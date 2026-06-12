"""grad_jvp_levers.py — isolate the staged-jvp COMPILE scaling and test levers.

Times ONLY the two expensive stages (perturbation full_evolution_batched +
spectrum get_Cl_batched) under forward-mode, at configurable (B, l_max, k_chunk),
for THREE schemes:
  jvp   : the current per-direction jax.jvp (compile + 1-direction warm)
  lin   : jax.linearize once, then apply the tangent map P times (compile-once,
          P cheap applies -- the throughput lever)
By sweeping B (fix l_max) vs l_max (fix B) you see which stage's COMPILE is
super-linear in which axis; by comparing jvp vs lin you see the linearize win.

Throwaway compile cache so every compile is COLD and visible.
Run via srun on a GPU node, PYTHONPATH=$(pwd).
  GJL_LMAX(128)  GJL_B(2)  GJL_KCHUNK(100)  GJL_P(2)  GJL_SCHEME(jvp|lin|both)
"""
import os, time, tempfile
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# GJL_CACHE=<dir> uses a PERSISTENT cache (to test whether the big jvp compiles
# are captured + reused across processes); default = throwaway (cold every run).
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.environ.get(
    "GJL_CACHE", tempfile.mkdtemp(prefix="jaxcc_levers_"))
# optional XLA flags passed straight through
if os.environ.get("GJL_XLA_FLAGS"):
    os.environ["XLA_FLAGS"] = os.environ["GJL_XLA_FLAGS"]
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import Model, _recmodel_cpu
from scan.batched_grad import raw_dict, derived_and_tangents, _to_float

LMAX = int(os.environ.get("GJL_LMAX", 128))
B = int(os.environ.get("GJL_B", 2))
KCHUNK = int(os.environ.get("GJL_KCHUNK", 100))
P = int(os.environ.get("GJL_P", 2))
SCHEME = os.environ.get("GJL_SCHEME", "both")
WRT = "h,omega_b,omega_cdm,n_s,ln10As".split(",")[:P]
P = len(WRT)


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def partition_stage(fn, primals_full):
    """Return (f_arr, diffs, statics) so f_arr(*diffs)=fn(*primals_full)."""
    diffs, statics = [], []
    for p in primals_full:
        d, s = eqx.partition(p, eqx.is_inexact_array)
        diffs.append(d); statics.append(s)

    def f_arr(*ds):
        return fn(*[eqx.combine(d, s) for d, s in zip(ds, statics)])
    return f_arr, tuple(diffs), tuple(statics)


def time_jvp(label, fn, primals_full, tangents_list):
    """Per-direction jax.jvp: compile (1st dir) + warm (2nd dir, cache hit)."""
    f_arr, diffs, _ = partition_stage(fn, primals_full)
    t0 = time.perf_counter()
    out, dot = jax.jvp(f_arr, diffs, tangents_list[0]); jax.block_until_ready((out, dot))
    cold = time.perf_counter() - t0
    warm = float('nan')
    if len(tangents_list) > 1:
        t0 = time.perf_counter()
        _, dot = jax.jvp(f_arr, diffs, tangents_list[1]); jax.block_until_ready(dot)
        warm = time.perf_counter() - t0
    print(f"  [jvp] {label:26s} cold {cold:8.1f}s   warm/dir {warm:7.2f}s   "
          f"(compile~{cold-warm:.1f}s)  peak {peak_gb():.2f} GB", flush=True)
    return out, dot


def time_linearize(label, fn, primals_full, tangents_list):
    """jax.linearize once (compile), then apply the linear tangent map P times."""
    f_arr, diffs, _ = partition_stage(fn, primals_full)
    t0 = time.perf_counter()
    out, lin = jax.linearize(f_arr, *diffs)
    # force the linear map to compile by applying once and blocking
    dot0 = lin(*tangents_list[0]); jax.block_until_ready((out, dot0))
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    for t in tangents_list:
        dot = lin(*t)
    jax.block_until_ready(dot)
    applyP = time.perf_counter() - t0
    print(f"  [lin] {label:26s} compile {cold:8.1f}s   apply x{len(tangents_list)} "
          f"{applyP:7.2f}s ({applyP/len(tangents_list):.2f}s/dir)   peak {peak_gb():.2f} GB",
          flush=True)
    return out, lin


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX} B={B} kchunk={KCHUNK} P={P} "
          f"scheme={SCHEME} xla={os.environ.get('XLA_FLAGS','')}", flush=True)
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=False,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    rng = np.random.default_rng(0)
    raw_ps = [raw_dict(h=0.6736 + 0.01 * rng.normal(),
                       omega_cdm=0.12 + 0.002 * rng.normal(),
                       n_s=0.965 + 0.005 * rng.normal()) for _ in range(B)]
    full_ps, params_dots = derived_and_tangents(model, raw_ps, WRT)
    params_batch = _to_float(jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps))

    PE = model.PE; SS = model.SS
    cpu = jax.devices('cpu')[0]
    gpu = jax.devices('gpu')[0]

    # ---- primal forward to get BG, PT (needed as primals for the two jvps) ----
    pre_BG = model._pre_recomb_batched(params_batch)
    ri_cpu = jax.device_put(pre_BG.recomb_inputs, cpu)
    p_cpu = jax.device_put(params_batch, cpu)
    hy = _recmodel_cpu(model.RecModel, True)
    recomb = jax.device_put(hy((ri_cpu, p_cpu)), gpu)
    BG = model._get_BG_batched(params_batch, pre_BG, recomb)
    PT = PE.full_evolution_batched((BG, params_batch), k_chunk_size=KCHUNK)
    jax.block_until_ready(PT)
    print("primal forward done\n", flush=True)

    # derived-param tangents (BG_dot) for each direction via the cheap stages.
    # For the lever test we only need *some* valid tangents to drive the two big
    # stages; build BG_dot per direction with the (cheap) staged jvps.
    from scan.batched_grad import jvp_stage
    BG_dots = []
    for pdot in params_dots:
        _, pre_BG_dot = jvp_stage(model._pre_recomb_batched, (params_batch,), (pdot,))
        ri_dot = jax.device_put(pre_BG_dot.recomb_inputs, cpu)
        p_dot = jax.device_put(pdot, cpu)
        _, recomb_dot = jvp_stage(lambda ri, p: hy((ri, p)),
                                  (ri_cpu, p_cpu), (ri_dot, p_dot))
        recomb_dot = jax.device_put(recomb_dot, gpu)
        _, BG_dot = jvp_stage(model._get_BG_batched,
                              (params_batch, pre_BG, recomb),
                              (pdot, pre_BG_dot, recomb_dot))
        BG_dots.append(BG_dot)

    # tangents for the perturbation stage: (BG_dot, pdot) per direction, filtered
    pert_fn = lambda bg, p: PE.full_evolution_batched((bg, p), k_chunk_size=KCHUNK)
    pert_tangents = [eqx.filter((BG_dots[j], params_dots[j]), eqx.is_inexact_array)
                     for j in range(P)]

    print("=== PERTURBATION stage (full_evolution_batched) ===", flush=True)
    if SCHEME in ("jvp", "both"):
        time_jvp("perturbation", pert_fn, (BG, params_batch), pert_tangents)
    if SCHEME in ("lin", "both"):
        _, lin_pert = time_linearize("perturbation", pert_fn, (BG, params_batch), pert_tangents)

    # tangents for the spectrum stage: (PT_dot, BG_dot, pdot). Need PT_dot; get it
    # from the perturbation jvp (one direction is enough structure-wise, but we
    # build per direction for the spectrum sweep).
    spec_fn = SS.get_Cl_batched
    PT_dots = []
    for j in range(P):
        _, PT_dot = jvp_stage(pert_fn, (BG, params_batch),
                              (BG_dots[j], params_dots[j]))
        PT_dots.append(PT_dot)
    spec_tangents = [eqx.filter((PT_dots[j], BG_dots[j], params_dots[j]),
                                eqx.is_inexact_array) for j in range(P)]

    print("=== SPECTRUM stage (get_Cl_batched) ===", flush=True)
    if SCHEME in ("jvp", "both"):
        time_jvp("spectrum", spec_fn, (PT, BG, params_batch), spec_tangents)
    if SCHEME in ("lin", "both"):
        time_linearize("spectrum", spec_fn, (PT, BG, params_batch), spec_tangents)

    print("\n[grad_jvp_levers] done", flush=True)


if __name__ == "__main__":
    main()

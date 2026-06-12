"""jvp_compile_levers.py — does smaller k_chunk (and/or XLA flags) tame the
staged-jvp COMPILE?

Hypothesis: the perturbation jvp compile blows up super-linearly in B because the
augmented ForwardMode-Kvaerno5 loop body is fused over vmap(k_chunk=100, B). A
SMALLER k_chunk under jvp shrinks that vmap axis -> smaller fusion -> faster
compile, at the cost of more (cached) chunks at runtime. This does NOT touch the
primal k_chunk=100 (the gradient is a separate path).

Times (cold = compile+run; run ~k_chunk-independent so cold DIFFS ~ compile diffs):
  - perturbation jvp at each k_chunk in JCL_KCHUNKS
  - spectrum jvp (one value)
GJL_XLA_FLAGS passes XLA_FLAGS through (e.g. opt-level) for a 2nd-config comparison.

Throwaway cache so compiles are COLD. Run via srun, PYTHONPATH=$(pwd).
  JCL_LMAX(128) JCL_B(8) JCL_KCHUNKS("100,25") JCL_XLA_FLAGS("")
"""
import os, time, tempfile
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = tempfile.mkdtemp(prefix="jaxcc_jcl_")
if os.environ.get("JCL_XLA_FLAGS"):
    os.environ["XLA_FLAGS"] = os.environ["JCL_XLA_FLAGS"]
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import Model, _recmodel_cpu
from scan.batched_grad import raw_dict, derived_and_tangents, jvp_stage, _to_float

LMAX = int(os.environ.get("JCL_LMAX", 128))
B = int(os.environ.get("JCL_B", 8))
KCHUNKS = [int(x) for x in os.environ.get("JCL_KCHUNKS", "100,25").split(",")]


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def main():
    print(f"devices={jax.devices()} lmax={LMAX} B={B} kchunks={KCHUNKS} "
          f"xla={os.environ.get('XLA_FLAGS','')}", flush=True)
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=False,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    rng = np.random.default_rng(0)
    raw_ps = [raw_dict(h=0.6736 + 0.01 * rng.normal(),
                       omega_cdm=0.12 + 0.002 * rng.normal(),
                       n_s=0.965 + 0.005 * rng.normal()) for _ in range(B)]
    full_ps, params_dots = derived_and_tangents(model, raw_ps, ["omega_cdm"])
    pb = _to_float(jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps))
    pdot = params_dots[0]
    cpu = jax.devices('cpu')[0]; gpu = jax.devices('gpu')[0]

    # ---- primal forward (only need BG; PT comes from k_chunk=100 for spectrum) ----
    pre_BG = model._pre_recomb_batched(pb)
    ri_cpu = jax.device_put(pre_BG.recomb_inputs, cpu); p_cpu = jax.device_put(pb, cpu)
    hy = _recmodel_cpu(model.RecModel, True)
    recomb = jax.device_put(hy((ri_cpu, p_cpu)), gpu)
    BG = model._get_BG_batched(pb, pre_BG, recomb)
    # BG_dot (cheap stages, one direction)
    _, pre_BG_dot = jvp_stage(model._pre_recomb_batched, (pb,), (pdot,))
    ri_dot = jax.device_put(pre_BG_dot.recomb_inputs, cpu); p_dot = jax.device_put(pdot, cpu)
    _, recomb_dot = jvp_stage(lambda ri, p: hy((ri, p)), (ri_cpu, p_cpu), (ri_dot, p_dot))
    recomb_dot = jax.device_put(recomb_dot, gpu)
    _, BG_dot = jvp_stage(model._get_BG_batched, (pb, pre_BG, recomb),
                          (pdot, pre_BG_dot, recomb_dot))
    print("primal forward + BG_dot done\n", flush=True)

    # ---- perturbation jvp COLD at each k_chunk ----
    print("=== PERTURBATION jvp cold (compile+run) vs k_chunk ===", flush=True)
    PT_ref = None
    for kc in KCHUNKS:
        t0 = time.perf_counter()
        PT_out, PT_dot = jvp_stage(
            lambda bg, p: model.PE.full_evolution_batched((bg, p), k_chunk_size=kc),
            (BG, pb), (BG_dot, pdot))
        jax.block_until_ready((PT_out, PT_dot))
        cold = time.perf_counter() - t0
        print(f"  k_chunk={kc:4d}: cold {cold:8.1f}s   peak {peak_gb():.2f} GB", flush=True)
        if kc == KCHUNKS[0]:
            PT_ref, PT_dot_ref = PT_out, PT_dot

    # ---- spectrum jvp COLD (uses k_chunk=KCHUNKS[0] PT) ----
    print("\n=== SPECTRUM jvp cold (compile+run) ===", flush=True)
    t0 = time.perf_counter()
    _, cl_dot = jvp_stage(model.SS.get_Cl_batched, (PT_ref, BG, pb),
                          (PT_dot_ref, BG_dot, pdot))
    jax.block_until_ready(cl_dot)
    print(f"  spectrum: cold {time.perf_counter()-t0:8.1f}s   peak {peak_gb():.2f} GB",
          flush=True)
    print("\n[jvp_compile_levers] done", flush=True)


if __name__ == "__main__":
    main()

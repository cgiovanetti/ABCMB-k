"""grad_compile_profile.py — per-stage jvp COMPILE vs WARM timing.

The batched AD gradient (scan/batched_grad.py) is correct (1.75e-5 vs single-path)
but its assembled staged-jvp compile is pathologically slow & super-linear in
(B, l_max). This script breaks staged_cl_and_grad apart and times EACH stage's
jvp separately -- cold (compile+run) and warm (run only) -- to find the culprit
fusion. Uses a THROWAWAY compile cache so nothing is masked by $SCRATCH.

Run via srun on a GPU node, PYTHONPATH=$(pwd).
  GCP_LMAX (128)  GCP_B (2)  GCP_WRT (omega_cdm,n_s)  GCP_KCHUNK (100)
"""
import os, time, tempfile
# fresh throwaway compile cache so every compile is COLD and visible
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = tempfile.mkdtemp(prefix="jaxcc_profile_")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import Model, _recmodel_cpu
from scan.batched_grad import raw_dict, derived_and_tangents, jvp_stage, _to_float

LMAX = int(os.environ.get("GCP_LMAX", 128))
B = int(os.environ.get("GCP_B", 2))
WRT = os.environ.get("GCP_WRT", "omega_cdm,n_s").split(",")
KCHUNK = int(os.environ.get("GCP_KCHUNK", 100))
P = len(WRT)


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def timed(label, fn):
    """Run fn() twice; report cold (compile+run) and warm (run). Returns result."""
    t0 = time.perf_counter()
    out = fn(); jax.block_until_ready(out)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = fn(); jax.block_until_ready(out)
    warm = time.perf_counter() - t0
    print(f"  {label:34s} cold {cold:8.1f}s   warm {warm:7.2f}s   "
          f"compile~{cold-warm:8.1f}s   peak {peak_gb():.2f} GB", flush=True)
    return out


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  B={B}  P={P}  kchunk={KCHUNK}  wrt={WRT}",
          flush=True)
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
    pdot0 = params_dots[0]   # one nuisance direction is enough to time the jvp compile

    PE = model.PE; SS = model.SS
    cpu = jax.devices('cpu')[0]
    try:
        gpu = jax.devices('gpu')[0]
    except Exception:
        gpu = None

    print("\n=== PRIMAL stages (forward only) ===", flush=True)
    pre_BG = timed("primal: _pre_recomb_batched",
                   lambda: model._pre_recomb_batched(params_batch))
    ri_cpu = jax.device_put(pre_BG.recomb_inputs, cpu)
    p_cpu = jax.device_put(params_batch, cpu)
    hy = _recmodel_cpu(model.RecModel, True)
    recomb = timed("primal: HyRex (CPU)", lambda: hy((ri_cpu, p_cpu)))
    if gpu is not None:
        recomb = jax.device_put(recomb, gpu)
    BG = timed("primal: _get_BG_batched",
               lambda: model._get_BG_batched(params_batch, pre_BG, recomb))
    PT = timed("primal: full_evolution_batched",
               lambda: PE.full_evolution_batched((BG, params_batch), k_chunk_size=KCHUNK))
    Cl = timed("primal: get_Cl_batched",
               lambda: SS.get_Cl_batched(PT, BG, params_batch))

    print("\n=== JVP stages (one direction) ===", flush=True)
    _, pre_BG_dot = timed("jvp: _pre_recomb_batched", lambda: jvp_stage(
        model._pre_recomb_batched, (params_batch,), (pdot0,)))
    ri_dot_cpu = jax.device_put(pre_BG_dot.recomb_inputs, cpu)
    p_dot_cpu = jax.device_put(pdot0, cpu)
    _, recomb_dot = timed("jvp: HyRex (CPU)", lambda: jvp_stage(
        lambda ri, p: hy((ri, p)), (ri_cpu, p_cpu), (ri_dot_cpu, p_dot_cpu)))
    if gpu is not None:
        recomb_dot = jax.device_put(recomb_dot, gpu)
    _, BG_dot = timed("jvp: _get_BG_batched", lambda: jvp_stage(
        model._get_BG_batched, (params_batch, pre_BG, recomb),
        (pdot0, pre_BG_dot, recomb_dot)))
    _, PT_dot = timed("jvp: full_evolution_batched", lambda: jvp_stage(
        lambda bg, p: PE.full_evolution_batched((bg, p), k_chunk_size=KCHUNK),
        (BG, params_batch), (BG_dot, pdot0)))
    _, cl_dot = timed("jvp: get_Cl_batched", lambda: jvp_stage(
        SS.get_Cl_batched, (PT, BG, params_batch), (PT_dot, BG_dot, pdot0)))

    print("\n[grad_compile_profile] done", flush=True)


if __name__ == "__main__":
    main()

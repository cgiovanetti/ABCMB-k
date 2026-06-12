"""grad_prod_shape.py — does the staged batched AD gradient COMPILE in bounded
time at PRODUCTION shape (lensing=True, large l_max)?

The handoff implied the production-shape compile is intractable. This runs the real
staged_cl_and_grad (scan/batched_grad.py) at a configurable l_max with lensing ON
(as the driver uses), small B, P=2, and a SMALL gradient k_chunk, and reports the
cold (compile+run) and warm times. Decisive tractability test.

Throwaway cache (cold). Run via srun, PYTHONPATH=$(pwd).
  GPS_LMAX(1024) GPS_B(2) GPS_KCHUNK(25) GPS_LENSING(1) GPS_WARM(0)
"""
import os, time, tempfile
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = tempfile.mkdtemp(prefix="jaxcc_gps_")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.batched_grad import raw_dict, derived_and_tangents, staged_cl_and_grad

LMAX = int(os.environ.get("GPS_LMAX", 1024))
B = int(os.environ.get("GPS_B", 2))
KCHUNK = int(os.environ.get("GPS_KCHUNK", 25))
LENSING = os.environ.get("GPS_LENSING", "1") != "0"
WARM = os.environ.get("GPS_WARM", "0") != "0"
WRT = ["omega_cdm", "n_s"]


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def main():
    print(f"devices={jax.devices()} lmax={LMAX} B={B} kchunk={KCHUNK} "
          f"lensing={LENSING} P={len(WRT)}", flush=True)
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=LENSING,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    rng = np.random.default_rng(0)
    raw_ps = [raw_dict(h=0.6736 + 0.01 * rng.normal(),
                       omega_cdm=0.12 + 0.002 * rng.normal(),
                       n_s=0.965 + 0.005 * rng.normal()) for _ in range(B)]
    full_ps, params_dots = derived_and_tangents(model, raw_ps, WRT)
    print(f"derived {B} cosmologies; starting staged gradient (cold)...", flush=True)

    t0 = time.perf_counter()
    out = staged_cl_and_grad(model, full_ps, params_dots, k_chunk_size=KCHUNK)
    jax.block_until_ready(out)
    cold = time.perf_counter() - t0
    print(f"[COLD] staged grad compile+run {cold:.1f}s (B={B},P={len(WRT)},lmax={LMAX},"
          f"kchunk={KCHUNK},lensing={LENSING}); peak {peak_gb():.2f} GB", flush=True)

    if WARM:
        t0 = time.perf_counter()
        out = staged_cl_and_grad(model, full_ps, params_dots, k_chunk_size=KCHUNK)
        jax.block_until_ready(out)
        warm = time.perf_counter() - t0
        print(f"[WARM] staged grad run {warm:.1f}s => {warm/B:.2f}s/cosmo-grad "
              f"(all P={len(WRT)}); compile~{cold-warm:.1f}s", flush=True)
    print("[grad_prod_shape] done", flush=True)


if __name__ == "__main__":
    main()

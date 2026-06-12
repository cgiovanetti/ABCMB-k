"""grad_prod_shape.py — production-shape throughput of the staged batched AD
gradient (Workstream A measurement).

Runs the real staged_cl_and_grad (scan/batched_grad.py) at a configurable l_max
with lensing ON (as the driver uses), P=2, a configurable k_chunk, and -- with
GPS_SHARD -- B-axis sharded across all visible GPUs (the same GSPMD path
call_batched uses). Reports the cold (compile+run) and WARM (second-call) times,
the per-direction throughput (s/cosmo/direction), and the per-device peak GPU
memory. The WARM number is the one that matters for the AD-vs-FD decision (the
compile is a ~5-min one-time-per-shape tax).

For a clean COLD compile number set GPS_COLDCACHE=1 (throwaway cache). The default
uses the persistent JAX_COMPILATION_CACHE_DIR so repeat sweeps don't recompile.
Run via srun, PYTHONPATH=$(pwd).
  GPS_LMAX(2508) GPS_B(32) GPS_KCHUNK(100) GPS_LENSING(1) GPS_SHARD(auto|0|1)
  GPS_COLDCACHE(0)
"""
import os, time, tempfile
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
if os.environ.get("GPS_COLDCACHE", "0") != "0":
    os.environ["JAX_COMPILATION_CACHE_DIR"] = tempfile.mkdtemp(prefix="jaxcc_gps_")
else:
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.join(os.environ.get("SCRATCH", "/pscratch/sd/c/carag"),
                     ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.batched_grad import raw_dict, derived_and_tangents, staged_cl_and_grad

LMAX = int(os.environ.get("GPS_LMAX", 2508))
B = int(os.environ.get("GPS_B", 32))
KCHUNK = int(os.environ.get("GPS_KCHUNK", 100))
LENSING = os.environ.get("GPS_LENSING", "1") != "0"
_shard_env = os.environ.get("GPS_SHARD", "auto").lower()
WRT = ["omega_cdm", "n_s"]

try:
    NDEV = len(jax.devices('gpu'))
except Exception:
    NDEV = 0
SHARD = (_shard_env == "1") or (_shard_env == "auto" and NDEV > 1)


def peak_gb():
    """Max over devices of peak_bytes_in_use (per-DEVICE peak, the sharded ceiling)."""
    try:
        return max(d.memory_stats()["peak_bytes_in_use"]
                   for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def main():
    print(f"devices={jax.devices()} lmax={LMAX} B={B} kchunk={KCHUNK} "
          f"lensing={LENSING} P={len(WRT)} shard={SHARD} ndev={NDEV}", flush=True)
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
    out = staged_cl_and_grad(model, full_ps, params_dots, k_chunk_size=KCHUNK,
                             shard=SHARD)
    jax.block_until_ready(out)
    cold = time.perf_counter() - t0
    print(f"[COLD] staged grad compile+run {cold:.1f}s (B={B},P={len(WRT)},lmax={LMAX},"
          f"kchunk={KCHUNK},lensing={LENSING},shard={SHARD})", flush=True)

    # WARM: run TWICE more; report the SECOND warm call (steadier, per the brief).
    warm = None
    for i in range(2):
        t0 = time.perf_counter()
        out = staged_cl_and_grad(model, full_ps, params_dots, k_chunk_size=KCHUNK,
                                 shard=SHARD)
        jax.block_until_ready(out)
        warm = time.perf_counter() - t0
        print(f"[WARM-{i+1}] staged grad run {warm:.1f}s", flush=True)
    pk = peak_gb()
    per_cd = warm / B / len(WRT)
    print(f"[RESULT] B={B} shard={SHARD} ndev={NDEV} | warm {warm:.1f}s "
          f"=> {warm/B:.3f} s/cosmo (all P={len(WRT)}), "
          f"{per_cd:.3f} s/cosmo/direction | per-device peak {pk:.2f} GB "
          f"(B_local={-(-B // max(NDEV,1)) if SHARD else B})", flush=True)
    print("[grad_prod_shape] done", flush=True)


if __name__ == "__main__":
    main()

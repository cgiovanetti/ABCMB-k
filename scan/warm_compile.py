"""warm_compile.py — one-time compile-cache warmer for the optimizer profile.

Compiles the EXACT shape the FD optimizer uses: call_batched(B=7, shard=True,
l_max=OPT_LMAX) through plik-lite chi2, runs it twice, prints the cold (compile)
vs warm (cached HLO) wall. The persistent JAX compile cache
($SCRATCH/.jax_cache_abcmb) is written on the first successful compile, so after
this exits, scan/profile_opt.py reuses it and pays seconds/eval instead of the
~10-15 min fresh compile that overran prior interactive walltimes.

Run via srun, PYTHONPATH=$(pwd). Env: OPT_LMAX (2508).
"""
import os, time
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
from scan.plik_lite import PlikLite
from scan.profile_opt import build, NUIS, CENTER, SIGMA, NDIM, FD_DELTA

LMAX = int(os.environ.get("OPT_LMAX", 2508))
pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)


def chi2_batched(phys_rows, ns):
    batch = [build(phys_rows[k], ns) for k in range(len(phys_rows))]
    out = model.call_batched(batch, shard=True)
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    return np.asarray(pl.profile_amplitude(m0)[0])


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}", flush=True)
    cache = os.path.join(_SCRATCH, ".jax_cache_abcmb")
    n0 = len(os.listdir(cache)) if os.path.isdir(cache) else 0
    print(f"cache entries before: {n0}", flush=True)

    # Build the exact FD stencil (B = 1 + 2*NDIM = 7) at a representative point.
    x = np.zeros(NDIM); ns = 0.9649
    pts = [x]
    for i in range(NDIM):
        e = np.zeros(NDIM); e[i] = FD_DELTA
        pts.append(x + e); pts.append(x - e)
    phys = CENTER + SIGMA * np.array(pts)
    print(f"stencil B={len(phys)} (shard=True, 1-device mesh on 1 GPU)", flush=True)

    t = time.perf_counter()
    c2 = chi2_batched(phys, ns)
    t_cold = time.perf_counter() - t
    print(f"COLD (compile+run): {t_cold:.1f}s   chi2[0]={c2[0]:.3f}", flush=True)

    t = time.perf_counter()
    c2b = chi2_batched(phys, ns)
    t_warm = time.perf_counter() - t
    print(f"WARM (cached HLO):  {t_warm:.2f}s   chi2[0]={c2b[0]:.3f}", flush=True)

    n1 = len(os.listdir(cache)) if os.path.isdir(cache) else 0
    print(f"cache entries after: {n1}  (+{n1-n0})", flush=True)
    print(f"speedup cold/warm = {t_cold/max(t_warm,1e-6):.0f}x", flush=True)


if __name__ == "__main__":
    main()

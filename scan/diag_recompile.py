"""diag_recompile.py — is call_batched(B=7) recompiling / leaking host RAM per call?

profile_opt.py OOM'd on the CPU backend (LLVM 'Unable to allocate section memory')
after ~27 min with ZERO POI completed. Hypothesis: the inline
`eqx.filter_jit(jax.vmap(self.RecModel), backend='cpu')` in _build_bgs_batched
(main.py:399) is rebuilt every call -> the CPU HyRex stage recompiles each call ->
slow + host-RAM accumulation of LLVM executables -> OOM.

This loops call_batched(B=7, shard=True) a handful of times and prints, per call:
  wall (s)         -- a ~2x jump (compile) vs steady (cached) distinguishes
  host VmRSS (GB)  -- monotonic growth => recompile/leak accumulation
The last two calls REUSE the first call's exact values (control): if wall/RSS
still grow on an identical-value repeat, the cache miss is wrapper-identity (not
value) driven -> confirms the inline-filter_jit bug.

Run via srun, PYTHONPATH=$(pwd). Cheap: ~6 calls.
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
from scan.profile_opt import build, CENTER, SIGMA, NDIM, FD_DELTA

LMAX = int(os.environ.get("OPT_LMAX", 2508))
pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6  # kB -> GB
    return float("nan")


def chi2_batched(phys_rows, ns):
    batch = [build(phys_rows[k], ns) for k in range(len(phys_rows))]
    out = model.call_batched(batch, shard=True)
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    return np.asarray(pl.profile_amplitude(m0)[0])


def stencil(x):
    pts = [x]
    for i in range(NDIM):
        e = np.zeros(NDIM); e[i] = FD_DELTA
        pts.append(x + e); pts.append(x - e)
    return CENTER + SIGMA * np.array(pts)


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}", flush=True)
    # 4 DIFFERENT points (as the optimizer would), then 2 IDENTICAL repeats of #0
    xs = [np.zeros(NDIM),
          np.array([0.5, 0.0, 0.0]),
          np.array([0.0, 0.5, -0.3]),
          np.array([-0.4, 0.2, 0.6])]
    nss = [0.9649, 0.9550, 0.9750, 0.9600]
    # control repeats (identical to call #0)
    xs += [np.zeros(NDIM), np.zeros(NDIM)]
    nss += [0.9649, 0.9649]

    print(f"  RSS start: {rss_gb():.2f} GB", flush=True)
    for i, (x, ns) in enumerate(zip(xs, nss)):
        phys = stencil(x)
        t = time.perf_counter()
        c2 = chi2_batched(phys, ns)
        dt = time.perf_counter() - t
        tag = "DIFF" if i < 4 else "REPEAT-of-#0"
        print(f"  call {i} [{tag}]: wall={dt:6.1f}s  RSS={rss_gb():6.2f} GB  "
              f"chi2[0]={c2[0]:.3f}", flush=True)
    print("\nVERDICT: steady wall + flat RSS => fine; growing RSS (esp. on the "
          "REPEAT calls) => per-call cache miss (inline filter_jit bug).", flush=True)


if __name__ == "__main__":
    main()

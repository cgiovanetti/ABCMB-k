"""precompute_hessian.py -- compute + cache the warm-start Hessian for the production
config (lcdm, l=2508, lowTT+lowEE) so the 6 POI_SLICE ranks LOAD it instantly instead
of each recomputing the flat ~18-min precond. Also validates the cache code path and
confirms the nuisance-Hessian conditioning is good at l=2508 (expect cond ~30-160, NOT
the ~2e4 seen at the under-constraining l=300 debug).

Run on debug (fits 30 min). NEVER login node.
"""
import os, time
os.environ.setdefault("PA_CONFIG", "scan/configs/lcdm.py")
import numpy as np
import scan.profile_prod_ad as P

theta_warm, prov = P._global_best_fit_physical()
print(f"[precompute] warm: {prov}", flush=True)
print(f"[precompute] cache path: {P._warm_hessian_cache_path()}", flush=True)
print(f"[precompute] LMAX={P.LMAX} lowtt={P.USE_LOWTT} lowee={P.USE_LOWEE}", flush=True)

t = time.perf_counter()
H = P._warm_precond_hessian(theta_warm)              # computes + saves (or loads if rerun)
print(f"[precompute] Hessian ready in {time.perf_counter()-t:.0f}s, shape={H.shape}", flush=True)

print("[precompute] per-POI nuisance-Hessian condition number:")
for poi in P.ORDER:
    hinv = P._hinv0_for_poi(H, P.ORDER.index(poi))
    cond = float(np.linalg.cond(np.linalg.inv(hinv)))
    print(f"   {poi:11s} cond = {cond:.1f}", flush=True)

# sanity: reload to confirm the cache round-trips and matches
H2 = P._warm_precond_hessian(theta_warm)
print(f"[precompute] cache round-trip max|dH| = {np.abs(H-H2).max():.2e} "
      f"(should be 0 -> loaded from disk)", flush=True)

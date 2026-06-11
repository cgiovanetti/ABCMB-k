"""tol_profconv_table.py — finalize the PROFILED tolerance-convergence study.

Loads the three profiled n_s scans (results/profile_prod_n_s_r{1e4,3e5,1e5}.npz),
each produced by the SAME profile_prod.py code with the SAME 7-pt grid / 3 Newton
iters and only rtol_large_k_PE differing. Re-extracts the central value + 1sigma
interval with the SAME interval()/sigma_parabola() used in production (the r1e5 run
timed out before its DONE block, so its interval was never written -- recompute it
from the saved chi2 grid). Reports the rtol-induced SHIFT vs the tightest rtol so
common extraction error cancels. Pure numpy/scipy; no JAX. Run via srun.
"""
import os, numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scan.profile_prod import interval, sigma_parabola

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TAGS = [("1e-4", "r1e4"), ("3e-5", "r3e5"), ("1e-5", "r1e5")]
NS_S = 0.0042   # Planck n_s sigma, for reporting shifts in % of sigma

rows = []
for rtol_str, tag in TAGS:
    f = os.path.join(RES, f"profile_prod_n_s_{tag}.npz")
    if not os.path.exists(f):
        print(f"  MISSING {f}"); continue
    d = np.load(f, allow_pickle=True)
    x = np.asarray(d["poi_grid"], float)
    y = np.asarray(d["chi2"], float)
    lo, x0, hi = interval(x, y, 1.0)          # Delta-chi2 = 1 -> 1 sigma
    sig_p = sigma_parabola(x, y)
    imin = int(np.nanargmin(y))
    rows.append(dict(rtol=rtol_str, argmin=x[imin], chi2min=float(np.nanmin(y)),
                     lo=lo, hi=hi, mid=0.5 * (lo + hi), hw=0.5 * (hi - lo),
                     sig_p=sig_p, iters=int(d["iter"]) if "iter" in d else -1,
                     done=bool(d["done"]) if "done" in d else False))

print(f"{'rtol':>6} {'iters':>5} {'done':>5} {'argmin':>9} {'chi2min':>9} "
      f"{'1sig_lo':>9} {'1sig_hi':>9} {'mid':>9} {'halfwidth':>9} {'parab_sig':>9}")
for r in rows:
    print(f"{r['rtol']:>6} {r['iters']:>5} {str(r['done']):>5} {r['argmin']:>9.5f} "
          f"{r['chi2min']:>9.2f} {r['lo']:>9.5f} {r['hi']:>9.5f} {r['mid']:>9.5f} "
          f"{r['hw']:>9.5f} {r['sig_p']:>9.5f}")

# convergence: shift vs the tightest rtol present (last row by construction = 1e-5)
ref = rows[-1]
print(f"\n=== SHIFT vs tightest rtol={ref['rtol']} (n_s sigma={NS_S}) ===")
print(f"{'rtol':>6} {'d(mid)':>10} {'d(mid)/sig':>11} {'d(halfwidth)':>13} {'d(hw)/hw_ref':>13}")
for r in rows:
    dmid = r['mid'] - ref['mid']; dhw = r['hw'] - ref['hw']
    print(f"{r['rtol']:>6} {dmid:>+10.5f} {dmid/NS_S*100:>+10.3f}% "
          f"{dhw:>+13.5f} {dhw/ref['hw']*100:>+12.3f}%")
print("\nLoosest rtol with |d(mid)| << 0.2% sigma (theory floor) & |d(hw)| << 1% wins.")

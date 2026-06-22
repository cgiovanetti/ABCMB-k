"""diag_consistency.py -- is BFGS stalling because the VALUE function (call_batched,
_chi2_from_out) and the GRADIENT function (staged-AD, _chi2_of_cls) are not the SAME
function? If they differ by ~0.1-0.3 in chi2, then minimizing the call_batched value
with the staged-AD gradient stalls at that mismatch (||g|| floor ~0.1-0.3, which is
exactly the observed median 0.135). Decisive test: evaluate BOTH chi2 paths at the
same rows and compare.

NEVER login node. ~one small AD block.
"""
import os
os.environ.setdefault("PA_CONFIG", "scan/configs/lcdm.py")
import numpy as np
import scan.profile_prod_ad as P

st = np.load("scan/results/profile_prod_ad_STATE.npz", allow_pickle=True)
POI_IDX = st["POI_IDX"]; PV = st["PV"]; x = np.array(st["x"], float)
g = np.array(st["g"], float); gn = np.abs(g).max(1)
N = len(PV)
worst = list(np.argsort(gn)[::-1][:4])
mid = list(np.argsort(np.abs(gn - 0.13))[:4])                 # 4 near-median rows
sel = np.array(sorted(set(worst + mid)))
print(f"[consistency] rows={list(sel)}  ||g||={[round(float(gn[b]),3) for b in sel]}", flush=True)

f_cb = P.fast_values_rows(POI_IDX[sel], x[sel], PV[sel])      # call_batched value path
chi2_ad, _ = P.ad_grad_rows(POI_IDX[sel], x[sel], PV[sel])    # staged-AD value path
print("\n row  POI         ||g||     f_callbatched   f_AD          diff(cb-AD)")
for k, b in enumerate(sel):
    print(f" {b:4d}  {P.ORDER[int(POI_IDX[b])]:9s} {gn[b]:7.3f}  {f_cb[k]:13.4f}  "
          f"{chi2_ad[k]:13.4f}  {f_cb[k]-chi2_ad[k]:+.4f}")
d = f_cb - chi2_ad
print(f"\n[summary] |f_cb - f_AD|: max={np.abs(d).max():.4f} mean={np.abs(d).mean():.4f}")
print(f"   median ||g|| in full set = {np.median(gn):.4f}")
print(f"   VERDICT: {'CONSISTENT (<0.01) -> stall is NOT value/grad mismatch; look elsewhere' if np.abs(d).max() < 0.01 else 'INCONSISTENT -> value(call_batched) != grad(staged-AD) chi2; BFGS stalls at this mismatch. Fix: use a CONSISTENT value-and-grad (the staged path returns both).'}")

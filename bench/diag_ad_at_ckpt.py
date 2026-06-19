"""diag_ad_at_ckpt.py -- is the gate's BFGS stall (||g||_FD ~ 20 >> GTOL=0.03,
chi2 dead flat) a TRUE stationary-point problem or just the FD noise floor?

Loads the timed-out gate checkpoint (results/profile_prod_ad_STATE.npz), then
evaluates the EXACT batched-AD gradient at the SAME stalled positions x and
compares to the stored FD gradient g. If ||g||_AD << ||g||_FD, the stall was FD
truncation/solver noise (the it0 calibration warned the FD step sat at its noise
floor, max-rel 1.2e-2 > 1e-2) and switching the lcdm config to grad_method='ad'
will let BFGS resume converging. If ||g||_AD ~ ||g||_FD, it's real
ill-conditioning and AD iterations are still needed but won't be a quick win.

Reuses the calibration's evenly-spaced 32-row subsample so the AD block shape
(B=32) hits the JAX compile cache from the gate's it0 calibration -> fast.

Run on GPU inside an srun (NEVER login node):
  PYTHONPATH=$(pwd) python -u bench/diag_ad_at_ckpt.py
"""
import os
os.environ.setdefault("PA_CONFIG", "scan/configs/lcdm.py")  # production data model
import numpy as np
import scan.profile_prod_ad as P

CKPT = "scan/results/profile_prod_ad_STATE.npz"
st = np.load(CKPT, allow_pickle=True)
POI_IDX = st["POI_IDX"]; PV = st["PV"]
x = np.array(st["x"], float)          # current BFGS position (g was evaluated here)
g = np.array(st["g"], float)          # FD gradient stored at x
best_f = np.array(st["best_f"], float)
it = int(st["it"])
N = len(PV)
gn_fd = np.abs(g).max(1)
print(f"[ckpt] it={it} N={N} P={P.P} grad_in_ckpt=FD  GTOL={P.GTOL}", flush=True)
print(f"[ckpt] ||g||_FD: max={gn_fd.max():.3e} median={np.median(gn_fd):.3e} "
      f"min={gn_fd.min():.3e}; chi2 min={best_f.min():.3f}", flush=True)

# calibration's exact subsample (covers all POIs/grid extents; B=32 cached shape)
n = min(max(P.FD_CALMIN, 1), N)
sel = np.unique(np.linspace(0, N - 1, n).astype(int))
print(f"[diag] evaluating EXACT AD grad at x for {len(sel)} rows (B={len(sel)}) ...",
      flush=True)
import time
t0 = time.perf_counter()
chi2_ad, G_ad = P.ad_grad_rows(POI_IDX[sel], x[sel], PV[sel])
print(f"[diag] AD grad done in {time.perf_counter()-t0:.0f}s", flush=True)

gn_ad = np.abs(G_ad).max(1)
gn_fd_sel = gn_fd[sel]
print("\n row  POI            ||g||_FD    ||g||_AD     ratio FD/AD")
for k, b in enumerate(sel):
    poi = P.ORDER[int(POI_IDX[b])]
    print(f" {b:4d}  {poi:12s}  {gn_fd_sel[k]:9.3e}  {gn_ad[k]:9.3e}   "
          f"{gn_fd_sel[k]/max(gn_ad[k],1e-30):8.2f}")

print(f"\n[summary] over {len(sel)} rows:")
print(f"   ||g||_FD : max={gn_fd_sel.max():.3e}  median={np.median(gn_fd_sel):.3e}")
print(f"   ||g||_AD : max={gn_ad.max():.3e}  median={np.median(gn_ad):.3e}")
print(f"   rows with ||g||_AD < GTOL={P.GTOL}: {int((gn_ad < P.GTOL).sum())}/{len(sel)}")
print(f"   VERDICT: {'FD-NOISE STALL (AD much smaller) -> switch to grad_method=ad' if np.median(gn_ad) < 0.3*np.median(gn_fd_sel) else 'AD comparable -> real conditioning, AD iters needed'}")
# also report per-POI AD ||g|| to see which directions are worst
print("\n[per-POI AD ||g|| max]:")
for pi in sorted(set(int(p) for p in POI_IDX[sel])):
    m = np.array([int(POI_IDX[b]) == pi for b in sel])
    print(f"   {P.ORDER[pi]:12s}  max||g||_AD={gn_ad[m].max():.3e}  "
          f"max||g||_FD={gn_fd_sel[m].max():.3e}")

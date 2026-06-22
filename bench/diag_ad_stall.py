"""diag_ad_stall.py -- why does AD BFGS stall at ||g||max~4.5 (it5..it8: 4.29,4.24,
4.61,4.98 rising) with chi2 dead-flat at 996.09?

Loads the latest gate checkpoint (it9) and answers, cheaply:
  1. Per-row ||g||_AD distribution -- is the bulk converging and only a few edge
     rows stuck (then the profile is basically fine), or is it systematic?
  2. WHERE are the stuck rows -- which POI, which grid index (edge vs interior),
     and are their nuisances pinned at the box (|x|~XBOX=5)?
  3. LINE-SEARCH PROBE at a few stuck rows: does ANY step -alpha*(Hinv@g) reduce
     the call_batched value f? If yes -> Hinv/line-search problem (a fresh clean
     inverse-Fisher restart fixes it). If no -> g is not a descent direction for f
     (gradient/value inconsistency -> deeper, a restart would NOT help).

Run on GPU inside srun/sbatch (debug queue, ~fits 30 min). NEVER login node.
"""
import os
os.environ.setdefault("PA_CONFIG", "scan/configs/lcdm.py")
import numpy as np
import scan.profile_prod_ad as P

CKPT = "scan/results/profile_prod_ad_STATE.npz"
st = np.load(CKPT, allow_pickle=True)
POI_IDX = st["POI_IDX"]; PV = st["PV"]
x = np.array(st["x"], float); g = np.array(st["g"], float)
Hinv = np.array(st["Hinv"], float); f = np.array(st["f"], float)
best_f = np.array(st["best_f"], float); it = int(st["it"])
N = len(PV); gn = np.abs(g).max(1)
NPTS = P.NPTS
print(f"[ckpt] it={it} N={N} P={P.P} GTOL={P.GTOL} XBOX={P.XBOX} grad={P.GRADMETHOD}", flush=True)

# (1) distribution
for thr in [0.03, 0.1, 0.3, 1.0, 2.0, 4.0]:
    print(f"   rows ||g||>{thr:<4}: {int((gn>thr).sum()):3d}/{N}")
print(f"   ||g|| median={np.median(gn):.3e} mean={gn.mean():.3e} max={gn.max():.3e}")
print(f"   chi2: min={best_f.min():.3f} f(min/med/max)={f.min():.2f}/{np.median(f):.2f}/{f.max():.2f}")

# (2) per-POI stuck counts + grid position of stuck rows + box pinning
print("\n[per-POI] stuck = ||g||>0.1; grid idx 0..%d (0,%d = +/-3sigma edges)" % (NPTS-1, NPTS-1))
for pi in range(P.D):
    sel = np.where(POI_IDX == pi)[0]
    if not len(sel):
        continue
    gsel = gn[sel]
    stuck = sel[gsel > 0.1]
    gidx = [int(np.where(sel == b)[0][0]) for b in stuck]    # position within this POI grid
    boxed = [int(np.max(np.abs(x[b])) > 0.98 * P.XBOX) for b in stuck]
    print(f"   {P.ORDER[pi]:11s} stuck {len(stuck):2d}/{len(sel)}  maxg={gsel.max():.2e}  "
          f"stuck_gridpos={gidx}  boxpinned={sum(boxed)}/{len(stuck)}")

# (3) line-search probe at the worst few stuck rows: does -alpha*(Hinv@g) reduce f?
probe = np.argsort(gn)[::-1][:4]                              # 4 worst rows
print(f"\n[probe] worst rows {list(probe)} (||g||={[round(float(gn[b]),2) for b in probe]}):")
alphas = [1.0, 0.5, 0.1, 0.03, 0.01]
# build one batch: for each probe row, base point + 5 trial steps along -Hinv@g
rows_b, Xtrial = [], []
for b in probe:
    d = -(Hinv[b] @ g[b])
    rows_b.append((b, None)); Xtrial.append(x[b])             # base
    for a in alphas:
        rows_b.append((b, a)); Xtrial.append(np.clip(x[b] + a * d, -P.XBOX, P.XBOX))
POI_b = np.array([POI_IDX[b] for (b, _) in rows_b])
PV_b = np.array([PV[b] for (b, _) in rows_b])
fvals = P.fast_values_rows(POI_b, np.array(Xtrial), PV_b)
k = 0
for b in probe:
    f0 = fvals[k]; k += 1
    line = f"   row {b:3d} ({P.ORDER[int(POI_IDX[b])]:9s}) f0={f0:.3f}:"
    best = f0
    for a in alphas:
        fa = fvals[k]; k += 1
        line += f"  a={a}:{fa-f0:+.3f}"
        best = min(best, fa)
    verdict = "DESCENT EXISTS (Hinv/LS issue)" if best < f0 - 1e-3 else "NO DESCENT (g vs f inconsistent?)"
    print(line + f"   -> {verdict}")

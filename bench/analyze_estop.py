#!/usr/bin/env python
"""Post-hoc validation of the chi2-plateau early-stop (scan/profile_prod_ad.py).

Reads a PA_BF_TRACE npz (per-iteration best_f history from a PA_FTOL=0 reference
run) and, for a sweep of (FTOL, FTOL_PATIENCE), replays the EXACT trigger logic
from bfgs_rows to find where the early-stop WOULD fire, then compares the sigma1
interval at that stop iteration against the fully-converged (final-iter) sigma1.

The deliverable question: does the early-stop bias the interval? Pass = the stop
iter is in the visibly-flat region AND |Dsigma1|/sigma_final << 0.05.

Usage (inside an srun; pure CPU numpy/scipy):
    python bench/analyze_estop.py scan/results/bf_trace_ln10As.npz
"""
import sys, os
import numpy as np

# Standalone (pure numpy/scipy) copies of the driver's interval/sigma_parabola so this
# analysis needs no heavy scan.profile_prod import. Kept BYTE-FAITHFUL to
# scan/profile_prod.py:interval / sigma_parabola -- if those change, update here.


def interval(x, y, level):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan, np.nan
    x, y = x[m], y[m]; o = np.argsort(x); x, y = x[o], y[o]
    try:
        from scipy.interpolate import PchipInterpolator
        p = PchipInterpolator(x, y - y.min())
        xs = np.linspace(x[0], x[-1], 40001); ys = p(xs)
    except Exception:
        xs = np.linspace(x[0], x[-1], 40001); ys = np.interp(xs, x, y - y.min())
    i = int(np.argmin(ys)); x0 = xs[i]; t = ys[i] + level

    def cross(side):
        seg, vs = (xs[:i + 1][::-1], ys[:i + 1][::-1]) if side < 0 else (xs[i:], ys[i:])
        k = np.where(vs >= t)[0]
        if len(k) == 0 or k[0] == 0:
            return np.nan
        j = k[0]; a, b, fa, fb = seg[j - 1], seg[j], vs[j - 1], vs[j]
        return a + (t - fa) * (b - a) / (fb - fa + 1e-30)
    return cross(-1), x0, cross(+1)


def sigma_parabola(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan
    d = y - y.min(); sel = d <= 4.0
    if sel.sum() < 3:
        sel = np.argsort(d)[:max(3, len(x) // 2)]
    a = np.polyfit(x[sel], y[sel], 2)[0]
    return np.nan if a <= 0 else 1.0 / np.sqrt(2.0 * a)


def sig1_halfwidth(grid, chi2):
    lo, mid, hi = interval(grid, chi2, 1.0)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return np.nan
    return 0.5 * (hi - lo)


def replay_trigger(bf_hist, ftol, patience):
    """Return the history index k (= iterations executed) at which the early-stop
    fires, mirroring bfgs_rows: after appending entry k (k>=1), once the window is
    long enough, stop when max per-row improvement over the window < ftol. Returns
    len-1 (ran to the end) if it never fires."""
    n = len(bf_hist)
    for k in range(1, n):
        hist_len = k + 1                       # bf_hist has k+1 entries after iter k
        if ftol > 0 and hist_len > patience:
            improve = float((bf_hist[k - patience] - bf_hist[k]).max())
            if improve < ftol:
                return k
    return n - 1


def main(path):
    d = np.load(path)
    bf = np.asarray(d["bf_hist"], float)       # (n_hist, N)
    PV = np.asarray(d["PV"], float)            # (N,) grid (single POI => PV is the grid)
    POI_IDX = np.asarray(d["POI_IDX"], int)
    n_hist, N = bf.shape
    npoi = len(np.unique(POI_IDX))
    print(f"trace {path}: {n_hist} history entries ({n_hist-1} BFGS iters), "
          f"N={N} rows, {npoi} POI(s)")
    if npoi != 1:
        print("WARNING: analysis assumes a SINGLE POI (PV == grid). Got "
              f"{npoi} POIs -- per-POI splitting not implemented here.")

    # ---- per-iteration sigma1 (both estimators) + window-improvement signal ----
    print("\n  k  iters   min_chi2   sig1_par   sig1_int   max_per_row_improve(win=3)")
    for k in range(n_hist):
        chi2 = bf[k]
        sp = sigma_parabola(PV, chi2)
        si = sig1_halfwidth(PV, chi2)
        if k >= 3:
            win = float((bf[k - 3] - bf[k]).max())
            wstr = f"{win:.2e}"
        else:
            wstr = "   --   "
        print(f"  {k:2d}  {k:5d}   {np.nanmin(chi2):8.3f}   "
              f"{sp:8.5f}   {si:8.5f}   {wstr}")

    sp_final = sigma_parabola(PV, bf[-1]); si_final = sig1_halfwidth(PV, bf[-1])
    print(f"\nFINAL (it{n_hist-1}): sigma1_parab={sp_final:.5f}  "
          f"sigma1_interval={si_final:.5f}")

    # ---- sweep candidate (FTOL, PATIENCE) ----
    print("\n=== early-stop trigger sweep (vs FINAL) ===")
    print(" FTOL    PAT  stop@it  iters_saved  sig1_par   d/sig_par   sig1_int   d/sig_int  verdict")
    for ftol in (3e-4, 5e-4, 1e-3, 2e-3, 5e-3):
        for pat in (3, 4, 5):
            k = replay_trigger(bf, ftol, pat)
            chi2 = bf[k]
            sp = sigma_parabola(PV, chi2); si = sig1_halfwidth(PV, chi2)
            dpar = (sp - sp_final) / sp_final if sp_final else np.nan
            dint = (si - si_final) / si_final if si_final else np.nan
            saved = (n_hist - 1) - k
            worst = max(abs(dpar), abs(dint))
            verdict = "OK" if worst < 0.05 else ("marginal" if worst < 0.1 else "BIAS")
            print(f" {ftol:.0e}  {pat:3d}  {k:6d}  {saved:11d}   "
                  f"{sp:8.5f}  {dpar:+8.4f}   {si:8.5f}  {dint:+8.4f}   {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "scan/results/bf_trace_ln10As.npz")

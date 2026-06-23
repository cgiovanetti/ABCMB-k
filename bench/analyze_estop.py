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


def sig1_series(bf, PV):
    """Per-iteration dchi2=1 interval half-width (single POI; PV is the grid)."""
    return np.array([sig1_halfwidth(PV, bf[k]) for k in range(len(bf))])


def replay_sigtol(s1, sigma_poi, sigtol, patience):
    """Replay the sigma1-STABILITY trigger (bfgs_rows): stop at history index k once the
    interval half-width has moved < sigtol*sigma_poi over the PATIENCE window (both ends
    finite). Returns k (= iters executed); len-1 if it never fires."""
    n = len(s1)
    for k in range(1, n):
        if sigtol > 0 and (k + 1) > patience:
            a, b = s1[k - patience], s1[k]
            if np.isfinite(a) and np.isfinite(b) and abs(b - a) / sigma_poi < sigtol:
                return k
    return n - 1


def analyze_one_poi(name, grid, chi2_hist, sigma, default=(1e-2, 3)):
    """Per-POI report: per-iter sigma1 + the sigma1-stability fire point at the default
    (SIGTOL,PATIENCE). Returns (s1_series, k_fire_at_default, si_final)."""
    n_hist = len(chi2_hist)
    s1 = np.array([sig1_halfwidth(grid, chi2_hist[k]) for k in range(n_hist)])
    si_final = s1[-1]
    print(f"\n--- POI {name}  (sigma={sigma:.5f}; final sigma1_int={si_final:.5f}) ---")
    print("   it  sig1_int   d(sig1)/sigma_to_final")
    for k in range(n_hist):
        dsig = abs(s1[k] - si_final) / sigma if np.isfinite(s1[k]) else np.nan
        print(f"   {k - 1 if k else 'X':>2}  {s1[k]:8.5f}   {dsig:18.4f}")
    sigtol, pat = default
    k = replay_sigtol(s1, sigma, sigtol, pat)
    dfire = abs(s1[k] - si_final) / sigma if np.isfinite(s1[k]) else np.nan
    tag = "FIRED" if k < n_hist - 1 else "ran-to-end (not enough iters to fire)"
    print(f"   default SIGTOL={sigtol:.0e} PAT={pat}: stop@it{k} (saved {n_hist-1-k}); "
          f"d(sigma1)={dfire:.4f}sigma  [{tag}]")
    return s1, k, si_final


def main(path):
    d = np.load(path)
    bf = np.asarray(d["bf_hist"], float)       # (n_hist, N)
    PV = np.asarray(d["PV"], float)            # (N,)
    POI_IDX = np.asarray(d["POI_IDX"], int)
    n_hist, N = bf.shape
    uniq = list(np.unique(POI_IDX))
    sig_order = np.asarray(d["sigma_order"], float) if "sigma_order" in d.files else None
    names = list(np.asarray(d["order"])) if "order" in d.files else None
    print(f"trace {path}: {n_hist} history entries ({n_hist-1} BFGS iters), "
          f"N={N} rows, {len(uniq)} POI(s)")
    SIGTOL, PAT = 1e-2, 3                       # the production default

    # ---- per-POI analysis (each POI's rows form its grid) ----
    per = {}
    for p in uniq:
        rows = np.where(POI_IDX == p)[0]
        name = (str(names[p]) if names is not None else f"poi{p}")
        sigma = float(sig_order[p]) if sig_order is not None else \
            sig1_halfwidth(PV[rows], bf[-1][rows])     # fallback: sig1_final
        per[p] = analyze_one_poi(name, PV[rows], bf[:, rows], sigma, (SIGTOL, PAT))

    # ---- GLOBAL trigger (the live bfgs_rows logic: stop when EVERY POI stable) ----
    print(f"\n=== GLOBAL sigma1-stability trigger (live logic; SIGTOL={SIGTOL:.0e} "
          f"PAT={PAT}) ===")
    k_global = n_hist - 1
    for k in range(1, n_hist):
        if (k + 1) <= PAT:
            continue
        ready, worst = True, 0.0
        for p in uniq:
            s1 = per[p][0]
            a, b = s1[k - PAT], s1[k]
            if not (np.isfinite(a) and np.isfinite(b)):
                ready = False; break
            sigma = float(sig_order[p]) if sig_order is not None else per[p][2]
            worst = max(worst, abs(b - a) / sigma)
        if ready and worst < SIGTOL:
            k_global = k; break
    fired = k_global < n_hist - 1
    print(f"  GLOBAL stop@it{k_global} (saved {n_hist-1-k_global} of {n_hist-1}) "
          f"-- {'FIRED' if fired else 'ran to end'}")
    # worst per-POI d(sigma1) at the global stop, vs each POI's final
    worst_d = 0.0
    for p in uniq:
        s1, _, sif = per[p]
        sigma = float(sig_order[p]) if sig_order is not None else sif
        if np.isfinite(s1[k_global]):
            worst_d = max(worst_d, abs(s1[k_global] - sif) / sigma)
    verdict = "OK" if worst_d < 0.02 else ("marginal" if worst_d < 0.05 else "BIAS")
    print(f"  worst per-POI d(sigma1) at the stop vs converged: {worst_d:.4f}sigma  [{verdict}]")
    print("  (the live trigger waits for the SLOWEST POI; speedup vs production MAXIT=18 "
          f"= 18/{k_global if fired else n_hist-1} iters)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "scan/results/bf_trace_calib.npz")

"""TODO items 20 / 22 on the EM-WIMP config (GPU): the u-basis gradient against the per-parameter one, and the
curvature along log10_m_chi with and without LINX's second-order term against the measured profile.

  MODE ab:    rows of the _v1emwimp log10_m_chi profile (all D dims free): (i) the per-parameter path
              (_LINX_UBASIS False), (ii) the u-basis path without, (iii) with the LINX term. (i) = (ii) to rounding;
              the profile slope and curvature along log10_m_chi (Schur complement over the other dims) of (ii) and
              (iii) against the profile's own value differences (a check, not used by the driver).
  MODE lm:    LM from the _v1emwimp starts of rows that crawled (h, Delta_Neff_init profiles), Gauss-Newton alone vs
              with the LINX term, PA_MAXIT iterations each: iterations to converge and the chi2 reached.

    PA_CONFIG=scan/configs/lcdm_mchi_dneff_linx_bbnlike_plikfull.py python scan/diag/gn_curv_check.py ab|lm
"""
import os, sys, time
import numpy as np
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
import scan.profile_prod_ad as d

MODE = sys.argv[1]
R = "/pscratch/sd/c/carag/ABCMB-k/scan/results"
IM = d.ORDER.index("log10_m_chi")
CURV = d.BBN.curv


def with_path(ubasis, curv):
    d._LINX_UBASIS = ubasis
    d.BBN.curv = CURV if curv else ()


def schur(g, H, j):
    o = [k for k in range(len(g)) if k != j]
    Hoo, Hjo = H[np.ix_(o, o)], H[j, o]
    return g[j] - Hjo @ np.linalg.solve(Hoo, g[o]), H[j, j] - Hjo @ np.linalg.solve(Hoo, Hjo)


if MODE == "ab":
    z = np.load(f"{R}/profile_prod_ad_log10_m_chi_v1emwimp.npz", allow_pickle=True)
    grid, c = z["poi_grid"], z["chi2"]
    JS = [int(s) for s in os.environ.get("GC_ROWS", "6,8,10,12").split(",")]
    FREE = np.tile(np.arange(d.D), (len(JS), 1))
    BASE = np.tile(d.CENTER, (len(JS), 1))
    X = np.empty((len(JS), d.D))
    pf = [d.ORDER.index(n) for n in z["nuis"]]
    for r, j in enumerate(JS):
        th = d.CENTER.copy(); th[IM] = grid[j]
        th[pf] = d.CENTER[pf] + d.SIGMA[pf] * z["xstar"][j]
        X[r] = (th - d.CENTER) / d.SIGMA
    out = {}
    for tag, ub, cv in (("param", False, False), ("ubasis", True, False), ("ubasis+linx", True, True)):
        with_path(ub, cv)
        t = time.time(); out[tag] = d.grad_gn_rows(BASE, FREE, X); log(f"{tag}: {time.time() - t:.0f}s")
    (cA, gA, HA), (cB, gB, HB), (cC, gC, HC) = out["param"], out["ubasis"], out["ubasis+linx"]
    log(f"u-basis vs per-parameter: chi2 max|d| {np.abs(cA - cB).max():.2e}; grad max|d|/max|g| "
        f"{np.abs(gA - gB).max() / np.abs(gA).max():.2e}; GN max|d|/max|H| {np.abs(HA - HB).max() / np.abs(HA).max():.2e}")
    h = d.SIGMA[IM] / np.diff(grid).mean()          # profile differences per scaled sigma of log10_m_chi
    np.savez("logs/gn_curv_check_ab.npz", JS=JS, X=X, **{f"{k}_{n}": v for k, o in out.items()
                                                          for n, v in zip("cgH", o)})
    for r, j in enumerate(JS):
        sl_m = (c[j + 1] - c[j - 1]) / 2 * h
        cu_m = (c[j + 1] - 2 * c[j] + c[j - 1]) * h ** 2
        sB, kB = schur(gB[r], HB[r], IM); sC, kC = schur(gC[r], HC[r], IM)
        log(f"m {10 ** grid[j]:5.1f} MeV: profile slope {sl_m:+.4f} (AD {sB:+.4f}); curvature measured {cu_m:.4f}, "
            f"GN {kB:.4f} ({kB / cu_m:.2f}x), GN+LINX {kC:.4f} ({kC / cu_m:.2f}x); H_mm GN {HB[r][IM, IM]:.3g} "
            f"GN+LINX {HC[r][IM, IM]:.3g}; eig min GN {np.linalg.eigvalsh(HB[r]).min():.2e}")

if MODE == "lm":
    picks = [p.split(":") for p in os.environ.get("GC_PICK", "h:2,h:6,Delta_Neff_init:4,Delta_Neff_init:7").split(",")]
    BASE, FREE, X0, XLO, XHI = [], [], [], [], []
    for poi, j in picks:
        z = np.load(f"{R}/profile_prod_ad_{poi}_v1emwimp.npz", allow_pickle=True)
        i = d.ORDER.index(poi); f = d.nuis_idx_of(i)
        b = np.where((z["rows_kind"] == 0) & (z["rows_j"] == int(j)))[0][0]
        base = d.CENTER.copy(); base[i] = z["poi_grid"][int(j)]
        BASE.append(base); FREE.append(f); X0.append(z["rows_x0"][b])
        lo, hi = d.scaled_box(np.array(f)); XLO.append(lo); XHI.append(hi)
        log(f"{poi}[{j}]: _v1emwimp final chi2 {z['rows_chi2'][b]:.3f}, m {10 ** (1.3 + 0.3 * z['rows_x'][b][f.index(IM)]):.1f} MeV")
    BASE, FREE, X0, XLO, XHI = map(np.array, (BASE, FREE, X0, XLO, XHI))
    maxit = int(os.environ.get("PA_MAXIT", "8"))
    for tag, cv in [v for v in (("GN+LINX", True), ("GN", False)) if v[0] in os.environ.get("GC_VARIANTS", "GN+LINX,GN").split(",")]:
        with_path(True, cv)
        S = d.lm_rows(BASE, FREE, XLO, XHI, X0, maxit=maxit, log_prefix=f"[{tag}]")
        conv = (S["gnorm"] < d.GTOL) | S["floor"]
        log(f"{tag}: chi2 {np.round(S['F'], 3).tolist()}; |g| {np.round(S['gnorm'], 3).tolist()}; converged "
            f"{conv.tolist()}; m {np.round(10 ** (1.3 + 0.3 * S['X'][np.arange(len(X0)), [list(f).index(IM) for f in FREE]]), 1).tolist()} MeV")

"""compare_profile_runs.py — overlay repeat runs of the LCDM+Neff full-plik profile.

Rows of two runs on the SAME grid must agree if the BFGS rows converged; any gap is
optimizer error. Every row's chi2 is an upper bound on the true profile at its POI
value, so the pointwise lower envelope over runs is the best available profile. A
parabola fit to that envelope gives an indicative best fit and 1-sigma.

`sig` per run is hardcoded because configs/lcdm_neff_plikfull_rc.py was edited in
place between _pfrc and _widecdm (xstar is stored in each run's own sigma units).

_v2fix ran on the Cl-step + AD fixes (branch xe-c1) and _v3 on the upstream merge with the
massive neutrino evolved: different chi2 surfaces (SURFACE), each offset by its own best
chi2; the envelope/parabola uses the newest surface present among the selected runs.

CPU only:  python scan/compare_profile_runs.py [TAG ...]   (default: every run in RUNS)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ORDER = ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"]
CEN_PF = [0.6736, 0.02237, 0.1200, 0.9649, 3.044, 0.0544, 3.044]      # lcdm_neff_plikfull.py
CEN_RC = [0.6662, 0.02229, 0.11654, 0.95986, 3.0446, 0.05933, 2.8268]  # lcdm_neff_plikfull_rc.py to 2026-10-19
CEN_V3 = [0.65895, 0.022221, 0.116773, 0.958684, 3.041751, 0.057532, 2.821114]  # rc recentred (massive nu)
CEN_V4 = CEN_V3[:6] + [-0.823]      # lcdm_dneff_linx_plikfull.py: the 7th parameter is Delta_Neff_init
SIG_OLD = [0.0054, 0.00015, 0.0012, 0.0042, 0.014, 0.0073, 0.2]
SIG_NEW = [0.0054, 0.00015, 0.0013, 0.0042, 0.014, 0.0073, 0.1]         # 2026-07-07 edit
SIG_V4 = SIG_NEW[:6] + [0.37]
RUNS = {"_pf": (CEN_PF, SIG_OLD), "_pfrc": (CEN_RC, SIG_OLD), "_widecdm": (CEN_RC, SIG_NEW),
        "_fixprec": (CEN_RC, SIG_NEW), "_fixprec2": (CEN_RC, SIG_NEW), "_v2": (CEN_RC, SIG_NEW),
        "_v2fix": (CEN_RC, SIG_NEW), "_v3": (CEN_V3, SIG_NEW), "_v4linx": (CEN_V4, SIG_V4)}
STYLE = {"_pf": ("#1baf7a", "s"), "_pfrc": ("#2a78d6", "o"), "_widecdm": ("#eb6834", "^"),
         "_fixprec": ("#eda100", "D"), "_fixprec2": ("#e87ba4", "v"), "_v2": ("#222222", "*"),
         "_v2fix": ("#7b3fbf", "P"), "_v3": ("#d62728", "h"), "_v4linx": ("#17becf", "X")}
XBOX = {"_v2": 10.0, "_v2fix": 10.0, "_v3": 10.0, "_v4linx": 10.0}                       # PA_XBOX; 5.0 otherwise
# chi2 surface of each run (runs on different surfaces are offset by their own best chi2 and never share
# an envelope): "pre-fix" = the original Cls; "xe-c1" = Cl-step + AD fixes; "massive-nu" = the upstream
# merge (#34-#50, #53) with the massive neutrino evolved (2026-10-19). The envelope uses the newest present.
SURFACE = {"_v2fix": "xe-c1", "_v3": "massive-nu", "_v4linx": "linx"}
SURF_ORDER = ["pre-fix", "xe-c1", "massive-nu", "linx"]
# runs whose 7th parameter is LINX's Delta_Neff_init: their Neff file is the Delta_Neff_init one and every row is
# placed at its derived Neff (npz derived_bbn[:, 1]) so it plots on the Neff axis with the others
LINX_RUNS = {"_v4linx"}


def _surf(tag):
    return SURFACE.get(tag, "pre-fix")


def load_rows(tags):
    """(tag, poi, chi2, pinned, theta[7]) for every row of every selected run."""
    rows = []
    for tag in tags:
        cen, sig = RUNS[tag]
        cen, sig = np.array(cen), np.array(sig)
        for poi in ORDER:
            fpoi = "Delta_Neff_init" if (tag in LINX_RUNS and poi == "Neff") else poi
            p = os.path.join(RES, f"profile_prod_ad_{fpoi}{tag}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            names = ["Neff" if str(n) == "Delta_Neff_init" else str(n) for n in d["nuis"]]
            sel = [k for k, n in enumerate(names) if n in ORDER]      # the BBN nuisances (q*, tau_n_fac) are not plotted
            nuis = [ORDER.index(names[k]) for k in sel]
            for i, c in enumerate(d["chi2"]):
                th = cen.copy()
                th[ORDER.index(poi)] = d["poi_grid"][i]
                th[nuis] = cen[nuis] + sig[nuis] * d["xstar"][i][sel]
                if tag in LINX_RUNS:
                    th[6] = d["derived_bbn"][i, 1]
                box = XBOX.get(tag, 5.0)
                rows.append((tag, poi, float(c), bool((np.abs(d["xstar"][i]) >= box - 1e-3).any()), th))
    return rows


def main():
    tags = sys.argv[1:] or list(RUNS)
    rows = load_rows(tags)
    smc = np.load(os.path.join(RES, "smc_plikfull_neff.npz"), allow_pickle=True)
    SMC = {str(k): (m, s) for k, m, s in zip(smc["cosmo_order"], smc["marg_mean"], smc["marg_std"])}
    gmins = {}
    for surf in SURF_ORDER:
        cs = [r[2] for r in rows if not r[3] and _surf(r[0]) == surf]
        if cs:
            gmins[surf] = min(cs)
            print(f"global best chi2 over unpinned rows, {surf} Cl surface: {gmins[surf]:.3f}")
    env_surf = [s for s in SURF_ORDER if s in gmins][-1]
    gmin = gmins[env_surf]
    print(f"{'POI':10s} {'envelope best':>13s} {'+/-1sig':>9s}   {'SMC mean +/- std':>20s}")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    axes = axes.ravel()
    for ax, poi in zip(axes, ORDER):
        j = ORDER.index(poi)
        env = {}
        for tag in tags:
            rr = sorted([r for r in rows if r[0] == tag and r[1] == poi], key=lambda r: r[4][j])
            if not rr:
                continue
            surf = _surf(tag)
            x = np.array([r[4][j] for r in rr]); c = np.array([r[2] for r in rr]) - gmins[surf]
            ok = ~np.array([r[3] for r in rr])
            col, mk = STYLE[tag]
            ax.plot(x[ok], c[ok], marker=mk, ms=6, lw=1.4, color=col, label=tag.lstrip("_"))
            ax.plot(x[~ok], np.minimum(c[~ok], 9.6), "x", ms=8, color=col)
            if surf != env_surf:
                continue
            for xi, ci in zip(x[ok], c[ok]):
                if ci < 30:
                    k = round(xi, 8); env[k] = min(env.get(k, np.inf), ci)
        if len(env) < 3:
            ax.set_title(f"{poi} (no envelope)"); continue
        ex = np.array(sorted(env)); ec = np.array([env[k] for k in ex])
        sel = ec - ec.min() < 8
        a2, a1, a0 = np.polyfit(ex[sel], ec[sel], 2)
        best, sig = -a1 / (2 * a2), (1 / np.sqrt(a2) if a2 > 0 else np.nan)
        xx = np.linspace(ex[sel].min(), ex[sel].max(), 300)
        ax.plot(xx, np.polyval([a2, a1, a0], xx), color="#222222", lw=1.2, ls="--",
                label="parabola fit to lower envelope")
        m, s = SMC[poi]
        ax.axvspan(m - s, m + s, color="#999999", alpha=0.18, lw=0, label="SMC mean ± 1σ")
        ax.axhline(1, color="#999999", lw=0.8, ls=":")
        ax.set_ylim(-0.2, 10); ax.set_title(poi); ax.set_xlabel(poi)
        ax.set_ylabel("χ² − best of the same Cl" if len(gmins) > 1 else f"χ² − global best ({gmin:.2f})")
        ax.grid(alpha=0.2, lw=0.6)
        print(f"{poi:10s} {best:13.5f} {sig:9.5f}   {m:10.5f} +/- {s:.5f}")
    axes[0].legend(fontsize=8, loc="upper center")
    axes[-1].axis("off")
    axes[-1].text(0.0, 0.95, "LCDM+Neff full-plik profiles\n(ABCMB-k, l_max 2508)\n\n"
                  "lines: per-run profile rows\n× : row with a nuisance pinned\n"
                  "    at the ±5σ box (clipped at 9.6)\ndashed: parabola to the pointwise\n"
                  "    lower envelope (fixed-Cl runs\n    only, if selected)\ngray band: SMC posterior, same\n"
                  "    likelihood + model\n\nsame grid, same physics: the\n_pfrc and _widecdm "
                  "curves should\ncoincide if the rows were converged",
                  va="top", fontsize=9, family="monospace")
    fig.tight_layout()
    out = os.path.join(RES, "profile_neff_runs_compared" + "".join(sys.argv[1:]) + ".png")
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()

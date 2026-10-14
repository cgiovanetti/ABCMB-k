"""compare_profile_runs.py — overlay repeat runs of the LCDM+Neff full-plik profile.

Rows of two runs on the SAME grid must agree if the BFGS rows converged; any gap is
optimizer error. Every row's chi2 is an upper bound on the true profile at its POI
value, so the pointwise lower envelope over runs is the best available profile. A
parabola fit to that envelope gives an indicative best fit and 1-sigma.

`sig` per run is hardcoded because configs/lcdm_neff_plikfull_rc.py was edited in
place between _pfrc and _widecdm (xstar is stored in each run's own sigma units).

_v2fix ran on the Cl-step + AD fixes (branch xe-c1), a different chi2 surface: its rows
are offset by their own best chi2, and the envelope/parabola uses only those runs when
any is selected.

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
CEN_RC = [0.6662, 0.02229, 0.11654, 0.95986, 3.0446, 0.05933, 2.8268]  # lcdm_neff_plikfull_rc.py
SIG_OLD = [0.0054, 0.00015, 0.0012, 0.0042, 0.014, 0.0073, 0.2]
SIG_NEW = [0.0054, 0.00015, 0.0013, 0.0042, 0.014, 0.0073, 0.1]         # 2026-07-07 edit
RUNS = {"_pf": (CEN_PF, SIG_OLD), "_pfrc": (CEN_RC, SIG_OLD), "_widecdm": (CEN_RC, SIG_NEW),
        "_fixprec": (CEN_RC, SIG_NEW), "_fixprec2": (CEN_RC, SIG_NEW), "_v2": (CEN_RC, SIG_NEW),
        "_v2fix": (CEN_RC, SIG_NEW)}
STYLE = {"_pf": ("#1baf7a", "s"), "_pfrc": ("#2a78d6", "o"), "_widecdm": ("#eb6834", "^"),
         "_fixprec": ("#eda100", "D"), "_fixprec2": ("#e87ba4", "v"), "_v2": ("#222222", "*"),
         "_v2fix": ("#7b3fbf", "P")}
XBOX = {"_v2": 10.0, "_v2fix": 10.0}                                    # PA_XBOX; 5.0 otherwise
FIXED_CL = {"_v2fix"}                                                   # Cl-step + AD fixes (xe-c1)


def load_rows(tags):
    """(tag, poi, chi2, pinned, theta[7]) for every row of every selected run."""
    rows = []
    for tag in tags:
        cen, sig = RUNS[tag]
        cen, sig = np.array(cen), np.array(sig)
        for poi in ORDER:
            p = os.path.join(RES, f"profile_prod_ad_{poi}{tag}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            nuis = [ORDER.index(str(n)) for n in d["nuis"]]
            for i, c in enumerate(d["chi2"]):
                th = cen.copy()
                th[ORDER.index(poi)] = d["poi_grid"][i]
                th[nuis] = cen[nuis] + sig[nuis] * d["xstar"][i]
                box = XBOX.get(tag, 5.0)
                rows.append((tag, poi, float(c), bool((np.abs(d["xstar"][i]) >= box - 1e-3).any()), th))
    return rows


def main():
    tags = sys.argv[1:] or list(RUNS)
    rows = load_rows(tags)
    smc = np.load(os.path.join(RES, "smc_plikfull_neff.npz"), allow_pickle=True)
    SMC = {str(k): (m, s) for k, m, s in zip(smc["cosmo_order"], smc["marg_mean"], smc["marg_std"])}
    gmins = {}
    for fixed in (False, True):
        cs = [r[2] for r in rows if not r[3] and (r[0] in FIXED_CL) == fixed]
        if cs:
            gmins[fixed] = min(cs)
            print(f"global best chi2 over unpinned rows, {'fixed' if fixed else 'pre-fix'} Cl: {gmins[fixed]:.3f}")
    env_fixed = True in gmins
    gmin = gmins[env_fixed]
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
            fixed = tag in FIXED_CL
            x = np.array([r[4][j] for r in rr]); c = np.array([r[2] for r in rr]) - gmins[fixed]
            ok = ~np.array([r[3] for r in rr])
            col, mk = STYLE[tag]
            ax.plot(x[ok], c[ok], marker=mk, ms=6, lw=1.4, color=col, label=tag.lstrip("_"))
            ax.plot(x[~ok], np.minimum(c[~ok], 9.6), "x", ms=8, color=col)
            if fixed != env_fixed:
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

"""collect_profiles.py -- gather the per-POI frequentist profile npz (written
independently by each POI_SLICE rank) into one summary: a table of best-fit +/- 1sigma
intervals with convergence certificates, compared to Planck 2018 and the validated SMC
posterior, plus an overlay Delta-chi^2 plot.

Usage (inside an srun/allocation; pure numpy+matplotlib, no GPU needed):
  python scan/collect_profiles.py [TAG] [PREFIX]
    TAG     default "_mn6"            (production POI_SLICE run)
    PREFIX  default "profile_prod_ad" (use "profile_prod" + TAG="" to read the
                                       entry-(a) FD profiles, e.g. for a dry test)

Reads scan/results/<PREFIX>_<poi><TAG>.npz. Core fields (poi_grid, chi2, sigma1=
[lo,mid,hi], sigma2) are required; certificate fields (gnorm, converged, gtol,
gradmethod) are optional so the entry-(a) npz (which lack them) also parse. Robust to
POIs not yet written (each POI_SLICE rank writes when it finishes).
"""
import os, sys, glob
import numpy as np

TAG = sys.argv[1] if len(sys.argv) > 1 else "_mn6"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "profile_prod_ad"


def _get(d, key, default):
    return np.array(d[key]) if key in d.files else default
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ORDER = ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion"]

# reference values (central, 1sigma)
PLANCK = {"h": (0.6736, 0.0054), "omega_b": (0.02237, 0.00015),
          "omega_cdm": (0.1200, 0.0012), "n_s": (0.9649, 0.0042),
          "ln10As": (3.044, 0.014), "tau_reion": (0.0544, 0.0073)}
SMC = {"h": (0.6789, 0.0062), "omega_b": (0.022385, 0.000155),
       "omega_cdm": (0.11992, 0.00137), "n_s": (0.96479, 0.00446),
       "ln10As": (3.0511, 0.0126), "tau_reion": (0.05887, 0.00597)}

rows = {}
for poi in ORDER:
    f = os.path.join(RES, f"{PREFIX}_{poi}{TAG}.npz")
    if not os.path.exists(f):
        continue
    try:
        d = np.load(f, allow_pickle=True)
    except Exception as ex:
        print(f"  (skip {poi}: {ex})"); continue
    rows[poi] = d

print(f"\n=== frequentist profile summary (TAG={TAG}) -- {len(rows)}/6 POIs present ===")
print(f"{'POI':11s} {'best-fit':>10s} {'-1sig':>9s} {'+1sig':>9s}  "
      f"{'conv':>7s} {'max|g|':>8s}  {'vs Planck':>10s} {'vs SMC':>8s}")
summary = {}
for poi in ORDER:
    if poi not in rows:
        print(f"{poi:11s} {'(pending)':>10s}")
        continue
    d = rows[poi]
    s1 = np.array(d["sigma1"], float)              # [lo, mid, hi]
    lo, mid, hi = s1
    grid = np.array(d["poi_grid"], float); chi2 = np.array(d["chi2"], float)
    ntot = len(grid)
    gn = _get(d, "gnorm", np.full(ntot, np.nan)).astype(float)
    conv = _get(d, "converged", np.zeros(ntot, bool))
    has_cert = "converged" in d.files
    gtol = float(d["gtol"]) if "gtol" in d.files else float("nan")
    nconv = int(np.nansum(conv)); maxg = float(np.nanmax(gn)) if np.isfinite(gn).any() else float("nan")
    half = 0.5 * (hi - lo)
    pc, ps = PLANCK[poi]; sc, ss = SMC[poi]
    dpl = (mid - pc) / ps if ps else float("nan")
    dsm = (mid - sc) / ss if ss else float("nan")
    conv_str = f"{nconv:3d}/{ntot:<3d}" if has_cert else f"  n/a({ntot})"
    print(f"{poi:11s} {mid:10.5f} {mid-lo:9.5f} {hi-mid:9.5f}  "
          f"{conv_str:>7s} {maxg:8.2e}  {dpl:+9.2f}s {dsm:+7.2f}s")
    summary[poi] = dict(best=mid, lo=lo, hi=hi, half=half, nconv=nconv, ntot=ntot,
                        maxg=maxg, gtol=gtol, dpl=dpl, dsm=dsm)

if rows:
    print(f"\nlegend: best-fit = Delta-chi^2 minimum; +/-1sig = Delta-chi^2=1 crossings;")
    print(f"        conv = rows with ||g||_AD<GTOL; vs X = (best - X_central)/X_sigma.")
    print(f"        gradmethod={str(rows[next(iter(rows))]['gradmethod']) if 'gradmethod' in rows[next(iter(rows))].files else '?'}, "
          f"GTOL={summary[next(iter(summary))]['gtol']:.0e}")

    # overlay plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = len(rows); fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, poi in zip(axes.ravel(), ORDER):
            if poi not in rows:
                ax.set_title(f"{poi} (pending)"); ax.axis("off"); continue
            d = rows[poi]
            grid = np.array(d["poi_grid"], float); chi2 = np.array(d["chi2"], float)
            dchi = chi2 - np.nanmin(chi2)
            ax.plot(grid, dchi, "-", color="0.6", lw=1, zorder=1)
            if "converged" in d.files:
                conv = np.array(d["converged"]).astype(bool)
                ax.scatter(grid[conv], dchi[conv], s=22, c="C0", label="converged", zorder=3)
                if (~conv).any():
                    ax.scatter(grid[~conv], dchi[~conv], s=22, c="C3", marker="x",
                               label="not certified", zorder=3)
            else:
                ax.scatter(grid, dchi, s=22, c="C0", zorder=3)
            for lv, c in [(1, "g"), (4, "orange")]:
                ax.axhline(lv, ls="--", lw=0.7, color=c)
            ax.axvline(PLANCK[poi][0], ls=":", color="k", lw=0.8)
            ax.axvspan(PLANCK[poi][0]-PLANCK[poi][1], PLANCK[poi][0]+PLANCK[poi][1],
                       color="k", alpha=0.07)
            ax.set_xlabel(poi); ax.set_ylabel(r"$\Delta\chi^2$")
            ax.set_ylim(0, 10)
            if poi == ORDER[0]:
                ax.legend(fontsize=8)
        fig.suptitle(f"ABCMB frequentist profiles (plik-lite+lowTT+lowEE), TAG={TAG} "
                     f"(dotted=Planck18, band=+/-1sig)")
        fig.tight_layout()
        out = os.path.join(RES, f"profiles_summary{TAG}.png")
        fig.savefig(out, dpi=120); print(f"\nsaved overlay -> {out}")
    except Exception as ex:
        print(f"plot skipped: {ex}")

    np.savez(os.path.join(RES, f"profiles_summary{TAG}.npz"),
             **{f"{p}_{k}": v for p, s in summary.items() for k, v in s.items()})
    print(f"saved summary -> {os.path.join(RES, f'profiles_summary{TAG}.npz')}")
else:
    print("no per-POI npz found yet (production still running/queued).")

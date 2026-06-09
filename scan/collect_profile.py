"""collect_profile.py — assemble the slice_*.npz from a profile grid scan into
the full chi^2 grid, project 1-D profile likelihoods for each scanned parameter,
report the minimum + Delta chi^2 = 1/4/9 (1/2/3 sigma) intervals, and plot.

Pure numpy + matplotlib (Agg) — light enough to run as the post-srun step of the
SLURM job (on a compute node; never the login node).

Usage:  python scan/collect_profile.py [out_dir]   (default scan/out_profile)
"""
import os, sys, glob
import numpy as np
from scan import profile_config as cfg


def interval(x, chi2, dlevel):
    """Delta chi2 = dlevel crossing on each side of the min (linear interp)."""
    imin = int(np.argmin(chi2)); target = chi2[imin] + dlevel
    def cross(up):
        rng = range(imin, len(x) - 1) if up else range(imin, 0, -1)
        for i in rng:
            j = i + 1 if up else i - 1
            if (chi2[i] - target) * (chi2[j] - target) <= 0:
                f = (target - chi2[i]) / (chi2[j] - chi2[i] + 1e-30)
                return x[i] + f * (x[j] - x[i])
        return np.nan
    return cross(False), cross(True)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out_profile")
    files = sorted(glob.glob(os.path.join(out, "slice_*.npz")))
    if not files:
        print(f"no slice_*.npz in {out}"); return

    Ntot = cfg.n_total()
    shape = tuple(cfg.NPTS[p] for p in cfg.SCAN_ORDER)
    axes = cfg.make_axes()
    flat = np.full(Ntot, np.inf)
    aplk = np.full(Ntot, np.nan)
    seen = np.zeros(Ntot, dtype=bool)
    for f in files:
        d = np.load(f)
        g = d["gidx"]
        flat[g] = d["chi2"]
        aplk[g] = d["A_planck"]
        seen[g] = True
    nmiss = int((~seen).sum())
    print(f"loaded {len(files)} slices; {seen.sum()}/{Ntot} grid points present"
          + (f"  (WARNING: {nmiss} MISSING -> filled +inf)" if nmiss else ""))

    chi2 = flat.reshape(shape)
    cmin = float(np.min(chi2))
    imin = np.unravel_index(int(np.argmin(chi2)), shape)
    print(f"\nglobal min chi2 = {cmin:.2f}  (A_planck = {aplk[np.argmin(flat)]:.5f})")
    print("best-fit grid point:")
    for k, p in enumerate(cfg.SCAN_ORDER):
        edge = " <-- on grid edge" if imin[k] in (0, shape[k]-1) else ""
        print(f"  {p:10s} = {axes[p][imin[k]]:.6g}{edge}")

    # 1-D profiles: min over all other axes
    profiles = {}
    print("\nprofiled 1-sigma (Delta chi2 = 1) intervals:")
    for k, p in enumerate(cfg.SCAN_ORDER):
        other = tuple(j for j in range(len(shape)) if j != k)
        prof = np.min(chi2, axis=other)
        profiles[p] = prof
        lo1, hi1 = interval(axes[p], prof, 1.0)
        best = axes[p][int(np.argmin(prof))]
        half = (hi1 - lo1) / 2 if np.isfinite(lo1) and np.isfinite(hi1) else np.nan
        print(f"  {p:10s} = {best:.6g}  [{lo1:.6g}, {hi1:.6g}]  (+/-{half:.3g})")

    np.savez(os.path.join(out, "profile_result.npz"),
             chi2=chi2, cmin=cmin, shape=np.array(shape),
             **{f"axis_{p}": axes[p] for p in cfg.SCAN_ORDER},
             **{f"prof_{p}": profiles[p] for p in cfg.SCAN_ORDER})
    print(f"\nsaved -> {os.path.join(out, 'profile_result.npz')}")

    # plot 1-D profiles
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(cfg.SCAN_ORDER)
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        ax = ax.ravel()
        for k, p in enumerate(cfg.SCAN_ORDER):
            x = axes[p]; y = profiles[p] - profiles[p].min()
            ax[k].plot(x, y, "o-")
            for d, c in [(1, "g"), (4, "orange"), (9, "r")]:
                ax[k].axhline(d, ls="--", lw=0.8, color=c)
            ax[k].axvline(cfg.CENTER[p], ls=":", color="gray", lw=0.8)
            ax[k].set_xlabel(p); ax[k].set_ylabel(r"$\Delta\chi^2$")
            ax[k].set_ylim(0, 12); ax[k].set_title(p)
        fig.suptitle(f"LCDM x plik-lite profile likelihoods  "
                     f"(min chi2 = {cmin:.1f}, N_data = 613)")
        fig.tight_layout()
        png = os.path.join(out, "profile_result.png")
        fig.savefig(png, dpi=110)
        print(f"saved -> {png}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()

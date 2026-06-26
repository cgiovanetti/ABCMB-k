"""wilks_collect.py -- merge per-rank wilks.py slices, compute the coverage / Wilks
statistics, and write the summary table + plots.

Usage (inside an srun; CPU only, no abcmb/GPU needed):
    python scan/wilks_collect.py <config_name><tag>
e.g.  python scan/wilks_collect.py lcdm           (merges wilks_lcdm_r*.npz)
      python scan/wilks_collect.py lcdm_neff_v1   (merges wilks_lcdm_neff_v1_r*.npz)

Pools the per-mock test statistics {t_poi} across all ranks (each rank fit a disjoint
slice of the mock population), compares the pooled distribution to chi2_1 (Wilks):
empirical coverage at the standard Delta-chi2 levels, KS test, moments; and the signed
pull vs N(0,1).  Writes wilks_<name>_merged.npz + per-POI png + a summary png.
"""
import os, sys, glob
import numpy as np
from scipy.stats import chi2 as _chi2dist, kstest, norm as _norm

_HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(_HERE, "results")

CHI2_1_MEDIAN = 0.4549364231
COV_LEVELS = {1.0: 0.6826894921, 2.7055434540954: 0.90,
              3.8414588206941: 0.95, 9.0: 0.9973002039}


def wilks_stats(t):
    t = np.asarray(t, float); t = t[np.isfinite(t)]; n = len(t)
    cov = {}
    for lvl, nominal in COV_LEVELS.items():
        frac = float(np.mean(t <= lvl))
        se = np.sqrt(max(nominal * (1 - nominal) / max(n, 1), 1e-12))
        cov[lvl] = dict(frac=frac, nominal=nominal, dev_sigma=(frac - nominal) / se)
    ks = kstest(t, _chi2dist(1).cdf)
    return dict(n=n, mean=float(t.mean()), median=float(np.median(t)),
                coverage=cov, ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                neg_frac=float(np.mean(t < -1e-6)))


def _plot_poi(poi, t, z, png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = np.asarray(t, float); t = t[np.isfinite(t)]
    z = np.asarray(z, float); z = z[np.isfinite(z)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    tc = np.clip(t, 0, None); hi = max(10, np.percentile(tc, 99))
    ax[0].hist(tc, bins=40, range=(0, hi), density=True, alpha=0.6, label="mocks")
    xs = np.linspace(1e-3, hi, 400)
    ax[0].plot(xs, _chi2dist(1).pdf(xs), "r-", lw=2, label=r"$\chi^2_1$")
    ax[0].set_xlabel(r"$t=\Delta\chi^2(\theta_{\rm true})$"); ax[0].set_ylabel("pdf")
    ax[0].set_title(f"{poi}: test statistic"); ax[0].legend()
    ts = np.sort(tc); ecdf = np.arange(1, len(ts) + 1) / len(ts)
    ax[1].plot(ts, ecdf, drawstyle="steps-post", label="empirical")
    ax[1].plot(xs, _chi2dist(1).cdf(xs), "r-", lw=2, label=r"$\chi^2_1$")
    for lv in (1.0, 3.8414588206941):
        ax[1].axvline(lv, ls=":", lw=0.8, color="gray")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("CDF"); ax[1].set_title("coverage"); ax[1].legend()
    ax[2].hist(z, bins=40, range=(-4, 4), density=True, alpha=0.6, label="mocks")
    zs = np.linspace(-4, 4, 400); ax[2].plot(zs, _norm.pdf(zs), "r-", lw=2, label="N(0,1)")
    ax[2].set_xlabel(r"pull $(\hat\theta-\theta_{\rm true})/\sigma$")
    ax[2].set_title(f"{poi}: pull"); ax[2].legend()
    fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)


def _plot_summary(pois, stats, png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.4 * len(pois) + 3, 4.2))
    x = np.arange(len(pois))
    for lvl, col, lab in [(1.0, "g", r"68.3% ($t<1$)"),
                          (3.8414588206941, "r", r"95% ($t<3.84$)")]:
        fr = [stats[p]["coverage"][lvl]["frac"] for p in pois]
        ax.plot(x, fr, "o-", color=col, label=lab)
        ax.axhline(COV_LEVELS[lvl], ls="--", lw=0.8, color=col)
    ax.set_xticks(x); ax.set_xticklabels(pois, rotation=30, ha="right")
    ax.set_ylabel("empirical coverage"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Wilks coverage vs nominal ($\\chi^2_1$)"); ax.legend()
    fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)


def main():
    if len(sys.argv) < 2:
        print("usage: python scan/wilks_collect.py <config_name><tag>"); sys.exit(1)
    key = sys.argv[1]
    files = sorted(glob.glob(os.path.join(RESDIR, f"wilks_{key}_r*.npz")))
    if not files:
        single = os.path.join(RESDIR, f"wilks_{key}.npz")
        if os.path.exists(single):
            files = [single]
        else:
            print(f"no wilks_{key}_r*.npz or wilks_{key}.npz in {RESDIR}"); sys.exit(1)
    print(f"[collect] {len(files)} slice file(s) for '{key}'")
    d0 = np.load(files[0], allow_pickle=True)
    pois = [str(p) for p in d0["pois"]]
    order = [str(p) for p in d0["order"]]
    theta_true = np.asarray(d0["theta_true"], float)
    merged = {f"t_{p}": [] for p in pois}
    merged.update({f"z_{p}": [] for p in pois})
    merged.update({f"gnorm_{p}": [] for p in pois})
    gnorm_global = []; chi2_global = []; ntot = 0
    cert = {p: {"t_shared": [], "t_exact": []} for p in pois}
    for f in files:
        d = np.load(f, allow_pickle=True)
        n = len(d["chi2_global"]); ntot += n
        gnorm_global.append(np.asarray(d["gnorm_global"]))
        chi2_global.append(np.asarray(d["chi2_global"]))
        for p in pois:
            merged[f"t_{p}"].append(np.asarray(d[f"t_{p}"]))
            merged[f"z_{p}"].append(np.asarray(d[f"z_{p}"]))
            if f"gnorm_{p}" in d.files:
                merged[f"gnorm_{p}"].append(np.asarray(d[f"gnorm_{p}"]))
            if f"cert_tshared_{p}" in d.files:
                cert[p]["t_shared"].append(np.asarray(d[f"cert_tshared_{p}"]))
                cert[p]["t_exact"].append(np.asarray(d[f"cert_texact_{p}"]))
    for k in merged:
        merged[k] = np.concatenate(merged[k]) if merged[k] else np.array([])
    gnorm_global = np.concatenate(gnorm_global); chi2_global = np.concatenate(chi2_global)
    print(f"[collect] pooled {ntot} mocks; global fit max||g||={gnorm_global.max():.2e}")

    print("\n===== WILKS / COVERAGE SUMMARY (expected: chi2_1) =====")
    stats = {}
    out = dict(key=key, order=np.array(order), pois=np.array(pois),
               theta_true=theta_true, ntot=ntot,
               gnorm_global=gnorm_global, chi2_global=chi2_global)
    for p in pois:
        t = merged[f"t_{p}"]; z = merged[f"z_{p}"]
        st = wilks_stats(t); stats[p] = st
        gtxt = ""
        if len(merged[f"gnorm_{p}"]):
            gtxt = f" max||g||cond={merged[f'gnorm_{p}'].max():.1e}"
        print(f"[{p}] n={st['n']} mean={st['mean']:.3f}(1.0) "
              f"median={st['median']:.3f}({CHI2_1_MEDIAN:.3f}) "
              f"KS={st['ks_stat']:.3f} p={st['ks_p']:.3f} neg={st['neg_frac']:.3f}{gtxt}")
        for lvl in (1.0, 3.8414588206941):
            c = st["coverage"][lvl]
            print(f"      cov(t<{lvl:.4g})={c['frac']:.3f} "
                  f"(nominal {c['nominal']:.3f}, {c['dev_sigma']:+.1f}sigma)")
        if cert[p]["t_shared"]:
            ts = np.concatenate(cert[p]["t_shared"]); te = np.concatenate(cert[p]["t_exact"])
            dt = np.abs(ts - te)
            print(f"      CERT shared-J vs per-mock-J: max|dt|={dt.max():.3f} "
                  f"med|dt|={np.median(dt):.3f} (n={len(dt)})")
            out[f"cert_tshared_{p}"] = ts; out[f"cert_texact_{p}"] = te
        out[f"t_{p}"] = t; out[f"z_{p}"] = z
        out[f"cov1_{p}"] = st["coverage"][1.0]["frac"]
        out[f"cov2_{p}"] = st["coverage"][3.8414588206941]["frac"]
        out[f"ks_{p}"] = st["ks_stat"]; out[f"ksp_{p}"] = st["ks_p"]
        _plot_poi(p, t, z, os.path.join(RESDIR, f"wilks_{key}_{p}.png"))
    np.savez(os.path.join(RESDIR, f"wilks_{key}_merged.npz"), **out)
    _plot_summary(pois, stats, os.path.join(RESDIR, f"wilks_{key}_summary.png"))
    print(f"\n[collect] wrote wilks_{key}_merged.npz + per-POI png + wilks_{key}_summary.png")


if __name__ == "__main__":
    main()

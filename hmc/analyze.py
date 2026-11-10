"""analyze.py — diagnostics of an hmc/run_lcdm.py run and its comparison with the CLASS MH reference.

  python hmc/analyze.py [TAG] [--ref hmc/results/ref/class_mh] [--burn K]

Prints acceptance / energy-error / divergence statistics, split R-hat and bulk ESS per parameter
(Stan's multi-chain estimator), the cost per effective sample, and the posterior means and sds
next to the reference chains (shift in reference sigmas, sd ratio); writes
hmc/results/lcdm<TAG>_triangle.png. Sampling draws only (the warmup is dropped), plus the first
--burn sampling trajectories (default 0).
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("tag", nargs="?", default="")
ap.add_argument("--ref", default=os.path.join(HERE, "results", "ref", "class_mh"))
ap.add_argument("--burn", type=int, default=0)
args = ap.parse_args()

d = np.load(os.path.join(HERE, "results", f"lcdm{args.tag}.npz"), allow_pickle=False)
names = [str(n) for n in d["names"]]
nw = int(d["n_warmup"])
th = np.asarray(d["h_theta"])                         # (n_iter, B, D)
n_iter, B, D = th.shape
s0 = nw + args.burn
x = th[s0:]                                           # (n, B, D) sampling draws
if len(x) < 4:
    sys.exit(f"only {len(x)} sampling trajectories so far (warmup {nw}, total {n_iter})")
# derived
H0 = 100.0 * x[..., 2]
Om = (x[..., 0] + x[..., 1]) / x[..., 2] ** 2
cols = {n: x[..., i] for i, n in enumerate(names)}
cols["H0"] = H0; cols["Omega_m"] = Om


def split_chains(c):
    """(n, B) -> (n//2, 2B)."""
    n = (c.shape[0] // 2) * 2
    return np.concatenate([c[:n // 2], c[n // 2:n]], axis=1)


def rhat(c):
    c = split_chains(c); n, m = c.shape
    W = c.var(axis=0, ddof=1).mean(); Bv = n * c.mean(axis=0).var(ddof=1)
    return np.sqrt(((n - 1) / n * W + Bv / n) / W)


def ess(c):
    """Stan's multi-chain bulk ESS (Geyer initial monotone sequence) on split chains."""
    c = split_chains(c); n, m = c.shape
    xc = c - c.mean(axis=0)
    f = np.fft.rfft(xc, n=2 * n, axis=0)
    acov = np.fft.irfft(f * np.conj(f), axis=0)[:n] / n          # (n, m), biased
    W = c.var(axis=0, ddof=1).mean()
    var_plus = (n - 1) / n * W + c.mean(axis=0).var(ddof=1)
    rho = 1.0 - (W - acov.mean(axis=1)) / var_plus
    rho[0] = 1.0
    # Geyer: sum pairs while positive, enforce monotone
    s, prev = 0.0, np.inf
    for t in range(0, n - 1, 2):
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        p = min(p, prev); prev = p
        s += p
    tau = -1.0 + 2.0 * s
    return n * m / max(tau, 1.0 / np.log10(n * m))


a = np.asarray(d["h_accept_prob"]); dH = np.asarray(d["h_dH"]); div = np.asarray(d["h_divergent"])
nst = np.asarray(d["h_n"]); eps = np.asarray(d["h_eps"]); wall = np.asarray(d["h_wall"])
print(f"run lcdm{args.tag}: {B} chains, {n_iter} trajectories ({nw} warmup), using {len(x)} x {B} = {len(x) * B} draws")
print(f"sampling: eps {eps[nw:].min():.3f}-{eps[nw:].max():.3f}, steps/trajectory {nst[nw:].mean():.2f}, "
      f"<accept prob> {a[nw:].mean():.3f}, accepted {np.asarray(d['h_accepted'])[nw:].mean():.3f}, "
      f"divergent {int(div[nw:].sum())}, |dH| median {np.median(np.abs(dH[nw:][np.isfinite(dH[nw:])])):.3g}")
n_grad = int(nst[s0:].sum())
print(f"cost: {int(nst.sum())} batched gradient calls in all ({n_grad} in the draws used), wall {wall.sum() / 3600:.2f} h "
      f"({wall[s0:].sum() / 3600:.2f} h for the draws used, {wall[s0:].sum() / max(n_grad, 1):.0f} s per call of {B})")

ref = None
try:
    import pandas as pd
    pd.options.future.infer_string = False
    from getdist import loadMCSamples
    ref = loadMCSamples(args.ref, settings={"ignore_rows": 0.3})
except Exception as e:
    print(f"(no reference: {e})")
rmap = {"omega_b": "omega_b", "omega_cdm": "omega_cdm", "H0": "H0", "ln10As": "logA", "n_s": "n_s",
        "tau_reion": "tau_reio", "A_planck": "A_planck", "Omega_m": "Omega_m"}
print(f"\n{'param':10s} {'Rhat':>6s} {'ESS':>7s} {'ESS/call':>8s}   {'HMC mean':>11s} {'HMC sd':>10s}" +
      (f"   {'ref mean':>11s} {'ref sd':>10s} {'shift/sd':>8s} {'sd ratio':>8s}" if ref else ""))
for n in ["omega_b", "omega_cdm", "H0", "ln10As", "n_s", "tau_reion", "A_planck", "Omega_m"]:
    c = cols[n]
    e = ess(c)
    line = (f"{n:10s} {rhat(c):6.3f} {e:7.0f} {e / max(n_grad, 1):8.2f}   {c.mean():11.6g} {c.std():10.4g}")
    if ref is not None:
        rm, rs = ref.mean(rmap[n]), ref.std(rmap[n])
        line += f"   {rm:11.6g} {rs:10.4g} {(c.mean() - rm) / rs:8.3f} {c.std() / rs:8.3f}"
    print(line)
if ref is not None:
    print(f"(reference: {args.ref}, R-1 from its .progress if present; MH chains after 30% burn-in)")

try:
    from getdist import MCSamples, plots
    import matplotlib
    matplotlib.use("Agg")
    pn = ["omega_b", "omega_cdm", "H0", "ln10As", "n_s", "tau_reion"]
    lab = [r"\omega_b", r"\omega_{cdm}", "H_0", r"\ln(10^{10}A_s)", "n_s", r"\tau"]
    hs = MCSamples(samples=np.stack([cols[p].ravel() for p in pn], axis=1), names=pn, labels=lab,
                   label=f"HMC, ABCMB-k ({len(x) * B} draws)")
    g = plots.get_subplot_plotter()
    if ref is not None:
        for p_h, p_r in (("ln10As", "logA"), ("tau_reion", "tau_reio")):
            ref.addDerived(ref.getParams().__dict__[p_r], name=p_h, label=lab[pn.index(p_h)])
        ref.label = "MH, CLASS (reference)"
        g.triangle_plot([ref, hs], pn, filled=[True, False], legend_loc="upper right")
    else:
        g.triangle_plot([hs], pn, filled=True)
    out = os.path.join(HERE, "results", f"lcdm{args.tag}_triangle.png")
    g.export(out)
    print(f"\nwrote {out}")
except Exception as e:
    print(f"(no triangle plot: {e})")

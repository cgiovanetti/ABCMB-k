"""tune_inner.py — OFFLINE tuner/validator for plik_full's inner nuisance profile.

Loads scan/results/clsref_lcdm.npz (theory Cls for a spread of cosmologies, saved once
by save_clsref.py) and, WITHOUT running ABCMB, checks that the gradient-only u-space
BFGS inner profile (jax.scipy.optimize.minimize) matches a scipy L-BFGS-B ground truth
across all cosmologies. The ground truth optimises the well-conditioned SCALED-z BOUNDED
problem (nu = start + scale*z, bounds [zlo,zhi]) -- the reliable reference (physical-nu
scipy is ill-conditioned). Fast (~2-4 min): clipy only.

Sweeps the BFGS maxiter (TI_MAXITS). Reports the worst chi^2 gap vs scipy + vmap/timing.
"""
import os, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize

MAXITS = [int(x) for x in os.environ.get("TI_MAXITS", "400,600,800").split(",")]


def main():
    print(f"jax devices: {jax.devices()}  MAXITS={MAXITS}", flush=True)
    from scan.plik_full import PlikFull, FLOAT_NAMES
    d = np.load(os.path.join(os.path.dirname(__file__), "results", "clsref_lcdm.npz"),
                allow_pickle=True)
    cls2d = jnp.asarray(d["cls2d"]); labels = [str(s) for s in d["labels"]]
    B = cls2d.shape[0]
    print(f"loaded clsref: {cls2d.shape}  labels={labels}", flush=True)

    pl = PlikFull()
    P = len(FLOAT_NAMES)
    start = jnp.asarray(pl.start); scale = jnp.asarray(pl.scale)
    lo = np.asarray(pl.lo); hi = np.asarray(pl.hi)

    # ---- scipy ground truth: SCALED-z BOUNDED (well-conditioned, reliable ref) ----
    obj_z = lambda z, cls: pl.penalized_chi2(cls, start + scale * z)
    vg = jax.jit(jax.value_and_grad(obj_z))
    zlo = (lo - np.asarray(pl.start)) / np.asarray(pl.scale)
    zhi = (hi - np.asarray(pl.start)) / np.asarray(pl.scale)
    bounds = [(None if not np.isfinite(a) else a, None if not np.isfinite(b) else b)
              for a, b in zip(zlo, zhi)]
    chi2_sci = np.empty(B)
    t0 = time.perf_counter()
    for b in range(B):
        clsb = cls2d[b]
        def vgb(z):
            v, g = vg(jnp.asarray(z), clsb)
            return float(v), np.asarray(g, float)
        res = minimize(vgb, np.zeros(P), jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 800, "ftol": 1e-13, "gtol": 1e-10})
        chi2_sci[b] = res.fun
    print(f"scipy ground-truth chi2: min={chi2_sci.min():.2f} max={chi2_sci.max():.2f} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- sweep BFGS maxiter ----
    print(f"\n{'maxit':>6s} {'worst_gap':>10s} {'mean_gap':>9s} {'wall_s':>8s}", flush=True)
    best = None
    for mit in MAXITS:
        prof = jax.jit(lambda cls, _m=mit: pl.profile(cls, maxit=_m))
        t0 = time.perf_counter()
        gaps = np.array([float(prof(cls2d[b])[0]) - chi2_sci[b] for b in range(B)])
        tc = time.perf_counter() - t0
        wg = float(np.max(np.abs(gaps))); mg = float(np.mean(np.abs(gaps)))
        flag = ""
        if best is None or wg < best[0]:
            best = (wg, mit); flag = " *"
        print(f"{mit:6d} {wg:10.4f} {mg:9.4f} {tc:8.1f}{flag}", flush=True)

    wg, mit = best
    print(f"\nBEST: maxit={mit} worst_gap={wg:.4f} chi^2  PASS(<0.1): {wg<0.1}", flush=True)

    # ---- per-cosmology detail + vmap/timing for the winner ----
    prof1 = jax.jit(lambda cls: pl.profile(cls, maxit=mit))
    print(f"\n{'label':16s} {'scipy':>11s} {'uBFGS':>11s} {'gap':>9s}", flush=True)
    for b in range(B):
        c2 = float(prof1(cls2d[b])[0])
        print(f"{labels[b]:16s} {chi2_sci[b]:11.4f} {c2:11.4f} {c2-chi2_sci[b]:+9.4f}", flush=True)

    profB = jax.jit(lambda c: jax.vmap(lambda x: pl.profile(x, maxit=mit))(c))
    t0 = time.perf_counter(); cB, _ = profB(cls2d); jax.block_until_ready(cB)
    tc = time.perf_counter() - t0
    t0 = time.perf_counter(); cB2, _ = profB(cls2d); jax.block_until_ready(cB2)
    tw = time.perf_counter() - t0
    per = np.array([float(prof1(cls2d[b])[0]) for b in range(B)])
    print(f"\nvmap compile {tc:.1f}s; warm {tw:.3f}s B={B} ({tw/B*1000:.1f} ms/cosmo); "
          f"vmap-vs-loop max|d|={float(np.max(np.abs(np.asarray(cB)-per))):.2e}", flush=True)


if __name__ == "__main__":
    import pandas as pd
    pd.options.future.infer_string = False
    main()

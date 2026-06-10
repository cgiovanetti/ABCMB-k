"""profile_batched.py — batched frequentist profile of n_s (the recommended design).

This is the production-shaped Option-B profiler: instead of scipy driving ONE
cosmology-batch (B=7) per optimizer step serially per POI (scan/profile_opt.py,
which both under-utilised the GPU and OOM'd on a host-memory leak), it runs ALL
parameter-of-interest (POI = n_s) optimisations IN LOCKSTEP. Each iteration is a
SINGLE call_batched over every POI's finite-difference stencil at once, so the
GPU sees a healthy batch and the ~55 s fixed per-call overhead amortises.

Optimiser: damped full-Hessian Newton over the cosmological nuisances
{h, omega_b, omega_cdm} in sigma-scaled coordinates. Per POI a 19-point central
stencil (centre + 6 axial +-, + 12 mixed +-+-) yields the gradient AND the full
symmetric 3x3 Hessian, so Newton walks the h-omega_cdm degeneracy directly
(affine-invariant -> no 100*theta_star reparametrisation needed). The A_s /
A_planck overall amplitude is profiled ANALYTICALLY (pl.profile_amplitude) so it
is not an optimiser dimension; tau is fixed (the prior-profiled tau is a cheap
add-on once the methodology is shown).

Batch sizing (HARD GPU constraint): with one massive neutrino the per-device peak
is ~0.60 GB * B (CLAUDE.md), so on one 80 GB A100 keep B <~ 110. 5 POI * 19 = 95
fits with margin and is a SINGLE compiled shape. For finer/ wider grids, shard
across the 4 GPUs of a node (call_batched(shard=True) auto-partitions B) -- that
is the production path; this 1-GPU run is the methodology demo.

Per-iteration it logs host VmRSS to confirm the main.py CPU-wrapper-cache fix
(no more ~0.12 GB/call leak). Checkpoints results/profile_batched_ns.npz each
iteration. Run via srun (1 GPU), PYTHONPATH=$(pwd).
Env: PB_LMAX(2508), PB_POI_LO/HI/N(0.955/0.975/5), PB_ITERS(4), PB_DELTA(0.4).
"""
import os, gc, time, itertools
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
from scan.plik_lite import PlikLite
from scan.profile_opt import build, NUIS, CENTER, SIGMA, NDIM

LMAX = int(os.environ.get("PB_LMAX", 2508))
NS_LO = float(os.environ.get("PB_POI_LO", 0.955))
NS_HI = float(os.environ.get("PB_POI_HI", 0.975))
NS_N = int(os.environ.get("PB_POI_N", 5))
ITERS = int(os.environ.get("PB_ITERS", 4))
DELTA = float(os.environ.get("PB_DELTA", 0.4))   # sigma-scaled FD step
XBOX = 4.0                                        # clamp |x| <= 4 sigma
STEP_CAP = 1.5                                    # clamp |dx_i| per Newton step
LAM = 1e-3                                        # Levenberg damping (PD safety)
BIG = 1e6

pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)

# --- stencil layout (NDIM=3): centre, 6 axial, 12 mixed = 19 points ----------
# Each entry is a (NDIM,) offset in units of DELTA.
_AX = []
for i in range(NDIM):
    e = np.zeros(NDIM); e[i] = 1.0
    _AX.append(("ax+", i, e)); _AX.append(("ax-", i, -e))
_MIX = []
for i, j in itertools.combinations(range(NDIM), 2):
    for si in (+1, -1):
        for sj in (+1, -1):
            o = np.zeros(NDIM); o[i] = si; o[j] = sj
            _MIX.append(("mix", (i, j, si, sj), o))
OFFSETS = [("c", None, np.zeros(NDIM))] + _AX + _MIX
NSTEN = len(OFFSETS)            # 19


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def eval_all(xs, ns_grid):
    """xs: (NS_N, NDIM) sigma-scaled centres. Build every POI's 19-pt stencil,
    run ONE call_batched, return (NS_N, NSTEN) amplitude-profiled chi^2."""
    batch = []
    for p in range(NS_N):
        for (_, _, off) in OFFSETS:
            phys = CENTER + SIGMA * (xs[p] + DELTA * off)
            batch.append(build(phys, ns_grid[p]))
    out = model.call_batched(batch, shard=True)            # B = NS_N*NSTEN
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    c2 = np.asarray(pl.profile_amplitude(m0)[0], dtype=float)   # (B,)
    c2 = np.where(np.isfinite(c2), c2, BIG)
    return c2.reshape(NS_N, NSTEN)


def grad_hess(c2_row):
    """c2_row: (NSTEN,) chi^2 on one POI's stencil -> (f0, g(NDIM), H(NDIM,NDIM))."""
    f = {("c", None): c2_row[0]}
    k = 1
    for (tag, idx, _) in OFFSETS[1:]:
        f[(tag, idx)] = c2_row[k]; k += 1
    f0 = f[("c", None)]
    g = np.zeros(NDIM); H = np.zeros((NDIM, NDIM))
    for i in range(NDIM):
        fp = f[("ax+", i)]; fm = f[("ax-", i)]
        g[i] = (fp - fm) / (2 * DELTA)
        H[i, i] = (fp + fm - 2 * f0) / (DELTA ** 2)
    for i, j in itertools.combinations(range(NDIM), 2):
        fpp = f[("mix", (i, j, +1, +1))]; fpm = f[("mix", (i, j, +1, -1))]
        fmp = f[("mix", (i, j, -1, +1))]; fmm = f[("mix", (i, j, -1, -1))]
        H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * DELTA ** 2)
    return f0, g, H


def newton_step(g, H):
    Hd = H + LAM * np.eye(NDIM)
    try:
        dx = -np.linalg.solve(Hd, g)
    except np.linalg.LinAlgError:
        dx = -g                                   # fall back to gradient descent
    # if Hessian not positive-definite, Newton can point uphill -> guard
    if not np.all(np.linalg.eigvalsh(Hd) > 0) or np.dot(g, dx) > 0:
        dx = -g / (np.linalg.norm(g) + 1e-12)     # normalised descent
    return np.clip(dx, -STEP_CAP, STEP_CAP)


def interval(x, y, level=1.0):
    y = np.asarray(y, float); x = np.asarray(x, float)
    m = np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    x, y = x[m], y[m]; i = int(np.argmin(y)); t = y[i] + level
    def cross(up):
        rng = range(i, len(x) - 1) if up else range(i, 0, -1)
        for q in rng:
            r = q + 1 if up else q - 1
            if (y[q] - t) * (y[r] - t) <= 0:
                fr = (t - y[q]) / (y[r] - y[q] + 1e-30); return x[q] + fr * (x[r] - x[q])
        return np.nan
    return cross(False), x[i], cross(True)


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  POI={NS_N}  B={NS_N*NSTEN}  "
          f"iters={ITERS}  delta={DELTA}", flush=True)
    ns_grid = np.linspace(NS_LO, NS_HI, NS_N)
    print(f"POI n_s grid: {ns_grid}", flush=True)

    xs = np.zeros((NS_N, NDIM))                   # all start at Planck centre
    best_chi2 = np.full(NS_N, np.inf); best_x = xs.copy()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True); npz = os.path.join(out, "profile_batched_ns.npz")

    print(f"  RSS start: {rss_gb():.2f} GB", flush=True)
    t_all = time.perf_counter()
    for it in range(ITERS):
        t0 = time.perf_counter()
        c2 = eval_all(xs, ns_grid)                # (NS_N, NSTEN), one call_batched
        f0 = c2[:, 0].copy()
        for p in range(NS_N):
            if f0[p] < best_chi2[p]:
                best_chi2[p] = f0[p]; best_x[p] = xs[p].copy()
            _, g, H = grad_hess(c2[p])
            xs[p] = np.clip(xs[p] + newton_step(g, H), -XBOX, XBOX)
        dt = time.perf_counter() - t0
        print(f"  iter {it}: chi2(centres)={np.array2string(f0, precision=2)} "
              f"min={best_chi2.min():.2f}  ({dt:.0f}s, RSS={rss_gb():.2f} GB)",
              flush=True)
        np.savez(npz, ns=ns_grid, chi2=best_chi2, xstar=best_x,
                 center=CENTER, sigma=SIGMA, iter=it, delta=DELTA)
        gc.collect()

    # final clean evaluation at the converged centres (no stencil bias)
    c2 = eval_all(best_x, ns_grid)
    final = c2[:, 0]
    for p in range(NS_N):
        if final[p] < best_chi2[p]:
            best_chi2[p] = final[p]
    np.savez(npz, ns=ns_grid, chi2=best_chi2, xstar=best_x,
             center=CENTER, sigma=SIGMA, iter=ITERS, delta=DELTA)
    print(f"\ntotal {time.perf_counter()-t_all:.0f}s -> {npz}", flush=True)

    j = int(np.nanargmin(best_chi2))
    lo, mid, hi = interval(ns_grid, best_chi2, 1.0)
    print(f"min chi2 = {best_chi2[j]:.2f} at n_s={ns_grid[j]:.4f}", flush=True)
    print("profiled chi2 curve:", flush=True)
    for p in range(NS_N):
        ph = CENTER + SIGMA * best_x[p]
        print(f"  n_s={ns_grid[p]:.4f}: chi2={best_chi2[p]:.3f} dchi2={best_chi2[p]-best_chi2[j]:+.3f}"
              f"  h={ph[0]:.4f} ob={ph[1]:.5f} ocdm={ph[2]:.5f}", flush=True)
    print(f"profiled n_s = {mid:.4f}  1sigma [{lo:.4f}, {hi:.4f}]  (+/-{(hi-lo)/2:.4f})",
          flush=True)
    print(f"Planck18 n_s = 0.9649 +/- 0.0042", flush=True)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        d = best_chi2 - best_chi2[j]
        fig, ax = plt.subplots(figsize=(6, 4.4))
        ax.plot(ns_grid, d, "o-")
        for lv, c in [(1, "g"), (4, "orange")]:
            ax.axhline(lv, ls="--", lw=0.8, color=c)
        ax.axvline(0.9649, ls=":", color="gray", lw=0.8, label="Planck n_s")
        ax.set_xlabel("n_s"); ax.set_ylabel(r"$\Delta\chi^2$ (profiled over h,$\omega_b$,$\omega_c$,$A_s$)")
        ax.set_ylim(0, max(5, float(np.nanmax(d)) * 1.1)); ax.legend()
        ax.set_title("batched-Newton profile of $n_s$ vs Planck plik-lite")
        fig.tight_layout(); png = npz.replace(".npz", ".png"); fig.savefig(png, dpi=120)
        print(f"saved -> {png}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)


if __name__ == "__main__":
    main()

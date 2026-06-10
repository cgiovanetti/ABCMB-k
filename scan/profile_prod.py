"""profile_prod.py — production LCDM frequentist profile vs Planck plik-lite.

Profiles each requested LCDM parameter (POI) in turn: for a grid of POI values it
MINIMIZES chi^2 over the other 5 LCDM parameters with damped full-Hessian Newton,
giving the profile-likelihood Delta-chi^2(POI) and its 1/2/3-sigma intervals.

Parameter handling (the "ironed out" production setup):
  * 6 LCDM params: h, omega_b, omega_cdm, n_s, ln10As (A_s=exp(ln10As)/1e10),
    tau_reion.  Whichever is the POI is held fixed on its grid; the other 5 are
    Newton nuisances in sigma-scaled coords -> a uniform 5-D / 51-point central
    stencil for EVERY POI, so all profiles share ONE compiled HLO.
  * A_s is a REAL optimizer dimension (ln10As) -> exact w.r.t. lensing
    non-linearity (the analytic-amplitude shortcut is only feasibility-grade).
  * A_planck is the pure multiplicative calibration -> profiled ANALYTICALLY with
    its Gaussian prior N(1, 0.0025) every evaluation (pl.profile_A, exact).
  * tau_reion carries its external Gaussian prior N(0.0544, 0.0073), added to chi^2
    at each point (whether tau is the POI or a nuisance). plik-lite alone cannot
    break A_s-tau, so the prior is what makes that subspace well-posed.

Optimizer: per POI value a 51-pt central stencil yields the gradient AND the full
symmetric 5x5 Hessian; Newton (Levenberg-damped, step-clamped, box-clamped) walks
all the LCDM degeneracies directly (affine-invariant -> no theta_star
reparametrization). Warm-started across the POI grid. Converges in a few steps.

Batched & sharded: every Newton iteration is ONE (or a few) call_batched(shard=True)
over the whole POI grid's stencils at once (B = chunk*51), auto-partitioned across
the node's GPUs. POI-chunked to a fixed batch so memory stays bounded and the HLO
is reused (B_cap from PROD_BCAP or n_dev*PROD_BLOCAL).

Multi-node (ONE job, NOT an array): each node owns POIs[procid::nprocs]
(SLURM_PROCID/SLURM_NPROCS) and shards its POIs over its own 4 GPUs -- no
cross-node traffic. Resumable: a POI whose results/profile_prod_<poi>.npz exists
with done=True is skipped.

Env: PROD_POIS (csv; default all 6), PROD_NPTS(13), PROD_NSIG(3.0), PROD_ITERS(6),
PROD_DELTA(0.4), PROD_LMAX(2508), PROD_BLOCAL(95), PROD_BCAP(0=auto), PROD_RESUME(1).
Run via srun, PYTHONPATH=$(pwd).
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

# ---------------- config ----------------
LMAX   = int(os.environ.get("PROD_LMAX", 2508))
NPTS   = int(os.environ.get("PROD_NPTS", 13))
NSIG   = float(os.environ.get("PROD_NSIG", 3.0))
ITERS  = int(os.environ.get("PROD_ITERS", 6))
DELTA  = float(os.environ.get("PROD_DELTA", 0.4))     # sigma-scaled FD step
BLOCAL = int(os.environ.get("PROD_BLOCAL", 95))       # target cosmologies/device
BCAP_ENV = int(os.environ.get("PROD_BCAP", 0))        # 0 => auto (n_dev*BLOCAL)
RESUME = os.environ.get("PROD_RESUME", "1") != "0"
XBOX, STEP_CAP, LAM, BIG = 4.0, 1.5, 1e-3, 1e6

# LCDM params: name -> (center, sigma, gauss_prior or None). ln10As: A_s=exp/1e10.
LCDM = {
    'h':         (0.6736,  0.0054,  None),
    'omega_b':   (0.02237, 0.00015, None),
    'omega_cdm': (0.1200,  0.0012,  None),
    'n_s':       (0.9649,  0.0042,  None),
    'ln10As':    (3.044,   0.014,   None),
    'tau_reion': (0.0544,  0.0073,  (0.0544, 0.0073)),
}
LCDM_ORDER = ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10As', 'tau_reion']
# non-LCDM fixed inputs (tau & A_s are now LCDM params, so NOT here)
FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}

POIS = os.environ.get("PROD_POIS", ",".join(LCDM_ORDER)).split(",")
POIS = [p.strip() for p in POIS if p.strip() in LCDM]
RANK = int(os.environ.get("SLURM_PROCID", 0))
NPROC = int(os.environ.get("SLURM_NPROCS", 1))

pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)
try:
    NDEV = len(jax.devices('gpu'))
except Exception:
    NDEV = 1
BCAP = BCAP_ENV if BCAP_ENV > 0 else max(1, NDEV) * BLOCAL


def build(vals):
    """vals: dict of LCDM param values -> full ABCMB param dict."""
    p = dict(FIXED)
    for k in ('h', 'omega_b', 'omega_cdm', 'n_s'):
        p[k] = float(vals[k])
    p['A_s'] = float(np.exp(vals['ln10As']) / 1e10)
    p['tau_reion'] = float(max(vals['tau_reion'], 5e-3))   # keep positive
    return p


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


# ---- 5-D stencil offsets (centre, 2N axial, 4*C(N,2) mixed) ----
def make_offsets(ndim):
    offs = [np.zeros(ndim)]
    for i in range(ndim):
        e = np.zeros(ndim); e[i] = 1.0
        offs.append(e.copy()); offs.append(-e)
    for i, j in itertools.combinations(range(ndim), 2):
        for si in (+1, -1):
            for sj in (+1, -1):
                o = np.zeros(ndim); o[i] = si; o[j] = sj
                offs.append(o)
    return np.array(offs)


def chi2_call(valdicts):
    """Run ONE call_batched over a list of LCDM valdicts -> (len,) total chi^2
    (A_planck profiled analytically w/ prior + tau prior)."""
    batch = [build(v) for v in valdicts]
    out = model.call_batched(batch, shard=True)
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    data = np.asarray(pl.profile_A(m0, with_prior=True)[0], dtype=float)
    tau = np.array([v['tau_reion'] for v in valdicts], dtype=float)
    c0, s0 = LCDM['tau_reion'][2]
    total = data + ((tau - c0) / s0) ** 2
    return np.where(np.isfinite(total), total, BIG)


def eval_grid(poi, poi_grid, xs, nuis, offs):
    """For each POI value build its 51-pt nuisance stencil, evaluate ALL in
    memory-bounded chunks -> (NPTS, NSTEN) chi^2."""
    nd = len(nuis); nsten = len(offs)
    cen = np.array([LCDM[n][0] for n in nuis]); sig = np.array([LCDM[n][1] for n in nuis])
    # flat list of valdicts, POI-major (stencil contiguous per POI)
    vds = []
    for p in range(len(poi_grid)):
        for s in range(nsten):
            phys = cen + sig * (xs[p] + DELTA * offs[s])
            vd = {nuis[i]: phys[i] for i in range(nd)}
            vd[poi] = poi_grid[p]
            vds.append(vd)
    poi_per_call = max(1, BCAP // nsten)
    chunk_pois = poi_per_call * nsten
    out = np.empty(len(vds), dtype=float)
    i = 0
    while i < len(vds):
        block = vds[i:i + chunk_pois]
        pad = chunk_pois - len(block)
        if pad:                                   # pad to fixed B (reuse HLO)
            block = block + [block[-1]] * pad
        c = chi2_call(block)
        out[i:i + chunk_pois - pad] = c[:chunk_pois - pad]
        i += chunk_pois
        gc.collect()
    return out.reshape(len(poi_grid), nsten)


def grad_hess(c2, offs, nd):
    f0 = c2[0]
    g = np.zeros(nd); H = np.zeros((nd, nd))
    # axial entries are at indices 1..2N in (i,+),(i,-) order
    fp = c2[1:1 + 2 * nd:2]; fm = c2[2:2 + 2 * nd:2]
    g = (fp - fm) / (2 * DELTA)
    np.fill_diagonal(H, (fp + fm - 2 * f0) / DELTA ** 2)
    k = 1 + 2 * nd
    for i, j in itertools.combinations(range(nd), 2):
        fpp, fpm, fmp, fmm = c2[k], c2[k + 1], c2[k + 2], c2[k + 3]; k += 4
        H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4 * DELTA ** 2)
    return f0, g, H


def newton_step(g, H, nd):
    Hd = H + LAM * np.eye(nd)
    try:
        dx = -np.linalg.solve(Hd, g)
    except np.linalg.LinAlgError:
        dx = -g
    if not np.all(np.linalg.eigvalsh(Hd) > 0) or np.dot(g, dx) > 0:
        dx = -g / (np.linalg.norm(g) + 1e-12)
    return np.clip(dx, -STEP_CAP, STEP_CAP)


def interval(x, y, level):
    """Delta-chi2=level crossings via shape-preserving (PCHIP) interpolation of
    the profile, then dense-grid root finding. The chi2 is DETERMINISTIC and
    smooth (in-batch noise floor MEASURED = 0, scan/noise_floor.py), so PCHIP
    crossings are sub-grid accurate -- far better than the old linear interp of
    a 0.5-sigma grid. Returns (lo, min_x, hi)."""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan, np.nan
    x, y = x[m], y[m]; o = np.argsort(x); x, y = x[o], y[o]
    try:
        from scipy.interpolate import PchipInterpolator
        p = PchipInterpolator(x, y - y.min())
        xs = np.linspace(x[0], x[-1], 40001); ys = p(xs)
    except Exception:                                   # fallback: dense-linear
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
    """Symmetric Gaussian sigma from a parabola fit to the points with
    Delta-chi2 <= 4 of the minimum (curvature -> sigma = 1/sqrt(0.5*d2chi2/dx2))."""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan
    d = y - y.min(); sel = d <= 4.0
    if sel.sum() < 3:
        sel = np.argsort(d)[:max(3, len(x) // 2)]
    a = np.polyfit(x[sel], y[sel], 2)[0]                 # chi2 ~ a x^2 + ...
    return np.nan if a <= 0 else 1.0 / np.sqrt(2.0 * a)


def profile_one(poi, outdir):
    nuis = [p for p in LCDM_ORDER if p != poi]; nd = len(nuis)
    offs = make_offsets(nd); nsten = len(offs)
    c, s = LCDM[poi][0], LCDM[poi][1]
    poi_grid = np.linspace(c - NSIG * s, c + NSIG * s, NPTS)
    npz = os.path.join(outdir, f"profile_prod_{poi}.npz")
    if RESUME and os.path.exists(npz):
        try:
            if bool(np.load(npz, allow_pickle=True)["done"]):
                print(f"[{poi}] resume: already done, skip", flush=True); return
        except Exception:
            pass
    print(f"[{poi}] grid {NPTS}pts in [{poi_grid[0]:.5f},{poi_grid[-1]:.5f}] "
          f"nuis={nuis} B={ (BCAP//nsten)*nsten } stencil={nsten}", flush=True)

    xs = np.zeros((NPTS, nd)); best = np.full(NPTS, np.inf); bestx = xs.copy()
    t0 = time.perf_counter()
    for it in range(ITERS):
        ti = time.perf_counter()
        c2 = eval_grid(poi, poi_grid, xs, nuis, offs)
        f0 = c2[:, 0].copy()
        for p in range(NPTS):
            if f0[p] < best[p]:
                best[p] = f0[p]; bestx[p] = xs[p].copy()
            _, g, H = grad_hess(c2[p], offs, nd)
            xs[p] = np.clip(xs[p] + newton_step(g, H, nd), -XBOX, XBOX)
        print(f"  [{poi}] iter {it}: min={best.min():.2f} "
              f"f0={np.array2string(f0, precision=1, max_line_width=200)} "
              f"({time.perf_counter()-ti:.0f}s RSS={rss_gb():.1f}GB)", flush=True)
        np.savez(npz, poi=poi, poi_grid=poi_grid, chi2=best, xstar=bestx,
                 nuis=np.array(nuis), iter=it, done=False)
    # clean final evaluation at converged nuisances
    c2 = eval_grid(poi, poi_grid, bestx, nuis, offs)
    best = np.minimum(best, c2[:, 0])
    lo1, mid, hi1 = interval(poi_grid, best, 1.0)
    lo2, _, hi2 = interval(poi_grid, best, 4.0)
    sig_p = sigma_parabola(poi_grid, best)               # Gaussian cross-check
    np.savez(npz, poi=poi, poi_grid=poi_grid, chi2=best, xstar=bestx,
             nuis=np.array(nuis), iter=ITERS, done=True,
             sigma1=np.array([lo1, mid, hi1]), sigma2=np.array([lo2, hi2]),
             sigma_parab=sig_p)
    j = int(np.nanargmin(best))
    print(f"[{poi}] DONE ({time.perf_counter()-t0:.0f}s): min chi2={best[j]:.2f} "
          f"at {poi}={poi_grid[j]:.5f}; 1sigma=[{lo1:.5f},{hi1:.5f}] "
          f"(PCHIP +/-{(hi1-lo1)/2:.5f}; parab sigma={sig_p:.5f}); "
          f"2sigma=[{lo2:.5f},{hi2:.5f}] -> {npz}", flush=True)
    _plot(poi, poi_grid, best, npz)


def _plot(poi, grid, chi2, npz):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        d = chi2 - np.nanmin(chi2)
        fig, ax = plt.subplots(figsize=(6, 4.3))
        ax.plot(grid, d, "o-")
        for lv, col in [(1, "g"), (4, "orange"), (9, "r")]:
            ax.axhline(lv, ls="--", lw=0.8, color=col)
        ax.axvline(LCDM[poi][0], ls=":", color="gray", lw=0.8, label=f"Planck {poi}")
        ax.set_xlabel(poi); ax.set_ylabel(r"$\Delta\chi^2$ (profiled)")
        ax.set_ylim(0, max(10, float(np.nanmax(d)) * 1.05)); ax.legend()
        ax.set_title(f"plik-lite profile of {poi}")
        fig.tight_layout(); fig.savefig(npz.replace(".npz", ".png"), dpi=120)
    except Exception as e:
        print(f"[{poi}] plot skipped: {e}", flush=True)


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    mine = POIS[RANK::NPROC]
    print(f"rank {RANK}/{NPROC} devices={jax.devices()} BCAP={BCAP} "
          f"POIs={POIS} -> mine={mine}", flush=True)
    for poi in mine:
        profile_one(poi, outdir)
    print(f"rank {RANK}: all POIs done", flush=True)


if __name__ == "__main__":
    main()

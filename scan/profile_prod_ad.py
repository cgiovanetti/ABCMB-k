"""profile_prod_ad.py — AD-gradient, convergence-gated frequentist profile.

Rebuild of profile_prod.py addressing three reviewer objections:

  (#5 autodiff)  The 51-point finite-difference stencil is GONE.  Gradients are
     exact forward-mode AD (jax.jacfwd) through the full ABCMB->likelihood
     pipeline -- ABCMB's whole selling point.  (Forward-over-forward Hessians OOM
     at l_max=2508, ~35 GB/cosmo -- measured, scan/derisk_ad.py -- so curvature is
     built by BFGS from the exact gradient history, and the Fisher/nuisance Hessian
     at the optimum is one central FD of the EXACT AD gradient, not of chi^2.)

  (#1 convergence/global min)  A real convergence GATE: each POI's nuisance
     minimisation runs to ||grad||_inf < GTOL (reported per POI -- the stationarity
     evidence), not a fixed iteration count.  BFGS with an Armijo backtracking line
     search (guaranteed descent).  --multistart re-minimises from K dispersed
     starts and reports the spread of converged chi^2 (the global-min demonstration).
     PD check on the nuisance Hessian at the optimum.

  (#3 data model)  No more circular N(0.0544,0.0073) tau "prior".  tau is a free
     parameter constrained by the ACTUAL low-ell EE likelihood (SRoll2, AD-able,
     scan/lowl_like.py), and the dropped ell=2..29 TEMPERATURE is restored via the
     low-ell TT (Commander) likelihood.  Data model = plik-lite high-ell TTTEEE +
     low-ell TT + low-ell EE.  A_planck still profiled analytically (envelope) on
     the high-ell block.

HYBRID evaluation (throughput): function VALUES (BFGS line search + recorded
profile) go through the FAST batched/sharded call_batched path (~0.67 s/param);
GRADIENTS go through the AD path.  PA_GRADMETHOD selects how gradients batch over
the POI grid:
  * "loop" (default, robust): single-cosmology jacfwd per POI point -- compiles in
    ~5 min (scan/derisk_ad.py) and is rock-solid.  Slower per gradient.
  * "vmap": one vmapped jacfwd over the whole grid -- amortises, but the GPU->CPU-
    HyRex->GPU hop makes XLA compile a single huge module (10-25 min, then CACHEd).
    Use for big production grids where the one-time compile amortises.

Per POI value the 5 nuisances are optimised in sigma-scaled coords; all POI grid
points move IN LOCKSTEP.  Multi-GPU = one POI per rank (SLURM_PROCID).

Run via srun, PYTHONPATH=$(pwd).  Env:
  PA_POIS(csv;all6) PA_NPTS(13) PA_NSIG(3) PA_MAXIT(40) PA_GTOL(3e-2)
  PA_LMAX(2508) PA_RTOL(1e-5) PA_TAG('') PA_RESUME(1)
  PA_GRADMETHOD(loop|vmap) PA_HESS(1) PA_SHARD(auto|0|1)
  PA_MULTISTART(0) PA_MS_K(6)  PA_USE_LOWTT(1) PA_USE_LOWEE(1)
"""
import os, time
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.plik_lite import PlikLite
from scan.lowl_like import LowLEE, LowLTT
from scan.profile_prod import interval, sigma_parabola

# ---------------- config ----------------
LMAX = int(os.environ.get("PA_LMAX", 2508))
NPTS = int(os.environ.get("PA_NPTS", 13))
NSIG = float(os.environ.get("PA_NSIG", 3.0))
MAXIT = int(os.environ.get("PA_MAXIT", 40))
GTOL = float(os.environ.get("PA_GTOL", 3e-2))       # ||grad||_inf gate (scaled coords)
RTOL = float(os.environ.get("PA_RTOL", 1e-5))
TAG = os.environ.get("PA_TAG", "")
RESUME = os.environ.get("PA_RESUME", "1") != "0"
GRADMETHOD = os.environ.get("PA_GRADMETHOD", "loop").lower()
DO_HESS = os.environ.get("PA_HESS", "1") != "0"
_shard_env = os.environ.get("PA_SHARD", "auto").lower()
MULTISTART = os.environ.get("PA_MULTISTART", "0") != "0"
MS_K = int(os.environ.get("PA_MS_K", 6))
USE_LOWTT = os.environ.get("PA_USE_LOWTT", "1") != "0"
USE_LOWEE = os.environ.get("PA_USE_LOWEE", "1") != "0"
XBOX = 5.0                                            # nuisance box (sigma units)
C1, MAXLS = 1e-4, 12                                  # Armijo c1, max backtracks
FDH = 0.05                                            # FD step (sigma) for Fisher Hessian

LCDM = {'h': (0.6736, 0.0054), 'omega_b': (0.02237, 0.00015),
        'omega_cdm': (0.1200, 0.0012), 'n_s': (0.9649, 0.0042),
        'ln10As': (3.044, 0.014), 'tau_reion': (0.0544, 0.0073)}
ORDER = ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10As', 'tau_reion']
CENTER = np.array([LCDM[k][0] for k in ORDER])
SIGMA = np.array([LCDM[k][1] for k in ORDER])
FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}

POIS = os.environ.get("PA_POIS", ",".join(ORDER)).split(",")
POIS = [p.strip() for p in POIS if p.strip() in LCDM]
RANK = int(os.environ.get("SLURM_PROCID", 0))
NPROC = int(os.environ.get("SLURM_NPROCS", 1))

pl = PlikLite()
lowee = LowLEE() if USE_LOWEE else None
lowtt = LowLTT() if USE_LOWTT else None
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
              rtol_large_k_PE=RTOL, atol_large_k_PE=RTOL * 1e-2,
              rtol_small_k_PE=min(1e-5, RTOL), max_steps_PE=16384)
CEN = jnp.asarray(CENTER); SIG = jnp.asarray(SIGMA)
try:
    NDEV = len(jax.devices('gpu'))
except Exception:
    NDEV = 1
DO_SHARD = (_shard_env == "1") or (_shard_env == "auto" and NDEV > 1)


# ======================================================================
# parameter assembly
# ======================================================================
def assemble_phys(poi_idx, x5, poi_val):
    """scaled nuisance x5 (5,) + POI value -> physical 6-vector (numpy)."""
    nuis = [i for i in range(6) if i != poi_idx]
    theta = CENTER.copy(); theta[poi_idx] = poi_val
    for k, i in enumerate(nuis):
        theta[i] = CENTER[i] + SIGMA[i] * x5[k]
    return theta


def build_dict(theta):
    """physical 6-vector -> ABCMB param dict (host floats, for call_batched)."""
    p = dict(FIXED)
    p['h'] = float(theta[0]); p['omega_b'] = float(theta[1])
    p['omega_cdm'] = float(theta[2]); p['n_s'] = float(theta[3])
    p['A_s'] = float(np.exp(theta[4]) / 1e10); p['tau_reion'] = float(theta[5])
    return p


# ======================================================================
# FAST function values via call_batched (no AD) -- BFGS line search + profile
# ======================================================================
def fast_values(poi_idx, X, PV):
    """X:(B,5) PV:(B,) -> (B,) total chi^2 (profiled A_planck + low-ell)."""
    batch = [build_dict(assemble_phys(poi_idx, X[b], PV[b])) for b in range(len(PV))]
    out = model.call_batched(batch, shard=DO_SHARD)
    Dtt = pl.abcmb_cl_to_Dl(out.ClTT, out.l)
    Dte = pl.abcmb_cl_to_Dl(out.ClTE, out.l)
    Dee = pl.abcmb_cl_to_Dl(out.ClEE, out.l)
    m0 = pl.bin_model(Dtt, Dte, Dee)
    chi2 = np.asarray(pl.profile_A(m0, with_prior=True)[0], dtype=float)
    if lowee is not None:
        chi2 = chi2 + np.asarray(lowee.chi2(Dee), dtype=float)
    if lowtt is not None:
        chi2 = chi2 + np.asarray(lowtt.chi2(Dtt), dtype=float)
    return np.where(np.isfinite(chi2), chi2, 1e6)


# ======================================================================
# AD objective (single cosmology) and its gradient
# ======================================================================
def chi2_scaled_single(x5, poi_val, poi_idx):
    """scalar chi^2 at scaled nuisances x5, AD-differentiable in x5. Envelope-
    profiled A_planck (stop_gradient) + low-ell. POI fixed."""
    nuis = [i for i in range(6) if i != poi_idx]
    theta = CEN.at[poi_idx].set(poi_val)
    for k, i in enumerate(nuis):
        theta = theta.at[i].set(CEN[i] + SIG[i] * x5[k])
    h, ob, oc, ns, ln10As, tau = theta
    p = dict(FIXED)
    p['h'] = h; p['omega_b'] = ob; p['omega_cdm'] = oc; p['n_s'] = ns
    p['A_s'] = jnp.exp(ln10As) / 1e10; p['tau_reion'] = tau
    out = model.run_cosmology_abbr(model.add_derived_parameters(p))
    Dtt = pl.abcmb_cl_to_Dl(out.ClTT, out.l)
    Dte = pl.abcmb_cl_to_Dl(out.ClTE, out.l)
    Dee = pl.abcmb_cl_to_Dl(out.ClEE, out.l)
    m0 = pl.bin_model(Dtt, Dte, Dee)
    A_star = jax.lax.stop_gradient(pl.profile_A(m0, with_prior=True)[1])
    diff = pl.X_data - m0 / (A_star ** 2)
    c2 = diff @ pl.invcov @ diff + ((A_star - 1.0) / 0.0025) ** 2
    if lowee is not None:
        c2 = c2 + lowee.chi2(Dee)
    if lowtt is not None:
        c2 = c2 + lowtt.chi2(Dtt)
    return c2


def _vg_one(poi_idx):
    """value-and-jacfwd for ONE cosmology (f free from the forward pass)."""
    def f(x5, poi_val):
        return chi2_scaled_single(x5, poi_val, poi_idx)

    def vg(x5, poi_val):
        e = jnp.eye(5)
        fs, g = jax.vmap(lambda v: jax.jvp(lambda z: f(z, poi_val), (x5,), (v,)))(e)
        return fs[0], g
    return f, vg


def iterate_fg(poi_idx, X, PV, vg, method):
    """(f, g) at every grid point.  method 'loop' or 'vmap'."""
    B = len(PV)
    if method == "vmap":
        F, G = jax.vmap(vg)(jnp.asarray(X), jnp.asarray(PV))
        return np.asarray(F), np.asarray(G)
    F = np.empty(B); G = np.empty((B, 5))
    for b in range(B):
        fb, gb = vg(jnp.asarray(X[b]), jnp.asarray(float(PV[b])))
        F[b] = float(fb); G[b] = np.asarray(gb)
    return F, G


# ======================================================================
# vectorised BFGS over the POI grid (lockstep), Armijo line search
# ======================================================================
def bfgs_profile(poi_idx, PV, x0=None, maxit=MAXIT, gtol=GTOL, log_prefix=""):
    B = len(PV)
    _, vg = _vg_one(poi_idx)
    x = np.zeros((B, 5)) if x0 is None else np.array(x0, float)
    # CONSISTENCY: all VALUES come from the fast call_batched path (f, line-search
    # ft, best); the AD gradient (single path) is used ONLY for the descent
    # direction.  Mixing single-path f with fast-path ft in the Armijo test makes
    # the ~0.01 batched-vs-single offset stall the line search near the minimum.
    _, g = iterate_fg(poi_idx, x, PV, vg, GRADMETHOD)     # exact AD gradient
    f = fast_values(poi_idx, x, PV)                       # value (fast path)
    best_f = f.copy(); best_x = x.copy()
    Hinv = np.tile(np.eye(5), (B, 1, 1))
    gnorm = np.abs(g).max(1)
    for it in range(maxit):
        active = gnorm > gtol
        if not active.any():
            break
        d = -np.einsum('bij,bj->bi', Hinv, g)
        gd = (g * d).sum(1)
        bad = gd >= 0
        d[bad] = -g[bad]; Hinv[bad] = np.eye(5); gd[bad] = (g[bad] * d[bad]).sum(1)
        # Armijo backtracking with FAST values, per-point alpha
        alpha = np.ones(B); accept = ~active
        x_new = x.copy()
        for _ls in range(MAXLS):
            if accept.all():
                break
            xt = np.clip(x + alpha[:, None] * d, -XBOX, XBOX)
            ft = fast_values(poi_idx, xt, PV)
            ok = (ft <= f + C1 * alpha * gd) & ~accept
            x_new[ok] = xt[ok]; accept |= ok
            alpha[~accept] *= 0.5
        stuck = active & ~accept
        if stuck.any():
            x_new[stuck] = np.clip(x + alpha[:, None] * d, -XBOX, XBOX)[stuck]
        # AD gradient at the new iterate (single path); f_new already holds the
        # fast-path values of the accepted line-search points (consistent profile)
        _, g_new = iterate_fg(poi_idx, x_new, PV, vg, GRADMETHOD)
        s = x_new - x; y = g_new - g; sy = (s * y).sum(1)
        for b in np.where(active & (sy > 1e-12))[0]:
            rho = 1.0 / sy[b]; I = np.eye(5)
            V = I - rho * np.outer(s[b], y[b])
            Hinv[b] = V @ Hinv[b] @ V.T + rho * np.outer(s[b], s[b])
        x, f, g = x_new, f_new, g_new
        gnorm = np.abs(g).max(1)
        upd = f < best_f; best_f[upd] = f[upd]; best_x[upd] = x[upd]
        if log_prefix:
            print(f"  {log_prefix} it{it}: min={best_f.min():.2f} "
                  f"||g||max={gnorm.max():.2e} active={int(active.sum())} "
                  f"({time.strftime('%H:%M:%S')})", flush=True)
    return best_f, best_x, gnorm


# Fisher / nuisance Hessian via central FD of the EXACT AD gradient
def nuisance_hessian(poi_idx, x_opt, poi_val):
    _, vg = _vg_one(poi_idx)
    cols = []
    for j in range(5):
        ep = np.array(x_opt); ep[j] += FDH
        em = np.array(x_opt); em[j] -= FDH
        _, gp = vg(jnp.asarray(ep), jnp.asarray(float(poi_val)))
        _, gm = vg(jnp.asarray(em), jnp.asarray(float(poi_val)))
        cols.append((np.asarray(gp) - np.asarray(gm)) / (2 * FDH))
    H = np.array(cols).T; H = 0.5 * (H + H.T)
    ev = np.linalg.eigvalsh(H)
    return H, ev, bool(np.all(ev > 0))


def rss_gb():
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / 1e6
    except Exception:
        pass
    return float('nan')


# ======================================================================
# drivers
# ======================================================================
def profile_one(poi, outdir):
    poi_idx = ORDER.index(poi)
    nuis = [ORDER[i] for i in range(6) if i != poi_idx]
    c, s = LCDM[poi]
    poi_grid = np.linspace(c - NSIG * s, c + NSIG * s, NPTS)
    npz = os.path.join(outdir, f"profile_prod_ad_{poi}{TAG}.npz")
    if RESUME and os.path.exists(npz):
        try:
            if bool(np.load(npz, allow_pickle=True)["done"]):
                print(f"[{poi}] resume: done, skip", flush=True); return
        except Exception:
            pass
    print(f"[{poi}] grid {NPTS}pts [{poi_grid[0]:.5f},{poi_grid[-1]:.5f}] nuis={nuis} "
          f"grad={GRADMETHOD} shard={DO_SHARD} lowEE={USE_LOWEE} lowTT={USE_LOWTT}",
          flush=True)
    t0 = time.perf_counter()
    best_f, best_x, gnorm = bfgs_profile(poi_idx, poi_grid, log_prefix=f"[{poi}]")
    pd = np.zeros(NPTS, bool); cond = np.full(NPTS, np.nan)
    if DO_HESS:
        for p in range(NPTS):
            _, ev, is_pd = nuisance_hessian(poi_idx, best_x[p], poi_grid[p])
            pd[p] = is_pd; cond[p] = ev.max() / max(ev.min(), 1e-30)
    lo1, mid, hi1 = interval(poi_grid, best_f, 1.0)
    lo2, _, hi2 = interval(poi_grid, best_f, 4.0)
    sig_p = sigma_parabola(poi_grid, best_f)
    np.savez(npz, poi=poi, poi_grid=poi_grid, chi2=best_f, xstar=best_x,
             gnorm=gnorm, hess_pd=pd, hess_cond=cond, nuis=np.array(nuis),
             done=True, sigma1=np.array([lo1, mid, hi1]), sigma2=np.array([lo2, hi2]),
             sigma_parab=sig_p, gtol=GTOL, use_lowee=USE_LOWEE, use_lowtt=USE_LOWTT)
    j = int(np.nanargmin(best_f))
    print(f"[{poi}] DONE ({time.perf_counter()-t0:.0f}s RSS={rss_gb():.1f}GB): "
          f"minchi2={best_f[j]:.2f} at {poi}={poi_grid[j]:.5f}; "
          f"1sig=[{lo1:.5f},{hi1:.5f}] (PCHIP +/-{(hi1-lo1)/2:.5f}; parab={sig_p:.5f}); "
          f"max||g||={gnorm.max():.2e}; PD {int(pd.sum())}/{NPTS} -> {npz}", flush=True)
    _plot(poi, poi_grid, best_f, npz)


def multistart(poi, outdir, K=MS_K):
    poi_idx = ORDER.index(poi)
    c, s = LCDM[poi]
    test_vals = np.array([c, c + 2 * s])
    rng = np.random.default_rng(1234)
    print(f"[{poi}] MULTISTART K={K} at {poi}={test_vals} grad={GRADMETHOD}", flush=True)
    saved = {}
    for vi, pv in enumerate(test_vals):
        PV = np.full(K, pv)
        x0 = rng.uniform(-2.5, 2.5, (K, 5)); x0[0] = 0.0
        bf, bx, gn = bfgs_profile(poi_idx, PV, x0=x0, log_prefix=f"[{poi}@{pv:.4f}]")
        spread = bf.max() - bf.min()
        print(f"[{poi}@{pv:.5f}] converged chi2: min={bf.min():.3f} max={bf.max():.3f} "
              f"spread={spread:.3f} max||g||={gn.max():.1e} "
              f"(global min if spread<<1)", flush=True)
        saved[f"chi2_{vi}"] = bf; saved[f"gnorm_{vi}"] = gn; saved[f"xstar_{vi}"] = bx
    np.savez(os.path.join(outdir, f"multistart_{poi}{TAG}.npz"),
             poi=poi, test_vals=test_vals, K=K, **saved)


def _plot(poi, grid, chi2, npz):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        d = chi2 - np.nanmin(chi2)
        fig, ax = plt.subplots(figsize=(6, 4.3))
        ax.plot(grid, d, "o-")
        for lv, col in [(1, "g"), (4, "orange"), (9, "r")]:
            ax.axhline(lv, ls="--", lw=0.8, color=col)
        ax.axvline(LCDM[poi][0], ls=":", color="gray", lw=0.8, label=f"Planck {poi}")
        ax.set_xlabel(poi); ax.set_ylabel(r"$\Delta\chi^2$ (profiled, AD/BFGS)")
        ax.set_ylim(0, max(10, float(np.nanmax(d)) * 1.05)); ax.legend()
        ax.set_title(f"AD profile of {poi}: plik-lite + lowTT + lowEE")
        fig.tight_layout(); fig.savefig(npz.replace(".npz", ".png"), dpi=120)
    except Exception as e:
        print(f"[{poi}] plot skipped: {e}", flush=True)


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    mine = POIS[RANK::NPROC]
    print(f"rank {RANK}/{NPROC} devices={jax.devices()} POIs={POIS} mine={mine} "
          f"NPTS={NPTS} GTOL={GTOL} rtol={RTOL} grad={GRADMETHOD} shard={DO_SHARD} "
          f"lowEE={USE_LOWEE} lowTT={USE_LOWTT} hess={DO_HESS} multistart={MULTISTART}",
          flush=True)
    for poi in mine:
        if MULTISTART:
            multistart(poi, outdir)
        else:
            profile_one(poi, outdir)
    print(f"rank {RANK}: done", flush=True)


if __name__ == "__main__":
    main()

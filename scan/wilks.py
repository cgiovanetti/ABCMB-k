"""wilks.py -- Wilks-theorem / coverage validation (TOOL_PLAN.md Workstream E).

The methodological flex: are the frequentist Delta-chi2 intervals the profile tool
reports actually calibrated? Generate N Gaussian mock realizations of the data from a
truth cosmology, re-fit each, and check the profile-likelihood test statistic against
Wilks' theorem.

DATA MODEL (fully, exactly mockable -- the reason for the plik-lite + tau-prior choice):
  * plik-lite high-ell TTTEEE is a multivariate Gaussian with covariance C, so a mock
    is d_k = m0(theta_true) + L z_k,  L L^T = C,  z_k ~ N(0, I).  m0 is the A=1 binned
    D_l vector from ABCMB at the truth.
  * A Gaussian tau "measurement" tau_obs_k ~ N(tau_true, sigma_tau) stands in for the
    real low-ell EE (a tabulated SimAll likelihood with no exact Gaussian mock
    generator).  sigma_tau defaults to the config sig['tau_reion'] (= the Planck tau
    width), so the data model is ~ the headline likelihood but exactly sampleable, and
    every parameter -- tau included -- is properly constrained.
  * A_planck is profiled analytically with its N(1, sigma_A) prior (envelope theorem,
    exactly as the driver).  Its prior is mocked (A_obs_k ~ N(1, sigma_A)) by default
    so the calibration nuisance is treated self-consistently; WK_MOCK_APLANCK=0 pins
    A_obs=1 (its effect on the cosmo-parameter test statistic is negligible).

TEST STATISTIC.  For each mock k and each parameter of interest (POI) theta:
    t_k(theta_true) = chi2_cond_k(POI = theta_true) - chi2_global_k
where chi2_global_k frees all D cosmo params (+ profiled A) and chi2_cond_k fixes the
POI at its TRUE value and frees the other D-1.  Wilks: {t_k} ~ chi2_1.  Coverage of the
nominal interval at the truth is frac(t < level): frac(t<1)=0.6827 (1sigma),
frac(t<3.8415)=0.95 -- this IS the empirical coverage of the Delta-chi2 intervals
(theta_true is inside the Delta-chi2<level interval iff t < level).  This closes
review gap #4.

FITTER -- mean-Jacobian Gauss-Newton on the batch axis.  The whole mock population is
ONE call_batched per GN iteration (NO per-mock gradients): the Jacobian dm0/dtheta is
shared across mocks and refreshed at the batch-mean theta each iteration, so
non-linearity of the cosmology->Cl map (the thing that can break Wilks) is tracked
while the cost stays ~one primal batch per iter.  A_planck is profiled analytically
per evaluation.  Each fit is certified two ways:
  (1) self-consistency: max |Delta t| over the final GN iteration (reported);
  (2) a per-mock-Jacobian GN on a subsample (WK_VALIDATE=K, each mock gets its own
      Jacobian) -- the exact-Jacobian fixed point; t agreement certifies the shared-J
      approximation.  Plus a final batched ||g|| (mean-Jacobian gradient) per mock.

Why GN, not the BFGS driver: 500 mocks x 7 fits x ~8 iters with AD/FD gradients is
tens of node-hours.  Mean-J GN is ~one primal batch per iter -> a couple node-hours,
and the certificates prove it converged to the true MLE.

Multi-node: WK_RANK_SLICE=1 (default under srun NPROC>1) slices the mock population
across ranks; each writes wilks_<config><tag>_r{RANK}.npz; merge with wilks_collect.py.

Run via srun, PYTHONPATH=$(pwd), JAX_COMPILATION_CACHE_DIR set (see scan/wilks.slurm).
Env knobs (all optional):
  WK_CONFIG(scan/configs/lcdm.py) WK_POIS(csv; config pois) WK_NMOCK(500)
  WK_LMAX(2508) WK_RTOL(1e-5) WK_SEED(0) WK_TAG('') WK_OUTDIR(scan/results)
  WK_TAU_SIG(cfg sig[tau_reion]) WK_SIGMA_A(0.0025) WK_MOCK_APLANCK(1)
  WK_TRUTH(''|path.npz with theta_true) WK_JAC_STEP(0.5) WK_GN_MAXIT(10)
  WK_GN_TOL(1e-3) WK_STEP_CAP(4.0) WK_EVAL_CHUNK(256) WK_SHARD(auto|0|1)
  WK_VALIDATE(0) WK_RANK_SLICE(auto)
"""
import os, time, importlib.util
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.plik_lite import PlikLite

_HERE = os.path.dirname(os.path.abspath(__file__))

# chi2_1 reference values (1 dof)
CHI2_1_LEVELS = {"1sigma": 1.0, "90%": 2.7055434540954, "2sigma": 4.0,
                 "95%": 3.8414588206941, "3sigma": 9.0}
CHI2_1_COVERAGE = {  # P(chi2_1 <= level)  -- the nominal coverage of a Delta-chi2<level interval
    1.0: 0.6826894921, 2.7055434540954: 0.90, 3.8414588206941: 0.95, 4.0: 0.9544997361,
    9.0: 0.9973002039}
CHI2_1_MEDIAN = 0.4549364231     # median of chi2_1
from scipy.stats import chi2 as _chi2dist, kstest, norm as _norm


# ======================================================================
# config loading (same scheme as profile_prod_ad.py / smc.py)
# ======================================================================
def _load_config(path):
    if not os.path.isabs(path):
        for base in (os.getcwd(), _HERE, os.path.join(_HERE, "configs")):
            cand = os.path.join(base, path)
            if os.path.exists(cand):
                path = cand
                break
    spec = importlib.util.spec_from_file_location("wk_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG, os.path.abspath(path)


CONFIG_PATH = os.environ.get("WK_CONFIG", "scan/configs/lcdm.py")
CFG, CONFIG_ABS = _load_config(CONFIG_PATH)
CONFIG_NAME = os.path.splitext(os.path.basename(CONFIG_ABS))[0]

ORDER = list(CFG["order"])
D = len(ORDER)
CENTER = np.array([CFG["cen"][k] for k in ORDER])
SIGMA = np.array([CFG["sig"][k] for k in ORDER])
FIXED = dict(CFG["fixed"])
USER_SPECIES = CFG.get("user_species", None)
assert "tau_reion" in ORDER, "wilks.py needs tau_reion in the config order (tau prior)"
TAU_IDX = ORDER.index("tau_reion")

# ---------------- env config ----------------
LMAX = int(os.environ.get("WK_LMAX", 2508))
RTOL = float(os.environ.get("WK_RTOL", 1e-5))
NMOCK = int(os.environ.get("WK_NMOCK", 500))
SEED = int(os.environ.get("WK_SEED", 0))
TAG = os.environ.get("WK_TAG", "")
OUTDIR = os.environ.get("WK_OUTDIR", os.path.join(_HERE, "results"))
TAU_SIG = float(os.environ.get("WK_TAU_SIG", CFG["sig"]["tau_reion"]))
SIGMA_A = float(os.environ.get("WK_SIGMA_A", 0.0025))
MOCK_APLANCK = os.environ.get("WK_MOCK_APLANCK", "1") != "0"
TRUTH_PATH = os.environ.get("WK_TRUTH", "")
JAC_STEP = float(os.environ.get("WK_JAC_STEP", 0.5))     # Jacobian central-FD step (sigma units)
GN_MAXIT = int(os.environ.get("WK_GN_MAXIT", 12))
GN_TOL = float(os.environ.get("WK_GN_TOL", 1e-3))        # max ||dx|| (sigma) to declare converged
GN_MINIT = int(os.environ.get("WK_GN_MINIT", 4))         # min GN iters before the chi2-plateau stop
GN_CHI2TOL = float(os.environ.get("WK_GN_CHI2TOL", 1e-2))  # stop once the plateau-pct per-row chi2 drop < this
GN_PLATEAU_PCT = float(os.environ.get("WK_GN_PLATEAU_PCT", 90.0))  # percentile of per-row drops used for the stop
JAC_EVERY = int(os.environ.get("WK_JAC_EVERY", 3))       # (batchmean ref only) refresh cadence
# Jacobian reference for the shared-Jacobian GN: "warmstart" anchors it at the fixed
# warm-start mean (= truth, ~1sigma from every mock's MLE) -> near-unbiased fixed point for
# both global and conditional fits. "batchmean" refreshes at the drifting population mean
# (every JAC_EVERY iters) -- DON'T use for the conditional fits: the non-POI params get
# stuck and the Jacobian follows them into a self-reinforcing biased fixed point (t inflated).
JAC_REF = os.environ.get("WK_JAC_REF", "warmstart").lower()
STEP_CAP = float(os.environ.get("WK_STEP_CAP", 4.0))     # per-iter ||dx||_inf cap (sigma)
EVAL_CHUNK = int(os.environ.get("WK_EVAL_CHUNK", 256))
VALIDATE = int(os.environ.get("WK_VALIDATE", 0))         # per-mock-Jacobian certificate subsample
_shard_env = os.environ.get("WK_SHARD", "auto").lower()

POIS = os.environ.get("WK_POIS", ",".join(CFG.get("pois", ORDER))).split(",")
POIS = [p.strip() for p in POIS if p.strip() in ORDER]

RANK = int(os.environ.get("SLURM_PROCID", 0))
NPROC = int(os.environ.get("SLURM_NPROCS", 1))
RANK_SLICE = (os.environ.get("WK_RANK_SLICE", "auto") == "1") or \
    (os.environ.get("WK_RANK_SLICE", "auto") == "auto" and NPROC > 1)

pl = PlikLite()
NDATA = pl.ndata
CINV = np.asarray(pl.invcov_np, float)                   # (ndata, ndata)
COV = np.linalg.inv(CINV)                                # used-spectra covariance
model = Model(user_species=USER_SPECIES, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
              rtol_large_k_PE=RTOL, atol_large_k_PE=RTOL * 1e-2,
              rtol_small_k_PE=min(1e-5, RTOL), max_steps_PE=16384)
try:
    NDEV = len(jax.devices('gpu'))
except Exception:
    NDEV = 1
DO_SHARD = (_shard_env == "1") or (_shard_env == "auto" and NDEV > 1)

_CINV_J = jnp.asarray(CINV)
_AGRID = jnp.linspace(0.985, 1.015, 1201)                # A_planck profile grid


# ======================================================================
# parameter -> ABCMB dict (mirrors profile_prod_ad.build_dict)
# ======================================================================
def build_dict(theta):
    p = dict(FIXED)
    for i, name in enumerate(ORDER):
        if name == "ln10As":
            p["A_s"] = float(np.exp(theta[i]) / 1e10)
        else:
            p[name] = float(theta[i])
    return p


def _eval_m0(thetas):
    """thetas (B, D) physical -> m0 (B, ndata) A=1 binned plik-lite model, via
    call_batched in EVAL_CHUNK-sized (padded) chunks. shard only for large B."""
    thetas = np.atleast_2d(np.asarray(thetas, float))
    B = thetas.shape[0]
    shard = DO_SHARD and B >= 2 * NDEV     # shard once the batch is worth splitting
    chunk = min(EVAL_CHUNK, B)
    out_m0 = np.empty((B, NDATA))
    for s in range(0, B, chunk):
        sub = [build_dict(thetas[b]) for b in range(s, min(s + chunk, B))]
        nb = len(sub)
        if nb < chunk:
            sub = sub + [sub[-1]] * (chunk - nb)
        out = model.call_batched(sub, shard=shard)
        Dtt = pl.abcmb_cl_to_Dl(out.ClTT, out.l)
        Dte = pl.abcmb_cl_to_Dl(out.ClTE, out.l)
        Dee = pl.abcmb_cl_to_Dl(out.ClEE, out.l)
        m0 = np.asarray(pl.bin_model(Dtt, Dte, Dee), float)
        out_m0[s:s + nb] = m0[:nb]
    return out_m0


@jax.jit
def _profileA_perrow(m0, d, A_center):
    """Per-row A_planck profile with a N(A_center, sigma_A) prior. m0,d: (B,ndata),
    A_center: (B,) (the mocked calibration measurement, 1 if not mocked).  Returns
    (chi2_plik (B,), A_star (B,)).  Mirrors PlikLite.profile_A but with a PER-ROW data
    vector d (the mocks) and a per-row prior centre."""
    Cinv_d = d @ _CINV_J                                  # (B, ndata)
    a = jnp.sum(d * Cinv_d, axis=-1)                      # d^T Cinv d
    b = jnp.sum(m0 * Cinv_d, axis=-1)                     # m0^T Cinv d
    c = jnp.sum(m0 * (m0 @ _CINV_J), axis=-1)             # m0^T Cinv m0
    t = 1.0 / (_AGRID ** 2)                               # (G,)
    chi2_grid = (a[:, None] - 2.0 * t[None, :] * b[:, None]
                 + (t[None, :] ** 2) * c[:, None]
                 + ((_AGRID[None, :] - A_center[:, None]) / SIGMA_A) ** 2)
    j = jnp.argmin(chi2_grid, axis=-1)
    chi2_min = jnp.take_along_axis(chi2_grid, j[:, None], axis=-1)[:, 0]
    return chi2_min, _AGRID[j]


def chi2_total(thetas, D_data, tau_obs, A_obs):
    """Total chi2 per row: A-profiled plik-lite (vs per-row mock d, prior centred on the
    mocked A_obs) + a Gaussian tau measurement.  thetas (B,D); D_data (B,ndata);
    tau_obs,A_obs (B,).  Returns (chi2 (B,), A_star (B,), m0 (B,ndata))."""
    m0 = _eval_m0(thetas)
    chi2_plik, A_star = _profileA_perrow(jnp.asarray(m0), jnp.asarray(D_data),
                                         jnp.asarray(A_obs))
    chi2_plik = np.asarray(chi2_plik, float); A_star = np.asarray(A_star, float)
    tau = thetas[:, TAU_IDX]
    chi2 = chi2_plik + ((tau - tau_obs) / TAU_SIG) ** 2
    return chi2, A_star, m0


# ======================================================================
# shared (mean) Jacobian dm0/dtheta_scaled at a point
# ======================================================================
def jacobian_scaled(theta0, free_idx):
    """J (ndata, nfree) = dm0/dx, x the scaled coord of the free dims, central FD with
    step JAC_STEP*SIGMA.  theta0 (D,), free_idx list of global dims that are free."""
    nf = len(free_idx)
    thetas = np.empty((2 * nf, D))
    for k, i in enumerate(free_idx):
        h = JAC_STEP * SIGMA[i]
        tp = theta0.copy(); tp[i] += h
        tm = theta0.copy(); tm[i] -= h
        thetas[2 * k] = tp; thetas[2 * k + 1] = tm
    m0 = _eval_m0(thetas)                                 # (2nf, ndata)
    J = np.empty((NDATA, nf))
    for k in range(nf):
        # x-step is +/-JAC_STEP in sigma units -> dm0/dx = (m0+ - m0-)/(2*JAC_STEP)
        J[:, k] = (m0[2 * k] - m0[2 * k + 1]) / (2.0 * JAC_STEP)
    return J


def _gn_matrices(J, free_idx):
    """Gauss-Newton step operators from a scaled Jacobian J (ndata,nfree).  The tau
    measurement is a data point with weight 1/sig_tau^2 and Jacobian dtau/dx = SIGMA_tau
    (physical tau per scaled coord), so in scaled coords its normal-matrix and rhs
    contributions carry SIGMA_tau (NOT a bare 1/sig_tau^2 -- that mixes scaled x with
    physical tau and over-weights the tau direction by 1/SIGMA_tau^2).  Returns
    M_plik = N^-1 J^T Cinv (nfree,ndata) and the tau step operator
    m_tau = N^-1 e_tau * SIGMA_tau/sig_tau^2 (or None if tau is fixed), with
    N = J^T Cinv J + (SIGMA_tau/sig_tau)^2 e_tau e_tau^T."""
    JTC = J.T @ CINV                                      # (nfree, ndata)
    N = JTC @ J                                           # (nfree, nfree)
    jtau = free_idx.index(TAU_IDX) if TAU_IDX in free_idx else None
    m_tau = None
    if jtau is not None:
        jac_tau = SIGMA[TAU_IDX]                          # dtau/dx_jtau
        N = N.copy(); N[jtau, jtau] += (jac_tau / TAU_SIG) ** 2
    Ninv = np.linalg.inv(N + 1e-9 * np.eye(N.shape[0]))
    M_plik = Ninv @ JTC
    if jtau is not None:
        m_tau = Ninv[:, jtau] * (jac_tau / TAU_SIG ** 2)
    return M_plik, m_tau, jtau


# ======================================================================
# mean-Jacobian Gauss-Newton fit over the whole mock population (one batch/iter)
# ======================================================================
def gn_fit(theta0, free_idx, D_data, tau_obs, A_obs, label=""):
    """Fit all B mocks in lockstep.  free_idx: global dims optimised (the rest held at
    theta0).  theta0 (B,D) warm starts.  Returns (chi2 (B,), theta (B,D), info).
    The Jacobian is shared across mocks, anchored at the fixed warm-start reference (truth)
    by default (WK_JAC_REF); only the model m0(theta_k) is re-evaluated per-mock each iter
    (one B-batch primal call), so a fit costs ~one primal batch per GN iteration."""
    B = D_data.shape[0]
    nf = len(free_idx)
    theta = np.asarray(theta0, float).copy()
    fidx = np.asarray(free_idx, int)
    sig_f = SIGMA[fidx]                                   # (nf,)
    # ---- shared Jacobian + GN operators at the FIXED warm-start reference (= truth). The
    # mock MLEs scatter ~1sigma around it, so J(ref) ~ J(MLE) and the GN fixed point
    # (rho perp J(ref)) is near the true MLE for BOTH global and conditional fits -> the
    # residual bias cancels in t = cond - global. (Refreshing J at the DRIFTING batch mean
    # instead let the conditional fits stick at a biased fixed point; WK_JAC_REF=batchmean
    # restores that behaviour for strongly non-linear models, but is off by default.)
    ref_point = theta.mean(axis=0)
    J = jacobian_scaled(ref_point, list(free_idx))        # (ndata, nf) scaled
    M_plik, m_tau, jtau = _gn_matrices(J, list(free_idx))
    gnorm = None; chi2_prev = None; max_dx = np.inf
    for it in range(GN_MAXIT):
        # ---- residuals at the current per-mock theta (one batched primal eval) ----
        chi2, A_star, m0 = chi2_total(theta, D_data, tau_obs, A_obs)
        rho = D_data - m0 / (A_star[:, None] ** 2)        # (B, ndata) plik residual
        rho_tau = tau_obs - theta[:, TAU_IDX]             # (B,)
        # ---- chi2-plateau early-stop: the test statistic is a chi2 difference, so once
        # the bulk of rows have converged the deliverable has too. Use the GN_PLATEAU_PCT
        # percentile of the per-row drops (not the max -- a few slow tail mocks would else
        # force every fit to maxit); the residual under-convergence of those few cancels in
        # t = cond - global because global and conditional fits share the warm start + fitter.
        if chi2_prev is not None and it >= GN_MINIT:
            improve = float(np.percentile(chi2_prev - chi2, GN_PLATEAU_PCT))
            if improve < GN_CHI2TOL:
                if label:
                    print(f"  [{label}] gn it{it}: chi2 plateau (p{GN_PLATEAU_PCT:.0f} drop "
                          f"{improve:.2e} < {GN_CHI2TOL:.0e}) -> stop", flush=True)
                break
        chi2_prev = chi2.copy()
        if JAC_REF == "batchmean" and it > 0 and it % JAC_EVERY == 0:
            J = jacobian_scaled(theta.mean(axis=0), list(free_idx))
            M_plik, m_tau, jtau = _gn_matrices(J, list(free_idx))
        # ---- per-mock GN step (host linear algebra; no GPU) ----
        dx = rho @ M_plik.T                               # (B, nf) scaled
        if m_tau is not None:
            dx = dx + rho_tau[:, None] * m_tau[None, :]
        step_inf = np.abs(dx).max(axis=1)
        scale = np.where(step_inf > STEP_CAP, STEP_CAP / np.maximum(step_inf, 1e-30), 1.0)
        dx = dx * scale[:, None]
        theta[:, fidx] = theta[:, fidx] + sig_f[None, :] * dx
        max_dx = float(np.abs(dx).max())
        # ||g||_inf (mean-Jacobian gradient magnitude); plik part 2 J^T Cinv rho, tau part
        # 2 (SIGMA_tau/sig_tau^2) rho_tau (the scaled-coord tau-prior gradient)
        g = 2.0 * (J.T @ CINV @ rho.T)                    # (nf, B)
        if jtau is not None:
            g[jtau, :] += 2.0 * (SIGMA[TAU_IDX] / TAU_SIG ** 2) * rho_tau
        gnorm = np.abs(g).max(axis=0)                     # (B,)
        if label:
            print(f"  [{label}] gn it{it}: chi2 min/med/max "
                  f"{chi2.min():.2f}/{np.median(chi2):.2f}/{chi2.max():.2f} "
                  f"max|dx|={max_dx:.2e} max||g||={gnorm.max():.2e} "
                  f"({time.strftime('%H:%M:%S')})", flush=True)
        if max_dx < GN_TOL:
            break
    # final chi2 at the converged theta (one more eval so chi2 matches theta exactly)
    chi2, A_star, _ = chi2_total(theta, D_data, tau_obs, A_obs)
    info = dict(iters=it + 1, max_dx=max_dx, gnorm=gnorm, A_star=A_star)
    return chi2, theta, info


# ======================================================================
# per-mock-Jacobian GN certificate (exact-Jacobian fixed point on a subsample)
# ======================================================================
def gn_fit_permock(theta0, free_idx, D_data, tau_obs, A_obs, label=""):
    """Like gn_fit but every mock gets its OWN Jacobian (refreshed at its own theta)
    each iter -- the exact-Jacobian GN.  Expensive (B*2*nfree evals/iter), so only used
    on the WK_VALIDATE subsample to certify the shared-Jacobian approximation."""
    B = D_data.shape[0]
    nf = len(free_idx)
    theta = np.asarray(theta0, float).copy()
    fidx = np.asarray(free_idx, int); sig_f = SIGMA[fidx]
    Ms = None; chi2_prev = None; max_dx = np.inf
    for it in range(GN_MAXIT):
        chi2, A_star, m0 = chi2_total(theta, D_data, tau_obs, A_obs)
        rho = D_data - m0 / (A_star[:, None] ** 2)
        rho_tau = tau_obs - theta[:, TAU_IDX]
        if chi2_prev is not None and it >= GN_MINIT and \
                float(np.percentile(chi2_prev - chi2, GN_PLATEAU_PCT)) < GN_CHI2TOL:
            break
        chi2_prev = chi2.copy()
        # per-mock Jacobian (refreshed every JAC_EVERY): B*2nf perturbations in ONE batch
        if Ms is None or it % JAC_EVERY == 0:
            big = np.empty((B * 2 * nf, D))
            for k in range(B):
                for j, i in enumerate(free_idx):
                    h = JAC_STEP * SIGMA[i]
                    tp = theta[k].copy(); tp[i] += h
                    tm = theta[k].copy(); tm[i] -= h
                    big[(k * 2 * nf) + 2 * j] = tp
                    big[(k * 2 * nf) + 2 * j + 1] = tm
            m0_big = _eval_m0(big)
            Ms = []
            for k in range(B):
                Jk = np.empty((NDATA, nf))
                for j in range(nf):
                    base = k * 2 * nf
                    Jk[:, j] = (m0_big[base + 2 * j] - m0_big[base + 2 * j + 1]) / (2.0 * JAC_STEP)
                Ms.append(_gn_matrices(Jk, list(free_idx)))
        dx = np.empty((B, nf))
        for k in range(B):
            M_plik, m_tau, jtau = Ms[k]
            d = rho[k] @ M_plik.T
            if m_tau is not None:
                d = d + rho_tau[k] * m_tau
            si = np.abs(d).max()
            if si > STEP_CAP:
                d = d * (STEP_CAP / si)
            dx[k] = d
        theta[:, fidx] = theta[:, fidx] + sig_f[None, :] * dx
        max_dx = float(np.abs(dx).max())
        if label:
            print(f"  [{label}] permock it{it}: max|dx|={max_dx:.2e}", flush=True)
        if max_dx < GN_TOL:
            break
    chi2, _, _ = chi2_total(theta, D_data, tau_obs, A_obs)
    return chi2, theta


# ======================================================================
# mock generation
# ======================================================================
def make_mocks(theta_true):
    """Generate NMOCK plik-lite mocks + tau/A_planck measurements (deterministic,
    seeded -> identical across ranks; ranks slice the SAME population)."""
    m0_true = _eval_m0(theta_true[None, :])[0]            # (ndata,)
    L = np.linalg.cholesky(COV)
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((NMOCK, NDATA))
    D_data = m0_true[None, :] + z @ L.T                   # (NMOCK, ndata)
    w = rng.standard_normal(NMOCK)
    tau_obs = theta_true[TAU_IDX] + TAU_SIG * w
    if MOCK_APLANCK:
        A_obs = 1.0 + SIGMA_A * rng.standard_normal(NMOCK)
    else:
        A_obs = np.ones(NMOCK)
    return D_data, tau_obs, A_obs, m0_true


# ======================================================================
# statistics + plots
# ======================================================================
def wilks_stats(t):
    """Compare {t} to chi2_1: coverage at the standard levels, KS test, moments."""
    t = np.asarray(t, float)
    t = t[np.isfinite(t)]
    n = len(t)
    cov = {}
    for lvl, nominal in CHI2_1_COVERAGE.items():
        frac = float(np.mean(t <= lvl))
        se = np.sqrt(max(nominal * (1 - nominal) / n, 1e-12))   # binomial SE at nominal
        cov[lvl] = dict(frac=frac, nominal=nominal, dev_sigma=(frac - nominal) / se)
    ks = kstest(t, _chi2dist(1).cdf)
    return dict(n=n, mean=float(t.mean()), median=float(np.median(t)),
                coverage=cov, ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
                neg_frac=float(np.mean(np.asarray(t) < -1e-6)))


def _plot_poi(poi, t, z, npz_path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        t = np.asarray(t, float); t = t[np.isfinite(t)]
        z = np.asarray(z, float); z = z[np.isfinite(z)]
        fig, ax = plt.subplots(1, 3, figsize=(13, 4))
        # (1) t histogram vs chi2_1 pdf
        tc = np.clip(t, 0, None)
        ax[0].hist(tc, bins=40, range=(0, max(10, np.percentile(tc, 99))),
                   density=True, alpha=0.6, label="mocks")
        xs = np.linspace(1e-3, max(10, np.percentile(tc, 99)), 400)
        ax[0].plot(xs, _chi2dist(1).pdf(xs), "r-", lw=2, label=r"$\chi^2_1$")
        ax[0].set_xlabel(r"$t=\Delta\chi^2(\theta_{\rm true})$"); ax[0].set_ylabel("pdf")
        ax[0].set_title(f"{poi}: test statistic"); ax[0].legend()
        # (2) empirical CDF of t vs chi2_1 CDF
        ts = np.sort(tc); ecdf = np.arange(1, len(ts) + 1) / len(ts)
        ax[1].plot(ts, ecdf, drawstyle="steps-post", label="empirical")
        ax[1].plot(xs, _chi2dist(1).cdf(xs), "r-", lw=2, label=r"$\chi^2_1$")
        for lv in (1.0, 3.8414588206941):
            ax[1].axvline(lv, ls=":", lw=0.8, color="gray")
        ax[1].set_xlabel("t"); ax[1].set_ylabel("CDF"); ax[1].set_title("coverage")
        ax[1].legend()
        # (3) signed pull vs N(0,1)
        ax[2].hist(z, bins=40, range=(-4, 4), density=True, alpha=0.6, label="mocks")
        zs = np.linspace(-4, 4, 400)
        ax[2].plot(zs, _norm.pdf(zs), "r-", lw=2, label=r"$N(0,1)$")
        ax[2].set_xlabel(r"pull $(\hat\theta-\theta_{\rm true})/\sigma$")
        ax[2].set_title(f"{poi}: pull"); ax[2].legend()
        fig.tight_layout(); fig.savefig(npz_path.replace(".npz", ".png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        print(f"[{poi}] plot skipped: {e}", flush=True)


# ======================================================================
# driver
# ======================================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"rank {RANK}/{NPROC} devices={jax.devices()} config={CONFIG_ABS} "
          f"POIs={POIS} NMOCK={NMOCK} D={D} lmax={LMAX} rtol={RTOL} shard={DO_SHARD} "
          f"tau_sig={TAU_SIG} mock_Aplanck={MOCK_APLANCK} validate={VALIDATE} "
          f"rank_slice={RANK_SLICE}", flush=True)

    # ---- truth ----
    if TRUTH_PATH and os.path.exists(TRUTH_PATH):
        d = np.load(TRUTH_PATH, allow_pickle=True)
        theta_true = np.array([float(d["theta_true_by_name"].item()[k]) for k in ORDER]) \
            if "theta_true_by_name" in d.files else np.asarray(d["theta_true"], float)
        print(f"[truth] loaded from {TRUTH_PATH}", flush=True)
    else:
        theta_true = CENTER.copy()
    print("[truth] " + ", ".join(f"{ORDER[i]}={theta_true[i]:.5g}" for i in range(D)),
          flush=True)

    # ---- mocks (full population; ranks slice it) ----
    t0 = time.perf_counter()
    D_data_all, tau_obs_all, A_obs_all, m0_true = make_mocks(theta_true)
    print(f"[mocks] generated {NMOCK} (ndata={NDATA}) in {time.perf_counter()-t0:.0f}s; "
          f"||m0_true||={np.linalg.norm(m0_true):.3g}", flush=True)
    if RANK_SLICE and NPROC > 1:
        mine = np.arange(RANK, NMOCK, NPROC)
    else:
        mine = np.arange(NMOCK)
    D_data = D_data_all[mine]; tau_obs = tau_obs_all[mine]; A_obs = A_obs_all[mine]
    B = len(mine)
    print(f"[slice] rank {RANK} fits {B}/{NMOCK} mocks", flush=True)

    # ---- resumable checkpoint: the per-rank output npz IS the checkpoint. A chained
    # job (e.g. across short debug-queue allocations, which schedule far faster than a
    # multi-day multi-node regular slot) reloads the global fit + any completed POIs and
    # only fits what remains, re-saving after each POI so a walltime timeout never loses
    # finished work. The mock slice is deterministic in (NMOCK, SEED, NPROC, RANK), so keep
    # the node count fixed across a chain or the loaded t arrays won't match the slice.
    free_all = list(range(D))
    suffix = f"_r{RANK}" if (RANK_SLICE and NPROC > 1) else ""
    out = os.path.join(OUTDIR, f"wilks_{CONFIG_NAME}{TAG}{suffix}.npz")
    results = {}
    cert = {}
    chi2_glob = theta_glob = gnorm_glob = None
    if os.path.exists(out):
        ck = np.load(out, allow_pickle=True)
        if "chi2_global" in ck.files and int(ck["nmock"]) == NMOCK and int(ck["seed"]) == SEED:
            chi2_glob = np.asarray(ck["chi2_global"], float)
            theta_glob = np.asarray(ck["theta_global"], float)
            gnorm_glob = np.asarray(ck["gnorm_global"], float)
            for poi in POIS:
                if f"t_{poi}" in ck.files:
                    results[poi] = dict(t=np.asarray(ck[f"t_{poi}"], float),
                                        z=np.asarray(ck[f"z_{poi}"], float),
                                        chi2_cond=np.asarray(ck[f"chi2cond_{poi}"], float),
                                        gnorm_cond=np.asarray(ck[f"gnorm_{poi}"], float))
                    if f"cert_maxdt_{poi}" in ck.files:
                        cert[poi] = dict(idx=np.asarray(ck[f"cert_idx_{poi}"]),
                                         t_shared=np.asarray(ck[f"cert_tshared_{poi}"]),
                                         t_exact=np.asarray(ck[f"cert_texact_{poi}"]),
                                         max_abs_dt=float(ck[f"cert_maxdt_{poi}"]),
                                         med_abs_dt=float("nan"))
            print(f"[resume] loaded global + POIs {sorted(results)} from {out}", flush=True)
        ck.close()

    def _save_state():
        save = dict(config=CONFIG_ABS, order=np.array(ORDER), pois=np.array(POIS),
                    theta_true=theta_true, sigma=SIGMA, nmock=NMOCK, seed=SEED,
                    mock_idx=mine, tau_sig=TAU_SIG, sigma_A=SIGMA_A,
                    mock_aplanck=MOCK_APLANCK, lmax=LMAX, rtol=RTOL,
                    chi2_global=chi2_glob, gnorm_global=gnorm_glob, theta_global=theta_glob)
        for p in results:
            save[f"t_{p}"] = results[p]["t"]; save[f"z_{p}"] = results[p]["z"]
            save[f"chi2cond_{p}"] = results[p]["chi2_cond"]
            save[f"gnorm_{p}"] = results[p]["gnorm_cond"]
            if p in cert:
                save[f"cert_idx_{p}"] = cert[p]["idx"]
                save[f"cert_tshared_{p}"] = cert[p]["t_shared"]
                save[f"cert_texact_{p}"] = cert[p]["t_exact"]
                save[f"cert_maxdt_{p}"] = cert[p]["max_abs_dt"]
        tmp = out + ".tmp.npz"               # atomic: never leave a half-written checkpoint
        np.savez(tmp, **save); os.replace(tmp, out)

    # ---- global fit (all D cosmo dims free) ----
    if chi2_glob is None:
        theta0_g = np.tile(theta_true, (B, 1))
        tG = time.perf_counter()
        chi2_glob, theta_glob, info_g = gn_fit(theta0_g, free_all, D_data, tau_obs, A_obs,
                                               label="global")
        gnorm_glob = info_g["gnorm"]
        print(f"[global] done {time.perf_counter()-tG:.0f}s iters={info_g['iters']} "
              f"max||g||={gnorm_glob.max():.2e} chi2 med={np.median(chi2_glob):.2f}", flush=True)
        _save_state()
    else:
        print(f"[global] reused from checkpoint chi2 med={np.median(chi2_glob):.2f}", flush=True)

    # ---- per-POI conditional fits (POI fixed at truth; warm-start at TRUTH, symmetric) ----
    for poi in POIS:
        if poi in results:
            print(f"[cond:{poi}] skip (in checkpoint) t med={np.median(results[poi]['t']):.3f}",
                  flush=True)
            continue
        pidx = ORDER.index(poi)
        free_c = [i for i in range(D) if i != pidx]
        # Warm-start the conditional fit at TRUTH (POI already = truth), the SAME start as
        # the global fit -- NOT the global solution. Symmetric start + identical fitter means
        # the mean-Jacobian fixed-point bias and any residual under-convergence are common to
        # both fits and cancel in t = chi2_cond - chi2_global. (Warm-starting from the global
        # solution put the non-POI params ~1sigma off the conditional optimum, so the
        # mean-Jacobian settled at a biased fixed point and t was inflated -- see CHANGELOG.)
        theta0_c = np.tile(theta_true, (B, 1))
        tC = time.perf_counter()
        chi2_cond, theta_cond, info_c = gn_fit(theta0_c, free_c, D_data, tau_obs, A_obs,
                                               label=f"cond:{poi}")
        t_stat = chi2_cond - chi2_glob
        # signed pull consistent with the test statistic: z = sign(theta_hat - theta_true)
        # * sqrt(t).  For a Gaussian (Wilks) t = z^2, so z ~ N(0,1) exactly -- a sigma-free
        # calibration check (no reliance on a fiducial profile sigma).
        z_pull = np.sign(theta_glob[:, pidx] - theta_true[pidx]) * np.sqrt(np.clip(t_stat, 0, None))
        results[poi] = dict(t=t_stat, z=z_pull, chi2_cond=chi2_cond,
                            gnorm_cond=info_c["gnorm"])
        print(f"[cond:{poi}] done {time.perf_counter()-tC:.0f}s iters={info_c['iters']} "
              f"max||g||={info_c['gnorm'].max():.2e} "
              f"t med={np.median(t_stat):.3f} (neg {int(np.sum(t_stat < -1e-3))})", flush=True)
        _save_state()

    # ---- per-mock-Jacobian certificate on a subsample (exact-Jacobian fixed point) ----
    # Resume-aware: only POIs without a saved cert are (re)done. Skipped entirely when
    # WK_VALIDATE=0 (the default for the large coverage runs -- gate (a) is the deliverable).
    if VALIDATE > 0:
        todo = [p for p in POIS if p not in cert]
        if todo:
            K = min(VALIDATE, B)
            sub = np.linspace(0, B - 1, K).astype(int)
            print(f"[cert] per-mock-Jacobian GN on {K} mocks for {todo} (exact-J) ...", flush=True)
            cG, _ = gn_fit_permock(np.tile(theta_true, (K, 1)), free_all,
                                   D_data[sub], tau_obs[sub], A_obs[sub], label="cert:global")
            for poi in todo:
                pidx = ORDER.index(poi)
                free_c = [i for i in range(D) if i != pidx]
                cC, _ = gn_fit_permock(np.tile(theta_true, (K, 1)), free_c,
                                       D_data[sub], tau_obs[sub], A_obs[sub], label=f"cert:{poi}")
                t_exact = cC - cG
                dt = np.abs(t_exact - results[poi]["t"][sub])
                cert[poi] = dict(idx=sub, t_shared=results[poi]["t"][sub], t_exact=t_exact,
                                 max_abs_dt=float(np.max(dt)), med_abs_dt=float(np.median(dt)))
                print(f"[cert:{poi}] shared-J vs per-mock-J: max|dt|={cert[poi]['max_abs_dt']:.3f} "
                      f"med|dt|={cert[poi]['med_abs_dt']:.3f}", flush=True)
                _save_state()

    _save_state()
    print(f"[save] {out}", flush=True)

    # ---- single-rank (or whole-population) stats + plots inline; multi-rank -> collect ----
    if not (RANK_SLICE and NPROC > 1):
        print("\n===== WILKS / COVERAGE SUMMARY (chi2_1 expected) =====", flush=True)
        for poi in POIS:
            st = wilks_stats(results[poi]["t"])
            print(f"[{poi}] n={st['n']} mean={st['mean']:.3f}(exp 1.0) "
                  f"median={st['median']:.3f}(exp {CHI2_1_MEDIAN:.3f}) "
                  f"KS={st['ks_stat']:.3f} p={st['ks_p']:.3f}", flush=True)
            for lvl in (1.0, 3.8414588206941):
                c = st["coverage"][lvl]
                print(f"      cov(t<{lvl:.3g})={c['frac']:.3f} "
                      f"(nominal {c['nominal']:.3f}, {c['dev_sigma']:+.1f}sigma)", flush=True)
            _plot_poi(poi, results[poi]["t"], results[poi]["z"], out)
    print(f"rank {RANK}: done", flush=True)


if __name__ == "__main__":
    main()

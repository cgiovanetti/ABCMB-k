"""Config-driven, lockstep frequentist profile-likelihood tool.

One sbatch in; per profiled parameter (POI), an npz + png out with the grid, the
profiled chi2, the best fit, 68/95% intervals and a per-point AD convergence certificate.

  1. Config-driven (PA_CONFIG=path/to/config.py). A config declares the parameter
     order, fiducials (cen), scale widths (sig), optional grid widths (grid_sig), the
     fixed dict, optional user_species, likelihood toggles, and NPTS/NSIG. Adding a
     parameter is a config edit, not a code edit.

  2. One lockstep batch across all POIs x grid points. Each "row" is one
     optimisation point (the POI fixed at its grid value, the other parameters
     free); the optimizer state is per-row arrays, so concatenating POIs just
     lengthens the batch. One POI subset per SLURM rank via PA_POI_SLICE.

  3. Optimizer: Levenberg-Marquardt on the Gauss-Newton curvature, AD only. Every
     gradient call pushes the free directions through ABCMB (forward mode) to the
     C_ell tangents dC_ell/dtheta and turns them into the chi2 gradient AND its
     Gauss-Newton matrix J^T W J (the high-ell likelihood's data covariance, config
     high_ell: full plik or ACT DR6 lite; its nuisances
     enter through the profiled-likelihood Schur complement; the low-ell
     likelihoods through their exact Hessian in D_ell), at the row's own point.
     Each iteration tries PA_LM_K damping values per active row in one batched
     value call and keeps the best that passes the gain-ratio test. Rows that
     moved get a new gradient; rows that did not keep theirs and raise the
     damping; converged and stalled rows leave the batches. Values for every
     acceptance test come from the call_batched path. A row is converged when
     max|g| < PA_GTOL. A row that stalls (no accepted step for PA_STALL_MAX
     iterations) while its Newton decrement 0.5 g'H^-1 g, the chi2 its Gauss-Newton
     model still expects to gain, is below PA_DTOL sits on the likelihood's noise
     floor: it counts as converged and is recorded as "floor" in the npz. The loop stops
     when no row is active (every row converged, stalled or on the floor), at PA_MAXIT,
     or with PA_SIGTOL > 0 once every POI's 68% interval is stable; the reason is logged
     and saved (npz stop_reason).

  4. Bounds and one-sided intervals (config, both optional). bounds = {name: (lo, hi)},
     None = open: physical limits, hard for the optimizer (projected steps), a POI's grid
     kept inside them, and the POI's intervals from the boundary-aware asymptotic
     thresholds of Cowan, Cranmer, Gross & Vitells (arXiv:1007.1727, the t~ statistic;
     the Feldman-Cousins likelihood-ratio ordering in the Gaussian limit), so an interval
     next to a bound comes out two-sided or as a one-sided limit as the data decide.
     box = {name: (lo, hi)} in scaled sigmas, None = -+PA_XBOX: the optimizer box of a
     free parameter (inside its bounds). A row on a box face counts as converged when chi2
     is flat outward there (outward slope below PA_GTOL: a direction the data leave open);
     when chi2 still falls outward the box, not the likelihood, holds it (npz at_box), so
     widen the box or set a bound. Each interval end is a threshold crossing, the bound,
     or open (below threshold up to the grid end); the interval is two-sided, an upper or
     lower limit, or unconstrained (npz ends1/ends95, kind1/kind95).

  5. Multimodality (scan/lm_tools.py). By default every POI gets PA_MS_K extra starts at
     PA_MS_NPTS of its grid points, spread +-PA_MS_SPREAD profile sigmas along the softest
     eigen-directions of the warm Gauss-Newton matrix on the row's free dims; they run in
     the same lockstep and leave the batch as they converge. A grid point is flagged when
     a start stalled, when converged starts sit at distinct points (> PA_MS_DX scaled
     sigma apart) or when they disagree in chi2 by > PA_MS_DCHI2. Each distinct minimum a
     spread start found at a grid point (not the warm start's) seeds a branch: an ordinary
     LM run from it at every grid point of that POI. The profile is the lower envelope
     over all starts and branches; the intervals are the union of the regions below
     threshold (one region for a unimodal profile). The per-POI npz keeps every row.

Run via srun, PYTHONPATH=$(pwd), JAX_COMPILATION_CACHE_DIR set. Env knobs (default):
  PA_CONFIG(required) PA_TAG('') PA_POIS(csv; the config order without the BBN nuisances)
  PA_POI_SLICE(0) PA_LMAX(2508) PA_NPTS(config npts, else 25) PA_NSIG(config nsig, else 3.0)
  PA_XBOX(5) PA_MAXIT(40) PA_GTOL(3e-2) PA_DTOL(1e-3; 0 = off) PA_SIGTOL(0 = off)
  PA_SIGTOL_PATIENCE(3) PA_RTOL(1e-5) PA_RESUME(1) PA_VAL_CHUNK(256) PA_AD_BCHUNK(64)
  PA_GRAD_SPECB(4) PA_LM_K(4) PA_STEP_MAX(1.0) PA_STALL_MAX(3) PA_WARM_LR(0) PA_TRACE_EVALS(0)
  PA_MS_K(3) PA_MS_NPTS(6) PA_MS_SPREAD(2.0) PA_MS_DX(0.5) PA_MS_DCHI2(0.1) PA_MS_WAIT(1)
  PA_MS_BRANCH_MAX(4) PA_BR_MAXIT(10) PA_FIT_DEG(4) PA_FIT_DCHI2(9) PA_CPU_DEVICES(16)
  PA_MLE_PREPASS(0) PA_MLE_MAXITER(40) PA_MLE_SIG_CLIP(3.0) PLF_MAXIT(3000; scan/plik_full.py)
"""
# Host devices for the HyRex jvp (scan/batched_grad.hyrex_cpu_mesh): must be set before JAX
# initialises. PA_CPU_DEVICES=0/1 keeps the single-device path.
import os as _os
_ncpu = _os.environ.get("PA_CPU_DEVICES", "16")
if _ncpu not in ("0", "1") and "xla_force_host_platform_device_count" not in _os.environ.get("XLA_FLAGS", ""):
    _os.environ["XLA_FLAGS"] = (_os.environ.get("XLA_FLAGS", "") +
                                f" --xla_force_host_platform_device_count={_ncpu}").strip()

import os, time, importlib.util
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# equinox runtime errors become NaNs instead of aborting (read by equinox at import): a failed ODE solve is
# then a NaN chi2 the step test rejects, and the vmapped LINX stage needs it -- under vmap a lax.cond runs
# both branches, and lineax's error_if inside the implicit solver's untaken branch would otherwise raise on
# the discarded garbage (scan/bbn_stage.py).
os.environ.setdefault("EQX_ON_ERROR", "nan")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
from scipy.stats import chi2 as _chi2_dist
from scipy.special import ndtr
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import Model
from abcmb import species as _species
from scan.lowl_like import LowLEE, LowLTT, LOWEE_TABLES, cl_to_Dl
from scan.batched_grad import staged_cl_and_grad, _to_float as _bg_to_float
from scan.lm_tools import (proj_grad, lm_step, newton_decrement, spread_starts, flag_point,
                           cluster_points, union_regions, is_multimodal)

_HERE = os.path.dirname(os.path.abspath(__file__))
CL1, CL2, CL95 = 0.682689492137, 0.954499736104, 0.95


# ======================================================================
# profile-interval helpers
# ======================================================================
def _interval_mode(x, y, bound=(-np.inf, np.inf)):
    """'pchip' for a profile whose points show two minima separated by a barrier above the roughness
    (lm_tools.is_multimodal; a polynomial would smooth the second mode away) or whose lowest point is a
    grid end that is not a physical bound (a profile still falling at the grid end, where a polynomial
    invents a turn-over), else 'fit'."""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    if m.sum() >= 2:
        xm, ym = x[m], y[m]; o = np.argsort(xm); xm, ym = xm[o], ym[o]
        j = int(np.argmin(ym))
        if (j == 0 and xm[0] != bound[0]) or (j == len(xm) - 1 and xm[-1] != bound[1]):
            return "pchip"
    return "pchip" if is_multimodal(x, y, FIT_DCHI2, MS_DCHI2) else "fit"


def _smooth_profile(x, y, bound=(-np.inf, np.inf)):
    """Dense Delta-chi2 curve (xs, ys) through a profile's points. 'fit': least-squares polynomial of
    degree FIT_DEG through the points within FIT_DCHI2 of the minimum (averages per-row scatter; a
    quartic follows the skew of tau / Neff); 'pchip' (_interval_mode, or too few points for the fit):
    shape-preserving interpolation through every point. None with fewer than 4 finite points."""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    if m.sum() < 4:
        return None
    x, y = x[m], y[m]; o = np.argsort(x); x, y = x[o], y[o]
    if _interval_mode(x, y, bound) == "fit":
        sel = (y - y.min()) <= FIT_DCHI2
        if sel.sum() >= FIT_DEG + 3:
            xs_, ys_ = x[sel], y[sel]; xc = xs_.mean()
            c = np.polyfit(xs_ - xc, ys_, FIT_DEG)
            xs = np.linspace(xs_.min(), xs_.max(), 40001); ys = np.polyval(c, xs - xc)
            return xs, ys - ys.min()
    from scipy.interpolate import PchipInterpolator
    xs = np.linspace(x[0], x[-1], 40001); ys = PchipInterpolator(x, y - y.min())(xs)
    return xs, ys - ys.min()


def wald_fit(x, y):
    """Parabola a (x - m)^2 + c through the points within Delta-chi2 <= 4 of the minimum: the Wald
    sigma = 1/sqrt(a), the vertex m, and the largest misfit of the parabola on those points
    (Delta-chi2 units; large = the profile is far from Gaussian). NaNs when no upward parabola fits."""
    x = np.asarray(x, float); y = np.asarray(y, float); k = np.isfinite(y)
    x, y = x[k], y[k]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    d = y - y.min(); sel = d <= 4.0
    if sel.sum() < 3:
        sel = np.argsort(d)[:3]
    a, b, c = np.polyfit(x[sel], y[sel], 2)
    if not a > 0:
        return np.nan, np.nan, np.nan
    m = -b / (2 * a)
    return 1.0 / np.sqrt(a), m, float(np.max(np.abs(np.polyval([a, b, c], x[sel]) - y[sel])))


def t_threshold(cl, d):
    """Asymptotic critical value of the boundary-aware statistic t~ at confidence cl for tested values
    d = (distance from the bound) / sigma >= 0 Wald sigmas inside the physical region (Cowan, Cranmer,
    Gross & Vitells 2011, arXiv:1007.1727): F(t) = 2 Phi(sqrt t) - 1 for t <= d^2 and
    Phi(sqrt t) + Phi((t + d^2) / (2 d)) - 1 beyond. Far from the bound this is the chi2_1 quantile
    (1 / 4 at 68.27 / 95.45%); at the bound the one-sided Phi^-1(cl)^2 (2.71 at 95%)."""
    d = np.maximum(np.asarray(d, float), 0.0)
    q = _chi2_dist.ppf(cl, 1)
    lo, hi = d ** 2, np.full_like(d, q)
    for _ in range(80):                   # F(d^2) < cl <= F(q): bisect the second branch
        t = 0.5 * (lo + hi)
        F = ndtr(np.sqrt(t)) + ndtr((t + d ** 2) / np.maximum(2 * d, 1e-300)) - 1.0
        lo, hi = np.where(F < cl, t, lo), np.where(F < cl, hi, t)
    return np.where(d ** 2 >= q, q, 0.5 * (lo + hi))


def poi_interval(x, y, cl, bound=(-np.inf, np.inf)):
    """Confidence interval at level cl from a profile (x, chi2). bound = the POI's physical (lo, hi)
    (+-inf = open). With no bound the threshold is the chi2_1 quantile; with one, t_threshold at each
    tested value (sigma from wald_fit, distance from the bound nearer the best fit), so the interval
    becomes a one-sided limit by itself when the data sit near the bound. Returns (lo, best, hi,
    regions): regions is the UNION of the regions below threshold as a list of (lo, hi) in increasing
    x (one region for a unimodal profile), lo/hi the outer ends of that union; an end equal to the
    bound is a one-sided limit, NaN means below threshold up to the grid end (end_kinds)."""
    sp = _smooth_profile(x, y, bound)
    if sp is None:
        return np.nan, np.nan, np.nan, []
    xs, ys = sp
    best = xs[int(np.argmin(ys))]
    thr = np.full_like(xs, _chi2_dist.ppf(cl, 1))
    blo, bhi = (float(b) for b in bound)
    sig = wald_fit(x, y)[0]
    near = [(d, dist) for d, dist in (((best - blo) / sig, xs - blo), ((bhi - best) / sig, bhi - xs))
            if np.isfinite(d)]
    if near:
        thr = t_threshold(cl, min(near, key=lambda t: t[0])[1] / sig)
    regions = union_regions(xs, ys, thr, (blo, bhi))     # non-empty: ys = 0 <= thr at the minimum
    return regions[0][0], best, regions[-1][1], regions


def end_kinds(lo, hi, bound, regions):
    """('crossing' | 'bound' | 'open') for each end of an interval from poi_interval: 'bound' is a
    one-sided limit at the POI's physical bound, 'open' is below threshold up to the grid end (the
    data leave the POI open there, or the grid is too narrow); ('none', 'none') without a profile."""
    if not regions:
        return "none", "none"
    k = lambda v, b: "open" if not np.isfinite(v) else ("bound" if v == b else "crossing")
    return k(lo, float(bound[0])), k(hi, float(bound[1]))


def interval_kind(ends):
    """'two-sided', 'upper limit' (no lower crossing), 'lower limit' (no upper crossing),
    'unconstrained' or 'none' from end_kinds."""
    lo, hi = ends
    if lo == "none":
        return "none"
    return {(True, True): "two-sided", (False, True): "upper limit", (True, False): "lower limit",
            (False, False): "unconstrained"}[(lo == "crossing", hi == "crossing")]


# ======================================================================
# config loading
# ======================================================================
def _load_config(path):
    """Load the CONFIG dict from a .py file (PA_CONFIG; a relative path is taken from the cwd)."""
    spec = importlib.util.spec_from_file_location("pa_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG, os.path.abspath(path)


if not os.environ.get("PA_CONFIG"):
    raise RuntimeError("PA_CONFIG is required: the path of a config file, e.g. scan/configs/<name>.py")
CONFIG_PATH = os.environ["PA_CONFIG"]
CFG, CONFIG_ABS = _load_config(CONFIG_PATH)

# BBN likelihood (config "bbn_like", needs "bbn": "linx"): Gaussian terms on LINX's Y_p(BBN) = 4 Y_He4 and
# 1e5 D/H, with LINX's nuisances profiled like every other parameter: tau_n_fac (prior N(1, tau_n_sig)) and
# one N(0, 1) log-normal rate shift per reaction of the network (q0, q1, ...; the reaction each one scales is
# printed at start-up). They are appended to ORDER here, so every row carries them as free dims and the
# priors are part of chi2. Config: {"Yp": (mean, sigma), "DH": (mean, sigma), "tau_n_sig": sigma}.
BBN_LIKE = CFG.get("bbn_like", None)
BBN_NUIS = []                                         # the appended BBN nuisance names
if BBN_LIKE is not None:
    if str(CFG.get("bbn", "")).lower() != "linx":
        raise ValueError(f"{CONFIG_ABS}: bbn_like needs bbn = linx")
    from scan.bbn_stage import LinxStage as _LinxStage
    _nq = _LinxStage.n_reactions(CFG.get("linx_reaction_net", "key_PRIMAT_2023"))
    BBN_NUIS = ["tau_n_fac"] + [f"q{i}" for i in range(_nq)]
    # prior means of the nuisance penalties: fixed, NOT the config cen (which is only a warm start)
    BBN_PRIOR_MEAN = {k: (1.0 if k == "tau_n_fac" else 0.0) for k in BBN_NUIS}
    CFG["order"] = list(CFG["order"]) + BBN_NUIS
    for _k in BBN_NUIS:                               # a config may give warm-start values (cen) for them
        CFG["cen"].setdefault(_k, BBN_PRIOR_MEAN[_k])
        CFG["sig"][_k] = float(BBN_LIKE["tau_n_sig"]) if _k == "tau_n_fac" else 1.0
        if "grid_sig" in CFG:
            CFG["grid_sig"][_k] = CFG["sig"][_k]

ORDER = list(CFG["order"])
D = len(ORDER)
P = D - 1                                            # free dims of a profile row
CENTER = np.array([CFG["cen"][k] for k in ORDER])
SIGMA = np.array([CFG["sig"][k] for k in ORDER])
# GRID placement (center + half-span-per-sigma) is DECOUPLED from the optimizer's
# scaled-coordinate system (CENTER/SIGMA stay the config guess, so the well-conditioned
# scaling never changes mid-run). The MLE pre-pass (PA_MLE_PREPASS) overwrites these with
# the joint MLE + its Fisher errors so every POI grid is centered on the actual optimum.
GRID_CEN = CENTER.copy()
# optional config 'grid_sig': grid half-span unit per POI (else the scaling SIGMA).
GRID_SIG = np.array([CFG.get("grid_sig", CFG["sig"])[k] for k in ORDER], float)
# optional config 'bounds' {name: (lo, hi)} (physical limits, None = open) and 'box' {name: (lo, hi)} (the
# optimizer box in scaled sigmas, None = -+PA_XBOX); see the module docstring, item 4
_BOUNDS, _BOX = CFG.get("bounds", {}) or {}, CFG.get("box", {}) or {}
for _k in list(_BOUNDS) + list(_BOX):
    if _k not in ORDER:
        raise ValueError(f"{CONFIG_ABS}: bounds / box given for {_k!r}, which is not in the order {ORDER}")
BLO = np.array([-np.inf if _BOUNDS.get(k, (None, None))[0] is None else float(_BOUNDS[k][0]) for k in ORDER])
BHI = np.array([np.inf if _BOUNDS.get(k, (None, None))[1] is None else float(_BOUNDS[k][1]) for k in ORDER])
if not ((BLO <= CENTER) & (CENTER <= BHI)).all():
    raise ValueError(f"{CONFIG_ABS}: cen outside the bounds for "
                     f"{[k for i, k in enumerate(ORDER) if not BLO[i] <= CENTER[i] <= BHI[i]]}")
FIXED = dict(CFG["fixed"])
USER_SPECIES = CFG.get("user_species", None)
# ABCMB evolves a massive neutrino only when the MassiveNeutrino species is passed; with
# N_nu_massive > 0 alone it silently puts all of Neff into massless neutrinos.
if float(FIXED.get("N_nu_massive", 0)) > 0 and _species.MassiveNeutrino not in tuple(USER_SPECIES or ()):
    raise ValueError(f"{CONFIG_ABS}: fixed N_nu_massive={FIXED['N_nu_massive']} needs "
                     f"user_species=(species.MassiveNeutrino,); without it ABCMB drops the massive neutrino")

# ---------------- env config (overrides config-file defaults) ----------------
# high-ell likelihood (config high_ell): "plik_full" (Planck 2018 full plik, scan/plik_full.py) or "act_lite" (ACT DR6
# TT/TE/EE CMB-only via candl, scan/act_lite.py). Each has its engine settings (ENGINE below) and default l_max.
HIGH_ELL = str(CFG.get("high_ell", "plik_full")).lower()
LMAX = int(os.environ.get("PA_LMAX", CFG.get("l_max", 6325 if HIGH_ELL == "act_lite" else 2508)))
NPTS = int(os.environ.get("PA_NPTS", CFG.get("npts", 25)))
NSIG = float(os.environ.get("PA_NSIG", CFG.get("nsig", 3.0)))
MAXIT = int(os.environ.get("PA_MAXIT", 40))
GTOL = float(os.environ.get("PA_GTOL", 3e-2))        # ||projected grad||_inf gate (scaled coords)
# Noise floor: a stalled row whose Newton decrement 0.5 g'H^-1 g is below PA_DTOL counts as converged ("floor").
DTOL = float(os.environ.get("PA_DTOL", "1e-3"))
# Optional early stop once every POI's 68% ends moved < PA_SIGTOL * grid_sig over PA_SIGTOL_PATIENCE iters.
# Off by default: the stop is convergence (every row converged, stalled or on the noise floor). For a
# likelihood too rough for rows to converge or stall (they keep taking noise-level steps) it saves the
# iterations to PA_MAXIT; it certifies no row.
SIGTOL = float(os.environ.get("PA_SIGTOL", "0"))
SIGTOL_PATIENCE = int(os.environ.get("PA_SIGTOL_PATIENCE", "3"))
RTOL = float(os.environ.get("PA_RTOL", 1e-5))
# Save / LoS lna grid: "recomb" (nodes move smoothly with the cosmology's recombination, perturbations.py) or
# "fixed" (the visibility grid frozen at the config centre: the setup of every result before 2026-11-07).
LNA_GRID = os.environ.get("PA_LNA_GRID", "recomb")
assert LNA_GRID in ("recomb", "fixed"), LNA_GRID
TAG =os.environ.get("PA_TAG", "")
RESUME = os.environ.get("PA_RESUME", "1") != "0"
VAL_CHUNK = int(os.environ.get("PA_VAL_CHUNK", "256"))      # max cosmologies per call_batched value call
AD_BCHUNK = int(os.environ.get("PA_AD_BCHUNK", "64"))       # max rows per staged-AD call (16 per GPU ~ 9 GB)
GRAD_SPECB = int(os.environ.get("PA_GRAD_SPECB", "4")) or None  # cosmologies per device per
    # spectrum-stage jvp call (the gradient's memory peak; sub-batching it costs no extra work)
# Multimodality (lm_tools): MS_K extra starts per row at MS_NPTS grid points per POI (0 = none),
# +-MS_SPREAD profile sigmas along the softest warm Gauss-Newton directions; converged starts more
# than MS_DX (scaled sigma, max-norm) apart are distinct minima, closer ones must agree in chi2 to
# MS_DCHI2; the interval stop waits for the extra starts (MS_WAIT). Branch seeds: every distinct
# minimum a spread start found (not the warm start's one at that point), at most MS_BRANCH_MAX per
# POI, each an ordinary LM run of BR_MAXIT iterations at every grid point of its POI.
MS_K = int(os.environ.get("PA_MS_K", "3"))
MS_NPTS = int(os.environ.get("PA_MS_NPTS", "6"))
MS_SPREAD = float(os.environ.get("PA_MS_SPREAD", "2.0"))
MS_DX = float(os.environ.get("PA_MS_DX", "0.5"))
MS_DCHI2 = float(os.environ.get("PA_MS_DCHI2", "0.1"))
MS_WAIT = os.environ.get("PA_MS_WAIT", "1") != "0"
MS_BRANCH_MAX = int(os.environ.get("PA_MS_BRANCH_MAX", "4"))
BR_MAXIT = int(os.environ.get("PA_BR_MAXIT", "10"))
# MLE PRE-PASS: solve the joint D-dim fit first (same optimizer, all parameters free), recenter
# every POI grid on it with a span from its Gauss-Newton errors, and warm-start there.
MLE_PREPASS = os.environ.get("PA_MLE_PREPASS", "0") != "0"
MLE_MAXITER = int(os.environ.get("PA_MLE_MAXITER", "40"))
MLE_SIG_CLIP = float(os.environ.get("PA_MLE_SIG_CLIP", "3.0"))  # grid sigma clamped to [1/clip, clip] x prior
POI_SLICE = os.environ.get("PA_POI_SLICE", "0") != "0"   # multi-node scale-out: each
# rank owns a DISJOINT subset of POIs (POIS[RANK::NPROC]) and runs its own lockstep,
# writing only its own per-POI npz -> no shared-file clobber, no MPI gather.
XBOX = float(os.environ.get("PA_XBOX", "5.0"))         # default free-dim box (scaled sigmas)
BOXLO = np.array([-XBOX if _BOX.get(k, (None, None))[0] is None else float(_BOX[k][0]) for k in ORDER])
BOXHI = np.array([XBOX if _BOX.get(k, (None, None))[1] is None else float(_BOX[k][1]) for k in ORDER])
if not ((BOXLO <= 0) & (BOXHI >= 0) & (BOXLO < BOXHI)).all():
    raise ValueError(f"{CONFIG_ABS}: each box must contain 0 (the config cen): "
                     f"{ {k: (BOXLO[i], BOXHI[i]) for i, k in enumerate(ORDER)} }")
FIT_DEG = int(os.environ.get("PA_FIT_DEG", "4"))       # _smooth_profile polynomial degree
FIT_DCHI2 = float(os.environ.get("PA_FIT_DCHI2", "9"))  # _smooth_profile: points within this of the min
WARM_LR = os.environ.get("PA_WARM_LR", "0") != "0"     # rows start at the warm-point linear response
# Levenberg-Marquardt: STEP_MAX caps each step (max-norm, sigma units): a far row's first steps can
# otherwise reach cosmologies where an ODE solve hits max_steps. LM_K damping values per iteration,
# a decade apart around the row's current lambda; a step is taken when its actual chi2 drop is at
# least LM_ETA of the Gauss-Newton prediction. A row whose LM_K candidates all fail keeps its point
# (and its gradient) and raises its lambda 1000x, so its next ladder starts at the most-damped step that just failed; STALL_MAX such iterations in a row, or
# lambda > LM_LAMBDA_MAX, stops it as stalled.
STEP_MAX = float(os.environ.get("PA_STEP_MAX", "1.0"))
LM_K = int(os.environ.get("PA_LM_K", "4"))
LM_LADDER = 10.0 ** (np.arange(LM_K) - 1)
LM_LAMBDA0, LM_LAMBDA_MIN, LM_LAMBDA_MAX = 1e-3, 1e-7, 1e8
LM_ETA = 1e-4
STALL_MAX = int(os.environ.get("PA_STALL_MAX", "3"))
TRACE_EVALS = os.environ.get("PA_TRACE_EVALS", "0") != "0"   # log every batch before evaluating it, so
#   a run that aborts inside ABCMB leaves the offending cosmologies as its last log lines

POIS = os.environ.get("PA_POIS", ",".join(k for k in ORDER if k not in BBN_NUIS)).split(",")
POIS = [p.strip() for p in POIS if p.strip() in ORDER]
# PA_POI2D="a,b": a 2D profile of the pair (profile_2d) instead of the 1D profiles; PA_NPTS2D points per axis unless
# the config gives grid2d = {name: (lo, hi, n)}
POI2D = [p.strip() for p in os.environ.get("PA_POI2D", "").split(",") if p.strip()]
NPTS2D = int(os.environ.get("PA_NPTS2D", "11"))
# PA_GRID2D='{"name": [v, ...]}' (JSON, explicit values) overrides the config grid2d per axis, e.g. one more row
# of an existing 2D profile under a new PA_TAG (the notebook merges tags by coordinate)
GRID2D_ENV = os.environ.get("PA_GRID2D", "")
if POI2D and (len(POI2D) != 2 or POI2D[0] == POI2D[1] or not set(POI2D) <= set(ORDER)):
    raise ValueError(f"PA_POI2D needs two distinct parameters of the order {ORDER}, got {POI2D}")
RANK = int(os.environ.get("SLURM_PROCID", 0))
NPROC = int(os.environ.get("SLURM_NPROCS", 1))

# A config may carry toy = {"chi2": f(theta_D) -> chi2 (jax)}: a toy objective in place of ABCMB and the
# likelihoods (scan/configs/toy_bimodal.py, pytests/test_profile_toy.py). Only values_rows / grad_gn_rows
# change (gradient by AD, curvature = the Hessian with its eigenvalues floored, standing in for J^T W J);
# the row machinery, the phases and the outputs are the production code.
TOY = CFG.get("toy", None)
USE_LOWTT = TOY is None and bool(CFG.get("use_lowtt", True))
USE_LOWEE = TOY is None and bool(CFG.get("use_lowee", True))
LOWEE = str(CFG.get("lowee", "sroll2")).lower()     # low-ell EE table: "sroll2" or "simall" (Planck 2018 lowE)
if LOWEE not in LOWEE_TABLES:
    raise ValueError(f"{CONFIG_ABS}: lowee must be one of {sorted(LOWEE_TABLES)}, not {LOWEE!r}")
# YHe: config bbn = "linx" (the LINX stage below), "table" (ABCMB's sBBN table, YHe(omega_b) with its tangent), or
# absent (fixed YHe / ABCMB's default)
BBN_TYPE = str(CFG.get("bbn", "")).lower()
if BBN_TYPE not in ("", "linx", "table"):
    raise ValueError(f"{CONFIG_ABS}: bbn must be 'linx', 'table' or absent, not {BBN_TYPE!r}")
if TOY is None:
    if HIGH_ELL not in ("plik_full", "act_lite"):
        raise ValueError(f"{CONFIG_ABS}: high_ell must be 'plik_full' or 'act_lite', not {HIGH_ELL!r}")
    lowee = LowLEE(LOWEE_TABLES[LOWEE]) if USE_LOWEE else None
    lowtt = LowLTT() if USE_LOWTT else None
    if HIGH_ELL == "plik_full":
        # full Planck plik (clipy): its 21 floated foreground/calibration nuisances (of 47; 26 fixed) are profiled
        # per cosmology at fixed ABCMB theory (scan/plik_full.py). The low-ell likelihoods take its A_planck.
        from scan.plik_full import PlikFull, FLOAT_NAMES as HIL_NAMES
        hil = PlikFull()
        I_CAL = HIL_NAMES.index("A_planck")
        ENGINE = dict(l_max_g=12, l_max_pol_g=10)
    else:
        # ACT DR6 lite (candl): A_act and P_act profiled per cosmology at fixed theory (scan/act_lite.py). The
        # low-ell likelihoods take calibration 1 (no A_planck: ACT's "ACT" = ACT DR6 + SRoll2 lowE). Engine: lensed
        # l_max 6325 (where the kept bins' windows end), unlensed / phiphi to 6325 + 5000 = the ACT paper's CLASS
        # range (chi2 within 0.006 of it; delta_l_max 1800 is up to 0.6 off), photon hierarchy 40 / 60, HALOFIT,
        # computed j_l, lensing streamed over l (scan/diag/act_*.py, CHANGELOG 2026-11-10).
        from scan.act_lite import ACTLite, FLOAT_NAMES as HIL_NAMES
        hil = ACTLite()
        I_CAL = None
        if LMAX < hil.l_window_max:
            raise ValueError(f"l_max {LMAX} < {hil.l_window_max}, where the ACT windows end")
        ENGINE = dict(l_max_g=40, l_max_pol_g=60, delta_l_max=5000, nonlinear="halofit",
                      use_bessel_tables=False, lensing_streamed=True)
    ENGINE.update(CFG.get("engine", {}) or {})          # config overrides

    def _lowl_chi2(nu, Dtt, Dee):
        """the low-ell likelihoods at the high-ell likelihood's calibration nu[I_CAL] (plik: A_planck, which Planck
        shares across its likelihoods, cobaya dividing the low-ell D_ell by A_planck^2, so it is profiled over all
        of them), or at 1 (I_CAL None)."""
        cal = 1.0 if I_CAL is None else nu[I_CAL]
        c = 0.0
        if lowtt is not None:
            c = c + lowtt.chi2(Dtt, calib=cal)
        if lowee is not None:
            c = c + lowee.chi2(Dee, calib=cal)
        return c

    # batched inner nuisance profile of the high-ell likelihood + low-ell: (chi2 (B,), nu* (B, n_nuisances))
    _hil_prof_B = jax.jit(lambda c, dtt, dee: hil.profile_batched(c, _lowl_chi2, (dtt, dee)))
    model = Model(user_species=USER_SPECIES, output_Cl=True, l_max=LMAX, lensing=True,
                  output_Pk=False, l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=RTOL, atol_large_k_PE=RTOL * 1e-2,
                  rtol_small_k_PE=min(1e-5, RTOL), max_steps_PE=16384,
                  lna_grid_mode="recomb" if LNA_GRID == "recomb" else "visibility",
                  **({"bbn_type": "table"} if BBN_TYPE == "table" else {}), **ENGINE)
    print(f"[engine] {HIGH_ELL}: l_max {LMAX}, " + ", ".join(f"{k} {v}" for k, v in ENGINE.items())
          + (", YHe from the sBBN table" if BBN_TYPE == "table" else ""), flush=True)
else:
    lowee = lowtt = hil = _hil_prof_B = model = None
    _toy_f = TOY["chi2"]
    _toy_chi2_B = jax.jit(jax.vmap(_toy_f))
    _toy_cgh_B = jax.jit(jax.vmap(lambda t: (_toy_f(t), jax.grad(_toy_f)(t), jax.hessian(_toy_f)(t))))
# BBN stage (bbn = "linx"): LINX per row gives YHe / Neff / T_nu_massless from omega_b and Delta_Neff_init.
# Delta_Neff_init is Delta Neff at the START of BBN, not the cosmology's (scan/bbn_stage.py).
BBN = None
LINX_ONLY = set()          # LINX inputs that are not ABCMB parameters (Delta_Neff_init, log10_m_chi, BBN nuisances)
if TOY is None and BBN_TYPE == "linx":
    from scan.bbn_stage import LinxStage
    for _k in ("omega_b", "Delta_Neff_init"):
        if _k not in ORDER:
            raise ValueError(f"{CONFIG_ABS}: bbn=linx needs {_k} in order")
    for _k in ("YHe", "Neff", "N_nu_massless", "T_nu_massless"):
        if _k in FIXED:
            raise ValueError(f"{CONFIG_ABS}: bbn=linx sets {_k}; remove it from fixed")
    # config linx_chi = {"g_chi": 4.0, "boson": False}: an EM-coupled thermal relic chi of mass 10**log10_m_chi MeV
    # in the plasma (the EM-WIMP model; abcmb/linx thermo.rho_chi); log10_m_chi is then a parameter of the run
    _chi = CFG.get("linx_chi") or {}
    BBN = LinxStage(CFG.get("linx_reaction_net", "key_PRIMAT_2023"), g_chi=float(_chi.get("g_chi", 0.)),
                    chi_is_boson=bool(_chi.get("boson", False)))
    if BBN.chi != ("log10_m_chi" in ORDER):
        raise ValueError(f"{CONFIG_ABS}: log10_m_chi in order needs linx_chi with g_chi != 0, and vice versa")
    LINX_ONLY = set(BBN.in_names) - {"omega_b"}
    I_OB, I_DN = ORDER.index("omega_b"), ORDER.index("Delta_Neff_init")
    # the stage inputs that are parameters of this run (global index -> stage column); the others stay at
    # their reference values (tau_n_fac 1, q 0)
    BBN_IDX = [ORDER.index(n) for n in BBN.in_names if n in ORDER]
    BBN_COLS = [BBN.in_names.index(ORDER[i]) for i in BBN_IDX]
    print(f"[bbn] LINX stage ({CFG.get('linx_reaction_net', 'key_PRIMAT_2023')}) on one host device, vmapped in "
          f"chunks of {BBN.b_chunk}" + (f", m_chi = 10**log10_m_chi MeV (g_chi {_chi['g_chi']})" if BBN.chi else "")
          + f"; {BBN.nq} reactions" + (": BBN likelihood on, nuisances " + ", ".join(
              f"{n} = {r}" for n, r in zip(BBN_NUIS[1:], BBN.reaction_names)) + f" and tau_n_fac (sigma "
              f"{BBN_LIKE['tau_n_sig']}); data Yp {BBN_LIKE['Yp']}, DH {BBN_LIKE['DH']}" if BBN_LIKE else ""),
          flush=True)
CEN = jnp.asarray(CENTER); SIG = jnp.asarray(SIGMA)
try:
    NDEV = len(jax.devices('gpu'))
except Exception:
    NDEV = 1
DO_SHARD = NDEV > 1


# ======================================================================
# rows: BASE (N,D) physical point with the fixed dims set, FREE (N,P') free-dim indices,
# X (N,P') free dims in scaled coords x = (theta - CENTER) / SIGMA
# ======================================================================
def nuis_idx_of(poi_idx):
    """global param indices (length P) that are FREE for a row with this POI."""
    return [i for i in range(D) if i != poi_idx]


def theta_of(base, free, x):
    """physical D-vector of one row (numpy)."""
    theta = np.array(base, float)
    theta[free] = CENTER[free] + SIGMA[free] * np.asarray(x, float)
    return theta


def phys_box(free):
    """(lo, hi) of the physical bounds of free dims in scaled coords (+-inf where open)."""
    return (BLO[free] - CENTER[free]) / SIGMA[free], (BHI[free] - CENTER[free]) / SIGMA[free]


def scaled_box(free):
    """(lo, hi) of a row's free dims in scaled coords: the box (config 'box', else +-XBOX) inside the
    physical bounds."""
    plo, phi = phys_box(free)
    return np.maximum(plo, BOXLO[free]), np.minimum(phi, BOXHI[free])


def build_dict(theta, bbn_y=None):
    """physical D-vector (ORDER) -> ABCMB param dict (host floats, call_batched).
    ln10As is stored as A_s; everything else passes through by name. With the BBN stage, bbn_y =
    (YHe, Neff, T_nu_massless) of this row replaces Delta_Neff_init."""
    p = dict(FIXED)
    for i, name in enumerate(ORDER):
        if name == "ln10As":
            p["A_s"] = float(np.exp(theta[i]) / 1e10)
        elif name in LINX_ONLY:
            continue
        else:
            p[name] = float(theta[i])
    if bbn_y is not None:
        p["YHe"], p["Neff"], p["T_nu_massless"] = (float(v) for v in bbn_y[:3])
    return p


def _bbn_X(thetas):
    """(N, n_in) LINX-stage inputs of physical D-vectors: omega_b, Delta_Neff_init and whichever BBN
    nuisances are parameters of this run; the rest at their reference values."""
    th = np.asarray(thetas, float).reshape(-1, D)
    X = BBN.x_default(th[:, I_OB], th[:, I_DN])
    for i, c in zip(BBN_IDX, BBN_COLS):
        X[:, c] = th[:, i]
    return X


def _bbn_values(thetas):
    """(N, 5) [YHe, Neff, T_nu_massless, YHe_BBN, DH] of physical D-vectors through the LINX stage; None
    without it."""
    if BBN is None:
        return None
    return BBN.values(_bbn_X(thetas))


def _bbn_chi2(thetas, ys):
    """(N,) the BBN likelihood of the rows: the Gaussian terms on Y_p(BBN) and D/H plus the Gaussian priors on
    the BBN nuisances, ((theta - prior mean) / prior sigma)^2; 0 without bbn_like."""
    if BBN_LIKE is None:
        return 0.0
    th = np.asarray(thetas, float).reshape(-1, D)
    c = (((ys[:, 3] - BBN_LIKE["Yp"][0]) / BBN_LIKE["Yp"][1]) ** 2
         + ((ys[:, 4] - BBN_LIKE["DH"][0]) / BBN_LIKE["DH"][1]) ** 2)
    for k in BBN_NUIS:
        i = ORDER.index(k)
        c = c + ((th[:, i] - BBN_PRIOR_MEAN[k]) / SIGMA[i]) ** 2
    return c


# PA_LNA_GRID=fixed: the visibility grid built once at the config centre and used for every row
# (Model.freeze_lna_grid); rebuilt per cosmology, its nodes would move jaggedly with theta.
if TOY is None and LNA_GRID == "fixed":
    model = model.freeze_lna_grid(build_dict(CENTER, None if BBN is None else _bbn_values([CENTER])[0]))
    print(f"[grid] lna grid frozen at the config centre: {model.specs['n_lna_PE']} nodes from "
          f"{float(model.specs['lna_grid_nodes'][0]):.6f}", flush=True)


def _bucket(n, cap):
    """batch size for n rows: the smallest NDEV * 2^k >= n, at most cap (few compiled shapes, padding <= 2x)."""
    b = NDEV
    while b < n and b < cap:
        b *= 2
    return min(b, cap)


def _padded(n, cap, *arrs):
    """split range(n) into chunks of <= cap rows, each padded (last row repeated) to its bucket;
    yields (start, stop, padded arrays)."""
    for s in range(0, n, cap):
        e = min(s + cap, n); m = _bucket(e - s, cap)
        idx = np.concatenate([np.arange(s, e), np.full(m - (e - s), e - 1)])
        yield s, e, [a[idx] for a in arrs]


# ======================================================================
# values via call_batched (no AD): every acceptance test uses these
# ======================================================================
def _chi2_from_out(out):
    Dtt = cl_to_Dl(out.ClTT, out.l)                  # for the low-ell likelihoods (muK^2 D_l)
    Dee = cl_to_Dl(out.ClEE, out.l)
    cls2d = hil.abcmb_cls_to_clik(out.ClTT, out.ClTE, out.ClEE, out.l)   # (B, ..., Lcol) its theory block
    # high-ell + low-ell, the high-ell nuisances (plik's A_planck shared with low-ell) profiled
    chi2 = np.asarray(_hil_prof_B(cls2d, Dtt, Dee)[0], dtype=float)
    return np.where(np.isfinite(chi2), chi2, 1e6)


def _trace(tag, BASE, FREE, X):
    if TRACE_EVALS:
        print(f"[eval] {tag} BEGIN {time.strftime('%H:%M:%S')} N={len(X)} ({' '.join(ORDER)})", flush=True)
        for b in range(len(X)):
            print(f"[eval]   row{b:3d} " + " ".join(f"{v:.7g}" for v in theta_of(BASE[b], FREE[b], X[b])),
                  flush=True)


def _toy_values(BASE, FREE, X):
    th = np.array([theta_of(BASE[b], FREE[b], X[b]) for b in range(len(X))])
    return np.array(_toy_chi2_B(jnp.asarray(th)), dtype=float)      # a copy: lm_rows updates it in place


def _toy_gn(BASE, FREE, X):
    """chi2, gradient and a PSD curvature of the toy objective in each row's scaled free coordinates:
    the Hessian with its eigenvalues floored stands in for the Gauss-Newton matrix (PSD like J^T W J,
    so a saddle is as invisible to it as to the production curvature)."""
    N, Pp = X.shape
    th = np.array([theta_of(BASE[b], FREE[b], X[b]) for b in range(N)])
    c, g, H = (np.array(v, dtype=float) for v in _toy_cgh_B(jnp.asarray(th)))
    G = np.empty((N, Pp)); HH = np.empty((N, Pp, Pp))
    for b in range(N):
        f = FREE[b]; s = SIGMA[f]
        G[b] = g[b, f] * s
        Hb = H[b][np.ix_(f, f)] * s[:, None] * s[None, :]
        ev, V = np.linalg.eigh(0.5 * (Hb + Hb.T))
        HH[b] = (V * np.maximum(ev, 1e-6 * max(float(ev.max()), 1e-12))) @ V.T
    return c, G, HH


def values_rows(BASE, FREE, X):
    """(N,) total chi^2 (profiled + low-ell) of the rows, through call_batched in bucketed chunks of
    <= VAL_CHUNK cosmologies (the per-device working set is set by the chunk, not by N)."""
    if TOY is not None:
        return _toy_values(BASE, FREE, X)
    N = len(X); out = np.empty(N)
    _trace("value", BASE, FREE, X)
    for s, e, (b, f, x) in _padded(N, VAL_CHUNK, BASE, FREE, X):
        th = [theta_of(b[k], f[k], x[k]) for k in range(len(x))]
        ys = _bbn_values(th)
        batch = [build_dict(th[k], None if ys is None else ys[k]) for k in range(len(x))]
        c = _chi2_from_out(model.call_batched(batch, shard=DO_SHARD))
        if ys is not None:
            c = c + _bbn_chi2(th, ys)
        out[s:e] = np.where(np.isfinite(c), c, 1e6)[:e - s]     # a failed solve (CMB or BBN) is rejected
    if TRACE_EVALS:
        print(f"[eval] value END {time.strftime('%H:%M:%S')} chi2 " + " ".join(f"{v:.3f}" for v in out), flush=True)
    return out


# ======================================================================
# gradient + Gauss-Newton curvature (AD only)
# ======================================================================
def _phys_to_derived(thD, bbn=None):
    """physical D-vector -> derived param dict, _to_float'd so the jvp tangent's
    inexact-array tree matches staged_cl_and_grad's _to_float'd primal. With the BBN stage, bbn =
    (y, J, th0) of the row: YHe / Neff / T_nu_massless enter as y + J (thD - th0)[omega_b, Delta_Neff_init],
    exact at th0 with the exact first derivative, so the jvp carries LINX's tangent without re-running it."""
    p = dict(FIXED)
    for i, name in enumerate(ORDER):
        if name == "ln10As":
            p["A_s"] = jnp.exp(thD[i]) / 1e10
        elif name in LINX_ONLY:
            continue
        else:
            p[name] = thD[i]
    if bbn is not None:
        y, J, th0 = bbn
        dx = jnp.stack([thD[i] - th0[i] for i in BBN_IDX])
        lin = jnp.asarray(y)[:3] + jnp.asarray(J)[:3][:, BBN_COLS] @ dx
        p["YHe"], p["Neff"], p["T_nu_massless"] = lin[0], lin[1], lin[2]
    return _bg_to_float(model.add_derived_parameters(p))


def _schur(Htt, Htn, Hnn, free):
    """Curvature of chi2 profiled over the nuisances nu at their optimum: H_tt - H_tn H_nn^-1 H_nt over
    the free nuisances (free (Q,) False for one sitting on a bound, held fixed). The gradient needs no
    correction (envelope theorem: d chi2/d nu = 0 at the optimum)."""
    m = free.astype(Htt.dtype)
    Htn = Htn * m[None, :]
    Hnn = Hnn * m[:, None] * m[None, :] + jnp.diag(1.0 - m)
    H = Htt - Htn @ jnp.linalg.solve(Hnn, Htn.T)
    return 0.5 * (H + H.T)


def _lowl_gn(like, D_, dD, A):
    """chi2, gradient and generalized Gauss-Newton curvature dD^T H dD of a low-ell likelihood in
    the D_ell it reads (ell = lmin..lmax) at calibration A: H is its exact AD Hessian there, with
    negative eigenvalues set to zero so the term stays positive semidefinite. Also its coupling to A
    (dD^T d2chi2/dD dA, (P',)) and d2chi2/dA2, for the Schur complement over the plik nuisances."""
    sl = slice(like.lmin, like.lmax + 1)
    f = lambda d, a: like.chi2(D_.at[sl].set(d), calib=a)
    d0 = D_[sl]
    c, gD = jax.value_and_grad(f)(d0, A)
    ev, V = jnp.linalg.eigh(jax.hessian(f)(d0, A))
    HD = (V * jnp.maximum(ev, 0.0)) @ V.T
    hDA = jax.jacfwd(jax.grad(f), argnums=1)(d0, A)
    hAA = jax.hessian(f, argnums=1)(d0, A)
    Jd = dD[:, sl]
    return c, Jd @ gD, Jd @ HD @ Jd.T, Jd @ hDA, hAA


def _gn_one(ClTT, ClTE, ClEE, dTT, dTE, dEE, nu):
    """total chi2, gradient (P',) and Gauss-Newton matrix (P',P') of ONE cosmology from its C_ell
    (n_l,) and C_ell tangents (P', n_l); nu = its high-ell nuisance optimum."""
    lvec = model.SS.ells
    cls2d = hil.abcmb_cls_to_clik(ClTT, ClTE, ClEE, lvec)                   # theory block (..., Lcol)
    dcls = hil.abcmb_cls_to_clik(dTT, dTE, dEE, lvec)                       # (P', ..., Lcol), linear
    res = hil.residual
    r = res(cls2d, nu)
    Jt = jax.vmap(lambda dc: jax.jvp(lambda c2: res(c2, nu), (cls2d,), (dc,))[1])(dcls)
    Jn = jax.jacfwd(lambda n: res(cls2d, n))(nu).T
    free = (nu - hil.lo > 1e-6 * hil.scale) & (hil.hi - nu > 1e-6 * hil.scale)
    W = hil.siginv
    c = r @ W @ r + hil.prior_penalty(nu)
    WJt = Jt @ W
    g = 2.0 * WJt @ r
    Htt = 2.0 * WJt @ Jt.T
    Htn = 2.0 * WJt @ Jn.T
    Hnn = 2.0 * (Jn @ W) @ Jn.T + hil.prior_hessian()
    for like, C, dC in ((lowtt, ClTT, dTT), (lowee, ClEE, dEE)):     # low-ell at the shared calibration
        if like is not None:
            cal = 1.0 if I_CAL is None else nu[I_CAL]
            cl_, gl, Hl, hA, hAA = _lowl_gn(like, cl_to_Dl(C, lvec), cl_to_Dl(dC, lvec), cal)
            c, g, Htt = c + cl_, g + gl, Htt + Hl
            if I_CAL is not None:
                Htn = Htn.at[:, I_CAL].add(hA)
                Hnn = Hnn.at[I_CAL, I_CAL].add(hAA)
    return c, g, _schur(Htt, Htn, Hnn, free)


_gn_B = jax.jit(jax.vmap(_gn_one))


def _gn_block(BASE, FREE, X):
    """One staged AD call over a block of rows: (chi2 (B,), grad (B,P'), GN (B,P',P')). Direction j
    of row b is SIGMA[i] e_i, i = FREE[b, j] (so gradient and curvature are in scaled coords)."""
    B, Pp = X.shape
    th_np = np.array([theta_of(BASE[b], FREE[b], X[b]) for b in range(B)])
    thetas = [jnp.asarray(t) for t in th_np]
    bbn = [None] * B
    if BBN is not None:                            # one batched LINX call (values + Jacobians) per block
        n0 = BBN.n_fallback; t0 = time.perf_counter()
        ys, Js = BBN.values_and_jac(_bbn_X(th_np))
        if TRACE_EVALS:
            print(f"[eval] bbn jac {B} rows {time.perf_counter() - t0:.0f}s, {BBN.n_fallback - n0} lanes "
                  f"recomputed without vmap", flush=True)
        bbn = [(ys[b], Js[b], th_np[b]) for b in range(B)]
    full_ps = [_phys_to_derived(t, bb) for t, bb in zip(thetas, bbn)]
    per = []                                       # [B][P'] filtered tangent dicts
    for b in range(B):
        dots = []
        for j in range(Pp):
            i = int(FREE[b, j])
            _, fd = jax.jvp(lambda t: _phys_to_derived(t, bbn[b]), (thetas[b],),
                            (jnp.zeros(D).at[i].set(SIG[i]),))
            dots.append(eqx.filter(fd, eqx.is_inexact_array))
        per.append(dots)
    params_dots = [jax.tree.map(lambda *xs: jnp.stack(xs), *[per[b][j] for b in range(B)])
                   for j in range(Pp)]
    (ClTT, ClTE, ClEE), cl_dots = staged_cl_and_grad(
        model, full_ps, params_dots, k_chunk_size=100, shard=DO_SHARD, spec_b_local=GRAD_SPECB)
    dTT, dTE, dEE = (jnp.stack([cd[s] for cd in cl_dots], axis=1) for s in range(3))  # (B,P',n_l)
    nu = _hil_prof_B(hil.abcmb_cls_to_clik(ClTT, ClTE, ClEE, model.SS.ells),
                     cl_to_Dl(ClTT, model.SS.ells), cl_to_Dl(ClEE, model.SS.ells))[1]
    c, g, H = _gn_B(ClTT, ClTE, ClEE, dTT, dTE, dEE, nu)
    c, g, H = np.array(c, float), np.array(g, float), np.array(H, float)   # copies: written below
    if BBN_LIKE is not None:                       # the BBN likelihood: its residuals' Gauss-Newton terms
        # and the nuisance priors, in the rows' scaled coordinates (direction j = SIGMA[i] e_i)
        c = c + _bbn_chi2(th_np, ys)
        col_of = dict(zip(BBN_IDX, BBN_COLS))
        sY, sDH = BBN_LIKE["Yp"][1], BBN_LIKE["DH"][1]
        for b in range(B):
            r = np.array([(ys[b, 3] - BBN_LIKE["Yp"][0]) / sY, (ys[b, 4] - BBN_LIKE["DH"][0]) / sDH])
            Jr = np.zeros((2, Pp))
            for j in range(Pp):
                i = int(FREE[b, j])
                if i in col_of:
                    Jr[0, j] = Js[b, 3, col_of[i]] * SIGMA[i] / sY
                    Jr[1, j] = Js[b, 4, col_of[i]] * SIGMA[i] / sDH
                if ORDER[i] in BBN_NUIS:
                    xj = (th_np[b, i] - BBN_PRIOR_MEAN[ORDER[i]]) / SIGMA[i]
                    g[b, j] += 2.0 * xj; H[b, j, j] += 2.0
            g[b] += 2.0 * Jr.T @ r; H[b] += 2.0 * Jr.T @ Jr
    return c, g, H


def grad_gn_rows(BASE, FREE, X):
    """(chi2 (N,), grad (N,P'), GN (N,P',P')) for every row, in bucketed blocks of <= AD_BCHUNK."""
    if TOY is not None:
        return _toy_gn(BASE, FREE, X)
    N, Pp = X.shape
    c = np.empty(N); G = np.empty((N, Pp)); H = np.empty((N, Pp, Pp))
    _trace("adgrad", BASE, FREE, X)
    for s, e, (b, f, x) in _padded(N, AD_BCHUNK, BASE, FREE, X):
        cb, gb, hb = _gn_block(b, f, x)
        c[s:e], G[s:e], H[s:e] = cb[:e - s], gb[:e - s], hb[:e - s]
    if TRACE_EVALS:
        print(f"[eval] adgrad END {time.strftime('%H:%M:%S')} |g|max " +
              " ".join(f"{v:.3g}" for v in np.abs(G).max(1)), flush=True)
    return c, G, H


# ======================================================================
# Levenberg-Marquardt over the row set (lockstep), inside the +-XBOX box
# ======================================================================
def _row_dec(G, H, X, XLO, XHI):
    """(N,) Newton decrement of every row (lm_tools.newton_decrement)."""
    return np.array([newton_decrement(G[b], H[b], X[b], XLO[b], XHI[b]) for b in range(len(X))])


def _envelope(pv, f):
    """(unique POI values, lowest chi2 at each): the profile of a row set with several starts per
    grid point."""
    u, inv = np.unique(np.asarray(pv, float), return_inverse=True)
    m = np.full(len(u), np.inf)
    np.minimum.at(m, inv, np.asarray(f, float))
    return u, m


def lm_rows(BASE, FREE, XLO, XHI, X0, POI_IDX=None, PV=None, maxit=MAXIT, gtol=None,
            log_prefix="", ckpt_path=None, resume_state=None, EXTRA=None):
    """Levenberg-Marquardt over a flat row set. BASE (N,D), FREE (N,P'), XLO/XHI/X0 (N,P'); POI_IDX/PV
    (N,) group the rows into profiles for the interval-stability stop (None: no early stop); several
    rows at one PV are starts of the same point and the profile takes their lowest chi2. EXTRA (N,)
    marks the multistart rows: with MS_WAIT the stop waits until none is active. Returns a dict of
    per-row arrays: X, F (values-path chi2), G, H (at X), gnorm (projected, exact AD), stalled, floor
    (stopped on the noise floor: a stall with Newton decrement < DTOL; not in stalled), lam, it, and stop
    (why the loop ended: "converged" | "sigtol" | "maxit"). The full
    state is checkpointed every iteration (ckpt_path); resume_state restarts from such a checkpoint.
    gtol None = GTOL at call time."""
    gtol = GTOL if gtol is None else gtol
    N, Pp = X0.shape
    if resume_state is not None:
        S = {k: np.array(resume_state[k]) for k in ("X", "F", "G", "H", "lam", "nfail", "stalled", "floor")}
        it0 = int(resume_state["it"])
        print(f"  {log_prefix} RESUME from it{it0}: min={S['F'].min():.2f}", flush=True)
    else:
        X = np.array(X0, float)
        F = values_rows(BASE, FREE, X)
        _, G, H = grad_gn_rows(BASE, FREE, X)
        bad = ~(np.isfinite(G).all(1) & np.isfinite(H).all((1, 2)))
        if bad.any():
            print(f"  {log_prefix} non-finite gradient/curvature at the start of rows "
                  f"{np.where(bad)[0].tolist()} -> stalled", flush=True)
        S = dict(X=X, F=F, G=np.where(bad[:, None], 0.0, G), H=np.where(bad[:, None, None], 0.0, H),
                 lam=np.full(N, LM_LAMBDA0), nfail=np.zeros(N, int), stalled=bad, floor=np.zeros(N, bool))
        it0 = 0
    X, F, G, H, lam, nfail, stalled, floor = (S[k] for k in ("X", "F", "G", "H", "lam", "nfail", "stalled",
                                                            "floor"))
    gproj = proj_grad(G, X, XLO, XHI)
    groups = {} if POI_IDX is None else {int(p): np.where(POI_IDX == p)[0] for p in np.unique(POI_IDX)}
    def _ends():
        out = {}
        for p, r in groups.items():
            pv, f = _envelope(PV[r], F[r])
            lo, _, hi, reg = poi_interval(pv, f, CL1, (BLO[p], BHI[p]))
            out[p] = (lo, hi, len(reg))
        return out
    hist = [_ends()]
    it = it0
    stop = "maxit"
    for it in range(it0, maxit):
        act = (np.abs(gproj).max(1) > gtol) & ~stalled & ~floor
        if not act.any():
            stop = "converged"
            break
        A = np.where(act)[0]
        cand = np.empty((LM_K, len(A), Pp)); pred = np.empty((LM_K, len(A)))
        for k in range(LM_K):
            for a, b in enumerate(A):
                d, pred[k, a] = lm_step(G[b], H[b], X[b], XLO[b], XHI[b], lam[b] * LM_LADDER[k], STEP_MAX)
                cand[k, a] = X[b] + d
        Fc = values_rows(np.tile(BASE[A], (LM_K, 1)), np.tile(FREE[A], (LM_K, 1)),
                         cand.reshape(-1, Pp)).reshape(LM_K, len(A))
        drop = F[A][None] - Fc
        ok = np.isfinite(Fc) & (drop > 0) & (drop >= LM_ETA * pred)
        kb = np.where(ok, Fc, np.inf).argmin(0)
        got = ok.any(0); ia = np.where(got)[0]
        moved, failed = A[got], A[~got]
        Xold, Fold = X[moved].copy(), F[moved].copy()
        X[moved] = cand[kb[ia], ia]; F[moved] = Fc[kb[ia], ia]
        lam[moved] = np.maximum(lam[moved] * LM_LADDER[kb[ia]], LM_LAMBDA_MIN)
        lam[failed] *= LM_LADDER[-1] * 10.0
        nfail[moved] = 0; nfail[failed] += 1
        if len(moved):
            _, Gm, Hm = grad_gn_rows(BASE[moved], FREE[moved], X[moved])
            fin = np.isfinite(Gm).all(1) & np.isfinite(Hm).all((1, 2))
            if not fin.all():                      # step back; treat as a failed step
                bb = moved[~fin]
                print(f"  {log_prefix} non-finite gradient at rows {bb.tolist()}: step undone", flush=True)
                X[bb], F[bb] = Xold[~fin], Fold[~fin]
                lam[bb] *= LM_LADDER[-1] * 10.0; nfail[bb] += 1
            G[moved[fin]], H[moved[fin]] = Gm[fin], Hm[fin]
        new = ((nfail >= STALL_MAX) | (lam > LM_LAMBDA_MAX)) & ~stalled & ~floor
        if new.any():                              # a stall with no gain left to find is on the noise floor
            nb = np.where(new)[0]
            low = _row_dec(G[nb], H[nb], X[nb], XLO[nb], XHI[nb]) < DTOL
            floor[nb[low]] = True; stalled[nb[~low]] = True
        gproj = proj_grad(G, X, XLO, XHI)
        if log_prefix:
            print(f"  {log_prefix} it{it}: min={F.min():.2f} ||g||max(active)="
                  f"{np.abs(gproj[act]).max():.2e} active={len(A)} moved={len(moved)} "
                  f"stalled={int(stalled.sum())} floor={int(floor.sum())} lam_med={np.median(lam[A]):.1e} "
                  f"({time.strftime('%H:%M:%S')})", flush=True)
        if ckpt_path:
            tmp = ckpt_path + ".tmp.npz"
            np.savez(tmp, BASE=BASE, FREE=FREE, X0=X0, X=X, F=F, G=G, H=H, lam=lam, nfail=nfail,
                     stalled=stalled, floor=floor, it=it + 1, gtol=gtol, run_sig=_run_signature())
            os.replace(tmp, ckpt_path)
        # ---- interval-stability early stop: every POI's 68% interval ends moved by < SIGTOL*grid_sig
        # over the window. A POI with active rows that took no step stalled rather than converged:
        # restart its window. Ends that stay NaN (off-grid) do not block. ----
        if not groups:
            continue
        stuck = [ORDER[p] for p, r in groups.items()
                 if act[r].any() and not np.isin(r, moved).any()]
        if stuck:
            print(f"  {log_prefix} no accepted step for {stuck} at it{it}; window restarted", flush=True)
            hist = [_ends()]
        else:
            hist.append(_ends())
        if SIGTOL > 0 and len(hist) > SIGTOL_PATIENCE:
            prev, cur = hist[-1 - SIGTOL_PATIENCE], hist[-1]
            worst, ready, nfin = 0.0, True, 0
            for p in groups:
                if prev[p][2] != cur[p][2]:                # the number of regions changed
                    ready = False
                for a, b in zip(prev[p][:2], cur[p][:2]):
                    if np.isfinite(a) and np.isfinite(b):
                        worst = max(worst, abs(b - a) / GRID_SIG[p]); nfin += 1
                    elif np.isfinite(a) != np.isfinite(b):
                        ready = False
            if ready and nfin and worst < SIGTOL:
                nx = 0 if EXTRA is None or not MS_WAIT else \
                    int(((np.abs(gproj).max(1) > gtol) & ~stalled & ~floor & EXTRA).sum())
                if nx:
                    print(f"  {log_prefix} intervals stable (worst move {worst:.2e} grid_sig) but "
                          f"{nx} extra starts still active -> continuing", flush=True)
                else:
                    print(f"  {log_prefix} intervals stable: worst move {worst:.2e} grid_sig < SIGTOL "
                          f"{SIGTOL:.0e} over {SIGTOL_PATIENCE} iters -> stop at it{it}", flush=True)
                    stop = "sigtol"
                    break
    nact = int(((np.abs(gproj).max(1) > gtol) & ~stalled & ~floor).sum())
    if stop == "maxit" and nact == 0:                 # the last iteration finished the rows
        stop = "converged"
    if log_prefix:
        why = {"converged": "no active row (each one converged, stalled, on the noise floor or held by the box)",
               "sigtol": f"68% intervals stable (PA_SIGTOL {SIGTOL:.0e}) with {nact} rows active",
               "maxit": f"MAXIT {maxit} reached with {nact} rows active"}[stop]
        print(f"  {log_prefix} stop at it{it}: {why}", flush=True)
    return dict(X=X, F=F, G=G, H=H, gnorm=np.abs(gproj).max(1), stalled=stalled, floor=floor, lam=lam, it=it,
                stop=stop)


# ======================================================================
# joint fit (MLE pre-pass) and the warm point
# ======================================================================
def _fisher_grid_sigma(H):
    """Per-parameter profile 1sigma from the joint scaled Gauss-Newton matrix H (= d2chi2/dxs2):
    cov = 2 H^-1, physical sigma_i = sqrt(2 (H^-1)_ii) SIGMA_i, clamped to [1/MLE_SIG_CLIP,
    MLE_SIG_CLIP] x SIGMA."""
    ev, V = np.linalg.eigh(0.5 * (H + H.T))
    Hinv = (V * (1.0 / np.maximum(ev, 1e-6))) @ V.T
    sig_scaled = np.sqrt(np.maximum(2.0 * np.diag(Hinv), 1e-12))
    return SIGMA * np.clip(sig_scaled, 1.0 / MLE_SIG_CLIP, MLE_SIG_CLIP)


def joint_fit(theta_start, maxit):
    """All D parameters free from theta_start: (theta, chi2, scaled GN matrix (D,D)) at the result.
    maxit=0 just evaluates the curvature at theta_start."""
    BASE = np.array(CENTER)[None]; FREE = np.arange(D)[None]
    xlo, xhi = scaled_box(FREE[0])
    X0 = np.clip((np.asarray(theta_start, float) - CENTER) / SIGMA, xlo, xhi)[None]
    if maxit == 0:
        c, _, H = grad_gn_rows(BASE, FREE, X0)
        return theta_of(BASE[0], FREE[0], X0[0]), float(c[0]), H[0]
    S = lm_rows(BASE, FREE, xlo[None], xhi[None], X0, maxit=maxit, log_prefix="[mle]")
    return theta_of(BASE[0], FREE[0], S["X"][0]), float(S["F"][0]), S["H"][0]


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
# lockstep driver across ALL POIs: base + multistart rows, flags, branches, envelope
# ======================================================================
KINDS = ("warm", "spread", "branch")      # row kinds, T["KIND"] = index into this


def _grid_for(poi):
    """NPTS points over GRID_CEN +- NSIG*GRID_SIG, inside the POI's physical bounds: a span that crosses a
    bound is shifted to start on it, keeping its width (the points past the bound go where a one-sided
    limit needs them), then clipped if the bounds are narrower than the span."""
    i = ORDER.index(poi)
    lo, hi = GRID_CEN[i] - NSIG * GRID_SIG[i], GRID_CEN[i] + NSIG * GRID_SIG[i]
    if lo < BLO[i]:
        lo, hi = BLO[i], hi + (BLO[i] - lo)
    if hi > BHI[i]:
        lo, hi = lo - (hi - BHI[i]), BHI[i]
    return np.linspace(max(lo, BLO[i]), min(hi, BHI[i]), NPTS)


def _ckpt_path(outdir, what):
    return os.path.join(outdir, f"profile_prod_ad_{what}{TAG}{'_r%d' % RANK if POI_SLICE else ''}.npz")


def _stage_rank(stage):
    """lm -> 0, branch -> 1 (the order of the phases)."""
    return 1 if stage.startswith("branch") else 0


def _run_signature():
    """What makes two runs the same run: the config file's bytes and the knobs that set the chi2 surface,
    the grids and the convergence test (not the code version). Stored in every npz and checkpoint;
    resuming or skipping a finished POI requires a match, so a changed setup under an old PA_TAG stops
    instead of silently skipping or mixing."""
    import hashlib
    h = hashlib.sha1(open(CONFIG_ABS, "rb").read())
    knobs = dict(LMAX=LMAX, RTOL=RTOL, NPTS=NPTS, NSIG=NSIG, XBOX=XBOX, GTOL=GTOL, DTOL=DTOL, MS_K=MS_K,
                 MS_NPTS=MS_NPTS, MS_SPREAD=MS_SPREAD, PLF_MAXIT=os.environ.get("PLF_MAXIT", "3000"))
    if LNA_GRID != "fixed":                 # results made before PA_LNA_GRID keep their signature
        knobs["LNA_GRID"] = LNA_GRID
    if POI2D:
        knobs["NPTS2D"] = NPTS2D
        if GRID2D_ENV:
            knobs["GRID2D"] = GRID2D_ENV
    h.update(repr(sorted(knobs.items())).encode())
    return h.hexdigest()[:16]


def _same_run(st):
    """does a loaded npz / checkpoint carry this run's signature?"""
    return "run_sig" in st.files and str(st["run_sig"]) == _run_signature()


def _save_table(path, T, stage):
    """checkpoint the row table after a phase."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, stage=stage, gtol=GTOL, run_sig=_run_signature(), **T)
    os.replace(tmp, path)


def _load_table(path, BASE, FREE):
    """(row table, stage) from a checkpoint whose base rows (the first len(BASE)) match this run's,
    else (None, None)."""
    if not (RESUME and os.path.exists(path)):
        return None, None
    try:
        st = np.load(path)
        n = len(BASE)
        if (st["BASE"].shape[0] >= n and np.allclose(st["BASE"][:n], BASE)
                and np.array_equal(st["FREE"][:n], FREE) and _same_run(st)):
            T = {k: st[k] for k in st.files if k not in ("stage", "gtol", "run_sig")}
            print(f"[resume] row table after stage {st['stage']} matches; continuing ({path})", flush=True)
            return T, str(st["stage"])
        print("[resume] row table from a different grid or setup (run_sig) -> fresh start", flush=True)
    except Exception as ex:
        print(f"[resume] could not load the row table ({ex}) -> fresh start", flush=True)
    return None, None


def _append(T, **rows):
    for k, v in rows.items():
        T[k] = np.concatenate([T[k], np.asarray(v, dtype=T[k].dtype)])


def _row_conv(T):
    """(converged, at_box) per row: converged = ||g||_AD < GTOL (projected) or stopped on the noise floor,
    not stalled and not held by the box. at_box = held by the box rather than the likelihood: a free dim on
    a box face (config 'box' / +-XBOX; a physical bound is a real limit and never counts) with chi2 still
    falling outward by more than GTOL per scaled sigma. On a face where chi2 is flat outward (a direction
    the data leave open) the row is converged."""
    plo, phi = phys_box(T["FREE"])
    X, G = T["X"], T["G"]
    on_lo = (X <= T["XLO"] + 1e-9) & (T["XLO"] > plo)
    on_hi = (X >= T["XHI"] - 1e-9) & (T["XHI"] < phi)
    at_box = (on_lo & (G > GTOL)).any(1) | (on_hi & (-G > GTOL)).any(1)
    return ((T["gnorm"] < GTOL) | T["floor"]) & ~at_box & ~T["stalled"], at_box


def _flags(T, pois_idx):
    """{(poi index, j): (flag, reason, reps, rows)} over the grid points: lm_tools.flag_point on the
    point's rows; reps = row indices of its distinct converged minima, lowest chi2 first."""
    conv, _ = _row_conv(T)
    out = {}
    for p in pois_idx:
        for j in range(NPTS):
            r = np.where((T["POI"] == p) & (T["J"] == j))[0]
            f, why, reps = flag_point(T["X"][r], T["F"][r], conv[r], T["stalled"][r], MS_DX, MS_DCHI2)
            out[(p, j)] = (f, why, [int(r[i]) for i in reps], r)
    return out


def profile_lockstep(pois, outdir):
    """Build the row set (every POI x grid point, plus MS_K spread starts at MS_NPTS points per POI),
    warm-start, run LM in lockstep, then flags -> branches, and write one npz/png per POI from the
    lower envelope over every row. Each phase checkpoints the row table (resumable)."""
    global GRID_CEN, GRID_SIG
    theta_warm = CENTER.copy(); Hwarm = [None]
    if MLE_PREPASS:
        theta_warm, chi2_mle, Hw = joint_fit(CENTER, MLE_MAXITER); Hwarm[0] = Hw
        GRID_CEN = theta_warm.copy(); GRID_SIG = _fisher_grid_sigma(Hw)
        print("[mle] recentered grids on the joint fit (chi2_min=%.3f):" % chi2_mle, flush=True)
        for i, nm in enumerate(ORDER):
            print(f"[mle]   {nm:>10}: cen {CENTER[i]:.5g} -> {GRID_CEN[i]:.5g}; grid-sig "
                  f"{GRID_SIG[i]:.3g}", flush=True)

    def _hwarm():
        if Hwarm[0] is None:
            _, _, Hwarm[0] = joint_fit(theta_warm, 0)
        return Hwarm[0]

    # ---- the base rows: one per (POI, grid point) ----
    pois_idx = [ORDER.index(p) for p in pois]
    rows_poi, rows_j, rows_pv, grid_of = [], [], [], {}
    for i in pois_idx:
        grid = _grid_for(ORDER[i]); grid_of[i] = grid
        rows_poi += [i] * NPTS; rows_j += list(range(NPTS)); rows_pv += list(grid)
    POI_IDX = np.array(rows_poi, int); J = np.array(rows_j, int); PV = np.array(rows_pv, float); N = len(PV)
    FREE = np.array([nuis_idx_of(i) for i in POI_IDX], int)
    BASE = np.tile(CENTER, (N, 1)); BASE[np.arange(N), POI_IDX] = PV
    XLO, XHI = map(np.array, zip(*[scaled_box(f) for f in FREE]))
    xw = (theta_warm - CENTER) / SIGMA
    print(f"[lockstep] N={N} base rows ({len(pois)} POIs x {NPTS}) P={P} D={D} shard={DO_SHARD} "
          f"warm={'joint fit' if MLE_PREPASS else 'config centre'}", flush=True)
    t0 = time.perf_counter()
    tpath = _ckpt_path(outdir, "TABLE")
    T, stage = _load_table(tpath, BASE, FREE)

    def _done_line(what):
        conv, at_box = _row_conv(T)
        print(f"[lockstep] {what} {time.perf_counter() - t0:.0f}s RSS={rss_gb():.1f}GB; rows "
              f"{len(T['PV'])} (" + ", ".join(f"{k} {int((T['KIND'] == i).sum())}" for i, k in enumerate(KINDS))
              + f"); converged {int(conv.sum())} (||g||_AD<{GTOL:.1e}, or a stall with 0.5g'H^-1g<{DTOL:.0e}: "
              f"{int((conv & T['floor']).sum())}); stalled {int(T['stalled'].sum())}; "
              f"held by the box {int(at_box.sum())}; max||g||_AD={T['gnorm'].max():.2e}; "
              f"lockstep stop: {str(T.get('lock_stop', '?'))}", flush=True)

    # ================= phase 1: lockstep LM over the base + spread rows =================
    if T is None:
        # the multistart rows copy their base row's structure; the spread itself needs X0 (below)
        KIND = np.zeros(N, int); START = np.zeros(N, int); ext = np.zeros(0, int); rep = np.zeros(0, int)
        if MS_K > 0:
            jsel = sorted(set(np.round(np.linspace(0, NPTS - 1, min(MS_NPTS, NPTS))).astype(int).tolist()))
            ext = np.array([b for b in range(N) if J[b] in jsel], int)
            rep = np.repeat(ext, MS_K)
            KIND = np.concatenate([KIND, np.ones(len(rep), int)])
            START = np.concatenate([START, np.tile(np.arange(1, MS_K + 1), len(ext))])
        BASE_f, FREE_f, XLO_f, XHI_f, POI_f, J_f, PV_f = (np.concatenate([a, a[rep]]) for a in
                                                         (BASE, FREE, XLO, XHI, POI_IDX, J, PV))
        Nf = len(PV_f)
        ckpt = _ckpt_path(outdir, "STATE")
        resume_state = None
        if RESUME and os.path.exists(ckpt):
            try:
                st = np.load(ckpt)
                if (st["BASE"].shape == BASE_f.shape and np.allclose(st["BASE"], BASE_f)
                        and np.array_equal(st["FREE"], FREE_f) and _same_run(st)):
                    resume_state = {k: st[k] for k in st.files}
                    print(f"[resume] LM checkpoint @ it{int(st['it'])} matches; continuing ({ckpt})", flush=True)
                else:
                    print("[resume] LM checkpoint from a different grid or setup (run_sig) -> fresh start", flush=True)
            except Exception as ex:
                print(f"[resume] could not load the LM checkpoint ({ex}) -> fresh start", flush=True)
        if resume_state is not None:
            X0_f = np.array(resume_state["X0"], float)
        else:
            X0 = np.clip(xw[FREE], XLO, XHI)
            # ---- linear-response warm start: each row starts at the Gaussian conditional optimum of
            # the warm point's Gauss-Newton matrix (rows far along a degeneracy need not walk there at
            # STEP_MAX per iteration); one batched value call scores both starts, each row keeps the
            # lower chi2. ----
            if WARM_LR:
                Hw = _hwarm(); Xlr = X0.copy()
                for b in range(N):
                    i, f = POI_IDX[b], FREE[b]
                    dpoi = (PV[b] - theta_warm[i]) / SIGMA[i]
                    Xlr[b] = np.clip(xw[f] - np.linalg.lstsq(Hw[np.ix_(f, f)], Hw[f, i] * dpoi,
                                                             rcond=None)[0], XLO[b], XHI[b])
                fk = values_rows(np.tile(BASE, (2, 1)), np.tile(FREE, (2, 1)), np.concatenate([X0, Xlr]))
                use = fk[N:] < fk[:N]
                X0 = np.where(use[:, None], Xlr, X0)
                print(f"[warm-lr] linear-response start kept for {int(use.sum())}/{N} rows; chi2 median "
                      f"{np.median(fk[:N]):.1f} -> {np.median(np.minimum(fk[:N], fk[N:])):.1f}", flush=True)
            # ---- multistart: MS_K spread starts around each selected row's start, +-MS_SPREAD profile
            # sigmas along the softest eigen-directions of the warm GN matrix on the row's free dims ----
            X0_f = X0
            if len(ext):
                Hw = _hwarm()
                Xe = np.concatenate([spread_starts(X0[b], Hw[np.ix_(FREE[b], FREE[b])], XLO[b], XHI[b],
                                                   MS_K, MS_SPREAD) for b in ext])
                X0_f = np.concatenate([X0, Xe])
                print(f"[multistart] {len(rep)} spread starts: {MS_K} per row at {len(jsel)} grid points per "
                      f"POI (indices {jsel}), +-{MS_SPREAD} profile sigma along the softest warm GN "
                      f"directions; max|dx| {np.abs(Xe - X0[rep]).max(1).mean():.2f} scaled sigma (mean)",
                      flush=True)
        S = lm_rows(BASE_f, FREE_f, XLO_f, XHI_f, X0_f, POI_IDX=POI_f, PV=PV_f, EXTRA=(KIND == 1),
                    log_prefix="[lock]", ckpt_path=ckpt, resume_state=resume_state)
        T = dict(BASE=BASE_f, FREE=FREE_f, XLO=XLO_f, XHI=XHI_f, POI=POI_f, J=J_f, PV=PV_f, KIND=KIND,
                 START=START, X0=X0_f, X=S["X"], F=S["F"], G=S["G"], H=S["H"],
                 gnorm=S["gnorm"], stalled=S["stalled"], floor=S["floor"],
                 lock_stop=np.array(S["stop"]))          # why the lockstep LM ended (not per row)
        stage = "lm"
        _save_table(tpath, T, stage)
        try:
            os.remove(ckpt)
        except Exception:
            pass
        _done_line("LM done")

    # ================= flags -> branches =================
    if _stage_rank(stage) < 1:
        flags = _flags(T, pois_idx)
        nflag = sum(v[0] for v in flags.values())
        print(f"[flags] {nflag} flagged grid points of {len(flags)}", flush=True)
        for (p, j), (f, why, reps, r) in flags.items():
            if f:
                print(f"  [{ORDER[p]}@{grid_of[p][j]:.5g}] {why}; known minima: "
                      + (", ".join(f"chi2={T['F'][i]:.3f}" for i in reps) or "none"), flush=True)
        # ---- branches: every distinct minimum a spread start found at a grid point that is not the
        # one the warm start reached there, clustered across the POI's grid points by MS_DX so each
        # branch is seeded once, at most MS_BRANCH_MAX per POI (lowest chi2 first); each seed is an
        # ordinary LM run from its point at every grid point of its POI ----
        conv, _ = _row_conv(T)
        seeds = []
        for p in pois_idx:
            cand = []
            for j in range(NPTS):
                r = np.where((T["POI"] == p) & (T["J"] == j))[0]
                w = r[T["KIND"][r] == 0]
                xw = T["X"][w[0]] if len(w) and conv[w[0]] else None
                for i in r[T["KIND"][r] == 1]:
                    if conv[i] and (xw is None or np.abs(T["X"][i] - xw).max() > MS_DX):
                        cand.append(int(i))
            if cand:
                cand = np.array(cand, int)
                _, reps = cluster_points(T["X"][cand], T["F"][cand], MS_DX)
                seeds += [(p, int(cand[i])) for i in reps[:MS_BRANCH_MAX]]
        if seeds:
            bB, bFR, bLO, bHI, bX0, bP, bJ, bPV, bST = ([] for _ in range(9))
            for s_i, (p, c) in enumerate(seeds):
                for j in range(NPTS):
                    b0 = int(np.where((T["POI"] == p) & (T["J"] == j) & (T["KIND"] == 0))[0][0])
                    bB.append(T["BASE"][b0]); bFR.append(T["FREE"][b0]); bLO.append(T["XLO"][b0]); bHI.append(T["XHI"][b0])
                    bX0.append(np.clip(T["X"][c], T["XLO"][b0], T["XHI"][b0])); bP.append(p); bJ.append(j)
                    bPV.append(T["PV"][b0]); bST.append(s_i)
            bB, bFR, bLO, bHI, bX0 = (np.array(a) for a in (bB, bFR, bLO, bHI, bX0))
            print(f"[branches] {len(seeds)} seeds -> {len(bX0)} rows: "
                  + ", ".join(f"{ORDER[p]}@{grid_of[p][T['J'][c]]:.5g} chi2={T['F'][c]:.3f}" for p, c in seeds),
                  flush=True)
            S3 = lm_rows(bB, bFR, bLO, bHI, bX0, POI_IDX=np.array(bP, int), PV=np.array(bPV, float),
                         maxit=BR_MAXIT, log_prefix="[branch]")
            _append(T, BASE=bB, FREE=bFR, XLO=bLO, XHI=bHI, POI=np.array(bP, int), J=np.array(bJ, int),
                    PV=np.array(bPV, float), KIND=np.full(len(bX0), 2, int), START=np.array(bST, int),
                    X0=bX0, X=S3["X"], F=S3["F"], G=S3["G"], H=S3["H"], gnorm=S3["gnorm"], stalled=S3["stalled"],
                    floor=S3["floor"])
            for s_i, (p, c) in enumerate(seeds):
                rb = np.where((T["KIND"] == 2) & (T["START"] == s_i) & (T["POI"] == p))[0]
                base = np.array([T["F"][(T["POI"] == p) & (T["J"] == j) & (T["KIND"] < 2)].min() for j in T["J"][rb]])
                below = T["F"][rb] < base - MS_DCHI2
                print(f"  branch {s_i} ({ORDER[p]}): converged {int(_row_conv(T)[0][rb].sum())}/{len(rb)}, "
                      f"below the earlier envelope at {int(below.sum())} grid points"
                      + (f" (by up to {float((base - T['F'][rb]).max()):.3f})" if below.any() else ""), flush=True)
        else:
            print("[branches] no distinct minimum beyond the warm starts' -> no branches", flush=True)
        stage = "branch"
        _save_table(tpath, T, stage)
    _done_line("all phases done")

    # ================= envelope + outputs =================
    conv, at_box = _row_conv(T)
    decT = _row_dec(T["G"], T["H"], T["X"], T["XLO"], T["XHI"])
    flags = _flags(T, pois_idx)
    ev = np.linalg.eigvalsh(T["H"])
    for poi in pois:
        i = ORDER.index(poi); grid = grid_of[i]
        rows = np.where(T["POI"] == i)[0]
        e = np.array([r[np.argmin(T["F"][r])] for r in (rows[T["J"][rows] == j] for j in range(NPTS))], int)
        chi2 = T["F"][e]
        bnd = (BLO[i], BHI[i])
        lo1, mid, hi1, reg1 = poi_interval(grid, chi2, CL1, bnd)
        lo2, _, hi2, reg2 = poi_interval(grid, chi2, CL2, bnd)
        lo95, _, hi95, reg95 = poi_interval(grid, chi2, CL95, bnd)
        ends1, ends2, ends95 = (end_kinds(a, b, bnd, r) for a, b, r in
                                ((lo1, hi1, reg1), (lo2, hi2, reg2), (lo95, hi95, reg95)))
        kind1, kind95 = interval_kind(ends1), interval_kind(ends95)
        sw, mw, dw = wald_fit(grid, chi2)
        extra = {}
        if BBN is not None:                          # BBN outputs of the envelope rows; the Delta_Neff_init
            yb = _bbn_values(np.array([theta_of(T["BASE"][r], T["FREE"][r], T["X"][r]) for r in e]))
            extra["derived_bbn"] = yb                # profile also mapped to Neff (monotone in the input)
            if poi == "Delta_Neff_init":
                neff = yb[:, 1]
                fn = lambda v: float(np.interp(v, grid, neff)) if np.isfinite(v) else np.nan
                extra.update(derived_neff=neff, sigma1_neff=np.array([fn(lo1), fn(mid), fn(hi1)]),
                             cl95_neff=np.array([fn(lo95), fn(hi95)]))
                print(f"[{poi}] as Neff: best {fn(mid):.4f}; 68% [{fn(lo1):.4f},{fn(hi1):.4f}]; "
                      f"95% [{fn(lo95):.4f},{fn(hi95):.4f}]; at the best fit YHe {np.interp(mid, grid, yb[:, 0]):.5f}, "
                      f"Yp(BBN) {np.interp(mid, grid, yb[:, 3]):.5f}, D/H {np.interp(mid, grid, yb[:, 4]):.3f}e-5",
                      flush=True)
        mmax = max(1, max(len(flags[(i, j)][2]) for j in range(NPTS)))
        minima_x = np.full((NPTS, mmax, P), np.nan); minima_chi2 = np.full((NPTS, mmax), np.nan)
        minima_kind = np.full((NPTS, mmax), "", dtype="U12")
        for j in range(NPTS):
            for m, rr in enumerate(flags[(i, j)][2]):
                minima_x[j, m] = T["X"][rr]; minima_chi2[j, m] = T["F"][rr]; minima_kind[j, m] = KINDS[T["KIND"][rr]]
        flag = np.array([flags[(i, j)][0] for j in range(NPTS)], bool)
        reason = np.array([flags[(i, j)][1] for j in range(NPTS)], dtype="U96")
        npz = os.path.join(outdir, f"profile_prod_ad_{poi}{TAG}.npz")
        np.savez(npz, poi=poi, run_sig=_run_signature(), poi_grid=grid, chi2=chi2, xstar=T["X"][e], gnorm=T["gnorm"][e],
                 converged=conv[e], stalled=T["stalled"][e], floor=T["floor"][e], dec=decT[e], at_box=at_box[e],
                 gn_eig=ev[e], hess_pd=ev[e].min(1) > 0,
                 nuis=np.array([ORDER[k] for k in nuis_idx_of(i)]), done=True,
                 sigma1=np.array([lo1, mid, hi1]), sigma2=np.array([lo2, hi2]),
                 cl95=np.array([lo95, hi95]),
                 bound=np.array([b if np.isfinite(b) else np.nan for b in bnd]),     # NaN = open
                 ends1=np.array(ends1), ends2=np.array(ends2), ends95=np.array(ends95), kind1=kind1,
                 kind95=kind95, stop_reason=str(T.get("lock_stop", "")),
                 regions1=np.array(reg1, float).reshape(-1, 2), regions2=np.array(reg2, float).reshape(-1, 2),
                 regions95=np.array(reg95, float).reshape(-1, 2),
                 sigma_wald=sw, wald_vertex=mw, wald_misfit=dw, interval_mode=_interval_mode(grid, chi2, bnd),
                 n_starts=np.array([int((T["J"][rows] == j).sum()) for j in range(NPTS)]),
                 n_minima=np.array([len(flags[(i, j)][2]) for j in range(NPTS)]), flag=flag, flag_reason=reason,
                 minima_x=minima_x, minima_chi2=minima_chi2, minima_kind=minima_kind,
                 rows_j=T["J"][rows], rows_kind=T["KIND"][rows], rows_start=T["START"][rows],
                 rows_x0=T["X0"][rows],
                 rows_x=T["X"][rows], rows_chi2=T["F"][rows], rows_gnorm=T["gnorm"][rows],
                 rows_converged=conv[rows], rows_stalled=T["stalled"][rows], rows_floor=T["floor"][rows],
                 rows_dec=decT[rows], rows_at_box=at_box[rows],
                 kinds=np.array(KINDS), ms=np.array([MS_K, MS_NPTS, MS_SPREAD, MS_DX, MS_DCHI2]),
                 gtol=GTOL, dtol=DTOL, use_lowee=USE_LOWEE, lowee=LOWEE, use_lowtt=USE_LOWTT, config=CONFIG_ABS,
                 high_ell=HIGH_ELL, **extra)
        j = int(np.nanargmin(chi2)); nval = int(np.isfinite(chi2).sum())
        edge = (j <= 1 and grid[0] != BLO[i]) or (j >= NPTS - 2 and grid[-1] != BHI[i])
        warn = ""
        if edge:
            warn = (f"  ** CENTERING: min at grid index {j}, not a physical bound (widen PA_NSIG / the config "
                    f"grid_sig, or PA_MLE_PREPASS=1)")
        for side, k, gend in (("lower", ends1[0], grid[0]), ("upper", ends1[1], grid[-1])):
            if k == "open":
                warn += (f"  ** OPEN {side} 68% end: below threshold up to the grid end {gend:.5g} (the data "
                         f"leave {poi} open there, or the grid is too narrow)")

        def _iv(lo, hi, ends):
            f = lambda v, k, g, s: f"{s}{g:.5f} (open)" if k == "open" else f"{v:.5f}" + (" (bound)" if k == "bound" else "")
            return f"[{f(lo, ends[0], grid[0], '<')}, {f(hi, ends[1], grid[-1], '>')}]"
        multi = f"; 68% UNION of {len(reg1)} regions {[(round(a, 5), round(b, 5)) for a, b in reg1]}" if len(reg1) > 1 else ""
        print(f"[{poi}] minchi2={chi2[j]:.2f} at {poi}={grid[j]:.5f}; 68% {_iv(lo1, hi1, ends1)} ({kind1}, "
              f"best {mid:.5f}); 95% {_iv(lo95, hi95, ends95)} ({kind95}); wald sigma {sw:.5f} (misfit {dw:.2f}); "
              f"converged {int(conv[e].sum())}/{nval} ({int((conv[e] & T['floor'][e]).sum())} on the noise floor), "
              f"stalled {int(T['stalled'][e].sum())}; "
              f"max||g||_AD={np.nanmax(T['gnorm'][e]):.2e}; {len(rows)} rows ("
              + ", ".join(f"{k} {int((T['KIND'][rows] == ki).sum())}" for ki, k in enumerate(KINDS))
              + f"); flagged {int(flag.sum())}/{NPTS} points; envelope from "
              + ", ".join(f"{k} {int((T['KIND'][e] == ki).sum())}" for ki, k in enumerate(KINDS)
                          if (T["KIND"][e] == ki).any())
              + f"{multi} -> {npz}{warn}", flush=True)
        _plot(poi, grid, chi2, npz, rows=(T["PV"][rows], T["F"][rows], T["KIND"][rows]), regions=reg1,
              bnd=bnd, ends=ends1)
    try:
        if os.path.exists(tpath):
            os.remove(tpath)                           # run finished cleanly
    except Exception:
        pass


# ======================================================================
# 2D profile (PA_POI2D="a,b"): both POIs fixed on a grid, every other parameter free
# ======================================================================
def _grid2d_for(poi):
    """the 2D grid of one POI: PA_GRID2D or config grid2d[poi] as a tuple (lo, hi, n) or a list of values, else
    PA_NPTS2D points over the POI's 1D span (inside its bounds)."""
    import json
    spec = dict(CFG.get("grid2d") or {}); spec.update(json.loads(GRID2D_ENV) if GRID2D_ENV else {})
    g = spec.get(poi)
    if isinstance(g, list):
        return np.asarray(g, float)
    if g is not None:
        return np.linspace(float(g[0]), float(g[1]), int(g[2]))
    full = _grid_for(poi)
    return np.linspace(full[0], full[-1], NPTS2D)


def _npz2d(outdir, pa, pb):
    part = f"_part{RANK}of{NPROC}" if (POI_SLICE and NPROC > 1) else ""
    return os.path.join(outdir, f"profile_prod_ad_2d_{pa}_{pb}{TAG}{part}.npz")


def profile_2d(pa, pb, outdir):
    """chi2 profiled over every other parameter on the (pa, pb) grid: one LM row per point, started at the
    config centre (with PA_WARM_LR at the better of it and the warm point's linear response). No spread
    starts or branches. With PA_POI_SLICE each rank takes every NPROC-th pa value and writes its own part
    npz (part<rank>of<nproc>); the parts together are the grid. npz: grid_a, grid_b, ja (this part's pa
    indices), chi2 / converged / at_box / stalled / floor (n_ja, n_b), xstar (n_ja, n_b, D-2), derived_bbn."""
    ia, ib = ORDER.index(pa), ORDER.index(pb)
    ga, gb = _grid2d_for(pa), _grid2d_for(pb)
    ja = np.arange(len(ga))[RANK::NPROC] if (POI_SLICE and NPROC > 1) else np.arange(len(ga))
    A, Bm = np.meshgrid(ga[ja], gb, indexing="ij")
    N, shape = A.size, A.shape
    free = np.array([k for k in range(D) if k not in (ia, ib)])
    FREE = np.tile(free, (N, 1))
    BASE = np.tile(CENTER, (N, 1)); BASE[:, ia] = A.ravel(); BASE[:, ib] = Bm.ravel()
    XLO, XHI = map(np.array, zip(*[scaled_box(f) for f in FREE]))
    print(f"[2d] {pa} x {pb}: {len(ga)} x {len(gb)} grid, this rank {len(ja)} x {len(gb)} = {N} rows, "
          f"{len(free)} free dims", flush=True)
    ckpt = _ckpt_path(outdir, f"STATE_2d_{pa}_{pb}")
    resume_state = None
    if RESUME and os.path.exists(ckpt):
        st = np.load(ckpt)
        if st["BASE"].shape == BASE.shape and np.allclose(st["BASE"], BASE) and _same_run(st):
            resume_state = {k: st[k] for k in st.files}
            print(f"[resume] 2d LM checkpoint @ it{int(st['it'])} matches; continuing", flush=True)
    if resume_state is not None:
        X0 = np.array(resume_state["X0"], float)
    else:
        X0 = np.clip(np.zeros((N, len(free))), XLO, XHI)
        if WARM_LR:
            _, _, Hw = joint_fit(CENTER, 0)
            dab = (BASE[:, [ia, ib]] - CENTER[[ia, ib]]) / SIGMA[[ia, ib]]
            resp = np.linalg.lstsq(Hw[np.ix_(free, free)], Hw[np.ix_(free, [ia, ib])], rcond=None)[0]
            Xlr = np.clip(-(dab @ resp.T), XLO, XHI)
            fk = values_rows(np.tile(BASE, (2, 1)), np.tile(FREE, (2, 1)), np.concatenate([X0, Xlr]))
            use = fk[N:] < fk[:N]
            X0 = np.where(use[:, None], Xlr, X0)
            print(f"[warm-lr] linear-response start kept for {int(use.sum())}/{N} rows", flush=True)
    S = lm_rows(BASE, FREE, XLO, XHI, X0, log_prefix="[2d]", ckpt_path=ckpt, resume_state=resume_state)
    T = dict(X=S["X"], G=S["G"], gnorm=S["gnorm"], floor=S["floor"], stalled=S["stalled"], FREE=FREE,
             XLO=XLO, XHI=XHI)
    conv, at_box = _row_conv(T)
    th = np.array([theta_of(BASE[b], FREE[b], S["X"][b]) for b in range(N)])
    extra = {}
    if BBN is not None:
        extra["derived_bbn"] = _bbn_values(th).reshape(shape + (-1,))
    rs = lambda v: np.asarray(v).reshape(shape)
    npz = _npz2d(outdir, pa, pb)
    np.savez(npz, poi=np.array([pa, pb]), run_sig=_run_signature(), grid_a=ga, grid_b=gb, ja=ja,
             chi2=rs(S["F"]), converged=rs(conv), at_box=rs(at_box), stalled=rs(S["stalled"]), floor=rs(S["floor"]),
             gnorm=rs(S["gnorm"]), xstar=S["X"].reshape(shape + (-1,)), theta=th.reshape(shape + (-1,)),
             nuis=np.array([ORDER[k] for k in free]), order=np.array(ORDER),
             bound=np.array([[b if np.isfinite(b) else np.nan for b in (BLO[i], BHI[i])] for i in (ia, ib)]),
             stop_reason=S["stop"], gtol=GTOL, dtol=DTOL, config=CONFIG_ABS, done=True, **extra)
    k = int(np.argmin(S["F"]))
    print(f"[2d] {pa} x {pb}: min chi2 {S['F'][k]:.3f} at {pa}={BASE[k, ia]:.5g}, {pb}={BASE[k, ib]:.5g}; converged "
          f"{int(conv.sum())}/{N} ({int((conv & S['floor']).sum())} on the noise floor), stalled "
          f"{int(S['stalled'].sum())}, held by the box {int(at_box.sum())} -> {npz}", flush=True)
    try:
        os.remove(ckpt)
    except Exception:
        pass


def _plot(poi, grid, chi2, npz, rows=None, regions=None, bnd=(-np.inf, np.inf), ends=None):
    """Delta-chi2 of the envelope; with rows = (pv, chi2, kind) every other row's end point too
    (starts, branches), and the 68% union regions shaded when there are several. Next to a physical
    bound (bnd) the thresholds are the boundary-aware ones and the bound is drawn; an open 68% end
    (ends from end_kinds) gets an arrow off the grid."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        i = ORDER.index(poi)
        c0 = np.nanmin(chi2); d = chi2 - c0
        fig, ax = plt.subplots(figsize=(6, 4.3))
        if rows is not None:
            pv, f, kind = rows
            env = np.interp(pv, grid, chi2)
            for ki, (mk, col, lab) in enumerate((("s", "0.65", "warm start"), ("^", "0.65", "spread start"),
                                                 ("+", "tab:purple", "branch"))):
                m = (kind == ki) & (f > env + 1e-9)
                if m.any():
                    ax.scatter(pv[m], f[m] - c0, s=16, marker=mk, c=col, label=lab, zorder=2)
        ax.plot(grid, d, "o-", label="profile (lower envelope)", zorder=3)
        if regions is not None and len(regions) > 1:
            for a, b in regions:
                ax.axvspan(grid[0] if not np.isfinite(a) else a, grid[-1] if not np.isfinite(b) else b,
                           color="g", alpha=0.08)
        xs = np.linspace(grid.min(), grid.max(), 400)
        sw = wald_fit(grid, chi2)[0]
        best = grid[int(np.nanargmin(chi2))]
        near = [(abs(best - b), sgn * (xs - b)) for b, sgn in ((bnd[0], 1), (bnd[1], -1)) if np.isfinite(b)]
        for cl, col, lab in ((CL1, "g", "68%"), (CL95, "orange", "95%")):
            thr = np.full_like(xs, _chi2_dist.ppf(cl, 1))
            if near and np.isfinite(sw):
                thr = t_threshold(cl, min(near, key=lambda t: t[0])[1] / sw)
            ax.plot(xs, thr, "--", lw=0.8, color=col, label=f"{lab} threshold")
        for b in bnd:
            if np.isfinite(b):
                ax.axvline(b, color="k", lw=1.0, label="physical bound")
        if ends is not None:
            thr1 = _chi2_dist.ppf(CL1, 1)
            for k, x0, dx in ((ends[0], grid[0], -1), (ends[1], grid[-1], 1)):
                if k == "open":
                    w = 0.08 * (grid[-1] - grid[0])
                    ax.annotate("", xy=(x0 + dx * w, 0.5 * thr1), xytext=(x0, 0.5 * thr1),
                                arrowprops=dict(arrowstyle="->", color="g", lw=1.2))
                    ax.text(x0, 0.62 * thr1, "68% open", color="g", fontsize=7,
                            ha="right" if dx < 0 else "left")
            ax.set_xlim(grid[0] - 0.1 * (grid[-1] - grid[0]), grid[-1] + 0.1 * (grid[-1] - grid[0]))
        ax.axvline(CENTER[i], ls=":", color="gray", lw=0.8, label=f"fiducial {poi}")
        ax.set_xlabel(poi); ax.set_ylabel(r"$\Delta\chi^2$ (profiled)")
        top = max(10, float(np.nanmax(d)) * 1.05)
        if rows is not None and np.isfinite(rows[1]).any():    # show the other rows too, within reason
            top = max(top, min(float(np.nanmax(rows[1])) - c0, 30.0) * 1.05)
        ax.set_ylim(0, top); ax.legend(fontsize=7)
        like = [{"plik_full": "full plik (nuisances profiled)",
                 "act_lite": "ACT DR6 lite (A_act, P_act profiled)"}[HIGH_ELL]] + ["lowTT"] * bool(USE_LOWTT) + [f"lowEE ({LOWEE})"] * bool(USE_LOWEE)
        if TOY is not None:
            like = ["toy objective " + os.path.basename(CONFIG_ABS)]
        ax.set_title(f"profile of {poi}: {' + '.join(like)}")
        fig.tight_layout(); fig.savefig(npz.replace(".npz", ".png"), dpi=120)
    except Exception as e:
        print(f"[{poi}] plot skipped: {e}", flush=True)


def _resume_skip(pois, outdir):
    """drop POIs whose npz is already done by this same run (RESUME). A finished npz under this PA_TAG from a
    different setup (run signature, _run_signature) stops the run: choose a new PA_TAG or move the file."""
    out = []
    for poi in pois:
        npz = os.path.join(outdir, f"profile_prod_ad_{poi}{TAG}.npz")
        if RESUME and os.path.exists(npz):
            try:
                st = np.load(npz, allow_pickle=True)
                done, same = bool(st["done"]), _same_run(st)
            except Exception:
                done = same = False
            if done and same:
                print(f"[{poi}] resume: done, skip", flush=True); continue
            if done:
                raise SystemExit(f"[{poi}] {npz} is a finished run with a different setup (config or knobs; "
                                 f"run_sig here {_run_signature()}): use a new PA_TAG or move the old file")
        out.append(poi)
    return out


def main():
    outdir = os.path.join(_HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    print(f"rank {RANK}/{NPROC} devices={jax.devices()} config={CONFIG_ABS} "
          f"POIs={POIS} NPTS={NPTS} NSIG={NSIG} GTOL={GTOL} DTOL={DTOL} rtol={RTOL} shard={DO_SHARD} "
          f"lowEE={LOWEE if USE_LOWEE else False} lowTT={USE_LOWTT} multistart={MS_K}x{MS_NPTS}pts "
          f"(spread {MS_SPREAD}, dx {MS_DX}, dchi2 {MS_DCHI2}) poi_slice={POI_SLICE} nproc={NPROC} "
          f"sigtol={SIGTOL} xbox={XBOX} bounds={_BOUNDS or None} box={_BOX or None}", flush=True)
    if POI2D:
        npz = _npz2d(outdir, *POI2D)
        if RESUME and os.path.exists(npz) and _same_run(np.load(npz, allow_pickle=True)):
            print(f"[2d] resume: {npz} done, skip", flush=True)
        else:
            profile_2d(*POI2D, outdir)
        print(f"rank {RANK}: done", flush=True)
        return
    if POI_SLICE and NPROC > 1:
        pois = _resume_skip(POIS[RANK::NPROC], outdir)     # each rank: a disjoint POI subset
    else:
        pois = _resume_skip(POIS, outdir) if RANK == 0 else []
    if pois:
        profile_lockstep(pois, outdir)
    print(f"rank {RANK}: done", flush=True)


if __name__ == "__main__":
    main()

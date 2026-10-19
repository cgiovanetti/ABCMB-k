"""Config-driven, lockstep frequentist profile-likelihood tool (TOOL_PLAN.md).

One sbatch in; per profiled parameter (POI), an npz + png out with the grid, the
profiled chi2, the best fit, 68/95% intervals (two-sided, or one-sided next to a
physical bound), and a per-point AD convergence certificate.

  1. Config-driven (PA_CONFIG=path/to/config.py). A config declares the parameter
     order, fiducials (cen), scale widths (sig), which parameters are POIs, the
     fixed dict, optional user_species, optional physical `bounds`, likelihood
     toggles, and NPTS/NSIG. Adding a parameter is a config edit, not a code edit.

  2. One lockstep batch across all POIs x grid points. Each "row" is one
     optimisation point (the POI fixed at its grid value, the other parameters
     free); the optimizer state is per-row arrays, so concatenating POIs just
     lengthens the batch. One POI subset per SLURM rank via PA_POI_SLICE.

  3. Optimizer: Levenberg-Marquardt on the Gauss-Newton curvature, AD only. Every
     gradient call pushes the free directions through ABCMB (forward mode) to the
     C_ell tangents dC_ell/dtheta and turns them into the chi2 gradient AND its
     Gauss-Newton matrix J^T W J (plik's own data covariance; the plik nuisances
     enter through the profiled-likelihood Schur complement; the low-ell
     likelihoods through their exact Hessian in D_ell), at the row's own point.
     Each iteration tries PA_LM_K damping values per active row in one batched
     value call and keeps the best that passes the gain-ratio test. Rows that
     moved get a new gradient; rows that did not keep theirs and raise the
     damping; converged and stalled rows leave the batches. Values for every
     acceptance test come from the call_batched path.

  4. Physical bounds (config `bounds`: {name: (lo, hi)}, None = open) are hard
     limits for the optimizer (projected steps) and for the grids. Intervals use
     the boundary-aware asymptotic thresholds of Cowan, Cranmer, Gross & Vitells,
     arXiv:1007.1727 (the t~ statistic), so a POI with a bound comes out two-sided
     or as a one-sided limit depending on the data; without one it is the usual
     Delta-chi2 = 1 / 4 crossing.

  5. Multimodality (scan/lm_tools.py). By default every POI gets PA_MS_K extra starts at
     PA_MS_NPTS of its grid points, spread +-PA_MS_SPREAD profile sigmas along the softest
     eigen-directions of the warm Gauss-Newton matrix on the row's free dims; they run in
     the same lockstep and leave the batch as they converge. A grid point is flagged when
     a start stalled, when converged starts sit at distinct points (> PA_MS_DX scaled
     sigma apart) or when they disagree in chi2 by > PA_MS_DCHI2. Flagged points are
     re-run with every known minimum deflated at once (Farrell, Birkisson & Funke 2015:
     the ordinary LM step d becomes d / (1 - grad(log M).d), M = prod_i (|x - x_i|^-p +
     alpha)), which converges to stationary points other than the known ones; each new
     one (a second minimum, or a saddle, whose chi2 is the barrier) seeds a branch: an
     ordinary LM run from it at every grid point of that POI. The profile is the lower
     envelope over all starts and branches; the intervals are the union of the regions
     below threshold (one region for a unimodal profile). The per-POI npz keeps every row.

Run via srun, PYTHONPATH=$(pwd), JAX_COMPILATION_CACHE_DIR set. Env knobs:
  PA_CONFIG(scan/configs/lcdm.py) PA_POIS(csv; config default)
  PA_NPTS(config) PA_NSIG(config) PA_MAXIT(40) PA_GTOL(3e-2)
  PA_LMAX(2508) PA_RTOL(1e-5) PA_TAG('') PA_RESUME(1)
  PA_LM_K(4) PA_STEP_MAX(1.0) PA_XBOX(5) PA_VAL_CHUNK(256) PA_AD_BCHUNK(64)
  PA_WARM_LR(0) PA_MLE_PREPASS(0) PA_POI_SLICE(0) PA_TRACE_EVALS(0)
  PA_MS_K(3) PA_MS_NPTS(6) PA_MS_SPREAD(2.0) PA_MS_DX(0.5) PA_MS_DCHI2(0.1) PA_MS_WAIT(1)
  PA_MS_ROUNDS(1) PA_MS_BRANCH_MAX(4) PA_BR_MAXIT(10) PA_DF_MAXIT(10) PA_DF_P(2) PA_DF_ALPHA(1)
  PA_MS_FORCE(0) PA_MS_BRANCH_FROM(new)   [the last two: shakeout knobs]
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
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import chi2 as _chi2_dist
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import Model
from abcmb import species as _species
from scan.plik_lite import PlikLite
from scan.lowl_like import LowLEE, LowLTT
from scan.batched_grad import staged_cl_and_grad, _to_float as _bg_to_float
from scan.lm_tools import (proj_grad, lm_step, deflate_step, deflated_resid, spread_starts,
                           flag_point, union_regions, is_multimodal)

_HERE = os.path.dirname(os.path.abspath(__file__))
CL1, CL2, CL95 = 0.682689492137, 0.954499736104, 0.95


# ======================================================================
# profile-interval helpers
# ======================================================================
def _interval_mode(x, y):
    """PA_INTERVAL, except that a profile whose points show two minima separated by a barrier above
    the roughness (lm_tools.is_multimodal) is interpolated: a polynomial would smooth the second
    mode away."""
    if INTERVAL != "fit" or is_multimodal(x, y, FIT_DCHI2, MS_DCHI2):
        return "pchip"
    return "fit"


def _smooth_profile(x, y):
    """Dense Delta-chi2 curve (xs, ys) through a profile's points. PA_INTERVAL=fit: least-squares
    polynomial of degree FIT_DEG through the points within FIT_DCHI2 of the minimum (averages per-row
    scatter; a quartic follows the skew of tau / Neff); pchip: shape-preserving interpolation through
    every point (also used for a multimodal profile, _interval_mode). None with fewer than 4 finite
    points."""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    if m.sum() < 4:
        return None
    x, y = x[m], y[m]; o = np.argsort(x); x, y = x[o], y[o]
    if _interval_mode(x, y) == "fit":
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
    sigma = 1/sqrt(a), the unconstrained vertex m (beyond a bound when the data prefer it), and the
    largest misfit of the parabola on those points (Delta-chi2 units; large = the Wald approximation
    behind the boundary-aware thresholds is poor). NaNs when no upward parabola fits."""
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


def poi_interval(x, y, cl, bound=(None, None)):
    """Confidence interval at level cl from a profile (x, chi2). bound = the POI's physical (lo, hi)
    limits (None = open). With no bound in play the threshold is the chi2_1 quantile on both sides.
    With one, the threshold is t_threshold at each tested value, sigma from wald_fit, so the interval
    becomes a one-sided limit by itself when the data sit near the bound. Returns (lo, best, hi,
    regions): regions is the UNION of the regions below threshold as a list of (lo, hi) in increasing
    x (one region for a unimodal profile), lo/hi are the outer ends of that union; an end equal to
    the bound is a one-sided limit, NaN means not reached within the grid."""
    sp = _smooth_profile(x, y)
    if sp is None:
        return np.nan, np.nan, np.nan, []
    xs, ys = sp
    i0 = int(np.argmin(ys)); best = xs[i0]
    thr = np.full_like(xs, _chi2_dist.ppf(cl, 1))
    blo, bhi = (np.nan if b is None else float(b) for b in bound)
    sig, _, _ = wald_fit(x, y)
    near = []
    if np.isfinite(blo):
        near.append(((best - blo) / sig, xs - blo))
    if np.isfinite(bhi):
        near.append(((bhi - best) / sig, bhi - xs))
    if near and np.isfinite(sig):
        _, dist = min(near, key=lambda t: t[0])          # the bound closest to the best fit
        thr = t_threshold(cl, dist / sig)
    regions = union_regions(xs, ys, thr, (blo, bhi))     # non-empty: ys = 0 <= thr at the minimum
    return regions[0][0], best, regions[-1][1], regions


def sigma_parabola(x, y):
    """Symmetric Gaussian sigma from a parabola fit to the points within Delta-chi2 <= 4."""
    return wald_fit(x, y)[0]


# ======================================================================
# config loading
# ======================================================================
def _load_config(path):
    """Load a CONFIG dict from a .py file (PA_CONFIG)."""
    if not os.path.isabs(path):
        # resolve relative to CWD, then to this file's dir, then to configs/
        for base in (os.getcwd(), _HERE, os.path.join(_HERE, "configs")):
            cand = os.path.join(base, path)
            if os.path.exists(cand):
                path = cand
                break
    spec = importlib.util.spec_from_file_location("pa_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG, os.path.abspath(path)


CONFIG_PATH = os.environ.get("PA_CONFIG", "scan/configs/lcdm.py")
CFG, CONFIG_ABS = _load_config(CONFIG_PATH)

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
# optional config 'grid': {poi: (lo, hi)} explicit grid range (overrides cen +- NSIG*grid_sig)
GRID_RANGE = dict(CFG.get("grid", {}))
# optional config 'bounds': {name: (lo, hi)} physical limits, None = open
_B = CFG.get("bounds", {})
BLO = np.array([-np.inf if _B.get(k, (None, None))[0] is None else float(_B[k][0]) for k in ORDER])
BHI = np.array([np.inf if _B.get(k, (None, None))[1] is None else float(_B[k][1]) for k in ORDER])
FIXED = dict(CFG["fixed"])
USER_SPECIES = CFG.get("user_species", None)
# ABCMB evolves a massive neutrino only when the MassiveNeutrino species is passed; with
# N_nu_massive > 0 alone it silently puts all of Neff into massless neutrinos.
if float(FIXED.get("N_nu_massive", 0)) > 0 and _species.MassiveNeutrino not in tuple(USER_SPECIES or ()):
    raise ValueError(f"{CONFIG_ABS}: fixed N_nu_massive={FIXED['N_nu_massive']} needs "
                     f"user_species=(species.MassiveNeutrino,); without it ABCMB drops the massive neutrino")

# ---------------- env config (overrides config-file defaults) ----------------
LMAX = int(os.environ.get("PA_LMAX", 2508))
NPTS = int(os.environ.get("PA_NPTS", CFG.get("npts", 25)))
NSIG = float(os.environ.get("PA_NSIG", CFG.get("nsig", 3.0)))
MAXIT = int(os.environ.get("PA_MAXIT", 40))
GTOL = float(os.environ.get("PA_GTOL", 3e-2))        # ||projected grad||_inf gate (scaled coords)
# interval-stability early-stop: stop once every POI's 68% interval ends have moved by
# < PA_SIGTOL * grid_sig over PA_SIGTOL_PATIENCE iters (the deliverable has converged to
# PA_SIGTOL sigmas). SIGTOL<=0 disables. The default 1e-2 was validated at full l
# (CHANGELOG 2026-06-23). The AD certificate is still recorded per row.
SIGTOL = float(os.environ.get("PA_SIGTOL", "1e-2"))
SIGTOL_PATIENCE = int(os.environ.get("PA_SIGTOL_PATIENCE", "3"))
RTOL = float(os.environ.get("PA_RTOL", 1e-5))
# Save / LoS grid: "fixed" = the visibility grid built once at the config centre and used for every
# row (Model.freeze_lna_grid). "visibility" = rebuilt per cosmology (before 2026-10-16): its nodes move
# with theta, which makes the Cls jagged in theta and the per-ell AD tangents wrong (CHANGELOG).
LNA_GRID = os.environ.get("PA_LNA_GRID", "fixed").lower()
TAG = os.environ.get("PA_TAG", "")
RESUME = os.environ.get("PA_RESUME", "1") != "0"
GRAD_KCHUNK = int(os.environ.get("PA_GRAD_KCHUNK", "100"))   # k_chunk for the batched AD gradient
VAL_CHUNK = int(os.environ.get("PA_VAL_CHUNK", "256"))      # max cosmologies per call_batched value call
AD_BCHUNK = int(os.environ.get("PA_AD_BCHUNK", "64"))       # max rows per staged-AD call (B_local=16 on a
    # 4-GPU node: 9 GB/device with PA_GRAD_SPECB=4 and all directions vmapped; B_local=32 is 14 GB
    # and only 3% faster per cosmology, 2026-10-14)
GRAD_DIRCHUNK = int(os.environ.get("PA_GRAD_DIRCHUNK", "0")) or None  # directions per vmapped
    # staged jvp (0 = all at once, fastest; smaller = memory fallback at ceil(P/chunk) primal solves)
GRAD_SPECB = int(os.environ.get("PA_GRAD_SPECB", "4")) or None  # cosmologies per device per
    # spectrum-stage jvp call (the gradient's memory peak; sub-batching it costs no extra work)
_shard_env = os.environ.get("PA_SHARD", "auto").lower()
# Multimodality (lm_tools): MS_K extra starts per row at MS_NPTS grid points per POI (0 = none),
# +-MS_SPREAD profile sigmas along the softest warm Gauss-Newton directions; converged starts more
# than MS_DX (scaled sigma, max-norm) apart are distinct minima, closer ones must agree in chi2 to
# MS_DCHI2; the interval stop waits for the extra starts (MS_WAIT). Flagged points go through
# MS_ROUNDS rounds of deflation (DF_MAXIT deflated steps; M = prod (r^-DF_P + DF_ALPHA), reach
# ~ DF_ALPHA^(-1/DF_P) scaled sigma) and branches (at most MS_BRANCH_MAX new stationary points per
# POI per round, BR_MAXIT iterations). Shakeout knobs: MS_FORCE=1 flags every multistart point;
# MS_BRANCH_FROM=any branches from every deflation end point, not only the new stationary ones.
MS_K = int(os.environ.get("PA_MS_K", "3"))
MS_NPTS = int(os.environ.get("PA_MS_NPTS", "6"))
MS_SPREAD = float(os.environ.get("PA_MS_SPREAD", "2.0"))
MS_DX = float(os.environ.get("PA_MS_DX", "0.5"))
MS_DCHI2 = float(os.environ.get("PA_MS_DCHI2", "0.1"))
MS_WAIT = os.environ.get("PA_MS_WAIT", "1") != "0"
MS_ROUNDS = int(os.environ.get("PA_MS_ROUNDS", "1"))
MS_BRANCH_MAX = int(os.environ.get("PA_MS_BRANCH_MAX", "4"))
MS_FORCE = os.environ.get("PA_MS_FORCE", "0") != "0"
MS_BRANCH_FROM = os.environ.get("PA_MS_BRANCH_FROM", "new")
BR_MAXIT = int(os.environ.get("PA_BR_MAXIT", "10"))
DF_MAXIT = int(os.environ.get("PA_DF_MAXIT", "10"))
DF_P = float(os.environ.get("PA_DF_P", "2"))
DF_ALPHA = float(os.environ.get("PA_DF_ALPHA", "1"))
USE_LOWTT = (os.environ["PA_USE_LOWTT"] != "0") if "PA_USE_LOWTT" in os.environ \
    else bool(CFG.get("use_lowtt", True))
USE_LOWEE = (os.environ["PA_USE_LOWEE"] != "0") if "PA_USE_LOWEE" in os.environ \
    else bool(CFG.get("use_lowee", True))
# Include the high-ell TTTEEE likelihood. DEFAULT ON. PA_USE_PLIK=0 only for low-ell-only DEBUG runs.
USE_PLIK = (os.environ["PA_USE_PLIK"] != "0") if "PA_USE_PLIK" in os.environ \
    else bool(CFG.get("use_plik", True))
# FULL Planck plik (clipy) vs plik-LITE for the high-ell TTTEEE likelihood. Config
# field high_ell: "plik_lite" (default) | "plik_full". Full plik carries 47
# foreground/calibration nuisances handled ENTIRELY inside the likelihood -- profiled
# per cosmology at FIXED ABCMB theory (no ABCMB re-run for nuisances). See scan/plik_full.py.
HIGH_ELL = os.environ.get("PA_HIGH_ELL", str(CFG.get("high_ell", "plik_lite"))).lower()
USE_PLIK_FULL = USE_PLIK and (HIGH_ELL == "plik_full")
WARM = os.environ.get("PA_WARM", "1") != "0"         # rows start at the warm point (else at CENTER)
# MLE PRE-PASS: solve the joint D-dim fit first (same optimizer, all parameters free), recenter
# every POI grid on it with a span from its Gauss-Newton errors, and warm-start there.
MLE_PREPASS = os.environ.get("PA_MLE_PREPASS", "0") != "0"
MLE_MAXITER = int(os.environ.get("PA_MLE_MAXITER", "40"))
MLE_SIG_CLIP = float(os.environ.get("PA_MLE_SIG_CLIP", "3.0"))  # grid sigma clamped to [1/clip, clip] x prior
POI_SLICE = os.environ.get("PA_POI_SLICE", "0") != "0"   # multi-node scale-out: each
# rank owns a DISJOINT subset of POIs (POIS[RANK::NPROC]) and runs its own lockstep,
# writing only its own per-POI npz -> no shared-file clobber, no MPI gather.
XBOX = float(os.environ.get("PA_XBOX", "5.0"))         # free-dim box (sigma units), inside any bounds
INTERVAL = os.environ.get("PA_INTERVAL", "fit")       # fit | pchip (see _smooth_profile)
FIT_DEG = int(os.environ.get("PA_FIT_DEG", "4"))       # _smooth_profile polynomial degree
FIT_DCHI2 = float(os.environ.get("PA_FIT_DCHI2", "9"))  # _smooth_profile: points within this of the min
WARM_LR = os.environ.get("PA_WARM_LR", "0") != "0"     # rows start at the warm-point linear response
# Levenberg-Marquardt: STEP_MAX caps each step (max-norm, sigma units): a far row's first steps can
# otherwise reach cosmologies where an ODE solve hits max_steps. LM_K damping values per iteration,
# a decade apart around the row's current lambda; a step is taken when its actual chi2 drop is at
# least LM_ETA of the Gauss-Newton prediction. A row whose LM_K candidates all fail keeps its point
# (and its gradient) and moves its lambda past the ladder; STALL_MAX such iterations in a row, or
# lambda > LM_LAMBDA_MAX, stops it as stalled.
STEP_MAX = float(os.environ.get("PA_STEP_MAX", "1.0"))
LM_K = int(os.environ.get("PA_LM_K", "4"))
LM_LADDER = 10.0 ** (np.arange(LM_K) - 1)
LM_LAMBDA0, LM_LAMBDA_MIN, LM_LAMBDA_MAX = 1e-3, 1e-7, 1e8
LM_ETA = 1e-4
STALL_MAX = int(os.environ.get("PA_STALL_MAX", "3"))
TRACE_EVALS = os.environ.get("PA_TRACE_EVALS", "0") != "0"   # log every batch before evaluating it, so
#   a run that aborts inside ABCMB leaves the offending cosmologies as its last log lines

POIS = os.environ.get("PA_POIS", ",".join(CFG.get("pois", ORDER))).split(",")
POIS = [p.strip() for p in POIS if p.strip() in ORDER]
RANK = int(os.environ.get("SLURM_PROCID", 0))
NPROC = int(os.environ.get("SLURM_NPROCS", 1))

pl = PlikLite()          # always built: its abcmb_cl_to_Dl feeds the low-ell likelihoods
lowee = LowLEE() if USE_LOWEE else None
lowtt = LowLTT() if USE_LOWTT else None
plf = None
_plf_prof_B = None       # jitted batched inner nuisance profile
if USE_PLIK_FULL:
    from scan.plik_full import PlikFull
    plf = PlikFull()
    _plf_prof_B = jax.jit(lambda c: plf.profile_batched(c))
model = Model(user_species=USER_SPECIES, output_Cl=True, l_max=LMAX, lensing=True,
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


def scaled_box(free):
    """(lo, hi) of a row's free dims in scaled coords: the physical bounds inside +-XBOX."""
    lo = np.maximum((BLO[free] - CENTER[free]) / SIGMA[free], -XBOX)
    hi = np.minimum((BHI[free] - CENTER[free]) / SIGMA[free], XBOX)
    return lo, hi


def build_dict(theta):
    """physical D-vector (ORDER) -> ABCMB param dict (host floats, call_batched).
    ln10As is stored as A_s; everything else passes through by name."""
    p = dict(FIXED)
    for i, name in enumerate(ORDER):
        if name == "ln10As":
            p["A_s"] = float(np.exp(theta[i]) / 1e10)
        else:
            p[name] = float(theta[i])
    return p


if LNA_GRID == "fixed":
    model = model.freeze_lna_grid(build_dict(CENTER))
    print(f"[grid] lna grid frozen at the config centre: {model.specs['n_lna_PE']} nodes from "
          f"{float(model.specs['lna_grid_nodes'][0]):.6f}", flush=True)
elif LNA_GRID != "visibility":
    raise ValueError(f"PA_LNA_GRID={LNA_GRID!r} (fixed|visibility)")


def _bucket(n, cap):
    """batch size for n rows: the smallest NDEV * 4^k >= n, at most cap. A few fixed sizes keep the
    compiled shapes few while converged and stalled rows leave the batches."""
    b = NDEV
    while b < n and b < cap:
        b *= 4
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
    Dtt = pl.abcmb_cl_to_Dl(out.ClTT, out.l)        # for the low-ell likelihoods (muK^2 D_l)
    Dee = pl.abcmb_cl_to_Dl(out.ClEE, out.l)
    if USE_PLIK_FULL:                                # FULL plik: inner-profile the 47 nuisances
        cls2d = plf.abcmb_cls_to_clik(out.ClTT, out.ClTE, out.ClEE, out.l)   # (B,6,Lcol)
        chi2 = np.asarray(_plf_prof_B(cls2d)[0], dtype=float)
    elif USE_PLIK:                                   # plik-LITE: profile A_planck analytically
        Dte = pl.abcmb_cl_to_Dl(out.ClTE, out.l)
        m0 = pl.bin_model(Dtt, Dte, Dee)
        chi2 = np.asarray(pl.profile_A(m0, with_prior=True)[0], dtype=float)
    else:                                  # low-ell-only DEBUG path (no high-ell)
        chi2 = np.zeros(np.shape(Dee)[:-1], dtype=float)
    if lowee is not None:
        chi2 = chi2 + np.asarray(lowee.chi2(Dee), dtype=float)
    if lowtt is not None:
        chi2 = chi2 + np.asarray(lowtt.chi2(Dtt), dtype=float)
    return np.where(np.isfinite(chi2), chi2, 1e6)


def _trace(tag, BASE, FREE, X):
    if TRACE_EVALS:
        print(f"[eval] {tag} BEGIN {time.strftime('%H:%M:%S')} N={len(X)} ({' '.join(ORDER)})", flush=True)
        for b in range(len(X)):
            print(f"[eval]   row{b:3d} " + " ".join(f"{v:.7g}" for v in theta_of(BASE[b], FREE[b], X[b])),
                  flush=True)


def values_rows(BASE, FREE, X):
    """(N,) total chi^2 (profiled + low-ell) of the rows, through call_batched in bucketed chunks of
    <= VAL_CHUNK cosmologies (the per-device working set is set by the chunk, not by N)."""
    N = len(X); out = np.empty(N)
    _trace("value", BASE, FREE, X)
    for s, e, (b, f, x) in _padded(N, VAL_CHUNK, BASE, FREE, X):
        batch = [build_dict(theta_of(b[k], f[k], x[k])) for k in range(len(x))]
        out[s:e] = _chi2_from_out(model.call_batched(batch, shard=DO_SHARD))[:e - s]
    if TRACE_EVALS:
        print(f"[eval] value END {time.strftime('%H:%M:%S')} chi2 " + " ".join(f"{v:.3f}" for v in out), flush=True)
    return out


# ======================================================================
# gradient + Gauss-Newton curvature (AD only)
# ======================================================================
def _phys_to_derived(thD):
    """physical D-vector -> derived param dict, _to_float'd so the jvp tangent's
    inexact-array tree matches staged_cl_and_grad's _to_float'd primal."""
    p = dict(FIXED)
    for i, name in enumerate(ORDER):
        if name == "ln10As":
            p["A_s"] = jnp.exp(thD[i]) / 1e10
        else:
            p[name] = thD[i]
    return _bg_to_float(model.add_derived_parameters(p))


def _schur_gn(r, Jt, Jn, W, Hprior, free):
    """Gradient and Gauss-Newton Hessian of chi2 = r^T W r + prior(nu), profiled over the
    nuisances nu at their optimum. r (nd,); Jt (P',nd) = dr/dtheta; Jn (Q,nd) = dr/dnu; Hprior
    (Q,Q) the prior's Hessian; free (Q,) False for a nuisance sitting on a bound (held fixed).
    Profiled curvature = H_tt - H_tn H_nn^-1 H_nt over the free nuisances; the gradient is the
    partial one (envelope theorem: d chi2/d nu = 0 at the optimum)."""
    WJt = Jt @ W
    g = 2.0 * WJt @ r
    Htt = 2.0 * WJt @ Jt.T
    m = free.astype(r.dtype)
    Htn = 2.0 * (WJt @ Jn.T) * m[None, :]
    Hnn = (2.0 * (Jn @ W) @ Jn.T + Hprior) * m[:, None] * m[None, :] + jnp.diag(1.0 - m)
    H = Htt - Htn @ jnp.linalg.solve(Hnn, Htn.T)
    return g, 0.5 * (H + H.T)


def _lowl_gn(like, D_, dD):
    """chi2, gradient and generalized Gauss-Newton curvature dD^T H dD of a low-ell likelihood in
    the D_ell it reads (ell = lmin..lmax): H is its exact AD Hessian there, with negative
    eigenvalues set to zero so the term stays positive semidefinite."""
    sl = slice(like.lmin, like.lmax + 1)
    f = lambda d: like.chi2(D_.at[sl].set(d))
    d0 = D_[sl]
    c, gD = jax.value_and_grad(f)(d0)
    ev, V = jnp.linalg.eigh(jax.hessian(f)(d0))
    HD = (V * jnp.maximum(ev, 0.0)) @ V.T
    Jd = dD[:, sl]
    return c, Jd @ gD, Jd @ HD @ Jd.T


def _gn_one(ClTT, ClTE, ClEE, dTT, dTE, dEE, nu):
    """total chi2, gradient (P',) and Gauss-Newton matrix (P',P') of ONE cosmology from its C_ell
    (n_l,) and C_ell tangents (P', n_l); nu = its plik nuisance optimum (full plik only)."""
    lvec = model.SS.ells
    Pp = dTT.shape[0]
    c, g, H = 0.0, jnp.zeros(Pp), jnp.zeros((Pp, Pp))
    if USE_PLIK_FULL:
        cls2d = plf.abcmb_cls_to_clik(ClTT, ClTE, ClEE, lvec)               # (6, Lcol)
        dcls = plf.abcmb_cls_to_clik(dTT, dTE, dEE, lvec)                   # (P', 6, Lcol), linear
        res = plf.residual
        r = res(cls2d, nu)
        Jt = jax.vmap(lambda dc: jax.jvp(lambda c2: res(c2, nu), (cls2d,), (dc,))[1])(dcls)
        Jn = jax.jacfwd(lambda n: res(cls2d, n))(nu).T
        free = (nu - plf.lo > 1e-6 * plf.scale) & (plf.hi - nu > 1e-6 * plf.scale)
        c = r @ plf.siginv @ r + plf.prior_penalty(nu)
        g, H = _schur_gn(r, Jt, Jn, plf.siginv, plf.prior_hessian(), free)
    elif USE_PLIK:
        Dl = lambda C: pl.abcmb_cl_to_Dl(C, lvec)
        m0 = pl.bin_model(Dl(ClTT), Dl(ClTE), Dl(ClEE))
        dm0 = jax.vmap(lambda a, b, e: jax.jvp(pl.bin_model, (Dl(ClTT), Dl(ClTE), Dl(ClEE)),
                                               (Dl(a), Dl(b), Dl(e)))[1])(dTT, dTE, dEE)
        A = jax.lax.stop_gradient(pl.profile_A(m0, with_prior=True)[1])
        r = pl.X_data - m0 / A ** 2
        c = r @ pl.invcov @ r + ((A - 1.0) / 0.0025) ** 2
        g, H = _schur_gn(r, -dm0 / A ** 2, (2.0 * m0 / A ** 3)[None, :], pl.invcov,
                         jnp.full((1, 1), 2.0 / 0.0025 ** 2), jnp.ones(1, bool))
    for like, C, dC in ((lowtt, ClTT, dTT), (lowee, ClEE, dEE)):
        if like is not None:
            cl_, gl, Hl = _lowl_gn(like, pl.abcmb_cl_to_Dl(C, lvec), pl.abcmb_cl_to_Dl(dC, lvec))
            c, g, H = c + cl_, g + gl, H + Hl
    return c, g, H


_gn_B = jax.jit(jax.vmap(_gn_one))


def _gn_block(BASE, FREE, X):
    """One staged AD call over a block of rows: (chi2 (B,), grad (B,P'), GN (B,P',P')). Direction j
    of row b is SIGMA[i] e_i, i = FREE[b, j] (so gradient and curvature are in scaled coords)."""
    B, Pp = X.shape
    thetas = [jnp.asarray(theta_of(BASE[b], FREE[b], X[b])) for b in range(B)]
    full_ps = [_phys_to_derived(t) for t in thetas]
    per = []                                       # [B][P'] filtered tangent dicts
    for b in range(B):
        dots = []
        for j in range(Pp):
            i = int(FREE[b, j])
            _, fd = jax.jvp(_phys_to_derived, (thetas[b],), (jnp.zeros(D).at[i].set(SIG[i]),))
            dots.append(eqx.filter(fd, eqx.is_inexact_array))
        per.append(dots)
    params_dots = [jax.tree.map(lambda *xs: jnp.stack(xs), *[per[b][j] for b in range(B)])
                   for j in range(Pp)]
    (ClTT, ClTE, ClEE), cl_dots = staged_cl_and_grad(
        model, full_ps, params_dots, k_chunk_size=GRAD_KCHUNK, shard=DO_SHARD,
        dir_chunk=GRAD_DIRCHUNK, spec_b_local=GRAD_SPECB)
    dTT, dTE, dEE = (jnp.stack([cd[s] for cd in cl_dots], axis=1) for s in range(3))  # (B,P',n_l)
    if USE_PLIK_FULL:
        nu = _plf_prof_B(plf.abcmb_cls_to_clik(ClTT, ClTE, ClEE, model.SS.ells))[1]
    else:
        nu = jnp.zeros((B, 1))
    c, g, H = _gn_B(ClTT, ClTE, ClEE, dTT, dTE, dEE, nu)
    return np.asarray(c, float), np.asarray(g, float), np.asarray(H, float)


def grad_gn_rows(BASE, FREE, X):
    """(chi2 (N,), grad (N,P'), GN (N,P',P')) for every row, in bucketed blocks of <= AD_BCHUNK."""
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
# Levenberg-Marquardt over the row set (lockstep), with physical bounds
# ======================================================================
def _proj_grad(G, X, XLO, XHI):
    """gradient with the components that push into an active bound zeroed (lm_tools.proj_grad)."""
    return proj_grad(G, X, XLO, XHI)


def _lm_step(g, H, x, xlo, xhi, lam):
    """one projected LM step capped at STEP_MAX (lm_tools.lm_step). Returns (d, GN predicted drop)."""
    return lm_step(g, H, x, xlo, xhi, lam, STEP_MAX)


def _envelope(pv, f):
    """(unique POI values, lowest chi2 at each): the profile of a row set with several starts per
    grid point."""
    u, inv = np.unique(np.asarray(pv, float), return_inverse=True)
    m = np.full(len(u), np.inf)
    np.minimum.at(m, inv, np.asarray(f, float))
    return u, m


def lm_rows(BASE, FREE, XLO, XHI, X0, POI_IDX=None, PV=None, maxit=MAXIT, gtol=GTOL,
            log_prefix="", ckpt_path=None, resume_state=None, EXTRA=None):
    """Levenberg-Marquardt over a flat row set. BASE (N,D), FREE (N,P'), XLO/XHI/X0 (N,P'); POI_IDX/PV
    (N,) group the rows into profiles for the interval-stability stop (None: no early stop); several
    rows at one PV are starts of the same point and the profile takes their lowest chi2. EXTRA (N,)
    marks the multistart rows: with MS_WAIT the stop waits until none is active. Returns a dict of
    per-row arrays: X, F (values-path chi2), G, H (at X), gnorm (projected, exact AD), stalled, lam,
    it. The full state is checkpointed every iteration (ckpt_path); resume_state restarts from such
    a checkpoint."""
    N, Pp = X0.shape
    if resume_state is not None:
        S = {k: np.array(resume_state[k]) for k in ("X", "F", "G", "H", "lam", "nfail", "stalled")}
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
                 lam=np.full(N, LM_LAMBDA0), nfail=np.zeros(N, int), stalled=bad)
        it0 = 0
    X, F, G, H, lam, nfail, stalled = (S[k] for k in ("X", "F", "G", "H", "lam", "nfail", "stalled"))
    gproj = _proj_grad(G, X, XLO, XHI)
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
    for it in range(it0, maxit):
        act = (np.abs(gproj).max(1) > gtol) & ~stalled
        if not act.any():
            break
        A = np.where(act)[0]
        cand = np.empty((LM_K, len(A), Pp)); pred = np.empty((LM_K, len(A)))
        for k in range(LM_K):
            for a, b in enumerate(A):
                d, pred[k, a] = _lm_step(G[b], H[b], X[b], XLO[b], XHI[b], lam[b] * LM_LADDER[k])
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
        stalled |= (nfail >= STALL_MAX) | (lam > LM_LAMBDA_MAX)
        gproj = _proj_grad(G, X, XLO, XHI)
        if log_prefix:
            print(f"  {log_prefix} it{it}: min={F.min():.2f} ||g||max(active)="
                  f"{np.abs(gproj[act]).max():.2e} active={len(A)} moved={len(moved)} "
                  f"stalled={int(stalled.sum())} lam_med={np.median(lam[A]):.1e} "
                  f"({time.strftime('%H:%M:%S')})", flush=True)
        if ckpt_path:
            tmp = ckpt_path + ".tmp.npz"
            np.savez(tmp, BASE=BASE, FREE=FREE, X0=X0, X=X, F=F, G=G, H=H, lam=lam, nfail=nfail,
                     stalled=stalled, it=it + 1, gtol=gtol)
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
                    int(((np.abs(gproj).max(1) > gtol) & ~stalled & EXTRA).sum())
                if nx:
                    print(f"  {log_prefix} intervals stable (worst move {worst:.2e} grid_sig) but "
                          f"{nx} extra starts still active -> continuing", flush=True)
                else:
                    print(f"  {log_prefix} intervals stable: worst move {worst:.2e} grid_sig < SIGTOL "
                          f"{SIGTOL:.0e} over {SIGTOL_PATIENCE} iters -> stop at it{it}", flush=True)
                    break
    return dict(X=X, F=F, G=G, H=H, gnorm=np.abs(gproj).max(1), stalled=stalled, lam=lam, it=it)


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
# deflated Newton walk out of the known minima (multimodality; lm_tools.deflate_step)
# ======================================================================
def deflated_rows(BASE, FREE, XLO, XHI, X0, MINS, NMIN, maxit=DF_MAXIT, gtol=GTOL, log_prefix="[defl]"):
    """Deflated Newton iteration (Farrell, Birkisson & Funke 2015) over a flat row set, every known
    stationary point of a row deflated at once: M(x) = prod_i (|x - x_i|^-DF_P + DF_ALPHA), and the
    step is the row's undamped Gauss-Newton step d rescaled to d / (1 - grad(log M).d), so it reuses
    g and the GN matrix; it is capped at STEP_MAX and projected into the box. A full Newton step
    toward a known point comes out reversed (the row walks away from it, up its valley, until the
    GN step points into another basin), a step away from it is shortened; no step is damped or
    rejected -- this is a bounded walk, not a descent. The walk ends when a row's projected gradient
    drops under gtol (a stationary point: `new` when > MS_DX from every known one, else `recaptured`),
    when it reaches the +-XBOX box (`escaped`), when ABCMB fails there (`stalled`, the row keeps its
    last point), or after maxit steps (`maxit`; the caller finishes those with ordinary LM).
    MINS (N, m, P') padded, NMIN (N,) the count per row. Returns per-row X, F (values path), G, H,
    gnorm, status."""
    N, Pp = X0.shape
    X = np.array(X0, float)
    _, G, H = grad_gn_rows(BASE, FREE, X)
    fin = np.isfinite(G).all(1) & np.isfinite(H).all((1, 2))
    G = np.where(fin[:, None], G, 0.0); H = np.where(fin[:, None, None], H, 0.0)
    status = np.array(["active"] * N, dtype="U12")
    status[~fin] = "stalled"
    mins_of = lambda b: MINS[b, :NMIN[b]]

    def _classify(b):
        if ((X[b] <= -XBOX + 1e-9) | (X[b] >= XBOX - 1e-9)).any():
            return "escaped"
        gp = proj_grad(G[b:b + 1], X[b:b + 1], XLO[b:b + 1], XHI[b:b + 1])[0]
        if np.abs(gp).max() < gtol:
            near = NMIN[b] > 0 and bool((np.abs(mins_of(b) - X[b]).max(1) <= MS_DX).any())
            return "recaptured" if near else "new"
        return "active"

    for b in np.where(status == "active")[0]:
        status[b] = _classify(b)
    for it in range(maxit):
        A = np.where(status == "active")[0]
        if not len(A):
            break
        Xc = X.copy()
        for b in A:
            d, _ = lm_step(G[b], H[b], X[b], XLO[b], XHI[b], LM_LAMBDA_MIN, np.inf)
            d = deflate_step(d, X[b], mins_of(b), DF_P, DF_ALPHA)
            dm = np.abs(d).max()
            if dm > STEP_MAX:
                d *= STEP_MAX / dm
            Xc[b] = np.clip(X[b] + d, XLO[b], XHI[b])
        _, Gc, Hc = grad_gn_rows(BASE[A], FREE[A], Xc[A])
        fin = np.isfinite(Gc).all(1) & np.isfinite(Hc).all((1, 2))
        for a, b in enumerate(A):
            if fin[a]:
                X[b], G[b], H[b] = Xc[b], Gc[a], Hc[a]
                status[b] = _classify(b)
            else:
                status[b] = "stalled"
        print(f"  {log_prefix} it{it}: walking={len(A)} " +
              " ".join(f"{k}={int((status == k).sum())}" for k in ("new", "recaptured", "escaped", "stalled")) +
              f" ({time.strftime('%H:%M:%S')})", flush=True)
    status[status == "active"] = "maxit"
    F = values_rows(BASE, FREE, X)
    gproj = proj_grad(G, X, XLO, XHI)
    return dict(X=X, F=F, G=G, H=H, gnorm=np.abs(gproj).max(1), status=status)


# ======================================================================
# lockstep driver across ALL POIs: base + multistart rows, flags, deflation, branches, envelope
# ======================================================================
KINDS = ("warm", "spread", "deflation", "branch")      # row kinds, T["KIND"] = index into this


def _grid_for(poi):
    """NPTS points over the config 'grid' range, else GRID_CEN +- NSIG*GRID_SIG, clipped to the
    POI's physical bounds (a clipped end lands exactly on the bound)."""
    i = ORDER.index(poi)
    lo, hi = GRID_RANGE.get(poi, (GRID_CEN[i] - NSIG * GRID_SIG[i], GRID_CEN[i] + NSIG * GRID_SIG[i]))
    return np.linspace(max(lo, BLO[i]), min(hi, BHI[i]), NPTS)


def _bound_of(i):
    return (None if not np.isfinite(BLO[i]) else BLO[i], None if not np.isfinite(BHI[i]) else BHI[i])


def _ckpt_path(outdir, what):
    return os.path.join(outdir, f"profile_prod_ad_{what}{TAG}{'_r%d' % RANK if POI_SLICE else ''}.npz")


def _stage_rank(stage):
    """lm -> 0, defl<r> -> 2r - 1, branch<r> -> 2r (the order of the phases)."""
    if stage.startswith("defl"):
        return 2 * int(stage[4:]) - 1
    if stage.startswith("branch"):
        return 2 * int(stage[6:])
    return 0


def _save_table(path, T, stage):
    """checkpoint the row table after a phase."""
    tmp = path + ".tmp.npz"
    np.savez(tmp, stage=stage, gtol=GTOL, **T)
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
                and np.array_equal(st["FREE"][:n], FREE) and float(st["gtol"]) == GTOL):
            T = {k: st[k] for k in st.files if k not in ("stage", "gtol")}
            print(f"[resume] row table after stage {st['stage']} matches; continuing ({path})", flush=True)
            return T, str(st["stage"])
        print("[resume] row table grid/gtol mismatch -> fresh start", flush=True)
    except Exception as ex:
        print(f"[resume] could not load the row table ({ex}) -> fresh start", flush=True)
    return None, None


def _append(T, **rows):
    for k, v in rows.items():
        T[k] = np.concatenate([T[k], np.asarray(v, dtype=T[k].dtype)])


def _row_conv(T):
    """(converged, at_box) per row: ||g||_AD < GTOL off the +-XBOX box and not stalled. A deflation
    row that ended on a stationary point (new or recaptured) qualifies like any other."""
    at_box = ((T["X"] <= -XBOX + 1e-9) | (T["X"] >= XBOX - 1e-9)).any(1)
    return (T["gnorm"] < GTOL) & ~at_box & ~T["stalled"], at_box


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


def _classify_rows(X, gnorm, stalled, mins_list):
    """status of finished rows: new / recaptured / stalled / maxit (lm_rows output + known minima)."""
    at_box = ((X <= -XBOX + 1e-9) | (X >= XBOX - 1e-9)).any(1)
    out = np.array(["maxit"] * len(X), dtype="U12")
    for b in range(len(X)):
        if stalled[b]:
            out[b] = "stalled"
        elif at_box[b]:
            out[b] = "escaped"
        elif gnorm[b] < GTOL:
            m = mins_list[b]
            out[b] = "recaptured" if len(m) and (np.abs(m - X[b]).max(1) <= MS_DX).any() else "new"
    return out


def profile_lockstep(pois, outdir):
    """Build the row set (every POI x grid point, plus MS_K spread starts at MS_NPTS points per POI),
    warm-start, run LM in lockstep, then MS_ROUNDS rounds of flags -> deflation -> branches, and
    write one npz/png per POI from the lower envelope over every row. Each phase checkpoints the row
    table (resumable)."""
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
          f"warm={'joint fit' if MLE_PREPASS else 'config centre' if WARM else 'cold'}", flush=True)
    t0 = time.perf_counter()
    tpath = _ckpt_path(outdir, "TABLE")
    T, stage = _load_table(tpath, BASE, FREE)

    def _done_line(what):
        conv, at_box = _row_conv(T)
        print(f"[lockstep] {what} {time.perf_counter() - t0:.0f}s RSS={rss_gb():.1f}GB; rows "
              f"{len(T['PV'])} (" + ", ".join(f"{k} {int((T['KIND'] == i).sum())}" for i, k in enumerate(KINDS))
              + f"); converged {int(conv.sum())} (||g||_AD<{GTOL:.1e}); stalled {int(T['stalled'].sum())}; "
              f"on the XBOX box {int(at_box.sum())}; max||g||_AD={T['gnorm'].max():.2e}", flush=True)

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
                        and np.array_equal(st["FREE"], FREE_f) and float(st["gtol"]) == GTOL):
                    resume_state = {k: st[k] for k in st.files}
                    print(f"[resume] LM checkpoint @ it{int(st['it'])} matches; continuing ({ckpt})", flush=True)
                else:
                    print("[resume] LM checkpoint grid/gtol mismatch -> fresh start", flush=True)
            except Exception as ex:
                print(f"[resume] could not load the LM checkpoint ({ex}) -> fresh start", flush=True)
        if resume_state is not None and "X0" in resume_state:
            X0_f = np.array(resume_state["X0"], float)
        else:
            X0 = np.clip(xw[FREE], XLO, XHI) if WARM else np.clip(np.zeros((N, P)), XLO, XHI)
            # ---- linear-response warm start: each row starts at the Gaussian conditional optimum of
            # the warm point's Gauss-Newton matrix (rows far along a degeneracy need not walk there at
            # STEP_MAX per iteration); one batched value call scores both starts, each row keeps the
            # lower chi2. ----
            if WARM and WARM_LR:
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
                 START=START, ROUND=np.zeros(Nf, int), X0=X0_f, X=S["X"], F=S["F"], G=S["G"], H=S["H"],
                 gnorm=S["gnorm"], stalled=S["stalled"], status=np.array(["lm"] * Nf, dtype="U12"),
                 barrier=np.full(Nf, np.nan))
        stage = "lm"
        _save_table(tpath, T, stage)
        try:
            os.remove(ckpt)
        except Exception:
            pass
        _done_line("LM done")

    # ================= rounds: flags -> deflation -> branches =================
    for rnd in range(1, MS_ROUNDS + 1):
        if _stage_rank(stage) < 2 * rnd - 1:
            flags = _flags(T, pois_idx)
            nflag = sum(v[0] for v in flags.values())
            flagged = [(k, v) for k, v in flags.items()
                       if v[0] or (MS_FORCE and (T["KIND"][v[3]] == 1).any())]
            print(f"[round {rnd}] {nflag} flagged grid points of {len(flags)}"
                  + (f" (+{len(flagged) - nflag} forced by PA_MS_FORCE)" if MS_FORCE else ""), flush=True)
            for (p, j), (f, why, reps, r) in flagged:
                print(f"  [{ORDER[p]}@{grid_of[p][j]:.5g}] {why or 'forced'}; known minima: "
                      + (", ".join(f"chi2={T['F'][i]:.3f}" for i in reps) or "none"), flush=True)
            if not flagged:
                stage = f"branch{rnd}"
                _save_table(tpath, T, stage)
                break
            # ---- deflation rows: the point's warm + spread starts (those within MS_DX of a known
            # minimum dropped), topped up to MS_K + 1 with spreads around its best minimum on that
            # minimum's own GN matrix; every known minimum of the point deflated at once ----
            dB, dFR, dLO, dHI, dX0, dP, dJ, dPV, dMIN, dSTART = ([] for _ in range(10))
            mmax = max(1, max(len(v[2]) for _, v in flagged))
            for (p, j), (f, why, reps, r) in flagged:
                mins = T["X"][reps] if reps else np.zeros((0, P))
                starts = T["X0"][r[T["KIND"][r] <= 1]]
                starts = np.unique(np.round(starts, 12), axis=0)
                if len(mins):
                    far = np.array([(np.abs(mins - s).max(1) > MS_DX).all() for s in starts], bool)
                    starts = starts[far]
                    if len(starts) < MS_K + 1:
                        b0 = reps[0]
                        starts = np.concatenate([starts, spread_starts(
                            T["X"][b0], T["H"][b0], T["XLO"][b0], T["XHI"][b0], MS_K + 1 - len(starts), MS_SPREAD)])
                b0 = int(r[0])
                for s_i, s in enumerate(starts):
                    dB.append(T["BASE"][b0]); dFR.append(T["FREE"][b0]); dLO.append(T["XLO"][b0]); dHI.append(T["XHI"][b0])
                    dX0.append(s); dP.append(p); dJ.append(j); dPV.append(T["PV"][b0]); dSTART.append(s_i)
                    m = np.full((mmax, P), np.nan); m[:len(mins)] = mins; dMIN.append((m, len(mins)))
            dB, dFR, dLO, dHI, dX0 = (np.array(a) for a in (dB, dFR, dLO, dHI, dX0))
            MINS = np.array([m for m, _ in dMIN]); NMIN = np.array([n for _, n in dMIN], int)
            print(f"[round {rnd}] deflation: {len(dX0)} walks from {len(flagged)} points, up to {mmax} minima "
                  f"deflated each (p={DF_P:g}, alpha={DF_ALPHA:g}, <= {DF_MAXIT} steps)", flush=True)
            R = deflated_rows(dB, dFR, dLO, dHI, dX0, MINS, NMIN, log_prefix=f"[defl{rnd}]")
            # ---- walks that ran out of steps: finish them with ordinary LM ----
            mx = np.where(R["status"] == "maxit")[0]
            if len(mx):
                print(f"[round {rnd}] {len(mx)} walks unfinished after {DF_MAXIT} steps -> ordinary LM", flush=True)
                S2 = lm_rows(dB[mx], dFR[mx], dLO[mx], dHI[mx], R["X"][mx], maxit=BR_MAXIT, log_prefix=f"[defl{rnd}-lm]")
                for k in ("X", "F", "G", "H", "gnorm"):
                    R[k][mx] = S2[k]
                R["status"][mx] = _classify_rows(S2["X"], S2["gnorm"], S2["stalled"],
                                                 [MINS[b, :NMIN[b]] for b in mx])
            n_new = int((R["status"] == "new").sum())
            print(f"[round {rnd}] deflation done: " + " ".join(
                f"{k}={int((R['status'] == k).sum())}" for k in ("new", "recaptured", "escaped", "stalled", "maxit")),
                flush=True)
            # ---- each new stationary point: chi2 against the point's best known minimum, and the
            # barrier along the straight segment between them (values only; an upper bound) ----
            bar = np.full(len(dX0), np.nan)
            new_rows = np.where(R["status"] == "new")[0]
            if len(new_rows):
                segB, segF, segX, owner = [], [], [], []
                ts = np.linspace(0.0, 1.0, 9)[1:-1]
                for a in new_rows:
                    if NMIN[a] == 0:
                        continue
                    xk = MINS[a, 0]
                    for t in ts:
                        segB.append(dB[a]); segF.append(dFR[a]); segX.append(xk + t * (R["X"][a] - xk)); owner.append(a)
                if segB:
                    fs = values_rows(np.array(segB), np.array(segF), np.array(segX)); owner = np.array(owner)
                    for a in new_rows:
                        if NMIN[a] == 0:
                            continue
                        p, j = dP[a], dJ[a]
                        fk = T["F"][flags[(p, j)][2][0]]
                        bar[a] = fs[owner == a].max() - max(fk, R["F"][a])
                        print(f"  [{ORDER[p]}@{grid_of[p][j]:.5g}] new stationary point chi2={R['F'][a]:.3f} "
                              f"({R['F'][a] - fk:+.3f} vs the known minimum); barrier along the segment "
                              f">= {bar[a]:.3f}; |dx|max={np.abs(R['X'][a] - MINS[a, 0]).max():.2f}", flush=True)
            _append(T, BASE=dB, FREE=dFR, XLO=dLO, XHI=dHI, POI=np.array(dP, int), J=np.array(dJ, int),
                    PV=np.array(dPV, float), KIND=np.full(len(dX0), 2, int), START=np.array(dSTART, int),
                    ROUND=np.full(len(dX0), rnd, int), X0=dX0, X=R["X"], F=R["F"], G=R["G"], H=R["H"],
                    gnorm=R["gnorm"], stalled=(R["status"] == "stalled"), status=R["status"], barrier=bar)
            stage = f"defl{rnd}"
            _save_table(tpath, T, stage)
        if _stage_rank(stage) < 2 * rnd:
            # ---- branches: an ordinary LM run from each new stationary point of this round at every
            # grid point of its POI (at most MS_BRANCH_MAX seeds per POI, lowest chi2 first) ----
            seeds = []
            for p in pois_idx:
                cand = np.where((T["POI"] == p) & (T["KIND"] == 2) & (T["ROUND"] == rnd)
                                & ((T["status"] == "new") if MS_BRANCH_FROM == "new" else True))[0]
                cand = cand[np.argsort(T["F"][cand])]
                seeds += [(p, int(c)) for c in cand[:MS_BRANCH_MAX]]
            if seeds:
                bB, bFR, bLO, bHI, bX0, bP, bJ, bPV, bST = ([] for _ in range(9))
                for s_i, (p, c) in enumerate(seeds):
                    for j in range(NPTS):
                        b0 = int(np.where((T["POI"] == p) & (T["J"] == j) & (T["KIND"] == 0))[0][0])
                        bB.append(T["BASE"][b0]); bFR.append(T["FREE"][b0]); bLO.append(T["XLO"][b0]); bHI.append(T["XHI"][b0])
                        bX0.append(np.clip(T["X"][c], T["XLO"][b0], T["XHI"][b0])); bP.append(p); bJ.append(j)
                        bPV.append(T["PV"][b0]); bST.append(s_i)
                bB, bFR, bLO, bHI, bX0 = (np.array(a) for a in (bB, bFR, bLO, bHI, bX0))
                print(f"[round {rnd}] branches: {len(seeds)} seeds -> {len(bX0)} rows: "
                      + ", ".join(f"{ORDER[p]}@{grid_of[p][T['J'][c]]:.5g} chi2={T['F'][c]:.3f}" for p, c in seeds),
                      flush=True)
                S3 = lm_rows(bB, bFR, bLO, bHI, bX0, POI_IDX=np.array(bP, int), PV=np.array(bPV, float),
                             maxit=BR_MAXIT, log_prefix=f"[branch{rnd}]")
                _append(T, BASE=bB, FREE=bFR, XLO=bLO, XHI=bHI, POI=np.array(bP, int), J=np.array(bJ, int),
                        PV=np.array(bPV, float), KIND=np.full(len(bX0), 3, int), START=np.array(bST, int),
                        ROUND=np.full(len(bX0), rnd, int), X0=bX0, X=S3["X"], F=S3["F"], G=S3["G"], H=S3["H"],
                        gnorm=S3["gnorm"], stalled=S3["stalled"],
                        status=np.array(["branch"] * len(bX0), dtype="U12"), barrier=np.full(len(bX0), np.nan))
                for s_i, (p, c) in enumerate(seeds):
                    rb = np.where((T["KIND"] == 3) & (T["ROUND"] == rnd) & (T["START"] == s_i) & (T["POI"] == p))[0]
                    base = np.array([T["F"][(T["POI"] == p) & (T["J"] == j) & (T["KIND"] < 3)].min() for j in T["J"][rb]])
                    below = T["F"][rb] < base - MS_DCHI2
                    print(f"  branch {s_i} ({ORDER[p]}): converged {int((T['gnorm'][rb] < GTOL).sum())}/{len(rb)}, "
                          f"below the earlier envelope at {int(below.sum())} grid points"
                          + (f" (by up to {float((base - T['F'][rb]).max()):.3f})" if below.any() else ""), flush=True)
            else:
                print(f"[round {rnd}] no new stationary point -> no branches", flush=True)
            stage = f"branch{rnd}"
            _save_table(tpath, T, stage)
    _done_line("all phases done")

    # ================= envelope + outputs =================
    conv, at_box = _row_conv(T)
    flags = _flags(T, pois_idx)
    ev = np.linalg.eigvalsh(T["H"])
    for poi in pois:
        i = ORDER.index(poi); grid = grid_of[i]; bnd = _bound_of(i)
        rows = np.where(T["POI"] == i)[0]
        e = np.array([r[np.argmin(T["F"][r])] for r in (rows[T["J"][rows] == j] for j in range(NPTS))], int)
        chi2 = T["F"][e]
        lo1, mid, hi1, reg1 = poi_interval(grid, chi2, CL1, bnd)
        lo2, _, hi2, reg2 = poi_interval(grid, chi2, CL2, bnd)
        lo95, _, hi95, reg95 = poi_interval(grid, chi2, CL95, bnd)
        sw, mw, dw = wald_fit(grid, chi2)
        mmax = max(1, max(len(flags[(i, j)][2]) for j in range(NPTS)))
        minima_x = np.full((NPTS, mmax, P), np.nan); minima_chi2 = np.full((NPTS, mmax), np.nan)
        minima_kind = np.full((NPTS, mmax), "", dtype="U12")
        for j in range(NPTS):
            for m, rr in enumerate(flags[(i, j)][2]):
                minima_x[j, m] = T["X"][rr]; minima_chi2[j, m] = T["F"][rr]; minima_kind[j, m] = KINDS[T["KIND"][rr]]
        flag = np.array([flags[(i, j)][0] for j in range(NPTS)], bool)
        reason = np.array([flags[(i, j)][1] for j in range(NPTS)], dtype="U96")
        npz = os.path.join(outdir, f"profile_prod_ad_{poi}{TAG}.npz")
        np.savez(npz, poi=poi, poi_grid=grid, chi2=chi2, xstar=T["X"][e], gnorm=T["gnorm"][e],
                 converged=conv[e], stalled=T["stalled"][e], at_box=at_box[e],
                 gn_eig=ev[e], hess_pd=ev[e].min(1) > 0,
                 nuis=np.array([ORDER[k] for k in nuis_idx_of(i)]), done=True,
                 sigma1=np.array([lo1, mid, hi1]), sigma2=np.array([lo2, hi2]),
                 cl95=np.array([lo95, hi95]), bound=np.array([np.nan if b is None else b for b in bnd]),
                 regions1=np.array(reg1, float).reshape(-1, 2), regions2=np.array(reg2, float).reshape(-1, 2),
                 regions95=np.array(reg95, float).reshape(-1, 2),
                 sigma_wald=sw, wald_vertex=mw, wald_misfit=dw, interval_mode=_interval_mode(grid, chi2),
                 n_starts=np.array([int((T["J"][rows] == j).sum()) for j in range(NPTS)]),
                 n_minima=np.array([len(flags[(i, j)][2]) for j in range(NPTS)]), flag=flag, flag_reason=reason,
                 minima_x=minima_x, minima_chi2=minima_chi2, minima_kind=minima_kind,
                 rows_j=T["J"][rows], rows_kind=T["KIND"][rows], rows_start=T["START"][rows],
                 rows_round=T["ROUND"][rows], rows_status=T["status"][rows], rows_x0=T["X0"][rows],
                 rows_x=T["X"][rows], rows_chi2=T["F"][rows], rows_gnorm=T["gnorm"][rows],
                 rows_converged=conv[rows], rows_stalled=T["stalled"][rows], rows_at_box=at_box[rows],
                 rows_barrier=T["barrier"][rows], kinds=np.array(KINDS),
                 ms=np.array([MS_K, MS_NPTS, MS_SPREAD, MS_DX, MS_DCHI2, DF_P, DF_ALPHA]),
                 gtol=GTOL, use_lowee=USE_LOWEE, use_lowtt=USE_LOWTT, config=CONFIG_ABS)
        j = int(np.nanargmin(chi2)); nval = int(np.isfinite(chi2).sum())
        edge = (j <= 1 and grid[0] != BLO[i]) or (j >= NPTS - 2 and grid[-1] != BHI[i])
        warn = ""
        if edge or not (np.isfinite(lo1) and np.isfinite(hi1)):
            why = []
            if edge:
                why.append(f"min at grid index {j} (not a physical bound)")
            if not np.isfinite(lo1):
                why.append("lower 68% end beyond the grid")
            if not np.isfinite(hi1):
                why.append("upper 68% end beyond the grid")
            warn = "  ** CENTERING: " + "; ".join(why) + " (widen PA_NSIG / the config grid, or PA_MLE_PREPASS=1)"
        kind = "one-sided" if any(v in [b for b in bnd if b is not None] for v in (lo1, hi1)) else "two-sided"
        multi = f"; 68% UNION of {len(reg1)} regions {[(round(a, 5), round(b, 5)) for a, b in reg1]}" if len(reg1) > 1 else ""
        print(f"[{poi}] minchi2={chi2[j]:.2f} at {poi}={grid[j]:.5f}; 68% [{lo1:.5f},{hi1:.5f}] ({kind}, "
              f"best {mid:.5f}); 95% [{lo95:.5f},{hi95:.5f}]; wald sigma {sw:.5f} (misfit {dw:.2f}); "
              f"converged {int(conv[e].sum())}/{nval}, stalled {int(T['stalled'][e].sum())}; "
              f"max||g||_AD={np.nanmax(T['gnorm'][e]):.2e}; {len(rows)} rows ("
              + ", ".join(f"{k} {int((T['KIND'][rows] == ki).sum())}" for ki, k in enumerate(KINDS))
              + f"); flagged {int(flag.sum())}/{NPTS} points; envelope from "
              + ", ".join(f"{k} {int((T['KIND'][e] == ki).sum())}" for ki, k in enumerate(KINDS)
                          if (T["KIND"][e] == ki).any())
              + f"{multi} -> {npz}{warn}", flush=True)
        _plot(poi, grid, chi2, npz, rows=(T["PV"][rows], T["F"][rows], T["KIND"][rows]), regions=reg1)
    try:
        if os.path.exists(tpath):
            os.remove(tpath)                           # run finished cleanly
    except Exception:
        pass


def _plot(poi, grid, chi2, npz, rows=None, regions=None):
    """Delta-chi2 of the envelope; with rows = (pv, chi2, kind) every other row's end point too
    (starts, deflation walks, branches), and the 68% union regions shaded when there are several."""
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        i = ORDER.index(poi); bnd = _bound_of(i)
        c0 = np.nanmin(chi2); d = chi2 - c0
        fig, ax = plt.subplots(figsize=(6, 4.3))
        if rows is not None:
            pv, f, kind = rows
            env = np.interp(pv, grid, chi2)
            for ki, (mk, col, lab) in enumerate((("s", "0.65", "warm start"), ("^", "0.65", "spread start"),
                                                 ("x", "tab:red", "deflation walk"), ("+", "tab:purple", "branch"))):
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
        for cl, col, lab in ((CL1, "g", "68%"), (CL95, "orange", "95%")):
            thr = np.full_like(xs, _chi2_dist.ppf(cl, 1))
            fin = [(b, s) for b, s in ((bnd[0], 1), (bnd[1], -1)) if b is not None]
            if fin and np.isfinite(sw):
                b, sgn = min(fin, key=lambda t: abs(grid[np.nanargmin(chi2)] - t[0]))
                thr = t_threshold(cl, sgn * (xs - b) / sw)
            ax.plot(xs, thr, "--", lw=0.8, color=col, label=f"{lab} threshold")
        for b in bnd:
            if b is not None:
                ax.axvline(b, color="k", lw=1.0, label="physical bound")
        ax.axvline(CENTER[i], ls=":", color="gray", lw=0.8, label=f"fiducial {poi}")
        ax.set_xlabel(poi); ax.set_ylabel(r"$\Delta\chi^2$ (profiled)")
        ax.set_ylim(0, max(10, float(np.nanmax(d)) * 1.05)); ax.legend(fontsize=7)
        like = ["full plik (nuisances profiled)" if USE_PLIK_FULL else "plik-lite"] if USE_PLIK else []
        like += ["lowTT"] * bool(USE_LOWTT) + ["lowEE"] * bool(USE_LOWEE)
        ax.set_title(f"profile of {poi}: {' + '.join(like)}")
        fig.tight_layout(); fig.savefig(npz.replace(".npz", ".png"), dpi=120)
    except Exception as e:
        print(f"[{poi}] plot skipped: {e}", flush=True)


def _resume_skip(pois, outdir):
    """drop POIs whose npz is already done (RESUME)."""
    out = []
    for poi in pois:
        npz = os.path.join(outdir, f"profile_prod_ad_{poi}{TAG}.npz")
        if RESUME and os.path.exists(npz):
            try:
                if bool(np.load(npz, allow_pickle=True)["done"]):
                    print(f"[{poi}] resume: done, skip", flush=True); continue
            except Exception:
                pass
        out.append(poi)
    return out


def main():
    outdir = os.path.join(_HERE, "results")
    os.makedirs(outdir, exist_ok=True)
    print(f"rank {RANK}/{NPROC} devices={jax.devices()} config={CONFIG_ABS} "
          f"POIs={POIS} NPTS={NPTS} NSIG={NSIG} GTOL={GTOL} rtol={RTOL} shard={DO_SHARD} "
          f"lowEE={USE_LOWEE} lowTT={USE_LOWTT} warm={WARM} multistart={MS_K}x{MS_NPTS}pts "
          f"(spread {MS_SPREAD}, dx {MS_DX}, dchi2 {MS_DCHI2}; rounds {MS_ROUNDS}; deflation p={DF_P:g} "
          f"alpha={DF_ALPHA:g}) poi_slice={POI_SLICE} nproc={NPROC} bounds="
          f"{ {k: (BLO[i], BHI[i]) for i, k in enumerate(ORDER) if np.isfinite(BLO[i]) or np.isfinite(BHI[i])} }",
          flush=True)
    if POI_SLICE and NPROC > 1:
        pois = _resume_skip(POIS[RANK::NPROC], outdir)     # each rank: a disjoint POI subset
    else:
        pois = _resume_skip(POIS, outdir) if RANK == 0 else []
    if pois:
        profile_lockstep(pois, outdir)
    print(f"rank {RANK}: done", flush=True)


if __name__ == "__main__":
    main()

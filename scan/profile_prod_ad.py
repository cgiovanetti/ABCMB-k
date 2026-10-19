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

Run via srun, PYTHONPATH=$(pwd), JAX_COMPILATION_CACHE_DIR set. Env knobs:
  PA_CONFIG(scan/configs/lcdm.py) PA_POIS(csv; config default)
  PA_NPTS(config) PA_NSIG(config) PA_MAXIT(40) PA_GTOL(3e-2)
  PA_LMAX(2508) PA_RTOL(1e-5) PA_TAG('') PA_RESUME(1)
  PA_LM_K(4) PA_STEP_MAX(1.0) PA_XBOX(5) PA_VAL_CHUNK(256) PA_AD_BCHUNK(64)
  PA_WARM_LR(0) PA_MLE_PREPASS(0) PA_MULTISTART(0) PA_MS_K(6) PA_POI_SLICE(0) PA_TRACE_EVALS(0)
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

_HERE = os.path.dirname(os.path.abspath(__file__))
CL1, CL2, CL95 = 0.682689492137, 0.954499736104, 0.95


# ======================================================================
# profile-interval helpers
# ======================================================================
def _smooth_profile(x, y):
    """Dense Delta-chi2 curve (xs, ys) through a profile's points. PA_INTERVAL=fit: least-squares
    polynomial of degree FIT_DEG through the points within FIT_DCHI2 of the minimum (averages per-row
    scatter; a quartic follows the skew of tau / Neff); pchip: shape-preserving interpolation through
    every point. None with fewer than 4 finite points."""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    if m.sum() < 4:
        return None
    x, y = x[m], y[m]; o = np.argsort(x); x, y = x[o], y[o]
    if INTERVAL == "fit":
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
    becomes a one-sided limit by itself when the data sit near the bound. Returns (lo, best, hi): an
    end equal to the bound is a one-sided limit, NaN means not reached within the grid."""
    sp = _smooth_profile(x, y)
    if sp is None:
        return np.nan, np.nan, np.nan
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
    inside = ys <= thr
    def end(step):
        j = i0
        while 0 <= j + step < len(xs) and inside[j + step]:
            j += step
        if j + step < 0 or j + step >= len(xs):          # still inside at the curve's end
            edge = xs[j]
            onb = (np.isfinite(blo) and abs(edge - blo) < 1e-12 * max(1, abs(blo))) or \
                  (np.isfinite(bhi) and abs(edge - bhi) < 1e-12 * max(1, abs(bhi)))
            return edge if onb else np.nan
        a, b = j, j + step                               # inside at a, outside at b
        fa, fb = ys[a] - thr[a], ys[b] - thr[b]
        return xs[a] + (xs[b] - xs[a]) * (-fa) / (fb - fa + 1e-300)
    return end(-1), best, end(+1)


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
MULTISTART = os.environ.get("PA_MULTISTART", "0") != "0"
MS_K = int(os.environ.get("PA_MS_K", 6))
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
    """gradient with the components that push into an active bound zeroed (KKT residual)."""
    at = ((X <= XLO + 1e-9) & (G > 0)) | ((X >= XHI - 1e-9) & (G < 0))
    return np.where(at, 0.0, G)


def _lm_step(g, H, x, xlo, xhi, lam):
    """one projected LM step: solve (H_FF + lam diag(H_FF)) d = -g_F on the dims not held at a bound,
    cap it at STEP_MAX, project into the box. Returns (d, Gauss-Newton predicted chi2 drop)."""
    act = ((x <= xlo + 1e-9) & (g > 0)) | ((x >= xhi - 1e-9) & (g < 0))
    F = np.where(~act)[0]
    d = np.zeros_like(x)
    if len(F):
        Hf = H[np.ix_(F, F)]
        dg = np.maximum(np.diag(Hf), 1e-6 * max(np.max(np.diag(Hf)), 1e-12))
        for _ in range(30):                        # raise lam until the damped matrix is PD
            try:
                L = np.linalg.cholesky(Hf + lam * np.diag(dg))
                break
            except np.linalg.LinAlgError:
                lam = max(lam, 1e-6) * 10.0
        else:
            L = np.diag(np.sqrt(dg * lam))
        d[F] = -np.linalg.solve(L.T, np.linalg.solve(L, g[F]))
    dm = np.abs(d).max()
    if dm > STEP_MAX:
        d *= STEP_MAX / dm
    d = np.clip(x + d, xlo, xhi) - x
    return d, -(g @ d + 0.5 * d @ H @ d)


def lm_rows(BASE, FREE, XLO, XHI, X0, POI_IDX=None, PV=None, maxit=MAXIT, gtol=GTOL,
            log_prefix="", ckpt_path=None, resume_state=None):
    """Levenberg-Marquardt over a flat row set. BASE (N,D), FREE (N,P'), XLO/XHI/X0 (N,P'); POI_IDX/PV
    (N,) group the rows into profiles for the interval-stability stop (None: no early stop). Returns
    a dict of per-row arrays: X, F (values-path chi2), G, H (at X), gnorm (projected, exact AD),
    stalled, lam, it. The full state is checkpointed every iteration (ckpt_path); resume_state
    restarts from such a checkpoint."""
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
            lo, _, hi = poi_interval(PV[r], F[r], CL1, (BLO[p], BHI[p]))
            out[p] = (lo, hi)
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
            np.savez(tmp, BASE=BASE, FREE=FREE, X=X, F=F, G=G, H=H, lam=lam, nfail=nfail,
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
                for a, b in zip(prev[p], cur[p]):
                    if np.isfinite(a) and np.isfinite(b):
                        worst = max(worst, abs(b - a) / GRID_SIG[p]); nfin += 1
                    elif np.isfinite(a) != np.isfinite(b):
                        ready = False
            if ready and nfin and worst < SIGTOL:
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
# lockstep driver across ALL POIs
# ======================================================================
def _grid_for(poi):
    """NPTS points over the config 'grid' range, else GRID_CEN +- NSIG*GRID_SIG, clipped to the
    POI's physical bounds (a clipped end lands exactly on the bound)."""
    i = ORDER.index(poi)
    lo, hi = GRID_RANGE.get(poi, (GRID_CEN[i] - NSIG * GRID_SIG[i], GRID_CEN[i] + NSIG * GRID_SIG[i]))
    return np.linspace(max(lo, BLO[i]), min(hi, BHI[i]), NPTS)


def _bound_of(i):
    return (None if not np.isfinite(BLO[i]) else BLO[i], None if not np.isfinite(BHI[i]) else BHI[i])


def profile_lockstep(pois, outdir):
    """Build ONE row set (all pois x NPTS grid points), warm-start, run LM in lockstep, and write
    one npz/png per POI."""
    global GRID_CEN, GRID_SIG
    theta_warm = CENTER.copy(); Hwarm = None
    if MLE_PREPASS:
        theta_warm, chi2_mle, Hwarm = joint_fit(CENTER, MLE_MAXITER)
        GRID_CEN = theta_warm.copy(); GRID_SIG = _fisher_grid_sigma(Hwarm)
        print("[mle] recentered grids on the joint fit (chi2_min=%.3f):" % chi2_mle, flush=True)
        for i, nm in enumerate(ORDER):
            print(f"[mle]   {nm:>10}: cen {CENTER[i]:.5g} -> {GRID_CEN[i]:.5g}; grid-sig "
                  f"{GRID_SIG[i]:.3g}", flush=True)
    # ---- assemble the flat row set ----
    rows_poi, rows_pv, row_of_poi = [], [], {}
    for poi in pois:
        idx = ORDER.index(poi); grid = _grid_for(poi)
        row_of_poi[poi] = (len(rows_pv), len(rows_pv) + NPTS, grid)
        rows_poi += [idx] * NPTS; rows_pv += list(grid)
    POI_IDX = np.array(rows_poi, int); PV = np.array(rows_pv, float); N = len(PV)
    FREE = np.array([nuis_idx_of(i) for i in POI_IDX], int)
    BASE = np.tile(CENTER, (N, 1)); BASE[np.arange(N), POI_IDX] = PV
    XLO, XHI = map(np.array, zip(*[scaled_box(f) for f in FREE]))
    # ---- warm starts: the free dims at the warm point (the config centre, or the joint fit) ----
    xw = (theta_warm - CENTER) / SIGMA
    X0 = np.clip(xw[FREE], XLO, XHI) if WARM else np.clip(np.zeros((N, P)), XLO, XHI)
    print(f"[lockstep] N={N} rows ({len(pois)} POIs x {NPTS}) P={P} D={D} shard={DO_SHARD} "
          f"warm={'joint fit' if MLE_PREPASS else 'config centre' if WARM else 'cold'}", flush=True)
    ckpt = os.path.join(outdir, f"profile_prod_ad_STATE{TAG}{'_r%d' % RANK if POI_SLICE else ''}.npz")
    resume_state = None
    if RESUME and os.path.exists(ckpt):
        try:
            st = np.load(ckpt)
            if (st["BASE"].shape == BASE.shape and np.allclose(st["BASE"], BASE)
                    and np.array_equal(st["FREE"], FREE) and float(st["gtol"]) == GTOL):
                resume_state = {k: st[k] for k in st.files}
                print(f"[resume] checkpoint @ it{int(st['it'])} matches; continuing ({ckpt})", flush=True)
            else:
                print("[resume] checkpoint grid/gtol mismatch -> fresh start", flush=True)
        except Exception as ex:
            print(f"[resume] could not load checkpoint ({ex}) -> fresh start", flush=True)
    # ---- linear-response warm start: each row starts at the Gaussian conditional optimum of the warm
    # point's Gauss-Newton matrix (rows far along a degeneracy need not walk there at STEP_MAX per
    # iteration); one batched value call scores both starts and each row keeps the lower chi2. ----
    if WARM and WARM_LR and resume_state is None:
        if Hwarm is None:
            _, _, Hwarm = joint_fit(theta_warm, 0)
        Xlr = X0.copy()
        for b in range(N):
            i, f = POI_IDX[b], FREE[b]
            dpoi = (PV[b] - theta_warm[i]) / SIGMA[i]
            Xlr[b] = np.clip(xw[f] - np.linalg.lstsq(Hwarm[np.ix_(f, f)], Hwarm[f, i] * dpoi,
                                                     rcond=None)[0], XLO[b], XHI[b])
        fk = values_rows(np.tile(BASE, (2, 1)), np.tile(FREE, (2, 1)), np.concatenate([X0, Xlr]))
        use = fk[N:] < fk[:N]
        X0 = np.where(use[:, None], Xlr, X0)
        print(f"[warm-lr] linear-response start kept for {int(use.sum())}/{N} rows; chi2 median "
              f"{np.median(fk[:N]):.1f} -> {np.median(np.minimum(fk[:N], fk[N:])):.1f}", flush=True)
    # ---- lockstep LM ----
    t0 = time.perf_counter()
    S = lm_rows(BASE, FREE, XLO, XHI, X0, POI_IDX=POI_IDX, PV=PV, log_prefix="[lock]",
                ckpt_path=ckpt, resume_state=resume_state)
    conv = S["gnorm"] < GTOL
    at_box = ((S["X"] <= -XBOX + 1e-9) | (S["X"] >= XBOX - 1e-9)).any(1)   # on the +-XBOX safety box:
    conv &= ~at_box                                                 # a zero projected gradient there is not a fit
    ev = np.linalg.eigvalsh(S["H"])                                  # (N, P) GN eigenvalues
    print(f"[lockstep] LM done {time.perf_counter() - t0:.0f}s RSS={rss_gb():.1f}GB; converged "
          f"{int(conv.sum())}/{N} (||g||_AD<{GTOL:.1e}); stalled {int(S['stalled'].sum())}; "
          f"on the XBOX box {int(at_box.sum())}; "
          f"max||g||_AD={S['gnorm'].max():.2e}", flush=True)
    # ---- write one npz/png per POI ----
    for poi in pois:
        s, e, grid = row_of_poi[poi]; sl = slice(s, e); i = ORDER.index(poi)
        chi2 = S["F"][sl]; bnd = _bound_of(i)
        lo1, mid, hi1 = poi_interval(grid, chi2, CL1, bnd)
        lo2, _, hi2 = poi_interval(grid, chi2, CL2, bnd)
        lo95, _, hi95 = poi_interval(grid, chi2, CL95, bnd)
        sw, mw, dw = wald_fit(grid, chi2)
        npz = os.path.join(outdir, f"profile_prod_ad_{poi}{TAG}.npz")
        np.savez(npz, poi=poi, poi_grid=grid, chi2=chi2, xstar=S["X"][sl], gnorm=S["gnorm"][sl],
                 converged=conv[sl], stalled=S["stalled"][sl], at_box=at_box[sl], lm_lambda=S["lam"][sl],
                 gn_eig=ev[sl], hess_pd=ev[sl].min(1) > 0,
                 nuis=np.array([ORDER[k] for k in nuis_idx_of(i)]), done=True,
                 sigma1=np.array([lo1, mid, hi1]), sigma2=np.array([lo2, hi2]),
                 cl95=np.array([lo95, hi95]), bound=np.array([np.nan if b is None else b for b in bnd]),
                 sigma_wald=sw, wald_vertex=mw, wald_misfit=dw, interval_mode=INTERVAL,
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
        print(f"[{poi}] minchi2={chi2[j]:.2f} at {poi}={grid[j]:.5f}; 68% [{lo1:.5f},{hi1:.5f}] ({kind}, "
              f"best {mid:.5f}); 95% [{lo95:.5f},{hi95:.5f}]; wald sigma {sw:.5f} (misfit {dw:.2f}); "
              f"converged {int(conv[sl].sum())}/{nval}, stalled {int(S['stalled'][sl].sum())}; "
              f"max||g||_AD={np.nanmax(S['gnorm'][sl]):.2e} -> {npz}{warn}", flush=True)
        _plot(poi, grid, chi2, npz)
    try:
        if os.path.exists(ckpt):
            os.remove(ckpt)                           # run finished cleanly
    except Exception:
        pass


def multistart(poi, outdir, K=MS_K):
    """global-minimum check: K random starts (+ the centre) at two values of the POI."""
    i = ORDER.index(poi); c, s = CENTER[i], SIGMA[i]
    test_vals = np.clip(np.array([c, c + 2 * s]), BLO[i], BHI[i])
    rng = np.random.default_rng(1234)
    print(f"[{poi}] MULTISTART K={K} at {poi}={test_vals}", flush=True)
    saved = {}
    f = np.array(nuis_idx_of(i)); xlo, xhi = scaled_box(f)
    for vi, pv in enumerate(test_vals):
        BASE = np.tile(CENTER, (K, 1)); BASE[:, i] = pv; FREE = np.tile(f, (K, 1))
        x0 = np.clip(rng.uniform(-2.5, 2.5, (K, P)), xlo, xhi); x0[0] = np.clip(0.0, xlo, xhi)
        S = lm_rows(BASE, FREE, np.tile(xlo, (K, 1)), np.tile(xhi, (K, 1)), x0, log_prefix=f"[{poi}@{pv:.4f}]")
        print(f"[{poi}@{pv:.5f}] converged chi2: min={S['F'].min():.3f} max={S['F'].max():.3f} "
              f"spread={S['F'].max() - S['F'].min():.3f} max||g||_AD={S['gnorm'].max():.1e} "
              f"(global min if spread<<1)", flush=True)
        saved[f"chi2_{vi}"] = S["F"]; saved[f"gnorm_{vi}"] = S["gnorm"]; saved[f"xstar_{vi}"] = S["X"]
    np.savez(os.path.join(outdir, f"multistart_{poi}{TAG}.npz"), poi=poi, test_vals=test_vals, K=K, **saved)


def _plot(poi, grid, chi2, npz):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        i = ORDER.index(poi); bnd = _bound_of(i)
        d = chi2 - np.nanmin(chi2)
        fig, ax = plt.subplots(figsize=(6, 4.3))
        ax.plot(grid, d, "o-")
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
          f"lowEE={USE_LOWEE} lowTT={USE_LOWTT} warm={WARM} multistart={MULTISTART} "
          f"poi_slice={POI_SLICE} nproc={NPROC} bounds="
          f"{ {k: (BLO[i], BHI[i]) for i, k in enumerate(ORDER) if np.isfinite(BLO[i]) or np.isfinite(BHI[i])} }",
          flush=True)
    if MULTISTART:
        for poi in POIS[RANK::NPROC]:
            multistart(poi, outdir)
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

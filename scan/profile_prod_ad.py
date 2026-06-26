"""Config-driven, lockstep frequentist profile-likelihood tool (TOOL_PLAN.md).

One sbatch in; per profiled parameter (POI), an npz + png out with the grid,
profiled chi2, best fit, 1sigma/2sigma PCHIP intervals, a per-point AD ||g||
convergence certificate, and the multistart spread.

  1. Config-driven (PA_CONFIG=path/to/config.py). A config declares the parameter
     order, fiducials (cen), scale widths (sig), which parameters are POIs, the
     fixed dict, optional user_species, likelihood toggles, and NPTS/NSIG. Adding
     a parameter (e.g. Neff) is a config edit, not a code edit. Ships
     configs/lcdm.py (6-param LCDM) and configs/lcdm_neff.py.

  2. One lockstep batch across all POIs x grid points x multistart replicas. Each
     "row" is one optimisation point; the BFGS state (x, Hinv, f, g) is per-row
     arrays, so concatenating POIs just lengthens the batch. Per-row direction j
     is the j-th free dim of that row's POI (P = D-1). One POI per SLURM rank is
     available via PA_RANK_SLICE for multi-node runs; a single task handles the
     full set.

  3. Gradient method (config field grad_method). "fdbatch" (the LCDM default)
     assembles central finite-difference gradients on the batch axis: the 2*P*N
     perturbed cosmologies go through call_batched in chunks of PA_FD_CHUNK, and
     iteration 0 calibrates the FD step against the exact batched-AD gradient
     until they agree (max-rel <= PA_FD_CALTOL). "ad" uses the exact gradient
     every iteration. Either way, all values for the line search and the recorded
     profile come from the call_batched path, and the final stationarity
     certificate (||g||_inf < GTOL, the nuisance Hessian, the PD check) is always
     from the exact AD gradient. grad_method ad/batched/loop/vmap stay available
     (PA_GRADMETHOD is a debug-only override).

Run via srun, PYTHONPATH=$(pwd), JAX_COMPILATION_CACHE_DIR set. Env knobs:
  PA_CONFIG(scan/configs/lcdm.py) PA_POIS(csv; config default)
  PA_NPTS(config) PA_NSIG(config) PA_MAXIT(40) PA_GTOL(3e-2)
  PA_LMAX(2508) PA_RTOL(1e-5) PA_TAG('') PA_RESUME(1)
  grad_method in config (fdbatch|ad|batched|loop|vmap); PA_GRADMETHOD = debug override
  PA_FD_STEP(1e-2) PA_FD_CHUNK(128)
  PA_FD_CALTOL(1e-2) PA_FD_CALMIN(32) PA_CAL_RETRIES(3)
  PA_HESS(1) PA_SHARD(auto|0|1) PA_WARM(1) PA_WARM_DIR(scan/results)
  PA_MULTISTART(0) PA_MS_K(6) PA_USE_LOWTT(cfg) PA_USE_LOWEE(cfg)
  PA_RANK_SLICE(0)  -- 1 => slice the ROW list across SLURM ranks (multi-node);
                       0 (default) => every rank runs the full lockstep set.
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
from scan.lowl_like import LowLEE, LowLTT
from scan.batched_grad import staged_chi2_and_grad, _to_float as _bg_to_float

_HERE = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# profile-interval helpers
# ======================================================================
def interval(x, y, level):
    """Delta-chi2 = level crossings of a profile, via shape-preserving (PCHIP)
    interpolation and dense-grid root finding. The chi2 is deterministic and
    smooth, so the crossings are sub-grid accurate. Returns (lo, min_x, hi)."""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    if m.sum() < 4:
        return np.nan, np.nan, np.nan
    x, y = x[m], y[m]; o = np.argsort(x); x, y = x[o], y[o]
    try:
        from scipy.interpolate import PchipInterpolator
        p = PchipInterpolator(x, y - y.min())
        xs = np.linspace(x[0], x[-1], 40001); ys = p(xs)
    except Exception:                                   # fallback: dense linear
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
    """Symmetric Gaussian sigma from a parabola fit to the points within
    Delta-chi2 <= 4 of the minimum (sigma = 1/sqrt(2a) for chi2 ~ a x^2)."""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan
    d = y - y.min(); sel = d <= 4.0
    if sel.sum() < 3:
        sel = np.argsort(d)[:max(3, len(x) // 2)]
    a = np.polyfit(x[sel], y[sel], 2)[0]                 # chi2 ~ a x^2 + ...
    return np.nan if a <= 0 else 1.0 / np.sqrt(2.0 * a)


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
P = D - 1                                            # free nuisance dims per row
CENTER = np.array([CFG["cen"][k] for k in ORDER])
SIGMA = np.array([CFG["sig"][k] for k in ORDER])
# GRID placement (center + half-span-per-sigma) is DECOUPLED from the optimizer's
# scaled-coordinate system (CENTER/SIGMA stay the config guess, so the well-conditioned
# scaling never changes mid-run). The MLE pre-pass (PA_MLE_PREPASS) overwrites these with
# the joint MLE + its Fisher errors so every POI grid is centered on the actual optimum
# with a span matched to the real uncertainty -- the model-agnostic fix for new physics.
GRID_CEN = CENTER.copy()
GRID_SIG = SIGMA.copy()
FIXED = dict(CFG["fixed"])
USER_SPECIES = CFG.get("user_species", None)

# ---------------- env config (overrides config-file defaults) ----------------
LMAX = int(os.environ.get("PA_LMAX", 2508))
NPTS = int(os.environ.get("PA_NPTS", CFG.get("npts", 25)))
NSIG = float(os.environ.get("PA_NSIG", CFG.get("nsig", 3.0)))
MAXIT = int(os.environ.get("PA_MAXIT", 40))
GTOL = float(os.environ.get("PA_GTOL", 3e-2))        # ||grad||_inf gate (scaled coords)
# chi2-plateau early-stop (legacy; off by default). The AD ||g|| grinds slowly
# toward a ~0.1 rtol-independent roughness floor it never reaches, so GTOL is not
# met, but chi2/intervals settle by ~it5. Stop a POI's lockstep once the
# slowest-converging row's best-chi2 has improved < FTOL over FTOL_PATIENCE iters.
# The post-loop AD ||g|| certificate still reports the honest floor. FTOL<=0
# disables. Superseded by the sigma1-stability trigger below.
FTOL = float(os.environ.get("PA_FTOL", "0"))         # max per-row chi2 improvement over window
FTOL_PATIENCE = int(os.environ.get("PA_FTOL_PATIENCE", "3"))
# sigma1-stability early-stop (the default trigger; model-agnostic, no per-model
# tuning). Stop once every POI's 1-sigma interval half-width has changed by
# < PA_SIGTOL * sigma(POI) over PA_SIGTOL_PATIENCE iters -- i.e. the deliverable
# (the interval) has converged to PA_SIGTOL sigmas. The tolerance is in
# sigma-units, so it transfers to any new cosmology on its first run (unlike the
# chi2 FTOL, whose scale is the solver roughness floor). It is robust to the ~0.1
# chi2 roughness (sigma1 is the PCHIP-smoothed crossing) and to edge-row lag (tail
# rows sit at dchi2~9, away from the dchi2=1 crossings). The post-loop AD ||g||
# certificate still runs. SIGTOL<=0 disables. The default 1e-2 was validated at
# full l (CHANGELOG 2026-06-23): on the worst-conditioned POI it fires at it5 and
# matches the converged interval to ~1e-4 sigma, ~2.6x faster wall-clock.
SIGTOL = float(os.environ.get("PA_SIGTOL", "1e-2"))  # interval half-width stability, in sigma units
SIGTOL_PATIENCE = int(os.environ.get("PA_SIGTOL_PATIENCE", "3"))
BF_TRACE = os.environ.get("PA_BF_TRACE", "")         # debug-only: dump per-iter best_f history
#   to this npz path for post-hoc early-stop validation. OFF by default (no prod impact).
RTOL = float(os.environ.get("PA_RTOL", 1e-5))
TAG = os.environ.get("PA_TAG", "")
RESUME = os.environ.get("PA_RESUME", "1") != "0"
# grad_method is a run-config field (a per-model analysis choice: FD iterations are
# fine for LCDM but risky for non-convex new-physics params), not an env var.
# PA_GRADMETHOD is a debug-only override that warns loudly when set
# (see scan/configs/*.py grad_method + TOOL_PLAN section 2).
_CFG_GRADMETHOD = str(CFG.get("grad_method", "fdbatch")).lower()
if "PA_GRADMETHOD" in os.environ:
    GRADMETHOD = os.environ["PA_GRADMETHOD"].lower()
    print(f"[profile] WARNING: PA_GRADMETHOD={GRADMETHOD} is a DEBUG-ONLY env override "
          f"and SHADOWS the config grad_method='{_CFG_GRADMETHOD}'. Production runs "
          f"should set grad_method in the config file, not the environment.", flush=True)
else:
    GRADMETHOD = _CFG_GRADMETHOD
GRAD_KCHUNK = int(os.environ.get("PA_GRAD_KCHUNK", "100"))   # k_chunk for batched AD grad
FD_STEP = float(os.environ.get("PA_FD_STEP", "1e-2"))        # central FD step (scaled coords)
FD_CHUNK = int(os.environ.get("PA_FD_CHUNK", "128"))        # cosmologies per primal call_batched (B_local=FD_CHUNK/n_dev)
FD_CALTOL = float(os.environ.get("PA_FD_CALTOL", "1e-2"))   # it0 fd-vs-ad max-rel target
FD_CALMIN = int(os.environ.get("PA_FD_CALMIN", "32"))       # min rows in the it0 calibration sample
CAL_RETRIES = int(os.environ.get("PA_CAL_RETRIES", "3"))
AD_BCHUNK = int(os.environ.get("PA_AD_BCHUNK", "64"))       # B per staged-AD certificate call
DO_HESS = os.environ.get("PA_HESS", "1") != "0"
_shard_env = os.environ.get("PA_SHARD", "auto").lower()
MULTISTART = os.environ.get("PA_MULTISTART", "0") != "0"
MS_K = int(os.environ.get("PA_MS_K", 6))
USE_LOWTT = (os.environ["PA_USE_LOWTT"] != "0") if "PA_USE_LOWTT" in os.environ \
    else bool(CFG.get("use_lowtt", True))
USE_LOWEE = (os.environ["PA_USE_LOWEE"] != "0") if "PA_USE_LOWEE" in os.environ \
    else bool(CFG.get("use_lowee", True))
# Include the plik-lite high-ell TTTEEE likelihood. DEFAULT ON. Set PA_USE_PLIK=0 ONLY for
# fast low-ell-only DEBUG runs (plik-lite needs theory Cls to l~2508, so it is meaningless
# at truncated LMAX) -- e.g. validating l-independent machinery like the chi2 early-stop.
USE_PLIK = (os.environ["PA_USE_PLIK"] != "0") if "PA_USE_PLIK" in os.environ \
    else bool(CFG.get("use_plik", True))
# FULL Planck plik (clipy) vs plik-LITE for the high-ell TTTEEE likelihood. Config
# field high_ell: "plik_lite" (default) | "plik_full". Full plik carries 47
# foreground/calibration nuisances handled ENTIRELY inside the likelihood -- profiled
# per cosmology at FIXED ABCMB theory (NO ABCMB re-run for nuisances), the whole point.
# See scan/plik_full.py. PA_HIGH_ELL is a debug override.
HIGH_ELL = os.environ.get("PA_HIGH_ELL", str(CFG.get("high_ell", "plik_lite"))).lower()
USE_PLIK_FULL = USE_PLIK and (HIGH_ELL == "plik_full")
WARM = os.environ.get("PA_WARM", "1") != "0"
# Use the config CENTER as the warm start (instead of reading the prior LCDM
# profile npz). For a RECENTERED re-run the config `cen` IS the joint MLE, so this
# gives every row the ideal continuation-like start (all non-POI dims at the MLE,
# POI fixed at its grid value) AND computes the warm-Hessian preconditioner at the
# MLE. The default (0) keeps the legacy `_global_best_fit_physical` behaviour.
WARM_FROM_CEN = os.environ.get("PA_WARM_FROM_CEN", "0") != "0"
WARM_DIR = os.environ.get("PA_WARM_DIR", os.path.join(_HERE, "results"))
# MLE PRE-PASS (model-agnostic auto-centering). Before any grid is built, solve the
# joint D-dim MLE (all params free, nuisances inner-profiled), recenter every POI grid
# on it, set the grid half-span from the MLE Fisher errors, warm-start there, and reuse
# the MLE Hessian as the BFGS preconditioner. Turns "hand-recenter each new model" into
# a one-line flag. Default OFF (legacy static-grid behaviour preserved).
MLE_PREPASS = os.environ.get("PA_MLE_PREPASS", "0") != "0"
MLE_MAXITER = int(os.environ.get("PA_MLE_MAXITER", "60"))     # L-BFGS-B iters for the joint solve
MLE_SIG_CLIP = float(os.environ.get("PA_MLE_SIG_CLIP", "3.0"))  # grid sigma clamped to [1/clip, clip] x prior
RANK_SLICE = os.environ.get("PA_RANK_SLICE", "0") != "0"
POI_SLICE = os.environ.get("PA_POI_SLICE", "0") != "0"   # multi-node scale-out: each
# rank owns a DISJOINT subset of POIs (POIS[RANK::NPROC]) and runs its own lockstep,
# writing only its own per-POI npz -> no shared-file clobber, no MPI gather. This is
# the node-scaling lever for few-hours wall-clock (the embarrassingly-parallel POI axis).
XBOX = 5.0                                            # nuisance box (sigma units)
C1, MAXLS = 1e-4, 12                                  # Armijo c1, max backtracks
FDH = 0.05                                            # FD step (sigma) for the AD Fisher Hessian
# Inverse-Fisher BFGS preconditioner: init Hinv from the (per-POI) nuisance
# Hessian at the warm-start global best fit. Turns the early steepest-descent
# steps into near-Newton steps -> converges in a few iters on the ill-conditioned
# CMB likelihood (kappa~1e3) instead of stalling. PA_PRECOND=0 reverts to eye.
PRECOND = os.environ.get("PA_PRECOND", "1") != "0"
HESS_CACHE = os.environ.get("PA_HESS_CACHE", "1") != "0"   # cache the warm Hessian to disk
# The warm-start D x D Hessian is a FLAT ~18-min cost (2D single-cosmo AD-grad evals) that
# depends ONLY on (theta_warm, l_max, likelihood) -- not on NPTS/POI/grid. Caching it to
# disk lets a precomputed Hessian be LOADED instantly (and shared across the POI_SLICE
# ranks, which otherwise each recompute it redundantly), removing ~18 min from the run's
# critical path. Load-if-present-else-compute, so an absent/mismatched cache just falls
# back to the original compute -- cannot break the run.

POIS = os.environ.get("PA_POIS", ",".join(CFG.get("pois", ORDER))).split(",")
POIS = [p.strip() for p in POIS if p.strip() in ORDER]
RANK = int(os.environ.get("SLURM_PROCID", 0))
NPROC = int(os.environ.get("SLURM_NPROCS", 1))

pl = PlikLite()          # always built: its abcmb_cl_to_Dl feeds the low-ell likelihoods
lowee = LowLEE() if USE_LOWEE else None
lowtt = LowLTT() if USE_LOWTT else None
# full-plik backend (only when high_ell=plik_full). Its inner-profile preconditioner
# (plf.Hprec) is set once by _setup_plik_full_precond() before any profile call.
plf = None
_plf_prof_B = None       # jitted batched inner profile (values path)
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
# parameter assembly (generic in ORDER; POI is the row's profiled dim)
# ======================================================================
def nuis_idx_of(poi_idx):
    """global param indices (length P) that are FREE for a row with this POI."""
    return [i for i in range(D) if i != poi_idx]


def assemble_phys(poi_idx, xP, poi_val):
    """scaled nuisance xP (P,) + POI value -> physical D-vector (numpy)."""
    theta = CENTER.copy(); theta[poi_idx] = poi_val
    for k, i in enumerate(nuis_idx_of(poi_idx)):
        theta[i] = CENTER[i] + SIGMA[i] * xP[k]
    return theta


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


# ======================================================================
# FAST function values via call_batched (no AD) -- the consistency-rule path
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


def fast_values_rows(POI_IDX, X, PV):
    """POI_IDX:(N,) int, X:(N,P), PV:(N,) -> (N,) total chi^2 (profiled + low-ell).
    Evaluated through call_batched in FD_CHUNK-sized chunks (memory-safe: per-device
    working set is set by FD_CHUNK/n_dev, NOT by N -- an unchunked call over all N
    rows OOMs once N is large, the same root cause as job 54369057/54362790)."""
    batch = [build_dict(assemble_phys(int(POI_IDX[b]), X[b], PV[b])) for b in range(len(PV))]
    return _chunked_call_batched(batch, FD_CHUNK)


def _chunked_call_batched(batch, chunk):
    """Evaluate a list of param dicts through call_batched in fixed-size chunks
    (last chunk PADDED to `chunk` to keep batch avals stable -> no recompile per
    new B). Returns concatenated chi^2 of length len(batch)."""
    N = len(batch)
    # Cap the chunk at the actual batch size: never pad a SMALL batch up to a large
    # `chunk`. This only SHRINKS the chunk (memory stays bounded by the caller's
    # `chunk`), and it kills the POI_SLICE line-search waste -- an 11-row fast_values
    # call was paying for 128 cosmologies (FD_CHUNK), ~12x/iter, which made small
    # per-rank batches line-search-bound (~30 min/iter at l=2508). N is constant within
    # a run (the full row set every call), so the B=N aval is stable -> one compile.
    chunk = min(chunk, N)
    out_chi2 = np.empty(N)
    for s in range(0, N, chunk):
        sub = batch[s:s + chunk]
        nsub = len(sub)
        if nsub < chunk:
            sub = sub + [sub[-1]] * (chunk - nsub)         # pad (replicate last)
        out = model.call_batched(sub, shard=DO_SHARD)
        out_chi2[s:s + nsub] = _chi2_from_out(out)[:nsub]
    return out_chi2


# ======================================================================
# fdbatch gradient (DEFAULT): central FD on the batch axis
# ======================================================================
def fdbatch_grad(POI_IDX, X, PV, step=None):
    """Central-FD gradient dchi2/dxP (scaled coords) at every row, assembled on
    the batch axis. For N rows x P dims, build 2*P*N perturbed cosmologies
    (x +/- step e_j), evaluate via _chunked_call_batched, finite-difference.
    Returns G (N,P). VALUES are NOT taken from here (consistency rule)."""
    if step is None:
        step = FD_STEP
    N = len(PV)
    # assemble the 2*P*N perturbed scaled-nuisance vectors
    batch = []
    for b in range(N):
        pidx = int(POI_IDX[b])
        for j in range(P):
            xp = np.array(X[b]); xp[j] += step
            xm = np.array(X[b]); xm[j] -= step
            batch.append(build_dict(assemble_phys(pidx, xp, PV[b])))
            batch.append(build_dict(assemble_phys(pidx, xm, PV[b])))
    chi2 = _chunked_call_batched(batch, FD_CHUNK)              # (2*P*N,)
    chi2 = chi2.reshape(N, P, 2)
    G = (chi2[:, :, 0] - chi2[:, :, 1]) / (2.0 * step)
    return G


# ======================================================================
# batched AD gradient (the it0 calibration + final certificate engine)
# ======================================================================
def _chi2_of_cls(ClTT, ClTE, ClEE):
    """Pure-jnp, batched envelope-profiled high-ell + low-ell chi2 from Cls.
    For full plik the 47 nuisances are inner-profiled (stop_gradient'd optimum), then
    the chi2 is differentiated in the Cls at the FIXED optimum -- exact by the envelope
    theorem, mirroring the single-A_planck profiling of plik-lite."""
    lvec = model.SS.ells
    Dtt = pl.abcmb_cl_to_Dl(ClTT, lvec); Dee = pl.abcmb_cl_to_Dl(ClEE, lvec)
    if USE_PLIK_FULL:
        cls2d = plf.abcmb_cls_to_clik(ClTT, ClTE, ClEE, lvec)                 # (B,6,Lcol)
        # envelope theorem: profile the nuisances at the PRIMAL cls (stop_gradient the
        # input so the outer jvp never traces tangents through the inner Hessian/solve),
        # then differentiate penalized_chi2 in cls at the FIXED optimum.
        nu_star = jax.lax.stop_gradient(
            plf.profile_batched(jax.lax.stop_gradient(cls2d))[1])             # (B,21)
        c2 = jax.vmap(plf.penalized_chi2)(cls2d, nu_star)
    elif USE_PLIK:
        Dte = pl.abcmb_cl_to_Dl(ClTE, lvec)
        m0 = pl.bin_model(Dtt, Dte, Dee)
        A_star = jax.lax.stop_gradient(pl.profile_A(m0, with_prior=True)[1])
        diff = pl.X_data - m0 / (A_star[..., None] ** 2)
        c2 = jnp.einsum("...i,ij,...j->...", diff, pl.invcov, diff) \
            + ((A_star - 1.0) / 0.0025) ** 2
    else:                                  # low-ell-only DEBUG path (no high-ell)
        c2 = jnp.zeros(jnp.shape(Dee)[:-1])
    if lowee is not None:
        c2 = c2 + lowee.chi2(Dee)
    if lowtt is not None:
        c2 = c2 + lowtt.chi2(Dtt)
    return c2


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


def _ad_grad_block(POI_IDX, X, PV):
    """One staged batched-AD gradient over a block of rows (no internal chunking).
    Returns (chi2 (B,), G (B,P)). The P tangents are per-row: direction j of row
    b is SIG[i] * e_i where i = the j-th free dim of row b's POI -- so the tangent
    dict for direction j varies row-by-row, which the staged push handles since
    params_dots[j] is the B-stacked tangent."""
    import equinox as eqx
    B = len(PV)
    thetas = [jnp.asarray(assemble_phys(int(POI_IDX[b]), X[b], PV[b])) for b in range(B)]
    full_ps = [_phys_to_derived(t) for t in thetas]
    # per-row, per-direction derived-param tangent
    per = []   # [B][P] filtered tangent dicts
    for b in range(B):
        nuis = nuis_idx_of(int(POI_IDX[b]))
        dots = []
        for j in range(P):
            i = nuis[j]
            tan = jnp.zeros(D).at[i].set(SIG[i])
            _, fd = jax.jvp(_phys_to_derived, (thetas[b],), (tan,))
            dots.append(eqx.filter(fd, eqx.is_inexact_array))
        per.append(dots)
    params_dots = [jax.tree.map(lambda *xs: jnp.stack(xs),
                                *[per[b][j] for b in range(B)]) for j in range(P)]
    chi2, grad = staged_chi2_and_grad(model, full_ps, params_dots, _chi2_of_cls,
                                      k_chunk_size=GRAD_KCHUNK, shard=DO_SHARD)
    return np.asarray(chi2, float), np.asarray(grad, float)


def ad_grad_rows(POI_IDX, X, PV, bchunk=None):
    """Exact batched-AD gradient over ALL rows, in B<=bchunk blocks (B_local=16
    is the measured sweet spot, so bchunk=64 on a 4-GPU node). Returns
    (chi2 (N,), G (N,P))."""
    if bchunk is None:
        bchunk = AD_BCHUNK
    N = len(PV)
    chi2 = np.empty(N); G = np.empty((N, P))
    for s in range(0, N, bchunk):
        e = min(s + bchunk, N)
        c, g = _ad_grad_block(POI_IDX[s:e], X[s:e], PV[s:e])
        chi2[s:e] = c; G[s:e] = g
    return chi2, G


# ---- single-cosmology AD value-and-grad (loop/vmap fallbacks + Hessian) ----
def _single_chi2_from_out(out):
    """single-cosmology total chi2 (high-ell profiled + low-ell), AD-able in `out`.
    Shared by chi2_scaled_single and _chi2_full_scaled. Full plik inner-profiles the
    47 nuisances (envelope-theorem: stop_gradient the optimum)."""
    Dtt = pl.abcmb_cl_to_Dl(out.ClTT, out.l); Dee = pl.abcmb_cl_to_Dl(out.ClEE, out.l)
    if USE_PLIK_FULL:
        cls2d = plf.abcmb_cls_to_clik(out.ClTT, out.ClTE, out.ClEE, out.l)   # (6,Lcol)
        nu_star = jax.lax.stop_gradient(plf.profile(jax.lax.stop_gradient(cls2d))[1])
        c2 = plf.penalized_chi2(cls2d, nu_star)
    elif USE_PLIK:
        Dte = pl.abcmb_cl_to_Dl(out.ClTE, out.l)
        m0 = pl.bin_model(Dtt, Dte, Dee)
        A_star = jax.lax.stop_gradient(pl.profile_A(m0, with_prior=True)[1])
        diff = pl.X_data - m0 / (A_star ** 2)
        c2 = diff @ pl.invcov @ diff + ((A_star - 1.0) / 0.0025) ** 2
    else:
        c2 = jnp.zeros(())
    if lowee is not None:
        c2 = c2 + lowee.chi2(Dee)
    if lowtt is not None:
        c2 = c2 + lowtt.chi2(Dtt)
    return c2


def chi2_scaled_single(xP, poi_val, poi_idx):
    """scalar chi^2 at scaled nuisances xP for ONE row, AD-able in xP."""
    nuis = nuis_idx_of(poi_idx)
    theta = CEN.at[poi_idx].set(poi_val)
    for k, i in enumerate(nuis):
        theta = theta.at[i].set(CEN[i] + SIG[i] * xP[k])
    p = dict(FIXED)
    for i, name in enumerate(ORDER):
        if name == "ln10As":
            p["A_s"] = jnp.exp(theta[i]) / 1e10
        else:
            p[name] = theta[i]
    out = model.run_cosmology_abbr(model.add_derived_parameters(p))
    return _single_chi2_from_out(out)


def _vg_one(poi_idx):
    """value-and-jacfwd for ONE cosmology with this POI."""
    def f(xP, poi_val):
        return chi2_scaled_single(xP, poi_val, poi_idx)

    def vg(xP, poi_val):
        e = jnp.eye(P)
        fs, g = jax.vmap(lambda v: jax.jvp(lambda z: f(z, poi_val), (xP,), (v,)))(e)
        return fs[0], g
    return f, vg


# ======================================================================
# unified gradient dispatch for BFGS ITERATIONS
# ======================================================================
def iterate_grad(POI_IDX, X, PV, method, fd_step=None):
    """G (N,P) at every row by the chosen ITERATION method. VALUES come from
    fast_values_rows separately (consistency rule)."""
    if method == "fdbatch":
        return fdbatch_grad(POI_IDX, X, PV, step=fd_step)
    if method in ("ad", "batched"):
        return ad_grad_rows(POI_IDX, X, PV)[1]
    # per-row single-cosmology AD (loop/vmap): build a vg per distinct POI
    N = len(PV); G = np.empty((N, P))
    vgs = {pi: _vg_one(pi)[1] for pi in sorted(set(int(p) for p in POI_IDX))}
    if method == "vmap":
        # group rows by POI so each group shares a vg aval
        for pi, vg in vgs.items():
            sel = np.where(POI_IDX == pi)[0]
            if not len(sel):
                continue
            _, g = jax.vmap(vg)(jnp.asarray(X[sel]), jnp.asarray(PV[sel]))
            G[sel] = np.asarray(g)
        return G
    for b in range(N):
        vg = vgs[int(POI_IDX[b])]
        _, gb = vg(jnp.asarray(X[b]), jnp.asarray(float(PV[b])))
        G[b] = np.asarray(gb)
    return G


def _interval_halfwidth(x, chi2):
    """1-sigma (dchi2=1) interval half-width of a single POI's profile, via the SAME
    PCHIP `interval` the final result uses. NaN if no clean dchi2=1 crossing yet (early
    iters, min at an edge) -- the sigma1-stability trigger holds until it is finite."""
    lo, _, hi = interval(x, chi2, 1.0)
    return 0.5 * (hi - lo) if (np.isfinite(lo) and np.isfinite(hi)) else np.nan


# ======================================================================
# vectorised BFGS over the ROW set (lockstep), Armijo line search
# ======================================================================
def bfgs_rows(POI_IDX, PV, x0=None, Hinv0=None, maxit=MAXIT, gtol=GTOL,
              fd_step=None, log_prefix="", ckpt_path=None, resume_state=None):
    """BFGS profile over a flat row set. POI_IDX:(N,), PV:(N,). Returns
    (best_f (N,), best_x (N,P), gnorm (N,)). Stable batch shapes: inactive rows
    keep riding the batch (no shrink).

    Hinv0 (N,P,P) seeds the per-row inverse-Hessian (inverse-Fisher preconditioner);
    None => identity. If ckpt_path is given the full BFGS state is written there
    after every iteration; resume_state (a dict from a prior ckpt) restarts from it
    so a walltime timeout loses at most one iteration."""
    N = len(PV)
    if resume_state is not None:
        x = np.array(resume_state["x"], float)
        Hinv = np.array(resume_state["Hinv"], float)
        f = np.array(resume_state["f"], float)
        g = np.array(resume_state["g"], float)
        best_f = np.array(resume_state["best_f"], float)
        best_x = np.array(resume_state["best_x"], float)
        start_it = int(resume_state["it"])
        gnorm = np.abs(g).max(1)
        print(f"  {log_prefix} RESUME from it{start_it}: min={best_f.min():.2f} "
              f"||g||max={gnorm.max():.2e}", flush=True)
    else:
        x = np.zeros((N, P)) if x0 is None else np.array(x0, float)
        g = iterate_grad(POI_IDX, x, PV, GRADMETHOD, fd_step=fd_step)
        f = fast_values_rows(POI_IDX, x, PV)              # value (fast path)
        best_f = f.copy(); best_x = x.copy()
        Hinv = np.tile(np.eye(P), (N, 1, 1)) if Hinv0 is None \
            else np.array(Hinv0, float)
        gnorm = np.abs(g).max(1)
        start_it = 0
    bf_hist = [best_f.copy()]            # chi2-plateau history (fresh on resume; rebuilds)
    # per-POI row grouping + sigma1 history for the sigma1-stability early-stop (the rows
    # of one POI share its POI_IDX; their PV values ARE that POI's grid).
    poi_rows = {int(p): np.where(POI_IDX == p)[0] for p in np.unique(POI_IDX)}
    def _cur_sig1():                     # {poi_idx: dchi2=1 interval half-width}
        return {p: _interval_halfwidth(PV[r], best_f[r]) for p, r in poi_rows.items()}
    sig1_hist = [_cur_sig1()]            # fresh on resume (rebuilds; worst case +PATIENCE iters)
    # rank-aware trace path: under POI_SLICE/RANK_SLICE each rank owns a disjoint POI
    # set, so a single shared path would clobber -- insert _r{RANK} (mirrors the ckpt).
    bf_trace_path = BF_TRACE
    if BF_TRACE and (POI_SLICE or RANK_SLICE):
        root, ext = os.path.splitext(BF_TRACE)
        bf_trace_path = f"{root}_r{RANK}{ext}"
    trace_prefix = None                  # debug-only: prior-job trace to prepend on resume
    if bf_trace_path and resume_state is not None and os.path.exists(bf_trace_path):
        try:
            trace_prefix = np.load(bf_trace_path)["bf_hist"]
        except Exception:
            trace_prefix = None
    def _write_trace():                  # cheap; called every iter so a walltime kill
        if not bf_trace_path:            # mid-BFGS still leaves a valid partial trace
            return
        hist = np.array(bf_hist)
        if trace_prefix is not None:     # drop the duplicated resumed state, then extend
            hist = np.concatenate([trace_prefix, hist[1:]], axis=0)
        tmp = bf_trace_path + ".tmp.npz"
        np.savez(tmp, bf_hist=hist, PV=PV, POI_IDX=POI_IDX, start_it=start_it,
                 sigma_order=SIGMA, order=np.array(ORDER))  # sigma per ORDER idx for sigma1 replay
        os.replace(tmp, bf_trace_path)
    for it in range(start_it, maxit):
        active = gnorm > gtol
        if not active.any():
            break
        d = -np.einsum('bij,bj->bi', Hinv, g)
        gd = (g * d).sum(1)
        bad = gd >= 0
        d[bad] = -g[bad]; Hinv[bad] = np.eye(P); gd[bad] = (g[bad] * d[bad]).sum(1)
        # Armijo backtracking with FAST values, per-row alpha
        alpha = np.ones(N); accept = ~active
        x_new = x.copy(); f_new = f.copy()
        for _ls in range(MAXLS):
            if accept.all():
                break
            xt = np.clip(x + alpha[:, None] * d, -XBOX, XBOX)
            ft = fast_values_rows(POI_IDX, xt, PV)
            ok = (ft <= f + C1 * alpha * gd) & ~accept
            x_new[ok] = xt[ok]; f_new[ok] = ft[ok]; accept |= ok
            alpha[~accept] *= 0.5
        stuck = active & ~accept
        if stuck.any():
            xt = np.clip(x + alpha[:, None] * d, -XBOX, XBOX)
            x_new[stuck] = xt[stuck]
            f_new[stuck] = fast_values_rows(POI_IDX, x_new, PV)[stuck]
        g_new = iterate_grad(POI_IDX, x_new, PV, GRADMETHOD, fd_step=fd_step)
        s = x_new - x; y = g_new - g; sy = (s * y).sum(1)
        # RELATIVE curvature condition (standard BFGS safeguard): only update when
        # the (s,y) pair is meaningfully positive-curvature. The old absolute
        # sy>1e-12 was effectively a no-op near convergence, where s,y -> 0 and a
        # noisy pair corrupts Hinv (the it2 ||g|| bounce seen in validation).
        snorm = np.linalg.norm(s, axis=1); ynorm = np.linalg.norm(y, axis=1)
        curv_ok = sy > 1e-8 * snorm * ynorm
        for b in np.where(active & curv_ok)[0]:
            rho = 1.0 / sy[b]; I = np.eye(P)
            V = I - rho * np.outer(s[b], y[b])
            Hinv[b] = V @ Hinv[b] @ V.T + rho * np.outer(s[b], s[b])
        x, f, g = x_new, f_new, g_new
        gnorm = np.abs(g).max(1)
        upd = f < best_f; best_f[upd] = f[upd]; best_x[upd] = x[upd]
        if log_prefix:
            print(f"  {log_prefix} it{it}: min={best_f.min():.2f} "
                  f"||g||max={gnorm.max():.2e} active={int(active.sum())} "
                  f"({time.strftime('%H:%M:%S')})", flush=True)
        if ckpt_path:                                    # resumable BFGS state
            tmp = ckpt_path + ".tmp.npz"
            np.savez(tmp, POI_IDX=POI_IDX, PV=PV, x=x, Hinv=Hinv, f=f, g=g,
                     best_f=best_f, best_x=best_x, it=it + 1, gnorm=gnorm,
                     fd_step=(FD_STEP if fd_step is None else fd_step), gtol=gtol)
            os.replace(tmp, ckpt_path)                    # atomic
        # chi2-plateau early-stop: once the SLOWEST-improving row (the max per-row drop
        # over the window) has plateaued below FTOL, every row has, so stop this POI.
        bf_hist.append(best_f.copy())
        _write_trace()
        # ---- sigma1-STABILITY early-stop (preferred): stop when EVERY POI's interval
        # half-width has moved < SIGTOL*sigma(POI) over the window (the interval has
        # converged to SIGTOL sigmas). Model-agnostic; no per-model tuning. ----
        sig1_hist.append(_cur_sig1())
        if SIGTOL > 0 and len(sig1_hist) > SIGTOL_PATIENCE:
            prev = sig1_hist[-1 - SIGTOL_PATIENCE]; cur = sig1_hist[-1]
            worst = 0.0; ready = True; n_finite = 0; n_offgrid = 0
            for p in poi_rows:
                a, b = prev[p], cur[p]
                fa, fb = np.isfinite(a), np.isfinite(b)
                if fa and fb:
                    worst = max(worst, abs(b - a) / GRID_SIG[p]); n_finite += 1
                elif (not fa) and (not fb):
                    n_offgrid += 1        # unbracketed the WHOLE window -> off-grid: a grid/
                    #                       centering problem, NOT something more iters fix.
                    #                       Do NOT let it block the stop (this is what burned a
                    #                       slice to the wall before; the pre-pass prevents it).
                else:
                    ready = False; break  # interval just forming/lost -> wait one more window
            if ready and n_finite > 0 and worst < SIGTOL:
                if log_prefix:
                    extra = (f" ({n_offgrid} POI(s) UNBRACKETED/off-grid -- check centering)"
                             if n_offgrid else "")
                    print(f"  {log_prefix} sigma1 stable: worst d(sigma1)/sigma "
                          f"{worst:.2e} < SIGTOL {SIGTOL:.0e} over {SIGTOL_PATIENCE} iters "
                          f"-> stop at it{it}{extra}", flush=True)
                break
        # ---- chi2-plateau early-stop (legacy alternative; needs solver-floor tuning) ----
        if FTOL > 0 and len(bf_hist) > FTOL_PATIENCE:
            improve = float((bf_hist[-1 - FTOL_PATIENCE] - best_f).max())
            if improve < FTOL:
                if log_prefix:
                    print(f"  {log_prefix} chi2 plateau: max per-row improve "
                          f"{improve:.2e} < FTOL {FTOL:.0e} over {FTOL_PATIENCE} iters "
                          f"-> stop at it{it}", flush=True)
                break
    return best_f, best_x, gnorm


# Fisher / nuisance Hessian via central FD of the EXACT AD gradient (per row)
def nuisance_hessian(poi_idx, x_opt, poi_val):
    _, vg = _vg_one(poi_idx)
    cols = []
    for j in range(P):
        ep = np.array(x_opt); ep[j] += FDH
        em = np.array(x_opt); em[j] -= FDH
        _, gp = vg(jnp.asarray(ep), jnp.asarray(float(poi_val)))
        _, gm = vg(jnp.asarray(em), jnp.asarray(float(poi_val)))
        cols.append((np.asarray(gp) - np.asarray(gm)) / (2 * FDH))
    H = np.array(cols).T; H = 0.5 * (H + H.T)
    ev = np.linalg.eigvalsh(H)
    return H, ev, bool(np.all(ev > 0))


# ======================================================================
# inverse-Fisher BFGS preconditioner (Hinv0 from the warm-start Hessian)
# ======================================================================
def _chi2_full_scaled(xs_full, theta0):
    """chi^2 at the physical point theta0 + SIG*xs_full (ALL D dims free, scaled).
    AD-able in xs_full. Mirrors chi2_scaled_single but with no fixed POI dim."""
    theta = jnp.asarray(theta0) + SIG * xs_full
    p = dict(FIXED)
    for i, name in enumerate(ORDER):
        if name == "ln10As":
            p["A_s"] = jnp.exp(theta[i]) / 1e10
        else:
            p[name] = theta[i]
    out = model.run_cosmology_abbr(model.add_derived_parameters(p))
    return _single_chi2_from_out(out)


def _full_grad_scaled(theta0, xs):
    """AD gradient of chi^2 w.r.t. ALL D scaled coords, at offset xs from theta0."""
    e = jnp.eye(D)
    fs, g = jax.vmap(lambda v: jax.jvp(lambda z: _chi2_full_scaled(z, theta0),
                                       (xs,), (v,)))(e)
    return fs[0], g


def _warm_hessian_cache_path():
    cfg = os.path.splitext(os.path.basename(CONFIG_ABS))[0]
    return os.path.join(WARM_DIR, f"warm_hessian_{cfg}_l{LMAX}"
                        f"_tt{int(USE_LOWTT)}_ee{int(USE_LOWEE)}"
                        f"{'' if USE_PLIK else '_noplik'}.npz")


def _setup_plik_full_precond(theta_ref=None):
    """No-op retained for call-site stability. The full-plik inner profile is now
    self-contained: it computes its OWN per-cosmology Hessian each call (a fixed
    reference Hessian converged slowly for far cosmologies -- the calibration
    nuisances make the Hessian cls-dependent). No external preconditioner setup."""
    return


def _warm_precond_hessian(theta_warm, h=FDH):
    """Full D x D chi^2 Hessian in SCALED coords at theta_warm, via central FD of
    the exact AD gradient (2D AD-grad evals; one-time, single cosmology). Returns
    H (D,D) symmetric. Cached to disk (PA_HESS_CACHE) keyed by config/l_max/likelihood
    + theta_warm -- a precomputed cache loads instantly and is shared across POI_SLICE
    ranks; an absent/mismatched cache just recomputes (the original behaviour)."""
    theta_w = np.asarray(theta_warm, float)
    cache = _warm_hessian_cache_path()
    if HESS_CACHE and os.path.exists(cache):
        try:
            d = np.load(cache)
            if (d["H"].shape == (D, D) and np.allclose(d["theta_warm"], theta_w, atol=1e-9)
                    and int(d["lmax"]) == LMAX and bool(d["lowtt"]) == USE_LOWTT
                    and bool(d["lowee"]) == USE_LOWEE):
                print(f"[precond] loaded cached warm Hessian ({cache})", flush=True)
                return np.array(d["H"], float)
            print(f"[precond] cache present but mismatched -> recompute", flush=True)
        except Exception as ex:
            print(f"[precond] cache load failed ({ex}) -> recompute", flush=True)
    x0 = jnp.zeros(D)
    cols = []
    for j in range(D):
        _, gp = _full_grad_scaled(theta_warm, x0.at[j].add(h))
        _, gm = _full_grad_scaled(theta_warm, x0.at[j].add(-h))
        cols.append(np.asarray((gp - gm) / (2.0 * h), float))
    H = np.array(cols).T
    H = 0.5 * (H + H.T)
    if HESS_CACHE:
        try:
            tmp = cache + f".tmp.r{RANK}.npz"                # per-rank tmp -> safe atomic
            np.savez(tmp, H=H, theta_warm=theta_w, lmax=LMAX,
                     lowtt=USE_LOWTT, lowee=USE_LOWEE)
            os.replace(tmp, cache)
            print(f"[precond] saved warm Hessian cache ({cache})", flush=True)
        except Exception as ex:
            print(f"[precond] cache save failed ({ex})", flush=True)
    return H


def _hinv0_for_poi(H, poi_idx):
    """Per-POI initial inverse-Hessian (P,P): invert the nuisance submatrix of the
    full warm Hessian (drop the POI row/col), regularised to PD."""
    nuis = nuis_idx_of(poi_idx)
    sub = 0.5 * (H[np.ix_(nuis, nuis)] + H[np.ix_(nuis, nuis)].T)
    ev = np.linalg.eigvalsh(sub)
    lo = float(ev.min())
    if lo <= 1e-6:                                   # floor to PD
        sub = sub + (abs(min(lo, 0.0)) + 1e-3) * np.eye(P)
    return np.linalg.inv(sub)


# ======================================================================
# MLE PRE-PASS: joint D-dim minimum + Fisher errors (model-agnostic auto-centering)
# ======================================================================
def _mle_cache_path():
    cfg = os.path.splitext(os.path.basename(CONFIG_ABS))[0]
    return os.path.join(WARM_DIR, f"mle_prepass_{cfg}_l{LMAX}"
                        f"_tt{int(USE_LOWTT)}_ee{int(USE_LOWEE)}"
                        f"{'' if USE_PLIK else '_noplik'}.npz")


def _fisher_grid_sigma(H):
    """Per-parameter PROFILE-likelihood 1sigma (the dchi2=1 half-width) from the joint
    SCALED Hessian H (= d2chi2/dxs2, xs=(theta-CENTER)/SIGMA). For a quadratic chi2 the
    profile error equals the marginal error: cov = 2 H^{-1}, so sigma_scaled_i =
    sqrt(2 (H^{-1})_ii); physical sigma_i = sigma_scaled_i * SIGMA_i. Eigen-floored for
    PD safety and clamped to [1/MLE_SIG_CLIP, MLE_SIG_CLIP] x the prior SIGMA so a noisy
    Hessian can never produce an absurd grid (it just falls back toward the prior guess)."""
    Hs = 0.5 * (H + H.T)
    ev, V = np.linalg.eigh(Hs)
    ev = np.maximum(ev, 1e-6)                         # floor to PD
    Hinv = (V * (1.0 / ev)) @ V.T
    var_scaled = np.maximum(2.0 * np.diag(Hinv), 1e-12)
    sig_scaled = np.sqrt(var_scaled)                  # in units of the prior SIGMA
    sig_scaled = np.clip(sig_scaled, 1.0 / MLE_SIG_CLIP, MLE_SIG_CLIP)
    return SIGMA * sig_scaled


def _joint_mle(theta_start):
    """Solve the joint D-dim MLE (ALL params free, plik nuisances inner-profiled) with
    L-BFGS-B on the exact AD gradient, then evaluate the Hessian at the optimum. Returns
    (theta_mle (D,), H_mle (D,D) scaled, chi2_min). Cached to disk (shared across
    POI_SLICE ranks / survives resume). The expensive Hessian DOUBLES as the BFGS
    preconditioner, so the only NET new cost vs the old warm-Hessian flow is the solve."""
    import scipy.optimize as _spo
    cache = _mle_cache_path()
    if HESS_CACHE and os.path.exists(cache):
        try:
            d = np.load(cache)
            if (d["theta_mle"].shape == (D,) and d["H"].shape == (D, D)
                    and int(d["lmax"]) == LMAX and bool(d["lowtt"]) == USE_LOWTT
                    and bool(d["lowee"]) == USE_LOWEE
                    and np.allclose(d["center"], CENTER, atol=1e-9)):
                print(f"[mle] loaded cached joint MLE ({cache})", flush=True)
                return (np.array(d["theta_mle"], float), np.array(d["H"], float),
                        float(d["chi2_min"]))
            print(f"[mle] cache present but mismatched -> recompute", flush=True)
        except Exception as ex:
            print(f"[mle] cache load failed ({ex}) -> recompute", flush=True)
    cen = jnp.asarray(CENTER)
    vg = jax.jit(lambda xs: _full_grad_scaled(cen, xs))      # compile once; fast thereafter
    nfev = [0]
    def fun(x):
        f, g = vg(jnp.asarray(x))
        nfev[0] += 1
        return float(f), np.asarray(g, float)
    x0 = (np.asarray(theta_start, float) - CENTER) / SIGMA
    t0 = time.perf_counter()
    res = _spo.minimize(fun, x0, jac=True, method="L-BFGS-B",
                        bounds=[(-6.0, 6.0)] * D,
                        options={"maxiter": MLE_MAXITER, "ftol": 1e-10, "gtol": 1e-6})
    theta_mle = CENTER + SIGMA * np.asarray(res.x, float)
    print(f"[mle] joint solve: chi2_min={float(res.fun):.3f} in {nfev[0]} evals "
          f"{time.perf_counter()-t0:.0f}s (L-BFGS-B {('ok' if res.success else 'stop:'+str(res.message))})",
          flush=True)
    H = _warm_precond_hessian(theta_mle)                      # DxD scaled Hessian AT the MLE
    if HESS_CACHE:
        try:
            tmp = cache + f".tmp.r{RANK}.npz"
            np.savez(tmp, theta_mle=theta_mle, H=H, chi2_min=float(res.fun),
                     center=CENTER, lmax=LMAX, lowtt=USE_LOWTT, lowee=USE_LOWEE)
            os.replace(tmp, cache)
            print(f"[mle] saved joint-MLE cache ({cache})", flush=True)
        except Exception as ex:
            print(f"[mle] cache save failed ({ex})", flush=True)
    return theta_mle, H, float(res.fun)


# ======================================================================
# warm starts from the entry-(a) global best fit
# ======================================================================
def _global_best_fit_physical():
    """Read the entry-(a) profile npz files (scan/results/profile_prod_<poi>.npz),
    find the single global best-fit (lowest chi2 across ALL POIs' grid points),
    and return its physical D-vector (in this config's ORDER). The npz stores,
    per POI: poi_grid (G,), chi2 (G,), xstar (G, P_old) scaled nuisances, nuis
    (P_old,) the non-POI param NAMES. We pick the min-chi2 (poi_val, xstar) pair
    and translate it into a full physical vector keyed by name; any ORDER param
    not present in the old LCDM run (e.g. Neff) defaults to its config fiducial.
    Returns (theta_phys (D,), provenance str) or (None, reason)."""
    best = None  # (chi2, poi_name, poi_val, {name: phys_value})
    used = []
    # the entry-(a) run was 6-param LCDM; map by NAME
    old_cen = {"h": 0.6736, "omega_b": 0.02237, "omega_cdm": 0.1200,
               "n_s": 0.9649, "ln10As": 3.044, "tau_reion": 0.0544}
    old_sig = {"h": 0.0054, "omega_b": 0.00015, "omega_cdm": 0.0012,
               "n_s": 0.0042, "ln10As": 0.014, "tau_reion": 0.0073}
    for poi in old_cen:
        f = os.path.join(WARM_DIR, f"profile_prod_{poi}.npz")
        if not os.path.exists(f):
            continue
        try:
            d = np.load(f, allow_pickle=True)
            grid = np.asarray(d["poi_grid"]); chi2 = np.asarray(d["chi2"])
            xstar = np.asarray(d["xstar"]); nuis = [str(s) for s in d["nuis"]]
            j = int(np.nanargmin(chi2))
            phys = {poi: float(grid[j])}
            for k, nm in enumerate(nuis):
                phys[nm] = old_cen[nm] + old_sig[nm] * float(xstar[j, k])
            used.append(f"{poi}:{chi2[j]:.2f}")
            if best is None or chi2[j] < best[0]:
                best = (float(chi2[j]), poi, float(grid[j]), phys)
        except Exception as e:
            print(f"[warm] skip {f}: {e}", flush=True)
    if best is None:
        return None, "no entry-(a) npz found"
    _, poi, _, phys = best
    theta = CENTER.copy()
    for i, nm in enumerate(ORDER):
        if nm in phys:
            theta[i] = phys[nm]
    prov = (f"global best chi2={best[0]:.2f} from {poi} profile "
            f"(npz fields poi_grid/chi2/xstar/nuis; "
            f"params {sorted(phys)} matched, others=config fiducial)")
    return theta, prov


def _warm_x0_for_rows(POI_IDX, PV, theta_warm):
    """Translate a physical warm-start vector theta_warm into per-row scaled
    nuisance starts x0 (N,P). For each row, the warm value of its free dims is
    (theta_warm[i] - CEN[i]) / SIG[i] (the POI dim itself is fixed at PV)."""
    N = len(PV); x0 = np.zeros((N, P))
    for b in range(N):
        nuis = nuis_idx_of(int(POI_IDX[b]))
        for k, i in enumerate(nuis):
            x0[b, k] = (theta_warm[i] - CENTER[i]) / SIGMA[i]
    return x0


# ======================================================================
# it0 calibration: tune the FD step against the exact AD gradient
# ======================================================================
def calibrate_fd_step(POI_IDX, X, PV):
    """Pick the central-FD step that best matches the exact batched-AD gradient on
    a subsample. Central FD has a U-shaped error vs step: truncation ~step^2 for
    LARGE steps, roundoff/solver-noise ~1/step for SMALL steps. The old code only
    halved and returned the LAST (smallest) step -> it walked straight into the
    noise floor (job 54442539 returned step=1.25e-3 at 8% error when step=1e-2 gave
    1.2%). Now we sweep a LADDER spanning both sides of PA_FD_STEP and return the
    BEST (lowest max-rel). Returns (step, max_rel, n_sample)."""
    N = len(PV)
    n = min(max(FD_CALMIN, 1), N)
    # evenly-spaced subsample across the row set (covers all POIs/grid extents)
    sel = np.unique(np.linspace(0, N - 1, n).astype(int))
    Ps, Xs, Vs = POI_IDX[sel], X[sel], PV[sel]
    _, G_ad = ad_grad_rows(Ps, Xs, Vs)
    denom = np.maximum(np.abs(G_ad), np.percentile(np.abs(G_ad), 90) + 1e-30)
    # ladder centred on PA_FD_STEP, biased upward (the noise floor is on the small
    # side, so probe larger steps too). CAL_RETRIES controls how far up we go.
    mults = sorted(set([0.5, 1.0] + [2.0 ** k for k in range(1, CAL_RETRIES + 1)]))
    ladder = [FD_STEP * m for m in mults]
    best_step, best_rel = ladder[0], np.inf
    for step in ladder:
        G_fd = fdbatch_grad(Ps, Xs, Vs, step=step)
        rel = np.abs(G_fd - G_ad) / denom
        max_rel = float(np.nanmax(rel))
        per_dir = np.nanmax(rel, axis=0)            # (P,) worst per direction
        flag = " *" if max_rel < best_rel else ""
        print(f"[cal] step={step:.2e} max-rel(fd,ad)={max_rel:.3e} "
              f"per-dir={np.array2string(per_dir, precision=2)} n={len(sel)}{flag}",
              flush=True)
        if max_rel < best_rel:
            best_step, best_rel = step, max_rel
    if best_rel > FD_CALTOL:
        print(f"[cal] WARNING: best max-rel {best_rel:.3e} (@step={best_step:.2e}) "
              f"exceeds target {FD_CALTOL:.1e} -- FD gradient is at its noise floor. "
              f"BFGS will use it for iterations (the certificate is exact AD); if "
              f"||g||_AD stalls above GTOL, switch this config to grad_method='ad'.",
              flush=True)
    return best_step, best_rel, len(sel)


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
    i = ORDER.index(poi)
    c, s = GRID_CEN[i], GRID_SIG[i]                  # MLE-recentered when PA_MLE_PREPASS ran
    return np.linspace(c - NSIG * s, c + NSIG * s, NPTS)


def profile_lockstep(pois, outdir):
    """Build ONE row set (all pois x NPTS grid points), warm-start, run BFGS in
    lockstep, then certify with the exact AD gradient + Hessian, and write one
    npz/png per POI."""
    # ---- MLE PRE-PASS (model-agnostic auto-centering): solve the joint optimum FIRST,
    # then center every grid on it with a Fisher-matched span. Must run BEFORE the grid
    # assembly (which reads GRID_CEN/GRID_SIG via _grid_for). The MLE Hessian is reused
    # as the BFGS preconditioner (H_prepass) so no Hessian is computed twice. ----
    H_prepass = None
    if MLE_PREPASS:
        global GRID_CEN, GRID_SIG
        theta_mle, H_prepass, chi2_mle = _joint_mle(CENTER)
        GRID_CEN = np.asarray(theta_mle, float)
        GRID_SIG = _fisher_grid_sigma(H_prepass)
        print("[mle] recentered grids on joint MLE (chi2_min=%.3f):" % chi2_mle, flush=True)
        for i, nm in enumerate(ORDER):
            print(f"[mle]   {nm:>10}: cen {CENTER[i]:.5g} -> {GRID_CEN[i]:.5g} "
                  f"({(GRID_CEN[i]-CENTER[i])/SIGMA[i]:+.2f} prior-sig); "
                  f"grid-sig {GRID_SIG[i]:.3g} (prior {SIGMA[i]:.3g}, "
                  f"{GRID_SIG[i]/SIGMA[i]:.2f}x)", flush=True)
    # ---- assemble the flat row set ----
    rows_poi = []; rows_pv = []; row_of_poi = {}
    for poi in pois:
        idx = ORDER.index(poi)
        grid = _grid_for(poi)
        start = len(rows_pv)
        for v in grid:
            rows_poi.append(idx); rows_pv.append(float(v))
        row_of_poi[poi] = (start, start + NPTS, grid)
    POI_IDX = np.array(rows_poi, int); PV = np.array(rows_pv, float)
    N = len(PV)

    # ---- full-plik inner-profile preconditioner (once, before any likelihood call) ----
    _setup_plik_full_precond(CENTER)

    # ---- optional multi-node: slice the row list across ranks ----
    if RANK_SLICE and NPROC > 1:
        mine = np.arange(RANK, N, NPROC)
    else:
        mine = np.arange(N)
    # (we keep the full row_of_poi mapping; rows not in `mine` are NaN-filled)

    # ---- warm starts ----
    x0 = np.zeros((N, P))
    prov = "cold (x0=0)"
    if WARM:
        if MLE_PREPASS:
            theta_warm = GRID_CEN.copy()       # the joint MLE found by the pre-pass
            prov = "joint MLE (PA_MLE_PREPASS)"
        elif WARM_FROM_CEN:
            theta_warm = CENTER.copy()
            prov = "config CENTER (PA_WARM_FROM_CEN; = joint MLE for a recentered run)"
        else:
            theta_warm, prov = _global_best_fit_physical()
        if theta_warm is not None:
            x0 = _warm_x0_for_rows(POI_IDX, PV, theta_warm)
        else:
            prov = f"cold (x0=0); warm unavailable: {prov}"
    print(f"[lockstep] N={N} rows ({len(pois)} POIs x {NPTS}) P={P} D={D} "
          f"grad={GRADMETHOD} shard={DO_SHARD} warm: {prov}", flush=True)

    Ps, Xs, Vs = POI_IDX[mine], x0[mine], PV[mine]

    # ---- resumable BFGS state: reload if a matching checkpoint exists ----
    ckpt = os.path.join(outdir, f"profile_prod_ad_STATE{TAG}"
                        f"{'_r%d' % RANK if (RANK_SLICE or POI_SLICE) else ''}.npz")
    resume_state = None
    if RESUME and os.path.exists(ckpt):
        try:
            st = np.load(ckpt, allow_pickle=True)
            ok = (len(st["PV"]) == len(Vs) and np.array_equal(st["POI_IDX"], Ps)
                  and np.allclose(st["PV"], Vs) and float(st["gtol"]) == GTOL)
            if ok:
                resume_state = {k: st[k] for k in
                                ("x", "Hinv", "f", "g", "best_f", "best_x", "it")}
                resume_state["fd_step"] = float(st["fd_step"]) \
                    if "fd_step" in st.files else FD_STEP
                print(f"[resume] BFGS checkpoint @ it{int(st['it'])} matches; "
                      f"continuing ({ckpt})", flush=True)
            else:
                print(f"[resume] checkpoint shape/grid/gtol mismatch -> fresh start",
                      flush=True)
        except Exception as ex:
            print(f"[resume] could not load checkpoint ({ex}) -> fresh start",
                  flush=True)

    # ---- it0 calibration (fdbatch only; skipped on resume) ----
    fd_step = FD_STEP; cal_maxrel = float('nan'); cal_n = 0
    if resume_state is not None:
        fd_step = resume_state.pop("fd_step", FD_STEP)
        print(f"[cal] resumed fd_step={fd_step:.2e} (calibration skipped)", flush=True)
    elif GRADMETHOD == "fdbatch":
        fd_step, cal_maxrel, cal_n = calibrate_fd_step(Ps, Xs, Vs)
        print(f"[cal] FINAL fd_step={fd_step:.2e} max-rel(fd,ad)={cal_maxrel:.3e} "
              f"(n={cal_n}; target<={FD_CALTOL:.1e})", flush=True)

    # ---- inverse-Fisher preconditioner: Hinv0 from the warm-start Hessian ----
    Hinv0 = None
    if PRECOND and resume_state is None and WARM and theta_warm is not None:
        try:
            tH = time.perf_counter()
            # reuse the MLE-pre-pass Hessian (computed AT the MLE) when available;
            # else compute the warm Hessian at theta_warm (legacy path)
            Hwarm = H_prepass if H_prepass is not None else _warm_precond_hessian(theta_warm)
            by_poi = {ORDER.index(p): _hinv0_for_poi(Hwarm, ORDER.index(p))
                      for p in pois}
            Hinv0 = np.stack([by_poi[int(Ps[b])] for b in range(len(Ps))])
            conds = {p: float(np.linalg.cond(np.linalg.inv(by_poi[ORDER.index(p)])))
                     for p in pois}
            print(f"[precond] inverse-Fisher Hinv0 from warm Hessian "
                  f"({time.perf_counter()-tH:.0f}s); nuis-Hessian cond per POI: "
                  f"{ {k: round(v,1) for k,v in conds.items()} }", flush=True)
        except Exception as ex:
            print(f"[precond] failed ({ex}) -> identity Hinv0", flush=True)
            Hinv0 = None

    # ---- lockstep BFGS over the (sliced) row set ----
    t0 = time.perf_counter()
    bf, bx, gn_iter = bfgs_rows(Ps, Vs, x0=Xs, Hinv0=Hinv0, fd_step=fd_step,
                                log_prefix="[lock]", ckpt_path=ckpt,
                                resume_state=resume_state)

    # ---- FINAL stationarity certificate: ALWAYS exact AD ----
    print(f"[cert] AD ||g|| certificate over {len(mine)} rows ...", flush=True)
    cert_chi2, G_cert = ad_grad_rows(Ps, bx, Vs)
    gnorm_ad = np.abs(G_cert).max(1)
    converged = gnorm_ad < GTOL

    # ---- scatter results back into full-length arrays ----
    best_f = np.full(N, np.nan); best_x = np.full((N, P), np.nan)
    gnorm_full = np.full(N, np.nan); conv_full = np.zeros(N, bool)
    best_f[mine] = bf; best_x[mine] = bx
    gnorm_full[mine] = gnorm_ad; conv_full[mine] = converged

    # ---- Hessian / PD per row (optional) ----
    pd_full = np.zeros(N, bool); cond_full = np.full(N, np.nan)
    if DO_HESS:
        for b in mine:
            _, ev, is_pd = nuisance_hessian(int(POI_IDX[b]), best_x[b], PV[b])
            pd_full[b] = is_pd; cond_full[b] = ev.max() / max(ev.min(), 1e-30)

    elapsed = time.perf_counter() - t0
    print(f"[lockstep] BFGS+cert done {elapsed:.0f}s RSS={rss_gb():.1f}GB; "
          f"converged {int(conv_full[mine].sum())}/{len(mine)} "
          f"(||g||_AD<{GTOL:.1e}); max||g||_AD={np.nanmax(gnorm_full):.2e}",
          flush=True)

    # ---- write one npz/png per POI ----
    for poi in pois:
        s, e, grid = row_of_poi[poi]
        sl = slice(s, e)
        chi2 = best_f[sl]; xstar = best_x[sl]; gnorm = gnorm_full[sl]
        conv = conv_full[sl]; pd = pd_full[sl]; cond = cond_full[sl]
        nuis = [ORDER[i] for i in nuis_idx_of(ORDER.index(poi))]
        lo1, mid, hi1 = interval(grid, chi2, 1.0)
        lo2, _, hi2 = interval(grid, chi2, 4.0)
        sig_p = sigma_parabola(grid, chi2)
        npz = os.path.join(outdir, f"profile_prod_ad_{poi}{TAG}.npz")
        np.savez(npz, poi=poi, poi_grid=grid, chi2=chi2, xstar=xstar,
                 gnorm=gnorm, converged=conv, hess_pd=pd, hess_cond=cond,
                 nuis=np.array(nuis), done=True,
                 sigma1=np.array([lo1, mid, hi1]), sigma2=np.array([lo2, hi2]),
                 sigma_parab=sig_p, gtol=GTOL, gradmethod=GRADMETHOD,
                 fd_step=fd_step, cal_maxrel=cal_maxrel, cal_n=cal_n,
                 use_lowee=USE_LOWEE, use_lowtt=USE_LOWTT, config=CONFIG_ABS)
        j = int(np.nanargmin(chi2))
        nconv = int(np.nansum(conv)); npt_valid = int(np.isfinite(chi2).sum())
        # CENTERING self-check: a min railing at idx 0/1 or N-1/N-2, or a nan interval
        # edge, means the grid does not bracket dchi2=1 (the failure the MLE pre-pass
        # fixes). Surface it LOUDLY rather than shipping a railed/half interval.
        edge = (j <= 1 or j >= NPTS - 2)
        railed = edge or not (np.isfinite(lo1) and np.isfinite(hi1))
        warn = ""
        if railed:
            why = []
            if j <= 1: why.append(f"min at idx {j} (LOW edge)")
            if j >= NPTS - 2: why.append(f"min at idx {j} (HIGH edge)")
            if not np.isfinite(lo1): why.append("lower 1sig off-grid")
            if not np.isfinite(hi1): why.append("upper 1sig off-grid")
            warn = ("  ** CENTERING WARNING: " + "; ".join(why)
                    + ("; rerun with PA_MLE_PREPASS=1 to auto-center" if not MLE_PREPASS
                       else "; pre-pass ran -- profile may be non-Gaussian, widen PA_NSIG"))
        print(f"[{poi}] minchi2={chi2[j]:.2f} at {poi}={grid[j]:.5f}; "
              f"1sig=[{lo1:.5f},{hi1:.5f}] (PCHIP +/-{(hi1-lo1)/2:.5f}; "
              f"parab={sig_p:.5f}); converged {nconv}/{npt_valid}; "
              f"max||g||_AD={np.nanmax(gnorm):.2e}; "
              f"PD {int(np.nansum(pd))}/{npt_valid} -> {npz}{warn}", flush=True)
        _plot(poi, grid, chi2, npz)

    # run finished cleanly -> drop the resumable BFGS checkpoint
    try:
        if os.path.exists(ckpt):
            os.remove(ckpt)
    except Exception:
        pass


def multistart(poi, outdir, K=MS_K):
    _setup_plik_full_precond(CENTER)             # full-plik inner-profile preconditioner (no-op for lite)
    poi_idx = ORDER.index(poi)
    i = poi_idx; c, s = CENTER[i], SIGMA[i]
    test_vals = np.array([c, c + 2 * s])
    rng = np.random.default_rng(1234)
    print(f"[{poi}] MULTISTART K={K} at {poi}={test_vals} grad={GRADMETHOD}", flush=True)
    saved = {}
    for vi, pv in enumerate(test_vals):
        PV = np.full(K, pv); POI_IDX = np.full(K, poi_idx, int)
        x0 = rng.uniform(-2.5, 2.5, (K, P)); x0[0] = 0.0
        fd_step = FD_STEP
        if GRADMETHOD == "fdbatch":
            fd_step, _, _ = calibrate_fd_step(POI_IDX, x0, PV)
        bf, bx, _ = bfgs_rows(POI_IDX, PV, x0=x0, fd_step=fd_step,
                              log_prefix=f"[{poi}@{pv:.4f}]")
        _, G = ad_grad_rows(POI_IDX, bx, PV)
        gn = np.abs(G).max(1)
        spread = bf.max() - bf.min()
        print(f"[{poi}@{pv:.5f}] converged chi2: min={bf.min():.3f} max={bf.max():.3f} "
              f"spread={spread:.3f} max||g||_AD={gn.max():.1e} "
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
        i = ORDER.index(poi)
        ax.axvline(CENTER[i], ls=":", color="gray", lw=0.8, label=f"fiducial {poi}")
        ax.set_xlabel(poi); ax.set_ylabel(r"$\Delta\chi^2$ (profiled, BFGS)")
        ax.set_ylim(0, max(10, float(np.nanmax(d)) * 1.05)); ax.legend()
        ax.set_title(f"profile of {poi}: plik-lite + lowTT + lowEE")
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
          f"POIs={POIS} NPTS={NPTS} NSIG={NSIG} GTOL={GTOL} rtol={RTOL} "
          f"grad={GRADMETHOD} shard={DO_SHARD} lowEE={USE_LOWEE} lowTT={USE_LOWTT} "
          f"hess={DO_HESS} warm={WARM} multistart={MULTISTART} "
          f"rank_slice={RANK_SLICE} poi_slice={POI_SLICE} nproc={NPROC}", flush=True)
    if MULTISTART:
        # multistart is per-POI; split POIs across ranks (cheap, independent)
        mine = POIS[RANK::NPROC] if not RANK_SLICE else POIS
        for poi in mine:
            multistart(poi, outdir)
    else:
        # lockstep: ONE batch across all POIs (or this rank's POI slice).
        if POI_SLICE and NPROC > 1:
            # MULTI-NODE scale-out: each rank runs its own lockstep over a DISJOINT
            # POI subset and writes only those POIs' npz (distinct filenames -> no
            # clobber, no MPI). Per-rank STATE checkpoint. This is the node-scaling
            # path for few-hours wall-clock; supersedes the broken RANK_SLICE.
            pois = _resume_skip(POIS[RANK::NPROC], outdir)
        elif RANK_SLICE:
            # row-slice across ranks: BROKEN for the per-POI npz write (every rank
            # clobbers it with NaN-filled rows). Use POI_SLICE instead.
            pois = _resume_skip(POIS, outdir)
        else:
            # single node: one rank does everything; idle the rest to avoid
            # duplicate work (legacy one-POI-per-rank is gone)
            pois = _resume_skip(POIS, outdir) if RANK == 0 else []
        if pois:
            profile_lockstep(pois, outdir)
    print(f"rank {RANK}: done", flush=True)


if __name__ == "__main__":
    main()

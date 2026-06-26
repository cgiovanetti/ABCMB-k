"""smc_plikfull.py — fast/slow tempered SMC on the full Planck plik, marginalising
the 21 foreground/calibration nuisances (the nuisance-marginalised Bayesian anchor;
a feasibility study of a real nuisance-marginalised Planck analysis on the batched
ABCMB engine).

Difference vs scan/smc.py (the plik-lite run): there the 6 cosmological params are
sampled and the single A_planck is profiled. Here the high-ell likelihood is the
full Planck 2018 plik TTTEEE (clipy; 21 floated foreground/calibration nuisances +
26 fixed), and all 21 floated nuisances are sampled alongside the cosmology -> the
particle is (Dc cosmo + 21 nuisance)-dimensional and the posterior is genuinely
nuisance-marginalised. Low-ell TT (Commander) + low-ell EE (SRoll2) are included
exactly as in smc.py (calib=1: A_planck's effect on low-ell is negligible vs the
low-ell error, so low-ell stays a pure function of the Cls).

The fast/slow split (the time saver): a nuisance-only move does not change the theory
Cls, so it does not re-run ABCMB -- exactly Cobaya's fast/slow oversampling. We
exploit this with Metropolis-within-Gibbs, two blocks per tempering stage:

  * Slow block (the Dc cosmology params): a proposal changes the Cls -> one
    Model.call_batched (the expensive GPU theory solve) + clipy + low-ell. A few
    moves per stage (SMC_MOVES_SLOW, default 2).
  * Fast block (the 21 nuisances): a proposal leaves the cosmology -- and hence the
    Cls and the low-ell chi^2 -- unchanged, so we reuse the cached per-particle clik
    C_l block and only re-evaluate clipy (the high-ell data chi^2) + the Gaussian/
    joint-SZ nuisance priors. No ABCMB. Oversampled (SMC_MOVES_FAST, default 20)
    because each move is nearly free relative to the theory solve.

Both block kernels are pi_beta-invariant, so their composition is a valid SMC
mutation kernel; the tempering / ESS / resampling / evidence machinery is identical
to smc.py (it only ever touches the per-particle tempered chi^2). The cost per stage
is ~ MOVES_SLOW * N theory solves (the fast moves are nearly free), so marginalising
21 nuisances costs about the same ABCMB time as the plik-lite run while delivering a
full nuisance-marginalised posterior.

Prior / likelihood (tempered) decomposition. Mirrors smc.py's flat-box scheme exactly
(max code reuse, low bug surface):
  prior  : flat box on every sampled coordinate. Cosmo box = CEN +- PRIOR_NSIG*SIG
           (tau floored). Nuisance box = the Planck uniform-prior bounds where finite,
           else start +- NUIS_BOX_NSIG*scale (wide enough that the Gaussian-prior
           tails are negligible at the edge).
  L^beta : the full penalised high-ell chi^2 (clipy bare data chi^2 + the Planck
           Gaussian + joint-SZ nuisance priors = plik_full.penalized_chi2) + low-ell,
           tempered by beta: 0 -> 1.
At beta=1, prior x L = flatbox x exp(-1/2 (data + lowl + nuisance-priors)) = the true
nuisance-marginalised posterior (the wide flat box only truncates negligible tails).
logZ is the evidence under this flat-box prior convention (not directly comparable to
the plik-lite logZ=-509.92: different parameter space + prior). Documented in the npz.

Eager Python loop around Model.call_batched + a jitted batched clipy: the pipeline is
GPU->CPU-HyRex->GPU and is not end-to-end jittable (TOOL_PLAN gotcha #1); the pipeline
is never wrapped here.

Env knobs (all optional):
  SMC_N(512) SMC_MOVES_SLOW(2) SMC_MOVES_FAST(20) SMC_ESS_TARGET(0.5)
  SMC_EVAL_CHUNK(128; ABCMB cosmologies/call_batched) SMC_CLIPY_CHUNK(32; nuisances/clipy call)
  SMC_LMAX(2508) SMC_SEED(0) SMC_RTOL(1e-5) SMC_MAXSTAGES(120)
  SMC_PRIOR_NSIG(5) SMC_NUIS_BOX_NSIG(6) SMC_TAU_FLOOR(0.01)
  SMC_OUT(scan/results/smc_plikfull) SMC_CONFIG(scan/configs/lcdm_plikfull.py)
  SMC_USE_LOWTT/SMC_USE_LOWEE(cfg) SMC_SHARD(auto|0|1) SMC_RESUME(path; auto _state.npz)

Run via srun, PYTHONPATH=$(pwd), JAX_COMPILATION_CACHE_DIR set (see smc_plikfull.slurm).
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
from scan.plik_full import PlikFull, FLOAT_NAMES
from scan.lowl_like import LowLEE, LowLTT

_HERE = os.path.dirname(os.path.abspath(__file__))
CHI2_INF = 1e6                      # finite stand-in for out-of-box / NaN chi2


# ======================================================================
# config loading (same scheme as smc.py / profile_prod_ad.py)
# ======================================================================
def _load_config(path):
    if not os.path.isabs(path):
        for base in (os.getcwd(), _HERE, os.path.join(_HERE, "configs")):
            cand = os.path.join(base, path)
            if os.path.exists(cand):
                path = cand
                break
    spec = importlib.util.spec_from_file_location("smc_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG, os.path.abspath(path)


CONFIG_PATH = os.environ.get("SMC_CONFIG", "scan/configs/lcdm_plikfull.py")
CFG, CONFIG_ABS = _load_config(CONFIG_PATH)
if CFG.get("high_ell", "plik_lite") != "plik_full":
    raise SystemExit(f"[smc-pf] config {CONFIG_ABS} is not high_ell='plik_full'; "
                     f"use scan/configs/lcdm_plikfull.py (or set high_ell). This "
                     f"script marginalises the FULL-plik nuisances.")

# ---- parameter vector: [cosmo..., 21 nuisances...] ----
COSMO_ORDER = list(CFG["order"]); Dc = len(COSMO_ORDER)
NUIS_ORDER = list(FLOAT_NAMES); Dn = len(NUIS_ORDER)          # 21
ORDER = COSMO_ORDER + NUIS_ORDER; D = Dc + Dn
CEN_C = np.array([CFG["cen"][k] for k in COSMO_ORDER])
SIG_C = np.array([CFG["sig"][k] for k in COSMO_ORDER])
FIXED = dict(CFG["fixed"])
USER_SPECIES = CFG.get("user_species", None)

# ---------------- env config ----------------
N = int(os.environ.get("SMC_N", 512))
EVAL_CHUNK = int(os.environ.get("SMC_EVAL_CHUNK", 128))      # ABCMB cosmologies / call_batched
CLIPY_CHUNK = int(os.environ.get("SMC_CLIPY_CHUNK", 32))     # nuisances / clipy batched eval
MOVES_SLOW = int(os.environ.get("SMC_MOVES_SLOW", 2))        # cosmo (theory) moves / stage
MOVES_FAST = int(os.environ.get("SMC_MOVES_FAST", 20))       # nuisance (clipy-only) moves / stage
ESS_TARGET = float(os.environ.get("SMC_ESS_TARGET", 0.5))
LMAX = int(os.environ.get("SMC_LMAX", 2508))
SEED = int(os.environ.get("SMC_SEED", 0))
OUT = os.environ.get("SMC_OUT", os.path.join(_HERE, "results", "smc_plikfull"))
RTOL = float(os.environ.get("SMC_RTOL", 1e-5))
MAXSTAGES = int(os.environ.get("SMC_MAXSTAGES", 120))
PRIOR_NSIG = float(os.environ.get("SMC_PRIOR_NSIG", 5.0))    # cosmo flat box half-width (sigma)
NUIS_BOX_NSIG = float(os.environ.get("SMC_NUIS_BOX_NSIG", 6.0))  # unbounded-nuisance box half-width
TAU_FLOOR = float(os.environ.get("SMC_TAU_FLOOR", 0.01))
USE_LOWTT = (os.environ["SMC_USE_LOWTT"] != "0") if "SMC_USE_LOWTT" in os.environ \
    else bool(CFG.get("use_lowtt", True))
USE_LOWEE = (os.environ["SMC_USE_LOWEE"] != "0") if "SMC_USE_LOWEE" in os.environ \
    else bool(CFG.get("use_lowee", True))
_shard_env = os.environ.get("SMC_SHARD", "auto").lower()
RESUME_PATH = os.environ.get("SMC_RESUME", "")
STATE_NPZ = OUT + "_state.npz"

try:
    NDEV = len(jax.devices('gpu'))
except Exception:
    NDEV = 1
DO_SHARD = (_shard_env == "1") or (_shard_env == "auto" and NDEV > 1)


# ======================================================================
# likelihood machinery: FULL plik (clipy) high-ell + low-ell TT/EE
# ======================================================================
plf = PlikFull()                       # full plik via clipy (prints its self-test)
pll = PlikLite()                       # used ONLY for abcmb_cl_to_Dl (low-ell D_ell)
lowee = LowLEE() if USE_LOWEE else None
lowtt = LowLTT() if USE_LOWTT else None
model = Model(user_species=USER_SPECIES, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
              rtol_large_k_PE=RTOL, atol_large_k_PE=RTOL * 1e-2,
              rtol_small_k_PE=min(1e-5, RTOL), max_steps_PE=16384)

# nuisance prior arrays (numpy host copies of the plik_full statics)
NU_START = np.asarray(plf.start, float)        # (21,) physical start
NU_SCALE = np.asarray(plf.scale, float)        # (21,) conditioning / ref scale
PRIOR_MEAN = np.asarray(plf.prior_mean, float)  # (21,) Gaussian-prior means (0 where uniform)
PRIOR_INVVAR = np.asarray(plf.prior_invvar, float)  # (21,) 1/sigma^2 (0 where uniform)
NU_LO = np.asarray(plf.lo, float)              # (21,) physical lower bounds (-inf unbounded)
NU_HI = np.asarray(plf.hi, float)              # (21,) physical upper bounds (+inf unbounded)
I_KSZ, I_ASZ = int(plf.i_ksz), int(plf.i_asz)
SZ_MEAN, SZ_SIG = float(plf.sz_mean), float(plf.sz_sig)


def prior_penalty_np(nu):
    """(M,21) physical nuisances -> (M,) Gaussian + joint-SZ chi^2 penalty (numpy).
    Mirrors PlikFull.prior_penalty exactly (folded into the tempered likelihood)."""
    nu = np.atleast_2d(nu)
    gauss = np.sum(PRIOR_INVVAR * (nu - PRIOR_MEAN) ** 2, axis=-1)
    sz = nu[..., I_KSZ] + 1.6 * nu[..., I_ASZ]
    return gauss + ((sz - SZ_MEAN) / SZ_SIG) ** 2


# jitted batched clipy bare data chi^2 (vmap of PlikFull.data_chi2 over a batch axis).
# cls2d_B: (b,6,lmax+1) muK^2 clik blocks; nu_B: (b,21) physical -> (b,) chi^2.
@jax.jit
def _clipy_data_chi2_batched(cls2d_B, nu_B):
    return jax.vmap(lambda c, n: plf.data_chi2(c, n))(cls2d_B, nu_B)


def clipy_penalized(cls_clik, nu):
    """(n,6,Lcol) cached clik blocks + (n,21) nuisances -> (n,) PENALISED high-ell
    chi^2 = clipy bare data chi^2 + Gaussian/joint-SZ priors. Evaluated in fixed-size
    CLIPY_CHUNK blocks (padded -> one compile, memory-safe)."""
    n = len(nu)
    out = np.empty(n)
    for s in range(0, n, CLIPY_CHUNK):
        cb = cls_clik[s:s + CLIPY_CHUNK]
        nb = nu[s:s + CLIPY_CHUNK]
        m = len(nb)
        if m < CLIPY_CHUNK:                     # pad last chunk -> fixed aval
            cb = np.concatenate([cb, np.repeat(cb[-1:], CLIPY_CHUNK - m, axis=0)], 0)
            nb = np.concatenate([nb, np.repeat(nb[-1:], CLIPY_CHUNK - m, axis=0)], 0)
        dat = np.asarray(_clipy_data_chi2_batched(jnp.asarray(cb), jnp.asarray(nb)),
                         dtype=float)[:m]
        out[s:s + m] = dat + prior_penalty_np(nu[s:s + m])
    return np.where(np.isfinite(out), out, CHI2_INF)


def build_cosmo_dict(theta_cosmo):
    """cosmo subvector (Dc, in COSMO_ORDER) -> ABCMB param dict (host floats)."""
    p = dict(FIXED)
    for i, name in enumerate(COSMO_ORDER):
        if name == "ln10As":
            p["A_s"] = float(np.exp(theta_cosmo[i]) / 1e10)
        else:
            p[name] = float(theta_cosmo[i])
    return p


def eval_theory_chunked(cosmo_dicts):
    """list of ABCMB param dicts -> (cls_clik (n,6,Lcol) np, lowl (n,) np).
    The expensive (SLOW) stage: one Model.call_batched per EVAL_CHUNK, converted to
    the clik C_l block (for clipy) and to low-ell D_ell -> low-ell chi^2 (calib=1)."""
    n = len(cosmo_dicts)
    Lcol = plf.Lcol
    cls_clik = np.empty((n, 6, Lcol))
    lowl = np.zeros(n)
    for s in range(0, n, EVAL_CHUNK):
        blk = cosmo_dicts[s:s + EVAL_CHUNK]
        nb = len(blk)
        if nb < EVAL_CHUNK:
            blk = blk + [blk[-1]] * (EVAL_CHUNK - nb)
        out = model.call_batched(blk, shard=DO_SHARD)
        cb = plf.abcmb_cls_to_clik(out.ClTT, out.ClTE, out.ClEE, out.l)  # (EVAL_CHUNK,6,Lcol)
        cls_clik[s:s + nb] = np.asarray(cb[:nb], dtype=float)
        if lowee is not None or lowtt is not None:
            Dtt = pll.abcmb_cl_to_Dl(out.ClTT, out.l)
            Dee = pll.abcmb_cl_to_Dl(out.ClEE, out.l)
            ll = np.zeros(EVAL_CHUNK)
            if lowee is not None:
                ll = ll + np.asarray(lowee.chi2(Dee), dtype=float)
            if lowtt is not None:
                ll = ll + np.asarray(lowtt.chi2(Dtt), dtype=float)
            lowl[s:s + nb] = ll[:nb]
    return cls_clik, lowl


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"]
                   for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


# ======================================================================
# prior box: cosmo (CEN +- PRIOR_NSIG*SIG, tau floored) + nuisance bounds/box
# ======================================================================
def _recenter_path():
    """Per-config recenter override: scan/results/smc_recenter_<config>.npz. If present,
    the cosmo flat-box is centered on its joint MLE `cen` with per-dim half-width `nsig`
    (sigma). This fixes the degeneracy-truncation that a STATIC LCDM-centered box causes
    once a correlated param (Neff) shifts + broadens the posterior: the box minimum no
    longer rails, so no marginal tail is clipped. Keyed by config so it auto-applies to
    the Neff run (lcdm_neff_plikfull) WITHOUT touching the LCDM run (no file -> legacy)."""
    cfg = os.path.splitext(os.path.basename(CONFIG_ABS))[0]
    return os.path.join(_HERE, "results", f"smc_recenter_{cfg}.npz")


def _prior_box():
    cen_c = CEN_C.copy(); nsig_c = np.full(Dc, PRIOR_NSIG)
    rc = _recenter_path()
    if os.path.exists(rc):
        try:
            d = np.load(rc, allow_pickle=True)
            if [str(x) for x in d["order"]] == COSMO_ORDER:
                cen_c = np.asarray(d["cen"], float)
                ns = np.asarray(d["nsig"], float)
                nsig_c = np.full(Dc, float(ns)) if ns.ndim == 0 else ns
                print(f"[smc-pf] PRIOR BOX RECENTERED on joint MLE ({rc}): cen shift "
                      f"{np.array2string((cen_c - CEN_C) / SIG_C, precision=2)} sigma_prior; "
                      f"box half-width nsig={np.array2string(nsig_c, precision=1)} -- "
                      f"prevents the Neff-degeneracy marginal truncation.", flush=True)
            else:
                print(f"[smc-pf] recenter file {rc} order mismatch -> legacy box", flush=True)
        except Exception as ex:
            print(f"[smc-pf] recenter load failed ({ex}) -> legacy box", flush=True)
    lo_c = cen_c - nsig_c * SIG_C
    hi_c = cen_c + nsig_c * SIG_C
    if "tau_reion" in COSMO_ORDER:
        ti = COSMO_ORDER.index("tau_reion")
        lo_c[ti] = max(lo_c[ti], TAU_FLOOR)
    # nuisance flat box: the Planck uniform bounds where finite, else start +- nsig*scale
    lo_n = np.where(np.isfinite(NU_LO), NU_LO, NU_START - NUIS_BOX_NSIG * NU_SCALE)
    hi_n = np.where(np.isfinite(NU_HI), NU_HI, NU_START + NUIS_BOX_NSIG * NU_SCALE)
    return np.concatenate([lo_c, lo_n]), np.concatenate([hi_c, hi_n])


LO, HI = _prior_box()


def in_box(theta):
    """theta (...,D) -> (...) bool, True if inside the flat prior box."""
    return np.all((theta >= LO) & (theta <= HI), axis=-1)


# ======================================================================
# full chi^2 (tempered likelihood) from scratch for a set of particles
# ======================================================================
def eval_particles(thetas):
    """thetas (M,D) -> (cls_clik (M,6,Lcol), lowl (M,), highell_pen (M,), chi2 (M,)).
    Out-of-box particles short-circuit to chi2=+inf (no theory eval). Used at INIT
    and on RESUME (recompute caches from saved particles)."""
    thetas = np.asarray(thetas, float)
    M = thetas.shape[0]
    Lcol = plf.Lcol
    cls_clik = np.zeros((M, 6, Lcol))
    lowl = np.full(M, CHI2_INF)
    highell = np.full(M, CHI2_INF)
    chi2 = np.full(M, np.inf)
    inside = in_box(thetas)
    idx = np.where(inside)[0]
    if len(idx) == 0:
        return cls_clik, lowl, highell, chi2
    cosmo_dicts = [build_cosmo_dict(thetas[b, :Dc]) for b in idx]
    cl_i, ll_i = eval_theory_chunked(cosmo_dicts)
    he_i = clipy_penalized(cl_i, thetas[idx, Dc:])
    cls_clik[idx] = cl_i
    lowl[idx] = ll_i
    highell[idx] = he_i
    c = he_i + ll_i
    chi2[idx] = np.where(np.isfinite(c), c, CHI2_INF)
    return cls_clik, lowl, highell, chi2


# ======================================================================
# SMC helpers (numpy; identical to scan/smc.py)
# ======================================================================
def logsumexp(logw):
    m = np.max(logw)
    if not np.isfinite(m):
        return -np.inf
    return m + np.log(np.sum(np.exp(logw - m)))


def ess_of_logw(logw):
    m = np.max(logw)
    w = np.exp(logw - m)
    s = w.sum()
    if s <= 0:
        return 0.0
    wn = w / s
    return 1.0 / np.sum(wn ** 2)


def normalized_weights(logw):
    m = np.max(logw)
    w = np.exp(logw - m)
    return w / w.sum()


def systematic_resample(weights, rng):
    Np = len(weights)
    positions = (rng.random() + np.arange(Np)) / Np
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    idx = np.searchsorted(cumsum, positions)
    return np.clip(idx, 0, Np - 1)


def find_delta_beta(chi2, beta, logW, ess_target_frac):
    """Bisection for delta-beta so the ESS of logW + (-delta_beta*chi2/2) ~ target."""
    Np = len(chi2)
    target = ess_target_frac * Np
    db_max = 1.0 - beta
    if db_max <= 0:
        return 0.0
    if ess_of_logw(logW - db_max * chi2 / 2.0) >= target:
        return db_max
    lo, hi = 0.0, db_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        e = ess_of_logw(logW - mid * chi2 / 2.0)
        if e < target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def weighted_cov(particles, weights):
    mean = np.average(particles, axis=0, weights=weights)
    dev = particles - mean
    cov = (dev * weights[:, None]).T @ dev
    cov = cov / (1.0 - np.sum(weights ** 2))
    return mean, cov


def _chol(prop_cov):
    try:
        return np.linalg.cholesky(prop_cov)
    except np.linalg.LinAlgError:
        return np.diag(np.sqrt(np.maximum(np.diag(prop_cov), 1e-16)))


# ======================================================================
# state persistence (resumable). Caches (cls_clik/lowl/highell) are RECOMPUTED on
# resume from the saved particles -> the state file stays small.
# ======================================================================
def _pack_rng(bg_state):
    import pickle
    return pickle.dumps(bg_state)


def _unpack_rng(buf):
    import pickle
    return pickle.loads(buf.tobytes())


def save_state(path, particles, chi2, logW, beta, logZ, rng, stage, trace, n_evals,
               wall_start, scale_c, scale_n):
    bg = rng.bit_generator.state
    np.savez(path,
             particles=particles, chi2=chi2, logW=logW, beta=beta, logZ=logZ,
             stage=stage, n_evals=n_evals, elapsed=time.perf_counter() - wall_start,
             scale_c=scale_c, scale_n=scale_n,
             # trace columns: stage, beta, dbeta, ess_pre, ess_post, acc_slow, acc_fast, logZ
             trace=np.array(trace, dtype=float) if trace else np.zeros((0, 8)),
             rng_state=np.frombuffer(_pack_rng(bg), dtype=np.uint8),
             order=np.array(ORDER), lo=LO, hi=HI,
             cosmo_order=np.array(COSMO_ORDER), nuis_order=np.array(NUIS_ORDER),
             center_c=CEN_C, sigma_c=SIG_C,
             N=N, moves_slow=MOVES_SLOW, moves_fast=MOVES_FAST, ess_target=ESS_TARGET,
             lmax=LMAX, seed=SEED, rtol=RTOL, use_lowtt=USE_LOWTT, use_lowee=USE_LOWEE,
             nuisances_marginalized=True, config=CONFIG_ABS, done=False)


def load_state(path):
    d = np.load(path, allow_pickle=True)
    rng = np.random.default_rng()
    rng.bit_generator.state = _unpack_rng(d["rng_state"])
    trace = [list(r) for r in np.atleast_2d(d["trace"])] if d["trace"].size else []
    return dict(
        particles=np.asarray(d["particles"]), chi2=np.asarray(d["chi2"]),
        logW=np.asarray(d["logW"]), beta=float(d["beta"]), logZ=float(d["logZ"]),
        stage=int(d["stage"]), n_evals=int(d["n_evals"]), elapsed=float(d["elapsed"]),
        scale_c=float(d["scale_c"]) if "scale_c" in d else 1.0,
        scale_n=float(d["scale_n"]) if "scale_n" in d else 1.0,
        trace=trace, rng=rng)


# ======================================================================
# the fast/slow SMC loop
# ======================================================================
def run_smc():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wall_start = time.perf_counter()

    resume_from = RESUME_PATH or (STATE_NPZ if os.path.exists(STATE_NPZ) else "")
    if resume_from and os.path.exists(resume_from):
        st = load_state(resume_from)
        particles = st["particles"]
        beta = st["beta"]; logZ = st["logZ"]; stage = st["stage"]
        n_evals = st["n_evals"]; trace = st["trace"]; rng = st["rng"]
        prev_elapsed = st["elapsed"]; logW = st["logW"]
        scale_c = st["scale_c"]; scale_n = st["scale_n"]
        print(f"[smc-pf] RESUME from {resume_from}: stage={stage} beta={beta:.4f} "
              f"logZ={logZ:.3f} n_evals={n_evals} (prev wall {prev_elapsed:.0f}s); "
              f"recomputing per-particle Cls caches ...", flush=True)
        cls_clik, lowl, highell, chi2_re = eval_particles(particles)
        n_evals += N
        # trust the recomputed chi2 (bit-stable theory) but keep the saved logW/beta
        chi2 = np.where(np.isfinite(chi2_re), chi2_re, CHI2_INF)
    else:
        rng = np.random.default_rng(SEED)
        particles = LO + (HI - LO) * rng.random((N, D))
        print(f"[smc-pf] INIT N={N} D={D} (cosmo {Dc} + nuis {Dn}) from flat box; "
              f"evaluating chi2 ...", flush=True)
        cls_clik, lowl, highell, chi2 = eval_particles(particles)
        n_evals = N
        beta = 0.0; logZ = 0.0; stage = 0; trace = []; prev_elapsed = 0.0
        logW = np.zeros(N)
        scale_c = 1.0; scale_n = 1.0
        bad = ~np.isfinite(chi2)
        if bad.any():
            print(f"[smc-pf] WARN {bad.sum()} init particles non-finite chi2; "
                  f"clamped to {CHI2_INF}", flush=True)
            chi2 = np.where(np.isfinite(chi2), chi2, CHI2_INF)
        print(f"[smc-pf] init chi2: min={chi2.min():.2f} med={np.median(chi2):.2f} "
              f"max={chi2.max():.2f}", flush=True)
        print(f"[smc-pf] per-device GPU peak after init eval: {peak_gb():.2f} GB "
              f"(EVAL_CHUNK={EVAL_CHUNK}, CLIPY_CHUNK={CLIPY_CHUNK})", flush=True)
        save_state(STATE_NPZ, particles, chi2, logW, beta, logZ, rng, stage, trace,
                   n_evals, wall_start, scale_c, scale_n)

    while beta < 1.0 - 1e-12 and stage < MAXSTAGES:
        stage += 1
        # ---- adaptive delta-beta (ESS of updated cumulative weights ~ target) ----
        db = find_delta_beta(chi2, beta, logW, ESS_TARGET)
        # pre-emptive resample when the schedule stalls on degenerate logW (smc.py fix)
        if db < 1e-8 and ess_of_logw(logW) < N - 1e-9:
            anc = systematic_resample(normalized_weights(logW), rng)
            particles = particles[anc].copy(); chi2 = chi2[anc].copy()
            cls_clik = cls_clik[anc].copy(); lowl = lowl[anc].copy()
            highell = highell[anc].copy(); logW = np.zeros(N)
            db = find_delta_beta(chi2, beta, logW, ESS_TARGET)
            print(f"[smc-pf] stage {stage}: pre-emptive resample (degenerate logW); "
                  f"re-found dbeta={db:.4e}", flush=True)
        if db <= 0:
            print(f"[smc-pf] stage {stage}: dbeta collapsed at beta={beta}; force->1",
                  flush=True)
            db = 1.0 - beta
        logw_incr = -db * chi2 / 2.0
        logZ += logsumexp(logW + logw_incr) - logsumexp(logW)
        logW = logW + logw_incr
        beta = min(beta + db, 1.0)
        w = normalized_weights(logW)
        ess_pre = ess_of_logw(logW)

        # ---- systematic resampling if ESS < target (reshuffle ALL caches) ----
        ess_post = ess_pre
        if ess_pre < ESS_TARGET * N:
            anc = systematic_resample(w, rng)
            particles = particles[anc].copy(); chi2 = chi2[anc].copy()
            cls_clik = cls_clik[anc].copy(); lowl = lowl[anc].copy()
            highell = highell[anc].copy()
            w = np.full(N, 1.0 / N); logW = np.zeros(N); ess_post = float(N)

        # ---- proposal covariances: separate cosmo (slow) and nuisance (fast) blocks ----
        _, cov = weighted_cov(particles, w)
        cov_c = cov[:Dc, :Dc] + 1e-8 * np.eye(Dc)
        cov_n = cov[Dc:, Dc:] + np.diag((1e-2 * NU_SCALE) ** 2) + 1e-12 * np.eye(Dn)
        Lc = _chol((2.38 ** 2 / Dc) * cov_c * scale_c ** 2)
        Ln = _chol((2.38 ** 2 / Dn) * cov_n * scale_n ** 2)

        # ---- SLOW block: cosmo moves (each = ONE batched theory solve) ----
        acc_slow = 0
        for _m in range(MOVES_SLOW):
            prop = particles.copy()
            prop[:, :Dc] += rng.standard_normal((N, Dc)) @ Lc.T
            inside = in_box(prop)
            idx = np.where(inside)[0]
            chi2_prop = np.full(N, np.inf)
            if len(idx):
                cosmo_dicts = [build_cosmo_dict(prop[b, :Dc]) for b in idx]
                cl_i, ll_i = eval_theory_chunked(cosmo_dicts)
                he_i = clipy_penalized(cl_i, prop[idx, Dc:])
                c = he_i + ll_i
                chi2_prop[idx] = np.where(np.isfinite(c), c, CHI2_INF)
                n_evals += len(idx)
            log_alpha = -0.5 * beta * (chi2_prop - chi2)
            accept = (np.log(rng.random(N)) < log_alpha) & np.isfinite(chi2_prop) & inside
            if accept.any():
                aidx = np.where(accept)[0]
                # map accepted global rows back to their position in idx for caches
                pos = {b: j for j, b in enumerate(idx)}
                jsel = np.array([pos[b] for b in aidx])
                particles[aidx, :Dc] = prop[aidx, :Dc]
                cls_clik[aidx] = cl_i[jsel]
                lowl[aidx] = ll_i[jsel]
                highell[aidx] = he_i[jsel]
                chi2[aidx] = chi2_prop[aidx]
            acc_slow += int(accept.sum())
        acc_slow_rate = acc_slow / (MOVES_SLOW * N)

        # ---- FAST block: nuisance moves (cosmo fixed -> REUSE Cls; clipy only) ----
        acc_fast = 0
        for _m in range(MOVES_FAST):
            nu_prop = particles[:, Dc:] + rng.standard_normal((N, Dn)) @ Ln.T
            prop = particles.copy(); prop[:, Dc:] = nu_prop
            inside = in_box(prop)
            he_prop = np.where(inside, clipy_penalized(cls_clik, nu_prop), CHI2_INF)
            chi2_prop = np.where(inside, he_prop + lowl, np.inf)
            log_alpha = -0.5 * beta * (chi2_prop - chi2)
            accept = (np.log(rng.random(N)) < log_alpha) & np.isfinite(chi2_prop) & inside
            particles[accept, Dc:] = nu_prop[accept]
            highell[accept] = he_prop[accept]
            chi2[accept] = chi2_prop[accept]
            acc_fast += int(accept.sum())
        acc_fast_rate = acc_fast / (MOVES_FAST * N)

        trace.append([stage, beta, db, ess_pre, ess_post, acc_slow_rate,
                      acc_fast_rate, logZ])
        print(f"[smc-pf] stage {stage:3d}: beta={beta:.5f} dbeta={db:.4e} "
              f"ESS_pre={ess_pre:.0f} ESS_post={ess_post:.0f} "
              f"acc_slow={acc_slow_rate:.3f} acc_fast={acc_fast_rate:.3f} "
              f"logZ={logZ:.3f} chi2min={chi2.min():.2f} "
              f"sc_c={scale_c:.2f} sc_n={scale_n:.2f} nev={n_evals} "
              f"({time.strftime('%H:%M:%S')}, {time.perf_counter()-wall_start:.0f}s)",
              flush=True)

        # ---- adapt each block's proposal scale toward a healthy acceptance ----
        if acc_slow_rate < 0.1:
            scale_c *= 0.5
        elif acc_slow_rate > 0.6:
            scale_c = min(scale_c * 1.5, 4.0)
        if acc_fast_rate < 0.1:
            scale_n *= 0.5
        elif acc_fast_rate > 0.6:
            scale_n = min(scale_n * 1.5, 4.0)

        save_state(STATE_NPZ, particles, chi2, logW, beta, logZ, rng, stage, trace,
                   n_evals, wall_start, scale_c, scale_n)

    # ======================================================================
    # finalize
    # ======================================================================
    weights = normalized_weights(logW)
    weights_uniform = bool(np.allclose(weights, 1.0 / N))
    marg_mean = np.average(particles, axis=0, weights=weights)
    marg_std = np.sqrt(np.average((particles - marg_mean) ** 2, axis=0, weights=weights))
    elapsed = prev_elapsed + (time.perf_counter() - wall_start)
    tr = np.array(trace, dtype=float) if trace else np.zeros((0, 8))

    print(f"\n[smc-pf] DONE: {stage} stages, beta={beta:.5f}, logZ={logZ:.4f}, "
          f"{n_evals} theory-evals, wall {elapsed:.0f}s", flush=True)
    print("[smc-pf] marginal posterior (mean +- std):", flush=True)
    for i, name in enumerate(ORDER):
        tag = "cosmo" if i < Dc else "nuis"
        print(f"    [{tag:5s}] {name:18s} {marg_mean[i]:.6g} +- {marg_std[i]:.4g}",
              flush=True)

    final = OUT + ".npz"
    np.savez(final,
             particles=particles, weights=weights, chi2=chi2,
             beta=beta, logZ=logZ, n_stages=stage, n_evals=n_evals,
             wall_seconds=elapsed,
             trace=tr,
             trace_cols=np.array(["stage", "beta", "dbeta", "ess_pre", "ess_post",
                                  "acc_slow", "acc_fast", "logZ"]),
             marg_mean=marg_mean, marg_std=marg_std,
             order=np.array(ORDER), cosmo_order=np.array(COSMO_ORDER),
             nuis_order=np.array(NUIS_ORDER), Dc=Dc, Dn=Dn,
             prior_lo=LO, prior_hi=HI, center_c=CEN_C, sigma_c=SIG_C,
             prior_nsig=PRIOR_NSIG, nuis_box_nsig=NUIS_BOX_NSIG,
             N=N, moves_slow=MOVES_SLOW, moves_fast=MOVES_FAST,
             ess_target=ESS_TARGET, lmax=LMAX, seed=SEED, rtol=RTOL,
             use_lowtt=USE_LOWTT, use_lowee=USE_LOWEE,
             nuisances_marginalized=True,       # the 21 plik nuisances are SAMPLED
             final_weights_uniform=weights_uniform,
             config=CONFIG_ABS, done=True)
    print(f"[smc-pf] wrote {final}", flush=True)
    return final


def main():
    print(f"[smc-pf] devices={jax.devices()} config={CONFIG_ABS}", flush=True)
    print(f"[smc-pf] N={N} D={D} (cosmo {Dc}: {COSMO_ORDER}; nuis {Dn}) "
          f"MOVES_SLOW={MOVES_SLOW} MOVES_FAST={MOVES_FAST} ESS_TARGET={ESS_TARGET} "
          f"LMAX={LMAX} rtol={RTOL} shard={DO_SHARD} lowTT={USE_LOWTT} "
          f"lowEE={USE_LOWEE} seed={SEED} EVAL_CHUNK={EVAL_CHUNK} "
          f"CLIPY_CHUNK={CLIPY_CHUNK}", flush=True)
    print(f"[smc-pf] cosmo box (CEN +- {PRIOR_NSIG} sigma), nuisance box "
          f"(bounds or start +- {NUIS_BOX_NSIG}*scale):", flush=True)
    for i, name in enumerate(ORDER):
        tag = "cosmo" if i < Dc else "nuis"
        print(f"    [{tag:5s}] {name:18s} [{LO[i]:.6g}, {HI[i]:.6g}]", flush=True)
    run_smc()


if __name__ == "__main__":
    main()

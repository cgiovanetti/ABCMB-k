"""smc.py — hand-rolled tempered Sequential Monte Carlo (SMC) on the batched
ABCMB likelihood (TOOL_PLAN.md Workstream D — the Bayesian anchor).

Gives the Bayesian posterior and the evidence (logZ) on the same data model the
frequentist profiles (scan/profile_prod_ad.py) use: plik-lite high-ell TTTEEE
(A_planck envelope-profiled, exactly as the driver does) + real low-ell TT
(Commander) + low-ell EE (SRoll2). log-likelihood = -chi2/2.

Why eager Python (not blackjax): the ABCMB pipeline is GPU -> CPU HyRex -> GPU and
is not end-to-end jittable (jitting the whole pipeline is the known ~20-min monolith
hang, TOOL_PLAN.md gotcha #1). So the SMC loop is hand-rolled in numpy around
`Model.call_batched` -- every likelihood evaluation of a population of particles is
one batched call; the pipeline is never wrapped in jax.jit/vmap here.

A_planck caveat: A_planck is profiled (envelope theorem, stop_gradient -- same as the
driver), not marginalized. This is acceptable given its tight N(1, 0.0025) prior; the
profiled and marginalized posteriors coincide to O(sigma_A^2) for a calibration
nuisance this tightly constrained. Noted in the npz (a_planck_profiled=True) and the
CHANGELOG.

Algorithm (tempered SMC, pi_beta ∝ prior x L^beta, beta: 0 -> 1):
  * N particles init from the flat prior box (CEN +- 5*SIG per param, tau floored
    at 0.01). N stays fixed the whole run -> one compile of the call_batched B aval.
  * adaptive beta ladder: each stage finds delta-beta by bisection so the ESS of the
    updated cumulative weights logW + (-delta_beta*chi2/2) is ~ESS_TARGET*N; beta
    accumulates to exactly 1.0 at the end. logW carries the running importance
    weights so the schedule + evidence stay correct even when a stage skips
    resampling (logW non-uniform). Incremental log-weight = -delta_beta*chi2_i/2.
  * logZ += logsumexp(logW+logw_incr) - logsumexp(logW) at each stage (the
    normalizer ratio; reduces to logmeanexp(logw_incr) right after a resample).
  * systematic resampling when ESS(logW) < ESS_TARGET*N (then logW back to uniform).
  * M random-walk Metropolis moves per stage targeting pi_beta; proposal covariance
    = 2.38^2/D * weighted particle covariance (recomputed each stage; +1e-8 diagonal
    jitter). Each move = ONE batched chi2 eval of all N proposals; accept/reject
    per-particle in numpy. Proposals outside the prior box -> reject (chi2=+inf).
    Acceptance rate logged per stage (healthy 0.15-0.5); if <0.1, the proposal is
    scaled down by 2 for the next stage.
  * State (particles, chi2, beta, logZ, rng) persisted to npz every stage ->
    resumable via SMC_RESUME=<path>.

Env knobs (all optional):
  SMC_N(512) SMC_MOVES(3) SMC_ESS_TARGET(0.5) SMC_LMAX(2508) SMC_SEED(0)
  SMC_OUT(scan/results/smc_lcdm) SMC_CONFIG(scan/configs/lcdm.py)
  SMC_RTOL(1e-5) SMC_MAXSTAGES(80) SMC_RESUME(path; auto-detects SMC_OUT_state.npz)
  SMC_USE_LOWTT(cfg) SMC_USE_LOWEE(cfg) SMC_SHARD(auto|0|1)

Run via srun, PYTHONPATH=$(pwd), JAX_COMPILATION_CACHE_DIR set (see scan/smc_lcdm.slurm).
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

_HERE = os.path.dirname(os.path.abspath(__file__))
CHI2_INF = 1e6                      # finite stand-in for out-of-box / NaN chi2


# ======================================================================
# config loading (same scheme as profile_prod_ad.py: PA_CONFIG-style dict)
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


CONFIG_PATH = os.environ.get("SMC_CONFIG", "scan/configs/lcdm.py")
CFG, CONFIG_ABS = _load_config(CONFIG_PATH)

ORDER = list(CFG["order"])
D = len(ORDER)
CENTER = np.array([CFG["cen"][k] for k in ORDER])
SIGMA = np.array([CFG["sig"][k] for k in ORDER])
FIXED = dict(CFG["fixed"])
USER_SPECIES = CFG.get("user_species", None)

# ---------------- env config ----------------
N = int(os.environ.get("SMC_N", 512))
# Per-call_batched evaluation chunk (cosmologies across ALL devices). The full
# population N is evaluated EVAL_CHUNK at a time so the per-device working set
# (~0.36 GB/cosmo-local peak at l2508+lensing, MEASURED from the 54369057 OOM)
# stays well under device memory. Default 128 -> B_local=32 on a 4-GPU node
# (~12 GB/device), safe on the 40 GB A100s the regular queue lands on (the
# "B~512/node" guidance in CLAUDE.md was measured on 80 GB nodes). Memory per
# call is set by EVAL_CHUNK/n_dev, NOT by N, so this is decoupled from the
# particle count. Each chunk is padded to EVAL_CHUNK -> ONE compile, reused.
EVAL_CHUNK = int(os.environ.get("SMC_EVAL_CHUNK", 128))
MOVES = int(os.environ.get("SMC_MOVES", 3))
ESS_TARGET = float(os.environ.get("SMC_ESS_TARGET", 0.5))
LMAX = int(os.environ.get("SMC_LMAX", 2508))
SEED = int(os.environ.get("SMC_SEED", 0))
OUT = os.environ.get("SMC_OUT", os.path.join(_HERE, "results", "smc_lcdm"))
RTOL = float(os.environ.get("SMC_RTOL", 1e-5))
MAXSTAGES = int(os.environ.get("SMC_MAXSTAGES", 80))
PRIOR_NSIG = float(os.environ.get("SMC_PRIOR_NSIG", 5.0))   # flat box half-width (sigma)
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
# prior box: flat CEN +- PRIOR_NSIG*SIG per param, tau floored
# ======================================================================
def _prior_box():
    lo = CENTER - PRIOR_NSIG * SIGMA
    hi = CENTER + PRIOR_NSIG * SIGMA
    if "tau_reion" in ORDER:
        ti = ORDER.index("tau_reion")
        lo[ti] = max(lo[ti], TAU_FLOOR)
    return lo, hi


LO, HI = _prior_box()


def in_box(theta):
    """theta (...,D) -> (...) bool, True if inside the flat prior box."""
    return np.all((theta >= LO) & (theta <= HI), axis=-1)


# ======================================================================
# likelihood machinery (MIRRORS profile_prod_ad.py; no driver import)
# ======================================================================
pl = PlikLite()
lowee = LowLEE() if USE_LOWEE else None
lowtt = LowLTT() if USE_LOWTT else None
model = Model(user_species=USER_SPECIES, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
              rtol_large_k_PE=RTOL, atol_large_k_PE=RTOL * 1e-2,
              rtol_small_k_PE=min(1e-5, RTOL), max_steps_PE=16384)


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


def _chi2_from_out(out):
    """BatchedOutput -> (B,) total chi^2 (A_planck-profiled plik-lite + low-ell)."""
    Dtt = pl.abcmb_cl_to_Dl(out.ClTT, out.l)
    Dte = pl.abcmb_cl_to_Dl(out.ClTE, out.l)
    Dee = pl.abcmb_cl_to_Dl(out.ClEE, out.l)
    m0 = pl.bin_model(Dtt, Dte, Dee)
    chi2 = np.asarray(pl.profile_A(m0, with_prior=True)[0], dtype=float)
    if lowee is not None:
        chi2 = chi2 + np.asarray(lowee.chi2(Dee), dtype=float)
    if lowtt is not None:
        chi2 = chi2 + np.asarray(lowtt.chi2(Dtt), dtype=float)
    return np.where(np.isfinite(chi2), chi2, CHI2_INF)


def peak_gb():
    """Max per-device GPU peak (GB) since reset; nan on CPU-only."""
    try:
        return max(d.memory_stats()["peak_bytes_in_use"]
                   for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def _eval_chunked(dicts):
    """Evaluate a list of param dicts through call_batched in fixed-size,
    memory-safe chunks. Returns a (len(dicts),) chi2 array.

    The per-device memory of one call_batched is set by (chunk size / n_dev),
    so chunking caps the working set independent of how many particles N we
    carry. Each chunk is PADDED to EVAL_CHUNK before the call so the B aval is
    constant -> ONE compile, reused across chunks AND across SMC stages."""
    n = len(dicts)
    out_chi2 = np.empty(n)
    for s in range(0, n, EVAL_CHUNK):
        blk = dicts[s:s + EVAL_CHUNK]
        nb = len(blk)
        if nb < EVAL_CHUNK:                       # pad last chunk -> fixed aval
            blk = blk + [blk[-1]] * (EVAL_CHUNK - nb)
        out = model.call_batched(blk, shard=DO_SHARD)
        out_chi2[s:s + nb] = _chi2_from_out(out)[:nb]
    return out_chi2


def chi2_batch(thetas):
    """thetas (M,D) physical -> (M,) total chi^2. Particles OUTSIDE the prior box
    are short-circuited to +inf (no theory eval); in-box rows are evaluated via
    call_batched in EVAL_CHUNK-sized chunks (memory-safe; see _eval_chunked)."""
    thetas = np.asarray(thetas, float)
    M = thetas.shape[0]
    chi2 = np.full(M, np.inf)
    inside = in_box(thetas)
    idx = np.where(inside)[0]
    if len(idx) == 0:
        return chi2
    sub = [build_dict(thetas[b]) for b in idx]
    chi2[idx] = _eval_chunked(sub)
    return chi2


# ======================================================================
# SMC helpers (numpy)
# ======================================================================
def logsumexp(logw):
    m = np.max(logw)
    if not np.isfinite(m):
        return -np.inf
    return m + np.log(np.sum(np.exp(logw - m)))


def logmeanexp(logw):
    m = np.max(logw)
    if not np.isfinite(m):
        return -np.inf
    return m + np.log(np.mean(np.exp(logw - m)))


def ess_of_logw(logw):
    """effective sample size of (unnormalized log) incremental weights."""
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
    """systematic resampling -> ancestor indices (N,)."""
    Np = len(weights)
    positions = (rng.random() + np.arange(Np)) / Np
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    idx = np.searchsorted(cumsum, positions)
    return np.clip(idx, 0, Np - 1)


def find_delta_beta(chi2, beta, logW, ess_target_frac):
    """Bisection for delta-beta in (0, 1-beta] so the ESS of the UPDATED cumulative
    weights logW + (-delta_beta*chi2/2) is ~ ess_target_frac*N. Carrying logW (the
    pre-stage cumulative log importance weights) makes the temper schedule correct
    even when a stage skipped resampling (logW non-uniform). Returns delta-beta."""
    Np = len(chi2)
    target = ess_target_frac * Np
    db_max = 1.0 - beta
    if db_max <= 0:
        return 0.0
    # ESS at full remaining step; if it's already >= target, take the whole rest.
    if ess_of_logw(logW - db_max * chi2 / 2.0) >= target:
        return db_max
    lo, hi = 0.0, db_max
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        e = ess_of_logw(logW - mid * chi2 / 2.0)
        if e < target:        # too aggressive, shrink
            hi = mid
        else:                 # room to push
            lo = mid
    return 0.5 * (lo + hi)


def weighted_cov(particles, weights):
    """weighted mean/cov of particles (N,D) with normalized weights (N,)."""
    mean = np.average(particles, axis=0, weights=weights)
    dev = particles - mean
    cov = (dev * weights[:, None]).T @ dev
    cov = cov / (1.0 - np.sum(weights ** 2))     # unbiased for normalized weights
    return mean, cov


# ======================================================================
# state persistence (resumable)
# ======================================================================
def save_state(path, particles, chi2, logW, beta, logZ, rng, stage, trace, n_evals,
               wall_start):
    bg = rng.bit_generator.state
    np.savez(path,
             particles=particles, chi2=chi2, logW=logW, beta=beta, logZ=logZ,
             stage=stage, n_evals=n_evals, elapsed=time.perf_counter() - wall_start,
             # trace columns: stage, beta, delta_beta, ess_pre, ess_post, acc, logZ
             trace=np.array(trace, dtype=float) if trace else np.zeros((0, 7)),
             rng_state=np.frombuffer(_pack_rng(bg), dtype=np.uint8),
             order=np.array(ORDER), lo=LO, hi=HI, center=CENTER, sigma=SIGMA,
             N=N, moves=MOVES, ess_target=ESS_TARGET, lmax=LMAX, seed=SEED,
             rtol=RTOL, use_lowtt=USE_LOWTT, use_lowee=USE_LOWEE,
             a_planck_profiled=True, config=CONFIG_ABS, done=False)


def _pack_rng(bg_state):
    import pickle
    return pickle.dumps(bg_state)


def _unpack_rng(buf):
    import pickle
    return pickle.loads(buf.tobytes())


def load_state(path):
    d = np.load(path, allow_pickle=True)
    rng = np.random.default_rng()
    rng.bit_generator.state = _unpack_rng(d["rng_state"])
    trace = [list(r) for r in np.atleast_2d(d["trace"])] if d["trace"].size else []
    return dict(
        particles=np.asarray(d["particles"]), chi2=np.asarray(d["chi2"]),
        logW=np.asarray(d["logW"]), beta=float(d["beta"]), logZ=float(d["logZ"]),
        stage=int(d["stage"]), n_evals=int(d["n_evals"]),
        elapsed=float(d["elapsed"]), trace=trace, rng=rng)


# ======================================================================
# the SMC loop
# ======================================================================
def run_smc():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    wall_start = time.perf_counter()

    resume_from = RESUME_PATH or (STATE_NPZ if os.path.exists(STATE_NPZ) else "")
    if resume_from and os.path.exists(resume_from):
        st = load_state(resume_from)
        particles = st["particles"]; chi2 = st["chi2"]
        beta = st["beta"]; logZ = st["logZ"]; stage = st["stage"]
        n_evals = st["n_evals"]; trace = st["trace"]; rng = st["rng"]
        prev_elapsed = st["elapsed"]; logW = st["logW"]
        print(f"[smc] RESUME from {resume_from}: stage={stage} beta={beta:.4f} "
              f"logZ={logZ:.3f} n_evals={n_evals} (prev wall {prev_elapsed:.0f}s)",
              flush=True)
    else:
        rng = np.random.default_rng(SEED)
        # init N particles uniformly in the flat prior box
        particles = LO + (HI - LO) * rng.random((N, D))
        print(f"[smc] INIT N={N} D={D} from prior box; evaluating chi2 ...",
              flush=True)
        chi2 = chi2_batch(particles)
        n_evals = N
        beta = 0.0; logZ = 0.0; stage = 0; trace = []; prev_elapsed = 0.0
        logW = np.zeros(N)                     # cumulative log importance weights
        # guard: replace any +inf (shouldn't happen — all init in-box) with a redraw
        bad = ~np.isfinite(chi2)
        if bad.any():
            print(f"[smc] WARN {bad.sum()} init particles had non-finite chi2; "
                  f"clamped to {CHI2_INF}", flush=True)
            chi2 = np.where(np.isfinite(chi2), chi2, CHI2_INF)
        print(f"[smc] init chi2: min={chi2.min():.2f} med={np.median(chi2):.2f} "
              f"max={chi2.max():.2f}", flush=True)
        print(f"[smc] per-device GPU peak after init eval: {peak_gb():.2f} GB "
              f"(EVAL_CHUNK={EVAL_CHUNK}, B_local={-(-EVAL_CHUNK // max(NDEV,1))})",
              flush=True)
        save_state(STATE_NPZ, particles, chi2, logW, beta, logZ, rng, stage, trace,
                   n_evals, wall_start)

    prop_scale = 1.0   # proposal shrink factor (halved if acceptance < 0.1)

    while beta < 1.0 - 1e-12 and stage < MAXSTAGES:
        stage += 1
        # ---- adaptive delta-beta by bisection (ESS of UPDATED cumulative weights
        #      ~ ESS_TARGET*N). logW carries pre-stage cumulative importance weights
        #      (uniform whenever the previous stage resampled). ----
        db = find_delta_beta(chi2, beta, logW, ESS_TARGET)
        # PRE-EMPTIVE RESAMPLE: when the cumulative weights logW are already
        # degenerate (ESS <= target), the bisection returns db ~ 0 (at mid=0 the ESS
        # is already <= target, so it shrinks toward zero) and beta cannot advance --
        # which is exactly the signal to resample. Reset logW to uniform and re-find a
        # real positive step. WITHOUT this, the schedule advanced beta by numerical-
        # noise db ~1e-16 and burned ~half its stages hovering at a fixed beta until
        # FP drift finally tripped the STRICT post-step resample test (ess_pre < target
        # never fires while ESS sits exactly at target). Efficiency only -- the old
        # runs were correct (db~0 stages add 0 to logZ), just ~2x too slow.
        if db < 1e-8 and ess_of_logw(logW) < N - 1e-9:
            anc = systematic_resample(normalized_weights(logW), rng)
            particles = particles[anc].copy()
            chi2 = chi2[anc].copy()
            logW = np.zeros(N)
            db = find_delta_beta(chi2, beta, logW, ESS_TARGET)
            print(f"[smc] stage {stage}: pre-emptive resample (degenerate logW); "
                  f"re-found dbeta={db:.4e}", flush=True)
        if db <= 0:
            print(f"[smc] stage {stage}: delta-beta collapsed to 0 at beta={beta}; "
                  f"forcing beta->1", flush=True)
            db = 1.0 - beta
        logw_incr = -db * chi2 / 2.0
        # evidence increment = ratio of normalizers (exact under conditional
        # resampling): logZ += logsumexp(logW + logw_incr) - logsumexp(logW).
        logZ += logsumexp(logW + logw_incr) - logsumexp(logW)
        logW = logW + logw_incr
        beta += db
        if beta > 1.0:
            beta = 1.0
        w = normalized_weights(logW)
        ess_pre = ess_of_logw(logW)

        # ---- systematic resampling if ESS < target (then logW back to uniform) ----
        ess_post = ess_pre
        if ess_pre < ESS_TARGET * N:
            anc = systematic_resample(w, rng)
            particles = particles[anc].copy()
            chi2 = chi2[anc].copy()
            w = np.full(N, 1.0 / N)
            logW = np.zeros(N)
            ess_post = float(N)

        # ---- proposal covariance: 2.38^2/D * weighted particle cov ----
        _, cov = weighted_cov(particles, w)
        cov = cov + 1e-8 * np.eye(D)
        prop_cov = (2.38 ** 2 / D) * cov * (prop_scale ** 2)
        try:
            L = np.linalg.cholesky(prop_cov)
        except np.linalg.LinAlgError:
            # fall back to a diagonal proposal if cov is not PD
            L = np.diag(np.sqrt(np.maximum(np.diag(prop_cov), 1e-16)))

        # ---- M random-walk Metropolis moves targeting pi_beta ----
        n_acc_total = 0
        for _m in range(MOVES):
            steps = rng.standard_normal((N, D)) @ L.T
            prop = particles + steps
            chi2_prop = chi2_batch(prop)        # ONE batched eval (out-of-box -> inf)
            n_evals += N
            # log pi_beta ∝ -beta*chi2/2 (flat prior inside the box, -inf outside).
            # out-of-box proposals already carry chi2=+inf -> log-accept = -inf.
            log_alpha = -0.5 * beta * (chi2_prop - chi2)
            log_u = np.log(rng.random(N))
            accept = (log_u < log_alpha) & np.isfinite(chi2_prop)
            particles[accept] = prop[accept]
            chi2[accept] = chi2_prop[accept]
            n_acc_total += int(accept.sum())
        acc_rate = n_acc_total / (MOVES * N)

        trace.append([stage, beta, db, ess_pre, ess_post, acc_rate, logZ])
        print(f"[smc] stage {stage:3d}: beta={beta:.5f} dbeta={db:.4e} "
              f"ESS_pre={ess_pre:.0f} ESS_post={ess_post:.0f} acc={acc_rate:.3f} "
              f"logZ={logZ:.3f} chi2min={chi2.min():.2f} "
              f"scale={prop_scale:.2f} nev={n_evals} "
              f"({time.strftime('%H:%M:%S')}, {time.perf_counter()-wall_start:.0f}s)",
              flush=True)

        # ---- adapt proposal scale if acceptance is unhealthy ----
        if acc_rate < 0.1:
            prop_scale *= 0.5
            print(f"[smc]   acc<0.1 -> shrink proposal scale to {prop_scale:.3f}",
                  flush=True)
        elif acc_rate > 0.6:
            prop_scale = min(prop_scale * 1.5, 4.0)

        # ---- persist state every stage (resumable) ----
        save_state(STATE_NPZ, particles, chi2, logW, beta, logZ, rng, stage, trace,
                   n_evals, wall_start)

    # ======================================================================
    # finalize: particles ~ posterior at the FINAL importance weights. The last
    # stage usually resampled (so logW is uniform), but if it didn't, logW carries
    # the residual weights -> use them so the marginals are unbiased either way.
    # ======================================================================
    weights = normalized_weights(logW)
    weights_uniform = bool(np.allclose(weights, 1.0 / N))
    marg_mean = np.average(particles, axis=0, weights=weights)
    marg_std = np.sqrt(np.average((particles - marg_mean) ** 2, axis=0,
                                  weights=weights))
    elapsed = prev_elapsed + (time.perf_counter() - wall_start)
    tr = np.array(trace, dtype=float) if trace else np.zeros((0, 7))

    print(f"\n[smc] DONE: {stage} stages, beta={beta:.5f}, logZ={logZ:.4f}, "
          f"{n_evals} chi2-evals, wall {elapsed:.0f}s", flush=True)
    print("[smc] marginal posterior (mean +- std), prior CEN, in-sigma-of-CEN:",
          flush=True)
    for i, name in enumerate(ORDER):
        dsig = (marg_mean[i] - CENTER[i]) / SIGMA[i]
        print(f"    {name:12s} {marg_mean[i]:.6g} +- {marg_std[i]:.4g}   "
              f"(CEN {CENTER[i]:.6g}, {dsig:+.2f} prior-sigma)", flush=True)

    final = OUT + ".npz"
    np.savez(final,
             particles=particles, weights=weights, chi2=chi2,
             beta=beta, logZ=logZ, n_stages=stage, n_evals=n_evals,
             wall_seconds=elapsed,
             trace=tr,  # [stage, beta, dbeta, ess_pre, ess_post, acc, logZ]
             trace_cols=np.array(["stage", "beta", "dbeta", "ess_pre",
                                  "ess_post", "acc", "logZ"]),
             marg_mean=marg_mean, marg_std=marg_std,
             order=np.array(ORDER), prior_lo=LO, prior_hi=HI,
             center=CENTER, sigma=SIGMA, prior_nsig=PRIOR_NSIG,
             N=N, moves=MOVES, ess_target=ESS_TARGET, lmax=LMAX, seed=SEED,
             rtol=RTOL, use_lowtt=USE_LOWTT, use_lowee=USE_LOWEE,
             a_planck_profiled=True,   # PROFILED not marginalized (see module doc)
             final_weights_uniform=weights_uniform,
             config=CONFIG_ABS, done=True)
    print(f"[smc] wrote {final}", flush=True)
    # leave the _state.npz in place (harmless; resume sees done in the final npz)
    return final


def main():
    print(f"[smc] devices={jax.devices()} config={CONFIG_ABS} N={N} D={D} "
          f"EVAL_CHUNK={EVAL_CHUNK} MOVES={MOVES} ESS_TARGET={ESS_TARGET} "
          f"LMAX={LMAX} rtol={RTOL} shard={DO_SHARD} lowTT={USE_LOWTT} "
          f"lowEE={USE_LOWEE} seed={SEED}", flush=True)
    print(f"[smc] prior box (CEN +- {PRIOR_NSIG} sigma):", flush=True)
    for i, name in enumerate(ORDER):
        print(f"    {name:12s} [{LO[i]:.6g}, {HI[i]:.6g}]", flush=True)
    run_smc()


if __name__ == "__main__":
    main()

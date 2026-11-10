"""run_act.py — lockstep HMC of ACT DR6 LCDM (massless neutrinos) through ABCMB-k (hmc/posterior_act.py).
The same sampler and schedule as run_lcdm.py (the Planck-lite run); differences: the data, two nuisances
(A_act, P_act), and Gauss-Newton steps from START to the ACT maximum before the metric and the chains.

  HMC_DATA     act (default: ACT DR6 lite + SRoll2 low-ell EE) | pact (+ low-ell TT + plik-lite cut)
  HMC_MODE     sample (default) | echeck (leapfrog energy error at eps 1/2, 1/4, 1/8 over T=1, then exit)
               | probe (two gradient calls of HMC_NCHAINS chains near START: timing and peak memory, then exit)
               | gn (the Gauss-Newton steps and the metric only, then exit)
  HMC_GN_STEPS 2       Gauss-Newton steps theta += Sigma_GN grad logp from START before the metric is taken
  HMC_NCHAINS  32      chains (one batched gradient call per leapfrog step)
  HMC_WARMUP 15, HMC_METRIC_AT 5,10, HMC_NSAMPLES 40, HMC_T 1.0,2.0, HMC_TARGET 0.8, HMC_EPS0 0.8, HMC_SEED 0
  HMC_TAG      ''      output hmc/results/<HMC_DATA><TAG>.npz (checkpoint every trajectory; resumes)
  HMC_AD_CHUNK 64      max cosmologies per staged-AD call
  HMC_SPECB    4       cosmologies per device per spectrum-stage jvp
  HMC_RTOL     1e-5    rtol_large_k_PE
  HMC_MU_CHUNK 0       lensing_mu_chunk (0: the lensing mu sums in one block)
  HMC_DELTA_L  5000    delta_l_max
  HMC_STAGE_LOG 0      1: log every staged-AD stage's time and GPU memory
"""
import os, sys, time
from hmc.posterior_act import ACTPosterior, NAMES, START, BOX     # sets XLA_FLAGS before jax starts
import numpy as np
import jax
from hmc.sampler import LockstepHMC, run

env = os.environ.get
DATA = env("HMC_DATA", "act")
MODE = env("HMC_MODE", "sample")
NCH = int(env("HMC_NCHAINS", "32"))
NW = int(env("HMC_WARMUP", "15"))
NS = int(env("HMC_NSAMPLES", "40"))
NGN = int(env("HMC_GN_STEPS", "2"))
METRIC_AT = tuple(int(v) for v in env("HMC_METRIC_AT", "5,10").split(",") if v.strip())
T_RANGE = tuple(float(v) for v in env("HMC_T", "1.0,2.0").split(","))
TARGET = float(env("HMC_TARGET", "0.8"))
EPS0 = float(env("HMC_EPS0", "0.8"))
SEED = int(env("HMC_SEED", "0"))
TAG = env("HMC_TAG", "")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, f"{DATA}{TAG}.npz")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def peak_gb():
    return max(d.memory_stats().get("peak_bytes_in_use", 0) for d in jax.devices()) / 2**30


log(f"[hmc] devices {jax.devices()} + {len(jax.devices('cpu'))} host; data {DATA}, mode {MODE}, {NCH} chains")
post = ACTPosterior(data=DATA, rtol=float(env("HMC_RTOL", "1e-5")), ad_chunk=int(env("HMC_AD_CHUNK", "64")),
                    spec_b_local=int(env("HMC_SPECB", "4")), stage_log=log if env("HMC_STAGE_LOG", "0") == "1" else None,
                    lensing_mu_chunk=int(env("HMC_MU_CHUNK", "0")), delta_l_max=int(env("HMC_DELTA_L", "5000")))


def oracle(th):
    t0 = time.time()
    lp, g = post.logp_and_grad(th)
    log(f"[call] {len(th)} cosmologies {time.time() - t0:.1f}s, {int(np.sum(~np.isfinite(lp)))} non-finite, "
        f"peak {peak_gb():.1f} GB")
    return lp, g


def save(history, theta, logp, grad, hmc, extra):
    h = {f"h_{k}": np.asarray(v) for k, v in history.items()}
    tmp = CKPT + ".tmp.npz"
    np.savez(tmp, names=np.array(NAMES), theta=theta, logp=logp, grad=grad, n_warmup=NW,
             **{f"s_{k}": v for k, v in hmc.state_dict().items()}, **extra, **h)
    os.replace(tmp, CKPT)


if MODE == "probe":
    rng = np.random.default_rng(SEED)
    th = np.clip(START + 0.5 * rng.standard_normal((NCH, len(NAMES))) * np.array(
        [1.7e-4, 2.1e-3, 8e-3, 1.6e-2, 7.5e-3, 9e-3, 3e-3, 4e-3]), BOX[:, 0] + 1e-6, BOX[:, 1] - 1e-6)
    for i in range(int(env("HMC_PROBE_CALLS", "2"))):
        lp, g = oracle(th)
    log(f"[probe] logp {np.round(lp[:4], 3)} ... finite {np.all(np.isfinite(lp))} {np.all(np.isfinite(g))}")
    sys.exit(0)

if os.path.exists(CKPT) and MODE == "sample":
    d = np.load(CKPT, allow_pickle=False)
    history = {k[2:]: list(d[k]) for k in d.files if k.startswith("h_")}
    theta, logp, grad = d["theta"], d["logp"], d["grad"]
    cov0, start = d["cov0"], d["start"]
    hmc = LockstepHMC(oracle, post.in_support, cov0, eps0=EPS0, T_range=T_RANGE, target=TARGET, seed=SEED)
    hmc.load_state_dict({k[2:]: d[k] for k in d.files if k.startswith("s_")})
    log(f"[hmc] resumed {CKPT} at iteration {len(history['theta'])}, eps_max {hmc.eps:.4f}")
else:
    start = START.copy()
    for it in range(NGN + 1):
        t0 = time.time()
        cov0, ev = post.laplace_cov(start)
        g = post.grad_at_laplace
        chi2 = sum(post.chi2_at_laplace.values())
        log(f"[gn] it {it} ({time.time() - t0:.0f}s): chi2 {chi2:.3f} (" +
            ", ".join(f"{k} {v:.3f}" for k, v in post.chi2_at_laplace.items()) +
            f"), predicted gain {float(g @ cov0 @ g):.3f}, peak {peak_gb():.1f} GB")
        log("[gn]   theta " + " ".join(f"{n} {v:.6g}" for n, v in zip(NAMES, start)))
        if it < NGN:
            start = np.clip(start + cov0 @ g, BOX[:, 0] + 1e-6, BOX[:, 1] - 1e-6)
    log("[hmc] Gauss-Newton metric: sd " + " ".join(f"{n} {s:.4g}" for n, s in zip(NAMES, np.sqrt(np.diag(cov0)))))
    # the same covariance in cobaya's names / units (a reference MH run's proposal)
    J = np.diag([1, 1, 100, 1, 1, 1, 1, 1.0])
    np.savetxt(os.path.join(OUT, f"{DATA}{TAG}_gn.covmat"), J @ cov0 @ J,
               header="omega_b omega_cdm H0 logA n_s tau_reio A_act P_act")
    if MODE == "gn":
        sys.exit(0)
    rng = np.random.default_rng(SEED + 12345)
    theta = rng.multivariate_normal(start, cov0, size=NCH)
    theta = np.clip(theta, BOX[:, 0] + 1e-6, BOX[:, 1] - 1e-6)
    logp, grad = oracle(theta)
    history = None
    hmc = LockstepHMC(oracle, post.in_support, cov0, eps0=EPS0, T_range=T_RANGE, target=TARGET, seed=SEED)
    log(f"[hmc] start: <logp> {np.mean(logp):.3f} (chi2 ~ {-2 * np.mean(logp):.2f}), "
        f"{int(np.sum(~np.isfinite(logp)))} non-finite")

if MODE == "echeck":
    r0 = np.random.default_rng(SEED + 7).standard_normal(theta.shape)
    med = []
    for n in (2, 4, 8):
        dH = hmc.trajectory(theta, logp, grad, 1.0 / n, n, r0=r0)[3]
        f = dH[np.isfinite(dH)]
        med.append(np.median(np.abs(f)))
        log(f"[echeck] eps {1.0 / n:.3f} ({n} steps, T=1): |dH| median {med[-1]:.4g} mean {np.mean(np.abs(f)):.4g} "
            f"max {np.max(np.abs(f)):.4g}, {int(np.sum(~np.isfinite(dH)))} dead")
    log(f"[echeck] ratios eps 1/2 : 1/4 : 1/8 = 1 : {med[1] / med[0]:.3f} : {med[2] / med[0]:.3f} "
        f"(exact gradient: 1 : 0.25 : 0.0625)")
    sys.exit(0)

history = run(hmc, theta, logp, grad, NW, NS, metric_at=METRIC_AT, history=history, log=log,
              callback=lambda h, th, lp, g: save(h, th, lp, g, hmc, dict(cov0=cov0, start=start)))
log(f"[hmc] done: {len(history['theta'])} trajectories, {hmc.n_calls} batched gradient calls")

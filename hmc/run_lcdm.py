"""run_lcdm.py — lockstep HMC of Planck 2018 LCDM (massless neutrinos) through ABCMB-k.

  HMC_MODE     sample (default) | echeck: the leapfrog energy-error check of the AD gradient (dH over a
               fixed time T=1 at eps 1/2, 1/4, 1/8 from the same momenta: ~eps^2 if the gradient is the
               gradient of logp), then exit
  HMC_NCHAINS  64      chains (one batched gradient call per leapfrog step)
  HMC_WARMUP   15      warmup trajectories (eps adapted every one; metric at HMC_METRIC_AT)
  HMC_METRIC_AT 5,10   warmup iterations at which the metric becomes the pooled draw covariance
  HMC_NSAMPLES 40      sampling trajectories after warmup
  HMC_T        1.0,2.0 integration-time range (whitened units; pi/2 decorrelates a Gaussian)
  HMC_TARGET   0.8     warmup target mean acceptance
  HMC_EPS0     0.8     initial eps (whitened units)
  HMC_SEED     0
  HMC_TAG      ''      output suffix: hmc/results/lcdm<TAG>.npz (checkpoint every trajectory; resumes)
  HMC_AD_CHUNK 64      max cosmologies per staged-AD call
  HMC_SPECB    4       cosmologies per device per spectrum-stage jvp
  HMC_RTOL     1e-5    rtol_large_k_PE
  HMC_LOWEE    simall  | sroll2
"""
import os, sys, time
from hmc.posterior import LCDMPosterior, NAMES, START, BOX     # sets XLA_FLAGS before jax starts
import numpy as np
import jax
from hmc.sampler import LockstepHMC, run

env = os.environ.get
MODE = env("HMC_MODE", "sample")
NCH = int(env("HMC_NCHAINS", "64"))
NW = int(env("HMC_WARMUP", "15"))
NS = int(env("HMC_NSAMPLES", "40"))
METRIC_AT = tuple(int(v) for v in env("HMC_METRIC_AT", "5,10").split(",") if v.strip())
T_RANGE = tuple(float(v) for v in env("HMC_T", "1.0,2.0").split(","))
TARGET = float(env("HMC_TARGET", "0.8"))
EPS0 = float(env("HMC_EPS0", "0.8"))
SEED = int(env("HMC_SEED", "0"))
TAG = env("HMC_TAG", "")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
CKPT = os.path.join(OUT, f"lcdm{TAG}.npz")


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


log(f"[hmc] devices {jax.devices()} + {len(jax.devices('cpu'))} host; mode {MODE}, {NCH} chains")
post = LCDMPosterior(rtol=float(env("HMC_RTOL", "1e-5")), ad_chunk=int(env("HMC_AD_CHUNK", "64")),
                     spec_b_local=int(env("HMC_SPECB", "4")), lowee=env("HMC_LOWEE", "simall"))


def oracle(th):
    t0 = time.time()
    lp, g = post.logp_and_grad(th)
    log(f"[call] {len(th)} cosmologies {time.time() - t0:.1f}s, {int(np.sum(~np.isfinite(lp)))} non-finite")
    return lp, g


def save(history, theta, logp, grad, hmc, extra):
    h = {f"h_{k}": np.asarray(v) for k, v in history.items()}
    tmp = CKPT + ".tmp.npz"
    np.savez(tmp, names=np.array(NAMES), theta=theta, logp=logp, grad=grad, n_warmup=NW,
             **{f"s_{k}": v for k, v in hmc.state_dict().items()}, **extra, **h)
    os.replace(tmp, CKPT)


def take_ownership(owner_path):
    """one writer per checkpoint: a still-running job named in the owner file (an earlier dev allocation or batch
    job) is cancelled before this run resumes, then this job becomes the owner."""
    import subprocess
    me = env("SLURM_JOB_ID", "")
    other = open(owner_path).read().strip() if os.path.exists(owner_path) else ""
    alive = lambda j: bool(subprocess.run(["squeue", "-h", "-j", j], capture_output=True, text=True).stdout.strip())
    if other and other != me and alive(other):
        log(f"[hmc] {CKPT} is owned by running job {other}: cancelling it")
        subprocess.run(["scancel", other])
        while alive(other):
            time.sleep(5)
    with open(owner_path, "w") as f:
        f.write(me)


if MODE == "sample":
    take_ownership(CKPT[:-4] + ".owner")

if os.path.exists(CKPT) and MODE == "sample":
    d = np.load(CKPT, allow_pickle=False)
    history = {k[2:]: list(d[k]) for k in d.files if k.startswith("h_")}
    theta, logp, grad = d["theta"], d["logp"], d["grad"]
    cov0 = d["cov0"]
    hmc = LockstepHMC(oracle, post.in_support, cov0, eps0=EPS0, T_range=T_RANGE, target=TARGET, seed=SEED)
    hmc.load_state_dict({k[2:]: d[k] for k in d.files if k.startswith("s_")})
    log(f"[hmc] resumed {CKPT} at iteration {len(history['theta'])}, eps_max {hmc.eps:.4f}")
else:
    t0 = time.time()
    cov0, ev = post.laplace_cov(START)
    log(f"[hmc] Gauss-Newton metric at START ({time.time() - t0:.0f}s): sd " +
        " ".join(f"{n} {s:.4g}" for n, s in zip(NAMES, np.sqrt(np.diag(cov0)))))
    log("[hmc] chi2 at START: " + ", ".join(f"{k} {v:.3f}" for k, v in post.chi2_at_laplace.items()))
    # the same covariance in cobaya's names / units: the reference MH run's proposal (hmc/reference)
    J = np.diag([1, 1, 100, 1, 1, 1, 1.0])
    np.savetxt(os.path.join(OUT, f"lcdm{TAG}_gn.covmat"), J @ cov0 @ J,
               header="omega_b omega_cdm H0 logA n_s tau_reio A_planck")
    rng = np.random.default_rng(SEED + 12345)
    theta = rng.multivariate_normal(START, cov0, size=NCH)
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
              callback=lambda h, th, lp, g: save(h, th, lp, g, hmc, dict(cov0=cov0)))
log(f"[hmc] done: {len(history['theta'])} trajectories, {hmc.n_calls} batched gradient calls")

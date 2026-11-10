"""sampler.py — lockstep multi-chain HMC for an oracle that evaluates a whole batch at once.

Every chain takes the same number of leapfrog steps with the same step size, so each leapfrog
step is ONE batched call of the oracle (for ABCMB: one batched staged-AD gradient over all
chains). Pure numpy; the oracle is any f(theta (B, D)) -> (logp (B,), grad (B, D)).

Kinetic energy 0.5 r.r in whitened coordinates z = L^-1 theta (Sigma = L L^T, the mass matrix
inverse), i.e. theta' = L r and r' = L^T grad_theta logp.

Each trajectory draws its integration time T ~ U(T_lo, T_hi) (shared by all chains) and takes
n = ceil(T / eps_max) steps of size T / n, then a Metropolis test per chain on the exact energy
change. A point outside the support or with a non-finite logp / gradient kills that chain's
trajectory (rejected; the oracle is then fed the chain's start point so the batch keeps its
shape). Any deterministic force gives a reversible, volume-preserving leapfrog, so the chains
target the posterior exactly; gradient errors only cost acceptance.

Warmup (all adaptation stops after it): eps_max is rescaled after every trajectory with the
leapfrog energy-error scaling, eps <- eps * ((1 - target) / (1 - mean accept))^(1/2) (clipped to
a factor 2 either way), and Sigma is replaced at the iterations in `metric_at` by the pooled
covariance of all chains' draws so far (the first `metric_burn` warmup iterations skipped).
"""
import json
import numpy as np


class LockstepHMC:
    def __init__(self, oracle, in_support, cov0, eps0=0.8, T_range=(1.0, 2.0), target=0.8,
                 seed=0, max_dH=1000.0):
        self.oracle = oracle
        self.in_support = in_support
        self.set_metric(np.asarray(cov0, float))
        self.eps = float(eps0)
        self.T_range = tuple(float(t) for t in T_range)
        self.target = float(target)
        self.max_dH = float(max_dH)
        self.rng = np.random.default_rng(seed)
        self.n_calls = 0

    # ---------------- metric ----------------
    def set_metric(self, cov):
        cov = 0.5 * (cov + cov.T)
        self.cov = cov
        self.L = np.linalg.cholesky(cov)

    # ---------------- one trajectory for all chains ----------------
    def _eval(self, th, alive, th_dummy):
        """oracle on the alive chains (dead ones get their start point); returns logp, grad, ok."""
        ok = alive & self.in_support(th)
        lp, g = self.oracle(np.where(ok[:, None], th, th_dummy))
        self.n_calls += 1
        lp = np.asarray(lp, float); g = np.asarray(g, float)
        ok = ok & np.isfinite(lp) & np.isfinite(g).all(axis=1)
        return np.where(ok, lp, -np.inf), np.where(ok[:, None], g, 0.0), ok

    def trajectory(self, theta, logp, grad, eps, n_steps, r0=None):
        """leapfrog all chains n_steps of size eps from (theta, logp, grad); returns the proposal
        (theta1, logp1, grad1), dH = H1 - H0 (inf for a dead trajectory) and the momenta used."""
        B, D = theta.shape
        L = self.L
        r = self.rng.standard_normal((B, D)) if r0 is None else np.array(r0, float)
        H0 = -logp + 0.5 * np.sum(r * r, axis=1)
        th, lp, g = theta.copy(), logp.copy(), grad.copy()
        alive = np.ones(B, bool)
        r_used = r.copy()
        r = r + 0.5 * eps * (g @ L)                   # r' = L^T grad, row-wise
        for s in range(n_steps):
            th = th + eps * (r @ L.T)                 # theta' = L r, row-wise
            lp, g, alive = self._eval(th, alive, theta)
            r = r + (0.5 if s == n_steps - 1 else 1.0) * eps * (g @ L)
        H1 = -lp + 0.5 * np.sum(r * r, axis=1)
        dH = np.where(alive, H1 - H0, np.inf)
        return th, lp, g, dH, r_used

    def step(self, theta, logp, grad):
        """one HMC transition for every chain; returns the new state and the trajectory's stats."""
        T = self.rng.uniform(*self.T_range)
        n = max(1, int(np.ceil(T / self.eps)))
        eps = T / n
        th1, lp1, g1, dH, _ = self.trajectory(theta, logp, grad, eps, n)
        div = ~np.isfinite(dH) | (np.abs(dH) > self.max_dH)
        a = np.where(div, 0.0, np.exp(np.minimum(0.0, -np.where(div, 0.0, dH))))
        acc = self.rng.uniform(size=len(a)) < a
        theta = np.where(acc[:, None], th1, theta)
        logp = np.where(acc, lp1, logp)
        grad = np.where(acc[:, None], g1, grad)
        return theta, logp, grad, dict(T=T, n=n, eps=eps, accept_prob=a, accepted=acc, dH=dH,
                                       divergent=div)

    def adapt_eps(self, mean_accept):
        """warmup step-size update from the chains' mean acceptance probability (leapfrog energy
        error ~ eps^2)."""
        f = np.sqrt((1.0 - self.target) / max(1.0 - mean_accept, 1e-3))
        self.eps *= float(np.clip(f, 0.5, 2.0))

    # ---------------- checkpointable state ----------------
    def state_dict(self):
        return dict(cov=self.cov, eps=self.eps, n_calls=self.n_calls,
                    rng_state=json.dumps(self.rng.bit_generator.state))

    def load_state_dict(self, d):
        self.set_metric(np.asarray(d["cov"], float))
        self.eps = float(d["eps"])
        self.n_calls = int(d["n_calls"])
        self.rng.bit_generator.state = json.loads(str(d["rng_state"]))


def pooled_cov(draws):
    """covariance of draws (n_iter, B, D) pooled over iterations and chains."""
    x = np.asarray(draws, float).reshape(-1, draws.shape[-1])
    return np.cov(x, rowvar=False)


def _print(msg):
    print(msg, flush=True)


def run(hmc, theta, logp, grad, n_warmup, n_samples, metric_at=(), metric_burn=2,
        history=None, callback=None, log=_print):
    """warmup + sampling. history: dict of per-iteration lists (resume: pass the checkpoint's); the
    iteration index is len(history['theta']). callback(history, theta, logp, grad) after every
    iteration (checkpointing). Returns history."""
    import time
    keys = ("theta", "logp", "accept_prob", "accepted", "dH", "divergent", "eps", "n", "T",
            "eps_max", "wall")
    history = history if history is not None else {k: [] for k in keys}
    while len(history["theta"]) < n_warmup + n_samples:
        it = len(history["theta"])
        warm = it < n_warmup
        if warm and it in metric_at and it > metric_burn:
            draws = np.asarray(history["theta"][metric_burn:it])
            hmc.set_metric(pooled_cov(draws))
            log(f"[hmc] it {it}: metric <- pooled covariance of {draws.shape[0]} x {draws.shape[1]} draws; "
                f"sd " + " ".join(f"{s:.4g}" for s in np.sqrt(np.diag(hmc.cov))))
        eps_max = hmc.eps
        t0 = time.time()
        theta, logp, grad, st = hmc.step(theta, logp, grad)
        wall = time.time() - t0
        if warm:
            hmc.adapt_eps(float(np.mean(st["accept_prob"])))
        for k, v in (("theta", theta), ("logp", logp), ("accept_prob", st["accept_prob"]),
                     ("accepted", st["accepted"]), ("dH", st["dH"]), ("divergent", st["divergent"]),
                     ("eps", st["eps"]), ("n", st["n"]), ("T", st["T"]), ("eps_max", eps_max),
                     ("wall", wall)):
            history[k].append(np.array(v))
        dHf = st["dH"][np.isfinite(st["dH"])]
        log(f"[hmc] it {it:4d} {'warm' if warm else 'samp'} T {st['T']:.3f} n {st['n']} eps {st['eps']:.4f} "
            f"<a> {np.mean(st['accept_prob']):.3f} acc {np.mean(st['accepted']):.3f} "
            f"div {int(st['divergent'].sum())} |dH| med {np.median(np.abs(dHf)) if dHf.size else np.nan:.3g} "
            f"max {np.max(np.abs(dHf)) if dHf.size else np.nan:.3g}  <logp> {np.mean(logp):.3f}  "
            f"{wall:.0f}s  -> eps_max {hmc.eps:.4f}")
        if callback is not None:
            callback(history, theta, logp, grad)
    return history

"""CPU tests of hmc/sampler.py on Gaussian targets (seconds; run on a compute node):
    python -m pytest -q hmc/test_sampler_toy.py
"""
import numpy as np
from hmc.sampler import LockstepHMC, run

D, B = 7, 64


def _target(seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D))
    C = A @ A.T + D * np.eye(D)
    s = 10.0 ** rng.uniform(-4, 0, D)                 # cosmology-like spread of scales
    C = C * s[:, None] * s[None, :]
    mu = rng.uniform(-1, 1, D) * s
    return mu, C


def _oracle(mu, C, grad_scale=1.0):
    P = np.linalg.inv(C)

    def f(th):
        d = th - mu
        return -0.5 * np.einsum("bi,ij,bj->b", d, P, d), -grad_scale * d @ P
    return f


def _sample(mu, C, cov0, grad_scale=1.0, n_warm=15, n_samp=150, seed=0, support=None):
    f = _oracle(mu, C, grad_scale)
    sup = support or (lambda th: np.ones(len(th), bool))
    hmc = LockstepHMC(f, sup, cov0, eps0=0.5, seed=seed)
    rng = np.random.default_rng(seed + 100)
    th = rng.multivariate_normal(mu, cov0, size=B)
    if support is not None:
        th = th[support(th)][:1].repeat(B, 0) if not support(th).all() else th
    lp, g = f(th)
    h = run(hmc, th, lp, g, n_warm, n_samp, metric_at=(5, 10), log=lambda m: None)
    return np.asarray(h["theta"][n_warm:]), h, hmc


def test_gaussian_moments():
    mu, C = _target()
    cov0 = C * np.outer(*(2 * [np.random.default_rng(3).uniform(0.5, 2.0, D)]))   # mis-scaled start
    x, h, hmc = _sample(mu, C, cov0)
    flat = x.reshape(-1, D)
    sd = np.sqrt(np.diag(C))
    # chains are nearly independent per draw at T ~ pi/2: generous 5-sigma MC bounds
    n_eff = flat.shape[0] / 3.0
    assert np.all(np.abs(flat.mean(0) - mu) < 5 * sd / np.sqrt(n_eff))
    assert np.all(np.abs(flat.std(0) / sd - 1) < 0.08)
    a = np.mean(h["accept_prob"][15:])
    assert 0.65 < a < 0.95, a


def test_energy_error_scales_as_eps2():
    """with the exact gradient the leapfrog energy error over a fixed time shrinks ~ eps^2; with a wrong
    gradient it does not (the check used on ABCMB's AD gradient)."""
    mu, C = _target()
    rng = np.random.default_rng(5)
    th = rng.multivariate_normal(mu, C, size=B)
    r0 = rng.standard_normal((B, D))
    for gs, expect_conv in ((1.0, True), (0.9, False)):
        f = _oracle(mu, C, gs)
        hmc = LockstepHMC(f, lambda t: np.ones(len(t), bool), C, seed=0)
        lp, g = f(th)
        med = []
        for n in (2, 4, 8):
            dH = hmc.trajectory(th, lp, g, 1.0 / n, n, r0=r0)[3]
            med.append(np.median(np.abs(dH)))
        ratio = med[0] / med[2]
        if expect_conv:
            assert 10 < ratio < 25, med          # eps 0.5 -> 0.125: x16
        else:
            assert ratio < 3, med


def test_wrong_gradient_still_exact():
    """any deterministic force keeps detailed balance: a 30%-wrong gradient costs acceptance, not
    correctness."""
    mu, C = _target()
    x, h, _ = _sample(mu, C, C, grad_scale=0.7, n_samp=200, seed=2)
    flat = x.reshape(-1, D)
    sd = np.sqrt(np.diag(C))
    assert np.all(np.abs(flat.std(0) / sd - 1) < 0.1)
    assert np.all(np.abs(flat.mean(0) - mu) < 6 * sd / np.sqrt(flat.shape[0] / 5.0))


def test_support_respected():
    mu, C = _target()
    lo = mu[0] - 0.5 * np.sqrt(C[0, 0])
    sup = lambda th: th[:, 0] > lo
    x, _, _ = _sample(mu, C, C, support=sup, n_samp=60)
    assert (x[..., 0] > lo).all()

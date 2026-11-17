"""scan/lm_tools.py on a toy double well: the LM step, the eigen-spread starts, minima clustering / flags, and the
union of below-threshold regions. CPU, no ABCMB."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scan.lm_tools import (proj_grad, lm_step, newton_decrement, fit_in_box,
                           spread_starts, cluster_points, flag_point, distinct_minima, union_regions, is_multimodal)

LO, HI = -np.full(2, 5.0), np.full(2, 5.0)


# chi2(x) = (x0^2 - 1)^2 + 0.3 x0 + 0.5 x1^2: minima near (+0.963, 0) and (-1.037, 0), the lower
# one at x0 < 0; the "Gauss-Newton" matrix is the Hessian with its eigenvalues floored (PSD, as J^T W J)
def f(x):
    return (x[0] ** 2 - 1) ** 2 + 0.3 * x[0] + 0.5 * x[1] ** 2


def g(x):
    return np.array([4 * x[0] * (x[0] ** 2 - 1) + 0.3, x[1]])


def gn(x):
    H = np.array([[12 * x[0] ** 2 - 4, 0.0], [0.0, 1.0]])
    ev, V = np.linalg.eigh(H)
    return (V * np.maximum(ev, 1e-2)) @ V.T


def descend(x, maxit=60, lam=1e-3, step_max=0.5, gtol=1e-8):
    """plain damped LM with the driver's gain test"""
    for _ in range(maxit):
        gg, H = g(x), gn(x)
        if np.abs(proj_grad(gg[None], x[None], LO[None], HI[None])).max() < gtol:
            break
        for k in range(6):
            d, pred = lm_step(gg, H, x, LO, HI, lam * 10.0 ** k, step_max)
            if f(x + d) < f(x) and f(x) - f(x + d) >= 1e-4 * pred:
                x = x + d; lam = max(lam * 10.0 ** (k - 1), 1e-7)
                break
        else:
            lam *= 1e3
    return x


X_RIGHT = descend(np.array([0.8, 0.3]))
X_LEFT = descend(np.array([-0.8, -0.3]))


def test_lm_descends_to_each_minimum():
    assert np.allclose(g(X_RIGHT), 0, atol=1e-6) and X_RIGHT[0] > 0.9
    assert np.allclose(g(X_LEFT), 0, atol=1e-6) and X_LEFT[0] < -1.0
    assert f(X_LEFT) < f(X_RIGHT)


def test_lm_step_respects_bounds():
    x = np.array([0.0, 4.9]); gg, H = np.array([0.0, -1.0]), np.eye(2)
    d, pred = lm_step(gg, H, x, LO, HI, 1e-7, 10.0)
    assert np.isclose((x + d)[1], 5.0) and pred > 0
    # a component pushing into an active bound is zeroed in the projected gradient
    assert proj_grad(np.array([[0.0, -1.0]]), np.array([[0.0, 5.0]]), LO[None], HI[None])[0, 1] == 0


def test_newton_decrement_is_the_remaining_gain():
    """On a quadratic chi2 the decrement is exactly chi2(x) - min chi2, a large gradient along a stiff direction
    is worth little, a dim held at a bound drops out, and a singular H with no gradient in its null space
    stays finite."""
    rng = np.random.default_rng(1)
    Q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    A = Q @ np.diag([400.0, 2.0, 1.0, 0.5]) @ Q.T                # one stiff direction, as in the BBN profile
    xs = np.array([0.3, -0.2, 0.1, 0.0]); lo, hi = -np.full(4, 5.0), np.full(4, 5.0)
    chi2 = lambda x: 0.5 * (x - xs) @ A @ (x - xs)
    x = xs + 0.01 * Q[:, 0] + 0.2 * Q[:, 2]
    gg = A @ (x - xs)
    assert np.isclose(newton_decrement(gg, A, x, lo, hi), chi2(x))
    assert np.isclose(np.abs(Q[:, 0] @ gg), 4.0) and np.isclose(0.5 * 4.0 ** 2 / 400.0, 0.02)
    assert np.isclose(np.abs(Q[:, 2] @ gg), 0.2) and np.isclose(newton_decrement(gg, A, x, lo, hi), 0.04)
    # dim 0 held at the bound its gradient pushes into: the gain of the 3-dim problem with x0 fixed
    lo2, hi2 = lo.copy(), hi.copy()
    if gg[0] > 0:
        lo2[0] = x[0]
    else:
        hi2[0] = x[0]
    F = [1, 2, 3]
    xF = x.copy(); xF[F] = x[F] - np.linalg.solve(A[np.ix_(F, F)], gg[F])
    assert np.isclose(newton_decrement(gg, A, x, lo2, hi2), chi2(x) - chi2(xF))
    assert np.isclose(newton_decrement(np.array([0.2, 0.0]), np.diag([1.0, 0.0]), np.zeros(2), LO, HI), 0.02)



def test_spread_starts_follow_the_softest_directions():
    H = np.array([[0.5, 0.0], [0.0, 8.0]])             # softest = x0 (profile sigma 2)
    x0 = np.array([0.1, 0.2])
    s = spread_starts(x0, H, LO, HI, 4, 2.0)
    assert np.allclose(s[0], x0 + [4.0, 0.0]) and np.allclose(s[1], x0 - [4.0, 0.0])
    assert np.allclose(np.abs(s[2] - x0), [0.0, 1.0]) and np.allclose(s[2] - x0, -(s[3] - x0))
    # a start that would leave the box is shortened along its direction, not clipped per coordinate
    H = np.array([[1.0, 0.9], [0.9, 1.0]])             # softest = (1, -1)/sqrt2, lambda 0.1
    s = spread_starts(np.array([4.0, -4.0]), H, LO, HI, 2, 2.0)   # +-2 sqrt(2/0.1)/sqrt2 = +-6.325
    assert any(np.allclose(r, [5.0, -5.0]) for r in s)
    assert any(np.allclose(r, [4.0 - 6.3246, -4.0 + 6.3246], atol=1e-3) for r in s)
    assert np.allclose(fit_in_box(np.zeros(2), np.array([20.0, 10.0]), LO, HI), [5.0, 2.5])


def test_cluster_and_flags():
    X = np.array([[1.0, 0.0], [1.02, 0.01], [-1.0, 0.0], [0.5, 0.5]])
    F = np.array([0.5, 0.52, 0.2, 3.0])
    labels, reps = cluster_points(X, F, 0.1)
    assert reps == [2, 0, 3] and labels.tolist() == [1, 1, 0, 2]
    conv = np.array([True, True, True, False]); st = np.zeros(4, bool)
    flag, why, r = flag_point(X, F, conv, st, 0.1, 0.1)
    assert flag and "2 distinct minima" in why and r == [2, 0]
    # one minimum, agreeing chi2: no flag
    flag, why, r = flag_point(X[:2], F[:2], conv[:2], st[:2], 0.1, 0.1)
    assert not flag and why == "" and r == [0]
    # one minimum, chi2 spread above the tolerance: flag
    flag, why, _ = flag_point(X[:2], np.array([0.5, 0.7]), conv[:2], st[:2], 0.1, 0.1)
    assert flag and "chi2 spread" in why
    # a stalled start flags, even with nothing converged
    flag, why, r = flag_point(X[:1], F[:1], np.zeros(1, bool), np.ones(1, bool), 0.1, 0.1)
    assert flag and why == "1 stalled" and r == []


def _segment_rise(fun, X):
    """barrier(i, j) for points X: the rise of fun along the segment between X[i] and X[j] above the higher end"""
    return lambda i, j: max(fun((1 - t) * X[i] + t * X[j]) for t in (0.25, 0.5, 0.75)) - max(fun(X[i]), fun(X[j]))


def test_flat_valley_is_one_minimum():
    """Converged starts far apart along a flat direction (a decoupling nuisance) at the same chi2 are one minimum
    with the barrier test, two without it; minima behind a barrier, or apart in chi2, stay distinct."""
    flat = lambda x: x[0] ** 2                                  # x1 open: chi2 flat along it
    X = np.array([[0.0, -3.0], [0.0, 0.5], [0.001, 4.0]])
    F = np.array([flat(x) for x in X]); conv = np.ones(3, bool); st = np.zeros(3, bool)
    assert flag_point(X, F, conv, st, 0.5, 0.1)[0]             # old rule: 3 "distinct minima"
    flag, why, r = flag_point(X, F, conv, st, 0.5, 0.1, barrier=_segment_rise(flat, X))
    assert not flag and why == "" and len(r) == 1
    labels, reps = distinct_minima(X, F, 0.5, 0.1, _segment_rise(flat, X))
    assert labels.tolist() == [0, 0, 0] and reps == [0]
    # the double well f of this file: two minima 0.6 apart in chi2 with a 0.7 barrier between them
    Xw = np.array([X_RIGHT, X_LEFT]); Fw = np.array([f(x) for x in Xw])
    flag, why, r = flag_point(Xw, Fw, np.ones(2, bool), np.zeros(2, bool), 0.1, 0.1, barrier=_segment_rise(f, Xw))
    assert flag and "2 distinct minima" in why and "1 behind a barrier" in why and r == [1, 0]
    # a start stuck on a plateau above the minimum (no barrier, chi2 higher by > dchi2): kept apart and flagged
    # as a slope or plateau, not a barrier
    plateau = lambda x: 1.0 - np.exp(-x[1] ** 2)               # minimum at x1 = 0, flat as |x1| grows
    Xp = np.array([[0.0, 0.0], [0.0, 6.0]]); Fp = np.array([plateau(x) for x in Xp])
    flag, why, r = flag_point(Xp, Fp, np.ones(2, bool), np.zeros(2, bool), 0.5, 0.1, barrier=_segment_rise(plateau, Xp))
    assert flag and "0 behind a barrier, 1 higher with none" in why and r == [0, 1]


def test_union_regions_and_multimodality():
    xs = np.linspace(-3, 3, 6001)
    ys = (xs ** 2 - 1) ** 2 * 4 + 0.3 * xs                 # two wells, barrier 4 at x = 0
    ys -= ys.min()
    reg = union_regions(xs, ys, 1.0)
    assert len(reg) == 2 and reg[0][1] < 0 < reg[1][0]
    for lo, hi in reg:                                     # ends sit on the threshold
        assert abs(np.interp(lo, xs, ys) - 1.0) < 2e-3 and abs(np.interp(hi, xs, ys) - 1.0) < 2e-3
    assert len(union_regions(xs, ys, 5.0)) == 1          # above the barrier: one region
    # inside at the curve's end: NaN, unless that end is the physical bound
    reg = union_regions(xs, ys, 1e3)
    assert len(reg) == 1 and np.isnan(reg[0][0]) and np.isnan(reg[0][1])
    reg = union_regions(xs, ys, 1e3, bound=(-3.0, None))
    assert reg[0][0] == -3.0 and np.isnan(reg[0][1])
    # a threshold that varies along x is accepted (the driver itself passes a scalar)
    assert len(union_regions(xs, ys, np.full_like(xs, 1.0))) == 2
    # the points of a bimodal profile vs a unimodal one with scatter on a flat bottom
    xg = np.linspace(-2, 2, 21)
    assert is_multimodal(xg, (xg ** 2 - 1) ** 2 * 4 + 0.3 * xg, 9.0, 0.1)
    rng = np.random.default_rng(0)
    assert not is_multimodal(xg, xg ** 2 + 1e-3 * rng.standard_normal(21), 9.0, 0.1)
    assert not is_multimodal(xg, xg ** 2, 9.0, 0.1)

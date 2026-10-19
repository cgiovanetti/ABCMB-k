"""lm_tools.py — pure-numpy pieces of the profile driver (scan/profile_prod_ad.py).

The projected Levenberg-Marquardt step, deflation of known stationary points (Farrell, Birkisson &
Funke 2015, SIAM J. Sci. Comput. 37, A2026), the deliberate multistart spread along the softest
Gauss-Newton directions, clustering of converged starts into distinct minima with the user's flags,
and the union of below-threshold regions of a profile. No JAX, no ABCMB, no likelihood: everything
here is exercised on a toy double well in pytests/test_lm_tools.py.

Coordinates are the driver's scaled ones, x = (theta - CENTER) / SIGMA, and H is d2chi2/dx2 there
(the Gauss-Newton matrix), so one profile sigma along an eigenvector v_k of H is sqrt(2 / lambda_k).
"""
import numpy as np


# ======================================================================
# projected Levenberg-Marquardt step
# ======================================================================
def proj_grad(G, X, XLO, XHI):
    """gradient with the components that push into an active bound zeroed (KKT residual)."""
    at = ((X <= XLO + 1e-9) & (G > 0)) | ((X >= XHI - 1e-9) & (G < 0))
    return np.where(at, 0.0, G)


def lm_step(g, H, x, xlo, xhi, lam, step_max):
    """one projected LM step: solve (H_FF + lam diag(H_FF)) d = -g_F on the dims not held at a bound,
    cap it at step_max (max-norm), project into the box. Returns (d, Gauss-Newton predicted chi2 drop)."""
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
    if dm > step_max:
        d *= step_max / dm
    d = np.clip(x + d, xlo, xhi) - x
    return d, -(g @ d + 0.5 * d @ H @ d)


# ======================================================================
# deflation (Farrell, Birkisson & Funke 2015)
# ======================================================================
def deflation(x, mins, p=2.0, alpha=1.0):
    """log M(x) and grad(log M)(x) for M(x) = prod_i (|x - x_i|^-p + alpha) over the known
    stationary points mins (m, P). No known points: (0, 0) -- M = 1, nothing deflated. The shift
    alpha makes M -> alpha^m far from every x_i, so the deflated step reverts to the ordinary one
    there; |x - x_i| ~ alpha^(-1/p) is the reach of the deflation."""
    mins = np.asarray(mins, float).reshape(-1, len(x))
    if len(mins) == 0:
        return 0.0, np.zeros_like(x, dtype=float)
    dx = x[None, :] - mins
    r = np.maximum(np.sqrt((dx ** 2).sum(1)), 1e-12)
    rp = r ** (-p)
    logM = float(np.log(rp + alpha).sum())
    dlogM = ((-p * rp / (r ** 2) / (rp + alpha))[:, None] * dx).sum(0)
    return logM, dlogM


def deflate_step(d, x, mins, p=2.0, alpha=1.0, tau_min=1e-3):
    """The deflated Newton step from the ordinary one. For the root problem g(x) = 0 deflated to
    M(x) g(x) = 0, the Jacobian is M J + g grad(M)^T and the Sherman-Morrison identity turns its
    Newton step into d / (1 - grad(log M).d), d = -J^-1 g the ordinary step (Farrell et al. 2015,
    sec. 3.1), so the deflated step costs no extra gradient or curvature. Near a known point the
    denominator shrinks or flips sign (the step is pushed away from it); far away it is 1. |tau| is
    floored at tau_min; the caller caps the step's length."""
    _, dl = deflation(x, mins, p, alpha)
    tau = 1.0 - float(dl @ d)
    if abs(tau) < tau_min:
        tau = tau_min if tau >= 0 else -tau_min
    return d / tau


def deflated_resid(x, g, mins, p=2.0, alpha=1.0):
    """log ||M(x) g||_inf: the residual of the deflated root problem, the quantity a deflated step
    must decrease. Infinite at a known point (M -> inf there), so the iteration cannot settle on one."""
    logM, _ = deflation(x, mins, p, alpha)
    return logM + np.log(max(float(np.abs(g).max()), 1e-300))


# ======================================================================
# multistart spread and minima clustering
# ======================================================================
def fit_in_box(x0, step, lo, hi):
    """x0 + t step with the largest t in [0, 1] that stays inside [lo, hi] (direction kept)."""
    t = 1.0
    for s, a, b, c in zip(step, x0, lo, hi):
        if s > 0 and a + s > c:
            t = min(t, (c - a) / s)
        elif s < 0 and a + s < b:
            t = min(t, (b - a) / s)
    return np.clip(x0 + max(t, 0.0) * step, lo, hi)


def spread_starts(x0, H, lo, hi, K, spread, eig_floor=1e-6):
    """K deliberately spread starts around x0: start k lies at +-spread profile sigmas along the
    softest eigen-direction number k // 2 of H (sign + for even k, - for odd k), one sigma along
    v_k being sqrt(2 / lambda_k); a start that would leave [lo, hi] is shortened along its direction.
    Eigenvalues are floored at eig_floor, so a flat direction gives a start on the box."""
    H = np.asarray(H, float)
    ev, V = np.linalg.eigh(0.5 * (H + H.T))
    ev = np.maximum(ev, eig_floor)
    out = np.empty((K, len(x0)))
    for k in range(K):
        j = (k // 2) % len(ev)
        s = 1.0 if k % 2 == 0 else -1.0
        out[k] = fit_in_box(np.asarray(x0, float), s * spread * np.sqrt(2.0 / ev[j]) * V[:, j],
                            np.asarray(lo, float), np.asarray(hi, float))
    return out


def cluster_points(X, F, dx):
    """Greedy clusters of the points X (n, P) by max-norm distance dx, visited in increasing F so
    the representative of each cluster is its lowest-F member. Returns (labels (n,), reps: list of
    row indices, one per cluster, in increasing F)."""
    X = np.asarray(X, float); F = np.asarray(F, float)
    labels = -np.ones(len(F), int); reps = []
    for i in np.argsort(F, kind="stable"):
        for c, r in enumerate(reps):
            if np.abs(X[i] - X[r]).max() <= dx:
                labels[i] = c
                break
        else:
            labels[i] = len(reps)
            reps.append(int(i))
    return labels, reps


def flag_point(X, F, conv, stalled, dx, dchi2):
    """The multimodality flags of one grid point from its starts: X (n, P) end points, F (n,) chi2,
    conv (n,) converged, stalled (n,). Flagged when any start stalled, when converged starts sit at
    distinct points (> dx apart in max-norm: distinct minima, whatever their chi2), or when converged
    starts at the same point disagree in chi2 by more than dchi2. Returns (flag, reason, reps) with
    reps the row indices of the distinct converged minima, lowest chi2 first."""
    conv = np.asarray(conv, bool); stalled = np.asarray(stalled, bool)
    F = np.asarray(F, float); X = np.asarray(X, float)
    reasons = []
    if stalled.any():
        reasons.append(f"{int(stalled.sum())} stalled")
    c = np.where(conv)[0]
    reps = []
    if len(c):
        labels, r = cluster_points(X[c], F[c], dx)
        reps = [int(c[i]) for i in r]
        if len(reps) > 1:
            reasons.append(f"{len(reps)} distinct minima (dchi2 {F[reps[-1]] - F[reps[0]]:.3f})")
        spread = max(float(F[c][labels == k].max() - F[c][labels == k].min()) for k in range(len(reps)))
        if spread > dchi2:
            reasons.append(f"chi2 spread {spread:.3f} at one point")
    return bool(reasons), "; ".join(reasons), reps


# ======================================================================
# profile shape: union of regions below threshold, multimodality of the points
# ======================================================================
def union_regions(xs, ys, thr, bound=(np.nan, np.nan)):
    """Connected regions {x : ys <= thr} of a dense curve (xs ascending; thr a scalar or an array on
    xs), each as (lo, hi) with the ends linearly interpolated at the threshold crossings. An end on
    the curve's edge is NaN (not reached within the curve) unless that edge is the physical bound.
    Returns the list of regions in increasing x."""
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    thr = np.broadcast_to(np.asarray(thr, float), xs.shape)
    inside = ys <= thr
    if not inside.any():
        return []
    f = ys - thr
    d = np.diff(inside.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0]
    if inside[0]:
        starts = np.r_[0, starts]
    if inside[-1]:
        ends = np.r_[ends, len(xs) - 1]
    blo, bhi = (np.nan if b is None else float(b) for b in bound)

    def cross(a, b):                                   # inside at a, outside at b
        return xs[a] + (xs[b] - xs[a]) * (-f[a]) / (f[b] - f[a] + 1e-300)

    def onb(x, b):
        return np.isfinite(b) and abs(x - b) < 1e-12 * max(1.0, abs(b))

    out = []
    for a, b in zip(starts, ends):
        lo = cross(a, a - 1) if a > 0 else (xs[0] if onb(xs[0], blo) else np.nan)
        hi = cross(b, b + 1) if b < len(xs) - 1 else (xs[-1] if onb(xs[-1], bhi) else np.nan)
        out.append((float(lo), float(hi)))
    return out


def is_multimodal(x, y, within, barrier):
    """True when the profile's points show more than one local minimum within `within` of the
    global minimum, each separated from it by a barrier higher than `barrier` (so per-point scatter
    on a flat bottom does not count). Grid ends count as minima when lower than their neighbour."""
    x = np.asarray(x, float); y = np.asarray(y, float); m = np.isfinite(y)
    x, y = x[m], y[m]
    if len(y) < 3:
        return False
    o = np.argsort(x); y = y[o]
    j = int(np.argmin(y))
    n = len(y)
    for i in range(n):
        if i == j or y[i] - y[j] > within:
            continue
        left = i == 0 or y[i] < y[i - 1]
        right = i == n - 1 or y[i] <= y[i + 1]
        if left and right:
            lo, hi = sorted((i, j))
            if y[lo + 1:hi].max() - max(y[i], y[j]) > barrier:
                return True
    return False

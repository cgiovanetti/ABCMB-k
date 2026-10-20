"""toy_bimodal.py — a 3-parameter toy objective for the multimodality machinery (no ABCMB).

  chi2(a, b, c) = ((a - delta tanh b) / s)^2                               POI a; the branch centre
                                                                            +-delta follows the sign of b
                + h [1 - exp(-(b - beta)^2 / 2w^2) - exp(-(b + beta)^2 / 2w^2)]   two wells in b at +-beta
                + D (1 - tanh b) / 2                                        the b < 0 branch sits D higher
                + k b^2                                                     a weak confinement, so the
                                                                            wells' tails are not flat
                + Bc (c - c0)^2                                             a stiff spectator

tanh is saturated inside the wells, so each branch is a full parabola in a and both exist at every a:
the exact profile is ~ min((a - delta)^2, (a + delta)^2 + D) / s^2 + k beta^2, bimodal with minima at
a = +-delta (profile_exact below minimizes it numerically, tails and the small well shifts included).
With delta = 1.1, D = 0.5 the branches cross at a = -0.11 (dchi2 1.47), so the 68% region is the union
of [0.10, 2.10] and [-1.81, -0.39] while the 95% region is one interval. The warm start sits in the
b > 0 well: without the spread starts and branches the driver follows that branch alone and is wrong
for a < -0.11. For a row the softest warm GN direction is b (curvature ~h/w^2 = 1.1, profile sigma
1.3), so the -2.5 sigma spread start lands at b ~ -0.4; for a <~ -0.6 it rolls into the other well,
which flags those points and seeds the branch that then covers every grid point.
pytests/test_profile_toy.py runs the driver on this config (PA_XBOX=10 so the b < 0 well is inside
the box, PA_MS_SPREAD=2.5, PA_NPTS=21).
"""
import numpy as np
import jax.numpy as jnp

DELTA, S, H, W, BETA, DOFF, K, BC, C0 = 1.1, 1.0, 1.6, 1.2, 3.0, 0.5, 0.005, 20.0, 0.3


def _chi2_ab(a, b, xp):
    """chi2 at c = c0 with the array module xp (jnp for the driver, np for the references)."""
    wells = H * (1.0 - xp.exp(-(b - BETA) ** 2 / (2 * W ** 2)) - xp.exp(-(b + BETA) ** 2 / (2 * W ** 2)))
    return ((a - DELTA * xp.tanh(b)) / S) ** 2 + wells + DOFF * (1.0 - xp.tanh(b)) / 2 + K * b ** 2


def chi2(theta):
    a, b, c = theta[0], theta[1], theta[2]
    return _chi2_ab(a, b, jnp) + BC * (c - C0) ** 2


def profile_exact(a, bgrid=np.linspace(-8.0, 8.0, 160001)):
    """chi2 profiled over (b, c) at the POI values a (c = c0 exactly; b on a fine grid, error < 1e-8)."""
    a = np.atleast_1d(np.asarray(a, float))[:, None]
    return _chi2_ab(a, bgrid[None, :], np).min(1)


def branch_exact(a, sign):
    """the chi2 of the b > 0 (sign=+1) or b < 0 (sign=-1) well alone: the local minimum over
    |b| in [beta - 0.8, 8] on that side (the well sits at |b| ~ beta; closer to 0 is the barrier)."""
    return profile_exact(a, np.linspace(BETA - 0.8, 8.0, 60001) * sign)


CONFIG = {
    "order": ["a", "b", "c"],
    "cen": {"a": 0.0, "b": BETA, "c": C0},
    "sig": {"a": 1.0, "b": 1.0, "c": 0.3},
    "grid_sig": {"a": 1.0, "b": 1.0, "c": 0.3},
    "pois": ["a"],
    "fixed": {},
    "user_species": None,
    "use_lowtt": False, "use_lowee": False, "use_plik": False,
    "npts": 11, "nsig": 3.0,
    "toy": {"chi2": chi2},
}

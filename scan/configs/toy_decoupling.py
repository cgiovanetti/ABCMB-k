"""toy_decoupling.py — a 3-parameter toy with a decoupling nuisance, for the flat-direction flags (no ABCMB).

  chi2(a, m, c) = ((a - A exp(-m)) / s)^2 + Bc (c - c0)^2

m acts only through eps = A exp(-m), which fades out as m grows (as m_chi does above ~40 MeV in the EM-WIMP
model, TODO item 20). At a POI value a > 0 the profile minimum sits at m = ln(A / a) (chi2 0); at a <= 0 the
data want eps < 0, so m runs to the box face and chi2 is flat along it there: every start that converges on
that plateau is the same minimum, wherever along m it stops. A start toward small m climbs a wall (eps grows
as e^-m), which the Gauss-Newton quadratic of the warm point does not see. pytests/test_profile_toy.py runs the
driver on it: no flags and no branches from the plateau, spread starts pulled back off the wall.
"""
import numpy as np
import jax.numpy as jnp

A, S, BC, C0 = 2.0, 1.0, 20.0, 0.3


def chi2(theta):
    a, m, c = theta[0], theta[1], theta[2]
    return ((a - A * jnp.exp(-m)) / S) ** 2 + BC * (c - C0) ** 2


def profile_exact(a, lo=-10.0, hi=10.0):
    """chi2 profiled over (m, c) with m in [lo, hi]: eps = A e^-m in [A e^-hi, A e^-lo]."""
    a = np.atleast_1d(np.asarray(a, float))
    eps = np.clip(a, A * np.exp(-hi), A * np.exp(-lo))
    return ((a - eps) / S) ** 2


CONFIG = {
    "order": ["a", "m", "c"],
    "cen": {"a": 0.0, "m": 0.0, "c": C0},
    "sig": {"a": 1.0, "m": 1.0, "c": 0.3},
    "fixed": {},
    "user_species": None,
    "nsig": 2.0,
    "toy": {"chi2": chi2},
}

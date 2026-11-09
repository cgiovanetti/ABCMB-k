"""toy_bounded.py — a 4-parameter toy objective for physical bounds, one-sided intervals and the optimizer box
(no ABCMB; pytests/test_profile_bounds.py).

  chi2(m, n, u, c) = m^2                     m >= 0, best fit on the bound: the 68 / 95% upper limits of the
                                             boundary-aware (Feldman-Cousins in this Gaussian limit) intervals
                                             are 1.000 / 1.960
                   + (n + 1)^2               n >= 0, the unconstrained best fit -1 lies outside: the limits
                                             solve Phi(sqrt(n^2 + 2n)) + Phi(n + 1) - 1 = CL
                   + U_SLOPE exp(-u)         no bound; chi2 falls toward large u and flattens: profiled as a
                                             POI it is open above (a lower limit, or unconstrained when
                                             U_SLOPE is too small to reach the threshold on the grid)
                   + BC (c - C0)^2           a stiff spectator

The config box holds u at u <= 0 (its cen), so every row with u free sits on that box face with outward
slope U_SLOPE per sigma: converged when U_SLOPE < PA_GTOL (chi2 flat there), held by the box otherwise.
U_SLOPE is read from the environment (TOY_U_SLOPE, default 0.01) when the driver loads the config.
"""
import os
import jax.numpy as jnp

U_SLOPE = float(os.environ.get("TOY_U_SLOPE", "0.01"))
BC, C0 = 20.0, 0.3


def chi2(theta):
    m, n, u, c = theta[0], theta[1], theta[2], theta[3]
    return m ** 2 + (n + 1.0) ** 2 + U_SLOPE * jnp.exp(-u) + BC * (c - C0) ** 2


CONFIG = {
    "order": ["m", "n", "u", "c"],
    "cen": {"m": 0.5, "n": 0.5, "u": 0.0, "c": C0},
    "sig": {"m": 1.0, "n": 1.0, "u": 1.0, "c": 0.3},
    "bounds": {"m": (0.0, None), "n": (0.0, None)},
    "box": {"u": (None, 0.0)},
    "fixed": {},
    "user_species": None,
    "nsig": 3.0,
    "toy": {"chi2": chi2},
}

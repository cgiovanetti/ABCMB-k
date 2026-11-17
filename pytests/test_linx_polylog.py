"""The vendored LINX polylogarithm (abcmb/linx/special_funcs.Li, TODO item 35): values are the series' unchanged, its
derivatives follow dLi_n/dz = Li_{n-1}/z, and a zero tangent gives a zero derivative at every order at z = 1, so
second derivatives through LINX's background (whose vector field already holds grad(rho_EM)) are finite. CPU; the
suite's jax_debug_nans also checks that no NaN is formed on the way."""
import os, sys
import numpy as np
import pytest
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
jax.config.update("jax_enable_x64", True)
from abcmb.linx.special_funcs import Li, _Li_series
from abcmb.linx import thermo


def d(f, x, k):
    """k-th derivative of f at x by nested jvp."""
    for _ in range(k):
        f = (lambda h: (lambda y: jax.jvp(h, (y,), (jnp.ones_like(y),))[1]))(f)
    return f(x)


ZS = np.array([-2.7, -1.0, -0.8, -0.3, 0.2, 0.7, 0.95, 1.05, 1.6, 2.6])
# where the series itself is accurate to <~1e-11 (on the negative axis in (-1.5, -0.5) and at |z| ~ 2.6 its values
# are good only to 1e-8..2e-6, which derivatives built on it inherit; scan/diag/li_series_accuracy.py)
Z_ACC = np.array([-0.3, 0.2, 0.7, 0.95, 1.05, 1.6])


@pytest.mark.parametrize("n", [2, 3, 4])
def test_values_are_the_series(n):
    assert np.array_equal(np.asarray(jax.vmap(lambda z: Li(n, z))(ZS)), np.asarray(jax.vmap(lambda z: _Li_series(n, z))(ZS)))


@pytest.mark.parametrize("n,k", [(2, 1), (2, 2), (3, 2), (4, 3)])
def test_derivatives_match_mpmath(n, k):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    for z in Z_ACC:
        ref = float(mp.re(mp.diff(lambda u: mp.polylog(n, u), mp.mpf(z), k)))
        assert abs(float(d(lambda u: Li(n, u), jnp.asarray(z), k)) - ref) <= 1e-9 * abs(ref), (n, k, z)


@pytest.mark.parametrize("mu", [0.0, jnp.zeros(())], ids=["concrete", "traced"])
def test_bose_einstein_at_mu_zero(mu):
    """rho = g pi^2/30 T^4 at mu = 0: its third T-derivative (z = 1, zero tangent) is finite and exact."""
    T0, g = 1.3, 2.0
    for k, exact in ((1, 4 * T0 ** 3), (2, 12 * T0 ** 2), (3, 24 * T0)):
        v = float(jax.jit(lambda T, m: d(lambda t: thermo.rho_massless_BE(t, m, g), T, k))(T0, mu))
        assert v == pytest.approx(g * np.pi ** 2 / 30 * exact, rel=1e-12)


def test_nonzero_tangent_at_one():
    """d^k/dmu^k Li_4(e^mu) at mu = 0 = Li_{4-k}(1): zeta(3), zeta(2); the third (Li_1(1), divergent) comes out
    finite, the price of a tangent rule linear in the tangent (reverse mode needs it)."""
    f = lambda m: Li(4, jnp.exp(m))
    assert float(d(f, jnp.zeros(()), 1)) == pytest.approx(1.2020569031595942, rel=1e-13)
    assert float(d(f, jnp.zeros(()), 2)) == pytest.approx(np.pi ** 2 / 6, rel=1e-13)
    assert np.isfinite(float(d(f, jnp.zeros(()), 3)))


def test_reverse_mode_and_forward_over_reverse():
    """LINX's dY holds jax.grad(rho_EM_std): the rule must transpose, and forward over it must stay finite at mu = 0."""
    g = jax.grad(lambda T: thermo.rho_massless_BE(T, 0., 2.0))
    T0 = jnp.asarray(1.3)
    assert float(g(T0)) == pytest.approx(2.0 * np.pi ** 2 / 30 * 4 * 1.3 ** 3, rel=1e-12)
    assert float(d(g, T0, 2)) == pytest.approx(2.0 * np.pi ** 2 / 30 * 24 * 1.3, rel=1e-12)
    assert float(jax.grad(jax.grad(lambda m: Li(4, jnp.exp(m))))(jnp.zeros(()))) == pytest.approx(np.pi ** 2 / 6,
                                                                                                   rel=1e-13)


def test_background_vector_field_second_derivatives_are_finite():
    """LINX's background dY (with an EM-coupled chi) differentiated twice along its state and along m_chi: finite."""
    from abcmb.linx.background import BackgroundModel
    bg = BackgroundModel(g_chi=4.0)
    Tg = jnp.asarray([8.0, 1.0, 0.1])
    f = lambda Y, m: jnp.stack(bg.dY(0.0, (Y[0], Y[1], Y[2]), (0.0, 0.5, m)))
    for T in Tg:
        Y = jnp.stack([jnp.log(10.0 / T), T, 0.99 * T])
        for v in jnp.eye(3):
            out = jax.jvp(lambda y: jax.jvp(lambda u: f(u, 20.0), (y,), (v,))[1], (Y,), (v,))[1]
            assert np.isfinite(np.asarray(out)).all(), (float(T), v)
        out = d(lambda m: f(Y, m), jnp.asarray(20.0), 2)
        assert np.isfinite(np.asarray(out)).all()

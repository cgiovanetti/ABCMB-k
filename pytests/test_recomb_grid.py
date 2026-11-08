"""The "recomb" lna grid (perturbations.recomb_grid_base / recomb_grid_weights): strictly increasing nodes on
[start, 0], nodes outside the moving window fixed, and Jacobian weights that integrate a visibility-like
bump to high accuracy wherever the core is shifted. CPU, no Model."""
import numpy as np
import jax.numpy as jnp

from abcmb import perturbations as pt

N = 300
CFG = pt.recomb_grid_cfg({})
START, MU0 = CFG[0], CFG[1]


def _bump_integral(lna, w, mu, sig=0.07):
    f = np.exp(-0.5 * ((np.asarray(lna) - mu) / sig) ** 2)
    return float(np.sum(np.asarray(w) * f)), sig * np.sqrt(2. * np.pi)


def test_base_grid():
    t0, w0, b, bp, jc = pt.recomb_grid_base(N, CFG)
    assert t0.shape == (N,) and t0[0] == START and t0[-1] == 0. and np.all(np.diff(t0) > 0)
    assert b[jc] == 1. and b[0] == 0. and b[-1] == 0.
    got, exact = _bump_integral(t0, w0, MU0)
    assert abs(got / exact - 1) < 1e-10


def test_shifted_core():
    t0, w0, b, bp, jc = pt.recomb_grid_base(N, CFG)
    for dmu in (-0.15, -0.02, 0.03, 0.15):
        lna = jnp.asarray(t0) + dmu * jnp.asarray(b)
        assert np.all(np.diff(np.asarray(lna)) > 0)
        assert np.array_equal(np.asarray(lna)[b == 0.], t0[b == 0.])          # nodes outside the window stay
        w = pt.recomb_grid_weights(lna, N, CFG)
        np.testing.assert_allclose(float(lna[jc] - t0[jc]), dmu, rtol=0, atol=1e-14)
        got, exact = _bump_integral(lna, w, MU0 + dmu)
        assert abs(got / exact - 1) < 1e-8, (dmu, got / exact - 1)

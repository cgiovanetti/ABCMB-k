"""LowLEE (scan/lowl_like.py) against cobaya's native low-ell EE likelihoods, for both tables:
planck_2018_lowl.EE (SimAll, the Planck 2018 lowE) and planck_2018_lowl.EE_sroll2. On the table nodes the
cubic interpolant must return cobaya's floor-lookup value. CPU."""
import numpy as np
import pytest
import jax.numpy as jnp
from cobaya.likelihoods.planck_2018_lowl.EE import EE
from cobaya.likelihoods.planck_2018_lowl.EE_sroll2 import EE_sroll2
from scan.lowl_like import LowLEE, LOWEE_TABLES


def _cobaya(cls, name):
    ref = cls.__new__(cls)                      # log_likelihood only needs the table and the class constants
    ref.probEE = np.loadtxt(LOWEE_TABLES[name])
    return ref


@pytest.mark.parametrize("name, cls", [("simall", EE), ("sroll2", EE_sroll2)])
def test_lowee_matches_cobaya_on_table_nodes(name, cls):
    ref, ours = _cobaya(cls, name), LowLEE(LOWEE_TABLES[name])
    rng = np.random.default_rng(1)
    peak = ref.probEE.argmax(0)                 # per-ell most likely table row
    for _ in range(8):
        j = np.clip(peak + rng.integers(-300, 300, size=peak.shape), 0, ref._nstepsEE - 3)
        D = np.zeros(40)
        D[2:30] = (j + 1e-6) * ref._stepEE      # just above node j, so cobaya's floor lands on j
        lnl_ref = ref.log_likelihood(D)
        lnl = -0.5 * float(ours.chi2(jnp.asarray(D)))
        assert abs(lnl - lnl_ref) < 1e-6, (name, lnl, lnl_ref)


def test_tables_differ():
    a, b = np.loadtxt(LOWEE_TABLES["simall"]), np.loadtxt(LOWEE_TABLES["sroll2"])
    assert a.shape == b.shape == (3000, 28)
    assert not np.allclose(a, b)

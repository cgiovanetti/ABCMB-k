"""Physical bounds, one-sided intervals and the optimizer box, end to end on scan/configs/toy_bounded.py
(CPU, no ABCMB). Expected values come from closed forms, not from the driver: the boundary-aware
(Feldman-Cousins in the Gaussian limit) upper limits 1.000 / 1.960 of a best fit on the bound, the limit of a
best fit one sigma outside, the crossing of the open-ended exponential, and the box rule (a row on a box face
converges when chi2 is flat outward, is held by the box when chi2 still falls)."""
import importlib
import os, sys
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import chi2 as chi2_dist, norm

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
CL1, CL95 = 0.682689492137, 0.95


def _driver(u_slope):
    os.environ.update(PA_CONFIG=os.path.join(ROOT, "scan", "configs", "toy_bounded.py"), PA_NPTS="21", PA_XBOX="5",
                      PA_MS_K="0", PA_TAG="_tb", PA_RESUME="0", PA_CPU_DEVICES="1", JAX_PLATFORM_NAME="cpu",
                      TOY_U_SLOPE=str(u_slope))
    import scan.profile_prod_ad as d
    return importlib.reload(d)                      # the driver reads its config at import


def _run(d, outdir, pois):
    d.profile_lockstep(pois, outdir)
    return {p: np.load(os.path.join(outdir, f"profile_prod_ad_{p}_tb.npz")) for p in pois}


def _n_limit(cl):
    """upper limit on n for chi2 = (n + 1)^2, n >= 0 (t~ = n^2 + 2n, sigma 1): F(t~; n) = cl."""
    return brentq(lambda n: ndtr(np.sqrt(n * n + 2 * n)) + ndtr(n + 1) - 1 - cl, 1e-6, 5)


@pytest.fixture(scope="module")
def flat(tmp_path_factory):
    d = _driver(0.01)
    return d, _run(d, str(tmp_path_factory.mktemp("flat")), ["m", "n", "u"])


@pytest.fixture(scope="module")
def falling(tmp_path_factory):
    d = _driver(0.5)
    return d, _run(d, str(tmp_path_factory.mktemp("falling")), ["m", "u"])


def test_threshold_limits():
    d = _driver(0.01)
    assert abs(float(d.t_threshold(0.95, 0.0)) - norm.ppf(0.95) ** 2) < 1e-9      # on the bound: 2.71
    assert abs(float(d.t_threshold(0.95, 20.0)) - chi2_dist.ppf(0.95, 1)) < 1e-12  # far away: 3.84


def test_best_fit_on_the_bound_gives_the_unified_upper_limits(flat):
    _, r = flat
    m = r["m"]
    assert m["poi_grid"][0] == 0.0 and abs(m["poi_grid"][-1] - 6.0) < 1e-12      # shifted onto the bound
    assert tuple(m["ends1"]) == ("bound", "crossing") and str(m["kind1"]) == "upper limit"
    assert str(m["kind95"]) == "upper limit"
    assert abs(m["sigma1"][2] - 1.0) < 2e-3 and abs(m["cl95"][1] - 1.960) < 3e-3, (m["sigma1"], m["cl95"])
    assert np.nanmax(m["bound"]) == 0.0 and np.isnan(m["bound"][1])


def test_best_fit_outside_gives_a_shorter_limit(flat):
    _, r = flat
    n = r["n"]
    assert str(n["kind1"]) == "upper limit" and str(n["kind95"]) == "upper limit"
    assert abs(n["sigma1"][2] - _n_limit(CL1)) < 3e-3, (n["sigma1"], _n_limit(CL1))
    assert abs(n["cl95"][1] - _n_limit(CL95)) < 3e-3, (n["cl95"], _n_limit(CL95))


def test_flat_box_face_converges_and_a_flat_poi_is_unconstrained(flat):
    _, r = flat
    for p in ("m", "n"):                         # u sits on its box face with outward slope 0.01 < GTOL
        assert r[p]["converged"].all() and not r[p]["at_box"].any()
        assert str(r[p]["stop_reason"]) == "converged"
    u = r["u"]                                   # Delta chi2 <= 0.2 over the grid
    assert tuple(u["ends1"]) == ("open", "open") and str(u["kind1"]) == "unconstrained"


def test_box_holds_a_falling_direction_and_an_open_poi_is_a_lower_limit(falling):
    _, r = falling
    m = r["m"]                                   # u on its box face with outward slope 0.5 > GTOL
    assert m["at_box"].all() and not m["converged"].any()
    assert abs(m["sigma1"][2] - 1.0) < 2e-3      # a constant offset: the m profile is unchanged
    u = r["u"]
    assert tuple(u["ends1"]) == ("crossing", "open") and str(u["kind1"]) == "lower limit"
    assert str(u["kind95"]) == "lower limit"
    lo1 = -np.log(chi2_dist.ppf(CL1, 1) / 0.5 + np.exp(-3.0))
    lo95 = -np.log(chi2_dist.ppf(CL95, 1) / 0.5 + np.exp(-3.0))
    assert abs(u["sigma1"][0] - lo1) < 2e-3 and abs(u["cl95"][0] - lo95) < 2e-3, (u["sigma1"], u["cl95"])
    assert u["converged"].all()                  # u is the POI: nothing free sits on a box face


def test_2d_profile_matches_the_exact_surface(tmp_path):
    """PA_POI2D path: chi2 profiled over (u, c) on an (m, n) grid is m^2 + (n + 1)^2 + 0.01 exactly (u on its flat
    box face at 0, c at C0), every row converged, nothing held by the box, both grids inside the bounds."""
    d = _driver(0.01)
    d.NPTS2D = 5
    d.profile_2d("m", "n", str(tmp_path))
    r = np.load(d._npz2d(str(tmp_path), "m", "n"))
    ga, gb = r["grid_a"], r["grid_b"]
    assert ga[0] == 0.0 and gb[0] == 0.0 and r["chi2"].shape == (5, 5)
    exact = ga[:, None] ** 2 + (gb[None, :] + 1.0) ** 2 + 0.01
    assert np.abs(r["chi2"] - exact).max() < 1e-6, r["chi2"] - exact
    assert r["converged"].all() and not r["at_box"].any() and str(r["stop_reason"]) == "converged"


def test_2d_extra_row_from_the_environment(tmp_path):
    """PA_GRID2D (explicit values) overrides one axis of the 2D grid: a single extra row, and the signature changes."""
    d = _driver(0.01)
    sig0 = d._run_signature()
    os.environ["PA_POI2D"], os.environ["PA_GRID2D"] = "m,n", '{"n": [0.25]}'
    try:
        d = importlib.reload(d)
        d.NPTS2D = 5
        assert d._run_signature() != sig0
        d.profile_2d("m", "n", str(tmp_path))
        r = np.load(d._npz2d(str(tmp_path), "m", "n"))
        assert r["chi2"].shape == (5, 1) and r["grid_b"][0] == 0.25
        assert np.abs(r["chi2"][:, 0] - (r["grid_a"] ** 2 + 1.25 ** 2 + 0.01)).max() < 1e-6
    finally:
        del os.environ["PA_POI2D"], os.environ["PA_GRID2D"]
        importlib.reload(d)

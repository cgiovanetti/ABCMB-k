"""The profile driver end to end on scan/configs/toy_decoupling.py (CPU, no ABCMB): a nuisance m whose effect
A e^-m fades out (TODO item 20). Where the data push m onto its plateau the spread starts converge at different m
with the same chi2: one flat valley, so no flags and no branches (without the barrier test each such point was
"distinct minima (dchi2 0.000)" and seeded a branch). Spread starts toward small m climb a wall the warm
Gauss-Newton quadratic does not see and are pulled back to at most PA_MS_START_DCHI2 above their row's start."""
import importlib
import os, sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    os.environ.update(PA_CONFIG=os.path.join(ROOT, "scan", "configs", "toy_decoupling.py"), PA_XBOX="10",
                      PA_NPTS="21", PA_MS_K="4", PA_MS_SPREAD="2.0", PA_MS_DCHI2="0.1", PA_TAG="_td", PA_RESUME="0",
                      PA_CPU_DEVICES="1", JAX_PLATFORM_NAME="cpu")
    import scan.profile_prod_ad as d
    d = importlib.reload(d)                         # the driver reads its config at import
    out = str(tmp_path_factory.mktemp("decoupling"))
    d.profile_lockstep(["a"], out)
    return d, np.load(os.path.join(out, "profile_prod_ad_a_td.npz"))


def test_plateau_raises_no_flags_and_no_branches(result):
    from scan.configs.toy_decoupling import profile_exact
    d, r = result
    grid, chi2 = r["poi_grid"], r["chi2"]
    # on the plateau (a < 0) a row stops at |g_m| < GTOL, and there chi2 - a^2 ~ 2|a| A e^-m = |g_m| (one e-fold per
    # scaled sigma): the profile sits at most GTOL above the exact one; elsewhere it is exact
    dev = chi2 - profile_exact(grid)
    assert dev.min() > -1e-9 and dev.max() < d.GTOL, np.c_[grid, chi2, profile_exact(grid)]
    assert np.abs(dev[grid > 0.1]).max() < 2e-3
    assert r["converged"].all() and not r["stalled"].any()
    kind, j, x = r["rows_kind"], r["rows_j"], r["rows_x"]
    im = list(r["nuis"]).index("m")
    # at a < 0 the converged starts sit far apart along m (the plateau) at one chi2 ...
    left = [jj for jj in range(len(grid)) if grid[jj] < -0.1 and (kind[j == jj] == 1).any()]
    assert left
    spread_m = max(np.ptp(x[(j == jj) & r["rows_converged"], im]) for jj in left)
    assert spread_m > d.MS_DX, spread_m
    # ... and count as one minimum: no flag, no branch
    assert not r["flag"].any(), r["flag_reason"]
    assert (r["n_minima"] == 1).all() and (kind == 2).sum() == 0


def test_spread_starts_pulled_back_off_the_wall(result):
    from scan.configs.toy_decoupling import chi2 as toy_chi2
    d, r = result
    kind, j = r["rows_kind"], r["rows_j"]
    th = lambda b, x: np.array(d.theta_of(np.r_[r["poi_grid"][j[b]], 0.0, 0.0], np.array([1, 2]), x))
    f0 = {jj: float(toy_chi2(th(b, r["rows_x0"][b]))) for b in np.where(kind == 0)[0] for jj in [j[b]]}
    rise = [float(toy_chi2(th(b, r["rows_x0"][b]))) - f0[j[b]] for b in np.where(kind == 1)[0]]
    assert max(rise) <= d.MS_START_DCHI2 + 1e-9, max(rise)
    # the wall is real: the unshortened spread along the softest warm direction would have gone far up it
    xm = r["rows_x0"][kind == 1][:, list(r["nuis"]).index("m")]
    assert xm.min() < -0.5 and xm.min() > -10.0

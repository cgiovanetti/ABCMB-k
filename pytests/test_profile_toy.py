"""The profile driver end to end on the toy bimodal objective of scan/configs/toy_bimodal.py (CPU, no
ABCMB): the spread starts find the second well at the left grid points, those points are flagged, the
branch carries the second minimum to every grid point, the lower envelope matches the exact profile
and the 68% interval comes out as a union of two regions. Without the
multistart the driver follows the warm start's branch alone and is wrong left of the crossing."""
import os, sys
import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

ENV = dict(PA_CONFIG=os.path.join(ROOT, "scan", "configs", "toy_bimodal.py"), PA_XBOX="10", PA_NPTS="21",
           PA_MS_SPREAD="2.5", PA_TAG="_toy", PA_RESUME="0", PA_CPU_DEVICES="1", JAX_PLATFORM_NAME="cpu")


@pytest.fixture(scope="module")
def driver():
    os.environ.update(ENV)
    import scan.profile_prod_ad as d
    return d


def _run(d, outdir, **overrides):
    saved = {k: getattr(d, k) for k in overrides}
    for k, v in overrides.items():
        setattr(d, k, v)
    try:
        d.profile_lockstep(["a"], outdir)
    finally:
        for k, v in saved.items():
            setattr(d, k, v)
    return np.load(os.path.join(outdir, "profile_prod_ad_a_toy.npz"))


def test_multistart_branches_recover_the_bimodal_profile(driver, tmp_path):
    from scan.configs.toy_bimodal import profile_exact, branch_exact, DELTA
    d = driver
    r = _run(d, str(tmp_path))
    grid, chi2 = r["poi_grid"], r["chi2"]
    exact = profile_exact(grid)
    assert np.abs(chi2 - exact).max() < 2e-3, np.c_[grid, chi2, exact]
    assert r["converged"].all() and not r["stalled"].any()
    # the warm-start rows alone are the b > 0 branch: wrong by > 1 left of the crossing
    kind, j = r["rows_kind"], r["rows_j"]
    warm = np.array([r["rows_chi2"][(kind == 0) & (j == jj)].min() for jj in range(len(grid))])
    assert np.abs(warm - branch_exact(grid, +1)).max() < 2e-3
    assert (warm - exact)[grid < -0.5].min() > 1.0
    # spread starts found the second well at some left grid points -> flags -> a branch (one seed, one
    # row per grid point; a further seed would add another full set)
    assert (kind == 1).sum() == 3 * 6
    nb = int((kind == 2).sum())
    assert nb >= len(grid) and nb % len(grid) == 0, nb
    # every grid point now knows both wells: the lowest one is the second well on the left (found by a
    # spread start or the branch, both below the warm branch by > 1) and the warm one on the right
    assert (r["n_minima"] >= 2).all() and r["flag"].all()
    assert all("distinct minima" in s for s in r["flag_reason"])
    left = grid < -0.11                                   # the branches cross at a = -0.114
    assert (r["minima_chi2"][left, 0] < warm[left] - 0.1).all()
    assert np.abs(r["minima_chi2"][~left, 0] - warm[~left]).max() < 2e-3
    assert set(r["minima_kind"][left][:, 0]) <= {"spread", "branch"}
    # intervals: a union of two regions at 68%, one region at 95% whose upper end runs off the grid
    # (NaN by design), pchip because the points are bimodal
    assert str(r["interval_mode"]) == "pchip"
    reg1, reg95 = r["regions1"], r["regions95"]
    assert reg1.shape == (2, 2) and reg95.shape == (1, 2), (reg1, reg95)
    assert np.allclose(reg1[1], [DELTA - 1.0, DELTA + 1.0], atol=0.1), reg1
    assert np.allclose(reg1[0], [-DELTA - np.sqrt(0.5), -DELTA + np.sqrt(0.5)], atol=0.1), reg1
    # best = the lowest node (pchip never undershoots its nodes): 1.2 on this grid, next to delta = 1.1
    assert np.allclose(r["sigma1"], [reg1[0, 0], DELTA, reg1[1, 1]], atol=0.16), r["sigma1"]
    assert abs(reg95[0, 0] - (-DELTA - np.sqrt(3.841 - 0.5))) < 0.1 and np.isnan(reg95[0, 1]), reg95


def test_without_multistart_the_wrong_branch_is_followed(driver, tmp_path):
    from scan.configs.toy_bimodal import profile_exact, branch_exact
    d = driver
    r = _run(d, str(tmp_path), MS_K=0)
    grid, chi2 = r["poi_grid"], r["chi2"]
    assert np.abs(chi2 - branch_exact(grid, +1)).max() < 2e-3
    assert (chi2 - profile_exact(grid))[grid < -0.5].min() > 1.0
    assert r["n_starts"].tolist() == [1] * len(grid) and not r["flag"].any()
    assert r["regions1"].shape == (1, 2)

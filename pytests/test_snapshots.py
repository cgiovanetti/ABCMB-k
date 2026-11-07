"""GPU-generated regression oracle: every scenario in fixtures/snapshots.npz is reproduced at rtol 5e-5.
Run on GPU, where the snapshots were generated. Skipped when snapshots.npz is absent."""

import os
import numpy as np
import pytest

from pytests.fixtures.scenarios import SCENARIOS, SCENARIO_NAMES

SNAPSHOTS_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'fixtures', 'snapshots.npz')


def load_snapshots():
    """Returns the npz handle (mmap'd). Keys are '<scenario>__<field>'."""
    if not os.path.exists(SNAPSHOTS_NPZ):
        raise FileNotFoundError(
            f"Snapshot file not found: {SNAPSHOTS_NPZ}. "
            f"Run pytests/fixtures/generate_snapshots.py to create it."
        )
    return np.load(SNAPSHOTS_NPZ, allow_pickle=True)


def assert_matches_snapshot(scenario, fields, rtol, atol):
    """Compare a dict of {field_name: array} to the snapshot keys '<scenario>__<field>'
    with np.testing.assert_allclose; report every failing field at once."""
    snap = load_snapshots()
    failures = []
    for fname, value in fields.items():
        key = f"{scenario}__{fname}"
        if key not in snap.files:
            failures.append(f"  missing key {key!r} in snapshot")
            continue
        ref = snap[key]
        cur = np.asarray(value)
        if cur.shape != ref.shape:
            failures.append(
                f"  {fname}: shape mismatch {cur.shape} vs ref {ref.shape}")
            continue
        diff = np.abs(cur - ref)
        ref_abs = np.abs(ref)
        # relative error where ref != 0, absolute elsewhere
        rel = np.where(ref_abs > 0, diff / np.maximum(ref_abs, 1e-300), 0.)
        max_rel = float(rel.max())
        max_abs = float(diff.max())
        try:
            np.testing.assert_allclose(cur, ref, rtol=rtol, atol=atol)
        except AssertionError as e:
            failures.append(
                f"  {fname}: max_rel={max_rel:.2e} max_abs={max_abs:.2e}\n"
                f"    rtol={rtol:.0e} atol={atol:.0e}\n"
                f"    {str(e).splitlines()[0]}")
    if failures:
        msg = f"Snapshot mismatch for scenario {scenario!r}:\n" + "\n".join(failures)
        raise AssertionError(msg)


pytestmark = pytest.mark.skipif(
    not os.path.exists(SNAPSHOTS_NPZ),
    reason=("snapshots.npz not present; run "
            "pytests/fixtures/generate_snapshots.py first"),
)


def _present_scenarios():
    """Read the manifest from snapshots.npz so the test parametrizes over
    exactly the scenarios that were generated (LINX may have been skipped)."""
    if not os.path.exists(SNAPSHOTS_NPZ):
        return []
    snap = load_snapshots()
    if '_scenarios' in snap.files:
        return list(snap['_scenarios'])
    # fallback: infer from keys
    return sorted({k.split('__', 1)[0] for k in snap.files if '__' in k})


SCENARIOS_TO_TEST = _present_scenarios()


@pytest.mark.parametrize("scenario", SCENARIOS_TO_TEST)
def test_snapshot_reproduces(scenario):
    """Re-run the scenario with the current code and compare to the stored snapshot."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from abcmb.main import Model

    scen = SCENARIOS[scenario]
    specs = scen["specs"]()
    user_species = scen["user_species"]()
    params = scen["params"]()

    model = Model(user_species=user_species, **specs)
    out = model(params)

    fields = {
        'ClTT': out.ClTT, 'ClTE': out.ClTE, 'ClEE': out.ClEE,
        'Pk':   out.Pk,   'l':    out.l,    'k':    out.k,
    }
    # rtol=5e-5 (atol=1e-18 swallows near-zero Cls at low ell / TE crossings) covers
    # cross-process XLA reduction-order drift (~1.2e-5 on Pk for the BBN scenarios) and
    # stays under rtol_large_k_PE=1e-4, so a real regression (>>1e-3) still fails.
    assert_matches_snapshot(scenario, fields, rtol=5e-5, atol=1e-18)

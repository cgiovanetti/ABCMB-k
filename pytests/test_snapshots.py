"""
Phase C done test: every scenario in snapshots.npz is reproduced by current
code to ~1e-12.

After Phase D+, this same test is what tells us when the refactor breaks
single-eval correctness. The B-axis isn't exercised here — that's parity.py.

Slow because each scenario runs full ABCMB. Skipped automatically if
pytests/fixtures/snapshots.npz is absent (e.g., on first checkout before
generate_snapshots.py has been run).
"""

import os
import pytest

from pytests.snapshots import (
    SNAPSHOTS_NPZ,
    load_snapshots,
    assert_matches_snapshot,
)
from pytests.fixtures.scenarios import SCENARIOS, SCENARIO_NAMES

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
    """Re-run the scenario with the current code and assert bit-precision
    match against the stored snapshot."""
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
    # rtol=1e-10 is intentionally tighter than the eventual refactor
    # parity tolerance (1e-9). Snapshot reproduction on the SAME code
    # path should be bit-precise modulo XLA scheduler nondeterminism.
    assert_matches_snapshot(scenario, fields, rtol=1e-10, atol=0.0)

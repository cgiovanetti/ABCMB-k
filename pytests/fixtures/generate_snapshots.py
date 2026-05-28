"""
Generate the Phase C snapshot fixture file.

Runs current ABCMB Model on each scenario in scenarios.SCENARIOS, captures
(ClTT, ClTE, ClEE, Pk) plus the l and k axes, and writes a single
pytests/fixtures/snapshots.npz. This is the parity oracle for everything
after Phase C — the refactored code must reproduce these arrays to ~1e-12.

Run on GPU (recommended) or CPU (slow but works):
    module load conda && conda activate actdr6 && \\
        python -u pytests/fixtures/generate_snapshots.py
"""

import os
import sys
import time

# Make the project root importable when invoked as `python path/to/this.py`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from abcmb.main import Model
from pytests.fixtures.scenarios import SCENARIOS, SCENARIO_NAMES

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(FIXTURES_DIR, 'snapshots.npz')


def run_scenario(name):
    scen = SCENARIOS[name]
    specs = scen["specs"]()
    user_species = scen["user_species"]()
    params = scen["params"]()

    print(f"\n[{name}] specs.bbn_type={specs.get('bbn_type', '')!r}  "
          f"input_tau_reion={specs['input_tau_reion']}  "
          f"user_species={user_species}", flush=True)

    t0 = time.perf_counter()
    model = Model(user_species=user_species, **specs)
    out = model(params)
    jax.block_until_ready(out.ClTT)
    jax.block_until_ready(out.ClTE)
    jax.block_until_ready(out.ClEE)
    jax.block_until_ready(out.Pk)
    dt = time.perf_counter() - t0

    arrays = {
        'ClTT': np.asarray(out.ClTT),
        'ClTE': np.asarray(out.ClTE),
        'ClEE': np.asarray(out.ClEE),
        'Pk':   np.asarray(out.Pk),
        'l':    np.asarray(out.l),
        'k':    np.asarray(out.k),
    }
    print(f"[{name}] done in {dt:.1f}s  ClTT[:3]={arrays['ClTT'][:3]}",
          flush=True)
    return arrays


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"jax.default_backend(): {jax.default_backend()}", flush=True)

    target = sys.argv[1:] if len(sys.argv) > 1 else SCENARIO_NAMES
    print(f"Generating scenarios: {target}", flush=True)

    snapshot = {}
    for name in target:
        if name not in SCENARIOS:
            print(f"WARNING: unknown scenario {name!r}, skipping",
                  flush=True)
            continue
        try:
            arrays = run_scenario(name)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        for key, arr in arrays.items():
            snapshot[f"{name}__{key}"] = arr

    snapshot['_scenarios'] = np.array(sorted(set(
        k.split('__', 1)[0] for k in snapshot.keys()
    )), dtype=object)
    print(f"\nWriting {NPZ_PATH} with {len(snapshot)} keys "
          f"({len(snapshot['_scenarios'])} scenarios)...", flush=True)
    np.savez(NPZ_PATH, **snapshot)
    print(f"Done. {NPZ_PATH}", flush=True)


if __name__ == "__main__":
    main()

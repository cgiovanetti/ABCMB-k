"""Diagnose the bbn_table/bbn_linx snapshot failures: is it run-to-run
nondeterminism (amplified by the recomb-concentrated vis-300 grid) or a real shift?

Runs bbn_table TWICE in one process (isolates nondeterminism) and compares both to
the committed snapshots.npz. Prints max rel diff for ClTT/ClEE/Pk.
"""
import os, sys, numpy as np
import jax
jax.config.update("jax_enable_x64", True)
_HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(_HERE))
import abcmb
assert "ABCMB-k" in abcmb.__file__, abcmb.__file__
from abcmb.main import Model
from pytests.fixtures.scenarios import SCENARIOS

def run(name):
    sc = SCENARIOS[name]
    m = Model(user_species=sc["user_species"](), **sc["specs"]())
    o = m(sc["params"]()); jax.block_until_ready(o.ClTT)
    return np.asarray(o.ClTT), np.asarray(o.ClEE), np.asarray(o.Pk)

def mx(a,b): return float(np.max(np.abs(a-b)/np.maximum(np.abs(b),1e-300)))

snap = np.load("pytests/fixtures/snapshots.npz")
for name in ("bbn_table",):
    tt1,ee1,pk1 = run(name)
    tt2,ee2,pk2 = run(name)
    print(f"## {name}", flush=True)
    print(f"  run1-vs-run2 (nondeterminism): TT {mx(tt1,tt2):.2e}  EE {mx(ee1,ee2):.2e}  Pk {mx(pk1,pk2):.2e}", flush=True)
    print(f"  run1-vs-npz:                   TT {mx(tt1,snap[name+'__ClTT']):.2e}  EE {mx(ee1,snap[name+'__ClEE']):.2e}  Pk {mx(pk1,snap[name+'__Pk']):.2e}", flush=True)

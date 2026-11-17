"""TODO item 35: LINX programs with the custom-jvp polylog ("new") vs the series differentiated directly ("old", LINX
as shipped). One cosmology (row 0 of logs/linx_curv_x.npy, m_chi 23.6 MeV), single-row programs, CPU.
    python scan/diag/li_linx_timing.py old|new   -> logs/li_linx_<mode>.npz"""
import os, sys, time
import numpy as np
os.environ.setdefault("EQX_ON_ERROR", "nan")
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
MODE = sys.argv[1]
from abcmb.linx import thermo, special_funcs
if MODE == "old":
    thermo.Li = special_funcs._Li_series          # thermo's BE/FD densities on the series' own derivatives
from scan.bbn_stage import LinxStage

S = LinxStage("key_recommended", g_chi=4.0, curv=("log10_m_chi",))
x = jnp.asarray(np.load("logs/linx_curv_x.npy")[0])
out = {}
progs = [("values", jax.jit(S._one_x)), ("jac", jax.jit(S._one_vj))] + ([("hess", jax.jit(S._one_vjk))] if MODE == "new" else [])
for name, f in progs:
    t = time.perf_counter(); r = jax.block_until_ready(f(x)); tc = time.perf_counter() - t
    t = time.perf_counter(); r = jax.block_until_ready(f(x)); tw = time.perf_counter() - t
    t = time.perf_counter(); r = jax.block_until_ready(f(x)); tw = min(tw, time.perf_counter() - t)
    r = r if isinstance(r, tuple) else (r,)
    for k, a in enumerate(r):
        out[f"{name}{k}"] = np.asarray(a)
    print(f"[{MODE}] {name}: first call {tc:.0f} s, warm {tw:.2f} s; finite {all(np.isfinite(np.asarray(a)).all() for a in r)}",
          flush=True)
np.savez(f"logs/li_linx_{MODE}.npz", **out)

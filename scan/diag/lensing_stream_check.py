"""lensing_stream_check.py — lensing_streamed=True gives the same lensed spectra as the matrix path.

  python scan/diag/lensing_stream_check.py <cfg>   # cfg from act_lmax_check, or "planck" (l_max 2508, tabulated j_l)
Value path, 4 cosmologies sharded over the visible GPUs; prints max |streamed / matrix - 1| per spectrum, timings, peaks.
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
from scan.diag.act_lmax_check import COMMON, CFGS, FIXED, cosmologies

cfg = sys.argv[1]
if cfg == "planck":
    spec = dict(COMMON, l_max=2508, l_max_g=12, l_max_pol_g=10, use_bessel_tables=True)
else:
    spec = dict(COMMON, **CFGS[cfg])
plist = [dict(FIXED, omega_b=t[0], omega_cdm=t[1], h=t[2], A_s=np.exp(t[3]) / 1e10, n_s=t[4], tau_reion=t[5])
         for t in cosmologies()[:4]]
out = {}
for c in (False, True):
    m = Model(**spec, lensing_streamed=c)
    for rep in range(2):
        t0 = time.time()
        o = m.call_batched(plist)
        cl = np.stack([np.asarray(o.ClTT), np.asarray(o.ClTE), np.asarray(o.ClEE), np.asarray(o.ClBB)], axis=1)
        dt = time.time() - t0
    peak = max(d.memory_stats().get("peak_bytes_in_use", 0) for d in jax.devices()) / 2**30
    out[c] = cl
    print(f"[{cfg}] streamed {c}: warm {dt:.1f}s, peak so far {peak:.1f} GB", flush=True)
a, b = out[False], out[True]
for i, s in enumerate(("TT", "TE", "EE", "BB")):
    den = np.sqrt(np.abs(a[:, 0] * a[:, 2])) if s == "TE" else np.abs(a[:, i])
    print(f"[{cfg}] {s}: max |streamed - matrix| / ref {np.max(np.abs(b[:, i] - a[:, i]) / den):.2e}", flush=True)

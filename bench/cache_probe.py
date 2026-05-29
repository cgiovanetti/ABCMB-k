"""Persistent-compile-cache probe: does a 2nd fresh process skip the compile?

Run twice with the SAME (fresh) cache dir. First process compiles cold; second
should hit the on-disk cache and skip it. Validates the job-array scaling plan
(compile amortized once across the whole array, not per task).

  python bench/cache_probe.py <cache_dir>

Prints: COLD/WARM call_batched wall time (first call in a fresh process).
"""
import os, sys, time
cache_dir = sys.argv[1]
os.makedirs(cache_dir, exist_ok=True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
from abcmb.main import Model

FID = {'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225, 'A_s': 2.12424e-9,
       'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245, 'TCMB0': 2.34865418e-4,
       'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
       'tau_reion': 0.0544, 'Delta_z_reion': 0.5, 'z_reion_He': 3.5,
       'Delta_z_reion_He': 0.5, 'exp_reion': 1.5}
B = 16
pl = [dict(FID, h=0.66 + 0.001 * i) for i in range(B)]
m = Model(user_species=None, output_Cl=True, l_max=800, lensing=False,
          output_Pk=True, output_k_max=0.5, l_max_g=12, l_max_pol_g=10,
          l_max_ur=17, l_max_ncdm=17)
t0 = time.perf_counter()
out = m.call_batched(pl, shard=False)
jax.block_until_ready(out.ClTT)
dt = time.perf_counter() - t0
print(f"CACHEPROBE first-call(compile-or-cache)={dt:.2f}s  cache_dir={cache_dir}",
      flush=True)

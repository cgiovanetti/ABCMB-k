"""act_stage_times.py — per-stage wall time of the batched value path (call_batched's stages) for one
act_lmax_check config, to see what the ACT l range costs: setup (pre-recomb + HyRex + get_BG), perturbations,
spectrum (LoS recurrence + lensing). Also n_k of the perturbation grid and the state size.

  python scan/diag/act_stage_times.py <cfg> <B>      # B sharded over the visible GPUs; 2 passes, the 2nd is timed
"""
import os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
from scan.diag.act_lmax_check import COMMON, CFGS, FIXED, cosmologies

cfg, B = sys.argv[1], int(sys.argv[2])
model = Model(**COMMON, **CFGS[cfg])
th = np.repeat(cosmologies(), -(-B // 8), axis=0)[:B]
plist = [dict(FIXED, omega_b=t[0], omega_cdm=t[1], h=t[2], A_s=np.exp(t[3]) / 1e10, n_s=t[4], tau_reion=t[5])
         for t in th]
Ny = 1 + sum(int(s.num_equations) for s in model.PE.species_list)
print(f"[{cfg}] B {B}: n_k perturbations {len(model.PE.k_axis_perturbations)} (cmb {model.specs['k_size_cmb']}), "
      f"n_k transfer {len(model.SS.k_axis_transfer)}, Ny {Ny}, l_top {model.SS.l_top}", flush=True)
shardfn, n_dev = model._make_shardfn(None)
for rep in range(2):
    t = [time.time()]
    full_ps = [model.add_derived_parameters(p) for p in plist]
    params_batch, BG = model._build_bgs_batched(full_ps, shardfn=shardfn)
    jax.block_until_ready(BG); t.append(time.time())
    PT = model.PE.full_evolution_batched((BG, params_batch), k_chunk_size=100)
    jax.block_until_ready(PT); t.append(time.time())
    cls = model.SS.get_Cl_batched(PT, BG, params_batch)
    jax.block_until_ready(cls); t.append(time.time())
    d = np.diff(t)
    print(f"[{cfg}] pass {rep}: setup {d[0]:.1f}s, perturbations {d[1]:.1f}s, spectrum {d[2]:.1f}s; "
          f"per cosmology per device {np.sum(d) * n_dev / B:.1f}s", flush=True)
peaks = [dv.memory_stats().get("peak_bytes_in_use", 0) / 2**30 for dv in jax.devices()]
print(f"[{cfg}] peak GB {max(peaks):.1f}", flush=True)

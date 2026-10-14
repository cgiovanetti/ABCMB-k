"""Name the big buffer in the spectrum-stage jvp: run the primal pipeline to stage 4 at B
cosmologies, then lower+compile the P=7-direction vmapped jvp of SS.get_Cl_batched alone and
print XLA's memory analysis plus the largest array shapes in the optimized HLO.
Usage: [LENSING=0] python hlo_stage5.py B"""
import os, sys, re
ROOT = "/pscratch/sd/c/carag/ABCMB-k-grad"
sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import abcmb
assert abcmb.__file__.startswith(ROOT)
from abcmb.main import Model
sys.path.insert(1, os.path.join(ROOT, "scan"))
from batched_grad import jvp_stage, _to_float
B = int(sys.argv[1]); P = 7
ORDER = ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"]
CEN = np.array([0.6662, 0.02229, 0.11654, 0.95986, 3.0446, 0.05933, 2.8268])
SIG = np.array([0.0054, 0.00015, 0.0013, 0.0042, 0.014, 0.0073, 0.1])
FIXED = {"YHe": 0.2454, "TCMB0": 2.34865418e-4, "N_nu_massive": 1, "T_nu_massive": 0.71611,
         "m_nu_massive": 0.06, "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
         "exp_reion": 1.5}
SPECS = dict(output_Cl=True, l_max=2508, lensing=os.environ.get("LENSING", "1") == "1",
             output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
             rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7, rtol_small_k_PE=1e-5, max_steps_PE=16384)
model = Model(**SPECS); SS = model.SS; PE = model.PE
print(f"N_k={len(PE.k_axis_perturbations)} N_kt={len(SS.k_axis_transfer)} n_mu={len(SS.lensing_mus)} "
      f"n_lens_ells={len(SS.lensing_ells)} n_ell_tab={len(SS.lensing_ells_indices)}", flush=True)
z = np.clip(np.random.default_rng(0).standard_normal((max(B, 1), 7)), -2, 2); z[0] = 0.
TH = CEN + SIG * z[:B]
def raw(th):
    p = dict(FIXED)
    for name, v in zip(ORDER, th):
        if name == "ln10As": p["A_s"] = jnp.exp(v) / 1e10
        else: p[name] = v
    return p
def derived(th): return _to_float(model.add_derived_parameters(raw(th)))
full_ps = [derived(jnp.asarray(th)) for th in TH]
pdots = []
for j in range(P):
    pdots.append(jax.tree.map(lambda *xs: jnp.stack(xs), *[
        eqx.filter(jax.jvp(derived, (jnp.asarray(th),), (jnp.zeros(7).at[j].set(SIG[j]),))[1],
                   eqx.is_inexact_array) for th in TH]))
pdot = jax.tree.map(lambda *xs: jnp.stack(xs), *pdots)               # (P, B, ...)
params_batch, BG = model._build_bgs_batched(full_ps, shardfn=None)
PT = PE.full_evolution_batched((BG, params_batch), k_chunk_size=100)
jax.block_until_ready(PT)
print("primal to stage 4 done", flush=True)
zeros_dot = lambda tree: jax.tree.map(lambda x: jnp.zeros((P,) + x.shape, x.dtype),
                                      eqx.filter(tree, eqx.is_inexact_array))
PT_dot, BG_dot = zeros_dot(PT), zeros_dot(BG)
def f(ptd, bgd, pd):
    return jvp_stage(SS.get_Cl_batched, (PT, BG, params_batch), (ptd, bgd, pd), n_dirs=P)
lowered = jax.jit(f).lower(PT_dot, BG_dot, pdot)
compiled = lowered.compile()
ma = compiled.memory_analysis()
print(f"MEMORY_ANALYSIS B={B} P={P}: args {ma.argument_size_in_bytes/1e9:.2f} GB, out "
      f"{ma.output_size_in_bytes/1e9:.3f} GB, TEMP {ma.temp_size_in_bytes/1e9:.2f} GB, "
      f"code {ma.generated_code_size_in_bytes/1e6:.1f} MB", flush=True)
txt = compiled.as_text()
sizes = []
for m in re.finditer(r"f64\[([0-9,]+)\]", txt):
    dims = [int(d) for d in m.group(1).split(",") if d]
    n = int(np.prod(dims)) if dims else 1
    if n * 8 > 2e8:
        line = txt[txt.rfind("\n", 0, m.start()) + 1: txt.find("\n", m.end())]
        sizes.append((n * 8 / 1e9, m.group(0), line.strip()[:160]))
seen = set(); top = []
for s in sorted(sizes, reverse=True):
    key = s[2]
    if key in seen: continue
    seen.add(key); top.append(s)
print("LARGEST f64 SHAPES in the optimized HLO (GB, shape, instruction):")
for s in top[:25]:
    print(f"  {s[0]:6.2f} GB  {s[1]:28s} {s[2]}")

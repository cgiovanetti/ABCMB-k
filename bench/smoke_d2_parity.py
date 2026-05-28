"""
Phase D.2 parity smoke: compare full_evolution_batched output to the
current single-call full_evolution path.

Two checks:
  (1) B=1: batched PerturbationTable[0] should match single-call
      PerturbationTable for the fiducial params.
  (2) B=2: batched PerturbationTable[i] should match single-call
      PerturbationTable for params i in {0, 1}.

Tolerance target ~1e-9 (XLA-fusion slack between vmap'd and direct paths).

Run on GPU:
    srun ... bash -c 'module load conda && conda activate actdr6 && \
        export PYTHONPATH=/pscratch/sd/c/carag/ABCMB-k:$PYTHONPATH && \
        python -u bench/smoke_d2_parity.py'
"""

import os
import sys
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

from abcmb.main import Model
from abcmb.perturbations import strip_bg_kappa

ELLMAX = 800
FIDUCIAL = {
    'h': 0.6736, 'omega_cdm': 0.120, 'omega_b': 0.02237,
    'A_s': 2.1e-9, 'n_s': 0.9649, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
PARAMS_TWO = [
    FIDUCIAL,
    {**FIDUCIAL, 'h': 0.68, 'omega_cdm': 0.118, 'A_s': 2.05e-9},
]


def _to_float(v):
    arr = jnp.asarray(v)
    if arr.dtype.kind in 'iub':
        return arr.astype(jnp.float64)
    return arr


def build_one_bg(model, params):
    full_p = model.add_derived_parameters(params)
    pre_bg = model.get_BG_pre_recomb(full_p)
    cpu_dev = jax.devices('cpu')[0]
    recomb_in_cpu = jax.device_put(pre_bg.recomb_inputs, cpu_dev)
    p_cpu = jax.device_put(full_p, cpu_dev)
    recomb_output = eqx.filter_jit(model.RecModel, backend='cpu')(
        (recomb_in_cpu, p_cpu))
    try:
        recomb_output = jax.device_put(recomb_output, jax.devices('gpu')[0])
    except Exception:
        pass
    recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)
    full_p = jax.tree_util.tree_map(_to_float, full_p)
    bg = model.get_BG(full_p, pre_bg, recomb_output)
    return full_p, bg


def stack_pytrees(pytrees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *pytrees)


def _max_rel_diff(a, b, label=""):
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        return float('inf'), f"{label}: SHAPE MISMATCH {a_arr.shape} vs {b_arr.shape}"
    if a_arr.size == 0:
        return 0.0, f"{label}: empty"
    diff = np.abs(a_arr - b_arr)
    ref = np.maximum(np.abs(b_arr), 1e-300)
    rel = diff / ref
    max_rel = float(rel.max())
    max_abs = float(diff.max())
    return max_rel, f"{label}: max_rel={max_rel:.2e}  max_abs={max_abs:.2e}"


def compare_PT_to_single(PT_batched, idx, PT_single, label, tol=1e-9):
    """Compare PerturbationTable[idx, ...] to a single PT. Returns
    (overall_max_rel, lines)."""
    fields = ['k', 'lna', 'delta_m', 'theta_b_prime',
              'metric_eta', 'metric_h_prime', 'metric_eta_prime',
              'metric_alpha', 'metric_alpha_prime']
    overall = 0.0
    lines = []
    for fname in fields:
        val_b = getattr(PT_batched, fname)
        val_s = getattr(PT_single, fname)
        # slice the B axis on the batched value
        val_b_sliced = val_b[idx]
        max_rel, line = _max_rel_diff(val_b_sliced, val_s,
                                      label=f"  {label}.{fname}")
        lines.append(line)
        overall = max(overall, max_rel)
    # species_perturbations is a dict
    sp_batched = PT_batched.species_perturbations
    sp_single = PT_single.species_perturbations
    for sname, qdict_b in sp_batched.items():
        for qname, qval_b in qdict_b.items():
            qval_s = sp_single[sname][qname]
            qval_b_sliced = qval_b[idx]
            max_rel, line = _max_rel_diff(
                qval_b_sliced, qval_s,
                label=f"  {label}.species[{sname}].{qname}")
            lines.append(line)
            overall = max(overall, max_rel)
    return overall, lines


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    PE = model.PE

    print("\n[setup] Build 2 BGs...", flush=True)
    setups = []
    for p in PARAMS_TWO:
        full_p, bg = build_one_bg(model, p)
        setups.append((full_p, bg))

    # ---------- single-call references via existing full_evolution ----------
    print("\n[single] full_evolution on each params (compile + run)...",
          flush=True)
    PT_singles = []
    for i, (full_p, bg) in enumerate(setups):
        t0 = time.perf_counter()
        PT_s = eqx.filter_jit(PE.full_evolution)((bg, full_p))
        jax.block_until_ready(PT_s.delta_m)
        print(f"  PT_single[{i}] in {time.perf_counter()-t0:.1f}s",
              flush=True)
        PT_singles.append(PT_s)

    # ---------- batched at B=1 ----------
    print("\n[batched B=1] strip + stack 1 BG; full_evolution_batched...",
          flush=True)
    bg0_strip = strip_bg_kappa(setups[0][1])
    BG_b1 = stack_pytrees([bg0_strip])
    p_b1 = stack_pytrees([setups[0][0]])

    t0 = time.perf_counter()
    PT_b1 = PE.full_evolution_batched((BG_b1, p_b1), k_chunk_size=100)
    jax.block_until_ready(PT_b1.delta_m)
    print(f"  B=1 done in {time.perf_counter()-t0:.1f}s", flush=True)

    overall_b1, lines_b1 = compare_PT_to_single(
        PT_b1, 0, PT_singles[0], label="B1[0]")
    print(f"  B=1 max_rel = {overall_b1:.2e}", flush=True)
    if overall_b1 > 1e-9:
        print("  field-by-field:", flush=True)
        for ln in lines_b1[:10]:
            print(f"   {ln}", flush=True)

    # ---------- batched at B=2 ----------
    print("\n[batched B=2] strip + stack 2 BGs; full_evolution_batched...",
          flush=True)
    bgs_strip = [strip_bg_kappa(bg) for _, bg in setups]
    full_ps = [fp for fp, _ in setups]
    BG_b2 = stack_pytrees(bgs_strip)
    p_b2 = stack_pytrees(full_ps)

    t0 = time.perf_counter()
    PT_b2 = PE.full_evolution_batched((BG_b2, p_b2), k_chunk_size=100)
    jax.block_until_ready(PT_b2.delta_m)
    print(f"  B=2 done in {time.perf_counter()-t0:.1f}s", flush=True)

    print("\n  B=2 element 0 vs single:", flush=True)
    overall_b2_0, lines_b2_0 = compare_PT_to_single(
        PT_b2, 0, PT_singles[0], label="B2[0]")
    print(f"    max_rel = {overall_b2_0:.2e}", flush=True)
    print("\n  B=2 element 1 vs single:", flush=True)
    overall_b2_1, lines_b2_1 = compare_PT_to_single(
        PT_b2, 1, PT_singles[1], label="B2[1]")
    print(f"    max_rel = {overall_b2_1:.2e}", flush=True)

    # ---------- verdict ----------
    print("\n" + "=" * 60, flush=True)
    print("PARITY SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  B=1 vs single: max_rel = {overall_b1:.2e}", flush=True)
    print(f"  B=2 vs single (element 0): max_rel = {overall_b2_0:.2e}",
          flush=True)
    print(f"  B=2 vs single (element 1): max_rel = {overall_b2_1:.2e}",
          flush=True)
    tol = 1e-9
    passed = all(x <= tol for x in (overall_b1, overall_b2_0, overall_b2_1))
    print(f"\n  tolerance: {tol:.0e}", flush=True)
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

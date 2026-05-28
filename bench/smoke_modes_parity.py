"""
Localize the D.2 parity failure: compare raw modes tensor (output of
_compute_modes_batched) to the single-call full_evolution intermediate.

If modes match at B=1, the bug is in make_output_table_batched (D.2).
If modes don't match at B=1, the bug is in _compute_modes_batched (D.1).
"""

import os
import sys
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from jax import vmap

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


@eqx.filter_jit
def single_call_modes(PE, BG, params):
    """Mirror full_evolution but stop at the raw modes tensor."""
    lna = jnp.linspace(BG.lna_transfer_start, 0., 500)
    res = vmap(PE.evolution_one_k, in_axes=[0, None, None])(
        PE.k_axis_perturbations, lna, (BG, params))
    # (N_k, Nlna, Ny) -> (Ny, Nlna, N_k)
    return res.transpose(2, 1, 0), lna


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    PE = model.PE
    full_p, bg = build_one_bg(model, FIDUCIAL)

    print("\n[single] compute modes via full_evolution-style vmap...",
          flush=True)
    t0 = time.perf_counter()
    modes_s, lna_s = single_call_modes(PE, bg, full_p)
    jax.block_until_ready(modes_s)
    print(f"  done in {time.perf_counter()-t0:.1f}s  shape={modes_s.shape}",
          flush=True)

    print("\n[batched B=1] strip + stack 1 BG + _compute_modes_batched...",
          flush=True)
    bg_strip = strip_bg_kappa(bg)
    BG_b = stack_pytrees([bg_strip])
    p_b = stack_pytrees([full_p])
    t0 = time.perf_counter()
    modes_b, lna_b = PE._compute_modes_batched((BG_b, p_b), k_chunk_size=100)
    jax.block_until_ready(modes_b)
    print(f"  done in {time.perf_counter()-t0:.1f}s  shape={modes_b.shape}",
          flush=True)

    # Compare lna and modes
    modes_b_sliced = np.asarray(modes_b[0])  # (Ny, Nlna, N_k)
    modes_s_arr = np.asarray(modes_s)
    lna_b_sliced = np.asarray(lna_b[0])
    lna_s_arr = np.asarray(lna_s)

    print("\n[compare]", flush=True)
    print(f"  lna  shape s={lna_s_arr.shape} b={lna_b_sliced.shape}",
          flush=True)
    print(f"  lna  max_abs_diff = {float(np.abs(lna_b_sliced - lna_s_arr).max()):.2e}",
          flush=True)

    if modes_s_arr.shape != modes_b_sliced.shape:
        print(f"  modes SHAPE MISMATCH: s={modes_s_arr.shape} "
              f"b={modes_b_sliced.shape}", flush=True)
        sys.exit(2)

    diff = np.abs(modes_b_sliced - modes_s_arr)
    ref = np.maximum(np.abs(modes_s_arr), 1e-300)
    rel = diff / ref
    print(f"  modes max_abs_diff = {float(diff.max()):.2e}", flush=True)
    print(f"  modes max_rel = {float(rel.max()):.2e}", flush=True)
    # which component is worst?
    worst = np.unravel_index(int(np.argmax(rel)), rel.shape)
    print(f"  worst index (Ny, Nlna, N_k) = {worst}", flush=True)
    print(f"    single = {modes_s_arr[worst]:.6e}", flush=True)
    print(f"    batched= {modes_b_sliced[worst]:.6e}", flush=True)

    # per-Ny-index worst rel
    print("\n  per-Ny worst rel:", flush=True)
    for i in range(modes_s_arr.shape[0]):
        r_i = (np.abs(modes_b_sliced[i] - modes_s_arr[i])
               / np.maximum(np.abs(modes_s_arr[i]), 1e-300))
        if r_i.max() > 1e-10:
            print(f"    Ny[{i}]: max_rel={float(r_i.max()):.2e}", flush=True)


if __name__ == "__main__":
    main()

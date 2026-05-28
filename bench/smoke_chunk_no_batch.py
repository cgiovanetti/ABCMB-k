"""
Test if removing the inner B-axis vmap makes chunk[1] correct.

Compare three vmap structures on chunk[1] (k_axis[100:200]):
  (1) Single-vmap reference (correct): vmap(evolution_one_k, in_axes=[0,None,None])
  (2) Single-vmap chunked: same as (1) but with chunk[1] only — does this fail?
  (3) Double-vmap (k outer, B inner B=1) like _evolve_chunk

If (2) is correct: the bug is the double-vmap structure or the singleton-B.
If (2) is wrong: the bug is somehow about vmapping just 100 high-k values.
"""

import os, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from jax import vmap

from abcmb.main import Model
from abcmb.perturbations import strip_bg_kappa

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
def single_full(PE, BG, params):
    lna = jnp.linspace(BG.lna_transfer_start, 0., 500)
    res = vmap(PE.evolution_one_k, in_axes=[0, None, None])(
        PE.k_axis_perturbations, lna, (BG, params))
    return res.transpose(2, 1, 0), lna


@eqx.filter_jit
def single_chunk_only(PE, BG, params, k_chunk):
    """Vmap only over a subset chunk of k, like _evolve_chunk but no B-axis."""
    lna = jnp.linspace(BG.lna_transfer_start, 0., 500)
    res = vmap(PE.evolution_one_k, in_axes=[0, None, None])(
        k_chunk, lna, (BG, params))
    return res.transpose(2, 1, 0), lna


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)
    model = Model(
        user_species=None, output_Cl=True, l_max=800, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    PE = model.PE
    full_p, bg = build_one_bg(model, FIDUCIAL)
    N_k = len(PE.k_axis_perturbations)
    K_CHUNK = 100
    print(f"N_k = {N_k}  K_CHUNK = {K_CHUNK}", flush=True)

    print("\n[ref] single-call full vmap (492 k's)", flush=True)
    modes_s, _ = single_full(PE, bg, full_p)
    jax.block_until_ready(modes_s)
    modes_s_arr = np.asarray(modes_s)  # (Ny, Nlna, N_k)

    k_chunk_1 = PE.k_axis_perturbations[K_CHUNK:2*K_CHUNK]
    s_slice_1 = modes_s_arr[:, :, K_CHUNK:2*K_CHUNK]

    print("\n[T1] single-vmap on chunk[1] ONLY (100 high-k values)",
          flush=True)
    t0 = time.perf_counter()
    res_T1, _ = single_chunk_only(PE, bg, full_p, k_chunk_1)
    jax.block_until_ready(res_T1)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    T1_arr = np.asarray(res_T1)  # (Ny, Nlna, K)
    diff_T1 = np.abs(T1_arr - s_slice_1).max()
    print(f"  T1 vs single[1]: max_abs={diff_T1:.2e}", flush=True)

    bg_strip = strip_bg_kappa(bg)
    BG_b = stack_pytrees([bg_strip])
    p_b = stack_pytrees([full_p])
    lna_b = vmap(lambda lts: jnp.linspace(lts, 0., 500))(
        BG_b.lna_transfer_start)

    print("\n[T2] double-vmap on chunk[1] (via _evolve_chunk)", flush=True)
    t0 = time.perf_counter()
    c = PE._evolve_chunk(k_chunk_1, lna_b, BG_b, p_b)
    jax.block_until_ready(c)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    T2_arr = np.asarray(c)[:, 0, :, :].transpose(2, 1, 0)  # (Ny, Nlna, K)
    diff_T2 = np.abs(T2_arr - s_slice_1).max()
    print(f"  T2 vs single[1]: max_abs={diff_T2:.2e}", flush=True)


if __name__ == "__main__":
    main()

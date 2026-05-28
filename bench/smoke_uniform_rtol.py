"""
Test: vmap 100 k modes that are ALL above k_split_PE (so all should use same rtol).

Compare to single full vmap result.

If still wrong: rtol mixing isn't the issue
If correct: the issue is mixed rtol under vmap
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from jax import vmap

from abcmb.main import Model

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


@eqx.filter_jit
def vmap_evol(PE, BG, params, k_subset):
    lna = jnp.linspace(BG.lna_transfer_start, 0., 500)
    res = vmap(PE.evolution_one_k, in_axes=[0, None, None])(
        k_subset, lna, (BG, params))
    return res.transpose(2, 1, 0), lna


def main():
    model = Model(
        user_species=None, output_Cl=True, l_max=800, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    PE = model.PE
    full_p, bg = build_one_bg(model, FIDUCIAL)
    k_axis = PE.k_axis_perturbations
    k_split = float(PE.specs["k_split_PE"])
    print(f"k_split_PE = {k_split}", flush=True)

    # Find all indices with k > k_split
    k_np = np.asarray(k_axis)
    above_idx = np.where(k_np > k_split)[0]
    below_idx = np.where(k_np <= k_split)[0]
    print(f"#k below split: {len(below_idx)}, #k above split: {len(above_idx)}",
          flush=True)
    print(f"first above-split idx: {above_idx[0]}, k={k_np[above_idx[0]]:.4e}",
          flush=True)

    print("\n[ref] full vmap (492)", flush=True)
    res_A, _ = vmap_evol(PE, bg, full_p, k_axis)
    jax.block_until_ready(res_A)
    A_arr = np.asarray(res_A)

    # Test 1: 100 k's all above split (should all use large_k rtol)
    test1_indices = above_idx[:100]
    test1_k = k_axis[test1_indices]
    print(f"\n[T1] 100 k's all above split. idx range "
          f"[{test1_indices.min()}, {test1_indices.max()}]",
          flush=True)
    res_1, _ = vmap_evol(PE, bg, full_p, test1_k)
    jax.block_until_ready(res_1)
    res_1_arr = np.asarray(res_1)
    s_slice_1 = A_arr[:, :, test1_indices]
    print(f"  T1 vs A[above-split]: max_abs={np.abs(res_1_arr - s_slice_1).max():.2e}",
          flush=True)

    # Test 2: 100 k's all below split
    test2_indices = below_idx[:100]
    test2_k = k_axis[test2_indices]
    print(f"\n[T2] 100 k's all BELOW split. idx range "
          f"[{test2_indices.min()}, {test2_indices.max()}]",
          flush=True)
    res_2, _ = vmap_evol(PE, bg, full_p, test2_k)
    jax.block_until_ready(res_2)
    res_2_arr = np.asarray(res_2)
    s_slice_2 = A_arr[:, :, test2_indices]
    print(f"  T2 vs A[below-split]: max_abs={np.abs(res_2_arr - s_slice_2).max():.2e}",
          flush=True)

    # Test 3: 200 mixed (100 below + 100 above), with below first
    test3_indices = np.concatenate([below_idx[-100:], above_idx[:100]])
    test3_k = k_axis[test3_indices]
    print(f"\n[T3] 200 k's mixed (below + above)", flush=True)
    res_3, _ = vmap_evol(PE, bg, full_p, test3_k)
    jax.block_until_ready(res_3)
    res_3_arr = np.asarray(res_3)
    s_slice_3 = A_arr[:, :, test3_indices]
    print(f"  T3 vs A[mixed]: max_abs={np.abs(res_3_arr - s_slice_3).max():.2e}",
          flush=True)


if __name__ == "__main__":
    main()

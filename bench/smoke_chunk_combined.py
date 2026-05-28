"""
Test what subsets of k_axis trigger the bug.

  (a) Full k_axis (492) → correct
  (b) chunk[1] only (100 high-k) → wrong
  (c) chunk[0] + chunk[1] (200 mixed) → ???
  (d) k_axis[0:200] = chunk[0] + chunk[1] same as (c) (yes)
  (e) chunk[0] + chunk[1] but with ARBITRARY low-k indices padding
"""

import os, sys, time
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
    print(f"BG.lna_transfer_start = {float(bg.lna_transfer_start)}",
          flush=True)
    K_CHUNK = 100
    k_axis = PE.k_axis_perturbations
    N_k = len(k_axis)
    print(f"N_k={N_k}", flush=True)

    print("\n[A] full vmap (492)", flush=True)
    res_A, _ = vmap_evol(PE, bg, full_p, k_axis)
    jax.block_until_ready(res_A)
    A_arr = np.asarray(res_A)

    test_idx_chunk1 = slice(K_CHUNK, 2*K_CHUNK)
    s_slice_1 = A_arr[:, :, test_idx_chunk1]

    print("\n[B] chunk[1] only (100)", flush=True)
    res_B, _ = vmap_evol(PE, bg, full_p, k_axis[test_idx_chunk1])
    jax.block_until_ready(res_B)
    B_arr = np.asarray(res_B)
    print(f"  B vs A[chunk1]: max_abs={np.abs(B_arr - s_slice_1).max():.2e}",
          flush=True)

    print("\n[C] chunk[0]+chunk[1] = k_axis[0:200] (200)", flush=True)
    res_C, _ = vmap_evol(PE, bg, full_p, k_axis[0:2*K_CHUNK])
    jax.block_until_ready(res_C)
    C_arr = np.asarray(res_C)
    print(f"  C vs A[0:200]: max_abs={np.abs(C_arr - A_arr[:,:,0:2*K_CHUNK]).max():.2e}",
          flush=True)

    print("\n[D] chunk[1] + chunk[2] = k_axis[100:300] (200)", flush=True)
    res_D, _ = vmap_evol(PE, bg, full_p, k_axis[K_CHUNK:3*K_CHUNK])
    jax.block_until_ready(res_D)
    D_arr = np.asarray(res_D)
    print(f"  D vs A[100:300]: max_abs={np.abs(D_arr - A_arr[:,:,K_CHUNK:3*K_CHUNK]).max():.2e}",
          flush=True)

    print("\n[E] all-high-k: k_axis[100:492] (392 high-k modes)",
          flush=True)
    res_E, _ = vmap_evol(PE, bg, full_p, k_axis[K_CHUNK:])
    jax.block_until_ready(res_E)
    E_arr = np.asarray(res_E)
    print(f"  E vs A[100:]: max_abs={np.abs(E_arr - A_arr[:,:,K_CHUNK:]).max():.2e}",
          flush=True)

    print("\n[F] just first 50 of chunk[1] (50 lowest k of chunk[1])",
          flush=True)
    res_F, _ = vmap_evol(PE, bg, full_p, k_axis[K_CHUNK:K_CHUNK+50])
    jax.block_until_ready(res_F)
    F_arr = np.asarray(res_F)
    print(f"  F vs A[100:150]: max_abs={np.abs(F_arr - A_arr[:,:,K_CHUNK:K_CHUNK+50]).max():.2e}",
          flush=True)


if __name__ == "__main__":
    main()

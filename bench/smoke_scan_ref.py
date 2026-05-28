"""
Get a per-k 'pure scan' reference: one-at-a-time evolve for chunk[1] k's, no vmap.
Compare to:
  - full vmap (current ref)
  - chunk[1]-only vmap
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
def single_k(PE, BG, params, k):
    lna = jnp.linspace(BG.lna_transfer_start, 0., 500)
    return PE.evolution_one_k(k, lna, (BG, params))  # (Nlna, Ny)


@eqx.filter_jit
def vmap_evol(PE, BG, params, k_subset):
    lna = jnp.linspace(BG.lna_transfer_start, 0., 500)
    res = vmap(PE.evolution_one_k, in_axes=[0, None, None])(
        k_subset, lna, (BG, params))
    return res.transpose(2, 1, 0), lna  # (Ny, Nlna, K)


def main():
    model = Model(
        user_species=None, output_Cl=True, l_max=800, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    PE = model.PE
    full_p, bg = build_one_bg(model, FIDUCIAL)
    k_axis = PE.k_axis_perturbations
    K_CHUNK = 100

    print("Get pure-per-k reference for chunk[1] (k_axis[100:200])...",
          flush=True)
    # Just do 5 representative ks to keep wall time reasonable
    sample_idx = [100, 105, 150, 155, 184, 185, 199]
    pure_results = {}
    for idx in sample_idx:
        k = k_axis[idx]
        sol = single_k(PE, bg, full_p, k)  # (Nlna, Ny)
        jax.block_until_ready(sol)
        pure_results[idx] = np.asarray(sol)
        print(f"  idx={idx} k={float(k):.4e} done", flush=True)

    print("\n[ref] full vmap (492)", flush=True)
    res_A, _ = vmap_evol(PE, bg, full_p, k_axis)
    jax.block_until_ready(res_A)
    A_arr = np.asarray(res_A)  # (Ny, Nlna, N_k)

    print("\n[chunk1] vmap chunk[1] (100)", flush=True)
    res_B, _ = vmap_evol(PE, bg, full_p, k_axis[K_CHUNK:2*K_CHUNK])
    jax.block_until_ready(res_B)
    B_arr = np.asarray(res_B)  # (Ny, Nlna, K=100)

    print("\nPer-sample compare:", flush=True)
    print(f"{'idx':>6} {'pure_norm':>12} {'full_vs_pure':>14} {'chunk_vs_pure':>14}",
          flush=True)
    for idx in sample_idx:
        pure_y = pure_results[idx]  # (Nlna, Ny)
        # full vmap result at idx (Ny, Nlna) -> (Nlna, Ny) for compare
        full_y = A_arr[:, :, idx].T
        chunk_y = B_arr[:, :, idx - K_CHUNK].T
        # restrict to non-zero
        norm = float(np.linalg.norm(pure_y))
        d_full = float(np.linalg.norm(full_y - pure_y))
        d_chunk = float(np.linalg.norm(chunk_y - pure_y))
        print(f"{idx:>6} {norm:>12.4e} {d_full:>14.4e} {d_chunk:>14.4e}",
              flush=True)


if __name__ == "__main__":
    main()

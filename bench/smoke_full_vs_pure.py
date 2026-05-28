"""
Compare full-vmap result (492 ks at once) against pure-single-k integration.

If full-vmap matches pure-single-k everywhere: full-vmap is ground truth, chunking has a bug.
If full-vmap deviates from pure-single-k: "chunking bug" is actually just vmap-induced
  numerical drift; the choice of batch composition (small-k + large-k mix) influences
  per-element accuracy in unpredictable ways.
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
    return PE.evolution_one_k(k, lna, (BG, params))


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
    N_k = len(k_axis)
    K_CHUNK = 100

    print("[full] full-vmap (492)", flush=True)
    res_A, _ = vmap_evol(PE, bg, full_p, k_axis)
    jax.block_until_ready(res_A)
    A_arr = np.asarray(res_A)  # (Ny, Nlna, N_k)

    print("\n[chunk1] chunk[1] (100)", flush=True)
    res_B, _ = vmap_evol(PE, bg, full_p, k_axis[K_CHUNK:2*K_CHUNK])
    jax.block_until_ready(res_B)
    B_arr = np.asarray(res_B)  # (Ny, Nlna, 100)

    # Pure single-k references at many indices
    sample_idx = [0, 10, 50, 99, 100, 110, 150, 155, 184, 185, 199,
                  200, 250, 300, 350, 400, 450, 491]
    pure_results = {}
    print("\n[pure] per-k pure single integrations:", flush=True)
    for idx in sample_idx:
        k = k_axis[idx]
        sol = single_k(PE, bg, full_p, k)
        jax.block_until_ready(sol)
        pure_results[idx] = np.asarray(sol)  # (Nlna, Ny)

    print(f"\n{'idx':>6} {'k':>11} {'norm':>11} "
          f"{'full_vs_pure':>14} {'chunk_vs_pure':>14} {'full_vs_chunk':>14}",
          flush=True)
    for idx in sample_idx:
        pure_y = pure_results[idx]
        full_y = A_arr[:, :, idx].T
        norm = float(np.linalg.norm(pure_y))
        d_full = float(np.linalg.norm(full_y - pure_y))
        if K_CHUNK <= idx < 2*K_CHUNK:
            chunk_y = B_arr[:, :, idx - K_CHUNK].T
            d_chunk = float(np.linalg.norm(chunk_y - pure_y))
            d_full_chunk = float(np.linalg.norm(full_y - chunk_y))
        else:
            d_chunk = float("nan")
            d_full_chunk = float("nan")
        print(f"{idx:>6} {float(k_axis[idx]):>11.4e} {norm:>11.3e} "
              f"{d_full:>14.2e} {d_chunk:>14.2e} {d_full_chunk:>14.2e}",
              flush=True)


if __name__ == "__main__":
    main()

"""Test if max_steps_PE is the cause."""

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
    for max_steps in [2048, 4096, 8192, 16384]:
        model = Model(
            user_species=None, output_Cl=True, l_max=800, lensing=False,
            output_Pk=True, output_k_max=0.5,
            l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
            max_steps_PE=max_steps,
        )
        PE = model.PE
        full_p, bg = build_one_bg(model, FIDUCIAL)
        k_axis = PE.k_axis_perturbations

        print(f"\n=== max_steps_PE = {max_steps} ===", flush=True)
        # ref
        res_A, _ = vmap_evol(PE, bg, full_p, k_axis)
        jax.block_until_ready(res_A)
        A_arr = np.asarray(res_A)
        # chunk[1]
        res_B, _ = vmap_evol(PE, bg, full_p, k_axis[100:200])
        jax.block_until_ready(res_B)
        B_arr = np.asarray(res_B)
        diff = float(np.abs(B_arr - A_arr[:, :, 100:200]).max())
        print(f"  chunk[1] vs ref[100:200] max_abs={diff:.2e}", flush=True)


if __name__ == "__main__":
    main()

"""
Drill into chunk[1] (k_indices 100-199) and see which k's are wrong.

Compare _evolve_chunk's per-k slice against single-call full-vmap's per-k slice.
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
def single_call_modes(PE, BG, params):
    lna = jnp.linspace(BG.lna_transfer_start, 0., 500)
    res = vmap(PE.evolution_one_k, in_axes=[0, None, None])(
        PE.k_axis_perturbations, lna, (BG, params))
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
    k_axis = np.asarray(PE.k_axis_perturbations)
    print(f"k_axis[0:5] = {k_axis[0:5]}", flush=True)
    print(f"k_axis[99:105] = {k_axis[99:105]}", flush=True)
    print(f"k_axis[199:205] = {k_axis[199:205]}", flush=True)
    print(f"k_axis[-5:] = {k_axis[-5:]}", flush=True)

    print("\n[ref] single full vmap...", flush=True)
    modes_s, lna_s = single_call_modes(PE, bg, full_p)
    jax.block_until_ready(modes_s)
    modes_s_arr = np.asarray(modes_s)  # (Ny, Nlna, N_k)

    bg_strip = strip_bg_kappa(bg)
    BG_b = stack_pytrees([bg_strip])
    p_b = stack_pytrees([full_p])
    lna_b = vmap(lambda lts: jnp.linspace(lts, 0., 500))(
        BG_b.lna_transfer_start)

    k_chunk_1 = PE.k_axis_perturbations[K_CHUNK:2*K_CHUNK]
    s_slice_1 = modes_s_arr[:, :, K_CHUNK:2*K_CHUNK]

    print("\n[!] _evolve_chunk on chunk[1] (first call)", flush=True)
    t0 = time.perf_counter()
    c = PE._evolve_chunk(k_chunk_1, lna_b, BG_b, p_b)
    jax.block_until_ready(c)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    c_arr = np.asarray(c)[:, 0, :, :].transpose(2, 1, 0)  # (Ny, Nlna, K)
    # Per-k max_abs across all (Ny, Nlna)
    diff = np.abs(c_arr - s_slice_1)
    print("\n  per-k max_abs_diff:", flush=True)
    for ki in range(K_CHUNK):
        per_k_max = float(diff[:, :, ki].max())
        # only print non-trivial
        if per_k_max > 1e-8:
            kval = k_axis[K_CHUNK + ki]
            print(f"    chunk[1] idx={ki:>3} (k_axis idx={K_CHUNK+ki}, k={kval:.4e})"
                  f": max_abs={per_k_max:.2e}", flush=True)
        elif ki < 5 or ki >= K_CHUNK - 5:
            kval = k_axis[K_CHUNK + ki]
            print(f"    chunk[1] idx={ki:>3} (k_axis idx={K_CHUNK+ki}, k={kval:.4e})"
                  f": max_abs={per_k_max:.2e}  (OK)", flush=True)


if __name__ == "__main__":
    main()

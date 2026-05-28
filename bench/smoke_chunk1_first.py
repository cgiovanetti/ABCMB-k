"""
Test if chunk[1] is wrong even when called FIRST (no prior chunks).

If chunk[1] wrong when called first: bug is k-values-dependent.
If chunk[1] correct when called first: bug is state-leakage from prior call.
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


def compare(label, c_arr, s_slice):
    diff = np.abs(c_arr - s_slice)
    ref = np.maximum(np.abs(s_slice), 1e-300)
    rel = diff / ref
    mask = np.abs(s_slice) > 1e-10
    if mask.any():
        max_rel = float(rel[mask].max())
    else:
        max_rel = float(rel.max())
    max_abs = float(diff.max())
    print(f"  {label}: max_abs={max_abs:.2e}  max_rel(meaningful)={max_rel:.2e}",
          flush=True)
    return max_abs, max_rel


def call_chunk(PE, k_chunk, lna_b, BG_b, p_b):
    c = PE._evolve_chunk(k_chunk, lna_b, BG_b, p_b)
    jax.block_until_ready(c)
    c_arr = np.asarray(c)[:, 0, :, :].transpose(2, 1, 0)
    return c_arr


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

    print("\n[ref] single-call full vmap...", flush=True)
    modes_s, lna_s = single_call_modes(PE, bg, full_p)
    jax.block_until_ready(modes_s)
    modes_s_arr = np.asarray(modes_s)

    bg_strip = strip_bg_kappa(bg)
    BG_b = stack_pytrees([bg_strip])
    p_b = stack_pytrees([full_p])
    lna_b = vmap(lambda lts: jnp.linspace(lts, 0., 500))(
        BG_b.lna_transfer_start)

    k_chunk_1 = PE.k_axis_perturbations[K_CHUNK:2*K_CHUNK]
    s_slice_1 = modes_s_arr[:, :, K_CHUNK:2*K_CHUNK]

    print("\n[!] Call chunk[1] FIRST (skip chunk[0])", flush=True)
    t0 = time.perf_counter()
    c1 = call_chunk(PE, k_chunk_1, lna_b, BG_b, p_b)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    compare("c1_first vs single[1]", c1, s_slice_1)

    # Also try chunk[2,3,4]
    for ci in range(2, 5):
        ks = PE.k_axis_perturbations[ci*K_CHUNK:(ci+1)*K_CHUNK]
        ss = modes_s_arr[:, :, ci*K_CHUNK:(ci+1)*K_CHUNK]
        if len(ks) == 0:
            break
        print(f"\n[!] Call chunk[{ci}] (size {len(ks)})", flush=True)
        t0 = time.perf_counter()
        c = call_chunk(PE, ks, lna_b, BG_b, p_b)
        print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
        compare(f"c{ci} vs single[{ci}]", c, ss)


if __name__ == "__main__":
    main()

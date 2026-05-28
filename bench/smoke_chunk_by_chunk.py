"""
Multiple chunks at k_chunk=100. Compare each chunk's output to the
corresponding slice of single-call full_evolution. If all chunks match,
the bug is in concatenation. If some chunks don't match, the bug is
multi-call (filter_jit cache effect, or closure aliasing).
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
    return res.transpose(2, 1, 0), lna  # (Ny, Nlna, N_k)


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

    print("\n[single] full vmap over k_axis...", flush=True)
    modes_s, lna_s = single_call_modes(PE, bg, full_p)
    jax.block_until_ready(modes_s)
    modes_s_arr = np.asarray(modes_s)  # (Ny, Nlna, N_k)
    print(f"  single shape {modes_s_arr.shape}", flush=True)

    print("\n[chunks] each chunk via _evolve_chunk, compare to slice of single",
          flush=True)
    bg_strip = strip_bg_kappa(bg)
    BG_b = stack_pytrees([bg_strip])
    p_b = stack_pytrees([full_p])
    lna_b = vmap(lambda lts: jnp.linspace(lts, 0., 500))(
        BG_b.lna_transfer_start)

    chunks_max_rel = []
    chunks_max_abs = []
    for chunk_idx, i in enumerate(range(0, N_k, K_CHUNK)):
        k_chunk = PE.k_axis_perturbations[i:i + K_CHUNK]
        t0 = time.perf_counter()
        c = PE._evolve_chunk(k_chunk, lna_b, BG_b, p_b)
        jax.block_until_ready(c)
        elapsed = time.perf_counter() - t0
        # c shape: (K, B=1, Nlna, Ny)
        # convert to (Ny, Nlna, K) like single's slice
        c_arr = np.asarray(c)[:, 0, :, :]  # (K, Nlna, Ny)
        c_arr = c_arr.transpose(2, 1, 0)  # (Ny, Nlna, K)
        # compare to single_modes[:, :, i:i+K]
        s_slice = modes_s_arr[:, :, i:i + c_arr.shape[2]]
        diff = np.abs(c_arr - s_slice)
        ref = np.maximum(np.abs(s_slice), 1e-300)
        rel = diff / ref
        # mask out near-zero noise
        mask = np.abs(s_slice) > 1e-10
        if mask.any():
            rel_masked = rel[mask]
            max_rel = float(rel_masked.max())
        else:
            max_rel = float(rel.max())
        max_abs = float(diff.max())
        chunks_max_rel.append(max_rel)
        chunks_max_abs.append(max_abs)
        print(f"  chunk[{chunk_idx}] i=[{i:>3},{i + c_arr.shape[2]:>3}) "
              f"K={c_arr.shape[2]:>3}  {elapsed:>5.1f}s  "
              f"max_abs={max_abs:.2e}  max_rel(meaningful)={max_rel:.2e}",
              flush=True)

    print(f"\nchunks summary: max max_abs={max(chunks_max_abs):.2e}  "
          f"max max_rel(meaningful)={max(chunks_max_rel):.2e}", flush=True)


if __name__ == "__main__":
    main()

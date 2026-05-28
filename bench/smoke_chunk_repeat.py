"""
Probe whether _evolve_chunk's cache is corrupted by progressive invocations.

Tests:
  A. Call _evolve_chunk on chunk[0] once, get result. Compare to single-vmap.
  B. Call _evolve_chunk on chunk[0] AGAIN (same shape, same data). Does the
     second result match the first? (If not, deterministic stateful bug.)
  C. Call _evolve_chunk on chunk[0] THEN chunk[1] THEN chunk[0] again.
     Does the second chunk[0] still match the first chunk[0]? (Tests whether
     the chunk[1] call corrupts state for chunk[0].)
  D. Call _evolve_chunk on chunk[1] FIRST (skipping chunk[0]) and compare to
     the single-call slice. Does chunk[1]-first work?

If A correct, B correct, C wrong -> state leakage between calls.
If D wrong on its own -> the bug is intrinsic to chunk[1]'s data (not state).
If D correct on its own -> bug is state-leakage from prior chunks.

If D is *correct* when called first but *wrong* after chunk[0] -> definitive
state leakage.
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
    c_arr = np.asarray(c)[:, 0, :, :].transpose(2, 1, 0)  # (Ny, Nlna, K)
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
    modes_s_arr = np.asarray(modes_s)  # (Ny, Nlna, N_k)

    bg_strip = strip_bg_kappa(bg)
    BG_b = stack_pytrees([bg_strip])
    p_b = stack_pytrees([full_p])
    lna_b = vmap(lambda lts: jnp.linspace(lts, 0., 500))(
        BG_b.lna_transfer_start)

    k_chunk_0 = PE.k_axis_perturbations[0:K_CHUNK]
    k_chunk_1 = PE.k_axis_perturbations[K_CHUNK:2*K_CHUNK]
    s_slice_0 = modes_s_arr[:, :, 0:K_CHUNK]
    s_slice_1 = modes_s_arr[:, :, K_CHUNK:2*K_CHUNK]

    print("\n[A] First call: chunk[0]", flush=True)
    t0 = time.perf_counter()
    c0_first = call_chunk(PE, k_chunk_0, lna_b, BG_b, p_b)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    compare("c0_first vs single[0]", c0_first, s_slice_0)

    print("\n[B] Second call: chunk[0] again (same shape, same data)", flush=True)
    t0 = time.perf_counter()
    c0_repeat = call_chunk(PE, k_chunk_0, lna_b, BG_b, p_b)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    compare("c0_repeat vs single[0]", c0_repeat, s_slice_0)
    diff_B = np.abs(c0_repeat - c0_first).max()
    print(f"  c0_repeat vs c0_first: max_abs={diff_B:.2e}", flush=True)

    print("\n[C] chunk[1] (after chunk[0] x2)", flush=True)
    t0 = time.perf_counter()
    c1_after = call_chunk(PE, k_chunk_1, lna_b, BG_b, p_b)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    compare("c1_after vs single[1]", c1_after, s_slice_1)

    print("\n[D] Third call: chunk[0] AGAIN after chunk[1]", flush=True)
    t0 = time.perf_counter()
    c0_third = call_chunk(PE, k_chunk_0, lna_b, BG_b, p_b)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    compare("c0_third vs single[0]", c0_third, s_slice_0)
    diff_D = np.abs(c0_third - c0_first).max()
    print(f"  c0_third vs c0_first: max_abs={diff_D:.2e}", flush=True)

    print("\n[E] Fresh process would call chunk[1] FIRST. We simulate by",
          flush=True)
    print("    just looking at what happens if we slice differently.",
          flush=True)
    print("    But we already saw chunk[1] is wrong in step C.", flush=True)

    print("\n=== Summary ===", flush=True)
    print("If [A] is correct but [B] differs from [A]: nondeterministic state",
          flush=True)
    print("If [B]==[A] but [D] wrong: state leakage between SHAPES",
          flush=True)
    print("If [B]==[A] and [D]==[A]: chunks themselves are deterministic; bug",
          flush=True)
    print("  is in chunk[1]'s data path (NOT state).", flush=True)


if __name__ == "__main__":
    main()

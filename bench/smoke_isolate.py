"""
Isolate where the D.1 bug lives. Three increasingly-batched paths run for
ONE k value (k_axis[300], a typical mid-k mode). All should give the same
(Nlna, Ny) trajectory for the fiducial params.

Path A: direct python call to PE.evolution_one_k(k, lna, (BG, params))
Path B: vmap at B=1 with unstripped, manually-singleton-extended BG
Path C: vmap at B=1 with stripped + stack_pytrees BG (= the production
        batched path's setup)
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


def _rel(a, b, label):
    a_a = np.asarray(a)
    b_a = np.asarray(b)
    if a_a.shape != b_a.shape:
        return f"{label}: SHAPE MISMATCH {a_a.shape} vs {b_a.shape}"
    diff = np.abs(a_a - b_a)
    ref = np.maximum(np.abs(b_a), 1e-300)
    rel = diff / ref
    return f"{label}: max_abs={float(diff.max()):.2e}  max_rel={float(rel.max()):.2e}"


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=800, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    PE = model.PE
    full_p, bg = build_one_bg(model, FIDUCIAL)
    k_axis = PE.k_axis_perturbations
    k = jnp.asarray(k_axis[300])  # single mid-k mode
    lna = jnp.linspace(bg.lna_transfer_start, 0., 500)
    print(f"k = {float(k):.4e}", flush=True)
    print(f"lna shape = {lna.shape}, lna[0]={float(lna[0]):.4e}",
          flush=True)

    # --------- A: direct python call (no jit, no vmap) ----------
    print("\n[A] direct evolution_one_k...", flush=True)
    t0 = time.perf_counter()
    a = PE.evolution_one_k(k, lna, (bg, full_p))
    jax.block_until_ready(a)
    print(f"  done in {time.perf_counter()-t0:.1f}s  shape={a.shape}",
          flush=True)

    # --------- B: vmap B=1 without stripping kappa ----------
    print("\n[B] vmap B=1, unstripped bg, manually expand singleton...",
          flush=True)
    # eqx.tree_at can't add a singleton; jax.tree.map((x: x[None])) on
    # a None leaf fails, but kappa_func is a diffrax.Solution
    # (registered pytree of arrays) — should accept x[None].
    try:
        bg_v1 = jax.tree.map(lambda x: x[None] if hasattr(x, 'shape') else x, bg)
        p_v1 = jax.tree.map(lambda x: jnp.asarray(x)[None], full_p)
        lna_v1 = lna[None, :]
        t0 = time.perf_counter()
        b = vmap(PE.evolution_one_k, in_axes=(None, 0, (0, 0)))(
            k, lna_v1, (bg_v1, p_v1))
        jax.block_until_ready(b)
        print(f"  done in {time.perf_counter()-t0:.1f}s  shape={b.shape}",
              flush=True)
        print(f"  {_rel(b[0], a, 'B[0] vs A')}", flush=True)
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
        b = None

    # --------- C: vmap B=1 with stripped + stack_pytrees ----------
    print("\n[C] vmap B=1, stripped kappa + stack_pytrees...", flush=True)
    bg_strip = strip_bg_kappa(bg)
    bg_c = stack_pytrees([bg_strip])
    p_c = stack_pytrees([full_p])
    lna_c = lna[None, :]
    t0 = time.perf_counter()
    c = vmap(PE.evolution_one_k, in_axes=(None, 0, (0, 0)))(
        k, lna_c, (bg_c, p_c))
    jax.block_until_ready(c)
    print(f"  done in {time.perf_counter()-t0:.1f}s  shape={c.shape}",
          flush=True)
    print(f"  {_rel(c[0], a, 'C[0] vs A')}", flush=True)

    # --------- and direct comparison of (B) vs (C) ----------
    if b is not None:
        print(f"\n  {_rel(b[0], c[0], 'B[0] vs C[0]')}", flush=True)


if __name__ == "__main__":
    main()

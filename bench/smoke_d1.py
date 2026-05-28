"""
Phase D.1 smoke test: call full_evolution_batched at B=2 and verify it
returns finite-shaped arrays. No correctness check — that's Phase D.4's
parity test.
"""

import os
import sys
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

from abcmb.main import Model
from abcmb.perturbations import strip_bg_kappa

ELLMAX = 800
FIDUCIAL = {
    'h': 0.6736, 'omega_cdm': 0.120, 'omega_b': 0.02237,
    'A_s': 2.1e-9, 'n_s': 0.9649, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
PARAMS_TWO = [
    FIDUCIAL,
    {**FIDUCIAL, 'h': 0.68, 'omega_cdm': 0.118},
]


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


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )

    print("[setup] Build 2 BGs sequentially...", flush=True)
    setups = []
    for p in PARAMS_TWO:
        full_p, bg = build_one_bg(model, p)
        setups.append((full_p, bg))

    print("[setup] Strip kappa_func + stack...", flush=True)
    PE = model.PE
    bgs_stripped = [strip_bg_kappa(bg) for _, bg in setups]
    full_ps = [fp for fp, _ in setups]
    BG_batch = stack_pytrees(bgs_stripped)
    params_batch = stack_pytrees(full_ps)
    print(f"  BG_batch.tau_tab.shape: {BG_batch.tau_tab.shape}", flush=True)
    print(f"  params_batch['h'].shape: {params_batch['h'].shape}", flush=True)

    print("[smoke] Calling full_evolution_batched (compile + run)...",
          flush=True)
    t0 = time.perf_counter()
    modes, lna_batch = PE.full_evolution_batched(
        (BG_batch, params_batch), k_chunk_size=100)
    jax.block_until_ready(modes)
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"  modes.shape: {modes.shape}  (expected (B, Ny, Nlna, N_k))",
          flush=True)
    print(f"  lna_batch.shape: {lna_batch.shape}", flush=True)
    print(f"  modes.dtype: {modes.dtype}", flush=True)
    print(f"  modes finite: {bool(jnp.all(jnp.isfinite(modes)))}",
          flush=True)
    print(f"  modes |min|, max, mean: "
          f"{float(jnp.min(modes)):.4e}, "
          f"{float(jnp.max(modes)):.4e}, "
          f"{float(jnp.mean(modes)):.4e}", flush=True)
    print(f"\nB=2 chunked path lives. Shape contract: "
          f"(B={modes.shape[0]}, Ny={modes.shape[1]}, "
          f"Nlna={modes.shape[2]}, N_k={modes.shape[3]})",
          flush=True)


if __name__ == "__main__":
    main()

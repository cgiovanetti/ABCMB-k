"""
Decompose the call_batched "setup" stage (per-cosmology BG build).

The profiler showed setup ~ 1.85 s/param, a linear O(B) floor. It is:
  get_BG_pre_recomb (GPU, jitted)
    + HyRex recombination (CPU, jitted backend='cpu')
    + get_BG (builds full Background incl. kappa diffrax solve, reion,
              decoupling) -- run EAGER in _build_one_bg (NOT jitted!)

This script times each sub-stage, and tests whether wrapping get_BG in
eqx.filter_jit removes the eager-BG cost. Tells us:
  - HyRex CPU floor (hard to batch)
  - eager-BG cost recoverable by jitting
  - whether get_BG_pre_recomb / get_BG can be vmapped over B

Run:
  srun ... python bench/probe_setup.py
"""

import os, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx

from abcmb.main import Model

ELLMAX = 800

FIDUCIAL = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}


def block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


def time_min(fn, reps=3):
    out = fn(); block(out)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(); block(out)
        best = min(best, time.perf_counter() - t0)
    return best, out


def _to_float(v):
    arr = jnp.asarray(v)
    if arr.dtype.kind in 'iub':
        return arr.astype(jnp.float64)
    return arr


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )

    params = model.add_derived_parameters(dict(FIDUCIAL))
    params = jax.tree_util.tree_map(_to_float, params)

    cpu_dev = jax.devices('cpu')[0]
    gpu_dev = jax.devices('gpu')[0] if any(
        d.platform == 'gpu' for d in jax.devices()) else cpu_dev

    # --- 1. get_BG_pre_recomb (jitted) ---
    def f_pre():
        return model.get_BG_pre_recomb(params)
    t_pre, pre_BG = time_min(f_pre)
    print(f"\n[1] get_BG_pre_recomb (jitted GPU): {t_pre:.3f}s", flush=True)

    # --- 2. HyRex (jitted cpu) ---
    recomb_inputs_cpu = jax.device_put(pre_BG.recomb_inputs, cpu_dev)
    params_cpu = jax.device_put(params, cpu_dev)
    rec_jit = eqx.filter_jit(model.RecModel, backend='cpu')
    def f_hyrex():
        return rec_jit((recomb_inputs_cpu, params_cpu))
    t_hyrex, recomb_output = time_min(f_hyrex)
    print(f"[2] HyRex (jitted CPU):             {t_hyrex:.3f}s", flush=True)

    recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)
    try:
        recomb_output = jax.device_put(recomb_output, gpu_dev)
    except Exception as e:
        print(f"  (device_put to gpu skipped: {type(e).__name__})", flush=True)

    # --- 3a. get_BG EAGER (as in current _build_one_bg) ---
    def f_bg_eager():
        return model.get_BG(params, pre_BG, recomb_output)
    t_bg_eager, bg = time_min(f_bg_eager, reps=3)
    print(f"[3a] get_BG EAGER:                  {t_bg_eager:.3f}s", flush=True)

    # --- 3b. get_BG JITTED ---
    get_BG_jit = eqx.filter_jit(model.get_BG)
    def f_bg_jit():
        return get_BG_jit(params, pre_BG, recomb_output)
    t_bg_jit, _ = time_min(f_bg_jit, reps=3)
    print(f"[3b] get_BG JITTED:                 {t_bg_jit:.3f}s", flush=True)

    # --- totals ---
    eager_total = t_pre + t_hyrex + t_bg_eager
    jit_total = t_pre + t_hyrex + t_bg_jit
    print(f"\nper-cosmology setup (current, eager BG): {eager_total:.3f}s",
          flush=True)
    print(f"per-cosmology setup (jitted BG):         {jit_total:.3f}s",
          flush=True)
    print(f"  -> jitting get_BG saves ~{t_bg_eager - t_bg_jit:.3f}s/cosmo",
          flush=True)
    print(f"\nHyRex CPU floor (hard to batch): {t_hyrex:.3f}s/cosmo "
          f"({100*t_hyrex/jit_total:.0f}% of jitted setup)", flush=True)

    # --- 4. can get_BG_pre_recomb vmap over B? quick test at B=4 ---
    print("\n[4] vmap test (get_BG_pre_recomb over B=4)...", flush=True)
    try:
        rng = np.random.default_rng(0)
        pl = []
        for _ in range(4):
            p = dict(FIDUCIAL)
            p['h'] = float(rng.uniform(0.65, 0.70))
            pl.append(jax.tree_util.tree_map(
                _to_float, model.add_derived_parameters(p)))
        pbatch = jax.tree.map(lambda *xs: jnp.stack(xs), *pl)
        vmapped_pre = eqx.filter_jit(
            lambda pb: jax.vmap(model.get_BG_pre_recomb)(pb))
        t0 = time.perf_counter()
        out = vmapped_pre(pbatch); block(out)
        print(f"     vmapped get_BG_pre_recomb B=4 (warm): "
              f"{time.perf_counter()-t0:.2f}s", flush=True)
        t0 = time.perf_counter()
        out = vmapped_pre(pbatch); block(out)
        print(f"     vmapped get_BG_pre_recomb B=4 (run):  "
              f"{time.perf_counter()-t0:.3f}s  -> "
              f"{(time.perf_counter()-t0)/4:.3f}s/cosmo", flush=True)
        print("     VMAP over B WORKS for get_BG_pre_recomb", flush=True)
    except Exception as e:
        print(f"     vmap FAILED: {type(e).__name__}: {str(e)[:120]}",
              flush=True)


if __name__ == "__main__":
    main()

"""
Phase A baseline benchmark.

Measures the current (vmap-over-k) ABCMB pipeline at B param-points to set
the bar Phase B has to beat. Three things get written:

  1. A table of (B, wall_model_per_params, wall_PE_per_params) wallclocks at
     B in {1, 4, 16, 64}. wall_PE strips HyRex/LINX/spectrum overhead to
     isolate the perturbation evolver — the part the refactor proposes to
     change.

  2. bench/baseline_stepcounts.npz: per-k step counts num_steps[k_i] for the
     fiducial params dict, plus the k_axis. Lets Phase B compare distributions.

  3. A baseline_summary.txt with the printed table.

The 64 perturbed params are sampled around the accuracy-test fiducial across
{h, omega_cdm, omega_b, A_s, n_s} at roughly Planck-2018 ±2-3σ.

Run on GPU. Activate conda first:
    module load conda && conda activate actdr6 && python bench/baseline.py
"""

import os
import sys
import time
import json
from contextlib import contextmanager

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import diffrax

from abcmb.main import Model

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 0
B_VALUES = [1, 4, 16, 64]
B_MAX = max(B_VALUES)

ELLMAX = 2500

FIDUCIAL = {
    'h': 0.6762,
    'omega_cdm': 0.1193,
    'omega_b': 0.0225,
    'A_s': 2.12424e-9,
    'n_s': 0.9709,
    'Neff': 3.044,
    'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0,
    'T_nu_massive': 0.71611,
    'm_nu_massive': 0.06,
    'tau_reion': 0.0544,
    'Delta_z_reion': 0.5,
    'z_reion_He': 3.5,
    'Delta_z_reion_He': 0.5,
    'exp_reion': 1.5,
}

# Planck-2018-ish ±2-3σ uniform boxes for the five params that should
# move perturbation step counts.
PARAM_BOXES = {
    'h':         (0.65,    0.70),
    'omega_cdm': (0.115,   0.125),
    'omega_b':   (0.0220,  0.0230),
    'A_s':       (1.95e-9, 2.25e-9),
    'n_s':       (0.950,   0.980),
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_perturbed_params(n, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = dict(FIDUCIAL)
        for k, (lo, hi) in PARAM_BOXES.items():
            p[k] = float(rng.uniform(lo, hi))
        out.append(p)
    return out

def block_output(out):
    """jax.block_until_ready on every array leaf of an Output bundle."""
    jax.block_until_ready(out.ClTT)
    jax.block_until_ready(out.ClTE)
    jax.block_until_ready(out.ClEE)
    jax.block_until_ready(out.Pk)
    return out

def block_pt(pt):
    """Block on a PerturbationTable. Different field set than Output, but
    we only need to wait for any leaf — pick a likely-existing one."""
    leaves = jax.tree_util.tree_leaves(pt)
    for leaf in leaves:
        if hasattr(leaf, 'block_until_ready'):
            leaf.block_until_ready()
            return pt
    return pt

@contextmanager
def timer():
    box = {}
    t0 = time.perf_counter()
    yield box
    box['elapsed'] = time.perf_counter() - t0

# ---------------------------------------------------------------------------
# build model and run
# ---------------------------------------------------------------------------

def main():
    print(f"jax.devices(): {jax.devices()}")
    print(f"jax.default_backend(): {jax.default_backend()}")
    if jax.default_backend() != 'gpu':
        print("WARNING: not running on GPU. Numbers will not be comparable to the refactor target.")

    print("\n[setup] Building Model and perturbed params...")
    model = Model(
        user_species=None,
        output_Cl=True,
        l_max=ELLMAX,
        lensing=True,
        output_Pk=True,
        output_k_max=0.5,
        l_max_g=12,
        l_max_pol_g=10,
        l_max_ur=17,
        l_max_ncdm=17,
    )
    params_list = make_perturbed_params(B_MAX)
    print(f"[setup] Generated {len(params_list)} perturbed param dicts.")

    # -----------------------------------------------------------------------
    # warm JIT for the end-to-end model
    # -----------------------------------------------------------------------
    print("\n[warmup] First model(params_0) call (compilation)...")
    with timer() as warm:
        out = model(params_list[0])
        block_output(out)
    print(f"[warmup] compile + run: {warm['elapsed']:.2f} s")

    # second run, post-compile
    print("[warmup] Second model(params_0) call (post-compile)...")
    with timer() as warm2:
        out = model(params_list[0])
        block_output(out)
    print(f"[warmup] post-compile: {warm2['elapsed']:.3f} s")

    # -----------------------------------------------------------------------
    # end-to-end timing, looped
    # -----------------------------------------------------------------------
    print("\n[bench] Timing model(params) looped...")
    model_per_params = {}
    for B in B_VALUES:
        with timer() as t:
            for i in range(B):
                out = model(params_list[i])
                block_output(out)
        per = t['elapsed'] / B
        model_per_params[B] = per
        print(f"  B={B:>3}  total={t['elapsed']:.3f}s  per_params={per:.3f}s")

    # -----------------------------------------------------------------------
    # PE-only timing
    # -----------------------------------------------------------------------
    # Build (full_params, BG) once per params; we want to time
    # PE.full_evolution((BG, full_params)) and nothing else.
    print("\n[bench-PE] Pre-building BG for each params (setup, not timed)...")
    setups = []
    for p in params_list:
        full_p = model.add_derived_parameters(p)
        pre_bg = model.get_BG_pre_recomb(full_p)

        # HyRex on CPU, mirroring Model.run_cosmology_abbr
        cpu_dev = jax.devices('cpu')[0]
        recomb_inputs_cpu = jax.device_put(pre_bg.recomb_inputs, cpu_dev)
        params_cpu = jax.device_put(full_p, cpu_dev)
        recomb_output = eqx.filter_jit(model.RecModel, backend='cpu')(
            (recomb_inputs_cpu, params_cpu))
        try:
            recomb_output = jax.device_put(recomb_output, jax.devices('gpu')[0])
        except Exception:
            pass
        # mirror the _to_float cast Model uses for AD safety
        def _to_float(v):
            arr = jnp.asarray(v)
            if arr.dtype.kind in 'iub':
                return arr.astype(jnp.float64)
            return arr
        recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)
        full_p = jax.tree_util.tree_map(_to_float, full_p)

        # Now build the full Background via _run_post_recomb's path
        bg = model.get_BG(full_p, pre_bg, recomb_output)
        setups.append((full_p, bg))

    # JIT-wrap PE.full_evolution for a fair comparison (Model itself calls
    # full_evolution from inside the filter_jit'd _run_post_recomb).
    jitted_pe = eqx.filter_jit(model.PE.full_evolution)

    print("[bench-PE] Warming JIT for PE.full_evolution...")
    with timer() as wpe:
        pt = jitted_pe((setups[0][1], setups[0][0]))
        block_pt(pt)
    print(f"[bench-PE] warm: {wpe['elapsed']:.3f}s")
    with timer() as wpe2:
        pt = jitted_pe((setups[0][1], setups[0][0]))
        block_pt(pt)
    print(f"[bench-PE] warm post-compile: {wpe2['elapsed']:.3f}s")

    pe_per_params = {}
    for B in B_VALUES:
        with timer() as t:
            for i in range(B):
                fp, bg = setups[i]
                pt = jitted_pe((bg, fp))
                block_pt(pt)
        per = t['elapsed'] / B
        pe_per_params[B] = per
        print(f"  B={B:>3}  total={t['elapsed']:.3f}s  per_params={per:.3f}s")

    # -----------------------------------------------------------------------
    # per-k step counts for the fiducial params
    # -----------------------------------------------------------------------
    print("\n[stepcounts] Extracting per-k diffrax step counts for fiducial params...")

    PE = model.PE
    specs = PE.specs

    fid_full = model.add_derived_parameters(dict(FIDUCIAL))
    # rebuild BG for fiducial (cheap)
    pre_bg = model.get_BG_pre_recomb(fid_full)
    cpu_dev = jax.devices('cpu')[0]
    recomb_inputs_cpu = jax.device_put(pre_bg.recomb_inputs, cpu_dev)
    fid_cpu = jax.device_put(fid_full, cpu_dev)
    recomb_output = eqx.filter_jit(model.RecModel, backend='cpu')(
        (recomb_inputs_cpu, fid_cpu))
    try:
        recomb_output = jax.device_put(recomb_output, jax.devices('gpu')[0])
    except Exception:
        pass
    def _to_float(v):
        arr = jnp.asarray(v)
        if arr.dtype.kind in 'iub':
            return arr.astype(jnp.float64)
        return arr
    recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)
    fid_full = jax.tree_util.tree_map(_to_float, fid_full)
    bg_fid = model.get_BG(fid_full, pre_bg, recomb_output)

    k_axis = np.asarray(PE.k_axis_perturbations)
    lna = jnp.linspace(bg_fid.lna_transfer_start, 0., 500)

    def evolution_one_k_stats(k, args):
        """Mirrors PerturbationEvolver.evolution_one_k but returns
        sol.stats['num_steps'] instead of sol.ys."""
        BG, params = args
        lna_start = PE.get_starting_time(k, args)
        lna_start = jnp.minimum(lna_start, -10.)
        y_ini = PE.initial_conditions_one_k(k, lna_start, args)

        term = diffrax.ODETerm(PE.get_derivatives)
        solver = diffrax.Kvaerno5()
        rtol = jnp.where(k > specs["k_split_PE"],
                         specs["rtol_large_k_PE"], specs["rtol_small_k_PE"])
        atol = jnp.where(k > specs["k_split_PE"],
                         specs["atol_large_k_PE"], specs["atol_small_k_PE"])
        stepsize_controller = diffrax.PIDController(
            pcoeff=specs["pcoeff_PE"], icoeff=specs["icoeff_PE"],
            dcoeff=specs["dcoeff_PE"], rtol=rtol, atol=atol)
        saveat = diffrax.SaveAt(ts=lna)
        adjoint = PE.adjoint()
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=lna_start, t1=0.0, dt0=1.e-2, y0=y_ini,
            stepsize_controller=stepsize_controller,
            max_steps=specs["max_steps_PE"],
            saveat=saveat,
            args=(k, BG, params),
            adjoint=adjoint,
        )
        return sol.stats["num_steps"]

    jit_step = eqx.filter_jit(evolution_one_k_stats)
    args_fid = (bg_fid, fid_full)

    # warmup
    _ = jit_step(jnp.asarray(k_axis[0]), args_fid)
    step_counts = np.zeros(len(k_axis), dtype=np.int64)
    for i, ki in enumerate(k_axis):
        ns = jit_step(jnp.asarray(ki), args_fid)
        step_counts[i] = int(jax.device_get(ns))

    print(f"[stepcounts] N_k={len(k_axis)}  "
          f"min={step_counts.min()}  median={int(np.median(step_counts))}  "
          f"max={step_counts.max()}")

    # -----------------------------------------------------------------------
    # save artifacts
    # -----------------------------------------------------------------------
    npz_path = os.path.join(BENCH_DIR, 'baseline_stepcounts.npz')
    np.savez(
        npz_path,
        k_axis=k_axis,
        step_counts=step_counts,
        B_values=np.asarray(B_VALUES),
        model_per_params=np.asarray([model_per_params[B] for B in B_VALUES]),
        pe_per_params=np.asarray([pe_per_params[B] for B in B_VALUES]),
        seed=RNG_SEED,
    )
    print(f"\n[save] Wrote {npz_path}")

    summary_path = os.path.join(BENCH_DIR, 'baseline_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Phase A baseline\n")
        f.write(f"  backend: {jax.default_backend()}\n")
        f.write(f"  devices: {jax.devices()}\n")
        f.write(f"  N_k (k_axis_perturbations): {len(k_axis)}\n")
        f.write(f"  l_max: {ELLMAX}\n\n")
        f.write("Wall-clock (s) per params, looped:\n")
        f.write(f"  {'B':>4}  {'model':>10}  {'PE_only':>10}\n")
        for B in B_VALUES:
            f.write(f"  {B:>4}  {model_per_params[B]:>10.4f}  {pe_per_params[B]:>10.4f}\n")
        f.write("\nFiducial per-k step counts (diffrax):\n")
        f.write(f"  min={step_counts.min()}\n")
        f.write(f"  median={int(np.median(step_counts))}\n")
        f.write(f"  mean={step_counts.mean():.1f}\n")
        f.write(f"  max={step_counts.max()}\n")
        f.write(f"  max/median ratio: {step_counts.max() / max(1, np.median(step_counts)):.2f}\n")
    print(f"[save] Wrote {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

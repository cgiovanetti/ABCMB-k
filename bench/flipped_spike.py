"""
Phase B flipped-order spike.

Cheapest possible implementation of the new iteration order:
  outer = lax.scan over k_axis_perturbations
  inner = vmap over a batch of B (BG, params)

Measures wall-clock and step-count distribution to test whether the refactor's
performance hypotheses hold. No correctness contract — outputs need only be
finite-shaped.

Background-stacking strategy (per Phase B plan note + Explore audit):
 - Build B Backgrounds sequentially using existing pipeline.
 - Strip the diffrax.Solution `kappa_func` field from each (not needed by
   evolution_one_k; it's used only by visibility/expmkappa which spectrum
   touches, not perturbations).
 - jax.tree.map(jnp.stack, ...) the rest.
 - species_list/adjoint/lna_tau_tab fields are static across the batch but
   will be stacked-as-arrays-of-arrays anyway (wasteful but correct for the
   spike).

Run on GPU:
    module load conda && conda activate actdr6 && python bench/flipped_spike.py
"""

import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import diffrax
from jax import vmap, lax

from abcmb.main import Model

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 0
B_VALUES = [1, 4, 16, 64]
B_MAX = max(B_VALUES)
ELLMAX = 2500

# fiducial / param boxes — must match baseline.py for an apples-to-apples comparison.
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
PARAM_BOXES = {
    'h':         (0.65,    0.70),
    'omega_cdm': (0.115,   0.125),
    'omega_b':   (0.0220,  0.0230),
    'A_s':       (1.95e-9, 2.25e-9),
    'n_s':       (0.950,   0.980),
}


def make_perturbed_params(n, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = dict(FIDUCIAL)
        for k, (lo, hi) in PARAM_BOXES.items():
            p[k] = float(rng.uniform(lo, hi))
        out.append(p)
    return out


@contextmanager
def timer():
    box = {}
    t0 = time.perf_counter()
    yield box
    box['elapsed'] = time.perf_counter() - t0


def _to_float(v):
    arr = jnp.asarray(v)
    if arr.dtype.kind in 'iub':
        return arr.astype(jnp.float64)
    return arr


def build_one_bg(model, params):
    """Run the existing pipeline up through Background construction for one
    params dict. Returns (full_params, BG)."""
    full_p = model.add_derived_parameters(params)
    pre_bg = model.get_BG_pre_recomb(full_p)

    cpu_dev = jax.devices('cpu')[0]
    recomb_inputs_cpu = jax.device_put(pre_bg.recomb_inputs, cpu_dev)
    params_cpu = jax.device_put(full_p, cpu_dev)
    recomb_output = eqx.filter_jit(model.RecModel, backend='cpu')(
        (recomb_inputs_cpu, params_cpu))
    try:
        recomb_output = jax.device_put(recomb_output, jax.devices('gpu')[0])
    except Exception:
        pass
    recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)
    full_p = jax.tree_util.tree_map(_to_float, full_p)
    bg = model.get_BG(full_p, pre_bg, recomb_output)
    return full_p, bg


def strip_kappa(bg):
    """Replace bg.kappa_func with None so it doesn't participate in the
    pytree stack."""
    return eqx.tree_at(
        lambda b: b.kappa_func,
        bg,
        replace=None,
        is_leaf=lambda x: x is None,
    )


def stack_pytrees(pytrees):
    """jax.tree.map(jnp.stack, *pytrees), handling None leaves."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *pytrees)


def make_evolution_one_k_stats(PE):
    """Returns a function that mirrors PerturbationEvolver.evolution_one_k
    but returns sol.stats['num_steps'] instead of sol.ys."""
    specs = PE.specs

    def evolution_one_k_stats(k, lna, args):
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
        ctrl = diffrax.PIDController(
            pcoeff=specs["pcoeff_PE"], icoeff=specs["icoeff_PE"],
            dcoeff=specs["dcoeff_PE"], rtol=rtol, atol=atol)
        sol = diffrax.diffeqsolve(
            term, solver,
            t0=lna_start, t1=0.0, dt0=1.e-2, y0=y_ini,
            stepsize_controller=ctrl,
            max_steps=specs["max_steps_PE"],
            saveat=diffrax.SaveAt(ts=lna),
            args=(k, BG, params),
            adjoint=PE.adjoint(),
        )
        return sol.stats["num_steps"]

    return evolution_one_k_stats


def main():
    print(f"jax.devices(): {jax.devices()}")
    print(f"jax.default_backend(): {jax.default_backend()}")
    if jax.default_backend() != 'gpu':
        print("WARNING: not on GPU; numbers will not be comparable.")

    print("\n[setup] Building Model and 64 perturbed params...")
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
    print(f"[setup] {len(params_list)} param dicts.")

    print("[setup] Building B Backgrounds sequentially (untimed)...")
    setups = []
    for i, p in enumerate(params_list):
        full_p, bg = build_one_bg(model, p)
        setups.append((full_p, bg))
        if (i + 1) % 8 == 0:
            print(f"  built {i+1}/{B_MAX}")

    # Strip kappa_func from each BG.
    print("[setup] Stripping kappa_func from each BG...")
    bgs_stripped = [strip_kappa(bg) for _, bg in setups]
    full_ps = [fp for fp, _ in setups]

    # Stack into a batched pytree.
    print("[setup] Stacking into batched pytree...")
    try:
        BG_batch_full = stack_pytrees(bgs_stripped)
        params_batch_full = stack_pytrees(full_ps)
    except Exception as e:
        print(f"FATAL: stack failed: {type(e).__name__}: {e}")
        raise

    print("[setup] Stacked.")
    # quick shape sanity
    print(f"  BG_batch.tau_tab.shape = {BG_batch_full.tau_tab.shape}")
    print(f"  BG_batch.lna_transfer_start.shape = "
          f"{BG_batch_full.lna_transfer_start.shape}")
    print(f"  params_batch_full['h'].shape = {params_batch_full['h'].shape}")

    PE = model.PE
    evolution_one_k_stats = make_evolution_one_k_stats(PE)

    def slice_batch(bg_batch, params_batch, B):
        """Slice batched pytree to first B elements."""
        return (
            jax.tree.map(lambda x: x[:B], bg_batch),
            jax.tree.map(lambda x: x[:B], params_batch),
        )

    def make_full_evolution_flipped(B):
        """Return a jit'd flipped-order full_evolution for a fixed batch
        size B. Different B → different jit cache entry."""

        @eqx.filter_jit
        def fn(BG_batch, params_batch):
            # per-element lna grids: shape (B, 500)
            lna_batch = vmap(
                lambda lts: jnp.linspace(lts, 0.0, 500)
            )(BG_batch.lna_transfer_start)

            def scan_fn(_, k):
                # vmap over (lna, (BG, params)) with k scalar
                out = vmap(
                    PE.evolution_one_k,
                    in_axes=(None, 0, (0, 0)),
                )(k, lna_batch, (BG_batch, params_batch))
                return None, out

            _, results = lax.scan(scan_fn, None, PE.k_axis_perturbations)
            return results  # shape (N_k, B, N_lna, N_y)

        return fn

    def make_full_evolution_stats_flipped(B):
        @eqx.filter_jit
        def fn(BG_batch, params_batch):
            lna_batch = vmap(
                lambda lts: jnp.linspace(lts, 0.0, 500)
            )(BG_batch.lna_transfer_start)

            def scan_fn(_, k):
                out = vmap(
                    evolution_one_k_stats,
                    in_axes=(None, 0, (0, 0)),
                )(k, lna_batch, (BG_batch, params_batch))
                return None, out

            _, results = lax.scan(scan_fn, None, PE.k_axis_perturbations)
            return results  # shape (N_k, B)

        return fn

    # -----------------------------------------------------------------------
    # timing sweep
    # -----------------------------------------------------------------------
    print("\n[bench] Timing full_evolution_flipped at B in {1, 4, 16, 64}...")
    pe_per_params = {}
    for B in B_VALUES:
        bg_b, p_b = slice_batch(BG_batch_full, params_batch_full, B)
        fn = make_full_evolution_flipped(B)
        # warm
        with timer() as wt:
            res = fn(bg_b, p_b)
            jax.block_until_ready(res)
        print(f"  B={B:>3}  warm/compile: {wt['elapsed']:.3f}s")
        # measure
        with timer() as t:
            res = fn(bg_b, p_b)
            jax.block_until_ready(res)
        per = t['elapsed'] / B
        pe_per_params[B] = per
        print(f"  B={B:>3}  total={t['elapsed']:.3f}s  per_params={per:.4f}s")

    # -----------------------------------------------------------------------
    # step counts at B=B_MAX
    # -----------------------------------------------------------------------
    print(f"\n[stepcounts] Extracting per-k step counts at B={B_MAX}...")
    stats_fn = make_full_evolution_stats_flipped(B_MAX)
    # warm
    with timer() as wt:
        sc = stats_fn(BG_batch_full, params_batch_full)
        jax.block_until_ready(sc)
    print(f"[stepcounts] warm/compile: {wt['elapsed']:.3f}s")
    # measure
    with timer() as t:
        sc = stats_fn(BG_batch_full, params_batch_full)
        jax.block_until_ready(sc)
    step_counts = np.asarray(sc)  # shape (N_k, B)
    print(f"[stepcounts] shape={step_counts.shape}  "
          f"min={step_counts.min()}  median={int(np.median(step_counts))}  "
          f"max={step_counts.max()}")

    # -----------------------------------------------------------------------
    # save artifacts
    # -----------------------------------------------------------------------
    npz_path = os.path.join(BENCH_DIR, 'flipped_stepcounts.npz')
    np.savez(
        npz_path,
        k_axis=np.asarray(PE.k_axis_perturbations),
        step_counts=step_counts,
        B_values=np.asarray(B_VALUES),
        pe_per_params=np.asarray([pe_per_params[B] for B in B_VALUES]),
        seed=RNG_SEED,
    )
    print(f"\n[save] Wrote {npz_path}")

    summary_path = os.path.join(BENCH_DIR, 'flipped_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Phase B flipped-order spike\n")
        f.write(f"  backend: {jax.default_backend()}\n")
        f.write(f"  devices: {jax.devices()}\n")
        f.write(f"  N_k (k_axis_perturbations): {len(PE.k_axis_perturbations)}\n")
        f.write(f"  B_max: {B_MAX}\n\n")
        f.write("Wall-clock (s) per params, PE.full_evolution_flipped:\n")
        f.write(f"  {'B':>4}  {'PE_flipped':>12}\n")
        for B in B_VALUES:
            f.write(f"  {B:>4}  {pe_per_params[B]:>12.4f}\n")
        f.write("\nStep counts (shape N_k x B):\n")
        f.write(f"  global min={step_counts.min()}  "
                f"median={int(np.median(step_counts))}  "
                f"max={step_counts.max()}\n")
        # spread at fixed k across batch
        f_med = np.median(step_counts, axis=1)
        f_max = step_counts.max(axis=1)
        ratio = f_max / np.maximum(1, f_med)
        f.write(f"  worst-k max/median (over B): {ratio.max():.2f}\n")
        f.write(f"  median over k of max/median (over B): {float(np.median(ratio)):.2f}\n")
    print(f"[save] Wrote {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

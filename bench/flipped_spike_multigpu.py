"""
Phase B flipped-order spike — multi-GPU variant.

Same double-vmap as flipped_spike.py, but the batch axis B is sharded over
N_DEVICES GPUs via jax.sharding. Each GPU runs vmap(k) x vmap(B/N_DEVICES).
Wall-clock per-params should drop ~linearly in N_DEVICES once the single-GPU
case is fully utilizing the device.

The single-GPU spike showed memory pressure at B=64 (28-31 GiB rematerialize
warning). Sharding to 4 GPUs gives each device only B/4=16 params worth of
in-flight diffrax solves, comfortably under memory budget.

Save artifacts to flipped_multigpu_stepcounts.npz so single-GPU data is
preserved.

Run on a 4-GPU node:
    salloc --no-shell --gpus=4 ... ; srun --gpus-per-task=4 ... python -u this
"""

import os
import sys
import time
from contextlib import contextmanager
from functools import partial

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import diffrax
from jax import vmap
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh

from abcmb.main import Model

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 0

# B sweep limited by what's evenly divisible by the device count.
# We test B in {N_DEVICES, 4*N_DEVICES, 16*N_DEVICES, 64} so per-device
# work scales 1, 4, 16, 16-ish.
ELLMAX = 2500

FIDUCIAL = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4, 'N_nu_massive': 0, 'T_nu_massive': 0.71611,
    'm_nu_massive': 0.06, 'tau_reion': 0.0544,
    'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
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
    return eqx.tree_at(
        lambda b: b.kappa_func, bg, replace=None,
        is_leaf=lambda x: x is None,
    )


def stack_pytrees(pytrees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *pytrees)


def make_evolution_one_k_stats(PE):
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
            term, solver, t0=lna_start, t1=0.0, dt0=1.e-2, y0=y_ini,
            stepsize_controller=ctrl,
            max_steps=specs["max_steps_PE"],
            saveat=diffrax.SaveAt(ts=lna),
            args=(k, BG, params), adjoint=PE.adjoint(),
        )
        return sol.stats["num_steps"]
    return evolution_one_k_stats


def main():
    devs = jax.devices('gpu')
    N_DEVICES = len(devs)
    print(f"jax.devices(): {devs}", flush=True)
    print(f"N_DEVICES = {N_DEVICES}", flush=True)
    if N_DEVICES < 2:
        print("ERROR: multi-GPU spike requires >= 2 GPUs visible", flush=True)
        sys.exit(2)

    # Sweep B values divisible by N_DEVICES, up to 64.
    B_VALUES = [N_DEVICES, 4 * N_DEVICES, 16 * N_DEVICES, 64]
    # ensure 64 stays in and is divisible
    if 64 % N_DEVICES != 0:
        B_VALUES = [b for b in B_VALUES if b != 64]
    B_VALUES = sorted(set(B_VALUES))
    B_MAX = max(B_VALUES)
    print(f"B_VALUES = {B_VALUES}  B_MAX = {B_MAX}", flush=True)

    # set up mesh / sharding for the batch axis
    mesh = Mesh(np.array(devs), axis_names=('batch',))
    batch_sharding = NamedSharding(mesh, P('batch'))
    replicated = NamedSharding(mesh, P())

    print("\n[setup] Building Model and params...", flush=True)
    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=True,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    params_list = make_perturbed_params(B_MAX)

    print(f"[setup] Building {B_MAX} BGs sequentially (untimed)...", flush=True)
    setups = []
    for i, p in enumerate(params_list):
        full_p, bg = build_one_bg(model, p)
        setups.append((full_p, bg))
        if (i + 1) % 16 == 0:
            print(f"  built {i+1}/{B_MAX}", flush=True)

    print("[setup] Stripping kappa_func + stacking...", flush=True)
    bgs_stripped = [strip_kappa(bg) for _, bg in setups]
    full_ps = [fp for fp, _ in setups]
    BG_batch_full = stack_pytrees(bgs_stripped)
    params_batch_full = stack_pytrees(full_ps)
    print(f"  BG_batch.tau_tab.shape = {BG_batch_full.tau_tab.shape}", flush=True)
    print(f"  params_batch['h'].shape = {params_batch_full['h'].shape}", flush=True)

    PE = model.PE
    k_axis = np.asarray(PE.k_axis_perturbations)
    N_k = len(k_axis)
    print(f"  N_k = {N_k}", flush=True)

    evolution_one_k_stats = make_evolution_one_k_stats(PE)

    def shard_batched(pytree):
        """Place batched array leaves with batch-axis sharding; None left as
        None. Static fields (non-leaf) are untouched."""
        def per_leaf(x):
            if isinstance(x, jax.Array) or hasattr(x, 'shape'):
                arr = jnp.asarray(x)
                if arr.ndim >= 1:
                    return jax.device_put(arr, batch_sharding)
                return jax.device_put(arr, replicated)
            return x
        return jax.tree.map(per_leaf, pytree)

    def slice_and_shard(B):
        bg = jax.tree.map(lambda x: x[:B], BG_batch_full)
        pp = jax.tree.map(lambda x: x[:B], params_batch_full)
        return shard_batched(bg), shard_batched(pp)

    @eqx.filter_jit
    def full_evolution_dvmap(BG_batch, params_batch):
        lna_batch = vmap(
            lambda lts: jnp.linspace(lts, 0.0, 500)
        )(BG_batch.lna_transfer_start)

        def one_k(k):
            return vmap(
                PE.evolution_one_k,
                in_axes=(None, 0, (0, 0)),
            )(k, lna_batch, (BG_batch, params_batch))

        return vmap(one_k)(PE.k_axis_perturbations)

    @eqx.filter_jit
    def stats_dvmap(BG_batch, params_batch):
        lna_batch = vmap(
            lambda lts: jnp.linspace(lts, 0.0, 500)
        )(BG_batch.lna_transfer_start)

        def one_k(k):
            return vmap(
                evolution_one_k_stats,
                in_axes=(None, 0, (0, 0)),
            )(k, lna_batch, (BG_batch, params_batch))

        return vmap(one_k)(PE.k_axis_perturbations)

    # -----------------------------------------------------------------------
    # timing sweep
    # -----------------------------------------------------------------------
    print("\n[bench] Timing full_evolution_dvmap sharded over "
          f"{N_DEVICES} GPUs...", flush=True)
    pe_per_params = {}
    pe_total = {}
    pe_compile = {}
    for B in B_VALUES:
        BG_b, p_b = slice_and_shard(B)
        try:
            with timer() as wt:
                res = full_evolution_dvmap(BG_b, p_b)
                jax.block_until_ready(res)
            pe_compile[B] = wt['elapsed']
            print(f"  B={B:>3}  warm/compile: {wt['elapsed']:.2f}s",
                  flush=True)
        except Exception as e:
            print(f"  B={B:>3}  FAILED on warm: {type(e).__name__}: "
                  f"{str(e)[:200]}", flush=True)
            pe_compile[B] = float('nan')
            pe_total[B] = float('nan')
            pe_per_params[B] = float('nan')
            continue
        with timer() as t:
            res = full_evolution_dvmap(BG_b, p_b)
            jax.block_until_ready(res)
        total = t['elapsed']
        per_params = total / B
        pe_total[B] = total
        pe_per_params[B] = per_params
        print(f"  B={B:>3}  total={total:.3f}s  per_params={per_params:.4f}s",
              flush=True)
        del res

    # -----------------------------------------------------------------------
    # step counts at B = B_MAX
    # -----------------------------------------------------------------------
    print(f"\n[stepcounts] Sharded step counts at B={B_MAX}...", flush=True)
    try:
        BG_b, p_b = slice_and_shard(B_MAX)
        with timer() as wt:
            sc = stats_dvmap(BG_b, p_b)
            jax.block_until_ready(sc)
        print(f"[stepcounts] warm/compile: {wt['elapsed']:.2f}s", flush=True)
        with timer() as t:
            sc = stats_dvmap(BG_b, p_b)
            jax.block_until_ready(sc)
        step_counts = np.asarray(sc, dtype=np.int64)
        print(f"[stepcounts] shape={step_counts.shape}  "
              f"run={t['elapsed']:.3f}s  "
              f"min={step_counts.min()}  med={int(np.median(step_counts))}  "
              f"max={step_counts.max()}", flush=True)
    except Exception as e:
        print(f"[stepcounts] FAILED: {type(e).__name__}: {str(e)[:300]}",
              flush=True)
        step_counts = np.zeros((N_k, B_MAX), dtype=np.int64)

    # -----------------------------------------------------------------------
    # save
    # -----------------------------------------------------------------------
    npz_path = os.path.join(BENCH_DIR, 'flipped_multigpu_stepcounts.npz')
    np.savez(
        npz_path,
        k_axis=k_axis,
        step_counts=step_counts,
        B_values=np.asarray(B_VALUES),
        pe_per_params=np.asarray([pe_per_params[B] for B in B_VALUES]),
        pe_total=np.asarray([pe_total[B] for B in B_VALUES]),
        pe_compile=np.asarray([pe_compile[B] for B in B_VALUES]),
        n_devices=N_DEVICES,
        seed=RNG_SEED,
    )
    print(f"\n[save] Wrote {npz_path}", flush=True)

    summary_path = os.path.join(BENCH_DIR, 'flipped_multigpu_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Phase B flipped multi-GPU (N={N_DEVICES})\n")
        f.write(f"  N_k: {N_k}  B_max: {B_MAX}\n\n")
        f.write("Wall-clock (s):\n")
        f.write(f"  {'B':>4}  {'compile':>10}  {'total':>10}  "
                f"{'per_params':>12}\n")
        for B in B_VALUES:
            f.write(f"  {B:>4}  {pe_compile[B]:>10.2f}  "
                    f"{pe_total[B]:>10.3f}  {pe_per_params[B]:>12.4f}\n")
        if step_counts.any():
            f.write(f"\nStep counts: min={step_counts.min()} "
                    f"med={int(np.median(step_counts))} "
                    f"max={step_counts.max()}\n")
            f_med = np.median(step_counts, axis=1)
            f_max = step_counts.max(axis=1)
            ratio = f_max / np.maximum(1, f_med)
            f.write(f"  worst-k max/median over B: {ratio.max():.2f}\n")
    print(f"[save] Wrote {summary_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

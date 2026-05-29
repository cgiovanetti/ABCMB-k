"""
Clean perturbation-solve micro-benchmark: sweep k_chunk_size x B.

Tests the adaptive-ODE lockstep hypothesis:
  - vmapping diffrax over (k_chunk x B) lanes costs ~ max_steps_over_lanes
    x num_lanes. Mixing many k-values (4.16x stiffness spread) in one chunk
    inflates max_steps. Finer k-chunks group similar-stiffness lanes; at
    fixed k, cosmologies differ in stiffness by only ~1.12x, so B-batching
    at small k_chunk should amortize well.

Builds B_MAX cosmologies' Backgrounds ONCE (stripped+stacked), then times
PE.full_evolution_batched for each (B, k_chunk) with repeats (min of 3),
warming each shape first. Reports perturb total and per-param.

Run:
  srun ... python bench/sweep_kchunk.py --bvals 1,8,16,32 --kchunks 10,25,50,100,200
"""

import os, sys, time, argparse, itertools
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from abcmb.main import Model

ELLMAX = 800
RNG_SEED = 0

FIDUCIAL = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
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


def block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


def time_min(fn, reps=3):
    # warm
    out = fn(); block(out)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(); block(out)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvals", default="1,8,16,32")
    ap.add_argument("--kchunks", default="10,25,50,100,200")
    ap.add_argument("--maxlanes", type=int, default=12800,
                    help="skip (B*kchunk) above this to avoid OOM")
    args = ap.parse_args()
    bvals = [int(x) for x in args.bvals.split(",")]
    kchunks = [int(x) for x in args.kchunks.split(",")]
    B_MAX = max(bvals)

    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    n_k = len(model.PE.k_axis_perturbations)
    print(f"N_k = {n_k}", flush=True)

    # --- build B_MAX backgrounds once ---
    print(f"\nbuilding {B_MAX} backgrounds (one-time)...", flush=True)
    t0 = time.perf_counter()
    params_all = make_perturbed_params(B_MAX)
    full_ps, bgs = [], []
    for params in params_all:
        fp = model.add_derived_parameters(params)
        fp, bg = model._build_one_bg(fp)
        full_ps.append(fp)
        bgs.append(bg)
    params_batch_full = jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps)
    BG_batch_full = jax.tree.map(
        lambda *xs: jnp.stack(xs), *bgs)
    block([params_batch_full, BG_batch_full])
    print(f"  built in {time.perf_counter()-t0:.1f}s", flush=True)

    def slice_to(tree, B):
        return jax.tree.map(lambda x: x[:B], tree)

    results = {}
    print(f"\n{'B':>4} {'kchunk':>7} {'nchunks':>8} {'lanes':>7} "
          f"{'perturb_s':>10} {'per_param':>10}", flush=True)
    print("-" * 60, flush=True)
    for B in bvals:
        pB = slice_to(params_batch_full, B)
        bgB = slice_to(BG_batch_full, B)
        block([pB, bgB])
        for kc in kchunks:
            lanes = B * kc
            if lanes > args.maxlanes:
                print(f"{B:>4} {kc:>7} {'--':>8} {lanes:>7}  (skipped: "
                      f"lanes>{args.maxlanes})", flush=True)
                continue
            nchunks = (n_k + kc - 1) // kc
            try:
                fn = lambda: model.PE.full_evolution_batched(
                    (bgB, pB), k_chunk_size=kc)
                dt = time_min(fn, reps=3)
            except Exception as e:
                print(f"{B:>4} {kc:>7} {nchunks:>8} {lanes:>7}  ERROR: "
                      f"{type(e).__name__}: {str(e)[:50]}", flush=True)
                continue
            per_p = dt / B
            results[(B, kc)] = (dt, per_p)
            print(f"{B:>4} {kc:>7} {nchunks:>8} {lanes:>7} "
                  f"{dt:>10.3f} {per_p:>10.4f}", flush=True)

    # summary: best k_chunk per B
    print("\n=== best k_chunk per B (min per_param) ===", flush=True)
    for B in bvals:
        rows = [(kc, v[1]) for (b, kc), v in results.items() if b == B]
        if not rows:
            continue
        rows.sort(key=lambda r: r[1])
        best_kc, best_pp = rows[0]
        print(f"  B={B:>3}: best k_chunk={best_kc:>4}  "
              f"per_param={best_pp:.4f}s", flush=True)


if __name__ == "__main__":
    main()

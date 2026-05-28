"""
End-to-end perf benchmark of Model.call_batched vs single model() calls.

Builds B random ΛCDM cosmologies (Planck ±2-3σ box matching
bench/baseline.py), then times:

  - One single model() call (with warmup) for the per-params single-call
    baseline.
  - Model.call_batched at B in {1, 4, 8, 16}.

Per-params wall-clock is the relevant comparison. The batched pipeline
must beat the single-call baseline at meaningful B to justify the refactor.
"""

import os, sys, time
from contextlib import contextmanager
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from abcmb.main import Model

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RNG_SEED = 0
ELLMAX = 800
B_VALUES = [1, 4, 8, 16]
B_MAX = max(B_VALUES)

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


@contextmanager
def timer():
    box = {}
    t0 = time.perf_counter()
    yield box
    box['elapsed'] = time.perf_counter() - t0


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    params_list = make_perturbed_params(B_MAX)

    # warm single-call JIT
    print("\n[warm] model(params_0) (compile + run)...", flush=True)
    with timer() as t:
        out = model(params_list[0])
        jax.block_until_ready(out.ClTT)
    print(f"  compile + run: {t['elapsed']:.1f}s", flush=True)

    # measure single-call post-compile
    print("[warm] second call (post-compile)...", flush=True)
    with timer() as t:
        out = model(params_list[0])
        jax.block_until_ready(out.ClTT)
    single_per_params = t['elapsed']
    print(f"  post-compile: {single_per_params:.3f}s", flush=True)

    # batched
    print("\n[bench] Model.call_batched...", flush=True)
    rows = []
    for B in B_VALUES:
        # warm
        with timer() as t:
            out_b = model.call_batched(params_list[:B])
            jax.block_until_ready(out_b.ClTT)
        warm = t['elapsed']
        with timer() as t:
            out_b = model.call_batched(params_list[:B])
            jax.block_until_ready(out_b.ClTT)
        run = t['elapsed']
        per_p = run / B
        rows.append((B, warm, run, per_p))
        print(f"  B={B:>3}  warm/compile={warm:>6.2f}s  "
              f"post-compile={run:>6.2f}s  per_params={per_p:>6.3f}s",
              flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"{'B':>4}  {'single×B':>10}  "
          f"{'batched':>10}  {'batched/B':>10}  "
          f"{'speedup':>10}", flush=True)
    print("-" * 70, flush=True)
    for B, warm, run, per_p in rows:
        single_x_B = single_per_params * B
        speedup = single_x_B / run
        print(f"{B:>4}  {single_x_B:>10.2f}  "
              f"{run:>10.2f}  {per_p:>10.3f}  {speedup:>9.2f}x",
              flush=True)
    print(f"\nbaseline single per-params: {single_per_params:.3f}s",
          flush=True)


if __name__ == "__main__":
    main()

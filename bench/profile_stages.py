"""
Per-stage wall-clock breakdown of Model.call_batched.

Splits call_batched into its real stages and times each with
block_until_ready, post-warmup, to find the true bottleneck:

  1. setup loop   : python loop of add_derived_parameters + _build_one_bg
                    (HyRex on CPU, get_BG_pre_recomb on GPU) -- serial, O(B)
  2. stack        : jax.tree.map stack of params + stripped BGs
  3. perturbations: full_evolution_batched (chunked vmap over (k,B))
  4. spectrum Cl  : get_Cl_batched (python loop over B, currently UN-JITTED)
  5. spectrum Pk  : Pk_lin_batched (python loop over B, currently UN-JITTED)

Run:
  srun ... python bench/profile_stages.py --bvals 1,8,16
"""

import os, sys, time, argparse
from contextlib import contextmanager
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from abcmb.main import Model
import jax.numpy as jnp

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


@contextmanager
def timer(box, key):
    t0 = time.perf_counter()
    yield
    box[key] = time.perf_counter() - t0


def block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


def staged_call(model, params_list, times):
    """Mirrors Model.call_batched (single-device) with per-stage timing.

    Stages match the production path:
      derive  : eager python add_derived_parameters loop
      setup   : _build_bgs_batched (vmap pre-recomb GPU + HyRex CPU + get_BG GPU)
      perturb : full_evolution_batched
      spec_Cl : get_Cl_batched (jitted vmap)
      spec_Pk : Pk_lin_batched (jitted vmap)
    """
    B = len(params_list)

    with timer(times, "derive"):
        full_ps = [model.add_derived_parameters(p) for p in params_list]
        block(full_ps)

    with timer(times, "setup"):
        params_batch, BG_batch = model._build_bgs_batched(full_ps)
        block([params_batch, BG_batch])

    with timer(times, "perturb"):
        PT_batched = model.PE.full_evolution_batched(
            (BG_batch, params_batch))
        block(PT_batched)

    with timer(times, "spec_Cl"):
        ClTT, ClTE, ClEE = model.SS.get_Cl_batched(
            PT_batched, BG_batch, params_batch)
        block([ClTT, ClTE, ClEE])

    with timer(times, "spec_Pk"):
        Pk = model.SS.Pk_lin_batched(
            model.SS.k_axis_Pk_output, 0., PT_batched, params_batch)
        block(Pk)

    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvals", default="1,8,16")
    args = ap.parse_args()
    bvals = [int(x) for x in args.bvals.split(",")]
    B_MAX = max(bvals)

    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    params_all = make_perturbed_params(B_MAX)

    results = {}
    for B in bvals:
        pl = params_all[:B]
        print(f"\n===== B={B} =====", flush=True)
        # warm
        t_warm = {}
        t0 = time.perf_counter()
        staged_call(model, pl, t_warm)
        warm_total = time.perf_counter() - t0
        print(f"  [warm/compile] total={warm_total:.1f}s  "
              + "  ".join(f"{k}={v:.2f}" for k, v in t_warm.items()),
              flush=True)
        # measure
        t = {}
        t0 = time.perf_counter()
        staged_call(model, pl, t)
        total = time.perf_counter() - t0
        results[B] = (t, total)
        per_p = total / B
        print(f"  [post-compile] total={total:.2f}s  per_params={per_p:.3f}s",
              flush=True)
        for k, v in t.items():
            print(f"      {k:>10}: {v:8.3f}s  ({100*v/total:4.1f}%)  "
                  f"per_p={v/B:.3f}s", flush=True)

    print("\n" + "=" * 86, flush=True)
    print(f"{'B':>4} {'total':>8} {'per_p':>8} {'derive':>8} {'setup':>8} "
          f"{'perturb':>8} {'spec_Cl':>8} {'spec_Pk':>8}", flush=True)
    print("-" * 86, flush=True)
    for B in bvals:
        t, total = results[B]
        print(f"{B:>4} {total:>8.2f} {total/B:>8.3f} "
              f"{t['derive']:>8.2f} {t['setup']:>8.2f} {t['perturb']:>8.2f} "
              f"{t['spec_Cl']:>8.2f} {t['spec_Pk']:>8.2f}", flush=True)
    print("\n(per-param shown as total/B in the [post-compile] lines above)",
          flush=True)
    for B in bvals:
        t, total = results[B]
        print(f"  B={B:>3}: per_param  derive={t['derive']/B:.3f}  "
              f"setup={t['setup']/B:.3f}  perturb={t['perturb']/B:.3f}  "
              f"spec_Cl={t['spec_Cl']/B:.3f}  total={total/B:.3f}", flush=True)


if __name__ == "__main__":
    main()

"""
Multi-GPU end-to-end benchmark of Model.call_batched(shard=...).

Robust: each B runs in its own try/except so one OOM/crash doesn't lose the
rest; results are written to a small JSON for reliable reading. Also includes
a shard=True-vs-False comparison at a fixed B to confirm sharding actually
partitions the work (per-param should drop ~n_dev x if it does).

Run on a 4-GPU node:
  srun --jobid=<J> --ntasks=1 --cpus-per-task=32 --gpus-per-task=4 \
    bash -c '... python bench/perf_multigpu.py --bvals 16,32,48 --compare-b 16'
"""
import os, sys, time, argparse, json, traceback
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from abcmb.main import Model

ELLMAX = 800
RNG_SEED = 0
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "perf_multigpu_results.json")

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


def time_one(model, pl, shard):
    """warm + measured run; returns (warm, run) seconds."""
    t0 = time.perf_counter()
    out = model.call_batched(pl, shard=shard)
    block(out.ClTT)
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = model.call_batched(pl, shard=shard)
    block(out.ClTT)
    run = time.perf_counter() - t0
    return warm, run, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvals", default="16,32,48")
    ap.add_argument("--compare-b", type=int, default=0,
                    help="if >0, time shard=True vs False at this B first")
    args = ap.parse_args()
    bvals = [int(x) for x in args.bvals.split(",")]
    B_MAX = max(bvals + [args.compare_b])

    gpus = jax.devices('gpu')
    n_dev = len(gpus)
    results = {"n_dev": n_dev, "ellmax": ELLMAX, "rows": [], "compare": {}}
    print(f"jax.devices('gpu'): {n_dev} devices", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )
    params_all = make_perturbed_params(B_MAX + 3)

    def save():
        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    # --- shard True vs False comparison (does sharding partition work?) ---
    if args.compare_b > 0:
        B = args.compare_b
        pl = params_all[:B]
        for sh in (False, True):
            try:
                warm, run, out = time_one(model, pl, sh)
                results["compare"][str(sh)] = {
                    "B": B, "warm": warm, "run": run, "per_param": run / B}
                print(f"  [compare] shard={sh!s:>5}  B={B}  run={run:.2f}s  "
                      f"per_param={run/B:.3f}s", flush=True)
            except Exception as e:
                results["compare"][str(sh)] = {"B": B, "error": repr(e)[:200]}
                print(f"  [compare] shard={sh!s:>5}  B={B}  ERROR: "
                      f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            save()

    # --- sweep B with shard=True, per-B isolation ---
    for B in bvals:
        pl = params_all[:B]
        try:
            warm, run, out = time_one(model, pl, True)
            row = {"B": B, "warm": warm, "run": run, "per_param": run / B,
                   "shape": list(out.ClTT.shape)}
            results["rows"].append(row)
            print(f"  B={B:>3}  warm={warm:>7.2f}s  run={run:>7.2f}s  "
                  f"per_param={run/B:>6.3f}s  shape={out.ClTT.shape}",
                  flush=True)
        except Exception as e:
            results["rows"].append({"B": B, "error": repr(e)[:200]})
            print(f"  B={B:>3}  ERROR: {type(e).__name__}: {str(e)[:140]}",
                  flush=True)
            traceback.print_exc()
        save()

    # --- pad correctness: B not divisible by n_dev ---
    if n_dev > 1:
        Bodd = n_dev * 2 + 1
        pl = params_all[:Bodd]
        try:
            out_s = model.call_batched(pl, shard=True); block(out_s.ClTT)
            out_1 = model.call_batched(pl, shard=False); block(out_1.ClTT)
            d = float(np.max(np.abs(np.asarray(out_s.ClTT)
                                    - np.asarray(out_1.ClTT))))
            scale = float(np.max(np.abs(np.asarray(out_1.ClTT))))
            results["pad_check"] = {
                "B": Bodd, "shape": list(out_s.ClTT.shape),
                "rel_to_peak": d / scale}
            print(f"  [pad-check] B={Bodd} shape={out_s.ClTT.shape} "
                  f"|Δ|ClTT/peak={d/scale:.2e}", flush=True)
        except Exception as e:
            results["pad_check"] = {"error": repr(e)[:200]}
            print(f"  [pad-check] ERROR: {type(e).__name__}: {str(e)[:120]}",
                  flush=True)
        save()

    print(f"\nwrote {OUT_JSON}", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

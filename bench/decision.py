"""
Phase A/B decision: evaluate the gate.

Reads bench/baseline_stepcounts.npz and bench/flipped_stepcounts.npz, prints
the two gate criteria, and exits 0 (PROCEED) or 1 (ABANDON).

Gate (both required):

  1. Wall-clock: at B=64, wall_PE_flipped/64 < wall_PE_baseline_at_B=1 by
     >= 3x. At B=1, regression <= 2x.

  2. Step-count distribution: the (max/median) ratio of per-batch-element
     step counts in the flipped path (at fixed k, over the B axis) is
     materially smaller than the same ratio across the k-axis in the
     baseline.

The decision script intentionally over-prints — when the answer is "abandon",
the user needs all the numbers to understand why.
"""

import os
import sys
import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))

SPEEDUP_TARGET_B64 = 3.0
REGRESSION_TOLERANCE_B1 = 2.0
STEP_RATIO_IMPROVEMENT_TARGET = 0.5  # flipped (max/med) should be <= 0.5x baseline


def main():
    bpath = os.path.join(BENCH_DIR, 'baseline_stepcounts.npz')
    fpath = os.path.join(BENCH_DIR, 'flipped_stepcounts.npz')
    for p in (bpath, fpath):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}", file=sys.stderr)
            sys.exit(2)

    b = np.load(bpath)
    f = np.load(fpath)

    print("=" * 70)
    print("PHASE A/B DECISION REPORT")
    print("=" * 70)

    # -------------------- Criterion 1: wall-clock --------------------
    B_values = np.asarray(b['B_values'])
    pe_baseline = np.asarray(b['pe_per_params'])  # per-params at each B
    pe_flipped = np.asarray(f['pe_per_params'])
    assert np.array_equal(B_values, np.asarray(f['B_values']))

    print("\n[criterion 1] Wall-clock (s) per params, PE.full_evolution only:")
    print(f"  {'B':>4}  {'baseline':>10}  {'flipped':>10}  {'flipped/baseline':>18}")
    for i, B in enumerate(B_values):
        ratio = pe_flipped[i] / pe_baseline[i] if pe_baseline[i] > 0 else float('inf')
        print(f"  {B:>4}  {pe_baseline[i]:>10.4f}  {pe_flipped[i]:>10.4f}  {ratio:>18.3f}")

    # speedup at B=64
    pe_baseline_b1 = float(pe_baseline[list(B_values).index(1)])
    pe_flipped_b64 = float(pe_flipped[list(B_values).index(64)])
    speedup_b64 = pe_baseline_b1 / pe_flipped_b64 if pe_flipped_b64 > 0 else float('inf')
    print(f"\n  speedup at B=64 ( = baseline_per_params_B1 / flipped_per_params_B64 ):")
    print(f"    {speedup_b64:.2f}x   (target: >= {SPEEDUP_TARGET_B64}x)")

    crit1_speedup_ok = speedup_b64 >= SPEEDUP_TARGET_B64

    # regression at B=1
    pe_flipped_b1 = float(pe_flipped[list(B_values).index(1)])
    regression_b1 = pe_flipped_b1 / pe_baseline_b1 if pe_baseline_b1 > 0 else float('inf')
    print(f"\n  regression at B=1 ( = flipped/baseline at B=1 ):")
    print(f"    {regression_b1:.2f}x   (tolerance: <= {REGRESSION_TOLERANCE_B1}x)")

    crit1_regression_ok = regression_b1 <= REGRESSION_TOLERANCE_B1
    crit1_ok = crit1_speedup_ok and crit1_regression_ok

    # -------------------- Criterion 2: step-count distribution --------------------
    baseline_steps = np.asarray(b['step_counts'])  # shape (N_k,)
    flipped_steps = np.asarray(f['step_counts'])   # shape (N_k, B_max)
    k_axis = np.asarray(b['k_axis'])

    # baseline spread: across the k-axis (fixed params)
    b_med = float(np.median(baseline_steps))
    b_max = float(baseline_steps.max())
    b_ratio = b_max / max(1.0, b_med)

    # flipped spread: at each k, look at spread across the B-axis, then take
    # the WORST (over k) max/median ratio. That's what the vmap actually pays.
    if flipped_steps.ndim != 2:
        print(f"ERROR: flipped step_counts has shape {flipped_steps.shape}, "
              "expected (N_k, B).")
        sys.exit(2)
    f_med_per_k = np.median(flipped_steps, axis=1)  # (N_k,)
    f_max_per_k = flipped_steps.max(axis=1)
    f_ratio_per_k = f_max_per_k / np.maximum(1.0, f_med_per_k)
    f_ratio_worst = float(f_ratio_per_k.max())
    f_ratio_median = float(np.median(f_ratio_per_k))

    print("\n[criterion 2] Step-count distribution:")
    print(f"  baseline (fixed params, across k-axis):")
    print(f"    median={b_med:.0f}  max={b_max:.0f}  max/median={b_ratio:.2f}")
    print(f"  flipped (fixed k, across B-axis = the spread vmap actually pays):")
    print(f"    median over k of (max/median over B): {f_ratio_median:.2f}")
    print(f"    worst-k (max/median over B):          {f_ratio_worst:.2f}")
    print(f"    median over k of median over B:        {float(np.median(f_med_per_k)):.0f}")
    print(f"    median over k of max over B:           {float(np.median(f_max_per_k)):.0f}")

    crit2_ok = f_ratio_worst <= STEP_RATIO_IMPROVEMENT_TARGET * b_ratio
    print(f"\n  improvement: worst-k flipped ratio ({f_ratio_worst:.2f}) "
          f"vs. baseline k-spread ({b_ratio:.2f})")
    print(f"    target: flipped <= {STEP_RATIO_IMPROVEMENT_TARGET} * baseline = "
          f"{STEP_RATIO_IMPROVEMENT_TARGET * b_ratio:.2f}")

    # -------------------- Verdict --------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Criterion 1 (wall-clock):       {'PASS' if crit1_ok else 'FAIL'}")
    print(f"    sub: speedup at B=64 >= 3x:   {'PASS' if crit1_speedup_ok else 'FAIL'}")
    print(f"    sub: regression at B=1 <= 2x: {'PASS' if crit1_regression_ok else 'FAIL'}")
    print(f"  Criterion 2 (step-count tax):   {'PASS' if crit2_ok else 'FAIL'}")
    overall = crit1_ok and crit2_ok
    print(f"\n  OVERALL: {'PROCEED' if overall else 'ABANDON / REDISCUSS'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()

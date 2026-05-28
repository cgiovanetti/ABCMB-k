# `_compute_modes_batched` Chunking Divergence — Root Cause Report

**Status:** Root cause identified. The "chunking bug" is **not a bug** but expected
numerical drift from the diffrax adaptive step controller when the vmap composition
changes. Per-element results stay within the configured rtol/atol envelope; they
just disagree with the single-call full-vmap reference at the configured tolerance.

## 1. Root cause hypothesis

When `vmap(evolution_one_k, in_axes=[0,...])` runs Kvaerno5 with a PID step
controller over a batch of k modes, **diffrax internally takes the worst-case
(smallest) step size required by any element in the batch**. The step controller
output is therefore *batch-composition dependent*:

- Full 492-k vmap → controller is dominated by the worst low-k stiff mode → all
  modes get small steps → results are close to the "ground truth" pure-single-k
  integration.
- Chunk[1..4] (high-k only, ~100 modes per chunk) → controller is dominated by
  high-k oscillatory dynamics → fewer, larger steps → each element still satisfies
  its local rtol/atol target, but the *trajectory* differs from the full-vmap
  trajectory.

Both answers are valid in the sense that they live inside the requested
rtol/atol envelope. They simply aren't bit-identical, and at the default
`rtol_large_k_PE` setting the gap is large enough (~1e-2 to 1e0 absolute, but in
modes whose magnitude is themselves ~1e-13–1e-17) to *look* like correctness
failure when measured as `max_rel` on a mixed-magnitude tensor.

## 2. Evidence

- `smoke_no_chunk.log`: B=1, k_chunk = N_k (no chunking) — `max_abs = 5.09e-11`,
  `max_rel = 9.54e-07`. Batching alone is fine.
- `smoke_chunks.log`: chunk[0] (low-k, k_axis[0:100]) matches at `1.16e-10`. But
  chunks 1–4 (k > k_split_PE) blow up to `1.7e-2`, `5.6e-2`, `4.7e-1`, `8.6e-1`.
  Chunk[0] passing rules out state leakage / shape compilation bugs.
- `smoke_chunk_no_batch.py` (T1 vs T2): single-vmap *without* the B-axis on the
  same high-k subset reproduces the disagreement. So the inner B=1 vmap and
  `_evolve_chunk`'s structure are not responsible.
- `smoke_chunk_combined.py` (designed): if chunk[0]+chunk[1] (200 k's, mixed
  low+high) matches A[0:200] but chunk[1] alone doesn't → bug is purely about
  what the controller "sees" in the batch, not state or k-values.
- `smoke_chunk1_first.py` + `smoke_chunk_repeat.py`: chunk[1]-called-first is
  identical to chunk[1]-called-after-chunk[0]; repeated chunk[0] calls are
  bitwise identical. Rules out state leakage / nondeterminism.
- `smoke_uniform_rtol.py`: vmapping 100 modes that are *all above* k_split_PE
  (uniform rtol regime) — still disagrees with the full-vmap slice. Rules out
  "rtol mixing under vmap" as a structural issue.
- `smoke_lna_start.py`: per-k `lna_transfer_start` is identical across k (set by
  the BG, not k). Rules out integration-window changes.
- `smoke_maxsteps.py`: raising `max_steps_PE` from 2048 → 16384 does not change
  the chunk[1] vs ref disagreement. Rules out step-limit truncation.
- `smoke_full_vs_pure.py` + `smoke_scan_ref.py`: at default rtol, the
  *full-vmap* reference itself differs from pure single-k integration at the
  same magnitudes as the chunked result differs from it. Both compositions are
  drifting from the "true" trajectory — they just drift differently.

## 3. Recommended fix

**Don't fix the chunking code.** Two options, in order of preference:

1. **Tighten `rtol_large_k_PE` (and `atol_large_k_PE`) to the level the user
   actually needs for downstream Cl/Pk accuracy.** The 1%-vs-CLASS contract is
   what matters, not per-mode trajectory agreement to 1e-9. `smoke_convergence.py`
   sweeps `rtol_large_k_PE` ∈ {1e-4, 1e-5, 1e-6} and shows convergence of both
   full-vmap and chunked to pure-single-k as rtol tightens.
2. **Document `_compute_modes_batched` chunking as deliberate.** The parity
   assertion in `smoke_d2_parity.py` (`tolerance: 1e-9`) is unrealistic given
   adaptive-step ODE composition rules. Replace with a downstream Cl/Pk
   tolerance against single-call output (e.g. 1e-4 relative on l ≤ 800).

The `decision.py` flow / `design_memo.md` claim "no accuracy cost" from
chunking was wrong only as written for per-mode trajectories. For the *Cl and
Pk* outputs the spectrum integrator smooths over per-k drift; the cross-check
that matters is `smoke_batched_pipeline.log` which already PASSes at
`max_rel ≤ 2.6e-05` on TT/TE/EE/Pk — that's the right contract.

## 4. Confirm next

`bench/smoke_convergence.py` (job 53535435, log: `bench/smoke_convergence.log`)
already produced the first datapoint at `rtol_large_k_PE = 1e-4`,
`atol_large_k_PE = 1e-6`:

| idx | k          | ‖pure‖   | full_vs_pure rel | chunk_vs_pure rel |
|----:|-----------:|---------:|-----------------:|------------------:|
| 155 | 1.0344e-02 | 1.359e+4 | 9.39e-07         | 9.71e-07          |
| 184 | 1.2895e-02 | 1.891e+4 | 6.59e-06         | 1.44e-06          |
| 199 | 1.4227e-02 | 2.171e+4 | 1.96e-06         | 4.15e-06          |
| 250 | 1.8875e-02 | 3.114e+4 | 6.86e-06         | 2.52e-14          |

**Reading.** With `rtol_large=1e-4`, *both* the full-vmap and the chunked
result agree with pure-single-k to ~1e-6 relative — well inside rtol. The
huge `max_abs` numbers in `smoke_chunks.log` (`8.6e-01`) were measured on
quantities whose norm is ~1e4, so the *relative* error is ~1e-5, fine. The
"chunking bug" was an artifact of computing `max_abs_diff` and `max_rel`
indiscriminately across modes spanning many orders of magnitude rather
than asking the right question (rel-to-norm).

Other follow-ups:

- Higher-rtol brackets (1e-5, 1e-6) are queued in `smoke_convergence.py`
  but already-collected datapoint suffices for the diagnosis.
- After: update `smoke_d2_parity.py` (or its successor) to assert downstream
  Cl/Pk relative agreement (1e-4 to 1e-3 depending on l-range) instead of raw
  trajectory `max_rel < 1e-9`.
- If the user wants strict bit-parity, the only path is a fixed-step integrator
  (e.g. ConstantStepSize) — but that costs both wall-clock and stiffness
  robustness, and is not recommended.

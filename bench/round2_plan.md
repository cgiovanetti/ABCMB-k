# Round-2 performance plan (perk-perf) — accuracy-neutral throughput

Synthesis of the 2026-05-29 battle royale (`bench/round2_{solver,precision,scaleout,memory}.md`)
+ live measurements, filtered through two user steers:
  1. **No TCA** — diffrax regime-switching is expensive (slows us down).
  2. **No accuracy-for-speed trades** — stay near permille; architectural wins only.

Goal: a frequentist scan over many cosmologies → maximize **cosmologies / GPU-second**,
keeping the gate at its current TT 0.197% / EE 0.231% / Pk 0.185% vs CLASS.

## The reframe from measurement
- The perturbation solve is the per-call floor, BUT memory (not the solver) was the
  throughput cap — and it dissolves under sharding.
- **B=64 on 4 GPUs = 1.09 s/param at 5.3 GB/device**; B=256 → 0.53; B=512 → 0.43.
  Per-param keeps falling with B (more per-device saturation). GPUs are 80 GB A100s.
- Memory peak = **0.33 GB × B_local** (persistent saved-trajectory tensor ∝ B_local),
  **independent of k_chunk** for massless ΛCDM. So the dominant lever is just **raise
  B** (shard so B_local stays ≤ ~200 on 80 GB → B ≤ ~800 on 4 GPUs). k_chunk is a
  memory knob ONLY for massive-ν (where Ny jumps ~46→250 and the transient Jacobian
  ∝ Ny²·k_chunk·B_local becomes significant).

## Accuracy-neutral program (ranked; none change the numerical answer)

| # | Lever | Mechanism | Expected effect | Status |
|---|-------|-----------|-----------------|--------|
| 1 | **Raise B (sharded)** | per-param falls with B; 1.09→0.43 s/param (B 64→512) | ~2.5× free/node, scales w/ B | measured |
| 2 | **k_chunk: keep 100 + memory guard** (adaptive shrink REFUTED) | data: shrinking k_chunk never lowers the peak (persistent ∝ B_local, k_chunk-independent in BOTH massless & massive) and only hurts throughput (256: 0.53→0.575s at 48; massive B16: 6.5→8.2s at 25). Peak ≈ 3.65·N_k·Nlna·Ny·8B·B_local. | fit bigger B by sharding/lower B, NOT k_chunk | DONE: default 100 + Ny-aware OOM warning |
| 3 | **Single multi-node job + compile cache** | ONE job, srun 1 worker/node (SLURM_PROCID), each shards its slice over the node's 4 GPUs; NO --array (Perlmutter touchy, only 2 queued jobs get priority); resumable + persistent `jax_compilation_cache_dir` | ~K× across K nodes | harness built (`scan/`); cache validating |
| 4 | **RHS CSE** (bit-identical) | get_derivatives recomputes aH/tau/tau_c; precompute once IF XLA isn't already eliding | ~1.0-1.3× per-param, ZERO accuracy cost | to probe (HLO/timing) |
| 5 | compile hygiene | donate output buffers, prealloc env, pad B | marginal + OOM relief | optional |

## DROPPED (with reason)
- **TCA** — user: regime switching expensive. (Both agents' top ceiling lever.)
- **Tolerance loosening** — 3e-4/3e-6 → EE 0.39% (>permille); 1e-3/1e-5 → TT 1.03% (gate fail). Measured, `bench/tol_bracket_results.json`.
- **float32 solve / fp32 storage** — no TCA ⇒ the bare stiff Thomson term can't hold fp32; fp32 storage propagates fp32 through make_output_table's fp64 metric sums. Accuracy trade.
- **Kvaerno3 / explicit/IMEX on a "smooth band"** — stiffness is k-independent (1/tau_c Thomson), so all modes stiff; lower order = accuracy trade.
- **Stiffness-homogeneous k-chunking** — accuracy-neutral but only ~1.1-1.35× (single stiffest mode at k_max is the irreducible per-chunk floor) AND complicates the adaptive-k_chunk memory story. Keep in reserve.
- **N_k reduction** — the perturbation k-axis is the cubic-spline source for the transfer function; spline accuracy IS the gate.

## Measurements (`bench/round2_sweep.jsonl`)
- **GPUs are A100-SXM4-80GB** (not 40GB) — 2× the headroom assumed.
- 4-GPU per-param vs B: B=64 → 1.09 (5.3 GB/dev); B=256 → 0.53 (21.1 GB/dev);
  **B=512 → 0.434 s/param (42.2 GB/dev, kchunk=48)**. Still falling, flattening.
  Throughput ≈ 0.9 → 1.9 → 2.3 cosmo/s/node (4 GPUs).
- **Memory model (CORRECTED): peak/device ≈ 0.33 GB × B_local, INDEPENDENT of
  k_chunk** (B_local = per-device cosmologies = B/n_dev sharded). The peak is the
  persistent saved-trajectory tensor (∝ B_local), NOT the transient solver
  workspace — so 512@kchunk48 (6144 lanes) and 256@kchunk100 (6400 lanes) sit on
  the SAME 0.33×B_local line despite differing lanes. On 80 GB (~66 GB usable),
  **B_local ≤ ~200 → B ≤ ~800 on 4 GPUs**.
- **Implication:** adaptive-k_chunk-for-memory is the WRONG lever (k_chunk doesn't
  move the peak). The memory knob is B_local: shard more / cap B per call. k_chunk
  is a throughput knob only (default 100; sensitivity test pending).
- tol bracket `bench/tol_bracket_results.json`: 3e-4→EE 0.39% (>permille);
  1e-3→TT 1.03% (fail). Tol-loosening rejected (user: stay near permille).
- compile cache: works for plain jit; ABCMB call_batched writes 0 entries
  (HyRex custom_vjp path). Best-effort only; one-time ~150s/job compile parallel
  across nodes is negligible vs hours of compute.

## Open item: snapshot fixtures stale on 80 GB A100
pytests/test_snapshots.py is RED on these 80 GB A100 nodes: ClTT max_rel ~4.4e-7
vs the rtol=1e-8 fixture tolerance (max_abs ~5e-17) — pure XLA-codegen drift vs
whatever environment generated the committed fixtures, NOT a regression (model()
matches CLASS at TT 0.197% here; this round only touched call_batched, not the
model() path). 4.4e-7 is ~250× below the rtol_large_k_PE=1e-4 solver tolerance.
RECOMMENDATION (user's call — it changes the test contract): either loosen the
snapshot rtol to ~1e-5 (still 10× under the solver floor; robust across GPUs/XLA)
or regenerate snapshots on the canonical node. Left unchanged in this commit.
- tol bracket: `bench/tol_bracket_results.json` (documents why tol-loosening is rejected).
- compile cache hit: `bench/cache_probe.log`.

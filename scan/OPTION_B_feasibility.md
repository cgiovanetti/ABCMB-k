# Option B feasibility: optimizer-based frequentist profile (forward-AD)

**Date:** 2026-06-09 · **Branch:** `perk-perf` · **1 node = 4× A100-80GB**

**Question.** Option B = profile likelihood by, at each value of the parameter of
interest (POI), *minimizing* χ² over the cosmological nuisances with a gradient
optimizer (instead of a dense grid). It needs ∂χ²/∂θ. Reverse-mode AD stores the
whole perturbation trajectory → OOM, so the question was whether **forward-mode**
AD works and is cheap enough. Measured with `scan/fwdad_validate.py` +
`scan/fwdad_scaling.py` (l_max=2508, lensed, 1 massive ν).

## Measured: forward-AD through ABCMB → plik-lite χ²

| quantity | value | note |
|---|---|---|
| `jacfwd(chi2)` traces end-to-end? | **YES** | add_derived_parameters + HyRex(CPU) + perturbations(GPU) + spectrum + χ², gradient finite |
| primal χ² | 610.18 | matches the standalone validation exactly |
| **peak memory, primal = jacfwd** | **1.20 GB → 1.20 GB** | **FLAT — forward-mode adds zero memory** |
| AD vs central finite-diff | 1e-4–5e-3 (5/6 params) | ω_b 6% is FD-reference-limited; AD is the exact value |
| **single jvp** (one direction) | **2.1× primal** | healthy forward-sensitivity cost |
| **`jacfwd`, all 6 params** | **~4× primal** | tangents **parallelized** (4× ≪ 6×2.1=12.8× serial) |

**Headline 1 — memory is FLAT** (1.20 GB → 1.20 GB). The reverse-mode blocker is
gone; ABCMB defaults to `adjoint=diffrax.ForwardMode` for exactly this reason.
Option B is **not** memory-limited.

**Headline 2 — the gradient is cheap (~4× a forward eval), not 33×.** A single
directional derivative (jvp) is 2.1× the primal, and `jax.jacfwd` propagates all
six tangents *together* for only ~4× total (marginal ~0.4× per extra direction).
The "~33×" in the first pass was a **measurement artifact**: it divided a
`jax.jit`-wrapped gradient (331 s) by an *un-wrapped* primal (10 s). On a
consistent baseline the ratio is ~4×.

**Implementation pitfall found:** do **not** wrap the cross-device χ²
(GPU→CPU-HyRex→GPU) in an outer `jax.jit` — XLA handles the embedded CPU-backend
stage poorly and everything runs ~8× slower (primal 10 s → 82 s). Use the
internal `filter_jit` path (call `jacfwd(chi2)(θ)` directly), or build the batched
gradient through `call_batched`'s existing batched methods (the 0.66 s/param path).

## Cost model (with the corrected ~4× gradient factor)

Per 1-D profile: **N_POI ≈ 40** values; at each, BFGS over **~5 nuisances**,
~**40 iterations**, each iteration = 1 gradient (~4× a forward) + ~2 line-search
forward evals = ~6 primal-equiv. Unit = one forward eval at 0.66 s (B≈256/node,
the measured batched throughput; forward-AD is memory-flat so the gradient runs at
the same batch).

| | primal-equiv / 1-D profile | node-hr / profile | **6 ΛCDM profiles** |
|---|---|---|---|
| **forward-AD** (~4×/grad) | 40·40·6 ≈ 9.6×10³ | ~1.8 | **~11 node-hr** |
| finite differences (~10×/grad) | 40·40·12 ≈ 1.9×10⁴ | ~3.5 | ~21 node-hr |
| **Option A, dense 6-D grid (15/dim)** | 1.1×10⁷ pts | — | **~2090 node-hr** |

So the optimizer route is **~180× cheaper** than the dense grid, and with the
corrected factor **forward-AD now beats finite differences ~2×** — use forward-AD
as the workhorse (exact, memory-flat), FD only as a no-AD fallback.

**Wall-clock:** vmap the optimizer across the N_POI points → each iteration is one
batched gradient (B = N_POI), sharded over 4 GPUs. A 1-D profile is **well under an
hour on 1 node**; the full ΛCDM set is **~1–2 h on 4–8 nodes** (debug/regular,
never premium).

## Verdict & next steps

**Option B is feasible and is the recommended approach.** Forward-AD is
memory-flat (the key result) and cheap (~4×/gradient), so the full ΛCDM profile
set is **~10–20 node-hours** (a couple hours wall on a few nodes) vs ~2000 for the
dense grid. Build order:
1. **Reparametrize h → 100·θ\*** (from the grid-scan finding — orthogonalizes the
   degeneracy the optimizer would otherwise crawl along).
2. **Wire the batched gradient through `call_batched`** (not `vmap` of the single
   path, and not an outer `jax.jit` of the cross-device χ²). Confirm the ~4×
   factor holds at production batch (B≈256) and measure the true per-POI gradient
   wall.
3. Optimizer: BFGS/L-BFGS over the nuisances, vmapped across the POI grid;
   envelope theorem for the prior-profiled A_planck; forward-AD **Hessian** at the
   minimum (forward-over-forward, ~N² jvps, N≈5 → cheap) for the Fisher errors.

Tools: `scan/fwdad_validate.py` (correctness + memory), `scan/fwdad_scaling.py`
(serial-vs-parallel + per-direction costs).

# Plan: Performance-first validation of the per-k / batched-params refactor

## Context

The refactor under consideration flips ABCMB's perturbation-evolution iteration order from "for one params, vmap over k" to "for one k, vmap over a batch of params." The motivating hypotheses are (a) **step-count tax**: diffrax's adaptive solver currently pays the worst-case step count across the k-axis batch on every vmap call, and varying params at fixed k should yield a much tighter step-count distribution; and (b) **throughput at large B**: B params × K k modes batches into wider GEMM-friendly shapes when B is large.

Both hypotheses are unproven and either one being wrong invalidates the design. The existing plan front-loaded snapshot fixtures, parity helpers, and HyRex/LINX vmap work — all of which is wasted effort if the performance gain doesn't materialize. **This plan moves the benchmark to the front and gates the rest of the refactor on its result.**

Auxiliary scope (snapshot fixtures, HyRex/LINX under vmap, batched Output API, frequentist likelihood layer) is explicitly deferred until after the benchmark decision.

## Shape of the change

```
  investigation  ──────────────────►   decision   ──────►   real refactor   ──────►   integration   ──────►   layer
                                                                                                              on top
  ┌──Phase A──┐   ┌──Phase B────┐    ┌──gate──┐    ┌──Phase C──┐ ┌──Phase D──┐ ┌──Phase E──┐ ┌──Phase F──┐  ┌──Phase G──┐
  │ profile   │   │ flipped     │    │ pass?  │    │ snapshots │ │ perturb.  │ │ spectrum  │ │ HyRex+LINX│  │ Output API│
  │ current   │ ► │ spike,      │ ►  │  →     │ ►  │ + parity  │►│  refactor │►│  refactor │►│ under     │► │ + batched │
  │ code, get │   │ measure     │    │ fail?  │    │ helpers   │ │ (real,    │ │ (real,    │ │ vmap      │  │ Model     │
  │ baseline  │   │ both signal │    │        │    │ (kept     │ │ batched,  │ │ batched,  │ │ (sequen-  │  │           │
  │ numbers   │   │ at B≤64     │    │  ↓     │    │ minimal)  │ │ B-axis,   │ │ B-axis)   │ │ tial→vmap)│  │           │
  └───────────┘   └─────────────┘    │ abandon│    └───────────┘ │ correct)  │ └───────────┘ └───────────┘  └───────────┘
                                     │ design │                  └───────────┘
                                     └────────┘                                                       Phase H: likelihood (stretch)
```

Order rationale (matches user feedback): investigation first, then the **core refactor (D + E)** before the auxiliary HyRex/LINX vmap work (F) and the API layer (G). HyRex/LINX are CPU and cheap; until the batched perturbations + spectrum work is correct, batching them gains us nothing.

Iteration-order flip:

```
current (perturbations.py:109-112)              proposed
───────────────────────────────────             ─────────────────────────────────
for params_i in batch:                          for k in k_axis:                  ← outer = lax.scan
    vmap(evolution_one_k,                          vmap(evolution_one_k,           ← inner = vmap over B
         in_axes=(0, None, None))                       in_axes=(None, None, 0))
        (k_axis, lna, (BG_i, params_i))               (k, lna, (BG_batch, params_batch))
```

## Phase A — Profile and baseline (current code, no edits)

Goal: numbers to compare Phase B against, and confirm perturbation evolution is in fact the bottleneck.

Build `bench/baseline.py` (new file, not in `abcmb/`):

1. Set up `Model` with default specs matching `pytests/accuracy_test.py` (lensing on, output_Pk on, l_max=2500). Use the fiducial param dict from there.
2. Generate `B_params` — a list of 64 param dicts perturbed around fiducial. Vary the parameters that actually drive perturbation step counts: `h`, `omega_cdm`, `omega_b`, `A_s`, `n_s`. Sample from a wide-but-physical box (≥1σ Planck), not just ε wiggles — step-count effects come from real cosmological variation, not noise.
3. Warm JIT: one untimed run.
4. Measure (using `time.perf_counter` and `jax.block_until_ready` on outputs):
   - **Wall-clock per params** for `model(params_i)` looped over B=1, 4, 16, 64.
   - **Wall-clock for `PE.full_evolution((BG, params))` only** at the same B values — strips HyRex/LINX/spectrum overhead so we're measuring the part we're proposing to change.
5. Step-count distribution: locally patch `evolution_one_k` to return `sol` instead of `sol.ys`, extract `sol.stats["num_steps"]`. Record `num_steps[k_i]` for one fiducial params dict across all `k_axis_perturbations`. Plot/save as histogram + per-k line.
6. Optional but cheap: a 30-second GPU profile via `jax.profiler.trace` on one full `model(params)` call. Confirms perturbations dominates and isn't dwarfed by bessel/LOS.

Done test: `bench/baseline.py` runs end-to-end, prints a table of (B, wall_per_params, wall_PE_only), and writes `bench/baseline_stepcounts.npz` containing per-k step counts. The PE-only time at B=64 sets the bar Phase B has to beat.

## Phase B — Flipped-order spike (the bet)

Goal: cheapest possible implementation of the new iteration order that produces honest wall-clock and step-count numbers. **No correctness contract** — outputs need only be finite and shaped right; parity testing is deferred until after the decision.

Build `bench/flipped_spike.py` (new file). Key insight that keeps this small: we don't need batched HyRex/LINX/BG-construction code — we can build the B `Background` objects **sequentially** in setup using the existing `Model.__call__`-up-to-`get_BG` path, then stack them with `jax.tree.map(lambda *xs: jnp.stack(xs), *BGs)` into a single batched pytree. That stacking is one-time setup, not part of the measurement.

Concrete steps:

1. Reuse the 64 perturbed param dicts from Phase A.
2. Setup (untimed): for each params_i, run `add_derived_parameters`, `get_BG_pre_recomb`, HyRex, and `get_BG` to produce 64 `Background` objects. Stack into `BG_batch` and `params_batch` (leading axis B). This may need `eqx.tree_at` or a manual stack since `Background` is an `eqx.Module` — write the stack helper inline; don't generalize it.
3. Define `evolution_one_k_batched(k, lna, BG_batch, params_batch)` = `vmap(PE.evolution_one_k, in_axes=(None, None, 0))(k, lna, (BG_batch, params_batch))`.
4. Define `full_evolution_flipped(BG_batch, params_batch)` that does `lax.scan` over `k_axis_perturbations`, calling `evolution_one_k_batched` on each step, returning a stacked array shape `(N_k, B, N_lna, N_y)`. Wrap in `eqx.filter_jit`.
5. Warm JIT, then time `full_evolution_flipped` at B ∈ {1, 4, 16, 64}.
6. For step counts: same `sol.stats["num_steps"]` extraction as Phase A, but now shape `(N_k, B)`. Save and compare distributions per-k.

Things that will probably trip and how to handle them in the spike (not generally — keep it local):

- **`Background` not stackable as-is**: it contains `interpax.CubicSpline` or method-bearing leaves. If `jax.tree.map(jnp.stack, ...)` fails on a leaf, just stack the underlying arrays manually for the fields that `evolution_one_k` actually reads (`aH`, `tau`, `tau_c`, etc. — see `perturbations.py:148-153, 224, 360-363`). Verify by inspection of `evolution_one_k`'s usage.
- **`specs` dict carrying non-array leaves**: not batched, no issue — it's static.
- **`adjoint` field**: not batched, no issue.
- **`vmap` over `params` dict**: each value is a jnp array of shape `()`; stacking gives shape `(B,)`. Confirmed traceable from reading `add_derived_parameters` — no Python branching on numeric values.

Done test: `bench/flipped_spike.py` runs end-to-end at B ∈ {1, 4, 16, 64} on GPU and prints (B, wall_PE_flipped, wall_PE_baseline_from_phase_A, ratio). Writes `bench/flipped_stepcounts.npz`. Outputs being finite (no NaNs) is sanity, not correctness.

## Decision rule (gate to the rest of the refactor)

Read both `.npz` files and decide based on **two** signals, both required:

1. **Wall-clock**: at B=64, `wall_PE_flipped / 64` < `wall_PE_baseline_at_B=1` by a meaningful margin (concrete target: ≥3×). At B=1, regression ≤2× is acceptable.
2. **Step-count distribution**: at fixed k, the spread of `num_steps` across the B-axis in flipped path is materially tighter than the spread across the k-axis in baseline. Quantitatively: ratio of (max/median) of per-batch-element step counts in the new path should be substantially smaller than the same ratio across k in the baseline.

If both pass → proceed with Phases C–H below, in the order given. If either fails → stop and re-discuss design before any further code changes.

## Phase C — Snapshot fixtures + parity helpers (post-decision prerequisite)

Smallest possible version of the original Phase 0. Now justified because the design has been validated, and everything after C overwrites the existing path so we need a parity oracle.

- `pytests/fixtures/generate_snapshots.py` — run current (pre-refactor, frozen at the Phase B commit) `Model` on 4–6 param dicts: vanilla ΛCDM, ΛCDM + massive ν, BBN table, BBN LINX, both reion parameterizations. Save `ClTT/ClTE/ClEE/Pk` plus a handful of `(k, lna)` transfer-function samples to `pytests/fixtures/snapshots.npz`. Commit the `.npz` on the refactor branch.
- `pytests/parity.py` — single helper `assert_batch_matches_loop(fn, params_list, tol)` that runs `fn` on a B-stacked batch and on a Python loop, asserts elementwise closeness.
- `pytests/snapshots.py` — single helper `assert_matches_snapshot(output, name, tol)`.

Done test: `pytest -k snapshot` reproduces every snapshot to ~1e-12 against frozen code. Existing `pytests/accuracy_test.py` still passes.

## Phase D — Per-k perturbation evolution with batched params (the real refactor)

The core edit. Touches `abcmb/perturbations.py` and the parts of `abcmb/background.py::Background` (and `BackgroundPreRecomb`) that `evolution_one_k` reads — these have to become valid batched pytrees with a leading B axis.

- Replace the iteration-order branch in `full_evolution` (`perturbations.py:109-112`) with the flipped form: `lax.scan` over `k_axis_perturbations`, inner `vmap(evolution_one_k, in_axes=(None, None, 0))` over the batched `(BG, params)` tuple. Drop the `if jax.default_backend() == 'gpu'` split — flipped form is the only path.
- Promote `Background` and `BackgroundPreRecomb` fields to carry a leading B axis. Methods like `aH`, `tau`, `tau_c` should keep their existing signatures; under vmap they "just work" if their internals are pure JAX. Read each method called by `evolution_one_k`, `get_derivatives`, and `initial_conditions_one_k` (see `perturbations.py:148-153, 224, 360-363, 196`) and confirm — fix any branch-on-numeric-value with `lax.cond` or `jnp.where`.
- `add_derived_parameters` must produce batched dicts. Python branching on key-presence is safe (static across batch); numeric branches inside (e.g., `input_N`, `input_Neff`) check existence not values, so they stay static. Verified by re-reading `main.py:368-612`.
- For this phase, HyRex/LINX still run sequentially (Python loop over the B param dicts, then stack outputs). This is slow but correct, and makes Phase D's parity test a clean isolation of the perturbations change.
- `PerturbationTable` gains a leading B axis on every field. The vmap'd `make_output_table` (`perturbations.py:318-404`) already broadcasts cleanly per-k; just lift the outer vmap.

Done test: `assert_batch_matches_loop` on transfer functions at sample `(k, lna)` across the snapshot params set, tolerance ~1e-9 (a touch of XLA-fusion slack on the scan).

## Phase E — Per-k spectrum integration

Touches `abcmb/spectrum.py::SpectrumSolver`. `get_Cl` already sums over k internally; refactor so each k contribution is computed across the B-axis at once and accumulated into `Cls[B, l]`. `Pk_lin` straight-vmaps over the B axis at fixed `k_axis_Pk_output`.

Done test: `assert_batch_matches_loop` on `ClTT/ClTE/ClEE/Pk`, tolerance ~1e-8 (Bessel interpolation + k-grid sums accumulate float noise).

## Phase F — HyRex + LINX under vmap (CPU)

Now that the GPU-side hot path is batched and correct, replace the sequential Python loop introduced in Phase D with proper vmap'd HyRex and LINX calls on CPU.

- Phase F.1 — HyRex: 20-line spike first — does `vmap(filter_jit(RecModel, backend='cpu'))` on a batch of two `(recomb_inputs, params)` tuples clean-compile? The known risk is `array_with_padding` (variable-length arrays). If clean → wire in. If not → diagnose; fallback is to keep the Python loop (HyRex is cheap; this isn't catastrophic at small B).
- Phase F.2 — LINX: same shape spike on a batch of `Delta_Neff_init` / `omega_b`.
- The `_to_float` cast workaround in `main.py:207-212, 233` has to extend to batched leaves — confirm.

Done test: `assert_batch_matches_loop` on `(xe, lna_xe, Tm, lna_Tm)` (HyRex) and `(Neff, T_nu_massless, YHe)` (LINX), tolerance ~1e-10.

## Phase G — Batched Model & Output API

Touches `main.py::Model.__call__`, `main.py::Output`, and `main.py::run_cosmology_abbr`.

- Input contract: `params` dict where every value is an array with leading dim B. A `Model.from_param_list([dict, dict, ...])` helper stacks for ergonomics.
- Output contract: `ClTT[B, l]`, `ClTE[B, l]`, `ClEE[B, l]`, `Pk[B, k]`, batched `BG`, batched `PT`. Document in docstrings.
- B=1 is a valid input (singleton leading axis). No special-casing.

Done test: snapshot tests reproduce when each params is passed as B=1. `pytests/accuracy_test.py` passes through batched `Model` with params wrapped to B=1 — same <1% threshold against `classy`. Loop-vs-batch parity holds for B ∈ {1, 2, 4, 8} on the snapshot set.

## Phase H — Frequentist likelihood layer (stretch)

Thin wrapper: `chi2_grid(params_grid, data, cov)`. Example notebook recovering a fiducial point from a grid. Done test: χ² minimum index recovers the fiducial input within grid resolution.

## Files touched (cumulative across phases)

- **A/B (investigation, no `abcmb/` edits)**: `bench/baseline.py`, `bench/flipped_spike.py`, `bench/*.npz`.
- **C**: `pytests/fixtures/generate_snapshots.py`, `pytests/fixtures/snapshots.npz`, `pytests/parity.py`, `pytests/snapshots.py`.
- **D**: `abcmb/perturbations.py` (most of it — full_evolution, full_evolution branch removed, PerturbationTable batched), `abcmb/background.py::Background` + `BackgroundPreRecomb`.
- **E**: `abcmb/spectrum.py::SpectrumSolver` (get_Cl, Pk_lin).
- **F**: `main.py` call sites for HyRex/LINX; `abcmb/hyrex/*` and `abcmb/linx/*` only if the vmap spikes show issues.
- **G**: `abcmb/main.py::Model.__call__`, `run_cosmology_abbr`, `Output`.
- **H**: new `abcmb/likelihood.py` or similar; `example_notebooks/`.

## Verification

- A: `python bench/baseline.py` on GPU produces baseline table + `baseline_stepcounts.npz`. Cross-check B=1 wall-clock against `time_tests.py` numbers.
- B: `python bench/flipped_spike.py` produces flipped table + `flipped_stepcounts.npz`. Outputs finite, no NaNs.
- Decision: short script loads both `.npz`, plots step-count distributions side-by-side, prints the decision rule's verdict.
- C–G: per-phase parity tests via `assert_batch_matches_loop` and `assert_matches_snapshot`.
- G: full `pytests/accuracy_test.py` (CLASS comparison, <1%) is the end-to-end contract.
- H: notebook recovers fiducial point.
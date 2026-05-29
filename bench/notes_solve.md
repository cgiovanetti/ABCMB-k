# notes_solve.md — Squeezing the batched perturbation solve (and the CPU setup)

Perf-engineering analysis of the `perk-refactor` batched pipeline in ABCMB-k.
Code-level, with real line numbers. Goal: drive `Model.call_batched(params_list)`
toward ~1 s/param on one A100 by attacking the dominant perturbation solve and
the un-batched per-cosmology CPU setup that will become the next floor.

**No code was run for this note** (login-node Python forbidden, GPU in use
elsewhere). Everything below is read from source + the verified `bench/`
artifacts. Where a number comes from an artifact it is cited; where it is an
estimate it is labelled ESTIMATE.

---

## 0. Verified baseline (re-read from artifacts, not trusted blindly)

From `bench/baseline_run.log` + `bench/baseline_summary.txt` (Phase A, single GPU,
post-compile). The log only times two stages directly:

| stage                          | per-params | share |
|--------------------------------|-----------:|------:|
| `model()` end-to-end           | 9.888 s    | 100%  |
| `PE.full_evolution` (PE-only)  | 8.151 s    | 82.4% |
| **everything else** (residual) | ~1.74 s    | 17.6% |

The log does NOT separately time `get_Cl` / `Pk_lin`. The ~1.74 s residual is
`get_BG_pre_recomb` (GPU) + HyRex CPU recomb + `Background` build + 2 device
transfers (the "setup" stage, §E) + `get_Cl` + `Pk_lin`. The Phase E
perf_batched asymptote (~12 s/params, dominated by the per-element spectrum loop)
is the evidence that `get_Cl` is the second-largest single component once
batched; in the single-call path it is folded into that 1.74 s residual. (Any
finer per-stage split below — e.g. "get_Cl ≈ 0.68 s" — is an estimate, not in the
logs.)

Per-k diffrax step counts, `N_k = 571` modes: **min=41, median=380, mean=447.3,
max=1579, max/median = 4.16.** That 4.16 is the worst-case-k tax.

From `bench/flipped_summary.txt` (Phase B flipped-order spike, single GPU, at the
single worst-case k, varying B):

| B  | per-params |
|---:|-----------:|
| 1  | 8.21 s |
| 4  | 5.12 s |
| 16 | 3.91 s |
| 64 | 3.43 s |

Flipped worst-k **max/median over B = 1.12** (the k-tax is essentially gone at
fixed k). Memory 28–31 GB at B=64 on a 40 GB A100 (near OOM).
4-GPU sharded (CHANGELOG Phase B): 0.93 s/params at B=64 (8.80x).

From `bench/perf_batched.py` numbers in CHANGELOG (ELLMAX=800, single A100):
single 9.51, batched B=1 22.64, B=4 14.46, B=8 12.95, B=16 12.08 s/params.
**Batched is currently SLOWER than single**, asymptoting at ~12 s because of the
Python-loop spectrum (Phase E). The modes side is faster (3.43 vs 8.14); the win
is hidden by the spectrum loop.

**Takeaway up front:** the single highest-value change is NOT in the solver knobs
— it is unblocking the spectrum loop (§E.0 / Phase E), because today the batched
path is net-slower than sequential and no solver tuning matters until that is
fixed. After that, the dominant *solver* lever is the small-k/large-k vmap split
(§A). Ranked list at the end.

---

## State-vector size (used throughout for memory math)

Computed from the species `num_equations` (species.py, verified by grep) with the
default cutoffs in `model_specs.load_specs` lines 31–34 (l_max_g=12, l_max_pol_g=10,
l_max_massless_nu=17). The ΛCDM `populate_species` set is
(DarkEnergy, ColdDarkMatter, Baryon, Photon, MasslessNeutrino):

```
Ny = 1 (metric eta, perturbations.py:329)
   + DarkEnergy.num_equations        = 0   (species.py:342, BackgroundFluid)
   + ColdDarkMatter.num_equations    = 1   (species.py:450)
   + Baryon.num_equations            = 2   (species.py:1052)
   + Photon: (l_max_g+1)+(l_max_pol_g+1) = 13+11 = 24  (species.py:1262-1264)
   + MasslessNeutrino: l_max_massless_nu+1 = 18         (species.py:571)
   = 46 equations (default ΛCDM)
```

NOTE: `design_memo.md §1.1` uses **Ny = 72** for its 10.52 GB-at-B=64 memory math.
That is the figure `K_CHUNK=100`/5.5 GB was sized against, so for memory accounting
I quote design_memo's numbers as-is; the discrepancy (72 vs the 46 I count for pure
ΛCDM) likely reflects a fuller default hierarchy in the spike config or how diffrax
pads internal stage storage. NOT load-bearing for any conclusion (all levers scale
linearly in Ny); but if someone re-derives memory, recount Ny from the actual
species/specs of the run rather than trusting 72.

Saved trajectory per `(k,B)` lane = `Nlna(500) × Ny × 8 B` = 184 KB (Ny=46) to
288 KB (Ny=72) (`_compute_modes_batched`, `lna = jnp.linspace(lts, 0., 500)` at
perturbations.py:179–181). design_memo §1.1: `571 × 64 × 500 × 72 × 8 B = 10.52 GB`
saved-ys at B=64. The implicit Kvaerno5 Jacobian workspace (`Ny²` floats/lane ×
stages, design_memo §1.2 ≈ 4.5 GB at B=64) dominates *transient* memory and is why
`K_CHUNK=100` was chosen (~5.5 GB at B=64 once chunked).

---

## The solver call (the thing every lever touches)

`PerturbationEvolver.evolution_one_k` (perturbations.py:382–448) is the leaf that
is double-vmapped. The diffrax config (lines 417–444):

```python
417  term = diffrax.ODETerm(self.get_derivatives)
418  solver = diffrax.Kvaerno5()                       # implicit, stiff-capable
420  rtol = jnp.where(k > self.specs["k_split_PE"],     # 0.01
424                   self.specs["rtol_large_k_PE"],    # 1e-4
                     self.specs["rtol_small_k_PE"])     # 1e-5
426  atol = jnp.where(k > self.specs["k_split_PE"],
                     self.specs["atol_large_k_PE"],     # 1e-6
                     self.specs["atol_small_k_PE"])     # 1e-10
432  stepsize_controller = diffrax.PIDController(
         pcoeff=0.25, icoeff=0.8, dcoeff=0.0,          # specs defaults
         rtol=rtol, atol=atol)
433  saveat = diffrax.SaveAt(ts=lna)                    # 500 points, dense=False
434  adjoint = self.adjoint()                           # ForwardMode (no bwd)
436  sol = diffrax.diffeqsolve(
         term, solver, t0=lna_start, t1=0.0, dt0=1e-2, y0=y_ini,
         stepsize_controller=stepsize_controller,
         max_steps=self.specs["max_steps_PE"],         # 2048
         saveat=saveat, args=(k,*args), adjoint=adjoint)
```

Crucial facts for everything below:
* It's **adaptive** (`PIDController`), **implicit** (`Kvaerno5`), saving 500
  fixed `ts`, `dense=False`, `ForwardMode` adjoint (no backward → memory budget is
  forward-only).
* `rtol`/`atol` are chosen via `jnp.where(k > k_split_PE, ...)` — i.e. **per-k as
  data**, NOT as a Python/static branch. Under `vmap` this becomes a per-lane
  *value*, but the controller still steps all lanes in lockstep (see §A).
* `lna_start = jnp.minimum(get_starting_time(k,args), -10.)` (line 406–411).
  `get_starting_time` does two 10000-point `jnp.interp` inversions (lines
  277–289) **per call, per lane** — that is real work multiplied by `K_CHUNK×B`.

---

## A. Adaptive-ODE lockstep stepping under vmap — THE central question

### What the controller actually does under vmap (confirmed from the config)

`diffrax.diffeqsolve` runs a single `while_loop` whose carry includes the scalar
step size `dt` and a boolean "accept". Under `vmap`, every leaf of that carry —
including `dt` and the accept flag — gains a batch axis, BUT the loop's
**predicate is a single scalar** (`jnp.any(still_running)`), and diffrax's
`PIDController` reduces the per-lane error estimate so that **a step is taken iff
it is acceptable; lanes that would have taken a larger step are forced to advance
at the smallest accepted `dt` across the batch.** This is exactly the "lockstep"
behaviour `chunking_debug_report.md` lines 24–31 documents:

> "Within a chunk, all lanes step together at the smallest step any lane needs
> (this is the 'lockstep tax' the flipped order reduces but doesn't eliminate)."

So: **CONFIRMED, lockstep stepping wastes work.** The total number of *integrator
steps executed* for a chunk = number of steps the *stiffest lane in that chunk*
needs. Every other lane is integrated with that same (smaller-than-it-needs) step
sequence — redundant function evaluations of `get_derivatives` (which loops over
all 5 species, perturbations.py:333–380) and redundant implicit Jacobian solves.

### How much headroom remains after the flip

The flip (vmap over B at fixed k) already collapsed the cross-*k* spread:
`max/median over B = 1.12` (`flipped_summary.txt`). At fixed k, the only spread
left is from cosmology variation (Planck ±2-3σ), which is tiny — so **within a
single-k vmap-over-B, lockstep waste is ~12% at most.** That part is nearly
solved.

The remaining waste is **within a k-chunk**: `_evolve_chunk` (perturbations.py:
126–148) vmaps over a *contiguous block of up to 100 k-modes* AND over B
simultaneously:

```python
142  def per_k(k, lna_b, BG_b, p_b):
143      return vmap(self.evolution_one_k, in_axes=(None, 0, (0, 0)))(k, lna_b, (BG_b, p_b))
147  return vmap(per_k, in_axes=(0, None, None, None))(k_chunk, lna_batch, BG_batch, params_batch)
```

This is `vmap(k_chunk) × vmap(B)` — a *single fused 2-D vmap* of size
`K_CHUNK × B` over one `diffeqsolve`. **All `K_CHUNK×B` lanes share one lockstep
step controller.** A chunk of `k_axis[i:i+100]` spans up to ~1 order of magnitude
in k → its step distribution is still wide (the chunk's runtime = its largest-k
lane's step count). `chunking_debug_report.md` lines 41–45 states this explicitly:

> "The lockstep stepping means a chunk's runtime is set by its worst (k, B) lane.
> With B fixed and k varying within a chunk, the worst lane is the largest k in
> the chunk."

So the flip removed the *cross-chunk* tax but **re-introduced an intra-chunk
k-tax bounded by the k-range inside each chunk.** With 6 chunks over `N_k=571`
(5×100 + 1×71, log-spaced from ~7e-6 to ~1.8) each chunk still spans a wide k
range. **This is the biggest remaining solver inefficiency.**

### Mitigation A1 (HIGHEST-VALUE solver change): k-homogeneous chunking

Right now chunks are *contiguous slices of the existing k-axis*
(`k_axis[i:i+k_chunk_size]`, perturbations.py:187). Because `get_k_axis_perturbations`
(model_specs.py:120–175) is monotonically increasing, each contiguous chunk is a
*band* of k. The stiffness (≈ step count) correlates strongly and monotonically
with k (small k = superhorizon = smooth = ~41 steps; large k = ~1579 steps per
baseline). So a contiguous chunk already groups *similar-stiffness* modes — which
is the right idea — BUT the chunk size is uniform (100), so the **high-k chunks
(stiff) and low-k chunks (smooth) get the same K_CHUNK and the same rtol envelope
via `jnp.where`, yet the stiff chunks dominate wall-clock.**

Two concrete improvements:

1. **Non-uniform chunk sizes by stiffness.** Make low-k (cheap) chunks *large*
   (e.g. 200) and high-k (expensive, near max=1579 steps) chunks *small* (e.g.
   32–50). The compile cost is one kernel per distinct chunk shape, so use only
   2–3 distinct sizes. Code sketch (replace the loop at perturbations.py:186–189):

   ```python
   # bands: (k_lo, k_hi, chunk_size). Boundaries from get_k_axis structure.
   # small-k smooth band -> big chunks; large-k stiff band -> small chunks.
   def _chunk_plan(k_axis, specs):
       # cheap heuristic: split at k_split_PE and at ~10*k_split_PE
       lo  = k_axis < specs["k_split_PE"]              # smooth, ~few steps
       mid = (k_axis >= specs["k_split_PE"]) & (k_axis < 0.1)
       hi  = k_axis >= 0.1                              # stiff, ~1579 steps
       return [(k_axis[lo], 200), (k_axis[mid], 100), (k_axis[hi], 40)]
   ```
   Then loop `_evolve_chunk` over each (band, size). **Expected speedup: 1.3–1.7x
   on the modes stage** — the stiff band shrinks its lockstep group so the cheap
   modes inside it stop being dragged; the smooth band amortizes more modes per
   kernel launch. ESTIMATE; bounded above by intra-band step spread.
   **Accuracy risk: NONE** — identical math, same rtol/atol per k via the existing
   `jnp.where`. Only the lane grouping changes (already proven benign in
   `chunking_debug_report.md`). **Confidence: HIGH.**

2. **Per-lane rtol/atol scaling within a chunk.** The controller already reads
   per-k rtol via `jnp.where(k > k_split_PE, ...)` (perturbations.py:420–430), so
   per-lane tolerances are *already plumbed through vmap as values*. But the split
   is binary (large vs small k). Lockstep means the *tightest* tolerance in the
   chunk sets the pace. **Loosening the high-k atol** (`atol_large_k_PE=1e-6`)
   would let stiff lanes accept bigger steps. Risk: the 1%-vs-CLASS gate. The
   CHANGELOG notes downstream Cl/Pk drift is already ~1e-5 at rtol_large=1e-4;
   there is likely 2-3x atol headroom before TT/EE move 1%. **Expected speedup:
   1.1–1.3x. Accuracy risk: MEDIUM (must run `pytests/accuracy_test.py`).
   Confidence: MEDIUM** — cheap to bracket (sweep atol_large_k_PE ∈ {1e-6, 3e-6,
   1e-5} and check max-rel Cl).

### Mitigation A2: split stiff small-scale from smooth large-scale into separate
vmap groups with *different solvers* (not just different chunk sizes)

Small-k superhorizon modes (min=41 steps) are smooth and **non-stiff** — they do
not need an implicit `Kvaerno5`. Run the `k < k_split_PE` band with an *explicit*
solver (`Dopri5` / `Tsit5`), which has far cheaper per-step cost (no Jacobian
solve). The large-k band keeps `Kvaerno5`. Because `solver = diffrax.Kvaerno5()`
is set unconditionally at perturbations.py:418, this needs a Python-level branch
*outside* the vmap (two `_evolve_chunk` variants, one per solver) — which the
band-chunking in A1 already gives you for free.

```python
# in evolution_one_k, make solver selectable by a static flag set per band:
solver = diffrax.Tsit5() if self._smooth_band else diffrax.Kvaerno5()
```
(`_smooth_band` a static eqx field, or pass two pre-built PE-like closures.)

**Expected speedup on the smooth band: 2-4x for those modes** (explicit step is
~5-10x cheaper than an implicit Newton solve; offset by needing more steps if the
mode is mildly stiff). Since smooth modes are already cheap (41 steps), the
*absolute* saving is modest, but combined with A1's larger chunks for that band
it compounds. **Accuracy risk: LOW-MEDIUM** — explicit solvers can mis-handle the
tight-coupling stiffness if applied to the wrong band; keep the boundary
conservative (`k_split_PE` or lower). Must run accuracy gate. **Confidence:
MEDIUM.**

### Mitigation A3 (high-ceiling, high-effort): fixed-step + dense output

The whole lockstep problem disappears if there is no adaptive controller. Replace
`PIDController` with a **fixed step count per band** (`diffrax.ConstantStepSize`
or a precomputed `StepTo` schedule) tuned so the stiffest mode in the band is
resolved, then *every lane does exactly the same number of steps* — no lockstep
waste because there is no adaptation, and the step count is *known at compile
time* so XLA can fully unroll/pipeline.

```python
stepsize_controller = diffrax.ConstantStepSize()
sol = diffrax.diffeqsolve(..., dt0=fixed_dt_for_band, max_steps=N_steps_band, ...)
```

The catch: you must pick `N_steps` per band ≥ what the stiffest mode needs, so
within a band you're back to paying the worst-lane cost — **but** you save the
controller's rejected-step overhead (PID rejects ~10-30% of trial steps) and you
get a static-shape loop XLA loves. Net win only realized if bands are stiffness-
homogeneous (A1). **Expected speedup: 1.2-1.5x stacked on A1. Accuracy risk:
HIGH** — fixed step is brittle across cosmologies (the stiffest mode shifts with
params); a too-coarse schedule fails the gate silently. **Confidence: LOW** for
production; worth a spike only after A1.

### A summary

The flip already won the *cross-k* battle (4.16 → 1.12). The remaining lockstep
waste is **intra-chunk** and is bounded by the k-range *inside* each contiguous
100-mode chunk. A1 (stiffness-homogeneous, non-uniform chunking) is the
highest-value, lowest-risk solver change. A2/A3 are follow-ons.

---

## B. `k_chunk_size` (default 100): serializing, recompiling, memory

`_compute_modes_batched` (perturbations.py:150–192) iterates k in a **Python
loop** over chunks (line 186–189) calling the JIT'd `_evolve_chunk`, then
`jnp.concatenate(chunks, axis=0)`.

### B1. Is chunking serializing work that could be one big vmap?

Yes, partially. The Python loop launches 6 sequential `_evolve_chunk` kernels.
Each kernel is internally `K_CHUNK×B`-parallel, but the **6 launches are
serial** and each pays JIT dispatch + a kernel-launch barrier. At small B the
device is underutilized within a chunk, so serialization hurts more.

**At B=8 the whole modes tensor is small enough for ONE big vmap.** Scaling
design_memo §1's B=64 figures down by 8×: saved-ys `10.52/8 = 1.3 GB`; transient
Jacobian `4.5/8 ≈ 0.6 GB`; XLA overhead (design_memo §1.4, ~6-8 GB at B=64, sub-
linear) maybe ~1-2 GB. **Total order ~3-4 GB at B=8 — fits comfortably on 40 GB.**
So at B≤8, set `k_chunk_size = N_k` (≥571): one kernel, no Python-loop
serialization, full k×B parallelism. **Expected speedup at B=8: 1.1–1.3x on
modes** (kills 5 of 6 launch barriers; lets XLA schedule the whole thing).
**Risk: NONE** (same code path, larger vmap). **Confidence: HIGH.**

At B=32: saved-ys `10.52/2 = 5.3 GB` + transient `~2.3 GB` + XLA overhead → order
~12-16 GB — still fits but getting tight given design_memo §1.4's large XLA
overhead. At B=64 the *measured* peak is 28-31 GB (flipped_summary.txt / design_memo
§1), which is why `K_CHUNK=100` (5.5 GB chunked) is mandatory at B=64.
**Recommendation: make `k_chunk_size` adaptive** = `min(N_k, floor(MEM_BUDGET /
per_lane_bytes / B))` rather than a hard 100.

### B2. Does a ragged final chunk recompile?

Yes — confirmed by design. `design_memo.md §5` (line 201) chose `K_CHUNK=100`
splitting 571 into 5×100 + 1×71; the **71-wide final chunk compiles a second kernel
variant** because `_evolve_chunk` is `@eqx.filter_jit` and its cache is keyed on
`k_chunk.shape` (docstring perturbations.py:131-134). So **2 compiles per run**,
amortized. Cheap fix: **pad the final chunk to K_CHUNK and slice the result**, so
all chunks share one shape → 1 compile. Saves one (potentially multi-second)
compile per fresh process. **Risk: NONE** (pad with a dummy k, discard).
**Confidence: HIGH.** Only matters for cold-start latency, not steady-state.

### B3. Does changing B recompile?

**Yes.** `_evolve_chunk` traces with `B` baked into every batched leaf's shape
(`lna_batch` is `(B, 500)`, `BG_batch`/`params_batch` leaves are `(B, ...)`).
Different B ⇒ different shapes ⇒ recompile. For a frequentist scan that fixes B
this is a one-time cost. **But if `call_batched` is ever called with a ragged
final batch** (e.g. 1000 params / B=64 → last batch of 1000-15×64=40), that last
batch recompiles. **Fix: pad the param batch to a fixed B and mask** (see §F).
**Confidence: HIGH** that this recompiles; mitigation is the §F padding.

### B4. Optimal chunk sizing — recommendation

* B ≤ 8: `k_chunk_size = N_k` (one vmap, no serialization).
* 8 < B ≤ 32: `k_chunk_size ≈ 200` (2-3 chunks), pad ragged.
* B = 64: keep ~100 but make it **stiffness-non-uniform per A1**, and pad ragged
  to kill the second compile.
* Always pad the final chunk to a uniform shape (B2).

---

## C. `saveat` density / output-table size

`evolution_one_k` saves at `lna = jnp.linspace(lts, 0., 500)` (perturbations.py:
100 for single, 179-181 for batched) with `SaveAt(ts=lna)`, `dense=False`. So
**500 lna points × Ny(46) per (k,B) lane.** `make_output_table` /
`make_output_table_batched` (perturbations.py:450-536, 224-248) consume the full
`(B, Ny, 500, N_k)` tensor.

### Is 500 more than the spectrum needs?

The spectrum LOS integral lives in `Cl_one_ell` (spectrum.py:661-842). It does
**not** re-sample PT onto a separate denser grid — it integrates directly on
`lna_axis = PT.lna[:-1]` (spectrum.py:685), i.e. the **same 500-point PT save
grid** is the line-of-sight quadrature grid (the `lax.scan` accumulator at
spectrum.py:805-828 runs over those `Nlna-1 = 499` points). The k-direction is
the part that gets re-interpolated: each PT field is cubic-splined from
`PT.k`(N_k=571) onto `k_axis_transfer` (spectrum.py:709-723). So **the 500 lna
points ARE the time-integration resolution**, not an oversampled intermediate.

design_memo.md §2.2 (lines 70-80) flags trimming Nlna to ~150-200 as "Moderate
feasibility, defer to Phase C." Since the spectrum integrates on this exact grid,
trimming directly cuts both the modes-tensor memory AND the LOS scan length —
double benefit, but also means accuracy is directly exposed. Headroom analysis:

* The perturbation trajectories are smooth in `lna` *except* across recombination
  (the visibility function is sharply peaked near `lna_decoupling`). A **uniform**
  500-point grid over `lts..0` (lts ≈ -14 to -10) puts most points where nothing
  happens. A **non-uniform save grid concentrated around recombination** could
  cut to ~300 points with equal or better accuracy.

  ```python
  # replace linspace(lts,0,500) with a grid dense near lna_decoupling:
  lna = concat_sorted(linspace(lts, lna_rec-1, 120),
                      linspace(lna_rec-1, lna_rec+1, 200),   # dense at recomb
                      linspace(lna_rec+1, 0., 120))           # ~440 -> trim to 300
  ```

* **Memory & bandwidth scale linearly in Nlna.** Cutting 500→300 cuts the saved-ys
  tensor and the `make_output_table` vmap work by 40%, directly easing the B=64
  OOM pressure (28-31 GB) and speeding the bilinear interp.

**Expected speedup: 1.1-1.2x on modes+PT memory/bandwidth; mainly an OOM-relief
lever that lets you raise B (which is the real win — see §F amortization).
Accuracy risk: MEDIUM** (must keep recomb resolution; run the gate).
**Confidence: MEDIUM.** Cheap to test: regenerate snapshots and check
`accuracy_test.py` at Nlna=300 non-uniform.

Secondary: `make_output_table` does several `vmap(s.rho_delta, ...)` reductions
over species (perturbations.py:506-523). These are O(Nlna×Nk×B×Nspecies) and run
*after* the solve. Trimming Nlna helps here too.

---

## D. float32 vs float64 in the hierarchy

`jax_enable_x64` is set at import in perturbations.py:13, background.py, main.py:26,
spectrum.py:13 — **everything is float64.** A100 FP64 is ~half FP32 throughput;
the implicit Jacobian solve in `Kvaerno5` is the FP-heavy part.

### Is mixed precision safe within 1%?

**Risk: HIGH, and structurally subtle.** Two distinct hazards:

1. **The accuracy gate.** rtol_large_k_PE=1e-4 is well above float32 eps (~1e-7),
   so the *integration* tolerance is float32-compatible in principle. But the
   *primordial-to-Cl* chain squares the transfer function and integrates over k
   (spectrum.py `integrate_Cl`), and the matter Pk squares delta_m
   (spectrum.py:191) — squaring amplifies relative error. ESTIMATE: float32 in
   the hierarchy alone likely keeps Cl within ~0.1-0.5%, *probably* under the 1%
   gate, but it is genuinely close and must be measured.

2. **`checkpointed_while_loop` / `filter_custom_vjp` integer-leaf workaround.**
   `_to_float` (main.py:245-250, 287-292) casts int/bool params to float64
   specifically because the diffrax custom-vjp path trips on integer leaves. If
   you drop to float32 you must cast to `float32`, not `float64`, *and* verify the
   HyRex `array_with_padding` (which carries int `padding_size`/`lastnum`)
   survives — those are deliberately kept int and the `_to_float` cast already
   has to coerce them. Mixed dtype across the GPU/CPU transfer boundary
   (main.py:253-263) is a footgun.

**Recommendation:** Do NOT globally drop to float32. Instead, a *targeted* mixed-
precision experiment: keep background/HyRex/optical-depth in float64 (cheap,
sensitive), run **only the perturbation hierarchy state `y` in float32** by
setting `y_ini` and the `get_derivatives` arithmetic to float32 while leaving
`lna`/`k` in float64. Diffrax respects the dtype of `y0`. **Expected speedup:
1.4-1.8x on the solve** (Jacobian solve is the bottleneck and is O(Ny²) FP).
**Accuracy risk: HIGH — this is the riskiest lever; gate it hard.** **Confidence:
LOW-MEDIUM** that it passes; HIGH that it's worth a *measured* spike because the
payoff is large and the solve is FP-bound. Cheap test: set `y0=y_ini.astype(f32)`
in `evolution_one_k`, run `accuracy_test.py`, report max-rel TT/EE/Pk.

---

## E. CPU per-cosmology setup is NOT batched — the coming floor

This is the lever the task flags as "may become the new bottleneck," and the
analysis confirms it.

### What `call_batched` does (main.py:187-238)

```python
209  for params in params_list:                       # PYTHON LOOP over B
213      full_params = self.add_derived_parameters(params)   # CPU python
214      full_p, bg = self._build_one_bg(full_params)        # HyRex on CPU, seq.
```

`add_derived_parameters` (main.py:448-692) is **pure Python + tiny jnp ops** run
*eagerly* (not jitted) — it loops over species computing `rho` at a couple of lna
values, does table interp / LINX, resolves the Neff/YHe triangle. It's cheap per
call (~ms) but it's `B` sequential Python calls.

`_build_one_bg` (main.py:240-265) is the expensive one:
```python
252  pre_BG = self.get_BG_pre_recomb(full_params)              # GPU jit, ~ small
254  recomb_inputs_cpu = jax.device_put(..., cpu_dev)          # GPU->CPU transfer
256  recomb_output = eqx.filter_jit(self.RecModel, backend='cpu')(...)  # HyRex CPU
259  recomb_output = jax.device_put(recomb_output, gpu)        # CPU->GPU transfer
264  bg = self.get_BG(...)                                     # GPU jit Background
```

### Why HyRex can't trivially vmap over B (but probably can)

HyRex (`hyrex/hyrex.py`) is NOT primarily a `while_loop` — it is built on
**diffrax `diffeqsolve`** stages. `recomb_model.get_history` (hyrex.py:172-207)
calls `helium_model(...)` then `hydrogen_model(...)`; both run `diffeqsolve` with
`Kvaerno3()`/`Tsit5()` on **fixed `SaveAt(ts=lna_axis)` grids** (helium.py:6,
hydrogen.py:366 `saveat = SaveAt(ts=lna_axis_2g)`, plus diffeqsolve calls at
hydrogen.py:371,482,754). There is exactly **one** data-dependent loop —
`eqx.internal.while_loop(..., kind="checkpointed", max_steps=...)` in
hydrogen.py:266 — which already has a *static* `max_steps` and uses
`array_with_padding` (class at array_with_padding.py) for fixed-shape output.

Important correction to a common assumption: **`recomb_model.__call__` is NOT
`@eqx.filter_jit`-decorated** (hyrex.py:146 is a plain method). The JIT is applied
*only at the call site*: `eqx.filter_jit(self.RecModel, backend='cpu')(...)`
(main.py:256, 300). So there is exactly one jit wrap, on CPU.

vmap-ability: the fixed-`SaveAt` diffeqsolve stages are trivially vmap-able over B
(adaptive-step lockstep tax applies, but recomb history varies little across
Planck-σ cosmologies → small spread). The single `checkpointed while_loop` is also
vmap-able (masked-while: runs to the worst lane's `max_steps`, which is already the
static cap). `array_with_padding` already gives fixed shapes. **So HyRex IS
vmap-able over B** — the blocker is engineering (Phase F deferred), not a hard
impossibility. The masked-while + adaptive-lockstep cost means a vmapped HyRex
won't be free, but it collapses B× Python dispatch / B× jit-cache lookups / B×
device transfers into one.

### The per-param floor it imposes

The "residual" ~1.0 s/params in the baseline (§0: 9.89 − 8.14 − 0.68 − 0.02 ≈
1.04 s) is this setup stage (pre_BG + HyRex CPU + 2 device transfers + Background
build). In the **current** `call_batched` it runs **B times sequentially**, so:

* **ESTIMATE floor: ~1.0 s × B**, *fully serial, on CPU*, with **no amortization
  at all.** At B=32 that's ~32 s of pure setup; at B=64, ~64 s. Per-params that's
  a *flat ~1.0 s/params* that does NOT decrease with B.

This is decisive: once §A/§E-spectrum bring the solve to <1 s/params, **the
un-batched setup at ~1.0 s/params becomes the entire budget** and the ~1 s/param
target is exactly the setup floor. **You cannot hit ~1 s/param without batching or
parallelizing this stage.** Quantified:

| stage (per-params)        | now (seq) | after solve+spectrum fixed | after setup batched |
|---------------------------|----------:|---------------------------:|--------------------:|
| modes (PE)                | ~3.4 (B64)| ~2 (A1)                    | ~2 |
| spectrum (vmap, §E.0)     | ~12 (loop)| ~0.7                       | ~0.7 |
| setup (HyRex etc, seq)    | ~1.0      | ~1.0 (UNCHANGED)           | ~0.1-0.3 |
| **total**                 | ~16       | ~3.7                       | ~3 → with multi-GPU ~1 |

### Mitigations

* **E-floor-1: vmap HyRex over B on CPU.** Wrap `_build_one_bg`'s recomb call in
  `vmap` over a stacked `recomb_inputs`/`params` batch. The while_loop becomes a
  masked-while (worst-lane iterations; small spread across cosmologies). **One CPU
  kernel instead of B.** Expected: setup floor ~1.0 → ~0.2-0.4 s/params (CPU
  while-loop doesn't parallelize across lanes the way a GPU does, but it
  eliminates B× Python dispatch + B× compile-cache lookups + B× device transfers).
  **Risk: LOW** (math identical; masked-while is standard). **Confidence: MEDIUM**
  on the magnitude — CPU vmap of a while_loop helps via dispatch/transfer
  elimination more than via SIMD; needs measuring.

* **E-floor-2: run HyRex on GPU, vmapped.** It currently runs CPU
  (`backend='cpu'`) "intentionally" (CLAUDE.md). But for a *batch* of B, a GPU
  vmap of the while_loop could beat CPU. Risk: the sequential solver is
  GPU-unfriendly per-lane, but B-parallel it may win. **Confidence: LOW**,
  speculative; spike only if E-floor-1 is insufficient.

* **E-floor-3: batch the device transfers.** Even keeping HyRex sequential, the
  two `jax.device_put` per element (main.py:254-260) are B× round-trips. Stack
  inputs once, transfer once. Cheap, **risk NONE, confidence HIGH** — but small
  absolute saving vs the while_loop cost.

* **E-floor-4: parallelize setup across the multi-GPU shard.** The Phase B 4-GPU
  result (0.93 s/params at B=64) shards the *solve*; shard the *setup* the same
  way (each device builds its B/4 BGs). With 4 devices the setup floor drops to
  ~1.0/4 = 0.25 s/params. **This is likely how the ~1 s target is actually hit.**

**E priority: E-floor-1 (vmap HyRex) is the structural fix; E-floor-4 (shard
setup) is the pragmatic one.** Both are needed to reach ~1 s/param.

---

## E.0. The spectrum loop (Phase E) — the actual current bottleneck (cross-ref)

Although the task scopes "solve + setup," the verified perf data
(`perf_batched.py`: batched asymptotes at ~12 s/params, *slower than single*)
shows the **spectrum Python loop dominates today** and must be fixed before any
solver tuning is observable. `get_Cl_batched`/`Pk_lin_batched` (spectrum.py:
616-659) loop over B calling `get_Cl`/`Pk_lin` per element (each `get_Cl` itself
already vmaps over ℓ at spectrum.py:594) — so the per-element calls pay full JIT
dispatch B times. The blocker is `BG.expmkappa` (spectrum.py:695, 729, 736) →
`self.kappa_func.evaluate(lna)` where `kappa_func` is a `diffrax.Solution`
(background.py:491,552,738-742) that does NOT stack across cosmologies (hence
`strip_bg_kappa`, perturbations.py:538-554).

**The fix is well-defined and localized** (CHANGELOG lines 166-187, CLAUDE.md):
`Background.kappa_func` is a `diffrax.Solution` produced by
`_tabulate_optical_depth(params)` (background.py:682-720, which calls
`diffeqsolve(..., saveat=SaveAt(dense=True))`) and consumed in `expmkappa(lna)`
via `self.kappa_func.evaluate(lna)` (background.py:738-742). The fix:

1. In `_tabulate_optical_depth`, change `saveat=SaveAt(dense=True)` to
   `saveat=SaveAt(ts=self.lna_tau_tab)` (or a recomb-focused grid) and return the
   sampled `(-sol.ys)` array instead of the `Solution`.
2. Add an array field `kappa_tab : jnp.array` to `Background` (replacing the
   `kappa_func : diffrax.solution` field at background.py:491), set it in
   `__init__` (background.py:552).
3. Rewrite `expmkappa` (background.py:738-742) to
   `jnp.exp(-tools.fast_interp(lna, self.lna_tau_tab[0], self.lna_tau_tab[-1],
   self.kappa_tab))` — **exactly mirroring how `tau` already works**
   (`tools.fast_interp(lna, self.lna_tau_tab[0], self.lna_tau_tab[-1],
   self.tau_tab)` at background.py:347).

After that: `kappa_func` is no longer a `diffrax.Solution`, so `Background`
**stacks cleanly** via `jax.tree.map(jnp.stack, ...)`, `strip_bg_kappa`
(perturbations.py:538-554) becomes unnecessary, and `get_Cl_batched`/
`Pk_lin_batched` (spectrum.py:616-659) collapse from Python loops to a single
outer `vmap` over (PT, BG, params). Downstream signatures (`visibility`,
`expmkappa` callers in spectrum.py:692-695) are unchanged.

NOTE: contrary to an earlier draft of this note, `Background` does **not** already
have `expmkappa_grid`/`bg_kappa_grid` placeholder fields — those do not exist; the
grid field must be added as step 2 above.

This is THE single highest-value change in the whole pipeline (turns ~12 s/params
into ~0.7 s/params per the Phase A single-call get_Cl=0.68 s). **Risk: LOW** (the
grid already exists; interp replaces a dense-Solution eval at the same tolerance).
**Confidence: HIGH.** It is outside the literal "solve" path but is the
prerequisite for the solve speedups to matter.

---

## F. Compilation / dispatch hygiene

### F1. `donate_argnums`
`_evolve_chunk` (perturbations.py:125, `@eqx.filter_jit`) allocates the full
`(K_CHUNK, B, Nlna, Ny)` output each call. For the chunked loop the inputs
(`lna_batch`, `BG_batch`, `params_batch`) are reused across chunks (read-only), so
donation of *those* is wrong, but the per-chunk `k_chunk` and the output buffer
can benefit. `eqx.filter_jit` supports `donate="all"`/`donate="warn"`. **Modest
saving (allocation churn), risk LOW, confidence MEDIUM.** Bigger lever is just not
chunking at small B (§B1).

### F2. Persistent compilation cache
Nothing in the repo sets `jax_compilation_cache_dir`. Cold start recompiles
everything (the multi-second Kvaerno5 vmap compile, ×2 for the ragged chunk, plus
HyRex). For a frequentist scan that restarts processes, set:
```python
jax.config.update("jax_compilation_cache_dir", "/pscratch/.../jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
```
**Pure cold-start latency win, risk NONE, confidence HIGH.** Doesn't touch steady
state but matters for SLURM job churn.

### F3. Pad B to a fixed size
As in §B3, ragged final batches recompile. Pad `params_list` to a multiple of a
fixed B and mask out the padding in the likelihood. **Risk NONE, confidence HIGH.**
Combine with F2 so a scan compiles once for B and reuses forever.

### F4. HyRex jit is fine; the redundancy is the per-call wrap, not double-jit
CORRECTION to a common assumption: `recomb_model.__call__` is a **plain method**
(hyrex.py:146), not `@eqx.filter_jit`-decorated — so there is no static double-jit
to remove. The real cost is that `call_batched` re-wraps `eqx.filter_jit(
self.RecModel, backend='cpu')(...)` **inside the per-element loop** (main.py:256),
constructing the filter_jit wrapper B times. Hoist the `eqx.filter_jit(
self.RecModel, backend='cpu')` wrapper out of the loop (build once, call B times),
or better, fold it into the vmapped HyRex of §E-floor-1. **Harmless to steady
state; minor dispatch tidy-up. Confidence: HIGH.**

---

## Ranked recommendations (highest value first)

| # | Change | Where (file:line) | Speedup (est) | Risk | Conf | Cheap test |
|---|--------|-------------------|--------------:|------|------|-----------|
| **1** | **Vmap the spectrum** (replace `kappa_func` diffrax.Solution with a `kappa_tab` array + `fast_interp` in `expmkappa`; drop the python loop → outer vmap) | background.py:491,552,682-720,738-742; spectrum.py:616-659 | **~12→~0.7 s/params** (single biggest win; unblocks everything) | LOW | HIGH | snapshot parity; mirror existing `tau`/`tau_tab` pattern |
| **2** | **Batch the CPU setup** (vmap HyRex over B; stack device transfers) — the coming floor | main.py:209-265; hyrex while_loops | ~1.0→~0.2-0.4 s/params | LOW | MED | time `_build_one_bg` looped vs vmapped at B=8 |
| **3** | **Stiffness-homogeneous, non-uniform k-chunking** (big chunks for smooth low-k, small for stiff high-k; A1) | perturbations.py:150-192 | modes 1.3-1.7x | NONE | HIGH | sweep chunk plan, compare modes wall-clock |
| **4** | **One big vmap at B≤8 / adaptive `k_chunk_size`** by memory (B1, B2 pad ragged) | perturbations.py:186-189 | modes 1.1-1.3x + kills 1 compile | NONE | HIGH | set k_chunk_size=N_k at B=8, time |
| **5** | **Shard setup + solve across 4 GPUs** (E-floor-4; Phase B proved solve sharding) | call_batched (new) | /4 on setup; restores linear scaling | MED | HIGH | reuse flipped_spike_multigpu pattern |
| **6** | **Explicit solver on the smooth small-k band** (A2) | perturbations.py:418 | smooth-band 2-4x (small abs) | MED | MED | Tsit5 on k<k_split, run gate |
| **7** | **Trim/redistribute `saveat` (500→~300 non-uniform at recomb)** (C) | perturbations.py:179-181 | 1.1-1.2x + OOM relief→higher B | MED | MED | regen snapshots, run gate at 300 |
| **8** | **Loosen `atol_large_k_PE`** within the stiff band (A1.2) | model_specs.py:66 | 1.1-1.3x | MED | MED | sweep {1e-6,3e-6,1e-5}, max-rel Cl |
| **9** | **float32 perturbation state `y`** only (D) | perturbations.py:414,436 | solve 1.4-1.8x | HIGH | LOW-MED | `y0.astype(f32)`, run accuracy gate |
| **10** | **Persistent compile cache + pad B** (F2,F3) | new config | cold-start only | NONE | HIGH | set cache dir, time 2nd job |

### The SINGLE highest-value change in the SOLVE path specifically

If we restrict to the solver (excluding #1 spectrum and #2 setup, which are
strictly larger wins but outside the "solve" kernel): **#3, stiffness-homogeneous
non-uniform k-chunking** in `_compute_modes_batched` (perturbations.py:186-189).
It directly attacks the only remaining lockstep tax (intra-chunk k-range, per
`chunking_debug_report.md`), is zero-accuracy-risk (identical math, only lane
grouping changes — already proven benign), needs no new diffrax features, and is
a ~20-line change to the existing Python chunk loop. Everything stiffer (A2/A3/D)
should be spiked only after #3 lands and is measured.

### The SINGLE highest-value change overall

**#1, vmap the spectrum** by replacing `Background.kappa_func` (a `diffrax.Solution`,
background.py:491,552) with a tabulated `kappa_tab` array consumed via
`tools.fast_interp` in `expmkappa` (background.py:738-742) — exactly the pattern
`tau`/`tau_tab` already uses (background.py:347). Today batched is *slower* than
sequential (perf_batched.log: B=16 → 12.08 s/params vs single 9.51) because of the
spectrum Python loop; no solver tuning is even observable until this is fixed. It
is the prerequisite for the whole batched design to pay out.

---

## Cheap measurement recipes (per lever, no full pipeline needed)

* **Modes-only timing** (isolate §A/§B): time `PE.full_evolution_batched(
  (BG_batch_stripped, params_batch))` alone at B∈{8,32,64} and chunk plans; the
  step-count distribution is already instrumented in `bench/baseline.py`.
* **Spectrum-only** (§E.0 / #1): build one PT, call `get_Cl` single vs a 2-element
  python-loop vs a prototype vmap; compare wall-clock and max-rel Cl. The grid
  fields exist so the interp prototype is local to `expmkappa`.
* **Setup floor** (§E / #2): time the `for params in params_list` loop
  (main.py:209-216) in isolation at B=8,32 — that *is* the floor; then prototype a
  single vmapped HyRex call on stacked inputs and compare.
* **Accuracy gate for #6-#9**: `cd pytests && pytest -s -vv` (the 1%-vs-CLASS
  contract) or the cheaper `test_snapshots.py` parity oracle for refactors that
  shouldn't change math (#3, #4).
* **Compile-cost levers** (§B2, §F): just print `time` of the first vs second
  `call_batched` in one process (compile vs steady); for §F2 compare two fresh
  processes with/without the cache dir.

---

## Appendix: confirmations that grounded the above

* Solver config (Kvaerno5, PIDController, per-k `jnp.where` rtol/atol, ForwardMode,
  saveat 500, max_steps 2048): perturbations.py:417-444. **Verified.**
* Double-vmap kernel (`vmap k_chunk × vmap B` around `evolution_one_k`, single
  fused lockstep controller): perturbations.py:142-148. **Verified.**
* Python-loop chunking (not lax.scan), contiguous `k_axis[i:i+k_chunk_size]`,
  `jnp.concatenate`: perturbations.py:186-191. **Verified.**
* Lockstep behaviour + intra-chunk worst-lane runtime: chunking_debug_report.md
  lines 24-31, 41-45. **Verified.**
* k-axis monotonic increasing (so contiguous chunk = k-band = stiffness-band):
  model_specs.py:120-175. **Verified.**
* Setup is a Python loop over B; `_build_one_bg` runs HyRex CPU sequentially with
  two device_put round-trips: main.py:209-265. **Verified.**
* HyRex = diffrax `diffeqsolve` (Kvaerno3/Tsit5) on fixed `SaveAt(ts=...)` grids
  (helium.py:542, hydrogen.py:371,482,754) + ONE `eqx.internal.while_loop`
  (hydrogen.py:266, static `max_steps`) + `array_with_padding`. `recomb_model.__call__`
  is a plain method (hyrex.py:146); jit applied only at call site
  `eqx.filter_jit(self.RecModel, backend='cpu')` (main.py:256,300). Hence
  vmap-able-in-principle (fixed-SaveAt diffeqsolve + masked-while) but currently
  sequential. **Verified.**
* Spectrum batched = python loop over B (spectrum.py:616-659); blocker is
  `kappa_func` diffrax.Solution in `expmkappa` (background.py:738-742) set by
  `_tabulate_optical_depth` with `SaveAt(dense=True)` (background.py:717). The
  field is declared `kappa_func : diffrax.solution` (background.py:491). NOTE: an
  earlier draft wrongly claimed `expmkappa_grid`/`bg_kappa_grid` placeholder fields
  already exist — they do NOT; the array field must be added (see §E.0).
  **Verified by direct read.**
* `tau` already uses `tools.fast_interp(lna, self.lna_tau_tab[0],
  self.lna_tau_tab[-1], self.tau_tab)` (the pattern to copy for `expmkappa`):
  background.py:347. **Verified.**
* Baseline/flipped numbers: bench/baseline_summary.txt, bench/flipped_summary.txt,
  CHANGELOG.txt lines 33-58,150-187. **Verified (re-read, not trusted blindly).**

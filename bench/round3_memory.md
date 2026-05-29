# Round 3 — Question A: reduce GPU memory per call (fit more B_local on one 80 GB A100)

Static analysis only. Cites `file:line`. Target: shrink the per-device peak so a
single 80 GB A100 holds more cosmologies (B_local = B/n_dev), raising single-node
throughput. Honors the hard constraints: stay near permille, no fp32/bf16 storage
unless ≪0.1% + gated, no diffrax regime switching, k_chunk stays 100.

## 0. What the peak actually is — re-derived from the code

Validated ground truth: per-device peak ≈ **3.65 · N_k · Nlna · Ny · 8B · B_local**,
independent of k_chunk. Let me map the 3.65 to actual tensors so we know what to cut.

The "1.0×" reference is the raw saved-ys tensor produced by the solve:

- `perturbations.py:178-192` `_compute_modes_batched`: each chunk from
  `_evolve_chunk` is `(K_CHUNK, B_local, Nlna=500, Ny)` (`perturbations.py:129`,
  `saveat=SaveAt(ts=lna)` at `:433`, `lna=linspace(lts,0,500)` at `:180`). The
  chunks are `jnp.concatenate(..., axis=0)` to `(N_k, B, Nlna, Ny)` then
  **`.transpose(1,3,2,0)` → `(B, Ny, Nlna, N_k)`** (`:191-192`).
  - **That transpose is a full out-of-place copy**: concatenated source `(N_k,B,Nlna,Ny)`
    AND the transposed `(B,Ny,Nlna,N_k)` both live simultaneously during the copy.
    That alone is **2.0×** the raw tensor at the transpose boundary.

- `make_output_table_batched` (`perturbations.py:224-248`) is `vmap(make_output_table)`
  over B. `make_output_table` (`:450-536`) **reads `modes` and emits a parallel set
  of derived (Nlna, N_k) fields**, all fp64, all the same (Nlna,N_k) footprint as one
  Ny-slice of modes:
  - `PerturbationTable` stored fields (`:571-582`): `delta_m`, `theta_b_prime`,
    `metric_eta`, `metric_h_prime`, `metric_eta_prime`, `metric_alpha`,
    `metric_alpha_prime` = **7 arrays of (Nlna,N_k)**.
  - `species_perturbations` dict (`:477-480`, the `output_perturbations` of each
    species): Photon→5 (`species.py:1388-1395`: delta,theta,sigma,G0,G2),
    Baryon→2 (`:1225-1229`), CDM→1 (`:547-548`), MasslessNu→3 (`:693-698`),
    DarkEnergy→0. = **11 arrays of (Nlna,N_k)** for massless ΛCDM.
  - So PT holds **~18 (Nlna,N_k) arrays = 18/Ny ≈ 18/46 ≈ 0.39×** the raw modes
    tensor (Ny≈46). PT does NOT scale with Ny — it's a fixed ~18 fields.
  - PLUS the transient sums in make_output_table (`:500-523`): `sum_rho_delta`,
    `sum_rho_plus_P_theta`, `sum_rho_plus_P_sigma`, `sum_rho_delta_m` (4 more
    (Nlna,N_k)) and per-species `vmap(s.rho_*)` temporaries (`:508-511`). These are
    transient but coexist with `modes` (still alive — needed for the species loop)
    and with the growing PT fields → another ~0.2-0.4× spike inside the vmap body.

So the 3.65× decomposes roughly as: **modes (1×) + transpose copy (still ~1× alive at
the boundary) + PT derived fields (~0.4×) + make_output_table transient sums/temporaries
(~0.3-0.6×) + spectrum working set (see §2, can co-peak).** The exact 3.65 is XLA's
high-water mark across these overlapping lifetimes; the point is **modes + its
transpose copy is the bulk, and modes stays alive through make_output_table.**

Concrete scale (massless ΛCDM, Ny=46, N_k=492, Nlna=500): raw modes =
492·46·500·8 = 0.0905 GB/B_local → ×3.65 ≈ 0.33 GB/B_local. Matches the measured
0.33. At 66 GB usable, B_local ≲ 200.

## 1. Ideas (mechanism / effect range / accuracy risk + gate / effort / prob / hidden floor)

### Idea A1 — Trim Nlna 500→~300 with a recomb-dense save grid  ★ #1 BET
**Mechanism.** Nlna is a *linear* multiplier on the WHOLE peak (modes, PT, spectrum
scan all carry it). The save grid is `jnp.linspace(BG.lna_transfer_start, 0., 500)`
(`perturbations.py:180`, and the single-call twin `:100`). The LoS integral
(`spectrum.py:676` `lna_axis = PT.lna[:-1]`, trapezoid weights `:792-793`, scan
`:817`) and Pk (`spectrum.py:293` `jnp.interp(lna, PT.lna, ...)`) read this SAME grid.
The visibility g(lna) (`spectrum.py:683`) is sharply peaked at recombination
(Δz~80, i.e. Δlna~0.06 around lna_rec≈-7), and the source terms vary fastest there;
the rest of the range (lna_transfer_start≈-15 → 0, ~15 e-folds) is smooth ISW/Doppler.
A **non-uniform grid dense around lna_rec and coarse elsewhere** carries the same
LoS/Pk accuracy with fewer points. CLASS samples the source perturbations on ~70-150
non-uniform conformal-time points, not 500 uniform e-folds — strong prior that 500
uniform is over-resolved.

**Effect.** Nlna 500→300 is a flat **0.6× on the entire peak** → B_local 200→**~333**
(massless), or equivalently fit B=64 in 3.2 GB/dev instead of 5.3. 500→250 → 0.5× →
B_local→~400. This is the single largest accuracy-neutral lever because it hits
*every* term in the 3.65 at once AND speeds the LoS scan (fewer scan steps) and the
diffrax `SaveAt` (fewer dense-interp evaluations per accepted step).

**Accuracy risk + gate.** Medium-low IF the grid stays recomb-dense; the gate risk is
under-resolving the visibility spike (smooths the acoustic peaks → TT/EE bias) or the
trapezoid weights `:792-793` assuming uniform Δlna (`delta_lna = PT.lna[-1]-PT.lna[-2]`,
`:677`) — **that uniform-spacing assumption MUST be replaced with per-interval
`jnp.diff(lna_axis)` trapezoid weights if the grid becomes non-uniform** (else silent
bias). Cheap gate: regenerate Cls at Nlna∈{500,400,350,300,250} on a *uniform* grid
first (one-line change at `:180` + `:792-793` already uniform) and watch the
pytests/accuracy_test.py max-rel vs CLASS — find where TT/EE/Pk crosses ~0.05% extra.
Then, separately, swap to a recomb-dense non-uniform grid (tanh-clustered around
`BG.lna_rec`) with proper non-uniform trapezoid weights and confirm it recovers the
Nlna=500 number at lower point count. **Effort.** Low for the uniform sweep (1-2 lines);
moderate for the non-uniform grid (need a grid builder using `BG.lna_rec` +
non-uniform trapezoid weights at `spectrum.py:792`, and the Pk `jnp.interp` is already
grid-agnostic). **Prob.** High that 500→~350 is permille-neutral on a uniform grid; very
high that a recomb-dense 250-300 matches 500. **Hidden floor.** The diffrax solve cost
is set by the *stiffest mode's step count*, not by Nlna (SaveAt is interpolation onto
the dense solution, cheap) — so Nlna cuts memory + LoS-scan time but NOT the dominant
solver wall-time. It's a memory/B_local lever first, a modest speed lever second.

### Idea A2 — Stream/fuse modes→PT→Cl per k-chunk so the full (B,Ny,Nlna,N_k) never materializes  ★ #2
**Mechanism.** Today: all N_k chunks are concatenated into one `(B,Ny,Nlna,N_k)`
tensor (`perturbations.py:191`), transposed (the 2× copy), turned into a full PT, and
only THEN does the spectrum consume it. The LoS Cl integral is
`∫ dk source(k,lna)·j_l(kχ)/k` — **a sum over k that is naturally chunk-decomposable**:
`Cl = Σ_chunks ∫_{k in chunk} ...`. The Pk is a `jnp.interp` over k (`spectrum.py:300`).
If we evolve k-chunk → build only that chunk's PT slice → fold into per-ell LoS
accumulators (transferT0..E are already (N_k,) accumulators built by the scan over
*lna*, `spectrum.py:796-819`) → discard the chunk, **the persistent footprint drops to
~one chunk's worth of modes (N_k=100 not 492) plus the (B,Nell) Cl accumulators.**
Persistent peak would fall from `3.65·492·…` toward `~(100/492)·raw + accumulators`,
i.e. a **~3-5× reduction** of the persistent tensor (the chunked modes become
transient, freed per chunk).

**What blocks it (quantified).** The spectrum does a **global cubic spline over the
FULL PT.k axis**: `interp_column = CubicSpline(jnp.log10(PT.k), col, ...)`
(`spectrum.py:700`, used at `:705-714`) maps the N_k≈492 perturbation grid onto the
N_k_transfer=2500 transfer grid (`k_axis_transfer`, `:184`/`:675`). A cubic spline is
GLOBAL — every transfer-grid k depends (through the tridiagonal solve) on ALL 492
source-grid points → **you cannot evaluate the spline until all 492 k-modes exist.**
That is the single thing forcing materialization of the full k-axis. So a clean
per-chunk stream is blocked at the spline.
*Two ways around it, both real work:*
  (a) **Restructure the spline to be the only thing that sees full-k:** keep modes
  chunked and transient, but accumulate into the *18 PT fields* (Nlna,N_k) as you go
  (each chunk writes its k-columns). The PT (18·Nlna·N_k, ~0.39× raw) is far smaller
  than the modes tensor (Ny·Nlna·N_k); if modes are freed per chunk, persistent peak ≈
  PT (0.39×) + one chunk modes (0.2×) + spectrum, i.e. **~2-2.5× smaller than today's
  3.65.** This keeps the global spline (operates on the assembled PT) and only removes
  the full *modes* tensor. **This is the high-value, lower-risk subset of A2.**
  (b) Replace the global cubic spline with a *local* (per-chunk) interpolation so the
  LoS can accumulate per chunk — REJECTED: round2_plan.md:38 flags the cubic spline
  over the perturbation k-axis as *the accuracy gate itself* ("spline accuracy IS the
  gate"). Local interpolation is an accuracy trade → out of bounds.

**Effect.** Variant (a): persistent peak ~3.65→~1.6-2.0× → B_local 200→**~360-450**.
**Accuracy risk.** Variant (a) is bit-identical-in-spirit (same spline on same
assembled PT; only the *construction order* of PT changes from "build full modes then
slice" to "accumulate columns per chunk"). The diffrax PID step noise is already
chunk-dependent and accepted (`perturbations.py:155-160`,
bench/chunking_debug_report.md), so per-chunk PT assembly introduces nothing new.
Gate: snapshot/accuracy test. **Effort.** HIGH — `make_output_table` (`:450-536`)
currently takes the whole `modes` and does species loops; refactoring it to accept and
scatter per-k-chunk columns into pre-allocated PT buffers, while keeping the
`theta_b_prime` backward-calc (`:497`) and metric sums (`:519-523`) correct per column,
is a real rewrite. Donation of the chunk buffer (`donate_argnums` on `_evolve_chunk`)
pairs with it. **Prob.** Moderate (correctness fiddly; the species `output_perturbations`
and the metric sums must be re-expressed column-wise). **Hidden floor.** PT itself
(0.39×) + the spectrum working set (§2) become the new floor; you can't go below ~PT +
one chunk + spectrum. And the transpose copy (§0, 2× spike) must be eliminated too
(write modes already in (B,Ny,Nlna,N_k) layout or transpose per-chunk) or it re-imposes
a 2× spike on the chunk.

### Idea A3 — Kill the full-modes transpose copy; free `modes` before/inside make_output_table
**Mechanism.** Two flat wins independent of A1/A2:
  (i) `perturbations.py:191-192` builds the concatenated `(N_k,B,Nlna,Ny)` AND its
  transpose `(B,Ny,Nlna,N_k)` — a full duplicate live at once (**2× the raw tensor at
  that instant**, the dominant contributor to "3.65"). Fix: have `_evolve_chunk` emit
  each chunk already transposed to `(B,Ny,Nlna,K_CHUNK)` (move the transpose *inside*
  the per-chunk JIT where it's 1/5 the size, `perturbations.py:142-148`), then
  `concatenate(axis=3)`. The concatenate still copies once, but you never hold the
  whole `(N_k,B,Nlna,Ny)` and its full transpose simultaneously — the spike drops from
  2× to ~1.2×.
  (ii) `modes` stays alive through the entire `make_output_table` species loop
  (`:506-516`) because it's the input. Once `species_perturbations` (`:477`) and the
  metric sums (`:500-523`) are computed, `modes` is dead — but under one fused vmap'd
  jit, XLA keeps it until the PerturbationTable is returned. `donate_argnums` on
  `make_output_table_batched` (`:224`) lets XLA alias/free the modes buffer as PT
  fields are written.
**Effect.** Removing the 2× transpose spike alone could take 3.65→**~2.6-2.9×** →
B_local 200→**~250-280**, with ZERO accuracy impact (pure layout/scheduling).
**Accuracy risk.** None (bit-identical — same arithmetic, different buffer layout).
Gate: snapshot test (rtol). **Effort.** Low-moderate (transpose relocation is a few
lines; donation needs the buffer not to be reused). **Prob.** High the transpose win
lands; donation is XLA-dependent (may already partially elide). **Hidden floor.** The
peak then reverts to whichever of {make_output_table transient sums, spectrum working
set} is next-highest — likely the spectrum (§2) becomes the binding peak, which is why
A3 pairs with A5.

### Idea A4 — `donate_argnums` / buffer aliasing across the batched stages
**Mechanism.** In `call_batched` (`main.py:291-303`) the live set after the modes solve
is BG_batch + modes/PT_batched + params_batch + (Cls, Pk). `full_evolution_batched`
(`perturbations.py:194-222`) holds `modes` then `PT`; `get_Cl_batched`
(`spectrum.py:616-640`) and `Pk_lin_batched` (`:642-650`) consume PT. Mark:
  - `make_output_table_batched` (`:224`): donate `modes_batch` (dead after PT built).
  - The `_evolve_chunk` chunk buffers are loop-local and already freed per iteration
    (Python list `chunks`, `:185-189`) — **do NOT donate `lna_batch`/`BG_batch`/
    `params_batch`**, they're reused across chunks and by the spectrum (the brief's
    "NOT chunk-loop inputs" caveat).
  - `get_Cl_batched`/`Pk_lin_batched` both read PT_batched; PT must stay alive until
    BOTH finish (`main.py:298-302`). Reorder so Pk (reads only PT.delta_m + params,
    `spectrum.py:297-300`) runs first and frees its slice, or fuse Cl+Pk into one
    jit so PT is consumed once. Marginal.
**Effect.** Donating modes overlaps with A3(ii): **~0.3-0.5× off the transient peak**,
i.e. helps fit ~10-20% more B_local. **Accuracy risk.** None. **Effort.** Low.
**Prob.** Moderate (XLA donation only frees if the donated buffer truly isn't reused;
under vmap the aliasing analysis sometimes declines). **Hidden floor.** Donation can't
beat the persistent PT + spectrum working set; it only trims transient double-buffering.

### Idea A5 — B-axis chunk the spectrum (if it's the secondary peak after A1/A3)
**Mechanism.** `get_Cl_batched` is `vmap(get_Cl, in_axes=0)` over B (`spectrum.py:639`).
Inside, `get_Cl` does `vmap(Cl_one_ell)` over ~Nell ells (`:594`); each `Cl_one_ell`
builds the source arrays `delta_g…alpha_prime` (`:705-714`) and the 4 source terms
`sourceT0..sourceE` (`:717-732`), each shape **(Nlna-1, N_k_transfer=2500)**
(`k_axis_transfer = geomspace(1e-4,0.4,2500)`, `:184`). These 4 sources are the scan
`xs` (`:812`) → under the ell-vmap the live set is **4·Nell·(Nlna-1)·2500·8B**. At
ELLMAX=800, Nell≈55 → 4·55·499·2500·8 ≈ **2.2 GB per B_local**; at ELLMAX=2500,
Nell≈98 → **~3.9 GB per B_local**. The `jax.checkpoint` on the scan body (`:817-819`)
already removed the (Nell,Nlna,Nk) *integrand* backward-rematerialization (the comment
cites ~21 GiB) but the **forward source `xs` tensor (Nell,Nlna,2500) is still resident.**
This is comparable to the persistent PT tensor — so after A1/A3 shrink the modes tensor,
**the spectrum likely becomes the binding peak.**
*Levers:*
  (i) **B-chunk the spectrum**: run `get_Cl_batched` over B in sub-batches (e.g. B/4)
  so the spectrum working set is `(B/4)·(spectrum set)`, sequenced. Pure scheduling,
  bit-identical. Cuts the spectrum peak ~linearly in the chunk count.
  (ii) **ell-chunk inside get_Cl**: vmap `Cl_one_ell` over ell *sub-chunks* with a
  python loop (mirrors the k_chunk pattern in perturbations) so (Nell,…) → (Nell/c,…)
  resident. Bit-identical (the cubic-spline-over-ell at `:599-601` runs after, on the
  assembled raw Cls, only ~98 numbers — unaffected).
  (iii) The interpolated source arrays (`:705-714`, ~14 of them at (499,2500)) are
  computed then consumed into 4 sources; they're transient but co-peak with the sources
  → ell-chunking shrinks them too.
**Effect.** Removes the spectrum as the binding constraint after A1/A3 → lets B_local
ride the (now-smaller) PT tensor. Could be the difference between B_local~280 and ~400
once modes are shrunk. **Accuracy risk.** None (scheduling/chunking, identical math).
Gate: snapshot test. **Effort.** Low (B-chunk: a python loop around the vmap in
`get_Cl_batched`; ell-chunk: a loop in `get_Cl`). **Prob.** High. **Hidden floor.** The
4 source arrays at minimum one-ell-chunk (4·(Nlna)·2500·8B ≈ 40 MB/B_local) + the
transfer accumulators; can't go below that.

### Idea A6 (CONDITIONAL re-propose) — fp32 storage of the SAVED modes ONLY, gated
**Mechanism.** round2_plan.md:35 rejected fp32 storage because "fp32 storage propagates
fp32 through make_output_table's fp64 metric sums." But the *gate-relevant* contract is
the downstream Cl/Pk at permille, and the saved trajectories already carry diffrax PID
noise at `rtol_large_k_PE=1e-4` (`perturbations.py:155-159`) — i.e. the modes are only
~1e-4 accurate to begin with, **~250× looser than fp32's ~1e-7 relative.** Storing
`modes` (and only modes) as fp32 while *upcasting to fp64 at the make_output_table
boundary* (so the metric sums `:500-523` and theta_b_prime `:497` stay fp64) keeps the
arithmetic fp64; only the *saved-buffer dtype* is halved → **the persistent modes tensor
halves (1×→0.5×), peak 3.65→~3.0-3.1×** → B_local 200→~240. **Accuracy risk.** The
concern is real: fp32 round-off (~1e-7) on a quantity already at 1e-4 should be
negligible, BUT differencing (e.g. `theta_g-theta_b` in `:497`, or the metric sums of
many species) can amplify cancellation. **This is why it's CONDITIONAL and last.**
Cheap gate: store modes fp32, upcast at the make_output_table input, run
pytests/accuracy_test.py — accept ONLY if TT/EE/Pk stay ≪0.1% above the fp64 baseline
(0.197/0.231/0.185%). If it moves any by >0.03%, drop it. **Effort.** Low (cast at
`_evolve_chunk` output + upcast at `make_output_table` input). **Prob.** Moderate (the
diffrax tol argument is strong, but cancellation in the metric sums is the unknown — must
measure). **Hidden floor.** Only halves the modes term; PT (fp64) + spectrum (fp64)
unaffected → caps at ~0.65× total, less than A1.

## 2. Ranked shortlist (top 5)

1. **A1 — Trim Nlna (500→~300, recomb-dense non-uniform grid).** The only lever that is
   a *flat multiplier on the entire 3.65 peak* (modes + PT + LoS scan) AND speeds the
   LoS scan. 500→300 = 0.6× → B_local 200→~333; 500→250 → ~400. Low effort for the
   uniform sweep that establishes the headroom; moderate for the non-uniform grid (must
   replace the uniform trapezoid weights at `spectrum.py:792-793` with per-interval
   `jnp.diff`). Accuracy-neutral if recomb-dense. **Highest benefit×prob÷(risk×effort).**

2. **A2(a) — Accumulate PT per k-chunk; never materialize the full (B,Ny,Nlna,N_k)
   modes tensor.** Keep the global cubic spline (the gate) operating on the assembled
   PT, but build PT column-by-column from transient per-chunk modes so the *modes*
   tensor (the Ny·… bulk) is freed per chunk. Persistent peak 3.65→~1.6-2.0× → B_local
   ~360-450. High effort (rewrite make_output_table column-wise) but the biggest
   structural win and accuracy-neutral. The spline over PT.k is the only true blocker
   and A2(a) routes around it without touching it.

3. **A3 — Eliminate the modes transpose copy + donate modes.** Move the transpose
   inside `_evolve_chunk` so the full `(N_k,B,Nlna,Ny)`+transpose never coexist (2×→1.2×
   spike). Pure layout, bit-identical, low effort. 3.65→~2.6-2.9× → B_local ~250-280.
   The cheapest *guaranteed* (zero-accuracy-risk) win; do it regardless.

4. **A5 — B-/ell-chunk the spectrum.** After A1/A3 shrink the modes tensor, the
   spectrum's forward source `xs` `(Nell,499,2500)` (~2.2 GB/B_local @ELLMAX800,
   ~3.9 @2500; `spectrum.py:812`) becomes the binding peak. B-chunking
   `get_Cl_batched` (a python loop around the B-vmap at `:639`) or ell-chunking inside
   `get_Cl` (around the ell-vmap at `:594`) cuts it linearly, bit-identical, low effort.
   Necessary to *realize* A1/A3's B_local gains rather than just shifting the bottleneck.

5. **A4 — donate_argnums across stages + reorder Cl/Pk.** Trim transient double-
   buffering (donate `modes_batch` into `make_output_table_batched`; run Pk before Cl or
   fuse them). ~10-20% more B_local, zero accuracy risk, low effort, but XLA-donation is
   not guaranteed to fire under vmap. Supporting lever, not a headliner.

(A6 fp32-modes-storage is the *conditional* re-proposal: only ~0.65× and must clear the
accuracy gate; keep in reserve behind A1-A5.)

## 3. #1 bet + cheapest de-risking measurement

**#1 bet: A1 (trim Nlna).** It is a flat multiplier on the *entire* peak, the lowest-
effort first step, and the prior (CLASS uses ~70-150 non-uniform source-time points vs
ABCMB's 500 uniform e-folds; the visibility is a narrow recomb spike) strongly suggests
500 uniform is over-resolved. Combined with A3 (free) and A5 (to stop the spectrum
becoming the new wall), the realistic accuracy-neutral target is **B_local ≈ 350-450 on
one 80 GB A100** (vs ~200 today) — i.e. roughly **2× more cosmologies per node** before
adding the structural A2(a) rewrite, which can push further.

**Cheapest GPU/gate measurement to de-risk A1:** change the single literal at
`perturbations.py:180` (`jnp.linspace(lts, 0., 500)` → `400/350/300/250`) and the twin
at `:100`, leave the uniform trapezoid weights as-is (still valid for a uniform grid),
and run **`pytests/accuracy_test.py`** (TT/EE/Pk vs CLASS) at each Nlna. One short GPU
session, ~5 model() evaluations — no batched run needed since the *per-mode* accuracy is
what's at stake. The first Nlna at which TT/EE/Pk exceed ~0.05% over the 0.197/0.231/
0.185% baseline is the uniform-grid floor; the recomb-dense non-uniform grid then
recovers headroom below it. As a direct memory check, separately run
`call_batched(B=64, shard=False)` at Nlna=500 vs 300 and read
`gpus[0].memory_stats()['peak_bytes_in_use']` — it should drop ~0.6× (5.3→~3.2 GB),
confirming Nlna is the flat multiplier the model predicts and that B_local headroom
grows ∝ 500/Nlna.

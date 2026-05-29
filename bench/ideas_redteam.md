# ideas_redteam.md — Skeptical red-team of the proposed `call_batched` wins

**Author:** performance red-team + numerical-methods pass, 2026-05-29 (branch `perk-refactor`).
**Mandate:** for each floated win (A–H), find the HIDDEN FLOOR that makes it underperform,
say whether it survives, give a net expected speedup *after* the floor, and rank the
algorithmic/precision wins by (speedup × prob-success ÷ accuracy-risk). Goal context:
drive `Model.call_batched` toward ~1 s/param.

**Method:** static analysis only — no Python/GPU run. All wall-clock numbers are quoted
from in-tree artifacts and cross-checked. Code facts are cited by `file:line` from a direct
read of `abcmb/{main,perturbations,spectrum,background,model_specs}.py`,
`pytests/{accuracy_test,test_snapshots}.py`, and the three `bench/notes_*.md`.

---

## 0. The ground truth that reframes everything (read this first)

The freshest measured artifact is **`bench/profile_stages.log`** (today, per-stage fences,
post-compile). It is *more authoritative* than the older `perf_batched.log` because it
splits the four stages of the real `call_batched` body. The headline:

| B  | total/params | setup (seq CPU) | perturb (GPU) | spec_Cl (py loop) | spec_Pk |
|---:|-------------:|----------------:|--------------:|------------------:|--------:|
| 1  | 22.46 s      | **7.09 s**      | 12.16 s       | 3.18 s            | 0.02 s  |
| 8  | 12.62 s      | **6.87 s**      | 2.56 s        | 3.18 s            | 0.01 s  |
| 16 | 12.41 s      | **6.96 s**      | 1.97 s        | 3.46 s            | 0.01 s  |

**Three facts this table forces, and they overturn the older narrative:**

1. **Setup is the #1 cost at every B≥8, not the spectrum.** It is a *flat ~7 s/params*,
   amortizes zero, and at B=16 is 56% of wall-clock. The older notes (`notes_solve.md` §E,
   `notes_strategy.md` §1a) *estimated* setup at "~1.0 s/params"; the measurement says
   **~7 s/params — 7× worse than every prior estimate.** Any plan that defers setup to
   second place is mis-prioritized against this data.
   - **Corroboration:** `sweep_kchunk.py` reports building 32 BGs took **244.6 s ≈ 7.6
     s/cosmo** (sweep_kchunk.log) — an independent confirmation of the ~7 s setup floor.
   - **Stale estimate flagged:** `probe_setup.py`'s header says "the profiler showed setup
     ~1.85 s/param." That was an earlier/different probe (single fiducial, likely a warmer
     path) and is **contradicted by the actual `call_batched` per-stage fence (7 s)**. Trust
     `profile_stages.log`: in the real loop the per-cosmology BG build is ~7 s.

2. **`spec_Cl` is NOT O(B)-growing the way the loop story implied.** It is **~3.2 s/params,
   essentially flat** from B=1→16 (3.18 → 3.18 → 3.46). At B=1 there is no loop yet it still
   costs 3.18 s. So the "~7–8 s/params python-loop dispatch tax" inferred in
   `notes_strategy.md` §1a does **not** appear in the direct measurement. The spectrum cost
   is dominated by the **actual `Cl_one_ell` LoS work** (vmap over the lensing ells × scan
   over 499 lna × cubic-spline transfer interp onto N_k_transfer≈2500), not by per-element
   JIT dispatch. This has a direct consequence for win B.

3. **perturb already amortizes hard and is no longer the bottleneck:** 12.16 (B=1) → 1.97
   (B=16) s/params. The double-vmap flip works as advertised. At B=16 it is only 16% of
   wall-clock.

So the *real* per-params budget at B=16 is **setup 7.0 + perturb 2.0 + spec_Cl 3.5 +
spec_Pk 0.0 ≈ 12.5 s**, and the dominant terms are **setup (flat) and spec_Cl (flat)**, not
perturb. The 1 s/param target requires killing *both* flat terms.

**Config caveat (matters for the gate):** `profile_stages.py`, `probe_setup.py`, and
`sweep_kchunk.py` all run `lensing=False`, which gives N_k=492. The accuracy gate
(`accuracy_test.py`) and the older `perf_batched` numbers use `lensing=True`, N_k=571,
ellmax=2500. Setup is lensing-independent (HyRex / reion Newton / device transfers), so the
7 s floor holds for the gate config too. But **spec_Cl is LARGER with lensing on** — the
`lensed_Cls` Wigner-matrix path (spectrum.py:437-572) runs only when `lensing=True`. So the
measured 3.2 s spec_Cl is a *lower bound* for the production (lensed) config, which
*strengthens* the case for fixing the spectrum, but still as a dispatch + sharding enabler,
not a 10× single-GPU win.

> Where this red-team disagrees with the older `notes_*.md`, it is because the per-stage
> measurement now exists; the older notes worked from a single end-to-end number and
> over-attributed cost to the spectrum loop and under-attributed it to setup.

---

## PART 1 — RED-TEAM of every proposed win (A–H)

### Summary table

| Win | Hidden floor / failure mode | Survives? | Net expected speedup (end-to-end, after the floor) | Confidence |
|-----|------------------------------|-----------|----------------------------------------------------|------------|
| **A. jit+vmap the eager setup (get_BG)** | The ~7 s is NOT one jittable kernel. It is (i) `add_derived_parameters` eager Python with `sys.exit()` branches + Python `for s in species_list` loops (main.py:448-692); (ii) `get_BG_pre_recomb` — **already** `@eqx.filter_jit` (main.py:317), so "jit it" buys ~0; (iii) HyRex on CPU behind 2× `jax.device_put` round-trips (main.py:254-260); (iv) `get_BG` run **eager** (not jitted) in `_build_one_bg`, containing a `lax.cond` reion branch (main.py:439) whose chosen `ReionizationModelFromTau` runs an **`optx.Newton` root-find** (background.py:1014-1017) + the kappa `diffrax` solve (background.py:708) + several eager `vmap(visibility)` scans (background.py:555-564). A single GPU `jit` of get_BG removes the eager-dispatch part only; the HyRex CPU solve, the Newton solve, and the B× transfers remain. The real lever is **vmap the whole setup over B on CPU** (= win #2/E) — a harder change blocked-in-practice (not in principle) by: Newton root-find (vmaps but locksteps, may not converge uniformly), the GPU/CPU `device_put` boundary, and the int `array_with_padding` leaves. | **PARTIALLY** | "Just jit get_BG" ≈ small (pre_BG already jitted; probe_setup tested get_BG eager vs jitted). The structural win is **vmap HyRex+reion+transfers over B**: collapses B× dispatch + B× transfers + B× `eqx.filter_jit(RecModel,...)` re-wrap (main.py:256). Realistic **7.0 → ~1.5–3 s/params** (CPU vmap helps via dispatch/transfer elimination, NOT SIMD; the lockstepped CPU diffrax + Newton still run worst-lane). | MED |
| **B. tabulate kappa_func → vmap spectrum** | Blocker analysis is **correct** (kappa_func is a `diffrax.Solution`, background.py:491,552,741; sole `.evaluate` at :741) and the fix is clean. BUT measured spec_Cl is **flat ~3.2 s/params** — it does NOT explode with B. The older "12 → 0.7 s" projection is wrong: removing the python loop swaps B sequential `Cl_one_ell` graphs for one vmapped-over-B graph, but the *arithmetic* (LoS scan × lensing ells × transfer interp) is identical and is what costs the 3.2 s. `get_Cl` is a **plain method** (spectrum.py:574), so XLA caches its graph after iter 0 → iters 1..B are cheap-dispatch + real compute; the flat curve proves dispatch is a *small* fraction. Hidden floor: **the LoS compute (~3 s) is largely irreducible by vmap** (already vmap'd over ells internally). New risk: vmap-over-B nests `vmap(Cl_one_ell over Nell)` inside `vmap(B)` → (B, Nell, Nlna, Nk) working set can OOM at B=32/64 (the `jax.checkpoint` at spectrum.py:827 mitigates but adds ~2× recompute). | **YES (but smaller than billed)** | **3.2 → ~2–2.5 s/params single-GPU** (kills per-element dispatch, shares bessel constants; does NOT collapse the LoS arithmetic). Real value: it **unblocks B-sharding of the spectrum** and removes `strip_bg_kappa`/`BG_list` cruft. Bet for cleanliness + sharding-enablement, **not** a 10× single-GPU spectrum win. | MED-HIGH |
| **C. tune k_chunk / one big vmap** | At B≤8 a single `k_chunk_size=N_k` kills 5 of 6 launch barriers — but perturb at B=8 is already only **2.56 s/params** (20%) and at B=16 **1.97 s** (16%). Floor: **perturb is no longer the bottleneck**, so a 1.3× win is ~0.5 s off a 12.5 s budget = **~4% end-to-end.** Ragged final chunk (N_k=492 or 571, not ÷100) recompiles a 2nd kernel (perturbations.py:131-134 cache keyed on k_chunk.shape) — cold-start only. One-big-vmap at B≥32 risks OOM (design_memo: 28–31 GB peak at B=64, K_CHUNK=100 mandatory). | **YES but low ROI now** | **~1.0–1.05× end-to-end.** Worth it only *after* setup+spectrum are fixed, when perturb re-emerges as the floor. | HIGH (works), HIGH (low-priority now) |
| **D. 4-GPU shard over B** | Shards the *GPU* work (perturb, and spectrum if B is done). But **setup runs on CPU sequentially** — sharding the GPU does nothing for the 7 s/params flat CPU setup unless you *also* shard setup across ranks (4 ranks × B/4 BGs). The Phase-B 0.93 s/params number is **PE-only** (CHANGELOG CAVEAT; `notes_strategy.md` §1c) — it excludes setup AND spectrum. Honest floor with setup unsharded: ≈ setup 7.0 + perturb/4 + spec/4 ≈ **7 + 0.5 + 0.9 ≈ 8.4 s/params**, barely better than single-GPU because the dominant term isn't on the GPU. | **ONLY IF setup is sharded/batched first** | With setup *also* sharded: ~1.75 (setup/4) + 0.5 (perturb) + ~0.9 (spec) ≈ **~3 s/params**. Without setup sharded: **~8 s, near-useless.** D is a multiplier on A/B/E, not a standalone win. | MED |
| **E. float32 hierarchy ODE** | A100 fp64 ≈ ½ fp32; Kvaerno5's implicit Jacobian/LU is FP-heavy, so the upside (~1.4–1.8× on perturb) is real. Floors: (1) perturb is only **16–20%** of the budget now → 1.8× = ~1 s off 12.5 = **~8% end-to-end**; (2) **accuracy**: the tight-coupling RHS term `R/aH/tau_c·(θ_g−θ_b)` (perturbations.py:497) and the metric `4πGa²ρδ` sums (perturbations.py:370-371) are large-cancellation/stiff ratios fp32 (~1e-7 eps) can corrupt; the Cl chain squares the transfer (spectrum.py:834-836), amplifying rel error; (3) the `_to_float`→float64 cast (main.py:245-250) is load-bearing for the checkpointed_while_loop custom-vjp — fp32 means a fp32 cast surviving the GPU/CPU boundary AND the int `array_with_padding` leaves. High blast radius for a small share. | **PROBABLY (gate-risky), low-priority** | perturb 2.0 → ~1.2 s best case = **~1.06× end-to-end.** Not worth the 1%-gate risk until setup+spectrum dominate less. | LOW-MED |
| **F. reduce saveat / N_k / l-sampling** | saveat=500 IS the LoS time grid (spectrum.py:685 `lna_axis=PT.lna[:-1]`; scan at :826 over those 499 pts) — trimming cuts BOTH modes memory AND spec_Cl compute (double benefit). Floor: most of the 500 uniform pts sit where nothing happens; recomb is a sharp visibility spike near lna_rec; a naive uniform cut risks the gate, a recomb-dense non-uniform grid holds it. l-sampling: `get_Cl` **already** cubic-splines from the sparse computed `lensing_ells_indices` up to all ells (spectrum.py:594,598-601) → "fewer ells" has little headroom without re-tuning spline knots. N_k=571 is CLASS-style adaptive (model_specs.py:120-175); modes not obviously redundant. | **PARTIALLY** | saveat 500→~300 non-uniform: perturb ~1.15× AND spec_Cl ~1.3× (scan 499→~300) → **~1.1–1.15× end-to-end** + real OOM relief enabling higher B. Best single *algorithmic* lever on spec_Cl. Gate-risk MED. | MED |
| **G. lower kappa ODE rtol=1e-10** | `_tabulate_optical_depth` (background.py:706) uses rtol=atol=1e-10, run **once per cosmology at BG-build time** (inside setup). Loosening to 1e-7 cuts steps in *that one scalar ODE* (1 state, lna 0→−10). Floor: it is a **microscopic fraction of the 7 s setup** — setup is dominated by HyRex (full xe/Tm recomb), the reion Newton solve, and 2× device transfers, NOT this scalar quadrature. Saves maybe tens of ms. | **YES but negligible** | **~1.00–1.01× end-to-end.** Numerically safe (1e-7 is still 3 orders tighter than the 1e-4 PE rtol that sets the Cl floor), but the time saved is in the noise. | HIGH |
| **H. donate_argnums + persistent compile cache** | Pure **cold-start/compile** win (compiles 96–224 s in profile_stages warm rows). Zero steady-state per-params effect. donate pitfall: donating the reused chunk-loop inputs `BG_batch`/`params_batch` is **wrong** — read every chunk (perturbations.py:186-189) → donation frees them after chunk 0, corrupting chunk 1+; only the fresh output buffer is safe. Persistent cache is HLO-keyed: safe across reruns of identical code; a semantically-neutral edit that nonetheless perturbs HLO (exactly the scheduler-shift effect documented in test_snapshots.py:67-70) correctly misses the cache; the dangerous stale-hit direction is guarded by the HLO key. | **YES (scoped)** | **0× steady-state; large on cold start** (~100–220 s compile per fresh SLURM process). Worth it for scan job-churn; irrelevant to the per-params target. donate: outputs only, never the chunk inputs. | HIGH |

### Targeted answers to the specific hunts in the mandate

**Does jit/vmap of get_BG over B work, or do `array_with_padding` / `lax.cond` reion / the kappa solve block it?**
- `array_with_padding` does **not** block vmap — fixed-shape output. HyRex's one
  data-dependent loop is a `checkpointed while_loop` with a *static* `max_steps`
  (hydrogen.py ~:266 per notes_solve appendix), which vmaps as a masked-while (worst lane);
  its diffrax stages use fixed `SaveAt(ts=...)` → vmap-clean.
- The **`lax.cond` reion branch** (main.py:439-444) keys on `self.specs["input_tau_reion"]`,
  a **static** bool — a Python-time branch, NOT a per-element data branch; it does not block
  vmap. BUT the chosen `ReionizationModelFromTau` runs **`optx.root_find` (Newton)**
  (background.py:1014-1017) — a data-dependent iterative solve. It vmaps but locksteps and is
  a real per-cosmology cost hidden inside setup.
- The **kappa diffrax solve** (background.py:708) returns a `diffrax.Solution` — this is why
  `Background` doesn't stack (win B). It does NOT block vmapping the *setup build*; it blocks
  stacking the *resulting BG objects*. Win B fixes the stacking; orthogonal to vmapping setup.
- **Verdict:** setup CAN be vmapped over B on CPU in principle (notes are right), but it is
  the **#2-hardest** change, not a quick "wrap in jit." Per the new data it is the **#1
  highest-value** change because setup is 7 s/params flat.

**Does the chunked perturb path recompile when B changes or when the last chunk is ragged? Cost?**
- **Yes to both.** `_evolve_chunk` is `@eqx.filter_jit` (perturbations.py:125) keyed on the
  shapes of `k_chunk` and every B-leaf. Different B → recompile (all leaves carry B). Ragged
  final chunk (e.g. 492 = 4×100+92, or 571 = 5×100+71) → a 2nd kernel variant
  (perturbations.py:131-134 docstring). Warm/compile rows in profile_stages.log show **total
  compile 96 s (B=1), 128 s (B=8), 224 s (B=16)** — compile grows with B and is dozens to
  >200 s, amortized per process. Fixes: pad ragged chunk to 100 and slice (kills 1 of 2
  compiles); pad B to a fixed size for scans (avoids per-batch recompiles); persistent cache (H).

**Memory wall: at what B does one A100 OOM?**
- Per design_memo §1 (quoted in notes_solve §B): saved-ys `N_k×B×500×Ny×8` ≈ 10.52 GB at
  B=64 (Ny=72); transient Kvaerno5 Jacobian ≈ 4.5 GB; XLA overhead ~6–8 GB; **measured peak
  28–31 GB / 40 GB at B=64** with XLA rematerialization warnings (flipped_run.log:35). So
  **B≈64 is the single-A100 ceiling at K_CHUNK=100**; one-big-vmap (C) at B≥32 would breach
  it. This caps single-GPU amortization right where it starts paying — confirming D
  (sharding) is needed to push B higher, but only after setup is off the critical path.

**Is spec_Cl really O(B) eager, or does XLA cache the get_Cl compile so only iter 0 is slow?**
- **It is NOT a per-iteration-recompile problem and NOT dominated by dispatch.** `get_Cl` is
  **not** `@eqx.filter_jit` (spectrum.py:574 plain method), called in the python loop
  (spectrum.py:642-645). Each call traces/dispatches, but XLA caches the compiled graph after
  element 0 (same shapes), so iters 1..B are cheap-dispatch + real compute. The **measured
  flat ~3.2 s/params** (B=1 has no loop yet costs 3.18) proves the cost is the **`Cl_one_ell`
  arithmetic**: CubicSpline transfer interp (spectrum.py:709-723) onto N_k_transfer≈2500, the
  LoS `lax.scan` over 499 lna for each lensing ell, plus `lensed_Cls` Wigner sums. **vmap-over-B
  therefore cannot beat the loop by 10×** — it removes only the (small) per-element dispatch and
  lets XLA share bessel constants/fuse. **This is the single most important correction to the
  prior notes:** win B is real but its single-GPU payoff is ~1.3–1.5× on the spectrum stage,
  not 10–17×. Its true value is enabling B-sharding of spec_Cl across GPUs.

**donate_argnums / persistent cache correctness:**
- `eqx.filter_jit(..., donate=...)` donates by argument; donating chunk-loop inputs
  (`lna_batch`/`BG_batch`/`params_batch`) is **incorrect** — reused for every chunk
  (perturbations.py:186-189); donation frees them after chunk 0 and corrupts chunk 1+. Only
  the fresh output is safe. Persistent cache is HLO-keyed: safe for identical code; an edit
  that perturbs XLA scheduling (the exact effect documented in test_snapshots.py:67-70)
  correctly misses; the stale-hit direction is guarded by the HLO key. Safe if you don't
  hand-roll a key.

---

## PART 2 — ALGORITHMIC / PRECISION wins, ranked by (speedup × prob-success ÷ accuracy-risk)

Ranking is **end-to-end** against the measured 12.5 s/params at B=16, weighted by how much of
the budget the lever touches (setup 56%, spec_Cl 28%, perturb 16%).

| Rank | Lever | Stage touched (share) | Raw stage speedup | Accuracy risk | End-to-end effect | Why this rank |
|------|-------|----------------------|-------------------|---------------|-------------------|---------------|
| **1** | **Non-uniform `saveat`, 500→~300 dense at recomb** (perturbations.py:100,179-181) | perturb (16%) **and** spec_Cl (28%) — the scan length (spectrum.py:826) | perturb ~1.15×, spec_Cl ~1.3× | **MED** — must keep recomb resolution (visibility spike); run accuracy_test.py + regen snapshots | **~1.1–1.15× end-to-end + OOM relief → higher B.** The *only* algorithmic lever hitting two stages at once, and it directly shrinks the 28% spec_Cl term. Cheap to bracket. | Hits the largest *reducible* compute (the LoS scan) and the modes memory simultaneously. |
| **2** | **kappa→`kappa_tab` + vmap spectrum** (background.py:491,552,741; spectrum.py:616-659) — win B | spec_Cl (28%) | ~1.3–1.5× (dispatch + constant-sharing; NOT the LoS arithmetic) | **LOW** (mirrors existing `tau`/`tau_tab`, background.py:347) — one subtlety: `grad(BG.visibility)` (spectrum.py:693) becomes piecewise-const under `fast_interp`; the 10000-pt κ grid ≫ the 500-pt query, so invisible to the gate; CubicSpline fallback already imported (spectrum.py:9) | **~1.1–1.2× single-GPU; real prize is enabling B-sharding of the spectrum (win D).** | Low risk, removes `strip_bg_kappa`/`BG_list` cruft, prerequisite for sharding spec_Cl. Overbilled in prior notes — rank by *enablement*, not raw speed. |
| **3** | **float32 perturbation state `y` only** (perturbations.py:414,436) — win E | perturb (16%) | ~1.4–1.8× | **HIGH** — tight-coupling cancellation, metric sums, transfer²; gate hard | **~1.05–1.08× end-to-end.** | Real FP upside, tiny budget share now + highest gate risk. Defer until perturb dominates. |
| **4** | **Loosen `atol_large_k_PE` 1e-6→3e-6** in the stiff band (model_specs.py:66) | perturb (16%) | ~1.1–1.3× (lockstep paces to tightest lane) | **MED** — sweep {1e-6,3e-6,1e-5}, max-rel Cl vs gate | **~1.03–1.05× end-to-end.** | Cheap, small share; lockstep tax already 1.12 (notes_strategy §3), thin headroom. |
| **5** | **kappa ODE rtol 1e-10→1e-7** (background.py:706) — win G | setup (56%) but a *microscopic* sub-part | n/a (one scalar ODE) | LOW (1e-7 ≫ the 1e-4 PE rtol Cl floor) | **~1.00–1.01× end-to-end.** | Touches the big stage but the wrong part of it; saves tens of ms. Safe but pointless. |
| **6** | **Fewer lensing ells / fewer N_k** | spec_Cl / perturb | low | MED-HIGH | **~1.0×** | N_k is CLASS-adaptive (model_specs.py:120-175); raw ells already sparse + splined (spectrum.py:594-601). Little redundancy without re-tuning knots. Lowest ROI. |

### Precision notes with citations

- **float32 precedent in CMB/Boltzmann:** CAMB/CLASS run the Einstein-Boltzmann hierarchy in
  **double precision** by design; the tight-coupling expansion exists precisely because the
  θ_g−θ_b difference loses precision even in fp64. ABCMB integrates the *full* hierarchy with
  that coupling term explicit (perturbations.py:497), so fp32 there is *more* exposed than
  CLASS, not less. There is no established fp32 Boltzmann-hierarchy precedent to lean on
  (emulator-side fp32 like CosmoPower nets is a different computation). **Treat fp32 as a
  measured spike, not a default.**
- **kappa rtol=1e-10 vs the 1% gate:** the gate is set by `rtol_large_k_PE=1e-4`
  (model_specs.py:64), already ~1e-5 downstream Cl drift (CHANGELOG Phase E). κ feeds
  `visibility`/`expmkappa` (background.py:741,768) → the source terms — but 1e-7 is still 3
  orders tighter than the PE floor, so loosening to 1e-7 is numerically safe. It just doesn't
  save meaningful wall-clock.
- **saveat 500→200/300:** trajectories are smooth in lna except across the recomb visibility
  spike. A uniform cut risks under-resolving recomb; a recomb-dense non-uniform grid is the
  safe form. The one place a real accuracy/speed tradeoff lives, and worth the gate run
  because it hits two stages.
- **max_steps_PE=2048 vs observed max=1595:** measured worst-(k,B) step count is 1595
  (flipped_summary.txt; baseline max=1579), so 2048 is a ~1.3× safety margin. Lowering the
  *cap* does **not** speed steady-state (adaptive solves stop when converged, not at the cap)
  — it only risks silent truncation if any scan cosmology needs >cap. **Leave max_steps
  alone**; it is a correctness guard, not a perf knob.

---

## PART 3 — The bets and the traps

### BET ON (in this order)

1. **Batch/vmap the CPU setup over B (win A's real form = win #2 / E-floor-1).**
   This is the **#1 measured cost** (7.0 s/params flat, 56% at B=16) and every prior note
   *underestimated it 7×*. Collapse the B× `_build_one_bg` python loop (main.py:212-216), the
   B× `device_put`s (main.py:254-260), and the B× `eqx.filter_jit(self.RecModel, ...)` re-wrap
   (main.py:256, hoist it!) into one vmapped CPU pass. Even modest dispatch/transfer
   elimination here dwarfs any perturb tuning. **Highest value; MED confidence on magnitude;
   MED effort.**

2. **kappa→kappa_tab + vmap the spectrum (win B / #2).**
   Low-risk, removes `strip_bg_kappa`/`BG_list` cruft, and — critically — is the
   *prerequisite* for sharding the 28% spec_Cl term across GPUs. Bet for the **~1.2×
   single-GPU + sharding-enablement**, not the overbilled 10×. **LOW risk, MED-HIGH confidence.**

3. **Multi-GPU shard over B (win D) — ONLY after #1 and #2.**
   The 0.93 s Phase-B number is PE-only; the honest end-to-end win materializes only when setup
   and spectrum are *also* batched/shardable. With #1+#2 done, 4-GPU sharding of
   setup+perturb+spectrum is the path from ~3 s to ~1 s/param. **The single route to the target.**

### TRAPS (do NOT spend GPU-hours here yet)

1. **float32 hierarchy (win E).** Highest accuracy risk (tight-coupling cancellation,
   transfer², the load-bearing `_to_float` cast) for a lever touching only 16% of the budget —
   best case ~1.06× end-to-end. Large blast radius (GPU/CPU dtype boundary, int padding leaves,
   custom-vjp). Trap until perturb is the floor.

2. **Lowering kappa ODE rtol (win G) and lowering max_steps_PE.** G saves tens of ms in a 7 s
   stage (~1.01× end-to-end); max_steps is a correctness guard, not a perf knob (adaptive
   solves don't run to the cap). Both are time sinks disguised as wins.

3. **k_chunk tuning / one-big-vmap (win C) *as a current priority*.** perturb is already only
   16% and amortizing beautifully (1.97 s at B=16); chunk tuning is ~1.0–1.05× end-to-end now
   and risks OOM at B≥32. Worthwhile *only after* setup+spectrum are fixed and perturb
   re-emerges as the floor. Don't sequence it first.

### One-line verdict

The prior notes optimized for the wrong floor. The fresh per-stage measurement
(`profile_stages.log`) shows **setup is 7 s/params flat (56%)** and **spec_Cl is 3.2 s/params
flat (28%)** while perturb has collapsed to 16%. **Bet GPU-hours on batching the CPU setup
first, then tabulating kappa to vmap the spectrum, then sharding all three stages across 4
GPUs.** The headline solver levers everyone reaches for (k-chunking, float32, rtol/saveat,
max_steps) are small-share, gate-risky, or both — and none of them moves the two flat terms
that actually own the budget.

---

## Appendix: load-bearing code facts (verified by direct read)

- `profile_stages.log` (today, post-compile, per-stage fences): setup 7.09/6.87/6.96 s at
  B=1/8/16; spec_Cl 3.18/3.18/3.46; perturb 12.16/2.56/1.97; spec_Pk ~0.02. **Freshest and
  most authoritative split.** Config: `lensing=False`, N_k=492.
- `sweep_kchunk.log`: building 32 BGs took 244.6 s ≈ 7.6 s/cosmo — corroborates the setup floor.
- `probe_setup.py` header claims "setup ~1.85 s/param" — an earlier/different probe,
  contradicted by the call_batched fence (7 s). Trust profile_stages.log.
- `get_Cl` is a **plain method**, not `@eqx.filter_jit` (spectrum.py:574); python-loop batched
  at spectrum.py:642-645. XLA caches its graph across iters → cost is arithmetic, not dispatch.
- kappa_func is a `diffrax.Solution` (background.py:491 field, :552 set via
  `_tabulate_optical_depth`, :741 the sole `.evaluate`); `_tabulate_optical_depth` uses
  rtol=atol=1e-10, `SaveAt(dense=True)` (background.py:706,717). Fix mirrors `tau`/`tau_tab`
  `tools.fast_interp` (background.py:347).
- `_evolve_chunk` `@eqx.filter_jit`, cache keyed on `k_chunk.shape` + B-leaf shapes
  (perturbations.py:125-134); ragged final chunk → 2nd compile; B-change → recompile.
- Setup chain: python `for params in params_list` (main.py:212) → `_build_one_bg`
  (main.py:240-265) → `get_BG_pre_recomb` (already jitted, :317) → 2× `jax.device_put`
  (:254-260) → `eqx.filter_jit(self.RecModel, backend='cpu')` **re-wrapped inside the loop**
  (:256) → `get_BG` with `lax.cond` on static `input_tau_reion` (:439) →
  `ReionizationModelFromTau` `optx.root_find` Newton (background.py:1014-1017).
- Memory at B=64: peak 28–31 GB/40 GB, XLA rematerialization (flipped_run.log:35; design_memo
  §1). B≈64 is the single-A100 ceiling at K_CHUNK=100.
- Accuracy gate: `pytests/accuracy_test.py` asserts max-rel TT/EE/Pk ≤ 0.01 vs CLASS
  (lines 135-137); `pytests/test_snapshots.py` rtol=1e-8/atol=1e-18 (line 71) — will change
  bit-for-bit under win B (interp vs dense-eval), so **regenerate snapshots after B**.
- Step counts: baseline min=41/median=380/max=1579 (max/median=4.16); flipped worst-k
  max/median over B=1.12 (baseline_summary.txt, flipped_summary.txt). max_steps_PE=2048
  (model_specs.py:60) is a guard, not a knob.

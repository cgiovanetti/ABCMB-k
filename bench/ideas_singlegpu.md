# ideas_singlegpu.md — Single-A100 plan to drive `Model.call_batched` toward ~1 s/param

**Author:** JAX perf-engineering pass, 2026-05-29 (branch `perk-refactor`). **Static analysis + design only; no code run.**
**Scope:** ONE A100, no multi-GPU. Build on `bench/notes_solve.md`, `bench/notes_spectrum.md`, `bench/notes_strategy.md`; do not repeat them. This note is anchored to the NEW per-stage `block_until_ready` data in `bench/profile_stages.log`, which the three prior notes did not have — and which materially changes the ranking, because it shows the setup stage is **7 s of eager-execution overhead**, an order of magnitude larger than the prior notes' ~0.5–1.0 s/cosmo estimate (notes_strategy §1a; notes_solve §E).

---

## 0. The measured ledger we are designing against (from `bench/profile_stages.log`)

Post-compile, per-stage, per-param (s/param = stage_total / B):

| B  | total/p | setup/p | stack/p | perturb/p | spec_Cl/p | spec_Pk/p |
|---:|--------:|--------:|--------:|----------:|----------:|----------:|
| 1  | 22.46   | **7.094** | 0.007 | 12.161 | 3.180 | 0.017 |
| 8  | 12.624  | **6.866** | 0.006 | 2.558  | 3.181 | 0.013 |
| 16 | 12.412  | **6.962** | 0.005 | 1.968  | 3.461 | 0.014 |

Single-cosmology fused `model()` (jitted) post-compile: **~9.5 s/param** (CHANGELOG/baseline; perf_batched warm 9.51).

**Three facts the per-stage data nails down that the prior notes could only estimate:**

1. **`setup` is a flat ~7.0 s/param that does NOT amortize with B** (7.094 → 6.866 → 6.962). This is the dominant term at every B≥8. It is pure **eager (un-jitted) execution** of `get_BG` (and the heavy work inside `Background.__init__`), run B times sequentially in the python `for` loop at `main.py:212-216`. In the single-cosmology path the identical work is *fused inside `_run_post_recomb`'s `@eqx.filter_jit`* (main.py:345, which calls `get_PTBG` → `get_BG` at main.py:406, all under one jit) so it costs nearly nothing. **This 7 s is the headline target, and it is NOT a compute floor — it is dispatch/eager overhead.** notes_solve §E sized this stage at ~1.0 s/cosmo from the single-call residual; the measured 7 s shows eager execution of the same code is ~7× slower than the jitted version, because every `jnp`/`vmap`/`diffeqsolve` op in `Background.__init__` (background.py:529-564) launches as its own un-fused kernel with full python+dispatch latency.

2. **`perturb` amortizes well and is already near its single-GPU floor:** 12.16 (B=1) → 2.56 (B=8) → 1.97 (B=16) s/param. At B=1 it is *worse* than the single path's ~8 s (double-vmap + per-chunk dispatch overhead at tiny B, as the task states); by B=16 it is ~2 s/param and falling. This matches flipped_summary's PE-only asymptote (~3.4 s at B=64 *with* the worst-k tax; the call_batched numbers are lower here because ELLMAX/saveat differ — treat the *trend* as load-bearing, not the absolute).

3. **`spec_Cl` is a flat ~3.2 s/param that does NOT amortize** (3.18 → 3.18 → 3.46): exactly the python-loop-over-B signature (`spectrum.py:642`), confirming notes_spectrum §10. `spec_Pk` is negligible (~0.014 s/param).

**Decomposition of where the ~12.4 s/param at B=16 goes:** setup 6.96 (56%) + perturb 1.97 (16%) + spec_Cl 3.46 (28%) + spec_Pk 0.01 + stack 0.005. **The two flat stages (setup + spec_Cl) are 10.4 of 12.4 s/param — 84% — and neither amortizes.** They are the whole game. The prior notes correctly identified spec_Cl (#1 in both) but **under-weighted setup by ~7×**; the new data flips setup to co-equal with (in fact slightly larger than) the spectrum loop.

---

## 1. Kill the eager-setup 7 s/param  — THE highest-value single-GPU change

### 1.1 Diagnosis: why setup is 7 s and why it is flat in B

`call_batched` (main.py:209-216) runs, for each of B cosmologies, **eagerly**:
```python
for params in params_list:
    full_params = self.add_derived_parameters(params)   # main.py:213  CPU python, ~ms
    full_p, bg  = self._build_one_bg(full_params)       # main.py:214  THE 7 s
```
`_build_one_bg` (main.py:240-265) does:
- `get_BG_pre_recomb(full_params)` — main.py:252. This IS `@eqx.filter_jit` (main.py:317), so it is *one* compiled call. Cheap.
- two `jax.device_put` GPU↔CPU round-trips + `eqx.filter_jit(self.RecModel, backend='cpu')(...)` — main.py:254-257. HyRex on CPU, one jitted call. Moderate.
- `get_BG(full_params, pre_BG, recomb_output)` — main.py:264. **This is the killer.** `get_BG` (main.py:410-446) is **NOT `@eqx.filter_jit`** (the docstring at main.py:415 says so explicitly). It runs `lax.cond` → `Background(...)` **eagerly**. `Background.__init__` (background.py:501-564) does, *op-by-op, un-fused, in python*:
  - `ReionModel(self, params)` + `xe_reion` correction + `array_with_padding` + 4× `_finite_pad` `jnp.where` (background.py:529-550),
  - `self._tabulate_optical_depth(params)` — background.py:552 — a **full `diffrax.diffeqsolve` (Kvaerno5, rtol=atol=1e-10, max_steps=2048, dense)** over lna∈[0,-10] (notes_spectrum §3.2). One adaptive implicit ODE solve, eager.
  - `vmap(self.visibility)` over **1500** lna points (background.py:555-556) — each `visibility` call evaluates the dense `kappa_func.evaluate` (the diffrax Solution) — a dense interp eval ×1500, eager.
  - `vmap(self.aH) * self.tau_c` over **5000** lna points (background.py:562-563), eager.
  - `_tabulate_kappa_d` / `tau` / argmin/argmax reductions, eager.

  Eager means: no XLA fusion, every primitive is a separate dispatched kernel with python + launch latency, and the GPU sits mostly idle between micro-kernels. That is why ~1 s of *jitted* work (the single-path residual) becomes ~7 s *eager*, and why it is flat in B (each element pays the full eager cost independently; there is no batch dimension to amortize dispatch over).

### 1.2 Fix: run the whole BG construction under ONE jit, vmapped over B

Wrap `get_BG_pre_recomb` + HyRex-bridge + `get_BG` into a single `@eqx.filter_jit` callable and `jax.vmap` it over the B param dicts. HyRex must stay CPU; the rest is GPU. Two-tier structure:

```python
# main.py — new methods on Model

@eqx.filter_jit                       # GPU; vmapped over B
def _pre_recomb_batched(self, params_batch):
    # vmap the EXISTING get_BG_pre_recomb body over the leading B axis.
    return jax.vmap(lambda p: BackgroundPreRecomb(
        p, self.species_list, self.RecModel, adjoint=self.adjoint))(params_batch)

def _hyrex_batched(self, recomb_inputs_batch, params_batch):
    # CPU, vmapped. ONE compiled CPU kernel for all B (see 1.3 on the while_loop).
    f = eqx.filter_jit(jax.vmap(self.RecModel), backend='cpu')
    return f((recomb_inputs_batch, params_batch))

@eqx.filter_jit                       # GPU; vmapped over B
def _build_bg_batched(self, params_batch, pre_BG_batch, recomb_batch):
    # vmap the EXISTING get_BG (lax.cond reion branch is fine under vmap — see 1.4)
    return jax.vmap(self.get_BG, in_axes=(0, 0, 0))(
        params_batch, pre_BG_batch, recomb_batch)
```

`call_batched` setup block becomes (replacing main.py:209-221):
```python
full_ps      = [self.add_derived_parameters(p) for p in params_list]   # cheap python
params_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps)
params_batch = jax.tree.map(_to_float, params_batch)                    # once, batched

pre_BG_batch = self._pre_recomb_batched(params_batch)                   # 1 GPU jit, vmap B
ri_cpu  = jax.device_put(pre_BG_batch.recomb_inputs, cpu_dev)           # ONE transfer
p_cpu   = jax.device_put(params_batch, cpu_dev)                         # ONE transfer
recomb_batch = self._hyrex_batched(ri_cpu, p_cpu)                       # 1 CPU jit, vmap B
recomb_batch = jax.device_put(recomb_batch, gpu_dev)                    # ONE transfer
recomb_batch = jax.tree.map(_to_float, recomb_batch)
BG_batch = self._build_bg_batched(params_batch, pre_BG_batch, recomb_batch)  # 1 GPU jit, vmap B
```
Now there are **3 jitted calls + 3 device transfers total**, not 3B calls + 2B transfers. `add_derived_parameters` stays a python loop (it is genuinely ~ms and does table/LINX/Neff branching that is hostile to vmap; keep it eager — see 1.5).

### 1.3 What vmaps cleanly vs needs care

| Component (line) | Under `vmap`-over-B? | Notes |
|---|---|---|
| `BackgroundPreRecomb.__init__` (main.py:343) | **Clean.** | Already `@eqx.filter_jit` for B=1; vmap just adds a batch axis. Pure tabulation of conformal time + RecombInputs packing. |
| `_tabulate_optical_depth` diffrax solve (background.py:552,682) | **Clean but lockstep.** | Adaptive Kvaerno5 vmapped over B → lockstep stepping at the stiffest cosmology's step count (notes_strategy §3). Spread across Planck-σ cosmologies is tiny (~1.12 worst-case, by analogy to the PE flip), so the lockstep tax is ~10%, not B×. Net: ONE solve's wall-time for all B instead of B solves. **This alone is most of the 7→~0.5 s win.** |
| `vmap(visibility)` over 1500 pts (background.py:555-556) | **Clean.** Becomes `(B,1500)`. | After the §2 kappa-tabulation fix this is a `fast_interp`, even cheaper. |
| `vmap(aH)*tau_c` over 5000 pts (background.py:562-563) | **Clean.** Becomes `(B,5000)`. | Pure elementwise + reductions; fuses beautifully under jit. |
| `lax.cond` reion branch (main.py:439) | **Needs care, but OK.** | `self.specs["input_tau_reion"]` is a **python static bool** closed over `self`, not data → it is the SAME branch for all B → `lax.cond` collapses to a static branch and vmaps without divergence. Safe as-is. (If it were per-element data it would force both branches; it is not.) |
| HyRex `array_with_padding` (hyrex) | **Needs care.** | `padding_size`/`lastnum` are int leaves with **static shapes** (notes_solve §E). Under vmap they gain a batch axis but the SHAPE stays static (the cap), so `array_with_padding` survives. The `_finite_pad` (background.py:544-550) already replaces inf with `lastval` → no NaN under batched AD. |
| HyRex `eqx.internal.while_loop(kind="checkpointed", max_steps=...)` (hydrogen.py:266) | **Needs care.** Masked-while. | Vmapped checkpointed while runs to the worst lane's iterations, capped at the existing static `max_steps`. Math identical; cost = worst-lane on CPU. notes_solve §E confirms vmap-ability. |
| `add_derived_parameters` (main.py:448-692) | **Do NOT vmap.** | Python control flow + table interp + optional LINX + Neff/YHe triangle + `for s in species` loops. Keep eager python loop (it is ~ms/cosmo). |

### 1.4 Risk on the `lax.cond` + the eager BG argmax/argmin

`Background.__init__` finds `lna_rec`/`lna_visibility_stop`/`lna_transfer_start` via `argmax`/`argmin` over fixed grids (background.py:557-564). Under vmap these become per-lane argmax/argmin — **fully supported**, returns `(B,)` scalars. No control-flow hazard. The only subtlety is that these were computed *eagerly at construction* before; moving them under jit means they are traced once and reused — strictly better.

### 1.5 Projected setup/param after the fix

The 7 s eager cost decomposes (estimated, from the op inventory in 1.1) as roughly: optical-depth diffrax solve ~3 s eager, the two big `vmap`s (1500+5000 pts) eagerly dispatched ~2 s, HyRex bridge + transfers ~1.5 s, reion/padding ~0.5 s. Under one jit + vmap:
- the diffrax solve becomes ONE lockstep solve for all B (≈ its single eager-fused cost ÷ ~nothing, +10% lockstep): **~0.2–0.4 s total**, i.e. **~0.01–0.05 s/param at B≥8**;
- the 1500/5000-pt vmaps fuse and run on `(B,·)` arrays: **~0.1–0.3 s total**;
- HyRex CPU vmapped: the dominant residual; CPU while-loop doesn't SIMD across lanes well, but collapses B× dispatch + B× transfers into 1×. notes_solve §E estimates ~0.2–0.4 s/param; with the transfers batched, **~0.3–0.6 s/param** is the honest CPU floor here.

**Projected setup: ~7.0 → ~0.4–0.7 s/param at B≥8**, dominated by the (still-sequential-per-lane) HyRex CPU while-loop. That is a **10–17× reduction on the largest stage.** Even if HyRex vmap underperforms (CPU SIMD is weak), the eager→jit conversion of the GPU portion (`get_BG`) alone removes ~4–5 s/param of pure dispatch overhead.

**Confidence: HIGH that get_BG-under-jit removes most of it** (it is literally the single-path's free behavior, just not wired into call_batched). **MEDIUM on the absolute HyRex residual** (needs measuring). This is the change with the largest, most certain payoff.

---

## 2. Vmap the spectrum (kappa tabulation) — second co-equal win

This is `notes_spectrum.md` §4–§5 in full; I do not repeat the line-by-line edit. Summary of the design and the projection against the NEW data:

- **Root cause:** `Background.kappa_func` is a `diffrax.Solution` (background.py:491, set at :552), read via `kappa_func.evaluate(lna)` in `expmkappa` (background.py:738-742). A `Solution` does not survive `jax.tree.map(jnp.stack, ...)`, so `get_Cl_batched`/`Pk_lin_batched` (spectrum.py:616-659) are forced into a **python loop over B** (spectrum.py:642, :655) — the flat 3.2 s/param.
- **Fix (background.py only):** replace `kappa_func` with an `expmkappa_tab` array tabulated on the shared class-level `lna_tau_tab` grid (background.py:59, 10000 pts), built once in `__init__` by consuming the diffrax `Solution` as a transient local, then discarded; rewrite `expmkappa` to `tools.fast_interp` — **exactly mirroring `tau`/`tau_tab` (background.py:347).** Then `Background` is a pure-array PyTree and stacks cleanly.
- **Payoff (spectrum.py):** `get_Cl_batched`/`Pk_lin_batched` collapse from python loops to a single `@eqx.filter_jit` `jax.vmap(self.get_Cl, in_axes=(0,0,0))` over the stacked `BG_batch`. `Pk_lin` reads no BG so it is trivially vmappable already.
- **Bonus this note adds:** the §2 fix is ALSO a precondition for §1's `_build_bg_batched` to return a *stackable* `BG_batch`. Today even if you jit `get_BG`, the returned `Background` carries the `kappa_func` Solution and cannot be vmapped/stacked — which is why `strip_bg_kappa` (perturbations.py:538) exists. **§2 must land before §1's batched-BG can be one stacked PyTree.** So the true dependency order is: §2 (kappa array) → §1 (vmap get_BG) → §3 (perturb). Do §2 first; it unblocks both the spectrum AND the setup batching, and lets you delete `strip_bg_kappa` entirely.

### 2.1 Projected spec_Cl/param after the fix

Today: flat 3.2 s/param (B-length python loop, each `get_Cl` paying full JIT dispatch + the dense `kappa_func.evaluate` ×~500 lna ×N_ell). After: ONE batched-LoS jit, the inner `vmap(Cl_one_ell)` over N_ell nested inside `vmap` over B; bessel tables + k-axis are shared constants. notes_spectrum §10 projects ~10× at B=32, ~15–17× at B=64. Against the measured 3.2 s/param flat, a conservative single-A100 projection: **spec_Cl ~3.2 → ~0.3–0.5 s/param at B≥16** (the single-call get_Cl is ~0.68 s in the Phase-A residual; batched-vmap amortizes its dispatch so per-param falls below that). **spec_Pk stays ~0.01.**
**Risk: LOW** (interp replaces dense-eval at the same tolerance; one autodiff subtlety — `grad(visibility)` becomes piecewise-constant, mitigated by the 10000-pt grid being 20× finer than the 500-pt PT.lna integration axis; CubicSpline fallback if the 1% gate moves — notes_spectrum §6.2). **Confidence: HIGH.**

---

## 3. Perturb amortization + k_chunk sizing

### 3.1 Model perturb/param as f(B, k_chunk) from the data

Measured perturb/param (s): B=1 → 12.16, B=8 → 2.56, B=16 → 1.97 (all at the **default k_chunk_size=100**, N_k=492, 5 chunks: 4×100 + 1×92). Fit the shape `T_perturb(B)/B ≈ a + b/B`:
- Using B=8 (2.56) and B=16 (1.97): solving 2.56 = a + b/8, 1.97 = a + b/16 → b/16 = 0.59 → b ≈ 9.4, a ≈ 1.38. So **per-param asymptotes to ~1.4 s** (the irreducible batched-solve cost per cosmology at the configured tolerances) with a ~9.4 s "fixed per-call overhead" term spread over B. B=1's 12.16 is dominated by that fixed term + double-vmap/per-chunk dispatch at B=1 (consistent with a ≈1.4 plus ~10.8 overhead — the task's "WORSE than single at B=1").
- Extrapolating: B=32 → ~1.4 + 9.4/32 ≈ **1.7 s/param**; B=64 → ~1.4 + 9.4/64 ≈ **1.55 s/param** (memory permitting — see 3.3). The asymptote ~1.4 s is the single-GPU perturb floor at default tolerances; squeezing it further needs the notes_solve §A levers (stiffness-homogeneous chunking 1.3–1.7×, float32 1.4–1.8×, atol loosening) — all second-wave, accuracy-gated.

### 3.2 Is one big vmap (k_chunk = N_k) better than chunk=100 at small B?

**Yes at B≤8, per notes_solve §B1, and the new data supports it:** the "fixed per-call overhead" term b≈9.4 s is exactly the kind of cost that 5 serial `_evolve_chunk` kernel launches (4 extra launch barriers + python dispatch + the ragged-92 second compile) inflate. At small B the device is underutilized *within* a 100-mode chunk, so serializing 5 chunks wastes more than it saves. Setting `k_chunk_size = N_k = 492` runs ONE fused k×B vmap, killing 4 launch barriers and the ragged-chunk recompile. Expected: shaves a meaningful slice of the b/B term at B≤8 (notes_solve estimates modes 1.1–1.3×). **Risk: NONE** (same math, bigger vmap). **Only constraint is memory (3.3).**

### 3.3 Memory: pick k_chunk and B within 40 GB

Saved-trajectory tensor = `N_k × B × N_lna × N_y × 8 bytes`. Using the task's figure (N_lna=500, N_y=72 — the design_memo padding-inclusive count; notes_solve §"State-vector size" flags 72 vs the 46 pure-ΛCDM count, scales linearly either way) and N_k=492:

| | per-param saved-ys | one-big-vmap (k_chunk=N_k) saved-ys | chunk=100 transient saved-ys |
|---|---|---|---|
| B=8  | 492×500×72×8 = 0.142 GB | ×8 = **1.13 GB** | ×100/492 of that ≈ 0.23 GB live + concat 1.13 |
| B=16 | | ×16 = **2.27 GB** | |
| B=32 | | ×32 = **4.53 GB** | |
| B=64 | | ×64 = **9.07 GB** | (matches design_memo ~10.5 GB saved-ys) |

Saved-ys is only part of peak. The **transient Kvaerno5 Jacobian/LU workspace** is the other big chunk (design_memo §1.2 ≈ 4.5 GB at B=64 *chunked to 100*; it scales with the live vmap width = k_chunk×B). For a single big vmap (k_chunk=N_k=492) the transient is 492/100 ≈ 4.9× the chunk=100 transient → at B=64 that is ~22 GB transient + 9 GB saved-ys ≈ 31 GB — which is exactly the measured 28–31 GB near-OOM (notes_strategy §2). So:

**Recommendation (single A100, 40 GB):**
- **B ≤ 16:** `k_chunk_size = N_k` (one vmap). Saved-ys ≤ 2.3 GB; transient ~4.9× the chunk-100 figure but at B=16 that is small (~2–3 GB). Total well under 40 GB. **Fastest, simplest.**
- **B = 32:** `k_chunk_size ≈ 200` (3 chunks), pad ragged to a uniform shape. Keeps transient ~2× chunk-100 (~tens of % of 40 GB). Saved-ys 4.5 GB.
- **B = 64:** keep `k_chunk_size = 100` (mandatory — the only config proven to fit at 28–31 GB), pad the ragged final chunk to kill the 2nd compile (notes_solve §B2).

**Sweet spot for ~1 s/param ambition:** the per-param perturb floor (~1.4 s) is reached by B≈16–32; pushing B higher buys little on perturb and risks OOM. **B=16–32 with k_chunk = N_k (B=16) or 200 (B=32) is the perturb optimum on one A100.** Going to B=64 is a memory/throughput tradeoff that does NOT lower per-param meaningfully (1.55 vs 1.7) and forces small chunks.

---

## 4. End-to-end budget with all three fixed (single A100)

Projected per-param, post-compile, by stage (using the measured flat/amortizing behavior + the §1/§2/§3 projections):

| stage | now (B=16) | after §1 (jit+vmap setup) | after §2 (vmap spectrum) | after §3 (k_chunk=N_k) |
|---|---:|---:|---:|---:|
| setup   | 6.96 | **0.4–0.7** | 0.4–0.7 | 0.4–0.7 |
| stack   | 0.005 | 0.005 | ~0 (BG_batch reused, no strip) | ~0 |
| perturb | 1.97 | 1.97 | 1.97 | **~1.6–1.9** |
| spec_Cl | 3.46 | 3.46 | **0.3–0.5** | 0.3–0.5 |
| spec_Pk | 0.014 | 0.014 | 0.014 | 0.014 |
| **TOTAL/param** | **12.4** | ~5.8–6.1 | ~2.7–3.1 | **~2.4–3.0** |

Projected per-param at several B (all three fixes; B=16/32 use k_chunk=N_k/200, B=64 uses 100):

| B  | setup | perturb | spec_Cl | spec_Pk | **TOTAL/param** |
|---:|------:|--------:|--------:|--------:|----------------:|
| 8  | ~0.6  | ~2.4    | ~0.6    | 0.01    | **~3.6** |
| 16 | ~0.5  | ~1.8    | ~0.4    | 0.01    | **~2.7** |
| 32 | ~0.5  | ~1.7    | ~0.35   | 0.01    | **~2.6** |
| 64 | ~0.45 | ~1.6    | ~0.3    | 0.01    | **~2.4** |

### 4.1 Honest single-GPU floor

**The single-A100 floor is ~2.4–2.7 s/param, NOT ~1 s, and NOT the ~3–5 s the strategy note claimed — but for a reason the strategy note got partly wrong.**

- notes_strategy §1b/§1c claimed **~5 s/param single-GPU**, built on a setup ("CPU tail") estimate of ~0.5–1.0 s and a spectrum-vmap estimate of ~1.4 s, with perturb ~3.0–3.4 s. **The new data revises two of those three:** (a) the spectrum, once vmapped, is **smaller** than they thought (~0.3–0.5, not 1.4 — they used a single-call get_Cl figure without crediting batch amortization); (b) the setup, once jitted+vmapped, is **comparable** to their estimate (~0.5 s), because the measured 7 s was almost entirely *eager-execution overhead that jit removes* — exactly the term they did not credit (they treated the 7 s as if it were the irreducible ~1 s CPU tail, never seeing the eager penalty). (c) Their perturb ~3.0–3.4 s used the B=64 flipped-spike worst-k-tax number; the call_batched data shows perturb amortizes to ~1.6–1.9 s/param at the configured tolerances/saveat. Netting these, the **realistic single-GPU floor is ~2.4–2.7 s/param**, *better* than their ~5 s, because they over-counted both setup (treated eager penalty as irreducible) and spectrum.

- **Why ~1 s is still out of reach on ONE A100, even after these fixes:** the perturb stage asymptotes at **~1.4–1.9 s/param** (§3.1) — an implicit Kvaerno5 solve of a ~(N_y) stiff system, 492 modes × B lanes, bandwidth/latency-bound on the small per-cell LU (notes_strategy §2 (iii): device 70–78% full, rematerializing at B=64). That ~1.5 s perturb floor + ~0.5 s HyRex CPU floor ≈ **2 s is the hard single-GPU wall** absent a precision/solver change. To break below ~2 s on one A100 you MUST add a second-wave lever: float32 perturbation state (notes_solve §D, 1.4–1.8× on the solve, HIGH accuracy risk) and/or stiffness-homogeneous chunking (notes_solve §A1, 1.3–1.7×, zero accuracy risk) and/or trimmed saveat (notes_solve §C). Stacking the zero-risk ones (A1 chunking + saveat trim) could bring perturb ~1.6 → ~1.0–1.2 and total → ~1.8–2.0 s/param. **A clean ≤1 s/param on a single A100 is not reachable** without float32 (gated hard) — and even then it brushes ~1.2–1.5 s. **The ~1 s target genuinely requires multi-GPU sharding over B** (notes_strategy §1c, §6; out of this note's single-GPU scope), which restores linear scaling the single-A100 memory saturation prevents.

**Bottom line for §4:** the three single-GPU fixes take per-param from **12.4 → ~2.4–2.7 s** (a ~5× win, mostly from setup+spectrum). That is the single-GPU floor at default accuracy. Second-wave accuracy-gated levers can reach ~1.8–2.0 s. ~1 s needs multi-GPU.

---

## 5. Compilation strategy

1. **Persistent compile cache** (notes_solve §F2; nothing in-repo sets it). For a frequentist scan that restarts processes, set at import:
   ```python
   jax.config.update("jax_compilation_cache_dir", "/pscratch/sd/c/carag/ABCMB-k/.jax_cache")
   jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
   jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
   ```
   The warm/compile rows in profile_stages.log show **95.8 s (B=1) → 223.7 s (B=16)** of compile — entirely amortizable across SLURM jobs at fixed B. **Risk: NONE. Cold-start only.**

2. **Pad B to a fixed size** (notes_solve §F3). Every batched leaf bakes B into its shape, so a ragged final batch (e.g. 1000 params / B=64 → last batch of 40) **recompiles every jitted stage** (`_pre_recomb_batched`, `_build_bg_batched`, `full_evolution_batched`, `get_Cl_batched`). Pad `params_list` to a multiple of a fixed B and mask the padding in the downstream likelihood. **Risk: NONE.** Combine with the cache so a scan compiles once per B and reuses forever.

3. **Pad the ragged final k-chunk** to a uniform shape (notes_solve §B2). Today N_k=492 / chunk=100 → 4×100 + 1×92 = **two `_evolve_chunk` compiles**. Pad the 92-chunk to 100, slice the result → one compile. (Moot if you use k_chunk=N_k at B≤16, which is the recommendation anyway.)

4. **`donate_argnums`** (notes_solve §F1). On the new batched stages, the inputs are read-only across the (single) call, so donation helps the **output buffers** of `full_evolution_batched` (the big `(N_k,B,500,N_y)` tensor) and `get_Cl_batched`. Use `eqx.filter_jit(..., donate="warn")` to let XLA reuse the saved-ys buffer for the PT and the PT buffer for the Cls, easing the B=32/64 peak. **Risk: LOW** (donation of inputs you don't reuse). **Confidence: MEDIUM** on the magnitude.

5. **One static B per Model build.** Because `self.specs` and `B` together key every cache, run a scan with a fixed `Model` instance and a fixed padded B; never re-instantiate `Model` per batch (it re-traces `species_list`/specs-dependent graphs).

---

## 6. The ONE change to do first, and its expected effect

**Do §2 first: replace `Background.kappa_func` (diffrax.Solution, background.py:491,552) with a tabulated `expmkappa_tab` array consumed via `tools.fast_interp` in `expmkappa` (background.py:738-742), mirroring `tau`/`tau_tab` (background.py:347).**

Rationale — it is the keystone that unblocks BOTH flat stages:
1. **Immediate spectrum win:** turns `get_Cl_batched`/`Pk_lin_batched` from python loops into one `jax.vmap` → **spec_Cl 3.2 → ~0.4 s/param** (the 28% slice).
2. **Enables the setup fix:** once `kappa_func` is gone, `Background` is a pure-array PyTree, so `get_BG` can be jitted-and-vmapped to return ONE stacked `BG_batch` (§1) — without §2, the batched `get_BG` would still emit un-stackable Solutions and you'd be stuck with `strip_bg_kappa` and a per-element BG list. So §2 is the prerequisite for the §1 setup attack on the 56% slice.
3. It deletes `strip_bg_kappa` (perturbations.py:538) and removes the strip/un-strip duplication in `call_batched` (main.py:220-221).

**Expected end-to-end effect of §2 alone:** at B=16, **12.4 → ~9.0 s/param** (spec_Cl 3.46 → ~0.4). Then §1 (now unblocked) takes it to **~2.7 s/param**, and §3 to **~2.4–2.7 s/param**. §2 is low-risk (LOW; interp at same tolerance, one piecewise-constant `grad(visibility)` subtlety with a CubicSpline fallback), localized to `background.py` + two `spectrum.py` wrappers + one `main.py` stacking line, and is the documented Path Forward in CHANGELOG/CLAUDE.md. **It is both the highest-value first move and the structural enabler for everything else in this plan.**

Gate after §2: `cd pytests && pytest -s -vv` (1%-vs-CLASS); regenerate `test_snapshots.py` fixtures (interp ≠ dense-eval bit-for-bit, deliberate). Report max-rel TT/EE/Pk.

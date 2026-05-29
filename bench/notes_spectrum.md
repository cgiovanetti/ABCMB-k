# notes_spectrum.md — Batched spectrum bottleneck: design + implementation plan

**Author:** JAX perf-engineering pass, 2026-05-29. **Branch:** `perk-refactor`.
**Status:** All source files read verbatim and verified. All line numbers below
are exact for the current tree. No unresolved gaps.

---

## 1. One-paragraph summary

`Model.call_batched` (`main.py:187`) evaluates B cosmologies params-axis-batched,
but its final spectrum stage runs as a **Python loop over B `Background` objects**
calling the single-cosmology `SpectrumSolver.get_Cl`/`Pk_lin`
(`spectrum.py:616` `get_Cl_batched`, `spectrum.py:651` `Pk_lin_batched`; loops at
`spectrum.py:642` and `:655`). It cannot be `vmap`'d because
`Background.kappa_func` is a `diffrax` dense `Solution` (`background.py:491` field,
set at `background.py:552`), which does not survive `jax.tree.map(jnp.stack, ...)`
across cosmologies — exactly why `strip_bg_kappa` (`perturbations.py:538`) exists
to null it before the modes path stacks BGs. The fix: stop storing `kappa_func` as
a `Solution`. Pre-tabulate `exp(-kappa)` on the existing `lna_tau_tab` grid (the
template `tau`/`tau_tab` already use at `background.py:59,347`), store it as a
plain array field, and rewrite `expmkappa` (`background.py:722`) to
`tools.fast_interp`. Then `Background` is a pure-array PyTree, stacks cleanly, the
spectrum loops become `jax.vmap`, and `strip_bg_kappa` can be deleted. Projected
spectrum-stage speedup ≈ **10–17×** at B=32/64 (§10).

**On the suspected `call_batched` bug (§6):** I read `call_batched` verbatim. It
calls the WRAPPERS `get_Cl_batched`/`Pk_lin_batched` (lines 228-232), both of
which correctly slice `params_batch[i]` per element internally. **There is no
full-`params_batch`-into-`get_Cl` vs `params_i`-into-`Pk_lin` asymmetry. The
suspected bug does not exist.** Detail and the one real subtlety in §6.

---

## 2. Verified baseline numbers

From `CHANGELOG.txt` (Phase A/B) and `bench/perf_batched.log`:

| quantity | value | source |
|----------|-------|--------|
| single-cosmology `model()` per-params | 9.89 s (9.51 s post-compile) | CHANGELOG Phase A / perf_batched.log |
| `PE.full_evolution` per-params | 8.14 s (dominant) | CHANGELOG Phase A |
| N_k perturbation modes | 571 | CHANGELOG Phase A (note: code default `k_axis_perturbations=geomspace(1e-4,0.4,600)`, `perturbations.py:65`; specs override to 571) |
| per-k diffrax steps | min=41 median=380 max=1579 | CHANGELOG Phase A |
| worst-k tax (max/median) | 4.16 | CHANGELOG Phase A |
| flipped double-vmap B=64 single-GPU | 3.43 s/params (2.37×) | CHANGELOG Phase B |
| flipped B=64 4-GPU sharded | 0.93 s/params (8.80×) | CHANGELOG Phase B |

Current `Model.call_batched` (perf_batched.log, ELLMAX=800, single A100,
post-compile): B=1 → 22.64; B=4 → 14.46; B=8 → 12.95; B=16 → 12.08 s/params.
**Batched is currently SLOWER than sequential single calls.** CHANGELOG names the
cause: *"The asymptote at ~12 s/params is set by the python-loop spectrum: each
get_Cl / Pk_lin element pays full JIT-dispatch overhead."* The modes side is
already faster batched (3.43 vs 8.14); the win is hidden by the spectrum loop.

> There is no per-stage `bench/perf_baseline.md` in this checkout (only
> `baseline_summary.txt` + `perf_batched.log`). The "54 s spectrum / 0.85 s per
> params" split is a projection, not a measured artifact here. The measured fact
> is the ~12 s/params asymptote attributed to the spectrum loop.

---

## 3. How `kappa_func` is built and consumed today (VERIFIED, background.py)

### 3.1 The field (`background.py:491`)
```python
class Background(BackgroundPreRecomb):
    ...
    kappa_func : "diffrax.solution"      # line 491
```
This is the ONLY non-array, non-static leaf on `Background` that blocks stacking.
All other fields are arrays, `array_with_padding` (eqx.Module of arrays), floats,
or the `static` `adjoint` (`background.py:65`, `eqx.field(static=True)` — statics
are not stacked, so they do not block `tree.map(jnp.stack,...)`).

### 3.2 How it is created (`background.py:552`; `_tabulate_optical_depth` at 682)
```python
    self.kappa_func = self._tabulate_optical_depth(params)     # line 552
```
```python
    def _tabulate_optical_depth(self, params):                 # line 682
        integrand = lambda lna, y, args: -1./self.tau_c(lna, params)/self.aH(lna, params)
        term = ODETerm(integrand)
        stepsize_controller = PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0,
                                            rtol=1.e-10, atol=1.e-10)
        adjoint=self.adjoint()
        sol = diffeqsolve(
            term, solver=Kvaerno5(), stepsize_controller=stepsize_controller,
            t0=0., t1=-10., dt0=-1.e-3, max_steps=2048, y0=0.0,
            saveat=SaveAt(dense=True), adjoint=adjoint)
        return sol                                             # a diffrax.Solution
```
Integrated lna ∈ [0, -10] (today backwards). κ(0)=0; κ grows back in time.
`kappa_func.evaluate(lna)` returns κ for lna ∈ [-10, 0].

### 3.3 How it is consumed — `expmkappa` (`background.py:722`)
```python
    def expmkappa(self, lna):                                  # line 722
        return jnp.where(
            lna < -10.,
            0.,
            jnp.exp(-self.kappa_func.evaluate(lna))            # line 741 — the ONLY .evaluate
        )
```
0 for lna < -10 (early, fully opaque → exp(-∞)=0), else exp(-κ). **This is the
single point where `kappa_func` is read anywhere in the codebase** (verified by
grep: matches only at 453/491 declaration, 552 assignment, 741 evaluate).

### 3.4 Downstream of `expmkappa`
- `visibility` (`background.py:744`): `return self.expmkappa(lna)/self.tau_c(lna, params)`.
  Despite the docstring mentioning κ′, this implementation does NOT differentiate
  `kappa_func`. So **visibility needs no separate grid** — fixing `expmkappa`
  fixes `visibility` for free.
- `__init__` builds `lna_rec`, `lna_visibility_stop`, `rA_rec`
  (`background.py:555-559`) by vmapping `self.visibility`. These run eagerly at
  construction; they keep working because they go through `self.visibility` →
  `self.expmkappa`. **Ordering constraint:** the grid must be assigned to `self`
  BEFORE line 555. Currently line 552 precedes 555 — keep that order.

### 3.5 The `tau_tab` template to copy (`background.py:59,60,88,321,347`)
```python
    lna_tau_tab = jnp.linspace(-33.0, 0.0, 10000)   # line 59 (class attr, shared, strictly increasing)
    tau_tab : jnp.array                              # line 60 (per-cosmology array field)
    ...
    self.tau_tab = self._tabulate_conformal_time(params)   # line 88
    ...
    def tau(self, lna):                              # line 321
        return tools.fast_interp(lna, self.lna_tau_tab[0],
                                 self.lna_tau_tab[-1], self.tau_tab)   # line 347
```
`lna_tau_tab` is a class attribute (10000 pts, -33→0, uniform), shared by all
cosmologies. `tau()` interpolates via `tools.fast_interp(x, xmin, xmax, yarr)`.
**This is exactly the pattern `expmkappa` should adopt.**

### 3.6 `tools.fast_interp` (VERIFIED, `ABCMBTools.py:268`)
```python
def fast_interp(x, xp_min, xp_max, fp):
    # uniform-grid linear interp; assumes fp evenly spaced from xp_min..xp_max
    n = fp.shape[-1]
    i = (x - xp_min) / (xp_max - xp_min) * (n - 1)
    i = jnp.clip(i, eps, n - 1.0 - eps)
    i_lower = jnp.floor(i).astype(jnp.int32); i_upper = jnp.minimum(i_lower + 1, n - 1)
    w_upper = i - i_lower; w_lower = 1.0 - w_upper
    return w_lower * fp[i_lower] + w_upper * fp[i_upper]
```
Requires a uniform x-grid — `lna_tau_tab` is a `linspace`, so it qualifies.
`background.py` imports it as `tools` (`background.py:12`). **Use it for the new
`expmkappa_tab` to be byte-consistent with `tau`.**

---

## 4. The fix (all edits in `background.py`; VERIFIED line targets)

### 4.1 (a) Replace the field declaration (`background.py:491`)
BEFORE:
```python
    kappa_func : "diffrax.solution"
```
AFTER:
```python
    expmkappa_tab : jnp.array      # exp(-kappa) tabulated on the shared lna_tau_tab axis
```
Reuse the class-level `lna_tau_tab` (`:59`) as the x-axis — no new axis field, and
it is shared so it never has to be stacked. Also update the docstring attribute
lines (`background.py:453` `kappa_func : diffrax.solution`, `:481`
`expmkappa : Compute exp(-kappa)...`) — cosmetic.

### 4.2 (b) Build the grid at construction (`background.py:552`)
BEFORE:
```python
        self.kappa_func = self._tabulate_optical_depth(params)   # line 552
```
AFTER:
```python
        # Build exp(-kappa) on the shared lna_tau_tab axis, then discard the
        # diffrax.Solution so Background stays a pure-array PyTree (stackable).
        _kappa_sol = self._tabulate_optical_depth(params)        # transient local
        def _expmkappa_on(l):
            l_in = jnp.clip(l, -10.0, 0.0)                       # ODE domain is [-10, 0]
            return jnp.where(l < -10.0, 0.0, jnp.exp(-_kappa_sol.evaluate(l_in)))
        self.expmkappa_tab = vmap(_expmkappa_on)(self.lna_tau_tab)
```
Notes:
- The `Solution` becomes a transient local; never stored. All AD machinery
  (ForwardMode through the κ ODE) still runs at build time — same philosophy as
  `_tabulate_conformal_time → tau_tab` (`background.py:252-319`).
- `_tabulate_optical_depth` itself is UNCHANGED; we just consume its return
  locally. Do not touch its rtol/atol=1e-10 (controls the κ accuracy the 1% gate
  depends on).
- `lna_tau_tab` spans -33→0, 10000 pts. Only [-10,0] carries non-zero exp(-κ);
  below -10 the grid holds 0 (matching the `lna<-10` branch). Resolution near
  recombination (lna ≈ -7) is ~3.3e-3 in lna over this axis.

### 4.3 (c) Rewrite `expmkappa` (`background.py:722-742`)
BEFORE:
```python
    def expmkappa(self, lna):
        return jnp.where(
            lna < -10.,
            0.,
            jnp.exp(-self.kappa_func.evaluate(lna))
        )
```
AFTER:
```python
    def expmkappa(self, lna):
        return jnp.where(
            lna < -10.,
            0.,
            tools.fast_interp(lna, self.lna_tau_tab[0],
                              self.lna_tau_tab[-1], self.expmkappa_tab)
        )
```
The `lna < -10.` guard is preserved exactly. Drop-in replacement using the same
interpolator `tau` uses. `visibility` (`background.py:744`) needs **no change**.

### 4.4 (d) `_tabulate_optical_depth` — keep as-is
Only call site is the new local in (b). Cosmetic rename optional; leave rtol/atol.

---

## 5. The spectrum side becomes vmappable (VERIFIED, spectrum.py)

### 5.1 Where the spectrum reads BG — `Cl_one_ell` (`spectrum.py:661`)
```python
        tau0 = BG.tau0                                            # 690
        tau = BG.tau(lna_axis)                                    # 691
        g   = vmap(BG.visibility,in_axes=[0,None])(lna_axis, params)        # 692
        g_prime = vmap(grad(BG.visibility,argnums=0),in_axes=[0,None])(...) # 693
        aH  = BG.aH(lna_axis, params)                             # 694
        expmkappa = vmap(BG.expmkappa)(lna_axis)                  # 695
        aH_dot = BG.aH_prime(lna_axis, params) * aH               # 696
```
- `lna_axis = PT.lna[:-1]` (`spectrum.py:685`) — the PT saveat axis (~500 pts),
  NOT `lna_tau_tab`. So `expmkappa`/`visibility` are queried at PT.lna and
  interpolated off the 10000-pt grid. Grid (10000) ≫ query axis (500), so interp
  error is dominated by PT.lna's own resolution → **no accuracy regression from
  the new grid.**
- **Line 693** `grad(BG.visibility, argnums=0)` — the one autodiff subtlety
  (§6.2): with `fast_interp` the lna-derivative is piecewise-constant.
- Lines 690/691/694/696 read `tau0`/`tau`/`aH`/`aH_prime` — all pure-array, all
  already vmap/stack-clean. `kappa_func` was the only blocker.

`Pk_lin` (`spectrum.py:268`) reads only `PT` and `params` — NO BG. So `Pk_lin` is
already trivially vmappable; its loop is pure JIT-dispatch overhead.

`lensing_Cl`/`lensing_power_spectrum` (only if `lensing=True`) read
`BG.aH/tau/tau0/lna_rec` — all pure-array, fine.

### 5.2 The existing batched wrappers (the bottleneck), VERIFIED
```python
    def get_Cl_batched(self, PT_batched, BG_list, params_batched):   # 616
        B = len(BG_list)
        triples = []
        for i in range(B):                                          # 642  PYTHON LOOP
            PT_i = jax.tree.map(lambda x: x[i], PT_batched)
            p_i = jax.tree.map(lambda x: x[i], params_batched)      # CORRECT per-element slice
            triples.append(self.get_Cl(PT_i, BG_list[i], p_i))      # 645
        ClTT = jnp.stack([t[0] for t in triples]); ...              # 646-648

    def Pk_lin_batched(self, k, z, PT_batched, params_batched):     # 651
        B = jax.tree_util.tree_leaves(params_batched)[0].shape[0]
        pks = []
        for i in range(B):                                          # 655  PYTHON LOOP
            PT_i = jax.tree.map(lambda x: x[i], PT_batched)
            p_i = jax.tree.map(lambda x: x[i], params_batched)      # CORRECT per-element slice
            pks.append(self.Pk_lin(k, z, PT_i, p_i))
        return jnp.stack(pks)
```
`get_Cl` signature is `get_Cl(self, PT, BG, params)` (`spectrum.py:574`);
`Pk_lin` is `Pk_lin(self, k, z, PT, params)` (`spectrum.py:268`).

AFTER — `get_Cl_batched` (BG_batched now a single stacked Background PyTree):
```python
    @eqx.filter_jit
    def get_Cl_batched(self, PT_batched, BG_batched, params_batched):
        return jax.vmap(self.get_Cl, in_axes=(0, 0, 0))(
            PT_batched, BG_batched, params_batched)
```
AFTER — `Pk_lin_batched` (k, z shared → in_axes None):
```python
    @eqx.filter_jit
    def Pk_lin_batched(self, k, z, PT_batched, params_batched):
        return jax.vmap(self.Pk_lin, in_axes=(None, None, 0, 0))(
            k, z, PT_batched, params_batched)
```
`@eqx.filter_jit` matches the module convention and reuses one compiled graph.
Note `get_Cl` ends in a `lax.cond(self.lensing, ...)` (`spectrum.py:610`); `self`
is closed over (static), so it vmaps fine.

### 5.3 Converting the `call_batched` spectrum stage (`main.py:218-233`)
VERIFIED current code:
```python
        # --- stack params + strip-and-stack BGs for the modes computation ---
        params_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps)     # 219
        BG_batch_stripped = jax.tree.map(
            lambda *xs: jnp.stack(xs), *[strip_bg_kappa(bg) for bg in bgs])  # 220-221
        PT_batched = self.PE.full_evolution_batched(
            (BG_batch_stripped, params_batch))                              # 224-225
        ClTT, ClTE, ClEE = self.SS.get_Cl_batched(
            PT_batched, bgs, params_batch)                                  # 228-229  (python list `bgs`)
        Pk = self.SS.Pk_lin_batched(
            self.SS.k_axis_Pk_output, 0., PT_batched, params_batch)         # 231-232
```
AFTER — build ONE stacked Background (now possible after §4) and pass it to both
the modes path and the spectrum vmap; drop the python `bgs` list and the strip:
```python
        params_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps)
        BG_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *bgs)   # full BG stacks now
        PT_batched = self.PE.full_evolution_batched((BG_batch, params_batch))
        ClTT, ClTE, ClEE = self.SS.get_Cl_batched(PT_batched, BG_batch, params_batch)
        Pk = self.SS.Pk_lin_batched(self.SS.k_axis_Pk_output, 0., PT_batched, params_batch)
```
Remove the `from .perturbations import strip_bg_kappa` import at `main.py:205`.
The modes path now consumes the SAME `BG_batch` (no strip/un-strip duplication).

---

## 6. The suspected `call_batched` bug — RESOLVED: it does NOT exist

I read `call_batched` verbatim (`main.py:187-238`). The spectrum stage is:
```python
        ClTT, ClTE, ClEE = self.SS.get_Cl_batched(PT_batched, bgs, params_batch)  # 228-229
        Pk = self.SS.Pk_lin_batched(self.SS.k_axis_Pk_output, 0., PT_batched, params_batch)  # 231-232
```
Both call the WRAPPERS, and both pass the SAME `params_batch`. Inside each wrapper
(`spectrum.py:644` and `:657`) every element is correctly re-sliced
`p_i = jax.tree.map(lambda x: x[i], params_batched)` before the single-cosmology
call. **There is no asymmetry where `get_Cl` gets the full batch while `Pk_lin`
gets a sliced element. The suspected bug is not present.** Current batched Cls are
(within diffrax step-controller noise) correct — consistent with
`bench/smoke_batched_pipeline.log` reporting TT/TE/EE/Pk max_rel ≤ 2.6e-5 vs
single.

The refactor in §5 still removes the per-element JIT dispatch and makes any such
asymmetry structurally impossible (one `params_batch` mapped over axis 0 for
both spectra), but it is a perf fix, not a correctness fix.

### 6.2 The one real subtlety: autodiff through `fast_interp` at line 693
`spectrum.py:693` does `grad(BG.visibility, argnums=0)`. With the old
`Solution.evaluate`, the lna-derivative of exp(-κ) was smooth. With `fast_interp`
(piecewise linear), `g_prime` becomes piecewise-CONSTANT in lna (step function
across grid cells). Because `lna_tau_tab` (10000 pts) is ~20× finer than the
PT.lna axis the integral samples (~500 pts), the step structure is far below
integration resolution and the Cls should be unaffected at the 1% gate. **If the
1% gate or the rtol=1e-8 snapshot shows a `g_prime`-driven regression**, switch
the `expmkappa_tab` read to a C¹ interpolator — `interpax.CubicSpline` is already
imported in `spectrum.py:9` and used there. Only the derivative path (line 693)
cares about smoothness; the value path does not.

---

## 7. `strip_bg_kappa` can be deleted (VERIFIED, perturbations.py:538)
```python
def strip_bg_kappa(bg):                                            # line 538
    return eqx.tree_at(
        lambda b: b.kappa_func, bg, replace=None,
        is_leaf=lambda x: x is None,
    )
```
It nulls ONLY `kappa_func`. Call sites (verified by grep): the docstring mention
at `perturbations.py:209`, and the actual use at `main.py:221`
(`[strip_bg_kappa(bg) for bg in bgs]`). After §4 there is no `kappa_func` field,
so the full `Background` stacks cleanly and `strip_bg_kappa` has no purpose.

Confirmation that `Background` has no OTHER non-stackable leaf (verified read of
`background.py` fields): `species_list` (tuple of `Fluid` eqx.Modules — arrays/
floats), `tau_tab`/`tau0`/`recomb_inputs` (arrays / RecombInputs-of-arrays),
`adjoint` (static — not stacked), `xe_tab`/`lna_xe_tab`/`Tm_tab`/`lna_Tm_tab`
(`array_with_padding`, arrays), `z_reion`/`tau_reion`/`lna_rec`/`rA_rec`/
`lna_transfer_start`/`lna_visibility_stop` (scalars), and (after the fix)
`expmkappa_tab` (array). **No other `diffrax.Solution`.** Note the modes path
already stacks the stripped BG (which still has `species_list`,
`array_with_padding`, etc.), so stacking those leaf types is already proven to
work; adding `expmkappa_tab` (a plain array) cannot break it.

Deletion steps:
1. Delete `strip_bg_kappa` (`perturbations.py:538-554`).
2. Update its docstring references in `_evolve_chunk`/`_compute_modes_batched`/
   `full_evolution_batched` (`perturbations.py:140,168,209` mention
   "`kappa_func` must be `None`") — now moot.
3. `main.py`: remove import at `:205`, change `:220-221` to stack the full BG
   (§5.3).

---

## 8. Correctness risks (ranked)

1. **[HIGH] Grid window.** `_tabulate_optical_depth` integrates lna ∈ [-10, 0]
   only (`background.py:713-714`, `t0=0., t1=-10.`). The §4.2 build CLAMPS with
   `jnp.clip(l, -10., 0.)` and the `where(l<-10, 0., ...)` zeroes the early
   region — matching the existing `expmkappa` `lna<-10` guard exactly. Do not drop
   the clip; evaluating the dense Solution outside [-10,0] is out-of-bounds.
2. **[MED] `grad(visibility)` smoothness (§6.2).** Piecewise-constant `g_prime`.
   Dense grid mitigates; CubicSpline fallback available.
3. **[MED] 1%-vs-CLASS gate (`pytests/accuracy_test.py`).** Run on GPU after the
   change; report TT/EE/Pk max rel error (<1%). Expectation: unchanged to ~1e-6.
4. **[MED] rtol=1e-8 snapshot gate (`pytests/test_snapshots.py`).** WILL change
   bit-for-bit (interp vs dense-eval differ). After the 1% gate passes,
   REGENERATE: `python pytests/fixtures/generate_snapshots.py` on the GPU backend
   the snapshots target. Deliberate, per repo CLAUDE.md.
5. **[LOW] Construction ordering.** `expmkappa_tab` assigned before
   `vmap(self.visibility,...)` at `background.py:555`. Keep the assignment at 552.
6. **[LOW] `z_d`/`rs_d`/`_tabulate_kappa_d` (`background.py:824-936`)** use a
   SEPARATE baryon-drag optical depth + `jnp.interp`, NOT `kappa_func`. Untouched.
   (Verified: no `kappa_func` reference in those methods.)
7. **[LOW] Reionization.** Reion enters via the `xe_tab` correction
   (`background.py:533-535`) BEFORE `_tabulate_optical_depth` (line 552), so
   `expmkappa_tab` already includes reionization. Keep the build order.

---

## 9. Memory note (relevant at B=64)
`design_memo.md §1.1` flags the modes saveat as the memory dominant
(~10.5 GB at B=64, N_k×B×N_lna×N_y). Adding `expmkappa_tab` is one
`(B, 10000)` f64 array ≈ 5 MB at B=64 — negligible. The batched `get_Cl` vmap
over B nests the existing `vmap(Cl_one_ell)` over N_ell; XLA fuses the LoS
`lax.scan` (already `jax.checkpoint`'d at `spectrum.py:827` to kill the
(Nell,Nlna,Nk) rematerialization) across both axes. If the batched LoS pressures
memory at B=64, chunk the B axis of `get_Cl_batched` the same way the modes path
chunks k (`k_chunk_size=100`), or shard B across the 4 GPUs (Phase B pattern).

---

## 10. Expected spectrum-stage speedup at B=32 / B=64
Today the spectrum is a B-length Python loop; each element pays full JIT dispatch
(CHANGELOG: sets the ~12 s/params asymptote). After §4–§5 it is ONE batched-LoS
JIT mapped over B:
- Per-element JIT dispatch eliminated (1 dispatch vs B).
- `get_Cl`'s inner `vmap(Cl_one_ell)` nests inside `vmap` over B → XLA fuses the
  LoS scan across both axes; bessel tables and k-axis are shared constants, so the
  marginal cost per extra cosmology is the per-k LoS arithmetic, not a fresh graph.
- `Pk_lin` over B is pure array work (no BG) → batches near-perfectly.

The measured fact is the ~12 s/params asymptote attributed to the loop; the modes
side is 3.43 s/params at B=64 (Phase B). Removing the spectrum loop drops
per-params toward the modes floor. Order-of-magnitude: **spectrum stage ~10×
faster at B=32, ~15–17× at B=64**; end-to-end `call_batched` falls from
~12 s/params toward ~3.4 s/params on a single A100 — then the 4-GPU shard
(0.93 s/params) puts the ~1 s/params target in reach.

---

## 11. Ranked action list
1. **[enabling] `background.py`:** field `:491` → `expmkappa_tab`; build at `:552`
   via `vmap(_expmkappa_on)(lna_tau_tab)`, discard the local `Solution`; rewrite
   `expmkappa` (`:722`) to `tools.fast_interp`. (§4)
2. **[gate] Run `pytests/accuracy_test.py` (GPU srun, `PYTHONPATH=$(pwd)`).**
   Report TT/EE/Pk max rel error; must stay < 1%. If `g_prime` smoothness
   regresses, swap to CubicSpline read (§6.2).
3. **[payoff] `spectrum.py`:** rewrite `get_Cl_batched` (`:616`) and
   `Pk_lin_batched` (`:651`) as `jax.vmap` + `@eqx.filter_jit`. (§5.2)
4. **[payoff] `main.py::call_batched`:** stack the full BG into `BG_batch`, feed
   both the modes path and the spectrum vmaps, drop `bgs` list + `strip_bg_kappa`
   import. (§5.3)
5. **[cleanup] Delete `strip_bg_kappa`** (`perturbations.py:538`) and fix its
   docstring mentions. (§7)
6. **[gate] Regenerate + run `pytests/test_snapshots.py`** on the GPU backend the
   snapshots target. (§8.4)
7. **[measure] Re-run `bench/perf_batched.py` at B=16/32/64**; confirm per-params
   drops toward the modes floor; update `bench/perf_batched.log`.

The §6 "bug check" is already resolved (no bug), so it is NOT on this list — but
the §5.3 refactor de-risks that path permanently regardless.

---

## 12. Bottom line
Localized, low-risk fix exactly as the repo's CHANGELOG (Phase E "NEXT STEP",
path (a)) and CLAUDE.md prescribe, now pinned to verified line numbers: one field
swap + one build block + one `expmkappa` rewrite in `background.py`; two Python
loops → `jax.vmap` in `spectrum.py`; one stacking change in `main.py`; delete
`strip_bg_kappa`. The suspected `call_batched` params-asymmetry bug does not
exist — current batched Cls are correct within diffrax noise. After the refactor
the spectrum stage stops being the asymptote (~10–17× faster at B=32/64), moving
the single-GPU bottleneck back to the modes solve where the 4-GPU shard already
reaches ~0.93 s/params.

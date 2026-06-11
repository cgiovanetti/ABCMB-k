# Design: batched AD gradients on the per-k pipeline (the throughput unlock)

**Date:** 2026-06-11 · **Branch:** `perk-perf` · **Author:** Claude (frequentist_adversary session)

## The point

The `perk-perf` refactor exists to compute **one k-mode for all params at once** (params-axis
batching + k-chunking + GPU sharding) — `Model.call_batched`. The AD-profile driver
(`scan/profile_prod_ad.py`) uses it for function VALUES but computes GRADIENTS through the
OLD single-cosmology path (`run_cosmology_abbr`, ~85 s/gradient/point, un-amortized). So the
expensive half — the autodiff that is the whole reason for the differentiable solver — does
**not** use the new k-distribution strategy. This memo scopes closing that gap.

## Why the two obvious things don't work (measured this session)

1. **`jax.jacfwd(call_batched)`** — `call_batched` takes a *list of param dicts* and derives
   params with an EAGER Python loop (`add_derived_parameters` has `sys.exit`/`bbn_type`
   branches, not traceable/vmappable). And the Jacobian of a B->B map is a dense `(B,B,P)`
   block — B× wasted work; we only want the block-diagonal `∂χ²_b/∂θ_b`.

2. **`vmap(jacfwd(single_path))` or `jacfwd(one-function batched pipeline)`** — puts the WHOLE
   GPU→CPU(HyRex)→GPU pipeline into ONE traced/jitted/vmapped XLA module. MEASURED: ~20–25 min
   compile-to-first-iteration, sometimes an effective hang (`scan/derisk_ad.py` vmap+jit;
   `scan/val_loop*.log`). This is the monolith pathology — the same reason OPTION_B says never
   wrap the cross-device χ² in an outer `jax.jit`.

## The key realization

`call_batched` AVOIDS the monolith because it does NOT compile the whole pipeline as one unit.
It eager-orchestrates (Python) a sequence of SEPARATELY-`@eqx.filter_jit`'d batched stages with
`device_put` hops between them:

```
add_derived_parameters (eager python, per-cosmo)        # main.py:622  -- NOT traceable as-is
_build_bgs_batched:                                     # main.py:392
    _pre_recomb_batched   (filter_jit, vmap, GPU)       # main.py:376
    _recmodel_cpu(...,True)(inputs)  (jit, vmap, CPU)   # HyRex, device_put hops
    _get_BG_batched       (filter_jit, vmap, GPU)       # main.py:384
full_evolution_batched (eager orchestration of...)      # perturbations.py:286
    _compute_modes_batched / _evolve_chunk per k-chunk  (filter_jit) # perturbations.py:201,234
    make_output_table_batched (filter_jit)              # perturbations.py:316
get_Cl_batched         (filter_jit, vmap, GPU)          # spectrum.py:617
Pk_lin_batched         (filter_jit, vmap, GPU)          # spectrum.py:643
```

**Forward-mode must mirror this exactly: apply `jax.jvp` to EACH already-jitted batched stage
in eager Python, threading the tangent between stages.** Each stage's `jvp` is a separate,
small compile (jvp-of-an-already-jitted-fn); none of them spans the cross-device boundary as a
monolith. Because we differentiate the BATCHED stages, the gradient INHERITS the params-axis
batching, the k-chunking, and the sharding for free. Forward-mode is the right mode: it's
memory-flat (de-risk: jacfwd of the single path was 4.7 GB vs 1.2 GB primal; reverse-mode OOMs
storing the trajectory) and we only have P=5 nuisance directions.

## Proposed implementation

### New entry point (additive — does NOT touch `call_batched`)
`Model.call_batched_grad(params_list, wrt_keys, shard=None, k_chunk_size=100)`
→ returns `(ClTT,ClTE,ClEE, dCl/dθ for θ in wrt_keys)` batched over B, OR more usefully a
χ²-aware variant that takes the likelihood closure and returns `(chi2 (B,), grad (B,P))`.

### Step A — derivation + its tangent (eager, per-cosmology, cheap ~ms)
The non-traceable `add_derived_parameters` stays OUTSIDE the differentiated GPU region. For each
cosmology and each of the P directions, forward-mode through JUST the derivation:
```
full_p, full_p_dot_j = jax.jvp(add_derived_parameters, (raw_p,), (raw_p_dot_j,))
```
(de-risk confirmed `add_derived_parameters` traces under forward-mode for the fixed ΛCDM path —
the `sys.exit` branches are static and don't fire). Stack over B → `params_batch` (primal) and
`params_batch_dot` (P, B, ...). For ΛCDM the raw→derived map is mostly identity + a few smooth
functions, so this is trivial and robust.

### Step B — staged forward-mode through the batched pipeline (P tangents, vmapped)
Push all P tangents at once by vmapping each stage's `jvp` over the P-direction axis (primal
shared) — this is what `jacfwd` does internally, but applied PER STAGE so no monolith:
```
def push(stage, primals, tangents_P):          # tangents_P: leading P axis
    return jax.vmap(lambda t: jax.jvp(stage, primals, t)[1])(tangents_P)
pre_BG      = stage(params_batch)                          # primal, once
pre_BG_dot  = push(_pre_recomb_batched, (params_batch,), (params_batch_dot,))
# HyRex: device_put primal+tangent to CPU, push(_recmodel_cpu(...,True)), reshard back
BG, BG_dot  = primal + push(_get_BG_batched, ...)
PT, PT_dot  = primal + push(full_evolution_batched, (BG,params), (BG_dot,params_dot))
Cl, Cl_dot  = primal + push(get_Cl_batched, (PT,BG,params), (PT_dot,BG_dot,params_dot))
```
Reuse `call_batched`'s `shardfn` on primal AND tangent so the gradient is B-sharded too (the
win scales with GPUs). `full_evolution_batched` is itself eager-orchestrated over k-chunks, so
`jvp` of it stays staged (jvp of each jitted `_evolve_chunk`) — NOT a monolith. (If that turns
out to compile poorly, fall back to threading tangents through the k-chunk loop explicitly.)

### Step C — χ² and its tangent
The likelihoods (`plik_lite` + `lowl_like`) are already pure-jnp/differentiable, so
`(chi2, chi2_dot) = jax.jvp(chi2_from_cls, (Cl,), (Cl_dot,))` — or just contract Cl_dot through
the (already linear-ish) binning. Returns `(chi2 (B,), grad (B,P))`.

## Validation contract (non-negotiable)
The batched-AD gradient MUST match the single-path `jacfwd` gradient (the de-risk's proven-exact
AD) to ~solver tolerance (rtol=1e-5) at several test cosmologies and all P directions. That is
the correctness gate, run BEFORE any throughput claim. Then: (1) memory sweep vs B (forward-mode
is flat per tangent, but P tangents × B-saved-trajectory needs the round-2/3-style measurement);
(2) throughput vs the single-path loop (expected ~10–20×: a batched forward is ~0.67 s/param and
P=5 tangents add ~sublinear, vs 85 s/point single).

## Risks / unknowns (ranked)
1. **Perturbation k-chunk tangent threading** — `full_evolution_batched`'s eager k-chunk loop
   under `jvp`. Most likely fine (staged), but the diffrax `ForwardMode` adjoint × k-chunk ×
   P-tangent interaction is the part to de-risk FIRST with a tiny l_max run.
2. **HyRex forward-diff under vmap+device_put** — de-risk did it single-cosmo; batched + P
   tangents + the CPU hop needs a check (recomb depends on ω_b/ω_cdm/h, so tangents must flow).
3. **Memory** — P=5 tangents alongside the ∝B saved-trajectory tensor. Measure; shard to fit.
4. **`jvp` of `@eqx.filter_jit` methods** — standard, but eqx filtering of the tangent pytree
   needs care (use `eqx.filter_jvp` / partition static vs array leaves).

## Effort & files
- `abcmb/main.py`: new `call_batched_grad` (+ a `_build_bgs_batched`-with-tangents helper).
- `abcmb/perturbations.py`: possibly a tangent-threading `full_evolution_batched` variant.
- `scan/profile_prod_ad.py`: wire as `PA_GRADMETHOD=batched`; drop the slow loop for production.
- Core-code changes → write proposed diffs to a file for review first (global CLAUDE.md rule),
  validate against single-path jacfwd, THEN apply.
- Estimate: ~1 focused GPU session to de-risk the perturbation-stage jvp (risk #1) + memory, then
  ~1 session to wire `call_batched_grad` end-to-end + validate + throughput. The payoff is the
  whole reason for the branch: the frequentist gradient finally rides the k-distributed pipeline.

## Recommended first step
De-risk risk #1 in isolation: a tiny script that `jax.jvp`s `full_evolution_batched` (staged)
at small l_max + small B + P=2 tangents, checks the tangent matches single-path `jacfwd`, and
times the compile. If that's clean and not a monolith, the rest is mechanical assembly.

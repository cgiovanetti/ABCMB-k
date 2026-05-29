# Battle-royale brief — perf round 2 (perk-perf, post-keystone)

**READ THIS FIRST. The `bench/ideas_*.md` and `bench/notes_*.md` memos are STALE
on priorities** — they were written *before* the keystone landed. Do not re-derive
problems that are already solved. This brief is the current ground truth.

## The goal (from the user, 2026-05-29)
A **full frequentist profile/scan over a 20–40-parameter model**. This means a
VERY large number of cosmologies will be evaluated. The metric that matters is
**sustained throughput — cosmologies per GPU-second** (and per wall-clock-second
across whatever hardware we can grab). Two framings of success, either or both:
  (a) batch **B ≫ 64** cosmologies per `call_batched` and still be ≤ ~1 s/param, or
  (b) drive per-param **≪ 1 s/param**.
"No refactor too large." Up to **2 interactive GPU allocations** for preliminary
testing (the orchestrator runs the GPU jobs; agents do static analysis only).

## What is ALREADY DONE (do not propose these — they are committed on perk-perf)
1. **expmkappa tabulated** — `Background.kappa_func` (a `diffrax.Solution`) is gone,
   replaced by `expmkappa_tab` array read via `interpax.interp1d(cubic)`. `Background`
   now stacks across cosmologies. `strip_bg_kappa` deleted.
2. **Spectrum vmapped** — `get_Cl_batched`/`Pk_lin_batched` are a single
   `@eqx.filter_jit jax.vmap` over B. spec_Cl went 3.2 → 0.01 s/param.
3. **Setup batched + vmapped** — `_build_bgs_batched` runs vmapped pre-recomb (GPU),
   vmapped HyRex (CPU), vmapped get_BG (GPU): O(1) jits + transfers, not O(B). setup
   went 7.0 (eager) → ~2.0 s/param at B=16, amortizing toward ~0.4 at large B.
4. **4-GPU sharding over B** — `call_batched(shard=)` uses `Mesh` +
   `NamedSharding(P('batch'))`, GSPMD auto-partition, no collectives. B padded to a
   multiple of n_dev. Each device builds+solves B/n_dev cosmologies.

## CURRENT measured perf (A100, ELLMAX=800, lensing=False, post-compile, per param)
Single-GPU B=16: setup 2.01 + perturb 1.97 + spec 0.01 ≈ **4.0 s/param**.
4-GPU sharded (bench/perf_multigpu_results.json):
  B=16 → 3.24, B=32 → 1.82, B=48 → 1.34, B=64 → 1.13 s/param.
Per-param is **still falling with B** at B=64. The CHANGELOG 4-GPU B=64 = 1.13 s/param.

## THE current bottleneck — read carefully
The **perturbation solve** is now the sole floor (`perturbations.py`):
- `evolution_one_k` (perturbations.py:382-448): per k-mode, an **adaptive implicit
  Kvaerno5** diffrax solve, PIDController, `saveat=SaveAt(ts=lna)` (500 pts),
  ForwardMode adjoint, `max_steps_PE=2048`. rtol/atol chosen per-k via
  `jnp.where(k>k_split_PE=0.01, large_k(1e-4/1e-6), small_k(1e-5/1e-10))`.
- `_evolve_chunk` (perturbations.py:125-148): `@eqx.filter_jit`, `vmap(k_chunk) ×
  vmap(B)` around evolution_one_k. **ALL k_chunk×B lanes share ONE lockstep PID
  controller** → a chunk's wall-clock = its worst (k,b) lane's step count.
- `_compute_modes_batched` (perturbations.py:150-192): **python loop** over
  contiguous k-chunks of size 100, then `jnp.concatenate(axis=0)`.
- N_k ≈ 492 (lensing=False) to 571 (lensing=True). k-axis is monotonic
  (model_specs.py:120-175), so a contiguous chunk = a k-band = a stiffness band.
- Step counts (baseline, single-cosmology): min=41 median=380 **max=1579**,
  max/median = **4.16** (the worst-case-k tax). After the params-first flip,
  worst-k max/median **over B = 1.12** at fixed k → sharding B can't beat one
  worst-k solve's latency; the single-worst-k floor is ~0.8 s/param.
- State vector Ny ≈ 46 (ΛCDM; l_max_g=12, l_max_pol_g=10, l_max_ur=17). Memory at
  B=64, k_chunk=100: ~28-31 GB / 40 GB on one A100 (near OOM). Sharding /n_dev the
  per-device memory, which is why bigger B fits on 4 GPUs.

## Levers NOT yet tried (from the stale memos — re-evaluate against CURRENT floor)
- **A1 stiffness-homogeneous / non-uniform k-chunking**: big chunks for smooth
  low-k, small for stiff high-k. Claimed 1.3-1.7× on modes, ZERO accuracy risk.
  NOT done (sweep_kchunk only varied a single uniform size, found 100 "optimal" —
  but only among uniform sizes).
- **A2 explicit solver (Tsit5/Dopri) on the smooth small-k band** (k<k_split_PE):
  non-stiff, no implicit Jacobian. 2-4× on those modes.
- **A3 fixed-step / StepTo schedule** per band — kills lockstep adaptivity, static
  shapes XLA loves. HIGH accuracy risk.
- **D float32 / mixed precision** perturbation state y (Jacobian/LU is FP-heavy on
  A100, fp64 is ½ fp32). 1.4-1.8× on solve, HIGH gate risk. (tf32 also possible.)
- **C saveat trim/redistribute** 500→~300 dense-at-recomb: cuts modes memory AND
  the LoS scan; OOM relief → bigger B. MED gate risk.
- **A1.2 loosen atol_large_k_PE** 1e-6→3e-6: lockstep paces to tightest lane.
- Persistent compile cache + pad-B-to-fixed-size: cold-start only.
- **Multi-node / >4 GPU data parallelism**: embarrassingly parallel over B; the
  premium queue (account m3166_g) can grab more than one node. Pure throughput.

## Hard constraints / gotchas
- `_to_float` casts int/bool params to float64 before filter_jit (custom_vjp/AD
  safety for checkpointed_while_loop). Don't strip it; fp32 must respect it.
- HyRex + LINX run on CPU intentionally (sequential solvers); GPU re-transfer in
  try/except for CPU-only friendliness.
- Accuracy gate = `pytests/accuracy_test.py` (max-rel TT/EE/Pk ≤ 1% vs CLASS;
  current TT 0.197% EE 0.231% Pk 0.185%). `pytests/test_snapshots.py` rtol=1e-8.
  Any theory-affecting change → run accuracy_test + regen snapshots.
- diffrax adaptive solves stop at convergence, NOT at max_steps; lowering the cap
  is a correctness guard, not a perf knob.
- Login-node python is FORBIDDEN; all runs go through srun. Agents: static analysis
  only — propose, don't run.

## What to produce
A ranked shortlist of CONCRETE levers to push throughput further, each with:
  - mechanism + exact file:line touch points,
  - expected effect (throughput / per-param), stated as a range,
  - accuracy risk + the cheap test to gate it,
  - implementation effort + probability of success,
  - the HIDDEN FLOOR that could make it underperform (be skeptical).
Rank by (throughput gain × prob success ÷ (risk × effort)). Flag your top 1-2 bets
and what preliminary GPU measurement would de-risk them fastest.

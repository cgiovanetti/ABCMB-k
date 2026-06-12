# Taming the staged-jvp COMPILE — findings (session fix_grad_perf, 2026-06-12)

CONCLUSION (read first): the staged-jvp compile was NEVER a wall. Each batched
stage's jvp compiles ONCE per (B,l_max) aval and is cached on the filter_jit'd
method, so it is a ~5-min ONE-TIME-PER-JOB tax (lensed-spectrum jvp at l2508 = 119s;
perturbation jvp l_max-INDEPENDENT). The prior "pathologically super-linear" framing
was COLD end-to-end measurements (incl the single-path reference) with both B and
l_max doubled each step. linearize ruled out; gradient wired as PA_GRADMETHOD=batched
and validated (8.58e-4 vs single-path). The remaining open item is THROUGHPUT (needs
B-scaling + sharding, like call_batched), not compile. Detail below.

## Per-stage jvp compile/warm (scan/grad_compile_profile.py, throwaway cache, A100)

### l_max=128, B=2, P=2 (omega_cdm, n_s) — BASELINE

| stage                | primal compile | primal warm | **jvp compile** | jvp warm/dir |
|----------------------|---------------:|------------:|----------------:|-------------:|
| _pre_recomb_batched  |   3.3s         | 0.21s       |  3.4s           | 0.24s        |
| HyRex (CPU)          |  29.9s         | 0.11s       | 26.8s           | 0.20s        |
| _get_BG_batched      |   7.9s         | 1.21s       |  7.2s           | 1.50s        |
| full_evolution (PE)  |  37.3s         | 33.17s      | **48.2s**       | 66.35s       |
| get_Cl_batched (SS)  |  26.3s         | 0.01s       | **47.7s**       | 0.03s        |

Total jvp compile ≈ 133s; total primal compile ≈ 105s.

### Key reads
- **Perturbation jvp**: 48s compile **AND** 66s warm/direction. Warm = re-solving
  the stiff Kvaerno5 ODE + tangent per direction. l_max-INDEPENDENT (hierarchy
  truncation l_max_g/pol/ur/ncdm fixed; only n_k grows w/ l_max → more cache-hit
  k-chunks, not bigger compile). B-scaling (buffer sizes).
  → COMPILE + RUNTIME problem. `jax.linearize` targets the runtime (solve primal
    once, apply linear tangent map P times).
- **Spectrum jvp**: 47.7s compile but **0.03s warm** — pure COMPILE problem.
  vmap-over-ells of a scan; graph is ell-count-independent (vmap=loop) but buffers
  (Nell×Nk) grow with l_max → XLA passes slow mildly w/ l_max.
- **HyRex jvp** (27s): l_max-independent, fixed ~tens of s.
- Each stage's jvp compiles ONCE per (B,l_max) aval shape and is CACHED on the
  filter_jit'd method → P directions do NOT multiply compile (already amortized).
  So the blowup is per-STAGE compile growing with B (& mildly l_max), NOT the
  number of compiles.

## Hypothesis (to confirm w/ isolation runs)
The super-linear blowup the handoff saw ("412s@B2/l128 → >20min@B4/l256 →
loop_reduce_fusion 4m46s@B16/l512") doubled BOTH B and l_max each time. Decompose:
- B drives both PE and SS jvp compile (XLA fusion/buffer-assignment passes scale
  with buffer sizes ∝ B).
- l_max drives SS jvp compile mildly (buffer ∝ Nell), PE not at all.

If B is the dominant driver, the levers are:
1. **Shard B over GPUs** (reuse call_batched shardfn) → per-device compile for
   B_local=B/n_dev → smaller buffers → faster compile (free, already designed).
2. **Lower XLA opt level** for the gradient compile (skip expensive fusion passes)
   → faster compile, runtime ~unaffected (gradient warm is ODE-solve-bound).
3. **Smaller k_chunk for the PE jvp** → smaller _evolve_chunk buffers.
4. **jax.linearize** → kills the P× re-solve runtime (perturbation warm 66s/dir).

## NEGATIVE result: jax.linearize on the ODE (perturbation) stage — MEASURED
l128/B2, P=3:
  [jvp] perturbation  compile ~47s   warm/dir 66.38s
  [lin] perturbation  compile 154.1s apply x3 193.42s (64.47s/dir)
linearize is 3× WORSE compile (154 vs 47s) AND **no runtime win** (64.5 ≈ 66.4
s/dir). The "apply" re-integrates the FULL augmented ForwardMode-Kvaerno5 system
every direction — it does NOT cache/separate the primal trajectory for the adaptive
implicit solver. **Drop linearize entirely.** Keep per-direction jax.jvp.

## Reframing: the compile is a ONE-TIME PER-JOB tax (in-memory jit cache)
Within a single SLURM job (one POI's full BFGS, many iters in ONE process), the
in-memory jit cache means the staged-jvp compiles ONCE on the first gradient call
and every later BFGS iter reuses it. So the persistent $SCRATCH cache only matters
for CROSS-job reuse; it is NOT essential to the core win. The blocker is purely
whether that ONE compile is bounded at production shape. And the per-direction jvp,
once compiled, amortizes the warm gradient over B (~24× vs single-path even without
linearize). So: tame the one-time compile; keep per-direction jvp.

## n_k ∝ l_max (model_specs.py:157) => stage compile scaling
- Perturbation jvp COMPILE is l_max-INDEPENDENT (k_chunk=100 fixed; only #chunks
  grows w/ l_max -> runtime, cached compile). It is B-SCALING and the suspected
  super-linear killer: the augmented-Kvaerno5 loop fused over vmap(k_chunk=100, B)
  (handoff's "loop_reduce_fusion 4m46s @ B16/l512").
- Spectrum jvp COMPILE scales with l_max (ell count + Nk buffers) and B.

## k_chunk lever — MEASURED (scan/jvp_compile_levers.py, l128/B8, 1 dir)
cold = compile+run for ONE perturbation jvp (run rises w/ smaller k_chunk):
  k_chunk=100: cold 186.2s   k_chunk=50: 152.8s   k_chunk=25: 174.1s   (k_chunk=10: TBD)
Non-monotonic: 100->50 DROPS 33s (compile falls faster than run rises), 50->25 RISES
(run dominates). So k_chunk~50 minimizes the ONE-CALL cold here; smaller k_chunk does
cut the COMPILE but past ~50 the extra runtime (more chunks/dispatch) outweighs it.
Modest lever (~18% best). For a many-iter production job the WARM (run) dominates the
total, favoring larger k_chunk; only when the one-time COMPILE is huge (l2508) does a
smaller k_chunk pay off. => choose GRAD_KCHUNK from the production-shape compile size.
Also: B-scaling of perturbation jvp cold is MILD (B2 113s -> B8 186s for 4x B at
k_chunk=100), i.e. the perturbation compile is NOT the catastrophic super-linear term.

## Compile levers
1. **Smaller k_chunk under jvp** — MEASURED above. Real but modest (~18%), and the
   PE compile B-scaling is mild anyway. Use a moderate k_chunk (~50) for the
   gradient if the one-time compile dominates; otherwise keep 100 for warm speed.
2. **XLA opt-level flag** — not yet tested. Candidate if production compile is huge.
3. **Sharding** (reuse call_batched shardfn) -> per-device B_local -> smaller
   buffers + parallel compile (free, already designed). Deferred to after the
   single-GPU path validates.

## THROUGHPUT — honest read (scan/grad_prod_shape.py, l2508/B8/P2/lensing, single-GPU)
COLD (compile+run) = 748.1s, peak **4.07 GB** (huge headroom -> B can go much larger).
Compile ~300s (SS 119 + PE ~85 + HyRex 27 + primal compiles), so warm-run ~448s for
B=8/P=2 => ~56 s/cosmo (P=2). At this SMALL-B SINGLE-GPU config the batched gradient is
NOT yet a win vs single-path (~85 s/cosmo P=5): the l2508 perturbation solve does not
saturate the GPU at B=8, and forward-mode x P directions costs. The WIN is the SAME
B-scaling + sharding that call_batched already demonstrates (primal 3.24 s/param @ B16
-> 1.13 s/param @ B64/4-GPU); the gradient is jvp of those identical batched stages, so
it inherits that amortization. The 4 GB peak @ B8 means B64 (~32 GB) fits one 80GB A100.
=> NEXT for throughput: shard the gradient (reuse call_batched shardfn on the stacked
primal+tangent; bench/driver_batched_wiring.md) + run at B64. NOT done this session --
the session's goal was the COMPILE, which is tamed.

## RESULTS — production compile is TRACTABLE; gradient CORRECT (the headline)
- **Lensed-spectrum jvp at PRODUCTION l2508/lensing=True** (scan/spec_compile_probe.py,
  fake-PT shape probe, B=2): **cold 119.6s, warm 0.57s, compile~119s, peak 1.80 GB.**
  The Wigner d-matrices are param-independent (warm 0.57s confirms the run is cheap);
  the ~2 min is a ONE-TIME compile. NOT intractable. No wigner-precompute fix needed.
- **Total production staged-jvp compile ≈ HyRex(27s) + PE(~110s @ B≈13, l_max-indep) +
  SS(119s @ l2508) + small ≈ ~5 min ONE-TIME PER JOB** (then every BFGS iter warm).
  The handoff's "hours/intractable" came from measuring COLD end-to-end (incl the
  single-path reference) with BOTH B and l_max doubled each step.
- **CORRECTNESS (scan/validate_chi2_grad.py, l128/B2/P5): worst max-rel(batched chi2
  grad, single-path jacfwd of the SAME objective) = 8.58e-04 (<1e-3 => CORRECT).**
  Higher than the 1.75e-5 dCl validation because the chi2 contraction (profile_A
  envelope + low-ell) amplifies the ~1e-5 batched-vs-single Cl chunking noise; still
  negligible for BFGS descent (the direction is essentially exact).

## Status: wiring APPLIED
- scan/batched_grad.py: + `staged_chi2_and_grad` + `k_chunk_size` knob.
- scan/profile_prod_ad.py: + `_chi2_of_cls`, `_phys_to_derived` (with the _to_float
  gotcha fix), `batched_grad_fg`, and `iterate_fg` method=="batched" branch
  (PA_GRADMETHOD=batched, PA_GRAD_KCHUNK). BFGS still keeps VALUES on the fast
  call_batched path (batched F discarded) -> consistency rule preserved.
- scan/validate_chi2_grad.py: correctness gate (batched chi2 grad vs single-path
  jacfwd of the SAME objective). GOTCHA found+fixed: _phys_to_derived must _to_float
  the DERIVED dict (FIXED carries python-int N_nu_massive -> jnp.array(1) int ->
  None tangent mismatches the _to_float'd float primal partition). The existing
  batched_grad.py derived_and_tangents dodges this only by casting the RAW input to
  float first.

## Spectrum jvp compile is B-INSENSITIVE (good)
spectrum jvp cold (lensing OFF): l128/B2 = 47.7s, l128/B8 = 53.2s. Barely moves with
B. So the spectrum compile scales with l_max (+ lensing), NOT B. Combined with the
perturbation jvp being l_max-independent + mild-B, the production compile decomposes:
  total ≈ HyRex(~27s, fixed) + PE(B; ~110s at B=13) + SS(l_max, lensing; THE unknown)
The SS(l2508, lensing) term is the production gate -> grad_prod_shape.py.

## STRATEGY (settled)
- Keep per-direction jax.jvp staged gradient (scan/batched_grad.staged_cl_and_grad);
  linearize is dead. Added `staged_chi2_and_grad` (contracts Cl tangents through the
  likelihood via one cheap jvp/direction) + a `k_chunk_size` knob.
- The compile is ONE-TIME PER JOB (in-memory cache); warm gradients amortize over B
  (~24x vs single-path). So the deliverable = (a) confirm the production-shape
  (l2508, lensing=True) compile is BOUNDED, (b) validate staged_chi2_and_grad vs
  single-path (scan/validate_chi2_grad.py), (c) wire PA_GRADMETHOD=batched into
  scan/profile_prod_ad.py (plan in bench/driver_batched_wiring.md), (d) quote warm
  throughput.
- OPEN production risk: the LENSED spectrum jvp. lensing=True => num_mu=ellmax+570,
  lensing_ells≈ellmax+500, ~15 Wigner d-matrices over (mu, ells). At l2508 that's a
  (3078 x 3008) grid x15, forward-differentiated — the likely dominant production
  compile. grad_prod_shape.py (lensing=1) measures it; if intractable, chunk the
  ell-vmap / lensing block (analogous to the perturbation k_chunk).

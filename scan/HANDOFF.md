# HANDOFF — frequentist_adversary session (2026-06-11)

Branch `perk-perf` (NOT merged to main). This session: (1) adversarially reviewed the
existing LCDM frequentist profile, (2) fixed 3 of the 5 gaps, (3) made the autodiff
gradient ride the per-k batched pipeline. Full detail in `CHANGELOG.txt` entries
2026-06-11 (b) and (c), and `bench/batched_ad_design.md`. This file is the fast orientation.

## TL;DR state
- **Data model (#3): DONE + validated.** Real Planck low-ell TT+EE replace the circular tau prior.
- **Autodiff (#5): DONE.** Exact AD gradients, proven, wired into the driver.
- **Convergence (#1): implemented + demonstrated; a stall bug found & fixed.** Full converged
  run + multistart not yet executed (throughput-gated).
- **Batched AD on the per-k pipeline: CORRECT (1.75e-5 vs single-path), but compile-blocked.**
  The gradient now rides the k-distribution refactor; the assembled staged-jvp COMPILE is
  pathologically slow — that's the #1 next task.
- **GPU: released. Working tree: clean except pre-existing entry-(a) result binaries (untracked).**

## The 5 review gaps (from the assessment at session start)
1. Optimizer global-min / convergence not demonstrated  -> **addressed** (gate+BFGS+stall fix; multistart coded, not run)
2. Theory-vs-CLASS parameter-bias not validated         -> NOT addressed
3. Data model limited + circular tau prior              -> **DONE** (low-ell TT+EE)
4. Coverage (Wilks vs Feldman-Cousins) assumed          -> NOT addressed
5. Finite differences instead of autodiff               -> **DONE** (exact AD; + batched-AD now correct)

## Key files (this session)
- `scan/plik_lite.py`        — (pre-existing) high-ell plik-lite TTTEEE chi2.
- `scan/lowl_like.py`        — NEW. AD-able low-ell EE (SRoll2) + TT (Commander). Splines
                               precomputed at init (compile fix). Validated vs cobaya native.
- `scan/validate_lowl.py`    — NEW. CPU validation of lowl_like vs cobaya (EE 0.052, TT 0.000).
- `scan/profile_prod_ad.py`  — NEW. AD-gradient BFGS driver: ||g||<GTOL gate, PD check,
                               multistart mode, hybrid eval (fast call_batched VALUES + AD grad).
                               PA_GRADMETHOD=loop (default, robust) | vmap (slow compile).
- `scan/profile_prod_ad.slurm` — NEW. Production submit (one POI per GPU-task, resumable).
- `scan/derisk_ad.py`        — NEW. Proved AD grad exact vs FD; Hessian OOMs (=> BFGS).
- `scan/derisk_batched_ad.py`— NEW. Risk-#1 de-risk: jvp through full_evolution_batched is
                               staged (1.55x) + correct (5e-4 vs FD).
- `scan/batched_grad.py`     — NEW. **The batched AD gradient** (staged forward-mode, no
                               core-code edits). Validated end-to-end vs single-path jacfwd: 1.75e-5.
- `scan/batched_grad_timing.py` — NEW. Throughput probe (compile-blocked; see below).
- `bench/batched_ad_design.md`  — NEW. Design + all de-risk/result records. READ THIS for the refactor.
- `scan/profile_prod.py`     — OLD (entry-a) FD-stencil driver. Superseded by profile_prod_ad.py.

## How to resume each open thread (all need a GPU: salloc ... --gpus=N ...)

### A. Tame the staged-jvp COMPILE (the throughput unlock — highest value)
The batched AD gradient is correct but `scan/batched_grad.py`'s assembled staged-jvp compiles
super-linearly (412s @ B2/lmax128 -> >20min @ B4/lmax256; XLA `loop_reduce_fusion` 4m46s @
B16/lmax512). The single ISOLATED stage-jvp compiled in 111s, so the slowness is the ASSEMBLED
graph. Plan (bench/batched_ad_design.md "THROUGHPUT" section):
  1. Instrument `staged_cl_and_grad` to time each stage's jvp compile separately -> find the culprit
     (prior: the k-chunked perturbation jvp `_evolve_chunk`/`_compute_modes_batched`).
  2. Try: per-stage `filter_jit` boundaries (so $SCRATCH cache captures each, chopping the giant
     fusion) / a single `vmap`-over-P jvp instead of the P-direction python loop / XLA flags.
  3. Then measure warm throughput (scan/batched_grad_timing.py), wire PA_GRADMETHOD=batched into
     profile_prod_ad.py, retire the loop, and reuse call_batched's shardfn for the multi-GPU win.
Repro the finding: `BGT_LMAX=256 BGT_B=4 python scan/batched_grad.py` (and batched_grad_timing.py).

### B. Finish the #1 demonstration (convergence + global min)
Stall bug is FIXED (commit 76127ca) but not re-confirmed at scale.
  - Confirm no-stall + reach the gate: `PA_POIS=n_s PA_NPTS=3 PA_MAXIT=20 PA_GRADMETHOD=loop \
    python scan/profile_prod_ad.py` (loop AD ~85s/grad/pt; ~20min compile-to-first-iter; cached).
  - Global-min: `PA_MULTISTART=1 PA_MS_K=4 PA_POIS=n_s python scan/profile_prod_ad.py`.
  - NOTE these are SLOW via the loop path -> ideally do them AFTER thread A makes the gradient fast.

### C. Production AD profile (after A)
`sbatch scan/profile_prod_ad.slurm` (defaults: 6 POIs, NPTS=9, loop, one POI/GPU, regular qos).
Once thread A lands, switch PA_GRADMETHOD=batched + raise NPTS=13.

### D. Untaken review gaps
- #2: run an MCMC on the SAME likelihood (plik-lite+lowTT+lowEE, no tau prior) and an
  ABCMB-vs-CLASS profile cross-check; quantify the parameter-bias floor.
- #4: justify Wilks for LCDM; plan Feldman-Cousins / MC coverage for extension params.

## Gotchas (will bite the next person)
- NEVER `jax.jit`/`vmap`/`jacfwd` the WHOLE cross-device pipeline as one fn -> ~20min monolith /
  hang (the GPU->CPU-HyRex->GPU `device_put` fuses). Always stage (jvp per already-jitted stage).
- AD Hessian (forward-over-forward) OOMs at l_max=2508 (~35 GB/cosmo). Use BFGS + FD-of-AD-grad.
- Batched-AD: `_to_float` the derived params before jvp (int N_nu_massive -> None under
  inexact-partition mismatches its float tangent).
- BFGS hybrid eval: keep ALL values on ONE path (the fast call_batched one); mixing single-path
  reference f with fast-path line-search ft stalls Armijo (the bug fixed in 76127ca).
- Low-ell EE/TT chi2 carry a large additive constant (cancels in Delta-chi2); min chi2 ~1012 with
  low-ell vs ~607 plik-only is expected, not a bug.
- PYTHONPATH=$(pwd) inside every srun (else `import abcmb` hits the sibling ../ABCMB checkout).

## Commits (this session, on perk-perf; latest first)
93237c8 batched AD throughput finding (compile is the bottleneck)
1d74214 batched AD gradient WORKS end-to-end (1.75e-5 vs single-path)
794caf7 de-risk batched AD risk #1 cleared (1.55x staged)
6976771 design memo: batched AD on the per-k pipeline
0b11134 CHANGELOG: stall diagnosis + fix
76127ca fix BFGS line-search stall (consistent fast-path values)
0aee3b3 AD profile: precompute low-ell splines + production slurm
8645b53 frequentist hardening: AD-grad BFGS driver + low-ell TT/EE

## Untracked (not committed — decide later)
`scan/results/profile_prod_*.png|npz`, `scan/results/tol_sweep_*.npz` — the PREVIOUS session's
entry-(a) 6-POI production profile deliverables (binaries; left untracked by that session).

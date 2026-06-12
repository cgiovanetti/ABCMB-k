# TOOL_PLAN — the parameter-estimation tool (canonical plan, 2026-06-12)

## 0. Goal (verbatim from the user) and the competitive bar

> "I want a tool where I can give it a new cosmology ABCMB-style and in a few
> hours I have the optimum +/- 1/2 sigma."

**Competitive bar:** CLASS runs LCDM+Neff with Metropolis-Hastings in ~22 hours.
The tool must be close to that or faster. Node-days is not competitive.

**Decisions already made (do not re-litigate):**
- Frequentist profile likelihoods are the headline deliverable; the batched
  per-k pipeline (`Model.call_batched`) is the engine.
- Gradients ARE used — but for **convergence certificates** (the per-point
  ‖g‖∞<GTOL gate, PD check, multistart) and BFGS descent, NOT as a throughput
  play. Forward-mode AD with P directions costs about the same as a P-point FD
  stencil; its value is exactness (FD was measured 1–7% wrong at steep
  curvature, `scan/derisk_ad.py`).
- 13 grid points per POI was judged too coarse by the user. Points are nearly
  free on the batch axis → **NPTS ≥ 25** from now on.
- SMC (sequential Monte Carlo) on the SAME likelihood is the Bayesian
  anchor (review gap #2) and gives the evidence. Coverage mocks on the batch
  axis (gap #4) are the methodological selling point. Pretrained-emulator
  pitch is dropped.
- User constraints from earlier rounds still hold: rtol_large_k_PE=1e-5 for
  production (1e-4 biases the interval midpoint by 19% of sigma), no
  tol-loosening / fp32, no job arrays, k_chunk=100 primal.

**What already exists and works** (read `CHANGELOG.txt` 2026-06-11/12 + `scan/HANDOFF.md`):
- `scan/profile_prod_ad.py` — BFGS profile driver: AD gradients, Armijo line
  search with all VALUES on the fast `call_batched` path (consistency rule,
  commit 76127ca), ‖g‖ gate + PD check + multistart mode. Production-proven at
  NPTS=13 (entry-(a) ran 6 LCDM POIs in 2h49m / 3 nodes with the older Newton driver).
- `scan/batched_grad.py` — the staged batched-AD gradient
  (`staged_cl_and_grad`, `staged_chi2_and_grad`): jvp per already-jitted
  batched stage, threading tangents, NO core edits. CORRECT (8.58e-4 vs
  single-path on the chi2 gradient, `scan/validate_chi2_grad.py`). Compile is
  a ~5-min one-time-per-job tax (`bench/grad_compile_findings.md`). **Not yet
  sharded** — measured only at B=8 single-GPU, where it loses to the loop path.
- Likelihood: plik-lite TTTEEE (`scan/plik_lite.py`, A_planck envelope-profiled)
  + real low-ell TT/EE (`scan/lowl_like.py`), validated vs cobaya.

## 1. Cost model (what the plan is priced on)

Measured primal throughput (4×A100-80GB node, l_max=2508 lensed, per cosmology):
B=64 sharded → 1.13 s; B=512 sharded → 0.44 s (≈225 s wall per batch call).
Measured gradient (UNSHARDED, B=8, P=2, l2508): 52.3 s/cosmo warm = ~26
s/cosmo/direction ≈ 4–5× the single-GPU primal per direction.

Projected tool run (ΛCDM+Neff, D=7 ⇒ 7 POIs × NPTS=25 × ~2 starts ≈ 400
points, P=6 free dims per point), IF the sharded gradient hits 2–4× primal per
direction at B_local≥32:
- per BFGS iter ≈ P×(grad/dir) + ~3 value batches ≈ 7–12 s/point
- 400 points ⇒ 50–80 min/iter on ONE node; 10–15 warm-started iters
- ⇒ **8–20 h on one node, 2–5 h on 4 nodes.** Beats the 22 h bar with a
  stationarity certificate MH cannot provide.

**The load-bearing unknown is the sharded-gradient throughput.** Workstream A
measures it first; everything else is priced off that number.

**Fallback if AD-sharding disappoints** (> ~6× primal/direction): build the
gradient by central FD *in the batch axis* — 2P extra cosmologies per point per
iter ≈ 12×0.44 ≈ 5.3 s/point, same ballpark. The chi2 noise floor at fixed
shapes is measured ZERO (deterministic, `scan/noise_floor.py`), so FD is
viable; its known weakness is step-size sensitivity at steep curvature, so
keep the AD gradient for the FINAL gate check at the optimum even in this
fallback. Either branch meets the bar; do not stall on this decision.

## 2. Workstream A — shard the staged AD gradient + measure (FIRST, blocking)

Goal: `staged_cl_and_grad` / `staged_chi2_and_grad` run B-axis-sharded over all
visible GPUs, like `call_batched` already does. Recipe (see
`bench/driver_batched_wiring.md` §"Open: SHARDING"):

1. Extract the shardfn builder from `Model.call_batched` (abcmb/main.py:273–285)
   into a small helper — `Model._make_shardfn()` returning `(shardfn, n_dev)`
   (Mesh over `jax.devices('gpu')`, `NamedSharding(P('batch'))` on ndim≥1
   leaves, replicated for scalars) — and have `call_batched` use it. Pure
   refactor, no behavior change. (This repo is the playground; direct edits OK.)
2. In `scan/batched_grad.py`, add `shard=None` to `staged_cl_and_grad` and
   `staged_chi2_and_grad`. When sharding: pad the batch to a multiple of n_dev
   (replicate the last cosmology; slice padding off all outputs), then apply
   shardfn to (a) `params_batch` after stacking, (b) EACH `pdot` tangent dict,
   (c) `recomb` and `recomb_dot` after the CPU→GPU `device_put` (mirror
   `_build_bgs_batched`, abcmb/main.py:415–430). **Shard BEFORE the stages** —
   sharding after builds everything on device 0 and OOMs (proven for the primal).
3. Wire `PA_SHARD` through `batched_grad_fg` in `scan/profile_prod_ad.py`
   (the values path already shards; only the gradient path is new).

Gates:
- **Correctness:** extend `scan/validate_chi2_grad.py` to compare sharded vs
  unsharded at l_max=128, B=4, P=2 on a 4-GPU node. Expect agreement at
  ~1e-12 (GSPMD partitions the same program) — anything worse than 1e-6 means
  a sharding bug, stop and fix.
- **Throughput (the measurement):** extend `scan/grad_prod_shape.py` with a
  `GPS_SHARD` knob; run l2508/lensing=1/P=2 at B=32, 64, 128 on 4 GPUs, warm
  numbers. Report s/cosmo/direction vs the primal s/param at the same B.
  Decision: ≤4× primal/dir → proceed with AD gradients; >6× → flip the driver
  default to batch-axis FD (§1 fallback) and keep AD for the final gate only.
- Memory check: gradient peak was 4.07 GB at B=8 P=2 (~0.5 GB/cosmo). Expect
  B_local≈64–96 to be the comfortable per-device ceiling on the grad path;
  record the measured number in the CHANGELOG.

SLURM: ONE interactive allocation (`--gpus=4`), `PYTHONPATH=$(pwd)`,
`module load conda && conda activate actdr6` inside every srun. Never the
login node. Persistent compile cache: `JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache_abcmb`.

**RESULT (2026-06-12 — Workstream A DONE; CHANGELOG entry (c)).** Sharding is
correct: sharded-vs-unsharded chi2 1.5e-9, grad 3.3e-5 (GSPMD kernel/solver
noise, ~35× under the 1.16e-3 batched-AD-vs-truth floor; the gate threshold was
recalibrated to chi2<1e-6 / grad<5e-4 — the "~1e-12" expectation assumed
bit-identical partitioning, which an adaptive ODE solver does not give).
Throughput: **5.50 s/cosmo/direction at B=64** (B_local=16, the sweet spot;
4.87× the B=64 primal); grad path costs ~0.5–0.84 GB/cosmo/device; B_local=32
degrades (knee inferred from cold-time blowup — weak evidence, but immaterial).
**ORCHESTRATOR VERDICT (supersedes the §1 decision rule's open band):** the
4.87× landed "in between", and the implementing agent's "AD ≈ FD cost"
tie-break was an arithmetic slip — central FD costs 2 primal evals/direction
(2.26 s at B=64; 0.88 s riding B=512 batches), so FD is 2.4–6× CHEAPER per
direction than AD. Decision: **PA_GRADMETHOD=fdbatch (central FD on the batch
axis) for BFGS iteration gradients, with (a) an iteration-0 calibration of the
FD step against the AD gradient and (b) the FINAL stationarity gate ‖g‖<GTOL +
Hessian always from the exact AD gradient.** AD-only stays available via the
env knob (priced ~3.7 h/iter/node for 400 points vs ~40 min for fdbatch).

## 3. Workstream B — driver → tool (`scan/profile_prod_ad.py` evolves into it)

Target invocation experience: a config (python dict or small yaml) declaring
parameter names, fiducials, prior-ish widths (the SIG scaling), which are POIs,
the FIXED dict, optional `user_species` for new physics, and likelihood
toggles. Output per POI: grid, profiled chi2, best fit, 1σ/2σ PCHIP intervals,
per-point convergence status, multistart spread. One sbatch in, npz + png out.

Changes, in order of value:

1. **Generalize the parameter spec.** `ORDER`/`FIXED`/`CEN`/`SIG` are module
   constants today; lift them into a config object so ΛCDM+Neff (or +any
   ABCMB param, e.g. user_species params) is a config edit, not a code edit.
   `add_derived_parameters` already passes unknown keys through as jnp arrays.
   Neff specifically: it's already a FIXED key — just move it into ORDER with
   CEN=3.044, SIG≈0.2.
2. **One lockstep batch across ALL POIs × grid points × starts** (today: one
   POI per SLURM rank, B=NPTS within a rank). This is structurally clean even
   though different POIs have different free dims: the tangent dicts are
   per-cosmology anyway — direction j for batch element b is "the j-th free
   dim of b's POI" (P = D−1 is uniform). The BFGS state (x, H, f) is already
   per-point arrays; concatenating POIs just lengthens B. Multistart replicas
   are more rows in the same batch.
3. **NPTS=25 default** (user wants denser; cost on the batch axis is marginal).
4. **Warm starts:** initialize every point's nuisances at the global ΛCDM
   best-fit (entry-(a) values, in `scan/results/profile_prod_*.npz`); for
   extension runs, at the ΛCDM best-fit + extension fiducial. (Optional
   2-wave variant — inner grid points first, outer warm-started from them —
   only if iteration counts stay high; not needed for v1.)
5. **Keep batch shapes STABLE.** The staged-jvp compile is cached per (B,
   l_max) aval — every new B pays ~5 min. The driver already masks converged
   points (`active`); inactive points keep riding the batch (wasted compute
   but no recompile). Optional later: shrink in power-of-2 buckets. Do NOT
   shrink B per-iteration to the active count.
6. **Multi-node = slice the point list across nodes** (one slurm job,
   one task per node, each task shards its slice over its 4 GPUs — the
   `scan/scan_multinode.slurm` + `scan_slice.py` pattern). NO job arrays.

Gate: re-run the 6-POI ΛCDM profile with the new driver at NPTS=25 and check
the intervals reproduce entry-(a) (h 0.6763 [0.6702,0.6831], etc.) within the
quoted precision, now WITH per-point ‖g‖<GTOL convergence + multistart spread.

## 4. Workstream C — the headline benchmark: ΛCDM+Neff vs the 22 h bar

Run the §3 tool on ΛCDM+Neff (7 POIs × 25 pts, plik-lite + lowTT + lowEE,
rtol=1e-5, multistart on at least the Neff POI). Record wall-clock × nodes and
the full convergence evidence. Deliverable sentence: "full frequentist
profiles for ΛCDM+Neff, every point converged to ‖g‖<GTOL, in X h on Y
Perlmutter GPU nodes" with X×(Y normalized to 1 node) ≲ 22 h MH equivalent.
This number goes in the paper / README. Also closes review gap #2's
"extension" half when compared against the published Neff posterior.

## 5. Workstream D — SMC posterior on the same likelihood (Bayesian anchor)

Sequential Monte Carlo, tempered: particles move from prior to posterior
through π_β ∝ prior × L^β, β: 0→1 adaptively. Every step evaluates the
likelihood for ALL particles at once = one `call_batched` call. ~20–40
tempering stages × 2–5 MH moves ≈ 100–300 batched evals ⇒ ~3–8 h on one node
at N=512 particles. Output: full posterior + the evidence (free, from the
incremental weight normalizations) — model comparison for extensions.

**Design constraint (important):** the ABCMB likelihood is NOT end-to-end
jittable (GPU → CPU HyRex → GPU; jitting the whole pipeline is the known
~20-min monolith / hang). So do NOT use blackjax's jitted SMC loop on it.
**Hand-roll the SMC driver in eager Python** (~150 lines of numpy around
`fast_values`-style batched chi2):
- init N=512 particles from the prior (for a defensible evidence) — priors:
  flat boxes matching the profile SIG ranges is fine for the anchor;
- adaptive Δβ by bisection so resampled ESS ≈ 0.5N;
- systematic resampling;
- M=2–5 random-walk MH moves per stage, proposal = 2.38²/D × weighted particle
  covariance (recompute each stage); each move is ONE batched chi2 eval of all
  N proposals (accept/reject per-particle in numpy);
- log-evidence = Σ log mean(incremental weights);
- batch shape stays (N,) the whole run — one compile.
Reuse the driver's chi2 machinery (`fast_values` / `_chi2_of_cls`); no AD
needed. Validation: marginal means/σ vs the profile intervals and vs Planck
2018 chains (they should agree at the ~0.1σ level for ΛCDM; differences
profile-vs-marginal are themselves a result for Neff).

Do NOT pip-install into `actdr6`. blackjax is unnecessary under this design;
if some library becomes genuinely needed, the ONE allowed new conda env is the
route.

## 6. Workstream E — coverage mocks (later; the methodological flex)

Feldman–Cousins / Wilks validation: generate mock data vectors from the
best-fit Cls + the plik-lite covariance, put the MOCKS on the batch axis, and
re-fit hundreds of them in lockstep with the §3 driver (the likelihood
closure takes a per-batch-element data vector — small change: `pl.X_data`
becomes (B, ndata)). Empirical coverage of the Δχ² intervals = review gap #4
closed, and a capability nobody runs because it's normally too expensive.
Defer until A–C are done.

## 7. Gotchas (hard-won; violating these costs hours)

1. NEVER `jax.jit`/`vmap`/`jacfwd` the whole cross-device pipeline as one fn
   (GPU→CPU-HyRex→GPU fuses into a ~20-min monolith / hang). Stage everything.
2. `_to_float` every derived-param dict before any jvp (int `N_nu_massive` →
   None tangent mismatches the float primal partition).
3. Armijo consistency rule (commit 76127ca): ALL values (reference f,
   line-search ft, recorded profile) from ONE path — the fast `call_batched`
   path. AD/batched-grad supplies ONLY directions/curvature.
4. Shard BEFORE the setup stages; pad B to a multiple of n_dev; slice padding
   off outputs.
5. Batch-shape churn = compile churn (~5 min per new (B, l_max) aval). Pad and
   mask; keep shapes stable across iterations and jobs.
6. `PYTHONPATH=$(pwd)` inside EVERY srun (else `import abcmb` resolves to the
   sibling `../ABCMB` checkout and your edits silently do nothing).
7. Every shell: `module load conda && conda activate actdr6 && …`. Never run
   Python on the login node. ONE interactive allocation per session
   (`--gpus=4`, qos=interactive); scancel when done; never premium; no job arrays.
8. AD Hessian (fwd-over-fwd) OOMs at l2508 (~35 GB/cosmo) — curvature comes
   from BFGS history; the final nuisance Hessian is one central FD of the
   EXACT AD gradient.
9. Low-ell TT/EE chi2 carry a large additive constant — min chi2 ~1012 with
   low-ell vs ~607 plik-only is expected, not a bug.
10. Production rtol_large_k_PE=1e-5 (1e-4 biases interval midpoints 19% of σ;
    3e-5 is the quick-scan compromise at 3.2%).
11. `pl.profile_A` envelope-profiling uses `stop_gradient` — exact by the
    envelope theorem; don't "fix" it.
12. Memory: primal ~0.33 GB/cosmo/device (massless); grad path ~0.5 GB/cosmo
    measured at B=8 — re-measure under sharding before picking production B_local.

## 8. Order of work and session sizing

| # | Workstream | Size | Blocking? |
|---|-----------|------|-----------|
| 1 | A: shard gradient + MEASURE | 1 GPU session (~3–4 h incl compiles) | yes — prices everything |
| 2 | B: driver → tool (config, lockstep multi-POI, NPTS=25, warm starts) | 1–2 sessions | needs A's verdict (AD vs FD branch) |
| 3 | B-gate: ΛCDM reproduction at NPTS=25 | overnight sbatch (regular qos, mail flags) | — |
| 4 | C: ΛCDM+Neff headline run | overnight sbatch, 2–4 nodes | needs 2 |
| 5 | D: SMC driver + ΛCDM(+Neff) posterior | 1 session + overnight | independent of A/B (uses primal only) — parallelizable |
| 6 | E: coverage mocks | later | needs 2 |

Log every session in `CHANGELOG.txt` (reverse-chron). Update `scan/HANDOFF.md`
pointers if direction changes. Production sbatch files get
`--mail-type=ALL --mail-user=cgiovanetti@lbl.gov`.

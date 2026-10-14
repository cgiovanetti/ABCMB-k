# Handoff: ABCMB-k gradient speedups, two bug fixes, solver evaluations (2026-10-14)

Everything below lives in the worktree `/pscratch/sd/c/carag/ABCMB-k-grad` on three stacked
branches of the ABCMB-k repo (the branches are in ABCMB-k's `.git`, so they survive deleting the
worktree). Nothing in `/pscratch/sd/c/carag/ABCMB-k` (xe-c1, chunk 3's checkout) was touched.
Merge order after chunk 3 finishes: `grad-vmap-dirs` -> `pt-per-kchunk` -> `pe-step-ts`.
Review diffs: `scan/.grad_bench/{grad_vmap_dirs,pt_per_kchunk,pe_step_ts}.diff` (each against the
branch below it). Full measurements: the 2026-10-14 entries at the top of `CHANGELOG.txt`.

## 1. grad-vmap-dirs (118514f): one primal solve per gradient

`scan/batched_grad.py staged_cl_and_grad` looped the P directions and every `jax.jvp` re-solved
the full primal (1 + P primal solves). Now the directions are stacked and pushed through ONE
vmapped jvp per stage (`jvp_stage(..., n_dirs=)`), the primal Cls read off the same jvp.
B=8, 1 A100: 167 -> 34 s/cosmo (upstream jacfwd: 61). Gradients unchanged (8e-14 vs the
production run). Knob: `PA_GRAD_DIRCHUNK` (directions per jvp; 0 = all, the default; chunking
re-solves the primal and is a poor trade).

## 2. pt-per-kchunk (049b353 .. 2a46965): memory, so bigger batches fit

- `abcmb/perturbations.py`: `full_evolution_batched` builds the output table per k-chunk
  (`_table_chunk`) and never materialises the (B, Ny, Nlna, N_k) modes tensor;
  `make_output_table{,_batched}` take an optional `k`. Snapshot tests pass; value path identical.
- `scan/batched_grad.py`: the spectrum jvp is the memory peak (~1.2 GiB/cosmology at P=7:
  f64[P,B,3079,3007] lensing products) but 2.5% of the time, so it runs on sub-batches of
  `spec_b_local` cosmologies per device (driver `PA_GRAD_SPECB`, default 4). No extra work.
- Driver defaults: `PA_AD_BCHUNK` 64 (B_local 16 at 9 GB/device; 32 is 14 GB, 64 is 28 GB).
- Result: 34 (B=8) -> 24.8 (16) -> 24.1 (32) -> 22.0 (64) s/cosmo; 85% is the perturbation jvp.
- `BG_MEMLOG=1` prints per-stage GB and seconds; `scan/.grad_bench/hlo_stage5.py` lists the
  largest HLO buffers of the spectrum jvp.

## 3. pe-step-ts (3ce9c0e, f13cfc7): accuracy fix + HyRex fix

### 3a. BUG: production saves the modes through in-step interpolation (also upstream)

`SaveAt(ts=lna)` with Kvaerno5 evaluates its third-order Hermite interpolant inside each accepted
step; across the wide adaptive steps that is 2.3e-3 (peak-normalised) in ClTT/ClTE at low ell and
4e-4 relative above ell 30 against a tight reference (Kvaerno5 rtol 1e-7 stepped onto the grid;
GRADSOLVE's independent Rodas5P agrees with the reference to 7e-7). It is the known "+2.5e-3
low-ell offset vs CLASS". Evidence: `scan/.grad_bench/gs/interp_check.py` + log.
Fix: spec `step_ts_PE` (default True) -> `step_ts=stop_gradient(lna[:-1])` on `_FrozenMeshPID`,
a `PIDController` subclass whose controller decisions carry no tangent, plus
`t0=stop_gradient(lna_start)`: the step mesh is data under forward-mode AD (as in modax and
record-and-replay adjoints). This is REQUIRED, not cosmetic: a step clipped onto a save point can
be tiny, its error norm ~0, and the differentiated controller's `err**(-1/order)` tangent is
non-finite ("linear solver received non-finite input" in the staged jvp).
- CLASS (pytests scenario): TT at ell=2 0.43% -> 0.17%, max TT 0.43% -> 0.18%, EE unchanged.
- AD vs central FD (`fd_check.py`, h=0.01 sigma): omega_b 2.5e-3 -> 2.0e-4, omega_cdm
  2.4e-3 -> 9.8e-4, Neff 3.2e-3 -> 2.2e-3 (old path -> new). More consistent, not less.
- Cost: +12% perturbation steps (median 594 -> 790/mode); value 5.4 -> 6.05 s/cosmo (B=32);
  gradient 24.8 -> 25.4 s/cosmo (B=16).
- TO DO when merging: regenerate `pytests/fixtures/snapshots.npz` (values move by the numbers
  above, test_snapshots will fail until then). `step_ts_PE=False` restores the old saves.
- Untested combination: `step_ts_PE=False` now also uses the frozen-mesh controller (AD
  semantics change slightly in that mode too); the FD comparison's "old path" column used the
  pre-branch gradient, not this combination.
- Upstream ABCMB has the same code: PR candidate (user opens PRs).

### 3b. HyRex jvp sharded over host devices

The HyRex stage of the gradient was 22 s at B=16: not the primal (0.85 s) but the 7-direction
jvp, one serial CPU program over all B x P lanes, and it does NOT shrink when more GPUs share the
batch (~140 s per 64-row block on a 4-GPU node vs ~370 s of perturbations). The old wrapper
(`_recmodel_cpu`, `filter_jit(..., backend='cpu')`) silently gathers sharded inputs onto device 0.
`scan/batched_grad.py hyrex_cpu_mesh`: a plain `filter_jit` on inputs placed on a
`NamedSharding` over `n_use` = largest divisor of B <= number of host devices; mesh/shardings
cached per (RecModel, n_use) (a fresh Mesh per call is a fresh jit key = recompile).
- Needs `XLA_FLAGS=--xla_force_host_platform_device_count=N` before JAX starts; the driver
  sets it from `PA_CPU_DEVICES` (default 16; 0/1 = old single-device path; `BG_HYREX_SHARD=0`
  also disables it). The value path keeps the single-device wrapper (its primal is faster there).
- In the pipeline the stage takes TWO calls to settle: 65 s, 22 s, then 4.3 s per gradient call
  at B=16 on 16 devices (standalone 22 -> 4.3 s). A profile run gets the 5x from iteration 3 on.
- Outputs change at ~1e-7 relative (CPU partitioning of the HyRex solve), i.e. the known
  batch-composition noise level.

## 4. Solver evaluations (no code to merge)

- GRADSOLVE (`/pscratch/sd/c/carag/gradsolve`): not a speedup. Fused kernels need dim<=64 and a
  scalar-arithmetic RHS; its JAX engines are the same lockstep vmapped loop as diffrax (Rodas5P
  record 51 s + replay 42 s vs 37 s production, one cosmology). Its accuracy exposed 3a.
- modax (`/pscratch/sd/c/carag/modax`, env `/global/common/software/m3166/carag/envs/modax`,
  py3.11 + jax 0.10.2 cuda13, the one allowed new env): one trajectory per CUDA thread, Rodas5P +
  Enzyme Jacobians + sparsity, forward sensitivities. Tests: 63 passed, 1 error on Perlmutter.
  Synthetic 46-variable hierarchy (`scan/.grad_bench/modax/synth_bench.py`, stiffer than ABCMB,
  solvers disagree at 7e-2 on it, timings indicative): forward 5.3 s/cosmo at 572 trajectories,
  0.46 s/cosmo at 16 cosmologies (vs 5.4 real value path); 7-direction sensitivities 13.3 s/cosmo
  at 16 cosmologies (29x its forward; vs ~20 s/cosmo for ABCMB-k's perturbation jvp); diffrax on
  the same toy 78 s/cosmo. So modax would make batched VALUES ~10x cheaper and gradients ~1.5x,
  which favours the surrogate/hybrid profiler over the pure gradient driver.
  Port constraints: RHS as fixed-size scalar tuples with constant indices (generate the
  hierarchy unrolled); background (x_e, T_m, tau; aH analytic) as scalar fit coefficients in the
  per-trajectory parameter row (closed-over tables = 64 KiB constant memory baked at build, and
  Enzyme crashed on a runtime-indexed one); `exp` not `tanh` (no Enzyme rule); shared `t_span`
  (start every mode at the earliest per-k start); the jvp rule integrates one sensitivity column
  per PARAMETER (`sens_param_columns`), so a background in p needs a small modax fork for
  per-tangent-direction seeds (`modax/_sensitivity.py`). Kernel compiles: ~2.5 min forward,
  ~7 min with sensitivities. Estimate: 1-2 weeks of focused work.

## 5. Harness files (this directory)

`bench_fvu.py` (copy of the fvu bench, ABCMB-k path, knobs DIR_CHUNK/SPEC_B/SHARD/LENSING/
STEP_TS), `hyrex_bench.py`, `hlo_stage5.py`, `fd_check.py`, `gs/gs_bench.py`,
`gs/interp_check.py`, `gs/accuracy_stepts.py`, `modax/synth_bench.py`; logs `log_*.log`.

# scan/: frequentist profile likelihoods with ABCMB-k

One driver (`profile_prod_ad.py`) profiles every parameter of interest (POI) of a config on a grid,
with Levenberg-Marquardt on Gauss-Newton curvature from forward-mode AD through the batched engine.
Deliverables (LCDM + Neff, Planck 2018 full plik TTTEEE + low-ell Commander TT and SRoll2 EE, one massive neutrino, l_max 2508):

| tag | data / BBN | Neff best [68%] [95%] | min chi2 | run |
|---|---|---|---|---|
| `_v3` | CMB, YHe fixed, POI Neff | 2.818 [2.596, 3.052] [2.395, 3.287] | 2759.06 | job 60025795, 3h52m on 4 nodes |
| `_v4linx` | CMB, YHe/Neff/T_nu from LINX, POI Delta_Neff_init | 2.850 [2.659, 3.050] [2.484, 3.250] | 2759.10 | job 60057736, 5h24m on 4 nodes |
| `_v5bbn` | `_v4linx` + Y_p (Aver 2026) + D/H (Cooke 2018) | 2.966 [2.886, 3.047] [2.809, 3.126] | 2761.09 | job 60388582, 6h46m on 4 nodes |

All seven POIs per run, numbers and convergence counts: `CHANGELOG.txt`. `_v5bbn` is biased: its BBN
nuisance priors sat at the config cen (the prepass fit) instead of 1 and 0. `_v6bbn` (same config, priors at
their fixed means, recomb lna grid) replaces it.

## Files

| file | role |
|---|---|
| `profile_prod_ad.py` | the driver: config, rows, LM lockstep, multistart, intervals, npz/png output |
| `lm_tools.py` | pure numpy: projected LM step, Newton decrement, spread starts, clustering/flags, union of regions |
| `batched_grad.py` | `staged_cl_and_grad`: C_ell and their tangents, one `jax.jvp` per batched engine stage |
| `plik_full.py` | full plik via clipy; 21 nuisances profiled per cosmology at fixed theory (bounded SPG) |
| `lowl_like.py` | JAX Planck 2018 low-ell TT (Commander) and EE (SRoll2) likelihoods |
| `bbn_stage.py` | `LinxStage`: batched LINX values and Jacobians on the host CPU (config `bbn: linx`) |
| `compare_profile_runs.py` | overlays runs, lower envelope, parabola fit (CPU only) |
| `configs/lcdm_dneff_linx_plikfull.py` | `_v4linx` config |
| `configs/lcdm_dneff_linx_bbnlike_plikfull.py` | `_v5bbn` config (cen/grid_sig = the joint fit of job 60327767) |
| `configs/toy_bimodal.py` | toy objective for `pytests/test_profile_toy.py` |
| `profile_prod_linx_dneff.slurm`, `profile_prod_linx_bbnlike.slurm` | the `_v4linx` / `_v6bbn` launchers |
| `results/` | outputs (git-ignored) |

## Reproducing

```bash
cd /pscratch/sd/c/carag/ABCMB-k
sbatch scan/profile_prod_linx_dneff.slurm      # _v4linx -> logs/abcmb_pf_linx_<jobid>.out
sbatch scan/profile_prod_linx_bbnlike.slurm    # _v6bbn  -> logs/abcmb_pf_bbn_<jobid>.out
```

Each launcher takes 4 GPU nodes on `regular` (account m5403_g), one task per node with `PA_POI_SLICE=1`
(two POIs per node, n_s alone), and sets the environment: `module load conda; conda activate actdr6`,
`PYTHONPATH` = repo root + `/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy`,
`JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache_abcmb`, `EQX_ON_ERROR=nan`. A walltime timeout is resumed
by resubmitting the same command (done POIs are skipped; the row table and LM state are checkpointed).
`_v3` used `configs/lcdm_neff_plikfull_rc.py` and `profile_prod_plikfull_neff48.slurm` with the driver
of that time; they exist only in the tag `pre-trim-2026-11-06` (set the launcher's account to m5403_g).

Outputs: `results/profile_prod_ad_<POI><TAG>.{npz,png}`. The npz holds `poi_grid`, `chi2` (the lower
envelope), `xstar`, `sigma1` (lo, best, hi), `sigma2`, `cl95`, `regions1/2/95`, the convergence labels,
the flags, the full row table (`rows_*`) and, for LINX runs, `derived_bbn` (YHe, Neff, T_nu, Y_p, D/H)
plus `derived_neff` / `sigma1_neff` / `cl95_neff` on the Delta_Neff_init profile. Comparison figure:

```bash
python scan/compare_profile_runs.py _v3 _v4linx _v5bbn   # -> results/profile_neff_runs_compared_v3_v4linx_v5bbn.png
```

CPU tests of the machinery (no ABCMB, seconds), on an allocation, never the login node:

```bash
salloc --no-shell --nodes=1 --qos=interactive --constraint=cpu --time=00:30:00 --account=m5403
srun --jobid=<id> --ntasks=1 --cpus-per-task=16 bash -c 'module load conda && conda activate actdr6 &&
  cd /pscratch/sd/c/carag/ABCMB-k && export PYTHONPATH=$(pwd):$PYTHONPATH JAX_PLATFORM_NAME=cpu &&
  python -m pytest -q pytests/test_lm_tools.py && python -m pytest -q pytests/test_profile_toy.py'
```

## Method

A row is one POI fixed at one grid value with the other parameters free, in scaled coordinates
x = (theta - cen) / sig. All rows of a rank's POIs run in one lockstep batch; converged and stalled rows
leave it, and batch sizes are padded to NDEV x 2^k so few shapes compile.

Each iteration computes, per active row, the C_ell and their tangents along every free direction
(`batched_grad.staged_cl_and_grad`: pre-recombination, HyRex on a host-device mesh, background,
perturbations with the directions vmapped, spectrum sub-batched by `PA_GRAD_SPECB`), and from them the
chi2 gradient and the Gauss-Newton matrix J^T W J at the row's own point:
- plik: the residual and its data covariance; the floated nuisances, profiled per cosmology at fixed
  theory, enter through the Schur complement H_tt - H_tn H_nn^-1 H_nt (gradient by the envelope theorem;
  nuisances on a bound held fixed);
- low-ell TT/EE: their exact AD Hessian in D_ell with negative eigenvalues set to zero. They take plik's
  calibration A_planck (as in cobaya), so A_planck is profiled over plik + low-ell together and their
  A_planck curvature and cross terms enter the Schur complement;
- BBN data (config `bbn_like`): the Y_p and D/H residual Jacobians from the LINX stage, plus the priors.

`PA_LM_K` damping values a decade apart are tried in one batched value call (steps projected onto the
+-`PA_XBOX` box, capped at `PA_STEP_MAX`); a row takes the best step whose chi2 drop is at
least 1e-4 of the GN prediction, otherwise it raises its damping and is stalled after `PA_STALL_MAX`
failures. Every acceptance test uses `call_batched` values; there are no finite differences anywhere.
Gauss-Newton / Fisher scoring with LM damping follows MSTW 2008 (arXiv:0901.0002), Bond, Jaffe & Knox
(astro-ph/9708203) and im3shape (arXiv:1302.0183).

Smooth theory: the save / line-of-sight lna grid is `lna_grid_mode="recomb"` (perturbations.py): a base grid
fixed by physics constants whose recombination core moves with the cosmology's visibility mean, with the
map's Jacobian as LoS weights, so the nodes are smooth in theta and tied to no reference cosmology.
`PA_LNA_GRID=fixed` restores the visibility grid frozen at the config centre (`Model.freeze_lna_grid`). The
visibility-adaptive grid rebuilt per row moved its nodes jaggedly. The engine side carries Hermite tau and
exp(-kappa) lookups with exact slopes, the visibility derivative from d exp(-kappa)/d lna = exp(-kappa)/(tau_c aH),
C^1 x_e and j_l lookups, Tsit5 optical depth, Debye j_l through u_3 at
large x and `dtmax_PE` 0.03 for k < 0.06, so AD tangents agree with finite differences.

LINX (config `bbn: linx`): the POI is Delta_Neff_init, Delta Neff at the START of BBN (Neff ~ 3.044 +
0.262 x; the true Neff is read back per row). `LinxStage` computes (YHe, Neff, T_nu, Y_p, D/H) and their
Jacobian for all rows of a call on one host device (vmapped in chunks of 64; batched two-jvp Jacobian,
lanes that fail recomputed alone); the engine runs with `bbn_type=""` and receives y + J (theta - theta_row),
so the staged jvp carries LINX's exact tangent. With `bbn_like`, tau_n_fac and q0..q11 (key_recommended
rate shifts) are appended to the order and profiled in every row.

Multimodality: `PA_MS_K` spread starts at `PA_MS_NPTS` grid points per POI, +-`PA_MS_SPREAD` profile sigmas
along the softest GN eigendirections. A point is flagged for a stalled start, converged starts more than
`PA_MS_DX` apart, or starts disagreeing by more than `PA_MS_DCHI2`. Each distinct spread-start minimum
seeds a branch (an LM run at every grid point of its POI). The profile is the lower envelope of all rows.

Stopping: a row converges at max|g| < `PA_GTOL`, or when it stalls with Newton decrement 0.5 g'H^-1 g
below `PA_DTOL` ("floor": the likelihood's jitter, e.g. D/H at ~5e-5, is above what GTOL demands along a
stiff direction). The run stops when every POI's 68% ends moved less than `PA_SIGTOL` x grid_sig over
`PA_SIGTOL_PATIENCE` iterations (after the spread starts finish), or at `PA_MAXIT`.

Intervals: a degree-`PA_FIT_DEG` polynomial through the points within `PA_FIT_DCHI2` of the minimum, or
pchip through every point when the profile shows two minima; the interval is the union of the regions
below the chi2_1 quantiles: 1 (68.27%), 3.84 (95%, the quoted 95% intervals) and 4 (95.45%, `sigma2` in the npz). A Delta_Neff_init profile is also mapped to Neff through each
row's derived Neff (`derived_neff`, `sigma1_neff`, `cl95_neff`, and the "as Neff" log line).

MLE prepass (`PA_MLE_PREPASS=1`): the same LM with every parameter free from the config cen; the grids are
recentred on the joint fit with Fisher profile sigmas sqrt(2 (H^-1)_ii) sig, clamped to
[1/`PA_MLE_SIG_CLIP`, `PA_MLE_SIG_CLIP`] x sig, and rows warm-start there. Use it when the config centre
is not the best fit, then copy the result into the config and run production with it off (as for `_v5bbn`).
Copying a fit into cen must not change the likelihood: re-evaluate chi2 at the previous best fit first (the
BBN nuisance priors have fixed means, 1 and 0, whatever cen holds).

## Environment knobs

| knob | default | meaning |
|---|---|---|
| `PA_CONFIG` | required | config file, relative to the cwd (the launchers `cd` to the repo root) |
| `PA_TAG` | `''` | suffix of every output and checkpoint |
| `PA_POIS` | the config order (no BBN nuisances) | comma-separated POIs |
| `PA_POI_SLICE` | 0 | 1: rank r of n profiles `POIS[r::n]` and writes its own checkpoints |
| `PA_LMAX` | 2508 | engine l_max |
| `PA_NPTS` / `PA_NSIG` | config `npts` / `nsig` | grid points per POI / half-span in config `grid_sig` units |
| `PA_XBOX` | 5 | free-dim box in scaled sigmas; a row on it is not converged (launchers 10) |
| `PA_MAXIT` | 40 | LM iterations (launchers 30) |
| `PA_GTOL` | 0.03 | gradient test, max-norm in scaled coordinates |
| `PA_DTOL` | 1e-3 | noise floor for stalled rows; 0 = off |
| `PA_SIGTOL` / `PA_SIGTOL_PATIENCE` | 1e-2 / 3 | interval-stability stop; `PA_SIGTOL<=0` = off |
| `PA_RTOL` | 1e-5 | `rtol_large_k_PE` (atol = 1e-2 rtol) |
| `PA_LNA_GRID` | recomb | save / LoS grid: `recomb` (moves smoothly with recombination) or `fixed` (frozen at the config centre; every result before 2026-11-07) |
| `PA_RESUME` | 1 | skip done POIs, resume from the checkpoints |
| `PA_VAL_CHUNK` | 256 | max cosmologies per value call |
| `PA_AD_BCHUNK` | 64 | max rows per gradient block (`_v5bbn` launcher: 32) |
| `PA_GRAD_SPECB` | 4 | cosmologies per device per spectrum-stage jvp |
| `PA_LM_K` | 4 | damping values per iteration |
| `PA_STEP_MAX` | 1.0 | step cap, max-norm, scaled sigmas |
| `PA_STALL_MAX` | 3 | failed iterations before a row is stalled |
| `PA_WARM_LR` | 0 | start rows at the warm point's linear response (launchers 1) |
| `PA_TRACE_EVALS` | 0 | log every batch before evaluating it (launchers 1) |
| `PA_MS_K` / `NPTS` / `SPREAD` / `DX` / `DCHI2` / `WAIT` / `BRANCH_MAX` | 3 / 6 / 2.0 / 0.5 / 0.1 / 1 / 4 | multistart (above) |
| `PA_BR_MAXIT` | 10 | LM iterations of a branch |
| `PA_FIT_DEG` | 4 | interval polynomial degree |
| `PA_FIT_DCHI2` | 9 | fit window above the minimum; also the two-minima test's range |
| `PA_CPU_DEVICES` | 16 | host devices for the HyRex jvp (0 or 1 = one device) |
| `PA_MLE_PREPASS` | 0 | joint fit first, grids recentred on it |
| `PA_MLE_MAXITER` | 40 | LM iterations of the prepass |
| `PA_MLE_SIG_CLIP` | 3.0 | clamp of the prepass grid sigmas |
| `PLF_MAXIT` | 3000 | SPG iterations of the plik nuisance profile (`plik_full.py`) |

## Gotchas

- Never `jit` / `vmap` / `jacfwd` the whole cross-device pipeline (GPU, CPU HyRex, GPU) as one function: it
  compiles for ~20 min or hangs. Differentiate stage by stage, as `batched_grad.py` does.
- `_to_float` every derived-parameter dict before a jvp: an integer leaf (`N_nu_massive`) gets no tangent
  and the tangent tree no longer matches the primal.
- Shard before the setup stages, pad B to a multiple of the device count, slice the padding off.
- Every new batch shape compiles (minutes for the AD path, cached in `JAX_COMPILATION_CACHE_DIR`); keep
  shapes stable across iterations and jobs.
- `PYTHONPATH` must start with the repo root in every srun: `actdr6` has the sibling
  `/pscratch/sd/c/carag/ABCMB` editable-installed, and without it `import abcmb` loads that tree and edits
  here do nothing. A script's own directory is `sys.path[0]`, so run `python scan/<x>.py` from the root
  with the root on `PYTHONPATH`; when in doubt print `abcmb.__file__`.
- The low-ell likelihoods are -2 ln L up to an additive constant, so absolute chi2 values are offset: a
  minimum near 2759 (full plik + low-ell) is the expected scale; only Delta chi2 is used.
- Production `rtol_large_k_PE` is 1e-5 (1e-4 shifts interval midpoints by 19% of sigma).
- GPU memory scales with rows per GPU x AD directions: shake out a launcher in `--qos=debug` with one
  production-size gradient block (the first `_v5bbn` job, 60327767, OOMed at 16 rows x 19 directions).

## Open items

- The 17 `_v5bbn` stalls are harmless: Newton decrements 2e-6 .. 2.3e-4, below `PA_DTOL`.
- Wilks / coverage calibration deferred (only needed if a second mode appears within a few Delta chi2).
- 2D joint profiles for a few parameter pairs.
- Vendored LINX (`abcmb/linx`) and LINX main have diverged: the inert-radiation normalization fix and the
  dense-grid / log-log output are not in LINX main yet (`/pscratch/sd/c/carag/LINX/rho_extra_norm.diff`,
  `dense_grid_loglog.diff`), and main's finite t0 / `SaveAt(t0, t1, dense)` is not in the vendored copy.
- The EM-coupled WIMP is not implemented (needs a LINX thermodynamics extension exposed as a stage input).
- `pytests/fixtures/snapshots.npz` predates the vendored-LINX changes: regenerate it on GPU.

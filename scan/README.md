# scan/: frequentist profile likelihoods with ABCMB-k

One driver (`profile_prod_ad.py`) profiles every parameter of interest (POI) of a config on a grid,
with Levenberg-Marquardt on Gauss-Newton curvature from forward-mode AD through the batched engine.
Deliverables (LCDM + Neff, Planck 2018 full plik TTTEEE + low-ell Commander TT and SRoll2 EE (SimAll in `_v7simall`), one massive neutrino, l_max 2508):

| tag | data / BBN | Neff best [68%] [95%] | min chi2 | run |
|---|---|---|---|---|
| `_v3` | CMB, YHe fixed, POI Neff | 2.818 [2.596, 3.052] [2.395, 3.287] | 2759.06 | job 60025795, 3h52m on 4 nodes |
| `_v4linx` | CMB, YHe/Neff/T_nu from LINX, POI Delta_Neff_init | 2.850 [2.659, 3.050] [2.484, 3.250] | 2759.10 | job 60057736, 5h24m on 4 nodes |
| `_v5bbn` (biased) | `_v4linx` + Y_p (Aver 2026) + D/H (Cooke 2018); nuisance priors at the config cen | 2.966 [2.886, 3.047] [2.809, 3.126] | 2761.09 | job 60388582, 6h46m on 4 nodes |
| `_v6bbn` | `_v5bbn` with the nuisance priors at 1 / 0, recomb lna grid | 2.982 [2.902, 3.064] [2.825, 3.143] | 2763.15 | job 60988361, 4h56m on 4 nodes |
| `_v7simall` | `_v4linx` with SimAll lowE (the Planck 2018 base_nnu likelihood), recomb lna grid, MLE prepass | running | | job 61091033 |
| `_v1act` | LCDM (Neff 3.044), ACT DR6 lite + SRoll2 lowE, YHe from the sBBN table, l_max 6325 + 5000 | queued | | job 61111914, 6 nodes |

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
| `act_lite.py` | ACT DR6 TT/TE/EE CMB-only (candl); A_act, P_act profiled per cosmology at fixed theory (config `high_ell: act_lite`) |
| `lowl_like.py` | JAX Planck 2018 low-ell TT (Commander) and EE (SRoll2 or SimAll: config `lowee`) likelihoods |
| `bbn_stage.py` | `LinxStage`: batched LINX values and Jacobians on the host CPU (config `bbn: linx`) |
| `compare_profile_runs.py` | overlays runs, lower envelope, parabola fit (CPU only) |
| `profiles.ipynb` | self-contained notebook: loads `results/` and plots 1D / 2D profiles, nuisances, convergence |
| `configs/lcdm_dneff_linx_plikfull.py` | `_v4linx` config |
| `configs/lcdm_dneff_linx_simall_plikfull.py` | `_v7simall` config: `_v4linx` with `lowee: "simall"` |
| `configs/lcdm_dneff_linx_bbnlike_plikfull.py` | `_v6bbn` config (cen/grid_sig = the joint fit of job 60327767) |
| `configs/lcdm_mchi_dneff_linx_bbnlike_plikfull.py` | EM-coupled WIMP: `_v6bbn` + log10_m_chi (1D `_v1emwimp`, 2D `_v1emwimp2d`) |
| `configs/lcdm_actlite.py` | LCDM with ACT DR6 lite + SRoll2 lowE (`_v1act`), one 0.06 eV neutrino, YHe from the sBBN table |
| `configs/toy_bimodal.py` | toy objective for `pytests/test_profile_toy.py` |
| `configs/toy_bounded.py` | toy objective for `pytests/test_profile_bounds.py` (bounds, one-sided limits, the box) |
| `profile_prod_linx_dneff.slurm`, `profile_prod_linx_bbnlike.slurm` | the `_v4linx` / `_v6bbn` launchers |
| `profile_prod_emwimp.slurm` | the EM-WIMP launcher (1D; 2D with `PA_POI2D`) |
| `profile_prod_act.slurm` | the ACT launcher (`_v1act`; candl from the venv `envs/abcmb-k`) |
| `diag/` | ACT l-range checks (`act_lmax_check.py`: delta_l_max / hierarchy / truncation vs the ACT paper's CLASS range), `act_lite_check.py` (= candl), `lensing_stream_check.py` |
| `results/` | outputs (git-ignored) |

## Reproducing

```bash
cd /pscratch/sd/c/carag/ABCMB-k
sbatch scan/profile_prod_linx_dneff.slurm      # _v4linx -> logs/abcmb_pf_linx_<jobid>.out
PA_CONFIG=scan/configs/lcdm_dneff_linx_simall_plikfull.py PA_TAG=_v7simall PA_MLE_PREPASS=1 \
    sbatch scan/profile_prod_linx_dneff.slurm  # _v7simall (SimAll lowE)
sbatch scan/profile_prod_linx_bbnlike.slurm    # _v6bbn  -> logs/abcmb_pf_bbn_<jobid>.out
sbatch scan/profile_prod_emwimp.slurm         # _v1emwimp (EM WIMP, 1D) -> logs/abcmb_pf_emwimp_<jobid>.out
# the 2D profile: set the pair in the environment (sbatch --export would split it at the comma)
PA_POI2D=log10_m_chi,Delta_Neff_init PA_TAG=_v1emwimp2d PA_AD_BCHUNK=32 sbatch scan/profile_prod_emwimp.slurm
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
python scan/compare_profile_runs.py _v3 _v4linx _v6bbn   # -> results/profile_neff_runs_compared_v3_v4linx_v6bbn.png
```

CPU tests of the machinery (no ABCMB, seconds), on an allocation, never the login node:

```bash
salloc --no-shell --nodes=1 --qos=interactive --constraint=cpu --time=00:30:00 --account=m5403
srun --jobid=<id> --ntasks=1 --cpus-per-task=16 bash -c 'module load conda && conda activate actdr6 &&
  cd /pscratch/sd/c/carag/ABCMB-k && export PYTHONPATH=$(pwd):$PYTHONPATH JAX_PLATFORM_NAME=cpu &&
  python -m pytest -q pytests/test_lm_tools.py pytests/test_profile_toy.py pytests/test_profile_bounds.py'
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

ACT (config `high_ell: act_lite`, `scan/act_lite.py`): ACT DR6 TT/TE/EE CMB-only through candl's data, the
residual and covariance in plik_full's interface, A_act (prior N(1, 0.003)) and P_act (flat [0.9, 1.1]) profiled per
cosmology by a 2-parameter Newton and entering the Schur complement like plik's nuisances; the low-ell likelihoods
take calibration 1. Engine for it (overridable by config `engine`): lensed l_max 6325 (the kept windows end there),
delta_l_max 5000 (unlensed and phiphi to 11325, the ACT paper's CLASS range: chi2 within 0.006 of it, while 1800 is up
to 0.6 off), photon hierarchy 40 / 60, HALOFIT, computed j_l, `lensing_streamed` (the matrix lensing's jvp needs ~60
GB per row). Config `bbn: table` takes YHe(omega_b) from the sBBN table. The venv
`/global/common/software/m3166/carag/envs/abcmb-k` (actdr6 + candl) is needed; `profile_prod_act.slurm` activates it.

LINX (config `bbn: linx`): the POI is Delta_Neff_init, Delta Neff at the START of BBN (Neff ~ 3.044 +
0.262 x; the true Neff is read back per row). `LinxStage` computes (YHe, Neff, T_nu, Y_p, D/H) and their
Jacobian for all rows of a call on one host device (vmapped in chunks of 64; batched two-jvp Jacobian,
lanes that fail recomputed alone); the engine runs with `bbn_type=""` and receives y + J (theta - theta_row),
so the staged jvp carries LINX's exact tangent. With `bbn_like`, tau_n_fac and q0..q11 (key_recommended
rate shifts) are appended to the order and profiled in every row.
Config `linx_chi = {"g_chi": 4, "boson": False}` adds an EM-coupled thermal relic chi of mass
10**`log10_m_chi` MeV, a LINX-only parameter like Delta_Neff_init: its annihilation heats the photons relative
to the neutrinos (abcmb/linx thermo.rho_chi, background.dY, taken from em_dm; the background then solves with
Kvaerno5, 1024 steps). Above ~30 MeV the model is LCDM.

2D profiles (`PA_POI2D=a,b`): one LM row per point of the (a, b) grid, every other parameter free, started
at the config centre (with `PA_WARM_LR`, at the better of it and the warm point's linear response); no spread
starts or branches. With `PA_POI_SLICE` each rank takes every NPROC-th a value and writes
`results/profile_prod_ad_2d_<a>_<b><TAG>_part<k>of<n>.npz` (grid_a, grid_b, ja, chi2 / converged / at_box / xstar /
theta / derived_bbn on its rows); `profiles.ipynb` stitches the parts. Contours at the 2-parameter thresholds.

Multimodality: `PA_MS_K` spread starts at `PA_MS_NPTS` grid points per POI, +-`PA_MS_SPREAD` profile sigmas
along the softest GN eigendirections. A point is flagged for a stalled start, converged starts more than
`PA_MS_DX` apart, or starts disagreeing by more than `PA_MS_DCHI2`. Each distinct spread-start minimum
seeds a branch (an LM run at every grid point of its POI). The profile is the lower envelope of all rows.

Stopping: a row converges at max|g| < `PA_GTOL`, or when it stalls with Newton decrement 0.5 g'H^-1 g
below `PA_DTOL` ("floor": the likelihood's jitter, e.g. D/H at ~5e-5, is above what GTOL demands along a
stiff direction). The run stops when no row is active (each converged, stalled, on the floor or held by the
box) or at `PA_MAXIT`; the log line `stop at itN: ...` and the npz `stop_reason` say which. `PA_SIGTOL > 0`
adds an interval-stability stop (every POI's 68% ends moved less than `PA_SIGTOL` x grid_sig over
`PA_SIGTOL_PATIENCE` iterations, after the spread starts finish): off by default, it is for a likelihood too
rough for rows to converge or stall, and certifies no row.

Bounds and the box (config, both optional): `bounds = {name: (lo, hi)}` (None = open) are physical limits,
hard for the optimizer, and a POI's grid is kept inside them (a span crossing a bound is shifted onto it,
width kept). `box = {name: (lo, hi)}` in scaled sigmas (None = -+`PA_XBOX`) is the optimizer box of a free
parameter, inside its bounds. A row on a box face is converged when chi2 is flat outward there (outward slope
below `PA_GTOL`: a direction the data leave open); when chi2 still falls outward the box holds it (`at_box`,
not converged): widen the box or give a bound. A physical bound is a real limit and never counts as the box.

Intervals: a degree-`PA_FIT_DEG` polynomial through the points within `PA_FIT_DCHI2` of the minimum, or
pchip through every point when the profile shows two minima or is still falling at a grid end that is not a
bound; the interval is the union of the regions below threshold. Without a bound the thresholds are the
chi2_1 quantiles: 1 (68.27%), 3.84 (95%, the quoted 95% intervals) and 4 (95.45%, `sigma2` in the npz). Next to
a bound they are the boundary-aware asymptotic thresholds of Cowan, Cranmer, Gross & Vitells
(arXiv:1007.1727, the t~ statistic): the likelihood-ratio ordering of Feldman & Cousins, exact for a Gaussian
profile, so the interval turns into a one-sided limit by itself when the data sit at the bound (best fit on
the bound: 68 / 95% upper limits 1.00 / 1.96 sigma). Each end is a crossing, the bound, or open (below
threshold up to the grid end: the data leave the POI open there, or the grid is too narrow), and the
interval is two-sided, an upper or a lower limit, or unconstrained (npz `ends1` / `ends95`, `kind1` /
`kind95`; the log prints an open end as `>grid end (open)`). Full Feldman-Cousins (toys) is needed only where
the profile is far from Gaussian near a bound. A Delta_Neff_init profile is also mapped to Neff through each
row's derived Neff from LINX (`derived_neff`, `sigma1_neff`, `cl95_neff`, and the "as Neff" log line).
Test: `pytests/test_profile_bounds.py` on `configs/toy_bounded.py`.

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
| `PA_XBOX` | 5 | default free-dim box in scaled sigmas (config `box` overrides per parameter); a row the box holds is not converged (launchers 10) |
| `PA_MAXIT` | 40 | LM iterations (launchers 30) |
| `PA_GTOL` | 0.03 | gradient test, max-norm in scaled coordinates |
| `PA_DTOL` | 1e-3 | noise floor for stalled rows; 0 = off |
| `PA_SIGTOL` / `PA_SIGTOL_PATIENCE` | 0 / 3 | interval-stability stop, off by default (rough likelihoods only) |
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
| `PA_POI2D` | `''` | `a,b`: a 2D profile of the pair instead of the 1D profiles |
| `PA_NPTS2D` | 11 | points per 2D axis, unless the config gives `grid2d = {name: (lo, hi, n)}` |
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

- Wilks / coverage calibration deferred (only needed if a second mode appears within a few Delta chi2).
- 2D joint profiles for a few parameter pairs.
- Vendored LINX (`abcmb/linx`) and LINX main have diverged: the inert-radiation normalization fix and the
  dense-grid / log-log output are not in LINX main yet (`/pscratch/sd/c/carag/LINX/rho_extra_norm.diff`,
  `dense_grid_loglog.diff`), and main's finite t0 / `SaveAt(t0, t1, dense)` is not in the vendored copy.
- The EM-coupled WIMP is not implemented (needs a LINX thermodynamics extension exposed as a stage input).
- `pytests/fixtures/snapshots.npz` predates the vendored-LINX changes: regenerate it on GPU.

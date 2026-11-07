# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

`cgiovanetti/ABCMB-k` (remote `origin`) is a fork of ABCMB (`TonyZhou729/ABCMB`, remote `upstream`, pushes disabled; arXiv:2602.15104), the Python+JAX differentiable CMB Boltzmann solver. The fork adds a params-axis batched (per-k) engine in `abcmb/` and a frequentist profile-likelihood analysis in `scan/`: start at `scan/README.md`; results, decisions and known issues are in `CHANGELOG.txt`.

Siblings in `/pscratch/sd/c/carag/`: `../ABCMB/` (the install other projects use) and `../BBN_Hubble/` (cobaya + OLE MCMC; see the parent `/pscratch/sd/c/carag/CLAUDE.md`, which also covers HPC setup).

## Install & test

```bash
pip install -e .                          # editable install
pip install -r pytest_requirements.txt    # test deps (adds classy)
```

- `pytests/accuracy_test.py`: the 1%-vs-CLASS gate and the contract for theory changes (needs `classy`). classy 3.3.4's default `delta_l_max=500` vs ABCMB's `l_max/2` fails TT by ~1.2% at l~2500; compare with CLASS given ABCMB's `delta_l_max`.
- `pytests/test_snapshots.py`: parity against frozen output (`pytests/fixtures/snapshots.npz`) at `rtol=5e-5`, `atol=1e-18`, no classy. Run it on GPU (`JAX_PLATFORM_NAME=gpu` inside srun), the backend the snapshots were made on; regenerate with `python pytests/fixtures/generate_snapshots.py` after a deliberate change (the current file is stale, see `CHANGELOG.txt`).
- `pytests/test_fixed_grid.py` (`Model.freeze_lna_grid`, GPU in practice); `pytests/test_lm_tools.py` and `test_profile_toy.py` (the profile machinery on toys, CPU); `halofit_test.py`, `k_tail_test.py` (upstream).

CI forces CPU; `pytests/conftest.py` enables x64 and `jax_debug_nans`. Run long suites one file at a time. `time_tests.py` is a hand-run timing script. Pins that matter: `jax==0.8.1`, `equinox==0.13.2`, `optimistix==0.0.11` (`setup.cfg`).

## Architecture

`abcmb.main.Model(**specs)(params)` returns an `Output` (Cls, Pk, grids, Background, PerturbationTable, params). Stages: `add_derived_parameters` (CPU, Python; BBN branch from `specs["bbn_type"]`: `""` user YHe, `"table"`, or `"linx"`) -> `get_BG_pre_recomb` (GPU) -> HyRex (CPU, `backend='cpu'`) -> `_run_post_recomb` (GPU: Background with reionization, perturbations, line-of-sight integral). HyRex and LINX run on CPU by design; the `try/except` around the device transfers keeps CPU-only runs working.

- **`Delta_Neff_init` (the `"linx"` input) is Delta Neff *at the start of BBN*, not the cosmology's Delta Neff.** It sets the inert radiation at `T_start`; that species is not reheated by e+e- annihilation, so `Neff ~ 3.044 + 0.262*Delta_Neff_init` here (-7 -> 1.21, 0 -> 3.044, 7 -> 4.88; since the 2026-10-22 normalization fix in the vendored LINX; LINX main and `../ABCMB` still give 0.272). The true `Neff` is `Neff_vec[-1]`, which is why `Neff`/`N_nu_massless` may not also be supplied. Never treat the input as `Neff - 3.044`.

Module map (`abcmb/`): `main.py` (`Model`, `Output`, `call_batched`/`BatchedOutput`, `lna_grid`/`freeze_lna_grid`, parameter derivation); `model_specs.py` (`load_specs` defaults, `populate_species`, k axes); `species.py` (the `Fluid` interface and built-in species: the extension point, see `ABCMB_Fluids.ipynb`); `background.py`; `perturbations.py` (`PerturbationEvolver`, `PerturbationTable`); `spectrum.py` (`SpectrumSolver`, tabulated j_l in `bessel_tab/`, Debye form at large x); `halofit.py`; `hyrex/` (recombination); `linx/` (vendored BBN, diverged from LINX main: edit it only for ABCMB's BBN coupling); `ABCMBTools.py`; `constants.py`.

JAX / eqx conventions:
- Major objects are `eqx.Module`s and most public methods are `eqx.filter_jit`; a non-array, non-static field silently breaks tracing.
- `specs` must hold only JAX-friendly leaves: `Model.__init__` pops `adjoint` before storing it. Mirror that for similar config.
- `_to_float` casts int/bool params to float64 before any `filter_jit` (integer leaves break `checkpointed_while_loop` / `filter_custom_vjp` under outer AD). Don't strip it.
- Set `jax_enable_x64` at import in any new numerical module.

Batched pipeline: `Model.call_batched(params_list, shard=None, k_chunk_size=100)` returns a `BatchedOutput` (Cls, Pk, l, k, params):
1. `add_derived_parameters` per cosmology in a Python loop (not vmappable).
2. `_build_bgs_batched`: vmapped `BackgroundPreRecomb` (GPU), HyRex (CPU) and `get_BG` (GPU). `Background` stacks because `exp(-kappa)` is tabulated (`expmkappa_tab`, cubic interpolation so `grad(visibility)` stays C^1).
3. `PE.full_evolution_batched`: k chunks of 100, `vmap(k_chunk) x vmap(B)` in `_evolve_chunk`, output table built per chunk.
4. `SpectrumSolver.get_Cl_batched` / `Pk_lin_batched`: one `filter_jit` vmap over B.
With more than one GPU (or `shard=True`) the inputs are B-sharded (`Mesh` + `NamedSharding`) **before** the setup stages, B padded to a multiple of the device count; sharding after the setup builds everything on device 0 and OOMs. Batched modes do not bit-match the single-call path (step-controller noise within rtol); compare Cls peak-normalized (~1e-6). Defaults: `n_lna_PE=300` visibility grid, `dtmax_PE=0.03` for k < 0.06, `rtol_large_k_PE=1e-4` (production profiles use 1e-5); GPU peak scales with B/n_dev.

## When making changes

- If you change the background, perturbations, recombination or spectrum, run the accuracy test and report the max relative error on TT/EE/Pk (threshold 1%), plus the snapshot test.
- New species: subclass `Fluid` like `species.MasslessNeutrino` / `species.DarkEnergy` and pass it via `user_species`, not the LCDM default list.
- Log substantive work in `CHANGELOG.txt` (reverse chronological, terse); keep `scan/README.md` current.

**USER CONSTRAINTS (hard — honor these):** stay near permille (no tol-loosening /
float32 / fp32-bf16 storage — all measured and rejected); no TCA / diffrax
regime-switching; no SLURM job arrays (Perlmutter is touchy; ≤2 queued jobs get
priority); k_chunk stays 100 (smaller is slower, no memory benefit); do not lower
l_max for massive neutrinos.

## GPU access (do not wait for the user)

Request your own NERSC allocations when ABCMB code needs to run (account `m5403_g` for GPU, `m5403` for CPU).

**NEVER run Python (or pytest, or anything that touches JAX/CUDA) on the login node.** Every Python invocation, even a one-line import, goes through `srun --jobid=...` against an active allocation.

**Always export `PYTHONPATH=$(pwd):$PYTHONPATH` (CWD `/pscratch/sd/c/carag/ABCMB-k`) inside the srun shell.** The shared `actdr6` env has the sibling `/pscratch/sd/c/carag/ABCMB/` checkout editable-installed; without the override `import abcmb` resolves to that checkout and your edits here appear to do nothing. A script's own directory comes first on `sys.path`, so a script placed in another checkout's root imports that checkout's `abcmb` whatever `PYTHONPATH` says; when in doubt print `abcmb.__file__`.

Up to two interactive allocations at a time are allowed: use both for independent runs, and `scancel` them when done. Non-interactive pattern for per-call fresh shells (`logs/` is git-ignored):

```bash
# allocate (returns immediately; note the JOBID it prints):
salloc --no-shell --nodes=1 --qos=interactive --time=02:00:00 \
       --constraint=gpu --gpus=1 --account=m5403_g
# capture the JOBID once (e.g. echo <id> > logs/.jobid), then for each run:
srun --jobid=$(cat logs/.jobid) --ntasks=1 --cpus-per-task=32 \
     --gpus-per-task=1 bash -c \
     'module load conda && conda activate actdr6 && export PYTHONPATH=$(pwd):$PYTHONPATH && python <script>'
# free the node when done:
scancel $(cat logs/.jobid) && rm logs/.jobid
```

For parallel runs use distinct JOBID files (`logs/.jobid_a`, `logs/.jobid_b`). CPU-only work (the toy tests, LINX checks): the same with `--constraint=cpu --account=m5403` and no GPU flags.

## Special instructions

You are free to ignore all memories and instructions about making diffs in this repo; this repo (AND THIS REPO ALONE) is your playground.  You're free to use git to its fullest potential, staging and pushing commits, making new branches, and anything else that helps you stay organized.  But you are expected to stay organized; don't generate so much garbage that we can't find anything, and clear out stale artifacts as necessary (you may ignore memories and previous instructions about saving certain artifacts).

You are allowed to make ONE new conda environment for this project, though there is likely already a conda environment suitable for your purposes.  Do not modify any existing conda environments, make ONE new one and modify it as you need, if necessary.  You may request up to two interactive GPU allocations at a time to use for any purpose, unless I give you local instructions that indicate otherwise.
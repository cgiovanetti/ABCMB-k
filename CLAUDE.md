# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

This is the **standalone ABCMB repository** (`TonyZhou729/ABCMB`, arXiv:2602.15104) — a Python+JAX, fully differentiable Boltzmann solver for the CMB. It is the source-of-truth, PyPI-published version of ABCMB.

It is distinct from sibling checkouts in the parent workspace (`/pscratch/sd/c/carag/`):
- `../ABCMB/` — local install used by other projects in the workspace
- `../BBN_Hubble/` — research project that wraps ABCMB inside cobaya theory classes and runs MCMC with the OLE emulator

When the user asks about ABCMB internals (the Model class, perturbation hierarchy, recombination, spectrum integration), work here. When they ask about cobaya integration, MCMC, or emulator training, that lives in `../BBN_Hubble/` — see the parent `/pscratch/sd/c/carag/CLAUDE.md`.

## Install & test

```bash
pip install -e .                                  # editable install
pip install -r pytest_requirements.txt            # test deps (adds classy)
cd pytests && pytest -s -vv                       # accuracy test vs. CLASS
```

Test hierarchy on this checkout:
- **`pytests/accuracy_test.py`** — the 1%-vs-CLASS gate. Requires a working `classy` install (locally: a built CLASS variant on `PYTHONPATH`). This is the contract for theory-affecting changes.
- **`pytests/test_snapshots.py`** — the lighter parity oracle at `rtol=1e-8`, `atol=1e-18` against frozen ABCMB output (`pytests/fixtures/snapshots.npz`). No `classy` dependency. Appropriate for refactor work where you don't want to rebuild CLASS. Regenerate snapshots with `python pytests/fixtures/generate_snapshots.py` if a deliberate code change shifts XLA scheduling.

CI forces CPU (`JAX_PLATFORM_NAME=cpu`); `pytests/conftest.py` enables x64 and `jax_debug_nans`. The snapshot tests need to run on the same backend the snapshots were generated against (currently GPU — set `JAX_PLATFORM_NAME=gpu` explicitly inside srun).

`time_tests.py` (top-level) is a hand-run JIT warm-up / wall-clock benchmark, not part of pytest.

Pinned dependencies that matter: `jax==0.8.1`, `equinox==0.13.2`, `optimistix==0.0.11` (see `setup.cfg`). Don't loosen these without testing.

## Big-picture architecture

The pipeline is orchestrated by `abcmb.main.Model` (an `eqx.Module`). `Model(**specs)(params)` runs end-to-end and returns an `Output` bundle (ClTT/ClTE/ClEE/Pk + grids + Background + PerturbationTable + full params).

Stages, in execution order — note the **deliberate device split**:

1. **`add_derived_parameters` (CPU, Python)** — fills defaults, resolves the Neff / N_nu_massless / YHe triangle, and selects the BBN branch from `specs["bbn_type"]`: `""` (user-supplied YHe), `"table"` (interp PArthENoPE `sBBN_2025_CLASS.txt`), or `"linx"` (run bundled LINX). Non-trivial: read this function before adding new cosmological parameters. New unknown keys are passed through as `jnp.array(...)` to avoid retracing.
2. **`get_BG_pre_recomb` (GPU, `eqx.filter_jit`)** — tabulates conformal time and packs `RecombInputs` (TCMB, nH, H on HyRex's lna axis).
3. **HyRex (`abcmb/hyrex/`) on CPU** — `RecModel((recomb_inputs, params))` is `filter_jit(..., backend='cpu')`. Outputs are moved back to GPU.
4. **`_run_post_recomb` (GPU, `eqx.filter_jit`)** — builds the full `Background` (incl. reionization, optical depth, decoupling), runs the perturbation hierarchy, integrates the line-of-sight transfer, returns Cls and Pk.

HyRex and LINX run on CPU even when JAX has GPU devices. This is intentional: their solvers are sequential and serve as JAX-traceable but cheap CPU stages. Everything else (background ODEs, perturbations, spectrum) is GPU. The `try/except` around `jax.devices('gpu')` in `main.py` is how the code stays CPU-only friendly.

### Module map (under `abcmb/`)

- **`main.py`** — `Model`, `Output`. Pipeline glue + parameter derivation. Adds `Model.call_batched(params_list, shard=)` + `BatchedOutput` for params-axis batched (and optionally multi-GPU sharded) evaluation, and the batched setup helpers `_build_bgs_batched` / `_pre_recomb_batched` / `_get_BG_batched`.
- **`model_specs.py`** — `load_specs` (run options with defaults), `populate_species` (assembles the species tuple from ΛCDM defaults + `user_species`), `get_k_axis_perturbations` / `get_k_axis_transfer`.
- **`species.py`** — base `Fluid` (`eqx.Module`) interface (`rho`, `P`, `w`, `y_ini`, `y_prime`, `rho_delta`, `rho_plus_P_theta`, `rho_plus_P_sigma`) plus all built-in species: `Photon`, `Baryon`, `ColdDarkMatter`, `MasslessNeutrino`, `MassiveNeutrino`, `DarkEnergy`. **This is the extension point** — new physics = new `Fluid` subclass. The `ABCMB_Fluids.ipynb` notebook walks through this.
- **`background.py`** — `BackgroundPreRecomb` (pre-recomb stage) and `Background` (full, with reionization), plus `ReionizationModelFromZ` / `ReionizationModelFromTau` (branched via `lax.cond` on `specs["input_tau_reion"]`).
- **`perturbations.py`** — `PerturbationEvolver` and `PerturbationTable`. Drives diffrax through the Einstein–Boltzmann hierarchy in synchronous gauge with the tight-coupling approximation. Branch `perk-refactor` adds `_evolve_chunk`, `_compute_modes_batched`, `full_evolution_batched`, `make_output_table_batched` (the `strip_bg_kappa` helper was removed on `perk-perf` once `Background` became stackable).
- **`spectrum.py`** — `SpectrumSolver`. Line-of-sight integral with tabulated spherical Bessel functions (`bessel_tab/`); produces Cls and the linear matter Pk. `get_Cl_batched` / `Pk_lin_batched` are a single `@eqx.filter_jit` `jax.vmap` over the batch axis (on `perk-perf`; were Python loops on `perk-refactor` — see "Batched pipeline" below).
- **`hyrex/`** — bundled HyRex recombination (`xe`, `Tm` evolution) using `array_with_padding` for variable-length arrays through JIT.
- **`linx/`** — **bundled** LINX (BBN). This is a vendored copy frozen with this ABCMB version; it is *not* the same code path as the standalone `../LINX/` checkout or the `../BBN_Hubble/OLE`-aware copies. Edit this only when ABCMB's BBN coupling specifically needs it.
- **`ABCMBTools.py`** — interpolation helpers (`bilinear_interp`, etc.) used across modules.
- **`constants.py`** — physical constants in ABCMB's units (eV, cm, Mpc).

### JAX / eqx conventions to respect

- All major objects are `eqx.Module`s; nearly every public method is wrapped in `eqx.filter_jit`. Adding non-array, non-static fields will silently break tracing.
- `specs` is a plain dict held on `Model`. It must not contain non-JAX leaves like `diffrax.adjoint` classes — `Model.__init__` explicitly pops `adjoint` out before storing `specs`. Mirror this pattern if you add similar config.
- `_to_float` in `run_cosmology_abbr` casts int/bool params to float64 before any `filter_jit`. This is a known workaround for `checkpointed_while_loop` / `filter_custom_vjp` not accepting integer leaves under outer AD. Don't strip it.
- `jax_enable_x64` is set at module import in several files. New modules that do any numerics should do the same.
- The HyRex/LINX → GPU re-transfer is wrapped in `try/except Exception: pass` for CPU-only runs. Preserve that pattern when adding cross-device stages.

### Batched (per-k) pipeline (branch `perk-refactor`; perf done on `perk-perf`)

`Model.call_batched(params_list, shard=None)` is the user-facing entrypoint for
params-axis-batched evaluation. It:

1. Derives params eagerly (`add_derived_parameters`, a Python loop — it has
   `sys.exit`/species-loops/bbn branching, so it is NOT vmapped; it is ~ms/cosmo).
2. Builds the batched `Background` via `_build_bgs_batched`: vmapped
   `BackgroundPreRecomb` construction (GPU), vmapped `RecModel`/HyRex (CPU), and
   vmapped `get_BG` (GPU) — one `eqx.filter_jit` + one device transfer each way
   instead of O(B). The full `Background` stacks because `kappa_func` is gone
   (see below); there is no more `strip_bg_kappa` / python BG list.
3. Calls `PE.full_evolution_batched((BG_batch, params_batch))` → batched
   `PerturbationTable`. Internally chunks the k-axis (`k_chunk_size=100` default;
   a sweep — `bench/sweep_kchunk.py` — confirms 100 is optimal) and runs
   `vmap(k_chunk) × vmap(B)` around `evolution_one_k` inside `_evolve_chunk`.
4. `SpectrumSolver.get_Cl_batched` / `Pk_lin_batched` are now a single
   `@eqx.filter_jit` `jax.vmap` over `B` on the stacked `BG_batch` (no python
   loop).

When `shard` is True (or `None` with >1 visible GPU), the stacked inputs are
B-axis sharded via `jax.sharding` `Mesh` + `NamedSharding(P('batch'))` BEFORE the
setup, so every GPU stage auto-partitions (GSPMD, no collectives — the pipeline
is embarrassingly parallel over B) and each device builds/solves only `B/n_dev`
cosmologies. `B` is padded to a multiple of the device count and the padding is
sliced off the output. **Shard before the setup, not after** — sharding after
`_build_bgs_batched` builds all B on device 0 and OOMs at B=64.

Returns a `BatchedOutput` (Cls/Pk/l/k/params; BG and PT are *not* stored).

PERF (ELLMAX=800, A100, post-compile, per param; see `CHANGELOG.txt` 2026-05-29):
single-GPU `call_batched` B=16 went 12.4 → 4.0 s/param; 4-GPU sharded reaches
1.13 s/param at B=64 (still falling with B). The win was killing two FLAT
eager-dispatch costs (the spectrum python loop, and `get_BG` run eagerly), NOT
solver tuning — the perturbation solve already amortizes 12→2 s/param.

`Background.kappa_func` (a `diffrax.Solution`) is **gone** — replaced by
`expmkappa_tab`, a plain array tabulated on the shared `lna_tau_tab` axis and read
via `interpax.interp1d(method="cubic")` in `expmkappa` (cubic, not linear, so
`grad(visibility)` stays C¹ for ClTE). That is what lets `Background` stack across
cosmologies. The keystone change is in `background.py`; `strip_bg_kappa` was
deleted from `perturbations.py`.

The chunked path inside `_compute_modes_batched` produces per-mode trajectories
that don't bit-match the single-call vmap reference. This is *not* a bug; it's
diffrax PID step-controller noise within the configured `rtol_large_k_PE` (1e-4
default). The contract that matters is downstream Cl/Pk agreement, which
`bench/validate_keystone.py` shows is ~1e-6 peak-normalized vs single-call. (Use
peak-normalized error, not pointwise relative — ClTE crosses zero ~6×, so
pointwise relative error blows up at the crossings even when agreement is
excellent.) See `bench/chunking_debug_report.md` for the original diagnosis.

## When making changes

- The accuracy test against CLASS is the contract. If you change anything in the background, perturbations, recombination, or spectrum modules, run `pytest -s -vv` and report the max relative error on TT/EE/Pk — the threshold is 1%.
- New `Fluid` species should follow `species.MasslessNeutrino` or `species.DarkEnergy` as templates and be passed in via the `user_species` tuple, not added to the ΛCDM default list.
- For HPC / NERSC / GPU job setup, see the parent workspace CLAUDE.md (`/pscratch/sd/c/carag/CLAUDE.md`). Don't duplicate that here.
- Log substantive sessions in `CHANGELOG.txt` (reverse-chronological, BBN_Hubble format).
- The branch-specific plan is in `plan.md` (canonical /ultraplan output, kept for history). The Phase A/B baseline numbers, design memo, and chunking-bug closeout report all live in `bench/` and are referenced from `CHANGELOG.txt`.

## GPU access (do not wait for the user)

Request your own NERSC GPU allocations when ABCMB code needs to run.

**NEVER run Python (or pytest, or anything that touches JAX/CUDA) on the login node.** Every Python invocation — even a one-line import smoke test — must go through `srun --jobid=…` against an active allocation. Login-node compute is shared and the user has explicitly forbidden it.

**Always export `PYTHONPATH=$(pwd):$PYTHONPATH` (assuming CWD is `/pscratch/sd/c/carag/ABCMB-k`) inside the srun shell.** The shared `actdr6` env has the sibling `/pscratch/sd/c/carag/ABCMB/` checkout editable-installed; without the override, `import abcmb` resolves to that checkout, not this one — your edits to `abcmb/*.py` in this directory will appear to do nothing.

**Use both allowed allocations when you can.** NERSC permits **up to two concurrent interactive allocations** for this account; if you have two independent jobs to run (e.g., baseline + spike benchmarks), allocate two nodes and run them in parallel instead of serializing. Don't leave the second slot idle to be polite — wall-clock is the cost. Always `scancel` allocations when done so they don't sit idle.

Non-interactive pattern that works inside Claude Code's per-call fresh shells:

```bash
# allocate (returns immediately with a JOBID line on stderr/stdout):
salloc --no-shell --nodes=1 --qos=interactive --time=02:00:00 \
       --constraint=gpu --gpus=1 --account=m3166_g
# capture the JOBID once (e.g., echo > bench/.jobid), then for each run:
srun --jobid=$(cat bench/.jobid) --ntasks=1 --cpus-per-task=32 \
     --gpus-per-task=1 bash -c \
     'module load conda && conda activate actdr6 && python <script>'
# free the node when done:
scancel $(cat bench/.jobid) && rm bench/.jobid
```

For parallel runs use distinct JOBID files (`bench/.jobid_a`, `bench/.jobid_b`, etc.).

## Current task

Major refactor of ABCMB.  The goal is to output **per k mode** to take better advantage of GPU parallelization.  Right now each power spectrum calculation is limited by the worst k to solve, and we're already vmapping to get just that far.  Instead, we'd like to refactor so I start with e.g. a grid of parameters and then compute just one k mode for all of those parameters at once.  I repeat for each k mode, and then at the end collapse back into a power spectrum to use to evaluate a likelihood in a frequentist-style analysis.

## Special instructions

You are free to ignore all memories and instructions about making diffs in this repo; this repo (AND THIS REPO ALONE) is your playground.  You're free to use git to its fullest potential, staging and pushing commits, making new branches, and anything else that helps you stay organized.  But you are expected to stay organized; don't generate so much garbage that we can't find anything, and clear out stale artifacts as necessary (you may ignore memories and previous instructions about saving certain artifacts).

You are allowed to make ONE new conda environment for this project, though there is likely already a conda environment suitable for your purposes.  Do not modify any existing conda environments, make ONE new one and modify it as you need, if necessary.  You may request up to two interactive GPU allocations at a time to use for any purpose, unless I give you local instructions that indicate otherwise.
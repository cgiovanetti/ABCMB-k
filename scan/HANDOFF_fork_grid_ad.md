# HANDOFF — ABCMB-k per-ell AD tangents are jagged: the cosmology-dependent save/LoS grid (2026-10-16)

## Task
Make ABCMB-k's forward-mode AD Cl tangents agree with finite differences ell by ell, at no or small cost,
without moving production accuracy (near permille). Upstream ABCMB had a milder form of the same problem and
is fixed (below). The fork is not, because its grid moves every node, not just the start.

## The mechanism (established upstream, 2026-10-15; CHANGELOG "2026-10-15 (night)")
- The line-of-sight (LoS) quadrature nodes are the perturbation save grid `PT.lna`. If node positions depend
  on theta, Cl(theta) inherits the quadrature's dependence on node placement. That dependence is zero
  physically and small numerically, but jagged: local slopes far exceed the trend. AD returns the local
  slope, so per-ell tangents go wrong where the true tangent is small (omega_b TT at ell < 30 worst).
- Upstream grid = `linspace(lna_transfer_start, 0, 500)`. Only the start moves, through the tau_c*aH = 0.008
  crossing (omega_b moves it most, then Neff, omega_cdm; h and tau barely).
  - A +-2e-3 scan of the start moves low-ell TT by ~1e-5, with local slopes +-0.02 per unit against a
    trend of -0.002.
  - Every LoS input on the nodes has AD = FD with respect to the start (<= 1.5e-3: PT values, g, g', xe,
    expmkappa, tau, sources). Only the assembled Cl derivative is off, by 13-36x.
  - A C1 tau lookup and a frozen lensing grid changed nothing.
- Upstream fix: `lna_transfer_start = jax.lax.stop_gradient(jnp.interp(...))`.
  - Per-ell AD vs FD (FD 0.2 sigma), before -> after: omega_b TT 2e-1 -> 4e-2, omega_cdm TE 5e-2 -> 1e-2,
    Neff TT 1e-1 -> 2e-3.
  - chi2-like gradient unchanged; no timing cost.
  - The user's own upstream v2 (`grad_bumps` adcaee3) has the same line; another session found it smooth
    (memory `project_grad_bumps_check`).

## The fork
- Code, ABCMB-k main c197eed (code = b9dfcc3):
  - `background.py:584`: `lna_transfer_start = jnp.interp(...)` is continuous and differentiable, with no
    stop_gradient.
  - `perturbations.py:23` `make_lna_grid` (default `lna_grid_mode="visibility"`, `n_lna_PE=300`,
    `lna_grid_n_dense=4000`, floor 0.4, smooth 0.3): `nodes = jnp.interp(linspace(0,1,n), cdf, u)`. The
    CDF comes from the visibility g(theta) on a dense grid starting at `lna_transfer_start`, so every node
    moves with theta, piecewise linearly (linear inversion over the dense CDF).
  - Called at `perturbations.py:177/262/318`. LoS trapezoid weights come from `jnp.diff(PT.lna)`
    (`spectrum.py:794-802`).
- Measured with `adprobe.py`: lensed, production-like specs (l_max 2508, rtol 1e-5, 1 massive nu,
  delta_l_max 500). Entries are max|AD - FD| / max|FD|, FD 0.05 / 0.2 sigma:

  | | fork main | fork, whole grid stop_gradient | upstream + fix |
  |---|---|---|---|
  | omega_b TT | 0.38 / 0.56 | 0.74 / 0.81 | 0.034 / 0.038 |
  | omega_b TE | 0.39 / 1.4 | 0.76 / 2.0 | 0.0066 / 0.0069 |
  | omega_cdm TT | 5.7e-3 / 0.20 | 0.095 / 0.30 | 2.3e-3 / 1.4e-3 |
  | omega_cdm TE | 0.075 / 1.6 | 1.2 / 2.4 | 0.019 / 0.013 |

  - Absolute, relative to Cl per sigma: the omega_b TT tangent is ~2e-3 (ell < 100) and 6e-3 (ell > 100).
    The AD error is 9e-4 (ell < 30) and 1e-4 (ell > 100) on fork main, and 2e-3 / 3e-4 with the grid
    frozen.
  - FD at 0.05 and 0.2 sigma disagree with each other, so the fork's Cl(theta) is itself jagged. Neither
    AD nor small-step FD is clean; use scans (below) rather than single FD pairs to judge a fix.
  - Freezing the whole grid makes it WORSE. In the fork, node placement changes the Cls at a level
    comparable to the low-ell omega_b tangent, unlike upstream (~1e-5), so the upstream one-liner does not
    transfer.
- chi2 level (`upstream_check.py fork_dtl`, chi2-like functional, AD vs FD 0.05 / 0.2 sigma):
  - h 3.7e-4 / 4.2e-4, omega_b 5.8e-4 / 2.1e-3 (FD noise 1.6e-3), omega_cdm 1.0e-3 / 6.8e-4,
    Neff 7.8e-4 / 2.3e-4.
  - Profiles converge on chi2 gradients (GTOL 0.03), so the practical cost is roughness of the chi2
    surface. Report the effect of a fix on that too, not only on the per-ell metric.

## Candidate approaches (none tried yet; most economical first)
1. **Fiducial (frozen) grid.** Build the visibility grid once at a fiducial cosmology and reuse it for
   every theta. Node positions are then data by construction and Cl(theta) loses the placement jitter.
   - Check: accuracy across the profile parameter ranges against a tight reference
     (`fork_interp_check.py`-style); per-ell AD vs FD; and scan roughness.
   - The start must still precede the tight-coupling crossing for every cosmology in range (margin or min).
   - Likely free. This is the recommended first experiment.
2. **Smooth grid motion.** Keep a theta-dependent grid but make node positions smooth in theta, for example
   a smoother CDF inversion or an analytic map anchored on lna_rec and the visibility width. AD with the
   grid derivative then matches FD. Values are unchanged at the fiducial.
3. **More nodes** (`n_lna_PE` 300 -> 400-500). This shrinks the placement dependence, but n_lna is the
   dominant per-call GPU memory factor; check the cost against the user's constraints.
4. **Upstream's configuration:** `lna_grid_mode="uniform"`, `n_lna_PE=500` + stop_gradient on the start.
   It is validated, but costs ~1.6x memory and LoS time, which is what the visibility grid was introduced
   to save (CHANGELOG 2026-05-30).

## Tools (all on disk)
- `scan/.upstream_pr/adprobe/` (untracked):
  - `adprobe.py TREE TAG LENSING(0|1)`: per-ell AD vs FD for omega_b and omega_cdm at both FD steps;
    `adprobe.py cmp TAG...` prints the table.
  - `lprofile.py` (ell bands), `abs_scale.py` (absolute scale), `crossprobe.py`, `timing.py` (warm value +
    7-direction gradient).
  - `lts_scan.py` / `probe_lts.py`: per-quantity derivative with respect to the grid start. These need an
    `if "lts_override" in params:` hook after `background.py`'s `lna_transfer_start`.
  - Fork results: `probe_fk_base_L1.npz`, `probe_fk_sg_L1.npz`; upstream results: `probe_*.npz`, `log_*.log`.
- `scan/.upstream_pr/upstream_check.py TAG`: 7 directions + chi2-like functional, AD vs FD. A TAG starting
  with `fork` runs the ABCMB-k checkout; other TAGs run `trees/TAG`.
- `scan/.upstream_pr/grad_bumps/gb_check.py scan TAG`: another session's 40-point scans of Cls and AD
  tangents along omega_b and Neff. A smooth AD tangent along the scan is the clearest proof a fix works.
  Written for upstream trees; adapt it for the fork.
- `scan/.v2analysis/fork_interp_check.py` + `run_fic.sh`: fork accuracy vs an rtol-1e-7 reference and
  `call_batched` timing (B=16, 1 GPU).
- Tests: `pytests/test_snapshots.py`, `pytests/accuracy_test.py`. Regenerate snapshots
  (`pytests/fixtures/generate_snapshots.py`, or `scan/.v2analysis/run_fork_tests.sh`) whenever values change.
- Run pattern:
  - Allocate: `salloc --no-shell --nodes=1 --qos=interactive --constraint=gpu --gpus=4 --account=m3166_g`
    (background it, timeout 7200000).
  - Build variants: `git archive <commit>` into a temp tree, patch with sed.
  - Run: `srun --jobid=<id> --overlap` with `PYTHONPATH=<tree>` and `CUDA_VISIBLE_DEVICES` per lane.
  - Clean up: delete trees afterwards and `scancel` when done. Probe compile ~3-6 min; an adprobe run is
    ~7 min warm.

## Constraints (from the user, CLAUDE.md files and memory)
- Never run python on the login node. Every python call goes through srun with
  `module load conda && conda activate actdr6`. Account m3166_g; never the premium queue.
- ABCMB-k is a playground: branch, commit, `git push origin`. Never push to ABCMB (the `upstream` push URL
  is disabled). No new worktrees or clones (`ABCMB-k-grad` is the one sanctioned exception, not yours to
  modify). Leave the uncommitted `CLAUDE.md` edit in the checkout alone.
- ABCMB-k hard constraints: stay near permille; no tolerance loosening or float32; no TCA or
  regime-switching; no SLURM job arrays; k_chunk stays 100; do not lower l_max for massive neutrinos.
- Gradients are AD, not FD. Physics correctness, no hand cuts. Keep edits minimal.
- Ask before any production relaunch.
- `upstream-fixes` (9276f7d) is the user's hand-transcription source for ABCMB: do not touch it.

## Where things stand
- ABCMB-k main c197eed is pushed. The interpolation fix there is `dtmax_PE=0.03` for k < 0.06 (+3.3%),
  replacing step_ts.
- Pending, needs user approval: rerun the 7 `_v2fix` profiles on main (no row is under GTOL on the current
  Cls). The fork AD fix should land before that rerun.

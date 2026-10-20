# NaN AD gradient / max_steps abort in the LCDM+Neff full-plik profile — root cause (2026-10-01)

Investigated by a subagent on interactive 59164522 (m5403_g, 1x4 A100). Scripts import
`ppa_f312ab5.py` (= the driver that 59149970 ran); `run.sh` = production env. Spot-checked
against `scan/plik_full.py` and the logs below.

## Failing computation: the full-plik inner SPG profile, not ABCMB
`PlikFull.profile` (`scan/plik_full.py:218-273`), reached eagerly from `_chi2_of_cls`. The
AD-path Cls are finite and match the value path to <=2e-11 peak-normalised
(`repro_nan1.log`); replaying the profile offline on the stashed Cls reproduces the NaN at
h row 9 with no ABCMB in the loop (`spg_study1.log` §0).

## Mechanism (n_s row 1, `spg_onset_ns1.log`)
converged state -> roundoff gives `sy = -1.2e-6` -> `alpha = AMAX = 1e10` (`:266`) -> the
projected step is unbounded in the box-free nuisances (calib_100T, calib_217T, A_planck have
bounds (None, None); gal545/galf no upper bound) -> all 10 halvings fail -> unconditional move
to `z + d/1024` (`:262`): calib_217T = -1.28 -> clipy `1/sqrt(cal)` -> NaN, never rejected.
That row sat at alpha = AMAX on 468/800 iterations; every row takes 2-10 unconditional moves.
h row 9 in the 59149970 executable: finite at k=799, NaN at k=800, finite for k=801..1000
(`spg_bisect_maxit.log`, `spg_step800.log`, `spg_kscan.log`) -> executable-specific.
Latent: `(a_sz*a_cib)**0.5` has a non-finite gradient exactly at 0 (not hit here).

## NaN -> 7-node abort (59041065, 09-28 validation)
NaN gradient -> NaN trial point `[h, nan x6]` -> conformal-time solve `abcmb/background.py:303`
hits max_steps -> raise mode: CpuCallback error -> XLA "Replicated computation launch failed
... Aborting". Reproduced verbatim (`nanrow_raise.log`, exit 134). Timing +26.3 min matches.

## Intermittency (`autotune_*.log`)
XLA GPU autotuning at compile time: 4 fresh compiles of the same program on the same inputs
gave 4 different bit-results, 1/4 NaN at h row 9. With `--xla_gpu_autotune_level=0`: 4/4
bit-identical (and NaN on this input). Raise-mode programs are never written to the persistent
cache (host callbacks), so ABCMB recompiles each run; nan-mode reruns load 59149970's cached
executables and reproduce its outcome. The SPG is chaotic in its input: 1e-13..1e-9 Cl
perturbations move the profiled chi2 by up to 0.11 and occasionally give NaN — plausibly the
driver's long-standing "~0.1 chi2 roughness floor" (not tested).

## Why one NaN row makes all 11 rows non-finite
Equinox runtime checks reduce the failure flag over the vmapped batch (`unvmap_any`, also
across shards); in nan mode the whole output is NaN-filled. Sites for a NaN cosmology:
diffrax `diffeqsolve` (`background.py:303`, HyRex, `:722`), `optx.root_find` in
`ReionizationModelFromTau` (`background.py:1036`), `lineax.linear_solve` in interpax
`CubicSpline` (`spectrum.py:599-601,700`). With all three throw=False only the NaN row is bad
(`nanrow_patched_all.log`).

## Recommended fixes (not applied)
1. SPG (`plik_full.py`): don't move on a failed line search (`z_new = where(accepted, z_acc, z)`);
   keep the previous alpha when `sy <= 0`; reject non-finite trial f/g; optional
   projected-gradient stop.
2. Physical boxes for the free nuisances (calibrations, A_planck > 0 within ~±50 sigma; finite
   upper bounds for gal545/galf; A_sz, A_cib_217 slightly above 0).
3. Driver: never build a trial from a non-finite direction (ac5687a `_safe_grad` covers most).
4. Pipeline: pre-validate non-finite rows in `call_batched`; keep EQX_ON_ERROR=nan.

Established: offline replay, abort repro, the three mixing sites, the n_s row 1 chain,
executable dependence, autotune determinism. Inferred: 59041065 / 09-28 followed this chain.

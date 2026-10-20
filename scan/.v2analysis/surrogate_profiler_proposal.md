# Batching-native profiler: fit the χ² surface once, then profile the fit

Proposal from the `_v2fix` session (2026-10-13), updated 2026-10-14 with the grad-vmap-dirs numbers.
The user's question that prompted it: "Is there no way to build a profiler around batching...?"
Status: an idea for review, not built. The aim is all 1D profiles AND the 2D joint profiles
(the user's goal: "if we can do even a few combinations that's a win for ABCMB").

## Costs it is built on (production specs: l_max 2508, lensing, 1 massive ν, PE rtol 1e-5)

| What | Rate | Source |
|---|---|---|
| Batched value, fork | 4.5 s/cosmo at B=64 on one 40 GB A100 (9.1 GB); ~1.1 s/cosmo per 4-GPU node at 256/node | fvu bench, 2026-10-13 |
| 7-direction AD gradient, fork after grad-vmap-dirs | 34 s/cosmo at B=8 per A100 (12.7 GB); B_local ≤ 8 on 40 GB | ABCMB-k-grad/scan/.grad_bench |
| 7-direction AD gradient, fork before | 167 s/cosmo at B=8 | fvu bench |

So a gradient now costs about 8 batched values (it was about 37).

## The method

1. **Sample.** Evaluate 10⁴–2×10⁴ cosmologies over the region that matters (Δχ² ≲ 12, a ball of
   radius ≈ 3.5 in whitened coordinates). Whiten with the best-fit Hessian: it is trustworthy now
   (rebuilt with the fixed AD, positive definite, min eigenvalue 0.122), and it barely changes
   along the profile (row Hessians were within 3–7% of it). The plik nuisances are already
   profiled inside every χ² evaluation, so the surface is 7-dimensional.
   - Cost: about 3–7 node-hours in 256-per-node calls. Every node and every batch is independent,
     which is the fork's best case.
2. **Fit and profile.** Fit a smooth model to χ² (a full quartic in 7D has 330 coefficients, so
   10⁴ points is ample) and minimize it analytically at every grid point. Every 1D profile and
   every 2D profile comes out of the same samples.
3. **Re-evaluate exactly.** Compute the true χ² at each predicted conditional minimum: one more
   batched call (about 150 rows for the 1D set; e.g. 15×15 = 225 per 2D pair).
   - At a true conditional minimum χ² is stationary in the other parameters, so a location error
     δ (in σ units) raises the true χ² only by about |δ|². Being 0.1σ off costs about 0.01 in Δχ².
   - The re-evaluated value can never fall below the true profile: it is a safe upper bound.
   - Where the fit is poor (expected: the non-Gaussian upper side of Neff), add samples there and
     refit.
4. **Validate** against `_v2fix`, an independent gradient-based answer for all 7 profiles.

This is NOT per-row finite differencing: nothing depends on a small step, the fit spans σ-scale
distances and models curvature explicitly, and every reported number is an exact χ² evaluation.
It is gradient-free, so the risk moves to fit fidelity; step 4 is the guard.

## Budget, before vs after the gradient fix (estimates, not measured)

| | Gradient driver, old fork | Gradient driver, grad-vmap-dirs | Surrogate |
|---|---|---|---|
| All 7 1D profiles | ~25–30 node-h | ~12 node-h (iteration 35 → ~15 min) | 5–10 node-h for 1D **and** 2D |
| Each 2D pair | ~14 node-h | ~6 node-h | ~0 extra (one more re-eval call) |

The gradient fix narrows the gap for 1D. The surrogate's remaining edge is mainly the 2D profiles.

On the other session's "values must be 30–45× cheaper than a gradient" argument: that costs a
LOCAL gradient-free optimizer that rebuilds a ~2n+1 stencil every iteration for every row. The
surrogate builds one GLOBAL model amortized over all 147 1D rows and all 21 possible pairs, so
that threshold doesn't apply directly. Please judge it on the budget above.

## Open questions to chew on

1. **Fit fidelity.** Is a 7D quartic in whitened coordinates enough over Δχ² ≲ 12? Known
   non-Gaussian features: Neff upper side, the h–Neff degeneracy. Alternatives: higher degree
   (quintic 792 coefficients, sextic 1716), GP/RBF, or local fits along the predicted profile
   paths. Test with held-out residuals measured NEAR the conditional-minimum curves, not
   averaged over the ball.
2. **χ² noise floor.** Sources: the inner nuisance profile (`plf.profile_batched` tolerance) and
   batch-composition step-controller noise (~1e-7 relative in the Cls, negligible). Measure it:
   re-evaluate a handful of points in different batch compositions. Least squares averages
   noise, but it sets how far the re-evaluated rows can be trusted.
3. **Sampling design.** In 7D most of the ball's volume sits near its shell, while the profile
   paths (1D curves, 2D sheets) run through the interior. Suggested two stages: (a) Sobol fill of
   the whitened ellipsoid; (b) after the first fit, small clouds around every predicted
   conditional minimum, then refit. Respect the nuisance box (PA_XBOX=10 scaling-σ) and
   physical bounds (τ > 0, etc.).
4. **Hybrid.** Use the surrogate's predicted minima as warm starts for 1–2 iterations of the now
   ~5× faster AD driver and its ‖g‖ certificate: the surrogate does the global/2D work, AD
   certifies. Possibly the best of both.
5. **Acceptance test.** Row by row against `_v2fix`: re-evaluated surrogate rows at or below the
   `_v2fix` rows (within the noise floor) mean it is at least as good; rows clearly above mark fit
   failures. Compare the 1σ intervals too.

## Pointers (ABCMB-k, branch xe-c1)

- Batched value path: `scan/profile_prod_ad.py`: `build_dict(assemble_phys(...))` ->
  `model.call_batched(dicts, shard=)` -> `_chi2_from_out` (full plik with the nuisances
  inner-profiled by `plf.profile_batched`, plus low-ℓ TT/EE). Chunked caller:
  `_chunked_call_batched` (pads the last chunk to keep one compiled shape).
- Config (centre, scaling σ, grid σ): `scan/configs/lcdm_neff_plikfull_rc.py` (`cen`, `sig`,
  `grid_sig`).
- Whitening Hessian: `scan/results/warm_hessian_lcdm_neff_plikfull_rc_l2508_tt1_ee1.npz`
  (keys H, theta_warm, sigma, fdh, lmax, lowtt, lowee); check the units of H against `sigma` (the
  driver rescales it: `_rescale_hessian`).
- Validation target: `scan/results/profile_prod_ad_{POI}_v2fix.npz` (poi_grid, chi2, xstar,
  nuis, converged, gnorm, sigma1_fit). Nuisances are stored in scaling-σ units:
  θ = cen + sig·xstar. Loader: `load_rows` in `scan/compare_profile_runs.py`. omega_cdm and
  ln10As arrive when chunk 3 (allocation 59781927) finishes.
- Global best χ² on the fixed Cl: 2757.60.

## Constraints

- AD, not FD, for any gradient (user preference). Values at σ-scale separations for a fit are fine.
- No premium QOS; charge m3166_g. Interactive allocations: chunk 3 holds one until it finishes.
- ABCMB-k's CLAUDE.md user constraints: no tolerance loosening / fp32; k_chunk 100; no job arrays.
- Do not touch the `/pscratch/sd/c/carag/ABCMB-k` checkout until chunk 3 is done (it must stay on xe-c1).

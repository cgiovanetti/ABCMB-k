# Sampling with ABCMB out of the box: open items

Collected while building the lockstep HMC (2026-11-10). Status in brackets; newest notes at the end of an item.

## Upstream ABCMB

1. **A gradient standard JAX samplers can call** [open, highest value]. `jax.grad` (what blackjax / numpyro NUTS
   call) does not work as shipped: the default adjoint is diffrax `ForwardMode` (forward only), and
   `RecursiveCheckpointAdjoint` ran but gave O(1)-wrong omega_b / omega_cdm gradients (2026-09-15; never re-measured
   after #47). Proposal: a `custom_vjp` around params -> C_ell whose backward pass is the forward-mode Jacobian over
   the few input parameters (one vmapped jvp); then `jax.grad` of any scalar likelihood is right and costs one
   jacfwd. Single-chain NUTS would then work unmodified.
2. **Batched parameters** [planned: the -k port]. Many-chain lockstep sampling needs one call per leapfrog step for
   all chains. Upstream `vmap` flattens out past B ~ 16 and B = 64 did not compile in 20 min (2026-10-13 bench). Port
   the staged forward-mode batched gradient (`scan/batched_grad.py`), not only `call_batched`.
3. **Smooth BBN-table YHe** [deprioritized 2026-11-10: the sampling will take YHe from LINX, item 14]. `bbn_type="table"` interpolates YHe(omega_b, Neff) bilinearly, so
   dYHe/domega_b is piecewise constant (kinks every ~5e-5 in omega_b). Harmless for HMC correctness, but a non-smooth
   force; bicubic would fix it. Same code in -k.
4. **Silent spec footguns** [open, small]. Unknown spec keys are kept silently (`load_specs`), so a typo is a no-op:
   the scan configs and `hmc/posterior.py` pass `l_max_ur` / `l_max_ncdm`, read nowhere (the real keys are
   `l_max_massless_nu` / `l_max_massive_nu`; same values, so no effect). `N_nu_massive > 0` without
   `user_species=(MassiveNeutrino,)` silently evolves no massive neutrino. Both deserve a warning.

5. **Streamed lensing for gradients at high l_max** [tracked elsewhere]. The matrix lensing needs ~60 GB per
   cosmology under the gradient at ACT's l_max; ABCMB-k a3af514 `lensing_streamed`. The ACT session tracks the upstream
   PR in `/pscratch/sd/c/carag/ABCMB/TODO_streamed_lensing.md`.

## ABCMB-k

6. **Engine-level batched value + gradient API** [open]. One call `(theta batch, directions) -> C_ell + tangents`,
   including the derived-parameter tangents. Today the caller builds them: the profile driver with a per-row,
   per-direction Python jvp loop; `hmc/posterior.py` with one `jit(vmap(jvp))` of `add_derived_parameters` (works
   with `bbn_type="table"`).
7. **JAX likelihoods as a package module** [open]. plik-lite was cut in the 11-06 trim (restored to `hmc/plik_lite.py`);
   low-ell TT/EE live in `scan/lowl_like.py`; clipy's SimAll EE is a floor lookup with zero gradient (keep the
   cubic-interpolated table). Missing for the usual combinations: DESI BAO (needs r_d, D_M, D_H tangents) and the BBN
   likelihood (LinxStage already gives Jacobians). ACT DR6 lite: `scan/act_lite.py` (candl data, in the profile driver since 02a7c9f) can be reused.
8. **HyRex tangent sharding needs `XLA_FLAGS=--xla_force_host_platform_device_count=N` before JAX starts** [open].
   Without it the HyRex jvp is one serial CPU loop. Set it up in the engine, or warn.
9. **Compile cost** [document]. 6-14 min per new batch shape; keep shapes fixed (pad) and set
   `JAX_COMPILATION_CACHE_DIR` so restarts reuse it.

## hmc/ (this directory)

10. **Config-driven posterior** [open]: parameters, priors and likelihoods from a config like `scan/configs/`;
   `LCDMPosterior` is hard-coded to 7 parameters.
11. **Shorter warmup, multi-node fan-out** [open]: the Gauss-Newton metric was already near-optimal (|dH| 0.10 vs 0.088
    for a perfectly whitened Gaussian), so ~4 warmup trajectories should do; after a shared warmup the chains can be
    split over nodes (4 nodes: ~3 h wall instead of ~9).
12. **Start point** [open]: start at a Gauss-Newton step from the guess, not a fixed point (the Planck massive-nu best
    fit was ~30 chi2 off along theta_s with massless nu). The ACT driver already does this.
13. **Derived parameters** [open]: theta_s, sigma8 (needs Pk) in the output.
14. **LINX in the sampling** [next, after `_v1` finishes cleanly; user 2026-11-10]. YHe / Neff / T_nu from LINX
    instead of the BBN table, as the profile driver does: `scan/bbn_stage.LinxStage` gives the batch's values and
    exact Jacobians on the host CPU (~145 s per 32 rows without chi, against ~480 s per 64-chain ABCMB gradient
    call), and the engine takes y + J (theta - theta0) so the jvp carries LINX's tangent. Optional: the BBN
    likelihood (Y_p, D/H) with tau_n_fac and the rate shifts q as extra HMC parameters (~20 dims in all).
15. **Validate the halofit HMC against the MH reference** [optional]: predicted agreement <= 0.10 sigma (ABCMB-k
    halofit vs CLASS HMcode, linear response); a full `_v2` run would confirm it directly (~10 node-h; or fold it into
    the LINX run, item 14, and compare against a LINX-YHe CLASS reference).
16. **ABCMB-k vs high-precision CLASS residual** [open, small]: <= 0.08 sigma along omega_cdm / h / n_s, consistent
    with a ~3e-5 relative acoustic-scale offset (TT +-3e-4 oscillation above l 500). Candidates: HyRex vs HyRec,
    constants, the "recomb" lna grid; needs theta_s from ABCMB.

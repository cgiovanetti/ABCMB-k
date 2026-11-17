# ABCMB-k: open items

One list for the whole repo (profile driver, HMC, engine, upstream ABCMB). Add new items here, not in per-directory
files. Item numbers are stable IDs (other notes cite them), not priorities. Status in brackets; newest notes at the
end of an item. Items 1-17 came from `hmc/TODO.md` (merged 2026-11-12). Closed 2026-11-16 (user): 7 (JAX likelihood
module: not ours), 10 (config-driven HMC posterior: a collaborator's), 15 (halofit HMC validation: not needed), 18
(`_v1emwimp` run: folded into 20), 19 (profiles.ipynb: synced), 23 (more 2D profiles: m_chi-Neff was the only
interesting pair), 28 (Delta_Neff_init dilution note), 30 (`ABCMB-k_emwimp/` deleted; `neff_terms.py` kept in
`scan/diag/`).

## High priority (user, 2026-11-16)

31. **Fixed-iteration KenCarp4 perturbation solver** [-k done 2026-11-16; upstream ABCMB next]. diffrax's KenCarp4 with the Newton while loop
    replaced by `n_iter` unrolled chord iterations (Jacobian and its explicit inverse once per step, no gradient
    through them); valid for nonlinear species, unlike PR #63's single-iteration LinearKenCarp4. ABCMB-k:
    `abcmb/fixed_kencarp4.py`, opt-in `PE_solver="kencarp_fixed"`, `kencarp_n_iter` (default 10);
    `pytests/test_fixed_kencarp4.py` (stiff toys: value and tangent errors as Kvaerno5's at the same tolerance).
    Minimal patch against diffrax's own file: `/pscratch/sd/c/carag/ABCMB/.pr_review/kencarp63/fixed_kencarp4_vs_diffrax.diff`
    (bit-identical to the ABCMB-k module). The Nov-7 upstream review (same directory, README.md): N=8-10 with the
    inverse, 1.7-1.9x faster than Kvaerno5 for values (with the PR's looser dtmax for KenCarp). 2026-11-16, -k batched
    at production settings, same step control (`scan/diag/kencarp_bench.py`, B=32 on 4 A100, warm; results in
    `logs/kencarp_bench/`): with N=10 and rtol_limber_k_PE = 1e-4 (now the default for kencarp_fixed) values 131 s vs
    Kvaerno5 162 s (1.24x), gradients (6 directions) 360 s vs 553 s (1.54x); TT/EE <= 1.4e-4 and Clpp <= 1.8e-4 against
    Kvaerno5 at 10-100x tighter tolerances (Kvaerno5 8.4e-5 / 1.2e-4), tangents within 1.5e-5 of peak of Kvaerno5's.
    The first gradient benchmark (686 s, 0.81x) carried a tangent through M^-1 (gdt has a numerically zero, not
    symbolic, tangent): stop_gradient on M^-1 alone halved it. Tried and dropped: a custom stage tangent (K chord
    iterations on the linearised stage equation) is unstable below K~6 (the once-per-step Jacobian is too far from the
    stage Jacobians for ABCMB's stiff modes) and saves only 11% at K=6 (309 s). Floor of this design ~250 s; lower needs
    per-stage Jacobians (the PR's cost) or work outside the solver. Kvaerno5 stays the default; kencarp_fixed is a
    1.24x / 1.54x opt-in.
22 (see below). **LINX gradient speedup**: done 2026-11-16 with item 20.

## Runs and decisions

24. **`_v1act`** (ACT DR6 lite LCDM profile) [ACT session]. 61111914 timed out 11-13 at 10 h: omega_b and n_s done
    (21/21 converged), the other four at 1-2 active rows. Resume 61404156 (11-16) on 4 nodes with
    PA_POIS=h,omega_cdm,ln10As,tau_reion and the r4 / r5 LM checkpoints copied to r2 / r3 (POIS[rank::4]; originals in
    logs/act/); the prepass reruns first (not checkpointed). When done: results table + delete STATE_v1act_r4/r5.

## Profile driver (scan/)

20. **Gauss-Newton crawl along decoupling nuisances** [fix on main 2026-11-16; open: a production check and one NaN
    lane]. With a parameter whose effect fades out (m_chi -> heavy) free, the gradient and the true curvature go as
    eps', J^T J as eps'^2, so GN overshoots along it (`_v1emwimp`: 60 iterations, 2-8 rows stalled per node). The
    missing curvature is LINX's own second-order term sum_k (dchi2/dy_k) d2y_k/dx_a dx_b, not ABCMB's: with it the
    profile curvature along log10_m_chi matches the converged profile's value differences to <= 3% at 11-26 MeV,
    where GN alone is 0.58x-0.06x (CHANGELOG 2026-11-16). On main: `LinxStage(curv=...)` (forward over forward
    through LINX; needed the vendored-LINX polylog fix, thermo._Li_BE), config `linx_curv` (default log10_m_chi with
    chi on), the matrix then |eigenvalues|; the second-order program runs on the host CPU next to the engine.
    LM race from four `_v1emwimp` starts (scan/diag/gn_curv_check.py lm): in one iteration both Delta_Neff_init rows
    reached the chi2 the GN run ended at after 60 iterations (2763.456 vs 2763.453; 2764.84 vs 2764.48); the run
    then aborted (below). Flat directions: converged starts count as distinct minima only when their chi2 differ by
    > MS_DCHI2 or a barrier separates them (values along the segment), branches only from lower or barrier-separated
    minima, and spread starts > PA_MS_START_DCHI2 above their row's start are pulled back
    (pytests/test_profile_decoupling.py). Open: (a) in the race's second iteration the batched second-order program
    returned a non-finite lane, and the per-lane fallback then compiled the single-row program on 4 host devices at
    once (LLVM out of host memory, abort); `hessian_rows` now has no per-lane fallback (that row keeps GN) and the
    cause is being traced (scan/diag/linx_curv_nan.py on the iteration-2 rows, logs/linx_curv_x_it2.txt). (b) LINX's
    CPU programs never hit the persistent compile cache (a new key per process): each job compiles the Jacobian
    program (~17 min) and the second-order one (~45 min) afresh. (c) A production re-run (`_v1emwimp` or a 1D
    log10_m_chi + Delta_Neff_init slice) to measure iterations against the 60 of the GN run. Generic limit: a
    decoupling inside ABCMB (EDE z_c / theta_i as f_EDE -> 0) needs ABCMB's own second derivatives (item 21).
22. **LINX gradient speedup** [done 2026-11-16, with item 20]. LINX runs push the engine's 9 inputs (h, omega_b,
    omega_cdm, n_s, ln10As, tau_reion, YHe, Neff, T_nu) and chain LINX's Jacobian (`_gn_block_linx`) instead of ~20
    parameter directions; equal to the per-parameter path to 1e-12 (gradient) / 2e-15 (GN). Warm cost per block:
    ABCMB with 9 directions vs 20, plus the LINX Jacobian (~190 s per 32-row chunk) as before. `PA_AD_BCHUNK` can
    roughly double for the same memory (not yet shaken out).
33. **Wilks / coverage calibration of the profile intervals** [deferred; from scan/README.md]: toys only if a second
    mode ever appears within a few Delta chi2.

## Upstream ABCMB

1. **A gradient standard JAX samplers can call** [open, highest value]. `jax.grad` (what blackjax / numpyro NUTS
   call) does not work as shipped: the default adjoint is diffrax `ForwardMode` (forward only), and
   `RecursiveCheckpointAdjoint` ran but gave O(1)-wrong omega_b / omega_cdm gradients (2026-09-15; never re-measured
   after #47). Proposal: a `custom_vjp` around params -> C_ell whose backward pass is the forward-mode Jacobian over
   the few input parameters (one vmapped jvp); then `jax.grad` of any scalar likelihood is right and costs one
   jacfwd. Single-chain NUTS would then work unmodified.
2. **Batched parameters** [planned: the -k port, item 25]. Many-chain lockstep sampling needs one call per leapfrog
   step for all chains. Upstream `vmap` flattens out past B ~ 16 and B = 64 did not compile in 20 min (2026-10-13
   bench). Port the staged forward-mode batched gradient (`scan/batched_grad.py`), not only `call_batched`.
3. **Smooth BBN-table YHe** [deprioritized 2026-11-10: the sampling will take YHe from LINX, item 14].
   `bbn_type="table"` interpolates YHe(omega_b, Neff) bilinearly, so dYHe/domega_b is piecewise constant (kinks every
   ~5e-5 in omega_b). Harmless for HMC correctness, but a non-smooth force; bicubic would fix it. Same code in -k.
4. **Silent spec footguns** [open, small]. Unknown spec keys are kept silently (`load_specs`), so a typo is a no-op:
   the scan configs and `hmc/posterior.py` pass `l_max_ur` / `l_max_ncdm`, read nowhere (the real keys are
   `l_max_massless_nu` / `l_max_massive_nu`; same values, so no effect). `N_nu_massive > 0` without
   `user_species=(MassiveNeutrino,)` silently evolves no massive neutrino. Both deserve a warning.
5. **Streamed lensing for gradients at high l_max** [tracked elsewhere]. The matrix lensing needs ~60 GB per
   cosmology under the gradient at ACT's l_max; ABCMB-k a3af514 `lensing_streamed`. The upstream PR is tracked in
   `/pscratch/sd/c/carag/ABCMB/TODO_streamed_lensing.md`.
   Caveat (2026-11-16): under `jax.grad` the streamed path needs MORE memory than the matrix path (item 34), so
   don't port it as-is.
25. **Port the -k engine into ABCMB main as opt-in options** [planned 2026-11-07, not started]. The batched path
   (`call_batched`, tabulated exp(-kappa) with Hermite lookup and exact slope, `visibility_prime`), the "recomb" lna
   grid and the AD fixes, selected by specs rather than replacing upstream's single-cosmology path. Covers item 2.
26. **Upstream bug-fix PRs** [the user opens them]. #52 / #53 / #54 open; not yet sent: the LINX sync and the OLE
   parameter-order fix. Candidate diffs in `scan/.upstream_pr/`.
27. **Upstream test suite** [proposed 2026-10-20, awaiting go-ahead]. Three tiers: CPU CI (accuracy vs CLASS,
   gradients, extensibility null tests, perf proxies), then nightly Perlmutter GPU benchmarks.
34. **Lensing Wigner matrices: speed + memory** [parked 2026-11-16; user: the contributor may be mistaken, leave it].
   A contributor said the Wigner calculation is redundant ("a static variable recomputed in a loop") and would save
   a lot of memory. Found: (a) speed: upstream `lensed_Cls` rebuilds its 12 d-matrices with 12 separate scans on
   every call (0.6 s of 5.9 s at l_max 2500; 2.6 s of 15.9 s at 6325 + 5000); one fused scan fixes it, spectra
   unchanged to 1e-15, peak memory unchanged. Ready: branch `lensing_wigner` in ../ABCMB (no commits) +
   `ABCMB/.pr_review/lensing_wigner/lensing_wigner.diff`. (b) memory: the Wigner matrices are most of the forward
   peak (1.28 of 1.77 GiB default; 11.5 of 14.6 GiB at the ACT range). Lensing-core temp in (n_mu, n_ell) arrays,
   matrix path forward / jvp x6 / grad = 21 / 39 / 34; -k streamed = ~0 / ~0 / 56 (its scan over l stores the
   per-l Wigner rows as residuals). Options if revisited: mirror symmetry (d1-1, d2-2, d3-1 are reflected d11,
   d22, d31; ~25-40% of the Wigner memory) or a mu-chunked + `jax.checkpoint` matrix path (all three AD modes).
   Numbers and scripts: `ABCMB/.pr_review/lensing_wigner/NOTES.md`.

## ABCMB-k engine

6. **Engine-level batched value + gradient API** [open]. One call `(theta batch, directions) -> C_ell + tangents`,
   including the derived-parameter tangents. Today the caller builds them: the profile driver with a per-row,
   per-direction Python jvp loop; `hmc/posterior.py` with one `jit(vmap(jvp))` of `add_derived_parameters` (works
   with `bbn_type="table"`).
8. **HyRex tangent sharding needs `XLA_FLAGS=--xla_force_host_platform_device_count=N` before JAX starts** [open].
   Without it the HyRex jvp is one serial CPU loop. Set it up in the engine, or warn.
9. **Compile cost** [document]. 6-14 min per new batch shape; keep shapes fixed (pad) and set
   `JAX_COMPILATION_CACHE_DIR` so restarts reuse it.
21. **AD second derivatives are inaccurate** [open, after the checkpoint; user: "seems like a bug that should be
   fixed"]. Per-ell d2C_l by jvp-of-jvp is off 2e-5..7e-4 (TT, l > 500), 0.5-3% (TE/EE) and 85% (l < 30) against
   value second differences: enough to wreck an exact Hessian (Gauss-Newton matches the value curvature to <1%, so
   the profiles are unaffected). Suspects: C^1 / linear interpolation tables, adaptive step sizes frozen under AD. The
   LINX Delta_Neff_init^2 nested jvp NaN is fixed (vendored-LINX polylog at z = 1, 2026-11-16). Harness
   `/pscratch/sd/c/carag/ABCMB-k_lnagrid/hess/`. No longer blocks item 20 (its curvature is LINX's).
32. **diffrax's KenCarp4 (with its Newton while loop) for CPU runs** [open; user 2026-11-16]. On CPU the while
   loop is cheap (no GPU lockstep or launch overhead), so diffrax.KenCarp4 itself may cut CPU runtime against
   Kvaerno5 (6 stages vs 7; the Nov-7 GPU review had it at 5.99 / 18.19 s vs Kvaerno5 8.10 / 20.23 s massless /
   massive, with PR #63's looser dtmax). Wire it as another `PE_solver` option ("kencarp_diffrax":
   `diffrax.KenCarp4()` with `MultiTerm(zero explicit term, the vector field)`, as in the review's tree), then time it
   on a CPU node against Kvaerno5 for values and forward-mode gradients at matched accuracy.

## hmc/

11. **Shorter warmup, multi-node fan-out** [open]: the Gauss-Newton metric was already near-optimal (|dH| 0.10 vs
    0.088 for a perfectly whitened Gaussian), so ~4 warmup trajectories should do; after a shared warmup the chains
    can be split over nodes (4 nodes: ~3 h wall instead of ~9).
12. **Start point** [open]: start at a Gauss-Newton step from the guess, not a fixed point (the Planck massive-nu best
    fit was ~30 chi2 off along theta_s with massless nu). The ACT driver already does this.
13. **Derived parameters** [open]: theta_s (item 17), sigma8 (needs Pk) in the output.
14. **LINX in the sampling** [next, after `_v1` finishes cleanly; user 2026-11-10]. YHe / Neff / T_nu from LINX
    instead of the BBN table, as the profile driver does: `scan/bbn_stage.LinxStage` gives the batch's values and
    exact Jacobians on the host CPU (~145 s per 32 rows without chi, against ~480 s per 64-chain ABCMB gradient
    call), and the engine takes y + J (theta - theta0) so the jvp carries LINX's tangent. Optional: the BBN
    likelihood (Y_p, D/H) with tau_n_fac and the rate shifts q as extra HMC parameters (~20 dims in all).
16. **ABCMB-k vs high-precision CLASS residual** [open, small]: <= 0.08 sigma along omega_cdm / h / n_s, consistent
    with a ~3e-5 relative acoustic-scale offset (TT +-3e-4 oscillation above l 500). Candidates: HyRex vs HyRec,
    constants, the "recomb" lna grid; needs theta_s from ABCMB (item 17).
17. **theta_s as an ABCMB derived parameter** [open; user 2026-11-12; upstream ABCMB and -k]. Neither code outputs
    the acoustic scale. Add 100 theta_s (CLASS convention: r_s / D_A at the visibility peak, `100*theta_s`) and
    theta_* (at tau = 1), with tangents, from the Background. Needed to chase item 16 (compare to CLASS's 1.04186 at
    the posterior mean), to report it in HMC/MH chains, and to sample in theta_s instead of h if a run ever needs it.

## Physics notes for the write-up

29. **Profiles vs posteriors for Neff**: Planck 2018 base_nnu's posterior mean (2.92) sits 0.08 above its own best
    fit (2.836); `_v7simall` reproduces the best fit (2.846). Expect MCMC means ~0.1 above profile best fits.

# ABCMB-k: open items

One list for the whole repo (profile driver, HMC, engine, upstream ABCMB). Add new items here, not in per-directory
files. Item numbers are stable IDs (other notes cite them), not priorities. Status in brackets; newest notes at the
end of an item. Items 1-17 came from `hmc/TODO.md` (merged 2026-11-12). Closed 2026-11-16 (user): 7 (JAX likelihood
module: not ours), 10 (config-driven HMC posterior: a collaborator's), 15 (halofit HMC validation: not needed), 18
(`_v1emwimp` run: folded into 20), 19 (profiles.ipynb: synced), 23 (more 2D profiles: m_chi-Neff was the only
interesting pair), 28 (Delta_Neff_init dilution note), 30 (`ABCMB-k_emwimp/` deleted; `neff_terms.py` kept in
`scan/diag/`). Closed 2026-11-18: 24 (`_v1act` done: `scan/README.md`, CHANGELOG 2026-11-18).

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
    2026-11-17: at ACT settings kencarp_fixed is slower than Kvaerno5 (675 vs 610 s warm values: 4th order takes
    ~1.85x the steps at high k, and the dense inverse dominates the step). **`PE_solver="kvaerno5_fixed"`**
    (`abcmb/fixed_kvaerno5.py`, CHANGELOG 2026-11-17) fixes both: Kvaerno5's tableau, 6 chord iterations, a structured
    inverse from the Jacobian's sparsity (`Model.with_pe_structure`), convergence term in the error estimate. Warm:
    plik engine values 47.8 s / gradients 218 s (Kvaerno5 162 / 553), ACT values 182 s (610), gradients 757 s;
    accuracy as Kvaerno5's. 2026-11-18 (CHANGELOG): checks done (ACT warm gradients 1849 -> 814 s; edge / m_nu gradients; snapshot Pk
    difference is Kvaerno5's own error; CLASS test unchanged); driver switch PA_PE_SOLVER in (default kvaerno5);
    IMEX KenCarp4 tested and not pursued; upstream main serial 1.9x / 2.8x. Open: (a) the default decision (user)
    + regenerate snapshots with it; (b) compile time (gradients ~1000-1150 s per shape vs 180-400): fori_loop with
    unroll; (c) apply the structured inverse as hub + block solves instead of a dense M^-1 (the dense M^-1 v now
    dominates the step); (d) upstream form (the diffrax-minimal patch is still the KenCarp one); (e) PA knob for
    kvaerno5_n_iter (config engine block works today).
    Branch `fixed-solver` (worktree ../ABCMB-k-solver; main carries the solver as of b920b05 and is not touched by
    this work): (b) done there, kvaerno5_jit_vf (RHS traced once under a nested jit) cuts compile ~2x for values and
    ~2.4x for gradients at no warm cost (ACT gradients ~985 -> ~410 s); (c) tried, kvaerno5_factored is +7% for values
    and -36% for gradients, left off. To merge when the solver is folded in.
22 (see below). **LINX gradient speedup**: done 2026-11-16 with item 20.

## Runs and decisions

## Profile driver (scan/)

20. **Gauss-Newton crawl along decoupling nuisances** [fixed on main 2026-11-16 (8968fb1); open: a production
    re-run, the user's call]. With a parameter whose effect fades out (m_chi -> heavy) free, the gradient and the
    true curvature go as eps', J^T J as eps'^2, so GN overshoots along it (`_v1emwimp`: 60 iterations, 2-8 rows
    stalled per node). The missing curvature is LINX's own second-order term sum_k (dchi2/dy_k) d2y_k/dx_a dx_b,
    not ABCMB's: with it the profile curvature along log10_m_chi matches the converged profile's value differences to
    <= 3% at 11-26 MeV, where GN alone is 0.58x-0.06x. On main: `LinxStage(curv=...).hessian_rows` (forward over
    forward through LINX; needed the vendored-LINX polylog fix, thermo._Li_BE), config `linx_curv` (default
    log10_m_chi with chi on), the matrix then |eigenvalues|; the second-order program runs on the host CPU next to the
    engine (~440 s per 32-row chunk warm, ~3.5x the Jacobian's 180 s). LM race, same four `_v1emwimp` starts, 8
    iterations each (job 61436899, logs/gn_curv_check_race.log): GN + LINX term reaches the chi2 `_v1emwimp` ended at
    on all four rows and converges the two Delta_Neff_init rows (|g| 0.025, 0.017) that stalled there; GN alone
    leaves Delta_Neff_init[4] 8.6 above (|g| 27) and [7] unconverged. Per iteration 13.0 vs 8.4 min on these 4-row
    blocks (the second-order program is longer than a 4-row engine call; at production blocks of 24 it overlaps the
    engine). Flat directions: converged starts count as distinct minima only when their chi2 differ by > MS_DCHI2
    or a barrier separates them, branches only from lower or barrier-separated minima, spread starts
    > PA_MS_START_DCHI2 above their row's start pulled back (pytests/test_profile_decoupling.py). Notes: a batched
    second-order lane once came back non-finite (first race, iteration 2; the per-lane recompile it triggered ran the
    host out of memory); it did not recur at the same points (32 batched evaluations finite), and such a row now keeps
    GN for that iteration. LINX's CPU programs do hit the persistent compile cache when the code and the calling
    thread match (the race job loaded both). Generic limit: a decoupling inside ABCMB (EDE z_c / theta_i as
    f_EDE -> 0) needs ABCMB's own second derivatives (item 21).
    Gap seen in `_v1act` (2026-11-18): a STALLED start still seeds a branch whatever its chi2. ln10As@3.07's stalled
    spread start sat at the point's known minimum (543.878) and its 21-row branch (~6 node-h at the ACT range) found
    nothing below the envelope. Candidate: a stalled start seeds a branch only when it is lower by > MS_DCHI2.
22. **LINX gradient speedup** [done 2026-11-16, with item 20]. LINX runs push the engine's 9 inputs (h, omega_b,
    omega_cdm, n_s, ln10As, tau_reion, YHe, Neff, T_nu) and chain LINX's Jacobian (`_gn_block_linx`) instead of ~20
    parameter directions; equal to the per-parameter path to 1e-12 (gradient) / 2e-15 (GN). Warm cost per block:
    ABCMB with 9 directions vs 20, plus the LINX Jacobian (~190 s per 32-row chunk) as before. `PA_AD_BCHUNK` can
    roughly double for the same memory (not yet shaken out).
33. **Wilks / coverage calibration of the profile intervals** [deferred; from scan/README.md]: toys only if a second
    mode ever appears within a few Delta chi2.
35. **LINX polylog: second derivatives through the background were NaN** [fixed in ABCMB-k 2026-11-17; upstream LINX
    and ../ABCMB: diff ready, the user sends]. thermo's Bose-Einstein densities use Li(n, exp(mu/T)) at mu = 0 (z = 1),
    and LINX's `Li` is a complex series in log z whose derivatives beyond the first are NaN there; the background's
    dY already holds grad(rho_EM), so any second derivative through BackgroundModel (any input) got inf * 0 = NaN
    although z's tangent is 0 (item 21's Delta_Neff_init^2 NaN; it blocked item 20). Fix (special_funcs.py, replacing
    8968fb1's concrete-mu=0 stopgap): `Li` is a custom_jvp around the unchanged series (`_Li_series`) with
    dLi_n/dz = Li_{n-1}/z, closing on Li_1 = -log|1-z| and Li_0 = z/(1-z); for n <= 2 the coefficient is 0 at exactly
    z = 1 (exact for the zero tangent at mu = 0; a nonzero tangent there, a truly divergent derivative, comes out finite).
    The rule is linear in the tangent, as LINX's jax.grad inside dY needs (a predicate on the tangent cannot transpose).
    Values and LINX Jacobians bit-identical; warm one-cosmology programs values 4.33 / 4.45 s, Jacobian 20.78 / 21.05 s
    (new / old), compiles within 3%; LINX Hessian rows equal the stopgap's (checked against central differences).
    pytests/test_linx_polylog.py. Diff for LINX main: /pscratch/sd/c/carag/LINX-scratch/polylog_safe_derivative.diff
    (applies to bfba07c; checked there under jax_debug_nans with env linx312: values unchanged, derivatives at mu = 0 exact)
    (applies to ../ABCMB's vendored abcmb/linx/special_funcs.py too). Side note, pre-existing: the series' values are
    good to <~1e-11 on the positive axis but only 1e-8..1e-7 for z in (-1.5, -0.5) and ~2e-6 at |z| ~ 2.6 (n = 4);
    harmless at LINX's z = +-1 (Li_4(-1) 2e-14, Li_3(-1) 3e-8) (scan/diag/li_series_accuracy.py).

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

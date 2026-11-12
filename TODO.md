# ABCMB-k: open items

One list for the whole repo (profile driver, HMC, engine, upstream ABCMB). Add new items here, not in per-directory
files. Item numbers are stable IDs (other notes cite them), not priorities. Status in brackets; newest notes at the
end of an item. Items 1-17 came from `hmc/TODO.md` (merged 2026-11-12).

## Runs and decisions

18. **EM-WIMP 1D profile `_v1emwimp`** [resumed 2026-11-12: job 61174328, PA_MAXIT=60 (the cap counts across resumes;
    the rows were at it22-24), 4 nodes (rank 3 is done and idles; a different node count would restart from scratch)]. Jobs 61054076 and 61084162 both hit the 12 h limit.
    Done: log10_m_chi (m_chi > 10.29 MeV at 95%, > 14.4 MeV at 68%; the 2D gives 10.1) and n_s. The other six POIs
    are checkpointed (`scan/results/profile_prod_ad_STATE_v1emwimp_r0-2.npz`) with 22-25 of 78 rows per node still
    active, mostly the light-chi spread starts, and 2-5 grid points per profile not yet converged. Resume with the
    same command (`sbatch scan/profile_prod_emwimp.slurm`, 4 nodes: another node count deals the POIs out differently
    and the checkpoints no longer match), ~48 node-h per 12 h; or wait for item 20.
19. **`scan/profiles.ipynb`** [user]. The user's working copy has the Delta_Neff_init / Neff fix; the copy on
    Perlmutter has their uncommitted edits. Sync and commit.
24. **`_v1act`** (ACT DR6 lite LCDM profile, job 61111914) [queued; ACT session].

## Profile driver (scan/)

20. **Gauss-Newton crawl along decoupling nuisances** [open, after the 11-11 checkpoint]. With a parameter whose
    effect fades out (here m_chi -> heavy) free, the residuals change by eps*c_i: the gradient and the true curvature
    go as eps, J^T J as eps^2, so the GN step along it diverges. In `_v1emwimp` the GN curvature along log10_m_chi is
    5-20x below the measured profile curvature at 23-40 MeV, GN asks for +1.6..+50 dex, and LM damps hard (lambda
    1-1e3). LINX's m_chi / Delta_Neff_init AD derivatives match central differences (<~1%, 1.8 MeV-1 GeV): not a
    gradient bug. Generic: log-parametrized couplings, EDE z_c / theta_i as f_EDE -> 0. Fixes: exact curvature along
    the one or two soft directions (AD Hessian-vector products; needs item 21), or a model-side upper box (log10 m_chi
    <= 2: the profile is flat to <1e-3 above 45 MeV). Also: spread starts along a flat direction land on the box
    faces (a 1.8 MeV start at Delta chi2 up to 1700); cap the spread. Diagnostics:
    `/pscratch/sd/c/carag/ABCMB-k_emwimp/v1/look_gn_mass.py`, `look_lam.py`.
22. **LINX gradient speedup** [open, after the checkpoint]. The driver pushes all 19 free directions through ABCMB,
    but with LINX the engine sees only 8 effective inputs (the BBN nuisances act through YHe / Neff / T_nu): push 8
    and chain LINX's Jacobian, ~2.4x cheaper gradients for BBN-likelihood runs.
23. **More 2D joint profiles** [open]. The 2D driver (`PA_POI2D`, `profile_2d`) works (`_v1emwimp2d`). LCDM + Neff
    candidates: Neff-h, Neff-omega_cdm; ~14 node-h per pair.

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
25. **Port the -k engine into ABCMB main as opt-in options** [planned 2026-11-07, not started]. The batched path
   (`call_batched`, tabulated exp(-kappa) with Hermite lookup and exact slope, `visibility_prime`), the "recomb" lna
   grid and the AD fixes, selected by specs rather than replacing upstream's single-cosmology path. Covers item 2.
26. **Upstream bug-fix PRs** [the user opens them]. #52 / #53 / #54 open; not yet sent: the LINX sync and the OLE
   parameter-order fix. Candidate diffs in `scan/.upstream_pr/`.
27. **Upstream test suite** [proposed 2026-10-20, awaiting go-ahead]. Three tiers: CPU CI (accuracy vs CLASS,
   gradients, extensibility null tests, perf proxies), then nightly Perlmutter GPU benchmarks.

## ABCMB-k engine

6. **Engine-level batched value + gradient API** [open]. One call `(theta batch, directions) -> C_ell + tangents`,
   including the derived-parameter tangents. Today the caller builds them: the profile driver with a per-row,
   per-direction Python jvp loop; `hmc/posterior.py` with one `jit(vmap(jvp))` of `add_derived_parameters` (works
   with `bbn_type="table"`).
7. **JAX likelihoods as a package module** [open]. plik-lite was cut in the 11-06 trim (restored to
   `hmc/plik_lite.py`); low-ell TT/EE live in `scan/lowl_like.py` (SRoll2 or SimAll, config `lowee`); clipy's SimAll
   EE is a floor lookup with zero gradient (keep the cubic-interpolated table). Missing for the usual combinations:
   DESI BAO (needs r_d, D_M, D_H tangents) and the BBN likelihood (LinxStage already gives Jacobians). ACT DR6 lite:
   `scan/act_lite.py` (candl data, in the profile driver since 02a7c9f) can be reused.
8. **HyRex tangent sharding needs `XLA_FLAGS=--xla_force_host_platform_device_count=N` before JAX starts** [open].
   Without it the HyRex jvp is one serial CPU loop. Set it up in the engine, or warn.
9. **Compile cost** [document]. 6-14 min per new batch shape; keep shapes fixed (pad) and set
   `JAX_COMPILATION_CACHE_DIR` so restarts reuse it.
21. **AD second derivatives are inaccurate** [open, after the checkpoint; user: "seems like a bug that should be
   fixed"]. Per-ell d2C_l by jvp-of-jvp is off 2e-5..7e-4 (TT, l > 500), 0.5-3% (TE/EE) and 85% (l < 30) against
   value second differences: enough to wreck an exact Hessian (Gauss-Newton matches the value curvature to <1%, so
   the profiles are unaffected). Suspects: C^1 / linear interpolation tables, adaptive step sizes frozen under AD. The
   LINX Delta_Neff_init^2 nested jvp returns NaN. Harness `/pscratch/sd/c/carag/ABCMB-k_lnagrid/hess/`. Blocks item 20.

## hmc/

10. **Config-driven posterior** [open]: parameters, priors and likelihoods from a config like `scan/configs/`;
    `LCDMPosterior` is hard-coded to 7 parameters.
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
15. **Validate the halofit HMC against the MH reference** [optional]: predicted agreement <= 0.10 sigma (ABCMB-k
    halofit vs CLASS HMcode, linear response); a full `_v2` run would confirm it directly (~10 node-h; or fold it
    into the LINX run, item 14, and compare against a LINX-YHe CLASS reference).
16. **ABCMB-k vs high-precision CLASS residual** [open, small]: <= 0.08 sigma along omega_cdm / h / n_s, consistent
    with a ~3e-5 relative acoustic-scale offset (TT +-3e-4 oscillation above l 500). Candidates: HyRex vs HyRec,
    constants, the "recomb" lna grid; needs theta_s from ABCMB (item 17).
17. **theta_s as an ABCMB derived parameter** [open; user 2026-11-12; upstream ABCMB and -k]. Neither code outputs
    the acoustic scale. Add 100 theta_s (CLASS convention: r_s / D_A at the visibility peak, `100*theta_s`) and
    theta_* (at tau = 1), with tangents, from the Background. Needed to chase item 16 (compare to CLASS's 1.04186 at
    the posterior mean), to report it in HMC/MH chains, and to sample in theta_s instead of h if a run ever needs it.

## Physics notes for the write-up

28. **Delta_Neff_init is inert radiation fixed at LINX's T_start**, so a chi that annihilates after T_start dilutes
    it: one unit adds 0.21 to the final Neff at m_chi = 21 MeV, 0.26 above 60 MeV. The EM-WIMP results are reported
    in LINX's final Neff, so they are right; describe the model as extra radiation present at the start of BBN.
29. **Profiles vs posteriors for Neff**: Planck 2018 base_nnu's posterior mean (2.92) sits 0.08 above its own best
    fit (2.836); `_v7simall` reproduces the best fit (2.846). Expect MCMC means ~0.1 above profile best fits.

## Housekeeping

30. **Scratch workspaces** [after item 18]. `/pscratch/sd/c/carag/ABCMB-k_emwimp/` (one-off LINX / GN / notebook
    diagnostics): delete, keeping `neff_terms/neff_terms.py` (the `_v6bbn` chi2 split by likelihood). Keep
    `/pscratch/sd/c/carag/planck2018_tables/` (the PLA 68% table PDF and the extracted base_nnu pages).

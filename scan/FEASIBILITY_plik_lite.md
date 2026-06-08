# Feasibility: ABCMB + Planck plik-lite for a frequentist profile

**Date:** 2026-06-08 · **Branch:** `perk-perf` · **Hardware:** 1 node = 4× A100-80GB (Perlmutter)

**Verdict (short):** *Feasible today* for ΛCDM and ΛCDM+1-extension-parameter
profiles via a sharded brute/adaptive grid scan — a full profile is **hours on a
handful of nodes**. The unit cost at Planck resolution is **~0.66 s/cosmology**
(l_max=2508, lensed, massless ν, one 4-GPU node, memory-limited at B=256). The
plik-lite χ² is fully validated against the literature best fit. Beyond ~5
effective scan dimensions the brute grid hits the curse of dimensionality and
the differentiable-optimizer route (below) is the recommended next step.

---

## 1. What was built

All in `scan/` (this is the frequentist-scan area; `scan_slice.py` is the
multi-node harness whose `summarize` hook these replace):

- **`plik_lite.py`** — a pure-JAX re-implementation of cobaya's
  `PlanckPlikLite` binning, arranged so the per-cosmology pieces (bin theory D_l,
  inverse-covariance contraction, profile the calibration nuisance) vmap over the
  batch axis. Turns a `(B, n_ell)` `BatchedOutput` block into a `(B,)` χ² vector
  — the artifact a scan keeps. Loads the on-disk data at
  `…/Neal_ACTDR6/cobaya_packages/data/planck_2018_pliklite_native/` (613 bins =
  215 TT + 199 TE + 199 EE, ℓ→2508, full 613×613 covariance).
- **`validate_plik.py`** — correctness gate (below).
- **`bench_plik_runtime.py`** — realistic runtime sweep at l_max=2508 + lensing.
- **`profile_demo.py`** — an end-to-end 2D frequentist profile (`profile_demo.png`).

### Normalization / convention bridge (validated, not assumed)
ABCMB returns **raw, dimensionless** C_l (the (ΔT/T)² angular power). plik-lite
wants D_l = ℓ(ℓ+1)C_l/2π in **μK²**. The bridge is
`D_l = ℓ(ℓ+1)/(2π) · C_l · T_CMB_μK²` with T_CMB = 2.7255e6 μK (ABCMB's
`TCMB0=2.349e-4 eV` ⇒ 2.7255 K). A_planck enters as `model → model / A_planck²`.

---

## 2. Correctness — VALIDATED

Run at the Planck 2018 base-ΛCDM best fit (massless ν, l_max=2508, lensed):

| check | result | expected |
|---|---|---|
| D_l^TT(ℓ=220) | **5737 μK²** | ~5700 (first acoustic peak) |
| full TTTEEE χ² (A_planck=1) | 610.2 | — |
| full TTTEEE χ² (A_planck profiled) | **607.6** @ A=1.0012 | — |
| χ² / N_data | **0.99** | ~1 |
| profile-demo global min χ² | **584.5** | literature plik_lite ≈ 585 |
| batched vs single χ² | Δ ≈ **0.1** | (noise floor, see §5) |

The min χ² landing on the literature value confirms binning, covariance, the
T_CMB² normalization, **and** the TE/EE sign conventions are all correct.

---

## 3. Runtime — the key measurement

The CHANGELOG perf numbers (0.44 s/param, B=512) are at **l_max=800, lensing
off**. A Planck plik-lite run needs **l_max≥2508 with lensing** (the data is
lensed). Both the solve and the LoS scan grow with l_max, so the unit cost was
measured, not extrapolated.

**l_max=2508, lensed, massless ΛCDM, post-compile (incl. the plik-lite χ²):**

| B (per node) | B_local (per GPU) | s/param | cosmo/s/node | notes |
|---|---|---|---|---|
| 16  | 4  | 3.75 | 0.27 | dispatch-bound |
| 64  | 16 | 1.30 | 0.77 | |
| 128 | 32 | 0.89 | 1.12 | |
| **256** | **64** | **0.67** | **1.49** | **memory-limited max** |
| 512 | 128 | — | — | **OOM** (~49 GB/dev needed) |

Single-GPU (`shard=False`): B=8 → 8.0, B=32 → 3.34 s/param. The 4-GPU sharded
path is ~**3.7× faster** at equal B_local — near-linear scaling, as expected for
an embarrassingly-parallel B-axis (GSPMD, no collectives).

**One massive ν (0.06 eV, the Planck baseline) is essentially free:** B=64 →
1.272, B=128 → 0.889, B=256 → **0.670** s/param — within noise of the massless
numbers, and it still fits at B=256. The lensed l_max=2508 photon hierarchy
dominates cost/memory so thoroughly that the extra ncdm hierarchy barely
registers. So the *production* unit cost (Planck uses 1 massive ν) is the same
**~0.67 s/param**.

**Takeaways**
- **Unit cost ≈ 0.66 s/cosmology** at the memory-limited max (B=256, B_local=64)
  on one 4-GPU node. ~1.5× the l_max=800 toy cost — the lensed Planck tail is
  not free but is not prohibitive.
- **Memory, not the solver, caps B per node.** B=256 (B_local=64) is the safe
  max; B=512 (B_local=128, ~49 GB/dev est.) OOMs on 80-GB cards. (The `peakGB`
  column under-reports the transient; the internal estimator + the OOM are the
  real guide.) The persistent saved-trajectory tensor ∝ B_local is the binding
  term — independent of k_chunk.
- **Compile** is ~3–5 min per (B, k_chunk) config, cached persistently on
  `$SCRATCH` ⇒ a once-per-shape cost amortized across all scan tasks/nodes.
- **Throughput ≈ 1.5 cosmo/s/node, linear in nodes.** ~5.4k/hr/node,
  ~130k/day/node.

---

## 4. Scanning strategy for the profile

### 4a. The architecture-specific trick: collapse the amplitude analytically
C_l scales **linearly** with A_s, and the calibration enters as `model/A_planck²`,
so the whole combination `α = (A_s/A_s_ref)/A_planck²` is a single multiplicative
amplitude on a spectrum computed at a *reference* A_s. Profiling it is **closed
form** (`χ²_min = a − b²/c`, see `PlikLite.profile_amplitude`) — **no extra ABCMB
runs**. This removes A_s *and* A_planck from the scan grid for free.
(Exact up to the small lensing non-linearity in A_s; promote A_s to a real
dimension for a publication-grade result.)

So the parameters that actually need an ABCMB evaluation for a ΛCDM/plik-lite
profile are **{h, ω_b, ω_cdm, n_s}** — 4D. (τ is unconstrained by high-ℓ TT/TE/EE
alone — degenerate with the amplitude — so it needs an external τ prior from
low-ℓ EE, or a joint low-ℓ likelihood; it is *not* a scan dimension here.)

### 4b. Demonstrated: 2D grid profile (works end-to-end)
`profile_demo.py` lays a 25×25 grid over (n_s, ω_cdm), evaluates all 625
cosmologies through `call_batched(shard=True)` in padded B=256 batches, profiles
the amplitude analytically, and profiles each parameter out. Result (see
`profile_demo.png`): clean elliptical Δχ² surface with 1/2/3σ contours and
parabolic 1D profiles. **625 cosmologies in 641 s (incl. compile)** on one node.
Found n_s = 0.962 ± 0.003, ω_cdm = 0.1213 (intervals tight because h, ω_b, τ were
held fixed — this is a *sliced* 2D profile, not the full marginalization).

### 4c. Cost model for the full profile (brute grid, 0.66 s/param, linear nodes)
N_grid points × 0.66 s ÷ N_nodes:

| scan | grid | points | 1 node | 8 nodes | 16 nodes |
|---|---|---|---|---|---|
| ΛCDM 4D | 15/dim | 5.1e4 | 9.3 h | 1.2 h | 35 min |
| ΛCDM 4D | 20/dim | 1.6e5 | 29 h | 3.7 h | 1.8 h |
| ΛCDM+1 5D | 15/dim | 7.6e5 | 139 h | 17 h | 8.7 h |
| ΛCDM+2 6D | 15/dim | 1.1e7 | 2080 h | 260 h | 130 h |

**Brute grid is practical to ~4–5 effective dimensions** (ΛCDM, or ΛCDM + one
extension like N_eff or Σmν) — a one-off profile is hours on 8–16 nodes. Use a
coarse grid → adaptive refinement near the minimum (the demo's ω_cdm interval was
finer than one grid step, so adaptivity is needed regardless). At 6D the brute
grid blows up.

### 4d. Recommended for higher dimensions: differentiable optimizer profile
ABCMB is differentiable — the point of the code. The scalable frequentist profile
is: for each of ~30 values of the parameter of interest, **minimize χ² over the
≤4 nuisance cosmo params via a JAX gradient optimizer**, and **vmap that optimizer
over the 30 profile points** so each iteration is one batched (grad-augmented)
`call_batched`. Rough cost: ~30 points × ~40 iters × (~2.5× a forward call) ≈ **~1
node-hour per 1-D profile** — ~10× cheaper than the 4D brute grid and dimension-
independent in the nuisances. **Open item:** reverse-mode AD through the lensed
l_max=2508 batched solve is memory-heavy (diffrax adjoint); its per-device memory
must be checked before committing — this is the single most valuable next
experiment.

---

## 5. Caveats / things to nail before publication

1. **Profile-interval precision floor.** Batched χ² differs from single-call χ² by
   ~0.1 (chunked-vmap diffrax PID noise, the documented ~1e-4 Cl floor). Since
   Δχ²=1 is 1σ, this is ~10% of a 1σ interval. Mitigate by smoothing/fitting the
   profile curve and re-evaluating the minimum + Δχ² crossings with single
   (unchunked) calls, or by tightening `rtol_large_k_PE`.
2. **A_s amplitude approximation** (§4a) — exact only up to lensing non-linearity.
3. **τ needs an external prior** (low-ℓ EE) — not constrained by plik-lite alone.
4. **ABCMB vs CLASS accuracy (~0.2% TT)** sets a χ² systematic floor. The min
   χ²=584.5 matching the literature shows it does not bias the minimum at present
   accuracy, but a coherent bias should be checked against the profile location
   for precise intervals.
5. **Massive neutrinos are NOT a cost concern** here (measured: 1×0.06 eV is
   within noise of massless at l_max=2508+lensing, and fits at B=256). Validate
   the χ² with the massive ν included for the final number (the demo profiled
   massless).

---

## 6. Bottom line

The runtime **is good enough** for frequentist profiling of ΛCDM and one-extension
models with the existing batched/sharded architecture: validated likelihood,
~0.66 s/cosmology, linear node scaling, and an amplitude-collapse trick that
removes two nuisances for free. A full ΛCDM profile is a few node-hours. The
natural follow-ups are (i) a memory check of the differentiable-optimizer profile
for higher-dim models, and (ii) wiring `plik_lite.chi2_from_abcmb` into
`scan_slice.py`'s `summarize` hook to ship the multi-node scan.

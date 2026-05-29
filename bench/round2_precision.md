# round2_precision.md — Floating-point / tolerance levers for the batched solve

Battle-royale round 2, precision/tolerance specialist. Branch `perk-perf`,
post-keystone. **Static analysis only — no Python/GPU run.** All code facts cited
`file:line` from a direct read of `abcmb/{perturbations,main,model_specs,spectrum,
background,species}.py` and the two gates. Wall-clock numbers quoted from in-tree
artifacts (`CHANGELOG.txt`, `bench/profile_stages.log`, `bench/flipped_summary.txt`,
`bench/accuracy_cubic.log`).

---

## 0. The thing the whole brief turns on: ABCMB integrates the STIFF hierarchy with NO TCA

This is the single most important fact for every precision lever below, and it is
NOT spelled out in the prior memos. **ABCMB has no tight-coupling-approximation
integration phase.** `get_starting_time` (perturbations.py:251-291) and the
`R_tc=0.0015`/`R_large=0.07` thresholds (model_specs.py:56-57) only choose the
*start time* `lna_start` of a *single* diffrax solve (perturbations.py:406-411).
From `lna_start` to today, `evolution_one_k` integrates the **full** Einstein-
Boltzmann hierarchy with the bare tight-coupling source terms explicit:

- Baryon: `theta_prime = -theta + cs2*k**2*delta/aH + R/tau_c/aH*(theta_g-theta)`
  (species.py:1221).
- Photon: `theta_prime = k**2/aH*(delta/4.-sigma) + (theta_b-theta)/aH/tau_c`
  (species.py:1371); and every photon/polarization multipole carries a
  `-F[L]/aH/tau_c` Thomson damping term (species.py:1372-1385).

`tau_c = 1/(a·n_e·σ_T·c)` (background.py:670-691). Deep in tight coupling
`tau_c → 0`, so `1/(aH·tau_c)` is **enormous** (this is exactly why CLASS/CAMB
switch to a TCA expansion: integrating `R/(aH·tau_c)·(θ_g − θ_b)` directly is the
classic stiff, catastrophic-cancellation term — `θ_g` and `θ_b` are driven to be
nearly equal precisely so that `(θ_g − θ_b)` times a huge coefficient stays
finite). ABCMB relies on `lna_start` being chosen late enough (`R_tc` threshold)
that `1/(aH·tau_c)` is merely "large, implicit-solvable" rather than "infinite,"
and lets Kvaerno5 (an implicit, stiff solver) eat the residual stiffness. That is
why the solver is implicit and why the small-k band gets a *tighter* tol
(rtol 1e-5/atol 1e-10, model_specs.py:63,65): small-k modes start integrating
earlier relative to their dynamics and sit in the stiff regime longer.

**Consequence for fp32 (lever #1):** the `(θ_g − θ_b)` difference is a genuine
large-cancellation term multiplied by a coefficient that can reach 1e3–1e6 in
units of 1/Mpc near `lna_start`. In fp64 (eps ≈ 2.2e-16) a cancellation of two
O(1) numbers agreeing to 6 digits leaves ~1e-10 relative noise × 1e6 coefficient
= ~1e-4 absolute in `theta_prime` — acceptable. In fp32 (eps ≈ 1.2e-7) the same
cancellation leaves ~1e-1 relative × 1e6 = catastrophic. **This is not a "fp64 by
tradition" situation; it is a structural cancellation that fp32 cannot absorb in
the bare term ABCMB integrates.** CLASS being fp64 is a *weaker* precedent than it
looks — CLASS never integrates this term explicitly (it has TCA); ABCMB does, so
ABCMB is *more* fp32-exposed than CLASS, not less. See lever #1 for the only
fp32 variant that could survive.

---

## 1. Gate headroom: what is spent vs available (quantified)

Current gate (`bench/accuracy_cubic.log`, lensing=True, ELLMAX=2500, the production
config):

| channel | max-rel vs CLASS | gate | margin to gate |
|---------|-----------------:|-----:|---------------:|
| TT | 0.1967% | 1% | **5.1×** |
| EE | 0.2309% | 1% | **4.3×** |
| Pk | 0.1845% | 1% | **5.4×** |

**How much of that 0.2% is already spent by the existing rtol=1e-4 diffrax noise?**
The CHANGELOG (keystone entry) and `bench/smoke_batched_pipeline.log` state the
chunked-vs-single-call Cl/Pk agreement is ~2.6e-5 relative, and TT/EE/Pk "were
already at the ~3e-4 diffrax-noise floor." So:

- **The ~0.2% gate error is dominated by ABCMB-vs-CLASS *physics/discretization*
  differences (k-grid, l-spline, recomb model, LoS quadrature), NOT by the
  rtol=1e-4 ODE noise.** The ODE noise contributes only ~3e-4 *relative to itself*,
  i.e. ~0.03% — about 1/7 of the 0.2% budget.
- **Therefore loosening the ODE tolerance does NOT eat the full 0.8% margin
  linearly.** It eats the *ODE-noise* sub-budget. Going rtol 1e-4 → 3e-4 roughly
  triples the ODE noise floor (≈0.03% → ≈0.1%), which adds in quadrature with the
  ~0.2% physics floor to give ≈0.22% — still 4.5× under the gate. Going 1e-4 →
  1e-3 (10×) would push ODE noise to ~0.3%, total ≈0.36% — still 2.8× under.

**This is the key quantitative correction to a naive "huge headroom" read:** the
headroom is large (5×) but the ODE tolerance is a *sub-dominant* contributor, so
loosening it buys real step-count reduction with comfortable gate margin, but it
is NOT going to consume the whole 5× — the physics floor caps how far it can
matter. Conversely, that same physics floor means there is MORE tolerance headroom
than the ODE-noise number alone suggests, because the gate is measured end-to-end
and the ODE noise adds in quadrature against a larger fixed term.

---

## 2. The levers, with hidden floors

### Lever P1 — Loosen the STIFF-band tolerance (rtol_large_k_PE / atol_large_k_PE)

**Mechanism.** All `K_CHUNK×B` lanes in `_evolve_chunk` share ONE lockstep PID
controller (perturbations.py:142-148; confirmed in notes_solve §A). The chunk's
wall-clock = the step count of its *tightest-tolerance, stiffest* lane. The stiff
band (k > k_split_PE=0.01) is the bulk of the k-axis (N_k≈492, only a handful of
modes are below 0.01) and carries the max=1579-step lanes (notes_solve §0). Its
tol is rtol_large_k_PE=1e-4, atol_large_k_PE=1e-6 (model_specs.py:64,66), plumbed
per-k as a *value* via `jnp.where(k > k_split_PE, ...)` (perturbations.py:420-430)
— so loosening it is a one-line spec change, no code restructure.

PID step size scales roughly as `(tol)^(1/(order+1))`. Kvaerno5 is order 5, so
step size ∝ `tol^(1/6)`. A 3× tol loosening → step size ×3^(1/6) ≈ ×1.20 → step
count ×0.83. A 10× loosening → ×10^(1/6) ≈ ×1.47 → step count ×0.68. rtol and atol
both gate, so loosen **both** in the stiff band (the perturbation amplitudes range
over many orders so atol matters for the small components — the high-ℓ multipoles
`F[L]` are tiny, atol-limited).

**Expected solve speedup.** Step count ×0.83 (3×) to ×0.68 (10×) on the stiff band,
which sets the lockstep pace → **modes stage 1.15–1.45×**. Note: this is the
*purest* lever for the lockstep-paced solve because it directly lowers the pace-
setting lane's step count, unlike chunking which only regroups lanes.

**Accuracy risk + exact gate test.** MEDIUM. Per §1, rtol 1e-4→3e-4 lands ≈0.22%
(quadrature), 1e-4→1e-3 lands ≈0.36% — both safely under 1%, but must be *measured*
because the quadrature model is an estimate. **Test: bracket sweep** — set
`rtol_large_k_PE ∈ {1e-4, 3e-4, 1e-3}` and `atol_large_k_PE ∈ {1e-6, 3e-6, 1e-5}`
(9 cells, but really a diagonal: (1e-4,1e-6) baseline, (3e-4,3e-6), (1e-3,1e-5)),
run `pytests/accuracy_test.py` (the 1%-vs-CLASS gate) at each, record max-rel
TT/EE/Pk. The test prints all three (accuracy_test.py:122,126,133). This is THE
cheapest gate-relevant measurement in the whole brief: one spec dict change, one
pytest run (~160 s each per accuracy_cubic.log), no code edit. Also time the modes
stage (bench/profile_stages.py perturb fence) at each tol to confirm the step-count
→ wall-clock translation.

**Effort.** TRIVIAL (edit model_specs.py:64,66 defaults, or pass via specs).
**Prob success.** HIGH (≈0.85) for 3× loosening passing the gate; MEDIUM (≈0.5)
for 10×.

**Hidden floor.** (a) The lockstep pace is set by the *stiffest* lane, and step
size scales only as `tol^(1/6)` — a 10× tol change is only a 1.47× step reduction,
so the lever saturates fast. (b) If the small-k band (rtol 1e-5) ever shares a
chunk with the large-k band, the small-k tol would re-set the pace — but the k-axis
is monotonic (model_specs.py:120-175) and only the *first* chunk straddles
k_split_PE=0.01, so this is one chunk out of 5. (c) **The real ceiling: rtol=1e-4
is already close to where the *physics* floor (0.2%) dominates;** pushing past
~3e-4 buys diminishing solve time for rising gate risk. Sweet spot is almost
certainly (3e-4, 3e-6): ~1.2× modes for ~+0.02% gate cost.

---

### Lever P2 — Kvaerno3 instead of Kvaerno5 on the stiff band

**Mechanism.** `solver = diffrax.Kvaerno5()` is set unconditionally
(perturbations.py:418). Kvaerno5 is a 5th-order ESDIRK; at rtol=1e-4 the solver
order is *higher than the tolerance needs*. Order-p adaptive solvers pick step size
∝ `tol^(1/(p+1))`; lower order = smaller steps but **far cheaper per step** (Kvaerno3
has fewer implicit stages → fewer Ny×Ny Newton/LU solves per step, and the LU is the
FP-heavy part). At a loose rtol the per-step saving usually beats the step-count
penalty. Kvaerno3 is already imported in spectrum.py:6 and used by HyRex
(notes_solve §E references Kvaerno3 in helium), so it is a known-good solver in this
codebase.

This needs a Python-level (static) branch outside the vmap — `evolution_one_k` would
pick the solver from a static flag, OR (cleaner) the stiffness-homogeneous chunking
(the brief's lever A1) gives per-band `_evolve_chunk` variants for free, and each band
picks its solver. For a quick spike, just swap line 418 globally and run the gate.

**Expected solve speedup.** 1.1–1.4× on the stiff band (order-3 vs order-5 tradeoff
at rtol 1e-4 is empirically favorable for moderately stiff problems; ESTIMATE). Could
be neutral or negative if the hierarchy is stiffer than Kvaerno3 handles well at
1e-4 — must measure step counts.

**Accuracy risk + test.** MEDIUM. Order-3 at the same rtol should hold the gate (the
tolerance, not the order, sets the error target), but the *constant* in the error
estimate differs. **Test:** swap perturbations.py:418 to `Kvaerno3()`, run
`pytests/accuracy_test.py` AND time the perturb fence at B=16. If gate passes and
step count drops, keep it (possibly band-selective).

**Effort.** TRIVIAL for global swap; LOW for band-selective. **Prob success.** MEDIUM
(≈0.5) — genuinely uncertain whether order-3 wins at this stiffness/tol; cheap to find
out. **Hidden floor.** If Kvaerno3's step count rises more than its per-step cost
falls (possible for the stiffest high-k lanes), it's a wash or loss. Kvaerno3 may also
need *more* steps right through recombination where the source is sharp. **Pairs
naturally with P1** (loosen tol → lower order becomes more favorable).

---

### Lever P3 — Mixed precision: fp32 perturbation state `y`, fp64 everything else

**Mechanism.** Everything is fp64 (`jax_enable_x64` at perturbations.py:13,
main.py:27, background.py, spectrum.py:18). A100 fp64 ≈ ½ fp32 throughput; the
Kvaerno5 implicit Jacobian factorization/solve (O(Ny²)=46²≈2100 FLOP/lane/stage) is
the FP-heavy inner kernel. diffrax respects the dtype of `y0`, so casting `y_ini`
to fp32 (perturbations.py:414) while keeping `lna`/`k`/background in fp64 would run
the state, the RHS arithmetic, and the Jacobian in fp32.

**Expected solve speedup.** 1.4–1.8× on the solve *if it survives* (the LU is fp-bound
and fp64→fp32 ≈ 2× there; offset by mixed-precision cast overhead and possibly more
steps from a noisier error estimate).

**Accuracy risk + test.** **HIGH — this is the riskiest lever and §0 explains why it
is structurally, not just numerically, exposed.** The bare tight-coupling term
`R/tau_c/aH*(theta_g-theta)` (species.py:1221) is a large-cancellation × large-
coefficient term that ABCMB integrates *explicitly* (no TCA). fp32 eps ≈ 1.2e-7;
the `(θ_g − θ_b)` difference can lose 6+ digits in tight coupling, and the
coefficient `R/(aH·tau_c)` reaches ~1e3–1e6 1/Mpc near `lna_start`. fp32 cannot
hold this. The metric sums `4πGa²ρδ` (perturbations.py:370,519) sum 5 species'
densities spanning many orders (photon vs CDM) — also cancellation-prone in fp32.
And the Cl chain *squares* the transfer (spectrum.py:825,827) and Pk squares delta_m
(spectrum.py:302), doubling relative error.

**The only fp32 variant worth a spike:** keep `lna_start` conservative (the existing
`R_tc=0.0015` threshold already delays the start past the worst stiffness), cast ONLY
`y0` to fp32, and **mask the gate test to the stiff band's downstream Cl** — if even
the conservative-start fp32 fails, the lever is dead. **Test:** `y0 = y_ini.astype(
jnp.float32)` at perturbations.py:414 (and verify `get_derivatives` arithmetic
follows the y dtype — the `4πG` constants are fp64, so the products promote to fp64
unless explicitly downcast; you may need `a = jnp.exp(lna).astype(f32)` etc. to
actually get fp32 FLOPs, otherwise XLA keeps it fp64 and you measure NOTHING). Run
`pytests/accuracy_test.py`, report max-rel TT/EE/Pk.

**Critical footguns at the boundaries:**
- `_to_float` (main.py:287-292, 347-351, 401-406) casts int/bool params to
  **float64** for the checkpointed_while_loop custom_vjp. If you go fp32 you must
  decide: params stay fp64 (then RHS promotes to fp64 — no fp32 win) or params go
  fp32 (then re-verify the custom_vjp integer-leaf workaround still holds, and the
  HyRex `array_with_padding` int leaves survive the GPU/CPU transfer). **Mixed dtype
  across the main.py:296-305 device boundary is a silent-promotion footgun.**
- background quantities (`aH`, `tau_c`, `expmkappa`) are fp64; multiplying fp32 `y`
  by fp64 `aH` promotes to fp64. To actually run fp32 FLOPs you must downcast inside
  `get_derivatives` — which is exactly where the cancellation lives. **You cannot
  get the fp32 *speed* without exposing the fp32 *cancellation*.** This is the
  hidden floor that likely kills the lever.

**Effort.** MEDIUM (the cast is one line; making it *actually* fp32 and not silently-
promoted-back-to-fp64 is fiddly; the boundary verification is the real work).
**Prob success.** LOW-MEDIUM (≈0.25) that it passes the gate; HIGH that it's worth ONE
measured spike because the payoff is large and the answer is currently unknown.
**Hidden floor:** silent fp64 re-promotion (you measure no speedup and conclude
wrongly), AND the structural cancellation in the no-TCA term (you measure a gate
failure that no amount of tuning fixes without adding a TCA phase — a much larger
refactor).

---

### Lever P4 — tf32 / bf16 matmul precision

**Mechanism.** `jax.default_matmul_precision('tensorfloat32')` or per-op precision
hints make XLA use TF32 (19-bit mantissa, ~8× A100 throughput) for matmuls.

**Assessment: NO win here, and I'm confident.** The solve has **no large matmuls**.
The Kvaerno5 Jacobian is Ny×Ny = 46×46 per lane — a *tiny* LU, batched over
K_CHUNK×B lanes. That is a batched-small-LU / batched-small-solve workload, which is
**latency- and memory-bound, not matmul-throughput-bound**; TF32 tensor cores do
nothing for a 46×46 LU (no GEMM call large enough to dispatch to tensor cores).
`get_derivatives` is elementwise + small `jnp.concatenate` (perturbations.py:333-380)
— no matmul. The spectrum DOES have larger reductions (CubicSpline interp, the LoS
scan, Wigner sums in lensed_Cls) but spec_Cl is already 0.01 s/param post-keystone
(CHANGELOG) — irrelevant. **Verdict: skip.** The only place tf32 could matter is if a
future change introduced a dense Jacobian assembled as a GEMM; it isn't there now.

**Effort/risk/floor:** N/A — the floor is "the solve has no qualifying matmul," so the
expected speedup is ~1.0×. Do not spend GPU-hours.

---

### Lever P5 — saveat density (500 → ~300, non-uniform dense at recomb)

**Mechanism.** `lna = jnp.linspace(lts, 0., 500)` with `SaveAt(ts=lna)`,
`dense=False` (perturbations.py:100 single, 179-181 batched). This is NOT an
oversampled intermediate — it IS the line-of-sight time-integration grid: the
spectrum scans directly on `lna_axis = PT.lna[:-1]` (spectrum.py:676, scan at
:817 over those 499 points). So trimming Nlna cuts BOTH (a) the saved-ys tensor
memory + `make_output_table` bandwidth (perturbations.py:500-523, O(Nlna×Nk×B)),
AND (b) the LoS scan length.

**Is this a *precision* lever?** Partly — `saveat` does NOT change the *adaptive
integration* (the PID controller steps at its own pace regardless of save points;
SaveAt only interpolates the dense path onto the save grid). So fewer save points
does **not** speed the ODE *solve* itself — it speeds the *output table build* and
the *LoS scan*. Post-keystone the LoS scan is folded into the 0.01 s/param spec
stage, so the saveat win is now almost entirely **memory/OOM relief → enables higher
B**, which is the real throughput lever (sharding payoff grows with B per the brief).

**Critical precision subtlety (the floor):** the perturbation trajectories are smooth
in lna *except* across recombination, where the visibility function `g = -dκ/dτ ·
expmkappa` is a sharp spike near `lna_decoupling`. The LoS source terms
(spectrum.py:717-732) are all weighted by `g` or `expmkappa`. A *uniform* 500→300 cut
under-resolves the visibility spike and will move TT/EE (the SW + Doppler terms peak
there). A **non-uniform grid dense at recomb** (e.g. 120 pts lts→rec-1, 200 pts
rec±1, 120 pts rec+1→0) holds accuracy at ~440→trim 300. This requires changing the
linspace at perturbations.py:179-181 to a `concat_sorted` of three bands keyed on
`BG.lna_rec` (already available — used in spectrum.py:419 lensing).

**Expected effect.** ~1.0× on the *solve* (doesn't touch it), ~1.15× on
make_output_table + ~40% memory relief on the modes tensor → the OOM ceiling moves
from B≈64 to B≈90-100 on one A100, and quarters again under 4-GPU sharding. The
throughput value is the higher-B headroom, not the per-param solve time.

**Accuracy risk + test.** MEDIUM (visibility-spike under-resolution). **Test:** change
the save grid to recomb-dense 300, regenerate snapshots (`python
pytests/fixtures/generate_snapshots.py` on GPU), then run BOTH
`pytests/test_snapshots.py` (parity — will legitimately change, so regen first) AND
`pytests/accuracy_test.py` (the 1% gate — this is the real check). Report max-rel
TT/EE/Pk; TT is the sensitive one (visibility-weighted SW).

**Effort.** LOW-MEDIUM (grid construction + snapshot regen). **Prob success.** MEDIUM
(≈0.55). **Hidden floor:** (a) it does not touch the actual solve floor, only memory/
table; (b) the LoS scan length reduction is now nearly free anyway post-keystone; (c)
the dominant value (higher B) only pays off *with sharding*, so it's a multiplier on
the multi-GPU lever, not standalone. Down-rank vs P1.

---

### Lever P6 — dt0 / PID coefficients

**Mechanism.** `dt0=1e-2` (perturbations.py:438), PID `pcoeff=0.25, icoeff=0.8,
dcoeff=0.` (model_specs.py:67-69). dt0 is the *initial* step; if it's far from the
controller's preferred first step, the first few steps are rejected/re-tried (wasted
RHS evals). PID coeffs tune how aggressively the step size adapts; the diffrax default
for a PID controller is (pcoeff=0.4, icoeff=0.3, dcoeff=0) — the current
(0.25, 0.8, 0.) is *more integral-heavy*, which can over/under-shoot and cause more
rejected steps depending on the problem.

**Expected speedup.** 1.0–1.1× (marginal). A better-tuned dt0 saves a handful of
startup steps; better PID coeffs reduce rejected-step fraction (PID typically rejects
10-30% of trial steps). Across all K_CHUNK×B lanes the rejected-step saving is real
but small.

**Accuracy risk + test.** LOW (these don't change the *target* tolerance, only the
path to it — the converged solution is the same to within rtol). **Test:** sweep
`dt0 ∈ {1e-3, 1e-2, 1e-1}` and try the diffrax-default PID (0.4, 0.3, 0.); run the
gate (should be unchanged to ~1e-5) and time the perturb fence. **Effort.** TRIVIAL.
**Prob success.** MEDIUM (≈0.5) for a measurable win; HIGH that it's safe. **Hidden
floor:** the current coeffs were likely already tuned (the 1.12 max/median over B in
flipped_summary.txt suggests the controller is well-behaved); marginal headroom.
Bundle into the P1 sweep — it's the same harness.

---

### Lever P7 — atol on the SMALL-band and the high-ℓ multipoles

**Mechanism.** atol_small_k_PE=1e-10 (model_specs.py:65) is very tight. The small-k
band (k<0.01) is few modes but tight-tol; if it ever shares the first chunk with
large-k modes it sets the lockstep pace for that chunk. Also, atol gates the *small*
state components — the high-ℓ photon/neutrino multipoles `F[L]`, `G[L]` are tiny and
atol-limited; 1e-10 may be tighter than the 1% gate needs.

**Expected/risk/test:** sub-lever of P1; fold the small-band atol into the same sweep
(`atol_small_k_PE ∈ {1e-10, 1e-9}`). LOW additional effort. **Hidden floor:** the
small-k band is so few modes that loosening it only helps the *one* straddling chunk;
small absolute win. Include for completeness, don't prioritize.

---

## 3. Ranked shortlist

Ranking metric: (solve-stage speedup × prob-success) ÷ (accuracy-risk × effort),
end-to-end-aware (the solve is now ~1.6–2.0 s/param, the single-GPU floor per the
brief; spectrum and setup are no longer the bottleneck post-keystone, so a solve win
DOES move the per-param number unlike in the old redteam analysis).

| Rank | Lever | Solve speedup | Risk | Effort | Prob | Score rationale |
|------|-------|--------------:|------|--------|-----:|-----------------|
| **1** | **P1 loosen stiff-band rtol/atol (3e-4/3e-6)** | 1.15–1.3× | MED | TRIVIAL | HIGH | Directly lowers the lockstep pace-setting lane's step count; one-line spec change; cheapest gate test; 4–5× headroom confirmed |
| **2** | **P2 Kvaerno3 on stiff band** | 1.1–1.4× | MED | TRIVIAL→LOW | MED | Order-5 is overkill at rtol 1e-4; cheaper per-step LU; trivial global swap to spike; pairs with P1 |
| **3** | **P6 dt0 + PID retune** | 1.0–1.1× | LOW | TRIVIAL | MED | Cuts rejected-step waste; same sweep harness as P1; safe but marginal |
| **4** | **P5 non-uniform saveat 500→300 dense-at-recomb** | ~1.0× solve, +40% mem | MED | LOW-MED | MED | Doesn't touch the solve; value is OOM relief → higher B → bigger sharding payoff. A multi-GPU multiplier, not a solve win |
| **5** | **P3 fp32 perturbation state** | 1.4–1.8× IF it survives | **HIGH** | MED | LOW | Biggest raw upside but structurally exposed (no-TCA cancellation §0) + silent-fp64-repromotion + boundary footguns. Worth ONE measured spike, not a bet |
| — | P4 tf32/bf16 | ~1.0× | — | — | — | **Dead.** No qualifying matmul in the solve (46×46 LU is latency-bound). Skip |
| — | P7 small-band atol | <1.05× | LOW | TRIVIAL | MED | Fold into P1 sweep; tiny absolute win |

### Top-5, one paragraph each

**1. Loosen the stiff-band tolerance (P1).** The lockstep PID controller paces every
`K_CHUNK×B` lane to the stiffest, tightest-tol lane (perturbations.py:142-148), and
the stiff band (k>0.01, the bulk of N_k) carries that pace at rtol_large_k_PE=1e-4 /
atol_large_k_PE=1e-6 (model_specs.py:64,66). Loosening to (3e-4, 3e-6) shrinks the
pace-setting step count by ~1/`3^(1/6)`≈0.83 → ~1.2× on the modes stage. The gate has
5× headroom (TT 0.197%, EE 0.231%, Pk 0.185% vs 1%) and — crucially — the ODE noise
is only ~1/7 of that 0.2% (the rest is physics floor), so a 3× tol loosening adds
only ~+0.02% in quadrature. One-line spec change, cheapest gate test in the brief.

**2. Kvaerno3 on the stiff band (P2).** Kvaerno5 (perturbations.py:418) is 5th-order;
at rtol=1e-4 that order is higher than the tolerance needs. Kvaerno3 takes smaller
steps but with far fewer implicit stages → fewer Ny×Ny Newton/LU solves per step
(the FP-heavy inner kernel). At loose tol the per-step saving usually beats the step-
count penalty → 1.1–1.4×. Trivial global swap to spike (Kvaerno3 already imported,
spectrum.py:6); ideally band-selective via the stiffness-homogeneous chunking. Pairs
multiplicatively with P1 (looser tol favors lower order). Genuinely uncertain whether
it wins at this stiffness — but the test is free.

**3. dt0 + PID retune (P6).** dt0=1e-2 (perturbations.py:438) and the integral-heavy
PID (0.25, 0.8, 0.) (model_specs.py:67-69) govern startup-step waste and rejected-
step fraction (PID rejects 10-30% of trials). The diffrax default (0.4, 0.3, 0.) and a
better dt0 could shave the rejected-step tax. Marginal (1.0-1.1×) but LOW-risk (the
converged solution is unchanged to ~rtol) and rides the same sweep harness as P1.

**4. Non-uniform saveat (P5).** `linspace(lts,0,500)` (perturbations.py:179-181) is
both the save grid and the LoS quadrature grid. It does NOT change the adaptive solve,
so it is not a solve-time lever; its value post-keystone is ~40% modes-tensor memory
relief → the single-A100 OOM ceiling moves from B≈64 to ~B≈90, multiplying the
sharding payoff (which grows with B). Must be recomb-dense (the visibility spike at
lna_rec drives TT/EE); a uniform cut fails the gate, a 3-band concat holds it.

**5. fp32 perturbation state (P3) — spike, don't bet.** Casting only `y0` to fp32
(perturbations.py:414) could give 1.4-1.8× on the FP-bound implicit solve. But §0:
ABCMB integrates the tight-coupling term `R/tau_c/aH·(θ_g−θ_b)` (species.py:1221)
**explicitly with no TCA**, a large-cancellation × large-coefficient (1e3-1e6)
construct that fp32 (eps 1e-7) structurally cannot hold — and you cannot get the
fp32 *speed* without downcasting inside `get_derivatives` where the cancellation
lives, plus the `_to_float`→fp64 cast (main.py:287-292) and the GPU/CPU boundary are
silent-promotion footguns. Worth exactly ONE measured spike because the answer is
unknown and the payoff is large, but the prior is failure.

---

## 4. The #1 bet and the cheapest de-risking measurement

**#1 BET: Lever P1 — loosen the stiff-band tolerance to (rtol_large_k_PE=3e-4,
atol_large_k_PE=3e-6), with P2 (Kvaerno3) and P6 (dt0/PID) folded into the same
sweep.** This is the highest (speedup × prob ÷ risk × effort): it directly attacks
the lockstep pace, it's a one-line spec change, the gate has a measured 5× headroom
of which the ODE tolerance only spends ~1/7, and the test is the cheapest in the
brief. P1+P2+P6 together plausibly reach ~1.3–1.5× on the modes stage (the single-GPU
floor), i.e. ~1.6 s/param → ~1.1-1.2 s/param, with the dominant lever (P1) being
zero-effort and the others rideable on the same harness.

**Cheapest GPU/gate measurement to de-risk it (do this first, ~3 srun runs):**

1. **Bracket the gate (≈160 s each).** In a single srun, run `pytests/accuracy_test.py`
   at three tolerance cells by editing only the specs passed to `Model(...)` (the test
   constructs the model at accuracy_test.py:55-66 — add `rtol_large_k_PE=`,
   `atol_large_k_PE=` kwargs there):
   - baseline (1e-4, 1e-6) — reproduce TT 0.197% / EE 0.231% / Pk 0.185%,
   - (3e-4, 3e-6),
   - (1e-3, 1e-5).
   Record max-rel TT/EE/Pk for each. This directly confirms the §1 quadrature model
   and finds the loosest tol that holds the 1% gate. **This single run answers
   whether P1 is safe — it is the highest-information, lowest-cost measurement.**

2. **Time the modes stage (one srun, B=16).** Run `bench/profile_stages.py` (it has a
   per-stage perturb fence) at baseline tol and at the loosest gate-passing tol from
   step 1. The ratio of the `perturb` fences is the realized solve speedup — confirms
   the step-count → wall-clock translation (step count ∝ tol^(1/6) is theory; measure
   it).

3. **(Optional, same srun as 2) Kvaerno3 spike.** Swap perturbations.py:418 to
   `Kvaerno3()`, rerun the gate + perturb fence. If gate passes and the fence drops,
   keep it. ~5 min of edit + run.

All three fit in one ~30-minute interactive GPU allocation. **Order: gate bracket
(step 1) before any timing — if the gate doesn't hold, the speedup is moot.** Export
`PYTHONPATH=$(pwd):$PYTHONPATH` inside srun so `import abcmb` resolves to THIS
checkout (per CLAUDE.md), and set `JAX_PLATFORM_NAME=gpu` for the snapshot backend if
P5 is later spiked.

### What I am confident is a trap

- **tf32/bf16 (P4): dead.** No matmul in the solve large enough for tensor cores;
  the 46×46 batched LU is latency-bound. Zero expected win — don't spend GPU-hours.
- **Global fp32 (not the targeted P3 variant): dead.** The no-TCA tight-coupling
  cancellation (§0) makes a blanket fp64→fp32 switch fail the gate with near-
  certainty. Only the *measured, targeted, conservative-start* spike is defensible,
  and even that is a low-prior probe, not a bet.
- **Lowering max_steps_PE (2048): not a perf knob.** Adaptive solves stop at
  convergence, not at the cap; observed max is 1579/1595 (notes_solve §0). Lowering
  the cap only risks silent truncation in a scan cosmology that needs more steps. The
  cap is a correctness guard. Leave it.

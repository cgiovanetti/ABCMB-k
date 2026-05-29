# round2_solver.md — Attacking the batched perturbation ODE solve

Perf round 2, branch `perk-perf`, post-keystone. The solve is now the single-GPU
floor (perturb 1.97 s/param of the 4.0 total at B=16; `CHANGELOG.txt:71`). This
note is a skeptical, code-grounded enumeration of ways to cut the per-solve cost.
**No code was run** (login-node python forbidden); everything is read from source
with `file:line` citations + the verified `bench/` artifacts.

The leaf being optimized is `evolution_one_k` (`perturbations.py:382-448`), an
adaptive **implicit Kvaerno5** diffrax solve with a PID controller, double-vmapped
`vmap(k_chunk) × vmap(B)` inside `_evolve_chunk` (`perturbations.py:142-148`),
Python-looped over contiguous 100-mode k-chunks in `_compute_modes_batched`
(`perturbations.py:186-191`).

---

## 0. The two structural facts that reframe everything

Reading the actual RHS, two things stand out that the stale memos did not weight
correctly. They change the ranking.

### Fact 1 — there is NO tight-coupling approximation (TCA). The solve is *all-stiff*.

`get_starting_time` (`perturbations.py:251-291`) despite its docstring about
"tight coupling" does **only one thing**: it computes `lna_start`, the time at
which integration *begins* (`perturbations.py:406-414`). The comment at
`perturbations.py:416` says "Settings for post-tight coupling" — i.e. the code
**starts the integration after the TCA window** and integrates the *full*
Einstein–Boltzmann hierarchy (including the stiff `±1/(aH·τ_c)` Thomson terms,
`species.py:1371-1385`, `species.py:1221`) the whole way to `lna=0`. There is no
reduced-order TCA system being integrated and no TCA→full switch event. So:

- The stiffness source is the **Thomson scattering rate `1/τ_c`** in the photon
  hierarchy (`species.py:1371,1372,1373,1377,1378,1382,1385`) and baryon θ
  (`species.py:1221`). At early times (large `1/τ_c`) the photon-baryon system is
  extremely stiff; that is *exactly why Kvaerno5 (implicit) is used* and why a
  naive explicit solver would die there.
- But the code already dodges the *worst* of it by starting late (`R_tc=0.0015`,
  `model_specs.py:56`; `lna_start = min(get_starting_time, -10.)`,
  `perturbations.py:411`). After the TCA window, `1/τ_c` is still large near
  recombination but decays; the stiffness is moderate-and-decaying, not extreme.

This is the load-bearing realization for **A2 (explicit solver)** and
**A4 (proper TCA)** below.

### Fact 2 — the RHS recomputes the background 5×/step, and each is a species-loop.

`BG.aH(lna,params)` is **not a table lookup**. It calls `H` → `rho_tot`
(`background.py:148,167,102-123`), a Python loop summing `rho` over all 5 species
analytically (`background.py:121-122`), then a sqrt. Within ONE `get_derivatives`
call (`perturbations.py:333-380`), `aH(lna,params)` is recomputed independently at:

1. `perturbations.py:356` (metric block)
2. `species.py:1357` Photon.y_prime
3. `species.py:1212` Baryon.y_prime
4. `species.py:1134` Baryon.cs2 (called from Baryon.y_prime line 1213)
5. `species.py:670` MasslessNeutrino.y_prime

That is **5 identical `rho_tot` species-loops + 5 sqrts per RHS evaluation**, all
at the same `lna`. Likewise `τ_c` (which calls `xe`→`fast_interp` + `nH`,
`background.py:688-691`) is recomputed at `species.py:1358` (Photon),
`species.py:1215` (Baryon), and inside `cs2` (`species.py:1134`) — 3×. And `τ`
(a `fast_interp`, `background.py:348`) at `species.py:1359,671` — 2×. `cs2`
(`species.py:1101-1134`) itself calls `Tm`, `TCMB`, `xe`, `mean_mass` (another
`xe`), `aH`, `τ_c`. This is **redundant work multiplied by `K_CHUNK×B×steps`** —
it lands inside the innermost vmapped loop body.

This is the load-bearing realization for **B1 (RHS CSE / precompute bg)**, which
the stale memos *entirely missed* (they treated the RHS as a fixed cost).

---

## State vector and step distribution (the cost model)

Ny = 46 for default ΛCDM (`notes_solve.md:73-81`, recomputed: metric 1 + CDM 1 +
Baryon 2 + Photon (13+11=24) + MasslessNu 18). Kvaerno5 cost per step is
**O(Ny²)** for the Jacobian + O(Ny³) for the LU of the implicit Newton system,
i.e. the per-step cost is dominated by a ~46×46 dense solve plus several RHS
evaluations for the Jacobian (Kvaerno5 has ~5 implicit stages → multiple RHS +
one LU per stage).

Per-k step counts (`N_k≈492`, single cosmology, `CHANGELOG.txt:116`):
**min=41, median=380, max=1579, max/median=4.16.** After the params-flip, worst-k
**max/median over B = 1.12** (`notes_solve.md:50`). So at *fixed k* over B,
lockstep is nearly gone; the residual lockstep tax is **intra-chunk** (the
k-spread inside one 100-mode contiguous chunk). Total cost ≈ Σ_chunks (chunk's
worst-k step count) × Ny²·B·(per-step constant).

---

# THE IDEAS

For each: mechanism, expected per-solve speedup RANGE, accuracy risk vs the
1%-vs-CLASS gate (`pytests/accuracy_test.py`; current TT 0.197% EE 0.231% Pk
0.185%), effort, P(success), and the **hidden floor**.

---

## B1 — RHS common-subexpression elimination (precompute background once/step) ★ TOP BET

**Mechanism.** Compute `aH`, `τ_c`, `τ`, `cs2`, `R` (and the `rho`/`P` per species
needed for the metric sum) **once** at the top of `get_derivatives`
(`perturbations.py:333`) and thread them into each species' `y_prime` instead of
letting each species recompute them. Concretely:

- `get_derivatives` already has `aH` (`perturbations.py:356`). Add `tau`, `tau_c`,
  and the per-species `rho`/`P` (already summed at `perturbations.py:363-368`).
- Change the species `y_prime` signature to accept a small `bg_locals` struct
  (aH, tau, tau_c, cs2, R, …) instead of `(BG, params, …)`, so the photon/baryon/
  neutrino RHS read precomputed scalars (`species.py:670-671,886-887,1212-1215,
  1357-1359`) rather than calling `BG.aH(...)`/`BG.tau_c(...)` again.
- `aH` collapses from 5 `rho_tot` species-loops → 1; `τ_c` from 3 → 1 (one `xe`
  fast_interp instead of three); `τ` from 2 → 1; `cs2` is computed once.

This is *pure CSE*: identical math, identical numerics (the values are bit-equal,
they are functions of `lna`/`params` only). XLA's CSE pass *may* already eliminate
some of it — but `rho_tot`'s Python species-loop and the repeated `jnp.where`
inside `xe`/`Tm`/`fast_interp` (`background.py:604-614,658-668`) are exactly the
kind of control-flow-laden code XLA's CSE often **fails** to fully dedupe across
method-call boundaries, especially through the `jnp.where`-gated `fast_interp` and
the `jnp.sqrt` in `H`. The redundancy is visible and large.

**Expected speedup: 1.15–1.5× on the solve.** The RHS is evaluated several times
per step (Kvaerno5 stages + Jacobian). If the background recompute is, say,
25–40% of RHS cost (5× `rho_tot`-loops + sqrts + 3× `xe` fast_interps + a `cs2`
that itself does 4 table lookups), cutting it to 1× shaves a real fraction of
*every* RHS call, hence every step, for *every* k and B. This compounds with the
Jacobian: a cheaper RHS also makes the finite-difference / autodiff Jacobian
assembly cheaper. Lower bound 1.15× if XLA already deduped most of it; upper bound
1.5× if it didn't.

**Accuracy risk: NONE** (CSE — bit-identical). Gate = `test_snapshots.py` should
pass at `rtol=1e-8` *unmodified*; if it doesn't, the refactor changed math and is
buggy. This is the cleanest possible gate.

**Effort: MEDIUM** (touches the `y_prime` signature across all species in
`species.py` + the call site `perturbations.py:376-378` + `make_output_table`'s
mirror at `perturbations.py:497`). ~80–150 lines, mechanical.

**P(success): HIGH (0.8).**

**Hidden floor.** (1) XLA's CSE may *already* be catching most of it, in which
case the win is only ~1.1×. This is the single biggest uncertainty and is **cheap
to settle** — see the measurement at the end. (2) The Jacobian is taken through
the RHS by diffrax autodiff; restructuring the RHS to pass precomputed scalars
risks accidentally making one of those scalars a *constant* w.r.t. `y` when it
should not be — but `aH`,`τ`,`τ_c`,`cs2` are all functions of `lna` and params
*only*, never of `y`, so they are genuinely constant w.r.t. the Newton solve at a
fixed step. Passing them as precomputed is therefore not just safe, it is *more
correct* for the Jacobian (the implicit solver should not be differentiating
through `xe` interpolation w.r.t. `y` anyway — it doesn't, since they don't depend
on `y`). No floor there. (3) If the autodiff Jacobian was the dominant cost (LU,
not RHS), then RHS CSE helps less than hoped — but Jacobian *assembly* still calls
the RHS, so it still helps.

---

## A1 — Stiffness-homogeneous, non-uniform k-chunking ★ #2

**Mechanism.** Replace the uniform 100-mode contiguous chunking
(`perturbations.py:186-189`) with **non-uniform chunks sized by stiffness band**.
Because `get_k_axis_perturbations` is monotonic (`model_specs.py:120-175`), a
contiguous chunk = a k-band = a stiffness band. The lockstep controller paces each
chunk to its **worst (largest-k) lane**, so a wide chunk drags its cheap low-k
lanes. Make stiff (high-k) chunks *small* and smooth (low-k) chunks *large*.

**Concrete chunk plan from the k-axis structure.** The k-axis spans
`k_min ≈ k_min_tau0/tau0_fid = 0.1/14186.68 ≈ 7.05e-6` to
`k_max = k_max_tau0_over_l_max/tau0_fid·l_max` (`model_specs.py:128-129`), with the
step law transitioning around `k_rec_fid = 2π/rs_rec_fid ≈ 0.0434`
(`model_specs.py:126`) via the `tanh((k-k_rec)/k_rec/k_step_transition)` term
(`model_specs.py:136-137`). The rtol/atol split is at `k_split_PE = 0.01`
(`perturbations.py:420-430`). Step count correlates monotonically with k (min=41
at small k → max=1579 at large k). A 3-band plan:

| band | k-range | est. step regime | chunk size | distinct shapes |
|------|---------|------------------|-----------:|----------------:|
| smooth super-horizon | `k < k_split_PE = 0.01` | ~41–~150 | 200 | 1 |
| transition | `0.01 ≤ k < ~0.1` | ~150–~600 | 100 | 1 |
| stiff sub-horizon | `k ≥ ~0.1` | ~600–1579 | 40 | 1 |

Implement as the `_chunk_plan` sketch in `notes_solve.md:216-221`, looping
`_evolve_chunk` over each (band-slice, size). **Distinct chunk shapes = number of
distinct sizes used = 3** (plus 1 ragged tail each if not padded → pad to kill
those). So **3 extra compiles vs the current 2** — a one-time cold-start cost,
amortized over a scan.

**Quantifying the realistic ceiling (be skeptical).** The brief's claim of
1.3–1.7× is **optimistic and must be discounted**, for two reasons:

1. The flip already collapsed the *cross-chunk* tax. The remaining tax is the
   *intra-chunk* k-spread. For the **stiff band** (`k≥0.1`), the step count goes
   from ~600 to 1579 — a max/median *within that band* of maybe ~1.5–2.0. Shrinking
   the chunk from 100→40 narrows the lockstep group, recovering a fraction of that
   spread. But the *worst* 40-mode chunk (containing k_max) still runs at 1579
   steps regardless — you cannot beat the single stiffest mode's latency. The win
   is only on the *other* sub-chunks of the stiff band that no longer wait for
   k_max.
2. The smooth band (`k<0.01`) is *already cheap* (41–150 steps). Making its chunk
   bigger (200) saves kernel-launch overhead and lets those modes amortize, but
   their absolute cost is small, so the absolute saving is small.

**Honest expected speedup: 1.1–1.35× on the solve** (I trim the brief's upper
bound). The hard ceiling is set by: total cost ≈ Σ_chunk (worst-k-in-chunk steps).
With the current 5×100 layout the stiff chunk[4] (k~0.1→1.8) runs at ~1579;
re-banding to 40-wide stiff chunks means only the *last* 40-chunk runs at ~1579 and
the earlier stiff chunks run at their own (lower) worst-k. Summing the step-count
histogram (min=41/med=380/max=1579) under the two layouts is the right back-of-
envelope, and it lands around 1.2× — not 1.7×.

**Accuracy risk: NONE.** Identical math, only lane grouping changes — proven benign
in `chunking_debug_report.md` (chunk[0]+chunk[1] bit-matches the 0:200 slice;
per-mode drift is within-rtol noise, Cl/Pk agree at ~2.6e-5,
`chunking_debug_report.md:34,75,146`). Gate: `test_snapshots.py` (downstream Cl/Pk
contract), not per-mode bit-parity.

**Effort: LOW** (~20–30 lines in `_compute_modes_batched`, plus a `_chunk_plan`
helper). **P(success): HIGH (0.85)** for *some* win; the *magnitude* is the
uncertain part.

**Hidden floor.** (1) The single stiffest mode (k_max, 1579 steps) is an
**irreducible latency floor** for whichever chunk contains it — no chunking helps
that one chunk. Since the solve is `Σ_chunk worst-k-steps`, and the k_max chunk is
already the dominant term, A1 only trims the *non-dominant* chunks. The ceiling is
therefore *less* than `max/median = 4.16` would suggest. (2) Smaller chunks =
more kernel launches = more dispatch barriers; at 40-wide there are ~12 launches
vs 5 today, partially offsetting the lockstep win at small B where the device is
underutilized per launch. (3) More distinct shapes = more compiles (mitigated by
padding to uniform shapes within each size class).

---

## B2 — Reduce N_k for the perturbation grid (skeptical: probably NO)

**Mechanism.** N_k≈492 perturbation modes are computed, then **cubic-splined** onto
the denser `k_axis_transfer` for the LoS integral (`spectrum.py:700-711`,
`CubicSpline(log10(PT.k), col)(log10(k_axis))`) and `jnp.interp`'d for Pk
(`spectrum.py:300,338`). So the perturbation k-axis is the *spline source grid*.
Could it be coarsened for a frequentist scan?

**Skeptical verdict: largely NO, with a narrow exception.** The k-axis is
CLASS-style adaptive precisely so the cubic spline of the transfer function is
accurate. The transfer function oscillates (acoustic peaks); under-sampling it
aliases the peaks and shifts Cl. The 1%-vs-CLASS gate currently sits at ~0.2% —
there is **not** 5× headroom to coarsen. CLASS's `k_step_sub=0.05`
(`model_specs.py:37`) is already tuned near the accuracy edge. The narrow
exception: **the lensing/Pk-only tail** (`k > k_max_cmb`, added at
`model_specs.py:151-170` with fixed `step=0.005`) is used only for Pk and high-l
lensing; for a `lensing=False` Cl-focused frequentist scan those extra modes are
**not needed** and are already excluded (N_k 492 vs 571). So this is already
handled for the Cl-only case.

**Expected speedup: ~1.0–1.1×** (only if the scan is Pk-insensitive and you trim
the `k_step_sub` slightly — high risk for low reward). **Accuracy risk: HIGH**
(directly moves Cl peaks). **Effort: LOW** (change specs). **P(success): LOW
(0.25)** of passing the gate with a meaningful N_k cut. **Hidden floor:** the
spline accuracy *is* the gate; any N_k cut that helps perf hurts accuracy roughly
1:1. **Do not pursue** beyond confirming the lensing tail is off for Cl-only scans.

---

## A4 — Implement a real TCA (reduced-order stiff-window system) (high ceiling, high effort)

**Mechanism.** Today the code *avoids* the stiff pre-TCA-end window by starting
late (`lna_start = min(get_starting_time, -10.)`, `perturbations.py:411`) — but
between `lna_start` and recombination the photon-baryon system is still stiff
(`1/τ_c` large), which is why Kvaerno5 + the high step counts. A genuine TCA
integrates a **reduced system** (photon θ and σ slaved to baryon θ via the
leading-order tight-coupling expansion, dropping the stiff `±θ_b/τ_c` terms) up to
a switch time, then hands off to the full hierarchy. CLASS/CAMB do exactly this;
it is the standard cure for this stiffness. It would let the early portion use a
**non-stiff explicit** solver and far fewer steps.

**Expected speedup: 1.3–2.0× on the solve** IF the stiff early steps dominate the
1579-step worst case (plausible: the early `1/τ_c`-driven steps are where adaptive
controllers cluster). **Accuracy risk: MEDIUM** (TCA is a controlled expansion;
CLASS validates it, but the switch time and expansion order must be right; ABCMB's
init conditions at `perturbations.py:414` already assume adiabatic/TCA-style ICs,
so the machinery is half-there). **Effort: HIGH** (a new reduced-order RHS + a
switch; this is a real solver feature, ~200+ lines, and the switch under vmap must
be a `lax.cond`/event, not a Python branch). **P(success): MEDIUM (0.4)** —
the physics is standard but the JAX/diffrax-under-vmap implementation is delicate.

**Hidden floor.** (1) Under vmap the TCA→full switch must happen at a *per-lane*
time → either an event (diffrax supports `Event`, but composing it with the
lockstep controller and `SaveAt(ts=...)` is fiddly) or a fixed conservative switch
`lna` shared across lanes (simpler, slightly suboptimal). (2) The current ICs are
set at `lna_start` *after* the TCA window — re-introducing a TCA phase means moving
`lna_start` earlier (into the stiff region) and integrating the reduced system
there, which is where the savings come from but also where the expansion must be
accurate. (3) If the 1579 worst-case steps are actually dominated by *recombination
crossing* (visibility-peak region) rather than the pre-recomb stiff window, TCA
helps less. **Worth a spike only after B1/A1, and only if a step-location profile
shows the steps cluster pre-recombination.**

---

## A2 — Explicit/IMEX solver on the genuinely non-stiff band (modest, gated)

**Mechanism.** The smooth super-horizon band (`k < k_split_PE = 0.01`, min=41
steps) modes are non-stiff *after* the late start. For those, Kvaerno5's implicit
Newton/LU is overkill; an explicit `Tsit5`/`Dopri5` (no Jacobian, no LU) has
~5–10× cheaper per-step cost. Requires a **Python-level solver branch per band**
(the band-chunking of A1 gives this for free): `solver = Tsit5() if smooth_band
else Kvaerno5()` (`perturbations.py:418`).

**The skeptical question — is the small-k band genuinely non-stiff?** Stiffness
here = the `1/(aH·τ_c)` Thomson terms (`species.py:1371-1385,1221`), which are
**independent of k** (τ_c is a background quantity). So the small-k modes are
*just as stiff in the photon-baryon sector* as large-k modes — they simply need
**fewer steps** because the *oscillation frequency* `k·τ` is low (the solution is
smooth in time), not because the system is less stiff. **This is the hidden floor
that kills A2 as stated:** an explicit solver's stability is limited by the stiff
eigenvalue (`~1/τ_c`), not by k. Tsit5 on the small-k band near recombination
would hit the same `1/τ_c` stiffness and either reject steps catastrophically or
need a tiny `dt` — *more* steps, not fewer. The 41-step count for small-k is
achieved *because Kvaerno5 is implicit and steps over the stiff `1/τ_c`*. So
**A2 only works in combination with A4** (a real TCA that removes the `1/τ_c`
stiffness from the integrated system); on the current all-stiff RHS, an explicit
solver is unsafe everywhere recombination matters.

**IMEX (KenCarp3/4)** is the more defensible variant: treat the stiff `1/τ_c`
photon-baryon coupling implicitly and the rest explicitly. But diffrax's IMEX
support is limited and would require splitting the RHS into stiff/non-stiff terms
(a real refactor), and the implicit part still needs the LU on the coupled
photon-baryon block. Net: complex, uncertain.

**Expected speedup: 1.0–1.3× on the smooth band only (small absolute)** — and only
*after* A4. **Accuracy risk: MEDIUM-HIGH** (explicit on stiff = silent blow-up or
step explosion). **Effort: MEDIUM** (alone) / HIGH (with the RHS split for IMEX).
**P(success): LOW (0.25)** as a standalone; it is really a *rider on A4*.
**Hidden floor: stiffness is k-independent — the small-k cheapness comes from
Kvaerno5's implicitness, not from non-stiffness, so removing implicitness there
backfires.** Demote this; the memos over-rated it.

---

## A3 — Fixed-step / StepTo schedule (kills lockstep, high gate risk)

**Mechanism.** Replace `PIDController` (`perturbations.py:432`) with
`ConstantStepSize` or a precomputed `StepTo` schedule per band, sized so the
stiffest mode in the band is resolved. Then every lane does *exactly* the same
number of steps — no lockstep waste (there is no adaptation), and the step count is
static → XLA fully unrolls/pipelines, and you save the PID's rejected-step
overhead (~10–30% of trial steps).

**Expected speedup: 1.2–1.4× stacked on A1** (the saved rejected steps + static
shapes). **Accuracy risk: HIGH** — a fixed schedule that is fine for the fiducial
cosmology can be too coarse for a corner of a 20–40-param scan (the stiffest mode's
location *shifts* with `omega_b`, `H0`, `YHe` → τ_c shifts → required `dt` shifts),
and a too-coarse step **fails the gate silently** (no error, just wrong Cl). For a
*frequentist scan over a wide param box* this is the worst possible failure mode.
**Effort: MEDIUM.** **P(success): LOW (0.2)** for production robustness.

**Hidden floor.** The whole point of adaptivity is robustness across the param box;
a frequentist scan *explores the box edges* where a fixed schedule is least
trustworthy. You would need a schedule sized for the worst corner → back to paying
worst-case everywhere, eroding the win. **Only viable if a per-cosmology cheap
pre-pass picks the schedule (adaptive-but-static), which is its own complexity.**
Spike only after A1+B1, and gate hard across the param box, not just fiducial.

---

## J1 — Exploit Jacobian sparsity / structured Newton (high ceiling, hard in diffrax)

**Mechanism.** Kvaerno5 does an implicit Newton with a dense LU on the Ny×Ny=46×46
system per stage per step (`perturbations.py:418`). But the hierarchy Jacobian is
**near-banded**: each multipole `F[l]` couples only to `F[l±1]` (tridiagonal in l;
`species.py:1377` `L·F[L-1]-(L+1)·F[L+1]`, same for ν `species.py:688` and
polarization `species.py:1382`), plus the dense `1/τ_c` coupling rows
(photon θ↔baryon θ, σ↔G) and the metric row coupling to all `delta`/`theta`
(`perturbations.py:370-371`). So the Jacobian is **block-banded with a few dense
rows/columns** (metric + the Thomson-coupled photon/baryon block), not fully dense.
A banded or block-structured linear solve would be O(Ny·bandwidth) instead of
O(Ny³).

**Skeptical verdict: real structure, but hard to exploit in diffrax.** diffrax's
Kvaerno5 uses a `lineax` linear solver for the Newton step; by default a dense LU.
`lineax` *does* support structured operators, but plumbing a custom
banded/block-sparse operator through `diffrax`'s implicit solver is **not a
supported knob** — it would require subclassing the solver or the root-finder
(`diffrax` exposes `root_finder` on `Kvaerno5`, and `optimistix`/`lineax` allow a
custom linear solver). Effort is HIGH and the API surface is fragile across the
pinned `diffrax`/`optimistix==0.0.11`/`equinox` versions (`setup.cfg`).

**Expected speedup: 1.3–2.5× on the LU portion** if Ny were large — but **here is
the hidden floor: Ny=46 is small.** A 46×46 dense LU is ~33k FLOPs; the RHS
evaluations for the Jacobian (Kvaerno5 assembles the Jacobian by autodiff through
the RHS, which is the *5×-redundant-background* RHS of Fact 2) are likely
**comparable to or larger than** the LU itself. So the LU is probably *not* the
dominant per-step cost at Ny=46 — the **Jacobian assembly (RHS autodiff) is**,
which means **B1 (cheaper RHS) attacks the same cost more cheaply.** Banded LU only
pays off at much larger Ny (e.g. massive-ν runs with `3×18=54` extra equations →
Ny~100). **For default ΛCDM, J1 is dominated by B1.**

**Effort: HIGH. P(success): LOW-MED (0.3).** **Demote for ΛCDM; reconsider only for
large-Ny extended models.** A cheaper partial win: tell Kvaerno5 to **reuse the
Jacobian across steps** (quasi-Newton / `diffrax` doesn't recompute the Jacobian
every step if the root finder converges; check whether `optimistix`'s
`Newton`/`Chord` is configured — a `Chord` (frozen-Jacobian) root finder reuses the
factorization across iterations, cheaper than full Newton). Worth checking the
default.

---

## C — Trim/redistribute `saveat` (500→~300, recomb-dense) — OOM relief, not solve speed

**Mechanism.** `evolution_one_k` saves at `lna=linspace(lts,0,500)`
(`perturbations.py:179-181,433`), and the spectrum integrates the LoS on *that same
grid* (`spectrum.py:676`). Redistribute to ~300 points concentrated at
recombination (`notes_solve.md:388-401`). **Important nuance for THIS round:**
`SaveAt(ts=...)` with adaptive stepping does **not** reduce the number of
*integrator steps* — the solver still steps adaptively and just interpolates onto
the save grid. So C does **not speed the solve's step count**; it reduces (a) the
saved-ys memory/bandwidth and (b) the LoS scan length (`spectrum.py:805` runs over
`Nlna-1`). The solve-speed benefit is **indirect**: less memory → fits larger B →
better amortization (the real throughput lever per the brief's framing (a)).

**Expected effect: 1.0× on solve step count; 1.1–1.2× on PT/LoS bandwidth; ~1.4×
OOM headroom** (500→~300 cuts the (B,Ny,500,Nk) tensor 40%, easing the 28–31 GB/40
GB pressure at B=64, `battle_royale_brief.md:54`). That headroom **lets B grow**,
which is where throughput actually improves. **Accuracy risk: MEDIUM** (must keep
recomb resolution; the visibility peak is sharp). Gate: regen snapshots + accuracy
test at the new grid. **Effort: LOW. P(success): MED-HIGH (0.7).** **Hidden floor:**
it doesn't touch the per-solve step count at all, so if you are single-GPU
*compute*-bound (not memory-bound), C does nothing for you; its value is purely as
an **OOM-relief enabler for larger B**, which only helps if you are memory-bound
before you are compute-bound.

---

## A1.2 — Loosen `atol_large_k_PE` (1e-6 → 3e-6) within the stiff band

**Mechanism.** Lockstep paces to the tightest tolerance lane. The stiff band uses
`atol_large_k_PE=1e-6` (`model_specs.py:66`); loosening it lets stiff lanes accept
bigger steps → fewer steps. `chunking_debug_report.md:60-65` notes downstream Cl/Pk
already converge at `rtol_large=1e-4`; there is plausibly 2–3× atol headroom.

**Expected speedup: 1.05–1.2× on the stiff band.** **Accuracy risk: MEDIUM**
(directly the gate). **Effort: TRIVIAL** (one spec). **P(success): MED (0.5).**
**Hidden floor:** atol on a state vector spanning many magnitudes (delta~1e4 vs
high-l multipoles~1e-13) is a blunt instrument — loosening it preferentially
corrupts the *small* components (high-l photon multipoles), which feed ClEE/ClTE
polarization (the EE gate is already the tightest at 0.231%). The win is small and
the gate is close. **Cheap to bracket** (sweep {1e-6,3e-6,1e-5}, run accuracy
test) — do it as a free rider alongside A1, but don't count on it.

---

## D — float32 / mixed precision on the perturbation state y (big-but-risky)

**Mechanism.** Everything is float64 (`perturbations.py:13`). A100 FP64 is ~½ FP32;
the Kvaerno5 Jacobian/LU is FP-heavy. Run only `y` in float32 (`y0=y_ini.astype(
f32)`, `perturbations.py:414,436`), keep `lna`/`k`/background in float64.

**Expected speedup: 1.4–1.8× on the solve.** **Accuracy risk: HIGH** — the gate.
`rtol_large=1e-4 > fp32 eps(1e-7)`, so the integration tolerance is fp32-compatible
in principle, but the Cl chain *squares* the transfer function and Pk squares
delta_m (`spectrum.py`), amplifying relative error; and the high-l multipoles
(~1e-13) lose all precision in fp32 (eps·magnitude underflows relative to the
delta~1e4 components in the same vector). **Effort: MEDIUM** (must respect
`_to_float` casting to fp32 not fp64, `main.py`; mixed-dtype across the CPU/GPU
transfer is a footgun — `notes_solve.md:434-450`). **P(success): LOW-MED (0.35)**
of passing the gate.

**Hidden floor.** (1) The state vector's dynamic range (1e4 to 1e-17) is **larger
than fp32 can represent relative to its top element** — the small multipoles, which
carry the damping-tail and polarization info, are exactly the EE/TE-sensitive ones,
and fp32 will zero them relative to delta. (2) `tf32` (the cheaper middle ground on
A100 tensor cores) only helps matmuls; the Kvaerno5 LU is element-wise/triangular,
not a matmul, so tf32 may not even engage. (3) diffrax's adaptive controller error
norm in fp32 may chatter near its own eps, *increasing* rejected steps. **Spike
only after the safe wins; gate against the *full* accuracy test, not snapshots.**

---

## E1 — `get_starting_time`'s two 10000-pt interp inversions per lane

**Mechanism.** `get_starting_time` (`perturbations.py:277-289`) builds a
10000-point `linspace` and does **two `jnp.interp` inversions** (`f1`, `f2`) **per
(k,B) lane**, every solve. `f1 = τ_c·aH` and `f2 = k/aH` over 10000 lna points —
that is 10000 `aH` evaluations (each a `rho_tot` species-loop!) ×2, per lane. This
is *outside* the ODE loop (once per solve, not per step), so it's a fixed
per-solve cost, but it is **vmapped over K_CHUNK×B**, so it's `10000×2×Ny-ish work
× K_CHUNK×B`. The `f1` curve depends only on `lna` and params (NOT k) — it is
**identical across all k in a batch** and recomputed K_CHUNK times redundantly.
Only `f2 = k/aH` depends on k (and only through the scalar `k` multiplier — `aH`
over the grid is k-independent!).

**Optimization:** precompute `f1` and the `1/aH` grid **once per cosmology**
(outside the k-vmap), pass them in, and reduce `get_starting_time` to two cheap
`jnp.interp` lookups on precomputed grids (the `k/aH` inversion becomes
`jnp.interp(R_large, k*inv_aH_grid, lna_grid)` with `inv_aH_grid` shared). This
moves 10000×2 `aH`-species-loops from per-(k,B)-lane to per-cosmology.

**Expected speedup: 1.02–1.1× on the solve** (it's a one-time-per-solve cost, not
per-step, so it's a small fraction of total — but it's vmapped K_CHUNK× redundantly
and `aH` is expensive). The 10000-pt grid over `aH`-species-loops is non-trivial:
if `get_starting_time` is, say, 3–8% of the solve, deduping the k-redundant part
recovers most of that. **Accuracy risk: NONE** (CSE — same values). **Effort: LOW**
(precompute in `_compute_modes_batched`, pass through). **P(success): HIGH (0.8)**.
**Hidden floor:** it's a small fraction of the per-solve cost (one-shot, not
per-step), so even fully eliminated the win is bounded by that fraction. Also, the
10000-pt grid may already be cheap if XLA fuses it. Bundle with B1; don't pursue
alone.

---

## Multi/throughput levers (out of "solve" scope but the actual throughput answer)

The brief's success metric is **cosmologies/GPU-second**, and the solve already
*amortizes well* over B (12.2 → 2.0 s/param B=1→16, `CHANGELOG.txt:23`). The
biggest throughput lever is **larger B + more GPUs**, not per-solve tuning:

- **G1 — push B higher with C's memory relief + sharding.** 4-GPU B=64 = 1.13
  s/param and *still falling with B* (`battle_royale_brief.md:33-34`). C (saveat
  trim) + sharding quarters per-device memory → B=128–256 likely fits → per-param
  keeps dropping. **This is the highest-throughput lever and is nearly free** (C is
  LOW effort, sharding exists). Speedup: continues the observed B-scaling, plausibly
  to ~0.6–0.8 s/param at B≥128.
- **G2 — multi-node data parallelism.** Embarrassingly parallel over B (no
  collectives, `battle_royale_brief.md:28`); the premium queue can grab >1 node.
  Pure linear throughput in #GPUs. Effort: SLURM + a top-level scatter. P(success):
  HIGH. This is how a real 20–40-param scan gets done regardless of per-solve tuning.

These bound the *practical* answer: per-solve tuning (B1/A1) buys ~1.3–1.8×
compounded; B-scaling + nodes buys the order-of-magnitude.

---

# RANKING

Rank metric: (speedup × P(success)) ÷ (risk × effort), risk/effort on 1=low..3=high.

| # | idea | speedup | P | risk | effort | score | verdict |
|---|------|--------:|--:|-----:|-------:|------:|---------|
| **1** | **B1 RHS CSE (precompute bg 1×/step)** | 1.15–1.5× | 0.8 | 1 | 2 | **~0.5** | DO FIRST. Zero accuracy risk, attacks a *measured* redundancy (5× aH/step). |
| **2** | **A1 stiffness-homogeneous chunking** | 1.1–1.35× | 0.85 | 1 | 1 | **~1.0** | DO. Cheapest, zero-risk; magnitude modest (k_max floor). |
| 3 | E1 dedup get_starting_time | 1.02–1.1× | 0.8 | 1 | 1 | ~0.8 | Bundle with B1 (free CSE rider). |
| 4 | G1 larger-B + C saveat relief | continues B-scaling | 0.7 | 2 | 1 | high* | Throughput king; C is the enabler. |
| 5 | A1.2 loosen atol_large | 1.05–1.2× | 0.5 | 2 | 1 | ~0.3 | Free rider on A1; bracket it, don't bank on it. |
| 6 | A4 real TCA | 1.3–2.0× | 0.4 | 2 | 3 | ~0.2 | High ceiling, only after a step-location profile justifies it. |
| 7 | D float32 state | 1.4–1.8× | 0.35 | 3 | 2 | ~0.1 | Big-but-risky; dynamic-range floor likely fails EE gate. |
| 8 | J1 banded Jacobian | 1.3–2.5× (large Ny) | 0.3 | 2 | 3 | low | Dominated by B1 at Ny=46; revisit for massive-ν. |
| 9 | A2 explicit/IMEX | 1.0–1.3× (band) | 0.25 | 3 | 2 | low | Stiffness is k-independent → backfires without A4. Demote. |
| 10 | A3 fixed-step | 1.2–1.4× | 0.2 | 3 | 2 | low | Brittle across the scan's param box. |
| — | B2 reduce N_k | ~1.0–1.1× | 0.25 | 3 | 1 | low | Spline accuracy IS the gate. Avoid (lensing tail already off for Cl-only). |

\*G1/G2 score off-scale on throughput but are out of the literal "solve kernel"
scope; flagged because they are the real answer to "cosmologies/GPU-second."

---

# TOP BET + cheapest de-risking measurement

**#1 bet: B1 — common-subexpression-eliminate the RHS background recompute.**
It is the only lever that attacks a redundancy I can *see in the source*
(`aH`→`rho_tot` species-loop computed 5×/RHS, `τ_c`→`xe` 3×, `τ` 2×;
Fact 2 above), has **zero accuracy risk** (bit-identical CSE, gated by the
unmodified `rtol=1e-8` snapshot test), and compounds with everything else (a
cheaper RHS makes the Jacobian assembly cheaper too). Its only real uncertainty is
*how much XLA's CSE already catches* — which is exactly what the cheap measurement
settles.

**Cheapest de-risking measurement (one GPU, minutes):**
Inside an `srun` allocation (`PYTHONPATH=$(pwd):$PYTHONPATH`, `module load conda &&
conda activate actdr6`), time **one compiled `_evolve_chunk`** call on a single
fixed-stiff k over B=16, two ways:
1. baseline RHS (current `get_derivatives`);
2. a hacked `get_derivatives` that computes `aH`, `tau`, `tau_c` **once** and
   stuffs them into a module-level cache / passes them as closure constants to the
   species `y_prime` (a throwaway monkeypatch, not the real refactor).

Compare post-compile wall-clock of `block_until_ready` on the two. If (2) is
≥1.15× faster, B1 is worth the full refactor; if it's <1.05×, XLA already deduped
it and we pivot to A1 + G1. This isolates the RHS cost from the rest of the
pipeline and needs neither CLASS nor the full `call_batched`.

A second, even cheaper measurement that informs A1 *and* A4 simultaneously: dump
the **per-step `t` (lna) locations** from one `evolution_one_k` at k_max (set
`SaveAt(steps=True)` on a throwaway copy, or read diffrax `sol.stats`). If the 1579
steps cluster *pre-recombination* (stiff `1/τ_c` window) → A4/TCA has a high
ceiling; if they cluster *at the visibility peak* → TCA won't help and A1 + B1 are
the whole game. This step-location histogram is the single most informative number
not yet measured.

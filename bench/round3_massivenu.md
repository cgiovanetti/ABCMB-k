# Round 3 — Question B: architectural speedups for massive neutrinos

Static analysis only. All claims cite `file:line` in this checkout (branch `perk-perf`).
Measured ground truth: B=16 single-GPU, one massive ν = 6.47 s/param vs ~1 s massless
sharded; peak 9.5 GB at B=16 (`bench/round2_massive.jsonl`, `CHANGELOG.txt:46`).

---

## 0. What the code actually does (the crux, read first)

### State-vector size
- `MassiveNeutrino.num_equations = 3 * (l_max_massive_nu + 1)` — `species.py:744-745`.
  With the default `l_max_massive_nu = 17` (`model_specs.py:34`) that is **3 × 18 = 54
  equations** for ONE massive species.
- The "3" is the **number of momentum bins N_q = 3** for the *perturbation* hierarchy
  (`q_3p`, `w_3p` are length-3, `species.py:729-730`; the `for i in range(3)` loops at
  `species.py:849, 891, 941, 974, 1006`).
- Baseline massless Ny ≈ 46 (Photon 13+11=24 `species.py:1262-1264`; massless ν 18
  `species.py:571`; Baryon 2; CDM 1; metric η 1). **+54 ⇒ Ny ≈ 100** for one massive ν.
  (The brief/CHANGELOG quote "Ny≈83" and "Ny≈250" — both are loose; 83 was an *empirical
  fit* of the 0.60 GB/B_local memory line, 250 a stale guess. The code-exact number is
  46→100. Either way the lever is the same: cut the +54.)

### N_q INCONSISTENCY (background vs perturbations)
- **Background** ρ/P use the **5-point** grid `q_5p`/`w_5p` (`species.py:773-777, 810-813`).
- **Perturbations** (`y_ini`, `y_prime`, the moment integrals `rho_delta`,
  `rho_plus_P_theta`, `rho_plus_P_sigma`) use the **3-point** grid `q_3p`/`w_3p`
  (`species.py:849, 891, 941, 974, 1006`).
  So ABCMB *already* runs the minimal N_q=3 in the part that drives Ny. There is little
  headroom to cut N_q below 3 (CLASS's lowest `ncdm_quadrature_strategy` is also 3 nodes;
  going to 2 is known to mis-track the free-streaming suppression). **The big Ny lever is
  l_max, not N_q.**

### Is the q-loop a python loop that serializes under vmap? (brief item #3)
**No — and this is important to get right.** `for i in range(3)` (`species.py:891`) is a
*trace-time* Python loop: `q = self.q_3p[i]` indexes a static array with a Python int, so
the loop fully **unrolls** into 3 statically-sized blocks inside one `jnp.concatenate`
(`species.py:913-915`). Under the double-vmap (`perturbations.py:142-148`) it vectorizes
cleanly — there is no per-q dynamic dispatch and no `lax.scan` over q. The cost it adds is
**graph size / FLOPs**, not serialization: the RHS HLO is ~2.2× longer (54 vs 24 ν-eqns),
so each Kvaerno5 stage and its Jacobian factorization are bigger. That, plus the q-integral
recompute below, is the 6×.

### Why 6× slower (brief item #3 diagnosis)
Three multiplicative contributions, in order of importance:
1. **Bigger stiff system.** Kvaerno5 is implicit; per accepted step it does a Newton solve
   whose linear-algebra cost scales worse-than-linearly in Ny (the dense Jacobian block for
   the coupled hierarchy is O(Ny²) to form and the implicit solve O(Ny²)–O(Ny³) depending on
   sparsity exploitation). Ny 46→100 (≈2.2×) ⇒ roughly 2.2²≈4.8× on the dominant term.
2. **Redundant live background q-integration in the RHS (brief item #5 — confirmed, and it's
   worse than "just massive-ν").** `BG.aH` (`background.py:169`) → `H` → `rho_tot`
   (`background.py:120-123`) loops over *all* species calling `.rho()` **live every call**.
   `MassiveNeutrino.rho` is the 5-point q-integral with `exp`+`sqrt` (`species.py:773-779`).
   `aH` is called **independently ~5-6× per single RHS evaluation**: once in the metric block
   (`perturbations.py:356`) and once inside *each* species' `y_prime` (Photon
   `species.py:1357`, massless ν `:670`, baryon `:1212`, massive ν `:886`, baryon.cs2 `:1134`).
   So one RHS call redoes the massive-ν 5-point integral ~5-6 times, and the whole `rho_tot`
   species sum ~5-6 times, none of it cached. `tau`/`xe`/`Tm`/`expmkappa` ARE tabulated+interpolated
   (`background.py:322-348, 559-563`) — but `aH`/`rho_tot`/`tau_c` are NOT. For massless ΛCDM
   each `.rho()` is one cheap `exp`; with massive ν it's a 5-wide reduction, so this redundancy
   that was negligible becomes a real per-step tax exactly when ν is massive.
3. **More steps (smaller contribution).** The ncdm `q/ε` advection terms (`species.py:900-910`)
   add fast oscillatory modes at high q after the ν go non-relativistic; the PID controller
   takes somewhat more steps. This is the smallest of the three and is genuinely physical.

Net: ≈4.8 (system size) × ≈1.2-1.3 (RHS recompute) ≈ 6×. The system-size term is the floor;
the recompute term is free to recover.

---

## Ranked shortlist (top 5)

### 1. **CSE / pre-tabulate `aH` (and the `rho_tot` species sum) so the massive-ν q-integral isn't recomputed 5-6× per RHS step.** ★ #1 BET
**Mechanism.** `aH(lna)` is a pure function of `lna` and `params`; it is recomputed from
scratch (full `rho_tot` species loop incl. the massive-ν 5-point integral) every time any
species' `y_prime` asks for it — ~5-6× per RHS call (`background.py:169`→`120`; callers
`perturbations.py:356`, `species.py:670,886,1134,1212,1357`). Two accuracy-NEUTRAL forms:
(a) **CSE within a step** — compute `aH` (and `metric_h_prime`/`metric_eta_prime`, which
already are) once in `get_derivatives` and thread it into each `y_prime` via the existing
`args` tuple (`perturbations.py:374`) instead of each species recomputing it; or
(b) **tabulate `aH` on `lna_tau_tab`** exactly like `tau`/`expmkappa` already are
(`background.py:559-563`) and read it with `interpax.interp1d` — `aH` is monotone-smooth so
a cubic on the 10000-pt grid is ≫permille-accurate. (b) also kills the *background-build*
recompute, not just the RHS one.
**Effect.** Removes ~4-5 redundant 5-point q-integrals + species sums per RHS step. Expected
~1.15-1.35× on the *massive* per-param (the recompute term in the 6× breakdown); also a smaller
win for massless ΛCDM (cheaper, but `aH` is still recomputed 5-6×/step there too — this is a
general lever that massive-ν merely amplifies). **Bit-identical for (a); ≪permille for (b).**
**Accuracy risk.** (a) None (algebraic CSE). (b) interpolation error of a smooth monotone
function on 10k points — gate once.
**Effort.** (a) Low-moderate (thread `aH` through the `y_prime` signature — touches every
species, but mechanical). (b) Low (mirror the `expmkappa_tab` pattern, one new field +
interp method). **Prob success: high.**
**Hidden floor.** Doesn't touch the O(Ny²) implicit-solve term (#1 in the 6× breakdown), so
the ceiling of this lever alone is ~1.3×. Best paired with a hierarchy-size cut below.

### 2. **Lower `l_max_massive_nu` from 17 → ~7-9 (the dominant Ny lever).**
**Mechanism.** Ny_massive = 3 × (l_max+1). 17→8 is 54→27 eqns, halving the massive
contribution and pulling total Ny 100→73. This cuts BOTH the persistent memory tensor
(∝ Ny, `round2_plan.md` memory model) AND the per-step implicit-solve cost (∝ ~Ny²,
the #1 term in the 6× breakdown). It is the single biggest combined mem+time lever.
**Why it's plausibly over-truncated.** ABCMB sets massive = massless = 17 (`model_specs.py:33-34`),
but the physics differs: massive ν free-stream less (they slow down and cluster), so their
high-ℓ multipole tail is *less* excited than the massless tail at the ℓ that matter for
CMB/Pk. CLASS's default `l_max_ncdm` is **17** too, BUT CLASS applies the ncdm **fluid
approximation** above a trigger (see #4) so it rarely integrates all 17 ncdm multipoles to
late times — ABCMB integrates the full 17-deep hierarchy for the *entire* run, which is the
expensive regime CLASS specifically avoids. So 17 is likely conservative *given that ABCMB
never truncates*. A truncation-error closure (the `(lmax+1)/aH/tau` term, `species.py:910`)
already damps the top mode, so moderate l_max is self-stabilizing.
**Effect.** l_max 17→8: ~0.5× massive Ny ⇒ persistent mem ↓ ~27% (100→73 total Ny) and
per-param ↓ via the Ny² term — plausibly 1.4-1.8× on the *massive* solve. 17→11 is the safe
conservative cut (~1.2-1.3×).
**Accuracy risk.** MODERATE and gateable — this is a physics truncation, must clear permille
vs CLASS on TT/EE/Pk *with a massive ν*. Sweep l_max ∈ {17,13,11,9,7} and watch the gate.
**Effort.** Trivial (one spec default; already plumbed through `num_ells_per_bin`).
**Prob success: high** for a modest cut (→11-13), medium for an aggressive one (→7-8).
**Hidden floor.** N_q=3 and the 3 lowest moments (δ,θ,σ) are physically load-bearing and
can't be cut; you're only trimming the free-streaming tail.

### 3. **Static (switch-free) fluid/reduced-hierarchy closure at high q — NOT a runtime regime switch.**
**Mechanism.** CLASS truncates each ncdm momentum bin to a (δ,θ,σ) fluid once non-relativistic.
The user has BANNED a *runtime diffrax regime switch* (constraint 2 — expensive under vmap).
But there is a **switch-free** variant: keep the full hierarchy for the *single* relevant bin
(or use a *fixed, low* l_max for the high-q bins and a higher l_max only for the lowest-q bin,
since the q/ε advection coupling that excites high multipoles scales with q — `species.py:900-910`).
I.e. a **q-dependent static l_max**: e.g. l_max = (12, 8, 6) for q_3p = (0.91, 3.38, 7.79)
instead of a flat 17 for all three. Set once at construction (`species.py:744`); no runtime
branch, vmaps cleanly. This is a structured version of #2 that exploits the physics (high-q
modes free-stream into high ℓ *more*, but contribute *less* to the moments by the 1/q² and
1/ε weights at `species.py:947,979,1012) — actually the dominant moment contribution is the
LOW-q bin, which argues for keeping low-q deep and trimming high-q).
**Effect.** Comparable to #2 but better accuracy-per-equation-saved: could reach
Ny_massive 54→~26 (12+8+6) with less accuracy loss than a flat l_max=8.
**Accuracy risk.** MODERATE, gateable; same gate as #2 but needs a per-bin sweep to tune the
profile. Architectural (no regime switch), honors constraint 2.
**Effort.** Moderate (the `range(3)` loops at `species.py:891` etc. would need per-bin
`num_ells_per_bin[i]`; the `jnp.concatenate` already supports ragged bins). The state-vector
indexing (`i*self.num_ells_per_bin`, `species.py:897,945,977,1010`) becomes a cumulative
offset — a clean refactor.
**Prob success: medium** (more tuning than #2 for a similar payoff). **Hidden floor:** the
low-q bin still needs a deepish hierarchy for the σ/Pk suppression; can't go fluid everywhere.

### 4. **Pure-fluid closure ONLY at the relevant k (super-horizon / low-k modes), set per-k at trace time.**
**Mechanism.** The fluid approximation is exact for modes still outside the horizon /
deep-radiation era. ABCMB already branches tolerances on k (`k_split_PE`,
`perturbations.py:420-430`); one could *statically* choose a reduced-l_max ncdm hierarchy
for the low-k chunk and the full one for high-k, since `_evolve_chunk` already compiles
per-k-chunk (`perturbations.py:126-148`). Because k-chunks are separate JIT cache entries,
a per-chunk static l_max is **switch-free** (chosen at trace time by chunk index, not at
runtime inside diffrax).
**Effect.** Saves Ny only on the low-k chunks (a fraction of N_k≈492); modest overall
(~1.05-1.15×) and complicates the chunk machinery.
**Accuracy risk.** Low IF the cut is restricted to genuinely super-horizon modes; gateable.
**Effort.** Moderate-high (per-chunk species construction; interacts with the k_chunk=100
default the user froze). **Prob success: medium-low.** **Hidden floor:** the stiff,
expensive modes are the *high-k* ones (`round2_plan.md` notes the single stiffest mode at
k_max is the per-chunk floor) — low-k modes are already cheap, so trimming them saves little
wall-clock even though it saves equations. **Lower priority than #2/#3.**

### 5. **Vectorize the q-loop as a true (3, l_max) batched operation + tabulate `tau_c` (secondary CSE).**
**Mechanism.** (a) Rewrite the `for i in range(3)` hierarchy (`species.py:891-915`) as a
single vmapped/array op over a (3, num_ells_per_bin) block. This does NOT change Ny or FLOPs,
but produces a tighter HLO (one fused kernel vs 3 unrolled blocks + a concat), which can
shave XLA overhead and Jacobian-assembly time, and makes a per-bin l_max (#3) cleaner.
(b) Also tabulate `tau_c` (`background.py:670-691`) — currently live `nH·xe` interp per call,
called in baryon.cs2/y_prime (`species.py:1134,1213,1215`); smaller than the `aH` win but
same pattern.
**Effect.** ~1.05-1.15× (XLA codegen / kernel-fusion only); bit-identical (a) /
≪permille (b). Mostly a *codegen* and *enabler* lever, not a headline win.
**Accuracy risk.** None (a) / interpolation-gate (b).
**Effort.** Moderate (a is a real rewrite of the hierarchy kernel; b is easy).
**Prob success: medium.** **Hidden floor:** XLA may already fuse the unrolled blocks well,
so the measured benefit could be near zero — probe HLO/timing before investing.

---

## #1 BET and the cheapest de-risking measurement

**#1 bet: Lever 1 (tabulate/CSE `aH` so the live massive-ν q-integral + `rho_tot` species
loop isn't recomputed 5-6× per RHS step), implemented as the tabulated `aH` variant (1b)
mirroring the existing `expmkappa_tab` pattern.** Rationale: it is the only top lever that is
**accuracy-neutral (≪permille, gateable once) AND benefits BOTH massive and massless paths**,
it directly attacks the confirmed redundant-recompute term in the 6× breakdown, and it's low
effort (one tabulated field + an `interpax.interp1d` reader, copying `background.py:559-563`).
It also *amplifies* the value of every Ny-cut lever (#2/#3) because a smaller hierarchy with a
cheap `aH` compounds. If you want strictly bit-identical, do variant 1a (thread `aH` through
`get_derivatives`→`y_prime`) first.

The single biggest *combined* mem+time win is Lever 2 (lower `l_max_massive_nu`), but it's an
accuracy trade that must clear the gate, so it ranks #2; pair it with the #1 bet.

**Cheapest GPU/CLASS de-risking measurement (one srun, ~15 min):**
Run `pytests/accuracy_test.py` (the existing <permille-vs-CLASS gate) with **one massive ν
enabled** (`user_species=(MassiveNeutrino,)`, `N_nu_massive=1`, `m_nu_massive≈0.06`,
`Neff≈3.044` as in `bench/mem_throughput_sweep.py:24,75-78`) at the default
`l_max_massive_nu=17`, then **re-run at l_max ∈ {13, 11, 9, 7}** and a single `aH`-tabulated
variant. One script, one allocation:
- Confirms the massive-ν accuracy baseline vs CLASS exists at all (the current gate is
  massless ΛCDM — there may be no massive-ν gate yet; establishing it is prerequisite for #2/#3).
- The l_max sweep directly reads off the accuracy headroom for the #2/#3 Ny cut (where TT/EE/Pk
  cross permille tells you the safe l_max).
- The `aH`-tabulated point confirms #1 stays ≪permille.
De-risks #1, #2, and #3 simultaneously, and tells you whether a massive-ν gate needs building
before any of this can be claimed.

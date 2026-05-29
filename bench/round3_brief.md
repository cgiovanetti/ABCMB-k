# Battle-royale brief — round 3 (perk-perf): memory/call + massive-ν speed

READ `bench/round2_plan.md` first (round-2 synthesis + measured numbers). This brief
adds the round-2 VALIDATED ground truth and the two new questions. Static analysis
only — DO NOT run python/GPU (orchestrator runs tests). Cite `file:line`. Be
skeptical: for each idea give mechanism, expected effect (range), accuracy risk +
the cheap gate, effort, prob success, and the HIDDEN FLOOR. Rank by
(benefit × prob ÷ (risk × effort)).

## Validated ground truth (round 2, A100-80GB, measured)
- Pipeline: `Model.call_batched` shards the B axis over GPUs (GSPMD, P('batch')),
  each device handles B_local = B/n_dev cosmologies. Per-param falls with B
  (1.09 s @B=64 → 0.44 @B=512); throughput ≈ raise B + add nodes.
- **Per-device GPU peak ≈ 3.65 · N_k · Nlna · Ny · 8B · B_local** — the PERSISTENT
  saved-trajectory tensor. INDEPENDENT of k_chunk (the transient Kvaerno5 workspace
  is negligible at kc≤100). Matches massless ΛCDM (Ny≈46 → 0.33 GB/B_local) and one
  massive ν (Ny≈83 → 0.60 GB/B_local) exactly. N_k≈492 (lensing=False), Nlna=500.
- So fitting more B/node ⇔ shrinking this persistent tensor (Nlna, Ny, the 3.65
  overhead, or not materializing it), since k_chunk can't help.
- HARD CONSTRAINTS (user): (1) stay near PERMILLE — no accuracy-for-speed trades
  (tol-loosening, float32/bf16/fp32-storage were measured & REJECTED; re-propose
  only if you can argue it stays ≪0.1% AND gate it). (2) NO TCA / diffrax
  regime-switching — it's expensive under vmap (slows us down). (3) k_chunk stays
  100 (data: smaller is strictly slower, no mem benefit).

## QUESTION A — reduce memory per call (fit more B_local on one 80 GB GPU)
The peak is the persistent tensor 3.65·N_k·Nlna·Ny·8B·B_local. Investigate, with
file:line from the actual code:
- **Nlna (=500, the save grid)**: `perturbations.py` `_compute_modes_batched`
  (lna=linspace(lts,0,500)) and `make_output_table`; `spectrum.py` integrates the
  LoS on this SAME grid (PT.lna). Non-uniform dense-at-recomb grid → fewer points
  same accuracy? Is it accuracy-neutral or a gate risk? It cuts BOTH memory and the
  spectrum scan.
- **The 3.65× overhead**: what is resident beyond the raw saved-ys (N_k·B·Nlna·Ny)?
  Map it: the raw modes tensor, the `PerturbationTable` derived fields
  (`perturbations.py` make_output_table → delta_m, theta_b_prime, metric_*,
  species_perturbations dict), and the spectrum working set (`spectrum.py`
  get_Cl_batched/Cl_one_ell). Which of these coexist in `main.py call_batched`
  (BG_batch + PT_batched + params_batch + Cls)? Can intermediates be freed
  (donate_argnums) or never materialized?
- **Streaming / not materializing the full (B,Ny,Nlna,N_k) tensor**: today modes →
  full PT → spectrum. Could the pipeline stream per-k-chunk (evolve a k-chunk →
  fold into the LoS/Pk accumulators → discard) so the full tensor never exists?
  What blocks it (the spectrum does a global cubic-spline over PT.k → all k needed
  at once; quantify how much that forces materialization).
- **donate_argnums / buffer aliasing** across full_evolution_batched → PT → Cls.
- **B-axis chunking of the spectrum** if the spectrum working set (not the solve) is
  a secondary peak.
Goal: how much smaller can mem/B_local get, accuracy-neutral, and what's the
resulting max B_local on 80 GB?

## QUESTION B — architectural speedups for massive neutrinos
Massive ν is BOTH the memory hog (Ny 46→~83) AND ~6× slower per solve (B=16
single-GPU: 6.5 s/param vs ~1 s massless sharded; bench/round2_massive.jsonl).
Read `abcmb/species.py` MassiveNeutrino (num_equations, the momentum q-grid, the
moment/q-integration, the ncdm Boltzmann hierarchy), and how it enters
`perturbations.py` (get_derivatives, initial_conditions, make_output_table) and the
background (`background.py` massive-ν ρ/P q-integrals). Investigate:
- **Momentum grid N_q**: how many q-points, how chosen, how integrated? num_equations
  = N_q·(l_max_ncdm+1) → Ny. Fewer q-points (Gauss-Laguerre/Gauss-Legendre with
  fewer nodes, or analytic moments) → proportionally smaller Ny → less mem AND
  faster. Accuracy-gated — quantify the headroom (CLASS uses ~5-15 q; what does
  ABCMB use, and is it over-resolved?).
- **l_max_ncdm (=17)**: is the massive-ν multipole hierarchy over-truncated/under?
  Lowering it cuts Ny.
- **Why 6× slower**: is it just Ny (bigger system → bigger Jacobian/RHS) or extra
  stiffness from the ncdm hierarchy? Does the q-loop vectorize well under the
  existing double-vmap, or is there a python loop over q?
- **ncdm fluid approximation** (CLASS truncates the massive-ν hierarchy to a fluid
  δ/θ/σ once non-relativistic): would it help? BUT note constraint (2) — if it needs
  a runtime regime switch in diffrax it's likely rejected; is there a switch-free
  form (e.g. always-fluid at the relevant k, or a fixed reduced hierarchy)?
- Any redundant background q-integration recomputed per RHS call.
Goal: ranked architectural levers to make massive-ν cheaper in BOTH memory and time,
honoring the accuracy + no-regime-switch constraints.

## Deliverable
Write your memo to the assigned `bench/round3_*.md`, and return a ranked shortlist
(top ~5, one paragraph each) + your #1 bet + the cheapest GPU/gate measurement to
de-risk it.

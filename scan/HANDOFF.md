# HANDOFF — frequentist tool, 2026-06-22 (night)

Branch `perk-perf` (NOT merged to main). Canonical goal: **give a new ABCMB cosmology →
profile-likelihood optimum ±½σ in a FEW HOURS of compute** (must beat CLASS+MH ΛCDM+Neff
@ 22 h). Plan lineage: `scan/TOOL_PLAN.md` (strategy) → this doc (current state) →
`CHANGELOG.txt` top entries (detail). Read this FIRST. (Prior `HANDOFF` content, 2026-06-11,
is in git history — superseded.)

────────────────────────────────────────────────────────────────────────
## TL;DR — where we are
The frequentist profile tool **works and is queued to deliver**. **One job is queued
(54845799)** that produces the result in ~6.7 h. The **chi2-plateau early-stop** (the ~2×
idea) is now **IMPLEMENTED and mechanically validated** (default OFF, safe) — see
§"EARLY-STOP STATUS (2026-06-23)" below. What remains is **tuning FTOL on a real full-l
trace** (the smooth low-l proxy can't), then flipping the default to land the ~2× on every
run after the headline.

────────────────────────────────────────────────────────────────────────
## EARLY-STOP STATUS (2026-06-23, session "frequentist_3")
**DONE — implemented + mechanically validated, default OFF.**
- `profile_prod_ad.py:bfgs_rows` stops a POI's lockstep once the **MAX-over-rows** per-row
  best-chi2 improvement over `PA_FTOL_PATIENCE` (3) iters drops below `PA_FTOL`. MAX (not
  min-row) so lagging edge ±3σ rows aren't cut early. Post-loop AD ‖g‖ cert unchanged.
- **DEFAULT OFF (`PA_FTOL=0`)** — a premature stop biases the interval and the queued
  54845799 reads this `.py` at runtime; full-MAXIT is already ~6.7 h (< 22 h bar), so the
  early-stop is a pure speedup, not a correctness need. Ship OFF, enable once tuned.
- **Validated on a FAST proxy** (debug full-l is too slow: ~9–22 min/iter + a ~17-min
  staged-AD compile that does NOT persist across jobs). `PA_USE_PLIK=0` (new toggle,
  default ON) drops high-ell plik-lite (it needs l~2508; LMAX=900 gave chi2~185000
  garbage) → lowTT+lowEE only. `scan/configs/lowl_proxy.py` profiles tau over ln10As
  (2-param, seconds/iter at LMAX=100). `PA_BF_TRACE` dumps per-iter best_f;
  `bench/analyze_estop.py` replays the trigger. **Result: σ1 freezes by it1, Δσ1=+0.0000
  for all (FTOL 3e-4..5e-3, PATIENCE 3-5)** — mechanics sound.
- **What the proxy CANNOT do** → the real remaining task: the cond=1.0 proxy converges
  uniformly (no edge-lag) and is SMOOTH (no ~0.1 roughness floor), so it can't tune FTOL
  for the rough full problem. `FTOL=1e-3` is likely BELOW the roughness floor → would
  never fire → no speedup; the right FTOL is ~the roughness scale (**~0.05–0.1**), giving
  σ1 good to the ~0.02–0.03σ floor the result already lives with.

### REMAINING (to land the ~2×)
1. **Capture a real full-l per-iter trace.** `PA_BF_TRACE` is now rank-aware
   (POI_SLICE → `_r<rank>`) and wired into `profile_prod_ad.slurm` (FTOL stays 0 = full
   safe run). So the NEXT production submit (resubmit the headline OR run lcdm_neff)
   auto-captures `scan/results/bf_trace_prod_r<rank>.npz`. (54845799 itself won't — its env
   froze at submit; it runs full, no trace.)
2. **Tune:** `python bench/analyze_estop.py scan/results/bf_trace_prod_r0.npz` → pick the
   smallest FTOL whose Δσ1 « 0.02σ AND that actually fires above the roughness floor.
3. **Flip the default** `PA_FTOL` in `profile_prod_ad.py` to the tuned value; commit. The
   ~2× then applies to every subsequent run. **Keep the headline full** (precision bar).
   (The original 2026-06-22 diff + gotchas are in §"NEXT TASK (priority 1)" below — now
   IMPLEMENTED; kept for the rationale.)

**DO NOT cancel the pending regular job 54845799** (user instruction). It reads the `.py`
files at runtime, so any committed `.py` change applies to it automatically — no resubmit
needed. It won't start for ~2 days (deep regular queue).

────────────────────────────────────────────────────────────────────────
## Jobs / state
- **54845799** `abcmb_prof_mn6` — PENDING, regular, 6 nodes, 8 h walltime. The production
  run: `PA_POI_SLICE=1 PA_NPTS=11 PA_MAXIT=18 PA_TAG=_mn6 PA_GTOL=0.03 PA_HESS=0`, l=2508.
  Each rank does 1 POI × 11 grid pts. Writes `scan/results/profile_prod_ad_<poi>_mn6.npz`.
  Emails on start/end. Mid-BFGS + done-POI resumable. **Leave it queued.**
- No other jobs of ours. The interactive node from tonight (54853342) was released.

────────────────────────────────────────────────────────────────────────
## What was diagnosed + fixed tonight (all committed on perk-perf)
The AD gate kept timing out. Root causes, in order of discovery:
1. **Contaminated Hinv (dominant).** The resumed BFGS carried a preconditioner built from
   FD-noise (s,y) pairs → near-zero search steps. **Fix = fresh start** (clean inverse-Fisher
   Hinv0, computed/loaded automatically when there's no matching checkpoint). DEMO PROVED it:
   ln10As l=2508 fresh start drove ‖g‖ **153→16→10.5→5.6→3.06→2.47**, chi2 settled to 996.49
   by it2 (the contaminated run was stuck rising at ~4.5).
2. **value == gradient** (refuted a red herring): `f_callbatched == f_AD` to <1e-4. Not a
   mismatch.
3. **Roughness floor ~0.1 in chi2 (~0.1 in ‖g‖), rtol-INDEPENDENT.** `diag_fscan.py` showed
   rtol 1e-5 vs 1e-6 give identical roughness (0.077 vs 0.076). So GTOL=0.03 is BELOW the
   achievable floor and won't be met; the source is a DISCRETIZATION (k_chunk and/or the
   round-4 visibility lna grid), NOT the ODE tol. **Do NOT chase this with PA_RTOL=1e-6.**
   It does NOT hurt the deliverable: chi2 per grid pt is good to ~0.1 and that AVERAGES OUT
   in the 11-pt spline → intervals good to ~0.02–0.03σ.

Commits (newest first): `daec127` chunk-cap fix · `b632821` f-scan verdict ·
`034a922` collect_profiles.py · `2a19a74` warm-Hessian cache · `e1a5603` diagnosis +
POI-slice + prod launch · `aa1f6bb` lcdm grad_method=ad.

### Performance levers landed
- **Multi-node POI-slice** (`PA_POI_SLICE=1`): each rank owns disjoint POIs, writes its own
  npz → no clobber, no MPI. Validated on 2-node debug. This is the node-scaling path.
- **Warm-Hessian cache** (`PA_HESS_CACHE`, default on): precomputed to
  `scan/results/warm_hessian_lcdm_l2508_tt1_ee1.npz`. Loads in **0 s** (was ~18 min). l=2508
  conditioning confirmed good: cond 32–159 per POI.
- **Chunk cap** (`_chunked_call_batched: chunk=min(chunk,N)`): killed the POI-slice
  line-search waste (11-row call was paying for 128 cosmos). Per-iter 30→22 min → production
  fits 8 h (was ~9 h timeout risk).

### Tools / artifacts (all in `bench/` unless noted)
- `diag_ad_stall.py` — per-row ‖g‖ map + box-pinning + two-direction (−g and −Hinv·g) descent
  probe at a checkpoint.
- `diag_consistency.py` — f_callbatched vs f_AD at given rows.
- `diag_fscan.py` (+`.slurm`) — 1-D f-scan along −g at two rtols (roughness test).
- `precompute_hessian.py` (+`.slurm`) — compute+cache the warm Hessian, print conditioning.
- `scan/collect_profiles.py` — combine per-POI npz → table (vs Planck18 + SMC) + overlay
  plot. Usage: `python scan/collect_profiles.py _mn6` (run inside an srun). Dry-tested on the
  entry-(a) FD profiles: all 6 POIs within ±0.5σ of Planck AND the SMC posterior.

────────────────────────────────────────────────────────────────────────
## NEXT TASK (priority 1) — chi2-plateau early-stop → ~3 h  [IMPLEMENTED 2026-06-23 — historical]
**Why:** the demo showed chi2/intervals settle by ~it5 while ‖g‖ grinds slowly toward its
~0.1 roughness floor and never reaches GTOL=0.03 — so iters ~6–18 are wasted chasing an
unreachable certificate. Stopping when chi2 plateaus cuts the production ~6.7 h → ~3 h. The
AD ‖g‖ certificate still runs after the loop and reports the honest floor.

**The change** (in `scan/profile_prod_ad.py`, function `bfgs_rows`). Add near the env block:
```python
FTOL = float(os.environ.get("PA_FTOL", "1e-3"))        # chi2-plateau early-stop threshold
FTOL_PATIENCE = int(os.environ.get("PA_FTOL_PATIENCE", "3"))
```
In `bfgs_rows`, initialise `bf_hist = [best_f.copy()]` right before the `for it in range(...)`
loop (both the resume and fresh branches share `best_f` by then). Inside the loop, AFTER the
`upd = f < best_f; best_f[upd]=...; best_x[upd]=...` line and the ckpt write, add:
```python
    bf_hist.append(best_f.copy())
    if FTOL > 0 and len(bf_hist) > FTOL_PATIENCE:
        improve = float((bf_hist[-1 - FTOL_PATIENCE] - best_f).max())  # max per-row chi2 drop over window
        if improve < FTOL:
            print(f"  {log_prefix} chi2 plateau: max per-row improve {improve:.2e} "
                  f"< FTOL {FTOL:.0e} over {FTOL_PATIENCE} iters -> stop at it{it}", flush=True)
            break
```
Per-rank/POI independent; the post-loop cert + npz write are unchanged.

**Gotchas to respect:**
- Use MAX per-row improvement (the slowest row), NOT the min-row chi2 — edge (±3σ) rows settle
  later. The demo only logged min-row chi2, so the right PATIENCE/FTOL is UNVERIFIED → must
  validate before trusting.
- A premature stop = biased intervals. Default FTOL=1e-3 is conservative; do not loosen
  without the validation below.
- On resume, `bf_hist` starts fresh (history not checkpointed) — fine, it rebuilds; worst case
  it runs PATIENCE extra iters after a resume. Acceptable.

**Validation plan (do this BEFORE relying on it for the production):**
1. Debug job, low l for speed but enough to settle: e.g. `PA_POIS=ln10As PA_NPTS=7 PA_LMAX=900
   PA_MAXIT=20 PA_TAG=_estop PA_FTOL=1e-3` on 1 node. Confirm it STOPS only after the chi2
   profile is visibly flat, and that the resulting `sigma1` interval matches a no-early-stop
   reference run (`PA_FTOL=0`) to «0.05σ. (Low-l is fine — the plateau LOGIC is l-independent;
   you're testing the trigger, not the science.)
2. If the interval matches → the early-stop is safe; the queued 54845799 picks it up at runtime
   (it reads the patched `.py`). Expect it to stop ~it6–8 → ~3 h.
3. If it stops too early (interval shifts) → raise FTOL_PATIENCE to 4–5 and/or lower FTOL to
   3e-4, re-test.

Project rule: write the diff to a file for review first, then apply.

────────────────────────────────────────────────────────────────────────
## After the early-stop lands (in order)
1. **Let 54845799 run** (or resubmit `sbatch scan/profile_prod_ad.slurm` with the same
   POI_SLICE/NPTS/TAG env — it auto-resumes per-rank). Then `python scan/collect_profiles.py
   _mn6` → the headline ΛCDM table + plot. Compare to Planck18, SMC (`smc_lcdm.npz`), and the
   entry-(a) FD profiles.
2. **Workstream C:** ΛCDM+Neff via `scan/configs/lcdm_neff.py` (grad_method already "ad") on
   the same POI-slice harness — the competitive headline vs 22 h.
3. **Tighter certificate (optional):** find the roughness source — test `lna_grid_mode=
   "uniform"` (revert round-4) and/or larger `k_chunk`; NOT rtol (proven null).
4. Coverage mocks (Workstream E); intra-POI row-slice (needs the MPI gather) for >6-way scale.

────────────────────────────────────────────────────────────────────────
## Hard constraints (user)
Stay near permille (no tol-loosening / fp32). No TCA / diffrax regime-switching. k_chunk
stays 100. Don't lower l_max for massive ν. **Prefer debug queue; ≤1 self-initiated
interactive session and only if <2 already running, release fast; NEVER scancel other
sessions' jobs.** Scale GPUs (nodes), not walltime. **Do not cancel 54845799.**

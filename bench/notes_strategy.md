# ABCMB per-k Batched Refactor — Skeptical Performance Strategy

**Author:** performance-architect subagent pass, 2026-05-29 (branch `perk-refactor`)
**Mandate:** decide where GPU-hours go to reach the stated TARGET of **~1 s per parameter combination** for a frequentist scan of B cosmologies.

**Method note:** All wall-clock numbers below are quoted from in-tree artifacts (`CHANGELOG.txt`, `bench/design_memo.md`, `bench/perf_batched.log`, `bench/baseline_summary.txt`, `bench/flipped_summary.txt`, `bench/flipped_multigpu_summary.txt`, `bench/chunking_debug_report.md`) and cross-checked against each other. Code facts were read directly from `abcmb/{main,perturbations,spectrum,background,model_specs}.py` and are cited by line. The diffrax-vmap-lockstep claim in §3 was verified against the diffrax FAQ and GitHub issue #281 (citations in §3). **No Python/GPU was run for this analysis (login-node Python is forbidden and the GPUs are in use elsewhere); everything here is from reading artifacts + source + the web.** The one biggest un-measured quantity (the per-stage split of the 12 s end-to-end batched time) is exactly what §4's profiler is designed to confirm before any GPU-hour campaign.

**Solver config (confirmed `abcmb/model_specs.py:60-69` + `abcmb/perturbations.py:404-444`):** `Kvaerno5` implicit solver, `diffrax.ForwardMode` adjoint by default (no reverse AD in the frequentist path), `PIDController(pcoeff=0.25, icoeff=0.8, dcoeff=0.0)`, `max_steps_PE=2048`, `k_split_PE=0.01` with `rtol_large_k_PE=1e-4 / atol_large_k_PE=1e-6` (k>0.01) and `rtol_small_k_PE=1e-5 / atol_small_k_PE=1e-10` (k≤0.01). The PE saveat grid is a hard-coded `jnp.linspace(BG.lna_transfer_start, 0., 500)` (`perturbations.py:100,180`); the Cl source grid is a separate hard-coded `jnp.linspace(BG.lna_rec, 0., 500)` (`spectrum.py:419`). `lna_transfer_start` is per-cosmology (`background.py:564`), so the batched path correctly builds a per-element lna grid via `vmap(jnp.linspace)` (`perturbations.py:179-181`).

---

## 0. Verified baseline ledger (the only numbers we trust)

| Quantity | Value | Source |
|---|---|---|
| End-to-end `model()` per-params, post-compile | **9.89 s** (Phase A) / **9.51 s** (perf_batched warm) | baseline_summary; perf_batched.log |
| `PE.full_evolution` per-params (perturbations alone) | **8.14 s** | baseline_summary |
| Everything else (recomb + BG + spectrum + param-deriv) | 9.89 − 8.14 = **~1.75 s** | derived |
| N_k (perturbation k modes) | **571** | flipped_summary, design_memo |
| Per-k diffrax step counts (single-cosmology vmap-over-k) | min=41, median=380, mean=658, max=1579 | baseline_summary |
| Worst-case-k tax: max/median over k | **4.16** | baseline_summary |
| Flipped (params-first) double-vmap, single A100, B=64, PE-only | **3.43 s/params** (2.37× vs 8.14) | flipped_summary |
| Flipped PE-only at B=1 / B=4 / B=16 | 7.77 / 4.76 / 3.57 s/params | flipped_summary |
| Flipped worst-k step tax: max/median over B | **1.12** (median-k 1.02) | flipped_summary |
| Flipped, 4-GPU sharded over B, B=4/16/64, PE-only | 2.21 / 1.26 / **0.93** s/params | flipped_multigpu_summary |
| Peak memory, single A100 B=64 (PE) | **28.37–31.33 GB / 40 GB** (XLA rematerialization warning) | flipped_run.log:35, design_memo §1 |
| State vector size N_y | ~72 | design_memo §1.2 |
| PE saveat lna grid N_lna | 500 | perturbations.py:100,180 |
| End-to-end `call_batched` (spectrum loop included), single A100, ELLMAX=800 | B=1 22.64, B=4 14.46, B=8 12.95, B=16 12.08 s/params | perf_batched.log |
| Asymptote of end-to-end batched per-params | **~12 s** (set by python `get_Cl_batched` loop) | perf_batched.log, CHANGELOG Phase E |
| End-to-end B=2 Cl/Pk parity vs single | ClTT 1.4e-7, ClTE 3.0e-5, ClEE 1.6e-8, Pk 2.6e-5 (all under 1% gate) | smoke_batched_pipeline.log |

**Two distinct "batched" measurements that must not be confused:**
1. **PE-only flipped spike** (`flipped_summary.txt`): 3.43 s/params at B=64. Perturbation solve alone, double-vmap, no spectrum loop, no per-cosmology CPU tails. The optimistic number.
2. **End-to-end `call_batched`** (`perf_batched.log`): 12.08 s/params at B=16, *getting worse not better* as B grows toward a ~12 s floor. The *real* current pipeline, and it is **slower than the 9.51 s single call**. The 2.37× PE win is masked by the python `get_Cl_batched` loop + un-batched CPU tails.

**Code-confirmed structure of the spectrum tail (this is more nuanced than "the spectrum is a python loop"):**
- `SpectrumSolver.get_Cl_batched` (`spectrum.py:616-649`) is a **python `for i in range(B)` loop** (`spectrum.py:642`) calling `self.get_Cl(PT_i, BG_list[i], ...)`, because `get_Cl` reads `BG.visibility`/`BG.expmkappa` (`spectrum.py:692-695`) which depend on `BG.kappa_func` — a `diffrax.Solution` (`background.py:453,491,552`) consumed via `kappa_func.evaluate(lna)` (`background.py:741`). That `Solution` does not survive `jax.tree.map(jnp.stack, ...)`, which is why `strip_bg_kappa` (`perturbations.py:538`) exists and why the spectrum runs un-stacked.
- `SpectrumSolver.Pk_lin_batched` (`spectrum.py:651-659`) is **ALSO a python `for i in range(B)` loop** (`spectrum.py:655`), not a vmap. (Earlier draft mis-stated this — corrected.) However `Pk_lin` itself never touches `kappa_func` (`spectrum.py:268-347`; it only interpolates `PT` fields and reads scalar `params`), so **Pk_lin_batched is trivially convertible to an outer `vmap` with zero new infrastructure** — the only reason it is a loop is consistency with `get_Cl_batched`. The expensive line-of-sight Cl integral is the real cost; Pk is cheap.

The gap between (1) 3.43 s and (2) 12 s is the whole story of this document.

---

## 1. BUDGET MODEL (quantitative)

Decompose end-to-end per-params at batch B into three cost classes:

```
T_per_params(B) = T_cpu_serial / 1           # un-batched per-cosmology CPU work (HyRex+param-deriv+BG): already per-cosmology, flat per-params
                + T_gpu_pert(B) / B          # batched GPU perturbation solve: sublinear-growing total, ~constant-ish per-params at large B
                + T_spectrum_loop(B) / B     # Cl python loop ×B (Pk loop too, but cheap): strictly per-element, flat-to-rising per-params
```

### 1a. Pin the per-stage numbers

**Perturbation (GPU, batched).** PE-only per-params (flipped_summary): B=1 7.77, B=4 4.76, B=16 3.57, B=64 3.43 s/params. *Total* GPU wall = per-params × B: B=1→7.77 s, B=64→219.5 s. The absolute PE cost grows ~28× as B grows 64× — **sublinear, so there IS batching benefit, but the device is NOT idle.** It asymptotes near ~3.4 s/params: the worst-k tax (4.16) is being amortized away, but memory saturation (§2) caps the win.

**Cl spectrum loop (python ×B; Pk loop cheap).** Back out from perf_batched: end-to-end batched − PE-batched.
- Single `model()` is 9.51 s of which 8.14 is PE → single spectrum+BG+recomb+param ≈ 1.37 s (consistent with the 1.75 s Phase-A residual; use ~1.4–1.75 s).
- End-to-end batched at B=16 is 12.08 s/params; PE-batched at B=16 is 3.57 s/params. So **everything-but-PE in the batched path costs ~8.5 s/params at B=16** — ~6× the single-call ~1.4 s. The excess is `get_Cl_batched` python-loop dispatch overhead: each `get_Cl` element pays full JIT-dispatch + the `kappa_func.evaluate` Solution call inside `expmkappa`/`visibility`, B times, with no kernel overlap.
- Model: `T_Cl_loop(B)/B ≈ T_Cl_single + dispatch_overhead` ≈ 1.4 + ~7 s of per-element python/JIT-dispatch friction. Dominant at B≥8, flat-to-rising in B. Matches the observed ~12 s asymptote.
- **Caveat:** the ~7 s figure is an *inference* from `(end-to-end − PE) ≈ 8.5 s/params` at B=16, attributed to the Cl loop because the rest (Pk loop, BG build) is small/known. It is the single biggest un-measured assumption in this document — §4's profiler exists to confirm it.

**CPU tails (HyRex recomb + `add_derived_parameters` + BG build).** Run un-batched, once per cosmology, sequentially in `_build_one_bg` (`main.py:240-265`). HyRex is `eqx.filter_jit(self.RecModel, backend='cpu')` (`main.py:256`); param-derivation is plain Python/CPU (`main.py:448-692`, with several `for s in self.species_list` loops and CPU LINX/table branches). No isolated measurement exists; bounded above by the single-call non-PE budget. Estimate **t_cpu ≈ 0.5–1.0 s/cosmology** (sequential xe/Tm solve dominates). This adds a **flat 0.5–1.0 s/params floor batching cannot remove** — only CPU(i+1)/GPU(i) overlap or a CPU vmap of HyRex (Phase F, deferred) hides it.

### 1b. Achievable per-params if each stage is batched/fixed

| Stage | Current per-params (B=64) | After the obvious fix | Floor mechanism |
|---|---|---|---|
| Perturbation (GPU) | 3.43 s | 3.0–3.4 s (single GPU); ~0.85–0.93 s (4-GPU shard) | GPU compute, partially saturated |
| Cl spectrum + Pk | ~7–8.5 s (python loops) | **~1.4 s** once true-vmap (kappa_func tabulation) | one batched kernel, ~flat in B |
| CPU tails (HyRex+param+BG) | ~0.5–1.0 s | ~0.2–0.5 s if HyRex CPU-vmapped (Phase F) | sequential CPU solver |

**Single A100, all stages batched (no sharding):**
```
T ≈ 3.0–3.4 (PE)  +  1.4 (vmap spectrum)  +  0.5–1.0 (CPU tail)  ≈  4.9–5.8 s/params
```
**→ 1 s/param is NOT reachable on a single A100.** The PE solve alone is ~3.4 s/params at B=64 on one A100 (memory-pressured at 28–31 GB, XLA rematerializing — §2). Even with a free spectrum and zero CPU tail the single-GPU floor is the PE solve, ≈3 s.

**4-GPU sharded over B, all stages batched:**
```
T ≈ 0.85–0.93 (PE, 4-GPU)  +  ~0.4 (vmap spectrum, also shardable)  +  0.2–0.5 (CPU tail, shardable)  ≈  1.3–1.7 s/params
```
With CPU tails overlapped behind the GPU solve and the spectrum sharded too, **4-GPU lands in the ~1.3–1.7 s/params band.** The CHANGELOG's 0.93 s/params at B=64 4-GPU is **PE-only** — it excludes spectrum and CPU tail. The *honest* end-to-end 4-GPU number is **~1.3 s, not 0.93 s**, unless the spectrum and CPU work are also sharded and the CPU is hidden.

### 1c. Verdict for §1

**1 s/param requires the 4-GPU sharding the spike used; it is not reachable on a single A100.** The perturbation ODE solve is a hard ~3 s/params floor there because the A100 is already near memory saturation at B=64 and cannot absorb more batch lanes. Honest budget:
- **Single A100, fully optimized: ~5 s/params** (PE-bound).
- **4× A100, fully optimized + CPU hidden + spectrum sharded: ~1.3–1.7 s/params** (just brushes the target).
- The spike's **0.93 s is PE-only and optimistic**; treat **~1.3 s** as the realistic 4-GPU end-to-end number until measured. Reaching a clean ≤1 s likely needs 4-GPU **plus** a precision/saveat win (#5/#6) on the PE solve.

---

## 2. CONTRARIAN PREMISE CHECK (the most important section)

**The refactor's premise:** batching B cosmologies fills GPU lanes that a single cosmology leaves idle, so we get a large near-linear speedup.

**The data says this premise is mostly FALSE, and the real win is a different mechanism.** Three pieces of evidence:

**(i) A single cosmology already vmaps over 571 k modes** (`perturbations.py:110`, `vmap(self.evolution_one_k, in_axes=[0,None,None])`). The GPU is not idle for one cosmology — it is already running a 571-wide vmap of the Kvaerno5 implicit solve. There are not 64× free lanes waiting; there are ~571 lanes already occupied, and adding B multiplies the vmap cell count to 571×64 = 36 544 (design_memo §1.2). That is a saturation problem, not an idle-lanes opportunity.

**(ii) The single-GPU speedup is 2.37×, not ~64×.** flipped_summary: PE per-params 7.77 (B=1) → 3.43 (B=64). If batching filled idle lanes, per-params would collapse toward a tiny constant. Instead it asymptotes at ~3.4 s. The *total* GPU wall grows 7.77→219.5 s as B grows 1→64 (28×): the device is doing real, growing work — **compute/memory bound, not lane-starved.**

**(iii) Memory is pressured at 28–31 GB / 40 GB at B=64.** flipped_run.log:35 logs the literal XLA rematerialization warning: "Can't reduce memory use below 28.37GiB … only reduced to 31.33GiB." design_memo §1 attributes ~10.5 GB to saveat trajectory output and ~4.5 GB to Kvaerno5 Jacobian/LU workspace. A device 70–78% full of working set and forcing XLA rematerialization is **saturated**. You cannot push B much past 64 on one A100 without OOM (hence mandatory k-chunking to 100 → 5.5 GB).

**So where does the 2.37× actually come from?** Not free lanes — the **worst-case-k tax reduction, 4.16 → 1.12.** In the single-cosmology path every one of the 571 k modes pays the worst k's step count under one shared adaptive controller (max-step ~1579 vs median 380). In the flipped path, at *fixed k* across B cosmologically-similar cosmologies (Planck ±2–3σ), step counts are nearly uniform (max/median 1.12). The flipped order **stops the easy k modes from paying the hard k mode's step bill.** CHANGELOG confirms: "Step-count hypothesis CONFIRMED: baseline max/median over k = 4.16; flipped worst-k max/median over B = 1.12."

Naively the tax reduction should buy 4.16/1.12 ≈ 3.7×. We see 2.37×. The shortfall (3.7 → 2.37) is the saturation/memory tax: at B=64 the device rematerializes, eating part of the step-count win.

**Verdict — what gets us from 3.43 s → 1 s:**

| Path | Does it get us there? |
|---|---|
| (a) Multi-GPU sharding over B | **YES, and REQUIRED.** 4-GPU PE-only is 0.93 s. End-to-end realistically ~1.3 s. No single-GPU route to 1 s. |
| (b) The lockstep / step-count win | **ALREADY SPENT.** 4.16→1.12 is essentially the floor (1.0 = perfect). ≤~12% more recoverable (§3). Not a lever to 1 s. |
| (c) Batching the CPU + spectrum tails | **NECESSARY but NOT SUFFICIENT.** This is what makes the 3.43 PE number *real* — right now end-to-end is 12 s, so the tails are the #1 problem. Fixing them gets single-GPU end-to-end 12 → ~5 s, the precondition for the 4-GPU number to mean anything. Does not by itself reach 1 s. |

**Bottom line:** order of operations is **(c) then (a).** First make the end-to-end pipeline pay out the PE batching win it already has (kill the `get_Cl_batched` python loop, hide/vmap the CPU tail) — single-GPU end-to-end 12 → ~5 s. Then shard over B across 4 GPUs to reach ~1.3 s. The premise "batching fills idle lanes" is wrong; the correct framing is **"batching equalizes step counts so we stop paying the worst-k tax, and multi-GPU restores the linear scaling that single-GPU memory saturation prevents."**

---

## 3. LOCKSTEP-STEPPING headroom

**How much more is recoverable beyond 4.16 → 1.12?** Almost none.
- Perfect lockstep = ratio 1.00. We are at **1.12** (worst-k), **1.02** (median-k): 88% of the way from baseline 4.16 to ideal 1.00 on the worst k, 98% at the median.
- Remaining headroom is **at most ~12%** on the worst k, and only on the minority of k modes where the B cosmologies still disagree on step count. Across all k the realizable gain is **single-digit percent** of total PE time. Not worth GPU-hours as a primary lever.

**The diffrax mechanism (VERIFIED against diffrax docs/issues):** `vmap` over a diffrax `diffeqsolve` with an adaptive PID controller runs **all batch elements in lockstep on a single shared step schedule.** XLA cannot give different vmap lanes different control flow, so diffrax steps the whole batch until the last (worst) element finishes, masking already-converged lanes. Confirmed by:
- **diffrax FAQ** (https://docs.kidger.site/diffrax/further_details/faq/): "When solving multiple ODEs in parallel using `vmap`, the ODEs will need to have the same step size and max steps, which means some of the small ODEs may be overly-sampled on an extra fine step size that is determined by the stiffest small ODE … [for matching tolerances] aim to have roughly the same number of steps instead."
- **GitHub issue #281, "Ensemble simulations of small ODEs on GPUs"** (https://github.com/patrick-kidger/diffrax/issues/281): same lockstep behavior for ensembles.

A vmapped adaptive solve therefore takes `max_i(steps_i)` steps, not `mean_i(steps_i)`. This is exactly why the flipped order works — the FAQ's prescription ("aim to have roughly the same number of steps") is satisfied by batching cosmologies at fixed k (1.12) but violated by batching k at fixed cosmology (4.16). The 1.12 residual is the irreducible stiffness heterogeneity among Planck ±2–3σ cosmologies at a fixed k.

**Ways to push below 1.12, in order of cost/benefit:**
1. **Sort/bucket k modes by expected step count, chunk like-with-like.** Already partially happening via the `k_split_PE=0.01` rtol/atol split (`perturbations.py:420-430`); `chunking_debug_report.md` shows step behavior changes exactly at the k_split boundary (index 52). Tighter bucketing within the high-k branch could shave a little, but the report shows that branch is already uniform-rtol. **Low ROI.**
2. **Fixed-step (ConstantStepSize / Tsit5/Dopri5) for the well-characterized high-k oscillatory modes**, adaptive only for the stiff low-k tail. Removes the controller's worst-case coupling on the predictable part. **Medium effort/reward, accuracy risk** — `chunking_debug_report.md` warns fixed-step "costs both wall-clock and stiffness robustness" for bit-parity, but as a *speed* lever on k>k_split only it may merit one experiment (verify high-k modes are non-stiff enough first).
3. **Per-lane early termination / event-based stopping** — not supported by diffrax vmap (the whole point of lockstep). Unavailable.

**Conclusion:** lockstep is a *solved* lever. The 4.16→1.12 win is banked; <~12% remains and it is not where GPU-hours should go. Do not run a benchmark campaign trying to beat 1.12.

---

## 4. PROFILING RECIPE (copy-pasteable, minimal GPU cost)

Drop this in `bench/profile_call_batched.py`. ONE warm + ONE measured pass, with `block_until_ready` fences around each stage, plus a perfetto trace, plus a static cost/memory analysis. Total GPU cost ≈ 2 batched calls (one warm + one measured) ≈ a few minutes at B=16.

```python
# bench/profile_call_batched.py
# Run under srun (NEVER login node):
#   srun --jobid=$(cat bench/.jobid) --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
#        bash -c 'module load conda && conda activate actdr6 && \
#        PYTHONPATH=$(pwd):$PYTHONPATH python bench/profile_call_batched.py'
import os, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax, jax.numpy as jnp
import numpy as np
import equinox as eqx
from abcmb.main import Model
from abcmb.perturbations import strip_bg_kappa
print("abcmb from:", __import__("abcmb").__file__)   # MUST be ABCMB-k, not the sibling checkout
print("devices:", jax.devices())

B = 16
# Match perf_batched.py / baseline.py settings.
model = Model(user_species=None, output_Cl=True, l_max=800, lensing=True,
              output_Pk=True, output_k_max=0.5,
              l_max_g=12, l_max_pol_g=10, l_max_massless_nu=17, l_max_massive_nu=17)
FID = dict(h=0.6762, omega_cdm=0.1193, omega_b=0.0225, A_s=2.12424e-9, n_s=0.9709,
           tau_reion=0.0544)
rng = np.random.default_rng(0)
params_list = [{k: float(v*(1+0.01*rng.standard_normal())) for k, v in FID.items()}
               for _ in range(B)]

def fence(x):  # block on every array leaf
    return jax.block_until_ready(x)

# ---------- warm (compile everything) ----------
t0 = time.time(); fence(model.call_batched(params_list))
print(f"[warm] compile+run {time.time()-t0:.2f}s")

# ---------- measured, per-STAGE fences (replicates call_batched body) ----------
# Stage 1: build B Backgrounds (CPU HyRex serial loop + param derivation)
t0 = time.time()
full_ps, bgs = [], []
for p in params_list:
    fp = model.add_derived_parameters(p)
    fp, bg = model._build_one_bg(fp)
    full_ps.append(fp); bgs.append(bg)
fence([bg.tau_tab for bg in bgs])
t_bg = time.time() - t0

# Stage 2: batched perturbation solve (GPU)
t0 = time.time()
params_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps)
BG_stripped  = jax.tree.map(lambda *xs: jnp.stack(xs),
                            *[strip_bg_kappa(bg) for bg in bgs])
PT = fence(model.PE.full_evolution_batched((BG_stripped, params_batch)))
t_pe = time.time() - t0

# Stage 3a: Cl python loop  /  Stage 3b: Pk loop
t0 = time.time()
ClTT, ClTE, ClEE = model.SS.get_Cl_batched(PT, bgs, params_batch)
fence((ClTT, ClTE, ClEE)); t_cl = time.time() - t0
t0 = time.time()
Pk = fence(model.SS.Pk_lin_batched(model.SS.k_axis_Pk_output, 0., PT, params_batch))
t_pk = time.time() - t0

tot = t_bg + t_pe + t_cl + t_pk
print(f"[stage] bg/hyrex/param = {t_bg:.3f}s  ({t_bg/B*1000:.0f} ms/cosmo)")
print(f"[stage] perturbation   = {t_pe:.3f}s  ({t_pe/B:.3f} s/params)")
print(f"[stage] Cl loop        = {t_cl:.3f}s  ({t_cl/B:.3f} s/params)")
print(f"[stage] Pk loop        = {t_pk:.3f}s  ({t_pk/B:.3f} s/params)")
print(f"[stage] TOTAL          = {tot:.3f}s   ({tot/B:.3f} s/params)")

# ---------- perfetto trace ----------
os.makedirs("bench/trace", exist_ok=True)
with jax.profiler.trace("bench/trace"):
    fence(model.call_batched(params_list))
print("trace -> bench/trace/ ; scp plugins/profile/*/*.trace.json.gz -> https://ui.perfetto.dev")

# ---------- static FLOPs + bytes (NO execution) to diagnose GPU saturation ----------
# Lower+compile the hot PE chunk kernel and read its cost model.
k_chunk = model.PE.k_axis_perturbations[:100]
lna_batch = jax.vmap(lambda lts: jnp.linspace(lts, 0., 500))(BG_stripped.lna_transfer_start)
lowered = eqx.filter_jit(model.PE._evolve_chunk).lower(
    k_chunk, lna_batch, BG_stripped, params_batch)
comp = lowered.compile()
try:
    print("cost_analysis  :", comp.cost_analysis())     # dict incl 'flops', 'bytes accessed'
except Exception as e:
    print("cost_analysis n/a:", e)
try:
    print("memory_analysis:", comp.memory_analysis())   # temp_size_in_bytes, output_size, etc.
except Exception as e:
    print("memory_analysis n/a:", e)
```

**How to read the outputs for saturation:**
- `cost_analysis()['flops'] / t_pe` = achieved FLOP/s. Compare to A100 peak fp64 ≈ 9.7 TFLOP/s (tensor-core fp64 ≈ 19.5). If achieved ≪ peak → **memory/latency-bound, not compute-bound** → batching/float32 still has room. If near peak → compute-bound → only more GPUs help.
- `cost_analysis()['bytes accessed'] / t_pe` vs A100 HBM ≈ 1.5–2.0 TB/s. The Kvaerno5 implicit solve on a (571×B, 72) state is almost certainly **bandwidth/latency-bound** (small dense LU per cell), consistent with the 2.37× (not 64×) batching win.
- `memory_analysis().temp_size_in_bytes` — confirms the design_memo 28–31 GB and the real OOM headroom per B before scanning B.

**Cheapest variant (no internals replication):** add `block_until_ready` + `time.time()` fences inside `call_batched` (`main.py:218-233`) around the three existing stages — the `_build_one_bg` loop, `full_evolution_batched`, and `get_Cl_batched`/`Pk_lin_batched` — guarded by `if os.environ.get("ABCMB_PROFILE")`. Commit it; it costs nothing in production and is the single most useful instrumentation to land.

---

## 5. ROI-RANKED OPTIMIZATION TABLE

End-to-end per-params speedups unless noted. "Confidence" = how sure the number is given current evidence.

| # | Optimization | Expected speedup (end-to-end) | Effort | Risk | Confidence | Depends on |
|---|---|---|---|---|---|---|
| 1 | **kappa_func → (lna_grid, expmkappa_grid) tabulation, then true-vmap `get_Cl_batched`** (+ Pk_lin_batched, trivial) | **~2–2.4×** (kills the ~7–8 s Cl python loop, `spectrum.py:642`; end-to-end B=16 12 → ~5 s) | Medium (localized to `background.py`; replace `kappa_func.evaluate` at `:741` with `jnp.interp`/`tools.fast_interp`, mirroring `tau_tab`/`tau()` at `:347`) | Low–Med (keep visibility/expmkappa accuracy; 1% gate) | **High** — documented #1 bottleneck; fix spec'd in CHANGELOG "Path forward" + `proposed_chunking_fix.py` | none |
| 2 | **Batch HyRex/param-derivation (Phase F) OR overlap CPU(i+1) with GPU(i)** | **~1.1–1.3×** (removes 0.5–1.0 s/params serial CPU floor) | Med (CPU vmap of HyRex `array_with_padding`) / Low (double-buffer `_build_one_bg`) | Med (HyRex is sequential; vmap may not help — overlap is safer) | Med | profiling §4 to size t_cpu first |
| 3 | **Multi-GPU sharding over B** (`Mesh` + `NamedSharding`/`shard_map` over the batch axis) | **~3.7× on PE** (3.43→0.93 PE-only); end-to-end ~3–4× once #1 done | Med (wire mesh into `call_batched`; spike `flipped_spike_multigpu.py` already proved it) | Med (4× GPU cost; load-balance B across 4 ranks) | **High** — flipped_multigpu measured 0.93 s PE-only at B=64 | #1 (else you shard a python loop = no win) |
| 4 | **k_chunk_size tuning / single big vmap vs chunked** (`full_evolution_batched(..., k_chunk_size=)`) | 1.0–1.15× (memory headroom → larger B per device, not raw speed) | Low (it's a kwarg, default 100) | Low | Med | profiling memory_analysis |
| 5 | **float32 perturbation hierarchy** (keep BG/spectrum fp64) | ~1.3–1.6× on PE (halves Jacobian/LU workspace 4.5→2.25 GB → less rematerialization → more B/GPU; tensor-core fp32 faster) | Med | **High** (1%-vs-CLASS gate; tight-coupling + Thomson sensitive — design_memo §2.3) | Low–Med | accuracy_test.py pass |
| 6 | **Reduce PE saveat N_lna** (500 → ~150–200; `perturbations.py:100,180`) | ~1.1–1.2× + ~6 GB memory (10.5→~3–4 GB saveat) | Low | Med (spectrum interp accuracy; 1% gate) | Med | accuracy_test.py pass |
| 7 | **donate_argnums + persistent compile cache** (`jax_compilation_cache_dir`) | First-call only: kills 28–233 s recompile per shape; ~0 on steady-state per-params | Low | Low | High | none |
| 8 | **Pad B to fixed power-of-two shapes** (avoid recompile per scan batch size) | First-call only; avoids recompiling for each B | Low | Low (wastes lanes if B≪pad) | High | none |
| 9 | **Lockstep/step-count fixes** (fixed-step high-k branch) | **<1.1×** (4.16→1.12 already banked; ≤12% residual) | Med | Med (stiffness/accuracy) | Med — see §3 | profiling step counts |

**Table notes:**
- #1 and #3 are multiplicative and hit *different* terms (spectrum vs perturbation). Do #1 first or #3 buys nothing (sharding a python loop).
- #5/#6 are **memory** plays that buy speed indirectly by relieving the 28–31 GB saturation capping the single-GPU PE win at 2.37×. They are how you'd push single-GPU below ~3.4 s/params — but both risk the 1% gate, so second-wave.
- #7/#8 are free wins on *first-call/compile* latency (the 233 s B=64 compile from flipped_run.log:36), which matters for interactive iteration and for scans that vary B. They do nothing for steady-state per-params.

---

## 6. RECOMMENDATION

### First 2–3 changes (in order)
1. **kappa_func tabulation → true-vmap `get_Cl_batched`** (row #1). The python loop at `spectrum.py:642` is *the* reason end-to-end batched is 12 s instead of ~5 s. The fix is already designed in CHANGELOG ("Path forward" / `proposed_chunking_fix.py`): replace the `diffrax.Solution` `kappa_func` (`background.py:552,741`) with a pre-tabulated `(lna_grid, expmkappa_grid)` and rewrite `expmkappa`/`visibility` to `jnp.interp`/`tools.fast_interp` (the pattern `tau_tab`/`tau()` already use at `:347`). Then `Background` stacks cleanly, `strip_bg_kappa` can go away, and both `get_Cl_batched` and `Pk_lin_batched` become outer vmaps. Precondition for *any* multi-GPU number to be real. **Do this first.** Expected: end-to-end B=16 12 → ~5 s on a single A100.
2. **Profiling instrumentation** (§4) committed behind `ABCMB_PROFILE`, run once at B=16. We are flying blind on the actual t_cpu and on whether the PE solve is compute- or bandwidth-bound. One measured `cost_analysis()` + a perfetto trace tells us whether #5/#6 are worth the accuracy risk and confirms the 3.43 PE number end-to-end. Costs ~2 batched calls.
3. **Multi-GPU sharding over B** (row #3), *after* #1. The spike proved 0.93 s PE-only at 4-GPU; productionizing the shard takes end-to-end into the ~1.3 s band — the only route to the 1 s target.

### The single cheapest experiment to validate the premise before spending GPU hours
**Run the §4 profiler at B=16 on one already-allocated GPU and read three numbers:**
(a) the per-stage split (bg/hyrex vs PE vs Cl-loop vs Pk-loop) — confirms the Cl loop is the ~7–8 s culprit and sizes t_cpu;
(b) `cost_analysis()['flops'] / t_pe` vs A100 peak — confirms whether PE is compute-bound (→ multi-GPU is the only lever) or bandwidth/latency-bound (→ batching/float32 still has room);
(c) `memory_analysis().temp_size_in_bytes` — confirms the 28–31 GB saturation and the max B per A100.

This is **one short post-warm run (~minutes)** and it decides whether the money goes to multi-GPU (premise: saturated, need more devices) or to single-GPU memory plays (#5/#6). **Do not allocate the 4-GPU campaign until this profiler has confirmed the per-stage split and the saturation reading.** Everything in this document hangs on the un-measured assumption that the PE solve is the floor and the Cl loop is removable friction — the profiler verifies both in one shot.

---

## 7. Skeptic's caveats (read before trusting any number above)

- **The 0.93 s 4-GPU number is PE-only and wall-clock-only, not parity-checked** (CHANGELOG Phase B CAVEAT). The trajectories it produced were subject to the same diffrax step-controller noise as the chunking "bug." The honest end-to-end 4-GPU number is **unmeasured**; my ~1.3 s is a budget estimate, not data.
- **`perf_batched.log` (12–22 s) are the only true end-to-end measurements we have** and they show batched is currently *slower* than single. Any claim of speedup today is about the PE sub-stage, not the shipped pipeline.
- **The 2.37× single-GPU PE win is below the 3.7× the step-count tax (4.16/1.12) predicts.** The shortfall is memory saturation — the strongest evidence the A100 is the bottleneck (not idle lanes) and that 1 s/param needs multiple GPUs.
- **`Pk_lin_batched` is currently a python loop too** (`spectrum.py:655`), but a cheap one and trivially vmap-able since `Pk_lin` never reads `kappa_func`. The expensive loop is `get_Cl_batched`.
- **float32 (#5) is the only single-GPU lever with real upside, and the riskiest for the 1%-vs-CLASS contract.** Do not commit it without `accuracy_test.py`.
- **Lockstep is done. 4.16→1.12 is banked.** Do not run a benchmark campaign trying to beat 1.12 — §3 shows <12% is left.
- **perf_batched used ELLMAX=800** while baseline/flipped used l_max=2500; the single-call 9.51 s (perf_batched) vs 9.89 s (baseline) reflect that. The per-stage budget mixes these; treat the spectrum/Cl figures as order-of-magnitude until §4 re-measures at one fixed l_max.

Sources (diffrax lockstep, §3): [diffrax FAQ](https://docs.kidger.site/diffrax/further_details/faq/), [diffrax issue #281 — Ensemble simulations of small ODEs on GPUs](https://github.com/patrick-kidger/diffrax/issues/281), [diffeqsolve API](https://docs.kidger.site/diffrax/api/diffeqsolve/).

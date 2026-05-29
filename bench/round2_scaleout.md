# round2_scaleout.md — Systems / scale-out levers for the frequentist scan

Battle-royale round 2, **HPC / distributed-JAX systems specialist**. Branch
`perk-perf`, post-keystone. **Static analysis only — no Python/GPU run** (the
orchestrator runs the GPU jobs). All code facts cited `file:line`; SLURM patterns
quoted from `/pscratch/sd/c/carag/CLAUDE.md`, this repo's `CLAUDE.md`, and
`BBN_Hubble/nersc_train_gpu.slurm`.

My lens: **throw more hardware / better orchestration at it.** I do NOT touch the
ODE math (that's `round2_solver.md` / `round2_precision.md`). My metric is
**sustained throughput = cosmologies per GPU-second** and per wall-clock-second
across whatever hardware we can grab. The single-call solve is near its latency
floor (~0.8 s/param worst-k); my job is to make N cosmologies cheap in aggregate.

---

## 0. Ground truth I am designing against (measured, cited)

### 0.1 Current 4-GPU sharded end-to-end (`bench/perf_multigpu_results.json`)

| B  | per-param (s) | total (s) | warm/compile (s) |
|----|--------------:|----------:|-----------------:|
| 16 | 3.24          | 51.9      | 51.6 |
| 32 | 1.82          | 58.3      | 146.7 |
| 48 | 1.34          | 64.4      | 153.7 |
| 64 | **1.13**      | ~72       | (CHANGELOG) |

Per-param is **still falling at B=64** — the curve has not flattened. Throughput
(cosmologies/s on the 4-GPU node, post-compile) is the reciprocal × B:

| B  | per-param | **throughput (cosmo/s, 4 GPU)** |
|----|----------:|--------------------------------:|
| 16 | 3.24      | 4.9 |
| 32 | 1.82      | 17.6 |
| 48 | 1.34      | 35.8 |
| 64 | 1.13      | **56.6** |

Throughput is rising **super-linearly** in B (4.9 → 56.6 over B=16→64, an 11.5×
gain for a 4× B increase). That is the single most important systems fact in this
note: **we are nowhere near the throughput knee.** The job right now is not "make
the solve faster," it is "feed it bigger B and replicate the node."

### 0.2 The CPU "setup floor" is a MYTH on the current code — corrected

The stale `ideas_multigpu.md §0.1/§2` is built on a **7.0 s/param FLAT setup
floor** attributed to "HyRex + device transfers." **`bench/probe_setup.log`
refutes this directly:**

```
[2] HyRex (jitted CPU):             0.062s        <- the actual HyRex cost
[3a] get_BG EAGER:                  6.626s        <- the "7 s" was THIS
[3b] get_BG JITTED:                 1.156s
HyRex CPU floor (hard to batch): 0.062s/cosmo (5% of jitted setup)
```

The 7 s was **eager `get_BG` dispatch**, which the committed `_build_bgs_batched`
(`main.py:332-379`) already kills by jitting+vmapping `get_BG`
(`_get_BG_batched`, `main.py:324-330`). HyRex itself is **62 ms/cosmo** and is
already vmapped on CPU (`main.py:366`). The CHANGELOG confirms setup amortizes
"30.8 → 2.0 as B grows" at B=16 and "toward ~0.4 at large B." **So item 4 of my
remit (the CPU setup floor) is largely already solved** — I re-cost it in §4
below and conclude it is NOT the throughput ceiling for B≤256, but flag the two
genuine residual serial costs (the `add_derived_parameters` python loop and the
HyRex CPU↔GPU round-trips).

### 0.3 Launch-mode fact that the whole multi-node analysis turns on

There are **two incompatible JAX launch modes** in this workspace, and conflating
them is the #1 way to get multi-node wrong:

- **Spike / `perf_multigpu.py` mode (single process, 4 GPUs):**
  `srun --ntasks=1 --gpus-per-task=4` → `jax.devices()` returns
  `[Cuda0,Cuda1,Cuda2,Cuda3]` to **one** interpreter
  (`bench/multigpu_run.log:9`, `flipped_spike_multigpu.py:17`). This is what
  `call_batched(shard=)` is built for: one `Mesh(np.asarray(gpus))`
  (`main.py:238`), GSPMD over the B axis, no MPI, no `jax.distributed`.
- **BBN_Hubble production MCMC mode (4 processes, 1 GPU each):**
  `--ntasks-per-node=4` + `export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`
  (`nersc_train_gpu.slurm:7-8,78,182`) → each of 4 MPI ranks sees exactly ONE
  GPU. This is the **wrong** mode for `call_batched` sharding: each process would
  build its own `Mesh` of size 1 and never partition. It is the right mode for
  cobaya/MCMC (4 independent walkers), but **the scan harness must NOT inherit
  this `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` line** — that is the easiest
  copy-paste bug to introduce.

This distinction drives the entire multi-node design (§1).

---

## 1. Multi-NODE data parallelism — THE big lever (item 1)

The scan is embarrassingly parallel over cosmologies. There is **no reduction
anywhere** across cosmologies in the pipeline — verified:
`get_Cl_batched`/`Pk_lin_batched` are a plain `jax.vmap` over B with no
`psum`/`all_gather`/`all_reduce` (`spectrum.py:617-650`), and the perturbation
chunk-loop concatenates along the **k** axis, not the batch axis
(`ideas_multigpu.md §1.3`, verified). So K nodes can run **completely
independent** slices of the parameter grid.

### 1a vs 1b: job-array of single-node jobs vs one multi-node `jax.distributed` program

**RECOMMENDATION: (a) a SLURM job array of independent single-node 4-GPU jobs.
Do NOT use `jax.distributed.initialize()` + a global multi-node mesh.**

Mechanism for (a): a sbatch array, each task owns a contiguous slice of the
N-cosmology grid, each task is exactly the `perf_multigpu.py` single-process
4-GPU pattern (`--ntasks=1 --gpus-per-task=4`), each writes its slice's Cls/Pk
(or per-cosmology χ²) to a shard file keyed by `$SLURM_ARRAY_TASK_ID`. A trivial
post-pass concatenates the shards.

Why (a) beats (b) for a frequentist scan:

| Axis | (a) job array | (b) `jax.distributed` global mesh |
|------|---------------|------------------------------------|
| Coupling | zero — slices never talk | all ranks join a coordinator; one slow/failed rank stalls the world |
| Fault tolerance | one task OOMs/dies → re-queue that ONE task; rest finish | one rank dies → whole job dies, lose all nodes' work |
| Interconnect | none needed (no NVLink even used, `multigpu_run.log:1-8`) | cross-node collectives over Slingshot for a B-axis we never reduce — pure overhead |
| Scheduler fit | premium queue backfills small 1-node jobs easily; array tasks start as nodes free up | needs K nodes co-scheduled simultaneously (long queue wait for K≥4) |
| Code change | **~0** (wrap existing `call_batched`) | non-trivial: `jax.distributed.initialize(coordinator_address=...)`, global `Mesh` over (node×gpu), process-id plumbing, MPICH bootstrap |
| Compile cost | paid per task, but **shared via persistent cache** (§2) | paid once but all K nodes block on first-compile barrier |

The frequentist scan is the **textbook embarrassingly-parallel HTC workload**;
`jax.distributed` exists for models too big for one node (sharded *parameters*,
cross-device collectives), which is the opposite of our situation. Choosing (b)
here would add a coordinator, a cross-node failure domain, and Slingshot traffic
to gain **nothing** (we never reduce across the sharded axis). **(a) is correct.**

### Quantified throughput and what breaks "perfect K×"

Ideal: K nodes × 56.6 cosmo/s (B=64, §0.1) = **K × 56.6 cosmo/s**. For N=10⁶
cosmologies at K=8 nodes: 10⁶ / (8×56.6) ≈ **2208 s ≈ 37 min of steady-state
GPU**, plus overheads. What erodes the K× (in order of impact):

1. **Per-task compile tax (the dominant erosion).** Each fresh process pays
   ~52–150 s of compile at B=16–48 (`perf_multigpu_results.json` `warm` column;
   CHANGELOG B=1→16 compile 95.8→223.7 s). At B=64 a single task's *steady-state*
   run for, say, 64×16 = 1024 cosmos (16 calls) is 16×72 ≈ 1150 s, so a ~120 s
   compile is ~10% tax **if each task does many calls.** If you naively size array
   tasks to one `call_batched` each (1024 cosmos/task), compile is amortized; if
   you size them to ~B cosmos each, compile **dominates** (>2×). → **The
   persistent compile cache (§2) is what makes the job array viable**; it turns
   the 120 s into a ~2–5 s cache-load after the first task on each node-type.
2. **Scheduler latency / backfill.** Premium queue (`nersc_train_gpu.slurm:4`)
   start times for 1-node GPU jobs are minutes, not the hours a K-node co-schedule
   would wait. Array throttling (`%` limit) keeps the account under its
   concurrent-GPU cap. This is wall-clock, not GPU-hours — it does not cost the
   allocation, only latency-to-result.
3. **Result aggregation.** Each task writes a `(slice, n_l)` Cl array or, better,
   a `(slice,)` χ² vector (§3 likelihood folding). 10⁶ × 799 float64 Cls = 6.4 GB
   if you keep all spectra; 10⁶ χ² scalars = 8 MB. → **Fold the likelihood inside
   the task** so aggregation is trivial (§3 / item 3).
4. **The CPU setup per task.** Re-costed in §4: ~0.4 s/param amortized, NOT a
   ceiling for B≤256.
5. **Tail / load imbalance across tasks.** If the grid is sliced contiguously and
   stiffness correlates with a parameter (e.g. high ω_b → stiffer), one task's
   worst-k could be systematically worse. Mitigate by **interleaving** the grid
   (task t gets cosmologies t, t+K, t+2K, …) so every task sees a representative
   stiffness mix. Cheap, zero-risk.

### The concrete harness (recommended)

A `scan_array.slurm` (job array) + a thin `scan_slice.py` driver. Pattern follows
the parent CLAUDE.md `salloc/srun` mechanics but in batch form:

```bash
#!/bin/bash
#SBATCH -A m3166_g
#SBATCH -C gpu
#SBATCH -q premium                       # premium, as in nersc_train_gpu.slurm:4
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks=1                        # ONE process ...
#SBATCH --gpus-per-task=4                 # ... that sees all 4 GPUs  (NOT 4x1!)
#SBATCH --cpus-per-task=64                # whole socket for the CPU setup vmap
#SBATCH -J abcmb_scan
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --array=0-249%32                  # 250 slices, <=32 concurrent (account cap)

module load conda && conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH      # MANDATORY: this checkout, not sibling
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# persistent compile cache shared by every array task on $SCRATCH (§2):
export JAX_COMPILATION_CACHE_DIR=/pscratch/sd/c/carag/ABCMB-k/.jax_cache
# NOTE: deliberately do NOT set CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
#       (that is the MCMC 1-gpu-per-rank pattern; here we want 1 proc / 4 gpus)

srun --ntasks=1 --gpus-per-task=4 --cpus-per-task=64 \
     python -u bench/scan_slice.py \
       --grid grid.npz --task $SLURM_ARRAY_TASK_ID --ntasks 250 \
       --batch 64 --out chains/scan/slice_${SLURM_ARRAY_TASK_ID}.npz
```

`scan_slice.py` (sketch — interleaved slice, fixed padded B, fold to χ²):

```python
import os; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from abcmb.main import Model
# grid: (N, n_param) ; interleave so each task sees a representative stiffness mix
g = np.load(args.grid)["params"]; mine = g[args.task::args.ntasks]
model = Model(... l_max=800, lensing=False, output_Pk=True ...)   # build ONCE
B = args.batch
chi2 = []
for i in range(0, len(mine), B):
    block = list(mine[i:i+B])
    # pad to fixed B so every call hits the SAME cached HLO (cache + no recompile)
    pad = B - len(block); block += [block[-1]]*pad
    out = model.call_batched([row_to_dict(r) for r in block], shard=True)
    chi2.append(np.asarray(loglike(out))[:B-pad] if pad else np.asarray(loglike(out)))
np.savez(args.out, chi2=np.concatenate(chi2), idx=np.arange(args.task, len(g), args.ntasks))
```

Effort: **LOW** (no theory touch; a slurm file + a ~50-line driver). Risk: LOW.
Throughput effect: **near-linear in K** once §2 caps compile — the headline lever.

---

## 2. Compile-cost amortization (item 2)

### Mechanism

Set the persistent on-disk compilation cache at import (nothing in-repo sets it
today; `ideas_singlegpu.md §5.1`):

```python
jax.config.update("jax_compilation_cache_dir",
                  "/pscratch/sd/c/carag/ABCMB-k/.jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
```

(Equivalently `JAX_COMPILATION_CACHE_DIR` env, as in the slurm above.)

### Does it survive across nodes/jobs?

**Yes, with one caveat.** The cache key is a hash of the **HLO + XLA flags +
target (compute capability + jaxlib/CUDA versions)**. Perlmutter's GPU nodes are
**homogeneous A100s** (single `-C gpu` partition), so an HLO compiled on one node
is reused on every other node — the cache is a plain directory on `$SCRATCH`,
which is a **global Lustre filesystem visible from all compute nodes**. So:

- Task 0 (anywhere) compiles, writes ~N_stage entries to `.jax_cache`.
- Every subsequent task, **on any node**, finds the entries by hash and does a
  **deserialize-from-disk** (~1–5 s for the whole pipeline) instead of a
  70–220 s XLA compile.

Caveats / hidden floors:
1. **Lustre metadata storms.** If 32 array tasks start *simultaneously* and all
   miss-then-write the same keys, you get a thundering-herd of concurrent writes
   to one cache dir — Lustre small-file metadata contention, and possible
   duplicate compiles (the cache is not locked across processes by default).
   Mitigate: **warm the cache with ONE priming task first** (array dependency:
   submit task 0 alone, then `--dependency=afterok` the rest), so the herd hits a
   populated cache and only reads. Cheap, removes the storm.
2. **Cache key includes B (the shape).** Every batched leaf bakes B into its
   shape (`ideas_singlegpu.md §5.2`), so a ragged final batch recompiles every
   jitted stage. → **Pad every `call_batched` to a fixed B** (the driver sketch
   above does this) so the whole scan reuses ONE set of cached HLOs. This is the
   non-negotiable companion to the cache.
3. **Cache key includes XLA flags.** If §6's XLA flags differ between the priming
   task and the scan tasks, they miss the cache. Keep flags identical across all
   tasks (put them in the slurm `export` block, not ad-hoc).

### Quantified compile tax vs steady-state

- **Without cache, one `call_batched`/task (B=64):** compile ~120 s, run ~72 s →
  compile is **62%** of the task. Catastrophic for a job array.
- **Without cache, ~16 calls/task (1024 cosmos):** compile 120 s, run 1150 s →
  compile **9%**. Tolerable but wasteful (paid K=250× across the array = 8.3
  GPU-hours of pure recompile).
- **With cache + fixed B:** first task pays 120 s; all others pay ~2–5 s
  deserialize. Across 250 tasks the compile tax drops from ~8.3 GPU-hours to
  ~0.3 GPU-hours. **~25× reduction in compile waste.**

Effort: **TRIVIAL** (3 config lines + the pad, both already specced). Risk:
**NONE** (cold-start only; wrong cache just recompiles). Prob success: ~0.95
(the one risk is the Lustre storm, mitigated by priming).

---

## 3. Throughput vs latency reframing (item 3)

**Per-CALL latency is the WRONG headline metric for a scan.** The right metric is
**total GPU-hours for N cosmologies** = N × (per-param s) / (GPUs per node × 3600)
× (node-count is wall-clock, not GPU-hour). Equivalently: **cosmologies per
GPU-second**, which is `(B/n_dev) / total_call_time` and is what we should
maximize.

### Does bigger B raise throughput, or just hide fixed costs?

**It genuinely raises throughput up to a memory wall — it is NOT just hiding fixed
costs.** Evidence: the per-param curve (3.24→1.82→1.34→1.13 over B=16→64) is the
sum of (i) **amortizing** fixed costs (compile excluded post-warm; the eager
dispatch already killed; setup amortizing 2.0→0.4) and (ii) the **solve**, which
fills the GPU better at larger B. The CHANGELOG is explicit that the solve "amortizes
12→2 s/param" and that "sharding quarters per-device memory, so larger B fits."
Two distinct gains stack:
- **Single-A100 saturation:** `ideas_singlegpu.md §3.1`/`notes_strategy §2(iii)`
  report one A100 is "70–78% full" at B=64 single-GPU and rematerializing — i.e.
  one GPU is near its occupancy knee at B≈64.
- **4-GPU sharding** quarters per-device memory, so the *sharded* B=64 puts only
  16/device → each device is at the *B=16 occupancy*, far from saturated → the
  4-GPU per-param keeps falling well past B=64.

### Where does the per-param curve flatten, and is that memory- or solve-bound?

Extrapolating the 4-GPU points (per-param 3.24, 1.82, 1.34, 1.13 at B=16,32,48,64;
per-device load 4,8,12,16): the residual after the worst-k floor (~0.8 s/param,
`ideas_multigpu.md §1.4`) is shrinking ~as 1/B_local. Fit suggests:

| B (4-GPU) | per-device | projected per-param | projected cosmo/s (4 GPU) |
|-----------|-----------:|--------------------:|--------------------------:|
| 64        | 16         | 1.13 (measured)     | 56.6 |
| 128       | 32         | ~0.95–1.0           | ~128–135 |
| 256       | 64         | ~0.85–0.90 (≈ worst-k floor) | ~285–300 |
| 512       | 128        | ~0.85 (flat) **+ OOM risk** | memory-bound |

**The flattening is SOLVE-bound (worst-k floor ~0.8 s), reached around B≈256
(64/device).** Beyond that, per-param barely improves and **per-device memory
becomes the wall** — single-GPU B=64 already warned 28–31 GiB / 40 GiB
(`battle_royale_brief` line 54). Sharded, B=256 puts 64/device, i.e. the
single-GPU B=64 memory regime *per device* → that is the practical ceiling at
`k_chunk=100`. To push B higher you must drop `k_chunk_size` (more, smaller
chunks → less in-flight state), which slightly erodes the per-param gain
(`ideas_singlegpu.md §5` B=256 → k_chunk≈50).

**Sweet spot for max cosmologies/GPU-hour: B ≈ 256 on the 4-GPU node** (64/device,
~0.85–0.90 s/param, ~290 cosmo/s/node), i.e. **5× the throughput of the current
B=64 measurement, on the SAME hardware**, purely by raising B until the worst-k
floor and the per-device memory wall meet. This is the cheapest large win and it
needs **zero code change** — just call `call_batched` with bigger lists (and
possibly tune `k_chunk_size` down at the top end). **This is my single most
under-exploited lever and it is free.**

Hidden floor: the worst-k tax (max/median 1.12 over B, `flipped_multigpu_summary`)
means we never beat ~0.8 s/param by scaling B alone — past B≈256 the only further
throughput is **more nodes** (§1), not bigger B.

---

## 4. The CPU setup floor at scale (item 4)

### Re-cost against the CURRENT code (the stale memo is wrong here)

`probe_setup.log` (§0.2): HyRex jitted CPU = **0.062 s/cosmo**, and the "7 s" was
eager `get_BG` which `_build_bgs_batched` already jitted away. The committed
batched setup (`main.py:332-379`) does: one vmapped `_pre_recomb_batched` (GPU),
**one** `jax.device_put(...,cpu)` of the stacked recomb inputs (`main.py:364-365`),
**one** vmapped `RecModel` on CPU (`main.py:366`), **one** re-shard/transfer back
(`main.py:369-375`), one vmapped `_get_BG_batched` (GPU, `main.py:378`). So it is
**O(1) jits + O(1) transfers**, not O(B). CHANGELOG: setup amortizes to ~0.4
s/param at large B.

**Is the CPU stage the throughput ceiling at large B / many nodes? NO, for B≤256.**
At B=256, HyRex CPU = 256 × 62 ms = ~16 s **if it were serial**, but it is vmapped
→ the Eigen threadpool over 64 cores parallelizes it; even pessimistically ~0.06
s/param it is a *small* fraction of the 0.85 s/param solve. The CPU stage becomes
co-bottleneck only if (a) you starve it of cores (don't — the slurm asks
`--cpus-per-task=64`) or (b) B grows so large the *vmapped CPU solve's* lockstep
worst-lane dominates — not a concern at B≤256.

### The two GENUINE residual serial costs (the real items to watch)

1. **The eager `add_derived_parameters` python loop** (`main.py:252`):
   `full_ps = [self.add_derived_parameters(p) for p in params_list_run]`. This is
   a Python `for` over B, each call running species `rho` loops + bbn branching
   (`main.py:562-806`). The CLAUDE.md calls it "~ms/cosmo" — at B=256 that is
   ~hundreds of ms, **on the host, serial, NOT overlapped with the GPU**. It does
   not vmap (it has `sys.exit`, species loops, table interp). At small per-call B
   it is noise; across a 10⁶-cosmo scan it is ~minutes of pure host time. **Lever:
   it is trivially parallelizable across the host cores with a
   `multiprocessing.Pool` or just `np`-vectorize the ΛCDM-default branch** (the
   common case has no LINX). Effort LOW, gain SMALL (only matters at the very
   largest B). Watch it, don't prioritize it.
2. **The two HyRex CPU↔GPU device transfers** (`main.py:364-365`, `369-375`).
   These are `(B, n_lna, …)` arrays: recomb inputs in, `(xe, Tm, …)` out. At B=256
   over PCIe/NVLink-to-host these are ~tens of MB each way — sub-100 ms, batched
   (one transfer, not B). **Not a ceiling.** The CLAUDE.md try/except re-transfer
   pattern (`main.py:372-375`) is preserved correctly.

### Async / pipeline overlap of CPU-setup(i+1) with GPU-solve(i) — worth it NOW?

**No — DESCOPE it.** `ideas_multigpu.md §2.4` proposed a threaded pipeline to hide
a *7 s* CPU floor. That floor was a measurement artifact (eager get_BG); the real
CPU side (HyRex 0.062 s/param vmapped + transfers) is **already <10% of the solve**.
Building a `threading.Thread`/`queue.Queue` producer-consumer to overlap a
sub-0.1 s/param stage behind a 0.85 s/param solve buys at most ~10% and adds real
complexity (GIL interplay, JAX async dispatch ordering, error propagation across
threads). **Negative ROI now.** The one piece of it worth keeping is the
already-done "batch the transfers" (stack once, transfer once) — which the current
code does. Revisit overlap ONLY if a future precision/solver win drops the solve
below ~0.3 s/param and the CPU stage becomes comparable.

---

## 5. Sharding hygiene (item 5)

### Is `P('batch')` GSPMD the right primitive at 8/16/many GPUs?

**On ONE node, yes — keep it.** The pipeline has no cross-cosmology reduction
(verified §1), so GSPMD auto-partition over the B axis is exactly right and emits
**zero collectives** — the `NVLink is not used` warnings (`multigpu_run.log:1-8`)
are therefore harmless (we never touch the interconnect). At 4 GPUs/node this is
proven (`perf_multigpu_results.json`). Perlmutter is 4 A100/node, so "8/16 GPUs"
**only exists across nodes** — and across nodes you do NOT want one global mesh
(§1, the job array wins). So the answer is: **`P('batch')` on the single-node
4-GPU mesh, replicated across nodes by the job array.** No `shard_map`, no `pmap`
(both add machinery for collectives we don't have).

### Any collective sneaking in?

I checked the three sharded stages:
- `_pre_recomb_batched` / `_get_BG_batched` (`main.py:316-330`): pure vmapped
  per-cosmology construction, no reduction.
- `full_evolution_batched`: chunk concat is along **k** (unsharded), local to each
  device (`ideas_multigpu.md §1.3`, verified).
- `get_Cl_batched` / `Pk_lin_batched` (`spectrum.py:617-650`): `jax.vmap`, no
  `psum`/`all_gather`.

**No collective on the B-sharded path.** The ONE place GSPMD *could* inject an
all-gather is the **final output**: if the host pulls all `(B, n_l)` Cls back, the
B-sharded array is gathered. That gather is tiny and one-shot per call — and
**eliminable** by folding the likelihood per-shard (§3) so only a `(B,)` χ² vector
(or even a per-shard scalar) crosses the wire.

### Does the HyRex CPU gather/re-shard round-trip become a bottleneck as the mesh grows?

**This is the one real sharding-hygiene wart.** `_build_bgs_batched`
(`main.py:363-375`) does:
1. `jax.device_put(pre_BG_batch.recomb_inputs, cpu)` — **gathers the B-sharded
   array off the 4 GPUs onto one host** (an implicit all-gather to host).
2. vmapped HyRex on CPU.
3. `shardfn(recomb_batch)` — **re-scatters** the result back across the 4 GPUs
   (`main.py:370`).

On ONE node this is a host round-trip of `(B,…)` arrays — fine at B≤256 (§4). But
it means **every node-local mesh does a full gather→CPU→scatter mid-pipeline.** If
a future design ever DID go multi-node-single-mesh (§1b, which I'm recommending
*against*), this gather would become a cross-node all-gather to a single host —
a true bottleneck. **This is a second, independent reason the job-array (§1a)
beats the global mesh (§1b):** the CPU round-trip stays node-local and cheap.
Within the job array it's a non-issue. No change needed; just don't let it go
multi-node.

---

## 6. Other systems-level levers (item 6)

- **XLA flags (LOW effort, SMALL–MED gain, must be in the cache key).**
  `--xla_gpu_enable_latency_hiding_scheduler=true` and
  `--xla_gpu_enable_async_collectives` are irrelevant (no collectives). The one
  worth trying: `--xla_gpu_enable_command_buffer=` / CUDA-graph capture (reduces
  per-launch kernel dispatch overhead, which matters for the many-small-kernel
  Kvaerno5 inner loop). Gate: set it in the slurm export block so it's part of the
  compile-cache key (§2); measure per-param at fixed B. Risk LOW. Prob the gain is
  real: ~0.4.
- **`XLA_PYTHON_CLIENT_PREALLOCATE=false` — keep it (already set everywhere:
  `nersc_train_gpu.slurm:24`, parent CLAUDE.md).** With 4 GPUs in one process,
  preallocate=true would have each grab 90% and the on-host CPU JAX (HyRex) +
  multi-GPU could collide. With `false`, the allocator grows on demand. Hidden
  floor: `false` can *fragment* at the B=64 memory edge and OOM where `true` would
  have fit (preallocate avoids fragmentation). **If B=128/256 OOMs, try
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` + preallocate=true** as an alternative —
  but only after measuring, since it changes the multi-GPU-in-one-process
  behavior. MED risk, situational.
- **NCCL: N/A** — no collectives, no NCCL traffic. `MPICH_GPU_SUPPORT_ENABLED=1`
  (`nersc_train_gpu.slurm:25`) is for the MCMC MPI path; the scan's single-process
  4-GPU mode doesn't use it. Harmless to leave set; irrelevant.
- **MPS / multiple processes per GPU: SKIP.** MPS (CUDA Multi-Process Service)
  helps when many *small* processes underutilize a GPU. Our solve already
  saturates one A100 at B≈64/device (§3); packing more processes onto a GPU would
  contend for the same SMs and memory and **OOM** (we're at 28–31/40 GiB). The
  right "more work per GPU" lever is bigger B (§3), not more processes. Negative
  ROI.
- **Single process / 4 GPUs vs 4 processes / 1 GPU each on one node:** the
  single-process GSPMD mode is correct (§0.3). The 4-proc mode would force
  `jax.distributed` for a node we can do with one process — strictly worse here.
- **`donate_argnums` on the batched stages** (`ideas_singlegpu.md §5.4`): donate
  the big `(N_k,B,500,N_y)` PT buffer so XLA reuses it for the Cls — eases the
  B=128/256 memory edge, enabling the §3 sweet spot. Effort LOW, risk LOW, MED
  confidence. This is the cheapest enabler for pushing B past 64.
- **`--cpus-per-task=64` (whole socket)** for the scan tasks vs the MCMC's `-c 32`
  (`nersc_train_gpu.slurm:175`): the scan's one process owns the whole node, so
  give the vmapped-HyRex CPU stage all 64 cores. Free.

---

## 7. Ranked shortlist

Ranked by (throughput gain × prob success ÷ (risk × effort)). All are
**systems-only** — none touch the ODE math (that's the other two round2 memos).

| # | Lever | Throughput effect | Risk | Effort | Prob | Score |
|---|-------|-------------------|------|--------|------|-------|
| 1 | **Raise B to ~256 on the 4-GPU node** (§3) | **~5× node throughput** (56→~290 cosmo/s) | LOW (OOM at top) | ~0 (bigger list; maybe k_chunk↓) | 0.85 | ★★★★★ |
| 2 | **Job array of independent 1-node 4-GPU tasks** (§1a) | **~K× across K nodes** (the scale-out) | LOW | LOW (slurm + 50-line driver) | 0.9 | ★★★★★ |
| 3 | **Persistent compile cache + fixed-B padding** (§2) | removes ~25× compile waste; makes #2 viable | NONE | TRIVIAL | 0.95 | ★★★★★ |
| 4 | **Fold likelihood per-shard → emit (B,) χ²** (§3/§5) | kills aggregation + the only output all-gather | LOW | LOW | 0.9 | ★★★★ |
| 5 | **donate_argnums on batched stages** (§6) | enables B≥128 (eases #1's memory wall) | LOW | LOW | 0.6 | ★★★ |

Explicitly **descoped** (negative/zero ROI vs the brief's now-stale priors):
async CPU/GPU pipeline overlap (§4 — the 7 s floor was a myth), MPS (§6), and
`jax.distributed` multi-node single-mesh (§1b — wrong tool for embarrassingly
parallel HTC).

---

## 8. My #1 bet, and the cheapest measurement to de-risk it

**#1 bet: the combination "raise B to ~256 + job array + compile cache" — and
within that, the FREE part (raise B) is the single highest-leverage move.** The
4-GPU per-param curve is still falling steeply at B=64 (1.13 s) and the GPUs are
not saturated when sharded (16/device = B=16 occupancy). Pushing B to the
worst-k-floor knee (~256, 64/device) should hit ~0.85–0.9 s/param ≈ **~290
cosmo/s/node — 5× the current measured throughput on the exact same hardware, with
zero code change.** Replicating that node across a job array with a shared compile
cache then scales it ~linearly in node count. This is strictly cheaper and
higher-leverage than any solver/precision change because it spends only hardware
we already have access to and code that already exists.

**Cheapest de-risking measurement (1 GPU allocation, ~30 min):**
Extend `bench/perf_multigpu.py` `--bvals` to `64,96,128,192,256` on a single
4-GPU node (`salloc --no-shell --nodes=1 --gpus=4 ...`; `srun --ntasks=1
--gpus-per-task=4 --cpus-per-task=64`). Two things to read off:
1. **The per-param curve past B=64** — confirm it bends toward ~0.85 (solve floor)
   and find the exact B where it flattens. That number sets the production B.
2. **The OOM point** — the first B that throws (or where you must drop
   `k_chunk_size`). That sets the memory wall and tells you whether `donate` (#5)
   is needed.

That ONE sweep simultaneously validates lever #1 (the free 5×), bounds lever #3's
B-padding target, and tells you whether #5 (donate) is on the critical path —
i.e. it de-risks three of the top five at once.

**Second allocation (optional, parallel):** a 2-task job-array smoke test
(`--array=0-1`) writing slice files, with `JAX_COMPILATION_CACHE_DIR` set, to
confirm (a) task 1 deserializes the cache instead of recompiling (read the `warm`
time: should drop from ~120 s to a few s) and (b) the slice/aggregate plumbing
round-trips. That validates levers #2 and #3 end-to-end before committing the full
array.

# Multi-GPU end-to-end design for `Model.call_batched` — road to ~1 s/param

Static-analysis design memo (no Python/GPU run for this document). Builds on the
measured PE-only multi-GPU spike (`bench/flipped_spike_multigpu.py`,
`bench/multigpu_run.log`, `bench/flipped_multigpu_summary.txt`), the **measured
end-to-end per-stage profile** (`bench/profile_stages.log`, B=1/8/16 with
`block_until_ready` fences), the strategy/solve/spectrum notes, and direct reads
of `abcmb/{main,perturbations,hyrex/hyrex,hyrex/array_with_padding}.py`.

Everything below distinguishes **measured** (cited to a log/line) from **ESTIMATE**.

---

## 0. The hard numbers we are designing against

### 0.1 End-to-end per-stage, single GPU, post-compile (`bench/profile_stages.log`)

| B  | total/param | setup/param | perturb/param | spec_Cl/param | spec_Pk/param |
|----|------------:|------------:|--------------:|--------------:|--------------:|
| 1  | 22.459 s    | 7.094       | 12.161        | 3.180         | 0.017 |
| 8  | 12.624 s    | 6.866       | 2.558         | 3.181         | 0.013 |
| 16 | 12.412 s    | 6.962       | 1.968         | 3.461         | 0.014 |

Read off the **per-stage scaling with B** directly from these three measured rows
(this is the load-bearing evidence — not extrapolation):

- **setup ≈ 6.9–7.1 s/param, FLAT in B.** This is the killer. It does **not**
  amortize. setup *total* at B=16 is 111.4 s — it is 56% of the whole run
  (`profile_stages.log:38`). This is the per-cosmology serial CPU floor (HyRex +
  two device transfers + eager `add_derived_parameters`), confirmed flat.
- **perturb amortizes hard:** 12.16 → 2.56 → 1.97 s/param (B=1→8→16). The 4-GPU
  spike pushes this further (§0.2).
- **spec_Cl ≈ 3.2–3.5 s/param, FLAT in B** — it is a Python `for i in range(B)`
  loop (`spectrum.py:642`), so it pays full per-element JIT dispatch B times and
  never amortizes. At B=64 this projects to ~3.3 s/param ≈ **211 s total**.
- **spec_Pk negligible** (~0.014 s/param), also a python loop but cheap (no BG).
- **stack ≈ 0 s** (the `jnp.stack` of pytrees).

**Crucial correction to the "7 s is GPU compile" hypothesis:** setup stays ~7
s/param at B=16, i.e. it does NOT fall toward zero after the first cosmology. So
the eager GPU `get_BG_pre_recomb`/`get_BG` calls (`@eqx.filter_jit`, identical
scalar param shapes across cosmologies) **do hit the XLA compile cache** — they
are not the cost. The ~7 s is genuine **per-cosmology serial CPU work**: HyRex on
CPU (`eqx.filter_jit(self.RecModel, backend='cpu')`, `main.py:256`) plus the two
`jax.device_put` round-trips (`main.py:254,259`). This is the single most
important fact in the memo: **4 GPUs do nothing for a 7 s/param serial CPU floor.**

### 0.2 PE-only multi-GPU spike, 4× A100, B sharded over batch axis (`flipped_multigpu_summary.txt`)

| B  | compile (s) | total (s) | per-param (s) |
|----|------------:|----------:|--------------:|
| 4  | 31.92       | 8.820     | 2.2051        |
| 16 | 38.19       | 20.144    | 1.2590        |
| 64 | 73.18       | 59.229    | **0.9254**    |

Step counts at B=64 (sharded): min=43, med=383, max=1595; **worst-k max/median
over B = 1.12** (`flipped_multigpu_summary.txt:10-11`). The 0.93 s/param is
PE-ONLY: the spike builds BGs in an explicitly **untimed** loop
(`flipped_spike_multigpu.py:175-181`) and never runs the spectrum.

### 0.3 The honest end-to-end picture today (single GPU, projected to B=64)

Add the measured flat stages to the spike's PE number:

```
setup 7.0  +  perturb 0.93 (4-GPU) + spec_Cl 3.3 (loop) + spec_Pk 0.01  ≈ 11.2 s/param
```

Even with 4 GPUs on the solve, **today's pipeline is ~11 s/param**, dominated by
two stages the spike skipped (setup 7.0, spectrum 3.3). The 4-GPU PE win is real
but invisible until those two are fixed. **That is the whole problem this memo
addresses.**

---

## 1. Sharding design for the perturbation solve

### 1.1 What the spike did, and why it is the right primitive (keep it)

The spike uses **NOT** `pmap` and **NOT** `shard_map` — it uses the modern GSPMD
auto-partition path: `jax.sharding.Mesh` + `NamedSharding` + plain
`eqx.filter_jit` (`flipped_spike_multigpu.py:163-166, 198-227`):

```python
mesh = Mesh(np.array(devs), axis_names=('batch',))          # :163
batch_sharding = NamedSharding(mesh, P('batch'))            # shard the B axis
replicated     = NamedSharding(mesh, P())                   # scalars replicated
# device_put every ndim>=1 leaf on batch_sharding, ndim-0 on replicated, then:
res = full_evolution_dvmap(BG_b, p_b)                       # ordinary filter_jit
```

XLA's GSPMD sees `P('batch')` on the inputs and auto-partitions the whole vmap'd
program over the 4 devices with **zero collectives**. This is correct and should
be kept, because:

- **The PE solve is embarrassingly parallel over B.** Every `(k, b)` diffrax
  solve is independent; there is no reduction anywhere in
  `full_evolution_batched`. With `P('batch')` in and `P('batch')` out, GSPMD
  partitions cleanly with **no inter-device traffic**. The
  `NVLink is not used` warnings (`multigpu_run.log:1-8`) are therefore harmless —
  we never touch the interconnect.
- `pmap` is legacy/rigid (bakes the device axis into the program, forces
  leading-axis = device count, fights `eqx.filter_jit`). The JAX team steers
  users to `jit` + `NamedSharding`
  ([Distributed arrays & automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html),
  [Introduction to sharded computation](https://docs.jax.dev/en/latest/sharded-computation.html)).
- `shard_map` ([manual, collective-aware](https://docs.jax.dev/en/latest/notebooks/shard_map.html))
  is the tool **only** when you want each device to run its own un-vmapped program
  with explicit `psum`/`all_gather`. We have no collective, and the per-device
  program is already a clean vmap, so `shard_map` buys nothing here and would cost
  us GSPMD's ability to reason about the k-chunk loop (§1.3). `shard_map`'s
  in_specs/out_specs express slicing/concatenation across mesh axes
  ([jax.shard_map](https://docs.jax.dev/en/latest/_autosummary/jax.shard_map.html));
  we'd just be hand-reimplementing what GSPMD does for free. **Skip it.**

JAX API refs: [`jax.sharding` (Mesh / NamedSharding / PartitionSpec)](https://docs.jax.dev/en/latest/jax.sharding.html).

### 1.2 Recommended wrapper around `full_evolution_batched`

`call_batched` (`main.py:187-238`) builds BGs in a python loop, strips
`kappa_func` via `strip_bg_kappa`, stacks, and calls
`PE.full_evolution_batched((BG_batch_stripped, params_batch))`. The only change
for multi-GPU is: **shard the stacked inputs on the B axis before that call.**

```python
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

def _make_batch_mesh():
    devs = jax.devices('gpu')
    mesh = Mesh(np.asarray(devs), axis_names=('batch',))
    return mesh, NamedSharding(mesh, P('batch')), NamedSharding(mesh, P())

def _shard_on_batch(pytree, batch_sh, repl_sh):
    def per_leaf(x):
        if eqx.is_array(x):                       # robust leaf test (vs spike's hasattr)
            return jax.device_put(x, batch_sh if x.ndim >= 1 else repl_sh)
        return x                                  # static / None untouched
    return jax.tree.map(per_leaf, pytree)

# inside call_batched, after stacking:
mesh, batch_sh, repl_sh = _make_batch_mesh()
assert B % mesh.devices.size == 0          # PAD B to a multiple of n_devices first
BG_s = _shard_on_batch(BG_batch_stripped, batch_sh, repl_sh)
p_s  = _shard_on_batch(params_batch,      batch_sh, repl_sh)
PT   = self.PE.full_evolution_batched((BG_s, p_s))   # GSPMD auto-partitions
```

- **Pad B to a multiple of `n_devices`=4.** `NamedSharding(P('batch'))` requires
  the sharded dim divisible by the mesh size; the spike dodges this by only
  testing B∈{4,16,64} (`flipped_spike_multigpu.py:154-158`). Production must pad
  (replicate the last cosmology, mask the output) — small but a real correctness
  requirement (and avoids recompiling for ragged final batches).
- Use `eqx.is_array` as the leaf predicate; the spike's
  `isinstance(x, jax.Array) or hasattr(x,'shape')` (`:202-207`) is fragile
  (catches numpy scalars).
- GSPMD propagates input sharding to the output, so `PT` comes back **already
  B-sharded** — the spectrum stage (§3) consumes it without a re-`device_put`.

### 1.3 How the internal k-chunk python loop interacts with sharding (VERIFIED)

`_compute_modes_batched` (`perturbations.py:150-192`) loops
`for i in range(0, n_k, k_chunk_size)` (`:186`) over k-chunks, calls the JIT'd
`_evolve_chunk` (`:126`, `vmap(per_k=k_chunk) × vmap(B)` around `evolution_one_k`,
`:142-148`) per chunk, then `jnp.concatenate(chunks, axis=0)` (`:191`) — **and
axis 0 is the k axis** (chunk output is `(K_CHUNK, B, Nlna, Ny)`, concatenated to
`(N_k, B, Nlna, Ny)`, then `.transpose(1,3,2,0)` to `(B, Ny, Nlna, N_k)`,
`:190-192`). The python loop is **trace-time unrolled** into one HLO program with
ceil(571/100)=6 chunk subgraphs.

Under `jit` + B-axis `NamedSharding`, **every chunk subgraph is itself B-axis
partitioned.** Each device:
1. holds its B/4 slice of the stacked `(BG_batch, params_batch)`,
2. runs the full 6-chunk unrolled program on its local slice (identical loop on
   all devices; GSPMD replicates structure, shards data),
3. per chunk runs `vmap(k=100) × vmap(B/4=16 at B=64)` of `evolution_one_k`,
4. **concatenates its 6 chunk-results along k — a LOCAL concat** (k is unsharded),
   no cross-device communication.

So **yes — each device runs its own copy of the chunk loop on its own B-shard,
with zero inter-chunk device traffic.** This is exactly why the spike's
0.93 s/param at B=64 holds. (The spike's `full_evolution_dvmap` was unchunked, so
this confirms the *production* chunked path will shard equally cleanly.)

Memory: per-device in-flight state at B=64 is `B_local(16) × k_chunk(100)` worth
of Kvaerno5 state vs the single-GPU `64 × 100` that warned at 28–31 GiB
(`design_memo.md §1`). Sharding **quarters per-device memory** — the spike comment
notes B/4=16 is "comfortably under budget" (`flipped_spike_multigpu.py:9-11`).
This also means the sharded path could **raise `k_chunk_size`** (e.g. 200 → 3
unrolled chunks instead of 6), trimming fixed per-chunk overhead — a secondary
knob worth a sweep.

### 1.4 The load-balance ceiling (worst-k under sharding)

`notes_solve.md §A`: a vmap'd diffrax adaptive solve runs all lanes **in
lockstep** at the smallest accepted step (confirmed via the diffrax FAQ and issue
#281, cited there). So a device's wall-clock is set by the worst `(k,b)` solve in
its shard. The measured worst-k max/median over B = **1.12** at B=64 means the
worst k mode is worst for *every* cosmology, so each device's shard already
contains a near-worst case — we lose only ~12% to imbalance. **Good for
throughput, but it means sharding B can never beat one worst-k solve's latency.**
The spike's curve (B=4→2.21, B=16→1.26, B=64→0.93; gaps 0.95, 0.33) is flattening
toward an asymptote ~0.8 s/param — **we are near the floor**. Below ~0.8 s/param
needs k-axis load-balancing (stiffness-homogeneous chunking, `notes_solve.md §A1`),
which is out of scope here.

---

## 2. The CPU setup stage at scale — the real risk

This is where the spike's 0.93 s headline is a mirage and ~1 s end-to-end is won
or lost. **Measured: setup is 7.0 s/param, FLAT in B** (§0.1). At B=64 that is
**~448 s of pure serial work** — bigger than everything else combined, and 4 GPUs
do not touch it.

### 2.1 What's actually in the 7 s (VERIFIED `main.py:240-265`, `hyrex/hyrex.py`)

`_build_one_bg`, per cosmology:
1. `_to_float` + `add_derived_parameters` — eager Python, cheap (~ms).
2. `get_BG_pre_recomb` (GPU, `@eqx.filter_jit`) — compile-cached after #1 cosmo.
3. `device_put(recomb_inputs, cpu)` + `device_put(params, cpu)` — GPU→CPU.
4. **HyRex** `eqx.filter_jit(self.RecModel, backend='cpu')((...))` — a sequential
   CPU recombination solve. `recomb_model.get_history` (`hyrex.py:172-207`) runs
   `helium_model` then `hydrogen_model`, each a diffrax `diffeqsolve` on fixed
   `SaveAt(ts=...)` grids, plus one `eqx.internal.while_loop(kind="checkpointed",
   max_steps=...)` in hydrogen with `array_with_padding` for fixed-shape output.
5. `device_put(recomb_output, gpu)` — CPU→GPU.
6. `get_BG` (GPU, compile-cached).

Since the GPU stages compile-cache (confirmed by setup staying flat at 7 s, not
falling), the 7 s is HyRex + the two device round-trips + eager python, repeated B
times **serially with no amortization**. This is the binding floor.

### 2.2 Option (a): vmap HyRex over B on CPU — VIABLE (verified)

`recomb_model.__call__` is a **plain method** (`hyrex.py:146`), jit applied only
at the call site — so a `jax.vmap(self.RecModel)` is clean to wrap. The
vmap-blocking question is whether `array_with_padding` survives vmap over B. I
read it (`array_with_padding.py`): `__init__` does
`jnp.argmax(jnp.isinf(arr))`, `arr[self.lastnum]` (gather by traced int), and
`concat` does `lax.dynamic_update_slice(z, y, [x.size - padding_size])` with a
**traced** offset into a **statically-sized** buffer (`z = jnp.ones(x.size +
y.size)*inf`). **All of these are vmap-compatible** (vmap lifts gather and
dynamic_update_slice to batched variants; the buffer size is static and identical
across cosmologies). The one `checkpointed while_loop` (hydrogen) becomes a
masked-while running to the worst lane's `max_steps` (already a static cap). The
`idx_4He_equil` index is computed in `recomb_model.__init__` from the static
`lna_axis_full` (`hyrex.py:144`) — **batch-independent**, no per-cosmology static
shape. **Conclusion: HyRex IS vmap-able over B.** This is the cleanest CPU fix.

```python
cpu = jax.devices('cpu')[0]
recomb_in_batch  = jax.device_put(stack_pytrees(recomb_inputs_list), cpu)  # (B,...)
params_batch_cpu = jax.device_put(stack_pytrees(full_ps),            cpu)
rec_batched = eqx.filter_jit(jax.vmap(self.RecModel), backend='cpu')(
    (recomb_in_batch, params_batch_cpu))
```

**Projected gain (ESTIMATE):** vmap replaces B serial 0.4 s solves + B× python
dispatch + B× compile-cache lookups + 2B device transfers with **one** batched
CPU kernel and **one** pair of (B,...) transfers. On a CPU, vmap parallelizes via
the Eigen threadpool *and* eliminates per-element dispatch/transfer overhead; the
dispatch/transfer elimination is the bigger lever (per `notes_solve.md
E-floor-1`). Realistic: setup **7.0 → ~0.3–1.0 s/param** (the masked-while still
pays the worst lane, and CPU vmap won't 64× the flops, but the B× overheads
collapse). Even at the pessimistic end this is a 7–20× cut to the binding floor.
**Risk: LOW** (math identical; masked-while is standard). Verify the `lastnum`/
`lastval` int gathers don't trip an int-leaf issue under the existing `_to_float`
casts (they're consumed before the GPU AD path, so likely fine).

### 2.3 Option (b): multiprocess HyRex across the 64-core CPU (fallback)

If vmap surprises us, fall back to data parallelism: a persistent
`ProcessPoolExecutor(max_workers≈32)` each solving its B/N slice serially. Each
worker forces `JAX_PLATFORM_NAME=cpu`, pins `OMP_NUM_THREADS`/Eigen so they don't
oversubscribe the socket. Avoids the GIL and the vmap question entirely.

```python
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=32) as ex:           # persistent, reuse across grid
    recomb_outputs = list(ex.map(_hyrex_worker, per_cosmo_args))
```

**Projected (ESTIMATE):** 64 jobs / 32 workers ≈ 2 waves × 0.4 s + fork/pickle
overhead ≈ ~1–2 s total at B=64 → ~0.02–0.03 s/param. Caveat: forking JAX has
nontrivial startup (each re-imports JAX); only worth it if the pool is **reused
across the whole scan**, not one-shot. Keep the GPU `get_BG` on rank 0 — don't let
32 workers contend on one GPU. Best as the fallback if (a) hits a snag.

### 2.4 Option (c): pipeline CPU HyRex with GPU perturb of a previous batch (structural)

The architecturally correct answer: **overlap**. While the GPU runs the PE solve
for micro-batch *i*, a CPU thread builds Backgrounds (HyRex) for micro-batch
*i+1*. HyRex is CPU + diffrax C++ and releases the GIL, so a `threading.Thread`
feeding a `queue.Queue` of ready stacked-BGs works; the main thread pulls and
launches the sharded GPU solve (JAX async dispatch returns immediately).

```
HyRex(mb0) ─► [GPU PE(mb0)] [GPU PE(mb1)] [GPU PE(mb2)] ...
              HyRex(mb1)     HyRex(mb2)     ...           (CPU thread, concurrent)
```

**Projected effect:** even keeping HyRex *serial* (~6.4 s for a 16-cosmology
micro-batch) it fully hides behind the GPU PE solve of that micro-batch
(~16×1.26 ≈ 20 s at B=16 from the spike). The serial CPU floor leaves the
critical path entirely: end-to-end ≈ max(GPU_total, CPU_total) + one micro-batch
of fill latency, not the sum. **Stacks with (a):** vmap'd HyRex (§2.2) makes the
CPU side so small it hides trivially even at large micro-batches.

### 2.5 Recommended CPU strategy

**(c) pipeline as the structure + (a) vmap'd HyRex to shrink the CPU side.**
(b) is the fallback if vmap snags. The device transfers should also be batched
(stack once, transfer once) regardless — `notes_solve.md E-floor-3`, zero risk.
With (c), the 7 s/param setup floor is **removed from the critical path**.

---

## 3. Spectrum across shards

### 3.1 Prerequisite: tabulate `kappa_func` (kills the O(B) loop)

`get_Cl_batched` (`spectrum.py:616`) is a Python `for i in range(B)` loop
(`:642`) — measured ~3.3 s/param FLAT (§0.1), projecting to **~211 s at B=64**,
which would dwarf everything. The blocker is `Background.kappa_func` being a
`diffrax.Solution` (`background.py:491`, consumed via `.evaluate(lna)` in
`expmkappa`, `background.py:741`) that does not survive
`jax.tree.map(jnp.stack,...)` — hence `strip_bg_kappa` (`perturbations.py:538`).

The fix (fully specified in `notes_spectrum.md §4` + CLAUDE.md): replace
`kappa_func` with a pre-tabulated `expmkappa_tab` array on the shared
`lna_tau_tab` grid, built at construction by `vmap`-evaluating the transient
Solution, and rewrite `expmkappa` to `tools.fast_interp` — exactly the pattern
`tau`/`tau_tab` already use (`background.py:347`). Then `Background` stacks
cleanly, `strip_bg_kappa` is deleted, and the two python loops become
`jax.vmap` (`notes_spectrum.md §5.2`):

```python
@eqx.filter_jit
def get_Cl_batched(self, PT_b, BG_b, p_b):
    return jax.vmap(self.get_Cl, in_axes=(0, 0, 0))(PT_b, BG_b, p_b)
@eqx.filter_jit
def Pk_lin_batched(self, k, z, PT_b, p_b):
    return jax.vmap(self.Pk_lin, in_axes=(None, None, 0, 0))(k, z, PT_b, p_b)
```

### 3.2 Sharding the spectrum over the same B-mesh — NO cross-device reduction

The spectrum is **per-cosmology, embarrassingly parallel** — confirmed. The
line-of-sight integral for cosmology *b* uses only PT slice *b*, that cosmology's
`(lna_grid, expmkappa_tab)`, and the **replicated** Bessel/k/l tables (static
model state, not per-call data). There is **no cross-cosmology coupling**, so
**no `psum`/`all_gather`/`all_reduce` anywhere**. Feed the already-B-sharded `PT`
(from §1, GSPMD kept it `P('batch')`) and the B-sharded stacked `Background` into
the vmapped `get_Cl_batched`; GSPMD partitions the vmap over B exactly as the PE
solve. Output `(B, n_l)` is simply B-sharded. **Confirmed: no cross-device
reduction needed.** The only optional collective is a final tiny `all_gather` if
the host wants all Cls — and even that is avoidable if the per-cosmology
likelihood is computed per-shard and only a `(B,)` scalar is gathered (negligible).

The batched LoS may pressure memory at B=64 (the inner `vmap(Cl_one_ell)` over
N_ell nested under `vmap(B)`); the LoS scan is already `jax.checkpoint`'d
(`spectrum.py:827`, per `notes_spectrum.md §9`). If it OOMs, chunk the B axis of
`get_Cl_batched` like the modes path chunks k, or rely on the /4 from sharding.

### 3.3 Pk
Already cheap (~0.014 s/param) and shards identically. Never a bottleneck.

---

## 4. End-to-end 4-GPU budget

All stages on the 4-GPU mesh, with §2 CPU pipeline + §3 tabulation. Compile
(~70–90 s one-time, `flipped_multigpu_summary.txt`) is amortized over the scan and
excluded from per-param.

### B = 64 (16 per device)

| stage | strategy | per-param |
|-------|----------|----------:|
| setup HyRex | **pipelined (c) + vmap (a)** → hidden behind GPU | **~0** on critical path |
| perturb | sharded double-vmap | **0.925** (measured, spike) |
| spec_Cl | tabulated + vmap + sharded | ~0.1–0.3 (ESTIMATE) |
| spec_Pk | vmap | ~0.014 |
| GPU BG (eager, compile-cached) | sequential but cheap; or fold into pipeline | ~0.05 |

- **With the CPU pipeline:** end-to-end ≈ 0.925 + ~0.2 + ~0.05 ≈ **~1.2 s/param**.
  **~1 s is in reach**, binding constraint = the perturbation worst-k solve.
- **Without the pipeline (vmap'd-but-serial HyRex):** add ~0.3–1.0 → ~1.3–2.0
  s/param. **Without ANY CPU fix:** add the measured 7.0 → ~8 s/param. The CPU
  floor, not the GPUs, decides whether we hit budget.

The spec_Cl estimate is the **least-certain number**: the measured 3.3 s/param is
a python-loop element dominated by per-element JIT dispatch, not flops
(`notes_spectrum.md` projects 10–17× from removing the loop). Vmapped+sharded over
4 GPUs it should be well under 0.3 s/param, but this needs a real measurement
after §3.1 lands.

### B = 256 (64 per device)

- **perturb:** near the worst-k floor (§1.4); from the spike trend expect
  **~0.80–0.85 s/param** — only ~10% better than B=64.
- **memory:** per device holds 64 cosmologies × k_chunk state = 4× the B=64
  per-device load, which approaches the single-GPU B=64 OOM regime (28–31 GiB).
  **At B=256 drop `k_chunk_size` to ~50** (more chunks) to stay under 80 GiB — a
  knob, slightly erodes the per-param gain.
- **setup:** serial floor = 256×0.4 = 102 s if not pipelined → still 0.4 s/param.
  **At B=256 the CPU pipeline/vmap is mandatory**, not optional.
- **end-to-end (pipelined):** ~0.85 + ~0.2 + ~0.05 ≈ **~1.1 s/param**, basically
  the same as B=64. **Larger B buys throughput + compile amortization, NOT
  per-param latency.** The binding constraint at all B is the PE worst-k solve.

### Verdict

**~1 s/param end-to-end on 4 GPUs at B≈64 is achievable, contingent on (priority
order):**
1. **Pipeline (c) + vmap (a) HyRex** so the measured 7 s/param CPU floor leaves
   the critical path (§2). **Highest risk, highest leverage — without it the GPU
   sharding is moot.**
2. **Tabulate `kappa_func`, vmap+shard the spectrum** (§3) — else spec_Cl is the
   measured 3.3 s/param O(B) loop = +211 s at B=64.
3. Batch the device transfers; pad B to a multiple of 4.

Once all three land, the **binding constraint is the perturbation worst-k solve**
(~0.9 s/param, floor ~0.8). Beating that needs k-axis load-balancing — a separate
effort.

---

## 5. NERSC launch mechanics

### 5.1 Single process, 4 GPUs visible (RECOMMENDED)

This design wants **one Python process seeing all 4 GPUs** via `jax.devices('gpu')`
returning `[Cuda0..Cuda3]` — exactly what the spike got (`multigpu_run.log:9-10`,
`N_DEVICES = 4`). Natural fit for `Mesh(np.array(devs))` + GSPMD: one process owns
the mesh, no MPI, no `jax.distributed.initialize()`.

```bash
# parent /pscratch/sd/c/carag/CLAUDE.md GPU pattern, scaled to 4 GPUs:
salloc --no-shell --nodes=1 --qos=interactive --time=02:00:00 \
       --constraint=gpu --gpus=4 --account=m3166_g
srun --jobid=$(cat bench/.jobid) --ntasks=1 --cpus-per-task=64 \
     --gpus-per-task=4 bash -c \
     'module load conda && conda activate actdr6 \
      && export PYTHONPATH=$(pwd):$PYTHONPATH \
      && export OMP_NUM_THREADS=1 \
      && export XLA_PYTHON_CLIENT_PREALLOCATE=false \
      && python -u bench/<end_to_end_script>.py'
```

- `--ntasks=1 --gpus-per-task=4` → **one process, 4 GPUs** → `jax.devices('gpu')`
  returns all four to one interpreter. (Parent CLAUDE.md already says "change to
  gpus=4 if you want to test MCMC".)
- `--cpus-per-task=64` (not the usual 32): grab the whole socket for the HyRex
  pipeline/pool (§2). The spike used 32; the CPU stage needs more to keep up.
- Keep `XLA_PYTHON_CLIENT_PREALLOCATE=false` so 4 GPUs don't each grab 90% up
  front and collide. `PYTHONPATH=$(pwd)` is mandatory (this checkout vs the
  sibling editable install — parent CLAUDE.md).
- For the multiprocess HyRex fallback (§2.3): per worker pin `OMP_NUM_THREADS` /
  let each subprocess default to 1 CPU device so the 64 cores aren't oversubscribed.

### 5.2 4-process MPI (NOT recommended here)

`--ntasks=4 --gpus-per-task=1` + `jax.distributed.initialize()` → 4 processes, 1
GPU each, joined into a global mesh. This is the standard **multi-node** pattern
and is the **wrong tool for one node**: it forces `jax.distributed` coordination,
turns the embarrassingly-parallel B-shard into a cross-process distributed array
(adding collective machinery we proved we don't need, §3.2), and makes the CPU
HyRex pipeline awkward (4 procs each owning 1/4 of the socket vs one proc
orchestrating a 64-core pool). It only earns its keep at >4 GPUs / multi-node,
beyond this task's scope. **Use single process, `--ntasks=1 --gpus-per-task=4`.**

---

## 6. Recommended end-to-end 4-GPU architecture

```
ONE process, 4 GPUs (--ntasks=1 --gpus-per-task=4, --cpus-per-task=64)

mesh = Mesh(jax.devices('gpu'), ('batch',));  batch=P('batch'), repl=P()

for each micro-batch mb (B=64 grid = 4 micro-batches of 16):          # §2.4 pipeline
  ┌─ CPU thread (GIL released in diffrax C++): ─────────────────────┐
  │   HyRex(mb+1) VMAPPED over its 16 cosmologies on CPU   §2.2 (a)  │
  │   -> stack, TABULATE kappa as expmkappa_tab array     §3.1       │
  │   -> ONE (B,...) device_put to GPU                    §2.5       │
  └──────────────────────────────────────────────────────────────────┘
  ┌─ main thread / 4 GPUs: ─────────────────────────────────────────┐
  │   shard mb's stacked (BG, params) on P('batch')       §1.2       │
  │   PT = full_evolution_batched((BG_s, p_s))  [GSPMD, no comms]    │
  │        k-chunk loop runs per-device on its B-shard    §1.3       │
  │   Cl = get_Cl_batched(BG_s, PT)   [vmap, sharded, no comms] §3.2 │
  │   Pk = Pk_lin_batched(...)        [vmap, sharded]    §3.3        │
  └──────────────────────────────────────────────────────────────────┘

collect (B, n_l) Cls / (B, n_k) Pk  (already B-sharded; gather only if host
needs them, or compute per-shard likelihood and gather a (B,) scalar)   §3.2
```

**Primitive:** `jax.sharding.Mesh` + `NamedSharding(P('batch'))` + plain
`eqx.filter_jit` (GSPMD). NOT pmap, NOT shard_map — zero collectives in the
B-parallel pipeline.

**Projected per-param, B=64, compile amortized:**
- perturbation: **0.925 s** (measured spike) — **binding constraint**
- spectrum (Cl+Pk, vmapped+sharded): ~0.2 s (ESTIMATE, least certain)
- GPU BG (compile-cached eager): ~0.05 s
- HyRex: **~0 on critical path** (pipelined + vmapped; serial 7 s/param hidden)
- **end-to-end ≈ ~1.2 s/param → ~1 s is reached** with the three fixes. At B=256
  it stays ~1.1 s/param (near the worst-k floor).

**If we do nothing about CPU/spectrum**, the spike's 0.93 s/param is a mirage:
real end-to-end at B=64 = 0.93 (perturb) + 7.0 (serial HyRex, measured) + 3.3
(O(B) spectrum loop, measured) ≈ **~11.2 s/param** (matches §0.3), dominated by
the two stages the spike skipped. The wins, in order: (1) pipeline/vmap HyRex
(kills the measured 7.0), (2) tabulate+vmap+shard the spectrum (kills the
measured 3.3), (3) shard the PE solve over 4 GPUs (the spike's 0.93). Only with
all three does the 4-GPU sharding deliver ~1 s/param end-to-end.

---

## 7. Verification status

Resolved by direct source read this session (no longer "assumed"):
- **k-chunk concat is along the unsharded k axis** (`perturbations.py:191`,
  `axis=0` = K_CHUNK) → §1.3 "no inter-chunk device traffic" holds. **VERIFIED.**
- **HyRex is vmap-able over B**: `recomb_model.__call__` is a plain method
  (`hyrex.py:146`); `array_with_padding` uses only vmap-safe ops (argmax/isinf,
  gather-by-traced-int, `dynamic_update_slice` into a static buffer); the 4He
  index is batch-independent (`hyrex.py:144`). → §2.2 (a) is VIABLE, not just
  plausible. **VERIFIED.**
- **setup does NOT re-trace the GPU BG per cosmology** — measured flat 7 s/param
  at B=1/8/16 confirms compile-cache hits; the 7 s is HyRex+transfers, the serial
  CPU floor. **VERIFIED from `profile_stages.log`.**

Still worth confirming before/with implementation:
- The exact magnitude of the vmap'd-HyRex CPU speedup (§2.2 estimate 7→0.3–1.0
  s/param) — measure a single vmapped `RecModel` call at B=16 vs the looped
  `_build_one_bg` floor.
- spec_Cl after kappa tabulation + vmap + shard (§4's least-certain number) —
  measure once `expmkappa_tab` lands.
- Whether the batched LoS at B=64 over 4 GPUs needs B-axis chunking for memory
  (§3.2).

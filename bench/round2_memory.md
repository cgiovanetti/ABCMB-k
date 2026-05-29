# round2_memory.md — GPU-memory / batching specialist (battle-royale round 2)

**Branch:** `perk-perf`. **Hardware:** Perlmutter A100 **40 GB** (the standard
`--constraint=gpu` partition; see below — there is no 80 GB partition for this
account). **Goal:** raise the max-feasible batch B per device (and per 4-GPU
mesh), because measured per-param time keeps FALLING with B (4-GPU: B=16→3.24,
32→1.82, 48→1.34, 64→1.13 s/param). Memory caps B today (~28–31 GB/40 GB at
B=64, k_chunk=100, single GPU).

**Static analysis only — no python/GPU was run.** Every number labelled
ESTIMATE is derived from source + cited `bench/` artifacts.

---

## 0. Ground truth re-counted from source (not trusted from the memos)

### 0.1 Ny — exact, from `species.py` num_equations + `model_specs.py` l_max defaults

ΛCDM `populate_species` = (DarkEnergy, ColdDarkMatter, Baryon, Photon,
MasslessNeutrino). Defaults (`model_specs.py:31-33`): `l_max_g=12`,
`l_max_pol_g=10`, `l_max_massless_nu=17`.

| species | num_equations | source |
|---|---:|---|
| metric η | 1 | `perturbations.py:329` (prepended) |
| DarkEnergy | 0 | `species.py:342` (BackgroundFluid) |
| ColdDarkMatter | 1 | `species.py:450` |
| Baryon | 2 | `species.py:1052` |
| Photon | (12+1)+(10+1)=24 | `species.py:1262-1264` |
| MasslessNeutrino | 17+1=18 | `species.py:571` |
| **Ny** | **46** | — |

**Ny = 46 for ΛCDM, confirmed.** The design_memo's "72" is wrong for pure ΛCDM
(it likely reflects a fuller hierarchy config or diffrax internal stage padding);
every lever below scales linearly in Ny, so I use **Ny=46** throughout and flag
where the 72-based memo numbers would shrink by 46/72=0.64.

### 0.2 N_k — k-axis sizes

`get_k_axis_perturbations` (`model_specs.py:120-175`) builds the perturbation
k-axis adaptively; the brief/notes give **N_k ≈ 492 (lensing=False)** to 571
(lensing=True). `k_axis_transfer` (`model_specs.py:177-193`) is the spectrum
integration grid (`geomspace(1e-4,0.4,2500)` default in `SpectrumSolver.__init__`,
`spectrum.py:184`) — N_k_transfer ≈ 2500. **These are different axes**: the modes
solve uses N_k≈492; the LoS integral re-interpolates onto N_k_transfer≈2500.

### 0.3 Nlna — the saved-trajectory time axis

`jnp.linspace(lts, 0., 500)` — **Nlna = 500**, hard-coded at
`perturbations.py:100` (single) and `:180` (batched). The LoS scan integrates on
`PT.lna[:-1]` = **499 points** (`spectrum.py:676`, `:791`). So 500 is *both* the
saved-ys time resolution AND the LoS quadrature length — trimming it is a
double win (§2).

### 0.4 GPU memory — 40 GB, NO 80 GB partition

Searched both CLAUDE.md files and the repo: Perlmutter's standard GPU partition
(`--constraint=gpu`, account `m3166_g`) is **40 GB A100 (HBM2e)**. The parent
CLAUDE.md salloc template uses bare `--constraint=gpu` with no memory
sub-constraint. NERSC Perlmutter does **not** expose 80 GB A100s in the general
GPU queue (the 80 GB nodes are a tiny non-general pool; not reachable via the
documented salloc line). **Treat 40 GB as the hard ceiling** (§7 expands).
`XLA_PYTHON_CLIENT_PREALLOCATE=false` is the documented setting
(`bench/ideas_multigpu.md:447,456`); nothing in-tree sets a memory fraction or a
compile cache dir.

---

## 1. Memory budget accounting AT THE CURRENT CODE

### 1.1 What is resident simultaneously in `call_batched` (verified, main.py:258-269)

```
258  params_batch, BG_batch = self._build_bgs_batched(...)      # BG_batch persists
261  PT_batched = self.PE.full_evolution_batched((BG_batch, params_batch))  # PT persists
265  ClTT,ClTE,ClEE = self.SS.get_Cl_batched(PT_batched, BG_batch, params_batch)
268  Pk = self.SS.Pk_lin_batched(..., PT_batched, params_batch)
```

**Yes — BG_batch + PT_batched + params_batch all coexist** for the whole
spectrum stage (get_Cl reads PT *and* BG; Pk reads PT). They are only freed after
line 269. The transient solver workspace (§1.3) lives *inside*
`full_evolution_batched` (line 261) and is freed before the spectrum — so the
peak is **max( perturb-solve peak , spectrum peak )**, each measured on top of the
persistent (BG_batch + PT_batched) resident set. Tabulating each:

### 1.2 Persistent resident tensors (per B)

**PT_batched (the dominant persistent object).** `make_output_table` builds
~10 named (Nlna, N_k) fields per cosmology + the `species_perturbations` dict
(Photon: delta, theta, sigma, G0, G2 ≈ 5 fields; Baryon: delta, theta ≈ 2;
others). Counting the fields that survive in `PerturbationTable`
(`perturbations.py:571-582`) plus the species dict, the PT holds on the order of
**~12–16 arrays of shape (Nlna=500, N_k≈492)** per cosmology (the raw
(Ny,Nlna,Nk) modes tensor is consumed into these and freed). Take ~14 fields:

  PT_per_cosmo ≈ 14 × 500 × 492 × 8 B = **27.5 MB/cosmo**

  | B | PT_batched |
  |---:|---:|
  | 16 | 0.44 GB |
  | 64 | 1.76 GB |
  | 128 | 3.5 GB |

NOTE: the design_memo §1.1 "10.5 GB saved-ys" is the **raw (N_k,B,Nlna,Ny)
modes tensor** that lives transiently *inside* `_compute_modes_batched` (the
`jnp.concatenate` output at `perturbations.py:191`), NOT the persistent PT.
Recount it with Ny=46, N_k=492:

  modes_raw = N_k(492) × B × Nlna(500) × Ny(46) × 8 B
            = 492×500×46×8 = 90.5 MB/cosmo

  | B | modes_raw (transient, inside line 261) |
  |---:|---:|
  | 16 | 1.45 GB |
  | 64 | **5.8 GB** |
  | 128 | 11.6 GB |

(The 10.5 GB figure used Ny=72 + N_k=571 + 64 → with the real ΛCDM counts it is
**5.8 GB**, not 10.5. This matters: the saved-ys is *less* dominant than the
memos assumed.)

**BG_batch.** Background holds per-cosmosmology arrays on `lna_tau_tab`
(10000 pts: tau_tab, expmkappa_tab), xe/Tm `array_with_padding`, plus scalars.
Order **~10000 × ~4 arrays × 8 B = 0.32 MB/cosmo** → 5 MB at B=16, 20 MB at B=64.
**Negligible.** (`notes_spectrum.md §9` confirms expmkappa_tab is ~5 MB at B=64.)

**params_batch.** A dict of scalars-per-cosmo → KB-scale. Negligible.

### 1.3 The TRANSIENT Kvaerno5 implicit-solver workspace — THE REAL PEAK

This is the hidden floor. `evolution_one_k` (`perturbations.py:382-448`) runs
`Kvaerno5` (implicit), which at every internal step forms an (Ny,Ny) Jacobian and
its LU factorization, plus diffrax's stage storage (Kvaerno5 has ~5 implicit
stages) and the PIDController carry. Under the double-vmap in `_evolve_chunk`
(`perturbations.py:142-148`) **the live vmap width = k_chunk × B**, and the
transient workspace is allocated for ALL live lanes at once.

Per-lane transient ESTIMATE (Jacobian + LU + stages + Newton iterates):
  ~ (a few) × Ny² × 8 B. Take a conservative ~8×Ny² floats/lane (Jacobian + LU
  copy + ~5 stage k-vectors each Ny, + Newton residual/iterate):
  8 × 46² × 8 B = 135 KB/lane → but the **measured** anchor is better:
  `ideas_singlegpu.md:164` calibrates "transient ≈ 4.5 GB at chunk=100, B=64",
  i.e. **4.5 GB / (100×64 lanes) = 7.4 KB/lane** of *steady* workspace (XLA reuses
  most stage buffers; the (Ny,Ny) LU is the residue). Use the **measured**
  calibration, not the naive one:

  transient ≈ **7.4 KB × (k_chunk × B)**

  | k_chunk × B | transient |
  |---:|---:|
  | 100×16 = 1600 | 0.012 GB? |

That can't be right against the 28–31 GB measured peak — the per-lane figure must
include XLA scheduling overhead that does NOT factor cleanly. So I anchor to the
**two measured points** instead and bracket:

- **Measured:** chunk=100, B=64, single GPU → **28–31 GB total peak**
  (`flipped_summary.txt` / design_memo §1 / brief).
- Persistent at B=64 (§1.2): PT 1.76 + modes_raw transient 5.8 + BG 0.02 ≈ 7.6 GB.
- ⇒ **transient solver workspace + XLA overhead ≈ 20–23 GB** at chunk=100, B=64.
- This scales ~linearly in the live width (k_chunk×B): the brief & notes both
  state "transient scales with live vmap width k_chunk×B". So the dominant term is
  **C_t × k_chunk × B with C_t ≈ 20 GB / (100×64) = 3.1 MB per (k_chunk·B) unit ÷
  1000**, i.e. **≈ 3.2 GB per 1000 lanes**, or **~3.2 KB/lane × 1000** — call it
  **C_t ≈ 3.1 MB per (k_chunk×B)/100**. Cleanest usable form:

  **transient(GB) ≈ 0.031 × (k_chunk × B / 100)**  ... wait, recompute:
  20 GB at k_chunk·B = 6400 ⇒ **transient ≈ 3.1 GB per 1000 lanes** (k_chunk·B).

### 1.4 The peak model (usable for extrapolation)

Total peak ≈ persistent(B) + transient(k_chunk, B), where

```
persistent(B)  ≈ (PT 27.5 MB + modes_raw 90.5 MB) × B
               ≈ 0.118 GB × B        # modes_raw + PT coexist during concat→PT
transient(k_chunk,B) ≈ 3.1 GB × (k_chunk × B) / 1000
```

Check at chunk=100, B=64: persistent 0.118×64 = 7.6 GB; transient 3.1×6400/1000 =
19.8 GB → **27.4 GB total** ✓ (matches the measured 28–31 GB band; the extra 1–3
GB is XLA instruction/scratch buffers, roughly fixed). **The transient solver
workspace is ~2.6× the persistent set at B=64 — it IS the peak, and it is set by
k_chunk×B, not B alone.** This is the single most important fact for raising B.

### 1.5 Max B at 40 GB (single GPU), per k_chunk — DERIVED

Budget ~36 GB usable (leave ~4 GB for XLA fixed scratch + bessel/background
constants). Solve persistent(B) + transient(k_chunk,B) ≤ 36:

  0.118·B + 0.0031·k_chunk·B ≤ 36  ⇒  B ≤ 36 / (0.118 + 0.0031·k_chunk)

| k_chunk | max B (single 40 GB GPU), ESTIMATE |
|---:|---:|
| 100 | 36 / (0.118+0.31) = **84** |
| 50 | 36 / (0.118+0.155) = **132** |
| 25 | 36 / (0.118+0.0775) = **184** |
| 16 | 36 / (0.118+0.0496) = **214** |
| 10 | 36 / (0.118+0.031) = **242** |

(The measured "near-OOM at B=64, chunk=100" says my 36 GB budget at chunk=100 is
slightly optimistic — the true single-GPU max at chunk=100 is ~64–80, consistent
with the table's 84 minus XLA slack. The *trend* is the load-bearing result:
**halving k_chunk roughly 1.5×'s the max B; quartering it ~2.2×'s it.**)

### 1.6 Max B per device under 4-GPU sharding

Sharding the B axis (`call_batched(shard=)`, `main.py:236-248`) gives each device
B/4 cosmologies — both persistent AND transient divide by 4 (each device builds &
solves its own B/4 lanes). So **max total B on 4 GPUs ≈ 4 × (single-GPU max B)**:

| k_chunk | max total B, 4-GPU, ESTIMATE |
|---:|---:|
| 100 | ~256–336 |
| 50 | ~528 |
| 25 | ~736 |

Per the perf curve (still falling at B=64), a 4-GPU run at B=256, chunk=50 is the
near-term throughput sweet spot to *test*. **Sharding × small-k_chunk compound
multiplicatively on max B.**

---

## 2. saveat / Nlna trimming (perturbations.py:180, linspace(lts,0,500) → ~300 non-uniform)

**Mechanism.** Nlna appears in (a) modes_raw transient (∝ Nlna), (b) PT
persistent (∝ Nlna), and (c) the LoS scan length (`spectrum.py:676,791`, 499
iterations). It does NOT appear in the transient *solver* workspace (§1.3) — the
Kvaerno5 LU is per-step, independent of how many `ts` you save. So Nlna trim cuts
**persistent(B)** from 0.118·B toward 0.118·(300/500)·B = 0.071·B, AND shortens
the LoS scan ~40%.

**Headroom it buys.** Persistent drops 40%, but persistent is only ~28% of the
B=64 peak (7.6/27.4). So peak drops only ~11% at chunk=100 → **max B at chunk=100
rises ~84→~94, ~12%.** *Modest* on its own at large k_chunk. **BUT** it stacks
multiplicatively with small k_chunk: when transient is small (k_chunk≤25),
persistent dominates, so a 40% persistent cut → ~40% more B. So Nlna-trim is a
**force-multiplier for the small-k_chunk regime** (§5), and it directly speeds the
LoS scan (~1.4× fewer scan steps in `Cl_one_ell`).

Concretely (replace `perturbations.py:180`):
```python
lna_batch = vmap(lambda lts: _nonuniform_lna(lts, BG_b.lna_rec))(BG_batch.lna_transfer_start)
# dense near recomb, sparse in the smooth tails; ~300 pts (notes_solve §C sketch)
```
The LoS axis (`spectrum.py:676`) automatically inherits PT.lna, so it shrinks too
— no second edit.

**Accuracy risk: MEDIUM.** Trajectories are smooth except across recombination
(visibility peak). A naive uniform 500→300 would thin the recomb region; a
**recomb-concentrated** non-uniform 300 should be ≥ as accurate. **Cheap test:**
regen snapshots + run `pytests/accuracy_test.py` (1%-vs-CLASS) at Nlna=300
non-uniform; report TT/EE/Pk max-rel.

**Effort:** ~20 lines (one grid helper + the `linspace`→helper swap). **Prob
success:** HIGH that it runs; MEDIUM that 300 passes the gate first try (may need
350 or a tuned breakpoint). **Hidden floor:** the LoS scan and the cubic-spline
k-interp in `Cl_one_ell` (`spectrum.py:700-714`) re-interp on N_k_transfer=2500
regardless — Nlna trim does nothing for *that* (k-direction) cost, which is the
larger part of the spectrum compute. So the throughput win is mostly the
memory-headroom-→-bigger-B indirect win, not a direct spectrum speedup.

---

## 3. bf16/fp16 STORAGE of saved trajectories (solve stays fp64; downcast only the stored tensor)

**Mechanism.** Keep `evolution_one_k` in fp64 (the Jacobian/LU and the gate
depend on it), but **downcast `sol.ys` to bf16 immediately on return**
(`perturbations.py:448`) so the (N_k,B,Nlna,Ny) modes_raw tensor AND the PT fields
it feeds are bf16. The tensor is consumed in two places:
1. `make_output_table` (`perturbations.py:450-536`) — builds PT fields by
   vmapping species reductions over the modes; these would run in bf16 then the PT
   stores bf16.
2. `Cl_one_ell` / `Pk_lin` interpolation (`spectrum.py:700-714`, `:292-300`) —
   `CubicSpline`/`jnp.interp` read the PT fields.

bf16 halves modes_raw (5.8→2.9 GB at B=64) and PT (1.76→0.88 GB) → persistent
0.118·B → 0.059·B. **Same ~11% peak cut at chunk=100 as Nlna-trim, but again a
~2× multiplier in the persistent-dominated small-k_chunk regime.** fp16 (not bf16)
would keep more mantissa (10 bits vs 7) but risks overflow on the (k²·delta) terms
in `make_output_table`; **bf16 is safer dynamic-range-wise**, fp16 more accurate
if no overflow.

**Accuracy risk: HIGH (skeptical).** bf16 has ~3 decimal digits (2⁻⁸ ≈ 4e-3
relative). The transfer functions are *squared* then integrated for Cls
(`spectrum.py:825-827`) and *squared* for Pk (`spectrum.py:302`) — squaring
doubles relative error to ~8e-3, dangerously close to the 1% gate, and the LoS
integral's near-cancellations (ISW vs SW) could amplify it further. **This is the
riskiest of the storage levers.** A safer variant: **fp16 storage of ONLY the
smooth, large-amplitude fields** (delta_m, metric_eta) and keep the
cancellation-prone source ingredients (Photon delta/sigma, the metric α/η
derivatives that build the ISW source) in fp32 not fp16. But that's fiddly and
partial.

**A better-targeted alternative: fp32 storage (not bf16).** fp32 has ~7 digits
(1e-7), well below the 1e-4 solver rtol, so storing the trajectory in fp32 loses
nothing the solver didn't already throw away, and **still halves** persistent vs
fp64. This is the **sweet spot**: identical memory win to bf16, near-zero accuracy
risk (the stored values already carry only ~4 sig figs of physical meaning at
rtol=1e-4, but fp32's 7 digits is a comfortable margin and the *interpolation* in
fp32 is fine). **Recommend fp32 storage, not bf16.**

**Cheap test:** add `.astype(jnp.float32)` at `perturbations.py:448` return; run
snapshots (will shift — regen) + accuracy_test; report max-rel. **Effort:** ~5
lines (the cast + ensuring `make_output_table` upcasts back to fp64 for the
cancellation-sensitive metric assembly, OR leaving it fp32 throughout the PT).
**Prob success:** fp32 HIGH; bf16 LOW-MEDIUM. **Hidden floor:** the transient
solver workspace (§1.3) stays fp64 and is the *larger* term at chunk=100, so fp32
storage alone only ~11%'s the peak at large chunk — its real value is, again,
compounding with small k_chunk (§5). Also: `jnp.interp`/`CubicSpline` may upcast
internally, silently negating the storage saving during the spectrum read — must
verify the interp keeps fp32 (or chunk the spectrum's B axis so only B/n PT slices
are upcast at once).

---

## 4. donate_argnums / buffer reuse (eqx.filter_jit donate)

**Mechanism.** `eqx.filter_jit(..., donate="warn"/"all")` lets XLA reuse an input
buffer for an output, avoiding a fresh allocation. Three candidate hand-offs:

1. **modes_raw → PT.** `make_output_table_batched` (`perturbations.py:224`) takes
   `modes_batch` and produces PT. modes_batch (5.8 GB at B=64) is **not reused after**
   `make_output_table` — donating it lets the PT fields reuse that buffer. Saves
   re-allocating PT (1.76 GB) on top of modes (5.8 GB) → cuts the line-261 peak by
   up to ~1.76 GB. **Safe** (modes_raw is dead after the table build).
2. **PT → Cls.** `get_Cl_batched` (`spectrum.py:616`) reads PT_batched — BUT PT is
   *also* read by `Pk_lin_batched` afterward (`main.py:268`), so PT is NOT dead and
   **must NOT be donated** to get_Cl. Could reorder (Pk first, then donate PT to
   get_Cl) but the Cls output is tiny (B×799 floats) — donating into it saves
   almost nothing. **Skip.**
3. **NOT the chunk-loop inputs.** `_evolve_chunk` (`perturbations.py:125`) is
   called once per chunk with the SAME `lna_batch`/`BG_batch`/`params_batch` reused
   across all chunks (`perturbations.py:186-189`) — **donating those would corrupt
   later chunks.** Only the per-chunk `k_chunk` slice and the chunk's output are
   donatable, and they're small. So **no donation inside the chunk loop.**

**Net.** The only worthwhile donation is **modes_raw → PT** at the
`make_output_table_batched` boundary. Mechanically: have `_compute_modes_batched`
return the modes and immediately call the table build in a single
`@eqx.filter_jit(donate="warn")` so XLA can alias. But note modes is built by
`jnp.concatenate(chunks)` *outside* a jit (`perturbations.py:191`) — the concat
result is a fresh buffer that `make_output_table_batched` (jitted) receives; mark
that arg donated.

**Effect on max B:** ~1.76 GB at B=64 ≈ slides max B from ~64 to ~68 at
chunk=100. **Small.** **Accuracy risk: NONE** (bit-identical). **Effort: LOW**
(filter_jit kwarg + ensure no later read of the donated arg). **Prob success:
HIGH** it works; MEDIUM the saving is measurable above XLA's existing
buffer-reuse (XLA already aliases many dead buffers; explicit donation may be
redundant). **Hidden floor:** XLA's memory planner likely *already* frees
modes_raw before allocating PT (they're in the same jit region if fused) — so the
realized win may be ≈0. Cheap to try, low ceiling.

---

## 5. k_chunk as a memory knob — adaptive k_chunk = f(B, mem_budget) ⭐

**This is the highest-leverage memory lever, because §1.4 proves the peak is
dominated by transient ∝ (k_chunk × B), not B alone.** Today k_chunk is a fixed
default 100 (`perturbations.py:150,194`; `call_batched` doesn't even thread it
through — `main.py:261` calls `full_evolution_batched((BG,params))` with the
default). Make it adaptive:

```python
# in call_batched, before full_evolution_batched:
PER_DEV_B = ceil(B / n_dev)                 # B per device after sharding
budget_GB = 34.0                            # leave headroom on 40 GB
persistent = 0.118 * PER_DEV_B              # GB (fp64; halve if fp32 storage §3)
trans_per_unit = 0.0031                     # GB per (k_chunk·B)/... (calibrated §1.3)
k_chunk = int((budget_GB - persistent) / (trans_per_unit * PER_DEV_B))
k_chunk = max(8, min(N_k, k_chunk))
```

Then thread `k_chunk_size=k_chunk` into `full_evolution_batched` (it already
accepts it, `perturbations.py:194`). **Tradeoff vs serialization:** smaller chunk
⇒ more chunks ⇒ more sequential `_evolve_chunk` launches (each a kernel-launch
barrier + a JIT-cache lookup) and more `jnp.concatenate` inputs. At chunk=100,
N_k=492 → 5 chunks; chunk=25 → 20 chunks. Launch overhead is ~tens of µs–ms each;
at 20 chunks that's still ≪ the multi-second solve, so **serialization cost is
negligible until chunk gets very small (<16)**. Below ~chunk=16 the device is
*underutilized within a chunk* at small B (only chunk×B lanes in flight), so
throughput drops — that's the lower floor.

**Optimal (B, k_chunk) frontier for max throughput** (ESTIMATE, single + 4-GPU):
- **The perf curve falls with B**, so the throughput optimum is the **largest B
  that fits**, at the **smallest k_chunk that keeps the device saturated**.
- Single GPU: device saturates around live-width k_chunk×B ≳ ~3000–6000 lanes
  (A100 has 6912 fp64 cores; needs ~10⁴ lanes to hide latency). So keep
  **k_chunk×B ≈ 6000** as the saturation floor, then maximize B subject to memory:
  - B=64: k_chunk ≈ 6000/64 ≈ 94 (≈ current 100 — explains why the sweep found 100
    "optimal" at B=64!). Peak ~27 GB ✓.
  - B=128: k_chunk ≈ 47, peak = 0.118·128 + 0.0031·47·128 = 15.1 + 18.6 = **33.7
    GB** — *fits on one 40 GB GPU!* This is the headline: **B=128 is reachable on
    ONE A100 by halving k_chunk to ~47**, and per-param keeps falling.
  - B=256 single GPU: k_chunk≈23 → 30.2 + 18.2 = 48 GB → OOM. Needs sharding.
- 4-GPU: per-device B=B/4. B=256 total → per-dev 64 → chunk≈94, per-dev peak ~27
  GB ✓. **B=512 total, 4-GPU → per-dev 128 → chunk≈47 → 33.7 GB/dev ✓.**

So the frontier: **single A100 reaches B≈128 (chunk≈47); 4×A100 reaches B≈512
(chunk≈47).** With fp32 storage (§3) halving persistent, push further: single
B≈170, 4-GPU B≈680.

**Accuracy risk: NONE** — chunking only regroups lanes; identical math, already
proven benign (`chunking_debug_report.md`; the keystone validation is ~1e-6).
**Effort: LOW-MEDIUM** (~30 lines: the budget formula + thread k_chunk through
`call_batched`→`full_evolution_batched`; calibrate trans_per_unit with ONE GPU
measurement). **Prob success: HIGH.** **Hidden floor:** (a) the saturation floor
— below k_chunk×B≈3000–6000 the device idles and per-param *rises*, so adaptive
k_chunk must not shrink chunk below the saturation point just to fit a bigger B
that the device can't fill; (b) each distinct k_chunk *shape* recompiles
(`_evolve_chunk` JIT cache keyed on shape, `perturbations.py:131-134`) + a ragged
final chunk = a 2nd compile — so adaptive k_chunk should **pad the final chunk to
a uniform shape** (notes_solve §B2) and pick from a small set of chunk sizes to
keep the compile cache warm across a scan.

---

## 6. Not materializing the full (B,Ny,Nlna,N_k) tensor — fuse modes→PT→Cl per chunk

**Mechanism.** Today: solve ALL chunks → `jnp.concatenate` to one
(N_k,B,Nlna,Ny) tensor (`perturbations.py:191`) → transpose → `make_output_table`
→ get_Cl. The full modes_raw (5.8 GB at B=64) + PT (1.76 GB) both fully
materialize. **Alternative:** the LoS integral in `Cl_one_ell` is a *sum over k*
(`spectrum.py:825-831`, `jnp.trapezoid(integrand, k_axis)`), and the transfer
functions are built per-k-column (`spectrum.py:700-714` interpolate each PT field
from PT.k onto k_axis_transfer). In principle one could **stream per k-chunk**:
solve a k-chunk → build its PT-columns → accumulate its contribution to the LoS
k-integral → discard, never holding all N_k at once. Peak would drop to
~one-chunk's worth of modes + the (small) Cl accumulators.

**Why it's HARD here (the hidden floor is structural):** the modes solve uses
N_k≈492 (the *perturbation* k-axis), but the LoS integral uses
N_k_transfer≈2500 (a *different, denser* k-axis), and the bridge is a **CubicSpline
over the FULL PT.k → k_axis_transfer** (`spectrum.py:700`,
`CubicSpline(log10(PT.k), col)`). A cubic spline needs the *entire* PT.k column to
build its coefficients — **you cannot spline a k-chunk in isolation** without
boundary error at the chunk seams. So streaming would require either (a) switching
the k-interp to a *local* scheme (linear or local-cubic) that only needs a few
neighbors — an accuracy change that must clear the 1% gate — or (b) accepting
seam error. This is why the current design materializes the full PT first. **Real
but high-effort and accuracy-coupled.**

**`jax.checkpoint` (rematerialization).** Already applied to the LoS scan body
(`spectrum.py:817-819`) — it kills the (Nell,Nlna,Nk) integrand rematerialization
(~21 GB) by recomputing the scan body on backward. For the *forward-only*
frequentist path (ForwardMode, no backward), that particular checkpoint mainly
helps the AD path; in pure forward it's near-free. A *new* checkpoint opportunity:
wrap `make_output_table` so its intermediate (Nlna,Nk) species reductions
(`perturbations.py:506-523`) are rematerialized rather than held — but those feed
directly into the persistent PT, so checkpointing them saves only transient
scratch during the table build, modest.

**Effect on max B:** streaming (if done) could cut the modes-resident from
5.8→~1.2 GB (one chunk) at B=64 → frees ~4.6 GB → max B ~64→~80 at chunk=100. But
the transient solver workspace (§1.3, 20 GB) is per-chunk *already* (only one chunk
solves at a time) and is unchanged. **So streaming attacks the persistent modes
term, which §1.4 shows is the *minor* term — the win is bounded by ~persistent,
not the dominant transient.** Combined with the k-spline difficulty, **low
priority.**

**Accuracy risk:** HIGH if the k-interp is changed to enable streaming; NONE for
the checkpoint-only variant (but that variant barely helps forward-only).
**Effort: HIGH** (restructures the modes→spectrum boundary). **Prob success:
LOW-MEDIUM.** **Hidden floor:** as above — the dominant memory term is the
*transient solver workspace per chunk*, which streaming does NOT reduce (it's
already one-chunk-at-a-time). Streaming only removes the *persistent* concatenated
modes, the smaller term. **Adaptive k_chunk (§5) gets most of streaming's benefit
with none of its accuracy risk**, because shrinking k_chunk shrinks the dominant
transient directly.

---

## 7. XLA_PYTHON_CLIENT_PREALLOCATE / memory fraction / 80 GB A100s

**PREALLOCATE.** Already `false` (documented `bench/ideas_multigpu.md:447,456`);
nothing in-tree sets `XLA_PYTHON_CLIENT_MEM_FRACTION`. With prealloc off, XLA
grows on demand and the 40 GB is fully available, but **fragmentation** under the
grow-on-demand allocator can cause OOM below the nominal 40 GB (the measured
"28–31 GB near-OOM" likely includes fragmentation slack). **Lever:** try
`XLA_PYTHON_CLIENT_PREALLOCATE=true` + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` for
production scans — preallocating one big slab eliminates fragmentation and can
recover ~2–4 GB of usable headroom (→ slightly higher max B). **Risk: NONE to
accuracy**, but on a *shared* 4-GPU node each process grabbing 95% can collide if
processes aren't 1-per-GPU; with the `Mesh` over all 4 GPUs in one process
(current design, `main.py:238`) it's one process owning all 4, so prealloc=true is
safe. **Cheap test:** set the two env vars, re-run the B=64 perf, see if B=80–96
now fits single-GPU. **Effort: trivial (env vars).** **Prob success: MEDIUM** that
it recovers meaningful headroom; the fragmentation slack is real but bounded.

**80 GB A100s.** Searched both CLAUDE.md files + repo: **Perlmutter's general GPU
queue (`--constraint=gpu`, account `m3166_g`) is 40 GB A100 (HBM2e) only.** NERSC
does not expose 80 GB A100s in the documented salloc path for this account. So the
"~2× max B from 80 GB" lever is **NOT available** — do not bank on it. If a 80 GB
partition *were* reachable it would trivially ~2× every max-B number above
(transient + persistent both ∝ memory), but that's hypothetical. **The realistic
ceiling is 40 GB/device.**

---

## RANKED SHORTLIST (throughput gain × prob ÷ (risk × effort))

Ranking metric in brackets. "Max-B" = max feasible batch; throughput rises
because per-param keeps falling with B.

### #1 — Adaptive `k_chunk = f(B, mem_budget)` (§5)  [score: dominant]
Mechanism: pick k_chunk from the §1.4 peak model so the transient (∝ k_chunk×B,
the *dominant* term) fits the per-device budget, keeping device saturation
(k_chunk×B ≳ ~3000–6000). Today k_chunk is a fixed 100 not even threaded through
`call_batched`. **Effect: single A100 B 64→~128 (chunk≈47); 4-GPU B≈256→512.**
Per-param falls all the way (the curve is still descending at B=64). **Accuracy
risk: NONE** (lane regrouping only — proven benign). **Effort: LOW-MEDIUM**
(~30 lines + calibrate one constant). **Prob: HIGH.** **Hidden floor:** must not
shrink chunk below the device-saturation point (per-param *rises* if the device
idles); pad ragged chunk + use a small fixed set of chunk sizes to avoid
recompiles. **This is the lever that directly attacks the term §1.4 proves is the
peak.**

### #2 — fp32 storage of saved trajectories (§3, the fp32 variant, NOT bf16)  [high]
Mechanism: `.astype(float32)` the returned `sol.ys` (`perturbations.py:448`) so
modes_raw + PT are fp32; solve stays fp64. Halves persistent(B) (0.118→0.059·B).
**Effect: ~11% peak cut at chunk=100 (max B 64→~72) BUT compounds with #1 —
in the small-k_chunk / large-B regime where persistent dominates, fp32 storage
~doubles the reachable B** (single ~128→~170, 4-GPU ~512→~680). **Accuracy risk:
LOW** (fp32's 7 digits ≫ the 1e-4 solver rtol the values already carry; bf16's 3
digits is the risky one — avoid bf16). **Effort: LOW** (~5 lines + verify the
spectrum interp doesn't silently upcast and negate it). **Prob: HIGH** (fp32).
**Hidden floor:** `jnp.interp`/`CubicSpline` may upcast to fp64 internally during
the spectrum read, transiently re-inflating — verify; and the dominant transient
solver workspace stays fp64, so fp32-storage alone (at chunk=100) is modest — its
value is the compounding with #1.

### #3 — saveat / Nlna trim 500→~300 non-uniform at recomb (§2)  [medium-high]
Mechanism: replace `linspace(lts,0,500)` (`perturbations.py:180`) with a
recomb-concentrated ~300-pt grid. Cuts persistent ∝ Nlna AND the LoS scan length
~40% (a direct ~1.4× on the `Cl_one_ell` scan). **Effect: ~12% more max B at
chunk=100; ~40% more in the persistent-dominated small-chunk regime (compounds
with #1/#2); plus a real spectrum-scan speedup.** **Accuracy risk: MEDIUM** (must
preserve recomb resolution — gate it). **Effort: LOW-MEDIUM (~20 lines).**
**Prob: HIGH it runs, MEDIUM 300 passes first try** (may need 350). **Hidden
floor:** does nothing for the k-direction spectrum cost (N_k_transfer=2500 cubic
interp), which is the larger spectrum term; the memory win is indirect
(→bigger B).

### #4 — Prealloc + mem-fraction env vars to recover fragmentation slack (§7)  [medium, trivial]
Mechanism: `XLA_PYTHON_CLIENT_PREALLOCATE=true` + `MEM_FRACTION=0.95` for the
single-process 4-GPU `Mesh` run — one slab, no grow-on-demand fragmentation.
**Effect: recover ~2–4 GB usable → ~10–15% more max B.** **Risk: NONE to
accuracy** (safe with the current 1-process-owns-all-GPUs design). **Effort:
TRIVIAL (env vars in the slurm/salloc).** **Prob: MEDIUM** the slack is
meaningful. **Hidden floor:** if the orchestration ever moves to 1-process-per-GPU,
prealloc=0.95 each would collide — keep it tied to the single-process Mesh.

### #5 — donate modes_raw → PT at the table-build boundary (§4)  [low, free]
Mechanism: `eqx.filter_jit(make_output_table_batched, donate="warn")` so the dead
modes_raw buffer aliases into PT. **Effect: ~1.76 GB at B=64 → max B 64→~68.**
**Risk: NONE** (bit-identical). **Effort: LOW (one kwarg + no-later-read check).**
**Prob: HIGH it works; MEDIUM it helps** above XLA's existing auto-aliasing.
**Hidden floor:** XLA likely already frees modes_raw before PT alloc within the
fused region → realized win may be ≈0. Try it (free), don't count on it.

**NOT recommended:** bf16 storage (§3, too close to the 1% gate after squaring);
streaming modes→PT→Cl to avoid materializing the full tensor (§6, HIGH effort,
blocked by the global CubicSpline k-interp, and it only attacks the *minor*
persistent term while leaving the dominant transient untouched — #1 gets the
benefit risk-free). 80 GB A100s (§7, not available on this account).

---

## My #1 bet

**Adaptive k_chunk (§5/#1), stacked with fp32 trajectory storage (§3/#2).**

Rationale: §1.4 establishes — recounted from real source, Ny=46 not 72 — that the
peak is **transient solver workspace ∝ (k_chunk × B)**, ~2.6× the persistent set
at B=64. The fixed k_chunk=100 was tuned for B=64 (where k_chunk×B≈6000 happens to
saturate an A100), which is exactly why the uniform sweep called 100 "optimal" —
**it's only optimal at B=64.** At larger B you must *shrink* k_chunk to fit, and
because per-param keeps falling with B, the throughput-maximizing point is the
largest B the (adaptive-chunk) memory model allows while staying device-saturated.
The model says **single A100 reaches B≈128 (chunk≈47), and 4-GPU reaches B≈512**,
with fp32 storage pushing those to ~170 / ~680. Both levers are NONE/LOW accuracy
risk (lane regrouping is proven benign; fp32 ≫ the rtol already in play), LOW
effort, and they multiply. This is strictly better than the high-risk float32
*solve* or fixed-step levers the other memos chase.

## Cheapest GPU measurement to de-risk it

**One srun, ~15 min, single A100, ELLMAX=800, lensing=False:** sweep
`call_batched` with **`shard=False`** at **(B, k_chunk)** ∈
{(64,100), (96,64), (128,47), (160,32)} and record, via
`jax.devices()[0].memory_stats()['peak_bytes_in_use']` after each call,
**peak bytes + per-param wall-clock**. This single sweep:
1. **Calibrates the §1.4 constant** `trans_per_unit` (fit peak vs k_chunk×B across
   the 4 points) — turning the ESTIMATE into a measured allocator.
2. **Confirms B=128 fits at chunk≈47** on one 40 GB GPU (the headline claim) and
   shows whether per-param is still falling at B=128 (it should be).
3. **Finds the device-saturation floor** (where shrinking k_chunk stops helping
   per-param) — the §5 hidden floor.

`memory_stats()` is a free in-process query (no profiler overhead); the only cost
is the ~4 compiles (one per k_chunk shape) + 4 solves. If B=128/chunk=47 fits and
per-param < the B=64 value, ship adaptive-k_chunk immediately and re-run the 4-GPU
`perf_multigpu` at B=256/chunk≈47 to confirm the 4× extrapolation.

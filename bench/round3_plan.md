# Round-3 synthesis + forward plan (perk-perf) — mem/call & massive-ν

Continues `bench/round2_plan.md`. Round-3 question (user): reduce GPU memory PER CALL
so more cosmologies fit on one 80 GB node, and find architectural massive-ν
speedups. Battle-royale memos: `bench/round3_memory.md`, `bench/round3_massivenu.md`.

## The unifying finding
Per-device GPU peak ≈ **3.65 · N_k · Nlna · Ny · 8B · B_local** (persistent
saved-trajectory tensor; round 2). `Ny` is ALSO what makes massive-ν slow (O(Ny²)
implicit solve). So both questions reduce to shrinking that tensor. Measured ground:
massless Ny≈46 → 0.33 GB/B_local; one massive ν → `num_equations=3·(l_max_massive_nu+1)=54`
→ Ny≈100 → 0.60 GB/B_local AND ~6× slower (≈4.8× from Ny→100 × ~1.3× from a
redundant live background q-integral). N_q=3 (perturbations) is already minimal — NOT
a lever. q-loop is trace-unrolled (vmaps fine).

## DO FIRST — accuracy-neutral (user: "a good place to start")
1. **Transpose-kill** — `perturbations.py:_compute_modes_batched` (~:191) builds the
   concatenated `(N_k,B,Nlna,Ny)` tensor AND its full transpose at once → a ~2× spike,
   the single biggest memory contributor. Restructure so the transpose buffer isn't
   separately materialized (e.g. transpose per-chunk inside `_evolve_chunk`, or write
   the concat directly into the (B,Ny,Nlna,N_k) layout). **Bit-identical** → gate with
   `test_snapshots.py` (rtol=1e-5 now) + `peak_bytes_in_use` before/after at fixed B.
   Risk: must preserve exact ordering. [round3_memory.md #3]
2. **Tabulate `aH`** — `background.py`: `aH→rho_tot` (incl. the massive-ν q-integral +
   full species sum) is recomputed ~5–6×/RHS-step and is NOT cached, unlike
   `tau`/`xe`/`expmkappa` which ARE (`tau_tab`, `expmkappa_tab`). Mirror that pattern:
   tabulate `aH` on the shared lna grid in `Background.__init__`, read via interp.
   ≪permille (smooth background), ~1.15–1.35× on massive-ν, helps massless too. Gate:
   `accuracy_test.py` (1% vs CLASS) + snapshots. [round3_massivenu.md #1]
   (Round-1's "RHS CSE" lever, now concrete: XLA already CSEs within a trace, but the
   single per-step `aH` eval is analytic; tabulating makes it an interp lookup.)

## ACCURACY-GATED — user OK to explore (gate vs CLASS first)
3. **Nlna reduction** (`n_lna_PE` is now a spec, default 500; set at
   `perturbations.py:100,180`). The save grid is BOTH the perturbation output grid and
   the spectrum LoS quadrature grid, so trimming it is a flat multiplier on the whole
   memory peak AND the LoS scan → ~1.6× more B_local at Nlna=300.
   - **Step A (cheap):** uniform reduction. `accuracy_test.py` at `n_lna_PE ∈
     {400,350,300,250}`; find where TT/EE/Pk exceed baseline (0.197/0.231/0.185%) by
     more than ~0.05%. **User's read: uniform MIGHT hold, but probably needs uneven
     spacing — don't assume.**
   - **Step B (if uniform fails):** non-uniform **recomb-dense** grid (mirror CLASS's
     uneven sampling — dense near `lna_rec`/the visibility spike, sparse elsewhere),
     replacing the `linspace` at `perturbations.py:100,180`. MUST also fix the LoS
     trapezoid weights at `spectrum.py:~792` to per-interval `jnp.diff` — they
     currently assume uniform `delta_lna`, so a non-uniform grid silently biases the
     integral. [round3_memory.md #1, correctness flag]
4. (lower priority) **Stream PT per k-chunk** — accumulate PT columns from transient
   per-chunk modes so the full modes tensor never co-exists; peak 3.65→~1.6–2.0×.
   Blocker: the spectrum's global cubic spline over PT.k needs all k. Bigger refactor.
   [round3_memory.md #2]
5. (free riders) `donate_argnums` on the batched stages; B-/ell-chunk the spectrum if
   its LoS source tensor `(Nell,~499,~2500)` becomes the binding peak after 1+3.

## REJECTED / DO-NOT-PURSUE (user decisions)
- **Lower l_max for massive ν — NO** (user, 2026-05-29). So round3_massivenu.md #2/#3
  (l_max_massive_nu / per-bin l_max reduction) are OFF the table.
- TCA / diffrax regime-switching (expensive); tol-loosening, float32, fp32/bf16
  storage (accuracy); SLURM job arrays (Perlmutter); shrinking k_chunk (no mem benefit,
  slower). See round2_plan.md.

## Cheapest de-risk for the gated work
One CLASS-gated srun: `accuracy_test`-style sweep over `n_lna_PE ∈ {500,400,350,300}`
(massless) recording TT/EE/Pk vs CLASS, + a peak-mem read at Nlna=500 vs 300 (expect
~0.6× the persistent peak). For the aH-tabulation, add one accuracy_test point with the
tabulated aH and confirm ≪permille. `bench/mem_throughput_sweep.py` (has `--massive`)
measures the B_local/throughput payoff.

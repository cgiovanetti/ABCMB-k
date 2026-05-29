"""scan_slice.py — one task of an embarrassingly-parallel frequentist scan.

Each invocation owns a contiguous slice of a global parameter grid and evaluates
it through ``Model.call_batched(shard=True)`` on the single node's GPUs, in
fixed-size padded batches so every call hits the SAME compiled HLO (and therefore
the SAME persistent-cache entry across tasks/nodes).

Design (see bench/round2_scaleout.md + the user's Perlmutter guidance):
  * The scan is embarrassingly parallel over cosmologies — there is NO reduction
    across cosmologies anywhere in the pipeline. We use ONE multi-node SLURM job
    that srun-launches one independent worker per node (NOT a job array — Perlmutter
    is touchy about arrays and only ~2 queued jobs gain priority at a time; and NOT
    one ``jax.distributed`` global mesh — that would add cross-node traffic for an
    axis we never reduce). Each worker is an independent process that sees only its
    node's 4 GPUs and shards its slice over them (GSPMD, P('batch')).
  * Each worker owns ``grid[tid::ntask]`` (strided, so adjacent grid points land on
    different nodes — robust if the grid is ordered). tid/ntask come from
    SLURM_PROCID/SLURM_NPROCS (one multi-node job) or, as a fallback, the array vars.
  * Persistent XLA compile cache on $SCRATCH (set BEFORE importing jax) makes the
    70-220 s compile a once-per-(B,k_chunk) cost — amortized across *resumed* jobs
    (within one job the nodes start together and compile in parallel, which is fine).
  * Fixed B padding: the LAST batch of a slice is padded up to B (and sliced off)
    so it reuses the cached HLO instead of triggering a recompile.
  * Resumable: a worker whose slice_<tid>.npz already exists is skipped (set
    ABCMB_SCAN_RESUME=0 to force recompute), so a walltimed-out scan re-runs cleanly.

Env / argv (all optional; sensible defaults):
  ABCMB_SCAN_N        total grid size (default 4096)
  ABCMB_SCAN_B        per-call batch size, fixed (default 256)
  ABCMB_SCAN_KCHUNK   k_chunk_size (default 0 => call_batched default 100, optimal)
  ABCMB_SCAN_SEED     grid RNG seed (default 0)
  ABCMB_SCAN_OUT      output dir (default scan/out)
  ABCMB_SCAN_RESUME   skip workers whose output exists (default 1)
  ABCMB_CACHE_DIR     compile cache dir (default $SCRATCH/.jax_cache_abcmb)
  SLURM_PROCID / SLURM_NPROCS    (one multi-node job; else array vars; else 0/1)

Output: one ``slice_<taskid>.npz`` with the per-cosmology ClTT/ClTE/ClEE/Pk and
the grid params for this slice. Swap the ``summarize`` hook for a real likelihood
to emit only a (B,) chi2 vector instead (kills the only place a gather could occur
and shrinks output ~1000x).
"""
import os, sys, time

# --- compile cache MUST be configured before jax is imported/initialized ---
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
CACHE_DIR = os.environ.get("ABCMB_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
import jax.numpy as jnp

from abcmb.main import Model

# ----- parameter grid (replace with the real scan grid / sampler) -----
FIDUCIAL = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225, 'A_s': 2.12424e-9,
    'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5, 'z_reion_He': 3.5,
    'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
BOXES = {'h': (0.65, 0.70), 'omega_cdm': (0.115, 0.125),
         'omega_b': (0.0220, 0.0230), 'A_s': (1.95e-9, 2.25e-9),
         'n_s': (0.950, 0.980)}


def make_grid(n, seed):
    """Deterministic grid so every task generates the SAME global grid and just
    slices its own portion (no central coordinator needed)."""
    rng = np.random.default_rng(seed)
    draws = {k: rng.uniform(lo, hi, size=n) for k, (lo, hi) in BOXES.items()}
    out = []
    for i in range(n):
        p = dict(FIDUCIAL)
        for k in BOXES:
            p[k] = float(draws[k][i])
        out.append(p)
    return out


def summarize(out, valid):
    """Hook: turn a BatchedOutput into the artifact you keep. Default keeps the
    full spectra for the valid (non-padding) rows. Replace with a likelihood to
    emit a (B,) chi2 and shrink output ~1000x."""
    return dict(
        ClTT=np.asarray(out.ClTT)[:valid], ClTE=np.asarray(out.ClTE)[:valid],
        ClEE=np.asarray(out.ClEE)[:valid], Pk=np.asarray(out.Pk)[:valid],
        params={k: np.asarray(v)[:valid] for k, v in out.params.items()},
    )


def main():
    N      = int(os.environ.get("ABCMB_SCAN_N", 4096))
    B      = int(os.environ.get("ABCMB_SCAN_B", 256))
    KCHUNK = int(os.environ.get("ABCMB_SCAN_KCHUNK", 0))   # 0 => auto
    SEED   = int(os.environ.get("ABCMB_SCAN_SEED", 0))
    OUT    = os.environ.get("ABCMB_SCAN_OUT", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out"))
    ELLMAX = int(os.environ.get("ABCMB_SCAN_ELLMAX", 2500))
    LENS   = os.environ.get("ABCMB_SCAN_LENSING", "1") == "1"
    os.makedirs(OUT, exist_ok=True)

    # one multi-node job: SLURM_PROCID/SLURM_NPROCS; fallback to array vars; else 0/1
    tid   = int(os.environ.get("SLURM_PROCID",
                os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ntask = int(os.environ.get("SLURM_NPROCS",
                os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    kc_kw = {} if KCHUNK == 0 else {"k_chunk_size": KCHUNK}

    fn = os.path.join(OUT, f"slice_{tid:04d}.npz")
    if os.path.exists(fn) and os.environ.get("ABCMB_SCAN_RESUME", "1") == "1":
        print(f"[task {tid}] {fn} exists -> skip (resume)", flush=True)
        return

    gpus = jax.devices('gpu')
    print(f"[task {tid}/{ntask}] {len(gpus)} GPUs, cache={CACHE_DIR}, "
          f"N={N} B={B} kchunk={KCHUNK or 'auto'} ellmax={ELLMAX} lens={LENS}",
          flush=True)

    grid = make_grid(N, SEED)
    # strided slice: adjacent grid points go to different nodes (load balance if
    # the grid is ordered by, e.g., a varying parameter).
    my = grid[tid::ntask]
    print(f"[task {tid}] owns grid[{tid}::{ntask}] ({len(my)} cosmologies)", flush=True)

    model = Model(user_species=None, output_Cl=True, l_max=ELLMAX, lensing=LENS,
                  output_Pk=True, output_k_max=0.5, l_max_g=12, l_max_pol_g=10,
                  l_max_ur=17, l_max_ncdm=17)

    results, n_done, t_start = [], 0, time.perf_counter()
    for b0 in range(0, len(my), B):
        batch = my[b0:b0 + B]
        valid = len(batch)
        if valid < B:                       # pad to fixed B -> reuse cached HLO
            batch = batch + [batch[-1]] * (B - valid)
        t0 = time.perf_counter()
        out = model.call_batched(batch, shard=True, **kc_kw)
        jax.block_until_ready(out.ClTT)
        dt = time.perf_counter() - t0
        results.append(summarize(out, valid))
        n_done += valid
        print(f"[task {tid}] batch {b0//B}: {valid} cosmo in {dt:.2f}s "
              f"({dt/valid:.3f}s/param)", flush=True)

    # concatenate batches and save
    agg = {}
    for key in ("ClTT", "ClTE", "ClEE", "Pk"):
        agg[key] = np.concatenate([r[key] for r in results], axis=0)
    pkeys = results[0]["params"].keys()
    agg_params = {k: np.concatenate([r["params"][k] for r in results])
                  for k in pkeys}
    np.savez_compressed(fn, **agg, **{f"param_{k}": v for k, v in agg_params.items()})
    tot = time.perf_counter() - t_start
    print(f"[task {tid}] DONE {n_done} cosmo in {tot:.1f}s "
          f"({tot/max(n_done,1):.3f}s/param incl. compile) -> {fn}", flush=True)


if __name__ == "__main__":
    main()

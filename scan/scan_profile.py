"""scan_profile.py — one node's worker for the LCDM x plik-lite frequentist
profile grid scan.

Each invocation owns the strided slice grid[tid::ntask] of the global Cartesian
grid defined in scan/profile_config.py, evaluates it through
Model.call_batched(shard=True) on the node's GPUs in fixed-B padded batches, and
for each cosmology computes the profiled chi^2:

    chi2(point) = min_Aplanck [ chi2_pliklite(model/Aplanck^2) + ((Aplanck-1)/0.0025)^2 ]
                  + sum_priors ((theta - mu)/sigma)^2        # tau lowE prior

Output: scan/out_profile/slice_<tid>.npz with, for each cosmology in this slice,
the profiled chi2, A_planck, the global grid index, and every scanned parameter
value. (Tiny — ~bytes/cosmo — vs the full spectra.)

Mirrors scan_slice.py's harness: ONE multi-node job, one worker/node, strided
slice, fixed-B padding so every call reuses the cached HLO, persistent XLA cache
on $SCRATCH, resumable (skip if slice_<tid>.npz exists). See that file's header.

Env (all optional):
  ABCMB_PROFILE_B       per-call batch, fixed (default 256; memory-limited max
                        at l_max=2508+lensing on 80GB A100)
  ABCMB_PROFILE_OUT     output dir (default scan/out_profile)
  ABCMB_PROFILE_ELLMAX  l_max (default 2508 — plik-lite reaches 2508)
  ABCMB_PROFILE_RESUME  skip workers whose output exists (default 1)
  ABCMB_CACHE_DIR       compile cache dir (default $SCRATCH/.jax_cache_abcmb)
  SLURM_PROCID/SLURM_NPROCS  one multi-node job; else 0/1
"""
import os, time

# --- compile cache MUST be set before jax is imported ---
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

from abcmb.main import Model
from scan.plik_lite import PlikLite
from scan import profile_config as cfg


def main():
    B      = int(os.environ.get("ABCMB_PROFILE_B", 256))
    ELLMAX = int(os.environ.get("ABCMB_PROFILE_ELLMAX", 2508))
    OUT    = os.environ.get("ABCMB_PROFILE_OUT", os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "out_profile"))
    os.makedirs(OUT, exist_ok=True)

    tid   = int(os.environ.get("SLURM_PROCID", 0))
    ntask = int(os.environ.get("SLURM_NPROCS", 1))

    fn = os.path.join(OUT, f"slice_{tid:04d}.npz")
    if os.path.exists(fn) and os.environ.get("ABCMB_PROFILE_RESUME", "1") == "1":
        print(f"[task {tid}] {fn} exists -> skip (resume)", flush=True)
        return

    gpus = jax.devices('gpu')
    Ntot = cfg.n_total()
    print(f"[task {tid}/{ntask}] {len(gpus)} GPUs, B={B}, ellmax={ELLMAX}, "
          f"grid Ntot={Ntot} ({cfg.SCAN_ORDER} = "
          f"{[cfg.NPTS[p] for p in cfg.SCAN_ORDER]})", flush=True)

    grid = cfg.make_grid()
    idx = list(range(tid, Ntot, ntask))   # global flat indices owned here
    my = [grid[i] for i in idx]
    print(f"[task {tid}] owns grid[{tid}::{ntask}] = {len(my)} cosmologies",
          flush=True)

    pl = PlikLite()
    model = Model(user_species=None, output_Cl=True, l_max=ELLMAX, lensing=True,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10,
                  l_max_ur=17, l_max_ncdm=17)

    chi2_out  = np.empty(len(my))
    aplk_out  = np.empty(len(my))
    cols = {p: np.empty(len(my)) for p in cfg.SCAN_ORDER}

    t_start = time.perf_counter()
    for b0 in range(0, len(my), B):
        batch = my[b0:b0 + B]
        valid = len(batch)
        if valid < B:                          # pad to fixed B -> reuse cached HLO
            batch = batch + [batch[-1]] * (B - valid)
        t0 = time.perf_counter()
        out = model.call_batched(batch, shard=True)
        # plik-lite chi^2 with A_planck profiled analytically (incl. its prior)
        Dtt = pl.abcmb_cl_to_Dl(out.ClTT, out.l)
        Dte = pl.abcmb_cl_to_Dl(out.ClTE, out.l)
        Dee = pl.abcmb_cl_to_Dl(out.ClEE, out.l)
        m0 = pl.bin_model(Dtt, Dte, Dee)
        chi2_lik, A_best = pl.profile_A(m0, with_prior=True)
        chi2_lik = np.asarray(jax.block_until_ready(chi2_lik))[:valid]
        A_best   = np.asarray(A_best)[:valid]
        # add explicit Gaussian priors (tau lowE) using the INPUT grid values
        chi2 = chi2_lik.copy()
        for name, (mu, sig) in cfg.GAUSS_PRIORS.items():
            v = np.array([batch[i][name] for i in range(valid)])
            chi2 = chi2 + ((v - mu) / sig) ** 2
        # record
        chi2_out[b0:b0 + valid] = chi2
        aplk_out[b0:b0 + valid] = A_best
        for p in cfg.SCAN_ORDER:
            cols[p][b0:b0 + valid] = [batch[i][p] for i in range(valid)]
        dt = time.perf_counter() - t0
        print(f"[task {tid}] batch {b0//B}: {valid} cosmo in {dt:.1f}s "
              f"({dt/valid:.3f}s/param)  chi2 [{chi2.min():.1f},{chi2.max():.1f}]",
              flush=True)

    gidx = np.array(idx, dtype=np.int64)
    np.savez(fn, gidx=gidx, chi2=chi2_out, A_planck=aplk_out,
             **{f"param_{p}": cols[p] for p in cfg.SCAN_ORDER})
    tot = time.perf_counter() - t_start
    print(f"[task {tid}] DONE {len(my)} cosmo in {tot:.1f}s "
          f"({tot/max(len(my),1):.3f}s/param incl. compile) -> {fn}", flush=True)


if __name__ == "__main__":
    main()

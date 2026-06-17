#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=abcmb_profile_ad
#SBATCH --output=logs/abcmb_profile_ad_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# Config-driven, lockstep frequentist profile TOOL (scan/profile_prod_ad.py;
# TOOL_PLAN.md Workstream B). The FULL Planck low-ell data model: plik-lite
# high-ell TTTEEE + low-ell TT Commander + low-ell EE SRoll2 (NO circular tau prior),
# A_planck envelope-profiled.
#
# PARALLELISM (NEW): ONE lockstep batch across ALL POIs x grid points. The driver
# builds a single flat row set and runs BFGS over all rows at once, sharding the
# batch over the node's 4 GPUs (PA_SHARD=auto). So this is ONE task on ONE node
# with 4 GPUs -- NOT one-POI-per-GPU. (Multi-node = PA_RANK_SLICE=1 + ntasks=nodes,
# one task/node, each task shards its row slice over its 4 GPUs.)
#
# GRADIENTS: the CONFIG's grad_method picks the BFGS iteration gradient (lcdm->fdbatch
# central-FD on the batch axis, Workstream-A verdict; lcdm_neff->ad exact batched-AD for
# the possibly non-convex Neff direction). it0 calibrates fdbatch against exact AD by
# sweeping a STEP LADDER and keeping the BEST (the old halving walked into the FD noise
# floor -- job 54442539). The final ||g||<GTOL stationarity certificate is ALWAYS exact
# AD; the nuisance PD-Hessian is optional (PA_HESS, default 0 -- see below). BFGS is
# preconditioned with the inverse-Fisher Hinv0 from the warm-start Hessian (near-Newton
# first step; without it the ill-conditioned likelihood stalled at ||g||~50). VALUES
# always on the fast call_batched path (consistency rule).
#
# RESUMABLE (two levels, PA_RESUME=1): (1) MID-BFGS -- the lockstep BFGS state is
# checkpointed to results/profile_prod_ad_STATE.npz after EVERY iteration (atomic),
# so a 12h walltime timeout loses at most one iteration; a resubmit reloads it,
# skips calibration+preconditioner, and continues. (2) WHOLE-POI -- a POI whose
# results/profile_prod_ad_<poi>.npz has done=True is skipped. The JAX compile cache
# ($SCRATCH/.jax_cache_abcmb) makes the ~5-min staged compiles one-time across resumes.
#
# QOS: regular (production). NEVER premium (2x). Resize via env before sbatch, e.g.
#   PA_CONFIG=scan/configs/lcdm_neff.py sbatch scan/profile_prod_ad.slurm
# RESUME a timed-out run:
#   sbatch scan/profile_prod_ad.slurm        # same command; done POIs are skipped

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
mkdir -p logs scan/results

export PA_CONFIG=${PA_CONFIG:-scan/configs/lcdm.py}
export PA_LMAX=${PA_LMAX:-2508}
export PA_NPTS=${PA_NPTS:-25}
export PA_NSIG=${PA_NSIG:-3.0}
export PA_MAXIT=${PA_MAXIT:-40}
export PA_GTOL=${PA_GTOL:-3e-2}
export PA_RTOL=${PA_RTOL:-1e-5}
# grad_method comes from the CONFIG file (lcdm->fdbatch, lcdm_neff->ad), NOT here.
# PA_GRADMETHOD is a DEBUG-ONLY override (warns + shadows config) -- do not set it
# for production; leave it unset so the per-physics config choice governs.
export PA_FD_STEP=${PA_FD_STEP:-1e-2}
export PA_FD_CHUNK=${PA_FD_CHUNK:-128}   # cosmologies per primal call_batched (B_local=CHUNK/n_dev); 512 OOM'd job 54362790 on 40GB nodes
# PA_HESS=0: the per-row nuisance PD-Hessian is central-FD of the AD gradient over
# ALL rows (150 x 5cols x 2 x vmap(5 jvps) ~ 7500 single-cosmo jvps, MULTI-HOUR).
# The exact-AD ||g||<GTOL stationarity certificate (always run) IS the convergence
# proof; the PD check is supplementary and near-certain for a smooth LCDM profile.
# Run it cheaply post-hoc on the per-POI argmin rows if a PD certificate is wanted.
export PA_HESS=${PA_HESS:-0}
export PA_SHARD=${PA_SHARD:-auto}
export PA_WARM=${PA_WARM:-1}
export PA_RESUME=${PA_RESUME:-1}
export PA_RANK_SLICE=${PA_RANK_SLICE:-0}   # multi-node RANK_SLICE is BROKEN for the
# final per-POI npz write (every rank clobbers it with NaN-filled rows) -- needs an
# MPI gather before use. Keep 0 (single node + checkpoint/resume) until that lands.

# ONE task, 4 GPUs; the driver shards the lockstep row batch over them.
srun --gpu-bind=none python -u scan/profile_prod_ad.py

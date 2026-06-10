#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --job-name=abcmb_profile
#SBATCH --output=logs/abcmb_profile_%j.out
#
# Production LCDM frequentist profile vs Planck plik-lite. ONE multi-node job
# (NOT an array): each node runs one independent profile_prod.py worker that owns
# POIS[procid::nprocs] (so 6 LCDM POIs split 2/2/2 over 3 nodes) and shards each
# POI's Newton stencils over its own 4 GPUs. Tolerance-converged precision:
# rtol_large_k_PE=1e-5 (scan/tol_converge.py) -> ~2.2x the 1e-4 solve time, hence
# 3 nodes + 6h (per-POI resumable). max_steps_PE=16384 (tighter rtol needs it). No cross-node traffic, no
# jax.distributed. RESUMABLE: a POI whose results/profile_prod_<poi>.npz has
# done=True is skipped. The in-process JIT cache makes each worker pay the
# ~3 min compile once (the persistent $SCRATCH cache does not capture sharded
# executables, but the single per-process compile is negligible vs the run).
#
# QOS: regular (production). NEVER premium (2x charge). Resize via env before sbatch:
#   PROD_POIS, PROD_NPTS, PROD_NSIG, PROD_ITERS, PROD_BLOCAL.
#   e.g.  PROD_NPTS=9 sbatch scan/profile_prod.slurm

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
mkdir -p logs scan/results

export PROD_LMAX=${PROD_LMAX:-2508}
export PROD_NPTS=${PROD_NPTS:-13}
export PROD_NSIG=${PROD_NSIG:-3.0}
export PROD_ITERS=${PROD_ITERS:-5}
export PROD_BLOCAL=${PROD_BLOCAL:-95}
export PROD_POIS=${PROD_POIS:-h,omega_b,omega_cdm,n_s,ln10As,tau_reion}
export PROD_RESUME=${PROD_RESUME:-1}
# PE solver tolerance for the high-ell modes; tighten for tolerance-converged
# precision (set by scan/tol_converge.py). Default keeps the ABCMB 1e-4.
export PROD_RTOL_LARGE=${PROD_RTOL_LARGE:-1e-5}
export PROD_MAXSTEPS=${PROD_MAXSTEPS:-16384}

# One task per node; each owns all 4 GPUs and shards its POIs over them.
# Do NOT pin CUDA_VISIBLE_DEVICES per local rank -- that disables B-sharding.
srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/profile_prod.py

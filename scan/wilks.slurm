#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --job-name=abcmb_wilks
#SBATCH --output=logs/abcmb_wilks_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# Wilks-theorem / coverage validation (scan/wilks.py; TOOL_PLAN.md Workstream E).
# Generates WK_NMOCK Gaussian mock realizations of the plik-lite TTTEEE data vector
# (+ a mockable Gaussian tau prior standing in for the non-Gaussian low-ell EE, +
# a mockable A_planck calibration) from a truth cosmology's Cls, re-fits each mock on
# the BATCH axis with mean-Jacobian Gauss-Newton, and compares the profile-likelihood
# test statistic t = chi2_cond(POI=truth) - chi2_global to chi2_1 (Wilks).  The
# empirical coverage frac(t<1)=0.683, frac(t<3.84)=0.95 is the deliverable.
#
# PARALLELISM: ONE task per node, 4 GPUs/node; each rank shards its mock slice over the
# node's GPUs.  Multi-node = the mock population is sliced across ranks (WK_RANK_SLICE
# auto-on when NPROC>1); each rank writes wilks_<config><tag>_r<rank>.npz; the trailing
# collect step merges them.  For NMOCK=500, 1 node ~2 h/config; --nodes=2 ~halves it.
#
# Both configs (run twice; <=2 queued jobs get priority):
#   WK_CONFIG=scan/configs/lcdm.py       sbatch scan/wilks.slurm
#   WK_CONFIG=scan/configs/lcdm_neff.py  sbatch scan/wilks.slurm
# Scale: --nodes=2 sbatch scan/wilks.slurm   (mock slice auto-splits across nodes)

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
mkdir -p logs scan/results

export WK_CONFIG=${WK_CONFIG:-scan/configs/lcdm.py}
export WK_NMOCK=${WK_NMOCK:-500}
export WK_LMAX=${WK_LMAX:-2508}
export WK_RTOL=${WK_RTOL:-1e-5}
export WK_SEED=${WK_SEED:-0}
export WK_TAG=${WK_TAG:-}
export WK_SHARD=${WK_SHARD:-auto}
export WK_VALIDATE=${WK_VALIDATE:-16}   # per-mock-Jacobian certificate subsample (per rank)
# tau prior width defaults to the config sig[tau_reion] (the Planck tau width); the GN
# fitter settings (WK_JAC_STEP, WK_GN_MAXIT, WK_GN_TOL, WK_STEP_CAP, WK_EVAL_CHUNK)
# keep their wilks.py defaults.

srun --gpu-bind=none python -u scan/wilks.py

# ---- merge slices + coverage table + plots (CPU; safe to fail without losing the run) ----
CFGNAME=$(basename "${WK_CONFIG}" .py)
srun --nodes=1 --ntasks=1 --gpus-per-node=4 --gpu-bind=none \
     python -u scan/wilks_collect.py "${CFGNAME}${WK_TAG}" || \
     echo "[wilks.slurm] collect failed; run 'python scan/wilks_collect.py ${CFGNAME}${WK_TAG}' manually"

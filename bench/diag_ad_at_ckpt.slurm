#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=diag_ad_ckpt
#SBATCH --output=logs/diag_ad_ckpt_%j.out
#
# Debug-queue diagnostic: is the gate BFGS stall FD noise or real ill-conditioning?
# Evaluates the exact AD gradient at the timed-out checkpoint's stalled position
# and compares to the stored FD gradient. See bench/diag_ad_at_ckpt.py.

module load conda
conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}

srun --gpu-bind=none python -u bench/diag_ad_at_ckpt.py

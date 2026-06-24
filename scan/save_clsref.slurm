#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#SBATCH --job-name=save_clsref
#SBATCH --output=logs/save_clsref_%j.out
#
# FOUNDATIONAL GATE for the full-plik switch: confirm clipy (JAX-native clik
# reimplementation) imports in actdr6, self-tests, and works with ABCMB theory Cls
# (sane chi2, differentiable, vmappable). Debug queue (<=30 min). 1 GPU is enough
# (single ABCMB call). See scan/save_clsref.py.

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
# repo first (so `import abcmb` is THIS checkout), then the clipy package dir
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
mkdir -p logs

srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
     python scan/save_clsref.py

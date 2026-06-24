#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=val_fullplik_grad
#SBATCH --output=logs/val_fullplik_grad_%j.out
#
# Integration gate: full-plik value path + staged-AD consistency + envelope-theorem
# cosmology gradient (AD vs FD) via the actual driver. 1 GPU (B=4 tiny). Debug.

module load conda
conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
export PA_CONFIG=scan/configs/lcdm_plikfull.py
mkdir -p logs scan/results
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 python scan/validate_fullplik_grad.py

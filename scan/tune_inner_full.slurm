#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#SBATCH --job-name=tune_inner_full
#SBATCH --output=logs/tune_inner_full_%j.out
module load conda; conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
export TI_FULL=1 TI_MAXITS=8,12,20 TI_EIGFLOORS=0.01
mkdir -p logs
srun --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 python scan/tune_inner.py

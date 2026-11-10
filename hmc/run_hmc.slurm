#!/bin/bash
#SBATCH --account=m5403_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=abcmb_hmc
#SBATCH --output=logs/abcmb_hmc_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# Lockstep HMC of LCDM (massless nu, Planck lite + low-ell) on one node (hmc/run_lcdm.py; knobs in its docstring).
# Resumable from hmc/results/lcdm${HMC_TAG}.npz: resubmit the same command.
module load conda
conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false EQX_ON_ERROR=nan
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
export HMC_TAG=${HMC_TAG:-_v1}
mkdir -p logs hmc/results
srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 python -u hmc/run_lcdm.py

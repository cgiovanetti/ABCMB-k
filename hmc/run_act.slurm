#!/bin/bash
#SBATCH --account=m5403_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=abcmb_hmc_act
#SBATCH --output=logs/abcmb_hmc_act_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# Lockstep HMC of ACT DR6 LCDM on one node (hmc/run_act.py; knobs in its docstring, all taken from the environment).
# Resumable from hmc/results/${HMC_DATA}${HMC_TAG}.npz: resubmit the same command.
# candl + candl_data live in the abcmb-k venv layered on actdr6 (/global/common/software/m3166/carag/envs/abcmb-k).
module load conda
conda activate actdr6
source /global/common/software/m3166/carag/envs/abcmb-k/bin/activate
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONNOUSERSITE=1 PYTHONPATH=$(pwd):$PYTHONPATH OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95} EQX_ON_ERROR=nan
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
export HMC_DATA=${HMC_DATA:-act} HMC_TAG=${HMC_TAG:-_v1}
mkdir -p logs hmc/results
srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 python -u hmc/run_act.py

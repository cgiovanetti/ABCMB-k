#!/bin/bash
# dev runs on an interactive allocation: srun --jobid=<id> -N1 -n1 -c64 --gpus=4 --overlap hmc/srun_dev.sh
module load conda && conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false EQX_ON_ERROR=nan
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
python -u hmc/run_lcdm.py

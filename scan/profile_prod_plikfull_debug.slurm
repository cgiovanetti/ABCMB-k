#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --job-name=plikfull_dbg
#SBATCH --output=logs/plikfull_dbg_%j.out
#
# DEBUG-scale end-to-end full-plik profile: confirm the whole driver loop (warm start,
# lockstep BFGS, sigma1 early-stop, AD ||g|| certificate, npz/png writing) works with
# full plik before committing the multi-hour headline. 1 POI, tiny grid, few iters.
# The 20-min AD compile is CACHED from validate_fullplik_grad, so this fits in 30 min.

module load conda
conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
mkdir -p logs scan/results

export PA_CONFIG=${PA_CONFIG:-scan/configs/lcdm_plikfull.py}
export PA_POIS=${PA_POIS:-tau_reion}
export PA_NPTS=${PA_NPTS:-5}
export PA_NSIG=${PA_NSIG:-2.0}
export PA_MAXIT=${PA_MAXIT:-4}
export PA_GTOL=${PA_GTOL:-0.03}
export PA_SIGTOL=${PA_SIGTOL:-1e-2}
export PA_HESS=${PA_HESS:-0}
export PA_TAG=${PA_TAG:-_dbg}

srun --ntasks=1 --gpus-per-node=4 --cpus-per-task=64 python -u scan/profile_prod_ad.py

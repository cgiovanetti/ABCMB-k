#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --job-name=estop_ref
#SBATCH --output=logs/estop_ref_%j.out
#
# VALIDATION of the chi2-plateau early-stop (scan/profile_prod_ad.py).
# Runs ONE POI (ln10As -- worst-conditioned, slowest-settling => the right stress
# test) at low l with PA_FTOL=0 (NO early stop, runs to MAXIT) and dumps the
# per-iter best_f history via PA_BF_TRACE. Post-hoc: bench/analyze_estop.py replays
# the trigger at a sweep of (FTOL,PATIENCE) and reports Dsigma1 vs the converged
# interval. DEBUG qos (<=30 min). Resumable: a walltime kill keeps the per-iter
# checkpoint + the partial trace; resubmit accumulates more iters.
#
# Resubmit (accumulate more iters): sbatch scan/validate_estop.slurm

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
mkdir -p logs scan/results

# fail fast on any syntax error before burning GPU time
python -m py_compile scan/profile_prod_ad.py bench/analyze_estop.py || exit 1

export PA_CONFIG=${PA_CONFIG:-scan/configs/lcdm.py}
export PA_POIS=${PA_POIS:-ln10As}
export PA_LMAX=${PA_LMAX:-900}
export PA_NPTS=${PA_NPTS:-7}
export PA_NSIG=${PA_NSIG:-3.0}
export PA_MAXIT=${PA_MAXIT:-20}
export PA_GTOL=${PA_GTOL:-3e-2}
export PA_FTOL=0                                   # reference: NO early stop
export PA_HESS=0
export PA_SHARD=auto
export PA_WARM=1
export PA_RESUME=1
export PA_TAG=${PA_TAG:-_estopref}
export PA_BF_TRACE=${PA_BF_TRACE:-scan/results/bf_trace_ln10As.npz}

srun --gpu-bind=none python -u scan/profile_prod_ad.py
echo "=== post-hoc early-stop analysis ==="
python bench/analyze_estop.py $PA_BF_TRACE

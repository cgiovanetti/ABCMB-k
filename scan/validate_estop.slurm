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
# Uses the FAST low-ell-only proxy (scan/configs/lowl_proxy.py + PA_USE_PLIK=0):
# tau_reion profiled over ln10As, lowTT+lowEE only, LMAX=100 -> seconds/iter, so a
# full converge+plateau fits ONE 30-min debug job. (plik-lite needs l~2508; it is
# meaningless truncated -- a LMAX=900 attempt gave chi2~185000 garbage.) The early-
# stop is l-INDEPENDENT machinery, so the proxy is a faithful trigger test and STILL
# exercises the key gotcha: edge +/-3sigma POI rows settle later than central rows.
# Runs with PA_FTOL=0 (NO early stop, full MAXIT) and dumps the per-iter best_f
# history via PA_BF_TRACE. Post-hoc: bench/analyze_estop.py replays the trigger over
# a (FTOL,PATIENCE) sweep and reports Dsigma1 vs the converged interval. DEBUG qos.
# Resumable: a walltime kill keeps the checkpoint + partial trace (it accumulates).
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

export PA_CONFIG=${PA_CONFIG:-scan/configs/lowl_proxy.py}
export PA_USE_PLIK=0                               # low-ell only (proxy needs no high-ell)
export PA_POIS=${PA_POIS:-tau_reion}
export PA_LMAX=${PA_LMAX:-100}                     # lowTT/lowEE need only l<=29
export PA_NPTS=${PA_NPTS:-7}
export PA_NSIG=${PA_NSIG:-3.0}
export PA_MAXIT=${PA_MAXIT:-12}
# GTOL unreachably small: the 2-param proxy is cond=1.0 and would hit GTOL=3e-2 in ONE
# iter (gtol break), leaving no grind for the early-stop to engage. 1e-8 forces the loop
# to run to MAXIT while chi2 plateaus -- reproducing the production regime (||g|| grinds
# below GTOL is impossible, chi2 settles early) that the early-stop is designed for.
export PA_GTOL=${PA_GTOL:-1e-8}
export PA_FTOL=0                                   # reference: NO early stop (full trace)
export PA_HESS=0
export PA_SHARD=auto
export PA_WARM=1
export PA_RESUME=1
export PA_TAG=${PA_TAG:-_estopref}
export PA_BF_TRACE=${PA_BF_TRACE:-scan/results/bf_trace_lowl_tau.npz}

srun --gpu-bind=none python -u scan/profile_prod_ad.py
echo "=== post-hoc early-stop analysis ==="
python bench/analyze_estop.py $PA_BF_TRACE

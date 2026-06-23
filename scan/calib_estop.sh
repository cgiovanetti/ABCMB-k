#!/bin/bash
# Full-l sigma1-stability CALIBRATION / DEMONSTRATION run (1 POI, ln10As -- the
# slowest-settling stress case). Runs the REAL likelihood (plik-lite + lowTT + lowEE)
# at l=2508 to MAXIT with PA_SIGTOL=0 (NO early stop -> a full reference trajectory) and
# dumps the per-iter best_f trace. Post-hoc, bench/analyze_estop.py replays the sigma1-
# stability trigger over a (SIGTOL,PATIENCE) sweep: it shows where the trigger WOULD fire
# (iters_saved = the ~2x) and that the interval there matches the converged one (<0.02sigma).
# That demonstrates the speedup AND fixes the SIGTOL default -- model-agnostic, so it then
# applies to every NEW cosmology on its first run.
#
# Run inside an interactive GPU allocation (4 GPUs ideal; 1 GPU also works, slower):
#   srun --jobid=$JOBID --ntasks=1 --cpus-per-task=32 --gpus-per-task=4 \
#        bash scan/calib_estop.sh
# Resumable: a re-run resumes the per-iter checkpoint AND accumulates the trace.

set -e
module load conda
conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
mkdir -p logs scan/results

export PA_CONFIG=${PA_CONFIG:-scan/configs/lcdm.py}     # the REAL likelihood (full l)
export PA_POIS=${PA_POIS:-ln10As}                       # slowest-settling POI = stress test
export PA_LMAX=${PA_LMAX:-2508}
export PA_NPTS=${PA_NPTS:-7}                            # enough for a clean PCHIP interval
export PA_MAXIT=${PA_MAXIT:-14}                         # long enough to see sigma1 plateau + waste
export PA_GTOL=${PA_GTOL:-3e-2}
export PA_SIGTOL=0                                      # reference: NO early stop (full trace)
export PA_HESS=0
export PA_SHARD=auto
export PA_WARM=1                                        # cached l=2508 warm Hessian -> fast precond
export PA_RESUME=1
export PA_TAG=${PA_TAG:-_calib}
export PA_BF_TRACE=${PA_BF_TRACE:-scan/results/bf_trace_calib.npz}

echo "[calib] starting full-l reference run ($(date))"
python -u scan/profile_prod_ad.py
echo "[calib] === sigma1-stability analysis ==="
python bench/analyze_estop.py "$PA_BF_TRACE"

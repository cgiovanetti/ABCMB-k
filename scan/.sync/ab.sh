#!/bin/bash
# A/B of the profile optimizers on the merged ABCMB: $1 = old | new. 5 rows of the h profile.
module load conda && conda activate actdr6
R=/pscratch/sd/c/carag/ABCMB-k; S=$R/scan/.sync
export PYTHONPATH=$R:/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache_abcmb EQX_ON_ERROR=nan
export PA_CONFIG=$S/cfg_ab.py PA_POIS=h PA_NPTS=5 PA_NSIG=2.5 PA_XBOX=10 PA_WARM_LR=1 PA_MAXIT=12
export PA_GTOL=0.03 PA_RTOL=1e-5 PA_SIGTOL=0 PA_LMAX=2508 PA_RESUME=0 PA_TAG=_ab_$1
cd $R
if [ "$1" = old ]; then
  PA_GRADMETHOD=ad PA_WARM_FROM_CEN=1 PA_BFGS_UPDATE=0 PA_LS_BATCH=4 PA_FD_CHUNK=256 PA_HESS=0 python -u $S/driver_old.py
else
  PA_LM_K=4 PA_VAL_CHUNK=256 python -u $S/driver_new.py
fi

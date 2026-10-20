#!/bin/bash
module load conda && conda activate actdr6
R=/pscratch/sd/c/carag/ABCMB-k; S=$R/scan/.sync
export PYTHONPATH=$R:/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache_abcmb EQX_ON_ERROR=nan
export PA_CONFIG=$S/cfg_ab.py PA_NPTS=5 PA_NSIG=2.5 PA_XBOX=10 PA_LMAX=2508 PA_RTOL=1e-5
cd $R
for s in "$@"; do python -u $S/$s.py; echo "=== $s exit $?"; done

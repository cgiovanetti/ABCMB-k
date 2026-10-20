#!/bin/bash
# lane: run sync_check modes sequentially on the allocation (one node, 4 GPUs)
module load conda && conda activate actdr6
export OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache_abcmb EQX_ON_ERROR=nan
R=/pscratch/sd/c/carag/ABCMB-k; S=$R/scan/.sync
cd $S
for step in "$@"; do
  case $step in
    acc_new)  PYTHONPATH=$R python -u sync_check.py acc new ;;
    acc_old)  PYTHONPATH=$S/tree_main python -u sync_check.py acc old ;;
    prof_new) PYTHONPATH=$R python -u sync_check.py prof new ;;
    prof_old) PYTHONPATH=$S/tree_main python -u sync_check.py prof old ;;
    class)    python -u sync_check.py class ;;
    report)   python -u sync_check.py report ;;
  esac
  echo "=== $step exit $? $(date +%T)"
done

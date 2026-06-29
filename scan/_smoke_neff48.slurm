#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --job-name=neff48_smoke
#SBATCH --output=logs/neff48_smoke_%j.out
#
# Smoke test for the 48h Neff relaunch (job 55235706) + the analysis notebook.
#  (1) Execute the notebook's CODE CELLS against EXISTING _pf data (PA_NB_TAG=_pf) on the
#      compute node (CPU, Agg backend) -> validates the analysis logic end to end.
#  (2) Mini profile run on the SAME path the real job uses: recentered _rc config,
#      PA_WARM_FROM_CEN=1, NO pre-pass, 1 POI, NPTS=11 (cache-HIT shapes -- identical to one
#      POI_SLICE rank of the proven run 54980729), MAXIT=2, early-stop OFF, throwaway tag
#      _smoketest -> confirms compile-cache hit + real iteration + npz write, NO prepass hang.
# Run (1) first so the notebook check always lands even if (2) hits the 30-min wall.

module load conda
conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}

echo "########## (1) NOTEBOOK CELL EXECUTION (PA_NB_TAG=_pf, CPU) ##########"
PA_NB_TAG=_pf python - <<'PY'
import json, matplotlib
matplotlib.use('Agg')
nb = json.load(open('scan/analyze_profile_neff.ipynb'))
g = {}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    print(f'--- exec code cell {i} ---', flush=True)
    exec(compile(src, f'cell{i}', 'exec'), g)
print('NOTEBOOK_CELLS_OK', flush=True)
PY
echo "########## notebook check rc=$? ##########"

echo "########## (2) PROFILE STARTUP (recentered _rc, no prepass, 1 POI, MAXIT=2) ##########"
export PA_CONFIG=scan/configs/lcdm_neff_plikfull_rc.py
export PA_WARM_FROM_CEN=1
export PA_MLE_PREPASS=0
export PA_TAG=_smoketest          # throwaway -- deleted after; must NOT touch real _pfrc
export PA_POIS=tau_reion          # one POI
export PA_POI_SLICE=0
export PA_NPTS=11                 # SAME B as the proven run -> compile-cache HIT
export PA_NSIG=3.0
export PA_MAXIT=2                 # just prove it iterates
export PA_SIGTOL=0               # disable early-stop so it actually runs 2 iters
export PA_GTOL=0.03
export PA_RTOL=1e-5
export PA_HESS=0
srun --ntasks=1 --gpus-per-task=4 --cpus-per-task=64 \
     python -u scan/profile_prod_ad.py
echo "########## SMOKE DONE rc=$? ##########"

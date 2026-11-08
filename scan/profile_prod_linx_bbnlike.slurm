#!/bin/bash
#SBATCH --account=m5403_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --job-name=abcmb_pf_bbn
#SBATCH --output=logs/abcmb_pf_bbn_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# _v6bbn: the _v4linx profile plus the BBN likelihood (Y_p, D/H) with LINX's nuisances (tau_n_fac, 12 rate
# shifts) profiled, POIs dealt out over 4 nodes by PA_POI_SLICE. (_v5bbn had the nuisance priors centred on the
# config cen and the frozen lna grid.) Resumable: resubmit the same command.

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
export EQX_ON_ERROR=nan            # a failed ODE solve is a NaN chi2 the step test rejects, not an abort
export PA_TRACE_EVALS=1            # log every evaluated cosmology
mkdir -p logs scan/results

export PA_CONFIG=scan/configs/lcdm_dneff_linx_bbnlike_plikfull.py   # grid: npts / nsig from the config
export PA_MLE_PREPASS=${PA_MLE_PREPASS:-0}   # the config cen is the prepass result already
export PA_TAG=${PA_TAG:-_v6bbn}
export PA_POIS=${PA_POIS:-h,omega_cdm,Delta_Neff_init,n_s,omega_b,ln10As,tau_reion}
export PA_POI_SLICE=1
export PA_XBOX=10                  # wide grids push h/Neff valley shifts past 5 scaling-sigma
export PA_WARM_LR=1
export PA_AD_BCHUNK=32             # 8 rows per GPU: 16 with 19 directions OOMed a 40 GB A100
export PA_MAXIT=${PA_MAXIT:-30}

srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/profile_prod_ad.py

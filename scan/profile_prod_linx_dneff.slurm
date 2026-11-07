#!/bin/bash
#SBATCH --account=m5403_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --job-name=abcmb_pf_linx
#SBATCH --output=logs/abcmb_pf_linx_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# _v4linx: 7-POI LCDM + Delta_Neff_init profile (full plik + low-ell, YHe/Neff/T_nu from LINX per row), POIs dealt
# out over 4 nodes by PA_POI_SLICE. Resumable: after a walltime timeout resubmit the same command.

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

export PA_CONFIG=scan/configs/lcdm_dneff_linx_plikfull.py   # grid: npts / nsig from the config
export PA_TAG=${PA_TAG:-_v4linx}
export PA_POIS=${PA_POIS:-h,omega_cdm,Delta_Neff_init,n_s,omega_b,ln10As,tau_reion}
export PA_POI_SLICE=1
export PA_XBOX=10                  # wide grids push h/Neff valley shifts past 5 scaling-sigma
export PA_WARM_LR=1
export PA_MAXIT=${PA_MAXIT:-30}

srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/profile_prod_ad.py

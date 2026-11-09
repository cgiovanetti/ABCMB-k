#!/bin/bash
#SBATCH --account=m5403_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=abcmb_pf_emwimp
#SBATCH --output=logs/abcmb_pf_emwimp_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# EM-coupled WIMP (scan/configs/lcdm_mchi_dneff_linx_bbnlike_plikfull.py): the _v6bbn data and LINX nuisances,
# LCDM + Delta_Neff_init + log10_m_chi. 1D: the 8 POIs dealt out over 4 nodes (PA_POI_SLICE), tag _v1emwimp.
# 2D: sbatch --export=ALL,PA_POI2D=log10_m_chi,Delta_Neff_init,PA_TAG=_v1emwimp2d,PA_AD_BCHUNK=32 scan/profile_prod_emwimp.slurm
# (the config's grid2d rows dealt out over the nodes, one part npz per node). Resumable: resubmit the same command.

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # without it JAX caps at 75% of the GPU: 20 directions x 8 rows per GPU
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
export EQX_ON_ERROR=nan            # a failed ODE solve is a NaN chi2 the step test rejects, not an abort
export PA_TRACE_EVALS=1            # log every evaluated cosmology
mkdir -p logs scan/results

export PA_CONFIG=scan/configs/lcdm_mchi_dneff_linx_bbnlike_plikfull.py
export PA_MLE_PREPASS=${PA_MLE_PREPASS:-0}   # the config cen is the _v6bbn best fit
export PA_TAG=${PA_TAG:-_v1emwimp}
export PA_POIS=${PA_POIS:-h,omega_cdm,Delta_Neff_init,n_s,omega_b,ln10As,tau_reion,log10_m_chi}
export PA_POI_SLICE=1
export PA_XBOX=10                  # wide grids push h/Neff valley shifts past 5 scaling-sigma
export PA_WARM_LR=1
export PA_AD_BCHUNK=${PA_AD_BCHUNK:-24}   # 1D: 6 rows per GPU (8 x 20 directions peaked at 37.9 of 40 GB); 2D: pass 32
export PA_STEP_MAX=${PA_STEP_MAX:-2.0}   # rows that start far (light chi) reach their optimum in half the iterations
export PA_MAXIT=${PA_MAXIT:-30}

srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/profile_prod_ad.py

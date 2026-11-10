#!/bin/bash
#SBATCH --account=m5403_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=20:00:00
#SBATCH --job-name=abcmb_pf_act
#SBATCH --output=logs/abcmb_pf_act_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# _v1act: 6-POI LCDM profile with ACT DR6 lite (candl; A_act, P_act profiled) + SRoll2 lowE (scan/configs/
# lcdm_actlite.py), one POI per node (6 nodes, PA_POI_SLICE). Engine at the ACT l range (lensed 6325,
# delta_l_max 5000, hierarchy 40/60, halofit, streamed lensing): ~9x the Planck cost per row, 8 rows per GPU per
# gradient block,
# ~3.6 min per row-iteration per node (Planck: ~0.5). Resumable: after a walltime timeout resubmit the same command.
# candl lives in the venv /global/common/software/m3166/carag/envs/abcmb-k (layered on actdr6).

module load conda
conda activate actdr6
source /global/common/software/m3166/carag/envs/abcmb-k/bin/activate

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONNOUSERSITE=1
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
export EQX_ON_ERROR=nan            # a failed ODE solve is a NaN chi2 the step test rejects, not an abort
export PA_TRACE_EVALS=${PA_TRACE_EVALS:-1}
mkdir -p logs scan/results

export PA_CONFIG=${PA_CONFIG:-scan/configs/lcdm_actlite.py}
export PA_MLE_PREPASS=${PA_MLE_PREPASS:-1}
export PA_TAG=${PA_TAG:-_v1act}
export PA_POIS=${PA_POIS:-h,omega_cdm,n_s,omega_b,ln10As,tau_reion}
export PA_POI_SLICE=1
export PA_XBOX=10
export PA_WARM_LR=${PA_WARM_LR:-1}
export PA_MAXIT=${PA_MAXIT:-30}
export PA_AD_BCHUNK=${PA_AD_BCHUNK:-32}      # 8 rows per GPU x 5 directions
export PA_GRAD_SPECB=${PA_GRAD_SPECB:-2}     # spectrum-stage sub-batches (LoS / halofit intermediates ~3 GB per row)
export PA_VAL_CHUNK=${PA_VAL_CHUNK:-32}       # 8 values per GPU (16 per GPU reach 33 GB of 40)

srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/profile_prod_ad.py

#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=abcmb_smc_plikfull
#SBATCH --output=logs/abcmb_smc_plikfull_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# FAST/SLOW tempered SMC on the FULL Planck plik (clipy), MARGINALISING the 21
# foreground/calibration nuisances (scan/smc_plikfull.py; the nuisance-marginalised
# Bayesian anchor / feasibility study). Particle = (6 cosmo + 21 nuisance)-D.
#
# THE SPEED LEVER (fast/slow, like Cobaya oversampling): a nuisance-only move does
# NOT change the theory Cls, so it does NOT re-run ABCMB -- it reuses the cached
# per-particle clik C_l block and only re-evaluates clipy (the high-ell data chi2).
# Metropolis-within-Gibbs: SLOW block = cosmo (MOVES_SLOW theory solves/stage),
# FAST block = nuisances (MOVES_FAST clipy-only moves/stage, ~free). Cost per stage
# ~ MOVES_SLOW * N theory solves, so marginalising 21 nuisances costs about the same
# ABCMB time as the plik-LITE run (scan/smc.py) did.
#
# SAME data model as the frequentist full-plik profiles: full plik TTTEEE + low-ell
# TT (Commander) + low-ell EE (SRoll2), rtol_large_k_PE=1e-5, l_max=2508 lensed.
#
# MEMORY: per-device working set set by EVAL_CHUNK/n_dev (ABCMB) and CLIPY_CHUNK
# (clipy), NOT by N, so N can grow without OOM. Defaults sized for 40 GB A100s.
#
# RESUMABLE: state persisted every tempering stage to <SMC_OUT>_state.npz (the big
# per-particle Cls caches are NOT stored -- they are recomputed from the saved
# particles on resume, one theory round). A resubmit auto-detects that file.
#
# QOS: regular (production). NEVER premium (2x). NO job arrays.
# RESUME a timed-out run:  sbatch scan/smc_plikfull.slurm   (auto-resumes from _state.npz)

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
mkdir -p logs scan/results

export SMC_CONFIG=${SMC_CONFIG:-scan/configs/lcdm_plikfull.py}
export SMC_N=${SMC_N:-512}
export SMC_EVAL_CHUNK=${SMC_EVAL_CHUNK:-128}    # ABCMB cosmologies / call_batched (B_local=CHUNK/n_dev)
export SMC_CLIPY_CHUNK=${SMC_CLIPY_CHUNK:-32}   # nuisances / clipy batched eval
export SMC_MOVES_SLOW=${SMC_MOVES_SLOW:-2}      # cosmo (theory) moves / stage
export SMC_MOVES_FAST=${SMC_MOVES_FAST:-20}     # nuisance (clipy-only) moves / stage
export SMC_ESS_TARGET=${SMC_ESS_TARGET:-0.5}
export SMC_LMAX=${SMC_LMAX:-2508}
export SMC_RTOL=${SMC_RTOL:-1e-5}
export SMC_SEED=${SMC_SEED:-0}
export SMC_OUT=${SMC_OUT:-scan/results/smc_plikfull}
export SMC_SHARD=${SMC_SHARD:-auto}
# SMC_RESUME left unset: smc_plikfull.py auto-detects ${SMC_OUT}_state.npz if present.

srun --gpu-bind=none python -u scan/smc_plikfull.py

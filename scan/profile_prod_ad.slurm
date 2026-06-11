#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --job-name=abcmb_profile_ad
#SBATCH --output=logs/abcmb_profile_ad_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# AD-gradient, convergence-gated LCDM frequentist profile vs the FULL Planck low-ell
# data model (plik-lite high-ell TTTEEE + low-ell TT Commander + low-ell EE SRoll2;
# NO circular tau prior). scan/profile_prod_ad.py.
#
# PARALLELISM: one POI per GPU-TASK (NOT per node). The AD gradient path
# (PA_GRADMETHOD=loop) is single-GPU, so we pin 1 GPU per task (4 tasks/node) and
# split the 6 LCDM POIs across the global ranks (PA_SHARD=0 -> call_batched values
# also stay single-GPU). 2 nodes = 8 GPU-tasks -> all 6 POIs run concurrently, one
# per GPU. RESUMABLE: a POI whose results/profile_prod_ad_<poi>.npz has done=True is
# skipped.
#
# THROUGHPUT NOTE: the AD gradient goes through the un-batched single-cosmology path
# (~85 s/gradient/point), so cost ~ NPTS * 85 s * n_BFGS_iters per POI. PA_NPTS=9 +
# ~12 iters ~ 2.5 h/POI; the Fisher Hessian (PA_HESS) adds ~NPTS*10*85 s, so it is
# OFF by default here (the per-POI ||grad|| gate is the stationarity evidence).
# Fast batched AD (differentiating call_batched) is the throughput follow-up.
#
# QOS: regular (production). NEVER premium (2x). Resize via env before sbatch, e.g.
#   PA_NPTS=13 PA_HESS=1 sbatch scan/profile_prod_ad.slurm
# Quick debug (1 POI, debug qos, 30 min): use a separate --qos=debug submit.

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
mkdir -p logs scan/results

export PA_LMAX=${PA_LMAX:-2508}
export PA_NPTS=${PA_NPTS:-9}
export PA_NSIG=${PA_NSIG:-3.0}
export PA_MAXIT=${PA_MAXIT:-30}
export PA_GTOL=${PA_GTOL:-3e-2}
export PA_RTOL=${PA_RTOL:-1e-5}
export PA_GRADMETHOD=${PA_GRADMETHOD:-loop}
export PA_HESS=${PA_HESS:-0}
export PA_SHARD=${PA_SHARD:-0}
export PA_USE_LOWEE=${PA_USE_LOWEE:-1}
export PA_USE_LOWTT=${PA_USE_LOWTT:-1}
export PA_POIS=${PA_POIS:-h,omega_b,omega_cdm,n_s,ln10As,tau_reion}
export PA_RESUME=${PA_RESUME:-1}

# one task per GPU; SLURM binds 1 GPU/task. The driver splits POIs by SLURM_PROCID.
srun --gpu-bind=closest python -u scan/profile_prod_ad.py

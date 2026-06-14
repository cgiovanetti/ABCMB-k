#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=abcmb_smc_lcdm
#SBATCH --output=logs/abcmb_smc_lcdm_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# Hand-rolled tempered SMC posterior + evidence on the batched ABCMB likelihood
# (scan/smc.py; TOOL_PLAN.md Workstream D — the Bayesian anchor). SAME data model
# as the frequentist profiles: plik-lite high-ell TTTEEE (A_planck ENVELOPE-PROFILED)
# + low-ell TT Commander + low-ell EE SRoll2, rtol_large_k_PE=1e-5, l_max=2508 lensed.
#
# EAGER PYTHON loop around Model.call_batched: the pipeline is GPU->CPU-HyRex->GPU and
# is NOT end-to-end jittable, so the SMC tempering loop is hand-rolled numpy. Each
# likelihood eval of the N-particle population is done in SMC_EVAL_CHUNK-sized
# (4-GPU sharded) call_batched chunks: per-device memory is set by EVAL_CHUNK/n_dev,
# NOT by N, so the population can grow without OOM. Each chunk is padded to
# EVAL_CHUNK -> ONE compile of the call_batched B aval, reused across chunks/stages.
# (Default EVAL_CHUNK=128 -> B_local=32 ~12 GB/dev, safe on 40 GB A100s; the prior
# N=512 single-call run 54369057 OOM'd at B_local=128 ~49 GB on regular-queue nodes.)
#
# RESUMABLE: state persisted to <SMC_OUT>_state.npz every tempering stage. A resubmit
# auto-detects that file (SMC_RESUME also accepts an explicit path) and continues from
# the saved beta/logZ/particles/rng. The JAX compile cache makes the staged compiles
# one-time across resumes.
#
# QOS: regular (production). NEVER premium (2x). NO job arrays.
# RESUME a timed-out run:  sbatch scan/smc_lcdm.slurm   (auto-resumes from _state.npz)

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
mkdir -p logs scan/results

export SMC_CONFIG=${SMC_CONFIG:-scan/configs/lcdm.py}
export SMC_N=${SMC_N:-512}
export SMC_EVAL_CHUNK=${SMC_EVAL_CHUNK:-128}   # cosmologies per call_batched (B_local=CHUNK/n_dev)
export SMC_MOVES=${SMC_MOVES:-3}
export SMC_ESS_TARGET=${SMC_ESS_TARGET:-0.5}
export SMC_LMAX=${SMC_LMAX:-2508}
export SMC_RTOL=${SMC_RTOL:-1e-5}
export SMC_SEED=${SMC_SEED:-0}
export SMC_OUT=${SMC_OUT:-scan/results/smc_lcdm}
export SMC_SHARD=${SMC_SHARD:-auto}
# SMC_RESUME left unset: smc.py auto-detects ${SMC_OUT}_state.npz if it exists.

# ONE task, 4 GPUs; the N-particle batch shards over them inside call_batched.
srun --gpu-bind=none python -u scan/smc.py

#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --job-name=abcmb_scan
#SBATCH --output=logs/abcmb_scan_%j.out
#
# ONE multi-node job (NOT a job array). Each node runs one independent worker
# (scan_slice.py) that owns grid[procid::nprocs] and shards its slice over the
# node's 4 GPUs. There is NO cross-node communication and NO jax.distributed —
# each worker is a self-contained 4-GPU process. Throughput ~ #nodes x per-node.
#
# WHY one job, not an array: Perlmutter is touchy about job arrays and only ~2
# queued jobs gain priority at a time, so a wide --array trickles through. A
# single N-node job is scheduled (and prioritized) as ONE job. To go bigger than
# one job's node/walltime limits, submit sequential jobs: the run is RESUMABLE
# (scan_slice skips slices whose slice_<tid>.npz already exists) and the
# persistent compile cache on $SCRATCH makes the resumed job skip compilation.
#
# Scale: bump --nodes (and re-tune ABCMB_SCAN_N). Tunables (env, before sbatch):
#   ABCMB_SCAN_N (total grid), ABCMB_SCAN_B (fixed batch, default 256),
#   ABCMB_SCAN_KCHUNK (0=auto), ABCMB_SCAN_SEED, ABCMB_SCAN_ELLMAX,
#   ABCMB_SCAN_LENSING, ABCMB_SCAN_RESUME.
#   e.g.  ABCMB_SCAN_N=65536 sbatch --nodes=8 scan/scan_multinode.slurm

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
mkdir -p logs scan/out

export ABCMB_SCAN_N=${ABCMB_SCAN_N:-4096}
export ABCMB_SCAN_B=${ABCMB_SCAN_B:-256}
export ABCMB_SCAN_KCHUNK=${ABCMB_SCAN_KCHUNK:-0}

# One task per node, each task owns all 4 GPUs on its node and shards over them.
# IMPORTANT: do NOT pin CUDA_VISIBLE_DEVICES=$SLURM_LOCALID — that gives the
# process a single GPU and silently disables the B-axis sharding.
srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/scan_slice.py

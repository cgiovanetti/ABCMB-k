#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
# QOS policy (see top-level /pscratch/sd/c/carag/CLAUDE.md): NEVER use premium
# (2x charge). 'debug' = 30 min max / <=8 nodes / <=5 jobs, cheap & fast — used
# here for the validation run. For a big production grid, override to regular:
#   sbatch --qos=regular --time=HH:MM:SS scan/profile.slurm   (resumable).
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --job-name=abcmb_profile
#SBATCH --output=logs/abcmb_profile_%j.out
#
# Frequentist profile of 6-param LCDM against Planck plik-lite.
# ONE multi-node job (NOT an array). Each node runs scan_profile.py, owns
# grid[procid::nprocs] of the Cartesian grid in scan/profile_config.py, and
# shards its slice over the node's 4 GPUs (GSPMD, no cross-node traffic).
# Resumable: a re-submitted job skips slices whose slice_<tid>.npz exists and
# reuses the persistent compile cache on $SCRATCH.
#
# Scope here = VALIDATION run (2 nodes, 1 h, coarse grid ~5625 pts). For
# production: edit NPTS in scan/profile_config.py (denser) and bump --nodes /
# --time, then re-submit (resume picks up where it left off).
#
# Tunables (env, before sbatch): ABCMB_PROFILE_B (default 256),
#   ABCMB_PROFILE_ELLMAX (2508), ABCMB_PROFILE_RESUME (1).

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
mkdir -p logs scan/out_profile

export ABCMB_PROFILE_B=${ABCMB_PROFILE_B:-256}
export ABCMB_PROFILE_ELLMAX=${ABCMB_PROFILE_ELLMAX:-2508}

echo "=== profile scan: $(date) — nodes=$SLURM_NNODES B=$ABCMB_PROFILE_B ==="

# one task per node; each owns all 4 GPUs on its node and shards over them.
# Do NOT pin CUDA_VISIBLE_DEVICES — that would disable the B-axis sharding.
srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/scan_profile.py

echo "=== srun done, aggregating: $(date) ==="
# CPU-only numpy aggregation + profile plots (runs on this compute node).
python -u scan/collect_profile.py scan/out_profile

echo "=== profile scan complete: $(date) ==="

#!/bin/bash
#SBATCH --account=m3166
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --job-name=estop_analyze
#SBATCH --output=logs/estop_analyze_%j.out
#
# Pure-numpy post-hoc early-stop analysis (no GPU) on the accumulated proxy trace.
# CPU/shared qos so it does NOT touch GPU debug contention and starts fast.

module load conda
conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
srun python bench/analyze_estop.py ${TRACE:-scan/results/bf_trace_lowl_tau.npz}

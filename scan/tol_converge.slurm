#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:29:00
#SBATCH --job-name=tol_conv
#SBATCH --output=logs/tol_converge_%j.out
#
# Solver-tolerance convergence study, ONE rtol per debug job (so each fits the
# 30-min debug cap and several run concurrently). Pass TC_RTOLS via --export:
#   TC_RTOLS=1e-4 sbatch scan/tol_converge.slurm
#   TC_RTOLS=1e-5 sbatch scan/tol_converge.slurm
#   TC_RTOLS=1e-6 sbatch scan/tol_converge.slurm
# debug qos (NOT premium); separate QOS from gpu_interactive (whose slots are full).

module load conda
conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
mkdir -p logs
export TC_RTOLS=${TC_RTOLS:-1e-4}
echo "TC_RTOLS=$TC_RTOLS"
srun --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 python -u scan/tol_converge.py

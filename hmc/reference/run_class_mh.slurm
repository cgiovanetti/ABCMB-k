#!/bin/bash
#SBATCH --account=m5403
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=hmc_ref_class
#SBATCH --output=logs/hmc_ref_class_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# CLASS + cobaya MH reference for the ABCMB-k HMC run (hmc/reference/lcdm_class_mh.yaml): 8 chains x 16 cores.
# Resumable: resubmit the same command.
module load conda
conda activate actdr6
cd /pscratch/sd/c/carag/ABCMB-k
mkdir -p logs hmc/results/ref
export OMP_NUM_THREADS=16 OMP_PLACES=threads OMP_PROC_BIND=spread
srun --ntasks-per-node=8 --cpus-per-task=32 --cpu-bind=cores cobaya-run ${REF_OUT:+-o $REF_OUT} hmc/reference/lcdm_class_mh.yaml

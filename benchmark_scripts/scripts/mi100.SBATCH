#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:mi100:1
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --job-name=mi100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz1994@nyu.edu
#SBATCH --output=/home/zz1994/packages/ABCMB/benchmark_scripts/out/mi100.out

module purge

singularity exec --nv \
	    --overlay /scratch/zz1994/singgpu/overlay-15GB-500K.ext3:ro \
	    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
	    /bin/bash -c "source /ext3/env.sh;\    
        python /home/zz1994/packages/ABCMB/benchmark_scripts/benchmark.py"
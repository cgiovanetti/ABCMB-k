#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --job-name=abcmb_plikfull
#SBATCH --output=logs/abcmb_plikfull_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# FULL Planck plik (clipy) frequentist profile -- the extended analysis (full plik
# instead of plik-lite, + Neff). MULTI-NODE POI-slice: ONE job, one worker per node,
# each rank owns a DISJOINT subset of POIs (POIS[RANK::NPROC]) and shards its lockstep
# over the node's 4 GPUs. NO job arrays. Resumable (done POIs skipped; BFGS-state +
# JAX compile cache survive a walltime timeout -> resubmit the SAME command).
#
# The ~21 plik foreground/calibration nuisances are profiled INSIDE the likelihood at
# fixed ABCMB theory by a gradient-only SPG (scan/plik_full.py) -- no ABCMB re-run.
#
#   default config = lcdm_neff_plikfull (7-POI LCDM+Neff headline). Override e.g.
#   PA_CONFIG=scan/configs/lcdm_plikfull.py sbatch scan/profile_prod_plikfull.slurm
#   (the 6-POI LCDM full-plik cross-check vs the prior plik-lite headline).

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
# repo FIRST (so import abcmb is THIS checkout), then the clipy package dir
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
mkdir -p logs scan/results

export PA_CONFIG=${PA_CONFIG:-scan/configs/lcdm_neff_plikfull.py}
export PA_POI_SLICE=${PA_POI_SLICE:-1}        # disjoint POIs per node (multi-node scale-out)
export PA_LMAX=${PA_LMAX:-2508}
export PA_NPTS=${PA_NPTS:-11}
export PA_NSIG=${PA_NSIG:-3.0}
export PA_MAXIT=${PA_MAXIT:-18}
export PA_GTOL=${PA_GTOL:-0.03}
export PA_RTOL=${PA_RTOL:-1e-5}
export PA_SIGTOL=${PA_SIGTOL:-1e-2}           # sigma1-stability early-stop (~2.3x), on by default
export PA_HESS=${PA_HESS:-0}                   # skip the per-row nuisance PD-Hessian (AD ||g|| cert suffices)
export PA_TAG=${PA_TAG:-_pf}

# one task per node, each owns its node's 4 GPUs and shards its POI slice over them
srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/profile_prod_ad.py

#!/bin/bash
#SBATCH --account=m3166_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=abcmb_pf_neff
#SBATCH --output=logs/abcmb_pf_neff_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# 7-POI LCDM+Neff full-plik profile (scan/profile_prod_ad.py): Levenberg-Marquardt on the
# Gauss-Newton curvature (AD only), grids of 21 points over +-2.5 grid_sig around the config
# centre (= the joint best fit of an earlier run), linear-response warm start. 4 nodes, POIs
# dealt out by POI_SLICE (2 per node, n_s alone). History of the earlier launches: CHANGELOG.
# Resumable: a walltime timeout -> resubmit the SAME command (done POIs skipped, LM state and
# the JAX compile cache survive).

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
# A failed ODE solve becomes a NaN chi2 that the step test rejects (59041065 aborted all nodes on
# one max_steps hit); the trace logs every evaluated cosmology so a recurrence can be pinned down.
export EQX_ON_ERROR=nan
export PA_TRACE_EVALS=1
mkdir -p logs scan/results

export PA_CONFIG=scan/configs/lcdm_neff_plikfull_rc.py
export PA_TAG=${PA_TAG:-_v3}
export PA_POIS=${PA_POIS:-h,omega_cdm,Neff,n_s,omega_b,ln10As,tau_reion}
export PA_POI_SLICE=1
export PA_LMAX=2508
export PA_NPTS=21
export PA_NSIG=2.5                 # x grid_sig from the config
export PA_XBOX=10                  # wide grids push h/Neff valley shifts past 5 scaling-sigma
export PA_WARM_LR=1
export PA_LM_K=4                   # damping values per iteration (one batched value call)
export PA_VAL_CHUNK=256            # one value call per iteration (4 x <=42 rows)
export PA_MAXIT=${PA_MAXIT:-30}
export PA_GTOL=0.03
export PA_RTOL=1e-5
export PA_SIGTOL=1e-2              # interval-stability early stop

srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/profile_prod_ad.py

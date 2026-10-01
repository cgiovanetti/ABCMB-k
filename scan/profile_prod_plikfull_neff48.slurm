#!/bin/bash
#SBATCH --account=m5403_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --job-name=abcmb_pf_neff
#SBATCH --output=logs/abcmb_pf_neff_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cgiovanetti@lbl.gov
#
# FINAL relaunch of the 7-POI LCDM+Neff FULL-plik headline profile.
#
# WHY THIS FILE EXISTS (the two prior timeouts):
#   1) 54980729 (10h): static LCDM-centered grids. Freeing Neff pulls Neff->~2.83 and
#      the Neff-omega_cdm-H0 degeneracy drags omega_cdm/h down ~3sig/1.4sig, so those
#      POIs railed at the low grid edge, omega_cdm's dchi2=1 crossing fell off-grid (nan),
#      and the nan blocked the sigma1 early-stop -> that rank ran to the wall.
#   2) 55017292 (10h): PA_MLE_PREPASS=1. The pre-pass's single-point full-D AD-through-SPG
#      gradient is a NEW compile signature absent from the warm cache; all 7 ranks printed
#      "is compiling..." and produced ZERO output / ZERO cache writes / ZERO checkpoints
#      for the full 10h (monster compile or hang). The pre-pass is NOT used here.
#   3) 55661112 (_widecdm, 8h, completed but unconverged): the cached warm Hessian was
#      garbage (FD step 0.05 sigma on a rough gradient -> eig -5.4, wrong degeneracy valley),
#      shifted to cond ~1e6 -> rows railed at the +/-5 sigma box. Fixed 2026-09-28: Hessian
#      recomputed at PA_FDH=0.5 (matches the valley of converged rows), Hinv0 floors |eig|
#      at 1.0, cache keyed on SIGMA + FD step. Tag _fixprec.
#
# THIS RUN = the PROVEN compile path (no pre-pass) + the two centering fixes + padding:
#   - PA_CONFIG = lcdm_neff_plikfull_rc.py : grids RECENTERED on the joint MLE (no railing).
#   - PA_WARM_FROM_CEN=1 : warm start + BFGS preconditioner taken at that center (=the MLE).
#   - hardened sigma1 early-stop (an off-grid POI no longer blocks the stop) is in the driver.
#   - PA_TAG=_pfrc : FRESH tag so RESUME does NOT skip the stale mis-centered _pf npz (Jun25).
#   - --time=48:00:00 : generous padding (the proven path needs only a few hours/POI).
#   - 7 nodes = 7 POIs -> POI_SLICE gives exactly 1 POI/node (balanced, no straggler).
# Resumable: a walltime timeout -> resubmit the SAME command (done POIs skipped, BFGS
# state + JAX compile cache survive).

module load conda
conda activate actdr6

cd /pscratch/sd/c/carag/ABCMB-k
export PYTHONPATH=$(pwd):/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy:$PYTHONPATH
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPICH_GPU_SUPPORT_ENABLED=1
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/.jax_cache_abcmb}
# 59041065 aborted all 7 nodes at 26 min: one ODE solve hit max_steps (not reproduced in a
# traced rerun). NaN mode turns a failed solve into a NaN chi2 the line search rejects;
# the trace logs every evaluated cosmology so a recurrence can be pinned down (grep nan).
export EQX_ON_ERROR=nan
export PA_TRACE_EVALS=1
mkdir -p logs scan/results

export PA_CONFIG=scan/configs/lcdm_neff_plikfull_rc.py
export PA_WARM_FROM_CEN=1          # warm start + preconditioner at the config center (= joint MLE)
export PA_MLE_PREPASS=0            # explicit: do NOT run the pre-pass (it hung run #2)
export PA_TAG=${PA_TAG:-_fixprec}  # fixed warm-Hessian preconditioner (was _widecdm)
# PA_POIS (csv POI subset) passes through from the submit env if set; keep --nodes = #POIs
export PA_POI_SLICE=1              # 1 disjoint POI per node
export PA_LMAX=2508
export PA_NPTS=11
export PA_NSIG=3.0
export PA_MAXIT=30              # was 18; the 1-sigma trust radius (PA_STEP_MAX) needs a few extra iters on far rows
export PA_GTOL=0.03
export PA_RTOL=1e-5
export PA_SIGTOL=1e-2              # sigma1-stability early-stop (default)
export PA_HESS=0

srun --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 \
     python -u scan/profile_prod_ad.py

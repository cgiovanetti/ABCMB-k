# hmc/: Hamiltonian Monte Carlo with ABCMB-k

LCDM with three massless neutrinos (Neff 3.044, YHe from the BBN table), Planck 2018 plik-lite TTTEEE +
Commander low-ell TT + SimAll low-ell EE, sampled by HMC on ABCMB-k's forward-mode AD gradient.

| file | role |
|---|---|
| `sampler.py` | lockstep multi-chain HMC (numpy): every chain takes the same leapfrog steps, so one leapfrog step = one batched gradient call |
| `posterior.py` | `LCDMPosterior`: logp and its gradient for a batch of points (ABCMB-k staged jvp, `scan/batched_grad.py`; JAX likelihoods) and the Gauss-Newton covariance at a point |
| `plik_lite.py` | JAX plik-lite (restored from tag `pre-trim-2026-11-06`, scan/plik_lite.py) |
| `run_lcdm.py` | driver: Gauss-Newton metric at the Planck best fit, chains drawn from it, warmup, sampling, checkpoint every trajectory (env knobs in its docstring) |
| `analyze.py` | R-hat, ESS, cost per effective sample, comparison with the reference, triangle plot |
| `test_sampler_toy.py` | CPU tests of the sampler on Gaussians (moments, eps^2 energy error, exactness with a wrong gradient, support) |
| `reference/` | the same posterior by cobaya MH + CLASS 3.3.4 (`lcdm_class_mh.yaml`, `run_class_mh.slurm`, `eval_once.py`) |
| `results/` | outputs (git-ignored) |

## Method

Whitened coordinates z = L^-1 theta with Sigma = L L^T the metric (mass matrix inverse). Each trajectory draws an
integration time T ~ U(1, 2) (pi/2 decorrelates a Gaussian), shared by all chains, and takes ceil(T / eps_max) leapfrog
steps; a Metropolis test per chain on the exact energy change. Warmup only: eps_max is rescaled after each trajectory
by sqrt((1 - 0.8) / (1 - mean acceptance)) (leapfrog energy error ~ eps^2), and the metric becomes the pooled
covariance of all chains' draws at warmup iterations 5 and 10 (first 2 skipped). The starting metric is the
Gauss-Newton curvature at the Planck best fit (plik-lite's J^T C^-1 J, the low-ell terms' exact Hessian in D_ell).

Any deterministic force keeps the leapfrog reversible and volume preserving, so the chains target the posterior
exactly whatever the gradient's accuracy; a wrong gradient only lowers acceptance. The gradient is checked with no
finite differences by the energy error over a fixed time T = 1 (`HMC_MODE=echeck`): it falls as eps^2 only if the
force is the gradient of logp.

## Running

```bash
# dev, on an interactive 4-GPU allocation
HMC_MODE=echeck HMC_NCHAINS=8 HMC_TAG=_dev srun --jobid=<id> -N1 -n1 -c64 --gpus=4 --overlap --export=ALL hmc/srun_dev.sh
HMC_TAG=_v1 srun --jobid=<id> -N1 -n1 -c64 --gpus=4 --overlap --export=ALL hmc/srun_dev.sh   # resumes from results/lcdm_v1.npz
python hmc/analyze.py _v1                                                                     # CPU, on a compute node
```

## ACT DR6 (`posterior_act.py`, `run_act.py`, `run_act.slurm`)

The same sampler on ACT DR6 TT/TE/EE CMB-only (candl's differentiable lite likelihood) + SRoll2 low-ell EE
(`HMC_DATA=act`, the ACT paper's "ACT"; `pact` adds low-ell TT and plik-lite cut at TT 1000 / TE,EE 600 with
A_planck = A_act). Parameters: the 6 LCDM ones + A_act (prior N(1, 0.003)) + P_act; massless neutrinos as above.
candl lives in the venv `/global/common/software/m3166/carag/envs/abcmb-k` (layered on actdr6; the launcher
activates both).

Engine (why, in `hmc/diag/act_*.py` and CHANGELOG 2026-11-10): lensed l_max 6325 (the last l where a kept ACT
window is non-zero), delta_l_max 5000 (unlensed and phiphi to 11325, the ACT paper's CLASS range: chi2 within
0.006 of it; 1800 is up to 0.6 off), photon hierarchy 40/60, halofit, computed j_l, and `lensing_streamed` (the
matrix lensing needs ~60 GB per cosmology under the gradient). Spectrum-stage sub-batches of 2 per GPU.

```bash
HMC_MODE=gn sbatch --qos=debug --time=00:30:00 hmc/run_act.slurm      # one Gauss-Newton step + metric (shakeout)
sbatch hmc/run_act.slurm                                               # HMC_DATA=act HMC_TAG=_v1, 32 chains; resumes
```

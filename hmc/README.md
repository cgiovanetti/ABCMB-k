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

## Result (`_v1`, 2026-11-10)

64 chains, 12 warmup + 20 sampling trajectories: 9.8 h on one 4xA100 node (481 s per gradient call of 64
cosmologies). Acceptance 0.89, no divergences, R-hat 0.99-1.01, ESS 1420-1610 for 1280 draws. Against cobaya MH +
CLASS on the same posterior (`reference/`): means within 0.12 sigma (n_s -0.12, the rest <= 0.06), sds within 4%.
`python hmc/analyze.py _v1` reprints the table and writes `results/lcdm_v1_triangle.png`.

The n_s offset is lensing physics, not the codes: `_v1` lensed with the linear spectrum, the cobaya CLASS reference
with HMcode (cobaya's default). The linear response to that difference reproduces every HMC - MH shift within the
MC error; with the same physics ABCMB-k and high-precision CLASS agree to <= 0.08 sigma (`diag/`, CHANGELOG
2026-11-11). The HMC now lenses with halofit by default (halofit vs HMcode: <= 0.01 sigma here).

## Running

```bash
# dev, on an interactive 4-GPU allocation
HMC_MODE=echeck HMC_NCHAINS=8 HMC_TAG=_dev srun --jobid=<id> -N1 -n1 -c64 --gpus=4 --overlap --export=ALL hmc/srun_dev.sh
HMC_TAG=_v1 srun --jobid=<id> -N1 -n1 -c64 --gpus=4 --overlap --export=ALL hmc/srun_dev.sh   # resumes from results/lcdm_v1.npz
python hmc/analyze.py _v1                                                                     # CPU, on a compute node
```

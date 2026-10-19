<h1 align="center">
ABCMB-k<!-- omit from toc -->
</h1>
<h4 align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

</h4>

An experimental fork of ABCMB, refactored for speed.

## Third-party code
ABCMB's HALOFIT implementation (`abcmb/halofit.py`) is adapted from [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) (MIT License, Copyright (c) 2022 Differentiable Universe Initiative); see `LICENSE-jax_cosmo`.  If you use the non-linear matter power spectrum, we suggest you cite jax-cosmo ([Campagne et al. 2023](https://arxiv.org/abs/2302.05163)) and the HALOFIT papers, [Smith et al. 2003](https://arxiv.org/abs/astro-ph/0207664) and [Takahashi et al. 2012](https://arxiv.org/abs/1208.2701).

## Citation

If you use ABCMB to publish scientific research, we suggest you cite
```
@article{abcmb,
   title={{ABCMB: A Python+JAX Package for the Cosmic Microwave Background Power Spectrum}},
   volume={2026},
   ISSN={1475-7516},
   url={http://dx.doi.org/10.1088/1475-7516/2026/08/078},
   DOI={10.1088/1475-7516/2026/08/078},
   number={08},
   journal={Journal of Cosmology and Astroparticle Physics},
   publisher={IOP Publishing},
   author={Zhou, Zilu and Giovanetti, Cara and Liu, Hongwan},
   year={2026},
   month=Aug, pages={078} }
```




● You probably have used it, just under other names. Gauss–Newton is Newton's method with the true Hessian replaced
  by the Fisher matrix evaluated at the current point; statisticians call that Fisher scoring. Levenberg–Marquardt
  adds a damping knob that leans toward plain downhill steps when a step fails. It's the default algorithm behind
  scipy.optimize.curve_fit when no bounds are given (that one's from memory, not from these searches).

  Examples in the literature:

  - Particle physics: MSTW 2008 PDF fit (Martin, Stirling, Thorne, Watt, arXiv:0901.0002, EPJC 63, 189). They fit
    with Levenberg–Marquardt instead of MINUIT because the global fit has about 50 free parameters, beyond what
    MINUIT handles in practice (around 15). They built the χ² gradient and Hessian from analytic expressions plus
    finite differences for the theory derivatives; we'd get those derivatives from autodiff instead.
  - CMB: Bond, Jaffe & Knox 1998, "Estimating the power spectrum of the CMB" (astro-ph/9708203). Their iterative
    power-spectrum estimator takes Newton–Raphson steps using the Fisher matrix, the expectation of the curvature,
    instead of the curvature itself. That's Fisher scoring, the same idea applied to C_ℓ bandpowers. I'm going by
    secondary descriptions here and haven't read their iteration section directly.
  - Weak lensing: im3shape (Zuntz et al. 2013, arXiv:1302.0183), the DES shear code. It fits bulge and disc galaxy
    models to each galaxy's exposures by Levenberg–Marquardt maximum likelihood, as described in the DES Y1 shape
    catalogue paper (arXiv:1708.01533).

  So the plan for our profiles is Fisher scoring for the cosmological parameters, with a damping fallback. The
  Fisher matrix gets rebuilt at every point from the Cl derivatives that autodiff already gives us, rather than
  being fixed at the centre.

  Sources:
  - MSTW 2008, Parton distributions for the LHC (arXiv:0901.0002)
  - Bond, Jaffe & Knox, Estimating the power spectrum of the CMB (astro-ph/9708203)
  - Knox, Bond & Jaffe, CMB power spectrum estimation (astro-ph/9702110)
  - Maximum-likelihood CMB power spectrum from interferometer data (astro-ph/0201438), describing the BJK 
    Fisher-matrix variant
  - Zuntz et al., im3shape (arXiv:1302.0183)
  - DES Y1 weak lensing shape catalogues (arXiv:1708.01533)
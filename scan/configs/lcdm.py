"""lcdm.py — the canonical 6-parameter LCDM profile config.

Reproduces profile_prod_ad.py's original module-constant setup EXACTLY (the
entry-(a) production data model: plik-lite high-ell TTTEEE + low-ell TT
Commander + low-ell EE SRoll2; A_planck envelope-profiled; tau a free nuisance
constrained by the REAL low-ell EE data, no circular prior).

A config is a plain python dict named CONFIG. The driver (profile_prod_ad.py)
loads it via PA_CONFIG=<path-to-this-file>. Declared fields:

  order        : list[str]   sampled (POI-able) parameter names, in the
                             canonical vector order. D = len(order); each row's
                             P = D-1 nuisance free dims.
  cen          : dict        fiducial (center) value per name in `order`.
  sig          : dict        scale width per name (the sigma scaling for the
                             nuisance coordinates AND the POI-grid half-extent).
  pois         : list[str]   which `order` names get a profile grid by default
                             (overridable by PA_POIS).
  fixed        : dict        ABCMB params held fixed (passed straight through;
                             add_derived_parameters forwards unknown keys).
  user_species : tuple|None  extra Fluid species for new physics (None = LCDM).
  use_lowtt    : bool        include the low-ell TT (Commander) likelihood.
  use_lowee    : bool        include the low-ell EE (SRoll2) likelihood.
  npts         : int         POI grid points (PA_NPTS overrides).
  nsig         : float       POI grid half-extent in sigma (PA_NSIG overrides).
  grad_method  : str         BFGS iteration-gradient method, a PER-PHYSICS-MODEL
                             analysis choice (NOT an env var): "fdbatch" (central
                             FD on the batch axis -- ~2.4-6x cheaper/dir, fine for
                             LCDM) or "ad" (exact batched-AD iterations -- use for
                             non-convex new-physics params where FD truncation is
                             risky). The final stationarity certificate is ALWAYS
                             exact AD regardless. PA_GRADMETHOD is a DEBUG-ONLY
                             override and warns when set.
"""

CONFIG = {
    "order": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion"],
    "cen": {
        "h": 0.6736, "omega_b": 0.02237, "omega_cdm": 0.1200,
        "n_s": 0.9649, "ln10As": 3.044, "tau_reion": 0.0544,
    },
    "sig": {
        "h": 0.0054, "omega_b": 0.00015, "omega_cdm": 0.0012,
        "n_s": 0.0042, "ln10As": 0.014, "tau_reion": 0.0073,
    },
    "pois": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion"],
    "fixed": {
        "Neff": 3.044, "YHe": 0.2454, "TCMB0": 2.34865418e-4,
        "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
        "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
        "exp_reion": 1.5,
    },
    "user_species": None,
    "use_lowtt": True,
    "use_lowee": True,
    "npts": 25,
    "nsig": 3.0,
    "grad_method": "ad",        # was "fdbatch"; the full-l (l=2508) + low-ell 6-POI gate
                                # STALLED on FD: it0 calibration hit the FD noise floor
                                # (best max-rel 1.20e-2 > 1e-2 target) and BFGS plateaued
                                # at ||g||~20 >> GTOL=0.03 with chi2 dead-flat (job
                                # 54600444). Exact batched-AD has no truncation floor and
                                # is actually CHEAPER here (3 staged jvp blocks vs FD's
                                # 12 chunks of 2*P*N evals). See CHANGELOG 2026-06-19.
}

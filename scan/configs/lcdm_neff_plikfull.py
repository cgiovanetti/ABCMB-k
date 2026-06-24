"""lcdm_neff_plikfull.py — LCDM + Neff (D=7) profile config on the FULL Planck plik.

The headline extension run (the two user asks together):
  1. high_ell = "plik_full"  -> the FULL Planck 2018 high-ell plik TTTEEE (clipy
     backend, 47 foreground/calibration nuisances profiled INSIDE the likelihood at
     fixed ABCMB theory -- no ABCMB re-run for nuisances). Replaces plik-lite.
  2. Neff in the sampled `order` (CEN=3.044, SIG=0.2) -> a POI + a free nuisance for
     every other POI's profile. add_derived_parameters resolves the
     Neff / N_nu_massless / YHe triangle, so no theory-code change.

Otherwise identical to configs/lcdm_neff.py (same params/priors/grad_method). Low-ell
TT (Commander) + EE (SRoll2) unchanged. Run via PA_CONFIG=scan/configs/lcdm_neff_plikfull.py.
"""

CONFIG = {
    "order": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"],
    "cen": {
        "h": 0.6736, "omega_b": 0.02237, "omega_cdm": 0.1200,
        "n_s": 0.9649, "ln10As": 3.044, "tau_reion": 0.0544, "Neff": 3.044,
    },
    "sig": {
        "h": 0.0054, "omega_b": 0.00015, "omega_cdm": 0.0012,
        "n_s": 0.0042, "ln10As": 0.014, "tau_reion": 0.0073, "Neff": 0.2,
    },
    "pois": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"],
    "fixed": {
        # Neff REMOVED from FIXED (now sampled). The rest matches configs/lcdm.py.
        "YHe": 0.2454, "TCMB0": 2.34865418e-4,
        "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
        "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
        "exp_reion": 1.5,
    },
    "user_species": None,
    "high_ell": "plik_full",    # <-- FULL plik (clipy) instead of plik-lite
    "use_lowtt": True,
    "use_lowee": True,
    "npts": 25,
    "nsig": 3.0,
    # exact AD iteration gradients for the possibly non-convex Neff direction (no FD
    # truncation risk); the envelope theorem handles the inner-profiled plik nuisances.
    "grad_method": "ad",
}

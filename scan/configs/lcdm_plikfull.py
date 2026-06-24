"""lcdm_plikfull.py — 6-parameter LCDM profile on the FULL Planck plik TTTEEE.

Identical to configs/lcdm.py except high_ell="plik_full" (full Planck 2018 high-ell
plik via clipy; 47 nuisances profiled inside the likelihood at fixed theory) instead
of plik-lite. Use to cross-check the full-plik intervals against the prior plik-lite
6-POI headline (scan/results/profiles_summary_mn6.npz) before the LCDM+Neff run.
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
    "high_ell": "plik_full",    # <-- FULL plik (clipy) instead of plik-lite
    "use_lowtt": True,
    "use_lowee": True,
    "npts": 25,
    "nsig": 3.0,
    "grad_method": "ad",
}

"""lcdm_neff.py — LCDM + Neff (D=7) profile config (Workstream C).

Identical to configs/lcdm.py except Neff is moved out of the FIXED dict into the
sampled `order` set (CEN=3.044, SIG=0.2), so it can be a POI and is a free
nuisance for every other POI's profile. add_derived_parameters already resolves
the Neff / N_nu_massless / YHe triangle from Neff, so no code change is needed.

Workstream B only SHIPS this config; Workstream C runs the headline LCDM+Neff
benchmark with it.
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
    "use_lowtt": True,
    "use_lowee": True,
    "npts": 25,
    "nsig": 3.0,
}

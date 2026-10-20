"""lcdm_dneff_linx_plikfull.py — LCDM + Neff with BBN-consistent YHe from LINX (D=7), FULL plik.

The lcdm_neff_plikfull_rc.py profile with the BBN step done by LINX (config "bbn": "linx", scan/bbn_stage.py):
the POI Neff is replaced by LINX's input Delta_Neff_init, the Delta Neff at the START of BBN (inert radiation
at T_start; it is diluted by e+e- annihilation, so Neff ~ 3.045 + 0.272 * Delta_Neff_init), and YHe, Neff and
T_nu_massless are computed per row instead of YHe = 0.2454 fixed. The driver reports the profile in
Delta_Neff_init and maps it to Neff (derived_neff in the npz).

cen / sig / grid_sig: the _v3 best fit (2026-10-20, massive neutrino evolved) with Neff 2.8211 ->
Delta_Neff_init = (2.8211 - 3.045) / 0.272 = -0.823; sig 0.1 / 0.272 = 0.37; grid_sig 0.225 / 0.272 = 0.83,
so the Neff grid of _v3 (2.26 .. 3.38) is kept. YHe will now sit at the BBN value (~0.247 at this omega_b and
Neff) rather than 0.2454, so the other parameters' best fit moves slightly.
"""
from abcmb import species


CONFIG = {
    "order": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Delta_Neff_init"],
    "cen": {
        "h": 0.65895, "omega_b": 0.022221, "omega_cdm": 0.116773,
        "n_s": 0.958684, "ln10As": 3.041751, "tau_reion": 0.057532, "Delta_Neff_init": -0.823,
    },
    "sig": {
        "h": 0.0054, "omega_b": 0.00015, "omega_cdm": 0.0013,
        "n_s": 0.0042, "ln10As": 0.014, "tau_reion": 0.0073, "Delta_Neff_init": 0.37,
    },
    "grid_sig": {
        "h": 0.0145, "omega_b": 0.00021, "omega_cdm": 0.0032,
        "n_s": 0.0072, "ln10As": 0.0154, "tau_reion": 0.0062, "Delta_Neff_init": 0.83,
    },
    "pois": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Delta_Neff_init"],
    "fixed": {
        "TCMB0": 2.34865418e-4,
        "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
        "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
        "exp_reion": 1.5,
    },
    "user_species": (species.MassiveNeutrino,),
    "bbn": "linx",
    "linx_reaction_net": "key_PRIMAT_2023",
    "high_ell": "plik_full",
    "use_lowtt": True,
    "use_lowee": True,
    "npts": 25,
    "nsig": 3.0,
}

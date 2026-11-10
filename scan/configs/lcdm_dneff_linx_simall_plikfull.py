"""lcdm_dneff_linx_simall_plikfull.py — the _v4linx profile (LCDM + Delta_Neff_init, YHe / Neff / T_nu_massless
from LINX per row, full plik TTTEEE + Commander lowTT) with the Planck 2018 baseline SimAll lowE in place of
SRoll2: the likelihood of Planck 2018 base_nnu plikHM_TTTEEE_lowl_lowE (one 0.06 eV neutrino, YHe from BBN).

cen / sig / grid_sig: as lcdm_dneff_linx_plikfull.py (the _v3 best fit); run with PA_MLE_PREPASS=1, which
recentres the grids on the SimAll joint fit (tau ~0.005 lower than with SRoll2). Launcher:
PA_CONFIG=scan/configs/lcdm_dneff_linx_simall_plikfull.py PA_TAG=_v7simall PA_MLE_PREPASS=1 \
    sbatch scan/profile_prod_linx_dneff.slurm
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
    "lowee": "simall",
    "npts": 21,
    "nsig": 2.5,
}

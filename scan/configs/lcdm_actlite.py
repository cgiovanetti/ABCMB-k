"""lcdm_actlite.py — LCDM profile with ACT DR6 TT/TE/EE CMB-only (candl's lite likelihood, A_act and P_act profiled)
+ Planck SRoll2 low-ell EE: ACT's own "ACT" combination (DR6-ACT-lite act-lcdm.yaml). One 0.06 eV massive neutrino,
Neff 3.044, YHe from the sBBN table, as ACT's LCDM baseline.

Engine (scan/profile_prod_ad.py, high_ell act_lite): lensed l_max 6325, delta_l_max 5000, l_max_g / l_max_pol_g
40 / 60, HALOFIT, computed j_l, streamed lensing.

cen: the ACT + SRoll2 maximum for massless neutrinos (Gauss-Newton from the Planck point, 2026-11-10: chi2 552.893 ->
541.875) with h moved by the 0.06 eV neutrino's -0.0057; run with PA_MLE_PREPASS=1. sig / grid_sig: that fit's
Gauss-Newton marginal errors. Launcher: sbatch scan/profile_prod_act.slurm
"""
from abcmb import species


CONFIG = {
    "order": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion"],
    "cen": {
        "h": 0.6624, "omega_b": 0.022581, "omega_cdm": 0.12321,
        "n_s": 0.96732, "ln10As": 3.04991, "tau_reion": 0.05526,
    },
    "sig": {
        "h": 0.0080, "omega_b": 0.000166, "omega_cdm": 0.00213,
        "n_s": 0.0075, "ln10As": 0.0124, "tau_reion": 0.0057,
    },
    "fixed": {
        "TCMB0": 2.34865418e-4, "Neff": 3.044,
        "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
        "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
        "exp_reion": 1.5,
    },
    "user_species": (species.MassiveNeutrino,),
    "bbn": "table",
    "high_ell": "act_lite",
    "use_lowtt": False,
    "use_lowee": True,
    "lowee": "sroll2",
    "npts": 21,
    "nsig": 2.5,
}

"""lcdm_mchi_dneff_linx_bbnlike_plikfull.py — LCDM + Delta_Neff_init + an EM-coupled WIMP chi, LINX + BBN likelihood,
full plik.

lcdm_dneff_linx_bbnlike_plikfull.py (the _v6bbn setup: Planck full plik + low-ell TT/EE, Y_p and D/H with LINX's
nuisances profiled) plus a thermal relic chi in equilibrium with photons and electrons (Dirac fermion, g_chi = 4):
its annihilation heats the photons relative to the decoupled neutrinos, lowering Neff and shifting eta and the
n/p freeze-out (the em_dm model; abcmb/linx thermo.rho_chi, background.dY). Parameter log10_m_chi = log10(m_chi /
MeV); above ~30 MeV the model is LCDM, so its profile is open above (a lower limit). cen: the _v6bbn best fit,
log10_m_chi = 1.3 (20 MeV, LCDM-like); its 1D grid 5-80 MeV. The 2D grid (7-63 MeV x Delta_Neff_init -1.2..2.4,
11 x 10: at most 30 rows, one gradient block, per node)
covers where a LINX-only map at omega_b 0.02217 keeps Neff in the CMB range with BBN chi2 below ~30.
Launcher: scan/profile_prod_emwimp.slurm (1D, or 2D with PA_POI2D).
"""
from abcmb import species


CONFIG = {
    "order": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Delta_Neff_init", "log10_m_chi"],
    "cen": {
        "h": 0.6649, "omega_b": 0.022166, "omega_cdm": 0.1199,
        "n_s": 0.96097, "ln10As": 3.0469, "tau_reion": 0.056934, "Delta_Neff_init": -0.23625,
        "log10_m_chi": 1.3,
        "tau_n_fac": 0.99999, "q0": -0.018861, "q1": -0.67931, "q2": -0.72647, "q3": -0.52094, "q4": -0.0095582,
        "q5": -0.00017559, "q6": -1.5935e-05, "q7": 0.032914, "q8": -0.01188, "q9": 5.0377e-06,
        "q10": -6.2813e-06, "q11": -9.457e-10,
    },
    "sig": {
        "h": 0.0054, "omega_b": 0.00015, "omega_cdm": 0.0013,
        "n_s": 0.0042, "ln10As": 0.014, "tau_reion": 0.0073, "Delta_Neff_init": 0.37, "log10_m_chi": 0.3,
    },
    "grid_sig": {
        "h": 0.00822, "omega_b": 0.000153, "omega_cdm": 0.00167,
        "n_s": 0.00551, "ln10As": 0.0134, "tau_reion": 0.00606, "Delta_Neff_init": 0.309, "log10_m_chi": 0.24,
    },
    # the optimizer box of log10_m_chi (scaled sigmas): not below 1.8 MeV; above, -+PA_XBOX (the flat LCDM side
    # converges on the box face)
    "box": {"log10_m_chi": (-3.5, None)},
    # the 2D profile (PA_POI2D=log10_m_chi,Delta_Neff_init): (lo, hi, n) per axis
    "grid2d": {"log10_m_chi": (0.85, 1.8, 11), "Delta_Neff_init": (-1.2, 2.4, 10)},
    "fixed": {
        "TCMB0": 2.34865418e-4,
        "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
        "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
        "exp_reion": 1.5,
    },
    "user_species": (species.MassiveNeutrino,),
    "bbn": "linx",
    "linx_reaction_net": "key_recommended",
    "linx_chi": {"g_chi": 4.0, "boson": False},
    "bbn_like": {"Yp": (0.2458, 0.0013), "DH": (2.527, 0.030), "tau_n_sig": 0.000682},
    "high_ell": "plik_full",
    "use_lowtt": True,
    "use_lowee": True,
    "npts": 21,
    "nsig": 2.5,
}

"""lcdm_dneff_linx_bbnlike_plikfull.py — LCDM + Neff with LINX and the BBN LIKELIHOOD, full plik.

lcdm_dneff_linx_plikfull.py (YHe / Neff / T_nu from LINX per row, POI Delta_Neff_init) plus the primordial
abundance measurements as data: Gaussian terms on LINX's Y_p(BBN) = 4 Y_He4 and 1e5 D/H, with LINX's BBN
nuisances profiled with everything else -- tau_n_fac (prior N(1, 0.000682), i.e. 0.6 s on 879.4 s) and one
N(0, 1) log-normal rate shift per reaction of the network (q0 .. q11; the driver appends them to "order" and
prints which reaction each scales). Network key_recommended = PRIMAT 2023 with the Chen et al. 2025
n(p,g)d rate (arXiv:2509.11743) and updated d(p,g)3He / d+d rates. Data as in
bbn-on-cmb-bao-tension/CMB_BBN_Neff.py: Y_p 0.2458 +/- 0.0013 (Aver 2026), D/H 2.527 +/- 0.030 (Cooke 2018);
the LBT value (Hsyu+ 2020, 0.2436 +/- 0.0040) is the alternative for "Yp".

cen: the _v4linx best fit (CMB only, BBN-consistent YHe; 2026-10-24). The D/H term pulls Neff up (the
network's D/H at this point is ~2.4e-5), so the joint best fit moves: run with PA_MLE_PREPASS=1, which
recentres every grid on the joint fit before profiling.
"""
from abcmb import species


CONFIG = {
    "order": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Delta_Neff_init"],
    "cen": {
        "h": 0.66027, "omega_b": 0.022210, "omega_cdm": 0.117200,
        "n_s": 0.958440, "ln10As": 3.042300, "tau_reion": 0.057740, "Delta_Neff_init": -0.742,
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
    "linx_reaction_net": "key_recommended",
    "bbn_like": {"Yp": (0.2458, 0.0013), "DH": (2.527, 0.030), "tau_n_sig": 0.000682},
    "high_ell": "plik_full",
    "use_lowtt": True,
    "use_lowee": True,
    "npts": 25,
    "nsig": 3.0,
}

"""lcdm_dneff_linx_bbnlike_plikfull.py — LCDM + Delta_Neff_init with LINX and the BBN LIKELIHOOD, full plik.

lcdm_dneff_linx_plikfull.py plus the primordial abundances as data: Gaussian terms on LINX's Y_p(BBN) =
4 Y_He4 and 1e5 D/H, with LINX's nuisances profiled -- tau_n_fac (prior N(1, 0.000682), i.e. 0.6 s on 879.4 s)
and one N(0, 1) log-normal rate shift per reaction (q0 .. q11; the driver appends them to "order" and prints
which reaction each scales). Network key_recommended = PRIMAT 2023 with the Chen et al. 2025 n(p,g)d rate
(arXiv:2509.11743) and updated d(p,g)3He / d+d rates. Data: Y_p 0.2458 +/- 0.0013 (LBT, Aver 2026), D/H 2.527
+/- 0.030 (Cooke 2018).
cen (BBN nuisances included) / grid_sig: the PA_MLE_PREPASS=1 joint fit of this likelihood and its Fisher
sigmas (job 60327767). Launcher: scan/profile_prod_linx_bbnlike.slurm.
"""
from abcmb import species


CONFIG = {
    "order": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Delta_Neff_init"],
    "cen": {
        "h": 0.6649, "omega_b": 0.022166, "omega_cdm": 0.1199,
        "n_s": 0.96097, "ln10As": 3.0469, "tau_reion": 0.056934, "Delta_Neff_init": -0.23625,
        "tau_n_fac": 0.99999, "q0": -0.018861, "q1": -0.67931, "q2": -0.72647, "q3": -0.52094, "q4": -0.0095582,
        "q5": -0.00017559, "q6": -1.5935e-05, "q7": 0.032914, "q8": -0.01188, "q9": 5.0377e-06,
        "q10": -6.2813e-06, "q11": -9.457e-10,
    },
    "sig": {
        "h": 0.0054, "omega_b": 0.00015, "omega_cdm": 0.0013,
        "n_s": 0.0042, "ln10As": 0.014, "tau_reion": 0.0073, "Delta_Neff_init": 0.37,
    },
    "grid_sig": {
        "h": 0.00822, "omega_b": 0.000153, "omega_cdm": 0.00167,
        "n_s": 0.00551, "ln10As": 0.0134, "tau_reion": 0.00606, "Delta_Neff_init": 0.309,
    },
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
    "npts": 21,
    "nsig": 2.5,
}

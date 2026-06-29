"""lcdm_neff_plikfull_rc.py — LCDM + Neff (D=7), FULL plik, RECENTERED grids.

Identical to lcdm_neff_plikfull.py EXCEPT `cen` is moved to the LCDM+Neff JOINT
best-fit (the MLE) measured from the first headline run (job 54980729).

Why: freeing Neff pulls Neff down to ~2.83, and the Neff-omega_cdm-H0 degeneracy
drags the correlated params DOWN with it. Grids centered on the LCDM/Planck values
then sit ~1.4 sigma (h) / ~2.9 sigma (omega_cdm) HIGH, so those POIs' minima rail at
the low grid edge and their lower dchi2=1 crossing falls off the grid (omega_cdm
came back nan). A nan interval also blocked the sigma1 early-stop for that slice, so
it ran to the wall. Recentering on the joint MLE puts every minimum near grid-center,
captures both interval edges, and lets the early-stop fire.

Joint MLE adopted (lowest-chi2 profile-min row, chi2=2757.11):
  h=0.6662 omega_b=0.02229 omega_cdm=0.11654 n_s=0.95986 ln10As=3.0446
  tau_reion=0.05933 Neff=2.8268
sigma's are UNCHANGED (they set the +/-3sigma grid span and the sigma1-stop units).

Run with PA_WARM_FROM_CEN=1 so the warm start (and warm-Hessian preconditioner) use
this center (= the joint MLE) rather than the stale LCDM-only profile npz.
"""

CONFIG = {
    "order": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"],
    "cen": {
        "h": 0.6662, "omega_b": 0.02229, "omega_cdm": 0.11654,
        "n_s": 0.95986, "ln10As": 3.0446, "tau_reion": 0.05933, "Neff": 2.8268,
    },
    "sig": {
        "h": 0.0054, "omega_b": 0.00015, "omega_cdm": 0.0012,
        "n_s": 0.0042, "ln10As": 0.014, "tau_reion": 0.0073, "Neff": 0.2,
    },
    "pois": ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"],
    "fixed": {
        "YHe": 0.2454, "TCMB0": 2.34865418e-4,
        "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
        "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
        "exp_reion": 1.5,
    },
    "user_species": None,
    "high_ell": "plik_full",
    "use_lowtt": True,
    "use_lowee": True,
    "npts": 25,
    "nsig": 3.0,
    "grad_method": "ad",
}

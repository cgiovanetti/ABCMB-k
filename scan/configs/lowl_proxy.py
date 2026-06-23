"""lowl_proxy.py -- a FAST, well-conditioned DEBUG config for validating
l-independent machinery (the chi2-plateau early-stop), NOT a science config.

It drops the high-ell plik-lite likelihood (set PA_USE_PLIK=0 when using this --
plik-lite needs theory Cls to l~2508, so it is meaningless at a truncated LMAX)
and keeps only the low-ell TT (Commander, l=2..29) + EE (SRoll2, l=2..29). Those
need theory only to l~29, so the whole pipeline runs in seconds/iter at LMAX~100
-- many BFGS iters per 30-min debug job, enough to watch a profile converge AND
plateau.

PROFILE SHAPE: tau_reion is the POI; ln10As is the single free nuisance (the
A_s e^{-2tau} degeneracy the low-ell data partially breaks via the EE reion bump).
The other four LCDM params are FIXED at fiducial -- so this is a clean 2-param
(D=2, P=1) problem with NO low-ell-unconstrained flat directions to wreck the
preconditioner. It still exercises the early-stop's key gotcha (edge +/-3sigma POI
rows settle LATER than central rows -> the trigger's max-over-rows reduction must
wait for the slowest row), which is what we are validating.

Usage:
  PA_CONFIG=scan/configs/lowl_proxy.py PA_USE_PLIK=0 PA_LMAX=100 \
  PA_POIS=tau_reion python scan/profile_prod_ad.py
"""

CONFIG = {
    "order": ["ln10As", "tau_reion"],          # tau is POI; ln10As is the nuisance
    "cen": {"ln10As": 3.044, "tau_reion": 0.0544},
    "sig": {"ln10As": 0.014, "tau_reion": 0.0073},
    "pois": ["tau_reion"],
    "fixed": {
        # the four LCDM params low-ell can't constrain, held at fiducial:
        "h": 0.6736, "omega_b": 0.02237, "omega_cdm": 0.1200, "n_s": 0.9649,
        # standard ABCMB fixed block (mirrors lcdm.py):
        "Neff": 3.044, "YHe": 0.2454, "TCMB0": 2.34865418e-4,
        "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
        "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
        "exp_reion": 1.5,
    },
    "user_species": None,
    "use_lowtt": True,
    "use_lowee": True,
    "use_plik": False,                         # debug proxy: low-ell only (also pass PA_USE_PLIK=0)
    "npts": 7,
    "nsig": 3.0,
    "grad_method": "ad",
}

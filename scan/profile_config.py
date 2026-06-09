"""profile_config.py — the analysis definition for a frequentist profile of
LCDM against Planck plik-lite. Pure numpy (no jax) so both the GPU worker
(scan_profile.py) and the collector (collect_profile.py) import the SAME grid.

Edit this file to change the scan (ranges, resolution, which params are free,
priors, fixed cosmology). The worker and collector stay in sync automatically.

Design choices (see scan/FEASIBILITY_plik_lite.md):
  * 6 free LCDM params, ALL evaluated by ABCMB: h, omega_b, omega_cdm, n_s, A_s,
    tau_reion. (A_s is a REAL grid dimension, not the analytic-amplitude shortcut,
    so the lensing non-linearity in A_s is captured exactly.)
  * The ONLY analytic nuisance is the Planck calibration A_planck (profiled
    closed-form WITH its N(1, 0.0025) prior in plik_lite.profile_A).
  * tau_reion is unconstrained by high-l TTTEEE alone, so it carries the Planck
    2018 lowE Gaussian prior tau = 0.0544 +/- 0.0073, added to chi^2.
  * One massive neutrino (0.06 eV) — the Planck baseline; measured to cost the
    same as massless at l_max=2508+lensing.

Grid is a Cartesian product, enumerated deterministically (itertools.product in
SCAN_ORDER) so every node regenerates the identical global grid and just takes
its strided slice.
"""
import itertools
import numpy as np

# Planck 2018 base-LCDM (TT,TE,EE+lowE) best fit + 1-sigma — used to CENTER and
# SCALE the grid (so the exact best fit is a grid point when NPTS is odd).
CENTER = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200,
    'n_s': 0.9649, 'A_s': 2.0989e-9, 'tau_reion': 0.0544,
}
SIGMA = {
    'h': 0.0054, 'omega_b': 0.00015, 'omega_cdm': 0.0012,
    'n_s': 0.0042, 'A_s': 2.94e-11, 'tau_reion': 0.0073,
}

# scan order (fixes the flat-index <-> grid mapping) and points-per-dim.
SCAN_ORDER = ['h', 'omega_b', 'omega_cdm', 'n_s', 'A_s', 'tau_reion']
# VALIDATION grid: coarse, odd N on the well-constrained dims so the Planck best
# fit is a grid point and the min should land on chi^2 ~ 585. Total = 5*3*5*5*3*3
# = 3375 points (~19 min on 2 nodes at 0.67 s/param) -> finishes inside the 30-min
# debug-qos walltime (workers only save at the end, so the grid MUST complete).
# Bump for production (and submit to --qos=regular, which has no 30-min cap).
NPTS = {'h': 5, 'omega_b': 3, 'omega_cdm': 5, 'n_s': 5, 'A_s': 3, 'tau_reion': 3}
NSIG = 4.0  # grid spans CENTER +/- NSIG*sigma

# test hook: ABCMB_PROFILE_NPTS_OVERRIDE=2 collapses every dim to N points (e.g.
# a 2^6=64-point smoke grid). Leave unset for the real scan.
import os as _os
_ov = _os.environ.get("ABCMB_PROFILE_NPTS_OVERRIDE")
if _ov:
    NPTS = {p: int(_ov) for p in SCAN_ORDER}

# Gaussian priors ADDED to chi^2 (A_planck is handled separately/analytically).
GAUSS_PRIORS = {'tau_reion': (0.0544, 0.0073)}

# fixed cosmology (one massive nu = Planck baseline). YHe fixed (no BBN coupling
# here since N_eff is fixed); promote to bbn_type="table"/"linx" if N_eff varies.
FIXED = {
    'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
    'exp_reion': 1.5,
}


def make_axes():
    """1-D grid axis for each scanned parameter."""
    return {p: np.linspace(CENTER[p] - NSIG * SIGMA[p],
                           CENTER[p] + NSIG * SIGMA[p], NPTS[p])
            for p in SCAN_ORDER}


def n_total():
    n = 1
    for p in SCAN_ORDER:
        n *= NPTS[p]
    return n


def make_grid(axes=None):
    """Deterministic list of param dicts (FIXED + one combo each), global order.
    Index i in this list == the flat C-order index into the NPTS grid."""
    if axes is None:
        axes = make_axes()
    cols = [axes[p] for p in SCAN_ORDER]
    grid = []
    for combo in itertools.product(*cols):
        p = dict(FIXED)
        for name, val in zip(SCAN_ORDER, combo):
            p[name] = float(val)
        grid.append(p)
    return grid

"""Per-ell error diagnostic: WHERE do TT/EE errors live for each grid?

Computes CLASS once, then for each (label, n, mode, extra) config reports the
max TT/EE rel error AND the multipole where it occurs, plus binned max errors
(low ell<30, mid 30-1000, high>1000). Saves all Cl arrays to grid_diag.npz for
offline plotting. Tells us what a reduced grid must resolve, instead of blind
tuning.
"""
import os, json, traceback
import pandas as pd
pd.options.future.infer_string = False
from classy import Class
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
np.seterr(all='raise')
import abcmb
assert "ABCMB-k" in abcmb.__file__, abcmb.__file__
from abcmb.main import Model

ELLMIN, ELLMAX = 2, 2500
BASE = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
CONFIGS = [
    ("uniform-500", 500, "uniform", {}),
    ("vis-300 sm0.3 f0.4", 300, "visibility", dict(lna_vis_smooth=0.3, lna_vis_floor=0.4)),
    ("vis-300 sm0.5 f0.6", 300, "visibility", dict(lna_vis_smooth=0.5, lna_vis_floor=0.6)),
    ("rd-300 a8w.5 reion5", 300, "recomb_dense",
     dict(lna_recomb_amp=8.0, lna_recomb_width=0.5, lna_reion_amp=5.0,
          lna_reion_center=-2.1, lna_reion_width=0.7)),
    ("rd-350 a8w.5 reion5", 350, "recomb_dense",
     dict(lna_recomb_amp=8.0, lna_recomb_width=0.5, lna_reion_amp=5.0,
          lna_reion_center=-2.1, lna_reion_width=0.7)),
]


def build(n, mode, extra):
    return Model(user_species=None, output_Cl=True, l_max=ELLMAX, lensing=True,
                 output_Pk=True, output_k_max=0.5, l_max_g=12, l_max_pol_g=10,
                 l_max_ur=17, l_max_ncdm=17, n_lna_PE=n, lna_grid_mode=mode, **extra)


def run_class(model):
    fp = model.add_derived_parameters(dict(BASE))
    C = Class()
    C.set({"output": "mPk, tCl, pCl, lCl", "l_max_scalars": ELLMAX,
           "P_k_max_1/Mpc": model.specs["output_k_max"], "lensing": "yes",
           "accurate_lensing": 1, "H0": fp["h"]*100, "omega_b": fp["omega_b"],
           "omega_cdm": fp["omega_cdm"], "A_s": fp["A_s"], "n_s": fp["n_s"],
           "N_ur": fp["Neff"], "YHe": fp["YHe"], "N_ncdm": fp["N_nu_massive"],
           "reio_parametrization": "reio_camb", "tau_reio": BASE["tau_reion"],
           "reionization_width": BASE["Delta_z_reion"],
           "helium_fullreio_redshift": BASE["z_reion_He"],
           "helium_fullreio_width": BASE["Delta_z_reion_He"],
           "reionization_exponent": BASE["exp_reion"],
           "l_max_g": model.specs["l_max_g"], "l_max_pol_g": model.specs["l_max_pol_g"],
           "l_max_ur": model.specs["l_max_ur"], "l_max_ncdm": model.specs["l_max_ncdm"]})
    C.compute()
    cl = C.lensed_cl(ELLMAX)
    return cl["tt"][ELLMIN:], cl["ee"][ELLMIN:]


def binned(rel, ell):
    out = {}
    for name, lo, hi in (("lo<30", 2, 30), ("mid", 30, 1000), ("hi>1000", 1000, 2501)):
        m = (ell >= lo) & (ell < hi)
        out[name] = round(float(np.max(rel[m]) * 100), 4)
    return out


def main():
    m0 = build(*CONFIGS[0][1:])
    cltt, clee = run_class(m0)
    ell = np.arange(ELLMIN, ELLMAX + 1)
    print(f"# abcmb {abcmb.__file__}\n# CLASS done", flush=True)
    save = {"ell": ell, "class_tt": cltt, "class_ee": clee}
    for label, n, mode, extra in CONFIGS:
        try:
            out = build(n, mode, extra)(dict(BASE))
            tt = np.asarray(out.ClTT); ee = np.asarray(out.ClEE)
            jax.block_until_ready([tt, ee])
            rtt = np.abs(cltt - tt) / np.abs(cltt)
            ree = np.abs(clee - ee) / np.abs(clee)
            save[f"tt_{label}"] = tt; save[f"ee_{label}"] = ee
            print(f"\n## {label}", flush=True)
            print(f"   TT max {rtt.max()*100:.4f}% @ ell={ell[rtt.argmax()]}  bins={binned(rtt,ell)}", flush=True)
            print(f"   EE max {ree.max()*100:.4f}% @ ell={ell[ree.argmax()]}  bins={binned(ree,ell)}", flush=True)
        except Exception as e:
            print(f"## {label} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            traceback.print_exc()
    np.savez("bench/grid_diag.npz", **save)
    print("\n# saved bench/grid_diag.npz", flush=True)


if __name__ == "__main__":
    main()

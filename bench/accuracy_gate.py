"""Accuracy gate vs CLASS for the round-3 changes (aH-tabulation especially).

Runs ABCMB's single-call model() (on whatever backend JAX picks — pass
JAX_PLATFORM_NAME=gpu in the srun to use the GPU) and compares TT/EE/Pk to
classy at the same cosmology, for massless ΛCDM and (with --massive) one
massive neutrino. Mirrors pytests/accuracy_test.py but is parametrizable and
prints a JSON line so a driver can diff against the committed baseline
(TT 0.197% / EE 0.231% / Pk 0.185% massless).

  python bench/accuracy_gate.py --massive 0
  python bench/accuracy_gate.py --massive 1

The massive-ν point establishes a gate that did not previously exist.
"""
import os, sys, json, argparse, traceback
import pandas as pd
pd.options.future.infer_string = False  # cobaya/BAO-adjacent import safety
from classy import Class
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
np.seterr(all='raise')
from abcmb.main import Model
from abcmb import species

ELLMIN, ELLMAX = 2, 2500
BASE = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}


def run(massive):
    params = dict(BASE)
    if massive:
        params['N_nu_massive'] = 1
        params['Neff'] = 2.0308              # 2 massless + 1 massive ~ 3.044
    user_species = (species.MassiveNeutrino,) if params['N_nu_massive'] > 0 else None

    model = Model(user_species=user_species, output_Cl=True, l_max=ELLMAX,
                  lensing=True, output_Pk=True, output_k_max=0.5,
                  l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)
    fp = model.add_derived_parameters(params)

    C = Class()
    C.set({
        "output": "mPk, tCl, pCl, lCl", "l_max_scalars": ELLMAX,
        "P_k_max_1/Mpc": model.specs["output_k_max"], "lensing": "yes",
        "accurate_lensing": 1, "H0": fp["h"]*100, "omega_b": fp["omega_b"],
        "omega_cdm": fp["omega_cdm"], "A_s": fp["A_s"], "n_s": fp["n_s"],
        "N_ur": fp["Neff"], "YHe": fp["YHe"], "N_ncdm": fp["N_nu_massive"],
        "reio_parametrization": "reio_camb", "tau_reio": params["tau_reion"],
        "reionization_width": params["Delta_z_reion"],
        "helium_fullreio_redshift": params["z_reion_He"],
        "helium_fullreio_width": params["Delta_z_reion_He"],
        "reionization_exponent": params["exp_reion"],
        "l_max_g": model.specs["l_max_g"], "l_max_pol_g": model.specs["l_max_pol_g"],
        "l_max_ur": model.specs["l_max_ur"], "l_max_ncdm": model.specs["l_max_ncdm"],
    })
    if fp["N_nu_massive"] > 0:
        C.set({"m_ncdm": fp["m_nu_massive"], "T_ncdm": fp["T_nu_massive"]})
    C.compute()
    cl = C.lensed_cl(ELLMAX)
    cltt, clee = cl["tt"][ELLMIN:], cl["ee"][ELLMIN:]

    out = model(params)
    err_tt = float(np.max(np.abs(cltt - np.asarray(out.ClTT)) / cltt))
    err_ee = float(np.max(np.abs(clee - np.asarray(out.ClEE)) / clee))
    CLA_Pk = np.vectorize(C.pk)(np.asarray(out.k), 0.)
    err_pk = float(np.max(np.abs(CLA_Pk - np.asarray(out.Pk)) / CLA_Pk))
    return err_tt, err_ee, err_pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--massive", type=int, default=0)
    a = ap.parse_args()
    rec = {"massive": a.massive, "backend": jax.default_backend()}
    try:
        tt, ee, pk = run(bool(a.massive))
        rec.update(TT_pct=round(tt*100, 4), EE_pct=round(ee*100, 4),
                   Pk_pct=round(pk*100, 4),
                   pass_1pct=bool(max(tt, ee, pk) <= 0.01), ok=True)
    except Exception as e:
        rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:300]}")
        traceback.print_exc()
    print("GATE " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

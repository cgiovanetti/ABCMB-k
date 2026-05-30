"""Accuracy-vs-CLASS sweep over n_lna_PE (Step A of round-3 plan §3).

Computes CLASS ONCE (massless LCDM, the accuracy_gate.py BASE cosmology), then
builds an ABCMB Model at each n_lna_PE and reports TT/EE/Pk max-rel-error vs
CLASS. The k/ell grids do not depend on n_lna_PE, so the single CLASS solve is a
valid reference for every point. One process amortizes the CLASS solve + JAX
import; each n_lna_PE still triggers its own JIT compile (unavoidable).

Baseline (n_lna_PE=500, these nodes): TT 0.197% / EE 0.231% / Pk 0.185%.
Gate: flag the smallest n_lna whose TT/EE/Pk stays within ~0.05% of baseline.

  python bench/nlna_accuracy_sweep.py --massive 0 --nlna 500 450 400 350 300 250 200
"""
import os, sys, json, time, argparse, traceback
import pandas as pd
pd.options.future.infer_string = False  # cobaya/BAO-adjacent import safety
from classy import Class
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
np.seterr(all='raise')
import abcmb
from abcmb.main import Model
from abcmb import species

# Guard: this sweep is meaningless unless we import the ABCMB-k playground (which
# has n_lna_PE wired in), not the pip-editable parent /pscratch/.../ABCMB.
assert "ABCMB-k" in abcmb.__file__, (
    f"WRONG abcmb imported: {abcmb.__file__} — set PYTHONPATH=/pscratch/sd/c/carag/ABCMB-k")
print(f"# abcmb from {abcmb.__file__}", flush=True)

ELLMIN, ELLMAX = 2, 2500
BASE = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}


def build_model(massive, nlna):
    params = dict(BASE)
    if massive:
        params['N_nu_massive'] = 1
        params['Neff'] = 2.0308
    user_species = (species.MassiveNeutrino,) if params['N_nu_massive'] > 0 else None
    model = Model(user_species=user_species, output_Cl=True, l_max=ELLMAX,
                  lensing=True, output_Pk=True, output_k_max=0.5,
                  l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  n_lna_PE=nlna)
    return model, params


def run_class(model, params):
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
    return C, cl["tt"][ELLMIN:], cl["ee"][ELLMIN:]


def errors(out, cltt, clee, C):
    err_tt = float(np.max(np.abs(cltt - np.asarray(out.ClTT)) / cltt))
    err_ee = float(np.max(np.abs(clee - np.asarray(out.ClEE)) / clee))
    CLA_Pk = np.vectorize(C.pk)(np.asarray(out.k), 0.)
    err_pk = float(np.max(np.abs(CLA_Pk - np.asarray(out.Pk)) / CLA_Pk))
    return err_tt, err_ee, err_pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--massive", type=int, default=0)
    ap.add_argument("--nlna", type=int, nargs="+",
                    default=[500, 450, 400, 350, 300, 250, 200])
    a = ap.parse_args()

    # CLASS once (cosmology + k/ell grids are n_lna-independent).
    model0, params = build_model(bool(a.massive), a.nlna[0])
    C, cltt, clee = run_class(model0, params)
    print(f"# CLASS done. backend={jax.default_backend()} massive={a.massive}",
          flush=True)

    rows = []
    for nlna in a.nlna:
        rec = {"massive": a.massive, "nlna": nlna}
        try:
            model, params = build_model(bool(a.massive), nlna)
            t0 = time.perf_counter()
            out = model(params)
            jax.block_until_ready(jax.tree_util.tree_leaves(out.ClTT))
            wall = time.perf_counter() - t0
            tt, ee, pk = errors(out, cltt, clee, C)
            rec.update(TT_pct=round(tt*100, 4), EE_pct=round(ee*100, 4),
                       Pk_pct=round(pk*100, 4), compile_run_s=round(wall, 1),
                       ok=True)
        except Exception as e:
            rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
        rows.append(rec)
        print("GATE " + json.dumps(rec), flush=True)

    print("\n# n_lna   TT%      EE%      Pk%     compile+run(s)", flush=True)
    for r in rows:
        if r.get("ok"):
            print(f"# {r['nlna']:5d}  {r['TT_pct']:7.4f}  {r['EE_pct']:7.4f}  "
                  f"{r['Pk_pct']:7.4f}   {r['compile_run_s']:.1f}", flush=True)
        else:
            print(f"# {r['nlna']:5d}  FAILED: {r.get('err')}", flush=True)


if __name__ == "__main__":
    main()

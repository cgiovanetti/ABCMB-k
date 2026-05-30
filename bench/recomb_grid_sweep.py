"""Recomb-dense lna-grid accuracy sweep vs CLASS (Step B).

Target (user, stricter than the 0.05% plan gate): RECOVER the n_lna=500 uniform
baseline accuracy (TT 0.197% / EE 0.231% / Pk 0.185%) at a reduced point count.
Computes CLASS once, then runs uniform-500 (baseline reproduce) and a list of
recomb-dense (N, amp, width, reion_amp) configs, all vs the same CLASS reference.

  python bench/recomb_grid_sweep.py            # default config list
  CONFIGS='300,12,0.4,0;300,20,0.3,0' python bench/recomb_grid_sweep.py
"""
import os, sys, json, time, traceback
import pandas as pd
pd.options.future.infer_string = False
from classy import Class
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
np.seterr(all='raise')
import abcmb
assert "ABCMB-k" in abcmb.__file__, f"WRONG abcmb: {abcmb.__file__}"
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

# (label, n_lna, mode, extra_specs)
def cfgs():
    env = os.environ.get("CONFIGS")  # "vis,N,floor,gp; vis,N,floor,gp; ..."
    out = [("uniform-500", 500, "uniform", {})]
    if env:
        for tok in env.split(";"):
            parts = [p.strip() for p in tok.split(",")]
            _, n, floor, gp = parts
            out.append((f"vis N={n} f={floor} gp={gp}", int(n), "visibility",
                        dict(lna_vis_floor=float(floor), lna_vis_gprime_amp=float(gp))))
        return out
    grid = [(400, 0.5), (350, 0.5), (300, 0.3), (300, 0.5), (300, 0.8), (250, 0.5)]
    for n, floor in grid:
        out.append((f"vis N={n} f={floor}", n, "visibility",
                    dict(lna_vis_floor=float(floor))))
    return out


def build(n, mode, extra):
    return Model(user_species=None, output_Cl=True, l_max=ELLMAX, lensing=True,
                 output_Pk=True, output_k_max=0.5, l_max_g=12, l_max_pol_g=10,
                 l_max_ur=17, l_max_ncdm=17, n_lna_PE=n, lna_grid_mode=mode, **extra)


def run_class(model, params):
    fp = model.add_derived_parameters(params)
    C = Class()
    C.set({"output": "mPk, tCl, pCl, lCl", "l_max_scalars": ELLMAX,
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
           "l_max_ur": model.specs["l_max_ur"], "l_max_ncdm": model.specs["l_max_ncdm"]})
    C.compute()
    cl = C.lensed_cl(ELLMAX)
    return C, cl["tt"][ELLMIN:], cl["ee"][ELLMIN:]


def main():
    configs = cfgs()
    m0 = build(*configs[0][1:])
    C, cltt, clee = run_class(m0, dict(BASE))
    print(f"# abcmb {abcmb.__file__}\n# CLASS done backend={jax.default_backend()}", flush=True)

    rows = []
    for label, n, mode, extra in configs:
        rec = {"label": label, "n": n, "mode": mode}
        try:
            model = build(n, mode, extra)
            out = model(dict(BASE))
            jax.block_until_ready(jax.tree_util.tree_leaves(out.ClTT))
            tt = float(np.max(np.abs(cltt - np.asarray(out.ClTT)) / cltt))
            ee = float(np.max(np.abs(clee - np.asarray(out.ClEE)) / clee))
            CLA_Pk = np.vectorize(C.pk)(np.asarray(out.k), 0.)
            pk = float(np.max(np.abs(CLA_Pk - np.asarray(out.Pk)) / CLA_Pk))
            # grid spacing diagnostic
            lna = np.asarray(out.PT.lna)
            lrec = float(np.asarray(out.BG.lna_rec))
            near = np.abs(lna - lrec) < 0.1
            dmin = float(np.min(np.diff(lna)))
            dnear = float(np.median(np.diff(lna[near]))) if near.sum() > 1 else float('nan')
            rec.update(TT_pct=round(tt*100, 4), EE_pct=round(ee*100, 4),
                       Pk_pct=round(pk*100, 4), lna_rec=round(lrec, 3),
                       dmin=round(dmin, 4), d_near_rec=round(dnear, 4),
                       npts_near=int(near.sum()), ok=True)
        except Exception as e:
            rec.update(ok=False, err=f"{type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()
        rows.append(rec)
        print("GATE " + json.dumps(rec), flush=True)

    print("\n# label                     TT%      EE%      Pk%    d_near_rec  npts_near", flush=True)
    for r in rows:
        if r.get("ok"):
            print(f"# {r['label']:24s}  {r['TT_pct']:7.4f}  {r['EE_pct']:7.4f}  "
                  f"{r['Pk_pct']:7.4f}   {r['d_near_rec']:.4f}     {r['npts_near']}", flush=True)
        else:
            print(f"# {r['label']:24s}  FAILED {r.get('err')}", flush=True)


if __name__ == "__main__":
    main()

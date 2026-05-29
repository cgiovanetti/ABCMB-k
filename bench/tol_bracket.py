"""Accuracy bracket: how loose can the stiff-band PE tolerance go before the
1%-vs-CLASS gate moves? ABCMB runs on GPU, CLASS on CPU. Mirrors
pytests/accuracy_test.py (lensing=True, ellmax=2500) but sweeps
(rtol_large_k_PE, atol_large_k_PE) and reports max-rel TT/EE/Pk per config.

Run on 1 GPU:
  srun --jobid=<J> --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    bash -c '... python bench/tol_bracket.py'
"""
import os, time, json
# ABCMB on GPU; CLASS is CPU regardless.
import numpy as np
np.seterr(all='ignore')
from classy import Class
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model

ELLMIN, ELLMAX = 2, 2500
PARAMS = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225, 'A_s': 2.12424e-9,
    'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5, 'z_reion_He': 3.5,
    'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
# (rtol_large_k_PE, atol_large_k_PE); baseline first.
CONFIGS = [
    (1e-4, 1e-6),
    (3e-4, 3e-6),
    (1e-3, 1e-5),
    (3e-3, 3e-5),
]


def build_class():
    cp = {
        "output": "mPk, tCl, pCl, lCl", "l_max_scalars": ELLMAX,
        "P_k_max_1/Mpc": 0.5, "lensing": "yes", "accurate_lensing": 1,
        "H0": PARAMS['h'] * 100, "omega_b": PARAMS['omega_b'],
        "omega_cdm": PARAMS['omega_cdm'], "A_s": PARAMS['A_s'],
        "n_s": PARAMS['n_s'], "N_ur": PARAMS['Neff'], "YHe": PARAMS['YHe'],
        "N_ncdm": 0, "reio_parametrization": "reio_camb",
        "tau_reio": PARAMS['tau_reion'], "reionization_width": PARAMS['Delta_z_reion'],
        "helium_fullreio_redshift": PARAMS['z_reion_He'],
        "helium_fullreio_width": PARAMS['Delta_z_reion_He'],
        "reionization_exponent": PARAMS['exp_reion'],
        "l_max_g": 12, "l_max_pol_g": 10, "l_max_ur": 17, "l_max_ncdm": 17,
    }
    C = Class(); C.set(cp); C.compute()
    cl = C.lensed_cl(ELLMAX)
    return C, cl["tt"][ELLMIN:], cl["ee"][ELLMIN:]


def main():
    print("building CLASS reference...", flush=True)
    C, cltt, clee = build_class()
    rows = []
    for (rtol, atol) in CONFIGS:
        model = Model(user_species=None, output_Cl=True, l_max=ELLMAX,
                      lensing=True, output_Pk=True, output_k_max=0.5,
                      l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                      rtol_large_k_PE=rtol, atol_large_k_PE=atol)
        # warm + timed
        t0 = time.perf_counter(); out = model(PARAMS)
        jax.block_until_ready(out.ClTT); warm = time.perf_counter() - t0
        t0 = time.perf_counter(); out = model(PARAMS)
        jax.block_until_ready(out.ClTT); run = time.perf_counter() - t0
        tt = np.asarray(out.ClTT); ee = np.asarray(out.ClEE)
        Pk = np.asarray(out.Pk); kk = np.asarray(out.k)
        cpk = np.vectorize(C.pk)(kk, 0.)
        e_tt = float(np.max(np.abs(cltt - tt) / cltt))
        e_ee = float(np.max(np.abs(clee - ee) / clee))
        e_pk = float(np.max(np.abs(cpk - Pk) / cpk))
        row = {"rtol": rtol, "atol": atol, "err_tt": e_tt, "err_ee": e_ee,
               "err_pk": e_pk, "warm": round(warm, 2), "run": round(run, 2),
               "pass": e_tt <= 0.01 and e_ee <= 0.01 and e_pk <= 0.01}
        rows.append(row)
        print(f"  rtol={rtol:.0e} atol={atol:.0e}  TT={e_tt:.4%} EE={e_ee:.4%} "
              f"Pk={e_pk:.4%}  run={run:.2f}s  pass={row['pass']}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "tol_bracket_results.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n" + json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()

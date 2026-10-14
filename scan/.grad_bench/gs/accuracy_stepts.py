"""CLASS accuracy of the single-cosmology path with step_ts_PE off (production behaviour: save
through Kvaerno5's linear in-step interpolation) vs on (step onto the save grid). Same scenario
as pytests/accuracy_test.py; reports max |rel err| vs CLASS for TT/EE at ell<30 and ell>=30."""
import os, sys, re, json
ROOT = "/pscratch/sd/c/carag/ABCMB-k-grad"
sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import abcmb
assert abcmb.__file__.startswith(ROOT)
from abcmb.main import Model
from classy import Class
src = open(os.path.join(ROOT, "pytests", "accuracy_test.py")).read()
h = float(re.search(r"def test_accuracy_checker\(h\s*=\s*([0-9.]+)\)", src).group(1))
ellmax = 2500
params = {'h': h, 'omega_cdm': 0.1193, 'omega_b': 0.0225, 'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044,
          'YHe': 0.245, 'TCMB0': 2.34865418e-4, 'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
          "tau_reion": 0.0544, "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5, "exp_reion": 1.5}
specs = dict(user_species=None, output_Cl=True, l_max=ellmax, lensing=True, output_Pk=True, output_k_max=0.5,
             l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)
m0 = Model(**specs, step_ts_PE=False)
fp = m0.add_derived_parameters(params)
C = Class()
C.set({"output": "mPk, tCl, pCl, lCl", "l_max_scalars": ellmax, "P_k_max_1/Mpc": 0.5, "lensing": "yes",
       "accurate_lensing": 1, "H0": fp["h"] * 100, "omega_b": fp["omega_b"], "omega_cdm": fp["omega_cdm"],
       "A_s": fp["A_s"], "n_s": fp["n_s"], "N_ur": fp["Neff"], "YHe": fp["YHe"], "N_ncdm": fp["N_nu_massive"],
       "reio_parametrization": "reio_camb", "tau_reio": params["tau_reion"], "reionization_width": params["Delta_z_reion"],
       "helium_fullreio_redshift": params["z_reion_He"], "helium_fullreio_width": params["Delta_z_reion_He"],
       "reionization_exponent": params["exp_reion"], "l_max_g": 12, "l_max_pol_g": 10, "l_max_ur": 17, "l_max_ncdm": 17})
C.compute()
cl = C.lensed_cl(ellmax)
out = {}
for step_ts in (False, True):
    model = Model(**specs, step_ts_PE=step_ts)
    o = model(params)
    l = np.asarray(o.l); lo = l < 30; hi = ~lo
    tt = np.asarray(cl["tt"][2:]); ee = np.asarray(cl["ee"][2:])
    ett = np.abs(np.asarray(o.ClTT) - tt) / tt; eee = np.abs(np.asarray(o.ClEE) - ee) / ee
    rec = {"TT_max": float(ett.max()), "TT_l<30": float(ett[lo].max()), "TT_l>=30": float(ett[hi].max()),
           "TT_l2": float(ett[0]), "EE_max": float(eee.max()), "EE_l<30": float(eee[lo].max()), "EE_l>=30": float(eee[hi].max())}
    out[f"step_ts={step_ts}"] = rec
    print(f"VS_CLASS step_ts={step_ts}: " + json.dumps(rec), flush=True)
print("DONE", flush=True)

from classy import Class
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
file_dir = os.path.dirname(__file__)

import sys
sys.path.append(file_dir+'/../')
# print(os.getcwd())
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
from ABCMB.main import Model
import ABCMB.spectrum as spectrum
from scipy.interpolate import interp1d
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pytest
import numpy as np
np.seterr(all='raise') 

def test_accuracy_checker(h = 0.6762):
    ellmin = 2
    ellmax = 2500
    try:
        # ABCMB:
        specs = {
            "lensing" : True,
            "l_max" : ellmax
        }

        params = {
            'h': h,
            'omega_cdm': 0.1193,
            'omega_b': 0.0225,
            'A_s': 2.12424e-9,
            'n_s': 0.9709,
            'Neff': 3.044,
            'YHe': 0.245,
            'TCMB0': 2.34865418e-4,
            'T_nu': (4. / 11.)**(1. / 3.),
            'N_ncdm': 0,
            'T_ncdm': 0.71611,
        }

        model = Model(specs) # ZZ: model now takes ellmin, ellmax for Cls, and want_lensing
        ABC_Cls, ABC_ell = model.run_cosmology(params)
        ABC_tt = ABC_Cls[0] 
        ABC_te = ABC_Cls[1] 
        ABC_ee = ABC_Cls[2] 
        
        params = model.add_derived_parameters(params)

        # CLASS:
        CLASS_params = {
            "output": "mPk, tCl, pCl, lCl",
            "l_max_scalars" : specs["l_max"],
            "lensing" : "yes",
            "H0": params["h"]*100,
            "omega_b": params["omega_b"],
            "omega_cdm": params["omega_cdm"],
            "A_s" : params["A_s"],
            "n_s" : params["n_s"],
            "N_ur": params["N_ur"],
            "YHe": params["YHe"],
            "N_ncdm": params["N_ncdm"],
            "reio_parametrization" : "reio_camb",
            "z_reio" : 11,
            "reionization_width" : 0.5,
            "helium_fullreio_redshift" : 3.5,
            "helium_fullreio_width" : 0.5,
            "reionization_exponent" : 1.5,
            "l_max_g": 15,
            "l_max_pol_g": 10,
            "l_max_ur": 12, 
            "l_max_ncdm":17,
            "radiation_streaming_trigger_tau_over_tau_k" : 20000,
            "radiation_streaming_trigger_tau_c_over_tau" : 2000,
            "ur_fluid_trigger_tau_over_tau_k" : 10000, 
            "ncdm_fluid_trigger_tau_over_tau_k" : 15000
        }

        CLASS_Model = Class()
        CLASS_Model.set(CLASS_params)

        CLASS_Model.compute()
        cl = CLASS_Model.lensed_cl(ellmax)
        cltt=cl["tt"][ellmin:]
        ell = cl["ell"][ellmax:]

        # Compare all ells
        err_tt = abs(cltt-ABC_tt)/cltt
        print(err_tt.max())

        assert max(err_tt) <= 0.01, f"Accuracy check failed: {err_tt}"
    
    except Exception as e:
        pytest.fail(f"accuracy_checks raised an exception: {e}")

# print(test_accuracy_checker())
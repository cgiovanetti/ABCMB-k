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
from ABCMB import species
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
        # Setup

        params = {
            'h': 0.6762,
            'omega_cdm': 0.1193,
            'omega_b': 0.0225,
            'A_s': 2.12424e-9,
            'n_s': 0.9709,
            'Neff': 3.044,
            'YHe': 0.245,
            'TCMB0': 2.34865418e-4,
            'T_nu': (4./11.)**(1./3.),
            'N_ncdm': 0,
            'T_ncdm': 0.71611,
            'm_ncdm': 0.06,
        }

        specs = {
            "output_Cl" : True,
            "l_max" : ellmax,
            "lensing" : True,
            "output_Pk" : True,
            "output_k_max" : 0.5,
            "l_max_g" : 12,
            "l_max_pol_g" : 10,
            "l_max_ur" : 17,
            "l_max_ncdm" : 17
        }
        if params["N_ncdm"] > 0:
            user_species = (
                species.MassiveNeutrino,
            )
        else:
            user_species = None

        model = Model(
            user_species=user_species,
            input_specs=specs
        ) 
        params = model.add_derived_parameters(params)

        # CLASS
        CLASS_params = {
            "output": "mPk, tCl, pCl, lCl" if specs["lensing"] else "mPk, tCl, lCl",
            #"temperature_contributions" : "tsw",
            "l_max_scalars" : ellmax,
            "P_k_max_1/Mpc" : specs["output_k_max"],
            "lensing" : "yes" if specs["lensing"] else "no",
            "H0": params["h"]*100,
            "omega_b": params["omega_b"],
            "omega_cdm": params["omega_cdm"],
            "A_s" : params["A_s"],
            "n_s" : params["n_s"],
            "N_ur": params["N_ur"],
            "YHe": params["YHe"],
            "N_ncdm": params["N_ncdm"],
            #"reio_parametrization" : "reio_none",
            "reio_parametrization" : "reio_camb",
            "z_reio" : 11,
            "reionization_width" : 0.5,
            "helium_fullreio_redshift" : 3.5,
            "helium_fullreio_width" : 0.5,
            "reionization_exponent" : 1.5,
            "l_max_g": specs["l_max_g"],
            "l_max_pol_g": specs["l_max_pol_g"],
            "l_max_ur": specs["l_max_ur"], 
            "l_max_ncdm":specs["l_max_ncdm"],
            "radiation_streaming_trigger_tau_over_tau_k" : 20000,
            "radiation_streaming_trigger_tau_c_over_tau" : 2000,
            "ur_fluid_trigger_tau_over_tau_k" : 10000, 
            "ncdm_fluid_trigger_tau_over_tau_k" : 15000
        } 

        CLASS_Model = Class()
        CLASS_Model.set(CLASS_params)
        if params["N_ncdm"] > 0:
            CLASS_Model.set({"m_ncdm": params["m_ncdm"], "T_ncdm": params["T_ncdm"]})

        CLASS_Model.compute()
        if specs["lensing"]:
            cl = CLASS_Model.lensed_cl(ellmax)
        else:
            cl = CLASS_Model.raw_cl(ellmax)
        cltt=cl["tt"][ellmin:]
        ell = cl["ell"][ellmax:]

        # ABCMB

        data, label = model.run_cosmology(params)
        ells = label[0]

        ABC_tt = data[0] 
        ABC_te = data[1] 
        ABC_ee = data[2] 

        # Compare all ells
        err_tt = abs(cltt-ABC_tt)/cltt
        print(err_tt.max())

        ABC_Pk = data[3]
        ABC_k = label[1]
        CLA_Pk = np.vectorize(CLASS_Model.pk)(ABC_k, 0.)
        err_Pk = abs(CLA_Pk-ABC_Pk)/CLA_Pk

        assert max(err_tt) <= 0.01, f"Accuracy check failed: {err_tt}"
        assert max(err_Pk) <= 0.01, f"Accuracy check failed: {err_pk}"
    
    except Exception as e:
        pytest.fail(f"accuracy_checks raised an exception: {e}")

#print(test_accuracy_checker())
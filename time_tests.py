import sys
sys.path.append('../')
import os


import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
import abcmb.spectrum as spectrum
from scipy.interpolate import interp1d
import jax.numpy as jnp
import numpy as np


import time

h = 0.6762

specs = {
    "output_Cl" : True,
    "output_Pk" : True,
    "lensing" : True
}

model = Model(specs)

for i in range(2):
    start=time.time()
    # ABCMB:
    # params = {
    #     'h': jnp.asarray(h),
    #     'omega_cdm': jnp.asarray(0.1193),
    #     'omega_b': jnp.asarray(0.0225),
    #     'A_s': jnp.asarray(2.12424e-9),
    #     'n_s': jnp.asarray(0.9709),
    #     #'Neff': 3.044,
    #     #'Delta_Neff_init': jnp.asarray(0.),
    #     #'YHe': jnp.asarray(0.245),
    #     'TCMB0': jnp.asarray(2.34865418e-4),
    #     #'T_nu': jnp.asarray((4. / 11.)**(1. / 3.) * 2.34865418e-4),
    #     # 'N_ncdm': jnp.asarray(0.),
    #     # 'T_ncdm': jnp.asarray(0.71611),
    #     # 'm_ncdm': jnp.asarray(0.06)
    # }
    params = {}
    
    out, aux = model.run_cosmology(params)

    print(out[0])
    print(time.time()-start)

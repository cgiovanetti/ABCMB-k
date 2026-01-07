import sys
sys.path.append('/home/zz1994/packages/ABCMB')
sys.path.append('/home/zz1994/packages/ABCMB/ABCMB')
import os

import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
from ABCMB.main import Model
import ABCMB.spectrum as spectrum
from scipy.interpolate import interp1d
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import time

omega_bs = 0.02 + 0.001*np.arange(11)
times = np.zeros(omega_bs.size-1)

specs = {
    "output_Cl" : True,
    "output_Pk" : True,
    "lensing" : True
}

model = Model(specs)

def f(omega_b):
    params = {"omega_b" : omega_b}
    return model.run_cosmology(params)

for i, omega_b in enumerate(omega_bs):
    
    start=time.time()
    output, aux = f(jnp.array(omega_b))
    rtime=time.time() - start

    if i == 0:
        # Skip the first one for compile time.
        print("Compile time: {}s".format(rtime))
    else:
        # Save the run times
        times[i-1] = rtime

mean = np.mean(times)
sqdiff = (times - mean)**2
sigma2 = np.sum(sqdiff)/times.size
sigma = np.sqrt(sigma2)

print("Performed {} runs with average {}s +/- {}s".format(times.size, mean, sigma))
print(times)
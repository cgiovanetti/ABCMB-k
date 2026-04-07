import sys
sys.path.append('../')

import jax
print(jax.devices())
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model

import time

specs = {
    "output_Cl" : True,
    "output_Pk" : True,
    "lensing" : True,
}

model = Model(**specs)

for i in range(2):
    start=time.time()

    params = {}
    out= model(params)

    print(out.ClTT)
    print(time.time()-start)


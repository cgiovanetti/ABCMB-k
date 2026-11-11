"""print the exact parameter dict cobaya passes to CLASS for the MH reference (hmc/reference/lcdm_class_mh.yaml)"""
import os
import pandas as pd
pd.options.future.infer_string = False
import yaml
from cobaya.model import get_model
HERE = os.path.dirname(os.path.abspath(__file__))
info = yaml.safe_load(open(os.path.join(HERE, "..", "reference", "lcdm_class_mh.yaml")))
for k in ("output", "sampler", "resume"):
    info.pop(k, None)
model = get_model(info)
pt = dict(omega_b=0.0223631, omega_cdm=0.119912, H0=67.8825, logA=3.04285, n_s=0.965061, tau_reio=0.0537985,
          A_planck=1.00051)
model.loglikes(pt)
th = model.theory["classy"]
print("classy.pars:", dict(sorted(th.classy.pars.items())))

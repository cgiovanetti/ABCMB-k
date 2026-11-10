"""one-point check of the reference yaml (CLASS logp at the HMC START point): python hmc/reference/eval_once.py"""
import pandas as pd
pd.options.future.infer_string = False
import os, yaml
from cobaya.run import run
here = os.path.dirname(os.path.abspath(__file__))
info = yaml.safe_load(open(os.path.join(here, "lcdm_class_mh.yaml")))
info.pop("output"); info["resume"] = False
info["sampler"] = {"evaluate": {"override": {"omega_b": 0.022383, "omega_cdm": 0.12011, "H0": 67.32,
                                             "logA": 3.0448, "n_s": 0.96605, "tau_reio": 0.0543,
                                             "A_planck": 1.0}}}
run(info)

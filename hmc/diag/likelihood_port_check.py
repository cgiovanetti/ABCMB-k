"""The HMC's JAX likelihoods vs cobaya's native ones on identical CLASS spectra (the MH reference settings), at the CLASS
posterior mean and at the old HMC start point; and the Monte Carlo error of both samplers' posterior means
(chain-to-chain scatter). CPU only:  python hmc/diag/likelihood_port_check.py
"""
import os
import pandas as pd
pd.options.future.infer_string = False
import numpy as np
import yaml
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from cobaya.model import get_model
from scan.lowl_like import LowLTT, LowLEE, cl_to_Dl, LOWEE_TABLES
from hmc.plik_lite import PlikLite

HERE = os.path.dirname(os.path.abspath(__file__))
info = yaml.safe_load(open(os.path.join(HERE, "..", "reference", "lcdm_class_mh.yaml")))
for k in ("output", "sampler", "resume"):
    info.pop(k, None)
model = get_model(info)
plite, lowtt, lowee = PlikLite(), LowLTT(), LowLEE(table_path=LOWEE_TABLES["simall"])
POINTS = {"posterior mean": dict(omega_b=0.0223631, omega_cdm=0.119912, H0=67.8825, logA=3.04285, n_s=0.965061,
                                  tau_reio=0.0537985, A_planck=1.00051),
          "old start": dict(omega_b=0.022383, omega_cdm=0.12011, H0=67.32, logA=3.0448, n_s=0.96605, tau_reio=0.0543,
                            A_planck=1.0)}
names = ["planck_2018_highl_plik.TTTEEE_lite_native", "planck_2018_lowl.TT", "planck_2018_lowl.EE"]
for lab, pt in POINTS.items():
    ll = model.loglikes(pt, as_dict=True)[0]
    cob = [-2 * ll[n] for n in names]
    cl = model.provider.get_Cl(ell_factor=False, units="1")                 # dimensionless C_ell, ell 0..lmax
    ells = np.asarray(cl["ell"])[2:2509]
    c3 = [jnp.asarray(np.asarray(cl[k])[2:2509]) for k in ("tt", "te", "ee")]
    D = [cl_to_Dl(c, jnp.asarray(ells)) for c in c3]
    A = pt["A_planck"]
    jx = [float(plite.chi2(plite.bin_model(*D), A)), float(lowtt.chi2(D[0], calib=A)), float(lowee.chi2(D[2], calib=A))]
    print(f"{lab}: chi2 cobaya native (plik_lite, lowTT, lowEE) " + "  ".join(f"{v:9.3f}" for v in cob)
          + "  | HMC JAX " + "  ".join(f"{v:9.3f}" for v in jx) + "  | JAX - cobaya "
          + "  ".join(f"{a - b:+.4f}" for a, b in zip(jx, cob)))

# ---- Monte Carlo error of the posterior means ----
d = np.load(os.path.join(HERE, "..", "results", "lcdm_v1.npz"))
th = d["h_theta"][int(d["n_warmup"]):]                     # (n, B, D)
cm = th.mean(0)                                             # per-chain means (B, D)
sd = th.reshape(-1, th.shape[-1]).std(0)
print("\nHMC: per-chain-mean scatter / sqrt(B) in posterior sds (64 chains x 20 draws):")
print("   " + "  ".join(f"{n} {v:.3f}" for n, v in zip(d["names"], cm.std(0, ddof=1) / np.sqrt(len(cm)) / sd)))
ref = os.path.join(HERE, "..", "results", "ref")
cols = ["omega_b", "omega_cdm", "H0", "logA", "n_s", "tau_reio", "A_planck"]
means, allc = [], []
for i in range(1, 9):
    f = os.path.join(ref, f"class_mh.{i}.txt")
    hdr = open(f).readline().lstrip("#").split()
    x = np.loadtxt(f)
    x = x[int(0.3 * len(x)):]
    w = x[:, hdr.index("weight")]
    v = np.stack([x[:, hdr.index(c)] for c in cols], 1)
    means.append((w[:, None] * v).sum(0) / w.sum()); allc.append((w, v))
means = np.array(means)
W = np.concatenate([a for a, _ in allc]); V = np.concatenate([b for _, b in allc])
m = (W[:, None] * V).sum(0) / W.sum(); s = np.sqrt((W[:, None] * (V - m) ** 2).sum(0) / W.sum())
print("CLASS MH: per-chain-mean scatter / sqrt(8) in posterior sds (8 chains, 30% burn-in), and the halves split:")
print("   " + "  ".join(f"{n} {v:.3f}" for n, v in zip(cols, means.std(0, ddof=1) / np.sqrt(len(means)) / s)))
print("   first-half chains vs second-half chains mean difference / sd: "
      + "  ".join(f"{n} {v:+.3f}" for n, v in zip(cols, (means[:4].mean(0) - means[4:].mean(0)) / s)))

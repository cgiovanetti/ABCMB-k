"""ABCMB-k with halofit lensing (nonlinear="halofit", the engine otherwise as hmc/posterior.py) vs CLASS with halofit /
HMcode / linear lensing, default and high precision, at the CLASS posterior mean: plik-lite chi2 and the posterior
shift each spectrum difference predicts at linear order, -Sigma_post J^T W delta (Sigma_post from the _v1 HMC draws,
J = ABCMB-k's linear-lensing C_ell tangents from hmc/diag/out/theory_vs_class.npz). GPU node:
  python hmc/diag/halofit_check.py
"""
import os, time
from hmc.posterior import FIXED                                         # sets XLA_FLAGS before jax starts
import numpy as np
import jax
import jax.numpy as jnp
from classy import Class
from abcmb.main import Model
from scan.lowl_like import cl_to_Dl
from hmc.plik_lite import PlikLite

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "theory_vs_class.npz"))
TH, ells = d["theta"], d["ells"]
LMAX = int(ells[-1]); A = TH[6]
SIG = np.array([0.00015, 0.0012, 0.0054, 0.014, 0.0042, 0.0073])
NAMES = ["omega_b", "omega_cdm", "h", "ln10As", "n_s", "tau_reion", "A_planck"]
OBS = np.array([0.010, 0.036, -0.029, 0.048, -0.119, 0.063, 0.042])    # observed HMC - MH (analyze.py _v1)

ndev = len(jax.devices("gpu"))
pdict = dict(FIXED)
pdict.update(omega_b=TH[0], omega_cdm=TH[1], h=TH[2], A_s=float(np.exp(TH[3]) / 1e10), n_s=TH[4], tau_reion=TH[5])
t0 = time.time()
mh = Model(output_Cl=True, l_max=LMAX, lensing=True, output_Pk=False, l_max_g=12, l_max_pol_g=10,
           rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7, rtol_small_k_PE=1e-5, max_steps_PE=16384,
           lna_grid_mode="recomb", bbn_type="table", nonlinear="halofit")
o = mh.call_batched([pdict] * ndev)
clsAh = [np.asarray(c[0]) for c in (o.ClTT, o.ClTE, o.ClEE)]
assert np.array_equal(np.asarray(o.l), ells)
print(f"ABCMB-k halofit (value, incl. compile) {time.time() - t0:.0f}s; N_k {mh.specs.get('N_k', 'n/a')}", flush=True)
t0 = time.time()
o = mh.call_batched([pdict] * ndev)
print(f"ABCMB-k halofit warm value call ({ndev} cosmologies) {time.time() - t0:.0f}s", flush=True)

BASE = dict(output="tCl,pCl,lCl", lensing="yes", l_max_scalars=LMAX, N_ur=3.044, N_ncdm=0, delta_l_max=1254,
            omega_b=TH[0], omega_cdm=TH[1], H0=100 * TH[2], A_s=float(np.exp(TH[3]) / 1e10), n_s=TH[4],
            tau_reio=TH[5])
HP = dict(recombination="HyRec", l_logstep=1.025, l_linstep=20, perturbations_sampling_stepsize=0.05,
          l_switch_limber=30, hyper_sampling_flat=32, l_max_g=40, l_max_ur=35, l_max_pol_g=60,
          ur_fluid_approximation=2, ur_fluid_trigger_tau_over_tau_k=130, radiation_streaming_approximation=2,
          radiation_streaming_trigger_tau_over_tau_k=240, hyper_flat_approximation_nu=7000,
          transfer_neglect_delta_k_S_t0=0.17, transfer_neglect_delta_k_S_t1=0.05,
          transfer_neglect_delta_k_S_t2=0.17, transfer_neglect_delta_k_S_e=0.17, accurate_lensing=1,
          start_small_k_at_tau_c_over_tau_h=0.0004, start_large_k_at_tau_h_over_tau_k=0.05,
          tight_coupling_trigger_tau_c_over_tau_h=0.005, tight_coupling_trigger_tau_c_over_tau_k=0.008,
          start_sources_at_tau_c_over_tau_h=0.006)


def class_cls(extra):
    c = Class(); c.set({**BASE, **extra}); c.compute()
    le = c.lensed_cl(LMAX); out = [np.asarray(le[k])[ells] for k in ("tt", "te", "ee")]
    c.struct_cleanup(); c.empty()
    return out


S = {"ABCMB-k linear": [d["clsA"][i] for i in range(3)], "ABCMB-k halofit": clsAh,
     "CLASS linear": [d["clsC"][i] for i in range(3)], "CLASS hp linear": [d["clsH"][i] for i in range(3)],
     "CLASS halofit": class_cls(dict(non_linear="halofit")), "CLASS hmcode": class_cls(dict(non_linear="hmcode")),
     "CLASS hp halofit": class_cls({**HP, "non_linear": "halofit"})}

pl = PlikLite()
Cinv = np.asarray(pl.invcov); X = np.asarray(pl.X_data)


def binned(c3):
    return np.asarray(pl.bin_model(*[cl_to_Dl(jnp.asarray(c), jnp.asarray(ells), lmax=LMAX) for c in c3])) / A ** 2


M = {k: binned(v) for k, v in S.items()}
for k, m in M.items():
    print(f"plik-lite chi2 {k:17s} {(X - m) @ Cinv @ (X - m):9.3f}")

dcl = d["dclA"]
mA = M["ABCMB-k linear"]
Jb = np.stack([np.asarray(pl.bin_model(*[cl_to_Dl(jnp.asarray(dcl[s][j]), jnp.asarray(ells), lmax=LMAX)
                                         for s in range(3)])) / A ** 2 for j in range(6)], 1)
Jb = np.concatenate([Jb, (-2.0 * mA / A)[:, None] * 0.0025], 1)
scale = np.concatenate([SIG, [0.0025]])
hd = np.load(os.path.join(HERE, "..", "results", "lcdm_v1.npz"))
Sig = np.cov(hd["h_theta"][int(hd["n_warmup"]):].reshape(-1, 7) / scale, rowvar=False)
psd = np.sqrt(np.diag(Sig))


def shift(a, b):
    return -(Sig @ (Jb.T @ Cinv @ (M[a] - M[b]))) / psd


print("\npredicted posterior shift (theory 1 - theory 2), posterior sds:")
print(f"   {'':36s} " + " ".join(f"{n:>9s}" for n in NAMES))
for a, b in (("ABCMB-k halofit", "CLASS halofit"), ("ABCMB-k halofit", "CLASS hp halofit"),
             ("CLASS halofit", "CLASS hp halofit"), ("ABCMB-k linear", "CLASS hp linear"),
             ("CLASS linear", "CLASS hp linear"), ("ABCMB-k halofit", "CLASS hmcode"),
             ("ABCMB-k linear", "CLASS hmcode")):
    print(f"   {a + ' - ' + b:36s} " + " ".join(f"{v:+9.3f}" for v in shift(a, b)))
print(f"   {'observed HMC - MH':36s} " + " ".join(f"{v:+9.3f}" for v in OBS))


def rel(a, b):
    return a[0] / b[0] - 1, (a[1] - b[1]) / np.sqrt(b[0] * b[2]), a[2] / b[2] - 1


sel = ells >= 30
print("\nmax |relative difference| ell 30-2508, TT  TE/sqrt(TTEE)  EE:")
for a, b in (("ABCMB-k halofit", "CLASS halofit"), ("ABCMB-k halofit", "CLASS hp halofit"),
             ("CLASS halofit", "CLASS hp halofit")):
    print(f"   {a + ' / ' + b:36s} " + "  ".join(f"{np.abs(x[sel]).max():.2e}" for x in rel(S[a], S[b])))
np.savez(os.path.join(HERE, "out", "halofit_check.npz"), ells=ells, **{k.replace(" ", "_"): np.array(v)
                                                                       for k, v in S.items()})

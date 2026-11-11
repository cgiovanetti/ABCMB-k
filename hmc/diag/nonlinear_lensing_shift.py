"""Does the MH reference's non-linear lensing explain the HMC - MH posterior differences?

cobaya's classy wrapper passed non_linear='hmcode' (hmc/diag/cobaya_class_args.py); the HMC's ABCMB-k model lenses with
the linear spectrum. With ABCMB-k's own C_ell tangents (hmc/diag/out/theory_vs_class.npz) this predicts, at linear
order (plik-lite Fisher), the posterior shift between two theories delta = m_1 - m_2 at the same theta:
-(J^T W J)^-1 J^T W delta, and compares it with the observed HMC - MH shift (hmc/analyze.py _v1). CPU only.
"""
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from classy import Class
from scan.lowl_like import cl_to_Dl
from hmc.plik_lite import PlikLite

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "out", "theory_vs_class.npz"))
TH, ells = d["theta"], d["ells"]
LMAX = int(ells[-1]); A = TH[6]
SIG = np.array([0.00015, 0.0012, 0.0054, 0.014, 0.0042, 0.0073])        # the tangents' direction scale (posterior.SIG)
NAMES = ["omega_b", "omega_cdm", "h", "ln10As", "n_s", "tau_reion", "A_planck"]
# observed HMC - MH mean shifts in posterior sds (analyze.py _v1, H0 for h)
OBS = np.array([0.010, 0.036, -0.029, 0.048, -0.119, 0.063, 0.042])

BASE = dict(output="tCl,pCl,lCl", lensing="yes", l_max_scalars=LMAX, N_ur=3.044, N_ncdm=0, delta_l_max=1254,
            omega_b=TH[0], omega_cdm=TH[1], H0=100 * TH[2], A_s=float(np.exp(TH[3]) / 1e10), n_s=TH[4],
            tau_reio=TH[5])


def class_cls(extra):
    c = Class(); c.set({**BASE, **extra}); c.compute()
    le = c.lensed_cl(LMAX); out = [np.asarray(le[k])[ells] for k in ("tt", "te", "ee")]
    c.struct_cleanup(); c.empty()
    return out


pl = PlikLite()
Cinv = np.asarray(pl.invcov); X = np.asarray(pl.X_data)


def binned(c3):
    return np.asarray(pl.bin_model(*[cl_to_Dl(jnp.asarray(c), jnp.asarray(ells), lmax=LMAX) for c in c3])) / A ** 2


S = {"ABCMB-k (linear)": [d["clsA"][i] for i in range(3)], "CLASS linear": [d["clsC"][i] for i in range(3)],
     "CLASS hmcode": class_cls(dict(non_linear="hmcode")), "CLASS halofit": class_cls(dict(non_linear="halofit"))}
M = {k: binned(v) for k, v in S.items()}
for k, m in M.items():
    print(f"plik-lite chi2 {k:17s} {(X - m) @ Cinv @ (X - m):9.3f}")

dcl = d["dclA"]                                                         # (3, NC, n_l) per SIG_j e_j
mA = M["ABCMB-k (linear)"]
Jb = np.stack([np.asarray(pl.bin_model(*[cl_to_Dl(jnp.asarray(dcl[s][j]), jnp.asarray(ells), lmax=LMAX)
                                         for s in range(3)])) / A ** 2 for j in range(6)], 1)
Jb = np.concatenate([Jb, (-2.0 * mA / A)[:, None] * 0.0025], 1)
F = Jb.T @ Cinv @ Jb + np.diag([0.0] * 6 + [1.0])
scale = np.concatenate([SIG, [0.0025]])
psd = np.sqrt(np.diag(np.linalg.inv(F))) * scale


# the full posterior covariance (all likelihoods + priors) from the HMC draws, in the tangents' units; a theory
# change that only touches plik-lite moves the posterior mean by -Sigma_post J^T W delta at linear order
hd = np.load(os.path.join(HERE, "..", "results", "lcdm_v1.npz"))
draws = hd["h_theta"][int(hd["n_warmup"]):].reshape(-1, 7) / scale
Sig = np.cov(draws, rowvar=False)
psd_post = np.sqrt(np.diag(Sig))


def shift(m1, m2, fisher=False):
    if fisher:
        return -np.linalg.solve(F, Jb.T @ Cinv @ (m1 - m2)) * scale / psd
    return -(Sig @ (Jb.T @ Cinv @ (m1 - m2))) / psd_post


print("\npredicted posterior shift (theory 1 - theory 2), in posterior sds (Sigma_post from the HMC draws):")
print(f"   {'':34s} " + " ".join(f"{n:>9s}" for n in NAMES))
for a, b in (("CLASS linear", "CLASS hmcode"), ("ABCMB-k (linear)", "CLASS linear"),
             ("ABCMB-k (linear)", "CLASS hmcode"), ("CLASS halofit", "CLASS hmcode")):
    print(f"   {a + ' - ' + b:34s} " + " ".join(f"{v:+9.3f}" for v in shift(M[a], M[b])))
print(f"   {'observed HMC - MH (MC err ~0.04)':34s} " + " ".join(f"{v:+9.3f}" for v in OBS))

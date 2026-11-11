"""ABCMB-k (the HMC engine, hmc/posterior.py settings) vs CLASS 3.3.4 at one point of the Planck-lite LCDM posterior.

CLASS runs twice, both with LINEAR lensing like the `_v1` HMC: the MH reference's settings without its HMcode lensing
(defaults + N_ur 3.044, N_ncdm 0, delta_l_max 1254; "CLASS ref" below) and with the ACT DR6 high-precision block (arXiv:2503.14454 Fig. 48, minus the nonlinear / mPk keys),
which stands in for the truth. Prints derived quantities, chi2 per likelihood for each spectrum set (the same JAX
likelihoods), the plik-lite chi2 difference by block and ell range, and the parameter shift the ABCMB-k - CLASS spectrum
difference predicts at linear order, -(J^T W J)^-1 J^T W delta (J = ABCMB-k's C_ell tangents, W = plik-lite's inverse
covariance). Writes hmc/diag/out/theory_vs_class<tag>.{npz,png}.

  python hmc/diag/theory_vs_class.py [tag]        # on a GPU node, PYTHONPATH = the ABCMB-k root
"""
import os, sys, time
from hmc.posterior import LCDMPosterior, FIXED, NC, NAMES, SIG          # sets XLA_FLAGS before jax starts
import numpy as np
import jax
import jax.numpy as jnp
from classy import Class
from scan.lowl_like import cl_to_Dl
from abcmb.main import Model

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
T_UK = 2.7255e6

# the CLASS MH posterior mean (hmc/analyze.py _v1 table)
TH = np.array([0.0223631, 0.119912, 0.678825, 3.04285, 0.965061, 0.0537985, 1.00051])
print("point: " + " ".join(f"{n} {v:.6g}" for n, v in zip(NAMES, TH)), flush=True)

post = LCDMPosterior()
model = post.model
ells = np.asarray(post.ells)
LMAX = int(ells[-1])

# ---------------- ABCMB-k: Cls + tangents (the HMC's staged path), unlensed, derived ----------------
t0 = time.time()
reps = max(post.ndev, 1)
cls_k, dcl_k = post._cls_and_dots(np.repeat(TH[None], reps, axis=0))
clsA = [np.asarray(c[0]) for c in cls_k]                     # TT, TE, EE on ells
dclA = [np.asarray(d[0]) for d in dcl_k]                     # (NC, n_l) per SIG_j e_j
print(f"ABCMB-k lensed Cls + tangents {time.time() - t0:.0f}s", flush=True)

pdict = dict(FIXED)
pdict.update(omega_b=TH[0], omega_cdm=TH[1], h=TH[2], A_s=float(np.exp(TH[3]) / 1e10), n_s=TH[4], tau_reion=TH[5])
dp = model.add_derived_parameters(dict(pdict))
der_A = {k: float(dp[k]) for k in ("YHe", "Neff", "N_nu_massless", "T_nu_massless", "omega_m", "omega_r") if k in dp}
spec_kw = {k: v for k, v in model.specs.items() if k in (
    "l_max", "l_max_g", "l_max_pol_g", "rtol_large_k_PE", "atol_large_k_PE", "rtol_small_k_PE", "max_steps_PE",
    "lna_grid_mode", "bbn_type", "delta_l_max")}
print("ABCMB-k specs " + ", ".join(f"{k}={v}" for k, v in spec_kw.items()), flush=True)
t0 = time.time()
model_u = Model(output_Cl=True, l_max=model.specs["l_max"], lensing=False, output_Pk=False,
                l_max_g=model.specs["l_max_g"], l_max_pol_g=model.specs["l_max_pol_g"],
                rtol_large_k_PE=model.specs["rtol_large_k_PE"], atol_large_k_PE=model.specs["atol_large_k_PE"],
                rtol_small_k_PE=model.specs["rtol_small_k_PE"], max_steps_PE=model.specs["max_steps_PE"],
                lna_grid_mode=model.specs["lna_grid_mode"], bbn_type=model.specs["bbn_type"],
                delta_l_max=model.specs["delta_l_max"])
ou = model_u.call_batched([pdict] * reps)
clsAu = [np.asarray(c[0]) for c in (ou.ClTT, ou.ClTE, ou.ClEE)]
assert np.array_equal(np.asarray(ou.l), ells), "unlensed ell grid differs"
print(f"ABCMB-k unlensed {time.time() - t0:.0f}s", flush=True)

# ---------------- CLASS: reference settings and high precision ----------------
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
DER_C = ["YHe", "Neff", "100*theta_s", "z_rec", "rs_rec", "z_d", "rs_d", "z_reio", "tau_reio", "Omega_m"]


def run_class(extra):
    c = Class()
    c.set({**BASE, **extra})
    t0 = time.time()
    c.compute()
    le = c.lensed_cl(LMAX); ra = c.raw_cl(LMAX)
    der = c.get_current_derived_parameters(DER_C)
    out = ([np.asarray(le[k])[ells] for k in ("tt", "te", "ee")], [np.asarray(ra[k])[ells] for k in ("tt", "te", "ee")],
           der, time.time() - t0)
    c.struct_cleanup(); c.empty()
    return out


clsC, clsCu, der_C, tC = run_class({})
print(f"CLASS reference settings {tC:.0f}s", flush=True)
clsH, clsHu, der_H, tH = run_class(HP)
print(f"CLASS high precision {tH:.0f}s", flush=True)

# ---------------- derived quantities ----------------
print("\nderived: ABCMB-k " + ", ".join(f"{k} {v:.6g}" for k, v in der_A.items()))
for lab, d in (("CLASS ref", der_C), ("CLASS hp ", der_H)):
    print(f"         {lab} " + ", ".join(f"{k} {v:.6g}" for k, v in d.items()))
print(f"ABCMB-k Omega_m {(TH[0] + TH[1]) / TH[2] ** 2:.6g} (cdm + b; CLASS Omega_m includes the massless-nu "
      f"radiation term only through h)")

# ---------------- chi2 per likelihood, same JAX likelihoods ----------------
A = TH[6]
parts = {}
for lab, c3 in (("ABCMB-k", clsA), ("CLASS ref", clsC), ("CLASS hp", clsH)):
    parts[lab] = np.asarray(post._chi2_parts(tuple(jnp.asarray(c) for c in c3), A))
print("\nchi2 at the point (plik_lite, lowl_TT, lowl_EE, A prior):")
for lab, p in parts.items():
    print(f"  {lab:9s} " + "  ".join(f"{v:9.3f}" for v in p) + f"   total {p.sum():9.3f}")

# ---------------- plik-lite chi2 difference by block and ell range ----------------
from hmc.plik_lite import PlikLite
pl = PlikLite()


def binned(c3):
    D = [np.asarray(cl_to_Dl(jnp.asarray(c), jnp.asarray(ells), lmax=LMAX)) for c in c3]
    return np.asarray(pl.bin_model(*[jnp.asarray(d) for d in D])) / A ** 2


Cinv = np.asarray(pl.invcov); X = np.asarray(pl.X_data)
nb = [pl.nbin[s] for s in pl.used_specs]
blk = np.concatenate([[s] * n for s, n in zip(pl.used_specs, nb)])
lav = np.concatenate([np.asarray(pl._M[s] @ np.arange(pl.Lcol)) / np.asarray(pl._M[s].sum(1)) for s in pl.used_specs])
mA, mC, mH = binned(clsA), binned(clsC), binned(clsH)


def decompose(m_from, m_to, lab):
    """chi2(m_to) - chi2(m_from) = sum_i [C^-1 (2 r_from - ... )]: per-bin terms of -2 r.C^-1 d + d.C^-1 d."""
    r = X - m_from; d = (X - m_to) - r
    contrib = d * (Cinv @ (2 * r + d))
    print(f"\n{lab}: plik-lite chi2 difference {contrib.sum():+.3f}")
    edges = [30, 500, 1000, 1500, 2000, 2509]
    for s in pl.used_specs:
        row = [contrib[(blk == s) & (lav >= lo) & (lav < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])]
        print(f"   {s.upper()}  " + "  ".join(f"[{lo},{hi}) {v:+7.3f}" for (lo, hi), v in zip(zip(edges[:-1], edges[1:]), row))
              + f"   sum {contrib[blk == s].sum():+7.3f}")
    return contrib


cAC = decompose(mC, mA, "ABCMB-k - CLASS ref")
cAH = decompose(mH, mA, "ABCMB-k - CLASS hp")
cCH = decompose(mH, mC, "CLASS ref - CLASS hp")

# ---------------- parameter shift predicted by the spectrum difference ----------------
# tangents of the binned model per SIG_j e_j (bin_model is linear in D_ell); A_planck enters as -2 m / A
Jb = np.stack([np.asarray(pl.bin_model(*[jnp.asarray(cl_to_Dl(jnp.asarray(dclA[s][j]), jnp.asarray(ells), lmax=LMAX))
                                         for s in range(3)])) / A ** 2 for j in range(NC)], 1)
Jb = np.concatenate([Jb, (-2.0 * mA / A)[:, None] * 0.0025], 1)              # last column per 0.0025 in A_planck
F = Jb.T @ Cinv @ Jb + np.diag([0.0] * NC + [1.0])                          # + the A_planck prior (unit in these units)
scale = np.concatenate([SIG[:NC], [0.0025]])
post_sd = np.sqrt(np.diag(np.linalg.inv(F))) * scale
for lab, mref in (("CLASS ref", mC), ("CLASS hp", mH)):
    delta = mA - mref                                                       # ABCMB-k minus CLASS at the same theta
    dth = -np.linalg.solve(F, Jb.T @ Cinv @ delta) * scale
    print(f"\npredicted ABCMB-k - {lab} posterior shift (plik-lite Fisher, linear):")
    print("   " + "  ".join(f"{n} {v / s:+.3f} sd" for n, v, s in zip(NAMES, dth, post_sd)))

# ---------------- spectra ratios ----------------
def rel(a, b):
    tt = a[0] / b[0] - 1; ee = a[2] / b[2] - 1; te = (a[1] - b[1]) / np.sqrt(b[0] * b[2])
    return tt, te, ee


R = {"ABCMB-k / CLASS ref": (rel(clsA, clsC), rel(clsAu, clsCu)),
     "ABCMB-k / CLASS hp": (rel(clsA, clsH), rel(clsAu, clsHu)),
     "CLASS ref / CLASS hp": (rel(clsC, clsH), rel(clsCu, clsHu))}
print("\nmax |relative difference| over ell 30-2508 (lensed | unlensed), TT  TE/sqrt(TTEE)  EE:")
sel = ells >= 30
for lab, (le, un) in R.items():
    print(f"  {lab:21s} " + "  ".join(f"{np.abs(x[sel]).max():.2e}" for x in le) + "  |  "
          + "  ".join(f"{np.abs(x[sel]).max():.2e}" for x in un))

np.savez(os.path.join(OUT, f"theory_vs_class{TAG}.npz"), theta=TH, ells=ells, clsA=clsA, clsAu=clsAu, clsC=clsC,
         clsCu=clsCu, clsH=clsH, clsHu=clsHu, dclA=dclA, chi2_parts=np.array(list(parts.values())),
         contrib_AC=cAC, contrib_AH=cAH, contrib_CH=cCH, lav=lav, blk=blk)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
cols = {"ABCMB-k / CLASS ref": "#2a78d6", "ABCMB-k / CLASS hp": "#eb6834", "CLASS ref / CLASS hp": "#1baf7a"}
fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
for row, which in enumerate(("lensed", "unlensed")):
    for col, nm in enumerate(("TT", "TE / sqrt(TT EE)", "EE")):
        a = ax[row, col]
        for lab, (le, un) in R.items():
            a.plot(ells, (le if row == 0 else un)[col], lw=1.2, color=cols[lab], label=lab)
        a.axhline(0, color="0.6", lw=0.8)
        a.set_title(f"{which} {nm}", fontsize=10)
        a.set_xscale("log"); a.set_xlim(2, LMAX)
        a.grid(alpha=0.25)
ax[0, 0].legend(fontsize=8, frameon=False)
for a in ax[1]:
    a.set_xlabel(r"$\ell$")
ax[0, 0].set_ylabel("relative difference"); ax[1, 0].set_ylabel("relative difference")
fig.suptitle("ABCMB-k (HMC engine) vs CLASS 3.3.4 at the CLASS posterior mean", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"theory_vs_class{TAG}.png"), dpi=130)
print(f"\nwrote {OUT}/theory_vs_class{TAG}.(npz|png)")

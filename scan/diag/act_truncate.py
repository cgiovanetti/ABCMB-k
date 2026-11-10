"""chi2 change from feeding candl's ACT DR6 lite D_ell that stop at L (zero above), from the c6325_dl5000 spectra."""
import os, numpy as np, jax
jax.config.update("jax_enable_x64", True)
import candl, candl_data
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "act_lmax")
like = candl.Like(candl_data.ACT_DR6_TTTEEE); LT = int(like.ells[-1])
d = np.load(os.path.join(D, "c6325_dl5000.npz")); ells, cls = d["ells"], d["cls"]
def chi2(b, L):
    Dl = {}
    for i, s in enumerate(("TT", "TE", "EE")):
        full = np.zeros(LT + 1); m = ells <= L
        full[ells[m]] = cls[b, i, m] * ells[m] * (ells[m] + 1) / (2 * np.pi) * 2.7255e6 ** 2
        Dl[s] = full[2:]
    return float(like.chi_square({"Dl": Dl, "A_act": 1.0, "P_act": 1.0}))
for L in (5500, 6000, 6200, 6300, 6325):
    dd = [chi2(b, L) - chi2(b, 6325) for b in range(len(cls))]
    print(f"D_ell zero above {L}: chi2 - chi2(6325) = " + " ".join(f"{v:8.2f}" for v in dd))

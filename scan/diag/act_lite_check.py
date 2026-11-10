"""act_lite_check.py — scan/act_lite.ACTLite against candl's own chi_square, and its (A_act, P_act) profile
against a brute-force minimum, on the spectra of act_lmax_check (scan/results/act_lmax/c6325_dl5000.npz).
CPU; the abcmb-k venv (candl)."""
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
from scan.act_lite import ACTLite

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "act_lmax")
d = np.load(os.path.join(D, "c6325_dl5000.npz"))
act = ACTLite()
print(f"windows end at l {act.l_window_max}; {act.nbin} bins x 3")
for b in range(3):
    D3 = act.abcmb_cls_to_clik(*(jnp.asarray(d["cls"][b, i]) for i in range(3)), d["ells"])
    for nu in ([1.0, 1.0], [1.002, 0.995], [0.997, 1.006]):
        nu = jnp.asarray(nu)
        mine, ref = float(act.data_chi2(D3, nu)), float(act.candl_chi2(D3, nu))
        print(f"cosmology {b} nu {np.asarray(nu)}: ACTLite {mine:.6f}  candl {ref:.6f}  diff {mine - ref:.2e}")
    c, nu = act.profile(D3)
    f = lambda v: float(act.data_chi2(D3, jnp.asarray(v)) + act.prior_penalty(jnp.asarray(v)))
    bf = minimize(f, [1.0, 1.0], method="Nelder-Mead", options=dict(xatol=1e-10, fatol=1e-10, maxiter=4000))
    print(f"  profile: chi2 {float(c):.6f} at A {float(nu[0]):.6f} P {float(nu[1]):.6f}; "
          f"Nelder-Mead {bf.fun:.6f} at {bf.x[0]:.6f} {bf.x[1]:.6f}")

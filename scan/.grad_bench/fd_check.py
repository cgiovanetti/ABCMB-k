"""Central finite differences of the bench loss along scaling-sigma directions vs the staged AD
gradient (frozen step mesh + step_ts). Loads row 0 of a saved grad npz from bench_fvu (grad in
units of d loss / d (SIG[j] e_j)) and evaluates the loss with call_batched at CEN +- h*SIG[j]*e_j
for the directions given. Usage: python fd_check.py <grad_npz> [h=0.01] [dirs=0,2,4,6]"""
import os, sys, json
ROOT = "/pscratch/sd/c/carag/ABCMB-k-grad"
sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import abcmb
assert abcmb.__file__.startswith(ROOT)
from abcmb.main import Model
npz = sys.argv[1]; H = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
DIRS = [int(d) for d in (sys.argv[3] if len(sys.argv) > 3 else "0,2,4,6").split(",")]
ORDER = ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"]
CEN = np.array([0.6662, 0.02229, 0.11654, 0.95986, 3.0446, 0.05933, 2.8268])
SIG = np.array([0.0054, 0.00015, 0.0013, 0.0042, 0.014, 0.0073, 0.1])
FIXED = {"YHe": 0.2454, "TCMB0": 2.34865418e-4, "N_nu_massive": 1, "T_nu_massive": 0.71611,
         "m_nu_massive": 0.06, "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
         "exp_reion": 1.5}
SPECS = dict(output_Cl=True, l_max=2508, lensing=True, output_Pk=False, l_max_g=12, l_max_pol_g=10,
             l_max_ur=17, l_max_ncdm=17, rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
             rtol_small_k_PE=1e-5, max_steps_PE=16384, step_ts_PE=os.environ.get("STEP_TS", "1") == "1")
model = Model(**SPECS)
ELLS = jnp.asarray(model.SS.ells, dtype=jnp.float64)
def loss_one(TT, TE, EE):
    l = ELLS; m = l >= 2; w = jnp.where(m, 2. * l + 1., 0.)
    sTT, sEE = jnp.where(m, TT, 1.), jnp.where(m, EE, 1.)
    return jnp.sum(w * (jnp.log(sTT) + jnp.log(sEE) + jnp.where(m, TE, 0.) / jnp.sqrt(sTT * sEE)))
def raw(th):
    p = dict(FIXED)
    for name, v in zip(ORDER, th):
        if name == "ln10As": p["A_s"] = float(np.exp(v) / 1e10)
        else: p[name] = float(v)
    return p
thetas = [CEN]
for j in DIRS:
    for sgn in (+1, -1):
        th = CEN.copy(); th[j] += sgn * H * SIG[j]; thetas.append(th)
out = model.call_batched([raw(th) for th in thetas], shard=False)
L = np.asarray(jax.vmap(loss_one)(out.ClTT, out.ClTE, out.ClEE))
g_ad = np.load(npz)["grad"][0]; loss_ad = float(np.load(npz)["loss"][0])
print(f"loss centre: value path {L[0]:.6f}  AD run {loss_ad:.6f}  diff {L[0]-loss_ad:.2e}", flush=True)
res = {}
for n, j in enumerate(DIRS):
    fd = (L[1 + 2 * n] - L[2 + 2 * n]) / (2 * H)        # d loss / d (SIG[j] e_j)
    res[ORDER[j]] = {"AD": float(g_ad[j]), "FD": float(fd), "rel": float((g_ad[j] - fd) / abs(fd))}
    print(f"VS_FD {ORDER[j]:10s} AD {g_ad[j]:14.4f}  FD {fd:14.4f}  rel diff {res[ORDER[j]]['rel']:.2e}", flush=True)
print("RESULT " + json.dumps(res), flush=True)

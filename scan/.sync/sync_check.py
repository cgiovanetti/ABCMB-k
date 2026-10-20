"""Upstream sync check: old ABCMB-k main (scan/.sync/tree_main) vs upstream-sync (repo root).
Run with PYTHONPATH=<tree> so `import abcmb` / `import scan` resolve to that tree.

  python sync_check.py acc  TAG   accuracy_test cosmology (massless nu, l_max 2500) -> out/acc_TAG.npz
  python sync_check.py prof TAG   profile specs (l_max 2508, rtol 1e-5, grid frozen at CEN):
                                  k-axis sizes, warm value time B=64, warm gradient time B=44 x 6 dirs,
                                  omega_cdm scan (9 pts, +-0.5 grid_sig) values + AD -> out/prof_TAG.npz
  python sync_check.py class      CLASS at the acc cosmology, delta_l_max 1250 -> out/class_acc.npz
  python sync_check.py report     compare
"""
import os, sys, time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
mode = sys.argv[1]

ACC = {'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225, 'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044,
       'YHe': 0.245, 'TCMB0': 2.34865418e-4, 'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
       "tau_reion": 0.0544, "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5, "exp_reion": 1.5}
ORDER = ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"]
CEN = np.array([0.6662, 0.02229, 0.11654, 0.95986, 3.0446, 0.05933, 2.8268])
GSIG = np.array([0.0145, 0.00021, 0.0032, 0.0072, 0.0154, 0.0062, 0.225])
FIXED = {"YHe": 0.2454, "TCMB0": 2.34865418e-4, "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
         "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5, "exp_reion": 1.5}


def build(theta, xp=np):
    p = dict(FIXED)
    for i, n in enumerate(ORDER):
        p["A_s" if n == "ln10As" else n] = (xp.exp(theta[i]) / 1e10) if n == "ln10As" else theta[i]
    return p


if mode == "class":
    from classy import Class
    c = Class()
    c.set({"output": "mPk, tCl, pCl, lCl", "l_max_scalars": 2500, "P_k_max_1/Mpc": 0.5, "lensing": "yes",
           "accurate_lensing": 1, "H0": 67.62, "omega_b": 0.0225, "omega_cdm": 0.1193, "A_s": 2.12424e-9,
           "n_s": 0.9709, "N_ur": 3.044, "YHe": 0.245, "N_ncdm": 0, "reio_parametrization": "reio_camb",
           "tau_reio": 0.0544, "reionization_width": 0.5, "helium_fullreio_redshift": 3.5,
           "helium_fullreio_width": 0.5, "reionization_exponent": 1.5, "l_max_g": 12, "l_max_pol_g": 10,
           "l_max_ur": 17, "l_max_ncdm": 17, "delta_l_max": 1250})
    c.compute()
    cl = c.lensed_cl(2500)
    k = np.load(f"{OUT}/acc_new.npz")["k"]
    np.savez(f"{OUT}/class_acc.npz", tt=cl["tt"][2:], ee=cl["ee"][2:], te=cl["te"][2:], bb=cl["bb"][2:],
             pp=cl["pp"][2:], pk=np.vectorize(c.pk)(k, 0.))
    sys.exit()

if mode == "report":
    c = np.load(f"{OUT}/class_acc.npz")
    print("(1) accuracy_test cosmology vs CLASS (delta_l_max 1250): max rel err per band")
    bands = ((2, 30), (30, 500), (500, 1500), (1500, 2000), (2000, 2501))
    for tag in ("old", "new"):
        a = np.load(f"{OUT}/acc_{tag}.npz")
        for f in ("tt", "ee"):
            e = np.abs(a[f] / c[f] - 1)
            print(f"  {tag} {f.upper()}: " + "  ".join(f"l{lo}-{hi - 1} {e[lo - 2:hi - 2].max():.2e}" for lo, hi in bands))
        if "bb" in a.files:
            for f in ("bb", "pp"):
                e = np.abs(a[f] / c[f] - 1)
                print(f"  {tag} {f.upper()}: " + "  ".join(f"l{lo}-{hi - 1} {e[lo - 2:hi - 2].max():.2e}" for lo, hi in bands))
        pk = f"{np.max(np.abs(a['pk'] / c['pk'] - 1)):.2e}" if a['pk'].shape == c['pk'].shape else "(other k grid)"
        print(f"  {tag} Pk: {pk}   single-call warm {a['t_warm']:.1f} s")
    print("(2) profile specs, old -> new")
    o, n = np.load(f"{OUT}/prof_old.npz"), np.load(f"{OUT}/prof_new.npz")
    for key in ("n_k_pert", "n_k_transfer", "n_lensing_ells"):
        print(f"  {key}: {int(o[key])} -> {int(n[key])}")
    print(f"  value  B=64 warm: {o['t_val']:.1f} -> {n['t_val']:.1f} s;  gradient B=44 x 6 warm: {o['t_grad']:.1f} -> {n['t_grad']:.1f} s")
    ell = n["ell"]
    for f in ("tt", "te", "ee"):
        d = np.abs(n[f"val_{f}"] - o[f"val_{f}"])
        den = np.abs(o[f"val_{f}"]).max(1, keepdims=True) if f == "te" else np.abs(o[f"val_{f}"])
        r = (d / den).max(0)
        print(f"  Cl change {f.upper()} (64 cosmologies, TE / max|TE|): " +
              "  ".join(f"l{lo}-{hi - 1} {r[(ell >= lo) & (ell < hi)].max():.1e}" for lo, hi in bands))
    print("(3) omega_cdm scan +-0.5 grid_sig: drift = max|v - v0 - int AD| / Cl  (TE / max|TE|)")
    for tag, s in (("old", o), ("new", n)):
        fr = s["scan_x"]
        out = []
        for f in ("tt", "te", "ee"):
            v, ad = s[f"scan_{f}"], s[f"scan_ad_{f}"]
            intad = np.concatenate([np.zeros((1, v.shape[1])), np.cumsum(0.5 * (ad[1:] + ad[:-1]) * np.diff(fr)[:, None], 0)])
            dr = np.abs((v - v[0]) - intad).max(0)
            den = np.abs(v).max() if f == "te" else np.abs(v).max(0)
            r = dr / den
            out.append(f"{f.upper()} " + " ".join(f"{r[(ell >= lo) & (ell < hi)].max():.1e}" for lo, hi in bands))
        print(f"  {tag}: " + " | ".join(out))
    sys.exit()

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
import abcmb
print("abcmb from", abcmb.__file__, flush=True)

if mode == "acc":
    model = Model(user_species=None, output_Cl=True, l_max=2500, lensing=True, output_Pk=True, output_k_max=0.5,
                  l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)
    out = model(ACC); jax.block_until_ready(out.ClTT)
    t0 = time.time(); out = model(ACC); jax.block_until_ready(out.ClTT); tw = time.time() - t0
    extra = {} if getattr(out, "ClBB", None) is None else {"bb": np.asarray(out.ClBB), "pp": np.asarray(out.Clpp)}
    np.savez(f"{OUT}/acc_{sys.argv[2]}.npz", tt=np.asarray(out.ClTT), te=np.asarray(out.ClTE), ee=np.asarray(out.ClEE),
             pk=np.asarray(out.Pk), k=np.asarray(out.k), t_warm=tw, **extra)
    print(f"acc done, warm {tw:.1f} s", flush=True)
    sys.exit()

# ---- profile specs ----
from scan.batched_grad import staged_cl_and_grad, _to_float
model = Model(user_species=None, output_Cl=True, l_max=2508, lensing=True, output_Pk=False, l_max_g=12,
              l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17, rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
              rtol_small_k_PE=1e-5, max_steps_PE=16384)
model = model.freeze_lna_grid(build(CEN))
SS = model.SS
info = dict(n_k_pert=len(model.PE.k_axis_perturbations),
            n_k_transfer=len(SS.k_axis_transfer), n_lensing_ells=len(SS.lensing_ells))
print(info, flush=True)

rng = np.random.default_rng(0)
thetas = CEN + GSIG * rng.uniform(-2.5, 2.5, size=(64, 7))
batch = [build(t) for t in thetas]
out = model.call_batched(batch, shard=True); jax.block_until_ready(out.ClTT)
t0 = time.time(); out = model.call_batched(batch, shard=True); jax.block_until_ready(out.ClTT); t_val = time.time() - t0
print(f"value B=64 warm {t_val:.1f} s", flush=True)
vals = {f"val_{f}": np.asarray(getattr(out, "Cl" + f.upper())) for f in ("tt", "te", "ee")}


def derived(theta):
    return _to_float(model.add_derived_parameters(build(theta, jnp)))


def grad_call(ths, dirs):
    full_ps = [derived(jnp.asarray(t)) for t in ths]
    dots = []
    for j in dirs:
        per = []
        for t in ths:
            tan = jnp.zeros(7).at[j].set(GSIG[j])
            _, fd = jax.jvp(derived, (jnp.asarray(t),), (tan,))
            per.append(fd)
        dots.append(jax.tree.map(lambda *xs: jnp.stack(xs), *per))
    return staged_cl_and_grad(model, full_ps, dots, k_chunk_size=100, shard=True, spec_b_local=4)


ths = thetas[:44]
dirs = [0, 1, 2, 3, 4, 6]                         # the 6 free directions of a tau_reion row
cls, g = grad_call(ths, dirs); jax.block_until_ready(cls[0])
t0 = time.time(); cls, g = grad_call(ths, dirs); jax.block_until_ready(cls[0]); t_grad = time.time() - t0
print(f"gradient B=44 x 6 warm {t_grad:.1f} s", flush=True)

fr = np.linspace(-0.5, 0.5, 9)
sth = np.tile(CEN, (9, 1)); sth[:, 2] = CEN[2] + fr * GSIG[2]
scls, sg = grad_call(sth, [2])
scan = {f"scan_{f}": np.asarray(scls[i]) for i, f in enumerate(("tt", "te", "ee"))}
scan.update({f"scan_ad_{f}": np.asarray(sg[0][i]) for i, f in enumerate(("tt", "te", "ee"))})
np.savez(f"{OUT}/prof_{sys.argv[2]}.npz", ell=np.asarray(SS.ells), t_val=t_val, t_grad=t_grad, scan_x=fr,
         **info, **vals, **scan)
print("prof done", flush=True)

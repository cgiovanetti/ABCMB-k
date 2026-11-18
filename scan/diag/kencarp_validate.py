"""Validation of PE_solver="kencarp_fixed" against "kvaerno5" before making it the default (TODO.md item 31): values
(call_batched) and forward-mode gradients (staged_cl_and_grad) on setups the first benchmark did not cover. One
process per (setup, solver); compare with kencarp_validate_compare.py.

    python scan/diag/kencarp_validate.py --setup massless --solver kencarp_fixed --out <npz> [--reps 2]

setups: massless  (HMC: no massive neutrino, YHe from the BBN table, halofit)
        edges     (_v3 model: LCDM at +-4 sigma, N_nu_massless 0.6-2.7 i.e. Neff ~1.6-3.7, YHe 0.22-0.27, as LINX
                   rows reach)
        mnu       (_v3 model with m_nu 0.02 and 0.3 eV)
        act       (the profile driver's ACT DR6 engine: l_max 6325, delta_l_max 5000, l_max_g/pol 40/60, halofit,
                   computed j_l, streamed lensing)
        memory    (_v3 model at the LINX profile block: B=32 on 4 GPUs x 20 AD directions; peak memory per GPU)
"""
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from abcmb.main import Model
from abcmb import species
from scan.batched_grad import staged_cl_and_grad, _to_float

jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser()
ap.add_argument("--setup", required=True)
ap.add_argument("--solver", default="kvaerno5")
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--what", default="values,grad")
ap.add_argument("--out", required=True)
ap.add_argument("--spec", action="append", default=[], help="extra Model spec key=value (repeatable)")
a = ap.parse_args()

REION = {"Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5, "exp_reion": 1.5, "TCMB0": 2.34865418e-4}
MASSIVE = {"N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06}
PROF = dict(output_Cl=True, l_max=2508, lensing=True, output_Pk=False, l_max_g=12, l_max_pol_g=10,
            rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7, rtol_small_k_PE=1e-5, max_steps_PE=16384, lna_grid_mode="recomb")
CEN = np.array([0.022383, 0.12011, 0.6732, 3.0448, 0.96605, 0.0543])     # omega_b omega_cdm h ln10As n_s tau
SIG = np.array([0.00015, 0.0012, 0.0054, 0.014, 0.0042, 0.0073])
rng = np.random.default_rng(7)
B, ndir, spec_b = 16, 6, 4
if a.setup == "massless":
    kw = dict(PROF, bbn_type="table", nonlinear="halofit")
    fixed = dict(REION, N_nu_massive=0, Neff=3.044)
    th = np.array([0.022383, 0.12011, 0.6789, 3.0448, 0.96605, 0.0543]) + SIG * rng.normal(size=(B, 6))
    extra = {}
elif a.setup == "edges":
    kw = dict(PROF, user_species=(species.MassiveNeutrino,))
    fixed = dict(REION, **MASSIVE)
    th = CEN + 4 * SIG * rng.choice([-1, 1], size=(B, 6))
    extra = {"N_nu_massless": rng.uniform(0.6, 2.7, B), "YHe": rng.uniform(0.22, 0.27, B)}
elif a.setup == "mnu":
    kw = dict(PROF, user_species=(species.MassiveNeutrino,))
    fixed = dict(REION, N_nu_massive=1, T_nu_massive=0.71611, YHe=0.2454)
    th = CEN + SIG * rng.normal(size=(B, 6))
    extra = {"m_nu_massive": np.repeat([0.02, 0.3], B // 2)}
elif a.setup == "act":
    B, spec_b = 8, 1
    kw = dict(output_Cl=True, l_max=6325, lensing=True, output_Pk=False, rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
              rtol_small_k_PE=1e-5, max_steps_PE=16384, lna_grid_mode="recomb", bbn_type="table",
              user_species=(species.MassiveNeutrino,), l_max_g=40, l_max_pol_g=60, delta_l_max=5000,
              nonlinear="halofit", use_bessel_tables=False, lensing_streamed=True)
    fixed = dict(REION, Neff=3.044, **MASSIVE)
    th = np.array([0.022581, 0.12321, 0.6624, 3.04991, 0.96732, 0.05526]) + SIG * rng.normal(size=(B, 6))
    extra = {}
elif a.setup == "memory":
    B, ndir, spec_b = 32, 20, 4
    kw = dict(PROF, user_species=(species.MassiveNeutrino,))
    fixed = dict(REION, YHe=0.2454, **MASSIVE)
    th = CEN + SIG * rng.normal(size=(B, 6))
    extra = {}
else:
    raise SystemExit(f"unknown setup {a.setup}")
import ast
kw.update(PE_solver=a.solver, **{k: ast.literal_eval(v) for k, v in (s.split("=") for s in a.spec)})
model = Model(**kw)
if a.solver == "kvaerno5_fixed":            # structured inverse: the Jacobian's sparsity, found once
    model = model.with_pe_structure(dict(fixed, omega_b=th[0, 0], omega_cdm=th[0, 1], h=th[0, 2],
                                         A_s=np.exp(th[0, 3]) / 1e10, n_s=th[0, 4], tau_reion=th[0, 5],
                                         **{k: float(v[0]) for k, v in extra.items()}))
    print(f"[validate] pe_structure: {len(model.specs['pe_structure'][0])} hubs, "
          f"{len(model.specs['pe_structure'][1])} blocks", flush=True)
ndev = len(jax.devices())
DIRS = np.diag(SIG) if ndir == 6 else (rng.normal(size=(ndir, 6)) * SIG)    # tangent directions in theta space
xkeys = sorted(extra)


def derived(t, x):
    p = dict(fixed)
    p["omega_b"], p["omega_cdm"], p["h"] = t[0], t[1], t[2]
    p["A_s"] = jnp.exp(t[3]) / 1e10
    p["n_s"], p["tau_reion"] = t[4], t[5]
    p.update({k: x[i] for i, k in enumerate(xkeys)})
    return _to_float(model.add_derived_parameters(p))


derived_and_dots = jax.jit(jax.vmap(lambda t, x: (derived(t, x), jax.vmap(
    lambda v: jax.jvp(lambda tt: derived(tt, x), (t,), (v,))[1])(jnp.asarray(DIRS)))))
X = np.stack([extra[k] for k in xkeys], 1) if xkeys else np.zeros((B, 0))


def peak_gb():
    try:
        return max(d.memory_stats().get("peak_bytes_in_use", 0) for d in jax.devices()) / 2 ** 30
    except Exception:
        return float("nan")


out = dict(setup=a.setup, solver=a.solver, B=B, ndir=ndir, ndev=ndev, th=th, X=X)
for what in a.what.split(","):
    times = []
    for rep in range(a.reps):
        t0 = time.time()
        if what == "values":
            plist = []
            for b in range(B):
                p = dict(fixed, omega_b=th[b, 0], omega_cdm=th[b, 1], h=th[b, 2], A_s=np.exp(th[b, 3]) / 1e10,
                         n_s=th[b, 4], tau_reion=th[b, 5])
                p.update({k: float(extra[k][b]) for k in xkeys})
                plist.append(p)
            o = model.call_batched(plist, shard=ndev > 1)
            jax.block_until_ready(o.ClTT)
            out["values_cls"] = np.stack([np.asarray(c) for c in (o.ClTT, o.ClTE, o.ClEE, o.Clpp)])
        else:
            P_B, T_B = derived_and_dots(jnp.asarray(th), jnp.asarray(X))
            full_ps = [jax.tree.map(lambda v: v[b], P_B) for b in range(B)]
            params_dots = [jax.tree.map(lambda v: v[:, j], T_B) for j in range(ndir)]
            cls3, dots = staged_cl_and_grad(model, full_ps, params_dots, k_chunk_size=100, shard=ndev > 1,
                                            spec_b_local=spec_b)
            jax.block_until_ready((cls3, dots))
            out["grad_cls"] = np.stack([np.asarray(c) for c in cls3])
            out["grad_dcl"] = np.stack([np.stack([np.asarray(d[s]) for d in dots], 1) for s in range(3)])
        times.append(time.time() - t0)
        print(f"[validate] {a.setup} {a.solver} {what} rep {rep}: {times[-1]:.1f} s, peak {peak_gb():.1f} GiB/GPU",
              flush=True)
    out[f"{what}_t"] = np.array(times)
out["peak_gb"] = peak_gb()
np.savez(a.out, **out)
print(f"[validate] wrote {a.out}", flush=True)

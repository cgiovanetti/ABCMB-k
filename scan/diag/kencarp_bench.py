"""Kvaerno5 vs FixedKenCarp4 (abcmb/fixed_kencarp4.py) in the batched engine at the profile production
settings (_v3 model: one 0.06 eV neutrino, user YHe; l_max 2508 lensed, recomb lna grid, rtol 1e-5): values
(Model.call_batched) and forward-mode gradients along the 6 LCDM directions (scan/batched_grad.staged_cl_and_grad),
B cosmologies sharded over the node's GPUs. Each call runs twice on different cosmologies of the same shape: the
first time includes compilation, the second is the warm time. Saves the second call's Cls and tangents.

    python scan/diag/kencarp_bench.py --solver kencarp_fixed --n_iter 10 --B 32 --out <npz> [--what values,grad]
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
ap.add_argument("--solver", default="kvaerno5")
ap.add_argument("--n_iter", type=int, default=10)
ap.add_argument("--B", type=int, default=32)
ap.add_argument("--what", default="values,grad")
ap.add_argument("--out", required=True)
a = ap.parse_args()

FIXED = {"YHe": 0.2454, "TCMB0": 2.34865418e-4, "N_nu_massive": 1, "T_nu_massive": 0.71611, "m_nu_massive": 0.06,
         "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5, "exp_reion": 1.5}
CEN = np.array([0.022383, 0.12011, 0.6732, 3.0448, 0.96605, 0.0543])     # omega_b omega_cdm h ln10As n_s tau
SIG = np.array([0.00015, 0.0012, 0.0054, 0.014, 0.0042, 0.0073])
model = Model(user_species=(species.MassiveNeutrino,), output_Cl=True, l_max=2508, lensing=True, output_Pk=False,
              l_max_ur=17, l_max_ncdm=17, l_max_g=12, l_max_pol_g=10, rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
              rtol_small_k_PE=1e-5, max_steps_PE=16384, lna_grid_mode="recomb",
              PE_solver=a.solver, kencarp_n_iter=a.n_iter)
ndev = len(jax.devices())


def derived(th):
    p = dict(FIXED)
    p["omega_b"], p["omega_cdm"], p["h"] = th[0], th[1], th[2]
    p["A_s"] = jnp.exp(th[3]) / 1e10
    p["n_s"], p["tau_reion"] = th[4], th[5]
    return _to_float(model.add_derived_parameters(p))


derived_and_dots = jax.jit(jax.vmap(lambda th: (derived(th), jax.vmap(
    lambda v: jax.jvp(derived, (th,), (v,))[1])(jnp.diag(jnp.asarray(SIG))))))


def thetas(seed):
    return CEN + SIG * np.random.default_rng(seed).normal(size=(a.B, 6))


out = dict(solver=a.solver, n_iter=a.n_iter, B=a.B, ndev=ndev)
for what in a.what.split(","):
    times = []
    for seed in (1, 2):
        th = thetas(seed)
        t0 = time.time()
        if what == "values":
            plist = [{**FIXED, "omega_b": t[0], "omega_cdm": t[1], "h": t[2], "A_s": np.exp(t[3]) / 1e10,
                      "n_s": t[4], "tau_reion": t[5]} for t in th]
            o = model.call_batched(plist, shard=ndev > 1)
            jax.block_until_ready(o.ClTT)
            cls = [np.asarray(c) for c in (o.ClTT, o.ClTE, o.ClEE, o.Clpp)]
        else:
            P_B, T_B = derived_and_dots(jnp.asarray(th))
            full_ps = [jax.tree.map(lambda x: x[b], P_B) for b in range(a.B)]
            params_dots = [jax.tree.map(lambda x: x[:, j], T_B) for j in range(6)]
            cls3, dots = staged_cl_and_grad(model, full_ps, params_dots, k_chunk_size=100, shard=ndev > 1,
                                            spec_b_local=4)
            jax.block_until_ready((cls3, dots))
            cls = [np.asarray(c) for c in cls3]
            out[f"{what}_dcl"] = np.stack([np.stack([np.asarray(d[s]) for d in dots], 1) for s in range(3)])
        times.append(time.time() - t0)
        print(f"[bench] {a.solver} N={a.n_iter} {what} seed {seed}: {times[-1]:.1f} s", flush=True)
    out[f"{what}_cls"] = np.stack(cls)
    out[f"{what}_t_cold"], out[f"{what}_t_warm"] = times
np.savez(a.out, **out)
print(f"[bench] wrote {a.out}", flush=True)

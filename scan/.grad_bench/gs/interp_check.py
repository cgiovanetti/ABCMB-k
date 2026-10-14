"""Which side of the GRADSOLVE-vs-production modes difference is right? Production saves the
Kvaerno5 state at the lna grid through diffrax's interpolation inside each accepted step. Here
the same diffrax solve is forced to STEP ONTO the save grid (PIDController(step_ts=lna)) at the
production tolerance and at rtol 1e-7 (reference), and every variant is compared to that
reference on the modes and the Cls: production chunked path, step_ts 1e-5, and the saved
GRADSOLVE re-step / dense modes (gs_bench*.npz)."""
import os, sys, time, json
ROOT = "/pscratch/sd/c/carag/ABCMB-k-grad"
sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import diffrax
import abcmb
assert abcmb.__file__.startswith(ROOT)
from abcmb.main import Model
HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"]
CEN = np.array([0.6662, 0.02229, 0.11654, 0.95986, 3.0446, 0.05933, 2.8268])
FIXED = {"YHe": 0.2454, "TCMB0": 2.34865418e-4, "N_nu_massive": 1, "T_nu_massive": 0.71611,
         "m_nu_massive": 0.06, "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
         "exp_reion": 1.5}
SPECS = dict(output_Cl=True, l_max=2508, lensing=True, output_Pk=False, l_max_g=12, l_max_pol_g=10,
             l_max_ur=17, l_max_ncdm=17, rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
             rtol_small_k_PE=1e-5, max_steps_PE=16384)
model = Model(**SPECS); PE = model.PE; SS = model.SS
def raw(th):
    p = dict(FIXED)
    for name, v in zip(ORDER, th):
        if name == "ln10As": p["A_s"] = float(np.exp(v) / 1e10)
        else: p[name] = float(v)
    return p
full_ps = [model.add_derived_parameters(raw(CEN))]
params_batch, BG_batch = model._build_bgs_batched(full_ps)
BG = jax.tree.map(lambda x: x[0] if eqx.is_array(x) and x.ndim >= 1 else x, BG_batch)
params = {k: v[0] for k, v in params_batch.items()}
k_axis = PE.k_axis_perturbations
modes_prod, lna_b = PE._compute_modes_batched((BG_batch, params_batch), k_chunk_size=100)
modes_prod = np.asarray(modes_prod[0]); lna = lna_b[0]
print(f"production modes done; Nlna={len(lna)}", flush=True)

def solve_one_k(k, lna, args, rtol_scale, step_ts, max_steps):
    """evolution_one_k with the step controller forced onto the save grid (step_ts) and
    tolerances scaled by rtol_scale."""
    lna_start = jnp.minimum(PE.get_starting_time(k, args), -10.)
    y_ini = PE.initial_conditions_one_k(k, lna_start, args)
    rtol = jnp.where(k > PE.specs["k_split_PE"], PE.specs["rtol_large_k_PE"], PE.specs["rtol_small_k_PE"]) * rtol_scale
    atol = jnp.where(k > PE.specs["k_split_PE"], PE.specs["atol_large_k_PE"], PE.specs["atol_small_k_PE"]) * rtol_scale
    ctrl = diffrax.PIDController(pcoeff=PE.specs["pcoeff_PE"], icoeff=PE.specs["icoeff_PE"],
                                 dcoeff=PE.specs["dcoeff_PE"], rtol=rtol, atol=atol,
                                 step_ts=(lna if step_ts else None))
    sol = diffrax.diffeqsolve(diffrax.ODETerm(PE.get_derivatives), diffrax.Kvaerno5(),
                              t0=lna_start, t1=0.0, dt0=1e-2, y0=y_ini, stepsize_controller=ctrl,
                              max_steps=max_steps, saveat=diffrax.SaveAt(ts=lna), args=(k, *args),
                              adjoint=diffrax.ForwardMode())
    return sol.ys, sol.stats["num_steps"]
@eqx.filter_jit
def chunk(kc, lna, args, rtol_scale, step_ts, max_steps):
    return jax.vmap(lambda k: solve_one_k(k, lna, args, rtol_scale, step_ts, max_steps))(kc)
def modes_variant(rtol_scale, step_ts, max_steps=65536, K=100):
    ys, ns = [], []
    for i in range(0, len(k_axis), K):
        y, n = chunk(k_axis[i:i + K], lna, (BG, params), rtol_scale, step_ts, max_steps)
        ys.append(np.asarray(y)); ns.append(np.asarray(n))
    y = np.concatenate(ys, 0); n = np.concatenate(ns, 0)          # (N_k, Nlna, Ny)
    return np.transpose(y, (2, 1, 0)), n
t0 = time.time(); m_step5, n_step5 = modes_variant(1.0, True); print(f"step_ts rtol1e-5: {time.time()-t0:.0f} s, steps median {int(np.median(n_step5))} max {n_step5.max()}", flush=True)
t0 = time.time(); m_free5, n_free5 = modes_variant(1.0, False); print(f"free rtol1e-5 (unchunked-equivalent): {time.time()-t0:.0f} s, steps median {int(np.median(n_free5))} max {n_free5.max()}", flush=True)
t0 = time.time(); m_ref, n_ref = modes_variant(1e-2, True); print(f"step_ts rtol1e-7 REFERENCE: {time.time()-t0:.0f} s, steps median {int(np.median(n_ref))} max {n_ref.max()}", flush=True)
def cls_of(modes):
    PT = PE.make_output_table_batched(lna_b, jnp.asarray(modes)[None], (BG_batch, params_batch))
    return [np.asarray(c)[0] for c in SS.get_Cl_batched(PT, BG_batch, params_batch)]
variants = {"production_interp_1e-5": modes_prod, "diffrax_step_ts_1e-5": m_step5, "diffrax_free_1e-5": m_free5}
for tag in ("", "_dense"):
    f = os.path.join(HERE, f"gs_bench{tag}.npz")
    if os.path.exists(f):
        variants[f"gradsolve{tag or '_restep'}"] = np.load(f)["modes_gs"]
cl_ref = cls_of(m_ref)
scale = np.max(np.abs(m_ref), axis=(1, 2), keepdims=True) + 1e-300
out = {}
for name, m in variants.items():
    rel = np.abs(m - m_ref) / scale
    cl = cls_of(m)
    rec = {"modes_max_peaknorm": float(rel.max()),
           "modes_max_peaknorm_photon_vars": float(rel[6:30].max())}
    for i, nm in enumerate(("TT", "TE", "EE")):
        a, b = cl_ref[i][2:], cl[i][2:]
        rec[f"Cl{nm}_peaknorm"] = float(np.max(np.abs(b - a)) / np.max(np.abs(a)))
        rec[f"Cl{nm}_rel_l>30"] = float(np.max(np.abs(b - a)[28:] / np.abs(a)[28:]))
    out[name] = rec
    print(f"VS_REF {name}: " + json.dumps(rec), flush=True)
np.savez(os.path.join(HERE, "interp_check.npz"), m_ref=m_ref, m_step5=m_step5, m_free5=m_free5, lna=np.asarray(lna), k=np.asarray(k_axis))
print("DONE", flush=True)

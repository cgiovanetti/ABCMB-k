"""GRADSOLVE vs ABCMB-k's diffrax Kvaerno5 on the real perturbation system, ONE cosmology
(the profile centre), all N_k modes. Production specs (l_max 2508, lensing, 1 massive nu,
PE rtol 1e-5). Compares the saved modes on ABCMB's own lna grid and the Cls through the
unchanged table + spectrum stages, and times both solves warm.
gradsolve path: Rodas5P record (device recorder, batched while_loop) + replay with saveat
(bracket-and-re-step, or the order-4 dense extension with DENSE=1). Every mode starts at the
EARLIEST per-k start time (shared t0: the tight-coupling and super-horizon thresholds hold
earlier too and the adiabatic series is more accurate there); the ensemble is split at
k_split_PE so each half keeps its own (rtol, atol).
Usage: [DENSE=1] [RTOL=1e-5] python gs_bench.py"""
import os, sys, time, json
ROOT = "/pscratch/sd/c/carag/ABCMB-k-grad"
GS = "/pscratch/sd/c/carag/gradsolve"
sys.path.insert(0, ROOT); sys.path.insert(1, GS)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import abcmb
assert abcmb.__file__.startswith(ROOT), abcmb.__file__
from abcmb.main import Model
import gradsolve
assert gradsolve.__file__.startswith(GS), gradsolve.__file__
from gradsolve.solvers.rodas5p_replay import record_rodas5p
from gradsolve.solvers.rodas5p_step import rodas5p_advance, rodas5p_stages_and_dense, rodas5p_dense_eval
from gradsolve.solvers.dense import vmap_saveat, vmap_saveat_dense
DENSE = os.environ.get("DENSE", "0") == "1"
ORDER = ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"]
CEN = np.array([0.6662, 0.02229, 0.11654, 0.95986, 3.0446, 0.05933, 2.8268])
FIXED = {"YHe": 0.2454, "TCMB0": 2.34865418e-4, "N_nu_massive": 1, "T_nu_massive": 0.71611,
         "m_nu_massive": 0.06, "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
         "exp_reion": 1.5}
SPECS = dict(output_Cl=True, l_max=2508, lensing=True, output_Pk=False, l_max_g=12, l_max_pol_g=10,
             l_max_ur=17, l_max_ncdm=17, rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
             rtol_small_k_PE=1e-5, max_steps_PE=16384)
model = Model(**SPECS); PE = model.PE; SS = model.SS
DEV = jax.devices()[0]
def peak_gb():
    return DEV.memory_stats()["peak_bytes_in_use"] / 1e9
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
N_k = len(k_axis)
Ny = 1 + sum(int(s.num_equations) for s in PE.species_list)
print(f"N_k={N_k} Ny={Ny} k_split={PE.specs['k_split_PE']} specs rtol/atol small {PE.specs['rtol_small_k_PE']}/{PE.specs['atol_small_k_PE']} "
      f"large {PE.specs['rtol_large_k_PE']}/{PE.specs['atol_large_k_PE']} DENSE={DENSE}", flush=True)

# ---------------- reference: production chunked diffrax path, B=1 ----------------
def ref_modes():
    modes, lna_b = PE._compute_modes_batched((BG_batch, params_batch), k_chunk_size=100)
    return jax.block_until_ready((modes, lna_b))
t0 = time.time(); modes_ref, lna_b = ref_modes(); t_ref_cold = time.time() - t0
t0 = time.time(); modes_ref, lna_b = ref_modes(); t_ref_warm = time.time() - t0
modes_ref = modes_ref[0]                                   # (Ny, Nlna, N_k)
lna_grid = lna_b[0]                                        # (Nlna,)
PT_ref = PE.make_output_table_batched(lna_b, modes_ref[None], (BG_batch, params_batch))
cls_ref = jax.block_until_ready(SS.get_Cl_batched(PT_ref, BG_batch, params_batch))
print(f"[diffrax ref] modes cold {t_ref_cold:.1f} s, warm {t_ref_warm:.1f} s; Nlna={len(lna_grid)} "
      f"lna_grid[0]={float(lna_grid[0]):.3f} peak {peak_gb():.2f} GB", flush=True)

# ---------------- gradsolve: Rodas5P record + replay ----------------
args1 = (BG, params)
lna_start_k = jax.vmap(lambda k: jnp.minimum(PE.get_starting_time(k, args1), -10.))(k_axis)
T0 = float(jnp.min(lna_start_k)); T1 = 0.0
print(f"per-k lna_start range [{float(lna_start_k.min()):.3f}, {float(lna_start_k.max()):.3f}] -> shared t0={T0:.3f}", flush=True)
y0_all = jax.vmap(lambda k: PE.initial_conditions_one_k(k, jnp.float64(T0), args1))(k_axis)   # (N_k, Ny)

class Prob:
    name = "abcmb_perturbations"; dim = Ny; t0 = T0; t1 = T1; is_stiff = True
    def f_jax(self, t, y, p):
        return PE.get_derivatives(t, y, (p[0], BG, params))
prob = Prob()
f = prob.f_jax
p_all = np.asarray(k_axis)[:, None]
split = np.asarray(k_axis) > PE.specs["k_split_PE"]
groups = [("small_k", ~split, PE.specs["rtol_small_k_PE"], PE.specs["atol_small_k_PE"]),
          ("large_k", split, PE.specs["rtol_large_k_PE"], PE.specs["atol_large_k_PE"])]
RTOL_OVERRIDE = os.environ.get("RTOL"); ATOL_OVERRIDE = os.environ.get("ATOL")

def record_all():
    out = {}
    for name, m, rtol, atol in groups:
        rtol = float(RTOL_OVERRIDE) if RTOL_OVERRIDE else rtol
        atol = float(ATOL_OVERRIDE) if ATOL_OVERRIDE else atol
        yf, dts, n_acc, n_rej = record_rodas5p(f, np.asarray(y0_all)[m], p_all[m], T0, T1,
                                               rtol=rtol, atol=atol, dt0=1e-2, max_steps=16384)
        out[name] = (jnp.asarray(dts), n_acc, n_rej, rtol, atol)
    return out
t0 = time.time(); rec = record_all(); t_rec_cold = time.time() - t0
t0 = time.time(); rec = record_all(); t_rec_warm = time.time() - t0
for name, (dts, n_acc, n_rej, rtol, atol) in rec.items():
    print(f"[record {name}] n={len(n_acc)} rtol={rtol} atol={atol} accepted steps: min {n_acc.min()} "
          f"median {int(np.median(n_acc))} max {n_acc.max()}; rejected median {int(np.median(n_rej))} max {n_rej.max()}", flush=True)
print(f"[record] cold {t_rec_cold:.1f} s, warm {t_rec_warm:.1f} s, peak {peak_gb():.2f} GB", flush=True)

ts = jnp.asarray(lna_grid)
@jax.jit
def replay_all(y0_all, p_all_j, dts_small, dts_large):
    outs = []
    for m, dts in ((jnp.asarray(~split), dts_small), (jnp.asarray(split), dts_large)):
        pass
    return None
def replay():
    ys_parts = []
    for name, m, _, _ in groups:
        dts = rec[name][0]
        if DENSE:
            _, ys = vmap_saveat_dense(f, T0, jnp.asarray(y0_all)[m], jnp.asarray(p_all[m]), dts, ts,
                                      rodas5p_stages_and_dense, rodas5p_dense_eval)
        else:
            _, ys = vmap_saveat(f, T0, jnp.asarray(y0_all)[m], jnp.asarray(p_all[m]), dts, ts, rodas5p_advance)
        ys_parts.append(ys)                                 # (n_g, Nlna, Ny)
    ys = jnp.concatenate(ys_parts, axis=0)                  # k order preserved: small k first
    return jax.block_until_ready(ys)
replay_j = jax.jit(replay) if False else replay
t0 = time.time(); ys = replay(); t_rep_cold = time.time() - t0
t0 = time.time(); ys = replay(); t_rep_warm = time.time() - t0
print(f"[replay{' dense' if DENSE else ''}] cold {t_rep_cold:.1f} s, warm {t_rep_warm:.1f} s, peak {peak_gb():.2f} GB", flush=True)
modes_gs = jnp.transpose(ys, (2, 1, 0))                     # (Ny, Nlna, N_k)

# ---------------- compare modes and Cls ----------------
mr = np.asarray(modes_ref); mg = np.asarray(modes_gs)
scale = np.max(np.abs(mr), axis=(1, 2), keepdims=True) + 1e-300
rel = np.abs(mg - mr) / scale
print(f"[modes] max peak-normalized |diff| over all variables: {rel.max():.3e}; per-variable max "
      f"(first 12): {np.round(rel.max(axis=(1, 2))[:12], 7).tolist()}", flush=True)
late = lna_grid > -7.0
print(f"[modes] same, lna > -7 only: {rel[:, np.asarray(late), :].max():.3e}", flush=True)
PT_gs = PE.make_output_table_batched(lna_b, modes_gs[None], (BG_batch, params_batch))
cls_gs = jax.block_until_ready(SS.get_Cl_batched(PT_gs, BG_batch, params_batch))
res = {"t_ref_warm": round(t_ref_warm, 1), "t_record_warm": round(t_rec_warm, 1),
       "t_replay_warm": round(t_rep_warm, 1), "dense": DENSE}
for i, nm in enumerate(("TT", "TE", "EE")):
    a = np.asarray(cls_ref[i])[0][2:]; b = np.asarray(cls_gs[i])[0][2:]
    res[f"Cl{nm}_max_peaknorm_diff"] = float(np.max(np.abs(b - a)) / np.max(np.abs(a)))
    res[f"Cl{nm}_max_rel_diff_l>30"] = float(np.max(np.abs(b - a)[28:] / np.abs(a)[28:]))
print("RESULT " + json.dumps(res), flush=True)
np.savez(os.path.join(os.path.dirname(__file__), f"gs_bench{'_dense' if DENSE else ''}.npz"),
         modes_ref=mr, modes_gs=mg, lna=np.asarray(lna_grid), k=np.asarray(k_axis),
         clTT_ref=np.asarray(cls_ref[0])[0], clTT_gs=np.asarray(cls_gs[0])[0],
         clTE_ref=np.asarray(cls_ref[1])[0], clTE_gs=np.asarray(cls_gs[1])[0],
         clEE_ref=np.asarray(cls_ref[2])[0], clEE_gs=np.asarray(cls_gs[2])[0],
         n_acc_small=rec["small_k"][1], n_acc_large=rec["large_k"][1])
print("DONE", flush=True)

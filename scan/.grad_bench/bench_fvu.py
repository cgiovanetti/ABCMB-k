"""Gradient benchmark for the grad-vmap-dirs worktree (copy of the fork-vs-upstream
bench scan/.v2analysis/fvu/bench_fvu.py in the production checkout, ABCMB-k code path only).
ABCMB-k's staged forward-mode AD (scan/batched_grad.py) with the P=7 directions vmapped
inside each staged jvp. Production specs of the profile (l_max 2508, lensing, 1 massive nu,
PE rtol 1e-5, ...); the same B seeded cosmologies as the fvu bench, so the saved
fvu/out/k_{value,grad}_B<B>.npz of the production checkout are the reference (read only).

Usage: DIR_CHUNK=<n> [SHARD=1] python bench_fvu.py k MODE REPS B1 [B2 ...]
  MODE  value : Cls only (call_batched)
        grad  : loss + its 7-parameter gradient (per scaling sigma); the loss is a smooth
                reference-free Cl functional, sum_l (2l+1)[log TT + log EE + TE/sqrt(TT EE)]
  DIR_CHUNK  directions per vmapped jvp (0 or unset = all 7 at once; 1 = one at a time)
  SHARD=1    shard the B axis over all visible GPUs (driver path); default single device
Prints one RESULT json line per B (cold = first call incl. compile, warm = mean of REPS more),
with the max |diff| of loss / grad / Cls against the production reference npz."""
import os, sys, time, json
CODE, MODE, REPS = sys.argv[1], sys.argv[2], int(sys.argv[3])
BS = [int(b) for b in sys.argv[4:]]
HERE = "/pscratch/sd/c/carag/ABCMB-k-grad/scan/.grad_bench"
ROOT = "/pscratch/sd/c/carag/ABCMB-k-grad"
REF = "/pscratch/sd/c/carag/ABCMB-k/scan/.v2analysis/fvu/out"     # production fvu outputs, read only
DIR_CHUNK = int(os.environ.get("DIR_CHUNK", "0")) or None
SPEC_B = int(os.environ.get("SPEC_B", "0")) or None          # cosmologies/device per spectrum jvp call
SHARD = os.environ.get("SHARD", "0") == "1"
sys.path.insert(0, ROOT)
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import abcmb
assert abcmb.__file__ == os.path.join(ROOT, "abcmb", "__init__.py"), abcmb.__file__
from abcmb.main import Model
sys.path.insert(1, os.path.join(ROOT, "scan"))
import batched_grad
assert batched_grad.__file__.startswith(ROOT), batched_grad.__file__
from batched_grad import staged_cl_and_grad

ORDER = ["h", "omega_b", "omega_cdm", "n_s", "ln10As", "tau_reion", "Neff"]
CEN = np.array([0.6662, 0.02229, 0.11654, 0.95986, 3.0446, 0.05933, 2.8268])
SIG = np.array([0.0054, 0.00015, 0.0013, 0.0042, 0.014, 0.0073, 0.1])
FIXED = {"YHe": 0.2454, "TCMB0": 2.34865418e-4, "N_nu_massive": 1, "T_nu_massive": 0.71611,
         "m_nu_massive": 0.06, "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5,
         "exp_reion": 1.5}
SPECS = dict(output_Cl=True, l_max=2508, lensing=os.environ.get("LENSING", "1") == "1",
             step_ts_PE=os.environ.get("STEP_TS", "1") == "1",
             output_Pk=False, l_max_g=12, l_max_pol_g=10,
             l_max_ur=17, l_max_ncdm=17, rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
             rtol_small_k_PE=1e-5, max_steps_PE=16384)
if CODE != "k":
    sys.exit(f"unknown CODE {CODE}: this copy benchmarks the ABCMB-k path only")
model = Model(**SPECS)
NK = len(model.PE.k_axis_perturbations)
NL = len(model.SS.ells)
DEV = jax.devices()[0]
print(f"[{CODE}/{MODE}] abcmb={abcmb.__file__} devices={jax.devices()} N_k={NK} n_l={NL} "
      f"N_kt={len(model.SS.k_axis_transfer)} n_lensing_ells={len(model.SS.lensing_ells)} "
      f"n_mu={len(model.SS.lensing_mus)} n_ell_tab={len(model.SS.lensing_ells_indices)} "
      f"DIR_CHUNK={DIR_CHUNK} SPEC_B={SPEC_B} SHARD={SHARD} specs={SPECS}", flush=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)


def thetas(B):
    z = np.clip(np.random.default_rng(0).standard_normal((max(B, 1), 7)), -2, 2)
    z[0] = 0.
    return CEN + SIG * z[:B]


def raw(th, host=False):
    p = dict(FIXED)
    for name, v in zip(ORDER, th):
        if name == "ln10As":
            p["A_s"] = float(np.exp(v) / 1e10) if host else jnp.exp(v) / 1e10
        else:
            p[name] = float(v) if host else v
    return p


def to_float(tree):
    def f(v):
        a = jnp.asarray(v)
        return a.astype(jnp.float64) if a.dtype.kind in "iub" else a
    return jax.tree_util.tree_map(f, tree)


def derived(th):
    return to_float(model.add_derived_parameters(raw(th)))


def stack(trees):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


ELLS = jnp.asarray(model.SS.ells, dtype=jnp.float64)


def loss_one(TT, TE, EE):
    l = ELLS
    m = l >= 2
    w = jnp.where(m, 2. * l + 1., 0.)
    sTT, sEE = jnp.where(m, TT, 1.), jnp.where(m, EE, 1.)
    return jnp.sum(w * (jnp.log(sTT) + jnp.log(sEE) + jnp.where(m, TE, 0.) / jnp.sqrt(sTT * sEE)))


loss_b = jax.vmap(loss_one)


def cls_of(out):
    return out.ClTT, out.ClTE, out.ClEE


def peak_gb():
    try:
        return DEV.memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def prep_grad(TH):
    """derived primals (list) + per-direction tangents (per sigma), forward-mode through
    add_derived_parameters only (as the profile driver does)."""
    ps, dots = [], []
    for th in TH:
        th = jnp.asarray(th)
        ps.append(derived(th))
        dots.append([eqx.filter(jax.jvp(derived, (th,), (jnp.zeros(7).at[j].set(SIG[j]),))[1],
                                eqx.is_inexact_array) for j in range(7)])
    return ps, dots


def run_value(B):
    TH = thetas(B)
    out = model.call_batched([raw(th, host=True) for th in TH], shard=SHARD)
    res = (out.ClTT, out.ClTE, out.ClEE)
    return jax.block_until_ready(res)


def run_grad(B):
    TH = thetas(B)
    ps, dots = prep_grad(TH)
    params_dots = [stack([dots[b][j] for b in range(B)]) for j in range(7)]
    cls, cl_grads = staged_cl_and_grad(model, ps, params_dots, k_chunk_size=100, shard=SHARD,
                                       dir_chunk=DIR_CHUNK, spec_b_local=SPEC_B)
    val = loss_b(*cls)
    cols = [jax.jvp(loss_b, cls, tuple(cl_dot))[1] for cl_dot in cl_grads]
    g = jnp.stack(cols, axis=-1)
    return jax.block_until_ready((val, g, cls))


fn = run_value if MODE == "value" else run_grad
for B in BS:
    rec = {"code": CODE, "mode": MODE, "B": B, "N_k": NK}
    try:
        t0 = time.time(); r = fn(B); rec["cold_s"] = round(time.time() - t0, 1)
        ws = []
        for _ in range(REPS):
            t0 = time.time(); r = fn(B); ws.append(time.time() - t0)
        if ws:
            rec["warm_s"] = round(float(np.mean(ws)), 1)
            rec["warm_s_per_cosmo"] = round(float(np.mean(ws)) / B, 2)
        rec["peak_GB"] = round(peak_gb(), 2)
        if MODE == "value":
            np.savez(os.path.join(HERE, "out", f"{CODE}_value_B{B}.npz"), TT=r[0], TE=r[1], EE=r[2], theta=thetas(B), l=np.asarray(ELLS))
            rec["finite"] = bool(all(np.isfinite(np.asarray(x)).all() for x in r))
            fv = os.path.join(REF, f"k_value_B{B}.npz")
            if os.path.exists(fv):
                rv = np.load(fv)
                rec["ref_cls_maxrel"] = {k: float(np.max(np.abs(np.asarray(r[i]) - rv[k])[:, 2:]
                                                        / np.abs(rv[k])[:, 2:].max(1, keepdims=True)))
                                         for i, k in enumerate(("TT", "TE", "EE"))}
        else:
            tag = f"dc{DIR_CHUNK or 7}" + (f"_sb{SPEC_B}" if SPEC_B else "") + ("_sh" if SHARD else "")
            np.savez(os.path.join(HERE, "out", f"{CODE}_grad_B{B}_{tag}.npz"), loss=r[0], grad=r[1],
                     TT=r[2][0], TE=r[2][1], EE=r[2][2], theta=thetas(B), l=np.asarray(ELLS))
            rec["finite"] = bool(np.isfinite(np.asarray(r[1])).all())
            rec["grad_row0"] = [round(float(x), 4) for x in np.asarray(r[1])[0]]
            # --- accuracy vs the production checkout's one-direction-at-a-time run ---
            fg = os.path.join(REF, f"k_grad_B{B}.npz"); fv = os.path.join(REF, f"k_value_B{B}.npz")
            if os.path.exists(fg):
                rg = np.load(fg)
                dl = np.abs(np.asarray(r[0]) - rg["loss"]).max()
                dg = np.abs(np.asarray(r[1]) - rg["grad"]).max()
                rel = dg / np.abs(rg["grad"]).max()
                rec["ref_grad"] = {"max_abs_dloss": float(dl), "max_abs_dgrad": float(dg),
                                   "max_rel_dgrad": float(rel)}
            if os.path.exists(fv):
                rv = np.load(fv)
                rec["ref_cls_maxrel"] = {k: float(np.max(np.abs(np.asarray(r[2][i]) - rv[k])[:, 2:]
                                                        / np.abs(rv[k])[:, 2:].max(1, keepdims=True)))
                                         for i, k in enumerate(("TT", "TE", "EE"))}
    except Exception as ex:
        import traceback; traceback.print_exc()
        rec["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"
        rec["peak_GB"] = round(peak_gb(), 2)
    print("RESULT " + json.dumps(rec), flush=True)
print(f"[{CODE}/{MODE}] DONE", flush=True)

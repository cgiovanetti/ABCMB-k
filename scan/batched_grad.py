"""batched_grad.py — staged forward-mode AD on the per-k batched pipeline.

Implements the design in bench/batched_ad_design.md: instead of `jacfwd`-ing the
whole cross-device pipeline (the ~20 min monolith), we apply `jax.jvp` to EACH
already-`filter_jit`'d BATCHED stage separately, in eager Python, threading the
array tangent between stages -- exactly how `Model.call_batched` orchestrates the
PRIMAL. Because we differentiate the BATCHED stages, the gradient inherits the
params-axis batching + k-chunking (+ sharding, once wired). Risk #1 (the
perturbation stage) is validated in scan/derisk_batched_ad.py (1.55x compile,
5e-4 vs FD); this assembles the full chain and validates the END-TO-END Cl gradient
against the proven single-cosmology `jacfwd`.

This lives in scan/ and only CALLS the existing Model batched-stage methods
(`_pre_recomb_batched`, HyRex `_recmodel_cpu`, `_get_BG_batched`,
`PE.full_evolution_batched`, `SS.get_Cl_batched`) -- NO core-code edits. If it works
+ is fast, it gets promoted to `Model.call_batched_grad` later.

Run via srun on a GPU node, PYTHONPATH=$(pwd). Env: BG_LMAX(128), BG_WRT(omega_cdm,n_s).
"""
import os, time
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
from abcmb.main import Model
from abcmb.main import _recmodel_cpu

LMAX = int(os.environ.get("BG_LMAX", 128))
WRT = os.environ.get("BG_WRT", "omega_cdm,n_s").split(",")

FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}
ORDER = ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10As', 'tau_reion']
A_S = float(np.exp(3.044) / 1e10)


def raw_dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200, n_s=0.9649,
             ln10As=3.044, tau_reion=0.0544):
    return dict(FIXED, h=h, omega_b=omega_b, omega_cdm=omega_cdm, n_s=n_s,
                A_s=float(np.exp(ln10As) / 1e10), tau_reion=tau_reion)


# ---------- per-stage jvp helper (partition static, thread array tangents) ----------
def jvp_stage(fn, primals_full, tangents_diff):
    """fn(*primals_full) with forward-mode. primals_full: tuple of full pytrees
    (eqx.Modules / dicts). tangents_diff: tuple of DIFF pytrees (arrays where the
    primal is an inexact array, None at static), one per primal. Returns
    (primal_out_full, out_tangent_diff) where out_tangent_diff is filtered to the
    inexact-array leaves (ready to thread into the next stage)."""
    diffs, statics = [], []
    for p in primals_full:
        d, s = eqx.partition(p, eqx.is_inexact_array)
        diffs.append(d); statics.append(s)

    def f_arr(*ds):
        return fn(*[eqx.combine(d, s) for d, s in zip(ds, statics)])

    out, out_dot = jax.jvp(f_arr, tuple(diffs), tuple(tangents_diff))
    return out, eqx.filter(out_dot, eqx.is_inexact_array)


def diff_of(tree):
    return eqx.filter(tree, eqx.is_inexact_array)


def _to_float(tree):
    """cast int/bool leaves to float64 (as run_cosmology_abbr / _build_bgs_batched
    do) so primal and tangent pytrees have matching inexact-array structure."""
    def f(v):
        arr = jnp.asarray(v)
        return arr.astype(jnp.float64) if arr.dtype.kind in 'iub' else arr
    return jax.tree_util.tree_map(f, tree)


# ---------- the staged forward-mode push: params_batch -> (Cl, dCl) ----------
def staged_cl_and_grad(model, full_ps, params_dots):
    """full_ps: list[B] of derived param dicts (primal).
    params_dots: list[P] of derived-param-tangent dicts, each stacked over B
        (i.e., params_dots[j] is the dB-stacked tangent of params_batch in
        direction j). Returns (ClTT,ClTE,ClEE) primal (B,n_l) and a list[P] of
        (dClTT,dClTE,dClEE) tangents (B,n_l)."""
    PE = model.PE; SS = model.SS
    cpu = jax.devices('cpu')[0]
    try:
        gpu = jax.devices('gpu')[0]
    except Exception:
        gpu = None

    # stack primal params over B, cast int/bool->float (match tangent structure)
    params_batch = _to_float(jax.tree.map(lambda *xs: jnp.stack(xs), *full_ps))

    # ---- stage 1: pre-recomb (GPU) ----
    pre_BG = model._pre_recomb_batched(params_batch)

    # ---- stage 2: HyRex (CPU) ----
    ri_cpu = jax.device_put(pre_BG.recomb_inputs, cpu)
    p_cpu = jax.device_put(params_batch, cpu)
    hy = _recmodel_cpu(model.RecModel, True)
    recomb = hy((ri_cpu, p_cpu))
    if gpu is not None:
        recomb = jax.device_put(recomb, gpu)

    # ---- stage 3: get_BG (GPU) ----
    BG = model._get_BG_batched(params_batch, pre_BG, recomb)

    # ---- stage 4: perturbations (GPU, k-chunked) ----
    PT = PE.full_evolution_batched((BG, params_batch))

    # ---- stage 5: spectrum (GPU) ----
    ClTT, ClTE, ClEE = SS.get_Cl_batched(PT, BG, params_batch)

    # ===== tangents, one nuisance direction at a time =====
    grads = []
    for pdot in params_dots:               # pdot: dict stacked over B (direction j)
        # stage 1 jvp
        _, pre_BG_dot = jvp_stage(model._pre_recomb_batched, (params_batch,), (pdot,))
        # stage 2 jvp (HyRex on CPU): inputs (recomb_inputs, params), tangents
        # (pre_BG_dot.recomb_inputs, pdot). Move primal+tangent to CPU; result to GPU.
        ri_dot_cpu = jax.device_put(pre_BG_dot.recomb_inputs, cpu)
        p_dot_cpu = jax.device_put(pdot, cpu)
        _, recomb_dot = jvp_stage(lambda ri, p: hy((ri, p)),
                                  (ri_cpu, p_cpu), (ri_dot_cpu, p_dot_cpu))
        if gpu is not None:
            recomb_dot = jax.device_put(recomb_dot, gpu)
        # stage 3 jvp
        _, BG_dot = jvp_stage(model._get_BG_batched,
                              (params_batch, pre_BG, recomb), (pdot, pre_BG_dot, recomb_dot))
        # stage 4 jvp (perturbations)
        _, PT_dot = jvp_stage(lambda bg, p: PE.full_evolution_batched((bg, p)),
                              (BG, params_batch), (BG_dot, pdot))
        # stage 5 jvp (spectrum)
        _, cl_dot = jvp_stage(SS.get_Cl_batched, (PT, BG, params_batch),
                              (PT_dot, BG_dot, pdot))
        # cl_dot is the filtered tangent of (ClTT,ClTE,ClEE) -> a 3-tuple of arrays
        grads.append(cl_dot)
    return (ClTT, ClTE, ClEE), grads


# ---------- derived-param tangents w.r.t. raw nuisances (eager, per cosmo) ----------
def derived_and_tangents(model, raw_ps, wrt):
    """For each raw param dict, derive params and the tangent of the derived dict
    w.r.t. each wrt key (forward-mode through add_derived_parameters only).
    Returns (full_ps list[B], params_dots list[P] of B-stacked tangent dicts)."""
    full_ps = [model.add_derived_parameters(p) for p in raw_ps]
    P = len(wrt)
    per_cosmo_dots = []   # [B] of [P] tangent dicts
    for p in raw_ps:
        rd = {k: jnp.asarray(float(v)) for k, v in p.items()}
        dots_j = []
        for key in wrt:
            tangent = {k: (jnp.ones(()) if k == key else jnp.zeros(())) for k in rd}
            _, fd = jax.jvp(lambda d: model.add_derived_parameters(d), (rd,), (tangent,))
            dots_j.append(eqx.filter(fd, eqx.is_inexact_array))
        per_cosmo_dots.append(dots_j)
    # restack: params_dots[j] = stack over B of per_cosmo_dots[b][j]
    params_dots = []
    for j in range(P):
        params_dots.append(jax.tree.map(lambda *xs: jnp.stack(xs),
                                        *[per_cosmo_dots[b][j] for b in range(len(raw_ps))]))
    return full_ps, params_dots


# ---------- single-path reference: jacfwd of Cl per cosmology ----------
def single_path_cl_grad(model, raw_p, wrt):
    """ClTT/TE/EE and their gradients w.r.t. `wrt` for ONE cosmology, via the
    PROVEN single-cosmology jacfwd path. Returns dict key->(dTT,dTE,dEE)."""
    base = {k: jnp.asarray(float(v)) for k, v in raw_p.items()}

    def cl_of(scalar, key):
        d = dict(base); d[key] = scalar
        out = model.run_cosmology_abbr(model.add_derived_parameters(d))
        return out.ClTT, out.ClTE, out.ClEE
    res = {}
    for key in wrt:
        prim = jax.jacfwd(lambda s: cl_of(s, key))(base[key])
        res[key] = prim                      # tuple (dTT,dTE,dEE), each (n_l,)
    return res


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  wrt={WRT}", flush=True)
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=False,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    raw_ps = [raw_dict(), raw_dict(h=0.68, omega_cdm=0.119, n_s=0.96)]

    full_ps, params_dots = derived_and_tangents(model, raw_ps, WRT)
    print(f"derived {len(full_ps)} cosmologies; P={len(params_dots)} directions", flush=True)

    t0 = time.perf_counter()
    (ClTT, ClTE, ClEE), grads = staged_cl_and_grad(model, full_ps, params_dots)
    jax.block_until_ready(grads)
    print(f"[staged] value+grad compile+run {time.perf_counter()-t0:.1f}s "
          f"(B={len(raw_ps)}, P={len(WRT)})", flush=True)

    # ---- validate batched grad vs single-path jacfwd, cosmology 0 ----
    print("\n[validate] batched staged grad vs single-path jacfwd (cosmo 0):", flush=True)
    ref = single_path_cl_grad(model, raw_ps[0], WRT)
    specs = ['TT', 'TE', 'EE']
    worst = 0.0
    for j, key in enumerate(WRT):
        for s in range(3):
            bat = np.asarray(grads[j][s][0])       # batched dCl for cosmo 0, spectrum s
            rfd = np.asarray(ref[key][s])
            denom = np.maximum(np.abs(rfd), np.percentile(np.abs(rfd), 90) + 1e-30)
            rel = np.abs(bat - rfd) / denom
            m = float(np.nanmedian(rel)); worst = max(worst, m)
            print(f"  d Cl{specs[s]} / d {key:10s}: median rel = {m:.2e}", flush=True)
    print(f"\n  >>> worst MEDIAN rel(batched, single-path jacfwd) = {worst:.2e} "
          f"(<~1e-3 => batched AD gradient is CORRECT)", flush=True)
    print("[batched_grad] done", flush=True)


if __name__ == "__main__":
    main()

"""derisk_batched_ad.py — risk-#1 de-risk for batched AD on the per-k pipeline.

QUESTION (bench/batched_ad_design.md, risk #1): does forward-mode AD through the
BATCHED perturbation stage `PE.full_evolution_batched` (the k-chunked, params-axis
vmapped solver) (a) COMPILE in staged time (~ primal, NOT the ~20 min monolith we
get from vmap/jit of the whole cross-device pipeline), and (b) give a CORRECT
tangent?

ISOLATION: we don't need the upstream BG tangent or the single-cosmology reference
for THIS test.  We build a real (BG_batch, params_batch) primal, push a random
small tangent through `eqx.filter_jvp(full_evolution_batched)`, and check the
output tangent against a CENTRAL FINITE DIFFERENCE of full_evolution_batched ITSELF
(self-consistent).  That isolates the perturbation-stage differentiability + compile
behavior -- the make-or-break for the whole refactor.

Tiny scale (small l_max, B=2) so the compile is seconds-to-minutes; we care about
the RATIO jvp-compile / primal-compile (≈1-3x = staged/good; >>10x or a hang =
monolith/bad) and the jvp-vs-FD agreement.

Run via srun on a GPU node, PYTHONPATH=$(pwd).  Env: DRB_LMAX(128), DRB_EPS(1e-4).
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

LMAX = int(os.environ.get("DRB_LMAX", 128))
EPS = float(os.environ.get("DRB_EPS", 1e-4))

FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}
A_S = float(np.exp(3.044) / 1e10)
raw_ps = [
    dict(FIXED, h=0.6736, omega_b=0.02237, omega_cdm=0.1200, n_s=0.9649, A_s=A_S, tau_reion=0.0544),
    dict(FIXED, h=0.6800, omega_b=0.02240, omega_cdm=0.1190, n_s=0.9600, A_s=A_S, tau_reion=0.0550),
]


def make_tangent(tree, seed, scale=1e-3):
    """random small tangent on the inexact-array leaves of `tree` (None else)."""
    diff = eqx.filter(tree, eqx.is_inexact_array)
    leaves, td = jax.tree_util.tree_flatten(diff)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    tl = [scale * jax.random.normal(k, l.shape, l.dtype) for k, l in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(td, tl)


def add_scaled(tree, tan, eps):
    """tree + eps*tan on inexact leaves; static leaves untouched."""
    diff, static = eqx.partition(tree, eqx.is_inexact_array)
    diff2 = jax.tree_util.tree_map(lambda a, t: a + eps * t, diff, tan)
    return eqx.combine(diff2, static)


def inexact_leaves(tree):
    return jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  eps={EPS}", flush=True)
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=False,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    PE = model.PE

    # ---- primal: build BG_batch + params_batch, then the perturbation solve ----
    full_ps = [model.add_derived_parameters(p) for p in raw_ps]
    params_batch, BG_batch = model._build_bgs_batched(full_ps, shardfn=None)
    print(f"built BG_batch (B={len(raw_ps)}); n_k_pert={len(PE.k_axis_perturbations)}", flush=True)

    # partition static out so jvp sees pure-array inputs (clean tangent treedef
    # that aligns with the primal output under raw tree_leaves).
    bg_d, bg_s = eqx.partition(BG_batch, eqx.is_inexact_array)
    p_d, p_s = eqx.partition(params_batch, eqx.is_inexact_array)

    def f_arr(a, b):
        return PE.full_evolution_batched((eqx.combine(a, bg_s), eqx.combine(b, p_s)))

    t0 = time.perf_counter()
    PT = f_arr(bg_d, p_d); jax.block_until_ready(PT)
    t_primal_c = time.perf_counter() - t0
    t0 = time.perf_counter()
    PT = f_arr(bg_d, p_d); jax.block_until_ready(PT)
    t_primal = time.perf_counter() - t0
    print(f"[primal] full_evolution_batched: compile {t_primal_c:.1f}s, run {t_primal:.2f}s, "
          f"peak {peak_gb():.2f} GB", flush=True)

    # ---- forward-mode: jax.jvp over the array-partitioned inputs ----
    bg_dot = make_tangent(BG_batch, 1)   # diff-structured (None at static leaves)
    p_dot = make_tangent(params_batch, 2)
    print("pushing tangent via jax.jvp ...", flush=True)
    t0 = time.perf_counter()
    PT_p, PT_dot = jax.jvp(f_arr, (bg_d, p_d), (bg_dot, p_dot))
    jax.block_until_ready(PT_dot)
    t_jvp_c = time.perf_counter() - t0
    t0 = time.perf_counter()
    PT_p, PT_dot = jax.jvp(f_arr, (bg_d, p_d), (bg_dot, p_dot))
    jax.block_until_ready(PT_dot)
    t_jvp = time.perf_counter() - t0
    jdl = jax.tree_util.tree_leaves(PT_dot)
    finite = all(bool(np.all(np.isfinite(np.asarray(x)))) for x in jdl
                 if np.asarray(x).dtype.kind == 'f')
    print(f"[jvp] jax.jvp: compile {t_jvp_c:.1f}s, run {t_jvp:.2f}s, "
          f"peak {peak_gb():.2f} GB, tangent finite={finite}", flush=True)
    print(f"  >>> COMPILE RATIO jvp/primal = {t_jvp_c/max(t_primal_c,1e-9):.2f}x "
          f"(~1-3x = STAGED/good; >>10x or hang = monolith/bad)", flush=True)

    # ---- correctness: jvp tangent vs central FD of the stage ----
    # PT_dot (tangent of PT_p) and PTp/PTm (= f_arr outputs) share the f_arr
    # output treedef, so raw tree_leaves align 1:1.
    PTp = f_arr(jax.tree_util.tree_map(lambda x, t: x + EPS * t, bg_d, bg_dot),
                jax.tree_util.tree_map(lambda x, t: x + EPS * t, p_d, p_dot))
    PTm = f_arr(jax.tree_util.tree_map(lambda x, t: x - EPS * t, bg_d, bg_dot),
                jax.tree_util.tree_map(lambda x, t: x - EPS * t, p_d, p_dot))
    ppl = jax.tree_util.tree_leaves(PTp); pml = jax.tree_util.tree_leaves(PTm)
    print("\n[correctness] jvp tangent vs central-FD of the stage (per large leaf):", flush=True)
    worst = 0.0
    for jt, pp, pm in zip(jdl, ppl, pml):
        pp = np.asarray(pp)
        if pp.dtype.kind != 'f' or pp.size < 100:
            continue
        jt = np.asarray(jt); pm = np.asarray(pm)
        fd = (pp - pm) / (2 * EPS)
        denom = np.maximum(np.abs(fd), np.percentile(np.abs(fd), 90) + 1e-30)
        rel = np.abs(jt - fd) / denom
        m = float(np.nanmedian(rel)); mx = float(np.nanmax(rel)); worst = max(worst, m)
        print(f"  leaf shape={pp.shape}: median rel={m:.2e} max rel={mx:.2e}", flush=True)
    print(f"\n  >>> worst MEDIAN rel(jvp,FD) over large leaves = {worst:.2e} "
          f"(<~1e-3 => jvp through the per-k stage is CORRECT)", flush=True)
    print("[derisk_batched_ad] done", flush=True)


if __name__ == "__main__":
    main()

"""fwdad_validate.py — does FORWARD-mode AD work through the full
ABCMB -> plik-lite chi^2 pipeline, and what does it cost (memory + time)?

This is the make-or-break check for the optimizer-based frequentist profile
(Option B): per profile point we minimize chi^2 over the cosmological nuisances,
which needs the gradient d chi2 / d theta. Reverse-mode (vjp) stores the whole
perturbation trajectory -> OOM. Forward-mode (jvp) propagates tangents inline
(memory ~ primal) and costs ~N_theta x primal -- cheap because we only
differentiate the handful of nuisance directions. ABCMB defaults to
adjoint=diffrax.ForwardMode and uses forward-mode-compatible while loops, so this
*should* work.

We test:
  (1) single-cosmology  jacfwd(chi2)(theta) : does it trace cleanly through
      add_derived_parameters + HyRex(CPU) + perturbations(GPU) + spectrum + chi2?
      Correct? (vs central finite differences)  Time + memory factor vs primal.
  (2) batched  vmap(jacfwd(chi2))(Theta) over B points : does the production
      pattern (one gradient per POI point) vmap, and how does memory scale with B
      (sets the max batch for the optimizer)?

theta = [h, omega_b, omega_cdm, n_s, ln(10^10 A_s), tau_reion]; chi2 uses
A_planck = 1 (a smooth scalar -- the analytic A_planck/A_s profiling is handled
by the envelope theorem in production and is also smooth) plus the tau lowE prior.

Run via srun on a GPU node, PYTHONPATH=$(pwd).
Env: FWD_LMAX (2508), FWD_BLIST ("4,8,16") for the batched test.
"""
import os, time
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.plik_lite import PlikLite

LMAX = int(os.environ.get("FWD_LMAX", 2508))
BLIST = [int(x) for x in os.environ.get("FWD_BLIST", "4,8,16").split(",") if x]

FIXED = {
    'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
    'exp_reion': 1.5,
}
PNAMES = ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10As', 'tau_reion']
theta0 = jnp.array([0.6736, 0.02237, 0.1200, 0.9649, 3.044, 0.0544])
TAU_MU, TAU_SIG = 0.0544, 0.0073

pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)


def build(theta):
    h, ob, oc, ns, ln10As, tau = theta
    p = dict(FIXED)
    p['h'] = h; p['omega_b'] = ob; p['omega_cdm'] = oc; p['n_s'] = ns
    p['A_s'] = jnp.exp(ln10As) / 1e10; p['tau_reion'] = tau
    return p


def chi2(theta):
    """scalar chi^2(theta): full ABCMB -> plik-lite, A_planck=1, + tau prior."""
    p = build(theta)
    full = model.add_derived_parameters(p)
    out = model.run_cosmology_abbr(full)
    Dtt = pl.abcmb_cl_to_Dl(out.ClTT, out.l)
    Dte = pl.abcmb_cl_to_Dl(out.ClTE, out.l)
    Dee = pl.abcmb_cl_to_Dl(out.ClEE, out.l)
    m0 = pl.bin_model(Dtt, Dte, Dee)
    c2 = pl.chi2(m0, A_planck=1.0)
    return c2 + ((theta[5] - TAU_MU) / TAU_SIG) ** 2


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def main():
    print(f"devices: {jax.devices()}", flush=True)
    print(f"lmax={LMAX}  theta0={np.asarray(theta0)}", flush=True)

    # ---------- (1) single-cosmology primal ----------
    print("\n=== (1) single-cosmology ===", flush=True)
    t0 = time.perf_counter(); c0 = float(chi2(theta0)); jax.block_until_ready(c0)
    t_primal = time.perf_counter() - t0
    print(f"primal chi2 = {c0:.3f}   (wall incl compile {t_primal:.1f}s)", flush=True)
    # timed primal (post-compile)
    t0 = time.perf_counter(); c0b = float(chi2(theta0)); jax.block_until_ready(c0b)
    t_primal2 = time.perf_counter() - t0
    print(f"primal (post-compile) {t_primal2:.2f}s   peak {peak_gb():.2f} GB", flush=True)

    # ---------- forward-mode gradient ----------
    print("\n--- jacfwd(chi2) ---", flush=True)
    gfun = jax.jit(jax.jacfwd(chi2))
    try:
        t0 = time.perf_counter(); g = gfun(theta0); jax.block_until_ready(g)
        t_grad = time.perf_counter() - t0
        t0 = time.perf_counter(); g = gfun(theta0); jax.block_until_ready(g)
        t_grad2 = time.perf_counter() - t0
        g = np.asarray(g)
        print(f"jacfwd OK. grad = {g}", flush=True)
        print(f"finite: {np.all(np.isfinite(g))}   "
              f"wall incl compile {t_grad:.1f}s   post-compile {t_grad2:.2f}s", flush=True)
        print(f"TIME factor (grad/primal) = {t_grad2/max(t_primal2,1e-9):.2f}x   "
              f"peak {peak_gb():.2f} GB", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"jacfwd FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        g = None

    # ---------- finite-difference cross-check ----------
    if g is not None:
        print("\n--- finite-difference check (central) ---", flush=True)
        rel = np.full(len(theta0), np.nan)
        eps = np.array([5e-4, 5e-5, 5e-4, 1e-3, 5e-3, 5e-4])  # per-param steps
        for i in range(len(theta0)):
            tp = theta0.at[i].add(eps[i]); tm = theta0.at[i].add(-eps[i])
            gi = (float(chi2(tp)) - float(chi2(tm))) / (2 * eps[i])
            rel[i] = abs(gi - g[i]) / (abs(gi) + 1e-30)
            print(f"  d/d{PNAMES[i]:10s}: AD={g[i]:+.4e}  FD={gi:+.4e}  rel={rel[i]:.2e}",
                  flush=True)
        print(f"  max rel err AD-vs-FD = {np.nanmax(rel):.2e} "
              f"(FD limited by step + ABCMB solver noise ~1e-4)", flush=True)

    # ---------- (2) batched gradient (production pattern) ----------
    print("\n=== (2) batched vmap(jacfwd(chi2)) (one gradient per POI point) ===",
          flush=True)
    for B in BLIST:
        try:
            rng = np.random.default_rng(B)
            Theta = jnp.asarray(np.asarray(theta0)[None, :]
                                + rng.normal(0, 1, (B, len(theta0)))
                                * np.array([0.003, 1e-4, 8e-4, 3e-3, 0.01, 5e-3]))
            gb_fun = jax.jit(jax.vmap(jax.jacfwd(chi2)))
            t0 = time.perf_counter(); G = gb_fun(Theta); jax.block_until_ready(G)
            tcomp = time.perf_counter() - t0
            t0 = time.perf_counter(); G = gb_fun(Theta); jax.block_until_ready(G)
            trun = time.perf_counter() - t0
            G = np.asarray(G)
            print(f"  B={B:3d}: vmap(jacfwd) OK  finite={np.all(np.isfinite(G))}  "
                  f"compile {tcomp:.1f}s  run {trun:.2f}s ({trun/B:.2f}s/grad)  "
                  f"peak {peak_gb():.2f} GB", flush=True)
        except Exception as e:
            print(f"  B={B:3d}: FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
            break


if __name__ == "__main__":
    main()

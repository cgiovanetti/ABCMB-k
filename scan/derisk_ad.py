"""derisk_ad.py — feasibility gate for the AD-driven optimizer rebuild (#5/#1).

fwdad_validate.py already established: jacfwd(chi2) traces, is correct vs FD, is
memory-flat (~primal), and costs ~4x a forward eval. What it did NOT measure, and
what decides the optimizer architecture, is:

  (1) Does the ENVELOPE-profiled-A objective (the REAL production chi2, with
      A_planck analytically profiled under its prior and frozen via stop_gradient)
      give the correct gradient?  -> confirms we can drop the 51-pt FD stencil.
  (2) Forward-over-forward HESSIAN, single cosmology: correct (vs FD-of-AD-grad)?
      memory? time?  -> decides batched-Newton (need H on GPU) vs batched-BFGS
      (only need g; build H from gradient history on host).
  (3) vmap over the 13-POI grid of (f, g, H): peak memory on ONE A100-80GB.
      If it fits, batched-Newton over the whole POI grid in one sharded call is the
      clean rebuild; if not, fall back to batched-(f,g) + host BFGS.

theta = [h, omega_b, omega_cdm, n_s, ln10As, tau].  We differentiate w.r.t. the 5
NUISANCE directions for a chosen POI (the production inner problem).  A_planck is
profiled analytically w/ its N(1,0.0025) prior (envelope, stop_gradient); tau keeps
its lowE Gaussian here (the real lowE likelihood lands in scan/lowl_like.py).

Run via srun on a GPU node, PYTHONPATH=$(pwd).  Env: DR_LMAX(2508), DR_POI(n_s),
DR_NPOI(13), DR_DOBATCH(1).
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

LMAX = int(os.environ.get("DR_LMAX", 2508))
POI = os.environ.get("DR_POI", "n_s")
NPOI = int(os.environ.get("DR_NPOI", 13))
DOBATCH = os.environ.get("DR_DOBATCH", "1") != "0"

FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}
PNAMES = ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10As', 'tau_reion']
CENTER = np.array([0.6736, 0.02237, 0.1200, 0.9649, 3.044, 0.0544])
SIGMA = np.array([0.0054, 0.00015, 0.0012, 0.0042, 0.014, 0.0073])
TAU_MU, TAU_SIG = 0.0544, 0.0073
PIDX = PNAMES.index(POI)
NUIS_IDX = [i for i in range(6) if i != PIDX]   # 5 nuisance directions

pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
              rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
              rtol_small_k_PE=1e-5, max_steps_PE=16384)


def build(theta):
    h, ob, oc, ns, ln10As, tau = theta
    p = dict(FIXED)
    p['h'] = h; p['omega_b'] = ob; p['omega_cdm'] = oc; p['n_s'] = ns
    p['A_s'] = jnp.exp(ln10As) / 1e10; p['tau_reion'] = tau
    return p


def chi2_env(theta):
    """REAL production objective: plik-lite high-l with A_planck profiled
    analytically under its N(1,0.0025) prior (envelope: A* frozen via
    stop_gradient -> exact profiled gradient), plus the tau lowE Gaussian."""
    p = build(theta)
    out = model.run_cosmology_abbr(model.add_derived_parameters(p))
    Dtt = pl.abcmb_cl_to_Dl(out.ClTT, out.l)
    Dte = pl.abcmb_cl_to_Dl(out.ClTE, out.l)
    Dee = pl.abcmb_cl_to_Dl(out.ClEE, out.l)
    m0 = pl.bin_model(Dtt, Dte, Dee)
    A_star = jax.lax.stop_gradient(pl.profile_A(m0, with_prior=True)[1])
    diff = pl.X_data - m0 / (A_star ** 2)
    c2 = diff @ pl.invcov @ diff + ((A_star - 1.0) / 0.0025) ** 2
    return c2 + ((theta[5] - TAU_MU) / TAU_SIG) ** 2


def peak_gb():
    try:
        return max(d.memory_stats()["peak_bytes_in_use"] for d in jax.devices('gpu')) / 1e9
    except Exception:
        return float('nan')


def reset_peak():
    for d in jax.devices('gpu'):
        try:
            d.memory_stats  # noqa
        except Exception:
            pass
    # jax has no public peak reset; we report cumulative peak (monotone) and note it.


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  POI={POI} nuis={[PNAMES[i] for i in NUIS_IDX]}",
          flush=True)
    theta0 = jnp.asarray(CENTER)

    # ---------- primal ----------
    t0 = time.perf_counter(); c0 = float(chi2_env(theta0)); jax.block_until_ready(c0)
    print(f"\n[primal] chi2_env = {c0:.4f}  (incl compile {time.perf_counter()-t0:.1f}s)", flush=True)
    t0 = time.perf_counter(); c0 = float(chi2_env(theta0)); jax.block_until_ready(c0)
    tp = time.perf_counter() - t0
    print(f"[primal] post-compile {tp:.2f}s  peak {peak_gb():.2f} GB", flush=True)

    # ---------- (1) envelope gradient vs FD ----------
    print("\n=== (1) envelope-A gradient (jacfwd) vs central FD ===", flush=True)
    gfun = jax.jacfwd(chi2_env)
    t0 = time.perf_counter(); g = np.asarray(gfun(theta0)); jax.block_until_ready(g)
    tg = time.perf_counter() - t0
    t0 = time.perf_counter(); g = np.asarray(gfun(theta0)); jax.block_until_ready(g)
    tg2 = time.perf_counter() - t0
    print(f"jacfwd OK finite={np.all(np.isfinite(g))} incl-compile {tg:.1f}s "
          f"post {tg2:.2f}s ({tg2/max(tp,1e-9):.1f}x primal) peak {peak_gb():.2f} GB", flush=True)
    eps = np.array([5e-4, 5e-5, 5e-4, 1e-3, 5e-3, 5e-4])
    print("  param        AD            FD            rel", flush=True)
    relmax = 0.0
    for i in NUIS_IDX:
        tp_ = theta0.at[i].add(eps[i]); tm_ = theta0.at[i].add(-eps[i])
        gi = (float(chi2_env(tp_)) - float(chi2_env(tm_))) / (2 * eps[i])
        rel = abs(gi - g[i]) / (abs(gi) + 1e-30); relmax = max(relmax, rel)
        print(f"  {PNAMES[i]:10s}  {g[i]:+.5e}  {gi:+.5e}  {rel:.2e}", flush=True)
    print(f"  -> max rel(AD,FD) over nuisances = {relmax:.2e} "
          f"(FD limited by step + ~1e-5 solver tol)", flush=True)

    # ---------- (2) single-cosmology Hessian (forward-over-forward) ----------
    print("\n=== (2) Hessian jacfwd(jacfwd) single cosmology ===", flush=True)
    try:
        Hfun = jax.jacfwd(jax.jacfwd(chi2_env))
        t0 = time.perf_counter(); H = np.asarray(Hfun(theta0)); jax.block_until_ready(H)
        tH = time.perf_counter() - t0
        t0 = time.perf_counter(); H = np.asarray(Hfun(theta0)); jax.block_until_ready(H)
        tH2 = time.perf_counter() - t0
        Hn = H[np.ix_(NUIS_IDX, NUIS_IDX)]
        evals = np.linalg.eigvalsh(Hn)
        print(f"Hessian OK finite={np.all(np.isfinite(H))} incl-compile {tH:.1f}s "
              f"post {tH2:.2f}s ({tH2/max(tp,1e-9):.1f}x primal) peak {peak_gb():.2f} GB", flush=True)
        print(f"  5x5 nuisance-block eigenvalues: {np.array2string(evals, precision=3)}", flush=True)
        print(f"  PD? {np.all(evals > 0)}  cond={evals.max()/max(evals.min(),1e-30):.1e}", flush=True)
        # cross-check one Hessian entry vs FD-of-AD-gradient
        i = NUIS_IDX[0]
        gp = np.asarray(gfun(theta0.at[i].add(eps[i])))
        gm = np.asarray(gfun(theta0.at[i].add(-eps[i])))
        Hii_fd = (gp[i] - gm[i]) / (2 * eps[i])
        print(f"  H[{PNAMES[i]},{PNAMES[i]}]: AD={H[i,i]:+.4e} FD-of-grad={Hii_fd:+.4e} "
              f"rel={abs(Hii_fd-H[i,i])/(abs(Hii_fd)+1e-30):.2e}", flush=True)
        hess_ok = True
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Hessian FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        hess_ok = False

    # ---------- (3) batched over the POI grid: (f, g, H) vmap ----------
    if DOBATCH:
        c, s = CENTER[PIDX], SIGMA[PIDX]
        poi_grid = np.linspace(c - 3 * s, c + 3 * s, NPOI)
        Theta = np.tile(CENTER, (NPOI, 1)); Theta[:, PIDX] = poi_grid
        Theta = jnp.asarray(Theta)
        print(f"\n=== (3) vmap over {NPOI}-POI grid ===", flush=True)
        # (f,g): only jacfwd (memory-flat path) -> the BFGS-safe fallback
        try:
            fg = jax.jit(jax.vmap(lambda th: (chi2_env(th), jax.jacfwd(chi2_env)(th))))
            t0 = time.perf_counter(); F, G = fg(Theta); jax.block_until_ready((F, G))
            tc = time.perf_counter() - t0
            t0 = time.perf_counter(); F, G = fg(Theta); jax.block_until_ready((F, G))
            tr = time.perf_counter() - t0
            print(f"  vmap(f,g) B={NPOI}: OK compile {tc:.1f}s run {tr:.2f}s "
                  f"({tr/NPOI:.2f}s/pt) peak {peak_gb():.2f} GB  [BFGS path]", flush=True)
        except Exception as e:
            print(f"  vmap(f,g) FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
        # (f,g,H): forward-over-forward batched -> the Newton path
        if hess_ok:
            try:
                fgh = jax.jit(jax.vmap(lambda th: (chi2_env(th), jax.jacfwd(chi2_env)(th),
                                                   jax.jacfwd(jax.jacfwd(chi2_env))(th))))
                t0 = time.perf_counter(); F, G, H = fgh(Theta); jax.block_until_ready((F, G, H))
                tc = time.perf_counter() - t0
                t0 = time.perf_counter(); F, G, H = fgh(Theta); jax.block_until_ready((F, G, H))
                tr = time.perf_counter() - t0
                print(f"  vmap(f,g,H) B={NPOI}: OK compile {tc:.1f}s run {tr:.2f}s "
                      f"({tr/NPOI:.2f}s/pt) peak {peak_gb():.2f} GB  [Newton path]", flush=True)
                print("  -> Newton path FITS on one 80GB device for the whole POI grid",
                      flush=True)
            except Exception as e:
                print(f"  vmap(f,g,H) B={NPOI}: FAILED {type(e).__name__}: {str(e)[:160]}",
                      flush=True)
                print("  -> use batched-(f,g) + host BFGS instead", flush=True)

    print("\n[derisk] done", flush=True)


if __name__ == "__main__":
    main()

"""profile_opt.py — optimizer-based frequentist profile (Option B), test driver.

For each value of the parameter of interest (POI = n_s), MINIMIZE chi^2 over the
cosmological nuisances {h, omega_b, omega_cdm} with L-BFGS-B (warm-started across
POI). The A_s / A_planck overall amplitude is profiled ANALYTICALLY
(pl.profile_amplitude: chi2 = a - b^2/c), so it is not an optimizer dimension;
tau is fixed at 0.0544. Optimize in sigma-scaled coordinates for conditioning.

Gradient method (env GRAD):
  * "fd"  (DEFAULT) — central finite differences via ONE call_batched per
    evaluation over the (1 + 2*n_nuis)-point stencil. Uses the fast, proven
    0.66 s/param batched path. Robust; descent-quality gradients.
  * "fwdad" — ABCMB forward-mode AD (jax.jacfwd), single-cosmology path. This is
    the VALIDATED production gradient (memory-flat, ~4x a forward eval, scan/
    OPTION_B_feasibility.md). NOTE: do NOT wrap the cross-device chi2 in an outer
    jax.jit (pathological compile); we call jacfwd(obj) directly. Slower here only
    because the single-cosmology path isn't batched; production batches it through
    call_batched.

Checkpoints scan/results/profile_opt_ns.npz after EVERY POI (resilient to
walltime). Run via srun (1 GPU ok), PYTHONPATH=$(pwd).
Env: OPT_POI_LO/HI/N, OPT_LMAX, GRAD, OPT_FD_DELTA.
"""
import os, time
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
from scipy.optimize import minimize
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.plik_lite import PlikLite

LMAX = int(os.environ.get("OPT_LMAX", 2508))
NS_LO = float(os.environ.get("OPT_POI_LO", 0.945))
NS_HI = float(os.environ.get("OPT_POI_HI", 0.985))
NS_N = int(os.environ.get("OPT_POI_N", 9))
GRAD = os.environ.get("GRAD", "fd").lower()
FD_DELTA = float(os.environ.get("OPT_FD_DELTA", 0.3))   # in sigma-scaled units

A_S_REF = float(np.exp(3.044) / 1e10)
FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'tau_reion': 0.0544, 'Delta_z_reion': 0.5, 'z_reion_He': 3.5,
         'Delta_z_reion_He': 0.5, 'exp_reion': 1.5}
NUIS = ['h', 'omega_b', 'omega_cdm']
CENTER = np.array([0.6736, 0.02237, 0.1200])
SIGMA = np.array([0.0054, 0.00015, 0.0012])
NDIM = len(NUIS)

pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)


def build(phys, ns):
    p = dict(FIXED)
    p['h'] = float(phys[0]); p['omega_b'] = float(phys[1]); p['omega_cdm'] = float(phys[2])
    p['n_s'] = float(ns); p['A_s'] = A_S_REF
    return p


def chi2_batched(phys_rows, ns):
    """phys_rows: (B,3) physical nuisances -> (B,) amplitude-profiled chi^2."""
    batch = [build(phys_rows[k], ns) for k in range(len(phys_rows))]
    out = model.call_batched(batch, shard=True)   # proven path (bench/demo); on
    # 1 GPU this is a 1-device mesh. (shard=False at B=7 hit an XLA codegen bug.)
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    return np.asarray(pl.profile_amplitude(m0)[0])


def fun_and_grad_fd(x, ns):
    """x: (3,) sigma-scaled. central-FD gradient via one call_batched stencil."""
    pts = [x]
    for i in range(NDIM):
        e = np.zeros(NDIM); e[i] = FD_DELTA
        pts.append(x + e); pts.append(x - e)
    phys = CENTER + SIGMA * np.array(pts)         # (1+2n, 3)
    c2 = chi2_batched(phys, ns)
    f = c2[0]
    g = np.array([(c2[1 + 2 * i] - c2[2 + 2 * i]) / (2 * FD_DELTA) for i in range(NDIM)])
    return float(f), g


# --- forward-AD path (single cosmology; NO outer jax.jit) ---
def _obj_x(x, ns):
    out = model.run_cosmology_abbr(model.add_derived_parameters(
        build(CENTER + SIGMA * np.asarray(x), ns)))
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    return pl.profile_amplitude(m0)[0]


def fun_and_grad_ad(x, ns):
    xj = jnp.asarray(x); nsj = jnp.asarray(ns)
    f = float(_obj_x(xj, nsj))
    g = np.asarray(jax.jacfwd(_obj_x, argnums=0)(xj, nsj), dtype=float)
    return f, g


def interval_1sigma(x, y):
    y = np.asarray(y); x = np.asarray(x)
    m = np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    x, y = x[m], y[m]; i = int(np.argmin(y)); t = y[i] + 1.0
    def cr(up):
        rng = range(i, len(x) - 1) if up else range(i, 0, -1)
        for k in rng:
            j = k + 1 if up else k - 1
            if (y[k] - t) * (y[j] - t) <= 0:
                f = (t - y[k]) / (y[j] - y[k] + 1e-30); return x[k] + f * (x[j] - x[k])
        return np.nan
    return cr(False), x[i], cr(True)


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  GRAD={GRAD}", flush=True)
    ns_grid = np.linspace(NS_LO, NS_HI, NS_N)
    print(f"POI n_s grid ({NS_N}): {ns_grid}", flush=True)
    fg = fun_and_grad_fd if GRAD == "fd" else fun_and_grad_ad

    chi2min = np.full(NS_N, np.nan); xstar = np.full((NS_N, NDIM), np.nan)
    niters = np.full(NS_N, np.nan)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out, exist_ok=True); npz = os.path.join(out, "profile_opt_ns.npz")

    x0 = np.zeros(NDIM); t_all = time.perf_counter()
    for i, ns in enumerate(ns_grid):
        t0 = time.perf_counter()
        res = minimize(fg, x0, args=(ns,), jac=True, method="L-BFGS-B",
                       bounds=[(-6, 6)] * NDIM,
                       options={"maxiter": 50, "ftol": 1e-10, "gtol": 1e-6})
        dt = time.perf_counter() - t0
        chi2min[i] = res.fun; xstar[i] = res.x; niters[i] = res.nit; x0 = res.x
        phys = CENTER + SIGMA * res.x
        print(f"  n_s={ns:.4f}: chi2={res.fun:.3f} nit={res.nit} nfev={res.nfev}  "
              f"h={phys[0]:.4f} ob={phys[1]:.5f} ocdm={phys[2]:.5f}  ({dt:.0f}s)", flush=True)
        np.savez(npz, ns=ns_grid, chi2=chi2min, xstar=xstar, niters=niters,
                 center=CENTER, sigma=SIGMA, grad=GRAD)   # checkpoint every POI
    print(f"\ntotal {time.perf_counter()-t_all:.0f}s  -> {npz}", flush=True)

    lo, mid, hi = interval_1sigma(ns_grid, chi2min)
    j = int(np.nanargmin(chi2min))
    print(f"global min chi2 = {chi2min[j]:.2f} at n_s={ns_grid[j]:.4f}", flush=True)
    print(f"profiled n_s = {mid:.4f}  1sigma [{lo:.4f}, {hi:.4f}]  (+/-{(hi-lo)/2:.4f})",
          flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        d = chi2min - np.nanmin(chi2min)
        ax[0].plot(ns_grid, d, "o-")
        for lv, c in [(1, "g"), (4, "orange"), (9, "r")]:
            ax[0].axhline(lv, ls="--", lw=0.8, color=c)
        ax[0].axvline(0.9649, ls=":", color="gray", lw=0.8, label="Planck n_s")
        ax[0].set_xlabel("n_s"); ax[0].set_ylabel(r"$\Delta\chi^2$ (profiled)")
        ax[0].set_ylim(0, 12); ax[0].legend()
        ax[0].set_title(f"optimizer profile of $n_s$ vs plik-lite (grad={GRAD})")
        ax[1].plot(ns_grid, niters, "s-")
        ax[1].set_xlabel("n_s"); ax[1].set_ylabel("L-BFGS-B iterations")
        ax[1].set_title("iterations / POI (warm-started)")
        fig.tight_layout(); png = npz.replace(".npz", ".png"); fig.savefig(png, dpi=110)
        print(f"saved -> {png}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)


if __name__ == "__main__":
    main()

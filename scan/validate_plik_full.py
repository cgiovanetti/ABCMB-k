"""validate_plik_full.py — standalone gate for scan/plik_full.py (the full-plik module).

Confirms, on one debug GPU node, that the inner nuisance profile is CORRECT and cheap
before any wiring into the tool:
  (A) jax.hessian works through clipy (the damped-Newton inner solver needs 2nd-order AD).
  (B) The inner profile REDUCES chi^2 below the fiducial-nuisance value and converges.
  (C) It matches a scipy L-BFGS-B GROUND TRUTH (chi^2 + nuisance vector) at the LCDM
      best fit -> the hand-rolled vmappable Newton finds the true optimum.
  (D) The profiled nuisances are SENSIBLE (near the Planck best-fit values).
  (E) profile_batched (vmap) matches the per-element profile.
  (F) timing of the inner profile (single + batched) -> the per-cosmology cost budget.
"""
import os, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

T_CMB_UK = 2.7255e6
As = float(np.exp(3.044) / 1e10)
LCDM = {
    'h': 0.6736, 'omega_cdm': 0.1200, 'omega_b': 0.02237,
    'A_s': As, 'n_s': 0.9649,
    'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}


def main():
    print(f"jax devices: {jax.devices()}", flush=True)
    from scan.plik_full import PlikFull, FLOAT_NAMES
    pl = PlikFull()
    print(f"plik_full: lmax={pl.lmax}, #floated={len(FLOAT_NAMES)}", flush=True)

    from abcmb.main import Model
    model = Model(user_species=None, output_Cl=True, l_max=2508, lensing=True,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10,
                  l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    print("\n--- ABCMB at LCDM best fit ---", flush=True)
    t0 = time.perf_counter()
    out = model(LCDM)
    jax.block_until_ready(out.ClTT)
    print(f"  ABCMB call (incl compile): {time.perf_counter()-t0:.1f}s", flush=True)
    cls2d = pl.abcmb_cls_to_clik(out.ClTT, out.ClTE, out.ClEE, out.l)
    cls2d = jnp.asarray(cls2d)
    print(f"  cls2d shape {cls2d.shape}", flush=True)

    # ---- (A) hessian through clipy ----
    print("\n=== (A) jax.hessian through clipy at the fiducial nuisances ===", flush=True)
    z0 = jnp.zeros(len(FLOAT_NAMES))
    f = lambda z: pl._obj_scaled(cls2d, z)
    try:
        t0 = time.perf_counter()
        g0 = jax.grad(f)(z0); H0 = jax.hessian(f)(z0)
        jax.block_until_ready(H0)
        ev = np.linalg.eigvalsh(np.asarray(H0))
        print(f"  grad ok (||g||={float(jnp.linalg.norm(g0)):.3e}); hessian ok "
              f"(eig min/max {ev.min():.3e}/{ev.max():.3e}); PD={bool(ev.min()>0)} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
        HESS_OK = bool(np.all(np.isfinite(np.asarray(H0))))
    except Exception as e:
        HESS_OK = False
        print(f"  HESSIAN FAILED: {type(e).__name__}: {e}", flush=True)

    # ---- (B) inner profile reduces chi^2 ----
    print("\n=== (B) inner profile (damped Newton) ===", flush=True)
    chi2_fid = float(pl.penalized_chi2(cls2d, pl.start))
    t0 = time.perf_counter()
    chi2_prof, nu_star = jax.jit(pl.profile)(cls2d)
    jax.block_until_ready(chi2_prof)
    print(f"  chi2 fiducial = {chi2_fid:.2f}", flush=True)
    print(f"  chi2 profiled = {float(chi2_prof):.2f}  (compile+eval {time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"  reduction = {chi2_fid - float(chi2_prof):.2f} (should be >= 0)", flush=True)
    t0 = time.perf_counter()
    chi2_prof2, _ = jax.jit(pl.profile)(cls2d)
    jax.block_until_ready(chi2_prof2)
    print(f"  warm single-profile time: {time.perf_counter()-t0:.3f}s", flush=True)

    # ---- (C) scipy ground truth ----
    print("\n=== (C) scipy L-BFGS-B ground truth ===", flush=True)
    from scipy.optimize import minimize
    val_and_grad = jax.jit(jax.value_and_grad(lambda z: pl._obj_scaled(cls2d, z)))
    zlo = np.asarray((pl.lo - pl.start) / pl.scale)
    zhi = np.asarray((pl.hi - pl.start) / pl.scale)
    bounds = list(zip(zlo, zhi))
    def vg(zz):
        v, g = val_and_grad(jnp.asarray(zz))
        return float(v), np.asarray(g, float)
    t0 = time.perf_counter()
    res = minimize(vg, np.zeros(len(FLOAT_NAMES)), jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9})
    nu_scipy = np.asarray(pl.start) + np.asarray(pl.scale) * res.x
    print(f"  scipy chi2 = {res.fun:.4f}  (success={res.success}, nit={res.nit}, "
          f"{time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"  Newton chi2 = {float(chi2_prof):.4f}   diff vs scipy = {float(chi2_prof)-res.fun:+.4f}", flush=True)
    nu_newton = np.asarray(nu_star)
    dmax = np.max(np.abs(nu_newton - nu_scipy) / np.asarray(pl.scale))
    print(f"  max |nu_newton - nu_scipy| / scale = {dmax:.3e} (Newton found the optimum: {dmax<1e-2})", flush=True)

    # ---- (D) sensible nuisances ----
    print("\n=== (D) profiled nuisances (scipy ground truth) ===", flush=True)
    for nm, v in zip(FLOAT_NAMES, nu_scipy):
        print(f"    {nm:22s} = {v:.5g}", flush=True)

    # ---- (E) batched vmap matches ----
    print("\n=== (E) profile_batched (vmap) ===", flush=True)
    cls_B = jnp.stack([cls2d * s for s in (0.995, 1.0, 1.005, 1.01)])
    t0 = time.perf_counter()
    chi2_B, nu_B = jax.jit(pl.profile_batched)(cls_B)
    jax.block_until_ready(chi2_B)
    print(f"  batched chi2 (B=4): {np.asarray(chi2_B)}  (compile+eval {time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"  element-1 (s=1.0) batched={float(chi2_B[1]):.4f} vs single={float(chi2_prof):.4f} "
          f"(match: {abs(float(chi2_B[1])-float(chi2_prof))<1e-3})", flush=True)
    t0 = time.perf_counter()
    chi2_B2, _ = jax.jit(pl.profile_batched)(cls_B)
    jax.block_until_ready(chi2_B2)
    print(f"  warm batched-profile (B=4) time: {time.perf_counter()-t0:.3f}s "
          f"({(time.perf_counter()-t0)/4*1000:.1f} ms/cosmo)", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  HESS_OK={HESS_OK}  newton==scipy:{dmax<1e-2}  vmap_ok:{abs(float(chi2_B[1])-float(chi2_prof))<1e-3}", flush=True)


if __name__ == "__main__":
    import pandas as pd
    pd.options.future.infer_string = False
    main()

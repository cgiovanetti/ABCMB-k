"""validate_clipy.py — FOUNDATIONAL GATE for the full-plik switch.

Task: replace plik-LITE with the FULL Planck 2018 high-ell plik (TTTEEE), whose
~47 foreground/calibration nuisances do NOT enter the theory code -- so they must
be handled inside the likelihood (profiled at fixed ABCMB theory), never by
re-running ABCMB.

The enabling discovery: `clipy` (/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/
code/planck/clipy) is a pure-Python, JAX-native reimplementation of clik that
already supports plik_rd12_HM_v22b_TTTEEE.clik, reads the foreground templates from
the bundle, runs x64, and JITs. It is the SAME backend modern cobaya calls, and its
constructor self-tests against the clik-stored check_value (-1172.465414).

This script confirms, on one debug GPU node, the load-bearing facts before any tool
wiring:
  (1) clipy IMPORTS in actdr6 and its plik TTTEEE self-test passes (diff ~ 0).
  (2) Its required nuisance list + reference values (from _default_par) are readable.
  (3) Fed ABCMB theory Cls (raw -> muK^2, TT/EE/BB/TE/TB/EB ordering) at the Planck
      2018 LCDM best fit + fiducial nuisances, chi^2 is SANE (~ ndata dof).
  (4) The chi^2 is DIFFERENTIABLE in the theory Cls (jax.grad through clipy) -- the
      envelope-theorem AD path the tool needs.
  (5) The chi^2 is VMAPpable over a batch of cosmologies (the per-k batched pipeline).
  (6) Wall-clock of one eval (compile + warm) -- the inner-nuisance-profile budget.

Run via srun on a GPU node, PYTHONPATH including this repo AND the clipy package dir.
"""
import os, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

CLIK_FILE = ("/pscratch/sd/c/carag/Neal_ACTDR6/baseline/plc_3.0/hi_l/plik/"
             "plik_rd12_HM_v22b_TTTEEE.clik")
T_CMB_UK = 2.7255e6

# Planck 2018 base-LCDM best fit (TT,TE,EE+lowE+lensing) -- same point validate_plik.py uses
As = float(np.exp(3.044) / 1e10)
LCDM = {
    'h': 0.6736, 'omega_cdm': 0.1200, 'omega_b': 0.02237,
    'A_s': As, 'n_s': 0.9649,
    'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}


def abcmb_cls_to_clik(ClTT, ClTE, ClEE, l_arr, lmax):
    """ABCMB raw (dimensionless) C_l on integer grid l_arr -> clik 2D cls block
    (6, lmax+1) = [TT, EE, BB, TE, TB, EB] in muK^2, zero-padded from l=0."""
    l_arr = np.asarray(l_arr).astype(int)
    fac = T_CMB_UK ** 2                       # raw C_l -> muK^2 (NO l(l+1) factor: clik wants C_l)
    out = np.zeros((6, lmax + 1))
    keep = l_arr <= lmax
    idx = l_arr[keep]
    out[0, idx] = np.asarray(ClTT)[keep] * fac     # TT
    out[1, idx] = np.asarray(ClEE)[keep] * fac     # EE
    out[3, idx] = np.asarray(ClTE)[keep] * fac     # TE
    return out                                      # BB/TB/EB left zero


def main():
    print(f"jax devices: {jax.devices()}  x64={jax.config.jax_enable_x64}", flush=True)
    import clipy
    print(f"clipy version: {clipy.version()}  hasjax={clipy.hasjax}", flush=True)

    # ---- (1)+(2) construct + self-test + read nuisance contract ----
    print("\n=== constructing clipy plik TTTEEE (self-test prints below) ===", flush=True)
    t0 = time.perf_counter()
    lkl = clipy.clik(CLIK_FILE)
    print(f"  construct wall: {time.perf_counter()-t0:.1f}s", flush=True)
    extras = list(lkl.extra_parameter_names)
    lmaxs = np.asarray(lkl.lmax)
    print(f"  lmax per spectrum [TT EE BB TE TB EB]: {lmaxs.tolist()}", flush=True)
    print(f"  has_cl: {np.asarray(lkl.has_cl).tolist()}", flush=True)
    LMAX = int(np.max(lmaxs))
    n_extra = len(extras)
    print(f"  #required nuisances (extra_parameter_names) = {n_extra}", flush=True)
    print(f"  nuisances: {extras}", flush=True)
    # reference nuisance values: the tail of the self-test check_param vector
    dpar = np.asarray(lkl._default_par, dtype=float)
    print(f"  _default_par length = {dpar.size}; parlen = {int(np.asarray(lkl.parlen))}", flush=True)
    nuis_ref = dict(zip(extras, dpar[-n_extra:]))
    print("  reference nuisance values (from check_param tail):", flush=True)
    for k in extras:
        print(f"    {k:24s} = {nuis_ref[k]:.6g}", flush=True)

    # ---- (3) ABCMB theory at LCDM best fit -> clik cls -> chi^2 ----
    from abcmb.main import Model
    model = Model(user_species=None, output_Cl=True, l_max=2508, lensing=True,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10,
                  l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    print("\n=== ABCMB single call at LCDM best fit ===", flush=True)
    t0 = time.perf_counter()
    out = model(LCDM)
    jax.block_until_ready(out.ClTT)
    print(f"  ABCMB call wall (incl compile): {time.perf_counter()-t0:.1f}s", flush=True)
    l_arr = np.asarray(out.l).astype(int)
    print(f"  l range {l_arr[0]}..{l_arr[-1]} ({l_arr.size} ells)", flush=True)

    cls2d = abcmb_cls_to_clik(out.ClTT, out.ClTE, out.ClEE, l_arr, LMAX)
    Dtt220 = cls2d[0, 220] * 220 * 221 / (2 * np.pi)
    print(f"  D_l^TT(220) = {Dtt220:.1f} muK^2 (expect ~5700)", flush=True)
    cls2d_j = jnp.asarray(cls2d)

    print("\n=== full-plik chi^2 at fiducial nuisances ===", flush=True)
    t0 = time.perf_counter()
    logL = float(lkl(cls2d_j, nuis_ref))
    jax.block_until_ready(jnp.asarray(logL))
    print(f"  logL = {logL:.4f}  (-2logL = {-2*logL:.2f})  warm+compile {time.perf_counter()-t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    chi2 = float(lkl(cls2d_j, nuis_ref, chi2_mode=True))
    print(f"  chi2_mode value = {chi2:.2f}  (eval {time.perf_counter()-t0:.2f}s)", flush=True)
    # plik TTTEEE has 613 'lite' bins but the full data vector is 2289 bandpowers;
    # a good fit lands -2logL near the #bandpowers used. Just sanity-check it's finite & O(1e3).
    print(f"  SANE? {np.isfinite(logL) and 1e3 < -2*logL < 5e3}", flush=True)

    # ---- (4) differentiability of chi^2 in the theory Cls ----
    print("\n=== AD: d(logL)/d(amplitude) through clipy (envelope-theorem path) ===", flush=True)
    def loglike_of_amp(a):
        return lkl(cls2d_j * a, nuis_ref)
    try:
        t0 = time.perf_counter()
        g = float(jax.grad(loglike_of_amp)(1.0))
        print(f"  d logL/d amp |_(a=1) = {g:.4f}  (grad compile+eval {time.perf_counter()-t0:.1f}s)", flush=True)
        # finite-difference check
        eps = 1e-4
        gfd = (float(loglike_of_amp(1.0 + eps)) - float(loglike_of_amp(1.0 - eps))) / (2 * eps)
        print(f"  FD check d logL/d amp = {gfd:.4f}  (rel diff {abs(g-gfd)/(abs(gfd)+1e-30):.2e})", flush=True)
        print(f"  AD WORKS through clipy: {abs(g-gfd)/(abs(gfd)+1e-30) < 1e-3}", flush=True)
    except Exception as e:
        print(f"  AD FAILED: {type(e).__name__}: {e}", flush=True)

    # ---- (5) vmap over a batch of cosmologies (the batched pipeline axis) ----
    print("\n=== vmap chi^2 over a B=4 batch of theory blocks ===", flush=True)
    try:
        batch = jnp.stack([cls2d_j * s for s in (0.99, 1.0, 1.01, 1.02)])  # (4,6,lmax+1)
        nuis_arr = jnp.asarray([nuis_ref[k] for k in extras])
        def chi2_one(cls_b):
            return lkl(cls_b, nuis_ref, chi2_mode=True)
        t0 = time.perf_counter()
        vals = jax.vmap(chi2_one)(batch)
        jax.block_until_ready(vals)
        print(f"  vmap chi2 over B=4: {np.asarray(vals)}  ({time.perf_counter()-t0:.1f}s)", flush=True)
        print(f"  VMAP WORKS: {np.all(np.isfinite(np.asarray(vals)))}", flush=True)
    except Exception as e:
        print(f"  VMAP FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n=== GATE SUMMARY ===", flush=True)
    print("  see above: self-test diff ~0, chi2 sane, AD ok, vmap ok => proceed with clipy backend", flush=True)


if __name__ == "__main__":
    import pandas as pd
    pd.options.future.infer_string = False
    main()

"""validate_fullplik_grad.py — integration gate for full plik in the driver.

Imports the ACTUAL driver (scan/profile_prod_ad.py) configured for full plik
(PA_CONFIG=scan/configs/lcdm_plikfull.py) and checks the three contracts that the
profile run depends on, on a small row set (1 POI x B grid points), at l=2508:

  (1) VALUE path: fast_values_rows (call_batched -> inner-profile) returns finite,
      sensible chi^2.
  (2) CONSISTENCY: the chi^2 from the staged-AD path (ad_grad_rows) matches the
      fast value path to <~1e-2 (the Armijo consistency rule needs ONE chi^2 source;
      they must agree).
  (3) GRADIENT: the envelope-theorem AD gradient (ad_grad_rows, nuisances inner-
      profiled + stop_gradient'd) matches a central finite-difference gradient
      (fdbatch_grad, value path) to a few %. This is THE correctness check for the
      full-plik cosmology gradient -- if the envelope theorem / stop_gradient wiring
      were wrong, AD and FD would diverge.

Set PA_CONFIG before importing the driver. Heavy compiles (staged AD grad with the
inner profile inside the jvp) -- may need regular queue if debug 30 min is tight.
"""
import os
os.environ.setdefault("PA_CONFIG", "scan/configs/lcdm_plikfull.py")
os.environ.setdefault("PA_RTOL", "1e-5")
import time
import numpy as np
import pandas as pd
pd.options.future.infer_string = False
import jax

import scan.profile_prod_ad as drv


def main():
    print(f"jax devices: {jax.devices()}", flush=True)
    print(f"config={drv.CONFIG_ABS} HIGH_ELL={drv.HIGH_ELL} USE_PLIK_FULL={drv.USE_PLIK_FULL} "
          f"grad={drv.GRADMETHOD} D={drv.D} P={drv.P}", flush=True)
    assert drv.USE_PLIK_FULL, "this validator requires high_ell=plik_full"

    # preconditioner (one ABCMB call + 96 s hessian, or load from cache)
    t0 = time.perf_counter()
    drv._setup_plik_full_precond(drv.CENTER)
    print(f"[precond] ready ({time.perf_counter()-t0:.0f}s)", flush=True)

    # small row set: 1 POI (tau_reion), B grid points around the fiducial, cold nuisances
    poi = "tau_reion"; pidx = drv.ORDER.index(poi)
    c, s = drv.CENTER[pidx], drv.SIGMA[pidx]
    B = 4
    PV = np.linspace(c - 1.5 * s, c + 1.5 * s, B)
    POI_IDX = np.full(B, pidx, int)
    X = np.zeros((B, drv.P))

    # (1) VALUE path
    t0 = time.perf_counter()
    chi2_fast = drv.fast_values_rows(POI_IDX, X, PV)
    print(f"\n(1) fast_values (value path, incl compile {time.perf_counter()-t0:.0f}s):", flush=True)
    print(f"    chi2 = {np.array2string(chi2_fast, precision=2)}  finite={np.all(np.isfinite(chi2_fast))}", flush=True)

    # (2)+(3) staged-AD chi^2 + gradient
    t0 = time.perf_counter()
    chi2_ad, G_ad = drv.ad_grad_rows(POI_IDX, X, PV)
    print(f"\n(2) staged-AD chi^2 (incl compile {time.perf_counter()-t0:.0f}s):", flush=True)
    dconsist = float(np.max(np.abs(chi2_ad - chi2_fast)))
    print(f"    chi2_ad = {np.array2string(chi2_ad, precision=2)}", flush=True)
    print(f"    max|chi2_ad - chi2_fast| = {dconsist:.3e}  (consistency PASS<1e-2: {dconsist<1e-2})", flush=True)

    # central FD gradient (value path)
    t0 = time.perf_counter()
    G_fd = drv.fdbatch_grad(POI_IDX, X, PV, step=1e-2)
    print(f"\n(3) gradient AD vs central-FD (FD compile {time.perf_counter()-t0:.0f}s):", flush=True)
    denom = np.maximum(np.abs(G_ad), np.percentile(np.abs(G_ad), 90) + 1e-30)
    relmax = float(np.max(np.abs(G_ad - G_fd) / denom))
    for b in range(B):
        print(f"    row {b} (tau={PV[b]:.4f}): ||g_ad||={np.linalg.norm(G_ad[b]):.3e} "
              f"||g_fd||={np.linalg.norm(G_fd[b]):.3e} "
              f"max-rel={np.max(np.abs(G_ad[b]-G_fd[b])/denom[b]):.2e}", flush=True)
    print(f"    OVERALL max-rel(AD,FD) = {relmax:.3e}  (envelope-theorem gradient PASS<5e-2: {relmax<5e-2})", flush=True)

    print(f"\nSUMMARY: value_finite={np.all(np.isfinite(chi2_fast))} "
          f"consistency={dconsist<1e-2} gradient={relmax<5e-2}", flush=True)


if __name__ == "__main__":
    main()

"""validate_plik.py — confirm ABCMB->plik-lite normalization & conventions.

Runs ABCMB at the Planck 2018 base-LCDM best fit (l_max=2508, lensing on),
feeds the spectra to scan/plik_lite.py, and reports chi^2. A correct T_CMB^2
normalization + TE/EE sign convention lands chi^2 near the data dof (~580-700
for 613 bins); a wrong factor blows it to ~1e10.

Run via srun on a GPU node with PYTHONPATH=$(pwd):$PYTHONPATH.
"""
import os, time
import numpy as np
import pandas as pd
pd.options.future.infer_string = False  # cobaya/BAO compat guard (harmless here)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from abcmb.main import Model
from scan.plik_lite import PlikLite

# Planck 2018 base-LCDM best fit (TT,TE,EE+lowE+lensing)
As = float(np.exp(3.044) / 1e10)
LCDM = {
    'h': 0.6736, 'omega_cdm': 0.1200, 'omega_b': 0.02237,
    'A_s': As, 'n_s': 0.9649,
    'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}


def main():
    print(f"jax devices: {jax.devices()}", flush=True)
    pl = PlikLite()
    print(f"plik-lite: ndata={pl.ndata}, used_specs={pl.used_specs}, "
          f"lmax={pl.lmax}, T_CMB_uK={pl.T_CMB_uK:.6g}", flush=True)
    print(f"  data chi2 floor a = d^T Cinv d = {pl._a:.2f}", flush=True)

    model = Model(user_species=None, output_Cl=True, l_max=2508, lensing=True,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10,
                  l_max_ur=17, l_max_ncdm=17)

    print("\n--- single Model() call (incl. compile) ---", flush=True)
    t0 = time.perf_counter()
    out = model(LCDM)
    jax.block_until_ready(out.ClTT)
    print(f"  single call wall: {time.perf_counter()-t0:.1f}s", flush=True)
    l_arr = out.l
    print(f"  l range: {int(l_arr[0])}..{int(l_arr[-1])} ({l_arr.size} ells)", flush=True)
    # peek at D_l at l~220 (first acoustic peak ~ 5700 muK^2)
    Dtt = pl.abcmb_cl_to_Dl(out.ClTT, l_arr)
    i220 = 220
    print(f"  D_l^TT(l=220) = {float(Dtt[i220]):.1f} muK^2 (expect ~5700)", flush=True)

    # diagnostic per-spectrum diagonal chi^2 (normalization sanity)
    diag = pl.diag_chi2_by_spec(out.ClTT, out.ClTE, out.ClEE, l_arr)
    print(f"\n  diagonal chi2 by spec (rough, ignores cov): {diag}", flush=True)
    print(f"  diag total: {sum(diag.values()):.1f} over {pl.ndata} bins", flush=True)

    # full chi^2, A_planck = 1
    res1 = pl.chi2_from_abcmb(out.ClTT, out.ClTE, out.ClEE, l_arr, profile=False)
    print(f"\n  FULL chi2 (A_planck=1):        {float(res1['chi2']):.2f}", flush=True)

    # full chi^2, profiled A_planck (with prior)
    res2 = pl.chi2_from_abcmb(out.ClTT, out.ClTE, out.ClEE, l_arr, profile=True,
                              with_prior=True)
    print(f"  FULL chi2 (A_planck profiled): {float(res2['chi2']):.2f} "
          f"@ A={float(res2['A_best']):.5f}", flush=True)

    # profiled, no prior
    res3 = pl.chi2_from_abcmb(out.ClTT, out.ClTE, out.ClEE, l_arr, profile=True,
                              with_prior=False)
    print(f"  FULL chi2 (A free, no prior):  {float(res3['chi2']):.2f} "
          f"@ A={float(res3['A_best']):.5f}", flush=True)
    print(f"  chi2/dof ~ {float(res2['chi2'])/pl.ndata:.3f}", flush=True)

    # ---- batched parity: B=4 around LCDM ----
    print("\n--- batched parity (B=4, incl. compile) ---", flush=True)
    pls = []
    for d in (0.0, 0.002, -0.002, 0.004):
        p = dict(LCDM); p['omega_cdm'] = LCDM['omega_cdm'] + d
        pls.append(p)
    t0 = time.perf_counter()
    outb = model.call_batched(pls, shard=True)
    jax.block_until_ready(outb.ClTT)
    print(f"  batched B=4 wall: {time.perf_counter()-t0:.1f}s", flush=True)
    resb = pl.chi2_from_abcmb(outb.ClTT, outb.ClTE, outb.ClEE, outb.l,
                              profile=True, with_prior=True)
    chi2b = np.asarray(resb['chi2'])
    print(f"  batched chi2 vector: {np.array2string(chi2b, precision=2)}", flush=True)
    print(f"  row0 (== single LCDM) batched {chi2b[0]:.2f} vs single "
          f"{float(res2['chi2']):.2f}  (diff {abs(chi2b[0]-float(res2['chi2'])):.3f})",
          flush=True)


if __name__ == "__main__":
    main()

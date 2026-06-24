"""save_clsref.py — run ABCMB ONCE over a small spread of cosmologies and save the
clik (6,lmax+1) theory blocks, so the inner-nuisance-profile optimizer can be tuned
OFFLINE (clipy only, ~2 min/job) instead of paying the 165 s ABCMB compile each time.

Saves scan/results/clsref_lcdm.npz: cls2d (B,6,Lcol), params (list), labels.
The spread = LCDM best fit + +/-2 sigma perturbations of h, omega_cdm, ln10As, tau,
so the optimizer is exercised where the bound-hitting nuisances (xi_sz_cib, ksz_norm)
behave differently across the POI grid extent.
"""
import os, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

As = float(np.exp(3.044) / 1e10)
BASE = {
    'h': 0.6736, 'omega_cdm': 0.1200, 'omega_b': 0.02237,
    'A_s': As, 'n_s': 0.9649,
    'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
# (name, +/- delta) perturbations (2 sigma-ish)
PERTS = [
    ("base", None, 0.0),
    ("h", "h", +0.011), ("h", "h", -0.011),
    ("omega_cdm", "omega_cdm", +0.0024), ("omega_cdm", "omega_cdm", -0.0024),
    ("ln10As", "A_s", +0.028),   # handled specially below (ln10As -> A_s)
    ("tau", "tau_reion", +0.015), ("tau", "tau_reion", -0.015),
]


def main():
    print(f"jax devices: {jax.devices()}", flush=True)
    from abcmb.main import Model
    from scan.plik_full import PlikFull
    pl = PlikFull()
    model = Model(user_species=None, output_Cl=True, l_max=2508, lensing=True,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10,
                  l_max_ur=17, l_max_ncdm=17,
                  rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7,
                  rtol_small_k_PE=1e-5, max_steps_PE=16384)
    batch = []; labels = []
    for label, key, delta in PERTS:
        p = dict(BASE)
        if key == "A_s":
            p["A_s"] = float(np.exp(3.044 + delta) / 1e10)
        elif key is not None:
            p[key] = p[key] + delta
        batch.append(p); labels.append(f"{label}{'' if delta==0 else ('%+g'%delta)}")
    print(f"running call_batched over B={len(batch)} cosmologies ...", flush=True)
    t0 = time.perf_counter()
    out = model.call_batched(batch, shard=False)
    jax.block_until_ready(out.ClTT)
    print(f"  call_batched wall (incl compile): {time.perf_counter()-t0:.1f}s", flush=True)
    cls2d = np.asarray(pl.abcmb_cls_to_clik(out.ClTT, out.ClTE, out.ClEE, out.l))  # (B,6,Lcol)
    print(f"  cls2d shape {cls2d.shape}", flush=True)
    outpath = os.path.join(os.path.dirname(__file__), "results", "clsref_lcdm.npz")
    np.savez(outpath, cls2d=cls2d, labels=np.array(labels), lmax=pl.lmax)
    print(f"  saved {outpath}", flush=True)


if __name__ == "__main__":
    import pandas as pd
    pd.options.future.infer_string = False
    main()

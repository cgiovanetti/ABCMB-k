"""
Correctness + quick-perf validation for the kappa-tabulation keystone.

Checks:
  1. single model() runs, no NaN, sane Cls.
  2. call_batched at B=4 runs end-to-end (BG now stacks; spectrum vmaps).
  3. batched vs single Cl/Pk agreement (expect ~1e-5, same as pre-change
     diffrax step-controller noise envelope).
  4. quick per-stage timing of call_batched at B=4 post-warmup so we can
     see the spectrum-loop win immediately.

Run:
  srun ... python bench/validate_keystone.py
"""
import os, sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from abcmb.main import Model

ELLMAX = 800
RNG_SEED = 0

FIDUCIAL = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
PARAM_BOXES = {
    'h':         (0.65,    0.70),
    'omega_cdm': (0.115,   0.125),
    'omega_b':   (0.0220,  0.0230),
    'A_s':       (1.95e-9, 2.25e-9),
    'n_s':       (0.950,   0.980),
}


def make_perturbed_params(n, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        p = dict(FIDUCIAL)
        for k, (lo, hi) in PARAM_BOXES.items():
            p[k] = float(rng.uniform(lo, hi))
        out.append(p)
    return out


def block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


def maxrel(a, b):
    a = np.asarray(a); b = np.asarray(b)
    denom = np.maximum(np.abs(a), np.abs(b))
    m = denom > 0
    return float(np.max(np.abs(a[m] - b[m]) / denom[m]))


def scaled_err(a, b):
    """Max absolute error normalized by the spectrum's own peak amplitude.

    The physically meaningful error for a sign-changing spectrum (TE crosses
    zero ~6x over l=2-800): pointwise *relative* error explodes at zero
    crossings even when the absolute agreement is excellent. For a chi-square
    likelihood the error that matters is |Δ| relative to the spectrum scale.
    Returns (max_abs_err / max|spectrum|, argmax l-index, pointwise_relmax).
    """
    a = np.asarray(a); b = np.asarray(b)
    absd = np.abs(a - b)
    scale = np.max(np.abs(b))
    # also locate where the pointwise relative metric peaks (expect a zero crossing)
    denom = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(denom > 0, absd / denom, 0.0)
    idx = np.unravel_index(np.argmax(rel), rel.shape)
    li = idx[-1]
    return (float(np.max(absd) / scale), int(li), float(np.max(rel)),
            float(np.abs(b).reshape(-1, b.shape[-1])[idx[0] if b.ndim > 1 else 0, li])
            if b.ndim >= 1 else 0.0)


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )

    B = 4
    pl = make_perturbed_params(B)

    # --- 1. single model() on each param (reference) ---
    print("\n[1] single model() per param (reference)...", flush=True)
    singles = []
    for i, p in enumerate(pl):
        out = model(p)
        block(out.ClTT)
        nan = bool(np.any(~np.isfinite(np.asarray(out.ClTT))))
        singles.append(out)
        print(f"    param {i}: ClTT[100]={float(out.ClTT[100]):.6e}  "
              f"NaN={nan}", flush=True)

    # --- 2/3. call_batched and compare ---
    print("\n[2] call_batched(B=4) (warm)...", flush=True)
    t0 = time.perf_counter()
    outb = model.call_batched(pl)
    block(outb.ClTT)
    print(f"    warm/compile: {time.perf_counter()-t0:.1f}s", flush=True)

    print("[3] batched vs single agreement:", flush=True)
    sTT = jnp.stack([s.ClTT for s in singles])
    sTE = jnp.stack([s.ClTE for s in singles])
    sEE = jnp.stack([s.ClEE for s in singles])
    sPk = jnp.stack([s.Pk   for s in singles])
    tt = maxrel(outb.ClTT, sTT)
    te = maxrel(outb.ClTE, sTE)
    ee = maxrel(outb.ClEE, sEE)
    pk = maxrel(outb.Pk,   sPk)
    print(f"    pointwise max_rel:  TT={tt:.3e}  TE={te:.3e}  "
          f"EE={ee:.3e}  Pk={pk:.3e}", flush=True)
    # peak-normalized (physically meaningful) error
    for name, ob, sb in [("TT", outb.ClTT, sTT), ("TE", outb.ClTE, sTE),
                         ("EE", outb.ClEE, sEE), ("Pk", outb.Pk, sPk)]:
        s, li, rmax, val = scaled_err(ob, sb)
        print(f"    {name}: |Δ|/peak = {s:.3e}   "
              f"(pointwise relmax {rmax:.2e} at l-idx {li}, "
              f"|spectrum| there = {val:.2e})", flush=True)
    # gate on peak-normalized error (the chi-square-relevant measure)
    sc = lambda ob, sb: scaled_err(ob, sb)[0]
    scs = [sc(outb.ClTT, sTT), sc(outb.ClTE, sTE),
           sc(outb.ClEE, sEE), sc(outb.Pk, sPk)]
    ok = max(scs) < 1e-3
    print(f"    -> {'PASS' if ok else 'FAIL'} "
          f"(peak-normalized |Δ|/peak < 1e-3 on all)", flush=True)

    # --- 4. quick per-stage timing at B=4 (post-warmup) ---
    print("\n[4] call_batched(B=4) post-compile timing:", flush=True)
    t0 = time.perf_counter()
    outb = model.call_batched(pl)
    block(outb.ClTT)
    dt = time.perf_counter() - t0
    print(f"    total={dt:.2f}s  per_param={dt/B:.3f}s", flush=True)


if __name__ == "__main__":
    main()

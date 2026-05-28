"""
End-to-end smoke for Model.call_batched.

  - Build Model.
  - Single: model(FIDUCIAL) -> Output.
  - Batched B=2: model.call_batched([FIDUCIAL, alt_params]) -> BatchedOutput.
  - Compare BatchedOutput[0] to single Output (FIDUCIAL).

Target tolerance ~1e-4 on Cls/Pk (accounting for the 1e-5 PT drift we
already observed propagating through spectrum integration).

Run via srun on a GPU allocation with conda + PYTHONPATH set.
"""

import sys
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from abcmb.main import Model


ELLMAX = 800
FIDUCIAL = {
    'h': 0.6736, 'omega_cdm': 0.120, 'omega_b': 0.02237,
    'A_s': 2.1e-9, 'n_s': 0.9649, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5,
    'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}
ALT = {**FIDUCIAL, 'h': 0.68, 'omega_cdm': 0.118, 'A_s': 2.05e-9}


def report(label, a, b, tol):
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        print(f"  {label}: SHAPE MISMATCH {a_arr.shape} vs {b_arr.shape}",
              flush=True)
        return False
    diff = np.abs(a_arr - b_arr)
    ref = np.maximum(np.abs(b_arr), 1e-300)
    rel = diff / ref
    rel_max = float(rel.max())
    abs_max = float(diff.max())
    status = "OK" if rel_max <= tol else "FAIL"
    print(f"  {label}: max_abs={abs_max:.2e}  max_rel={rel_max:.2e}  [{status}]",
          flush=True)
    return rel_max <= tol


def main():
    print(f"jax.devices(): {jax.devices()}", flush=True)
    print(f"abcmb from: {__import__('abcmb').__file__}", flush=True)

    model = Model(
        user_species=None, output_Cl=True, l_max=ELLMAX, lensing=False,
        output_Pk=True, output_k_max=0.5,
        l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17,
    )

    print("\n[single] model(FIDUCIAL)...", flush=True)
    t0 = time.perf_counter()
    out_single = model(FIDUCIAL)
    jax.block_until_ready(out_single.ClTT)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)

    print("\n[batched B=2] model.call_batched([FIDUCIAL, ALT])...",
          flush=True)
    t0 = time.perf_counter()
    out_batched = model.call_batched([FIDUCIAL, ALT])
    jax.block_until_ready(out_batched.ClTT)
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    print(f"  ClTT batched shape: {out_batched.ClTT.shape}", flush=True)
    print(f"  Pk   batched shape: {out_batched.Pk.shape}", flush=True)

    print("\n[compare] BatchedOutput[0] vs single Output:", flush=True)
    TOL = 1e-4
    passed = True
    passed &= report("ClTT", out_batched.ClTT[0], out_single.ClTT, TOL)
    passed &= report("ClTE", out_batched.ClTE[0], out_single.ClTE, TOL)
    passed &= report("ClEE", out_batched.ClEE[0], out_single.ClEE, TOL)
    passed &= report("Pk",   out_batched.Pk[0],   out_single.Pk,   TOL)
    passed &= report("l",    out_batched.l,       out_single.l,    1e-15)
    passed &= report("k",    out_batched.k,       out_single.k,    1e-15)

    print("\n" + "=" * 60, flush=True)
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

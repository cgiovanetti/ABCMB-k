"""noise_floor.py — quantify the chi^2 numerical noise floor of the production path.

The frequentist profile interpolates Delta-chi^2(POI) crossings; that is only
meaningful if the NUMERICAL noise in chi^2 (from the diffrax rtol~1e-4 solve and
the k-axis chunking, which makes a cosmology's trajectory non-bit-identical across
batch positions / chunk sizes) is << the Delta-chi^2 ~ 1 features being resolved.

This measures that floor directly. For each of a few cosmologies we evaluate the
EXACT SAME parameters many times, through the SAME chi2 path the profiler uses
(call_batched(shard=True) -> plik-lite profile_A + tau prior), under conditions
that expose numerical noise:

  A) IN-BATCH replication: put R identical copies in ONE call_batched. Any spread
     across the copies is pure batch-position / chunk noise (same HLO, same call).
  B) CHUNK-SIZE variation: evaluate the same cosmology with different per-call
     batch sizes (=> different chunk boundaries) and compare.
  C) BATCH vs SINGLE: compare the batched chi2 to the single-cosmology path
     (run_cosmology_abbr), the cleanest "reference".

Reports, per cosmology: chi2 mean and peak-to-peak / std spread for (A), and the
(B),(C) offsets. The headline number is the in-batch ptp: if it is <~0.01 the
profile intervals are trustworthy to <1% of sigma; ~0.1 would be disqualifying.

Run via srun (1 GPU ok), PYTHONPATH=$(pwd). Env: NF_LMAX(2508), NF_R(24).
"""
import os
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
from scan.plik_lite import PlikLite

LMAX = int(os.environ.get("NF_LMAX", 2508))
R = int(os.environ.get("NF_R", 24))            # in-batch replications
FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}
TAU_C, TAU_S = 0.0544, 0.0073

pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)

# a few representative cosmologies (Planck centre + a couple offsets along n_s)
COSMOS = {
    "center":   dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200, n_s=0.9649, ln10As=3.044, tau=0.0544),
    "ns_lo":    dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200, n_s=0.9565, ln10As=3.044, tau=0.0544),
    "ns_hi":    dict(h=0.6736, omega_b=0.02237, omega_cdm=0.1200, n_s=0.9733, ln10As=3.044, tau=0.0544),
}


def build(c):
    p = dict(FIXED)
    p['h'] = c['h']; p['omega_b'] = c['omega_b']; p['omega_cdm'] = c['omega_cdm']
    p['n_s'] = c['n_s']; p['A_s'] = float(np.exp(c['ln10As']) / 1e10)
    p['tau_reion'] = c['tau']
    return p


def chi2_batched(cosmos):
    out = model.call_batched([build(c) for c in cosmos], shard=True)
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    data = np.asarray(pl.profile_A(m0, with_prior=True)[0], dtype=float)
    tau = np.array([c['tau'] for c in cosmos])
    return data + ((tau - TAU_C) / TAU_S) ** 2


def chi2_single(c):
    out = model.run_cosmology_abbr(model.add_derived_parameters(build(c)))
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    data = float(pl.profile_A(m0[None], with_prior=True)[0][0])
    return data + ((c['tau'] - TAU_C) / TAU_S) ** 2


def stats(x):
    x = np.asarray(x, float)
    return x.mean(), np.ptp(x), x.std()


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  R={R}", flush=True)
    for name, c in COSMOS.items():
        # (A) R identical copies in ONE call_batched
        cA = chi2_batched([dict(c) for _ in range(R)])
        mA, pA, sA = stats(cA)
        print(f"\n[{name}] n_s={c['n_s']:.4f}", flush=True)
        print(f"  (A) in-batch R={R} identical: mean={mA:.4f} ptp={pA:.2e} std={sA:.2e}",
              flush=True)
        # (B) different batch size => different chunk boundaries
        for B2 in (7, 13, 33):
            cB = chi2_batched([dict(c) for _ in range(B2)])
            print(f"  (B) B={B2:2d}: mean={cB.mean():.4f}  (offset vs A = {cB.mean()-mA:+.2e})",
                  flush=True)
        # (C) single-cosmology reference path
        try:
            cS = chi2_single(dict(c))
            print(f"  (C) single-path: {cS:.4f}  (offset vs A-mean = {cS-mA:+.2e})", flush=True)
        except Exception as e:
            print(f"  (C) single-path failed: {e}", flush=True)

    print("\nHEADLINE: the (A) ptp is the per-evaluation chi^2 noise floor. "
          "<~0.01 => intervals trustworthy to <1% of sigma; ~0.1 => disqualifying.",
          flush=True)


if __name__ == "__main__":
    main()

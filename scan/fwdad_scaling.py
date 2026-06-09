"""fwdad_scaling.py — is the forward-AD gradient SERIAL or PARALLEL over tangents?

The first validation measured jacfwd(chi2) ~33x primal for a 6-input gradient.
At B=1 the A100 is far from saturated, so jacfwd SHOULD propagate all 6 tangents
together for ~ the cost of one jvp. ~33x means either (a) the tangents are
computed serially (primal recomputed per direction), or (b) each single jvp is
already very expensive (augmented-ODE step blow-up). This script distinguishes:

  t_primal                     : forward eval
  t_jvp[i]  (per direction)    : ONE jvp (same compiled fn, different tangent ->
                                 no recompile) -> shows cheap (A_s/n_s: spectrum
                                 only) vs expensive (h/ob/ocdm: through the ODE)
                                 directions
  t_serial  = sum_i t_jvp[i]   : explicit 6 serial jvps
  t_jacfwd6 = jax.jacfwd       : JAX's vmapped-tangents Jacobian
  t_jacfwd3                    : only the 3 ODE params (h, ob, ocdm)

Verdicts: jacfwd6 ~= max(t_jvp) -> tangents PARALLEL (cheap gradient).
          jacfwd6 ~= t_serial    -> tangents SERIAL (the 33x is real overhead).
          t_jvp1  ~= t_primal    -> each jvp cheap; the 33x was pure serialization.
          t_jvp1 >> t_primal     -> augmented ODE itself is expensive.

Run via srun, PYTHONPATH=$(pwd). Env: FWD_LMAX (2508).
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

LMAX = int(os.environ.get("FWD_LMAX", 2508))
FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5}
PNAMES = ['h', 'omega_b', 'omega_cdm', 'n_s', 'ln10As', 'tau_reion']
theta0 = jnp.array([0.6736, 0.02237, 0.1200, 0.9649, 3.044, 0.0544])
pl = PlikLite()
model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
              output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17, l_max_ncdm=17)


def build(theta):
    h, ob, oc, ns, ln10As, tau = theta
    p = dict(FIXED); p['h'] = h; p['omega_b'] = ob; p['omega_cdm'] = oc
    p['n_s'] = ns; p['A_s'] = jnp.exp(ln10As) / 1e10; p['tau_reion'] = tau
    return p


def chi2(theta):
    p = build(theta)
    out = model.run_cosmology_abbr(model.add_derived_parameters(p))
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    return pl.chi2(m0, A_planck=1.0) + ((theta[5] - 0.0544) / 0.0073) ** 2


def timeit(f, *a, reps=2):
    f(*a); r = []
    for _ in range(reps):
        t = time.perf_counter(); o = f(*a); jax.block_until_ready(o)
        r.append(time.perf_counter() - t)
    return float(np.median(r)), o


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}", flush=True)
    jchi2 = jax.jit(chi2)
    t_primal, _ = timeit(jchi2, theta0)
    print(f"\nprimal: {t_primal:.2f}s", flush=True)

    # single jvp, reused across directions (one compile, many tangents)
    jvp1 = jax.jit(lambda th, v: jax.jvp(chi2, (th,), (v,))[1])
    print("\n--- per-direction single jvp (same compiled fn) ---", flush=True)
    tdir = {}
    for i in range(6):
        v = jnp.zeros(6).at[i].set(1.0)
        t, _ = timeit(jvp1, theta0, v)
        tdir[i] = t
        print(f"  jvp d/d{PNAMES[i]:10s}: {t:.2f}s  ({t/t_primal:.1f}x primal)", flush=True)
    t_serial = sum(tdir.values())
    print(f"  sum of 6 serial jvps = {t_serial:.2f}s ({t_serial/t_primal:.1f}x)", flush=True)

    # jacfwd over all 6 (JAX vmaps the tangents)
    jf6 = jax.jit(jax.jacfwd(chi2))
    t_jf6, _ = timeit(jf6, theta0)
    print(f"\njacfwd(6 inputs): {t_jf6:.2f}s ({t_jf6/t_primal:.1f}x primal)", flush=True)

    # jacfwd over only the 3 ODE params (h, omega_b, omega_cdm)
    def chi2_3(p3):
        th = theta0.at[0].set(p3[0]).at[1].set(p3[1]).at[2].set(p3[2])
        return chi2(th)
    jf3 = jax.jit(jax.jacfwd(chi2_3))
    t_jf3, _ = timeit(jf3, theta0[:3])
    print(f"jacfwd(3 ODE params): {t_jf3:.2f}s ({t_jf3/t_primal:.1f}x primal)", flush=True)

    print("\n=== VERDICT ===", flush=True)
    print(f"  max single jvp        = {max(tdir.values())/t_primal:.1f}x primal", flush=True)
    print(f"  jacfwd6 / max_jvp     = {t_jf6/max(tdir.values()):.2f}  "
          f"(~1 => tangents PARALLEL; ~6 => SERIAL)", flush=True)
    print(f"  jacfwd6 / serial_sum  = {t_jf6/t_serial:.2f}  "
          f"(<1 => jacfwd beats serial loop)", flush=True)


if __name__ == "__main__":
    main()

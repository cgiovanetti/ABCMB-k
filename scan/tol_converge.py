"""tol_converge.py — find the LOOSEST rtol_large_k_PE that still gives a
converged profile (rtol is the runtime bottleneck, so push it as large as possible).

The bug-vs-tolerance question is settled (round (d): |D| shrinks ~10x/decade =>
solver tolerance, not a bug). Here we map the CONVERGENCE THRESHOLD directly and
self-referentially: for each rtol, the BATCHED chi2 profile along n_s (at fixed
Planck-centre nuisances, a cheap proxy -- the rtol error is smooth in the params)
is fit to a parabola -> (n_s_min, sigma). The rtol-induced SHIFT of n_s_min vs the
TIGHTEST rtol in the run is the numerical bias; common parabola-fit error cancels
in the shift. The loosest rtol whose shift is << the ABCMB-vs-CLASS ~0.2%-of-sigma
theory floor (and below the 0.1%-sigma working target) is the answer.

Per rtol it also reports the warm solve time so we see the runtime payoff of
loosening. Saves results/tol_sweep_<rtol>.npz so several debug jobs (one or a few
rtol each) can be aggregated. Run via srun, PYTHONPATH=$(pwd).
Env: TC_RTOLS (csv), TC_LMAX(2508), TC_B(7 n_s points), TC_MAXSTEPS(16384).
"""
import os, time
_SCRATCH = os.environ.get("SCRATCH", "/pscratch/sd/c/carag")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(_SCRATCH, ".jax_cache_abcmb"))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
from scan.plik_lite import PlikLite

LMAX = int(os.environ.get("TC_LMAX", 2508))
RTOLS = [float(x) for x in os.environ.get("TC_RTOLS", "1e-4,5e-5,3e-5,2e-5,1e-5").split(",")]
MAXSTEPS = int(os.environ.get("TC_MAXSTEPS", 16384))
NPTS = int(os.environ.get("TC_NPTS", 9))
FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4, 'N_nu_massive': 1,
         'T_nu_massive': 0.71611, 'm_nu_massive': 0.06, 'Delta_z_reion': 0.5,
         'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5, 'exp_reion': 1.5, 'tau_reion': 0.0544}
NS_C, NS_S = 0.9649, 0.0042
# NARROW slice centred on the fixed-nuisance min (~0.962), +/-0.004 (~2 sigma of
# the fixed-nuisance parabola, sigma_p~0.0021) so the deg-2 vertex is clean -- a
# wide +/-3 sigma_planck slice reaches Delta-chi2~36 and the vertex fit jitters.
NS = 0.962 + np.linspace(-0.004, 0.004, NPTS)
A_S_REF = float(np.exp(3.044) / 1e10)
pl = PlikLite()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def batched_chi2(model):
    batch = [dict(FIXED, h=0.6736, omega_b=0.02237, omega_cdm=0.1200,
                  n_s=float(n), A_s=A_S_REF) for n in NS]
    out = model.call_batched(batch, shard=True)
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    return np.asarray(pl.profile_A(m0, with_prior=True)[0], dtype=float)


def fit_min_sigma(ns, chi2):
    """quadratic vertex + sigma=1/sqrt(2a) from chi2 ~ a(n_s-n0)^2."""
    a, b, c = np.polyfit(ns, chi2, 2)
    n0 = -b / (2 * a)
    sig = np.nan if a <= 0 else 1.0 / np.sqrt(2.0 * a)
    return n0, sig


def main():
    print(f"devices={jax.devices()} lmax={LMAX} rtols={RTOLS} NS(+/-3s,{NPTS}pts)", flush=True)
    rows = []
    for R in RTOLS:
        model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
                      output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17,
                      l_max_ncdm=17, rtol_large_k_PE=R, atol_large_k_PE=R * 1e-2,
                      rtol_small_k_PE=min(1e-5, R), atol_small_k_PE=1e-10,
                      max_steps_PE=MAXSTEPS)
        c1 = batched_chi2(model)                         # compile + solve
        t = time.perf_counter(); c2 = batched_chi2(model); dt = time.perf_counter() - t
        n0, sig = fit_min_sigma(NS, c2)
        rows.append((R, n0, sig, dt))
        np.savez(os.path.join(OUT, f"tol_sweep_{R:.0e}.npz"),
                 rtol=R, ns=NS, chi2=c2, n0=n0, sigma=sig, warm_s=dt)
        print(f"  rtol={R:.1e}: n_s_min={n0:.6f}  sigma={sig:.6f}  warm={dt:.0f}s", flush=True)
        del model

    # convergence table referenced to the TIGHTEST rtol in this run
    rows.sort(key=lambda r: r[0])                        # ascending rtol
    ref_n0 = rows[0][1]                                  # tightest
    print("\n=== CONVERGENCE (shift of n_s_min vs tightest rtol here) ===", flush=True)
    print(f"  reference rtol={rows[0][0]:.1e}: n_s_min={ref_n0:.6f}", flush=True)
    for R, n0, sig, dt in rows:
        shift_sig = (n0 - ref_n0) / NS_S
        print(f"  rtol={R:.1e}: dn_s_min={n0-ref_n0:+.2e} = {shift_sig*100:+.3f}% of sigma "
              f"| sigma={sig:.6f} | warm={dt:.0f}s", flush=True)
    print("  loosest rtol with |shift| << 0.2% (theory floor) / < 0.1% (target) wins.",
          flush=True)


if __name__ == "__main__":
    main()

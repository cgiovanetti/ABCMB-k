"""tol_converge.py — solver-tolerance convergence study for the frequentist profile.

The batched (k-chunked) and single-call paths differ by D = chi2_batched -
chi2_single ~ 0.1, and D VARIES ~0.027 over 2sigma in n_s -> a ~0.7%-of-sigma
systematic on the profile (scan/noise_floor.py). Is that a BUG or just the finite
PE solver tolerance (rtol_large_k_PE=1e-4, the loosest, sets the high-ell modes)?
This sweeps rtol_large_k_PE in {1e-4, 1e-5, 1e-6} (atol_large_k_PE scaled x1e-2)
and, at FIXED Planck-centre nuisances, measures along a 5-point n_s line:

  * chi2_batched(n_s)            (one call_batched, B=5)
  * chi2_single(n_s)            (run_cosmology_abbr) at the 3 key points
  * D = batched - single, and its VARIATION across n_s (the profile-biasing piece)
  * sigma from a parabola fit to the batched Delta-chi2(n_s)  (interval proxy)
  * wall time + GPU mem per rtol (to size the production cost)

VERDICTS:
  * D and its variation SHRINK ~10x per decade of rtol => tolerance-driven (NOT a
    bug); pick the rtol where the numerical piece is << the ABCMB-vs-CLASS ~0.2%
    theory floor. * D PLATEAUS as rtol tightens => a real chunking bug to fix.
  * sigma_batched(rtol) converging => the interval is tolerance-converged there.

Run via srun (1 GPU ok), PYTHONPATH=$(pwd). Env: TC_LMAX(2508), TC_RTOLS.
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
RTOLS = [float(x) for x in os.environ.get("TC_RTOLS", "1e-4,1e-5,1e-6").split(",")]
MAXSTEPS = int(os.environ.get("TC_MAXSTEPS", 16384))   # tighter rtol => more steps
FIXED = {'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
         'N_nu_massive': 1, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
         'Delta_z_reion': 0.5, 'z_reion_He': 3.5, 'Delta_z_reion_He': 0.5,
         'exp_reion': 1.5, 'tau_reion': 0.0544}
NS_C, NS_S = 0.9649, 0.0042
NS = NS_C + NS_S * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])   # 5-point n_s line
SINGLE_IDX = [0, 2, 4]                                      # -2s, centre, +2s
A_S_REF = float(np.exp(3.044) / 1e10)
pl = PlikLite()


def base():
    return dict(FIXED, h=0.6736, omega_b=0.02237, omega_cdm=0.1200, A_s=A_S_REF)


def m0_to_chi2(out):
    m0 = pl.bin_model(pl.abcmb_cl_to_Dl(out.ClTT, out.l),
                      pl.abcmb_cl_to_Dl(out.ClTE, out.l),
                      pl.abcmb_cl_to_Dl(out.ClEE, out.l))
    return np.asarray(pl.profile_A(m0, with_prior=True)[0], dtype=float)


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def gpu_gb(model):
    try:
        return jax.devices('gpu')[0].memory_stats().get('peak_bytes_in_use', 0) / 1e9
    except Exception:
        return float('nan')


def sigma_parab(ns, chi2):
    a = np.polyfit(ns, chi2 - chi2.min(), 2)[0]
    return np.nan if a <= 0 else 1.0 / np.sqrt(2.0 * a)


def main():
    print(f"devices: {jax.devices()}  lmax={LMAX}  rtols={RTOLS}", flush=True)
    print(f"n_s line: {NS}", flush=True)
    res = {}
    for R in RTOLS:
        aR = R * 1e-2
        model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
                      output_Pk=False, l_max_g=12, l_max_pol_g=10, l_max_ur=17,
                      l_max_ncdm=17, rtol_large_k_PE=R, atol_large_k_PE=aR,
                      rtol_small_k_PE=min(1e-5, R), atol_small_k_PE=1e-10,
                      max_steps_PE=MAXSTEPS)
        # batched (B=5)
        t = time.perf_counter()
        outb = model.call_batched([dict(base(), n_s=float(n)) for n in NS], shard=True)
        cb = m0_to_chi2(outb); tb = time.perf_counter() - t
        # single at the 3 key points
        cs = np.full(len(NS), np.nan); t = time.perf_counter()
        for i in SINGLE_IDX:
            outs = model.run_cosmology_abbr(model.add_derived_parameters(
                dict(base(), n_s=float(NS[i]))))
            m0s = pl.bin_model(pl.abcmb_cl_to_Dl(outs.ClTT, outs.l),
                               pl.abcmb_cl_to_Dl(outs.ClTE, outs.l),
                               pl.abcmb_cl_to_Dl(outs.ClEE, outs.l))
            cs[i] = float(pl.profile_A(m0s[None], with_prior=True)[0][0])
        ts = time.perf_counter() - t
        D = cs - cb
        sig = sigma_parab(NS, cb)
        res[R] = dict(cb=cb, cs=cs, D=D, sig=sig)
        print(f"\n=== rtol_large_k_PE={R:.0e} (atol={aR:.0e}) ===", flush=True)
        print(f"  batched chi2 : {np.array2string(cb, precision=4)}  ({tb:.0f}s B=5)", flush=True)
        print(f"  single  chi2 : {np.array2string(cs, precision=4)}  ({ts:.0f}s x{len(SINGLE_IDX)})", flush=True)
        Dv = D[SINGLE_IDX]
        print(f"  D=batched-single @[-2s,c,+2s]: {np.array2string(-Dv, precision=4)} "
              f"(min->max VARIATION = {np.nanmax(Dv)-np.nanmin(Dv):+.4f})", flush=True)
        print(f"  sigma_batched(parabola) = {sig:.6f}  (Planck n_s sigma={NS_S})", flush=True)
        print(f"  GPU peak ~{gpu_gb(model):.1f} GB, host RSS {rss_gb():.1f} GB", flush=True)
        del model

    print("\n=== CONVERGENCE ===", flush=True)
    Rs = sorted(res, reverse=True)
    for R in Rs:
        Dv = res[R]['D'][SINGLE_IDX]
        var = np.nanmax(Dv) - np.nanmin(Dv)
        print(f"  rtol={R:.0e}: |D|(centre)={abs(res[R]['D'][2]):.4f}  "
              f"D-variation(2s)={var:.4f}  sigma_b={res[R]['sig']:.6f}", flush=True)
    print("  D & variation shrinking ~10x/decade => tolerance (not a bug); "
          "sigma_b stabilizing => interval tolerance-converged.", flush=True)
    # profile-bias proxy: variation of D over 1 sigma ~ var/2; interval shift ~ that/(2/sigma)
    for R in Rs:
        Dv = res[R]['D'][SINGLE_IDX]; var = np.nanmax(Dv) - np.nanmin(Dv)
        bias_frac = (var / 2.0) / 2.0   # ~ (dD over 1 sigma)/slope(=2/sigma), in units of sigma
        print(f"  rtol={R:.0e}: est central-value bias ~ {bias_frac*100:.2f}% of sigma", flush=True)


if __name__ == "__main__":
    main()

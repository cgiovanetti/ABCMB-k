"""profile_demo.py — end-to-end frequentist profile likelihood demo.

Demonstrates the GPU-batched architecture doing a real frequentist profile
against Planck plik-lite:

  * Lay a 2D grid over (n_s, omega_cdm), other LCDM params fixed (h, omega_b,
    tau), reference A_s.
  * Evaluate ALL grid cosmologies' lensed Cls via Model.call_batched in fixed,
    sharded, padded batches (the embarrassingly-parallel scan kernel).
  * For each cosmology, profile the overall amplitude (A_s * calibration)
    analytically -> chi2_min(n_s, omega_cdm).
  * Profile each parameter out (min over the other) -> 1D profile curves; report
    the minimum and the Delta chi2 = 1 (1 sigma) interval.

This is the smallest scan that exercises the full machinery; the cost model for
the full 4D LCDM profile is in the feasibility writeup.

Env:
  PROF_NS   "0.93,1.00,25"      n_s   lo,hi,N
  PROF_OCDM "0.112,0.128,25"    omega_cdm lo,hi,N
  PROF_B    "64"                per-call batch (sharded over visible GPUs)
  PROF_LMAX "2508"
  PROF_OUT  "scan/profile_demo.npz"
"""
import os, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.plik_lite import PlikLite

As_ref = float(np.exp(3.044) / 1e10)
BASE = {
    'h': 0.6736, 'omega_cdm': 0.1200, 'omega_b': 0.02237, 'A_s': As_ref,
    'n_s': 0.9649, 'Neff': 3.044, 'YHe': 0.2454, 'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0, 'T_nu_massive': 0.71611, 'm_nu_massive': 0.06,
    'tau_reion': 0.0544, 'Delta_z_reion': 0.5, 'z_reion_He': 3.5,
    'Delta_z_reion_He': 0.5, 'exp_reion': 1.5,
}


def parse3(s, cast=float):
    a, b, n = s.split(",")
    return float(a), float(b), int(n)


def interval_1sigma(x, chi2):
    """Delta chi2 = 1 interval around the min via linear interpolation."""
    chi2 = np.asarray(chi2); x = np.asarray(x)
    imin = int(np.argmin(chi2)); target = chi2[imin] + 1.0
    def cross(lo_to_hi):
        idxs = range(imin, len(x) - 1) if lo_to_hi else range(imin, 0, -1)
        for i in idxs:
            j = i + 1 if lo_to_hi else i - 1
            if (chi2[i] - target) * (chi2[j] - target) <= 0:
                f = (target - chi2[i]) / (chi2[j] - chi2[i] + 1e-30)
                return x[i] + f * (x[j] - x[i])
        return np.nan
    return cross(False), x[imin], cross(True)


def main():
    NS = parse3(os.environ.get("PROF_NS", "0.93,1.00,25"))
    OC = parse3(os.environ.get("PROF_OCDM", "0.112,0.128,25"))
    B = int(os.environ.get("PROF_B", 64))
    LMAX = int(os.environ.get("PROF_LMAX", 2508))
    OUT = os.environ.get("PROF_OUT", "scan/profile_demo.npz")

    ns_ax = np.linspace(NS[0], NS[1], NS[2])
    oc_ax = np.linspace(OC[0], OC[1], OC[2])
    NS_N, OC_N = NS[2], OC[2]
    print(f"grid: n_s[{NS[0]},{NS[1]}]x{NS_N}  omega_cdm[{OC[0]},{OC[1]}]x{OC_N}"
          f"  = {NS_N*OC_N} cosmologies, B={B}, lmax={LMAX}", flush=True)

    # flatten grid (row-major: ns outer, oc inner)
    grid = []
    for ns in ns_ax:
        for oc in oc_ax:
            p = dict(BASE); p['n_s'] = float(ns); p['omega_cdm'] = float(oc)
            grid.append(p)
    Ntot = len(grid)

    pl = PlikLite()
    model = Model(user_species=None, output_Cl=True, l_max=LMAX, lensing=True,
                  output_Pk=False, l_max_g=12, l_max_pol_g=10,
                  l_max_ur=17, l_max_ncdm=17)

    chi2_flat = np.empty(Ntot)
    alpha_flat = np.empty(Ntot)
    t0 = time.perf_counter()
    for b0 in range(0, Ntot, B):
        batch = grid[b0:b0 + B]
        valid = len(batch)
        if valid < B:
            batch = batch + [batch[-1]] * (B - valid)
        tb = time.perf_counter()
        outb = model.call_batched(batch, shard=True)
        Dtt = pl.abcmb_cl_to_Dl(outb.ClTT, outb.l)
        Dte = pl.abcmb_cl_to_Dl(outb.ClTE, outb.l)
        Dee = pl.abcmb_cl_to_Dl(outb.ClEE, outb.l)
        m0 = pl.bin_model(Dtt, Dte, Dee)
        c2, alpha = pl.profile_amplitude(m0)
        c2 = np.asarray(jax.block_until_ready(c2))
        alpha = np.asarray(alpha)
        chi2_flat[b0:b0 + valid] = c2[:valid]
        alpha_flat[b0:b0 + valid] = alpha[:valid]
        dt = time.perf_counter() - tb
        print(f"  batch {b0//B}: {valid} cosmo in {dt:.1f}s "
              f"({dt/valid:.3f}s/param)  chi2 range [{c2[:valid].min():.1f},"
              f"{c2[:valid].max():.1f}]", flush=True)
    ttot = time.perf_counter() - t0
    print(f"grid done: {Ntot} cosmo in {ttot:.1f}s ({ttot/Ntot:.3f}s/param "
          f"incl. compile)", flush=True)

    chi2 = chi2_flat.reshape(NS_N, OC_N)   # [n_s, omega_cdm]
    # profiles
    prof_ns = chi2.min(axis=1)             # min over omega_cdm
    prof_oc = chi2.min(axis=0)             # min over n_s
    imin = np.unravel_index(np.argmin(chi2), chi2.shape)
    print(f"\nglobal min chi2 = {chi2.min():.2f} at "
          f"n_s={ns_ax[imin[0]]:.4f}, omega_cdm={oc_ax[imin[1]]:.5f}", flush=True)
    lo, mid, hi = interval_1sigma(ns_ax, prof_ns)
    print(f"profiled n_s     = {mid:.4f}  (1sigma [{lo:.4f}, {hi:.4f}], "
          f"+/-{(hi-lo)/2:.4f})", flush=True)
    lo, mid, hi = interval_1sigma(oc_ax, prof_oc)
    print(f"profiled omega_cdm = {mid:.5f}  (1sigma [{lo:.5f}, {hi:.5f}], "
          f"+/-{(hi-lo)/2:.5f})", flush=True)

    np.savez(OUT, chi2=chi2, ns_ax=ns_ax, oc_ax=oc_ax, prof_ns=prof_ns,
             prof_oc=prof_oc, alpha=alpha_flat.reshape(NS_N, OC_N),
             ttot=ttot, Ntot=Ntot)
    print(f"saved -> {OUT}", flush=True)

    # ---- plot (Agg, headless) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dchi2 = chi2 - chi2.min()
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        # 2D Delta chi2 surface with 1,2,3 sigma contours
        im = ax[0].contourf(oc_ax, ns_ax, dchi2, levels=30, cmap="viridis_r")
        cs = ax[0].contour(oc_ax, ns_ax, dchi2, levels=[2.30, 6.18, 11.83],
                           colors="w", linewidths=1.2)
        ax[0].clabel(cs, fmt={2.30: "1s", 6.18: "2s", 11.83: "3s"})
        ax[0].plot(oc_ax[imin[1]], ns_ax[imin[0]], "r*", ms=14)
        ax[0].set_xlabel(r"$\omega_{cdm}$"); ax[0].set_ylabel(r"$n_s$")
        ax[0].set_title(r"$\Delta\chi^2$ surface (plik-lite, A profiled)")
        fig.colorbar(im, ax=ax[0])
        # 1D profiles
        ax[1].plot(ns_ax, prof_ns - prof_ns.min(), "o-")
        for d, c in [(1, "g"), (4, "orange"), (9, "r")]:
            ax[1].axhline(d, ls="--", color=c, lw=0.8)
        ax[1].set_xlabel(r"$n_s$"); ax[1].set_ylabel(r"$\Delta\chi^2$ (profiled)")
        ax[1].set_ylim(0, 12); ax[1].set_title("profile of $n_s$")
        ax[2].plot(oc_ax, prof_oc - prof_oc.min(), "o-")
        for d, c in [(1, "g"), (4, "orange"), (9, "r")]:
            ax[2].axhline(d, ls="--", color=c, lw=0.8)
        ax[2].set_xlabel(r"$\omega_{cdm}$"); ax[2].set_ylabel(r"$\Delta\chi^2$ (profiled)")
        ax[2].set_ylim(0, 12); ax[2].set_title(r"profile of $\omega_{cdm}$")
        fig.tight_layout()
        png = OUT.replace(".npz", ".png")
        fig.savefig(png, dpi=110)
        print(f"saved -> {png}", flush=True)
    except Exception as e:
        print(f"plot skipped: {e}", flush=True)


if __name__ == "__main__":
    main()

"""act_lmax_check.py — does ACT DR6 lite (candl) need ABCMB run at the ACT paper's l_max_scalars 9500 /
delta_l_max 1800, or is l_max ~ 6325 (where the kept bins' windows end) enough?

  python hmc/diag/act_lmax_check.py run <cfg> [<cfg> ...]   # GPU: lensed Cls at 8 fixed cosmologies per cfg
  python hmc/diag/act_lmax_check.py compare                 # candl chi2 per cfg and cosmology (CPU is fine)

Value path (Model.call_batched, B sharded over the visible GPUs, 2 batches of 4). LCDM as in hmc/posterior.py
(three massless neutrinos, YHe from the BBN table, recomb lna grid, rtol 1e-5) with computed j_l (the tables stop
at l = 5000). Outputs hmc/results/act_lmax/<cfg>.npz.
"""
import os, sys, time
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "act_lmax")
COMMON = dict(output_Cl=True, lensing=True, output_Pk=False, l_max_ur=17, l_max_ncdm=17,
              rtol_large_k_PE=1e-5, atol_large_k_PE=1e-7, rtol_small_k_PE=1e-5, max_steps_PE=16384,
              lna_grid_mode="recomb", bbn_type="table", use_bessel_tables=False)
HF = dict(nonlinear="halofit")
CFGS = {
    "ref9500":      dict(l_max=9500, delta_l_max=1800, l_max_g=40, l_max_pol_g=60, **HF),  # the ACT paper's CLASS block
    "c6325":        dict(l_max=6325, delta_l_max=1800, l_max_g=40, l_max_pol_g=60, **HF),
    "c6325_g12":    dict(l_max=6325, delta_l_max=1800, l_max_g=12, l_max_pol_g=10, **HF),
    "c6325_dl3162": dict(l_max=6325, delta_l_max=3162, l_max_g=40, l_max_pol_g=60, **HF),
    "c6325_dl1000": dict(l_max=6325, delta_l_max=1000, l_max_g=40, l_max_pol_g=60, **HF),
    "c6325_dl2500": dict(l_max=6325, delta_l_max=2500, l_max_g=40, l_max_pol_g=60, **HF),
    "c6325_dl5000": dict(l_max=6325, delta_l_max=5000, l_max_g=40, l_max_pol_g=60, **HF),   # l_top = ref9500's
    "c6325_dl6500": dict(l_max=6325, delta_l_max=6500, l_max_g=40, l_max_pol_g=60, **HF),   # beyond ref9500
    "c7300":        dict(l_max=7300, delta_l_max=1800, l_max_g=40, l_max_pol_g=60, **HF),
    "c6000":        dict(l_max=6000, delta_l_max=1800, l_max_g=40, l_max_pol_g=60, **HF),
    "c6000_dl5325": dict(l_max=6000, delta_l_max=5325, l_max_g=40, l_max_pol_g=60, **HF),   # l_top = c6325_dl5000's
    "c6325_dl4000": dict(l_max=6325, delta_l_max=4000, l_max_g=40, l_max_pol_g=60, **HF),
    "c6325_lin":    dict(l_max=6325, delta_l_max=1800, l_max_g=40, l_max_pol_g=60),
}
NAMES = ("omega_b", "omega_cdm", "h", "ln10As", "n_s", "tau_reion")
START = np.array([0.022383, 0.12011, 0.6789, 3.0448, 0.96605, 0.0543])
SIG = np.array([0.00015, 0.0012, 0.0054, 0.014, 0.0042, 0.0073])
FIXED = {"TCMB0": 2.34865418e-4, "N_nu_massive": 0, "Neff": 3.044,
         "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5, "exp_reion": 1.5}


def cosmologies():
    z = np.random.default_rng(0).standard_normal((7, 6))
    return np.vstack([START, START + 2.0 * SIG * z])         # (8, 6): the start point + 7 at ~2 sigma


def run(cfg):
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    import jax
    jax.config.update("jax_enable_x64", True)
    from abcmb.main import Model
    assert "ABCMB-k" in __import__("abcmb").__file__
    th = cosmologies()
    plist = [dict(FIXED, omega_b=t[0], omega_cdm=t[1], h=t[2], A_s=np.exp(t[3]) / 1e10, n_s=t[4], tau_reion=t[5])
             for t in th]
    t0 = time.time()
    model = Model(**COMMON, **CFGS[cfg])
    print(f"[{cfg}] Model built {time.time() - t0:.0f}s, n_k transfer {len(model.SS.k_axis_transfer)}, "
          f"l_top {model.SS.l_top}", flush=True)
    cls, times = [], []
    for s in (0, 4):
        t0 = time.time()
        out = model.call_batched(plist[s:s + 4])
        c = np.stack([np.asarray(out.ClTT), np.asarray(out.ClTE), np.asarray(out.ClEE)], axis=1)   # (4, 3, n_l)
        times.append(time.time() - t0)
        cls.append(c)
        print(f"[{cfg}] batch {s // 4}: {times[-1]:.1f}s, finite {np.all(np.isfinite(c))}", flush=True)
    peaks = [d.memory_stats().get("peak_bytes_in_use", 0) / 2**30 for d in jax.devices()]
    print(f"[{cfg}] peak GB per device {' '.join(f'{p:.1f}' for p in peaks)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"{cfg}.npz"), theta=th, ells=np.asarray(out.l), cls=np.concatenate(cls),
             times=np.array(times), peaks=np.array(peaks))


def compare():
    import jax
    jax.config.update("jax_enable_x64", True)
    import candl, candl_data
    like = candl.Like(candl_data.ACT_DR6_TTTEEE)
    LT = int(like.ells[-1])                                    # theory input ell 2..8501
    T2 = 2.7255e6 ** 2
    res = {}
    for cfg in CFGS:
        f = os.path.join(OUT, f"{cfg}.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        ells, cls = d["ells"], d["cls"]
        chi2 = []
        for b in range(len(cls)):
            Dl = {}
            for i, s in enumerate(("TT", "TE", "EE")):
                full = np.zeros(LT + 1)
                m = ells <= LT
                full[ells[m]] = cls[b, i, m] * ells[m] * (ells[m] + 1) / (2 * np.pi) * T2
                Dl[s] = full[2:]
            chi2.append(float(like.chi_square({"Dl": Dl, "A_act": 1.0, "P_act": 1.0})))
        res[cfg] = (np.array(chi2), d["times"], d["peaks"], ells, cls)
    ref = res["ref9500"][0]
    print("chi2 (ACT DR6 lite, A_act = P_act = 1) per cosmology; 0 = START, 1-7 at ~2 sigma")
    print(f"{'cfg':>14} " + " ".join(f"{i:>9d}" for i in range(len(ref))) + "   max|d - d0|  warm s/4   peak GB")
    for cfg, (c, t, p, _, _) in res.items():
        dd = c - ref
        row = " ".join(f"{v:9.3f}" for v in (c if cfg == "ref9500" else dd))
        print(f"{cfg:>14} {row}   {np.max(np.abs(dd - dd[0])):10.4f}  {t[-1]:8.1f}  {max(p):7.1f}")
    print("(rows other than ref9500 are chi2 - chi2_ref; max|d - d0| = the part that is not a constant offset)")
    # spectra: max relative difference over the windows' range
    e0, c0 = res["ref9500"][3], res["ref9500"][4]
    for cfg, (_, _, _, e, c) in res.items():
        if cfg == "ref9500":
            continue
        for lo, hi in ((576, 3000), (3000, 5000), (5000, 6325)):
            m = (e >= lo) & (e <= hi); m0 = (e0 >= lo) & (e0 <= hi)
            r = [np.max(np.abs(c[:, i, m] / c0[:, i, m0] - 1)) if i != 1 else
                 np.max(np.abs(c[:, 1, m] - c0[:, 1, m0]) / np.sqrt(c0[:, 0, m0] * c0[:, 2, m0])) for i in range(3)]
            print(f"  {cfg:>14} l {lo}-{hi}: TT {r[0]:.2e}  TE/sqrt(TT EE) {r[1]:.2e}  EE {r[2]:.2e}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        for cfg in sys.argv[2:]:
            run(cfg)
    else:
        compare()

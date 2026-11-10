"""posterior_act.py — ACT DR6 LCDM posterior through ABCMB-k's batched engine, with its exact gradient by
forward-mode AD (scan/batched_grad.staged_cl_and_grad), for the lockstep HMC (hmc/run_act.py).

Data (`data`), as the ACT DR6 lite yamls define them (Neal_ACTDR6/DR6-ACT-lite/yamls/likelihoods):
  "act"   ACT DR6 TT/TE/EE CMB-only, candl's differentiable version (candl_data.ACT_DR6_TTTEEE; its chi_square,
          i.e. without candl's built-in tau / A_act priors) + Planck SRoll2 low-ell EE (calibration 1: cobaya's
          EE_sroll2 only takes A_planck when one is defined). The ACT paper's "ACT".
  "pact"  + Planck Commander low-ell TT + plik-lite cut to TT <= 1000, TE/EE <= 600 (PlanckActCut: bins with
          blmax above the cut removed), with A_planck = A_act in all three Planck pieces. "P-ACT".
Parameters (NAMES): omega_b, omega_cdm, h, ln10As, n_s, tau_reion, A_act (prior N(1, 0.003), as act-lcdm.yaml),
P_act (flat [0.9, 1.1], candl / cobaya's range). Cosmology as hmc/posterior.py: three massless neutrinos
(Neff 3.044), YHe from the BBN table.

Engine: lensed l_max 6325, the last multipole where any kept ACT bin's window is non-zero (candl windows,
hmc/diag/act_windows.py); D_ell is zero-padded to candl's 8501. delta_l_max 5000: unlensed C_l and C_l^phiphi to
11325, the range of the ACT paper's CLASS block (l_max_scalars 9500 + delta_l_max 1800), whose chi2 it reproduces
to 0.006 (hmc/diag/act_lmax_check.py; delta_l_max 1800 here is up to 0.6 off). Photon hierarchy l_max_g 40 /
l_max_pol_g 60 as that block; HALOFIT lensing, streamed over l (lensing_streamed: the matrix form's gradient
needs ~60 GB per cosmology at this l); computed j_l (the tables stop at l = 5000); rtol 1e-5 and the recomb lna
grid as the Planck runs.
"""
import os
_ncpu = os.environ.get("HMC_CPU_DEVICES", "16")
if _ncpu not in ("0", "1") and "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                               f" --xla_force_host_platform_device_count={_ncpu}").strip()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("EQX_ON_ERROR", "nan")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      os.path.join(os.environ.get("SCRATCH", "/pscratch/sd/c/carag"), ".jax_cache_abcmb"))

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from abcmb.main import Model
from scan.lowl_like import LowLEE, LowLTT, cl_to_Dl, LOWEE_TABLES
from scan.batched_grad import staged_cl_and_grad, _to_float

NAMES = ("omega_b", "omega_cdm", "h", "ln10As", "n_s", "tau_reion", "A_act", "P_act")
LATEX = (r"\omega_b", r"\omega_{cdm}", "h", r"\ln(10^{10}A_s)", "n_s", r"\tau_{reio}", r"A_{ACT}", r"P_{ACT}")
NC = 6                                              # cosmological parameters (through ABCMB)
BOX = np.array([(0.005, 0.1), (0.001, 0.99), (0.40, 1.00), (1.61, 3.91), (0.8, 1.2), (0.01, 0.8),
                (0.5, 1.5), (0.9, 1.1)])
A_PRIOR = (1.0, 0.003)
# start: the ACT maximum, found by two Gauss-Newton steps (chi2 552.893 -> 541.924 -> 541.875, predicted gains
# 11.03 / 0.05) from the Planck-lite HMC's start point (0.022383, 0.12011, 0.6789, 3.0448, 0.96605, 0.0543, 1, 1);
# run_act.py takes HMC_GN_STEPS more before the metric. SIG only scales the AD directions (the tangents are
# divided back): ACT DR6 posterior sds, roughly.
START = np.array([0.0225812, 0.123208, 0.668114, 3.04991, 0.967315, 0.0552572, 0.999994, 0.999875])
SIG = np.array([0.00017, 0.0021, 0.008, 0.016, 0.0075, 0.009, 0.003, 0.004])
FIXED = {"TCMB0": 2.34865418e-4, "N_nu_massive": 0, "Neff": 3.044,
         "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5, "exp_reion": 1.5}
ENGINE = dict(output_Cl=True, l_max=6325, delta_l_max=5000, lensing=True, output_Pk=False,
              l_max_g=40, l_max_pol_g=60, l_max_ur=17, l_max_ncdm=17, nonlinear="halofit",
              use_bessel_tables=False, lensing_streamed=True, max_steps_PE=16384, lna_grid_mode="recomb",
              bbn_type="table")
PLIK_CUT = {"tt": 1000, "te": 600, "ee": 600}


def _plik_cut():
    """plik-lite (hmc/plik_lite.py) restricted to the P-ACT bins: the bins with blmax above PLIK_CUT are
    removed from the data vector and the covariance before inverting (= PlanckActCut's 1e10 diagonal)."""
    from scipy.io import FortranFile
    from hmc.plik_lite import PlikLite
    pl = PlikLite()
    f = lambda n: os.path.join(pl.data_dir, n)
    blmax = np.loadtxt(f("blmax.dat")).astype(int) + 30
    data = np.loadtxt(f("cl_cmb_plik_v22.dat"))
    n = sum(pl.nbin.values())
    cov = FortranFile(f("c_matrix_plik_v22.dat"), "r").read_reals(dtype=float).reshape((n, n))
    cov = np.tril(cov) + np.tril(cov, -1).T
    keep, off = [], 0
    for s in ("tt", "te", "ee"):
        nb = pl.nbin[s]
        keep.append(off + np.where(blmax[:nb] <= PLIK_CUT[s])[0])
        off += nb
    keep = np.concatenate(keep)
    pl.keep = keep
    pl.X_data = jnp.asarray(data[keep, 1])
    pl.invcov = jnp.asarray(np.linalg.inv(cov[np.ix_(keep, keep)]))
    pl.ndata = len(keep)
    return pl


class ACTPosterior:
    def __init__(self, data="act", rtol=1e-5, ad_chunk=64, spec_b_local=4, stage_log=None, **engine):
        import candl, candl_data
        self.data = data
        spec = dict(ENGINE, rtol_large_k_PE=rtol, atol_large_k_PE=rtol * 1e-2, rtol_small_k_PE=min(1e-5, rtol))
        spec.update(engine)
        self.model = Model(**spec)
        self.ells = self.model.SS.ells
        self.ad_chunk = int(ad_chunk)
        self.spec_b_local = spec_b_local
        self.stage_log = stage_log
        try:
            self.ndev = len(jax.devices("gpu"))
        except Exception:
            self.ndev = 1
        act = candl.Like(candl_data.ACT_DR6_TTTEEE)
        LT = int(act.ells[-1])                                      # candl's theory input: ell 2..8501
        assert int(act.ells[0]) == 2 and spec["l_max"] <= LT
        lowee = LowLEE(table_path=LOWEE_TABLES["sroll2"])
        if data == "pact":
            lowtt, plite = LowLTT(), _plik_cut()
        elif data != "act":
            raise ValueError(f"data must be 'act' or 'pact', got {data!r}")
        ells = self.ells
        sig = jnp.asarray(SIG[:NC])
        model = self.model

        def derived(th6):
            p = dict(FIXED)
            p["omega_b"], p["omega_cdm"], p["h"] = th6[0], th6[1], th6[2]
            p["A_s"] = jnp.exp(th6[3]) / 1e10
            p["n_s"], p["tau_reion"] = th6[4], th6[5]
            return _to_float(model.add_derived_parameters(p))

        def derived_and_dots(th6):
            dots = jax.vmap(lambda v: jax.jvp(derived, (th6,), (v,))[1])(jnp.diag(sig))
            return derived(th6), dots

        self._derived_B = jax.jit(jax.vmap(derived_and_dots))

        def parts(cl3, nu):
            """chi2 pieces: ACT, low-ell EE, A_act prior[, low-ell TT, plik-lite cut]."""
            A, P = nu[0], nu[1]
            Dtt, Dte, Dee = (cl_to_Dl(c, ells, lmax=LT) for c in cl3)
            out = [act.chi_square({"Dl": {"TT": Dtt[2:], "TE": Dte[2:], "EE": Dee[2:]}, "A_act": A, "P_act": P}),
                   lowee.chi2(Dee, calib=A if data == "pact" else 1.0),
                   ((A - A_PRIOR[0]) / A_PRIOR[1]) ** 2]
            if data == "pact":
                out += [lowtt.chi2(Dtt, calib=A),
                        plite.chi2(plite.bin_model(Dtt[:plite.Lcol], Dte[:plite.Lcol], Dee[:plite.Lcol]), A)]
            return jnp.stack(out)

        self.part_names = ("act", "lowl_EE", "A_prior") + (("lowl_TT", "plik_cut") if data == "pact" else ())
        loglike = lambda cl3, nu: -0.5 * jnp.sum(parts(cl3, nu))
        self._chi2_parts = jax.jit(parts)
        self._vg_B = jax.jit(jax.vmap(jax.value_and_grad(loglike, argnums=(0, 1))))
        nn = len(NAMES) - NC
        signu = jnp.asarray(SIG[NC:])

        def hess_lin(cl3, dcl3, nu):
            """Hessian of loglike in theta / SIG with the theory LINEAR in theta (its C_ell tangents): the
            Gauss-Newton matrix of the Gaussian pieces, the exact curvature of the low-ell ones."""
            f = lambda d: loglike(tuple(c + d[:NC] @ dc for c, dc in zip(cl3, dcl3)), nu + d[NC:] * signu)
            return jax.hessian(f)(jnp.zeros(NC + nn))

        self._hess_lin = jax.jit(hess_lin)

    # ---------------------------------------------------------------
    def in_support(self, th):
        th = np.atleast_2d(th)
        return np.all((th > BOX[:, 0]) & (th < BOX[:, 1]), axis=1)

    def _cls_and_dots(self, th):
        """(B, 8) -> Cls 3 x (B, n_l) and tangents 3 x (B, NC, n_l) per SIG_j e_j."""
        B = len(th)
        P_B, T_B = self._derived_B(jnp.asarray(th[:, :NC]))
        full_ps = [jax.tree.map(lambda x: x[b], P_B) for b in range(B)]
        params_dots = [jax.tree.map(lambda x: x[:, j], T_B) for j in range(NC)]
        cls, dots = staged_cl_and_grad(self.model, full_ps, params_dots, k_chunk_size=100,
                                       shard=self.ndev > 1, spec_b_local=self.spec_b_local,
                                       stage_log=self.stage_log)
        dcl = tuple(jnp.stack([d[s] for d in dots], axis=1) for s in range(3))   # (B, NC, n_l)
        return cls, dcl

    def logp_and_grad(self, th):
        """theta (B, 8) inside BOX -> logp (B,), d logp / d theta (B, 8); NaN where a solve failed."""
        th = np.asarray(th, float)
        lp = np.empty(len(th)); g = np.empty_like(th)
        for s in range(0, len(th), self.ad_chunk):
            e = min(s + self.ad_chunk, len(th))
            cls, dcl = self._cls_and_dots(th[s:e])
            (v, (gcl, gnu)) = self._vg_B(cls, jnp.asarray(th[s:e, NC:]))
            gth = sum(jnp.einsum("bl,bjl->bj", gc, dc) for gc, dc in zip(gcl, dcl)) / SIG[:NC]
            lp[s:e] = np.asarray(v); g[s:e, :NC] = np.asarray(gth); g[s:e, NC:] = np.asarray(gnu)
        return lp, g

    def laplace_cov(self, th0):
        """covariance from the Gauss-Newton curvature at th0 (negative eigenvalues floored), the sampler's
        starting metric; also sets chi2_at_laplace and grad_at_laplace (d logp / d theta at th0)."""
        th0 = np.asarray(th0, float)[None]
        reps = max(self.ndev, 1)
        cls, dcl = self._cls_and_dots(np.repeat(th0, reps, axis=0))
        c0 = tuple(c[0] for c in cls); d0 = tuple(d[0] for d in dcl)
        nu0 = jnp.asarray(th0[0, NC:])
        H = np.asarray(self._hess_lin(c0, d0, nu0))
        self.chi2_at_laplace = dict(zip(self.part_names, np.asarray(self._chi2_parts(c0, nu0))))
        (_, (gcl, gnu)) = self._vg_B(tuple(c[:1] for c in cls), nu0[None])
        gth = sum(jnp.einsum("bl,bjl->bj", gc, dc[:1]) for gc, dc in zip(gcl, dcl))[0] / SIG[:NC]
        self.grad_at_laplace = np.concatenate([np.asarray(gth), np.asarray(gnu[0])])
        ev, V = np.linalg.eigh(-0.5 * (H + H.T))
        F = (V * np.maximum(ev, 1e-8 * ev.max())) @ V.T
        return np.linalg.inv(F) * SIG[:, None] * SIG[None, :], ev

    @staticmethod
    def derived(th):
        """H0 and Omega_m (massless neutrinos: no neutrino matter) of theta (..., 8)."""
        th = np.asarray(th, float)
        return {"H0": 100.0 * th[..., 2], "Omega_m": (th[..., 0] + th[..., 1]) / th[..., 2] ** 2}

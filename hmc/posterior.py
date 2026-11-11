"""posterior.py — Planck 2018 LCDM posterior through ABCMB-k's batched engine, with its exact
gradient by forward-mode AD (scan/batched_grad.staged_cl_and_grad).

Data: plik-lite TTTEEE (hmc/plik_lite.py, cobaya's native plik_lite) + Commander low-ell TT +
low-ell EE (SimAll, Planck 2018 baseline; or SRoll2), all JAX (scan/lowl_like.py; the EE table is
cubic-interpolated instead of cobaya's floor lookup, so it is differentiable). A_planck is shared:
every likelihood divides its D_ell by A_planck^2, prior N(1, 0.0025), as in cobaya.

Parameters (NAMES): omega_b, omega_cdm, h, ln10As, n_s, tau_reion, A_planck; flat priors on the
box BOX (cobaya's Planck ranges, H0 in [40, 100]). Three massless neutrinos (Neff 3.044, no
massive species); YHe from the BBN table (bbn_type "table", CLASS's sBBN file), so it follows
omega_b in value and tangent. Engine settings = the scan/ production profiles (l_max 2508, lensed,
rtol 1e-5, "recomb" lna grid), lensing potential from halofit (nonlinear="halofit", the default since 2026-11-10;
`_v1` lensed with the linear spectrum, which moved n_s by -0.12 sigma against the HMcode CLASS reference:
hmc/diag/nonlinear_lensing_shift.py).

`logp_and_grad(theta (B, 7))` pushes the 6 cosmological directions through every engine stage in
one staged jvp (the Cls and their tangents), then contracts the likelihood's reverse-mode gradient
in C_ell with them; A_planck enters only the likelihood. The B axis is sharded over the visible
GPUs, in blocks of at most `ad_chunk` cosmologies.

Import this module before anything else imports jax: it sets the host-device count HyRex's
tangent stage shards over (HMC_CPU_DEVICES, default 16).
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
import equinox as eqx
from abcmb.main import Model
from scan.lowl_like import LowLEE, LowLTT, cl_to_Dl, LOWEE_TABLES
from scan.batched_grad import staged_cl_and_grad, _to_float
from hmc.plik_lite import PlikLite

NAMES = ("omega_b", "omega_cdm", "h", "ln10As", "n_s", "tau_reion", "A_planck")
LATEX = (r"\omega_b", r"\omega_{cdm}", "h", r"\ln(10^{10}A_s)", "n_s", r"\tau_{reio}", r"A_{planck}")
NC = 6                                              # cosmological parameters (through ABCMB)
BOX = np.array([(0.005, 0.1), (0.001, 0.99), (0.40, 1.00), (1.61, 3.91), (0.8, 1.2), (0.01, 0.8),
                (0.9, 1.1)])
A_PRIOR = (1.0, 0.0025)
# start point: Planck 2018 TTTEEE+lowl+lowE best fit with h raised to keep theta_s with massless neutrinos (the
# published 0.6732 is for one 0.06 eV neutrino and sits ~5 conditional sigma off along theta_s here, ~30 in chi2);
# SIG = Planck posterior sds, only used to scale the AD directions (the tangents are divided back)
START = np.array([0.022383, 0.12011, 0.6789, 3.0448, 0.96605, 0.0543, 1.0])
SIG = np.array([0.00015, 0.0012, 0.0054, 0.014, 0.0042, 0.0073, 0.0025])
FIXED = {"TCMB0": 2.34865418e-4, "N_nu_massive": 0, "Neff": 3.044,
         "Delta_z_reion": 0.5, "z_reion_He": 3.5, "Delta_z_reion_He": 0.5, "exp_reion": 1.5}


class LCDMPosterior:
    def __init__(self, lmax=2508, rtol=1e-5, ad_chunk=64, spec_b_local=4, lowee="simall", nonlinear="halofit"):
        self.model = Model(output_Cl=True, l_max=lmax, lensing=True, output_Pk=False,
                           l_max_g=12, l_max_pol_g=10,
                           rtol_large_k_PE=rtol, atol_large_k_PE=rtol * 1e-2,
                           rtol_small_k_PE=min(1e-5, rtol), max_steps_PE=16384,
                           lna_grid_mode="recomb", bbn_type="table", nonlinear=nonlinear)
        self.ells = self.model.SS.ells
        self.ad_chunk = int(ad_chunk)
        self.spec_b_local = spec_b_local
        try:
            self.ndev = len(jax.devices("gpu"))
        except Exception:
            self.ndev = 1
        plite = PlikLite()
        lowtt = LowLTT()
        lowee_ = LowLEE(table_path=LOWEE_TABLES[lowee])
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
            """derived dict and its tangents along SIG_j e_j (leaves (NC, ...))."""
            dots = jax.vmap(lambda v: jax.jvp(derived, (th6,), (v,))[1])(jnp.diag(sig))
            return derived(th6), dots

        self._derived_B = jax.jit(jax.vmap(derived_and_dots))

        def loglike(cl3, A):
            Dtt, Dte, Dee = (cl_to_Dl(c, ells, lmax=lmax) for c in cl3)
            chi2 = (plite.chi2(plite.bin_model(Dtt, Dte, Dee), A)
                    + lowtt.chi2(Dtt, calib=A) + lowee_.chi2(Dee, calib=A))
            return -0.5 * chi2 - 0.5 * ((A - A_PRIOR[0]) / A_PRIOR[1]) ** 2

        self._loglike = loglike

        def chi2_parts(cl3, A):
            Dtt, Dte, Dee = (cl_to_Dl(c, ells, lmax=lmax) for c in cl3)
            return jnp.stack([plite.chi2(plite.bin_model(Dtt, Dte, Dee), A), lowtt.chi2(Dtt, calib=A),
                              lowee_.chi2(Dee, calib=A), ((A - A_PRIOR[0]) / A_PRIOR[1]) ** 2])

        self._chi2_parts = jax.jit(chi2_parts)
        self._vg_B = jax.jit(jax.vmap(jax.value_and_grad(loglike, argnums=(0, 1))))

        def hess_lin(cl3, dcl3, A):
            """Hessian of loglike in (theta_1..6 / SIG, A) with the theory LINEAR in theta (its C_ell
            tangents): the Gauss-Newton matrix of plik-lite, the exact curvature of the low-ell terms."""
            f = lambda d: loglike(tuple(c + d[:NC] @ dc for c, dc in zip(cl3, dcl3)), A + d[NC])
            return jax.hessian(f)(jnp.zeros(NC + 1))

        self._hess_lin = jax.jit(hess_lin)

    # ---------------------------------------------------------------
    def in_support(self, th):
        th = np.atleast_2d(th)
        return np.all((th > BOX[:, 0]) & (th < BOX[:, 1]), axis=1)

    def _cls_and_dots(self, th):
        """(B, 7) -> Cls 3 x (B, n_l) and tangents 3 x (B, NC, n_l) per SIG_j e_j."""
        B = len(th)
        P_B, T_B = self._derived_B(jnp.asarray(th[:, :NC]))
        full_ps = [jax.tree.map(lambda x: x[b], P_B) for b in range(B)]
        params_dots = [jax.tree.map(lambda x: x[:, j], T_B) for j in range(NC)]
        cls, dots = staged_cl_and_grad(self.model, full_ps, params_dots, k_chunk_size=100,
                                       shard=self.ndev > 1, spec_b_local=self.spec_b_local)
        dcl = tuple(jnp.stack([d[s] for d in dots], axis=1) for s in range(3))   # (B, NC, n_l)
        return cls, dcl

    def logp_and_grad(self, th):
        """theta (B, 7) inside BOX -> logp (B,), d logp / d theta (B, 7); NaN where a solve failed."""
        th = np.asarray(th, float)
        lp = np.empty(len(th)); g = np.empty_like(th)
        for s in range(0, len(th), self.ad_chunk):
            e = min(s + self.ad_chunk, len(th))
            cls, dcl = self._cls_and_dots(th[s:e])
            (v, (gcl, gA)) = self._vg_B(cls, jnp.asarray(th[s:e, NC]))
            gth = sum(jnp.einsum("bl,bjl->bj", gc, dc) for gc, dc in zip(gcl, dcl)) / SIG[:NC]
            lp[s:e] = np.asarray(v); g[s:e, :NC] = np.asarray(gth); g[s:e, NC] = np.asarray(gA)
        return lp, g

    def laplace_cov(self, th0):
        """covariance from the Gauss-Newton curvature at th0 (negative eigenvalues floored), the
        sampler's starting metric."""
        th0 = np.asarray(th0, float)[None]
        reps = max(self.ndev, 1)
        cls, dcl = self._cls_and_dots(np.repeat(th0, reps, axis=0))
        H = np.asarray(self._hess_lin(tuple(c[0] for c in cls), tuple(d[0] for d in dcl), th0[0, NC]))
        self.chi2_at_laplace = dict(zip(("plik_lite", "lowl_TT", "lowl_EE", "A_prior"),
                                        np.asarray(self._chi2_parts(tuple(c[0] for c in cls), th0[0, NC]))))
        ev, V = np.linalg.eigh(-0.5 * (H + H.T))
        F = (V * np.maximum(ev, 1e-8 * ev.max())) @ V.T
        s = np.concatenate([SIG[:NC], [1.0]])
        return np.linalg.inv(F) * s[:, None] * s[None, :], ev

    @staticmethod
    def derived(th):
        """H0 and Omega_m (massless neutrinos: no neutrino matter) of theta (..., 7)."""
        th = np.asarray(th, float)
        return {"H0": 100.0 * th[..., 2], "Omega_m": (th[..., 0] + th[..., 1]) / th[..., 2] ** 2}

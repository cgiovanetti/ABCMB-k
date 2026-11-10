"""act_lite.py — ACT DR6 TT/TE/EE CMB-only ("lite", foreground-marginalized) for the profile driver, with the
interface of plik_full.PlikFull, so the driver's Gauss-Newton / Schur-complement code is shared.

Data: candl's ACT DR6 likelihood (candl_data.ACT_DR6_TTTEEE; Louis et al. 2025, Calabrese et al. 2025): 45 bins
each of TT, TE, EE (centres 600-6126), window functions on D_ell for ell 2..8501 that are non-zero only up to
ell 6325, a Gaussian covariance. candl lives in the venv /global/common/software/m3166/carag/envs/abcmb-k.

Nuisances (FLOAT_NAMES), profiled per cosmology at fixed theory: A_act, the overall calibration (TT / A^2,
TE / A^2, EE / A^2; prior N(1, 0.003) as the ACT DR6 yamls), and P_act, the polarization efficiency (TE / P,
EE / P^2; flat on [0.9, 1.1]). Both divide the theory before binning, so the model is the binned spectra b
(computed once per cosmology) times 1/A^2, 1/(A^2 P), 1/(A^2 P^2), and the profile is a damped Newton on two
numbers. candl's built-in tau prior is not used: tau comes from the low-ell EE likelihood, as in ACT's own
"ACT" combination (act_dr6_cmbonly + planck_2018_lowl.EE_sroll2, which takes calibration 1 here).

Conventions as plik_full: abcmb_cls_to_clik (raw ABCMB C_l -> the likelihood's theory block, here D_ell in muK^2,
(..., 3, 8502) = TT, TE, EE on ell 0..8501), residual(block, nu) = data - model with data chi2 = r^T siginv r,
prior_penalty / prior_hessian, profile_batched(block_B, extra_fn, extra_args) -> (chi2 (B,), nu* (B, 2)).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

T_CMB_UK_DEFAULT = 2.7255e6
FLOAT_NAMES = ["A_act", "P_act"]
A_PRIOR = (1.0, 0.003)


class ACTLite:
    def __init__(self, T_CMB_uK=T_CMB_UK_DEFAULT, newton_iter=12):
        import candl
        import candl_data
        like = candl.Like(candl_data.ACT_DR6_TTTEEE)
        ells = np.asarray(like.ells)
        assert ells[0] == 2 and np.all(np.diff(ells) == 1)
        assert list(like.spec_order) == ["TT lxl", "TE lxl", "EE lxl"], like.spec_order
        self.T_CMB_uK = float(T_CMB_uK)
        self.lmax = int(ells[-1])                                   # 8501: candl's theory input range
        self.Lcol = self.lmax + 1
        # windows acting on D_ell[0..lmax]: (3, nbin, Lcol)
        W = np.zeros((3, like.N_bins[0], self.Lcol))
        for i in range(3):
            W[i][:, 2:] = np.asarray(like.window_functions[i]).T
        nz = np.where(np.any(W != 0, axis=(0, 1)))[0]
        self.l_window_max = int(nz[-1])                             # 6325
        self.W = jnp.asarray(W)
        self.data = jnp.asarray(np.asarray(like._data_bandpowers))
        cov = np.asarray(like.covariance)
        self.cov = cov
        self.siginv = jnp.asarray(np.linalg.inv(cov))
        self.nbin = like.N_bins[0]
        # nuisances: start, conditioning scale, Gaussian prior, bounds (physical)
        self.start = jnp.asarray([1.0, 1.0])
        self.scale = jnp.asarray([0.003, 0.003])
        self.prior_mean = jnp.asarray([A_PRIOR[0], 0.0])
        self.prior_invvar = jnp.asarray([1.0 / A_PRIOR[1] ** 2, 0.0])
        self.lo = jnp.asarray([0.5, 0.9])
        self.hi = jnp.asarray([1.5, 1.1])
        self.newton_iter = int(newton_iter)
        self._candl = like

    # ------------------------------------------------------------------
    def abcmb_cls_to_clik(self, ClTT, ClTE, ClEE, l_arr):
        """raw C_l on the integer grid l_arr -> (..., 3, Lcol) D_ell in muK^2 (TT, TE, EE), zero above
        l_arr's end (the windows of the kept bins end at 6325)."""
        l_arr = jnp.asarray(l_arr)
        keep = l_arr <= self.lmax
        idx = jnp.where(keep, l_arr, 0).astype(int)
        fac = l_arr * (l_arr + 1.0) / (2.0 * jnp.pi) * self.T_CMB_uK ** 2
        zeros = jnp.zeros(jnp.shape(ClTT)[:-1] + (self.Lcol,))
        sc = lambda C: zeros.at[..., idx].add(jnp.where(keep, C * fac, 0.0))
        return jnp.stack([sc(ClTT), sc(ClTE), sc(ClEE)], axis=-2)

    def binned(self, D3):
        """(3, Lcol) -> (3, nbin) at A = P = 1."""
        return jnp.einsum("sbl,sl->sb", self.W, D3)

    def _model(self, b, nu):
        A, P = nu[0], nu[1]
        return jnp.concatenate([b[0] / A ** 2, b[1] / (A ** 2 * P), b[2] / (A ** 2 * P ** 2)])

    def residual(self, D3, nu):
        """data - model (nbin * 3,), data chi2 = r^T siginv r; affine in D3."""
        return self.data - self._model(self.binned(D3), nu)

    def prior_penalty(self, nu):
        return jnp.sum(self.prior_invvar * (nu - self.prior_mean) ** 2)

    def prior_hessian(self):
        return 2.0 * jnp.diag(self.prior_invvar)

    def data_chi2(self, D3, nu):
        r = self.residual(D3, nu)
        return r @ self.siginv @ r

    # ------------------------------------------------------------------
    def profile(self, D3, extra=None):
        """(A_act, P_act) minimizing data chi2 + prior (+ extra(nu)) at fixed theory D3: damped Newton with the
        exact 2 x 2 Hessian in scaled coordinates, steps clipped to the bounds, a fixed number of iterations
        (the objective is a smooth function of two calibrations that sit within ~0.5% of 1). Returns
        (chi2 incl. extra, nu*)."""
        b = self.binned(D3)

        def f(z):
            nu = self.start + self.scale * z
            r = self.data - self._model(b, nu)
            c = r @ self.siginv @ r + self.prior_penalty(nu)
            return c if extra is None else c + extra(nu)

        zlo = (self.lo - self.start) / self.scale
        zhi = (self.hi - self.start) / self.scale
        g_f, h_f = jax.grad(f), jax.hessian(f)

        def step(z, _):
            g, H = g_f(z), h_f(z)
            ev, V = jnp.linalg.eigh(0.5 * (H + H.T))
            dz = -(V / jnp.maximum(ev, 1e-6 * jnp.max(jnp.abs(ev)))) @ (V.T @ g)
            zn = jnp.clip(z + dz, zlo, zhi)
            ok = jnp.isfinite(f(zn)) & (f(zn) <= f(z))
            return jnp.where(ok, zn, z), None

        z, _ = jax.lax.scan(step, jnp.zeros(2), None, length=self.newton_iter)
        nu = self.start + self.scale * z
        return f(z), nu

    def profile_batched(self, D3_B, extra_fn=None, extra_args=()):
        if extra_fn is None:
            return jax.vmap(lambda d: self.profile(d))(D3_B)
        return jax.vmap(lambda d, *a: self.profile(d, extra=lambda nu: extra_fn(nu, *a)))(D3_B, *extra_args)

    # ------------------------------------------------------------------
    def candl_chi2(self, D3, nu):
        """candl's own chi_square at (D3, nu), for validation."""
        Dl = {"TT": D3[0, 2:], "TE": D3[1, 2:], "EE": D3[2, 2:]}
        return self._candl.chi_square({"Dl": Dl, "A_act": nu[0], "P_act": nu[1]})

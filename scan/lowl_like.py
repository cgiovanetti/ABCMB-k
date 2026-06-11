"""lowl_like.py — JAX, autodiff-differentiable Planck 2018 low-ell likelihoods.

Two pieces, both faithful re-implementations of cobaya's NATIVE (clik-free) low-ell
likelihoods, re-arranged so they are pure-JAX, vmap/jacfwd-able, and consume the
SAME padded D_ell[0..lmax] (muK^2) arrays that scan/plik_lite.py already produces
from ABCMB output:

  * LowLEE  -- SRoll2 / SimAll low-ell EE (ell=2..29). cobaya stores a per-multipole
    log-prob TABLE probEE[3000, 28]: row j is the log-likelihood at
    D_ell^EE = j * 1e-4 muK^2.  cobaya's native code does a FLOOR lookup
    (`(D/step).astype(int)`) and sums -> piecewise-constant, NOT differentiable.
    We instead CUBIC-interpolate each column (the table step is 1e-4 muK^2, so the
    interpolant is faithful) -> smooth, C1, AD-able.  This REPLACES the circular
    N(0.0544,0.0073) tau "prior": tau is now constrained by FITTING the actual
    low-ell EE data through the predicted reionization-bump spectrum.

  * LowLTT  -- Commander Gibbs low-ell TT (ell=2..29), Gaussianized via the public
    "cl2x" change-of-variable: per multipole a cubic spline x(C_ell) maps the cl to
    a Gaussian variable; logL = sum log(dx/dC) [Jacobian] - 0.5 (x-mu)^T Cinv (x-mu).
    We replicate the spline with interpax cubic (+ its analytic derivative for the
    Jacobian) so the whole thing is AD-able.  Adds the ell=2..29 TEMPERATURE the
    plik-lite-only run was dropping.

Both return a *chi^2 contribution* = -2 ln L (up to an additive constant that
cancels in any Delta-chi^2 profile).  Validated against the cobaya native formula
in scan/validate_lowl.py.

Conventions:
  * Input is the padded D_ell[0..lmax] array, D_ell = ell(ell+1)C_ell/2pi in muK^2,
    exactly PlikLite.abcmb_cl_to_Dl(...) output.  We slice [2:30].
  * calib (= A_planck) enters as D_ell -> D_ell / calib^2, as in cobaya. Calibration
    is negligible vs low-ell uncertainty (cobaya's own note); default calib=1.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from interpax import interp1d

_DATA = "/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/data"


# ======================================================================
# low-ell EE (SRoll2)
# ======================================================================
class LowLEE:
    def __init__(self, table_path=os.path.join(_DATA, "planck_sroll2_lowE_native",
                                                "sroll2_prob_table.txt"),
                 lmin=2, lmax=29, step=1.0e-4):
        self.lmin, self.lmax, self.step = lmin, lmax, step
        prob = np.loadtxt(table_path)                       # (nsteps, nell)
        self.nsteps, self.nell = prob.shape
        assert self.nell == lmax - lmin + 1, (self.nell, lmax - lmin + 1)
        # D_ell grid the rows are sampled on (left edges, matching cobaya floor)
        self._Dgrid = jnp.asarray(np.arange(self.nsteps) * step)   # (nsteps,)
        self._prob = jnp.asarray(prob)                             # (nsteps, nell)
        self._Dmax = float((self.nsteps - 2) * step)

    def chi2(self, Dl_padded, calib=1.0):
        """Dl_padded: (..., lmax_pad+1) padded D_ell^EE in muK^2.  Returns (...,)
        chi^2 = -2 sum_ell lnP_ell(D_ell). Batched over leading axes."""
        D = Dl_padded[..., self.lmin:self.lmax + 1] / (calib ** 2)   # (..., nell)
        D = jnp.clip(D, 0.0, self._Dmax)                             # stay in table
        # per-column cubic interp: query D[...,c] into (Dgrid, prob[:,c])
        def _col(xq_c, f_c):
            return interp1d(xq_c, self._Dgrid, f_c, method="cubic", extrap=True)
        lnP = jax.vmap(_col, in_axes=(-1, -1), out_axes=-1)(D, self._prob)  # (...,nell)
        return -2.0 * jnp.sum(lnP, axis=-1)


# ======================================================================
# low-ell TT (Commander, Gaussianized cl2x)
# ======================================================================
class LowLTT:
    def __init__(self, data_dir=os.path.join(_DATA, "planck_2018_lowT_native"),
                 lmin=2, lmax=29):
        self.lmin, self.lmax = lmin, lmax
        nl = lmax - lmin + 1
        f = lambda n: os.path.join(data_dir, n)
        cov = np.loadtxt(f("cov.txt"))[lmin - 2:lmax + 1 - 2, lmin - 2:lmax + 1 - 2]
        self._covinv = jnp.asarray(np.linalg.inv(cov))
        self._mu = jnp.asarray(np.loadtxt(f("mu.txt"))[lmin - 2:lmax + 1 - 2])
        mu_sigma = np.loadtxt(f("mu_sigma.txt"))[lmin - 2:lmax + 1 - 2]    # fiducial D_ell
        self._spline_cl = jnp.asarray(np.loadtxt(f("cl2x_1.txt"))[:, lmin - 2:lmax + 1 - 2])  # (1000,nl)
        self._spline_x = jnp.asarray(np.loadtxt(f("cl2x_2.txt"))[:, lmin - 2:lmax + 1 - 2])    # (1000,nl)
        self.nl = nl
        # prior bounds (where |x| < 5, i.e. the well-sampled region), as in cobaya
        sc = np.asarray(self._spline_cl); sv = np.asarray(self._spline_x); nbins = sc.shape[0]
        self._lo = np.zeros(nl); self._hi = np.zeros(nl)
        for i in range(nl):
            j = 0
            while abs(sv[j, i] + 5) < 1e-4:
                j += 1
            self._lo[i] = sc[j + 2, i]
            j = nbins - 1
            while abs(sv[j, i] - 5) < 1e-4:
                j -= 1
            self._hi[i] = sc[j - 2, i]
        self._lo_j = jnp.asarray(self._lo); self._hi_j = jnp.asarray(self._hi)
        # offset so the fiducial gives chi^2 ~ 0 (cobaya normalizes the same way)
        self._offset = 0.0
        self._offset = float(self._neglogl_raw(jnp.asarray(mu_sigma)))

    def _x_and_dxdcl(self, theory):
        """theory: (..., nl) D_ell^TT.  Returns x(...,nl) and dx/dCl(...,nl)."""
        def _col(cl_c, knots_cl, knots_x):
            x = interp1d(cl_c, knots_cl, knots_x, method="cubic", extrap=True)
            dx = interp1d(cl_c, knots_cl, knots_x, method="cubic", derivative=1, extrap=True)
            return x, dx
        x, dx = jax.vmap(_col, in_axes=(-1, -1, -1), out_axes=(-1, -1))(
            theory, self._spline_cl, self._spline_x)
        return x, dx

    def _neglogl_raw(self, theory):
        """-2 * [ sum log(dx/dCl) - 0.5 (x-mu)^T Cinv (x-mu) ]  (no offset)."""
        x, dx = self._x_and_dxdcl(theory)
        jac = 2.0 * jnp.sum(jnp.log(jnp.abs(dx)), axis=-1)     # -2 * sum log dxdcl
        delta = x - self._mu
        gauss = jnp.einsum("...i,ij,...j->...", delta, self._covinv, delta)
        return gauss - jac

    def chi2(self, Dl_padded, calib=1.0):
        """Dl_padded: (..., lmax_pad+1) padded D_ell^TT muK^2.  Returns (...,) chi^2."""
        theory = Dl_padded[..., self.lmin:self.lmax + 1] / (calib ** 2)   # (..., nl)
        # clamp into the spline support (near the data min this is inactive)
        theory = jnp.clip(theory, self._lo_j, self._hi_j)
        return self._neglogl_raw(theory) - self._offset


# quick self-test placeholder (real validation in scan/validate_lowl.py)
if __name__ == "__main__":
    ee = LowLEE(); tt = LowLTT()
    print(f"LowLEE: table {ee.nsteps}x{ee.nell}, Dmax={ee._Dmax:.4f} muK^2")
    print(f"LowLTT: nl={tt.nl}, offset={tt._offset:.4f}, "
          f"prior lo={np.round(tt._lo,4)} hi={np.round(tt._hi,4)}")

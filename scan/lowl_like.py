"""lowl_like.py — JAX, autodiff-differentiable Planck 2018 low-ell likelihoods.

Two pieces, both faithful re-implementations of cobaya's NATIVE (clik-free) low-ell
likelihoods, re-arranged so they are pure-JAX, vmap/jacfwd-able, and consume the
SAME padded D_ell[0..lmax] (muK^2) arrays that scan/plik_lite.py already produces
from ABCMB output:

  * LowLEE  -- SRoll2 / SimAll low-ell EE (ell=2..29). cobaya stores a per-multipole
    log-prob TABLE probEE[3000, 28]: row j is the log-likelihood at
    D_ell^EE = j * 1e-4 muK^2.  cobaya's native code does a FLOOR lookup
    (`(D/step).astype(int)`) -> piecewise-constant, NOT differentiable.  We
    CUBIC-interpolate each column (the table step is 1e-4 muK^2, so the interpolant
    is faithful) -> smooth, C1, AD-able.  REPLACES the circular N(0.0544,0.0073)
    tau "prior": tau is now constrained by FITTING the actual low-ell EE data.

  * LowLTT  -- Commander Gibbs low-ell TT (ell=2..29), Gaussianized via the public
    "cl2x" change-of-variable: per multipole a cubic spline x(C_ell) maps the cl to
    a Gaussian variable; logL = sum log(dx/dC) [Jacobian] - 0.5 (x-mu)^T Cinv (x-mu).
    Adds the ell=2..29 TEMPERATURE the plik-lite-only run dropped.

PERF: the cubic spline DERIVATIVES (interpax approx_df, a tridiagonal solve per
column) are PRECOMPUTED at __init__ and passed to interp1d(..., fx=...) at call
time, so the traced graph only EVALUATES -- the 56 coefficient solves never enter
the jacfwd/vmap graph (otherwise the AD compile blows up to >10 min).

Both return a chi^2 contribution = -2 ln L (up to an additive constant that cancels
in any Delta-chi^2 profile).  Validated against the cobaya native formula in
scan/validate_lowl.py.  calib (= A_planck) enters as D_ell -> D_ell/calib^2 as in
cobaya; calibration is negligible vs low-ell uncertainty so default calib=1.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
from interpax import interp1d, Interpolator1D

_DATA = "/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/data"


def _cubic_fx(x, f):
    """precompute interpax cubic spline derivative coeffs (the tridiag solve)."""
    return Interpolator1D(jnp.asarray(x), jnp.asarray(f), method="cubic").derivs["fx"]


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
        self._Dgrid = jnp.asarray(np.arange(self.nsteps) * step)   # shared x (uniform)
        self._prob = jnp.asarray(prob)                             # (nsteps, nell)
        self._fx = _cubic_fx(self._Dgrid, self._prob)             # (nsteps, nell) precomputed
        self._Dmax = float((self.nsteps - 2) * step)

    def chi2(self, Dl_padded, calib=1.0):
        """Dl_padded: (..., lmax_pad+1) padded D_ell^EE in muK^2.  Returns (...,)
        chi^2 = -2 sum_ell lnP_ell(D_ell). Batched over leading axes."""
        D = Dl_padded[..., self.lmin:self.lmax + 1] / (calib ** 2)
        D = jnp.clip(D, 0.0, self._Dmax)
        def _col(xq_c, f_c, fx_c):
            return interp1d(xq_c, self._Dgrid, f_c, method="cubic", fx=fx_c, extrap=True)
        lnP = jax.vmap(_col, in_axes=(-1, -1, -1), out_axes=-1)(D, self._prob, self._fx)
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
        spline_cl = np.loadtxt(f("cl2x_1.txt"))[:, lmin - 2:lmax + 1 - 2]  # (1000,nl) x-knots
        spline_x = np.loadtxt(f("cl2x_2.txt"))[:, lmin - 2:lmax + 1 - 2]   # (1000,nl) x-vals
        self.nl = nl
        self._cl = jnp.asarray(spline_cl)
        self._xk = jnp.asarray(spline_x)
        # per-column precomputed cubic derivative coeffs (x differs per column)
        fxs = [np.asarray(_cubic_fx(spline_cl[:, c], spline_x[:, c])) for c in range(nl)]
        self._fx = jnp.asarray(np.array(fxs).T)                            # (1000,nl)
        # prior bounds (|x|<5 region), as in cobaya
        nbins = spline_cl.shape[0]
        self._lo = np.zeros(nl); self._hi = np.zeros(nl)
        for i in range(nl):
            j = 0
            while abs(spline_x[j, i] + 5) < 1e-4:
                j += 1
            self._lo[i] = spline_cl[j + 2, i]
            j = nbins - 1
            while abs(spline_x[j, i] - 5) < 1e-4:
                j -= 1
            self._hi[i] = spline_cl[j - 2, i]
        self._lo_j = jnp.asarray(self._lo); self._hi_j = jnp.asarray(self._hi)
        self._offset = 0.0
        self._offset = float(self._neglogl_raw(jnp.asarray(mu_sigma)))

    def _x_and_dxdcl(self, theory):
        """theory: (..., nl) D_ell^TT.  Returns x(...,nl), dx/dCl(...,nl)."""
        def _col(cl_c, knots_c, xk_c, fx_c):
            x = interp1d(cl_c, knots_c, xk_c, method="cubic", fx=fx_c, extrap=True)
            dxc = interp1d(cl_c, knots_c, xk_c, method="cubic", derivative=1,
                           fx=fx_c, extrap=True)
            return x, dxc
        x, dx = jax.vmap(_col, in_axes=(-1, -1, -1, -1), out_axes=(-1, -1))(
            theory, self._cl, self._xk, self._fx)
        return x, dx

    def _neglogl_raw(self, theory):
        x, dx = self._x_and_dxdcl(theory)
        jac = 2.0 * jnp.sum(jnp.log(jnp.abs(dx)), axis=-1)     # -2 sum log dxdcl
        delta = x - self._mu
        gauss = jnp.einsum("...i,ij,...j->...", delta, self._covinv, delta)
        return gauss - jac

    def chi2(self, Dl_padded, calib=1.0):
        """Dl_padded: (..., lmax_pad+1) padded D_ell^TT muK^2.  Returns (...,) chi^2."""
        theory = Dl_padded[..., self.lmin:self.lmax + 1] / (calib ** 2)
        theory = jnp.clip(theory, self._lo_j, self._hi_j)
        return self._neglogl_raw(theory) - self._offset


if __name__ == "__main__":
    ee = LowLEE(); tt = LowLTT()
    print(f"LowLEE: table {ee.nsteps}x{ee.nell}, Dmax={ee._Dmax:.4f} muK^2")
    print(f"LowLTT: nl={tt.nl}, offset={tt._offset:.4f}")

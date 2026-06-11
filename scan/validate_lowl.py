"""validate_lowl.py — validate the JAX low-ell EE/TT likelihoods (scan/lowl_like.py)
against cobaya's NATIVE (clik-free) reference formula, byte-for-byte the same data.

cobaya's planck_2018_lowl.EE / .TT are pure-python; we replicate their exact
log_likelihood here in numpy (the reference) and compare to our JAX versions on a
spread of test D_ell spectra.  The JAX versions differ from cobaya ONLY by:
  * EE: cubic interpolation of the prob table vs cobaya's floor lookup (the table
    step is 1e-4 muK^2, so the gap is a fraction of one table step -> tiny).
  * TT: interpax cubic spline vs scipy InterpolatedUnivariateSpline for the cl->x
    change of variable (different cubic flavor; sub-cosmic-variance).
We report max |Delta chi^2| over the test spectra (must be << 1, the scale of a 1
sigma shift) and confirm the JAX gradient is finite/non-zero.

CPU only (JAX_PLATFORMS=cpu) so it never touches a GPU.  Run via srun.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.interpolate import InterpolatedUnivariateSpline
from scan.lowl_like import LowLEE, LowLTT, _DATA

LMAXPAD = 2508
EE_LMIN, EE_LMAX = 2, 29


def pad(Dl_2_29, lmin=2):
    """embed a (28,) D_ell[2..29] into a padded [0..LMAXPAD] array."""
    a = np.zeros(LMAXPAD + 1); a[lmin:lmin + len(Dl_2_29)] = Dl_2_29
    return a


# ---------------- EE reference (cobaya native floor lookup) ----------------
def ee_ref_chi2(probEE, Dl_2_29, step=1e-4, calib=1.0):
    idx = (np.asarray(Dl_2_29) / (calib ** 2 * step)).astype(int)
    logL = np.take_along_axis(probEE, idx[np.newaxis, :], 0).sum()
    return -2.0 * logL


# ---------------- TT reference (cobaya native cl2x) ----------------
class TTRef:
    def __init__(self, data_dir, lmin=2, lmax=29):
        self.lmin, self.lmax = lmin, lmax
        f = lambda n: os.path.join(data_dir, n)
        cov = np.loadtxt(f("cov.txt"))[lmin - 2:lmax + 1 - 2, lmin - 2:lmax + 1 - 2]
        self._covinv = np.linalg.inv(cov)
        self._mu = np.loadtxt(f("mu.txt"))[lmin - 2:lmax + 1 - 2]
        self.mu_sigma = np.loadtxt(f("mu_sigma.txt"))[lmin - 2:lmax + 1 - 2]
        sc = np.loadtxt(f("cl2x_1.txt"))[:, lmin - 2:lmax + 1 - 2]
        sv = np.loadtxt(f("cl2x_2.txt"))[:, lmin - 2:lmax + 1 - 2]
        self._spl = [InterpolatedUnivariateSpline(sc[:, i], sv[:, i]) for i in range(lmax - lmin + 1)]
        self._dspl = [s.derivative() for s in self._spl]
        self._offset = 0.0
        self._offset = self.chi2(self.mu_sigma)

    def chi2(self, theory):
        x = np.array([s(c) for s, c in zip(self._spl, theory)])
        dx = np.array([d(c) for d, c in zip(self._dspl, theory)])
        logl = np.sum(np.log(dx)) - 0.5 * self._covinv.dot(x - self._mu).dot(x - self._mu)
        return -2.0 * logl - self._offset


def main():
    ee = LowLEE(); tt = LowLTT()
    probEE = np.asarray(ee._prob)
    ttref = TTRef(os.path.join(_DATA, "planck_2018_lowT_native"))

    print("=" * 70)
    print("LOW-ELL EE (SRoll2)")
    print("=" * 70)
    rng = np.random.default_rng(0)
    # test spectra: (a) exact grid points, (b) random in-support, (c) a realistic
    # low-ell EE bump shape scaled up/down
    Dmax = ee._Dmax
    ell = np.arange(2, 30)
    bump = 0.02 * np.exp(-((ell - 5) ** 2) / 50.0)          # crude reion bump, muK^2
    tests = {
        "grid-aligned": (np.floor(rng.uniform(50, 2900, 28)) * 1e-4),
        "random-in-support": rng.uniform(0.0005, Dmax * 0.9, 28),
        "reion-bump x1": np.clip(bump, 1e-4, Dmax),
        "reion-bump x0.5": np.clip(0.5 * bump, 1e-4, Dmax),
        "reion-bump x1.5": np.clip(1.5 * bump, 1e-4, Dmax),
    }
    maxd = 0.0
    for name, D in tests.items():
        ref = ee_ref_chi2(probEE, D)
        jx = float(ee.chi2(jnp.asarray(pad(D))))
        d = abs(jx - ref); maxd = max(maxd, d)
        print(f"  {name:20s}: cobaya={ref:10.4f}  jax={jx:10.4f}  |d|={d:.4f}")
    # gradient finite/non-zero
    g = jax.grad(lambda D: ee.chi2(D))(jnp.asarray(pad(tests["reion-bump x1"])))
    gn = np.asarray(g)[2:30]
    print(f"  EE max |Delta chi2| = {maxd:.4f}  (must be << 1)")
    print(f"  EE grad finite={np.all(np.isfinite(gn))} nonzero={np.any(gn!=0)} "
          f"||g||={np.linalg.norm(gn):.3e}")

    print("=" * 70)
    print("LOW-ELL TT (Commander cl2x)")
    print("=" * 70)
    base = ttref.mu_sigma.copy()
    tests_tt = {
        "fiducial (mu_sigma)": base,
        "fiducial x1.05": base * 1.05,
        "fiducial x0.95": base * 0.95,
        "random +-3%": base * (1 + rng.uniform(-0.03, 0.03, len(base))),
    }
    maxd = 0.0
    for name, D in tests_tt.items():
        ref = ttref.chi2(D)
        jx = float(tt.chi2(jnp.asarray(pad(D))))
        d = abs(jx - ref); maxd = max(maxd, d)
        print(f"  {name:20s}: cobaya={ref:10.4f}  jax={jx:10.4f}  |d|={d:.4f}")
    g = jax.grad(lambda D: tt.chi2(D))(jnp.asarray(pad(base * 1.02)))
    gn = np.asarray(g)[2:30]
    print(f"  TT max |Delta chi2| = {maxd:.4f}  (must be << 1; cosmic-variance scale)")
    print(f"  TT grad finite={np.all(np.isfinite(gn))} nonzero={np.any(gn!=0)} "
          f"||g||={np.linalg.norm(gn):.3e}")
    print(f"  TT offset (chi2 at fiducial should be ~0): jax={float(tt.chi2(jnp.asarray(pad(base)))):.4f}")


if __name__ == "__main__":
    main()

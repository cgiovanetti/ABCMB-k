"""plik_lite.py — JAX-friendly Planck 2018 plik-lite (foreground-marginalized)
Gaussian chi^2, wired to ABCMB output.

This is a faithful re-implementation of cobaya's
``cobaya.likelihoods.base_classes.planck_pliklite.PlanckPlikLite`` binning, but
arranged so the *expensive* per-cosmology pieces (binning theory spectra, the
inverse-covariance contraction, and profiling the calibration nuisance
A_planck) are pure-JAX and vmap over a leading batch axis. ABCMB's
``Model.call_batched`` returns spectra batched over B cosmologies; this kernel
turns that (B, n_ell) block into a (B,) chi^2 vector — exactly the artifact a
frequentist scan keeps.

Conventions (must match cobaya exactly):
  * Data file ``cl_cmb_plik_v22.dat`` columns: [L_av, D_l, err], where
    D_l = l(l+1) C_l / (2 pi) in muK^2.  613 bins = 215 TT + 199 TE + 199 EE.
  * bin_lmin_offset = 30; bins reach l = 2508.
  * cobaya works "directly with D_L not C_L": it pre-multiplies the raw bweights
    by 2 pi / (l (l+1)), then dots them against the theory D_l array. We replicate
    that, so our binning matrix acts on D_l (not C_l).
  * A_planck enters as model_binned -> model_binned / A_planck^2.

ABCMB gives *raw, dimensionless* C_l (the (dT/T)^2 angular power) on the integer
grid l = l_min .. l_max. We convert to D_l in muK^2 via
    D_l = l(l+1)/(2 pi) * C_l * T_CMB_uK^2 .
T_CMB_uK defaults to 2.7255e6 (Planck fiducial; ABCMB's TCMB0=2.34865418e-4 eV
== 2.7255 K). Validated empirically — a wrong factor makes chi^2 ~ 1e10.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

T_CMB_K_DEFAULT = 2.7255
T_CMB_UK_DEFAULT = T_CMB_K_DEFAULT * 1.0e6

# default data location (already on disk in this workspace)
DEFAULT_DATA_DIR = "/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/data/planck_2018_pliklite_native"


class PlikLite:
    """Loads plik-lite data once (numpy/host) and exposes JAX chi^2 kernels.

    Construct once, reuse across an entire scan. The binning matrices, data
    vector and inverse covariance are stored as jnp arrays so the chi^2 path is
    fully traceable / vmappable / jittable.
    """

    def __init__(self, data_dir=DEFAULT_DATA_DIR, use_cl=("tt", "te", "ee"),
                 T_CMB_uK=T_CMB_UK_DEFAULT):
        self.data_dir = data_dir
        self.use_cl = tuple(c.lower() for c in use_cl)
        self.T_CMB_uK = float(T_CMB_uK)

        # ---- .dataset constants (hard-coded from plik_lite_v22.dataset) ----
        nbintt, nbinte, nbinee = 215, 199, 199
        self.nbin = {"tt": nbintt, "te": nbinte, "ee": nbinee}
        self.lmax = 2508
        bin_lmin_offset = 30

        f = lambda n: os.path.join(data_dir, n)
        data = np.loadtxt(f("cl_cmb_plik_v22.dat"))           # (613, 3)
        blmin = np.loadtxt(f("blmin.dat")).astype(int) + bin_lmin_offset
        blmax = np.loadtxt(f("blmax.dat")).astype(int) + bin_lmin_offset
        weights = np.loadtxt(f("bweight.dat"))
        ls = np.arange(len(weights)) + bin_lmin_offset
        weights = weights * (2.0 * np.pi / ls / (ls + 1))     # work in D_L
        weights = np.hstack((np.zeros(bin_lmin_offset), weights))  # index by l

        # covariance (Fortran binary, lower-triangle, symmetrize) — as cobaya
        from scipy.io import FortranFile
        nbins_total = nbintt + nbinte + nbinee
        cov = FortranFile(f("c_matrix_plik_v22.dat"), "r").read_reals(
            dtype=float).reshape((nbins_total, nbins_total))
        cov = np.tril(cov) + np.tril(cov, -1).T

        # ---- per-spectrum binning matrices, acting on D_l[0..lmax] ----
        Lcol = self.lmax + 1
        self._M = {}
        order = ("tt", "te", "ee")
        for spec in order:
            nb = self.nbin[spec]
            M = np.zeros((nb, Lcol))
            for i in range(nb):
                lo, hi = blmin[i], blmax[i]
                M[i, lo:hi + 1] = weights[lo:hi + 1]
            self._M[spec] = M

        # ---- assemble the used data vector + invcov over used spectra ----
        offsets = {"tt": 0, "te": nbintt, "ee": nbintt + nbinte}
        used_idx = []
        for spec in order:
            if spec in self.use_cl:
                used_idx.append(np.arange(self.nbin[spec]) + offsets[spec])
        used_idx = np.hstack(used_idx)
        self.used_specs = tuple(s for s in order if s in self.use_cl)

        self.X_data_np = data[used_idx, 1]                    # D_l muK^2
        self.err_np = data[used_idx, 2]                       # diagonal errors
        self.invcov_np = np.linalg.inv(cov[np.ix_(used_idx, used_idx)])
        self.ndata = self.X_data_np.size

        # ---- jnp versions ----
        self.X_data = jnp.asarray(self.X_data_np)
        self.invcov = jnp.asarray(self.invcov_np)
        self.M = {s: jnp.asarray(self._M[s]) for s in self.used_specs}
        # precompute invcov @ d (for fast A_planck profiling)
        self._Cinv_d = self.invcov @ self.X_data
        self._a = float(self.X_data_np @ (self.invcov_np @ self.X_data_np))

        self.Lcol = Lcol

    # ------------------------------------------------------------------
    # ABCMB raw C_l  ->  padded D_l[0..lmax] in muK^2
    # ------------------------------------------------------------------
    def abcmb_cl_to_Dl(self, Cl, l_arr):
        """Convert one ABCMB raw-C_l array (on integer grid ``l_arr``) to a
        zero-padded D_l[0..lmax] array in muK^2. Works batched if Cl has a
        leading B axis (l_arr is shared, shape (n_ell,))."""
        l_arr = jnp.asarray(l_arr)
        fac = l_arr * (l_arr + 1.0) / (2.0 * jnp.pi) * (self.T_CMB_uK ** 2)
        Dl_sparse = Cl * fac                                   # (..., n_ell)
        # scatter onto a 0..lmax grid (l_arr are integers, l_min usually 2)
        Lcol = self.Lcol
        keep = l_arr <= self.lmax
        idx = jnp.where(keep, l_arr, 0)
        zeros = jnp.zeros(Cl.shape[:-1] + (Lcol,), dtype=Dl_sparse.dtype)
        contrib = jnp.where(keep, Dl_sparse, 0.0)
        Dl = zeros.at[..., idx].add(contrib)
        return Dl

    # ------------------------------------------------------------------
    # binned model from padded D_l arrays
    # ------------------------------------------------------------------
    def bin_model(self, Dtt=None, Dte=None, Dee=None):
        """Bin padded D_l[0..lmax] arrays -> concatenated model vector (A=1).
        Each D* may carry a leading B axis. Returns (..., ndata)."""
        Dmap = {"tt": Dtt, "te": Dte, "ee": Dee}
        parts = []
        for s in self.used_specs:
            D = Dmap[s]
            # M[s] is (nbin, Lcol); D is (..., Lcol) -> (..., nbin)
            parts.append(D @ self.M[s].T)
        return jnp.concatenate(parts, axis=-1)

    # ------------------------------------------------------------------
    # chi^2 at fixed A_planck
    # ------------------------------------------------------------------
    def chi2(self, model0, A_planck=1.0):
        """chi^2 from an A=1 model vector ``model0`` (..., ndata)."""
        diff = self.X_data - model0 / (A_planck ** 2)
        return jnp.einsum("...i,ij,...j->...", diff, self.invcov, diff)

    # ------------------------------------------------------------------
    # profile A_planck (closed form + optional Gaussian prior, vectorized)
    # ------------------------------------------------------------------
    def profile_A(self, model0, sigma_A=0.0025, with_prior=True, n_grid=4001,
                  A_lo=0.985, A_hi=1.015):
        """Minimize chi^2 over A_planck for each cosmology.

        model0 : (..., ndata) A=1 binned model.
        Returns (chi2_min, A_best), each (...).

        With the (Planck) Gaussian prior N(1, sigma_A) included, chi^2(A) is
        smooth; we evaluate it on a dense A grid (cheap: a few scalar coeffs per
        cosmo) and take the min. Without the prior, the closed form
            t* = b/c,  A* = 1/sqrt(t*),  chi2 = a - b^2/c
        is used (t = 1/A^2)."""
        Cinv_m0 = jnp.einsum("ij,...j->...i", self.invcov, model0)
        b = jnp.einsum("...i,i->...", model0, self._Cinv_d)        # m0 . Cinv d
        c = jnp.einsum("...i,...i->...", model0, Cinv_m0)          # m0 . Cinv m0
        a = self._a

        if not with_prior:
            t = b / c
            chi2_min = a - b * b / c
            A_best = 1.0 / jnp.sqrt(t)
            return chi2_min, A_best

        Agrid = jnp.linspace(A_lo, A_hi, n_grid)                   # (G,)
        t = 1.0 / (Agrid ** 2)                                     # (G,)
        # chi2(A) = a - 2 t b + t^2 c + ((A-1)/sigma)^2
        # broadcast: (...,1) with (G,)
        chi2_grid = (a - 2.0 * t * b[..., None] + (t ** 2) * c[..., None]
                     + ((Agrid - 1.0) / sigma_A) ** 2)            # (..., G)
        j = jnp.argmin(chi2_grid, axis=-1)
        chi2_min = jnp.take_along_axis(chi2_grid, j[..., None], axis=-1)[..., 0]
        A_best = Agrid[j]
        return chi2_min, A_best

    def profile_amplitude(self, model0):
        """Profile the OVERALL amplitude of the model analytically.

        Because C_l (hence the binned D_l) scales linearly with A_s, and the
        plik-lite calibration enters as model -> model / A_planck^2, the whole
        combination ``alpha = (A_s/A_s_ref) / A_planck^2`` is a single
        multiplicative amplitude on a spectrum computed at a *reference* A_s.
        Profiling it freely (A_s carries no informative prior; A_planck's tight
        prior is absorbed once A_s floats) is closed form:
            alpha* = b/c,   chi2_min = a - b^2/c,   (t=1/A^2 algebra, no prior)
        with a=d^T Cinv d, b=m0^T Cinv d, c=m0^T Cinv m0.

        NOTE: with *lensed* spectra this is exact only up to the (small) lensing
        non-linearity in A_s (lensing smoothing ~ A_s, primary ~ A_s). Good for
        a feasibility profile; promote A_s to a real grid/optimizer dim for a
        publication-grade result.

        model0 : (..., ndata) spectrum at the reference A_s, A_planck=1.
        Returns (chi2_min, alpha).
        """
        Cinv_m0 = jnp.einsum("ij,...j->...i", self.invcov, model0)
        b = jnp.einsum("...i,i->...", model0, self._Cinv_d)
        c = jnp.einsum("...i,...i->...", model0, Cinv_m0)
        alpha = b / c
        chi2_min = self._a - b * b / c
        return chi2_min, alpha

    # ------------------------------------------------------------------
    # convenience: full path from ABCMB batched spectra -> profiled chi^2
    # ------------------------------------------------------------------
    def chi2_from_abcmb(self, ClTT, ClTE, ClEE, l_arr, profile=True,
                        with_prior=True):
        """ClTT/ClTE/ClEE: (..., n_ell) ABCMB raw-C_l. Returns dict with
        chi2 (profiled or A=1), A_best, and the binned model vector."""
        Dtt = self.abcmb_cl_to_Dl(ClTT, l_arr)
        Dte = self.abcmb_cl_to_Dl(ClTE, l_arr)
        Dee = self.abcmb_cl_to_Dl(ClEE, l_arr)
        m0 = self.bin_model(Dtt, Dte, Dee)
        if profile:
            chi2_min, A_best = self.profile_A(m0, with_prior=with_prior)
            return {"chi2": chi2_min, "A_best": A_best, "model0": m0}
        return {"chi2": self.chi2(m0, 1.0), "A_best": jnp.ones(m0.shape[:-1]),
                "model0": m0}

    # ------------------------------------------------------------------
    # diagnostic: per-spectrum diagonal chi^2 (normalization sanity check)
    # ------------------------------------------------------------------
    def diag_chi2_by_spec(self, ClTT, ClTE, ClEE, l_arr):
        Dtt = self.abcmb_cl_to_Dl(ClTT, l_arr)
        Dte = self.abcmb_cl_to_Dl(ClTE, l_arr)
        Dee = self.abcmb_cl_to_Dl(ClEE, l_arr)
        m0 = self.bin_model(Dtt, Dte, Dee)
        out = {}
        off = 0
        for s in self.used_specs:
            nb = self.nbin[s]
            d = self.X_data_np[off:off + nb] - np.asarray(m0)[..., off:off + nb]
            e = self.err_np[off:off + nb]
            out[s] = float(np.sum((d / e) ** 2))
            off += nb
        return out

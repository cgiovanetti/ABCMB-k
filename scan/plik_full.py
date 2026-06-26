"""plik_full.py — FULL Planck 2018 high-ell plik (TTTEEE) for the frequentist tool.

Replaces plik-lite. The full plik carries 47 foreground/calibration nuisances that
do not enter the theory code, so they are handled entirely inside the likelihood
(profiled at fixed ABCMB theory Cls), never by re-running ABCMB. That is the whole
point of this module: one ABCMB call per cosmology, then a cheap inner optimisation
over the nuisances.

Backend: `clipy` (pure-Python, JAX-native clik reimplementation; same code modern
cobaya calls). Its constructor self-tests against the clik-stored check_value
(self-test diff 4e-6). We feed it ABCMB Cls (raw -> muK^2, TT/EE/BB/TE/TB/EB
ordering, 0..lmax) and a nuisance dict; it returns logL (chi^2 = -2 logL).

Nuisance treatment (matches the Planck 2018 baseline exactly -- the cobaya
planck_2018_highl_plik yamls):
  FLOATED (21), profiled per cosmology:
    A_cib_217, xi_sz_cib, A_sz, ksz_norm,            (CIB & SZ; uniform priors)
    ps_A_{100_100,143_143,143_217,217_217},          (point sources; uniform priors)
    gal545_A_{100,143,143_217,217},                  (TT dust; Gaussian priors)
    galf_TE_A_{100,100_143,100_217,143,143_217,217}, (TE dust; Gaussian priors)
    calib_100T, calib_217T, A_planck                 (calibration; Gaussian priors)
  Joint SZ prior: (ksz_norm + 1.6*A_sz) ~ N(9.5, 3.0).
  FIXED (26) at Planck values: cib_index=-1.3, galf_{TE,EE}_index=-2.4, galf_EE_A_*,
    A_cnoise_e2e_*_EE=1, A_sbpx_*_{TT,EE}=1, A_pol=1, calib_{100,143,217}P.

Profiling (not marginalising) the nuisances is the frequentist-consistent choice and
matches what the tool already does for the single plik-lite A_planck. The profiled
chi^2 is differentiable in the theory Cls by the envelope theorem (hold the nuisance
optimum fixed via stop_gradient), so the existing staged-AD gradient path is intact.

Conventions:
  * cls block to clipy: shape (6, lmax+1) = [TT, EE, BB, TE, TB, EB], muK^2 C_l
    (not D_l). raw ABCMB C_l -> muK^2 via * T_CMB_uK^2 (T_CMB_uK = 2.7255e6).
  * chi2_mode=True returns the bare data logL (no priors); we add the Planck priors
    ourselves for full transparency/control.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

CLIK_FILE_DEFAULT = ("/pscratch/sd/c/carag/Neal_ACTDR6/baseline/plc_3.0/hi_l/plik/"
                     "plik_rd12_HM_v22b_TTTEEE.clik")
CLIPY_PATH = "/pscratch/sd/c/carag/Neal_ACTDR6/cobaya_packages/code/planck/clipy"
T_CMB_UK_DEFAULT = 2.7255e6

# ---- the 26 FIXED nuisances (Planck 2018 baseline values; cobaya params_*.yaml) ----
FIXED_NUIS = {
    "cib_index": -1.3,
    "galf_TE_index": -2.4,
    "galf_EE_index": -2.4,
    "galf_EE_A_100": 0.055, "galf_EE_A_100_143": 0.040, "galf_EE_A_100_217": 0.094,
    "galf_EE_A_143": 0.086, "galf_EE_A_143_217": 0.21, "galf_EE_A_217": 0.70,
    "A_cnoise_e2e_100_100_EE": 1.0, "A_cnoise_e2e_143_143_EE": 1.0,
    "A_cnoise_e2e_217_217_EE": 1.0,
    "A_sbpx_100_100_TT": 1.0, "A_sbpx_143_143_TT": 1.0, "A_sbpx_143_217_TT": 1.0,
    "A_sbpx_217_217_TT": 1.0,
    "A_sbpx_100_100_EE": 1.0, "A_sbpx_100_143_EE": 1.0, "A_sbpx_100_217_EE": 1.0,
    "A_sbpx_143_143_EE": 1.0, "A_sbpx_143_217_EE": 1.0, "A_sbpx_217_217_EE": 1.0,
    "A_pol": 1.0,
    "calib_100P": 1.021, "calib_143P": 0.966, "calib_217P": 1.040,
}

# ---- the 21 FLOATED nuisances: (name, start, scale, gauss_prior(mean,sigma)|None, (lo,hi)) ----
# scale = characteristic step for conditioning (prior sigma where Gaussian; ref-dist scale
# otherwise). lo/hi = the cobaya uniform-prior box (None = unbounded).
_F = [
    # name                start    scale   gauss(mean,sigma)        bounds
    ("A_cib_217",          67.0,   10.0,   None,                    (0.0, 200.0)),
    ("xi_sz_cib",           0.1,    0.1,   None,                    (0.0, 1.0)),
    ("A_sz",                7.0,    2.0,   None,                    (0.0, 10.0)),
    ("ksz_norm",            3.0,    3.0,   None,                    (0.0, 10.0)),
    ("ps_A_100_100",      257.0,   24.0,   None,                    (0.0, 400.0)),
    ("ps_A_143_143",       47.0,   10.0,   None,                    (0.0, 400.0)),
    ("ps_A_143_217",       40.0,   12.0,   None,                    (0.0, 400.0)),
    ("ps_A_217_217",      104.0,   13.0,   None,                    (0.0, 400.0)),
    ("gal545_A_100",        8.6,    2.0,   (8.6,    2.0),           (0.0, None)),
    ("gal545_A_143",       10.6,    2.0,   (10.6,   2.0),           (0.0, None)),
    ("gal545_A_143_217",   23.5,    8.5,   (23.5,   8.5),           (0.0, None)),
    ("gal545_A_217",       91.9,   20.0,   (91.9,  20.0),           (0.0, None)),
    ("galf_TE_A_100",       0.130,  0.042, (0.130,  0.042),         (0.0, None)),
    ("galf_TE_A_100_143",   0.130,  0.036, (0.130,  0.036),         (0.0, None)),
    ("galf_TE_A_100_217",   0.46,   0.09,  (0.46,   0.09),          (0.0, None)),
    ("galf_TE_A_143",       0.207,  0.072, (0.207,  0.072),         (0.0, None)),
    ("galf_TE_A_143_217",   0.69,   0.09,  (0.69,   0.09),          (0.0, None)),
    ("galf_TE_A_217",       1.938,  0.54,  (1.938,  0.54),          (0.0, None)),
    ("calib_100T",          1.0002, 0.0007,(1.0002, 0.0007),        (None, None)),
    ("calib_217T",          0.99805,0.00065,(0.99805,0.00065),      (None, None)),
    ("A_planck",            1.0,    0.0025,(1.0,    0.0025),        (None, None)),
]
FLOAT_NAMES = [f[0] for f in _F]


class PlikFull:
    """Full plik TTTEEE via clipy, with an inner nuisance profile at fixed theory.

    Usage:
      pl = PlikFull()
      cls2d = pl.abcmb_cls_to_clik(ClTT, ClTE, ClEE, l_arr)   # (6, lmax+1) muK^2
      chi2, nu_star = pl.profile(cls2d)                        # profiled high-ell chi^2
      # batched:
      chi2_B, nu_B   = pl.profile_batched(cls2d_B)             # cls2d_B: (B,6,lmax+1)
    """

    def __init__(self, clik_file=CLIK_FILE_DEFAULT, T_CMB_uK=T_CMB_UK_DEFAULT,
                 maxit=30, maxls=10, c1=1e-4, eigfloor=0.1):
        import sys
        if CLIPY_PATH not in sys.path:
            sys.path.insert(0, CLIPY_PATH)
        import clipy
        self.clipy = clipy
        self.lkl = clipy.clik(clik_file)            # prints its self-test on construct
        self.T_CMB_uK = float(T_CMB_uK)
        self.maxit = int(maxit)                     # projected-Newton outer iterations
        self.maxls = int(maxls)                     # Armijo backtracks per iteration
        self.c1 = float(c1)                         # Armijo sufficient-decrease constant
        self.eigfloor = float(eigfloor)             # PD floor for the preconditioner eigenvalues
        self.Hprec = None                           # fixed PD Hessian (set by compute/set_preconditioner)

        lmaxs = np.asarray(self.lkl.lmax)
        self.lmax = int(np.max(lmaxs))              # 2508
        self.Lcol = self.lmax + 1

        # static arrays for the scaled-coordinate inner profile
        self.start = jnp.asarray([f[1] for f in _F])           # (21,) physical start
        self.scale = jnp.asarray([f[2] for f in _F])           # (21,) conditioning scale
        # Gaussian-prior arrays (mean, inv-sigma^2); 0 inv-var for uniform-prior params
        pm = np.zeros(len(_F)); piv = np.zeros(len(_F))
        for i, f in enumerate(_F):
            if f[3] is not None:
                pm[i] = f[3][0]; piv[i] = 1.0 / f[3][1] ** 2
        self.prior_mean = jnp.asarray(pm)
        self.prior_invvar = jnp.asarray(piv)
        # box bounds in PHYSICAL coords (inf where unbounded)
        lo = np.array([(-np.inf if f[4][0] is None else f[4][0]) for f in _F])
        hi = np.array([(np.inf if f[4][1] is None else f[4][1]) for f in _F])
        self.lo = jnp.asarray(lo); self.hi = jnp.asarray(hi)
        # SZ joint-prior indices: (ksz_norm + 1.6*A_sz) ~ N(9.5, 3.0)
        self.i_ksz = FLOAT_NAMES.index("ksz_norm")
        self.i_asz = FLOAT_NAMES.index("A_sz")
        self.sz_mean, self.sz_sig = 9.5, 3.0

        # constant fixed-nuisance dict (host floats); merged with the floated dict per call
        self._fixed = {k: float(v) for k, v in FIXED_NUIS.items()}

    # ------------------------------------------------------------------
    # ABCMB raw C_l -> clik (6, lmax+1) muK^2 block  (BB/TB/EB = 0)
    # ------------------------------------------------------------------
    def abcmb_cls_to_clik(self, ClTT, ClTE, ClEE, l_arr):
        """raw dimensionless C_l on integer grid l_arr -> (..., 6, lmax+1) muK^2.
        Leading batch axes on Cl* are supported (l_arr shared, shape (n_ell,))."""
        l_arr = jnp.asarray(l_arr)
        fac = self.T_CMB_uK ** 2                    # raw C_l -> muK^2 (no l(l+1): clik wants C_l)
        keep = l_arr <= self.lmax
        idx = jnp.where(keep, l_arr, 0).astype(int)
        bshape = jnp.shape(ClTT)[:-1]
        zeros = jnp.zeros(bshape + (self.Lcol,))
        def scatter(Cl):
            return zeros.at[..., idx].add(jnp.where(keep, Cl * fac, 0.0))
        tt = scatter(ClTT); ee = scatter(ClEE); te = scatter(ClTE)
        bb = jnp.zeros_like(tt)
        # stack on a NEW axis just before the ell axis -> (..., 6, Lcol)
        return jnp.stack([tt, ee, bb, te, bb, bb], axis=-2)

    # ------------------------------------------------------------------
    # bare data chi^2 from clipy (single cosmology; vmap for a batch)
    # ------------------------------------------------------------------
    def _full_dict(self, nu_phys):
        """floated physical vector (21,) -> full 47-key nuisance dict for clipy."""
        d = {FLOAT_NAMES[i]: nu_phys[i] for i in range(len(FLOAT_NAMES))}
        d.update(self._fixed)
        return d

    def data_chi2(self, cls2d, nu_phys):
        """-2 * bare data logL (no priors) at one cosmology. cls2d: (6, lmax+1)."""
        logL = self.lkl(cls2d, self._full_dict(nu_phys), chi2_mode=True)
        return -2.0 * logL

    def prior_penalty(self, nu_phys):
        """sum of Gaussian + joint-SZ chi^2 penalties on the floated nuisances."""
        gauss = jnp.sum(self.prior_invvar * (nu_phys - self.prior_mean) ** 2)
        sz = nu_phys[self.i_ksz] + 1.6 * nu_phys[self.i_asz]
        gauss = gauss + ((sz - self.sz_mean) / self.sz_sig) ** 2
        return gauss

    def penalized_chi2(self, cls2d, nu_phys):
        return self.data_chi2(cls2d, nu_phys) + self.prior_penalty(nu_phys)

    # ------------------------------------------------------------------
    # inner nuisance profile (scaled coords): gradient-only nonmonotone Spectral
    # Projected Gradient (SPG). See profile() for the full rationale. Short version:
    # clipy's 2nd derivative is unreliable (1st-order AD is exact, verified) so every
    # Hessian-Newton variant failed; two nuisances rail at uniform-prior bounds (the
    # joint-SZ prior) so the method must be genuinely bound-constrained. SPG (projected
    # BB steps + GLL nonmonotone Armijo) is gradient-only and projects onto the box, and
    # matches scipy L-BFGS-B to <0.02 chi^2 by maxit=800 (offline tuning, B=8 spread).
    # ------------------------------------------------------------------
    @property
    def zlo(self):
        return (self.lo - self.start) / self.scale

    @property
    def zhi(self):
        return (self.hi - self.start) / self.scale

    def _obj_z(self, cls2d, z):
        """penalized chi^2 in SCALED coords z (nu = start + scale*z; well-conditioned,
        curvature ~ O(1) since scale = the per-param prior/ref sigma)."""
        return self.penalized_chi2(cls2d, self.start + self.scale * z)

    # no-ops retained for driver call-site stability (no external preconditioner needed)
    def compute_preconditioner(self, cls2d_ref=None, eigfloor=None):
        return None

    def set_preconditioner(self, Hprec=None):
        return None

    def profile(self, cls2d, z0=None, maxit=None, maxls=None, eigfloor=None):
        """Inner-profile the 21 floated nuisances at fixed theory cls2d (6,lmax+1).
        Gradient-only Spectral Projected Gradient (SPG: projected gradient with
        Barzilai-Borwein steps + Armijo) in the bounded scaled-z box. Returns
        (chi2_prof, nu_star_phys).

        Why this design: (1) clipy's 2nd derivative is unreliable (verified: 1st-order AD
        is exact, every Hessian-Newton variant failed, gap 2.8-14 chi^2), so the method
        must be gradient-only -- BB steps build curvature from consecutive gradients.
        (2) Two nuisances rail at their uniform-prior bounds (xi_sz_cib->1, ksz_norm->0,
        joint-SZ prior); where the data wants them far outside the box the boundary
        gradient is steep, so a sigmoid-reparam unconstrained BFGS stalls short of the
        bound and loses several chi^2, while projection sets them exactly at the
        bound. (3) Scaled-z (scale = prior sigma) is well-conditioned, so SPG converges
        like scipy L-BFGS-B (~100 iters). Smooth in cls -> clean envelope-theorem
        gradient. Differentiable in cls2d only through the final eval; callers wanting the
        gradient stop_gradient the cls input to the SPG, then differentiate penalized_chi2
        at the fixed optimum. eigfloor accepted but unused."""
        # default 800 (SPG: worst gap 0.017 chi^2 vs scipy across a B=8 spread;
        # 600->0.067, 400->0.13). PLF_MAXIT env override is for cheap debug smoke tests
        # only -- leave it unset in production so the inner profile stays at 800.
        if maxit is None:
            maxit = int(os.environ.get("PLF_MAXIT", "800"))
        maxls = self.maxls if maxls is None else maxls
        zlo, zhi, c1 = self.zlo, self.zhi, self.c1
        f = lambda zz: self._obj_z(cls2d, zz)
        gradf = jax.grad(f)
        z = jnp.zeros(len(FLOAT_NAMES)) if z0 is None else jnp.asarray(z0)
        g = gradf(z)
        AMIN, AMAX = 1e-10, 1e10
        MNM = 10                                                 # GLL nonmonotone window

        def outer(carry, _):
            z, g, alpha, fbuf = carry
            fref = jnp.max(fbuf)                                 # GLL: compare vs max of last MNM f's
            d = jnp.clip(z - alpha * g, zlo, zhi) - z            # spectral projected-gradient dir
            gd = jnp.dot(g, d)
            def ls(carry, _):                                    # NONMONOTONE Armijo: first accepted lam
                lam, accepted, z_acc = carry
                zt = z + lam * d
                ok = (f(zt) <= fref + c1 * lam * gd) & (~accepted)
                return (lam * 0.5, accepted | ok, jnp.where(ok, zt, z_acc)), None
            z_tiny = z + (0.5 ** maxls) * d
            (_, accepted, z_acc), _ = jax.lax.scan(ls, (1.0, False, z_tiny), None, length=maxls)
            z_new = jnp.where(accepted, z_acc, z_tiny)
            g_new = gradf(z_new)
            s = z_new - z; y = g_new - g
            sy = jnp.dot(s, y); ss = jnp.dot(s, s)
            alpha_new = jnp.where(sy > 1e-30, jnp.clip(ss / sy, AMIN, AMAX), AMAX)  # BB step
            fbuf_new = jnp.concatenate([fbuf[1:], f(z_new)[None]])
            return (z_new, g_new, alpha_new, fbuf_new), None

        fbuf0 = jnp.full(MNM, f(z))
        (z_star, _, _, _), _ = jax.lax.scan(outer, (z, g, jnp.asarray(1.0), fbuf0), None, length=maxit)
        nu_star = self.start + self.scale * z_star
        return self.penalized_chi2(cls2d, nu_star), nu_star

    def profile_batched(self, cls2d_B, z0_B=None):
        """vmap of profile over a leading batch axis. cls2d_B: (B,6,lmax+1).
        Returns (chi2_B (B,), nu_star_B (B,21))."""
        if z0_B is None:
            return jax.vmap(lambda c: self.profile(c))(cls2d_B)
        return jax.vmap(lambda c, z: self.profile(c, z0=z))(cls2d_B, z0_B)

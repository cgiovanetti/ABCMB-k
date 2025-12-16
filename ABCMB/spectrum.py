import numpy as np
import jax.numpy as jnp
import equinox as eqx
import jax
from jax import vmap, jit, config, grad, lax
from diffrax import diffeqsolve, ODETerm, Dopri5, Kvaerno3, Kvaerno5, Tsit5, SaveAt, PIDController, DiscreteTerminatingEvent
from jax.scipy.interpolate import RegularGridInterpolator
from functools import partial
from interpax import CubicSpline
from scipy.special import spherical_jn

from . import ABCMBTools as tools
from . import constants as cnst

import os
file_dir = os.path.dirname(__file__)

config.update("jax_enable_x64", True)

bessel_l_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/l.txt", dtype="int"))
bessel_x_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/x.txt"))

# 2D arrays of tabulated spherical functions over l and x axes.
bessel_phi0_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/phi0.txt"))
bessel_phi1_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/phi1.txt"))
bessel_phi2_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/phi2.txt"))

try:
    gpus = jax.devices('gpu')
    bessel_l_tab = jax.device_put(
        bessel_l_tab, device=gpus[0])
    bessel_x_tab = jax.device_put(
        bessel_x_tab, device=gpus[0])
    bessel_phi0_tab = jax.device_put(
        bessel_phi0_tab, device=gpus[0])
    bessel_phi1_tab = jax.device_put(
        bessel_phi1_tab, device=gpus[0])
    bessel_phi2_tab = jax.device_put(
        bessel_phi2_tab, device=gpus[0])
except: 
    pass


def phi0(i, x):
    """
    Compute spherical Bessel function φ₀.

    First integer argument i indicates we are computing this for l = bessel_l_tab[i]
    float argument x is the argument of the bessel function.

    Parameters:
    -----------
    i : int
        Index into bessel_l_tab for multipole ℓ
    x : float
        Argument of spherical Bessel function

    Returns:
    --------
    float
        φ₀(x) for multipole ℓ = bessel_l_tab[i]

    Notes:
    ------
    1312.2697 Eq. (3.19a)
    """
    # Annoyingly the following line is not jit safe...
    # For now I am passing in the idx corresponding to the theoretically desired l.
    #idx = jnp.where(bessel_l_tab == ell)[0][0].item()
    return tools.fast_interp(x, bessel_x_tab.min(), bessel_x_tab.max(), bessel_phi0_tab[:, i])

def phi1(i, x):
    """
    Compute spherical Bessel function φ₁.

    Parameters:
    -----------
    i : int
        Index into bessel_l_tab for multipole ℓ
    x : float
        Argument of spherical Bessel function

    Returns:
    --------
    float
        φ₁(x) for multipole ℓ = bessel_l_tab[i]

    Notes:
    ------
    1312.2697 Eq. (3.19a)
    """
    #idx = jnp.where(bessel_l_tab == ell)[0][0].item()
    return tools.fast_interp(x, bessel_x_tab.min(), bessel_x_tab.max(), bessel_phi1_tab[:, i])

def phi2(i, x):
    """
    Compute spherical Bessel function φ₂.

    Parameters:
    -----------
    i : int
        Index into bessel_l_tab for multipole ℓ
    x : float
        Argument of spherical Bessel function

    Returns:
    --------
    float
        φ₂(x) for multipole ℓ = bessel_l_tab[i]

    Notes:
    ------
    1312.2697 Eq. (3.19a)
    """
    #idx = jnp.where(bessel_l_tab == ell)[0][0].item()
    return tools.fast_interp(x, bessel_x_tab.min(), bessel_x_tab.max(), bessel_phi2_tab[:, i])

class SpectrumSolver(eqx.Module):
    """
    CMB angular power spectrum computation.

    Computes temperature and polarization angular power spectra by
    integrating transfer functions over wavenumber and time.

    Attributes:
    -----------
    ells : jnp.array
        Multipole values for output power spectra
    ells_indices : jnp.array
        Indices into bessel_l_tab corresponding to ells
    lensing_ells : jnp.array
        Extended multipole range for lensing calculations
    lensing_ells_indices : jnp.array
        Indices into bessel_l_tab for lensing multipoles
    lensing : bool
        Whether to include gravitational lensing effects
    k_axis_transfer : jnp.array
        Wavenumber grid for transfer function integration (units: Mpc^{-1})
    k_axis_Pk_output : jnp.array
        Wavenumber grid for matter power spectrum output (units: Mpc^{-1})
    k_pivot : float
        Pivot scale for primordial power spectrum normalization (units: Mpc^{-1}, default: 0.05)
    scale_sw : float
        Multiplicative factor for Sachs-Wolfe term (default: 1.0)
    scale_isw : float
        Multiplicative factor for integrated Sachs-Wolfe term (default: 1.0)
    scale_dop : float
        Multiplicative factor for Doppler term (default: 1.0)
    scale_pol : float
        Multiplicative factor for polarization term (default: 1.0)

    Methods:
    --------
    primordial_spectrum : Compute primordial power spectrum
    Pk_lin : Compute linear matter power spectrum
    get_Cl : Compute angular power spectra for multiple ℓ
    Cl_one_ell : Compute angular power spectrum for single ℓ
    integrand_T0 : Compute SW+ISW temperature source integrand
    integrand_T1 : Compute ISW temperature source integrand
    integrand_T2 : Compute polarization temperature source integrand
    integrand_E : Compute E-mode polarization source integrand
    """

    ells         : jnp.array
    ells_indices : jnp.array

    lensing_ells : jnp.array
    lensing_ells_indices : jnp.array

    lensing : bool

    k_axis_transfer  : jnp.array
    k_axis_Pk_output : jnp.array

    k_pivot    : float = 0.05 # In 1/Mpc
    scale_sw  : float = 1.
    scale_isw : float = 1.
    scale_dop : float = 1.
    scale_pol : float = 1.

    def __init__(self,
                 ellmin=2,
                 ellmax=2500,
                 lensing=False,
                 k_axis_transfer=jnp.geomspace(1.e-4, 0.4, 2500),
                 k_axis_Pk_output=jnp.geomspace(1.e-4, 0.1, 100),
                 k_pivot=0.05,
                 scale_sw=1,
                 scale_isw=1,
                 scale_dop=1,
                 scale_pol=1):
        """
        Initialize CMB spectrum solver.

        Sets up multipole range, lensing configuration, and source term switches
        for computing angular power spectra.

        Parameters:
        -----------
        ellmin : int, optional
            Minimum multipole (default: 2)
        ellmax : int, optional
            Maximum multipole (default: 2500)
        lensing : bool, optional
            Whether to include lensing effects (default: True)
        k_pivot : float, optional
            Pivot scale for primordial spectrum (units: Mpc^{-1}, default: 0.05)
        scale_sw : float, optional
            Switch for Sachs-Wolfe term (default: 1)
        scale_isw : float, optional
            Switch for integrated Sachs-Wolfe term (default: 1)
        scale_dop : float, optional
            Switch for Doppler term (default: 1)
        scale_pol : float, optional
            Switch for polarization term (default: 1)
        """

        self.lensing = lensing

        self.ells = jnp.arange(ellmin, ellmax+1)
        ell_idx_min = jnp.where(bessel_l_tab<=ellmin)[0][-1]
        ell_idx_max = jnp.where(bessel_l_tab>=ellmax)[0][0]
        self.ells_indices = jnp.arange(ell_idx_min, ell_idx_max+1)
        
        if self.lensing:
            lensing_ellmax = ellmax+500
            lensing_ell_idx_max = jnp.where(bessel_l_tab>=lensing_ellmax)[0][0]
            self.lensing_ells = jnp.arange(ellmin, lensing_ellmax+1)
            self.lensing_ells_indices = jnp.arange(ell_idx_min, lensing_ell_idx_max+1)
        else:
            self.lensing_ells = self.ells
            self.lensing_ells_indices = self.ells_indices

        self.k_axis_transfer = k_axis_transfer
        self.k_axis_Pk_output = k_axis_Pk_output
        self.k_pivot    = k_pivot

        self.scale_sw  = scale_sw
        self.scale_isw = scale_isw
        self.scale_dop = scale_dop
        self.scale_pol = scale_pol

    def primordial_spectrum(self, k, params):
        """
        Compute primordial curvature power spectrum.

        Parameters:
        -----------
        k : float or array
            Wavenumber (units: Mpc^{-1})
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Primordial power spectrum P_R(k), units Mpc^3
        """
        return params['A_s']*(k/self.k_pivot)**(params['n_s']-1.) * (2*jnp.pi**2/k**3)

    def Pk_lin(self, k, z, PT, params):
        """
        Compute linear matter power spectrum at wavenumbers k and redshift z.

        Parameters:
        -----------
        k : float or array
            Wavenumber (Mpc^{-1})
        z : float
            Redshift to evaluate.
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Linear matter power spectrum P(k, z), units Mpc^3
        """

        lna = -jnp.log(1.+z)
    
        # vmapped interpolation over Nk (columns of the 2D arrays)
        interp_over_lna = jax.vmap(
            lambda y: jnp.interp(lna, PT.lna, y),
            in_axes=1  # loop over columns
        )

        delta_m_lna = interp_over_lna(PT.delta_m)  # shape (Nk,)

        # now interpolate over k
        delta_m = jnp.interp(k, PT.k, delta_m_lna)

        return delta_m**2 * self.primordial_spectrum(k, params)

    def Pk_cb(self, k, z, PT, params):
        """
        Compute linear Baryon+DarkMatter power spectrum at wavenumbers k and redshift z.
        Does not include any other massive species present.

        Parameters:
        -----------
        k : float or array
            Wavenumber (Mpc^{-1})
        z : float
            Redshift to evaluate.
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Linear Baryon+DarkMatter power spectrum P_cb(k, z), units Mpc^3
        """

        lna = -jnp.log(1.+z)
    
        # vmapped interpolation over Nk (columns of the 2D arrays)
        interp_over_lna = jax.vmap(
            lambda y: jnp.interp(lna, PT.lna, y),
            in_axes=1  # loop over columns
        )

        delta_dm_lna = interp_over_lna(PT.delta_dm)  # shape (Nk,)
        delta_b_lna   = interp_over_lna(PT.delta_b)    # shape (Nk,)

        # now interpolate over k
        delta_dm = jnp.interp(k, PT.k, delta_dm_lna)
        delta_b   = jnp.interp(k, PT.k, delta_b_lna)

        # total matter overdensity
        delta_m = (
            params['omega_b']   * delta_b +
            params['omega_cdm'] * delta_dm
        ) / params['omega_m']

        return delta_m**2 * self.primordial_spectrum(k, params)

    def lensing_power_spectrum(self, k, lna, PT, BG, params):
        """
        Computes the lensing power spectrum at wavenumbers k and redshift z.
        Eq.(3.15) in astro-ph/0601594

        Parameters:
        -----------
        k : float or array
            Wavenumber (Mpc^{-1})
        lna : float
            Scale factor
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Lensing matter power spectrum P(k, z), dimensionless.
        """
        a = jnp.exp(lna)
        z = 1./a - 1.
        aH = BG.aH(lna, params)

        Omega_m = params["omega_m"]/params["h"]**2
        Omega_L = params["omega_Lambda"]/params["h"]**2

        # Matter fraction over time after equality. 1 at early times and becomes Om0 today. 
        Om = (Omega_m * (1.+z)**3)/ ((Omega_m * (1.+z)**3) + Omega_L)

        Pk = self.Pk_lin(k, z, PT, params) # Mpc^3

        return 9./8./jnp.pi**2 * Om**2 * aH**4 * Pk / k

    def lensing_Cl(self, ells, PT, BG, params):
        """
        Angular lensing power spectrum at multipole ell.

        IMPORTANT: Assumes Limber approximation throughout, even at ell=2.

        Eq.(3.14) in astro-ph/0601594, except shifts ell -> ell+1/2 to match CLASS.

        Parameters:
        -----------
        ell : float or array
            Multipole
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        float or array
            Angular lensing matter power spectrum Cl^phiphi, dimensionless.
        """

        coeff = 8.*jnp.pi**2/(ells+0.5)**3
        chi = lambda lna : BG.tau0 - BG.tau(lna)

        def integrand_func(lna):
            k = (ells+0.5)/chi(lna)
            window = (chi(BG.lna_rec) - chi(lna))/chi(BG.lna_rec)/chi(lna)
            res = chi(lna)/BG.aH(lna, params) * window**2 * self.lensing_power_spectrum(k, lna, PT, BG, params)
            return res

        lna_axis = jnp.linspace(BG.lna_rec, 0., 500)
        integrand = vmap(integrand_func)(lna_axis)
        integrand = jnp.nan_to_num(integrand, nan=0.)
        return coeff*jnp.trapezoid(integrand, lna_axis, axis=0)

    def lensed_Cls(self, ells, ClTT_unlensed, ClTE_unlensed, ClEE_unlensed, PT, BG, params):
        """
        Compute lensed CMB power spectra.

        Applies gravitational lensing corrections to unlensed temperature
        and polarization power spectra using Wigner rotation matrices.

        Parameters:
        -----------
        ells : array
            Multipole values
        ClTT_unlensed : array
            Unlensed temperature power spectrum
        ClTE_unlensed : array
            Unlensed temperature-E-mode cross spectrum
        ClEE_unlensed : array
            Unlensed E-mode polarization power spectrum
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        tuple
            (ClTT, ClTE, ClEE) lensed power spectra
        """
        #beta = jnp.linspace(0., jnp.pi/16., 5000)
        #mu = jnp.cos(beta)
        # CLASS samples angle uniformly
        # 500 points is enough for lmax < 4000
        theta = jnp.linspace(0., jnp.pi/16., 500)

        # Flip mu so that mu is in ascending order, works better for trapz.
        mu = jnp.flip(jnp.cos(theta))

        # Compute lensing Cl
        Clpp = self.lensing_Cl(ells, PT, BG, params)

        # Wigner matrices needed in general and for temperature
        # Note that for all wigner matrices, the symmetry relation is dnm = (-1)^(m-n) x dmn
        d00 = tools.d00(mu, ells)
        d11 = tools.d1n(mu, ells, 1)
        d1m1 = tools.d1n(mu, ells, -1)
        d2m2 = tools.d2n(mu, ells, -2)
        dm11 = d1m1

        # Wigner matrices needed for polarization
        d22 = tools.d2n(mu, ells, 2)
        d31 = tools.d3n(mu, ells, 1)
        d40 = tools.d4n(mu, ells, 0)
        d3m3 = tools.d3n(mu, ells, -3)
        d4m4 = tools.d4n(mu, ells, -4)
        d20 = tools.d2n(mu, ells, 0)
        d3m1 = tools.d3n(mu, ells, -1)
        d4m2 = tools.d4n(mu, ells, -2)
        d02 = d20
        dm24 = d4m2

        # Lensing angular correlation function
        Cgl  = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ells*(ells+1)*Clpp*d11, axis=1
        ) # Nmu
        Cgl2 = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ells*(ells+1)*Clpp*dm11, axis=1
        ) # Nmu
        sigma2     = Cgl[-1] - Cgl
        Cgl    = Cgl[:, None]
        Cgl2   = Cgl2[:, None]
        sigma2 = sigma2[:, None]

        llp1   = ells*(ells+1)

        X000       = jnp.exp(-llp1*sigma2/4)
        X000_prime = -llp1/4.*X000
        X220       = 1./4.*jnp.sqrt((ells+2)*(ells-1)*ells*(ells+1))*jnp.exp(-(llp1-2)*sigma2/4.)
        X022       = jnp.exp(-(llp1-4)*sigma2/4)
        X022_prime = -(llp1-4)/4*X022
        X121       = -1./2.*jnp.sqrt((ells+2)*(ells-1))*jnp.exp(-(llp1-8./3.)*sigma2/4.)
        X132       = -1./2.*jnp.sqrt((ells+3)*(ells-2))*jnp.exp(-(llp1-20./3.)*sigma2/4.)
        X242       = 1./4.*jnp.sqrt((ells+4)*(ells+3)*(ells-2)*(ells-3))*jnp.exp(-(llp1-10.)*sigma2/4.)

        # Correlation functions
        ksi = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClTT_unlensed * (
                X000**2 * d00 \
                + 8./ells/(ells+1)*Cgl2*X000_prime**2*d1m1 \
                + Cgl2**2 * (X000_prime**2*d00 + X220**2*d2m2) \
                - d00
            ), 
            axis=1
        )

        ksip = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClEE_unlensed * (
                X022**2 * d22 \
                + 2*Cgl2*X132*X121*d31 \
                + Cgl2**2 * (X022_prime**2*d22 + X242*X220*d40) \
                - d22
            ), 
            axis=1
        )

        ksim = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClEE_unlensed * (
                X022**2 * d2m2 \
                + Cgl2*(X121**2*d1m1 + X132**2*d3m3) \
                + 1./2.*Cgl2**2 * (2*X022_prime**2*d2m2 + X220**2*d00 + X242**2*d4m4) \
                - d2m2
            ), 
            axis=1
        )

        ksix = 1./4./jnp.pi * jnp.sum(
            (2.*ells+1)*ClTE_unlensed * (
                X022*X000*d02 \
                + Cgl2 * 2*X000_prime/jnp.sqrt(llp1) * (X121*d11 + X132*d3m1) \
                + 1./2.*Cgl2**2 * ((2*X022_prime*X000_prime+X220**2)*d20+X220*X242*dm24) \
                - d02
            ), 
            axis=1
        )
        
        ClTT = 2.*jnp.pi * jnp.trapezoid(ksi[:, None]*d00, mu, axis=0) + ClTT_unlensed
        ClTE = 2.*jnp.pi * jnp.trapezoid(ksix[:, None]*d20, mu, axis=0) + ClTE_unlensed
        ClEE = 1./2. * 2.*jnp.pi * jnp.trapezoid(ksip[:, None]*d22+ksim[:, None]*d2m2, mu, axis=0) + ClEE_unlensed

        return (ClTT, ClTE, ClEE)

    def get_Cl(self, PT, BG, params):
        """
        Compute angular power spectra for multiple multipoles using lax.scan.

        Parameters:
        -----------
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        tuple
            (ClTT, ClTE, ClEE) angular power spectra
        """

        if jax.default_backend() == "gpu":
            
            tt_raw, te_raw, ee_raw = vmap(self.Cl_one_ell, in_axes=(0, None, None, None))(self.lensing_ells_indices, PT, BG, params)

        else:
            
            def scan_fun(_, idx):
                cltt, clte, clee = self.Cl_one_ell(idx, PT, BG, params)
                return None, jnp.array([cltt, clte, clee])

            _, Cls_raw = lax.scan(scan_fun, None, self.lensing_ells_indices)
            tt_raw = Cls_raw[:, 0]
            te_raw = Cls_raw[:, 1]
            ee_raw = Cls_raw[:, 2]

        # Cubic spline for smooth Cl over user requested ells
        lensing_ells = bessel_l_tab[self.lensing_ells_indices]
        tt_unlensed = CubicSpline(lensing_ells, tt_raw, check=False)(self.lensing_ells)
        te_unlensed = CubicSpline(lensing_ells, te_raw, check=False)(self.lensing_ells)
        ee_unlensed = CubicSpline(lensing_ells, ee_raw, check=False)(self.lensing_ells)

        def get_lensed_Cls():
            tt_lensed, te_lensed, ee_lensed = self.lensed_Cls(self.lensing_ells, tt_unlensed, te_unlensed, ee_unlensed, PT, BG, params)
            return (tt_lensed[self.ells-2], te_lensed[self.ells-2], ee_lensed[self.ells-2])

        def get_unlensed_Cls():
            return (tt_unlensed[self.ells-2], te_unlensed[self.ells-2], ee_unlensed[self.ells-2])

        return lax.cond(
            self.lensing,
            get_lensed_Cls,
            get_unlensed_Cls
        )

    def Cl_one_ell(self, idx, PT, BG, params):
        """
        Computes angular power spectrum for single multipole.

        Integrates transfer functions over wavenumber.

        Parameters:
        -----------
        idx : int
            Index into bessel_l_tab for multipole ℓ
        PT : perturbations.PerturbationTable
            Perturbation evolution table
        BG : background.Background
            Background cosmology module
        params : dict
            Dictionary of input and derived parameters

        Returns:
        --------
        tuple
            (C_ℓ^TT, C_ℓ^TE, C_ℓ^EE) angular power spectra
        """
        l = bessel_l_tab[idx]
        k_T0_axis = self.k_axis_transfer
        lna_axis = PT.lna

        ### TRANSFER FUNCTION ###
        # Background quantities, all Nlna 1D vectors
        tau0 = BG.tau0
        tau = BG.tau(lna_axis)
        g   = vmap(BG.visibility,in_axes=[0,None])(lna_axis, params)
        g_prime = vmap(grad(BG.visibility,argnums=0),in_axes=[0,None])(lna_axis, params) # Derivative of g w.r.t. lna
        aH  = BG.aH(lna_axis, params)
        expmkappa = vmap(BG.expmkappa)(lna_axis)
        aH_dot = BG.aH_prime(lna_axis, params) * aH # Derivative of aH w.r.t. conformal time tau.

        g         = g[:, None]
        g_prime   = g_prime[:, None]
        aH        = aH[:, None]
        expmkappa = expmkappa[:, None]
        aH_dot    = aH_dot[:, None]

        # Perturbations, all (Nk, Nlna) 2D vectors
        # Cubic Spline is necessary here for accuracy. 
        interp_column = lambda col : CubicSpline(jnp.log10(PT.k), col, check=False)(jnp.log10(k_T0_axis))

        # Found that this is much much faster than RegularGridInterpolator
        delta_g       = vmap(interp_column, in_axes=0, out_axes=0)(PT.delta_g)
        theta_b       = vmap(interp_column, in_axes=0, out_axes=0)(PT.theta_b)
        theta_b_prime = vmap(interp_column, in_axes=0, out_axes=0)(PT.theta_b_prime)
        sigma_g       = vmap(interp_column, in_axes=0, out_axes=0)(PT.sigma_g)
        Gg0           = vmap(interp_column, in_axes=0, out_axes=0)(PT.Gg0)
        Gg2           = vmap(interp_column, in_axes=0, out_axes=0)(PT.Gg2)
        eta           = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_eta)
        eta_prime     = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_eta_prime)
        alpha         = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_alpha)
        alpha_prime   = vmap(interp_column, in_axes=0, out_axes=0)(PT.metric_alpha_prime)

        # Source terms
        sourceT0 = self.scale_sw * g * (delta_g/4. + aH*alpha_prime) \
                + self.scale_isw * (
                    g * (eta - aH*alpha_prime - 2.*aH*alpha) \
                    + 2.*expmkappa * (aH*eta_prime - aH_dot*alpha - aH**2*alpha_prime)
                ) \
                + self.scale_dop * (
                    aH * (g*((theta_b_prime / k_T0_axis**2) + alpha_prime) \
                    + g_prime*((theta_b / k_T0_axis**2) + alpha))
                )

        sourceT1 = self.scale_isw * expmkappa * \
                ((aH*alpha_prime + 2.*aH*alpha - eta) * k_T0_axis)

        sourceT2 = self.scale_pol * g * (2*sigma_g + Gg0 + Gg2) / 8.

        sourceE  = jnp.sqrt(6) * sourceT2

        # Bessel functions
        chiT0 = jnp.outer(tau0-tau, k_T0_axis)

        # Note: our phi0's seem to be accurate up to lmax ~ 3000 or so.
        phi0_tab = phi0(idx, chiT0)
        transferT0 = jnp.trapezoid(
            sourceT0 / aH * phi0_tab,
            #sourceT0 / aH * spherical_jn(l, chiT0),
            lna_axis, axis=0
        )

        transferT1 = jnp.trapezoid(
            sourceT1 / aH * phi1(idx, chiT0),
            lna_axis, axis=0
        )

        transferT2 = jnp.trapezoid(
            sourceT2 / aH * phi2(idx, chiT0),
            lna_axis, axis=0
        )
        
        # TODO: Fix this!
        # epsilon_tab = jnp.sqrt(3./8.*(l+2)*(l+1)*l*(l-1)) / chiT0**2
        # #epsilon_tab = epsilon_tab.at[-1].set(jnp.zeros(k_T0_axis.size)) # Filter out the chiT0=0 part
        # epsilon_tab = epsilon_tab.at[-1].set(
        # jnp.where(
        #     l == 2,
        #     jnp.ones(k_T0_axis.size)/15.,
        #     jnp.zeros(k_T0_axis.size)
        # )
        # )
        # epsilon_tab *= phi0_tab

        epsilon_tab = phi0_tab / chiT0**2

        # Mask out the x=0 part. For l=2 this is 1/15, and for l>2 it's 0.
        epsilon_tab = epsilon_tab.at[-1].set(
            jnp.where(
                l == 2,
                jnp.ones(k_T0_axis.size)/15.,
                jnp.zeros(k_T0_axis.size)
            )
        )
        epsilon_tab *= jnp.sqrt(3./8.*(l+2)*(l+1)*l*(l-1))

        transferE = jnp.trapezoid(
            sourceE / aH * epsilon_tab,
            lna_axis, axis=0
        )

        del chiT0

        transferT = transferT0 + transferT1 + transferT2
        ### END OF TRANSFER FUNCTION ###

        ### LINE OF SIGHT INTEGRAL ###
        integrandTT = 4.*jnp.pi * params['A_s'] * (k_T0_axis/self.k_pivot)**(params['n_s']-1.) * transferT**2 / k_T0_axis
        integrandTE = 4.*jnp.pi * params['A_s'] * (k_T0_axis/self.k_pivot)**(params['n_s']-1.) * transferT*transferE / k_T0_axis
        integrandEE = 4.*jnp.pi * params['A_s'] * (k_T0_axis/self.k_pivot)**(params['n_s']-1.) * transferE**2 / k_T0_axis
        
        return (
            jnp.trapezoid(integrandTT, k_T0_axis),
            jnp.trapezoid(integrandTE, k_T0_axis),
            jnp.trapezoid(integrandEE, k_T0_axis)
        )
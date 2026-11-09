import jax
import jax.numpy as jnp
from jax import vmap

import equinox as eqx

from diffrax import diffeqsolve, ODETerm, Tsit5, Kvaerno5, PIDController, SaveAt, Event, ForwardMode

from . import thermo  
from . import const

rho_massless_BE_v = vmap(
    thermo.rho_massless_BE, in_axes=(0, None, None)
)
rho_massless_FD_v = vmap(
    thermo.rho_massless_FD, in_axes=(0, None, None)
)

class BackgroundModel(eqx.Module): 
    """Background model.

    Attributes
    ----------
    decoupled : bool, optional
        Whether neutrinos are always decoupled. Default is `False`.
    use_FD : bool, optional
        Whether to use Fermi-Dirac statistics for neutrinos, or a Maxwell-Boltzmann distribution. Default is `True`.
    collision_me : bool, optional
        Finite electron mass correction in energy transfer collision terms. Default is `True`.
    LO : bool, optional
        Whether to use leading order QED correction. Default is `True`.
    NLO : bool, optional
        Whether to use next-to-leading order QED correction. Default is True.
    g_chi : float, optional
        Internal degrees of freedom of an EM-coupled thermal relic chi (Dirac fermion 4). Default 0 = no chi.
    chi_is_boson : bool, optional
        Bose-Einstein chi if True, Fermi-Dirac if False. Default is `False`.
    adjoint : diffrax.adjoint
        Adjoint mode for diffrax solves.  Default is ForwardMode.
    """

    decoupled : bool
    use_FD : bool
    collision_me : bool
    LO : bool
    NLO : bool
    g_chi : float = eqx.field(static=True)
    chi_is_boson : bool = eqx.field(static=True)

    adjoint : "diffrax.adjoint" = eqx.field(static=True)

    def __init__(self, decoupled=False, use_FD=True, collision_me=True, LO=True, NLO = True, g_chi=0.,
                 chi_is_boson=False, adjoint = ForwardMode):

        self.decoupled = decoupled
        self.use_FD = use_FD
        self.collision_me = collision_me
        self.LO = LO
        self.NLO = NLO
        self.g_chi = g_chi
        self.chi_is_boson = chi_is_boson
        self.adjoint = adjoint

    @eqx.filter_jit
    def __call__(
        self, Delt_Neff_init, m_chi=0., T_start=const.T_start,
        T_end=const.T_end, rtol=1e-8, atol=1e-10,
        solver=None, max_steps=None, n_out=4096
    ):
        """ Calculate thermodynamics given an initial :math:`\\Delta N_\\mathrm{eff}`.

        Parameters
        ----------
        Delt_Neff_init : float
            Initial :math:`\\Delta N_\\mathrm{eff}`.  Can be positive or negative.  
        m_chi : float, optional
            Mass of the EM-coupled relic chi in MeV (used when g_chi != 0; never 0 then).
        T_EM_init : float
            Initial EM (and neutrino) temperature. Default is `const.T_start`. 
        T_EM_end : float 
            Final EM temperature to terminate integration at. Default is
            `const.T_end`. 
        rtol : float, optional
            Relative tolerance of the abundance solver. Default is `1e-8`.  
        atol : float, optional
            Absolute tolerance of the abundance solver. Default is `1e-10`. 
        max_steps : int, optional
            Maximum number of steps taken by the solver. Default is `4096`.
            Increasing this slows down the code, while decreasing this could
            mean that the solver cannot complete the solution.
        n_out : int, optional
            Number of log-spaced output times (first step to the end), read
            from the solver's dense interpolant. Default is `4096`.
        solver : Diffrax ODE solver 
            The Diffrax ODE solver to use. Default: Tsit5, or the implicit
            Kvaerno5 with chi (its annihilation makes the T_g equation stiff).

        Returns
        -------
        t_vec : array_like
            Times in s at which thermodynamics are saved.
        a_vec : array_like
            Scale factor at each point in time.
        rho_g_vec : array_like
            Energy density of photons in MeV^4 at each point in time.
        rho_nu_vec : array_like
            Energy density of one species of neutrinos in MeV^4 at each point in time.
        rho_extra_vec : array_like
            Energy density in MeV^4 of extra species at each point in time.
        """
        print('`\\         /´  ||||        ||||  |||||     ||||  ||||   ||||')
        print(' /\\_______/\\   ||||        ||||  |||||||   ||||   |||| ||||') 
        print(' ) __` ´__ (   ||||        ||||  |||| |||| ||||    |||||||')
        print('/  `-|_|-´  \\  ||||        ||||  ||||  |||| |||    ||||||| ')
        print('/   (_x_)   \\  ||||||||||  ||||  ||||   |||||||   |||| ||||')
        print('  )  `-´  (    ||||||||||  ||||  ||||    ||||||  ||||   ||||')
        print(' ')

        print('Compiling thermodynamics model...')

        if solver is None:          # chi annihilation makes the T_g equation stiff (Tsit5 fails at 20 MeV)
            solver = Kvaerno5() if self.g_chi != 0. else Tsit5()
        if max_steps is None:
            max_steps = 512 if self.g_chi == 0. else 1024

        lna_init = 0. 
        T_EM_init = T_start
        T_nu_init = T_EM_init 

        rho_extra_init = (7/8) * (4/11)**(4/3) * (
            thermo.rho_massless_BE(T_EM_init, 0., 2)
        ) * Delt_Neff_init

        Y0 = (lna_init, T_EM_init, T_nu_init)
        
        def T_EM_check(t, y, args, **kwargs): 
            return y[1] < T_end
            
        sol = diffeqsolve(
            ODETerm(self.dY), solver, args=(lna_init, rho_extra_init, m_chi),
            t0=0., t1=jnp.inf, dt0=None, y0=Y0, 
            saveat=SaveAt(steps=True, dense=True), event=Event(T_EM_check),
            stepsize_controller = PIDController(
                rtol=rtol, atol=atol
            ),
            max_steps=max_steps,
            adjoint = self.adjoint()
        )

        # The last legitimate step made by the solver, when T_g drops below
        # T_end (sol.ys has max_steps entries; the unused ones are inf).
        last_step_ind = jnp.max(
            jnp.argwhere(
                sol.ys[1] < T_end,
                size=max_steps
            )[:,0]
        )

        # Output on n_out log-spaced times from the first step to the event
        # step, read from the solver's dense (4th-order) interpolant rather
        # than on the adaptive steps themselves: the abundance network
        # interpolates a(t) and T(t) on this grid, and on the raw steps that
        # error (~1e-4 mid-step) moved with the step sequence, i.e. with the
        # inputs (1.5e-5 roughness in Y_p against Delt_Neff_init).
        t_vec = jnp.exp(jnp.linspace(
            jnp.log(sol.ts[0]), jnp.log(sol.ts[last_step_ind]), n_out
        ))
        # exp(log(.)) can land one ulp past the last step, outside the interpolant
        t_vec = t_vec.at[0].set(sol.ts[0]).at[-1].set(sol.ts[last_step_ind])
        lna_vec, T_EM_vec, T_nu_vec = jax.vmap(sol.evaluate)(t_vec)

        a_vec = jnp.exp(lna_vec)
        rho_g_vec = rho_massless_BE_v(T_EM_vec, 0., 2)
        rho_nu_vec = rho_massless_FD_v(T_nu_vec, 0., 2)

        # Trivial relation between rho_extra and a, normalized at the initial
        # state a = exp(lna_init) as in dY.
        rho_extra_vec = rho_extra_init * (jnp.exp(lna_init)**4 / a_vec**4)

        # Rescale a so that the present day CMB temperature is correct.
        final_a = const.T0CMB / T_EM_vec[-1]
        a_vec  *= final_a / a_vec[-1]

        P_extra_vec = rho_extra_vec / 3

        T_g_vec  = thermo.T_g(rho_g_vec)
        # chi goes out in the new-physics slot (rho_NP, P_NP) with the inert radiation: the abundance network
        # builds H(t) and a(t) from the standard EM sector + 3 rho_nu + rho_NP, so chi is counted once there,
        # while dY puts it in the EM sector for the photon heating.
        rho_extra_vec = rho_extra_vec + thermo.rho_chi_v(T_g_vec, m_chi, self.g_chi, self.chi_is_boson)
        P_extra_vec = P_extra_vec + thermo.p_chi_v(T_g_vec, m_chi, self.g_chi, self.chi_is_boson)
        rho_tot_vec = (
            thermo.rho_EM_std_v(T_g_vec) + 3 * rho_nu_vec + rho_extra_vec
        )

        Neff_vec = thermo.N_eff(rho_tot_vec, rho_g_vec)

        return (
            t_vec, a_vec, rho_g_vec, rho_nu_vec, 
            rho_extra_vec, P_extra_vec, Neff_vec
        )
    
    @eqx.filter_jit
    def dY(self, t, Y, args): 
        """ Differential equation for background quantities. 

        Parameters
        ----------
        t : float
            Time in s.  
        Y : tuple of floats
            The values of :math:`\\log a`, :math:`T_\\gamma` and :math:`T_\\nu`. 
        args : tuple of floats
            The initial value of :math:`\\log a`, and the initial energy density in MeV^4 of the extra inert relativistic species.

        Returns
        -------
        The time derivative of the quantities specified in `Y`. 
        """

        lna, T_g, T_nu = Y
        lna_init, rho_extra_init, m_chi = args

        # chi is part of the EM sector: its energy density and heat capacity enter the T_g equation, so its
        # annihilation heats the photons relative to the decoupled neutrinos (lower Neff, shifted eta and n/p).
        rho_chi, drho_chi_dT_g = thermo.rho_chi_and_deriv(T_g, m_chi, self.g_chi, self.chi_is_boson)
        p_chi = thermo.p_chi(T_g, m_chi, self.g_chi, self.chi_is_boson)
        rho_EM = thermo.rho_EM_std(T_g, LO=self.LO, NLO=self.NLO) + rho_chi
        rho_plus_p_EM = thermo.rho_plus_p_EM_std(T_g, LO=self.LO, NLO=self.NLO) + rho_chi + p_chi
        drho_EM_dT_g = thermo.drho_EM_dT_g_std(T_g, LO=self.LO, NLO=self.NLO) + drho_chi_dT_g

        rho_nu = 3*thermo.rho_nue_std(T_nu)
        rho_plus_p_nu = (4/3) * rho_nu
        drho_nu_dT_nu = 3*thermo.drho_nue_dT_nue_std(T_nu)

        rho_extra = rho_extra_init * jnp.exp(lna_init)**4 / jnp.exp(lna)**4 

        H = thermo.Hubble(rho_EM + rho_nu + rho_extra)

        C_rho_nue, C_rho_numu, _, _ = thermo.collision_terms_std(
            T_g, T_nu, T_nu, decoupled=self.decoupled, use_FD=self.use_FD, collision_me=self.collision_me
        )

        drho_EM_dt = -3 * H * rho_plus_p_EM - C_rho_nue - 2*C_rho_numu
        drho_nu_dt  = -3 * H * rho_plus_p_nu + C_rho_nue + 2*C_rho_numu

        dT_g_dt = drho_EM_dt / drho_EM_dT_g
        dT_nu_dt = drho_nu_dt / drho_nu_dT_nu

        return H, dT_g_dt, dT_nu_dt 

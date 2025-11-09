from jax import jit, config
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np
import equinox as eqx

import jax

import sys
import os
file_dir = os.path.dirname(__file__)

from .hyrex import hyrex
from . import cosmology, perturbations, spectrum, model_specs
from . import constants as cnst
from . import AbstractSpecies as AS
from .ABCMBTools import bilinear_interp

from .linx.background import BackgroundModel
from .linx.abundances import AbundanceModel
from .linx.nuclear import NuclearRates
from .linx import const as linxconst

config.update("jax_enable_x64", True)

class Model(eqx.Module):
    """
    Model configuration and computation manager.

    Creates instances of fluid species based on user input and organizes
    them for computation. Manages the full pipeline from background
    evolution through CMB power spectrum computation.

    Methods:
    --------
    run_cosmology : Compute CMB angular power spectra
    get_PTBG : Get perturbation table and background cosmology
    get_BG : Get background cosmology
    add_derived_parameters : Compute derived parameters
    """

    PE : perturbations.PerturbationEvolver
    SS : spectrum.SpectrumSolver
    RM : hyrex.recomb_model

    species_list       : tuple = ()
    perturbations_list : tuple = ()

    bbn_type                : str = ""
    linx_reaction_net       : str = ""
    
    PArthENoPE_CLASS_table  : Array #= eqx.field(converter=jnp.asarray)

    return_PTBG : bool

    ### ADDING SPECIES: add has_ parameter and add condition to append to tuple.
    # In the init, all species that are present within the model should be set to True.
    # All couplings present between species should be set to true. 
    def __init__(self,
                 precision_in = {},
                 user_species=None,
                 bbn_type = "",
                 linx_reaction_net = "key_PRIMAT_2023",
                 return_PTBG=False,
                 ):
        """
        Initialize Model instance.

        Sets up fluid species, recombination model, and spectrum solver
        based on configuration parameters.

        Parameters:
        -----------
        ellmin : int, optional
            Minimum multipole for CMB spectrum (default: 2)
        ellmax : int, optional
            Maximum multipole for CMB spectrum (default: 2500)
        lensing : bool, optional
            Whether to include lensing effects (default: False)
        has_MassiveNeutrinos : bool, optional
            Whether to include massive neutrinos (default: False)
        return_PTBG : bool, optional
            Whether to return perturbation table and background (default: False)
        bbn_type : str, optional
            BBN calculation method: "Table", "LINX", or "" for manual (default: "")
        linx_reaction_net : str, optional
            Nuclear reaction network for LINX (default: "key_PRIMAT_2023")
        """

        # Fill in all user defined and missing precision parameters
        precision = model_specs.load_specs(precision_in)

        # Populate all species
        self.species_list, self.perturbations_list = model_specs.populate_species(
            user_species,
            precision,
        )   

        # Initialize perturbation evolver
        k_axis_perturbations = model_specs.get_k_axis_perturbations(precision)
        self.PE = perturbations.PerturbationEvolver(
            self.perturbations_list, 
            k_axis_perturbations,
            precision["start_small_k"],
            precision["start_large_k"]
        )

        # Intialize spectrum solver
        k_axis_transfer = model_specs.get_k_axis_transfer(precision)
        self.SS = spectrum.SpectrumSolver(
            precision["l_min"],
            precision["l_max"],
            precision["lensing"],
            k_axis_transfer,
            k_pivot=precision["k_pivot"],
            switch_sw=precision["switch_sw"],
            switch_isw=precision["switch_isw"],
            switch_dop=precision["switch_dop"],
            switch_pol=precision["switch_pol"]
        )

        # Initialize recombination model
        self.RM = hyrex.recomb_model() # DO NOT CHANGE z1 FROM 0

        # Initialize BBN model
        self.PArthENoPE_CLASS_table = jnp.asarray(np.loadtxt(file_dir+'/sBBN_2025_CLASS.txt'))
        self.bbn_type = bbn_type
        self.linx_reaction_net = linx_reaction_net

        self.return_PTBG = return_PTBG

    ### JITTED OR JITTABLE FUNCTIONS ###

    # @jit
    @eqx.filter_jit
    def run_cosmology(self, params : dict):
        """
        Compute CMB angular power spectra for given parameters.

        Runs the full pipeline from background evolution through
        perturbation integration to CMB power spectrum computation.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        tuple
            (ℓ values, (C_ℓ^TT, C_ℓ^TE, C_ℓ^EE)) for computed multipoles
        """

        params = self.add_derived_parameters(params)
        PT, BG = self.get_PTBG(params)
        Cls = self.SS.get_Cl(PT, BG, params)
        ells = self.SS.ells
        
        if self.return_PTBG:
            return ells, Cls, PT, BG
        else:
            return ells, Cls

    # @jit
    @eqx.filter_jit
    def get_PTBG(self, params : dict):
        """
        Get perturbation table and background.

        Computes background and evolves perturbations for the given parameters.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        tuple
            (PerturbationTable, Background) objects
        """
        BG = self.get_BG(params)
        PE = perturbations.PerturbationEvolver(self.perturbations_list) 
        PT = PE.full_evolution((BG, params))

        return PT, BG

    @eqx.filter_jit
    def get_BG(self, params : dict):
        """
        Get background for given parameters.

        Parameters:
        -----------
        params : dict
            Cosmological parameters

        Returns:
        --------
        cosmology.Background
            Background object
        """
        # params = self.add_derived_parameters(params)
        BG = cosmology.Background(params, self.species_list, self.RM)
        return BG
    


    def add_derived_parameters(self, param_in : dict) -> dict:
        """
        Compute derived parameters.

        Calculates derived parameters from the fundamental parameters,
        including density parameters and ratios, and fills in default
        parameter values left unspecified by the user.

        Parameters:
        -----------
        params : dict
            Input parameters

        Returns:
        --------
        dict
            Extended parameter dictionary with derived quantities
        """
        # we do not want to do in-place updates so we can
        # recycle dicts if LINX option is used
        params = param_in.copy()

        if self.bbn_type=="Table" or self.bbn_type=="table":
            # add default params if user unspecified.  No YHe
            params['Neff']          = params.get("Neff", jnp.array(3.044))
            params['h']             = params.get('h', jnp.array(0.7))
            params['omega_cdm']     = params.get('omega_cdm', jnp.array(0.120))
            params['omega_b']       = params.get("omega_b", jnp.array(0.02238))
            params['A_s']           = params.get('A_s', jnp.array(2.e-9))
            params['n_s']           = params.get('n_s', jnp.array(0.965))
            params['TCMB0']         = params.get('TCMB0', jnp.array(2.34865418e-4))
            params['T_nu']          = params.get('T_nu', jnp.array(0.71611 * params['TCMB0']))
            params['T_ncdm']        = params.get('T_ncdm', jnp.array(0.71611))
            params['N_ncdm']        = params.get('N_ncdm', jnp.array(0.))
            params['m_ncdm']        = params.get('m_ncdm', jnp.array(0.6))
            params['z_reion']       = params.get('z_reion', jnp.array(11.0))
            params['Delta_z_reion'] = params.get('Delta_z_reion', jnp.array(0.5))
            params['z_reion_He']    = params.get('z_reion_He', jnp.array(3.5))
            params['Delta_z_reion_He'] = params.get('Delta_z_reion_He', jnp.array(0.5))

            # other derived params must be specified *before* BBN computation
            params['omega_m']      = params['omega_cdm'] + params['omega_b']
            params['R_b']          = params['omega_b'] / params['omega_m']
            params['omega_g']      = 8. * jnp.pi**3 * cnst.G / 45. / cnst.H0_over_h**2 / cnst.hbar**3 / cnst.c**3 * params['TCMB0']**4
            params['H0']           = params['h'] * cnst.H0_over_h
            params['N_ur']         = params['Neff'] - (params['T_ncdm'] / params['TCMB0'])**4 / (4. / 11.)**(4. / 3.) * params['N_ncdm']
            params['omega_nu']     = 7. / 8. * params['N_ur'] * (params['T_nu']/params['TCMB0'])**(4) * params['omega_g']
            params['omega_r']      = params['omega_g'] + params['omega_nu']
            params['R_nu']         = jnp.where(params['omega_r'] > 0.0, params['omega_nu'] / params['omega_r'], 0.0)
            params['omega_Lambda'] = params['h']**2 - params['omega_r'] - params['omega_m']
            
            # interpolate CLASS ParthENoPE table
            bbn = self.PArthENoPE_CLASS_table
            omegab_all = bbn[:, 0]
            DNeff_all = bbn[:, 1]
            YHe_all = bbn[:, 2]

            # we have to hardcode these values to be jit safe (alternatively we 
            # could read them in at runtime, but these tables don't update 
            # frequently)
            n2 = 13 
            n1 = 701

            omegab = omegab_all[:n1]
            DNeff = DNeff_all[::n1] 

            YHe_grid = YHe_all.reshape(n2, n1)
            
            # Neff = params["Neff"] # less extensible option
            a_bbn = cnst.TCMB_today*1e-6/0.01   # neutrino decoupling is well over by 10 keV, so 
                                                # compute Neff at a scale factor approximately 
                                                # corresponding to this temperature
            lna_bbn = jnp.log(a_bbn)

            # this is more extensible than just using params['Neff']; if the user includes i.e. interacting
            # dark radiation, the input parameter Neff tracks only the scaling of the neutrino
            # energy density
            Neff_BBN = (jnp.sum(jnp.asarray([s.rho(lna_bbn, params) for s in self.species_list])) - 
                    self.species_list[-2].rho(lna_bbn,params))/(self.species_list[-1].rho(lna_bbn,params)/params['Neff'])
            
            # last two args are user input omega_b and (Neff_BBN - 3.046) (MUST be 3.046 as 
            # this was assumed when constructing the PArthENoPE table)
            res_YHe = bilinear_interp(omegab, DNeff,YHe_grid, params['omega_b'],Neff_BBN - 3.046)

            # tabulated result is Yp_CMB
            params['YHe'] = res_YHe

        elif self.bbn_type=="LINX" or self.bbn_type=="Linx" or self.bbn_type=="linx":
            # first add params not specified by user.  No Neff or YHe
            params['h']             = params.get('h', jnp.array(0.7))
            params['omega_cdm']     = params.get('omega_cdm', jnp.array(0.120))
            params['omega_b']       = params.get("omega_b", jnp.array(0.02238))
            params['A_s']           = params.get('A_s', jnp.array(2.e-9))
            params['n_s']           = params.get('n_s', jnp.array(0.965))
            params['TCMB0']         = params.get('TCMB0', jnp.array(2.34865418e-4))
            params['T_nu']          = params.get('T_nu', jnp.array(0.71611 * params['TCMB0']))
            params['T_ncdm']        = params.get('T_ncdm', jnp.array(0.71611))
            params['N_ncdm']        = params.get('N_ncdm', jnp.array(0.))
            params['m_ncdm']        = params.get('m_ncdm', jnp.array(0.))
            params['z_reion']       = params.get('z_reion', jnp.array(11.0))
            params['Delta_z_reion'] = params.get('Delta_z_reion', jnp.array(0.5))
            params['z_reion_He']    = params.get('z_reion_He', jnp.array(3.5))
            params['Delta_z_reion_He'] = params.get('Delta_z_reion_He', jnp.array(0.5))

            if params.get("Neff") is not None:
                print("You have specified a value of Neff, but LINX instead expects a \n" \
                    "parameter 'Delt_Neff_init' which will be used to compute Neff.  Refer to LINX \n" \
                    "docs or https://arxiv.org/abs/2408.14538 for more information.")
                sys.exit()


            thermo_model_DNeff = BackgroundModel()
            (
                t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec 
            ) = thermo_model_DNeff(jnp.asarray(params['Delt_Neff_init']))

            params['Neff'] = Neff_vec[-1]

            # convert user input omega_b to eta_fac LINX expects
            eta_fac = params['omega_b'] * linxconst.Omegabh2_to_eta0/linxconst.eta0

            abundance_model = AbundanceModel(NuclearRates(nuclear_net=self.linx_reaction_net))

            abundances = abundance_model(
                rho_g_vec,
                rho_nu_vec,
                rho_NP_vec,
                P_NP_vec,
                t_vec=t_vec_ref,
                a_vec=a_vec_ref,  
                eta_fac = eta_fac,
                tau_n_fac = jnp.asarray(params.get("tau_n_fac", 1.0)),
                nuclear_rates_q = jnp.asarray( params.get("nuclear_rates_q", jnp.zeros( len(abundance_model.nuclear_net.reactions) )) )
                )
            
            # number abundance
            YHe_BBN = 4*abundances[5]
        
            # CMB uses real mass fraction
            Yp_CMB = 1./(4*cnst.mH/cnst.mHe*(1/YHe_BBN - 1) + 1)
            params['YHe'] = Yp_CMB

            # other derived params must be specified *after* BBN computation
            params['omega_m']      = params['omega_cdm'] + params['omega_b']
            params['R_b']          = params['omega_b'] / params['omega_m']
            params['omega_g']      = 8. * jnp.pi**3 * cnst.G / 45. / cnst.H0_over_h**2 / cnst.hbar**3 / cnst.c**3 * params['TCMB0']**4
            params['H0']           = params['h'] * cnst.H0_over_h
            params['N_ur']         = params['Neff'] - (params['T_ncdm'] / params['TCMB0'])**4 / (4. / 11.)**(4. / 3.) * params['N_ncdm']
            params['omega_nu']     = 7. / 8. * params['N_ur'] * (params['T_nu']/params['TCMB0'])**(4) * params['omega_g']
            # params['omega_nu']     = 7. / 8. * params['N_ur'] * (4. / 11.)**(4. / 3.) * params['omega_g']
            params['omega_r']      = params['omega_g'] + params['omega_nu']
            params['R_nu']         = jnp.where(params['omega_r'] > 0.0, params['omega_nu'] / params['omega_r'], 0.0)
            params['omega_Lambda'] = params['h']**2 - params['omega_r'] - params['omega_m']
        
        else:
            # if neither is specified, fill out the dict as usual.  
            # input params defaults
            params['Neff']          = params.get("Neff", jnp.array(3.044))
            params['h']             = params.get('h', jnp.array(0.7))
            params['omega_cdm']     = params.get('omega_cdm', jnp.array(0.120))
            params['omega_b']       = params.get("omega_b", jnp.array(0.02238))
            params['A_s']           = params.get('A_s', jnp.array(2.e-9))
            params['n_s']           = params.get('n_s', jnp.array(0.965))
            params['YHe']           = params.get('YHe', jnp.array(0.245))
            params['TCMB0']         = params.get('TCMB0', jnp.array(2.34865418e-4))
            params['T_nu']          = params.get('T_nu', jnp.array(0.71611 * params['TCMB0']))
            params['T_ncdm']        = params.get('T_ncdm', jnp.array(0.71611))
            params['N_ncdm']        = params.get('N_ncdm', jnp.array(0.))
            params['m_ncdm']        = params.get('m_ncdm', jnp.array(0.))
            params['z_reion']       = params.get('z_reion', jnp.array(11.0))
            params['Delta_z_reion'] = params.get('Delta_z_reion', jnp.array(0.5))
            params['z_reion_He']    = params.get('z_reion_He', jnp.array(3.5))
            params['Delta_z_reion_He'] = params.get('Delta_z_reion_He', jnp.array(0.5))

            # derived params
            params['omega_m']      = params['omega_cdm'] + params['omega_b']
            params['R_b']          = params['omega_b'] / params['omega_m']
            params['omega_g']      = 8. * jnp.pi**3 * cnst.G / 45. / cnst.H0_over_h**2 / cnst.hbar**3 / cnst.c**3 * params['TCMB0']**4
            params['H0']           = params['h'] * cnst.H0_over_h
            params['N_ur']         = params['Neff'] - (params['T_ncdm'] / params['TCMB0'])**4 / (4. / 11.)**(4. / 3.) * params['N_ncdm']
            params['omega_nu']     = 7. / 8. * params['N_ur'] * (params['T_nu']/params['TCMB0'])**(4) * params['omega_g']
            params['omega_r']      = params['omega_g'] + params['omega_nu']
            params['R_nu']         = jnp.where(params['omega_r'] > 0.0, params['omega_nu'] / params['omega_r'], 0.0)
            params['omega_Lambda'] = params['h']**2 - params['omega_r'] - params['omega_m']

        return params
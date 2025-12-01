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
    specs : dict

    species_list : tuple = ()
    species_dict : dict  #= eqx.field(static=True) # Dict as fields must be static
    #perturbed_species_list : tuple = ()
    #perturbed_species_dict : dict  = eqx.field(static=True) # Dict as fields must be static

    bbn_type                : str = ""
    linx_reaction_net       : str = ""
    
    PArthENoPE_CLASS_table  : Array #= eqx.field(converter=jnp.asarray)
    thermo_model_DNeff : BackgroundModel
    abundanceModel : AbundanceModel

    return_PTBG : bool

    

    ### ADDING SPECIES: add has_ parameter and add condition to append to tuple.
    # In the init, all species that are present within the model should be set to True.
    # All couplings present between species should be set to true. 
    def __init__(self,
                 input_specs = {},
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

        # Fill in all user defined and missing specs parameters
        specs = model_specs.load_specs(input_specs)
        self.specs = specs

        # Populate all species
        self.species_list, self.species_dict = model_specs.populate_species(
            user_species,
            specs,
        )   

        # Initialize perturbation evolver
        k_axis_perturbations, k_axis_Pk_output = model_specs.get_k_axis_perturbations(specs)
        self.PE = perturbations.PerturbationEvolver(
            self.species_list, 
            self.species_dict,
            k_axis_perturbations,
            specs
        )

        # Intialize spectrum solver
        k_axis_transfer = model_specs.get_k_axis_transfer(specs)
        self.SS = spectrum.SpectrumSolver(
            specs["l_min"],
            specs["l_max"],
            specs["lensing"],
            k_axis_transfer,
            k_axis_Pk_output,
            k_pivot=specs["k_pivot"],
            switch_sw=specs["switch_sw"],
            switch_isw=specs["switch_isw"],
            switch_dop=specs["switch_dop"],
            switch_pol=specs["switch_pol"]
        )

        # Initialize recombination model
        self.RM = hyrex.recomb_model() # DO NOT CHANGE z1 FROM 0

        # Initialize BBN model
        self.PArthENoPE_CLASS_table = jnp.asarray(np.loadtxt(file_dir+'/sBBN_2025_CLASS.txt'))
        self.bbn_type = bbn_type
        self.linx_reaction_net = linx_reaction_net
        
        # initialize LINX
        if self.bbn_type.lower() == "linx":
            self.thermo_model_DNeff = BackgroundModel()
            self.abundanceModel = AbundanceModel(NuclearRates(nuclear_net=self.linx_reaction_net)) 
        else:
            self.thermo_model_DNeff = None
            self.abundanceModel = None

        self.return_PTBG = return_PTBG

    # need this outside of the jit context
    # since we want LINX to run on CPU
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

        
        full_params = self.add_derived_parameters(params)
        output, aux = self.run_cosmology_abbr(full_params)
        return output, aux
        
    ### JITTED OR JITTABLE FUNCTIONS ###

    @eqx.filter_jit
    def run_cosmology_abbr(self, params : dict):
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

        # let the user know the code is compiling
        print("\n")
        print("            __")
        print("           /  \\")
        print("          /    \\")
        print("         /"+'\033[1m' +"   /\\"+"\033[0m"+" \\")
        print("            "+'\033[1m' +"/  \\"+"\033[0m"+" \\     _     _")
        print("        /  "+'\033[1m' +"/ /\\ \\"+"\033[0m"+" \\   / \\   / \\ ")
        print("          "+'\033[1m' +"/ /__\\ \\"+"\033[0m"+" \\_/"+'\033[1m' +"___"+"\033[0m"+"\\_/"+'\033[1m' +"___"+"\033[0m"+"\\   __")
        print("       / "+'\033[1m' +"/ ______ \\  | _ \\ / ___"+"\033[0m"+"\\_/  \\   _")
        print("        "+'\033[1m' +"/ /      \\ \\ |  _// /    | \\/"+"\033[0m"+" \\_/"+'\033[1m' +"_"+"\033[0m"+"\\  ")
        print("______/"+'\033[1m' +"/ /        \\ \\| _ \\\\ \\___ ||\\/||| - )"+"\033[0m"+"/\\  ")
        print("      "+'\033[1m' +"/_/          \\_\\___/ \\____|||  |||_-_)"+"\033[0m"+"  \\/\\ is compiling...")
        print("                                                 \\/\\")
        print("\n")

        PT, BG = self.get_PTBG(params)
        output = ()
        aux = ()

        if self.specs["output_Cl"]:
            Cls = self.SS.get_Cl(PT, BG, params)
            ells = self.SS.ells
            output += Cls
            aux += (ells,)
        
        if self.specs["output_Pk"]:
            Pk = self.SS.Pk_lin(self.SS.k_axis_Pk_output, 0., PT, params)
            output += (Pk,)
            aux += (self.SS.k_axis_Pk_output,)

        if self.return_PTBG:
            aux += (PT, BG)

        return output, aux

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
        PT = self.PE.full_evolution((BG, params))

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
        # we do not want to do in-place updates so we can
        # recycle dicts if LINX option is used
        params = param_in.copy()

        # Default parameters except Neff and YHe
        params['h']             = params.get('h', jnp.array(0.7))
        params['H0']            = params['h'] * cnst.H0_over_h
        params['omega_cdm']     = params.get('omega_cdm', jnp.array(0.120))
        params['omega_b']       = params.get("omega_b", jnp.array(0.02238))
        params['A_s']           = params.get('A_s', jnp.array(2.e-9))
        params['n_s']           = params.get('n_s', jnp.array(0.965))
        params['TCMB0']         = params.get('TCMB0', jnp.array(2.34865418e-4))
        params['z_reion']       = params.get('z_reion', jnp.array(11.0))
        params['Delta_z_reion'] = params.get('Delta_z_reion', jnp.array(0.5))
        params['z_reion_He']    = params.get('z_reion_He', jnp.array(3.5))
        params['Delta_z_reion_He'] = params.get('Delta_z_reion_He', jnp.array(0.5))

        # Here we fill in a fake omega_Lambda just so that the DE energy density can be computed in a loop.
        # This fake quantity will not be used in anything, and later the correct omega_Lambda will be computed.
        # Purely computational, no physics used or messed up.
        params['omega_Lambda'] = 0.

        # Massive neutrinos
        params['T_nu_massive']  = params.get('T_nu_massive', jnp.array(0.71611)) # Massive neutrino temperature, as a ratio to TCMB.
        params['N_nu_massive']  = params.get('N_nu_massive', jnp.array(0))  # Literal number of massive neutrinos
        params['m_nu_massive']  = params.get('m_nu_massive', jnp.array(0.06)) # Massive neutrino mass, in eV
        params['T_nu_massless'] = params.get('T_nu_massless', jnp.array(0.71636856)) # Massless neutrino temperature, as a ratio to TCMB

        ### CHECKING INPUT COMPATIBILITY ###

        input_N    = params.get('N_nu_massless') != None
        input_Neff = params.get('Neff') != None

        # If the user input both massless neutrino number and Neff, throw an error. Our code treats these as 1-to-1, see paper.
        if input_N and input_Neff:
            print("You can only input one of N_nu_massless or Neff, but got values N_nu_massless={} and Neff={}.".format(params["N_nu_massless"], params["Neff"]))
            sys.exit()

        # If the user input either N_massless or Neff, but requested LINX, throw an error. LINX will compute the correct values.
        if (input_N or input_Neff) and self.bbn_type.lower() == "linx":
            print(
                "You have specified a value for N_nu_massless and/or Neff, but LINX instead expects a \n" \
                    "parameter 'Delta_Neff_init' which will be used to compute Neff. Refer to LINX \n" \
                    "docs or https://arxiv.org/abs/2408.14538 for more information."
            )
            sys.exit()

        if not input_N and not input_Neff and self.bbn_type.lower() != "linx":
            params["N_nu_massless"] = 3 - params['N_nu_massive']
            input_N = True
            print(
                "You did not specify either N_nu_massless or Neff, and did not ask LINX to compute these quantities.\nN_nu_massless will be set to 3-N_nu_massive={}.".format(3-params["N_nu_massive"])
            )
            # Default to Neff mode with standard Neff=3.044. Infer T_nu_massless later.
            # params["Neff"] = 3.044
            # input_Neff = True
            # print(
            #     "You did not specify either T_nu_massless or Neff, and did not ask LINX to compute these quantities.\nNeff will be set to a fiducial value of {}.".format(params["Neff"])
            # )
        
        ### END OF INPUT COMPATIBILITY ###

        ### HELIUM FRACTION AND Neff ###
        # Regardless of bbn_type, these two parameters will be set by the end.

        lna_early = -23.
        a_early = jnp.exp(lna_early)

        # Case 1: The user specifies the true number of massless neutrinos. Note this is distinct from CLASS' N_ur which
        # is computed assuming T_massless = (4/11)^(1/3) x T_CMB.
        # Here, Neff will be inferred from the cosmological fluid content. 
        # In particular if the universe contains massive neutrinos, we account for the error incurred when using a late time
        # massive neutrino temperature which underestimates the massive neutrino energy density at early time, when Neff is set.
        # We account for this by adding the missing relativistic energy in massive neutrinos at early times to the massless fluid.
        # See detail in paper.
        if input_N:
            rho_g = 0.
            rho_nu = 0.
            rho_extra = 0.
            for s in self.species_list:
                rho = s.rho(lna_early, params)
                if s.name == "Photon":
                    rho_g += rho
                elif "neutrino" in s.name.lower():
                    rho_nu += rho
                else:
                    rho_extra += rho

            Neff_raw     = (rho_nu+rho_extra)/rho_g * (8./7.) * (11./4.)**(4./3.) # Uncorrected Neff using T_nu_massive today
            rho_nu_early = 7/8 * (params["N_nu_massless"] + params["N_nu_massive"]) * params["T_nu_massless"]**4 * rho_g # Correct using massless neutrino temp.
            params["Neff"] = (rho_nu_early+rho_extra)/rho_g * (8./7.) * (11./4.)**(4./3.)
            params["N_nu_massless"] = params["N_nu_massless"] + params["Neff"] - Neff_raw # Add difference to massless sector.

        if self.bbn_type.lower() == "table":
            # Applies if user requested BBN table to be user.
            # In this case Neff must already have been set, and can be used to interp YHe.

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

            # Comprehensive Neff, includes all relativsitic species at early times.
            Neff_BBN = params["Neff"]
            
            # last two args are user input omega_b and (Neff_BBN - 3.046) (MUST be 3.046 as 
            # this was assumed when constructing the PArthENoPE table)
            res_YHe = bilinear_interp(omegab, DNeff,YHe_grid, params['omega_b'],Neff_BBN - 3.046)

            # tabulated result is Yp_CMB
            params['YHe'] = res_YHe
        elif self.bbn_type.lower() == "linx":
            # Applies if user requested to run LINX.
            # For this branch to happen, Neff must NOT have already been set. 
            # Logic above has already accounted for this, since input_T and input_Neff must both be False
            # for LINX to execute.
            params['Delta_Neff_init'] = params.get('Delta_Neff_init', 0.)
            (
                t_vec_ref, a_vec_ref, rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec, Neff_vec 
            ) = eqx.filter_jit(self.thermo_model_DNeff,backend='cpu')(jnp.asarray(params['Delta_Neff_init']))

            # convert user input omega_b to eta_fac LINX expects
            eta_fac = params['omega_b'] * linxconst.Omegabh2_to_eta0/linxconst.eta0

            abundances = eqx.filter_jit(self.abundanceModel,backend='cpu')(
                rho_g_vec,
                rho_nu_vec,
                rho_NP_vec,
                P_NP_vec,
                t_vec=t_vec_ref,
                a_vec=a_vec_ref,  
                eta_fac = eta_fac,
                tau_n_fac = jnp.asarray(params.get("tau_n_fac", 1.0)),
                nuclear_rates_q = jnp.asarray( params.get("nuclear_rates_q", jnp.zeros( len(self.abundanceModel.nuclear_net.reactions) )) )
                )
  
            # number abundance
            try:
                params['Neff'] = jax.device_put(Neff_vec[-1],device=jax.devices('gpu')[0])
                YHe_BBN = jax.device_put(4*abundances[5],device=jax.devices('gpu')[0])
            except: # no GPU
                params['Neff'] = Neff_vec[-1]
                YHe_BBN = 4*abundances[5]
                pass
        
            # CMB uses real mass fraction
            Yp_CMB = 1./(4*cnst.mH/cnst.mHe*(1/YHe_BBN - 1) + 1)
            params['YHe'] = Yp_CMB

            # Now Neff has been set by LINX but massless neutrino number has yet to be calculated.
            # we now set the input_Neff flag to True so the branch below takes care of this.
            input_Neff = True
        else:
            # Applies if user wanted neither LINX or BBN table. 
            params['YHe'] = params.get('YHe', jnp.array(0.245))


        # Case 2: User specifies the total Neff of the universe, including neutrinos and all other relativistic species at early times.
        # Then we subtract off all relativistic energy densities from Neff and assign the remaining to massless neutrinos.
        # Since massless neutrino temperature is already specified here, the true derived parameter is N_nu_massless, the physical
        # number of massless neutrinos.
        # The philosophy is that if we're increasing Neff, we are not heating the existing neutrinos, we are adding extra neutrinos
        # at the same temperature. At the CMB level these are indistinguishable, but we chose the later convention. 
        # Note, if after the deduction there's not enough energy density for massless neutrinos (N_nu_massless < 0), ABCMB throws an error.
        if input_Neff:
            rho_g = 0.
            rho_extra = 0.
            for s in self.species_list:
                if s.name == "Photon":
                    rho = s.rho(lna_early, params)
                    rho_g += rho
                elif s.name != "MasslessNeutrino":
                    rho = s.rho(lna_early, params)
                    rho_extra += rho
            rho1nu = 7/8 * (4/11)**(4/3) * rho_g
            
            params['N_nu_massless'] = (params["Neff"] - rho_extra/rho1nu) * ((4/11)**(1/3) / params["T_nu_massless"])**4
            # if params['N_nu_massless'] < 0:
            #     print("ABCMB got a negative N_nu_massless. This is most likely because you included an extra relativistic fluid but did not\n"
            #     +"account for its contribution to Neff when inputting Neff. For this reason when studying BSM radiation we recommend inputting\n"
            #     +"N_nu_massless and T_nu_massless instead of Neff to safely fix the neutrino contributions.")
            #     sys.exit()

        # Loop over matter fluids to compute total matter density today.
        rho_m = 0.
        for s in self.species_list:
            if s.is_matter:
                rho_m += s.rho(0., params)
        params['omega_m']      = rho_m / (3 * cnst.H0_over_h**2/8/jnp.pi/cnst.G) # Fractional matter density
        params['R_b']          = params['omega_b'] / params['omega_m'] # Baryon fraction
    
        # Loop over all fluids and compute energy density at very early time, inferring radiation energy density this way.
        a_early = jnp.exp(-23.)
        rho_r  = 0.
        rho_nu = 0.
        for s in self.species_list:
            rho_r += s.rho(jnp.log(a_early), params)
            if "neutrino" in s.name.lower():
                rho_nu += s.rho(jnp.log(a_early), params)

        params['omega_r']      = rho_r * a_early**4 / (3 * cnst.H0_over_h**2/8/jnp.pi/cnst.G) # Fractional radiation density today
        params['R_nu']         = rho_nu / rho_r # Fractional radiation density in neutrinos, defined at early times. Used for setting adiabatic ICs.

        # Having inferred correct omega_m and omega_r, compute correct omega_Lambda
        params['omega_Lambda'] = params['h']**2 - params['omega_r'] - params['omega_m']

        return params
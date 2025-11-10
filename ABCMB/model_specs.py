import numpy as np
import jax.numpy as jnp
import equinox as eqx

from . import AbstractSpecies as AS
from . import species

def load_specs(input_specs):

    specs = {}

    ### OUTPUT RELATED specs PARAMS ###
    specs["l_min"]     = input_specs.get("l_min", 2)
    specs["l_max"]     = input_specs.get("l_max", 2500)
    specs["lensing"]   = input_specs.get("lensing", False)

    ### TODO: HYREX RELATED specs PARAMS ###

    ### Boltzmann Hierarchy Cutoffs ###
    specs["l_max_g"]     = input_specs.get("l_max_g", 15)
    specs["l_max_pol_g"] = input_specs.get("l_max_pol_g", 10)
    specs["l_max_ur"]    = input_specs.get("l_max_ur", 12)
    specs["l_max_ncdm"]  = input_specs.get("l_max_ncdm", 17)

    ### Perturbation k-grid resolution ###
    specs["k_step_sub"]             = input_specs.get("k_step_sub", 5.e-2)
    specs["k_step_super"]           = input_specs.get("k_step_super", 2.e-3)
    specs["k_step_transition"]      = input_specs.get("k_step_transition", 2.e-1)
    specs["k_step_super_reduction"] = input_specs.get("k_step_super_reduction", 1.e-1)
    specs["k_min_tau0"]             = input_specs.get("k_min_tau0", 1.e-1)
    specs["k_max_tau0_over_l_max"]  = input_specs.get("k_max_tau0_over_l_max", 1.8)

    ### Transfer integration k-grid resolution ###
    specs["k_transfer_linstep"] = input_specs.get("k_transfer_linstep", 4.5e-1)
    specs["k_transfer_logstep"] = input_specs.get("k_transfer_logstep", 170.)
    specs["k_pivot"]            = input_specs.get("k_pivot", 0.05)

    ### Set perturbations initial condition time ###
    specs["start_small_k"] = input_specs.get("start_small_k", 0.0015)
    specs["start_large_k"] = input_specs.get("start_small_k", 0.07)

    ### Physical contributions to CMB temperature transfer function ###
    specs["switch_sw"]  = input_specs.get("switch_sw", 1)
    specs["switch_isw"] = input_specs.get("switch_isw", 1)
    specs["switch_dop"] = input_specs.get("switch_dop", 1)
    specs["switch_pol"] = input_specs.get("switch_pol", 1)

    return specs

def populate_species_old(user_species, specs):
    diffrax_vector_idx = 2 # The first two indices (0 and 1) are always reserved for the metric perturbations.
    species_list = ()
    perturbed_species_list = ()
    perturbed_species_dict = {}

    dark_energy = AS.DarkEnergy()
    species_list = species_list + (dark_energy,)

    cold_dark_matter = AS.ColdDarkMatter(diffrax_vector_idx)
    species_list = species_list + (cold_dark_matter,)
    diffrax_vector_idx += cold_dark_matter.num_ell_modes # Add to total length of Diffrax vector

    baryon = AS.Baryon(diffrax_vector_idx) 
    species_list = species_list + (baryon,)
    diffrax_vector_idx += baryon.num_ell_modes # Add to total length of Diffrax vector

    photon = AS.Photon(diffrax_vector_idx, num_F_ell_modes=specs["l_max_g"]+1, num_G_ell_modes=specs["l_max_pol_g"])
    species_list = species_list + (photon,)
    diffrax_vector_idx += photon.num_ell_modes # Add to total length of Diffrax vector

    massless_neutrinos = AS.MasslessNeutrinos(diffrax_vector_idx, num_ell_modes=specs["l_max_ur"])
    species_list   = species_list + (massless_neutrinos,)
    diffrax_vector_idx += massless_neutrinos.num_ell_modes # Add to total length of Diffrax vector

    if user_species is not None:
        for species in user_species:
            fn = lambda s: s.delta_idx
            # update delta_idx, for which the user probably used default
            # value 0
            updated_species = eqx.tree_at(fn, species, diffrax_vector_idx)
            species_list = species_list + (updated_species,)
            diffrax_vector_idx += updated_species.num_ell_modes

    i = 0
    for species in species_list:
        if isinstance(species, AS.AbstractPerturbedFluid):
            perturbed_species_list = perturbed_species_list + (species, )
            perturbed_species_dict[species.name] = i
            i += 1

    return species_list, perturbed_species_list, perturbed_species_dict

def populate_species(user_species, specs):
    species_list = ()
    species_dict = {}

    lcdm_species = (
        species.DarkEnergy,
        species.ColdDarkMatter,
        species.Baryon,
        species.Photon,
        species.MasslessNeutrino
    )

    i = 0
    diffrax_vector_idx = 2
    for s in lcdm_species:
        instance = s(diffrax_vector_idx, specs) # Creates an instance of s. init is now consistent across all species
        species_list = species_list + (instance,)
        species_dict[instance.name] = i

        i += 1
        diffrax_vector_idx += instance.num_ell_modes

    if user_species is not None:
        for s in user_species:
            instance = s(diffrax_vector_idx, specs)
            species_list = species_list + (instance,)
            species_dict[instance.name] = i

            i += 1
            diffrax_vector_idx += instance.num_ell_modes

    return species_list, species_dict

def get_k_axis_perturbations(specs):
    ks = np.zeros(2000)

    H0_fid     = 2.255560e-04
    tau0_fid   = 1.418668e+04
    rs_rec_fid = 1.446279e+02
    k_rec_fid  = 2.*jnp.pi/rs_rec_fid

    k_min = specs["k_min_tau0"] / tau0_fid
    k_max = specs["k_max_tau0_over_l_max"] / tau0_fid * specs["l_max"]

    k = k_min   
    ks[0] = k
    i = 0
    while k < k_max:
        step = (specs["k_step_super"]
                + 0.5 * (jnp.tanh((k-k_rec_fid)/k_rec_fid/specs["k_step_transition"])+1.)
                * (specs["k_step_sub"]-specs["k_step_super"])) * k_rec_fid

        scale2 = H0_fid**2

        step *= (k**2/scale2+1.)/(k**2/scale2+1./specs["k_step_super_reduction"])

        k += step
        i += 1
        ks[i] = k

    specs["k_min"]     = k_min
    specs["k_max_cmb"] = k

    # If lensing is needed, we need to extend max k by some amount to accurately compute high-l lensing.
    if specs["lensing"]:
        k_max = k + 0.3
        
        while k < k_max:
            step = 0.005

            k += step
            i += 1
            ks[i] = k

    ks = jnp.array(ks[np.where(ks>0)])

    return ks

def get_k_axis_transfer(specs):
    ks = np.zeros(8000)

    k_period = 4.518444e-04

    k = specs["k_min"]
    ks[0] = k
    i = 0
    while k < specs["k_max_cmb"]:
        k = k \
            + k_period * specs["k_transfer_linstep"] * k \
            / (k + specs["k_transfer_linstep"]/specs["k_transfer_logstep"])
        i += 1
        ks[i] = k

    ks = jnp.array(ks[np.where(ks>0)])
    return ks
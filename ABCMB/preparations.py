import numpy as np
import jax.numpy as jnp
import equinox as eqx

from . import AbstractSpecies as AS

def load_precision(precision_in):

    precision = {}

    ### OUTPUT RELATED PRECISION PARAMS ###
    precision["l_min"]     = precision_in.get("l_min", 2)
    precision["l_max"]     = precision_in.get("l_max", 2500)
    precision["lensing"]  = precision_in.get("lensing", False)

    ### TODO: HYREX RELATED PRECISION PARAMS ###

    ### Boltzmann Hierarchy Cutoffs ###
    precision["l_max_g"]     = precision_in.get("l_max_g", 15)
    precision["l_max_pol_g"] = precision_in.get("l_max_pol_g", 10)
    precision["l_max_ur"]    = precision_in.get("l_max_ur", 12)
    precision["l_max_ncdm"]  = precision_in.get("l_max_ncdm", 17)

    ### Perturbation k-grid resolution ###
    precision["k_step_sub"]             = precision_in.get("k_step_sub", 5.e-2)
    precision["k_step_super"]           = precision_in.get("k_step_super", 2.e-3)
    precision["k_step_transition"]      = precision_in.get("k_step_transition", 2.e-1)
    precision["k_step_super_reduction"] = precision_in.get("k_step_super_reduction", 1.e-1)
    precision["k_min_tau0"]             = precision_in.get("k_min_tau0", 1.e-1)
    precision["k_max_tau0_over_l_max"]  = precision_in.get("k_max_tau0_over_l_max", 1.8)

    ### Transfer integration k-grid resolution ###
    precision["k_transfer_linstep"] = precision_in.get("k_transfer_linstep", 4.5e-1)
    precision["k_transfer_logstep"] = precision_in.get("k_transfer_logstep", 170.)
    precision["k_pivot"]            = precision_in.get("k_pivot", 0.05)

    ### Set perturbations initial condition time ###
    precision["start_small_k"] = precision_in.get("start_small_k", 0.0015)
    precision["start_large_k"] = precision_in.get("start_small_k", 0.07)

    ### Physical contributions to CMB temperature transfer function ###
    precision["switch_sw"]  = precision_in.get("switch_sw", 1)
    precision["switch_isw"] = precision_in.get("switch_isw", 1)
    precision["switch_dop"] = precision_in.get("switch_dop", 1)
    precision["switch_pol"] = precision_in.get("switch_pol", 1)

    return precision

def populate_species(user_species, precision):
    diffrax_vector_idx = 2 # The first two indices (0 and 1) are always reserved for the metric perturbations.
    species_list = ()
    perturbations_list = ()

    dark_energy = AS.DarkEnergy()
    species_list = species_list + (dark_energy,)

    # user species must be defined before CDM, since ABCMB expects
    # fixed indices for CDM, baryons, photons, and massive neutrinos

    if user_species is not None:
        for species in user_species:
            fn = lambda spec: spec.delta_idx
            # update delta_idx, for which the user probably used default
            # value 0
            updated_species = eqx.tree_at(fn, species, diffrax_vector_idx)
            species_list = species_list + (updated_species,)
            diffrax_vector_idx += updated_species.num_ell_modes

    # These perturbed species are always present in all runs.
    # massless neutrinos are last, photons are second to last, baryons third to last, CDM fourth to last.

    cold_dark_matter = AS.ColdDarkMatter(diffrax_vector_idx)
    species_list = species_list + (cold_dark_matter,)
    diffrax_vector_idx += cold_dark_matter.num_ell_modes # Add to total length of Diffrax vector

    baryon = AS.Baryon(dark_energy, diffrax_vector_idx) # CG switched order
    diffrax_vector_idx += baryon.num_ell_modes # Add to total length of Diffrax vector

    photon = AS.Photon(diffrax_vector_idx, baryon, num_F_ell_modes=precision["l_max_g"]+1, num_G_ell_modes=precision["l_max_pol_g"])
    diffrax_vector_idx += photon.num_ell_modes # Add to total length of Diffrax vector

    baryon = eqx.tree_at(lambda b : b.photon, baryon, photon)
    species_list = species_list + (baryon, photon,)

    massless_neutrinos = AS.MasslessNeutrinos(diffrax_vector_idx, num_ell_modes=precision["l_max_ur"])
    species_list   = species_list + (massless_neutrinos,)
    diffrax_vector_idx += massless_neutrinos.num_ell_modes # Add to total length of Diffrax vector

    for species in species_list:
        if isinstance(species, AS.AbstractPerturbedFluid):
            perturbations_list = perturbations_list + (species, )

    return species_list, perturbations_list

def get_k_axis_perturbations(precision):
    ks = np.zeros(2000)

    H0_fid     = 2.255560e-04
    tau0_fid   = 1.418668e+04
    rs_rec_fid = 1.446279e+02
    k_rec_fid  = 2.*jnp.pi/rs_rec_fid

    k_min = precision["k_min_tau0"] / tau0_fid
    k_max = precision["k_max_tau0_over_l_max"] / tau0_fid * precision["l_max"]

    k = k_min   
    ks[0] = k
    i = 0
    while k < k_max:
        step = (precision["k_step_super"]
                + 0.5 * (jnp.tanh((k-k_rec_fid)/k_rec_fid/precision["k_step_transition"])+1.)
                * (precision["k_step_sub"]-precision["k_step_super"])) * k_rec_fid

        scale2 = H0_fid**2

        step *= (k**2/scale2+1.)/(k**2/scale2+1./precision["k_step_super_reduction"])

        k += step
        i += 1
        ks[i] = k

    precision["k_min"]     = k_min
    precision["k_max_cmb"] = k

    # If lensing is needed, we need to extend max k by some amount to accurately compute high-l lensing.
    if precision["lensing"]:
        k_max = 1.0
        
        while k < k_max:
            step = 0.005

            k += step
            i += 1
            ks[i] = k

    ks = jnp.array(ks[np.where(ks>0)])

    return ks

def get_k_axis_transfer(precision):
    ks = np.zeros(8000)

    k_period = 4.518444e-04

    k = precision["k_min"]
    ks[0] = k
    i = 0
    while k < precision["k_max_cmb"]:
        k = k \
            + k_period * precision["k_transfer_linstep"] * k \
            / (k + precision["k_transfer_linstep"]/precision["k_transfer_logstep"])
        i += 1
        ks[i] = k

    ks = jnp.array(ks[np.where(ks>0)])
    return ks
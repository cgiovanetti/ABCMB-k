import jax
import jax.numpy as jnp
import equinox as eqx

class ModuleWithArrays(eqx.Module):
    # This mimics AbundanceModel with arrays created in __init__
    coefficients: jax.Array
    weights: jax.Array
    species_data: jax.Array  # Mimics species_Z, species_N, etc.
    dictionary : dict
    
    def __init__(self):
        # These arrays are created on default device (GPU) during init
        self.coefficients = jnp.array([1.0, 2.0, 3.0])
        self.weights = jnp.array([0.5, 0.3, 0.2])
        self.species_data = jnp.array([0, 1, 1, 1, 2, 2, 3, 4, 3, 2, 3, 5])  # Mimics species_Z
        self.dictionary = {'add':8.7}
    
    def __call__(self, x):
        # Return array to mimic LINX returning abundances
        return jnp.ones(12) * (jnp.sum(self.coefficients * self.weights) * x + self.dictionary['add'])
class ModuleWithPrimitives(eqx.Module):
    use_scaling: bool
    factor: float
    
    def __init__(self):
        self.use_scaling = True
        self.factor = 2.0
    
    @eqx.filter_jit
    def __call__(self, x):
        if self.use_scaling:
            return x * self.factor
        return x

class Caller(eqx.Module):
    # Mimics Model class
    withArrays : ModuleWithArrays  # Mimics abundanceModel
    withPrimitives : ModuleWithPrimitives

    def __init__(self, ):
        # When Model is created, abundanceModel is created with GPU arrays
        self.withArrays = ModuleWithArrays()
        self.withPrimitives = ModuleWithPrimitives()

    def __call__(self, params_dict):
        # Mimics run_cosmology - NOT jitted
        full_params = self.add_derived_parameters(params_dict)
        # Mimics run_cosmology_abbr - jitted, no backend (GPU default)
        result = self.run_cosmology_abbr(full_params)
        return result

    # NOT jitted - mimics add_derived_parameters
    def add_derived_parameters(self, params_dict):
        params = params_dict.copy()
        
        # Call CPU-jitted abundanceModel - mimics line 384 in main.py
        # This returns CPU arrays
        cpu_abundances = eqx.filter_jit(self.withArrays, backend='cpu')(params['Delta_Neff_init'])
        
        # Extract value and add to params - mimics line 442
        params['YHe'] = cpu_abundances[5]  # CPU scalar
        
        # Do some arithmetic - these stay on same device as inputs
        params['omega_m'] = params['omega_cdm'] + params['omega_b']  # GPU + GPU = GPU
        
        return params
    
    @eqx.filter_jit  # No backend - defaults to GPU
    def run_cosmology_abbr(self, params):
        # self.withArrays has GPU arrays from __init__
        # params['YHe'] is CPU scalar from CPU-jitted function
        # This creates device mismatch!
        return jnp.sum(self.withArrays.species_data) + params['YHe'] + params['omega_m']

caller = Caller()
# Create params as JAX arrays on GPU (default device)
params = {
    'Delta_Neff_init': jnp.asarray(0.),
    'omega_cdm': jnp.asarray(0.1193),
    'omega_b': jnp.asarray(0.0225)
}
res = caller(params)
print(res)

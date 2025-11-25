import jax
import jax.numpy as jnp
import equinox as eqx

class ModuleWithArrays(eqx.Module):

    coefficients: jax.Array
    weights: jax.Array
    dictionary : dict
    
    def __init__(self):
        self.coefficients = jnp.array([1.0, 2.0, 3.0])
        self.weights = jnp.array([0.5, 0.3, 0.2])
        self.dictionary = {'add':8.7}
    
    def __call__(self, x):
        return jnp.sum(self.coefficients * self.weights) * x + self.dictionary['add']

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
    withArrays : ModuleWithArrays
    withPrimitives : ModuleWithPrimitives

    def __init__(self, ):
        self.withArrays = ModuleWithArrays()
        self.withPrimitives = ModuleWithPrimitives()

    def __call__(self, a, b):
        # This mimics run_cosmology - NOT jitted
        params_dict = self.add_derived_parameters(a, b)
        # This mimics run_cosmology_abbr - jitted without backend spec (defaults to GPU)
        result = self.run_cosmology_abbr(params_dict)
        return result

    # NOT jitted - mimics add_derived_parameters
    def add_derived_parameters(self, a, b):
        # Call CPU-jitted function that returns arrays
        cpu_arrays = eqx.filter_jit(self.withArrays, backend='cpu')(a)
        
        # Create params dict with mix of CPU and GPU arrays
        # cpu_arrays is on CPU from the CPU-jitted function
        # The arithmetic operations create new arrays that stay on CPU
        params_dict = {
            'first': jnp.array([cpu_arrays, cpu_arrays+1, cpu_arrays+2, cpu_arrays+3, cpu_arrays+4]),  # CPU
            'rest': jnp.array([b, b+1, b+2, b+3, b+4, b+5, b+6])  # GPU (default)
        }
        return params_dict
    
    @eqx.filter_jit  # No backend specified - defaults to GPU
    def run_cosmology_abbr(self, params_dict):
        # This receives a dict with mixed-device arrays
        return jnp.sum(params_dict['first']) + jnp.sum(params_dict['rest'])

caller = Caller()
res = caller(1.,2.)
print(res)


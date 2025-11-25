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

    def __call__(self, input_dict):
        # NOT jitted - mimics run_cosmology
        full_params = self.add_derived_parameters(input_dict)
        # Jitted - mimics run_cosmology_abbr
        result = self.run_cosmology_abbr(full_params)
        return result

    # NOT jitted - mimics add_derived_parameters
    def add_derived_parameters(self, input_dict):
        # Start with input dict (Python floats)
        params = input_dict.copy()
        
        # Call CPU-jitted function with Python float
        # This returns array on CPU
        cpu_result = eqx.filter_jit(self.withArrays, backend='cpu')(params['a'])
        
        # Do arithmetic with CPU array and Python float
        # The Python float gets converted to JAX array on DEFAULT device (GPU)
        params['first'] = cpu_result + params['b']  # cpu_result is CPU, params['b'] becomes GPU
        
        # Also add some pure Python float operations that become GPU arrays
        params['rest'] = params['b'] * 2.0  # This becomes a GPU array
        
        return params
    
    @eqx.filter_jit  # No backend - defaults to GPU
    def run_cosmology_abbr(self, params):
        # params dict now has mixed devices:
        # params['first'] involves CPU array
        # params['rest'] is GPU array
        return params['first'] + params['rest']

caller = Caller()
input_params = {'a': 1.0, 'b': 2.0}  # Python floats, like time_tests.py
res = caller(input_params)
print(res)


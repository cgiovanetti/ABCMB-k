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
    some_array_on_cpu : jax.Array  # This will be on CPU

    def __init__(self, ):
        self.withArrays = ModuleWithArrays()
        self.withPrimitives = ModuleWithPrimitives()
        # Create array on CPU explicitly
        self.some_array_on_cpu = jax.device_put(jnp.ones(548), jax.devices('cpu')[0])

    def __call__(self, params_dict):
        # NOT jitted
        full_params = self.add_derived_parameters(params_dict)
        # Jitted - self has CPU array, params has GPU arrays
        result = self.run_cosmology_abbr(full_params)
        return result

    def add_derived_parameters(self, params_dict):
        # Just pass through - params are already on GPU
        return params_dict
    
    @eqx.filter_jit  # self has CPU array, params arg has GPU arrays
    def run_cosmology_abbr(self, params):
        # Mixed devices: self.some_array_on_cpu is CPU, params values are GPU
        return jnp.sum(self.some_array_on_cpu) + params['h']

caller = Caller()
# Create params on GPU (default device)
params = {'h': jnp.asarray(0.6762)}
res = caller(params)
print(res)

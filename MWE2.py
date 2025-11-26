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

    def __call__(self, input_params):
        # NOT jitted - mimics run_cosmology calling add_derived_parameters
        full_params = self.add_derived_parameters(input_params)
        # Jitted - mimics run_cosmology_abbr
        result = self.run_cosmology_abbr(full_params)
        return result

    # NOT jitted - mimics add_derived_parameters
    def add_derived_parameters(self, input_params):
        params = input_params.copy()
        
        # Call CPU-jitted function - returns scalar on CPU
        cpu_scalar = eqx.filter_jit(self.withArrays, backend='cpu')(params['a'])
        
        # Create array from CPU scalar - stays on CPU
        cpu_array = jnp.ones(548) * cpu_scalar
        
        # Explicitly move to GPU (mimics line 437 in main.py)
        gpu_array = jax.device_put(cpu_array, jax.devices('gpu')[0])
        
        # Build params dict with GPU array and Python float
        # The Python float will become a CPU scalar when passed to jitted function
        params['omega_b'] = gpu_array[0]  # GPU scalar
        params['h'] = params['b']  # Python float -> will be CPU scalar in jit
        
        return params
    
    @eqx.filter_jit  # No backend - defaults to GPU
    def run_cosmology_abbr(self, params):
        # params dict has mixed devices after pytree flattening
        return params['omega_b'] + params['h']

caller = Caller()
input_params = {'a': 1.0, 'b': 2.0}  # Python floats
res = caller(input_params)
print(res)


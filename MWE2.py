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
        # Return a large array to mimic LINX returning arrays like rho_g_vec
        return jnp.ones(548) * (jnp.sum(self.coefficients * self.weights) * x + self.dictionary['add'])

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
        # NOT jitted - mimics run_cosmology
        params_dict = self.add_derived_parameters(a, b)
        # Jitted - mimics run_cosmology_abbr
        result = self.run_cosmology_abbr(params_dict)
        return result

    # NOT jitted - mimics add_derived_parameters
    def add_derived_parameters(self, a, b):
        # CPU-jitted function returns large array (mimics LINX)
        cpu_large_array = eqx.filter_jit(self.withArrays, backend='cpu')(a)
        
        # Move to GPU explicitly (mimics the device_put at line 437)
        gpu_large_array = jax.device_put(cpu_large_array, jax.devices('gpu')[0])
        
        # Create dict with:
        # - 'first': large array on GPU (from CPU computation moved to GPU)
        # - 'rest': scalars that stay on CPU (default for scalars)
        params_dict = {
            'first': gpu_large_array,  # GPU array
            'rest': b  # Scalar stays on CPU
        }
        return params_dict
    
    @eqx.filter_jit  # No backend - defaults to GPU
    def run_cosmology_abbr(self, params_dict):
        # Mixed device dict causes error
        return jnp.sum(params_dict['first']) + params_dict['rest']

caller = Caller()
res = caller(1.,2.)
print(res)


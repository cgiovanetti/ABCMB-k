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
        # This mimics add_derived_parameters - not jitted, creates dict with mixed devices
        dynamic_dict = self.fun1(a, b)
        # This mimics run_cosmology_abbr - GPU jitted, receives mixed-device dict
        result = self.fun2(dynamic_dict)
        return result

    # not jitted - mimics add_derived_parameters
    def fun1(self, a, b):
        # CPU-jitted function returns array (mimics LINX functions)
        cpu_result = eqx.filter_jit(self.withArrays, backend='cpu')(a)
        
        # Create dictionary with mixed devices
        # 'first' is on CPU (from cpu_result which came from CPU-jitted function)
        # 'rest' is on GPU (default device for jnp.array)
        dynamic_dict = {
            'first': jnp.array([cpu_result, cpu_result+1, cpu_result+2, cpu_result+3, cpu_result+4]),
            'rest': jnp.array([b, b+1, b+2, b+3, b+4, b+5, b+6])
        }
        return dynamic_dict
    
    @eqx.filter_jit
    def fun2(self, dynamic_dict):
        # GPU-jitted function receives dict with mixed-device arrays
        return jnp.sum(dynamic_dict['first']) + jnp.sum(dynamic_dict['rest'])

caller = Caller()
res = caller(1.,2.)
print(res)
# print(Manipulator(res))


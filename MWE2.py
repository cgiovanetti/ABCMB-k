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
        one = self.fun1(a,b)
        two = self.fun2(one,a)
        return one + two

    # not jitted in my example
    def fun1(self, a, b):
        quantity1 = eqx.filter_jit(self.withArrays,backend='cpu')(a)
        quantity2 = self.withPrimitives(quantity1)
        return quantity2 
    
    @eqx.filter_jit
    def fun2(self, a, b):
        return a**2/b**2

caller = Caller()
res = caller(1.,2.)
print(res)
# print(Manipulator(res))


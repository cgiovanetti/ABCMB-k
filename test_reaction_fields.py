import jax
import jax.numpy as jnp
from ABCMB.linx.reactions import Reaction

print("=" * 60)
print("CHECKING REACTION FIELDS")
print("=" * 60)

rxn = Reaction(
    'npdg', (0, 1), (2, ), 4.7161402e9, 1.5, -25.81502, 
    spline_data='key_PRIMAT_2023/npdg.txt', 
    interp_type='linear'
)

print("\nChecking all Reaction fields:")
for attr_name in dir(rxn):
    if not attr_name.startswith('_'):
        try:
            val = getattr(rxn, attr_name)
            if not callable(val):
                is_jax = isinstance(val, jax.Array)
                print(f"  {attr_name}: {type(val).__name__}, is JAX array: {is_jax}")
                if is_jax:
                    print(f"    VALUE: {val}")
        except:
            pass

print("\n" + "=" * 60)
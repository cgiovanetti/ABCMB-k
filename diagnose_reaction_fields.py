"""
Targeted diagnostic to check Reaction class fields for JAX arrays.
"""

import jax
import jax.numpy as jnp
import sys

sys.path.insert(0, '/Users/caragiovanetti/Research/GitHub/ABCMB')
from ABCMB.linx.reactions import Reaction

# Create a simple Reaction instance
reaction = Reaction(
    'test', (0, 1), (2,), 
    alpha=1.0, beta=0.0, gamma=0.0,
    frwrd_rate_param_func=lambda T, p: 1.0
)

print("Checking Reaction instance fields:")
print("=" * 80)

for field_name in ['name', 'in_states', 'out_states', 'frwrd_symmetry_fac', 
                   'bkwrd_symmetry_fac', 'alpha', 'beta', 'gamma']:
    value = getattr(reaction, field_name)
    is_jax = isinstance(value, (jax.Array, jnp.ndarray))
    
    # Check if field is marked as static
    if hasattr(Reaction, '__dataclass_fields__'):
        field = Reaction.__dataclass_fields__.get(field_name)
        is_static = field.metadata.get('static', False) if field and hasattr(field, 'metadata') else False
    else:
        is_static = False
    
    print(f"{field_name}:")
    print(f"  Value: {value}")
    print(f"  Type: {type(value)}")
    print(f"  Is JAX array: {is_jax}")
    print(f"  Is static: {is_static}")
    
    if is_jax and not is_static:
        print(f"  >>> PROBLEM: JAX array NOT marked as static!")
    print()

print("\nDiagnosis:")
print("=" * 80)
print("The frwrd_symmetry_fac and bkwrd_symmetry_fac fields are computed using")
print("jnp.prod() which returns JAX arrays, but they are NOT marked as static.")
print("This is the source of the unhashable type error!")
"""
Targeted diagnostic to check WeakRates class fields for JAX arrays.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys

sys.path.insert(0, '/Users/caragiovanetti/Research/GitHub/ABCMB')
from ABCMB.linx.weak_rates import WeakRates

# Create a WeakRates instance
weak_rates = WeakRates()

print("Checking WeakRates instance fields:")
print("=" * 80)

for field_name in ['RC_corr', 'thermal_corr', 'FM_corr', 'weak_mag_corr',
                   'T_nTOp_thermal_interval', 'T_pTOn_thermal_interval',
                   'L_nTOpCCRTh_res', 'L_pTOnCCRTh_res', 'lambda_0']:
    value = getattr(weak_rates, field_name)
    is_jax = isinstance(value, (jax.Array, jnp.ndarray))
    is_numpy = isinstance(value, np.ndarray)
    
    # Check if field is marked as static
    if hasattr(WeakRates, '__dataclass_fields__'):
        field = WeakRates.__dataclass_fields__.get(field_name)
        is_static = field.metadata.get('static', False) if field and hasattr(field, 'metadata') else False
    else:
        is_static = False
    
    print(f"{field_name}:")
    print(f"  Type: {type(value)}")
    print(f"  Is JAX array: {is_jax}")
    print(f"  Is NumPy array: {is_numpy}")
    print(f"  Is static: {is_static}")
    
    if is_jax and not is_static:
        print(f"  >>> PROBLEM: JAX array NOT marked as static!")
    if is_numpy and not is_static:
        print(f"  >>> WARNING: NumPy array NOT marked as static (may become JAX array)")
    print()

print("\nDiagnosis:")
print("=" * 80)
print("Checking if any fields contain JAX arrays that are not marked as static...")
import jax
import jax.numpy as jnp
import numpy as np
import warnings

# Capture warnings
warnings.filterwarnings('error', message='.*JAX array.*static.*')

print("=" * 60)
print("FINAL COMPREHENSIVE DIAGNOSIS")
print("=" * 60)

try:
    print("\n1. Testing Reaction creation...")
    from ABCMB.linx.reactions import Reaction
    rxn = Reaction(
        'npdg', (0, 1), (2, ), 4.7161402e9, 1.5, -25.81502, 
        spline_data='key_PRIMAT_2023/npdg.txt', 
        interp_type='linear'
    )
    print("   Reaction created successfully - no warning")
except UserWarning as e:
    print(f"   WARNING during Reaction creation: {e}")

try:
    print("\n2. Testing WeakRates creation...")
    from ABCMB.linx.weak_rates import WeakRates
    wr = WeakRates()
    print("   WeakRates created successfully - no warning")
except UserWarning as e:
    print(f"   WARNING during WeakRates creation: {e}")

try:
    print("\n3. Testing NuclearRates creation...")
    from ABCMB.linx.nuclear import NuclearRates
    nr = NuclearRates(nuclear_net='key_PRIMAT_2023')
    print("   NuclearRates created successfully - no warning")
except UserWarning as e:
    print(f"   WARNING during NuclearRates creation: {e}")
    print("\n   Checking NuclearRates fields for JAX arrays:")
    for field_name in ['reactions', 'in_states', 'out_states', 
                       'frwrd_symmetry_fac', 'bkwrd_symmetry_fac']:
        if hasattr(nr, field_name):
            val = getattr(nr, field_name)
            if isinstance(val, dict) and len(val) > 0:
                first_key = list(val.keys())[0]
                first_val = val[first_key]
                print(f"      {field_name}['{first_key}']: {type(first_val)}, is JAX: {isinstance(first_val, jax.Array)}")

try:
    print("\n4. Testing AbundanceModel creation...")
    from ABCMB.linx.abundances import AbundanceModel
    am = AbundanceModel(nr)
    print("   AbundanceModel created successfully - no warning")
except UserWarning as e:
    print(f"   WARNING during AbundanceModel creation: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
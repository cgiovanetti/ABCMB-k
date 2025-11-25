import jax
import jax.numpy as jnp
import numpy as np
from ABCMB.linx.reactions import Reaction
from ABCMB.linx.weak_rates import WeakRates
from ABCMB.linx.nuclear import NuclearRates

print("=" * 60)
print("DETAILED NESTED STATIC FIELD DIAGNOSIS")
print("=" * 60)

# Test WeakRates in detail
print("\n1. WeakRates class - checking all fields:")
wr = WeakRates()
for field_name in ['T_nTOp_thermal_interval', 'T_pTOn_thermal_interval', 
                   'L_nTOpCCRTh_res', 'L_pTOnCCRTh_res', 'lambda_0']:
    if hasattr(wr, field_name):
        val = getattr(wr, field_name)
        print(f"   {field_name}:")
        print(f"      type: {type(val)}")
        print(f"      is JAX array: {isinstance(val, jax.Array)}")
        if isinstance(val, (list, tuple)) and len(val) > 0:
            print(f"      first element type: {type(val[0])}")

# Test NuclearRates in detail
print("\n2. NuclearRates class - checking all fields:")
nr = NuclearRates(nuclear_net='key_PRIMAT_2023')
for field_name in ['reactions', 'in_states', 'out_states', 
                   'frwrd_symmetry_fac', 'bkwrd_symmetry_fac',
                   'frwrd_rate_param', 'bkwrd_rate_param',
                   'frwrd_reaction_by_particle', 'bkwrd_reaction_by_particle']:
    if hasattr(nr, field_name):
        val = getattr(nr, field_name)
        print(f"   {field_name}:")
        print(f"      type: {type(val)}")
        print(f"      is JAX array: {isinstance(val, jax.Array)}")
        if isinstance(val, dict) and len(val) > 0:
            first_key = list(val.keys())[0]
            first_val = val[first_key]
            print(f"      first value type: {type(first_val)}")
            print(f"      first value is JAX array: {isinstance(first_val, jax.Array)}")
        elif isinstance(val, (list, tuple)) and len(val) > 0:
            print(f"      first element type: {type(val[0])}")

# Check Reaction objects within NuclearRates
print("\n3. Checking Reaction objects in NuclearRates.reactions:")
if len(nr.reactions) > 0:
    rxn = nr.reactions[0]
    for field_name in ['in_states', 'out_states', 'frwrd_symmetry_fac', 
                       'bkwrd_symmetry_fac', 'alpha', 'beta', 'gamma']:
        if hasattr(rxn, field_name):
            val = getattr(rxn, field_name)
            print(f"   {field_name}:")
            print(f"      type: {type(val)}")
            print(f"      is JAX array: {isinstance(val, jax.Array)}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
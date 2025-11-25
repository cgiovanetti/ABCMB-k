import jax
import jax.numpy as jnp
import numpy as np
from ABCMB.linx.reactions import Reaction
from ABCMB.linx.weak_rates import WeakRates
from ABCMB.linx.nuclear import NuclearRates
from ABCMB.linx.abundances import AbundanceModel

print("=" * 60)
print("DIAGNOSING STATIC FIELD JAX ARRAY WARNINGS")
print("=" * 60)

# Test 1: Reaction class
print("\n1. Testing Reaction class fields:")
rxn = Reaction(
    'npdg', (0, 1), (2, ), 4.7161402e9, 1.5, -25.81502, 
    spline_data='key_PRIMAT_2023/npdg.txt', 
    interp_type='linear'
)
print(f"   T9_vec type: {type(rxn.T9_vec)}")
print(f"   T9_vec is JAX array: {isinstance(rxn.T9_vec, jax.Array)}")
print(f"   mu_median_vec type: {type(rxn.mu_median_vec)}")
print(f"   mu_median_vec is JAX array: {isinstance(rxn.mu_median_vec, jax.Array)}")
print(f"   expsigma_vec type: {type(rxn.expsigma_vec)}")
print(f"   expsigma_vec is JAX array: {isinstance(rxn.expsigma_vec, jax.Array)}")

# Test 2: WeakRates class
print("\n2. Testing WeakRates class fields:")
wr = WeakRates()
print(f"   T_nTOp_thermal_interval type: {type(wr.T_nTOp_thermal_interval)}")
print(f"   T_nTOp_thermal_interval is JAX array: {isinstance(wr.T_nTOp_thermal_interval, jax.Array)}")
print(f"   L_nTOpCCRTh_res type: {type(wr.L_nTOpCCRTh_res)}")
print(f"   L_nTOpCCRTh_res is JAX array: {isinstance(wr.L_nTOpCCRTh_res, jax.Array)}")

# Test 3: AbundanceModel class
print("\n3. Testing AbundanceModel class fields:")
nuclear_net = NuclearRates(nuclear_net='key_PRIMAT_2023')
am = AbundanceModel(nuclear_net)
print(f"   species_Z type: {type(am.species_Z)}")
print(f"   species_Z is JAX array: {isinstance(am.species_Z, jax.Array)}")
print(f"   species_N type: {type(am.species_N)}")
print(f"   species_N is JAX array: {isinstance(am.species_N, jax.Array)}")
print(f"   species_excess_mass type: {type(am.species_excess_mass)}")
print(f"   species_excess_mass is JAX array: {isinstance(am.species_excess_mass, jax.Array)}")

# Test 4: NuclearRates.reactions
print("\n4. Testing NuclearRates.reactions field:")
print(f"   reactions type: {type(nuclear_net.reactions)}")
print(f"   reactions[0] type: {type(nuclear_net.reactions[0])}")
print(f"   reactions[0].T9_vec is JAX array: {isinstance(nuclear_net.reactions[0].T9_vec, jax.Array)}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
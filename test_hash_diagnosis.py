import jax
import jax.numpy as jnp
from ABCMB.linx.nuclear import NuclearRates
from ABCMB.linx.abundances import AbundanceModel
from ABCMB.linx.background import BackgroundModel

print("=" * 60)
print("HASH DIAGNOSIS TEST")
print("=" * 60)

# Test 1: Can we hash BackgroundModel?
print("\n1. Testing BackgroundModel hashability...")
try:
    bg_model = BackgroundModel()
    hash_val = hash(bg_model)
    print(f"   SUCCESS: BackgroundModel is hashable (hash={hash_val})")
except TypeError as e:
    print(f"   FAILED: BackgroundModel is NOT hashable: {e}")

# Test 2: Can we hash NuclearRates?
print("\n2. Testing NuclearRates hashability...")
try:
    nuclear_net = NuclearRates(nuclear_net="key_PRIMAT_2023")
    hash_val = hash(nuclear_net)
    print(f"   SUCCESS: NuclearRates is hashable (hash={hash_val})")
except TypeError as e:
    print(f"   FAILED: NuclearRates is NOT hashable: {e}")

# Test 3: Can we hash AbundanceModel?
print("\n3. Testing AbundanceModel hashability...")
try:
    abundance_model = AbundanceModel(NuclearRates(nuclear_net="key_PRIMAT_2023"))
    hash_val = hash(abundance_model)
    print(f"   SUCCESS: AbundanceModel is hashable (hash={hash_val})")
except TypeError as e:
    print(f"   FAILED: AbundanceModel is NOT hashable: {e}")

# Test 4: Inspect the problematic attributes
print("\n4. Inspecting NuclearRates attributes...")
nuclear_net = NuclearRates(nuclear_net="key_PRIMAT_2023")
print(f"   - reactions type: {type(nuclear_net.reactions)}")
print(f"   - reactions length: {len(nuclear_net.reactions)}")
print(f"   - reactions_names type: {type(nuclear_net.reactions_names)}")
print(f"   - in_states type: {type(nuclear_net.in_states)}")
print(f"   - out_states type: {type(nuclear_net.out_states)}")

# Test 5: Inspect Reaction attributes
print("\n5. Inspecting first Reaction attributes...")
if len(nuclear_net.reactions) > 0:
    rxn = nuclear_net.reactions[0]
    print(f"   - Reaction name: {rxn.name}")
    print(f"   - T9_vec type: {type(rxn.T9_vec)}")
    print(f"   - mu_median_vec type: {type(rxn.mu_median_vec)}")
    print(f"   - expsigma_vec type: {type(rxn.expsigma_vec)}")
    if rxn.T9_vec is not None:
        print(f"   - T9_vec is: {type(rxn.T9_vec).__name__}")

# Test 6: Inspect AbundanceModel attributes
print("\n6. Inspecting AbundanceModel attributes...")
abundance_model = AbundanceModel(NuclearRates(nuclear_net="key_PRIMAT_2023"))
print(f"   - species_Z type: {type(abundance_model.species_Z)}")
print(f"   - species_N type: {type(abundance_model.species_N)}")
print(f"   - species_A type: {type(abundance_model.species_A)}")
print(f"   - species_excess_mass type: {type(abundance_model.species_excess_mass)}")
print(f"   - species_spin type: {type(abundance_model.species_spin)}")
print(f"   - species_binding_energy type: {type(abundance_model.species_binding_energy)}")
print(f"   - species_mass type: {type(abundance_model.species_mass)}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
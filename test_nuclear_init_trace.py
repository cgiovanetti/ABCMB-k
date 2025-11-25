import jax
import jax.numpy as jnp
import equinox as eqx
from ABCMB.linx.reactions import Reaction

# Manually trace through NuclearRates.__init__ to find the culprit

print("=" * 60)
print("TRACING NuclearRates.__init__")
print("=" * 60)

# Simulate what happens in NuclearRates.__init__
interp_type = 'linear'
nuclear_net = 'key_PRIMAT_2023'

# Create one reaction
print("\n1. Creating a single Reaction...")
rxn = Reaction(
    'npdg', (0, 1), (2, ), 4.7161402e9, 1.5, -25.81502, 
    spline_data='key_PRIMAT_2023/npdg.txt', 
    interp_type=interp_type
)
print(f"   Reaction created")
print(f"   rxn.frwrd_symmetry_fac type: {type(rxn.frwrd_symmetry_fac)}, is JAX: {isinstance(rxn.frwrd_symmetry_fac, jax.Array)}")

# Create tuple of reactions
print("\n2. Creating tuple of reactions...")
reactions_tuple = tuple([rxn])
print(f"   Tuple created: {type(reactions_tuple)}")

# Create dicts
print("\n3. Creating dicts...")
in_states = {}
out_states = {}
frwrd_symmetry_fac = {}
bkwrd_symmetry_fac = {}

in_states[rxn.name] = rxn.in_states
out_states[rxn.name] = rxn.out_states
frwrd_symmetry_fac[rxn.name] = rxn.frwrd_symmetry_fac
bkwrd_symmetry_fac[rxn.name] = rxn.bkwrd_symmetry_fac

print(f"   frwrd_symmetry_fac['{rxn.name}'] type: {type(frwrd_symmetry_fac[rxn.name])}")
print(f"   Is JAX array: {isinstance(frwrd_symmetry_fac[rxn.name], jax.Array)}")

# Now try to create a simple eqx.Module with these fields
print("\n4. Creating test Module with these fields...")

class TestModule(eqx.Module):
    reactions: tuple = eqx.field(static=True)
    in_states: dict = eqx.field(static=True)
    frwrd_symmetry_fac: dict = eqx.field(static=True)
    
try:
    test_mod = TestModule(
        reactions=reactions_tuple,
        in_states=in_states,
        frwrd_symmetry_fac=frwrd_symmetry_fac
    )
    print("   TestModule created successfully - no warning!")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
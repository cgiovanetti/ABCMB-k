"""
Test to replicate the exact inline JIT pattern from main.py
"""
import jax
import jax.numpy as jnp
from ABCMB.linx.abundances import AbundanceModel
from ABCMB.linx.background import BackgroundModel
from ABCMB.linx.nuclear import NuclearRates

print("=" * 80)
print("INLINE JIT WITH BACKEND PARAMETER TEST")
print("=" * 80)

# Initialize models
print("\n1. Initializing models...")
abundance_model = AbundanceModel(NuclearRates(nuclear_net='key_PRIMAT_2023'))
background_model = BackgroundModel()

# Test 1: Inline JIT without backend (like your working case)
print("\n2. Testing INLINE JIT without backend parameter...")
try:
    # This mimics: result = jax.jit(model)(args)
    result = jax.jit(background_model)(jnp.array(0.0))
    print("   ✓ BackgroundModel inline JIT (no backend) - SUCCESS")
except Exception as e:
    print(f"   ✗ BackgroundModel inline JIT (no backend) - FAILED: {e}")

# Test 2: Inline JIT with backend='cpu' (like your failing case)
print("\n3. Testing INLINE JIT with backend='cpu' parameter...")
try:
    # This mimics: result = jax.jit(model, backend='cpu')(args)
    result = jax.jit(background_model, backend='cpu')(jnp.array(0.0))
    print("   ✓ BackgroundModel inline JIT (backend='cpu') - SUCCESS")
except Exception as e:
    print(f"   ✗ BackgroundModel inline JIT (backend='cpu') - FAILED")
    print(f"   Error: {type(e).__name__}: {e}")

# Test 3: Same for AbundanceModel
print("\n4. Testing AbundanceModel with inline JIT + backend='cpu'...")

# First get some dummy data for AbundanceModel
try:
    bg_result = jax.jit(background_model)(jnp.array(0.0))
    t_vec, a_vec, rho_g, rho_nu, rho_NP, P_NP, Neff = bg_result
    
    print("   Got background data for AbundanceModel test")
    
    # Now try inline JIT with backend='cpu' on AbundanceModel
    result = jax.jit(abundance_model, backend='cpu')(
        rho_g, rho_nu, rho_NP, P_NP,
        t_vec=t_vec, a_vec=a_vec,
        eta_fac=jnp.array(1.0),
        tau_n_fac=jnp.array(1.0),
        nuclear_rates_q=jnp.zeros(len(abundance_model.nuclear_net.reactions))
    )
    print("   ✓ AbundanceModel inline JIT (backend='cpu') - SUCCESS")
except Exception as e:
    print(f"   ✗ AbundanceModel inline JIT (backend='cpu') - FAILED")
    print(f"   Error: {type(e).__name__}: {e}")
    import traceback
    print("\n   Full traceback:")
    traceback.print_exc()

# Test 4: Two-step approach (create jitted function, then call)
print("\n5. Testing two-step approach (create jitted function first)...")
try:
    jitted_abundance = jax.jit(abundance_model, backend='cpu')
    print("   ✓ Created jitted function with backend='cpu'")
    
    result = jitted_abundance(
        rho_g, rho_nu, rho_NP, P_NP,
        t_vec=t_vec, a_vec=a_vec,
        eta_fac=jnp.array(1.0),
        tau_n_fac=jnp.array(1.0),
        nuclear_rates_q=jnp.zeros(len(abundance_model.nuclear_net.reactions))
    )
    print("   ✓ Called jitted function - SUCCESS")
except Exception as e:
    print(f"   ✗ Two-step approach - FAILED")
    print(f"   Error: {type(e).__name__}: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
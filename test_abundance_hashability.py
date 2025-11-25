import jax
import jax.numpy as jnp
from ABCMB.linx.nuclear import NuclearRates
from ABCMB.linx.abundances import AbundanceModel

print("Testing AbundanceModel hashability...")

# Create the models exactly as in main.py
nuclear_rates = NuclearRates(nuclear_net="key_PRIMAT_2023")
abundance_model = AbundanceModel(nuclear_rates)

print("\n1. Testing if AbundanceModel instance is hashable:")
try:
    hash(abundance_model)
    print("   ✓ AbundanceModel instance IS hashable")
except TypeError as e:
    print(f"   ✗ AbundanceModel instance is NOT hashable: {e}")

print("\n2. Testing if NuclearRates instance is hashable:")
try:
    hash(nuclear_rates)
    print("   ✓ NuclearRates instance IS hashable")
except TypeError as e:
    print(f"   ✗ NuclearRates instance is NOT hashable: {e}")

print("\n3. Checking AbundanceModel attributes for JAX arrays:")
for attr_name in dir(abundance_model):
    if not attr_name.startswith('_'):
        attr = getattr(abundance_model, attr_name)
        if isinstance(attr, jax.Array):
            print(f"   - {attr_name}: JAX Array (shape={attr.shape}, dtype={attr.dtype})")
        elif hasattr(attr, '__class__') and 'jax' in str(type(attr)):
            print(f"   - {attr_name}: {type(attr)}")

print("\n4. Checking NuclearRates attributes for JAX arrays:")
for attr_name in dir(nuclear_rates):
    if not attr_name.startswith('_'):
        attr = getattr(nuclear_rates, attr_name)
        if isinstance(attr, jax.Array):
            print(f"   - {attr_name}: JAX Array")
        elif isinstance(attr, dict):
            print(f"   - {attr_name}: dict with {len(attr)} items")
            for k, v in list(attr.items())[:2]:  # Show first 2 items
                if isinstance(v, jax.Array):
                    print(f"      * {k}: JAX Array")
                else:
                    print(f"      * {k}: {type(v)}")

print("\n5. Testing JIT compilation:")
try:
    jitted_model = jax.jit(abundance_model, backend='cpu')
    print("   ✓ JIT compilation succeeded")
except TypeError as e:
    print(f"   ✗ JIT compilation failed: {e}")

print("\n6. Testing if the issue is with calling the JIT'd function:")
if 'jitted_model' in locals():
    try:
        # Create dummy inputs
        rho_g_vec = jnp.ones(100)
        rho_nu_vec = jnp.ones(100)
        rho_NP_vec = jnp.zeros(100)
        P_NP_vec = jnp.zeros(100)
        t_vec = jnp.linspace(0, 1, 100)
        a_vec = jnp.linspace(1, 2, 100)
        eta_fac = jnp.array(1.0)
        tau_n_fac = jnp.array(1.0)
        nuclear_rates_q = jnp.zeros(len(nuclear_rates.reactions))
        
        result = jitted_model(
            rho_g_vec, rho_nu_vec, rho_NP_vec, P_NP_vec,
            t_vec=t_vec, a_vec=a_vec, eta_fac=eta_fac,
            tau_n_fac=tau_n_fac, nuclear_rates_q=nuclear_rates_q
        )
        print("   ✓ Calling JIT'd function succeeded")
    except Exception as e:
        print(f"   ✗ Calling JIT'd function failed: {type(e).__name__}: {e}")
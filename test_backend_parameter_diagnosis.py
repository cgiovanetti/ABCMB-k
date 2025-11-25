"""
Diagnostic test to understand why backend='cpu' causes hashing issues
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from ABCMB.linx.abundances import AbundanceModel
from ABCMB.linx.background import BackgroundModel

print("=" * 80)
print("BACKEND PARAMETER DIAGNOSIS")
print("=" * 80)

# Initialize models
print("\n1. Initializing models...")
abundance_model = AbundanceModel()
background_model = BackgroundModel()

print(f"   AbundanceModel initialized: {type(abundance_model)}")
print(f"   BackgroundModel initialized: {type(background_model)}")

# Test 1: JIT without backend parameter
print("\n2. Testing JIT WITHOUT backend parameter...")
try:
    jitted_abundance_no_backend = jax.jit(abundance_model)
    print("   ✓ AbundanceModel JIT (no backend) - SUCCESS")
except Exception as e:
    print(f"   ✗ AbundanceModel JIT (no backend) - FAILED: {e}")

try:
    jitted_background_no_backend = jax.jit(background_model)
    print("   ✓ BackgroundModel JIT (no backend) - SUCCESS")
except Exception as e:
    print(f"   ✗ BackgroundModel JIT (no backend) - FAILED: {e}")

# Test 2: JIT with backend='cpu' parameter
print("\n3. Testing JIT WITH backend='cpu' parameter...")
try:
    jitted_abundance_cpu = jax.jit(abundance_model, backend='cpu')
    print("   ✓ AbundanceModel JIT (backend='cpu') - SUCCESS")
except Exception as e:
    print(f"   ✗ AbundanceModel JIT (backend='cpu') - FAILED")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {e}")

try:
    jitted_background_cpu = jax.jit(background_model, backend='cpu')
    print("   ✓ BackgroundModel JIT (backend='cpu') - SUCCESS")
except Exception as e:
    print(f"   ✗ BackgroundModel JIT (backend='cpu') - FAILED")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {e}")

# Test 3: Check JAX array types in models
print("\n4. Inspecting JAX array types in models...")

def find_jax_arrays(obj, path=""):
    """Recursively find JAX arrays in an object"""
    arrays = []
    if isinstance(obj, jax.Array):
        arrays.append((path, type(obj), obj.device()))
    elif hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            arrays.extend(find_jax_arrays(value, f"{path}.{key}" if path else key))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            arrays.extend(find_jax_arrays(item, f"{path}[{i}]"))
    return arrays

abundance_arrays = find_jax_arrays(abundance_model)
print(f"\n   JAX arrays in AbundanceModel ({len(abundance_arrays)} found):")
for path, arr_type, device in abundance_arrays[:5]:  # Show first 5
    print(f"     - {path}: {arr_type.__name__}, device={device}")
if len(abundance_arrays) > 5:
    print(f"     ... and {len(abundance_arrays) - 5} more")

background_arrays = find_jax_arrays(background_model)
print(f"\n   JAX arrays in BackgroundModel ({len(background_arrays)} found):")
for path, arr_type, device in background_arrays[:5]:  # Show first 5
    print(f"     - {path}: {arr_type.__name__}, device={device}")
if len(background_arrays) > 5:
    print(f"     ... and {len(background_arrays) - 5} more")

# Test 4: Check if arrays are hashable
print("\n5. Testing array hashability...")
if abundance_arrays:
    test_path, test_type, test_device = abundance_arrays[0]
    test_array = eval(f"abundance_model.{test_path}")
    print(f"   Testing array: {test_path}")
    print(f"   Array type: {test_type}")
    print(f"   Array device: {test_device}")
    try:
        hash(test_array)
        print("   ✓ Array is hashable")
    except TypeError as e:
        print(f"   ✗ Array is NOT hashable: {e}")

# Test 5: Check JAX configuration
print("\n6. JAX Configuration:")
print(f"   JAX version: {jax.__version__}")
print(f"   Default backend: {jax.default_backend()}")
print(f"   Available backends: {jax.devices()}")

# Test 6: Test with explicit device placement
print("\n7. Testing with explicit device placement...")
try:
    with jax.default_device(jax.devices('cpu')[0]):
        jitted_abundance_device = jax.jit(abundance_model)
    print("   ✓ AbundanceModel JIT (with default_device context) - SUCCESS")
except Exception as e:
    print(f"   ✗ AbundanceModel JIT (with default_device context) - FAILED: {e}")

# Test 7: Test static_argnums approach
print("\n8. Testing alternative approaches...")
def wrapper_no_backend(model, *args):
    return model(*args)

def wrapper_with_backend(model, *args):
    return model(*args)

try:
    jitted_wrapper_no_backend = jax.jit(wrapper_no_backend, static_argnums=(0,))
    print("   ✓ Wrapper with static_argnums (no backend) - SUCCESS")
except Exception as e:
    print(f"   ✗ Wrapper with static_argnums (no backend) - FAILED: {e}")

try:
    jitted_wrapper_with_backend = jax.jit(wrapper_with_backend, static_argnums=(0,), backend='cpu')
    print("   ✓ Wrapper with static_argnums (backend='cpu') - SUCCESS")
except Exception as e:
    print(f"   ✗ Wrapper with static_argnums (backend='cpu') - FAILED: {e}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
"""
Minimal Working Example: Equinox Module JIT Compilation Issue with JAX Arrays

CONTEXT FOR EQUINOX TEAM:
We're encountering an "unhashable type: 'jaxlib._jax.ArrayImpl'" error when trying to
JIT-compile Equinox modules that contain JAX array attributes. This MWE demonstrates
the issue in its simplest form.

PROBLEM:
- Equinox modules with JAX array attributes cannot be hashed for JIT compilation
- This occurs even when arrays are created during __init__ and marked as static
- The issue prevents using jax.jit(module, backend='cpu') for CPU execution

WHAT WE'VE TRIED:
1. Marking arrays as static with eqx.field(static=True) - still fails
2. Converting arrays to tuples - still fails (tuples contain unhashable arrays)
3. Using different JAX array creation methods - all fail
4. Attempting to hash the module directly - fails with same error

This MWE shows both a FAILING case (with JAX arrays) and a WORKING case (with primitives).
"""

import jax
import jax.numpy as jnp
import equinox as eqx


# ============================================================================
# FAILING CASE: Module with JAX Arrays
# ============================================================================

class ModuleWithArrays(eqx.Module):
    """
    A minimal Equinox module containing JAX array attributes.
    This mirrors the structure of our actual problem.
    """
    # JAX array attributes created during initialization
    coefficients: jax.Array
    weights: jax.Array
    
    def __init__(self):
        # Create simple JAX arrays
        self.coefficients = jnp.array([1.0, 2.0, 3.0])
        self.weights = jnp.array([0.5, 0.3, 0.2])
    
    def __call__(self, x):
        """Simple computation using the arrays."""
        return jnp.sum(self.coefficients * self.weights) * x


# ============================================================================
# WORKING CASE: Module with Primitive Types
# ============================================================================

class ModuleWithPrimitives(eqx.Module):
    """
    A similar module but with only primitive types.
    This works fine with JIT compilation.
    """
    use_scaling: bool
    factor: float
    
    def __init__(self):
        self.use_scaling = True
        self.factor = 2.0
    
    def __call__(self, x):
        """Simple computation using primitives."""
        if self.use_scaling:
            return x * self.factor
        return x


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    print("=" * 70)
    print("EQUINOX JIT COMPILATION MWE")
    print("=" * 70)
    
    # Test 1: Working case with primitives
    print("\n1. WORKING CASE: Module with primitive types")
    print("-" * 70)
    module_primitives = ModuleWithPrimitives()
    print(f"Module created: {module_primitives}")
    
    try:
        jitted_primitives = jax.jit(module_primitives, backend='cpu')
        result = jitted_primitives(5.0)
        print(f"✓ JIT compilation successful!")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test 2: Failing case with JAX arrays
    print("\n2. FAILING CASE: Module with JAX arrays")
    print("-" * 70)
    module_arrays = ModuleWithArrays()
    print(f"Module created: {module_arrays}")
    print(f"  coefficients: {module_arrays.coefficients}")
    print(f"  weights: {module_arrays.weights}")
    
    try:
        print("\nAttempting JIT compilation...")
        jitted_arrays = jax.jit(module_arrays, backend='cpu')
        result = jitted_arrays(5.0)
        print(f"✓ Unexpected success: {result}")
    except TypeError as e:
        print(f"✗ JIT compilation failed with TypeError:")
        print(f"  {e}")
        print(f"\nThis is the core issue: JAX arrays in Equinox modules cannot be hashed")
        print(f"for JIT compilation, even though they are immutable.")
    
    # Test 3: Direct hashing attempt
    print("\n3. ADDITIONAL TEST: Direct hashing attempt")
    print("-" * 70)
    try:
        hash(module_primitives)
        print(f"✓ Module with primitives is hashable")
    except TypeError as e:
        print(f"✗ Module with primitives is not hashable: {e}")
    
    try:
        hash(module_arrays)
        print(f"✓ Module with arrays is hashable")
    except TypeError as e:
        print(f"✗ Module with arrays is not hashable:")
        print(f"  {e}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The issue is fundamental: Equinox modules containing JAX array attributes")
    print("cannot be hashed, which prevents JIT compilation with jax.jit(module).")
    print("\nThis is problematic because:")
    print("- We need to JIT-compile modules for CPU backend execution")
    print("- The arrays are immutable and created during initialization")
    print("- Marking fields as static doesn't resolve the issue")
    print("\nQuestion for Equinox team:")
    print("Is there a recommended pattern for JIT-compiling modules that contain")
    print("JAX arrays as attributes? Or is this a fundamental limitation?")
    print("=" * 70)


if __name__ == "__main__":
    main()
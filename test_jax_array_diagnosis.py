#!/usr/bin/env python3
"""
Simplified diagnostic script to identify unhashable JAX arrays in AbundanceModel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from ABCMB.linx.nuclear import NuclearRates
from ABCMB.linx.abundances import AbundanceModel
from ABCMB.linx.reactions import Reaction

def check_module_fields(module, module_name, max_depth=2, current_depth=0):
    """Check fields of an Equinox module for JAX arrays."""
    if current_depth >= max_depth:
        return []
    
    issues = []
    indent = "  " * current_depth
    
    print(f"{indent}Checking {module_name}:")
    
    # Get field names from the module
    if hasattr(module, '__dataclass_fields__'):
        field_names = module.__dataclass_fields__.keys()
    else:
        field_names = [name for name in dir(module) if not name.startswith('_')]
    
    for field_name in field_names:
        try:
            field_value = getattr(module, field_name)
        except:
            continue
        
        # Check for JAX arrays
        if isinstance(field_value, jax.Array):
            print(f"{indent}  [JAX ARRAY] {field_name}: shape={field_value.shape}, dtype={field_value.dtype}")
            issues.append({
                'module': module_name,
                'field': field_name,
                'type': 'jax.Array',
                'shape': field_value.shape,
                'dtype': field_value.dtype
            })
        
        # Check for numpy arrays
        elif isinstance(field_value, np.ndarray):
            print(f"{indent}  [NUMPY] {field_name}: shape={field_value.shape}, dtype={field_value.dtype}")
        
        # Check for nested modules (but limit depth)
        elif isinstance(field_value, eqx.Module) and current_depth < max_depth - 1:
            nested_issues = check_module_fields(
                field_value, 
                f"{module_name}.{field_name}",
                max_depth,
                current_depth + 1
            )
            issues.extend(nested_issues)
        
        # Check tuples/lists of modules (only first few items)
        elif isinstance(field_value, (tuple, list)) and len(field_value) > 0:
            if isinstance(field_value[0], eqx.Module):
                print(f"{indent}  [COLLECTION] {field_name}: {len(field_value)} items")
                # Only check first item to avoid explosion
                if current_depth < max_depth - 1:
                    nested_issues = check_module_fields(
                        field_value[0],
                        f"{module_name}.{field_name}[0]",
                        max_depth,
                        current_depth + 1
                    )
                    issues.extend(nested_issues)
    
    return issues

def main():
    print("="*80)
    print("JAX ARRAY HASHABILITY DIAGNOSIS")
    print("="*80)
    
    # Create the models
    print("\n1. Creating NuclearRates...")
    nuclear_rates = NuclearRates(nuclear_net='key_PRIMAT_2023')
    
    print("\n2. Creating AbundanceModel...")
    abundance_model = AbundanceModel(nuclear_rates)
    
    # Check for unhashable JAX arrays
    print("\n3. Scanning for JAX arrays (limited depth to avoid recursion)...")
    print()
    issues = check_module_fields(abundance_model, "AbundanceModel", max_depth=3)
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    if issues:
        print(f"\nFound {len(issues)} JAX array(s) in module attributes:")
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. {issue['module']}.{issue['field']}")
            print(f"   Shape: {issue['shape']}, Dtype: {issue['dtype']}")
        
        print("\n" + "="*80)
        print("ROOT CAUSE")
        print("="*80)
        print("\nJAX arrays stored as Equinox module attributes are unhashable.")
        print("This prevents JIT compilation from working.")
        
        print("\nThe JAX arrays are created in:")
        print("  - Reaction.__init__(): frwrd_symmetry_fac, bkwrd_symmetry_fac (from jnp.prod)")
        print("  - AbundanceModel.__init__(): species_Z, species_N, species_excess_mass, etc.")
        
    else:
        print("\nNo JAX arrays found in module attributes.")
    
    # Try to JIT compile
    print("\n" + "="*80)
    print("ATTEMPTING JIT COMPILATION")
    print("="*80)
    
    try:
        print("\nAttempting: jax.jit(abundance_model, backend='cpu')")
        jitted = jax.jit(abundance_model, backend='cpu')
        print("SUCCESS! (Unexpected if JAX arrays are present)")
    except TypeError as e:
        print(f"\nFAILED with TypeError:")
        print(f"  {str(e)[:200]}...")
        print("\nThis confirms JAX arrays in module attributes prevent JIT compilation.")

if __name__ == "__main__":
    main()
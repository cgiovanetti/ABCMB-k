#!/usr/bin/env python3
"""
Diagnostic script to identify unhashable JAX arrays in AbundanceModel hierarchy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from ABCMB.linx.nuclear import NuclearRates
from ABCMB.linx.abundances import AbundanceModel

def check_hashability(obj, path="root"):
    """Recursively check if an object and its attributes are hashable."""
    issues = []
    
    if isinstance(obj, eqx.Module):
        print(f"\n{'='*60}")
        print(f"Checking Equinox Module: {path}")
        print(f"Type: {type(obj).__name__}")
        print(f"{'='*60}")
        
        # Get all fields
        for field_name in dir(obj):
            if field_name.startswith('_'):
                continue
            
            try:
                field_value = getattr(obj, field_name)
            except:
                continue
            
            # Check if it's a JAX array
            if isinstance(field_value, jax.Array):
                field_path = f"{path}.{field_name}"
                print(f"  [JAX ARRAY] {field_name}: shape={field_value.shape}, dtype={field_value.dtype}")
                issues.append({
                    'path': field_path,
                    'type': 'jax.Array',
                    'shape': field_value.shape,
                    'dtype': field_value.dtype
                })
            
            # Check if it's a numpy array
            elif isinstance(field_value, np.ndarray):
                field_path = f"{path}.{field_name}"
                print(f"  [NUMPY ARRAY] {field_name}: shape={field_value.shape}, dtype={field_value.dtype}")
            
            # Check if it's a nested Module
            elif isinstance(field_value, eqx.Module):
                nested_issues = check_hashability(field_value, f"{path}.{field_name}")
                issues.extend(nested_issues)
            
            # Check if it's a tuple/list of Modules
            elif isinstance(field_value, (tuple, list)) and len(field_value) > 0:
                if isinstance(field_value[0], eqx.Module):
                    print(f"  [COLLECTION] {field_name}: {len(field_value)} items")
                    for i, item in enumerate(field_value):
                        nested_issues = check_hashability(item, f"{path}.{field_name}[{i}]")
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
    print("\n3. Scanning for JAX arrays in AbundanceModel hierarchy...")
    issues = check_hashability(abundance_model, "AbundanceModel")
    
    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    
    if issues:
        print(f"\nFound {len(issues)} JAX array(s) that will cause hashing issues:")
        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. {issue['path']}")
            print(f"   Type: {issue['type']}")
            print(f"   Shape: {issue['shape']}")
            print(f"   Dtype: {issue['dtype']}")
        
        print("\n" + "="*80)
        print("ROOT CAUSE")
        print("="*80)
        print("\nJAX arrays are unhashable because they are mutable and their values")
        print("can change. Equinox modules require all fields to be hashable for")
        print("JIT compilation to work properly.")
        
        print("\n" + "="*80)
        print("COMPARISON WITH BackgroundModel")
        print("="*80)
        print("\nBackgroundModel only stores boolean flags (hashable primitives).")
        print("It does NOT store any JAX arrays as module attributes.")
        print("All arrays are created dynamically during computation.")
        
    else:
        print("\nNo JAX arrays found - this shouldn't happen!")
    
    # Try to hash it
    print("\n" + "="*80)
    print("ATTEMPTING TO JIT COMPILE")
    print("="*80)
    
    try:
        print("\nAttempting: jax.jit(abundance_model, backend='cpu')")
        jitted = jax.jit(abundance_model, backend='cpu')
        print("SUCCESS! (This shouldn't happen if JAX arrays are present)")
    except TypeError as e:
        print(f"FAILED with TypeError: {e}")
        print("\nThis confirms the diagnosis: JAX arrays in module attributes")
        print("prevent JIT compilation.")

if __name__ == "__main__":
    main()
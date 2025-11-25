"""
Diagnostic script to find JAX arrays that are not marked as static in AbundanceModel hierarchy.
This will help identify the exact source of the "unhashable type: 'jaxlib._jax.ArrayImpl'" error.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any
import sys

# Import the classes we need to inspect
sys.path.insert(0, '/Users/caragiovanetti/Research/GitHub/ABCMB')
from ABCMB.linx.abundances import AbundanceModel
from ABCMB.linx.nuclear import NuclearRates, Reaction
from ABCMB.linx.weak_rates import WeakRates


def is_jax_array(obj: Any) -> bool:
    """Check if an object is a JAX array."""
    return isinstance(obj, (jax.Array, jnp.ndarray)) or hasattr(obj, '__array__')


def is_static_field(cls, field_name: str) -> bool:
    """Check if a field is marked as static in an Equinox module."""
    if hasattr(cls, '__dataclass_fields__'):
        field = cls.__dataclass_fields__.get(field_name)
        if field and hasattr(field, 'metadata'):
            return field.metadata.get('static', False)
    return False


def check_hashability(obj: Any) -> tuple[bool, str]:
    """Check if an object is hashable and return reason if not."""
    try:
        hash(obj)
        return True, "hashable"
    except TypeError as e:
        return False, str(e)


def inspect_object(obj: Any, obj_name: str = "root", depth: int = 0, max_depth: int = 5) -> list[dict]:
    """
    Recursively inspect an object and its attributes to find JAX arrays.
    
    Returns a list of dictionaries containing information about each attribute.
    """
    if depth > max_depth:
        return []
    
    results = []
    indent = "  " * depth
    
    # Get the class of the object
    obj_class = type(obj)
    
    print(f"{indent}Inspecting: {obj_name} (type: {obj_class.__name__})")
    
    # Check if this is an Equinox module
    is_eqx_module = isinstance(obj, eqx.Module)
    
    if is_eqx_module:
        print(f"{indent}  -> This is an Equinox Module")
        
        # Get all fields using eqx.tree_flatten_one_level
        try:
            leaves, treedef = jax.tree_util.tree_flatten_one_level(obj)
            field_names = list(obj.__dataclass_fields__.keys()) if hasattr(obj, '__dataclass_fields__') else []
            
            print(f"{indent}  -> Found {len(field_names)} fields")
            
            for field_name in field_names:
                try:
                    attr_value = getattr(obj, field_name)
                    is_static = is_static_field(obj_class, field_name)
                    is_jax = is_jax_array(attr_value)
                    is_hashable, hash_reason = check_hashability(attr_value)
                    
                    attr_type = type(attr_value).__name__
                    
                    result = {
                        'path': f"{obj_name}.{field_name}",
                        'class': obj_class.__name__,
                        'field_name': field_name,
                        'type': attr_type,
                        'is_jax_array': is_jax,
                        'is_static': is_static,
                        'is_hashable': is_hashable,
                        'hash_reason': hash_reason,
                        'depth': depth
                    }
                    
                    results.append(result)
                    
                    # Print immediate findings
                    status = []
                    if is_jax:
                        status.append("JAX_ARRAY")
                    if is_static:
                        status.append("STATIC")
                    if not is_hashable:
                        status.append("UNHASHABLE")
                    
                    status_str = ", ".join(status) if status else "OK"
                    print(f"{indent}    {field_name}: {attr_type} [{status_str}]")
                    
                    # Flag critical issues
                    if is_jax and not is_static:
                        print(f"{indent}      ⚠️  JAX ARRAY NOT MARKED AS STATIC!")
                    if is_jax and is_static:
                        print(f"{indent}      ℹ️  JAX array correctly marked as static (may cause warning)")
                    if not is_hashable and not is_static:
                        print(f"{indent}      🔴 UNHASHABLE AND NOT STATIC - LIKELY CULPRIT!")
                    
                    # Recursively inspect nested Equinox modules
                    if isinstance(attr_value, eqx.Module) and depth < max_depth:
                        nested_results = inspect_object(attr_value, f"{obj_name}.{field_name}", depth + 1, max_depth)
                        results.extend(nested_results)
                        
                except Exception as e:
                    print(f"{indent}    {field_name}: ERROR - {e}")
                    
        except Exception as e:
            print(f"{indent}  -> Error inspecting module: {e}")
    
    return results


def main():
    """Main diagnostic function."""
    print("=" * 80)
    print("JAX ARRAY STATIC FIELD DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Create a minimal AbundanceModel instance to inspect
    print("Creating AbundanceModel instance for inspection...")
    print()
    
    try:
        # We need to create an instance - use default parameters
        # This might fail, but we can still inspect the class structure
        abundance_model = AbundanceModel()
        
        print("Successfully created AbundanceModel instance")
        print()
        
        # Inspect the entire hierarchy
        results = inspect_object(abundance_model, "AbundanceModel", depth=0, max_depth=5)
        
        print()
        print("=" * 80)
        print("SUMMARY OF FINDINGS")
        print("=" * 80)
        print()
        
        # Filter and categorize results
        jax_not_static = [r for r in results if r['is_jax_array'] and not r['is_static']]
        jax_static = [r for r in results if r['is_jax_array'] and r['is_static']]
        unhashable_not_static = [r for r in results if not r['is_hashable'] and not r['is_static']]
        
        print(f"Total attributes inspected: {len(results)}")
        print(f"JAX arrays marked as static: {len(jax_static)}")
        print(f"JAX arrays NOT marked as static: {len(jax_not_static)}")
        print(f"Unhashable attributes NOT marked as static: {len(unhashable_not_static)}")
        print()
        
        if jax_not_static:
            print("🔴 CRITICAL: JAX arrays NOT marked as static:")
            print("-" * 80)
            for r in jax_not_static:
                print(f"  Path: {r['path']}")
                print(f"  Class: {r['class']}")
                print(f"  Type: {r['type']}")
                print(f"  Hashable: {r['is_hashable']} ({r['hash_reason']})")
                print()
        
        if unhashable_not_static:
            print("🔴 CRITICAL: Unhashable attributes NOT marked as static:")
            print("-" * 80)
            for r in unhashable_not_static:
                print(f"  Path: {r['path']}")
                print(f"  Class: {r['class']}")
                print(f"  Type: {r['type']}")
                print(f"  Is JAX array: {r['is_jax_array']}")
                print(f"  Hash reason: {r['hash_reason']}")
                print()
        
        if jax_static:
            print("ℹ️  JAX arrays correctly marked as static (these cause warnings but are OK):")
            print("-" * 80)
            for r in jax_static:
                print(f"  Path: {r['path']}")
                print(f"  Class: {r['class']}")
                print()
        
        # Final diagnosis
        print()
        print("=" * 80)
        print("DIAGNOSIS")
        print("=" * 80)
        
        if jax_not_static or unhashable_not_static:
            print("❌ Found attributes that need to be marked as static!")
            print()
            print("These attributes are causing the 'unhashable type' error:")
            for r in (jax_not_static + unhashable_not_static):
                print(f"  - {r['path']} in {r['class']}")
        else:
            print("✅ All JAX arrays appear to be marked as static.")
            print("The error may be coming from a different source.")
            print("Possible causes:")
            print("  1. Dynamic creation of JAX arrays during __init__")
            print("  2. JAX arrays in nested structures (lists, dicts)")
            print("  3. JAX arrays created in methods rather than fields")
        
    except Exception as e:
        print(f"Error creating AbundanceModel: {e}")
        print()
        print("Attempting to inspect class structure without instance...")
        print()
        
        # Inspect the class structure directly
        print("Inspecting AbundanceModel class fields:")
        if hasattr(AbundanceModel, '__dataclass_fields__'):
            for field_name, field in AbundanceModel.__dataclass_fields__.items():
                is_static = field.metadata.get('static', False) if hasattr(field, 'metadata') else False
                print(f"  {field_name}: {field.type} [{'STATIC' if is_static else 'NOT STATIC'}]")
        
        print()
        print("Inspecting NuclearRates class fields:")
        if hasattr(NuclearRates, '__dataclass_fields__'):
            for field_name, field in NuclearRates.__dataclass_fields__.items():
                is_static = field.metadata.get('static', False) if hasattr(field, 'metadata') else False
                print(f"  {field_name}: {field.type} [{'STATIC' if is_static else 'NOT STATIC'}]")
        
        print()
        print("Inspecting Reaction class fields:")
        if hasattr(Reaction, '__dataclass_fields__'):
            for field_name, field in Reaction.__dataclass_fields__.items():
                is_static = field.metadata.get('static', False) if hasattr(field, 'metadata') else False
                print(f"  {field_name}: {field.type} [{'STATIC' if is_static else 'NOT STATIC'}]")
        
        print()
        print("Inspecting WeakRates class fields:")
        if hasattr(WeakRates, '__dataclass_fields__'):
            for field_name, field in WeakRates.__dataclass_fields__.items():
                is_static = field.metadata.get('static', False) if hasattr(field, 'metadata') else False
                print(f"  {field_name}: {field.type} [{'STATIC' if is_static else 'NOT STATIC'}]")


if __name__ == "__main__":
    main()
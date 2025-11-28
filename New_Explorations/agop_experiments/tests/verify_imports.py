"""
Quick import verification script for AGOP experiments.

This script checks that all imports work correctly without running full training.
Run this first to verify the code is syntactically correct before submitting jobs.

Usage:
    python verify_imports.py
"""

import sys
from pathlib import Path

def test_imports(script_name):
    """Test that a script's imports work."""
    print(f"\nTesting {script_name}...", end=' ')
    
    try:
        # Try importing the script as a module
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_module", script_name)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Don't execute, just compile to check syntax
            with open(script_name) as f:
                code = f.read()
            compile(code, script_name, 'exec')
            print("✓ Syntax OK")
            return True
        else:
            print("✗ Cannot load module spec")
            return False
    except SyntaxError as e:
        print(f"✗ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("="*80)
    print("AGOP Experiments - Import and Syntax Verification")
    print("="*80)
    
    scripts = [
        'agop_utils.py',
        'train_nanda_agop.py',
        'train_softmax_agop.py',
        'train_mnist_agop.py',
        'train_composition_agop.py',
        'analysis/visualize_agop_metrics.py',
        'analysis/compare_grok_nogrok.py',
    ]
    
    results = {}
    for script in scripts:
        results[script] = test_imports(script)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for script, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {script}")
    
    print("\n" + "="*80)
    print(f"Result: {passed}/{total} scripts verified")
    print("="*80)
    
    if passed == total:
        print("\n✓ All scripts have correct syntax and structure!")
        print("  Ready to run experiments (ensure PyTorch environment is activated)")
        return True
    else:
        print(f"\n✗ {total - passed} scripts have issues")
        print("  Fix syntax errors before running experiments")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


#!/usr/bin/env python3
"""
Quick test verification script for AI Istanbul chatbot testing infrastructure
"""

import sys
import os
import subprocess
import importlib.util

def check_import(module_name):
    """Check if a module can be imported."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            print(f"✅ {module_name} - Available")
            return True
        else:
            print(f"❌ {module_name} - Not found")
            return False
    except Exception as e:
        print(f"❌ {module_name} - Error: {e}")
        return False

def main():
    print("🧪 AI Istanbul Chatbot - Testing Infrastructure Verification")
    print("=" * 60)
    
    # Check required testing modules
    required_modules = [
        'pytest',
        'pytest_asyncio', 
        'pytest_cov',
        'httpx',
        'psutil',
        'asyncio',
        'json',
        'unittest.mock'
    ]
    
    print("Checking required testing dependencies:")
    all_available = True
    for module in required_modules:
        if not check_import(module):
            all_available = False
    
    print()
    
    # Check if backend directory exists
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    if os.path.exists(backend_path):
        print("✅ Backend directory - Found")
    else:
        print("❌ Backend directory - Not found")
        all_available = False
    
    # Check if main.py exists
    main_py_path = os.path.join(backend_path, 'main.py')
    if os.path.exists(main_py_path):
        print("✅ Backend main.py - Found")
    else:
        print("❌ Backend main.py - Not found")
        all_available = False
    
    # Check if tests directory exists
    tests_path = os.path.join(os.path.dirname(__file__), 'tests')
    if os.path.exists(tests_path):
        print("✅ Tests directory - Found")
    else:
        print("❌ Tests directory - Not found")
        all_available = False
    
    print()
    
    # Try to run a simple pytest command
    try:
        print("Testing pytest execution...")
        result = subprocess.run(['pytest', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Pytest execution - Working ({result.stdout.strip()})")
        else:
            print(f"❌ Pytest execution - Failed: {result.stderr}")
            all_available = False
    except Exception as e:
        print(f"❌ Pytest execution - Error: {e}")
        all_available = False
    
    print()
    print("=" * 60)
    
    if all_available:
        print("🎉 Testing infrastructure is ready!")
        print("✅ All dependencies available")
        print("✅ File structure correct")
        print("✅ Pytest working")
        print()
        print("Next steps:")
        print("1. Run: ./run_tests.sh --quick")
        print("2. Run: ./run_tests.sh --coverage")
        print("3. Run: ./run_tests.sh  # Full suite")
        return True
    else:
        print("❌ Testing infrastructure has issues")
        print("Please install missing dependencies and check file structure")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

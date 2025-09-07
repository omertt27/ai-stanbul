#!/usr/bin/env python3
"""
Dependency checker and installer for AI-stanbul backend
Ensures all required packages are installed before starting the server
"""

import sys
import subprocess
import importlib

def install_package(package_name, package_spec=None):
    """Install a package using pip"""
    if package_spec is None:
        package_spec = package_name
    
    try:
        print(f"📦 Installing {package_name}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", package_spec
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def check_and_install_dependencies():
    """Check for required dependencies and install if missing"""
    print("🔍 Checking dependencies...")
    
    dependencies = [
        ("multipart", "python-multipart==0.0.6"),
        ("fuzzywuzzy", "fuzzywuzzy==0.18.0"),
        ("Levenshtein", "python-levenshtein==0.20.9"),
        ("fastapi", "fastapi==0.104.1"),
        ("uvicorn", "uvicorn[standard]==0.24.0"),
        ("sqlalchemy", "sqlalchemy==2.0.23"),
        ("openai", "openai==1.3.0"),
        ("requests", "requests==2.31.0"),
    ]
    
    missing_deps = []
    installed_deps = []
    
    for module_name, package_spec in dependencies:
        try:
            importlib.import_module(module_name)
            installed_deps.append(module_name)
        except ImportError:
            missing_deps.append((module_name, package_spec))
    
    print(f"✅ Found {len(installed_deps)} dependencies already installed")
    
    if missing_deps:
        print(f"📦 Installing {len(missing_deps)} missing dependencies...")
        for module_name, package_spec in missing_deps:
            success = install_package(module_name, package_spec)
            if not success:
                print(f"⚠️  Warning: Could not install {module_name}")
    else:
        print("✅ All dependencies are already installed")
    
    # Verify critical dependencies
    critical_deps = ["multipart", "fastapi", "uvicorn"]
    print("🔍 Verifying critical dependencies...")
    
    for dep in critical_deps:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep} verified")
        except ImportError:
            print(f"❌ Critical dependency {dep} not available")
            return False
    
    print("🚀 All dependencies ready!")
    return True

if __name__ == "__main__":
    success = check_and_install_dependencies()
    if success:
        print("✅ Dependencies check completed successfully")
        sys.exit(0)
    else:
        print("❌ Dependencies check failed")
        sys.exit(1)

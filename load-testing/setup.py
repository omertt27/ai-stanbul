#!/usr/bin/env python3
"""
AI Istanbul Load Testing Suite Setup

This script sets up the testing environment and installs dependencies.
"""

import os
import subprocess
import sys
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🔧 AI ISTANBUL LOAD TESTING SETUP")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install Python dependencies."""
    print("\n📦 Installing Python dependencies...")
    
    requirements_file = Path(__file__).parent / 'requirements.txt'
    
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found!")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"💥 Exception installing dependencies: {str(e)}")
        return False

def install_playwright():
    """Install Playwright browsers for frontend testing."""
    print("\n🌐 Installing Playwright browsers...")
    
    try:
        # Install playwright browsers
        result = subprocess.run([
            sys.executable, '-m', 'playwright', 'install'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Playwright browsers installed successfully")
            return True
        else:
            print(f"⚠️  Playwright installation warning:")
            print(result.stderr)
            print("   Frontend tests may not work properly")
            return True  # Don't fail setup for this
            
    except Exception as e:
        print(f"⚠️  Exception installing Playwright: {str(e)}")
        print("   Frontend tests may not work properly")
        return True  # Don't fail setup for this

def create_results_directory():
    """Create results directory for test outputs."""
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    print(f"✅ Results directory created: {results_dir}")
    return True

def verify_installation():
    """Verify that key packages are importable."""
    print("\n🔍 Verifying installation...")
    
    test_imports = [
        ('requests', 'HTTP client'),
        ('matplotlib', 'Chart generation'),
        ('pandas', 'Data analysis'),
        ('locust', 'Load testing framework'),
        ('psutil', 'System monitoring'),
        ('playwright', 'Frontend testing')
    ]
    
    failed_imports = []
    
    for package, description in test_imports:
        try:
            __import__(package)
            print(f"✅ {package:12} - {description}")
        except ImportError:
            print(f"❌ {package:12} - {description} (FAILED)")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Some packages failed to import: {', '.join(failed_imports)}")
        print("   Some tests may not work properly")
        return False
    
    print("\n🎉 All packages verified successfully!")
    return True

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🚀 SETUP COMPLETE - NEXT STEPS")
    print("=" * 60)
    print()
    print("1. Start your AI Istanbul application:")
    print("   cd /Users/omer/Desktop/ai-stanbul")
    print("   npm run dev  # Start frontend")
    print("   # Start backend server separately")
    print()
    print("2. Run load tests:")
    print("   cd load-testing")
    print("   python run_tests.py --help                    # Show options")
    print("   python run_tests.py --list                    # List available tests")
    print("   python run_tests.py --tests load stress       # Run specific tests")
    print("   python run_tests.py --env production          # Test production")
    print()
    print("3. View results:")
    print("   open results/load_test_report.html            # View HTML report")
    print("   ls results/                                   # Check result files")
    print()
    print("📚 For detailed documentation, see: README.md")

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Install Playwright (optional)
    install_playwright()
    
    # Create directories
    if not create_results_directory():
        return 1
    
    # Verify installation
    verify_installation()
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

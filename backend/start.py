#!/usr/bin/env python3
"""
Production-ready start script for AI-stanbul backend
Automatically detects and uses the best available backend version
"""
import sys
import os
import uvicorn
import importlib

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def check_dependencies():
    """Quick dependency check"""
    try:
        from check_dependencies import check_and_install_dependencies
        return check_and_install_dependencies()
    except Exception as e:
        print(f"⚠️  Dependency checker not available: {e}")
        return True  # Continue anyway

def test_imports():
    """Test if all required modules can be imported"""
    modules_to_test = [
        'database',
        'models', 
        'enhanced_chatbot',
        'fuzzywuzzy'
    ]
    
    missing_modules = []
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            missing_modules.append(module)
    
    return len(missing_modules) == 0

if __name__ == "__main__":
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print("🚀 AI-stanbul Backend Starting...")
    
    # Check dependencies first
    deps_ok = check_dependencies()
    if not deps_ok:
        print("⚠️  Some dependencies missing, but continuing...")
    
    print("🔍 Testing module imports...")
    
    # Test if we can use the full main.py
    if test_imports():
        print("✅ All modules available - using full backend (main.py)")
        app_module = "main:app"
    else:
        print("⚠️  Some modules missing - using standalone backend (main_standalone.py)")
        app_module = "main_standalone:app"
    
    print(f"🌐 Starting server on {host}:{port}")
    print(f"📁 Using module: {app_module}")
    
    # Start the uvicorn server
    try:
        uvicorn.run(
            app_module, 
            host=host, 
            port=port, 
            reload=False,
            log_level="info"
        )
    except Exception as e:
        print(f"❌ Failed to start with {app_module}: {e}")
        if app_module == "main:app":
            print("🔄 Falling back to standalone mode...")
            uvicorn.run(
                "main_standalone:app", 
                host=host, 
                port=port, 
                reload=False,
                log_level="info"
            )
        else:
            raise e

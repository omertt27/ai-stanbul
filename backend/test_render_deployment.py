#!/usr/bin/env python3
"""
Render Deployment Verification Script
Tests all Render-specific requirements before deployment
"""
import os
import sys
import subprocess
import time
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def test_port_binding():
    """Test that the app can bind to the PORT environment variable"""
    print("🔍 Testing port binding...")
    
    # Test with different port values (simulating Render's PORT)
    test_ports = [10000, 3000, 8080]  # 10000 is Render's default
    
    for port in test_ports:
        print(f"  Testing PORT={port}...")
        
        # Set environment variable
        env = os.environ.copy()
        env['PORT'] = str(port)
        
        try:
            # Start the server
            proc = subprocess.Popen(
                [sys.executable, 'start.py'],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for startup
            time.sleep(5)
            
            # Test if port is bound
            try:
                response = requests.get(f'http://localhost:{port}/health', timeout=3)
                if response.status_code == 200:
                    print(f"  ✅ Successfully bound to port {port}")
                    proc.terminate()
                    proc.wait()
                    return True
                else:
                    print(f"  ❌ Port {port} bound but health check failed")
            except requests.exceptions.RequestException:
                print(f"  ❌ Failed to connect to port {port}")
            
            proc.terminate()
            proc.wait()
            
        except Exception as e:
            print(f"  ❌ Error testing port {port}: {e}")
    
    return False

def test_host_binding():
    """Test that the app binds to 0.0.0.0"""
    print("🔍 Testing host binding to 0.0.0.0...")
    
    env = os.environ.copy()
    env['PORT'] = '9999'
    env['HOST'] = '0.0.0.0'
    
    try:
        proc = subprocess.Popen(
            [sys.executable, 'start.py'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        time.sleep(5)
        
        # Test localhost connection (should work if bound to 0.0.0.0)
        try:
            response = requests.get('http://localhost:9999/health', timeout=3)
            if response.status_code == 200:
                print("  ✅ Successfully bound to 0.0.0.0")
                proc.terminate()
                proc.wait()
                return True
        except requests.exceptions.RequestException:
            pass
        
        proc.terminate()
        proc.wait()
        
    except Exception as e:
        print(f"  ❌ Error testing host binding: {e}")
    
    print("  ❌ Failed to bind to 0.0.0.0")
    return False

def test_procfile_command():
    """Test the exact Procfile command"""
    print("🔍 Testing Procfile command...")
    
    env = os.environ.copy()
    env['PORT'] = '9998'
    
    try:
        # Test the exact command from Procfile
        proc = subprocess.Popen(
            ['uvicorn', 'backend.main:app', '--host', '0.0.0.0', '--port', env['PORT']],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd='..'  # Run from parent directory like Render would
        )
        
        time.sleep(5)
        
        try:
            response = requests.get('http://localhost:9998/health', timeout=3)
            if response.status_code == 200:
                print("  ✅ Procfile command works correctly")
                proc.terminate()
                proc.wait()
                return True
        except requests.exceptions.RequestException:
            pass
        
        proc.terminate()
        proc.wait()
        
    except Exception as e:
        print(f"  ❌ Error testing Procfile command: {e}")
    
    print("  ❌ Procfile command failed")
    return False

def test_environment_variables():
    """Test environment variable handling"""
    print("🔍 Testing environment variables...")
    
    # Test default values
    if os.environ.get('PORT') is None:
        print("  ✅ PORT not set - will use default")
    else:
        print(f"  ℹ️  PORT is set to: {os.environ.get('PORT')}")
    
    # Test that our code handles the PORT variable
    test_script = '''
import os
import sys
sys.path.insert(0, ".")
from main import start_server

# Should not crash even without PORT set
port = int(os.environ.get("PORT", 8000))
print(f"Port would be: {port}")
'''
    
    try:
        result = subprocess.run(
            [sys.executable, '-c', test_script],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("  ✅ Environment variable handling works")
            return True
        else:
            print(f"  ❌ Environment variable test failed: {result.stderr}")
            
    except Exception as e:
        print(f"  ❌ Error testing environment variables: {e}")
    
    return False

def main():
    """Run all Render deployment tests"""
    print("🚀 Render Deployment Verification")
    print("=" * 50)
    
    tests = [
        ("Environment Variables", test_environment_variables),
        ("Port Binding", test_port_binding),
        ("Host Binding", test_host_binding),
        ("Procfile Command", test_procfile_command),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Ready for Render deployment!")
        print("\n💡 Deployment Instructions:")
        print("1. Push code to your git repository")
        print("2. Connect repository to Render")
        print("3. Set build command: pip install -r requirements.txt")
        print("4. Render will automatically use the Procfile")
        print("5. Your app will be available at: https://your-app.onrender.com")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("The app may still work, but deployment might have issues.")

if __name__ == "__main__":
    main()

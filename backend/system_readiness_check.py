#!/usr/bin/env python3
"""
Comprehensive AI Istanbul System Readiness Check
Check all components and provide a complete status report
"""

import requests
import json
import sqlite3
import subprocess
import time
import os
from typing import Dict, List

def check_backend_server() -> Dict:
    """Check if backend server is running and responsive"""
    try:
        # Basic health check
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            return {"status": "✅ RUNNING", "message": "Backend server is responsive"}
        else:
            return {"status": "❌ ERROR", "message": f"Server returned {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "❌ DOWN", "message": f"Server not accessible: {str(e)}"}

def check_ai_endpoint() -> Dict:
    """Test the main AI endpoint functionality"""
    test_queries = [
        "hello",
        "restaurants in beyoglu", 
        "how to get from kadikoy to sultanahmet",
        "museums in istanbul"
    ]
    
    results = []
    for query in test_queries:
        try:
            response = requests.post(
                "http://localhost:8000/ai",
                json={"query": query, "session_id": "readiness_test"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                message_length = len(data.get("message", ""))
                results.append({
                    "query": query,
                    "status": "✅ SUCCESS",
                    "response_length": message_length
                })
            else:
                results.append({
                    "query": query, 
                    "status": "❌ FAILED",
                    "error": f"HTTP {response.status_code}"
                })
        except Exception as e:
            results.append({
                "query": query,
                "status": "❌ FAILED", 
                "error": str(e)
            })
    
    success_count = sum(1 for r in results if r["status"] == "✅ SUCCESS")
    success_rate = (success_count / len(test_queries)) * 100
    
    return {
        "status": "✅ WORKING" if success_rate >= 75 else "⚠️ PARTIAL" if success_rate > 0 else "❌ FAILED",
        "success_rate": f"{success_rate:.1f}%",
        "details": results
    }

def check_database() -> Dict:
    """Check database connectivity and data"""
    try:
        conn = sqlite3.connect('/Users/omer/Desktop/ai-stanbul/backend/app.db')
        cursor = conn.cursor()
        
        # Check places table
        cursor.execute("SELECT COUNT(*) FROM places")
        places_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT category, COUNT(*) FROM places GROUP BY category")
        categories = cursor.fetchall()
        
        # Check museums specifically
        cursor.execute("SELECT COUNT(*) FROM places WHERE category LIKE '%Museum%'")
        museums_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "✅ CONNECTED",
            "places_total": places_count,
            "museums_count": museums_count,
            "categories": dict(categories)
        }
    except Exception as e:
        return {"status": "❌ ERROR", "message": str(e)}

def check_frontend() -> Dict:
    """Check if frontend files exist and can be served"""
    frontend_path = "/Users/omer/Desktop/ai-stanbul/frontend"
    
    if not os.path.exists(frontend_path):
        return {"status": "❌ MISSING", "message": "Frontend directory not found"}
    
    required_files = [
        "package.json",
        "src/App.jsx", 
        "src/main.jsx",
        "index.html"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(frontend_path, file)):
            missing_files.append(file)
    
    if missing_files:
        return {
            "status": "⚠️ INCOMPLETE", 
            "message": f"Missing files: {', '.join(missing_files)}"
        }
    
    # Check if we can run npm commands
    try:
        result = subprocess.run(
            ["npm", "list"], 
            cwd=frontend_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return {"status": "✅ READY", "message": "Frontend files and dependencies OK"}
        else:
            return {"status": "⚠️ DEPS", "message": "Dependencies may need installation"}
    except Exception as e:
        return {"status": "⚠️ UNKNOWN", "message": f"Could not check dependencies: {str(e)}"}

def check_environment() -> Dict:
    """Check environment variables and configuration"""
    env_file = "/Users/omer/Desktop/ai-stanbul/backend/.env"
    
    if not os.path.exists(env_file):
        return {"status": "⚠️ MISSING", "message": ".env file not found"}
    
    try:
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        required_vars = ["OPENAI_API_KEY", "GOOGLE_MAPS_API_KEY"]
        found_vars = []
        
        for var in required_vars:
            if var in env_content:
                found_vars.append(var)
        
        if len(found_vars) == len(required_vars):
            return {"status": "✅ CONFIGURED", "vars_found": found_vars}
        else:
            missing = set(required_vars) - set(found_vars)
            return {
                "status": "⚠️ PARTIAL", 
                "vars_found": found_vars,
                "missing": list(missing)
            }
    except Exception as e:
        return {"status": "❌ ERROR", "message": str(e)}

def run_readiness_check():
    """Run complete system readiness check"""
    
    print("🔍 AI ISTANBUL SYSTEM READINESS CHECK")
    print("=" * 50)
    
    # Backend Server Check
    print("\n1. 🖥️  Backend Server Status")
    backend_result = check_backend_server()
    print(f"   {backend_result['status']} - {backend_result['message']}")
    
    # AI Endpoint Check  
    print("\n2. 🤖 AI Endpoint Functionality")
    ai_result = check_ai_endpoint()
    print(f"   {ai_result['status']} - Success Rate: {ai_result['success_rate']}")
    for detail in ai_result['details']:
        print(f"      • {detail['query'][:30]}... → {detail['status']}")
    
    # Database Check
    print("\n3. 🗄️  Database Status")
    db_result = check_database()
    if db_result['status'] == "✅ CONNECTED":
        print(f"   {db_result['status']} - {db_result['places_total']} places, {db_result['museums_count']} museums")
    else:
        print(f"   {db_result['status']} - {db_result['message']}")
    
    # Frontend Check
    print("\n4. 🎨 Frontend Status") 
    frontend_result = check_frontend()
    print(f"   {frontend_result['status']} - {frontend_result['message']}")
    
    # Environment Check
    print("\n5. ⚙️  Environment Configuration")
    env_result = check_environment()
    if env_result['status'] == "✅ CONFIGURED":
        print(f"   {env_result['status']} - All required variables found")
    else:
        print(f"   {env_result['status']} - {env_result.get('message', 'Partial configuration')}")
    
    # Overall Assessment
    print("\n" + "=" * 50)
    print("🎯 OVERALL READINESS ASSESSMENT")
    print("=" * 50)
    
    critical_systems = [backend_result, ai_result, db_result]
    ready_systems = sum(1 for system in critical_systems if system['status'].startswith('✅'))
    
    if ready_systems == 3 and frontend_result['status'].startswith('✅'):
        print("🎉 SYSTEM FULLY READY FOR USERS!")
        print("✅ All critical components are working")
        print("✅ AI endpoints responding correctly")
        print("✅ Database populated with content")
        print("✅ Frontend files available")
        
        print("\n🚀 TO START FOR USERS:")
        print("1. Backend is already running on http://localhost:8000")
        print("2. Start frontend: cd frontend && npm run dev")
        print("3. Access at http://localhost:5173")
        
    elif ready_systems >= 2:
        print("⚠️  SYSTEM MOSTLY READY (Minor Issues)")
        print("✅ Core functionality is working")
        print("⚠️  Some components need attention")
        
        print("\n🔧 RECOMMENDED ACTIONS:")
        if not backend_result['status'].startswith('✅'):
            print("- Fix backend server issues")
        if not ai_result['status'].startswith('✅'):
            print("- Debug AI endpoint problems")
        if not db_result['status'].startswith('✅'):
            print("- Check database connectivity")
        if not frontend_result['status'].startswith('✅'):
            print("- Set up frontend dependencies (npm install)")
        
    else:
        print("❌ SYSTEM NOT READY")
        print("Multiple critical issues need to be resolved")
        
        print("\n🆘 CRITICAL FIXES NEEDED:")
        if not backend_result['status'].startswith('✅'):
            print("- Backend server must be started")
        if not ai_result['status'].startswith('✅'):
            print("- AI functionality is broken")
        if not db_result['status'].startswith('✅'):
            print("- Database issues must be resolved")

if __name__ == "__main__":
    run_readiness_check()

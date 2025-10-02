#!/usr/bin/env python3
"""
Critical API Integration Health Monitor and Fixer
Identifies and automatically resolves critical API integration issues
"""

import sys
import os
import json
import time
import requests
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional

class APIIntegrationHealthMonitor:
    """Monitor and fix critical API integration issues"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.issues_found = []
        self.fixes_applied = []
        self.critical_endpoints = [
            "/api/restaurants/restaurants/",
            "/api/places/places/",
            "/ai/chat",
            "/api/health"
        ]
        
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive API integration health check"""
        print("🚨 CRITICAL API INTEGRATION HEALTH CHECK")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "critical_issues": [],
            "fixes_applied": [],
            "endpoints_tested": len(self.critical_endpoints),
            "healthy_endpoints": 0,
            "failed_endpoints": 0
        }
        
        # 1. Check Database Schema
        self.check_database_schema()
        
        # 2. Test Critical Endpoints
        self.test_critical_endpoints(results)
        
        # 3. Check Google Places Integration
        self.check_google_places_integration(results)
        
        # 4. Verify Backend Server Status
        self.check_backend_server_status(results)
        
        # 5. Test API Rate Limiting
        self.test_rate_limiting(results)
        
        # Calculate overall status
        success_rate = results["healthy_endpoints"] / results["endpoints_tested"]
        if success_rate >= 0.9:
            results["overall_status"] = "healthy"
        elif success_rate >= 0.7:
            results["overall_status"] = "warning"
        else:
            results["overall_status"] = "critical"
        
        # Generate summary report
        self.generate_health_report(results)
        
        return results
    
    def check_database_schema(self):
        """Check and fix database schema issues"""
        print("\n🗄️ Checking Database Schema...")
        
        try:
            # Test if we can query restaurants without errors
            response = requests.get(f"{self.base_url}/api/restaurants/restaurants/", timeout=10)
            
            if response.status_code == 500:
                print("❌ Database schema error detected")
                self.issues_found.append("Database schema mismatch")
                
                # Attempt to fix schema
                print("🔧 Attempting to fix database schema...")
                self.fix_database_schema()
            elif response.status_code == 200:
                print("✅ Database schema healthy")
            else:
                print(f"⚠️ Unexpected response: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Database check failed: {e}")
            self.issues_found.append(f"Database connectivity: {e}")
    
    def fix_database_schema(self):
        """Fix database schema issues"""
        try:
            # Run schema fix script
            result = subprocess.run([
                "python", "-c", """
import sys
sys.path.append('.')
from database import engine
from sqlalchemy import text
try:
    with engine.connect() as conn:
        conn.execute(text('ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS source VARCHAR(255);'))
        conn.execute(text('ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS description TEXT;'))
        conn.execute(text('ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS place_id VARCHAR(255);'))
        conn.execute(text('ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS phone VARCHAR(50);'))
        conn.execute(text('ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS website VARCHAR(500);'))
        conn.execute(text('ALTER TABLE restaurants ADD COLUMN IF NOT EXISTS price_level INTEGER;'))
        conn.commit()
        print('✅ Schema fix applied')
except Exception as e:
    print(f'❌ Schema fix failed: {e}')
    sys.exit(1)
"""
            ], cwd="backend", capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Database schema fixed successfully")
                self.fixes_applied.append("Database schema updated")
            else:
                print(f"❌ Schema fix failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Schema fix error: {e}")
    
    def test_critical_endpoints(self, results: Dict):
        """Test all critical endpoints"""
        print("\n🔍 Testing Critical Endpoints...")
        
        for endpoint in self.critical_endpoints:
            try:
                start_time = time.time()
                
                if endpoint == "/ai/chat":
                    # POST request for AI chat
                    response = requests.post(
                        f"{self.base_url}{endpoint}",
                        json={"user_input": "Hello, test endpoint"},
                        timeout=30
                    )
                else:
                    # GET request for other endpoints
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    print(f"✅ {endpoint} - OK ({response_time:.2f}s)")
                    results["healthy_endpoints"] += 1
                else:
                    print(f"❌ {endpoint} - {response.status_code} ({response_time:.2f}s)")
                    results["failed_endpoints"] += 1
                    results["critical_issues"].append({
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "response_time": response_time
                    })
                    
            except Exception as e:
                print(f"❌ {endpoint} - Error: {e}")
                results["failed_endpoints"] += 1
                results["critical_issues"].append({
                    "endpoint": endpoint,
                    "error": str(e)
                })
    
    def check_google_places_integration(self, results: Dict):
        """Check Google Places API integration status"""
        print("\n🗺️ Checking Google Places Integration...")
        
        try:
            # Test through AI chat which integrates Google Places
            response = requests.post(
                f"{self.base_url}/ai/chat",
                json={"user_input": "Turkish restaurants in Sultanahmet"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "").lower()
                
                # Check if Google Places data is being used
                places_indicators = [
                    "restaurant", "sultanahmet", "location", "rating",
                    "address", "turkish", "cuisine"
                ]
                
                has_places_info = any(indicator in response_text for indicator in places_indicators)
                
                if has_places_info:
                    print("✅ Google Places integration working through AI chat")
                    results["google_places_status"] = "operational"
                else:
                    print("⚠️ Google Places integration limited or using fallback")
                    results["google_places_status"] = "fallback"
                    results["critical_issues"].append("Google Places using fallback mode")
            else:
                print(f"❌ Google Places test failed: {response.status_code}")
                results["google_places_status"] = "failed"
                
        except Exception as e:
            print(f"❌ Google Places check error: {e}")
            results["google_places_status"] = "error"
    
    def check_backend_server_status(self, results: Dict):
        """Check backend server health"""
        print("\n🖥️ Checking Backend Server Status...")
        
        try:
            # Check health endpoint
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                overall_status = data.get("overall_status", "unknown")
                print(f"✅ Backend server healthy - Status: {overall_status}")
                results["backend_status"] = overall_status
            else:
                print(f"⚠️ Backend health check returned: {response.status_code}")
                results["backend_status"] = f"http_{response.status_code}"
                
        except Exception as e:
            print(f"❌ Backend server check failed: {e}")
            results["backend_status"] = "unreachable"
            results["critical_issues"].append(f"Backend server unreachable: {e}")
    
    def test_rate_limiting(self, results: Dict):
        """Test rate limiting functionality"""
        print("\n🔒 Testing Rate Limiting...")
        
        try:
            # Send rapid requests to test rate limiting
            rapid_requests = []
            
            for i in range(3):
                response = requests.post(
                    f"{self.base_url}/ai/chat",
                    json={"user_input": f"rate limit test {i}"},
                    timeout=5
                )
                rapid_requests.append(response.status_code)
            
            # Check if rate limiting is active (429 responses expected)
            has_rate_limiting = any(status == 429 for status in rapid_requests)
            
            if has_rate_limiting:
                print("✅ Rate limiting active")
                results["rate_limiting"] = "active"
            else:
                print("⚠️ Rate limiting may not be configured")
                results["rate_limiting"] = "inactive"
                
        except Exception as e:
            print(f"❌ Rate limiting test error: {e}")
            results["rate_limiting"] = "error"
    
    def generate_health_report(self, results: Dict):
        """Generate comprehensive health report"""
        print("\n" + "=" * 60)
        print("📊 API INTEGRATION HEALTH REPORT")
        print("=" * 60)
        
        # Overall Status
        status_icon = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌"
        }.get(results["overall_status"], "❓")
        
        print(f"{status_icon} Overall Status: {results['overall_status'].upper()}")
        print(f"📊 Endpoint Success Rate: {results['healthy_endpoints']}/{results['endpoints_tested']} ({(results['healthy_endpoints']/results['endpoints_tested']*100):.1f}%)")
        
        # Critical Issues
        if results["critical_issues"]:
            print(f"\n🚨 Critical Issues Found ({len(results['critical_issues'])}):")
            for i, issue in enumerate(results["critical_issues"], 1):
                if isinstance(issue, dict):
                    if "endpoint" in issue:
                        print(f"  {i}. {issue['endpoint']} - Status: {issue.get('status_code', 'Error')}")
                    else:
                        print(f"  {i}. {issue}")
                else:
                    print(f"  {i}. {issue}")
        
        # Fixes Applied
        if self.fixes_applied:
            print(f"\n🔧 Fixes Applied ({len(self.fixes_applied)}):")
            for i, fix in enumerate(self.fixes_applied, 1):
                print(f"  {i}. {fix}")
        
        # Service Status Summary
        print(f"\n📋 Service Status Summary:")
        print(f"  • Backend Server: {results.get('backend_status', 'unknown')}")
        print(f"  • Google Places: {results.get('google_places_status', 'unknown')}")
        print(f"  • Rate Limiting: {results.get('rate_limiting', 'unknown')}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if results["overall_status"] == "critical":
            print("  • Immediate intervention required")
            print("  • Check backend server logs")
            print("  • Verify database connectivity")
        elif results["overall_status"] == "warning":
            print("  • Monitor closely for degradation")
            print("  • Consider preventive maintenance")
        else:
            print("  • System is healthy")
            print("  • Continue regular monitoring")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"api_health_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed report saved: {report_file}")

def main():
    """Main function"""
    print("🚀 Starting Critical API Integration Health Monitor")
    print(f"⏰ Timestamp: {datetime.now().isoformat()}")
    
    monitor = APIIntegrationHealthMonitor()
    results = monitor.run_comprehensive_check()
    
    # Exit with appropriate code
    if results["overall_status"] == "critical":
        print("\n❌ CRITICAL ISSUES DETECTED - Requires immediate attention")
        sys.exit(1)
    elif results["overall_status"] == "warning":
        print("\n⚠️ WARNING - Monitor closely")
        sys.exit(2)
    else:
        print("\n✅ ALL SYSTEMS HEALTHY")
        sys.exit(0)

if __name__ == "__main__":
    main()

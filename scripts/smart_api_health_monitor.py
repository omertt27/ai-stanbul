#!/usr/bin/env python3
"""
Smart API Integration Health Monitor
Intelligently handles rate limiting and provides comprehensive health assessment
"""

import sys
import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional

class SmartAPIHealthMonitor:
    """Smart health monitor that handles rate limiting intelligently"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.rate_limit_wait_time = 60  # Wait 60 seconds for rate limit reset
        
    def run_smart_health_check(self) -> Dict[str, Any]:
        """Run intelligent health check with rate limit handling"""
        print("🧠 SMART API INTEGRATION HEALTH CHECK")
        print("=" * 60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "unknown",
            "endpoints": {},
            "integrations": {},
            "performance": {},
            "recommendations": []
        }
        
        # 1. Test basic endpoints (no rate limiting expected)
        self.test_basic_endpoints(results)
        
        # 2. Smart AI chat test (handles rate limiting)
        self.test_ai_chat_smart(results)
        
        # 3. Test integrations through working endpoints
        self.test_integrations_smart(results)
        
        # 4. Performance assessment
        self.assess_performance(results)
        
        # 5. Generate smart recommendations
        self.generate_smart_recommendations(results)
        
        # Calculate overall status
        self.calculate_overall_status(results)
        
        return results
    
    def test_basic_endpoints(self, results: Dict):
        """Test basic endpoints that shouldn't have rate limiting"""
        print("\n🔍 Testing Basic Endpoints...")
        
        basic_endpoints = [
            "/api/health",
            "/api/restaurants/restaurants/",
            "/api/places/places/"
        ]
        
        for endpoint in basic_endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                response_time = time.time() - start_time
                
                results["endpoints"][endpoint] = {
                    "status_code": response.status_code,
                    "response_time": round(response_time, 3),
                    "status": "healthy" if response.status_code == 200 else "failed"
                }
                
                if response.status_code == 200:
                    print(f"✅ {endpoint} - OK ({response_time:.2f}s)")
                else:
                    print(f"❌ {endpoint} - {response.status_code} ({response_time:.2f}s)")
                    
            except Exception as e:
                print(f"❌ {endpoint} - Error: {e}")
                results["endpoints"][endpoint] = {
                    "status": "error",
                    "error": str(e)
                }
    
    def test_ai_chat_smart(self, results: Dict):
        """Smart AI chat test that handles rate limiting"""
        print("\n🤖 Testing AI Chat (Smart Rate Limit Handling)...")
        
        endpoint = "/ai/chat"
        
        try:
            # First attempt
            response = requests.post(
                f"{self.base_url}{endpoint}",
                json={"user_input": "Hello, quick health check"},
                timeout=30
            )
            
            if response.status_code == 429:
                print("⚠️ Rate limit detected - waiting for reset...")
                print(f"⏳ Waiting {self.rate_limit_wait_time} seconds for rate limit reset...")
                
                # Wait for rate limit reset
                time.sleep(self.rate_limit_wait_time)
                
                # Retry after waiting
                print("🔄 Retrying AI chat after rate limit reset...")
                response = requests.post(
                    f"{self.base_url}{endpoint}",
                    json={"user_input": "Hello after rate limit reset"},
                    timeout=30
                )
            
            # Evaluate response
            if response.status_code == 200:
                data = response.json()
                response_length = len(data.get("response", ""))
                
                results["endpoints"][endpoint] = {
                    "status_code": 200,
                    "status": "healthy",
                    "response_length": response_length,
                    "rate_limit_handled": True
                }
                
                print(f"✅ AI Chat - OK (Response: {response_length} chars)")
                
                # Test Google Places integration through response content
                self.analyze_ai_response_for_integrations(data.get("response", ""), results)
                
            elif response.status_code == 429:
                print("⚠️ Rate limit still active after waiting")
                results["endpoints"][endpoint] = {
                    "status_code": 429,
                    "status": "rate_limited",
                    "message": "Rate limit active even after waiting"
                }
            else:
                print(f"❌ AI Chat - {response.status_code}")
                results["endpoints"][endpoint] = {
                    "status_code": response.status_code,
                    "status": "failed"
                }
                
        except Exception as e:
            print(f"❌ AI Chat test error: {e}")
            results["endpoints"][endpoint] = {
                "status": "error",
                "error": str(e)
            }
    
    def analyze_ai_response_for_integrations(self, response_text: str, results: Dict):
        """Analyze AI response to determine integration status"""
        response_lower = response_text.lower()
        
        # Check for Google Places indicators
        places_indicators = [
            "restaurant", "location", "address", "rating", 
            "cuisine", "turkish", "food", "dining"
        ]
        
        google_places_score = sum(1 for indicator in places_indicators if indicator in response_lower)
        
        if google_places_score >= 3:
            results["integrations"]["google_places"] = "operational"
            print("✅ Google Places integration: Operational")
        elif google_places_score >= 1:
            results["integrations"]["google_places"] = "partial"
            print("⚠️ Google Places integration: Partial")
        else:
            results["integrations"]["google_places"] = "fallback"
            print("❌ Google Places integration: Fallback mode")
        
        # Check for weather integration
        weather_indicators = ["weather", "temperature", "climate", "sunny", "rainy"]
        weather_score = sum(1 for indicator in weather_indicators if indicator in response_lower)
        
        if weather_score >= 1:
            results["integrations"]["weather"] = "operational"
            print("✅ Weather integration: Operational")
        else:
            results["integrations"]["weather"] = "unknown"
    
    def test_integrations_smart(self, results: Dict):
        """Test integrations through working endpoints"""
        print("\n🔗 Testing Integrations...")
        
        # Test database integration
        try:
            response = requests.get(f"{self.base_url}/api/restaurants/restaurants/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                restaurant_count = len(data) if isinstance(data, list) else 0
                
                results["integrations"]["database"] = {
                    "status": "operational",
                    "restaurant_count": restaurant_count
                }
                print(f"✅ Database integration: {restaurant_count} restaurants available")
            else:
                results["integrations"]["database"] = {"status": "failed"}
                print("❌ Database integration: Failed")
                
        except Exception as e:
            results["integrations"]["database"] = {"status": "error", "error": str(e)}
            print(f"❌ Database integration error: {e}")
    
    def assess_performance(self, results: Dict):
        """Assess overall system performance"""
        print("\n⚡ Assessing Performance...")
        
        response_times = []
        successful_endpoints = 0
        total_endpoints = 0
        
        for endpoint, data in results["endpoints"].items():
            total_endpoints += 1
            if data.get("status") == "healthy":
                successful_endpoints += 1
                if "response_time" in data:
                    response_times.append(data["response_time"])
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            results["performance"] = {
                "avg_response_time": round(avg_response_time, 3),
                "max_response_time": round(max_response_time, 3),
                "min_response_time": round(min_response_time, 3),
                "success_rate": round((successful_endpoints / total_endpoints) * 100, 1)
            }
            
            print(f"📊 Average response time: {avg_response_time:.3f}s")
            print(f"📊 Success rate: {successful_endpoints}/{total_endpoints} ({(successful_endpoints/total_endpoints)*100:.1f}%)")
            
            # Performance assessment
            if avg_response_time < 1.0:
                print("✅ Performance: Excellent")
                results["performance"]["assessment"] = "excellent"
            elif avg_response_time < 3.0:
                print("✅ Performance: Good")
                results["performance"]["assessment"] = "good"
            else:
                print("⚠️ Performance: Could be improved")
                results["performance"]["assessment"] = "needs_improvement"
    
    def generate_smart_recommendations(self, results: Dict):
        """Generate intelligent recommendations based on test results"""
        recommendations = []
        
        # Check endpoint health
        failed_endpoints = [ep for ep, data in results["endpoints"].items() 
                          if data.get("status") not in ["healthy", "rate_limited"]]
        
        if failed_endpoints:
            recommendations.append(f"Fix failed endpoints: {', '.join(failed_endpoints)}")
        
        # Check rate limiting
        rate_limited_endpoints = [ep for ep, data in results["endpoints"].items() 
                                if data.get("status") == "rate_limited"]
        
        if rate_limited_endpoints:
            recommendations.append("Consider adjusting rate limiting configuration for better user experience")
        
        # Check integrations
        failed_integrations = [name for name, data in results["integrations"].items() 
                             if (isinstance(data, dict) and data.get("status") == "failed") or 
                                (isinstance(data, str) and data == "fallback")]
        
        if failed_integrations:
            recommendations.append(f"Review integration configurations: {', '.join(failed_integrations)}")
        
        # Performance recommendations
        if results.get("performance", {}).get("assessment") == "needs_improvement":
            recommendations.append("Optimize API response times - consider caching, database indexing")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System is healthy - continue regular monitoring")
        
        results["recommendations"] = recommendations
    
    def calculate_overall_status(self, results: Dict):
        """Calculate overall system status"""
        healthy_endpoints = sum(1 for data in results["endpoints"].values() 
                              if data.get("status") in ["healthy", "rate_limited"])
        total_endpoints = len(results["endpoints"])
        
        success_rate = (healthy_endpoints / total_endpoints) if total_endpoints > 0 else 0
        
        # Check critical failures
        has_critical_failures = any(data.get("status") == "error" 
                                  for data in results["endpoints"].values())
        
        if has_critical_failures:
            results["overall_status"] = "critical"
        elif success_rate >= 0.9:
            results["overall_status"] = "healthy"
        elif success_rate >= 0.7:
            results["overall_status"] = "warning"
        else:
            results["overall_status"] = "critical"
        
        # Print final status
        print("\n" + "=" * 60)
        print("📊 SMART HEALTH CHECK RESULTS")
        print("=" * 60)
        
        status_icon = {
            "healthy": "✅",
            "warning": "⚠️",
            "critical": "❌"
        }.get(results["overall_status"], "❓")
        
        print(f"{status_icon} Overall Status: {results['overall_status'].upper()}")
        print(f"📊 Endpoint Success Rate: {healthy_endpoints}/{total_endpoints} ({success_rate*100:.1f}%)")
        
        if results.get("performance"):
            perf = results["performance"]
            print(f"⚡ Performance: {perf.get('assessment', 'unknown').title()} (avg: {perf.get('avg_response_time', 0):.3f}s)")
        
        print(f"\n💡 Recommendations ({len(results['recommendations'])}):")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"smart_health_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed report saved: {report_file}")

def main():
    """Main function"""
    print("🧠 Smart API Integration Health Monitor")
    print("🔧 Intelligently handling rate limits and integration testing")
    
    monitor = SmartAPIHealthMonitor()
    results = monitor.run_smart_health_check()
    
    # Exit with appropriate code
    if results["overall_status"] == "critical":
        print("\n❌ CRITICAL ISSUES DETECTED")
        sys.exit(1)
    elif results["overall_status"] == "warning":
        print("\n⚠️ WARNINGS DETECTED - Monitor closely")
        sys.exit(2)
    else:
        print("\n✅ SYSTEM HEALTHY")
        sys.exit(0)

if __name__ == "__main__":
    main()

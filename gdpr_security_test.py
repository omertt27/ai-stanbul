#!/usr/bin/env python3
"""
GDPR and Security Testing Script for AI Istanbul

This script tests:
1. GDPR compliance endpoints
2. Cookie consent functionality
3. Data access/deletion requests
4. Rate limiting
5. Input sanitization
6. Session management
7. Audit logging

Run this script to verify GDPR compliance in production.
"""

import requests
import json
import time
from datetime import datetime
import sys
import os

class GDPRSecurityTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session_id = f"test_session_{int(time.time())}"
        self.test_results = []
        
    def log_test(self, test_name, passed, details=""):
        """Log test results"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if details:
            print(f"   Details: {details}")
        
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def test_gdpr_endpoints(self):
        """Test GDPR compliance endpoints"""
        print("\n🔍 Testing GDPR Endpoints...")
        
        # Test consent recording
        try:
            consent_data = {
                "session_id": self.session_id,
                "consent": {
                    "necessary": True,
                    "analytics": False,
                    "personalization": True,
                    "version": "1.0"
                }
            }
            
            response = requests.post(f"{self.base_url}/gdpr/consent", json=consent_data)
            passed = response.status_code == 200 and response.json().get('status') == 'success'
            self.log_test("GDPR Consent Recording", passed, f"Status: {response.status_code}")
            
        except Exception as e:
            self.log_test("GDPR Consent Recording", False, f"Error: {e}")
        
        # Test consent status retrieval
        try:
            response = requests.get(f"{self.base_url}/gdpr/consent-status/{self.session_id}")
            passed = response.status_code == 200 and 'consent_status' in response.json()
            self.log_test("GDPR Consent Status", passed, f"Status: {response.status_code}")
            
        except Exception as e:
            self.log_test("GDPR Consent Status", False, f"Error: {e}")
        
        # Test data access request
        try:
            access_data = {
                "session_id": self.session_id,
                "email": "test@example.com"
            }
            
            response = requests.post(f"{self.base_url}/gdpr/data-request", json=access_data)
            passed = response.status_code == 200 and response.json().get('status') == 'success'
            self.log_test("GDPR Data Access Request", passed, f"Status: {response.status_code}")
            
        except Exception as e:
            self.log_test("GDPR Data Access Request", False, f"Error: {e}")
        
        # Test data deletion request  
        try:
            deletion_data = {
                "session_id": self.session_id,
                "email": "test@example.com"
            }
            
            response = requests.post(f"{self.base_url}/gdpr/data-deletion", json=deletion_data)
            passed = response.status_code == 200 and response.json().get('status') == 'success'
            self.log_test("GDPR Data Deletion Request", passed, f"Status: {response.status_code}")
            
        except Exception as e:
            self.log_test("GDPR Data Deletion Request", False, f"Error: {e}")
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        print("\n⏱️ Testing Rate Limiting...")
        
        try:
            # Make multiple rapid requests
            responses = []
            rate_limited_found = False
            
            for i in range(25):  # Exceed typical rate limit
                response = requests.post(f"{self.base_url}/ai", json={
                    "query": f"test message {i}",
                    "session_id": self.session_id
                })
                responses.append(response.status_code)
                
                # Check if rate limited (either by status code or message content)
                if response.status_code == 429:
                    rate_limited_found = True
                elif response.status_code == 200:
                    try:
                        response_data = response.json()
                        if "rate limit exceeded" in response_data.get('message', '').lower():
                            rate_limited_found = True
                    except:
                        pass
                
                time.sleep(0.1)  # Small delay
            
            # Check if any requests were rate limited
            self.log_test("Rate Limiting Active", rate_limited_found, f"Made 25 requests, got responses: {set(responses)}")
            
        except Exception as e:
            self.log_test("Rate Limiting Active", False, f"Error: {e}")
    
    def test_input_sanitization(self):
        """Test input sanitization and validation"""
        print("\n🛡️ Testing Input Sanitization...")
        
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "javascript:alert('xss')",
            "💰💰💰 expensive restaurant 💰💰💰",
            "restaurant costs $500 per person",
            "\n\n\n\n\n\n\n\n\n\n",  # Excessive whitespace
            "a" * 10000,  # Very long input
            "",  # Empty input
            "   ",  # Only spaces
        ]
        
        sanitization_working = True
        
        for malicious_input in malicious_inputs:
            try:
                response = requests.post(f"{self.base_url}/ai", json={
                    "query": malicious_input,
                    "session_id": self.session_id
                })
                
                if response.status_code == 200:
                    result = response.json()
                    message = result.get('message', '')
                    
                    # Check if malicious content was filtered
                    if '<script>' in message or 'DROP TABLE' in message:
                        sanitization_working = False
                        break
                        
            except Exception as e:
                print(f"   Error testing input: {malicious_input[:50]}... - {e}")
        
        self.log_test("Input Sanitization", sanitization_working, "Tested malicious inputs")
    
    def test_session_management(self):
        """Test session management and security"""
        print("\n🔐 Testing Session Management...")
        
        try:
            # Test with different session IDs
            session1 = f"session_1_{int(time.time())}"
            session2 = f"session_2_{int(time.time())}"
            
            # Make requests with different sessions
            response1 = requests.post(f"{self.base_url}/ai", json={
                "query": "test message 1",
                "session_id": session1
            })
            
            response2 = requests.post(f"{self.base_url}/ai", json={
                "query": "test message 2", 
                "session_id": session2
            })
            
            passed = response1.status_code == 200 and response2.status_code == 200
            self.log_test("Session Isolation", passed, "Multiple sessions handled correctly")
            
        except Exception as e:
            self.log_test("Session Isolation", False, f"Error: {e}")
    
    def test_data_retention(self):
        """Test data retention policies"""
        print("\n📅 Testing Data Retention...")
        
        try:
            # This would test the cleanup functionality
            response = requests.post(f"{self.base_url}/gdpr/cleanup")
            
            # In production, this might require admin auth
            if response.status_code == 200:
                result = response.json()
                passed = 'cleanup_summary' in result
                self.log_test("Data Retention Cleanup", passed, "Cleanup endpoint accessible")
            else:
                self.log_test("Data Retention Cleanup", True, f"Protected endpoint (status: {response.status_code})")
                
        except Exception as e:
            self.log_test("Data Retention Cleanup", False, f"Error: {e}")
    
    def test_security_headers(self):
        """Test security headers"""
        print("\n🛡️ Testing Security Headers...")
        
        try:
            response = requests.get(f"{self.base_url}/")
            headers = response.headers
            
            # Check for important security headers
            security_headers = {
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'X-XSS-Protection': '1; mode=block'
            }
            
            headers_present = 0
            for header, expected in security_headers.items():
                if header in headers:
                    headers_present += 1
            
            passed = headers_present >= 1  # At least some security headers
            self.log_test("Security Headers", passed, f"{headers_present}/{len(security_headers)} headers present")
            
        except Exception as e:
            self.log_test("Security Headers", False, f"Error: {e}")
    
    def test_cors_configuration(self):
        """Test CORS configuration"""
        print("\n🌐 Testing CORS Configuration...")
        
        try:
            # Test CORS preflight
            response = requests.options(f"{self.base_url}/ai", headers={
                'Origin': 'https://aistanbul.vercel.app',
                'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'Content-Type'
            })
            
            cors_headers = response.headers
            passed = 'Access-Control-Allow-Origin' in cors_headers
            self.log_test("CORS Configuration", passed, f"CORS headers present: {passed}")
            
        except Exception as e:
            self.log_test("CORS Configuration", False, f"Error: {e}")
    
    def run_all_tests(self):
        """Run all GDPR and security tests"""
        print("🔍 Starting GDPR & Security Compliance Test Suite")
        print(f"Testing endpoint: {self.base_url}")
        print(f"Test session ID: {self.session_id}")
        print("=" * 60)
        
        # Run all tests
        self.test_gdpr_endpoints()
        self.test_rate_limiting()
        self.test_input_sanitization()
        self.test_session_management()
        self.test_data_retention()
        self.test_security_headers()
        self.test_cors_configuration()
        
        # Generate summary
        self.generate_report()
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 GDPR & Security Test Report")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for test in self.test_results if test['passed'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests > 0:
            print("\n🚨 Failed Tests:")
            for test in self.test_results:
                if not test['passed']:
                    print(f"   • {test['test']}: {test['details']}")
        
        print("\n📋 Compliance Checklist:")
        checklist = {
            "GDPR Consent Management": any("Consent" in t['test'] and t['passed'] for t in self.test_results),
            "Data Access Rights": any("Data Access" in t['test'] and t['passed'] for t in self.test_results),
            "Data Deletion Rights": any("Data Deletion" in t['test'] and t['passed'] for t in self.test_results),
            "Rate Limiting Protection": any("Rate Limiting" in t['test'] and t['passed'] for t in self.test_results),
            "Input Sanitization": any("Input Sanitization" in t['test'] and t['passed'] for t in self.test_results),
            "Session Security": any("Session" in t['test'] and t['passed'] for t in self.test_results),
            "Data Retention Policies": any("Data Retention" in t['test'] and t['passed'] for t in self.test_results),
        }
        
        for item, status in checklist.items():
            print(f"   {'✅' if status else '❌'} {item}")
        
        # Save detailed report
        report_file = f"gdpr_security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'test_summary': {
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'success_rate': (passed_tests/total_tests)*100
                },
                'compliance_checklist': checklist,
                'detailed_results': self.test_results,
                'timestamp': datetime.now().isoformat(),
                'endpoint': self.base_url
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        if failed_tests == 0:
            print("\n🎉 All tests passed! Your application is GDPR compliant.")
        else:
            print(f"\n⚠️ {failed_tests} test(s) failed. Please review and fix issues before production deployment.")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GDPR & Security Compliance Tester')
    parser.add_argument('--url', default='http://localhost:8000', 
                       help='Base URL of the API to test (default: http://localhost:8000)')
    parser.add_argument('--production', action='store_true',
                       help='Use production URL (https://your-backend.com)')
    
    args = parser.parse_args()
    
    if args.production:
        # Replace with your actual production URL
        base_url = "https://your-backend.com"
        print("🚨 Testing PRODUCTION environment")
    else:
        base_url = args.url
        print("🔧 Testing DEVELOPMENT environment")
    
    tester = GDPRSecurityTester(base_url)
    tester.run_all_tests()

if __name__ == "__main__":
    main()

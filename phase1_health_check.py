#!/usr/bin/env python3
"""
Phase 1: Production Health Check Script
Tests all critical endpoints and verifies system readiness
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Tuple
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "https://api.aistanbul.net")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://aistanbul.net")
LLM_API_URL = os.getenv("LLM_API_URL", "")

# Test languages
TEST_LANGUAGES = ["en", "tr", "ar", "de", "fr", "es"]

# Test queries per language
TEST_QUERIES = {
    "en": "What are the best restaurants in Sultanahmet?",
    "tr": "Sultanahmet'te en iyi restoranlar nelerdir?",
    "ar": "ما هي أفضل المطاعم في سلطان أحمد؟",
    "de": "Was sind die besten Restaurants in Sultanahmet?",
    "fr": "Quels sont les meilleurs restaurants à Sultanahmet?",
    "es": "¿Cuáles son los mejores restaurantes en Sultanahmet?"
}

class HealthChecker:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "details": []
        }
        self.start_time = time.time()

    def print_header(self, text: str):
        """Print section header"""
        print(f"\n{Fore.CYAN}{'='*80}")
        print(f"{Fore.CYAN}{text:^80}")
        print(f"{Fore.CYAN}{'='*80}\n")

    def print_success(self, text: str):
        """Print success message"""
        print(f"{Fore.GREEN}✅ {text}")

    def print_error(self, text: str):
        """Print error message"""
        print(f"{Fore.RED}❌ {text}")

    def print_warning(self, text: str):
        """Print warning message"""
        print(f"{Fore.YELLOW}⚠️  {text}")

    def print_info(self, text: str):
        """Print info message"""
        print(f"{Fore.BLUE}ℹ️  {text}")

    def record_result(self, test_name: str, passed: bool, message: str, details: Dict = None):
        """Record test result"""
        if passed:
            self.results["tests_passed"] += 1
            self.print_success(f"{test_name}: {message}")
        else:
            self.results["tests_failed"] += 1
            self.print_error(f"{test_name}: {message}")
        
        self.results["details"].append({
            "test": test_name,
            "passed": passed,
            "message": message,
            "details": details or {}
        })

    def test_backend_health(self) -> Tuple[bool, str, Dict]:
        """Test backend health endpoint"""
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return True, "Backend is healthy", data
            else:
                return False, f"Backend returned status {response.status_code}", {}
        except Exception as e:
            return False, f"Backend connection failed: {str(e)}", {}

    def test_llm_health(self) -> Tuple[bool, str, Dict]:
        """Test LLM API health"""
        if not LLM_API_URL:
            return False, "LLM_API_URL not configured", {}
        
        try:
            # Test health endpoint
            response = requests.get(f"{LLM_API_URL.rstrip('/v1')}/health", timeout=10)
            if response.status_code == 200:
                return True, "LLM server is healthy", response.json()
            else:
                return False, f"LLM server returned status {response.status_code}", {}
        except Exception as e:
            return False, f"LLM server connection failed: {str(e)}", {}

    def test_chat_endpoint(self, language: str, query: str) -> Tuple[bool, str, Dict]:
        """Test chat endpoint with specific language"""
        try:
            payload = {
                "message": query,
                "language": language,
                "session_id": f"health_check_{language}_{int(time.time())}"
            }
            
            response = requests.post(
                f"{BACKEND_URL}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                response_text = data.get("response", "")
                
                # Check if response is meaningful (not empty, not error)
                if response_text and len(response_text) > 10:
                    return True, f"Chat works in {language}", {
                        "language": language,
                        "response_length": len(response_text),
                        "response_preview": response_text[:100]
                    }
                else:
                    return False, f"Empty or invalid response for {language}", data
            else:
                return False, f"Chat endpoint returned status {response.status_code}", {}
        except Exception as e:
            return False, f"Chat endpoint failed for {language}: {str(e)}", {}

    def test_frontend_loads(self) -> Tuple[bool, str, Dict]:
        """Test if frontend loads"""
        try:
            response = requests.get(FRONTEND_URL, timeout=10)
            if response.status_code == 200:
                return True, "Frontend loads successfully", {
                    "status_code": response.status_code,
                    "size": len(response.content)
                }
            else:
                return False, f"Frontend returned status {response.status_code}", {}
        except Exception as e:
            return False, f"Frontend connection failed: {str(e)}", {}

    def test_cors(self) -> Tuple[bool, str, Dict]:
        """Test CORS configuration"""
        try:
            headers = {
                "Origin": FRONTEND_URL,
                "Access-Control-Request-Method": "POST"
            }
            response = requests.options(f"{BACKEND_URL}/api/chat", headers=headers, timeout=10)
            
            cors_header = response.headers.get("Access-Control-Allow-Origin", "")
            if cors_header in ["*", FRONTEND_URL]:
                return True, "CORS configured correctly", {
                    "allowed_origin": cors_header
                }
            else:
                return False, "CORS not configured correctly", {
                    "allowed_origin": cors_header,
                    "expected": FRONTEND_URL
                }
        except Exception as e:
            return False, f"CORS check failed: {str(e)}", {}

    def run_all_tests(self):
        """Run all health checks"""
        self.print_header("🏥 PHASE 1 HEALTH CHECK - PRODUCTION READINESS")
        
        print(f"{Fore.BLUE}Configuration:")
        print(f"  Backend URL: {BACKEND_URL}")
        print(f"  Frontend URL: {FRONTEND_URL}")
        print(f"  LLM API URL: {LLM_API_URL or '❌ NOT SET'}")
        print()

        # Test 1: Backend Health
        self.print_header("1️⃣ Backend Health Check")
        passed, message, details = self.test_backend_health()
        self.record_result("Backend Health", passed, message, details)
        
        if passed:
            self.print_info(f"   Version: {details.get('version', 'unknown')}")
            self.print_info(f"   Environment: {details.get('environment', 'unknown')}")

        # Test 2: LLM Server Health
        self.print_header("2️⃣ LLM Server Health Check")
        passed, message, details = self.test_llm_health()
        self.record_result("LLM Health", passed, message, details)

        # Test 3: Frontend Loading
        self.print_header("3️⃣ Frontend Loading Check")
        passed, message, details = self.test_frontend_loads()
        self.record_result("Frontend Loading", passed, message, details)

        # Test 4: CORS Configuration
        self.print_header("4️⃣ CORS Configuration Check")
        passed, message, details = self.test_cors()
        self.record_result("CORS Configuration", passed, message, details)

        # Test 5: Multi-Language Chat Tests
        self.print_header("5️⃣ Multi-Language Chat Tests")
        for lang in TEST_LANGUAGES:
            query = TEST_QUERIES[lang]
            self.print_info(f"Testing {lang.upper()}: {query[:50]}...")
            passed, message, details = self.test_chat_endpoint(lang, query)
            self.record_result(f"Chat ({lang})", passed, message, details)
            
            if passed:
                preview = details.get("response_preview", "")
                self.print_info(f"   Response: {preview}...")
            
            # Rate limit protection
            time.sleep(1)

        # Summary
        self.print_header("📊 SUMMARY")
        total_tests = self.results["tests_passed"] + self.results["tests_failed"]
        success_rate = (self.results["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
        elapsed_time = time.time() - self.start_time

        print(f"{Fore.CYAN}Total Tests: {total_tests}")
        print(f"{Fore.GREEN}Passed: {self.results['tests_passed']}")
        print(f"{Fore.RED}Failed: {self.results['tests_failed']}")
        print(f"{Fore.YELLOW}Success Rate: {success_rate:.1f}%")
        print(f"{Fore.BLUE}Elapsed Time: {elapsed_time:.2f}s")
        print()

        # Status
        if self.results["tests_failed"] == 0:
            self.print_success("🎉 ALL TESTS PASSED! System is production ready!")
            return_code = 0
        elif success_rate >= 80:
            self.print_warning("⚠️  MOST TESTS PASSED. Review failed tests and fix issues.")
            return_code = 1
        else:
            self.print_error("❌ MULTIPLE TESTS FAILED. System is NOT production ready!")
            return_code = 2

        # Save results
        report_file = f"health_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.print_info(f"Detailed report saved to: {report_file}")
        
        return return_code

if __name__ == "__main__":
    # Check if required env vars are set
    if not BACKEND_URL or not FRONTEND_URL:
        print(f"{Fore.RED}❌ ERROR: Required environment variables not set!")
        print(f"{Fore.YELLOW}Please set:")
        print(f"  export BACKEND_URL=https://api.aistanbul.net")
        print(f"  export FRONTEND_URL=https://aistanbul.net")
        print(f"  export LLM_API_URL=https://your-runpod-url/v1")
        sys.exit(1)

    checker = HealthChecker()
    exit_code = checker.run_all_tests()
    sys.exit(exit_code)

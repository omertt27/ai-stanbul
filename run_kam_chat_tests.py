#!/usr/bin/env python3
"""
KAM AI Chat Comprehensive Test Runner
Tests 40 different scenarios for restaurants and attractions
Analyzes responses for accuracy, features, and performance
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Any
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"  # FastAPI default port
TEST_FILE = "test_kam_chat_comprehensive.json"
RESULTS_FILE = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_test_cases() -> Dict:
    """Load test cases from JSON file"""
    try:
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{Colors.FAIL}Error: Test file '{TEST_FILE}' not found{Colors.ENDC}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{Colors.FAIL}Error: Invalid JSON in test file: {e}{Colors.ENDC}")
        sys.exit(1)

def send_chat_request(user_input: str) -> Dict[str, Any]:
    """Send a chat request to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/chat",
            json={"message": user_input, "user_id": "test_user"},
            timeout=30
        )
        response.raise_for_status()
        return {
            "success": True,
            "response": response.json(),
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds()
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout",
            "response_time": 30.0
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "response_time": 0
        }

def analyze_response(test_case: Dict, api_response: Dict) -> Dict[str, Any]:
    """Analyze the API response against expected features"""
    analysis = {
        "test_id": test_case["id"],
        "category": test_case["category"],
        "subcategory": test_case["subcategory"],
        "input": test_case["input"],
        "success": api_response["success"],
        "response_time": api_response.get("response_time", 0),
        "features_detected": [],
        "features_missing": [],
        "issues": [],
        "score": 0,
        "response_preview": ""
    }
    
    if not api_response["success"]:
        analysis["issues"].append(f"API Error: {api_response.get('error', 'Unknown error')}")
        return analysis
    
    # Extract response text
    response_data = api_response.get("response", {})
    response_text = response_data.get("response", "").lower()
    analysis["response_preview"] = response_text[:200] + "..." if len(response_text) > 200 else response_text
    
    # Check for expected features
    expected_features = test_case.get("expected_features", [])
    
    # Feature detection patterns
    feature_patterns = {
        "location_filtering": ["beyoğlu", "sultanahmet", "kadıköy", "taksim", "fatih", "beşiktaş"],
        "restaurant_recommendations": ["restaurant", "dining", "eat", "food"],
        "4_results": ["1.", "2.", "3.", "4."],
        "cuisine_filtering": ["turkish", "seafood", "vegetarian", "vegan", "cuisine"],
        "dietary_filtering": ["vegetarian", "vegan", "halal", "kosher", "gluten-free"],
        "price_filtering": ["💰", "price", "budget", "affordable", "expensive"],
        "typo_correction": ["restaurant", "attraction", "istanbul"],  # Should work despite typos
        "operating_hours": ["open", "closed", "hours"],
        "category_filtering": ["museum", "monument", "park", "mosque"],
        "weather_appropriate": ["indoor", "outdoor", "rainy", "sunny"],
        "family_activities": ["family", "kids", "children"],
        "romantic_spots": ["romantic", "couples", "scenic"],
        "free_entry": ["free", "no admission", "no fee"],
        "successful_search": ["here are", "i found", "recommendations", "attractions"]
    }
    
    # Check each expected feature
    for feature in expected_features:
        patterns = feature_patterns.get(feature, [])
        detected = any(pattern in response_text for pattern in patterns)
        
        if detected:
            analysis["features_detected"].append(feature)
        else:
            analysis["features_missing"].append(feature)
    
    # Check response quality
    if len(response_text) < 50:
        analysis["issues"].append("Response too short")
    
    if "error" in response_text or "sorry" in response_text[:100]:
        analysis["issues"].append("Error message detected")
    
    if test_case["category"] == "restaurants":
        if "restaurant" not in response_text:
            analysis["issues"].append("No restaurant mention in response")
        # Check for 4 results
        result_count = sum(1 for i in range(1, 5) if f"{i}." in response_text)
        if result_count < 4:
            analysis["issues"].append(f"Expected 4 results, found {result_count}")
    
    if test_case["category"] == "places_attractions":
        if "attraction" not in response_text and "place" not in response_text and "site" not in response_text:
            analysis["issues"].append("No attractions mentioned in response")
    
    # Calculate score
    total_features = len(expected_features)
    detected_features = len(analysis["features_detected"])
    
    if total_features > 0:
        feature_score = (detected_features / total_features) * 70
    else:
        feature_score = 70
    
    # Response quality score (30 points)
    quality_score = 30
    if analysis["issues"]:
        quality_score -= len(analysis["issues"]) * 5
    if quality_score < 0:
        quality_score = 0
    
    # Response time bonus/penalty
    time_bonus = 0
    if api_response["response_time"] < 2.0:
        time_bonus = 5
    elif api_response["response_time"] > 10.0:
        time_bonus = -5
    
    analysis["score"] = min(100, feature_score + quality_score + time_bonus)
    
    return analysis

def print_test_result(analysis: Dict, test_num: int, total_tests: int):
    """Print a single test result"""
    status_color = Colors.OKGREEN if analysis["score"] >= 70 else Colors.WARNING if analysis["score"] >= 50 else Colors.FAIL
    
    print(f"\n{Colors.BOLD}[{test_num}/{total_tests}] Test #{analysis['test_id']}{Colors.ENDC}")
    print(f"Category: {Colors.OKCYAN}{analysis['category']} → {analysis['subcategory']}{Colors.ENDC}")
    print(f"Input: {Colors.OKBLUE}\"{analysis['input']}\"{Colors.ENDC}")
    print(f"Score: {status_color}{analysis['score']:.1f}/100{Colors.ENDC}")
    print(f"Response Time: {analysis['response_time']:.2f}s")
    
    if analysis["features_detected"]:
        print(f"{Colors.OKGREEN}✓ Features Detected: {', '.join(analysis['features_detected'])}{Colors.ENDC}")
    
    if analysis["features_missing"]:
        print(f"{Colors.WARNING}✗ Features Missing: {', '.join(analysis['features_missing'])}{Colors.ENDC}")
    
    if analysis["issues"]:
        print(f"{Colors.FAIL}⚠ Issues: {', '.join(analysis['issues'])}{Colors.ENDC}")
    
    if not analysis["success"]:
        print(f"{Colors.FAIL}✗ Test FAILED{Colors.ENDC}")

def print_summary(results: List[Dict]):
    """Print overall test summary"""
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    average_score = sum(r["score"] for r in results) / total_tests if total_tests > 0 else 0
    average_time = sum(r["response_time"] for r in results) / total_tests if total_tests > 0 else 0
    
    # Category breakdown
    restaurant_tests = [r for r in results if r["category"] == "restaurants"]
    attraction_tests = [r for r in results if r["category"] == "places_attractions"]
    
    restaurant_avg = sum(r["score"] for r in restaurant_tests) / len(restaurant_tests) if restaurant_tests else 0
    attraction_avg = sum(r["score"] for r in attraction_tests) / len(attraction_tests) if attraction_tests else 0
    
    # Score distribution
    excellent = sum(1 for r in results if r["score"] >= 90)
    good = sum(1 for r in results if 70 <= r["score"] < 90)
    fair = sum(1 for r in results if 50 <= r["score"] < 70)
    poor = sum(1 for r in results if r["score"] < 50)
    
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}TEST SUMMARY{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}Overall Results:{Colors.ENDC}")
    print(f"  Total Tests: {total_tests}")
    print(f"  Successful: {Colors.OKGREEN}{successful_tests}{Colors.ENDC}")
    print(f"  Failed: {Colors.FAIL}{total_tests - successful_tests}{Colors.ENDC}")
    print(f"  Average Score: {Colors.OKCYAN}{average_score:.1f}/100{Colors.ENDC}")
    print(f"  Average Response Time: {average_time:.2f}s\n")
    
    print(f"{Colors.BOLD}Category Performance:{Colors.ENDC}")
    print(f"  Restaurants: {Colors.OKCYAN}{restaurant_avg:.1f}/100{Colors.ENDC} ({len(restaurant_tests)} tests)")
    print(f"  Places & Attractions: {Colors.OKCYAN}{attraction_avg:.1f}/100{Colors.ENDC} ({len(attraction_tests)} tests)\n")
    
    print(f"{Colors.BOLD}Score Distribution:{Colors.ENDC}")
    print(f"  {Colors.OKGREEN}Excellent (90-100):{Colors.ENDC} {excellent} tests")
    print(f"  {Colors.OKGREEN}Good (70-89):{Colors.ENDC} {good} tests")
    print(f"  {Colors.WARNING}Fair (50-69):{Colors.ENDC} {fair} tests")
    print(f"  {Colors.FAIL}Poor (<50):{Colors.ENDC} {poor} tests\n")
    
    # Top issues
    all_issues = []
    for r in results:
        all_issues.extend(r["issues"])
    
    if all_issues:
        from collections import Counter
        issue_counts = Counter(all_issues)
        print(f"{Colors.BOLD}Most Common Issues:{Colors.ENDC}")
        for issue, count in issue_counts.most_common(5):
            print(f"  • {issue}: {count} occurrences")
    
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")

def main():
    """Main test execution"""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║         KAM AI CHAT COMPREHENSIVE TEST SUITE                  ║")
    print("║         Testing Restaurants & Attractions Features            ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    # Load test cases
    print(f"{Colors.OKBLUE}Loading test cases...{Colors.ENDC}")
    test_data = load_test_cases()
    test_cases = test_data["test_cases"]
    
    print(f"{Colors.OKGREEN}✓ Loaded {len(test_cases)} test cases{Colors.ENDC}")
    print(f"  - Restaurants: {test_data['categories']['restaurants']}")
    print(f"  - Places & Attractions: {test_data['categories']['places_attractions']}\n")
    
    # Check API availability
    print(f"{Colors.OKBLUE}Checking API availability...{Colors.ENDC}")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"{Colors.OKGREEN}✓ API is online and ready{Colors.ENDC}\n")
        else:
            print(f"{Colors.WARNING}⚠ API responded with status {response.status_code}{Colors.ENDC}\n")
    except requests.exceptions.RequestException:
        print(f"{Colors.FAIL}✗ API is not available at {API_BASE_URL}{Colors.ENDC}")
        print(f"{Colors.WARNING}Please start the backend server with: python app.py or uvicorn app:app{Colors.ENDC}\n")
        sys.exit(1)
    
    # Run tests
    print(f"{Colors.HEADER}Starting test execution...{Colors.ENDC}\n")
    results = []
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"{Colors.OKCYAN}Running test {idx}/{len(test_cases)}...{Colors.ENDC}", end="\r")
        
        # Send request
        api_response = send_chat_request(test_case["input"])
        
        # Analyze response
        analysis = analyze_response(test_case, api_response)
        results.append(analysis)
        
        # Print result
        print_test_result(analysis, idx, len(test_cases))
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Print summary
    print_summary(results)
    
    # Save results
    output = {
        "test_suite": test_data["test_suite"],
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(results),
        "results": results,
        "summary": {
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "average_score": sum(r["score"] for r in results) / len(results),
            "average_response_time": sum(r["response_time"] for r in results) / len(results),
        }
    }
    
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{Colors.OKGREEN}Results saved to: {RESULTS_FILE}{Colors.ENDC}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Test execution interrupted by user{Colors.ENDC}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Unexpected error: {e}{Colors.ENDC}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

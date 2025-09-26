#!/usr/bin/env python3
"""
Comprehensive Restaurant Advice Test Suite - 30 Tests
Tests restaurant recommendations using live Google Maps data
Focuses on accuracy, live data usage, and response formatting
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import re

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_RESULTS_FILE = f"restaurant_google_maps_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def make_request(query: str, session_id: str = "test_session") -> Dict[str, Any]:
    """Make a request to the AI Istanbul API"""
    try:
        response = requests.post(
            f"{BASE_URL}/ai/chat",
            json={"message": query, "session_id": session_id},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True,
                "response": data.get("response", ""),
                "status_code": response.status_code,
                "error": None
            }
        else:
            return {
                "success": False,
                "response": "",
                "status_code": response.status_code,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "response": "",
            "status_code": 0,
            "error": str(e)
        }

def analyze_restaurant_response(query: str, response: str) -> Dict[str, Any]:
    """Analyze if the response contains Google Maps live data indicators"""
    
    # Indicators that suggest live Google Maps data
    google_maps_indicators = [
        "google maps",
        "live recommendations",
        "ratings_total",
        "⭐⭐⭐⭐⭐",
        "⭐⭐⭐⭐",
        "/5.0",
        "reviews)",
        "highly-rated",
        "top-rated",
        "rating",
        "💰💰💰",
        "💰💰",
        "price level",
        "vicinity"
    ]
    
    # Count live data indicators
    live_indicators_found = []
    for indicator in google_maps_indicators:
        if indicator.lower() in response.lower():
            live_indicators_found.append(indicator)
    
    # Check for specific restaurant names (suggests real data)
    has_specific_restaurants = bool(re.search(r'\*\*[A-Z][a-zA-Z\s&]+Restaurant\*\*|\*\*[A-Z][a-zA-Z\s&]+Lokantası\*\*|\*\*[A-Z][a-zA-Z\s&]+Cafe\*\*', response))
    
    # Check for ratings format (X.X/5.0 format)
    has_ratings_format = bool(re.search(r'\d\.\d/5\.0', response))
    
    # Check for review counts
    has_review_counts = bool(re.search(r'\(\d+[,\d]*\s*reviews?\)', response))
    
    # Check for price indicators
    has_price_indicators = bool(re.search(r'💰+\s*(?:Budget|Moderate|Expensive)', response))
    
    # Check for location/vicinity info
    has_location_info = "📍" in response or "vicinity" in response.lower()
    
    # Check for proper formatting (no double asterisks)
    has_formatting_issues = "** **" in response or "****" in response
    
    # Determine if likely using live data
    live_data_score = len(live_indicators_found)
    uses_live_data = (
        live_data_score >= 3 or 
        has_ratings_format or 
        has_review_counts or
        "google maps" in response.lower()
    )
    
    return {
        "uses_live_data": uses_live_data,
        "live_data_score": live_data_score,
        "live_indicators_found": live_indicators_found,
        "has_specific_restaurants": has_specific_restaurants,
        "has_ratings_format": has_ratings_format,
        "has_review_counts": has_review_counts,
        "has_price_indicators": has_price_indicators,
        "has_location_info": has_location_info,
        "has_formatting_issues": has_formatting_issues,
        "response_length": len(response),
        "is_relevant": any(word in query.lower() for word in ["restaurant", "food", "eat", "dining", "cuisine"])
    }

def run_restaurant_tests() -> List[Dict[str, Any]]:
    """Run comprehensive restaurant advice tests"""
    
    # 30 diverse restaurant queries to test Google Maps integration
    test_queries = [
        # General restaurant queries
        "Best restaurants in Istanbul",
        "Top rated restaurants in Istanbul",
        "Highly rated restaurants in Istanbul",
        "Popular restaurants in Istanbul",
        "Good restaurants in Istanbul",
        
        # Cuisine-specific queries
        "Best Turkish restaurants in Istanbul",
        "Traditional Turkish restaurants in Istanbul", 
        "Authentic Turkish restaurants in Istanbul",
        "Best seafood restaurants in Istanbul",
        "Fresh seafood restaurants in Istanbul",
        "Vegetarian restaurants in Istanbul",
        "Vegan restaurants in Istanbul",
        "Italian restaurants in Istanbul",
        "Asian restaurants in Istanbul",
        "Japanese restaurants in Istanbul",
        
        # Location-based queries
        "Best restaurants in Sultanahmet",
        "Restaurants in Beyoğlu with good ratings",
        "Top restaurants in Galata area",
        "Kadıköy restaurant recommendations",
        "Restaurants near Galata Tower",
        "Dining in Taksim area",
        "Restaurants in Karaköy district",
        
        # Special dining experiences
        "Fine dining restaurants in Istanbul",
        "Luxury restaurants in Istanbul",
        "Rooftop restaurants in Istanbul", 
        "Restaurants with Bosphorus view",
        "Turkish breakfast places in Istanbul",
        "Best kebab restaurants in Istanbul",
        "Meze restaurants in Istanbul",
        "Restaurants for special occasions in Istanbul"
    ]
    
    results = []
    total_tests = len(test_queries)
    
    print(f"🚀 Starting Restaurant Google Maps Test Suite")
    print(f"📊 Running {total_tests} restaurant advice tests...")
    print(f"🎯 Testing live Google Maps data integration")
    print("-" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}/{total_tests}: {query}")
        
        # Make the request
        result = make_request(query)
        
        if result["success"]:
            # Analyze the response
            analysis = analyze_restaurant_response(query, result["response"])
            
            # Combine result with analysis
            test_result = {
                "test_number": i,
                "query": query,
                "success": True,
                "response": result["response"],
                "response_length": len(result["response"]),
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            # Print analysis
            status = "✅ LIVE DATA" if analysis["uses_live_data"] else "❌ STATIC DATA"
            print(f"   {status} (Score: {analysis['live_data_score']}/10)")
            
            if analysis["has_ratings_format"]:
                print(f"   ⭐ Has ratings format")
            if analysis["has_review_counts"]:
                print(f"   📊 Has review counts") 
            if analysis["has_price_indicators"]:
                print(f"   💰 Has price indicators")
            if analysis["has_formatting_issues"]:
                print(f"   ⚠️  Formatting issues detected")
                
        else:
            test_result = {
                "test_number": i,
                "query": query,
                "success": False,
                "response": result["response"],
                "error": result["error"],
                "status_code": result["status_code"],
                "analysis": {"uses_live_data": False},
                "timestamp": datetime.now().isoformat()
            }
            print(f"   ❌ FAILED: {result['error']}")
        
        results.append(test_result)
        
        # Small delay between requests
        time.sleep(0.5)
    
    return results

def generate_test_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - successful_tests
    
    # Analyze Google Maps integration
    live_data_tests = sum(1 for r in results if r.get("analysis", {}).get("uses_live_data", False))
    static_data_tests = successful_tests - live_data_tests
    
    # Analyze response quality
    avg_response_length = sum(len(r.get("response", "")) for r in results if r["success"]) // max(successful_tests, 1)
    
    # Analyze specific features
    tests_with_ratings = sum(1 for r in results if r.get("analysis", {}).get("has_ratings_format", False))
    tests_with_reviews = sum(1 for r in results if r.get("analysis", {}).get("has_review_counts", False))
    tests_with_prices = sum(1 for r in results if r.get("analysis", {}).get("has_price_indicators", False))
    tests_with_formatting_issues = sum(1 for r in results if r.get("analysis", {}).get("has_formatting_issues", False))
    
    # Calculate success rates
    live_data_rate = (live_data_tests / successful_tests * 100) if successful_tests > 0 else 0
    ratings_rate = (tests_with_ratings / successful_tests * 100) if successful_tests > 0 else 0
    reviews_rate = (tests_with_reviews / successful_tests * 100) if successful_tests > 0 else 0
    
    report = {
        "test_summary": {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0
        },
        "google_maps_integration": {
            "live_data_tests": live_data_tests,
            "static_data_tests": static_data_tests,
            "live_data_rate": live_data_rate,
            "live_data_success": live_data_rate >= 80  # Target: 80% should use live data
        },
        "response_quality": {
            "avg_response_length": avg_response_length,
            "tests_with_ratings": tests_with_ratings,
            "tests_with_reviews": tests_with_reviews,
            "tests_with_prices": tests_with_prices,
            "tests_with_formatting_issues": tests_with_formatting_issues,
            "ratings_rate": ratings_rate,
            "reviews_rate": reviews_rate
        },
        "recommendations": [],
        "timestamp": datetime.now().isoformat()
    }
    
    # Generate recommendations
    if live_data_rate < 80:
        report["recommendations"].append(f"🔧 Low live data usage ({live_data_rate:.1f}%) - Check Google Maps API integration")
    
    if tests_with_formatting_issues > 0:
        report["recommendations"].append(f"🎨 Fix formatting issues in {tests_with_formatting_issues} responses")
    
    if ratings_rate < 60:
        report["recommendations"].append(f"⭐ Low rating inclusion ({ratings_rate:.1f}%) - Improve rating display")
        
    if reviews_rate < 60:
        report["recommendations"].append(f"📊 Low review count inclusion ({reviews_rate:.1f}%) - Include review counts")
    
    if avg_response_length < 500:
        report["recommendations"].append("📝 Consider more detailed restaurant descriptions")
    elif avg_response_length > 2000:
        report["recommendations"].append("✂️ Consider shorter, more concise responses")
    
    return report

def main():
    """Run the restaurant Google Maps test suite"""
    print("🍽️ Restaurant Google Maps Integration Test Suite")
    print("=" * 60)
    
    try:
        # Test server connectivity
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)
        if health_response.status_code != 200:
            print(f"❌ Server not available at {BASE_URL}")
            return
        
        print(f"✅ Server is running at {BASE_URL}")
        
        # Run tests
        results = run_restaurant_tests()
        
        # Generate report
        report = generate_test_report(results)
        
        # Save results
        output_data = {
            "report": report,
            "test_results": results
        }
        
        with open(TEST_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎯 RESTAURANT GOOGLE MAPS TEST RESULTS")
        print("=" * 60)
        
        summary = report["test_summary"]
        google_maps = report["google_maps_integration"]
        quality = report["response_quality"]
        
        print(f"📊 Tests Run: {summary['total_tests']}")
        print(f"✅ Successful: {summary['successful_tests']} ({summary['success_rate']:.1f}%)")
        print(f"❌ Failed: {summary['failed_tests']}")
        print()
        
        print("🗺️ GOOGLE MAPS INTEGRATION:")
        print(f"   Live Data Tests: {google_maps['live_data_tests']}/{summary['successful_tests']} ({google_maps['live_data_rate']:.1f}%)")
        status = "✅ EXCELLENT" if google_maps['live_data_rate'] >= 80 else "⚠️ NEEDS IMPROVEMENT"
        print(f"   Status: {status}")
        print()
        
        print("📈 RESPONSE QUALITY:")
        print(f"   Avg Response Length: {quality['avg_response_length']} characters")
        print(f"   Responses with Ratings: {quality['tests_with_ratings']} ({quality['ratings_rate']:.1f}%)")
        print(f"   Responses with Reviews: {quality['tests_with_reviews']} ({quality['reviews_rate']:.1f}%)")
        print(f"   Responses with Prices: {quality['tests_with_prices']}")
        print(f"   Formatting Issues: {quality['tests_with_formatting_issues']}")
        print()
        
        if report["recommendations"]:
            print("💡 RECOMMENDATIONS:")
            for rec in report["recommendations"]:
                print(f"   {rec}")
            print()
        
        print(f"💾 Detailed results saved to: {TEST_RESULTS_FILE}")
        
        # Overall assessment
        if google_maps['live_data_rate'] >= 80 and summary['success_rate'] >= 95:
            print("🎉 OVERALL: EXCELLENT - Restaurant advice is using live Google Maps data!")
        elif google_maps['live_data_rate'] >= 60 and summary['success_rate'] >= 90:
            print("👍 OVERALL: GOOD - Most restaurant advice uses live data")
        else:
            print("⚠️ OVERALL: NEEDS IMPROVEMENT - Check Google Maps integration")
            
    except Exception as e:
        print(f"❌ Test suite failed: {str(e)}")
        return

if __name__ == "__main__":
    main()

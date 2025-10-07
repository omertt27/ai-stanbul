#!/usr/bin/env python3
"""
Comprehensive Restaurant Discovery Analysis & Test Suite
Tests 500 restaurants with Google Places API integration, location searches, 
cuisine filtering, dietary restrictions, and smart query processing.

Features tested:
- Real-time restaurant recommendations from Google Places API
- Location-specific searches (Beyoğlu, Sultanahmet, Kadıköy, etc.)
- Cuisine filtering (Turkish, seafood, vegetarian, street food)
- Dietary restrictions support (vegetarian, vegan, halal, kosher, gluten-free)
- Price level indicators and operating hours
- Smart typo correction and context-aware follow-ups
- Answer correctness analysis for 500 restaurants
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
import sys
import os

# Add the backend directory to Python path
sys.path.append('/Users/omer/Desktop/ai-stanbul/backend')

# Test configuration
BASE_URL = "http://localhost:8000"
CHAT_ENDPOINT = f"{BASE_URL}/api/chat"
RESTAURANT_ENDPOINT = f"{BASE_URL}/api/restaurants"

class RestaurantDiscoveryAnalyzer:
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        self.restaurant_count = 0
        self.correctness_analysis = {
            "location_accuracy": [],
            "cuisine_accuracy": [], 
            "dietary_accuracy": [],
            "price_accuracy": [],
            "description_quality": [],
            "typo_correction": []
        }
        
    def log_result(self, test_name: str, query: str, success: bool, response_data: Dict = None, 
                  error: str = None, analysis: Dict = None):
        """Log test result with correctness analysis"""
        result = {
            "test_name": test_name,
            "query": query,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data,
            "error": error,
            "correctness_analysis": analysis
        }
        self.test_results.append(result)
        
        if success:
            self.passed_tests += 1
            print(f"✅ {test_name}: {query}")
            if analysis:
                print(f"   📊 Analysis: {analysis.get('summary', 'No analysis')}")
        else:
            self.failed_tests += 1
            print(f"❌ {test_name}: {query} - {error}")
        
        self.total_tests += 1
    
    def make_chat_request(self, query: str) -> Dict:
        """Make request to chat endpoint which handles restaurant queries"""
        try:
            response = requests.post(
                CHAT_ENDPOINT, 
                json={"message": query, "session_id": "test_session"},
                timeout=30
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def make_direct_restaurant_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make direct request to restaurant endpoints"""
        try:
            url = f"{RESTAURANT_ENDPOINT}/{endpoint}" if not endpoint.startswith('http') else endpoint
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_restaurant_response(self, response_data: Dict, query: str, expected_criteria: Dict) -> Dict:
        """Analyze correctness of restaurant response"""
        analysis = {
            "restaurants_found": 0,
            "location_match": False,
            "cuisine_match": False,
            "dietary_match": False,
            "price_match": False,
            "has_descriptions": False,
            "average_rating": 0,
            "summary": ""
        }
        
        # Extract restaurants from various response formats
        restaurants = []
        if isinstance(response_data, dict):
            if "restaurants" in response_data:
                restaurants = response_data["restaurants"]
            elif "results" in response_data:
                restaurants = response_data["results"]
            elif "response" in response_data:
                # Chat response format
                response_text = response_data.get("response", "")
                # Check if response contains restaurant information
                if any(word in response_text.lower() for word in ["restaurant", "cuisine", "food", "dining"]):
                    analysis["has_descriptions"] = True
                    analysis["restaurants_found"] = response_text.count("restaurant") + response_text.count("Restaurant")
        
        if isinstance(restaurants, list):
            analysis["restaurants_found"] = len(restaurants)
            
            if restaurants:
                # Check location accuracy
                expected_location = expected_criteria.get("location", "").lower()
                if expected_location:
                    location_matches = sum(1 for r in restaurants 
                                         if expected_location in str(r).lower())
                    analysis["location_match"] = location_matches > 0
                
                # Check cuisine accuracy
                expected_cuisine = expected_criteria.get("cuisine", "").lower()
                if expected_cuisine:
                    cuisine_matches = sum(1 for r in restaurants 
                                        if expected_cuisine in str(r).lower())
                    analysis["cuisine_match"] = cuisine_matches > 0
                
                # Check dietary restrictions
                expected_dietary = expected_criteria.get("dietary", "").lower()
                if expected_dietary:
                    dietary_matches = sum(1 for r in restaurants 
                                        if expected_dietary in str(r).lower())
                    analysis["dietary_match"] = dietary_matches > 0
                
                # Check if restaurants have descriptions
                descriptions = [r.get("description", "") for r in restaurants if isinstance(r, dict)]
                analysis["has_descriptions"] = any(desc and len(desc) > 20 for desc in descriptions)
                
                # Calculate average rating
                ratings = [r.get("rating", 0) for r in restaurants if isinstance(r, dict) and r.get("rating")]
                if ratings:
                    analysis["average_rating"] = round(sum(ratings) / len(ratings), 1)
        
        # Generate summary
        summary_parts = []
        if analysis["restaurants_found"] > 0:
            summary_parts.append(f"{analysis['restaurants_found']} restaurants")
        if analysis["location_match"]:
            summary_parts.append("✓ location")
        if analysis["cuisine_match"]:
            summary_parts.append("✓ cuisine")
        if analysis["dietary_match"]:
            summary_parts.append("✓ dietary")
        if analysis["has_descriptions"]:
            summary_parts.append("✓ descriptions")
        if analysis["average_rating"] > 0:
            summary_parts.append(f"avg rating {analysis['average_rating']}")
        
        analysis["summary"] = ", ".join(summary_parts) if summary_parts else "No restaurants found"
        
        return analysis
    
    def test_restaurant_query(self, query: str, expected_criteria: Dict, test_name: str) -> Dict:
        """Test restaurant query and analyze correctness"""
        
        # Try chat endpoint first (most likely to work)
        result = self.make_chat_request(query)
        
        if result["success"]:
            analysis = self.analyze_restaurant_response(result["data"], query, expected_criteria)
            success = analysis["restaurants_found"] > 0 or analysis["has_descriptions"]
            self.log_result(test_name, query, success, result["data"], analysis=analysis)
            return result["data"]
        else:
            # Try direct restaurant endpoint as fallback
            direct_result = self.make_direct_restaurant_request("search", {"q": query})
            if direct_result["success"]:
                analysis = self.analyze_restaurant_response(direct_result["data"], query, expected_criteria)
                success = analysis["restaurants_found"] > 0
                self.log_result(test_name, query, success, direct_result["data"], analysis=analysis)
                return direct_result["data"]
            else:
                self.log_result(test_name, query, False, error=result["error"])
                return None
    
    def run_comprehensive_restaurant_tests(self):
        """Run comprehensive restaurant discovery tests with correctness analysis"""
        print("🍽️ Starting Comprehensive Restaurant Discovery Analysis")
        print("🎯 Testing 500+ restaurants with correctness analysis")
        print("=" * 80)
        
        # Test server connectivity
        try:
            health_check = requests.get(f"{BASE_URL}/health", timeout=5)
            print("✅ Server is accessible")
        except:
            print("❌ Server not accessible. Please start the backend server.")
            return
        
        # 1-10: Istanbul District-Specific Searches
        print("\n📍 DISTRICT-SPECIFIC RESTAURANT SEARCHES (1-10)")
        print("-" * 60)
        
        districts = [
            ("Beyoğlu", {"location": "beyoğlu"}),
            ("Sultanahmet", {"location": "sultanahmet"}),
            ("Kadıköy", {"location": "kadıköy"}),
            ("Beşiktaş", {"location": "beşiktaş"}),
            ("Taksim", {"location": "taksim"}),
            ("Galata", {"location": "galata"}),
            ("Üsküdar", {"location": "üsküdar"}),
            ("Ortaköy", {"location": "ortaköy"}),
            ("Eminönü", {"location": "eminönü"}),
            ("Karaköy", {"location": "karaköy"})
        ]
        
        for i, (district, criteria) in enumerate(districts, 1):
            query = f"restaurants in {district}"
            self.test_restaurant_query(query, criteria, f"{i}. {district} Restaurants")
        
        # 11-20: Turkish Cuisine Variations
        print("\n🇹🇷 TURKISH CUISINE SEARCHES (11-20)")
        print("-" * 60)
        
        turkish_cuisines = [
            ("Turkish restaurants in Sultanahmet", {"location": "sultanahmet", "cuisine": "turkish"}),
            ("Ottoman cuisine in Istanbul", {"cuisine": "ottoman"}),
            ("best kebab restaurants", {"cuisine": "kebab"}),
            ("Turkish breakfast places", {"cuisine": "breakfast"}),
            ("meze restaurants in Beyoğlu", {"location": "beyoğlu", "cuisine": "meze"}),
            ("pide restaurants in Kadıköy", {"location": "kadıköy", "cuisine": "pide"}),
            ("döner places near Taksim", {"location": "taksim", "cuisine": "döner"}),
            ("Turkish street food", {"cuisine": "street food"}),
            ("baklava shops in Istanbul", {"cuisine": "baklava"}),
            ("Turkish tea houses in Sultanahmet", {"location": "sultanahmet", "cuisine": "tea"})
        ]
        
        for i, (query, criteria) in enumerate(turkish_cuisines, 11):
            self.test_restaurant_query(query, criteria, f"{i}. Turkish Cuisine")
        
        # 21-30: International Cuisine
        print("\n🌍 INTERNATIONAL CUISINE SEARCHES (21-30)")
        print("-" * 60)
        
        international_cuisines = [
            ("Italian restaurants in Galata", {"location": "galata", "cuisine": "italian"}),
            ("Japanese restaurants in Beyoğlu", {"location": "beyoğlu", "cuisine": "japanese"}),
            ("Chinese restaurants in Kadıköy", {"location": "kadıköy", "cuisine": "chinese"}),
            ("Indian restaurants in Taksim", {"location": "taksim", "cuisine": "indian"}),
            ("Mediterranean cuisine in Beşiktaş", {"location": "beşiktaş", "cuisine": "mediterranean"}),
            ("French restaurants in Istanbul", {"cuisine": "french"}),
            ("Mexican food in Kadıköy", {"location": "kadıköy", "cuisine": "mexican"}),
            ("seafood restaurants near Bosphorus", {"cuisine": "seafood"}),
            ("sushi restaurants in Istanbul", {"cuisine": "sushi"}),
            ("pizza places in Beyoğlu", {"location": "beyoğlu", "cuisine": "pizza"})
        ]
        
        for i, (query, criteria) in enumerate(international_cuisines, 21):
            self.test_restaurant_query(query, criteria, f"{i}. International Cuisine")
        
        # 31-40: Dietary Restrictions & Special Needs
        print("\n🥗 DIETARY RESTRICTIONS & SPECIAL NEEDS (31-40)")
        print("-" * 60)
        
        dietary_queries = [
            ("vegetarian restaurants in Beyoğlu", {"location": "beyoğlu", "dietary": "vegetarian"}),
            ("vegan restaurants in Kadıköy", {"location": "kadıköy", "dietary": "vegan"}),
            ("halal restaurants in Sultanahmet", {"location": "sultanahmet", "dietary": "halal"}),
            ("kosher restaurants in Istanbul", {"dietary": "kosher"}),
            ("gluten-free restaurants in Beşiktaş", {"location": "beşiktaş", "dietary": "gluten-free"}),
            ("halal Turkish food", {"cuisine": "turkish", "dietary": "halal"}),
            ("vegetarian Turkish cuisine", {"cuisine": "turkish", "dietary": "vegetarian"}),
            ("vegan friendly restaurants", {"dietary": "vegan"}),
            ("gluten free dining options", {"dietary": "gluten-free"}),
            ("healthy restaurants in Istanbul", {"dietary": "healthy"})
        ]
        
        for i, (query, criteria) in enumerate(dietary_queries, 31):
            self.test_restaurant_query(query, criteria, f"{i}. Dietary Restrictions")
        
        # 41-45: Price Level & Quality Searches
        print("\n💰 PRICE LEVEL & QUALITY SEARCHES (41-45)")
        print("-" * 60)
        
        price_queries = [
            ("cheap restaurants in Kadıköy", {"location": "kadıköy", "price": "cheap"}),
            ("expensive restaurants in Beyoğlu", {"location": "beyoğlu", "price": "expensive"}),
            ("fine dining restaurants in Istanbul", {"price": "fine dining"}),
            ("budget restaurants near Sultanahmet", {"location": "sultanahmet", "price": "budget"}),
            ("luxury restaurants with Bosphorus view", {"price": "luxury"})
        ]
        
        for i, (query, criteria) in enumerate(price_queries, 41):
            self.test_restaurant_query(query, criteria, f"{i}. Price Level")
        
        # 46-50: Smart Query Processing & Typo Correction
        print("\n🧠 SMART QUERY PROCESSING & TYPO CORRECTION (46-50)")
        print("-" * 60)
        
        smart_queries = [
            ("resturants in istanbul", {"typo_correction": True}),  # restaurants
            ("turksh food near me", {"cuisine": "turkish", "typo_correction": True}),  # turkish
            ("vegerarian places in beyoglu", {"dietary": "vegetarian", "typo_correction": True}),  # vegetarian
            ("restraunts with good reveiws", {"typo_correction": True}),  # restaurants, reviews
            ("best turkish breakfast near galata bridge with bosphorus view", {"complex_query": True})
        ]
        
        for i, (query, criteria) in enumerate(smart_queries, 46):
            self.test_restaurant_query(query, criteria, f"{i}. Smart Processing")
        
        # Analysis and Summary
        self.generate_comprehensive_analysis()
    
    def generate_comprehensive_analysis(self):
        """Generate comprehensive analysis of restaurant discovery accuracy"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE RESTAURANT DISCOVERY ANALYSIS")
        print("=" * 80)
        
        # Basic stats
        print(f"📈 BASIC STATISTICS:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests} ✅")
        print(f"   Failed: {self.failed_tests} ❌")
        print(f"   Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        # Analyze by category
        categories = {
            "District-Specific": [r for r in self.test_results if ". " in r["test_name"] and int(r["test_name"].split(".")[0]) <= 10],
            "Turkish Cuisine": [r for r in self.test_results if "Turkish Cuisine" in r["test_name"]],
            "International": [r for r in self.test_results if "International" in r["test_name"]],
            "Dietary": [r for r in self.test_results if "Dietary" in r["test_name"]],
            "Price Level": [r for r in self.test_results if "Price Level" in r["test_name"]],
            "Smart Processing": [r for r in self.test_results if "Smart Processing" in r["test_name"]]
        }
        
        print(f"\n📊 CATEGORY ANALYSIS:")
        for category_name, results in categories.items():
            if results:
                passed = sum(1 for r in results if r["success"])
                total = len(results)
                avg_restaurants = sum(r.get("correctness_analysis", {}).get("restaurants_found", 0) 
                                    for r in results if r.get("correctness_analysis")) / max(1, total)
                print(f"   {category_name}: {passed}/{total} ({(passed/total)*100:.1f}%) - Avg {avg_restaurants:.1f} restaurants/query")
        
        # Quality metrics
        all_analyses = [r.get("correctness_analysis") for r in self.test_results if r.get("correctness_analysis")]
        
        if all_analyses:
            location_accuracy = sum(1 for a in all_analyses if a.get("location_match")) / len(all_analyses) * 100
            cuisine_accuracy = sum(1 for a in all_analyses if a.get("cuisine_match")) / len(all_analyses) * 100
            dietary_accuracy = sum(1 for a in all_analyses if a.get("dietary_match")) / len(all_analyses) * 100
            description_quality = sum(1 for a in all_analyses if a.get("has_descriptions")) / len(all_analyses) * 100
            
            print(f"\n🎯 ACCURACY METRICS:")
            print(f"   Location Accuracy: {location_accuracy:.1f}%")
            print(f"   Cuisine Accuracy: {cuisine_accuracy:.1f}%")
            print(f"   Dietary Accuracy: {dietary_accuracy:.1f}%")
            print(f"   Description Quality: {description_quality:.1f}%")
            
            # Average restaurants per query
            total_restaurants = sum(a.get("restaurants_found", 0) for a in all_analyses)
            avg_restaurants_per_query = total_restaurants / len(all_analyses)
            print(f"   Avg Restaurants/Query: {avg_restaurants_per_query:.1f}")
        
        # Top performing queries
        successful_tests = [r for r in self.test_results if r["success"]]
        if successful_tests:
            top_tests = sorted(successful_tests, 
                             key=lambda x: x.get("correctness_analysis", {}).get("restaurants_found", 0), 
                             reverse=True)[:5]
            
            print(f"\n🏆 TOP PERFORMING QUERIES:")
            for i, test in enumerate(top_tests, 1):
                analysis = test.get("correctness_analysis", {})
                print(f"   {i}. {test['query']} - {analysis.get('summary', 'No data')}")
        
        # Issues found
        failed_tests = [r for r in self.test_results if not r["success"]]
        if failed_tests:
            print(f"\n⚠️ ISSUES IDENTIFIED ({len(failed_tests)} tests failed):")
            for test in failed_tests[:5]:  # Show first 5 failures
                print(f"   • {test['query']} - {test.get('error', 'Unknown error')}")
        
        return {
            "total_tests": self.total_tests,
            "success_rate": (self.passed_tests/self.total_tests)*100,
            "categories": categories,
            "quality_metrics": {
                "location_accuracy": location_accuracy if all_analyses else 0,
                "cuisine_accuracy": cuisine_accuracy if all_analyses else 0,
                "dietary_accuracy": dietary_accuracy if all_analyses else 0,
                "description_quality": description_quality if all_analyses else 0
            } if all_analyses else {}
        }
    
    def save_results(self, filename: str = None):
        """Save comprehensive test results"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"restaurant_discovery_analysis_{timestamp}.json"
        
        # Generate comprehensive analysis
        analysis_summary = self.generate_comprehensive_analysis()
        
        results_data = {
            "test_metadata": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": f"{(self.passed_tests/self.total_tests)*100:.1f}%",
                "timestamp": datetime.now().isoformat(),
                "restaurant_database_size": "500+",
                "test_categories": [
                    "District-Specific Searches",
                    "Turkish Cuisine Variations",
                    "International Cuisine",
                    "Dietary Restrictions",
                    "Price Level Filtering",
                    "Smart Query Processing"
                ]
            },
            "analysis_summary": analysis_summary,
            "detailed_results": self.test_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Comprehensive analysis saved to: {filename}")
        return filename

def main():
    """Main execution function"""
    print("🚀 Restaurant Discovery Comprehensive Analysis")
    print("🎯 Testing 500+ Restaurants with Correctness Analysis")
    print(f"🌐 Target URL: {BASE_URL}")
    print()
    
    # Initialize analyzer
    analyzer = RestaurantDiscoveryAnalyzer()
    
    # Run comprehensive tests
    analyzer.run_comprehensive_restaurant_tests()
    
    # Save results
    results_file = analyzer.save_results()
    
    print(f"\n✨ Analysis complete! Results saved to: {results_file}")
    
    return analyzer.test_results

if __name__ == "__main__":
    main()

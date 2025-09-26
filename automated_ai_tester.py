#!/usr/bin/env python3
"""
Automated AI Istanbul Chatbot Tester
====================================

This script automatically tests the AI Istanbul chatbot by sending API requests
and evaluating the responses programmatically.
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import List, Dict, Any

# Import test data
from daily_talk_test_inputs import (
    RESTAURANT_FOOD_QUERIES,
    TRANSPORTATION_QUERIES,
    CULTURAL_SITES_QUERIES,
    SHOPPING_QUERIES,
    GENERAL_TRAVEL_QUERIES
)

class AIIstanbulTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results = []
        self.session = requests.Session()
        
    def test_api_connection(self) -> bool:
        """Test if the API is accessible."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            try:
                # Try alternative health check endpoint
                response = self.session.get(f"{self.base_url}/", timeout=5)
                return response.status_code in [200, 404]  # 404 is fine if no root endpoint
            except:
                return False
    
    def send_chat_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the chatbot API."""
        try:
            payload = {
                "message": message
            }
            
            # Try different possible endpoints
            endpoints = ["/ai/chat", "/chat", "/api/chat", "/chatbot", "/ask"]
            
            for endpoint in endpoints:
                try:
                    response = self.session.post(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                        timeout=30,
                        headers={'Content-Type': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        return {
                            "success": True,
                            "response": response.json(),
                            "status_code": response.status_code,
                            "endpoint": endpoint
                        }
                except requests.exceptions.RequestException:
                    continue
            
            return {
                "success": False,
                "error": "No working endpoint found",
                "status_code": None,
                "endpoint": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": None,
                "endpoint": None
            }
    
    def evaluate_response(self, query: str, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of a chatbot response."""
        if not response_data["success"]:
            return {
                "query": query,
                "success": False,
                "error": response_data.get("error", "Unknown error"),
                "scores": {
                    "accuracy": 0,
                    "relevance": 0,
                    "completeness": 0,
                    "cultural_awareness": 0,
                    "overall": 0
                }
            }
        
        try:
            # Extract response text
            api_response = response_data["response"]
            
            # Handle different response formats
            if isinstance(api_response, dict):
                response_text = (
                    api_response.get("response") or 
                    api_response.get("message") or 
                    api_response.get("answer") or 
                    str(api_response)
                )
            else:
                response_text = str(api_response)
            
            # Basic automated evaluation criteria
            scores = self._score_response(query, response_text)
            
            return {
                "query": query,
                "success": True,
                "response_text": response_text,
                "response_length": len(response_text),
                "endpoint_used": response_data["endpoint"],
                "scores": scores,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "query": query,
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
                "scores": {
                    "accuracy": 0,
                    "relevance": 0,
                    "completeness": 0,
                    "cultural_awareness": 0,
                    "overall": 0
                }
            }
    
    def _score_response(self, query: str, response: str) -> Dict[str, float]:
        """Score a response based on various criteria."""
        scores = {}
        
        # Response length score (basic completeness indicator)
        length_score = min(len(response) / 500, 1.0) * 5  # Normalize to 0-5
        scores["completeness"] = round(length_score, 2)
        
        # Relevance score (keyword matching)
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        relevance = len(query_words.intersection(response_words)) / len(query_words)
        scores["relevance"] = round(relevance * 5, 2)
        
        # Istanbul/Turkish context score
        turkish_terms = [
            "istanbul", "turkish", "turkey", "bosphorus", "sultanahmet", 
            "taksim", "galata", "beyoglu", "kadikoy", "eminonu",
            "hagia sophia", "blue mosque", "grand bazaar", "topkapi"
        ]
        turkish_score = sum(1 for term in turkish_terms if term in response.lower())
        scores["cultural_awareness"] = round(min(turkish_score / 3, 1.0) * 5, 2)
        
        # Practical information score (look for practical details)
        practical_indicators = [
            "hour", "time", "price", "cost", "address", "metro", "bus",
            "walk", "minute", "lira", "euro", "open", "closed", "recommend"
        ]
        practical_score = sum(1 for term in practical_indicators if term in response.lower())
        scores["accuracy"] = round(min(practical_score / 5, 1.0) * 5, 2)
        
        # Overall score (average of all scores)
        all_scores = [scores[key] for key in scores.keys()]
        scores["overall"] = round(sum(all_scores) / len(all_scores), 2)
        
        return scores
    
    def run_test_suite(self, max_tests: int = 20) -> Dict[str, Any]:
        """Run the complete test suite."""
        print("🤖 AUTOMATED AI ISTANBUL CHATBOT TESTING")
        print("=" * 50)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 API Base URL: {self.base_url}")
        
        # Check API connection
        print("\n🔍 Checking API connection...")
        if not self.test_api_connection():
            print("❌ API connection failed! Make sure the backend server is running.")
            return {"success": False, "error": "API connection failed"}
        
        print("✅ API connection successful!")
        
        # Prepare test queries
        all_queries = (RESTAURANT_FOOD_QUERIES + TRANSPORTATION_QUERIES + 
                      CULTURAL_SITES_QUERIES + SHOPPING_QUERIES + 
                      GENERAL_TRAVEL_QUERIES)
        
        test_queries = all_queries[:max_tests]
        
        print(f"\n🧪 Running {len(test_queries)} automated tests...")
        
        # Run tests
        test_results = []
        categories = [
            ("🍽️ Restaurant & Food", RESTAURANT_FOOD_QUERIES),
            ("🚌 Transportation", TRANSPORTATION_QUERIES),
            ("🏛️ Cultural Sites", CULTURAL_SITES_QUERIES),
            ("🛒 Shopping", SHOPPING_QUERIES),
            ("✈️ General Travel", GENERAL_TRAVEL_QUERIES)
        ]
        
        test_num = 1
        for category_name, queries in categories:
            if test_num > max_tests:
                break
                
            print(f"\n{category_name}")
            print("-" * 40)
            
            for query in queries:
                if test_num > max_tests:
                    break
                
                print(f"🤖 Test #{test_num}: ", end="", flush=True)
                
                # Send request to API
                response_data = self.send_chat_message(query)
                
                # Evaluate response
                result = self.evaluate_response(query, response_data)
                test_results.append(result)
                
                # Print result
                if result["success"]:
                    score = result["scores"]["overall"]
                    if score >= 4:
                        print(f"✅ EXCELLENT ({score}/5)")
                    elif score >= 3:
                        print(f"✅ GOOD ({score}/5)")
                    elif score >= 2:
                        print(f"⚠️ FAIR ({score}/5)")
                    else:
                        print(f"❌ POOR ({score}/5)")
                else:
                    print(f"❌ FAILED - {result.get('error', 'Unknown error')}")
                
                test_num += 1
                time.sleep(1)  # Rate limiting
        
        # Calculate summary statistics
        summary = self._generate_summary(test_results)
        
        # Save detailed results
        self._save_results(test_results, summary)
        
        return {
            "success": True,
            "summary": summary,
            "detailed_results": test_results
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate test summary statistics."""
        successful_tests = [r for r in results if r["success"]]
        failed_tests = [r for r in results if not r["success"]]
        
        if not successful_tests:
            return {
                "total_tests": len(results),
                "successful_tests": 0,
                "failed_tests": len(failed_tests),
                "success_rate": 0,
                "average_scores": {},
                "grade": "F",
                "recommendation": self._get_recommendation({}, len(failed_tests))
            }
        
        # Calculate average scores
        avg_scores = {}
        score_keys = ["accuracy", "relevance", "completeness", "cultural_awareness", "overall"]
        
        for key in score_keys:
            scores = [r["scores"][key] for r in successful_tests if "scores" in r]
            avg_scores[key] = round(sum(scores) / len(scores) if scores else 0, 2)
        
        # Determine overall grade
        overall_avg = avg_scores.get("overall", 0)
        if overall_avg >= 4.5:
            grade = "A"
        elif overall_avg >= 4.0:
            grade = "B"
        elif overall_avg >= 3.0:
            grade = "C"
        elif overall_avg >= 2.0:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": round(len(successful_tests) / len(results) * 100, 1),
            "average_scores": avg_scores,
            "grade": grade,
            "recommendation": self._get_recommendation(avg_scores, len(failed_tests))
        }
    
    def _get_recommendation(self, scores: Dict[str, float], failed_count: int) -> str:
        """Generate improvement recommendations."""
        if failed_count > 5:
            return "CRITICAL: Multiple API failures. Check backend server connectivity."
        
        overall = scores.get("overall", 0)
        
        if overall >= 4.0:
            return "EXCELLENT: Chatbot is production-ready with high-quality responses."
        elif overall >= 3.0:
            return "GOOD: Chatbot performs well but could benefit from minor improvements."
        elif overall >= 2.0:
            return "FAIR: Chatbot needs improvement in response quality and local knowledge."
        else:
            return "POOR: Significant improvements needed before production deployment."
    
    def _save_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = f"ai_istanbul_test_results_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "detailed_results": results,
                "test_info": {
                    "timestamp": datetime.now().isoformat(),
                    "base_url": self.base_url,
                    "total_tests": len(results)
                }
            }, f, indent=2, ensure_ascii=False)
        
        # Save summary report
        report_file = f"ai_istanbul_test_report_{timestamp}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🇹🇷 AI ISTANBUL CHATBOT - AUTOMATED TEST REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🌐 API Base URL: {self.base_url}\n\n")
            
            f.write("📊 SUMMARY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Tests: {summary['total_tests']}\n")
            f.write(f"Successful: {summary['successful_tests']}\n")
            f.write(f"Failed: {summary['failed_tests']}\n")
            f.write(f"Success Rate: {summary['success_rate']}%\n")
            f.write(f"Overall Grade: {summary['grade']}\n\n")
            
            f.write("📈 AVERAGE SCORES (0-5 scale):\n")
            f.write("-" * 30 + "\n")
            for key, value in summary['average_scores'].items():
                f.write(f"{key.replace('_', ' ').title()}: {value}/5\n")
            
            f.write(f"\n💡 RECOMMENDATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{summary['recommendation']}\n")
        
        print(f"\n💾 Results saved:")
        print(f"   📄 Detailed: {detailed_file}")
        print(f"   📝 Report: {report_file}")

def main():
    """Main function to run the automated tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated AI Istanbul Chatbot Tester")
    parser.add_argument("--url", default="http://localhost:8001", help="Backend API URL")
    parser.add_argument("--tests", type=int, default=20, help="Number of tests to run (max 20)")
    parser.add_argument("--quick", action="store_true", help="Run quick test (5 tests)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.tests = 5
    
    # Initialize tester
    tester = AIIstanbulTester(args.url)
    
    try:
        # Run test suite
        results = tester.run_test_suite(args.tests)
        
        if results["success"]:
            summary = results["summary"]
            
            print(f"\n🎯 FINAL RESULTS:")
            print("=" * 30)
            print(f"✅ Success Rate: {summary['success_rate']}%")
            print(f"📊 Overall Score: {summary['average_scores']['overall']}/5")
            print(f"🎓 Grade: {summary['grade']}")
            print(f"\n💡 {summary['recommendation']}")
            
            # Exit with appropriate code
            if summary['grade'] in ['A', 'B']:
                sys.exit(0)  # Success
            else:
                sys.exit(1)  # Needs improvement
        else:
            print(f"\n❌ Test suite failed: {results.get('error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

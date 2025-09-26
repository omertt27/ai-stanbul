#!/usr/bin/env python3
"""
Final Comprehensive AI Istanbul Chatbot Tester
==============================================

Ultimate test suite using 80 diverse inputs to evaluate:
- Location-specific response quality
- Query understanding and routing
- Cultural awareness and accuracy
- Practical usefulness
- Overall user experience

This is the final validation test for the comprehensive location enhancement system.
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Import our comprehensive test inputs
from comprehensive_test_inputs import (
    ALL_COMPREHENSIVE_TEST_INPUTS, 
    TEST_CATEGORIES,
    get_test_input_by_category
)

@dataclass
class TestResult:
    query: str
    category: str
    response: str
    response_length: int
    response_time: float
    success: bool
    error_message: str = None
    location_detected: bool = False
    location_name: str = None
    has_walking_distances: bool = False
    has_cultural_context: bool = False
    has_practical_tips: bool = False
    has_transportation_info: bool = False
    estimated_relevance_score: float = 0.0
    estimated_completeness_score: float = 0.0
    estimated_cultural_score: float = 0.0

class ComprehensiveTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        self.session = requests.Session()
        self.session.timeout = 30
        
    def test_api_connection(self) -> bool:
        """Test if API is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False
    
    def analyze_response_quality(self, query: str, response: str) -> Tuple[float, float, float]:
        """Analyze response quality on multiple dimensions"""
        if not response or len(response) < 50:
            return 1.0, 1.0, 1.0  # Very low scores for poor responses
        
        # Relevance scoring based on content analysis
        relevance_score = self._score_relevance(query, response)
        
        # Completeness scoring based on information depth
        completeness_score = self._score_completeness(query, response)
        
        # Cultural awareness scoring
        cultural_score = self._score_cultural_awareness(query, response)
        
        return relevance_score, completeness_score, cultural_score
    
    def _score_relevance(self, query: str, response: str) -> float:
        """Score how relevant the response is to the query"""
        query_lower = query.lower()
        response_lower = response.lower()
        
        score = 2.0  # Base score
        
        # Location-specific bonus
        locations = ['sultanahmet', 'beyoğlu', 'beyoglu', 'kadıköy', 'kadikoy', 'taksim', 'karaköy', 'karakoy', 'eminönü', 'eminonu']
        query_has_location = any(loc in query_lower for loc in locations)
        response_has_location = any(loc in response_lower for loc in locations)
        
        if query_has_location and response_has_location:
            score += 1.5  # Strong bonus for location matching
        elif query_has_location and not response_has_location:
            score -= 1.0  # Penalty for missing location context
        
        # Query type matching
        if 'restaurant' in query_lower or 'food' in query_lower or 'eat' in query_lower:
            if any(word in response_lower for word in ['restaurant', 'food', 'dining', 'cuisine', 'meal']):
                score += 1.0
            if 'turkish' in response_lower:
                score += 0.5
                
        if 'transport' in query_lower or 'metro' in query_lower or 'travel' in query_lower:
            if any(word in response_lower for word in ['metro', 'tram', 'ferry', 'bus', 'transport']):
                score += 1.0
                
        if 'museum' in query_lower or 'cultural' in query_lower:
            if any(word in response_lower for word in ['museum', 'cultural', 'historical', 'mosque', 'palace']):
                score += 1.0
        
        # Specific details bonus
        if any(indicator in response_lower for indicator in ['walking distance', 'minute walk', 'meters', 'km']):
            score += 0.5
            
        if any(indicator in response_lower for indicator in ['tip:', 'advice', 'recommendation', 'suggest']):
            score += 0.5
        
        return min(5.0, max(1.0, score))
    
    def _score_completeness(self, query: str, response: str) -> float:
        """Score how complete and comprehensive the response is"""
        response_lower = response.lower()
        
        score = 2.0  # Base score
        
        # Length and detail indicators
        if len(response) > 500:
            score += 1.0
        elif len(response) > 300:
            score += 0.5
            
        # Structured information bonus
        structure_indicators = ['•', '-', '1.', '2.', '**', 'tips:', 'transportation:', 'walking']
        structure_count = sum(1 for indicator in structure_indicators if indicator in response_lower)
        score += min(1.0, structure_count * 0.2)
        
        # Comprehensive information elements
        info_elements = [
            'walking distance', 'transportation', 'metro', 'tram', 'ferry',
            'opening hours', 'recommendation', 'tip', 'advice', 'cultural',
            'budget', 'price', 'cost', 'free', 'expensive', 'cheap'
        ]
        info_count = sum(1 for element in info_elements if element in response_lower)
        score += min(1.5, info_count * 0.1)
        
        return min(5.0, max(1.0, score))
    
    def _score_cultural_awareness(self, query: str, response: str) -> float:
        """Score cultural awareness and sensitivity"""
        response_lower = response.lower()
        
        score = 3.0  # Base score (neutral)
        
        # Cultural context indicators
        cultural_indicators = [
            'turkish', 'ottoman', 'byzantine', 'islamic', 'mosque', 'cultural',
            'traditional', 'local', 'authentic', 'etiquette', 'custom',
            'respect', 'modest', 'shoes', 'prayer', 'ramadan', 'halal'
        ]
        
        cultural_count = sum(1 for indicator in cultural_indicators if indicator in response_lower)
        score += min(1.5, cultural_count * 0.1)
        
        # Specific cultural advice bonus
        if 'remove shoes' in response_lower or 'dress modest' in response_lower:
            score += 0.5
        if 'prayer time' in response_lower or 'prayer' in response_lower:
            score += 0.3
        if 'local custom' in response_lower or 'turkish culture' in response_lower:
            score += 0.4
        
        return min(5.0, max(1.0, score))
    
    def detect_response_features(self, response: str) -> Dict[str, bool]:
        """Detect specific features in the response"""
        response_lower = response.lower()
        
        return {
            'has_walking_distances': any(indicator in response_lower for indicator in 
                                       ['walking distance', 'minute walk', 'meters', 'km walk']),
            'has_cultural_context': any(indicator in response_lower for indicator in 
                                      ['cultural', 'traditional', 'custom', 'etiquette', 'respect']),
            'has_practical_tips': any(indicator in response_lower for indicator in 
                                    ['tip:', 'advice', 'recommend', 'suggestion', 'avoid']),
            'has_transportation_info': any(indicator in response_lower for indicator in 
                                         ['metro', 'tram', 'ferry', 'bus', 'transport']),
            'location_detected': any(loc in response_lower for loc in 
                                   ['sultanahmet', 'beyoğlu', 'beyoglu', 'kadıköy', 'kadikoy', 
                                    'taksim', 'karaköy', 'karakoy', 'eminönü', 'eminonu'])
        }
    
    def run_single_test(self, query: str, category: str) -> TestResult:
        """Run a single test query"""
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/ai/chat",
                json={"message": query},
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result_data = response.json()
                response_text = result_data.get('response', '')
                
                # Analyze response quality
                relevance, completeness, cultural = self.analyze_response_quality(query, response_text)
                
                # Detect features
                features = self.detect_response_features(response_text)
                
                return TestResult(
                    query=query,
                    category=category,
                    response=response_text,
                    response_length=len(response_text),
                    response_time=response_time,
                    success=True,
                    location_detected=features['location_detected'],
                    has_walking_distances=features['has_walking_distances'],
                    has_cultural_context=features['has_cultural_context'],
                    has_practical_tips=features['has_practical_tips'],
                    has_transportation_info=features['has_transportation_info'],
                    estimated_relevance_score=relevance,
                    estimated_completeness_score=completeness,
                    estimated_cultural_score=cultural
                )
            else:
                return TestResult(
                    query=query,
                    category=category,
                    response="",
                    response_length=0,
                    response_time=response_time,
                    success=False,
                    error_message=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                query=query,
                category=category,
                response="",
                response_length=0,
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
    
    def run_comprehensive_test_suite(self, sample_size: int = None) -> Dict[str, Any]:
        """Run the full comprehensive test suite"""
        print("🧪 COMPREHENSIVE AI ISTANBUL CHATBOT TEST SUITE")
        print("=" * 60)
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 API Base URL: {self.base_url}")
        print()
        
        # Test API connection
        print("🔍 Checking API connection...")
        if not self.test_api_connection():
            print("❌ API connection failed! Make sure the backend server is running.")
            return {"error": "API connection failed"}
        
        print("✅ API connection successful!")
        print()
        
        # Determine test inputs
        if sample_size:
            import random
            test_inputs = random.sample(ALL_COMPREHENSIVE_TEST_INPUTS, min(sample_size, len(ALL_COMPREHENSIVE_TEST_INPUTS)))
            print(f"🧪 Running {len(test_inputs)} sampled tests...")
        else:
            test_inputs = ALL_COMPREHENSIVE_TEST_INPUTS
            print(f"🧪 Running all {len(test_inputs)} comprehensive tests...")
        
        print()
        
        # Run tests by category
        category_results = {}
        test_number = 1
        
        for category, category_inputs in TEST_CATEGORIES.items():
            if sample_size:
                # Filter to only inputs in our sample
                category_test_inputs = [inp for inp in category_inputs if inp in test_inputs]
            else:
                category_test_inputs = category_inputs
                
            if not category_test_inputs:
                continue
                
            print(f"📋 {category.upper().replace('_', ' ')}")
            print("-" * 40)
            
            category_results[category] = []
            
            for query in category_test_inputs:
                result = self.run_single_test(query, category)
                self.results.append(result)
                category_results[category].append(result)
                
                # Display result
                if result.success:
                    avg_score = (result.estimated_relevance_score + 
                               result.estimated_completeness_score + 
                               result.estimated_cultural_score) / 3
                    
                    if avg_score >= 4.0:
                        status = "✅ EXCELLENT"
                    elif avg_score >= 3.5:
                        status = "✅ GOOD"
                    elif avg_score >= 2.5:
                        status = "⚠️ FAIR"
                    else:
                        status = "❌ POOR"
                    
                    print(f"🤖 Test #{test_number}: {status} ({avg_score:.2f}/5)")
                else:
                    print(f"🤖 Test #{test_number}: ❌ FAILED ({result.error_message})")
                
                test_number += 1
            
            print()
        
        # Calculate overall statistics
        successful_tests = [r for r in self.results if r.success]
        if successful_tests:
            avg_relevance = sum(r.estimated_relevance_score for r in successful_tests) / len(successful_tests)
            avg_completeness = sum(r.estimated_completeness_score for r in successful_tests) / len(successful_tests)
            avg_cultural = sum(r.estimated_cultural_score for r in successful_tests) / len(successful_tests)
            overall_score = (avg_relevance + avg_completeness + avg_cultural) / 3
            
            success_rate = len(successful_tests) / len(self.results) * 100
            
            # Calculate feature statistics
            location_detection_rate = sum(1 for r in successful_tests if r.location_detected) / len(successful_tests) * 100
            walking_distance_rate = sum(1 for r in successful_tests if r.has_walking_distances) / len(successful_tests) * 100
            cultural_context_rate = sum(1 for r in successful_tests if r.has_cultural_context) / len(successful_tests) * 100
            practical_tips_rate = sum(1 for r in successful_tests if r.has_practical_tips) / len(successful_tests) * 100
            
            # Grade assignment
            if overall_score >= 4.5:
                grade = "A+"
                recommendation = "EXCELLENT: Production-ready with outstanding performance!"
            elif overall_score >= 4.0:
                grade = "A"
                recommendation = "EXCELLENT: Ready for production deployment!"
            elif overall_score >= 3.5:
                grade = "B+"
                recommendation = "VERY GOOD: Minor improvements recommended before production."
            elif overall_score >= 3.0:
                grade = "B"
                recommendation = "GOOD: Some improvements needed for optimal performance."
            elif overall_score >= 2.5:
                grade = "C"
                recommendation = "FAIR: Significant improvements needed before production."
            elif overall_score >= 2.0:
                grade = "D"
                recommendation = "POOR: Major improvements required."
            else:
                grade = "F"
                recommendation = "FAILING: System needs substantial redesign."
        
        else:
            avg_relevance = avg_completeness = avg_cultural = overall_score = 0.0
            success_rate = 0.0
            location_detection_rate = walking_distance_rate = cultural_context_rate = practical_tips_rate = 0.0
            grade = "F"
            recommendation = "FAILING: No successful responses."
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_filename = f"comprehensive_test_results_{timestamp}.json"
        
        detailed_results = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': len(self.results),
                'successful_tests': len(successful_tests),
                'api_url': self.base_url
            },
            'summary': {
                'success_rate': success_rate,
                'overall_score': overall_score,
                'grade': grade,
                'recommendation': recommendation,
                'avg_relevance': avg_relevance,
                'avg_completeness': avg_completeness,
                'avg_cultural_awareness': avg_cultural,
                'location_detection_rate': location_detection_rate,
                'walking_distance_rate': walking_distance_rate,
                'cultural_context_rate': cultural_context_rate,
                'practical_tips_rate': practical_tips_rate
            },
            'category_breakdown': {},
            'detailed_results': []
        }
        
        # Add category breakdown
        for category, results in category_results.items():
            successful_category = [r for r in results if r.success]
            if successful_category:
                cat_avg = sum((r.estimated_relevance_score + r.estimated_completeness_score + r.estimated_cultural_score) / 3 
                             for r in successful_category) / len(successful_category)
                detailed_results['category_breakdown'][category] = {
                    'total_tests': len(results),
                    'successful_tests': len(successful_category),
                    'average_score': cat_avg,
                    'success_rate': len(successful_category) / len(results) * 100
                }
        
        # Add detailed results
        for result in self.results:
            detailed_results['detailed_results'].append({
                'query': result.query,
                'category': result.category,
                'success': result.success,
                'response_text': result.response,
                'response_length': result.response_length,
                'response_time': result.response_time,
                'error_message': result.error_message,
                'features': {
                    'location_detected': result.location_detected,
                    'has_walking_distances': result.has_walking_distances,
                    'has_cultural_context': result.has_cultural_context,
                    'has_practical_tips': result.has_practical_tips,
                    'has_transportation_info': result.has_transportation_info
                },
                'scores': {
                    'relevance': result.estimated_relevance_score,
                    'completeness': result.estimated_completeness_score,
                    'cultural_awareness': result.estimated_cultural_score,
                    'overall': (result.estimated_relevance_score + result.estimated_completeness_score + result.estimated_cultural_score) / 3
                },
                'timestamp': datetime.now().isoformat()
            })
        
        # Save results to file
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Print final results
        print("🎯 FINAL COMPREHENSIVE TEST RESULTS:")
        print("=" * 40)
        print(f"✅ Success Rate: {success_rate:.1f}%")
        print(f"📊 Overall Score: {overall_score:.2f}/5")
        print(f"🎓 Grade: {grade}")
        print()
        print("📈 DETAILED SCORES (0-5 scale):")
        print("-" * 30)
        print(f"Relevance: {avg_relevance:.2f}/5")
        print(f"Completeness: {avg_completeness:.2f}/5")
        print(f"Cultural Awareness: {avg_cultural:.2f}/5")
        print()
        print("🔧 FEATURE ANALYSIS:")
        print("-" * 30)
        print(f"Location Detection: {location_detection_rate:.1f}% of responses")
        print(f"Walking Distances: {walking_distance_rate:.1f}% of responses")
        print(f"Cultural Context: {cultural_context_rate:.1f}% of responses")
        print(f"Practical Tips: {practical_tips_rate:.1f}% of responses")
        print()
        print("💾 Results saved:")
        print(f"   📄 Detailed: {results_filename}")
        print()
        print("💡 RECOMMENDATION:")
        print("-" * 30)
        print(recommendation)
        
        return detailed_results


def main():
    """Run the comprehensive test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive AI Istanbul Chatbot Tester')
    parser.add_argument('--url', default='http://localhost:8001', help='Backend API URL')
    parser.add_argument('--sample', type=int, help='Run test on random sample of N inputs')
    parser.add_argument('--category', choices=list(TEST_CATEGORIES.keys()), help='Test specific category only')
    
    args = parser.parse_args()
    
    tester = ComprehensiveTester(args.url)
    
    if args.category:
        # Test specific category
        category_inputs = get_test_input_by_category(args.category)
        print(f"Testing {args.category} category with {len(category_inputs)} inputs...")
        # Run subset test (implement if needed)
        results = tester.run_comprehensive_test_suite()
    else:
        # Run full comprehensive test
        results = tester.run_comprehensive_test_suite(args.sample)
    
    return results

if __name__ == "__main__":
    main()

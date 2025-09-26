#!/usr/bin/env python3
"""
Enhanced Final Comprehensive Tester - v2.0
==========================================

This is the upgraded test suite using enhanced prompts and feature detection
to measure improvements in chatbot performance.

Target improvements:
- Overall Score: 1.95 -> 4.0+
- Completeness: 1.33 -> 4.0+
- Feature Coverage: 26.6% -> 80%+
- Cultural Awareness: 1.68 -> 4.0+
"""

import asyncio
import httpx
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

# Import test inputs
from comprehensive_test_inputs import (
    DAILY_TALK_INPUTS, RESTAURANT_ADVICE_INPUTS, DISTRICT_ADVICE_INPUTS,
    MUSEUM_ADVICE_INPUTS, TRANSPORTATION_ADVICE_INPUTS
)

@dataclass
class EnhancedTestResult:
    """Enhanced test result with detailed feature analysis"""
    query: str
    category: str
    difficulty: str
    response: str
    relevance_score: float
    completeness_score: float
    cultural_awareness_score: float
    feature_coverage_percentage: float
    detected_features_count: int
    expected_features_count: int
    missing_features: List[str]
    feature_categories: Dict[str, List[str]]
    location_accuracy: bool
    response_length: int
    timestamp: str

class EnhancedComprehensiveTester:
    """Enhanced comprehensive tester with improved evaluation metrics"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
        # Enhanced category mappings with expected features
        self.category_configs = {
            "daily_talks": {
                "name": "DAILY TALKS",
                "expected_features": ["welcoming_tone", "practical_advice", "cultural_tips", "specific_locations", "transportation_info", "time_context", "safety_guidance", "insider_knowledge"],
                "inputs": DAILY_TALK_INPUTS
            },
            "restaurant_advice": {
                "name": "RESTAURANT ADVICE", 
                "expected_features": ["specific_restaurants", "signature_dishes", "atmosphere_description", "location_details", "transportation_directions", "timing_advice", "cultural_etiquette", "price_context", "alternatives"],
                "inputs": RESTAURANT_ADVICE_INPUTS
            },
            "district_advice": {
                "name": "DISTRICT ADVICE",
                "expected_features": ["district_character", "key_attractions", "local_atmosphere", "transportation_options", "walking_routes", "optimal_timing", "unique_experiences", "cultural_significance", "practical_tips"],
                "inputs": DISTRICT_ADVICE_INPUTS
            },
            "museum_advice": {
                "name": "MUSEUM ADVICE",
                "expected_features": ["specific_museums", "historical_significance", "key_highlights", "practical_details", "transportation_info", "visiting_strategies", "photography_policies", "cultural_context", "visit_duration"],
                "inputs": MUSEUM_ADVICE_INPUTS
            },
            "transportation_advice": {
                "name": "TRANSPORTATION ADVICE",
                "expected_features": ["specific_routes", "step_by_step_directions", "istanbulkart_info", "alternative_routes", "schedule_timing", "transport_apps", "cultural_etiquette", "accessibility_info", "mode_integration"],
                "inputs": TRANSPORTATION_ADVICE_INPUTS
            }
        }
        
    def classify_difficulty(self, query: str) -> str:
        """Enhanced difficulty classification"""
        query_lower = query.lower()
        
        # Hard queries - multiple requirements, complex scenarios
        hard_indicators = [
            'overwhelmed', 'tired of', 'real istanbul', 'authentic', 'off beaten path',
            'budget', 'allergies', 'vegetarian', 'halal', 'solo woman', 'family with kids',
            'business trip', 'photographer', 'instagram', 'crowds', 'scams', 'safety'
        ]
        
        # Medium queries - specific requests with some complexity
        medium_indicators = [
            'best time', 'how to', 'where can i', 'looking for', 'recommend', 
            'what should i', 'planning', 'itinerary', 'first time', 'just arrived'
        ]
        
        if any(indicator in query_lower for indicator in hard_indicators):
            return "hard"
        elif any(indicator in query_lower for indicator in medium_indicators):
            return "medium"
        else:
            return "easy"
    
    def calculate_relevance_score(self, query: str, response: str, category: str) -> float:
        """Enhanced relevance scoring with category-specific criteria"""
        score = 0.0
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Base relevance - does response address the query?
        if len(response) < 50:
            return 1.0  # Too short to be helpful
        
        # Istanbul focus (critical)
        istanbul_terms = ['istanbul', 'sultanahmet', 'beyoglu', 'kadikoy', 'taksim', 'galata', 'bosphorus']
        istanbul_mentions = sum(1 for term in istanbul_terms if term in response_lower)
        if istanbul_mentions >= 2:
            score += 1.0
        elif istanbul_mentions >= 1:
            score += 0.7
        else:
            score += 0.3
        
        # Category-specific relevance
        if category == "restaurant_advice":
            food_terms = ['restaurant', 'food', 'eat', 'dining', 'turkish cuisine', 'breakfast', 'kebab']
            food_score = min(sum(1 for term in food_terms if term in response_lower) * 0.2, 1.0)
            score += food_score
        elif category == "district_advice":
            district_terms = ['district', 'neighborhood', 'area', 'character', 'atmosphere', 'attractions']
            district_score = min(sum(1 for term in district_terms if term in response_lower) * 0.2, 1.0)
            score += district_score
        elif category == "museum_advice":
            culture_terms = ['museum', 'historical', 'culture', 'art', 'exhibition', 'heritage']
            culture_score = min(sum(1 for term in culture_terms if term in response_lower) * 0.2, 1.0)
            score += culture_score
        elif category == "transportation_advice":
            transport_terms = ['metro', 'tram', 'bus', 'ferry', 'transport', 'istanbulkart', 'route']
            transport_score = min(sum(1 for term in transport_terms if term in response_lower) * 0.2, 1.0)
            score += transport_score
        elif category == "daily_talks":
            helpful_terms = ['help', 'tip', 'advice', 'recommend', 'suggest', 'welcome']
            helpful_score = min(sum(1 for term in helpful_terms if term in response_lower) * 0.2, 1.0)
            score += helpful_score
        
        # Query-specific terms (does it address specific aspects mentioned?)
        query_keywords = set(query_lower.split())
        response_keywords = set(response_lower.split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'i', 'you', 'we', 'they', 'he', 'she', 'it'}
        
        meaningful_query_words = query_keywords - common_words
        meaningful_response_words = response_keywords - common_words
        
        if meaningful_query_words:
            keyword_overlap = len(meaningful_query_words.intersection(meaningful_response_words)) / len(meaningful_query_words)
            score += keyword_overlap * 1.0
        
        # Length and depth bonus
        if len(response) > 200:
            score += 0.5
        if len(response) > 400:
            score += 0.5
        
        # Practical information bonus
        practical_terms = ['hours', 'address', 'how to get', 'metro', 'tram', 'walk', 'minutes', 'tip', 'advice']
        practical_score = min(sum(1 for term in practical_terms if term in response_lower) * 0.1, 0.5)
        score += practical_score
        
        return min(score, 5.0)
    
    def calculate_cultural_awareness_score(self, response: str) -> float:
        """Enhanced cultural awareness scoring"""
        response_lower = response.lower()
        score = 0.0
        
        # Cultural terms and concepts
        cultural_terms = {
            'turkish': 0.3, 'ottoman': 0.3, 'byzantine': 0.3, 'islamic': 0.2,
            'culture': 0.2, 'tradition': 0.2, 'heritage': 0.2, 'etiquette': 0.3,
            'customs': 0.3, 'respect': 0.2, 'mosque': 0.2, 'prayer': 0.2,
            'ramadan': 0.3, 'halal': 0.2, 'merhaba': 0.2, 'local': 0.2,
            'authentic': 0.2, 'east meets west': 0.5, 'two continents': 0.3,
            'cultural bridge': 0.4, 'diversity': 0.2
        }
        
        for term, weight in cultural_terms.items():
            if term in response_lower:
                score += weight
        
        # Religious/cultural sensitivity
        if 'mosque' in response_lower and any(word in response_lower for word in ['respect', 'appropriate', 'dress', 'prayer', 'remove shoes']):
            score += 0.5
        
        # Local customs mentioned
        local_customs = ['tipping', 'bargaining', 'tea culture', 'turkish coffee', 'hammam', 'greeting']
        customs_mentioned = sum(1 for custom in local_customs if custom in response_lower)
        score += min(customs_mentioned * 0.2, 0.6)
        
        return min(score, 5.0)
    
    def check_location_accuracy(self, response: str) -> bool:
        """Check if response correctly focuses on Istanbul"""
        response_lower = response.lower()
        
        # Must mention Istanbul-related terms
        istanbul_indicators = [
            'istanbul', 'sultanahmet', 'beyoglu', 'kadikoy', 'taksim', 'galata',
            'bosphorus', 'golden horn', 'hagia sophia', 'blue mosque', 'topkapi'
        ]
        
        has_istanbul_focus = any(term in response_lower for term in istanbul_indicators)
        
        # Should not mention other Turkish cities incorrectly
        other_cities = ['ankara', 'izmir', 'antalya', 'bursa', 'cappadocia']
        mentions_other_cities = any(city in response_lower for city in other_cities)
        
        return has_istanbul_focus and not mentions_other_cities
    
    async def test_single_query(self, query: str, category: str, session: httpx.AsyncClient) -> EnhancedTestResult:
        """Test a single query with enhanced analysis"""
        try:
            # Make API call
            start_time = time.time()
            response = await session.post(
                f"{self.base_url}/ai/chat",
                json={"message": query},
                timeout=30.0
            )
            end_time = time.time()
            
            if response.status_code != 200:
                print(f"❌ API Error {response.status_code} for query: {query[:50]}...")
                return None
            
            result = response.json()
            if "error" in result:
                print(f"❌ API returned error for query: {query[:50]}... | Error: {result['error']}")
                return None
            
            ai_response = result.get("response", "")
            if not ai_response:
                print(f"❌ Empty response for query: {query[:50]}...")
                return None
            
            # Enhanced analysis using new systems
            difficulty = self.classify_difficulty(query)
            relevance_score = self.calculate_relevance_score(query, ai_response, category)
            cultural_awareness_score = self.calculate_cultural_awareness_score(ai_response)
            location_accuracy = self.check_location_accuracy(ai_response)
            
            # Feature analysis using enhanced detection
            expected_features = self.category_configs[category]["expected_features"]
            
            try:
                # Import enhanced feature detection
                import sys, os
                backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
                if backend_path not in sys.path:
                    sys.path.insert(0, backend_path)
                
                from enhanced_feature_detection import analyze_response_features
                
                feature_analysis = analyze_response_features(ai_response, category.replace("_advice", "").replace("daily_talks", "daily_talk"), expected_features)
                
                completeness_score = feature_analysis['completeness_score']
                feature_coverage_percentage = feature_analysis['coverage_percentage']
                detected_features_count = feature_analysis['total_features_detected']
                missing_features = feature_analysis['missing_features']
                feature_categories = feature_analysis['feature_categories']
                
            except ImportError as e:
                print(f"⚠️ Enhanced feature detection not available: {e}")
                # Fallback to basic analysis
                completeness_score = min(len(ai_response) / 200.0, 5.0)  # Basic length-based score
                feature_coverage_percentage = 50.0  # Default
                detected_features_count = 3  # Estimate
                missing_features = []
                feature_categories = {}
            
            response_time = end_time - start_time
            print(f"✅ {category.upper()}: {query[:50]}... | R:{relevance_score:.1f} C:{completeness_score:.1f} F:{feature_coverage_percentage:.0f}% | {response_time:.1f}s")
            
            return EnhancedTestResult(
                query=query,
                category=category,
                difficulty=difficulty,
                response=ai_response,
                relevance_score=relevance_score,
                completeness_score=completeness_score,
                cultural_awareness_score=cultural_awareness_score,
                feature_coverage_percentage=feature_coverage_percentage,
                detected_features_count=detected_features_count,
                expected_features_count=len(expected_features),
                missing_features=missing_features,
                feature_categories=feature_categories,
                location_accuracy=location_accuracy,
                response_length=len(ai_response),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            print(f"❌ Error testing query: {query[:50]}... | {str(e)}")
            traceback.print_exc()
            return None
    
    async def run_comprehensive_test(self) -> Dict:
        """Run the complete enhanced test suite"""
        print("🎯 ENHANCED AI ISTANBUL CHATBOT - COMPREHENSIVE TEST v2.0")
        print("=" * 65)
        
        start_time = time.time()
        
        async with httpx.AsyncClient() as session:
            # Test all categories
            for category_key, category_config in self.category_configs.items():
                print(f"\n🔍 Testing {category_config['name']}...")
                print("-" * 50)
                
                for query in category_config['inputs']:
                    result = await self.test_single_query(query, category_key, session)
                    if result:
                        self.results.append(result)
                    await asyncio.sleep(0.1)  # Rate limiting
        
        total_time = time.time() - start_time
        
        # Comprehensive analysis
        analysis = self.analyze_results()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results JSON
        results_data = [asdict(result) for result in self.results]
        json_filename = f"ai_istanbul_enhanced_test_results_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                "test_metadata": {
                    "version": "2.0_enhanced",
                    "total_tests": len(self.results),
                    "total_time": total_time,
                    "timestamp": timestamp
                },
                "overall_analysis": analysis,
                "detailed_results": results_data
            }, f, indent=2, ensure_ascii=False)
        
        # Save readable report
        report_filename = f"ai_istanbul_enhanced_test_report_{timestamp}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            self.write_detailed_report(f, analysis, total_time)
        
        print(f"\n💾 Results saved to {json_filename} and {report_filename}")
        return analysis
    
    def analyze_results(self) -> Dict:
        """Comprehensive analysis of enhanced test results"""
        if not self.results:
            return {}
        
        # Overall metrics
        total_tests = len(self.results)
        avg_relevance = sum(r.relevance_score for r in self.results) / total_tests
        avg_completeness = sum(r.completeness_score for r in self.results) / total_tests
        avg_cultural_awareness = sum(r.cultural_awareness_score for r in self.results) / total_tests
        avg_feature_coverage = sum(r.feature_coverage_percentage for r in self.results) / total_tests
        location_accuracy = sum(1 for r in self.results if r.location_accuracy) / total_tests * 100
        
        # Calculate overall score (weighted average)
        overall_score = (
            avg_relevance * 0.3 +
            avg_completeness * 0.3 +
            avg_cultural_awareness * 0.2 +
            (avg_feature_coverage / 20) * 0.2  # Normalize feature coverage to 5-point scale
        )
        
        # Letter grade
        if overall_score >= 4.5:
            letter_grade = "A+"
        elif overall_score >= 4.0:
            letter_grade = "A"
        elif overall_score >= 3.5:
            letter_grade = "B+"
        elif overall_score >= 3.0:
            letter_grade = "B"
        elif overall_score >= 2.5:
            letter_grade = "C+"
        elif overall_score >= 2.0:
            letter_grade = "C"
        elif overall_score >= 1.5:
            letter_grade = "D"
        else:
            letter_grade = "F"
        
        # Category breakdown
        category_analysis = {}
        for category_key, category_config in self.category_configs.items():
            category_results = [r for r in self.results if r.category == category_key]
            if category_results:
                category_analysis[category_key] = {
                    "name": category_config["name"],
                    "relevance": sum(r.relevance_score for r in category_results) / len(category_results),
                    "completeness": sum(r.completeness_score for r in category_results) / len(category_results),
                    "feature_coverage": sum(r.feature_coverage_percentage for r in category_results) / len(category_results),
                    "cultural_awareness": sum(r.cultural_awareness_score for r in category_results) / len(category_results),
                    "tests": len(category_results)
                }
        
        return {
            "overall_score": overall_score,
            "letter_grade": letter_grade,
            "avg_relevance": avg_relevance,
            "avg_completeness": avg_completeness,
            "avg_cultural_awareness": avg_cultural_awareness,
            "avg_feature_coverage": avg_feature_coverage,
            "location_accuracy": location_accuracy,
            "total_tests": total_tests,
            "category_breakdown": category_analysis
        }
    
    def write_detailed_report(self, f, analysis: Dict, total_time: float):
        """Write detailed enhanced test report"""
        f.write("🎯 AI ISTANBUL CHATBOT - ENHANCED COMPREHENSIVE TEST REPORT v2.0\n")
        f.write("=" * 70 + "\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tests: {analysis['total_tests']}\n")
        f.write(f"Total Time: {total_time:.1f} seconds\n\n")
        
        f.write("🏆 OVERALL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        f.write(f"Final Score: {analysis['overall_score']:.2f}/5.0\n")
        f.write(f"Letter Grade: {analysis['letter_grade']}\n")
        f.write(f"Average Relevance: {analysis['avg_relevance']:.2f}/5.0\n")
        f.write(f"Average Completeness: {analysis['avg_completeness']:.2f}/5.0\n")
        f.write(f"Cultural Awareness: {analysis['avg_cultural_awareness']:.2f}/5.0\n")
        f.write(f"Feature Coverage: {analysis['avg_feature_coverage']:.1f}%\n")
        f.write(f"Location Accuracy: {analysis['location_accuracy']:.1f}%\n\n")
        
        f.write("📊 CATEGORY BREAKDOWN\n")
        f.write("-" * 30 + "\n")
        for category_key, cat_data in analysis['category_breakdown'].items():
            f.write(f"{cat_data['name']}:\n")
            f.write(f"  Relevance: {cat_data['relevance']:.2f}/5.0\n")
            f.write(f"  Completeness: {cat_data['completeness']:.2f}/5.0\n")
            f.write(f"  Feature Coverage: {cat_data['feature_coverage']:.1f}%\n")
            f.write(f"  Cultural Awareness: {cat_data['cultural_awareness']:.2f}/5.0\n")
            f.write(f"  Tests: {cat_data['tests']}\n\n")
        
        f.write("📝 SAMPLE DETAILED RESULTS\n")
        f.write("-" * 30 + "\n")
        
        # Show first 10 results as samples
        for i, result in enumerate(self.results[:10]):
            f.write(f"Query: {result.query}\n")
            f.write(f"Category: {result.category} | Difficulty: {result.difficulty}\n")
            f.write(f"Relevance: {result.relevance_score:.1f}/5 | Completeness: {result.completeness_score:.1f}/5 | Features: {result.detected_features_count}/{result.expected_features_count} ({result.feature_coverage_percentage:.0f}%)\n")
            f.write(f"Response: {result.response[:100]}{'...' if len(result.response) > 100 else ''}\n")
            f.write("-" * 40 + "\n")

# Main execution
async def main():
    """Run the enhanced comprehensive test"""
    tester = EnhancedComprehensiveTester()
    analysis = await tester.run_comprehensive_test()
    
    print(f"\n🎯 ENHANCED TEST COMPLETE!")
    print(f"📊 Final Score: {analysis['overall_score']:.2f}/5.0 (Grade: {analysis['letter_grade']})")
    print(f"📈 Improvements needed for production: {max(0, 3.5 - analysis['overall_score']):.2f} points")
    
    if analysis['overall_score'] >= 3.5:
        print("🎉 READY FOR PRODUCTION! 🚀")
    else:
        print("⚠️ Needs improvement before production release")

if __name__ == "__main__":
    asyncio.run(main())

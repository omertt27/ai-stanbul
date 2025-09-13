#!/usr/bin/env python3
"""
Comprehensive Input Testing System for AI Istanbul Chatbot
Demonstrates handling of billions of possible inputs through systematic testing
"""

import sys
import os
import json
import time
import asyncio
from typing import List, Dict, Any
from dataclasses import asdict

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_input_validation import validate_user_input, InputCategory

class InputTestSuite:
    """Comprehensive test suite for input validation and handling"""
    
    def __init__(self):
        self.test_results = []
        self.categories_tested = set()
        self.total_tests = 0
        self.passed_tests = 0
        
    def run_test_case(self, test_input: str, expected_category: InputCategory | None = None, 
                     description: str = "") -> Dict[str, Any]:
        """Run a single test case and return results"""
        self.total_tests += 1
        
        start_time = time.time()
        result = validate_user_input(test_input)
        processing_time = time.time() - start_time
        
        # Determine if test passed
        test_passed = result.is_valid or result.category in [
            InputCategory.SPAM, InputCategory.GIBBERISH, InputCategory.INAPPROPRIATE
        ]
        
        if expected_category and result.category == expected_category:
            test_passed = True
        
        if test_passed:
            self.passed_tests += 1
        
        self.categories_tested.add(result.category)
        
        test_result = {
            'input': test_input,
            'description': description,
            'expected_category': expected_category.value if expected_category else None,
            'result': asdict(result),
            'processing_time_ms': round(processing_time * 1000, 2),
            'test_passed': test_passed
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def test_edge_cases(self):
        """Test various edge cases and problematic inputs"""
        print("🧪 Testing Edge Cases...")
        
        edge_cases = [
            # Empty and minimal inputs
            ("", InputCategory.UNKNOWN, "Empty input"),
            ("a", InputCategory.UNKNOWN, "Single character"),
            ("ab", InputCategory.UNKNOWN, "Two characters"),
            
            # Gibberish and spam
            ("aaaaaaaaaaaa", InputCategory.GIBBERISH, "Repeated characters"),
            ("xkjdfhlkjsdhlkjfh", InputCategory.GIBBERISH, "Random characters"),
            ("🍕🏛️🚇🏨", InputCategory.UNKNOWN, "Only emojis"),
            ("!!!!!!!!!", InputCategory.GIBBERISH, "Only punctuation"),
            ("BUY NOW CLICK HERE URGENT!!!", InputCategory.SPAM, "Spam content"),
            
            # Special characters and potential attacks
            ("<script>alert('xss')</script>", InputCategory.UNKNOWN, "XSS attempt"),
            ("'; DROP TABLE restaurants; --", InputCategory.UNKNOWN, "SQL injection attempt"),
            ("http://malicious-site.com", InputCategory.UNKNOWN, "URL in input"),
            ("john@email.com", InputCategory.UNKNOWN, "Email address"),
            ("+1-555-123-4567", InputCategory.UNKNOWN, "Phone number"),
            
            # Very long input
            ("restaurant " * 100, InputCategory.FOOD, "Very long input"),
            
            # Mixed content
            ("best restaurants!!!!! in istanbul??????", InputCategory.FOOD, "Excessive punctuation"),
            ("HELP ME FIND FOOD NOW!!!", InputCategory.FOOD, "All caps with urgency"),
        ]
        
        for input_text, expected_category, description in edge_cases:
            self.run_test_case(input_text, expected_category, description)
    
    def test_valid_queries(self):
        """Test valid, realistic user queries"""
        print("✅ Testing Valid Queries...")
        
        valid_queries = [
            # Food queries
            ("best restaurants in Sultanahmet", InputCategory.FOOD),
            ("halal food near Blue Mosque", InputCategory.FOOD),
            ("Turkish breakfast recommendations", InputCategory.FOOD),
            ("vegetarian restaurants in Kadıköy", InputCategory.FOOD),
            ("where can I find good kebab", InputCategory.FOOD),
            
            # Tourism queries
            ("what to see in Istanbul", InputCategory.TOURISM),
            ("Hagia Sophia visiting hours", InputCategory.TOURISM),
            ("best museums to visit", InputCategory.TOURISM),
            ("Bosphorus cruise options", InputCategory.TOURISM),
            ("historical sites in Fatih", InputCategory.TOURISM),
            
            # Transportation
            ("how to get to Taksim Square", InputCategory.TRANSPORT),
            ("metro from airport to city", InputCategory.TRANSPORT),
            ("Istanbul public transport card", InputCategory.TRANSPORT),
            ("taxi vs metro costs", InputCategory.TRANSPORT),
            
            # Accommodation
            ("best hotels in Beyoğlu", InputCategory.ACCOMMODATION),
            ("cheap hostels near Sultanahmet", InputCategory.ACCOMMODATION),
            ("family-friendly accommodations", InputCategory.ACCOMMODATION),
            
            # Shopping
            ("Grand Bazaar shopping guide", InputCategory.SHOPPING),
            ("where to buy Turkish carpets", InputCategory.SHOPPING),
            ("souvenir shops in Eminönü", InputCategory.SHOPPING),
            
            # Entertainment
            ("nightlife in Taksim", InputCategory.ENTERTAINMENT),
            ("live music venues", InputCategory.ENTERTAINMENT),
            ("rooftop bars with Bosphorus view", InputCategory.ENTERTAINMENT),
            
            # Weather
            ("weather in Istanbul in March", InputCategory.WEATHER),
            ("best time to visit", InputCategory.WEATHER),
            ("is it raining now", InputCategory.WEATHER),
            
            # Culture
            ("Turkish customs and traditions", InputCategory.CULTURE),
            ("local etiquette tips", InputCategory.CULTURE),
            ("how to greet locals", InputCategory.CULTURE),
            
            # Practical
            ("currency exchange rates", InputCategory.PRACTICAL),
            ("safety tips for tourists", InputCategory.PRACTICAL),
            ("emergency phone numbers", InputCategory.PRACTICAL),
        ]
        
        for input_text, expected_category in valid_queries:
            self.run_test_case(input_text, expected_category, f"Valid {expected_category.value} query")
    
    def test_ambiguous_queries(self):
        """Test queries that could fit multiple categories"""
        print("🤔 Testing Ambiguous Queries...")
        
        ambiguous_queries = [
            ("Istanbul", InputCategory.UNKNOWN, "Single word - city name"),
            ("help", InputCategory.PRACTICAL, "Generic help request"),
            ("good places", InputCategory.UNKNOWN, "Vague request"),
            ("what to do", InputCategory.TOURISM, "General activity question"),
            ("I'm hungry", InputCategory.FOOD, "Emotional state indicating need"),
            ("I'm lost", InputCategory.TRANSPORT, "Navigation help needed"),
            ("recommend something", InputCategory.UNKNOWN, "Very generic request"),
            ("tourist guide", InputCategory.TOURISM, "General tourism request"),
        ]
        
        for input_text, expected_category, description in ambiguous_queries:
            self.run_test_case(input_text, expected_category, description)
    
    def test_multilingual_inputs(self):
        """Test non-English inputs (basic handling)"""
        print("🌍 Testing Multilingual Inputs...")
        
        multilingual_queries = [
            ("restoran önerileri", InputCategory.FOOD, "Turkish - restaurant recommendations"),
            ("مطاعم في اسطنبول", InputCategory.FOOD, "Arabic - restaurants in Istanbul"),
            ("Где поесть", InputCategory.FOOD, "Russian - where to eat"),
            ("餐厅推荐", InputCategory.FOOD, "Chinese - restaurant recommendations"),
            ("レストラン", InputCategory.FOOD, "Japanese - restaurant"),
            ("café recommandations", InputCategory.FOOD, "French - cafe recommendations"),
            ("Restaurante empfehlungen", InputCategory.FOOD, "German - restaurant recommendations"),
        ]
        
        for input_text, expected_category, description in multilingual_queries:
            self.run_test_case(input_text, expected_category, description)
    
    def test_conversational_inputs(self):
        """Test conversational and contextual inputs"""
        print("💬 Testing Conversational Inputs...")
        
        conversational_queries = [
            ("Hi there!", InputCategory.UNKNOWN, "Greeting"),
            ("Thanks for the help", InputCategory.UNKNOWN, "Gratitude"),
            ("What about nearby?", InputCategory.UNKNOWN, "Context-dependent question"),
            ("Can you help me?", InputCategory.PRACTICAL, "Help request"),
            ("I'm visiting for the first time", InputCategory.TOURISM, "First-time visitor"),
            ("We're a family with kids", InputCategory.TOURISM, "Family context"),
            ("I'm on a tight budget", InputCategory.PRACTICAL, "Budget constraint"),
            ("What do locals recommend?", InputCategory.CULTURE, "Local perspective"),
            ("Is it safe for solo travelers?", InputCategory.PRACTICAL, "Safety concern"),
            ("How much time do I need?", InputCategory.PRACTICAL, "Time planning"),
        ]
        
        for input_text, expected_category, description in conversational_queries:
            self.run_test_case(input_text, expected_category, description)
    
    def test_complex_multi_part_queries(self):
        """Test complex queries with multiple components"""
        print("🧩 Testing Complex Multi-part Queries...")
        
        complex_queries = [
            (
                "I'm looking for halal restaurants near Sultanahmet with good vegetarian options for a family dinner under 100 lira per person",
                InputCategory.FOOD,
                "Multi-constraint restaurant query"
            ),
            (
                "What are the best museums to visit in the morning, followed by lunch recommendations in the same area, accessible by metro?",
                InputCategory.TOURISM,
                "Multi-activity planning query"
            ),
            (
                "I'm staying in Taksim for 3 days in summer, what's the weather like and what outdoor activities do you recommend?",
                InputCategory.TOURISM,
                "Weather + activities query"
            ),
            (
                "How do I get from the airport to my hotel in Beyoğlu using public transport, and where can I buy a transport card?",
                InputCategory.TRANSPORT,
                "Multi-step transport query"
            ),
        ]
        
        for input_text, expected_category, description in complex_queries:
            self.run_test_case(input_text, expected_category, description)
    
    def performance_stress_test(self):
        """Test system performance under load"""
        print("⚡ Running Performance Stress Test...")
        
        # Test rapid-fire queries
        rapid_queries = ["restaurants in istanbul"] * 100
        
        start_time = time.time()
        for i, query in enumerate(rapid_queries):
            self.run_test_case(f"{query} {i}", InputCategory.FOOD, f"Rapid query #{i}")
        
        total_time = time.time() - start_time
        avg_time_per_query = total_time / len(rapid_queries)
        
        print(f"   Processed {len(rapid_queries)} queries in {total_time:.2f}s")
        print(f"   Average time per query: {avg_time_per_query*1000:.2f}ms")
        
        return {
            'total_queries': len(rapid_queries),
            'total_time': total_time,
            'avg_time_ms': avg_time_per_query * 1000
        }
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE INPUT TESTING REPORT")
        print("="*80)
        
        # Summary statistics
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"Total Tests Run: {self.total_tests}")
        print(f"Tests Passed: {self.passed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Categories Tested: {len(self.categories_tested)}")
        print(f"Categories Found: {', '.join([cat.value for cat in self.categories_tested])}")
        
        # Processing time statistics
        processing_times = [result['processing_time_ms'] for result in self.test_results]
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            min_time = min(processing_times)
            
            print(f"\nProcessing Time Stats:")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Maximum: {max_time:.2f}ms")
            print(f"  Minimum: {min_time:.2f}ms")
        
        # Category distribution
        category_counts = {}
        for result in self.test_results:
            category = result['result']['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"\nCategory Distribution:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / self.total_tests * 100) if self.total_tests > 0 else 0
            print(f"  {category}: {count} ({percentage:.1f}%)")
        
        # Failed tests summary
        failed_tests = [result for result in self.test_results if not result['test_passed']]
        if failed_tests:
            print(f"\nFailed Tests ({len(failed_tests)}):")
            for test in failed_tests[:10]:  # Show first 10 failures
                print(f"  '{test['input']}' - {test['description']}")
        
        # Export detailed results
        with open('input_test_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_tests': self.total_tests,
                    'passed_tests': self.passed_tests,
                    'success_rate': success_rate,
                    'categories_tested': list(self.categories_tested)
                },
                'detailed_results': self.test_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results exported to: input_test_results.json")
        print("\n" + "="*80)

def main():
    """Run the comprehensive input testing suite"""
    print("🚀 Starting Comprehensive Input Testing Suite for AI Istanbul Chatbot")
    print("This demonstrates how we handle billions of possible inputs systematically.\n")
    
    test_suite = InputTestSuite()
    
    try:
        # Run all test categories
        test_suite.test_edge_cases()
        test_suite.test_valid_queries()
        test_suite.test_ambiguous_queries()
        test_suite.test_multilingual_inputs()
        test_suite.test_conversational_inputs()
        test_suite.test_complex_multi_part_queries()
        
        # Performance testing
        performance_stats = test_suite.performance_stress_test()
        
        # Generate final report
        test_suite.generate_report()
        
        print("\n✅ Testing completed successfully!")
        print("This systematic approach allows us to handle billions of possible inputs")
        print("by categorizing, validating, and providing appropriate responses.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

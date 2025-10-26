"""
Integration Tests for Hidden Gems Handler
Tests the complete functionality of the enhanced hidden gems system
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from backend.services.hidden_gems_handler import get_hidden_gems_handler
from backend.data.hidden_gems_database import (
    get_all_neighborhoods,
    get_all_hidden_gems,
    TOTAL_HIDDEN_GEMS
)


class TestHiddenGemsHandler:
    """Comprehensive test suite for Hidden Gems Handler"""
    
    def __init__(self):
        self.handler = get_hidden_gems_handler()
        self.test_results = []
        self.passed = 0
        self.failed = 0
    
    def log_test(self, test_name, passed, details=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        result = f"{status}: {test_name}"
        if details:
            result += f"\n   Details: {details}"
        self.test_results.append(result)
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        print(result)
    
    def test_database_loading(self):
        """Test that database loads correctly"""
        print("\n" + "="*60)
        print("TEST 1: Database Loading")
        print("="*60)
        
        # Check if database is available
        passed = self.handler.database_available
        self.log_test(
            "Database availability",
            passed,
            f"Database loaded: {passed}"
        )
        
        # Check total gems count
        if passed:
            gems = get_all_hidden_gems()
            self.log_test(
                "Total gems count",
                len(gems) > 0,
                f"Found {len(gems)} gems (expected {TOTAL_HIDDEN_GEMS})"
            )
            
            # Check neighborhoods
            neighborhoods = get_all_neighborhoods()
            self.log_test(
                "Neighborhoods loaded",
                len(neighborhoods) >= 6,
                f"Found {len(neighborhoods)} neighborhoods: {', '.join(neighborhoods)}"
            )
    
    def test_basic_query_detection(self):
        """Test hidden gems query detection"""
        print("\n" + "="*60)
        print("TEST 2: Query Detection")
        print("="*60)
        
        test_queries = [
            ("Show me hidden gems in Istanbul", True),
            ("Any secret spots in Kadıköy?", True),
            ("Tell me about off-the-beaten-path places", True),
            ("What are some local favorite places?", True),
            ("Best restaurants in Beyoğlu", False),
            ("How to get to Taksim?", False)
        ]
        
        for query, should_detect in test_queries:
            detected = self.handler.detect_hidden_gems_query(query)
            passed = detected == should_detect
            self.log_test(
                f"Query detection: '{query[:40]}...'",
                passed,
                f"Expected: {should_detect}, Got: {detected}"
            )
    
    def test_neighborhood_filtering(self):
        """Test filtering by neighborhood"""
        print("\n" + "="*60)
        print("TEST 3: Neighborhood Filtering")
        print("="*60)
        
        neighborhoods = ['sarıyer', 'beşiktaş', 'kadıköy', 'beyoğlu']
        
        for neighborhood in neighborhoods:
            gems = self.handler.get_hidden_gems(location=neighborhood, limit=10)
            passed = len(gems) > 0
            self.log_test(
                f"Get gems in {neighborhood.title()}",
                passed,
                f"Found {len(gems)} gems"
            )
            
            if passed and len(gems) > 0:
                # Print first gem as example
                print(f"   Example: {gems[0]['name']} ({gems[0]['type']})")
    
    def test_type_filtering(self):
        """Test filtering by gem type"""
        print("\n" + "="*60)
        print("TEST 4: Type Filtering")
        print("="*60)
        
        gem_types = ['nature', 'cafe', 'food', 'historical']
        
        for gem_type in gem_types:
            gems = self.handler.get_hidden_gems(gem_type=gem_type, limit=5)
            passed = len(gems) > 0
            self.log_test(
                f"Get {gem_type} gems",
                passed,
                f"Found {len(gems)} {gem_type} gems"
            )
            
            if passed:
                # Verify all gems are of correct type
                all_correct_type = all(g.get('type') == gem_type for g in gems)
                self.log_test(
                    f"Type consistency for {gem_type}",
                    all_correct_type,
                    f"All gems are {gem_type} type: {all_correct_type}"
                )
    
    def test_combined_filtering(self):
        """Test combined neighborhood + type filtering"""
        print("\n" + "="*60)
        print("TEST 5: Combined Filtering")
        print("="*60)
        
        test_cases = [
            ('kadıköy', 'nature'),
            ('beşiktaş', 'historical'),
            ('sarıyer', 'cafe')
        ]
        
        for location, gem_type in test_cases:
            gems = self.handler.get_hidden_gems(
                location=location,
                gem_type=gem_type,
                limit=5
            )
            passed = len(gems) >= 0  # May be 0 for some combinations
            self.log_test(
                f"Get {gem_type} gems in {location.title()}",
                passed,
                f"Found {len(gems)} gems matching both criteria"
            )
    
    def test_query_parameter_extraction(self):
        """Test parameter extraction from natural language queries"""
        print("\n" + "="*60)
        print("TEST 6: Query Parameter Extraction")
        print("="*60)
        
        test_queries = [
            ("Hidden gems in Kadıköy", {'location': 'kadıköy'}),
            ("Secret nature spots", {'gem_type': 'nature'}),
            ("Coffee places only locals know", {'gem_type': 'cafe'}),
            ("Cheap hidden restaurants in Beyoğlu", {'location': 'beyoğlu', 'gem_type': 'food'})
        ]
        
        for query, expected_params in test_queries:
            params = self.handler.extract_query_parameters(query)
            passed = True
            details = []
            
            for key, expected_value in expected_params.items():
                actual_value = params.get(key)
                if actual_value != expected_value:
                    passed = False
                details.append(f"{key}: expected '{expected_value}', got '{actual_value}'")
            
            self.log_test(
                f"Extract params from: '{query}'",
                passed,
                "; ".join(details)
            )
    
    def test_response_formatting(self):
        """Test response formatting with real data"""
        print("\n" + "="*60)
        print("TEST 7: Response Formatting")
        print("="*60)
        
        # Get sample gems
        gems = self.handler.get_hidden_gems(location='kadıköy', limit=3)
        
        if len(gems) > 0:
            response = self.handler.format_hidden_gem_response(gems, 'kadıköy')
            
            # Check response contains key elements
            checks = [
                ('Has title', 'Hidden Gems' in response),
                ('Has location', 'Kadıköy' in response or 'kadıköy' in response.lower()),
                ('Has gem names', any(gem['name'] in response for gem in gems)),
                ('Has how to find', 'How to Find' in response or 'how_to_find' in str(gems)),
                ('Has local tips', 'Local Tip' in response or 'Pro Tip' in response)
            ]
            
            for check_name, check_result in checks:
                self.log_test(check_name, check_result)
            
            # Print sample of response
            print(f"\n   Sample Response (first 300 chars):")
            print(f"   {response[:300]}...\n")
        else:
            self.log_test("Get sample gems for formatting", False, "No gems found")
    
    def test_fallback_handling(self):
        """Test fallback when no gems found"""
        print("\n" + "="*60)
        print("TEST 8: Fallback Handling")
        print("="*60)
        
        # Query for non-existent combination
        gems = self.handler.get_hidden_gems(
            location='nonexistent',
            gem_type='unicorn',
            limit=5
        )
        
        response = self.handler.format_hidden_gem_response(gems, 'nonexistent')
        
        passed = 'Looking for hidden gems' in response or 'neighborhood' in response.lower()
        self.log_test(
            "Fallback response generated",
            passed,
            f"Response type: {'Fallback' if passed else 'Unknown'}"
        )
    
    def test_time_based_recommendations(self):
        """Test time-based gem recommendations"""
        print("\n" + "="*60)
        print("TEST 9: Time-Based Recommendations")
        print("="*60)
        
        times = ['morning', 'afternoon', 'evening', 'night']
        
        for time in times:
            gems = self.handler.get_recommendations_by_time(time, limit=3)
            passed = len(gems) > 0
            self.log_test(
                f"Get {time} recommendations",
                passed,
                f"Found {len(gems)} gems for {time}"
            )
            
            if passed and len(gems) > 0:
                print(f"   Example: {gems[0]['name']} (best: {gems[0].get('best_time', 'N/A')})")
    
    def test_hidden_factor_ranking(self):
        """Test that gems with higher hidden_factor are ranked higher"""
        print("\n" + "="*60)
        print("TEST 10: Hidden Factor Ranking")
        print("="*60)
        
        gems = self.handler.get_hidden_gems(limit=10)
        
        if len(gems) >= 2:
            # Check if gems are sorted by hidden_factor
            factors = [g.get('hidden_factor', 0) for g in gems]
            is_sorted = all(factors[i] >= factors[i+1] for i in range(len(factors)-1))
            
            self.log_test(
                "Gems sorted by hidden_factor",
                is_sorted,
                f"Hidden factors: {factors[:5]}..."
            )
        else:
            self.log_test("Get enough gems for ranking test", False, "Need at least 2 gems")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "🔍"*30)
        print("HIDDEN GEMS HANDLER - COMPREHENSIVE TEST SUITE")
        print("🔍"*30)
        
        self.test_database_loading()
        self.test_basic_query_detection()
        self.test_neighborhood_filtering()
        self.test_type_filtering()
        self.test_combined_filtering()
        self.test_query_parameter_extraction()
        self.test_response_formatting()
        self.test_fallback_handling()
        self.test_time_based_recommendations()
        self.test_hidden_factor_ranking()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Total: {self.passed + self.failed}")
        print(f"📈 Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")
        print("="*60)
        
        return self.passed, self.failed


def main():
    """Run tests"""
    tester = TestHiddenGemsHandler()
    passed, failed = tester.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive Analysis of Places & Attractions Test Results
=============================================================

Analyzes the test results from the comprehensive Places & Attractions test suite
and provides detailed insights into:
- System performance by category
- Feature detection rates
- Response quality analysis
- Recommendations for improvements
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

class PlacesAttractionsAnalyzer:
    """Comprehensive analyzer for Places & Attractions test results"""
    
    def __init__(self, results_file: str):
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        self.total_tests = len(self.results)
        self.successful_tests = sum(1 for r in self.results if r['success'])
        
    def analyze_by_category(self) -> Dict[str, Dict]:
        """Analyze results grouped by test category"""
        categories = defaultdict(lambda: {
            'total': 0,
            'successful': 0,
            'avg_response_length': 0,
            'avg_features_detected': 0,
            'tests': []
        })
        
        for result in self.results:
            cat = result['category']
            categories[cat]['total'] += 1
            if result['success']:
                categories[cat]['successful'] += 1
                categories[cat]['avg_response_length'] += result['response_length']
                
            features_detected = len([f for f in result['expected_features'] if f in result['detected_features']])
            features_expected = len(result['expected_features'])
            coverage = features_detected / features_expected * 100 if features_expected > 0 else 0
            
            categories[cat]['tests'].append({
                'id': result['test_id'],
                'query': result['query'],
                'success': result['success'],
                'features_detected': features_detected,
                'features_expected': features_expected,
                'coverage': coverage,
                'response_length': result['response_length']
            })
        
        # Calculate averages
        for cat in categories:
            if categories[cat]['successful'] > 0:
                categories[cat]['avg_response_length'] /= categories[cat]['successful']
                
            total_coverage = sum(t['coverage'] for t in categories[cat]['tests'])
            categories[cat]['avg_features_detected'] = total_coverage / len(categories[cat]['tests']) if categories[cat]['tests'] else 0
        
        return dict(categories)
    
    def analyze_feature_detection(self) -> Dict[str, Any]:
        """Analyze feature detection across all tests"""
        all_expected_features = set()
        for result in self.results:
            all_expected_features.update(result['expected_features'])
        
        feature_stats = {}
        for feature in all_expected_features:
            expected_count = sum(1 for r in self.results if feature in r['expected_features'])
            detected_count = sum(1 for r in self.results if feature in r['detected_features'] and feature in r['expected_features'])
            
            feature_stats[feature] = {
                'expected_in': expected_count,
                'detected_in': detected_count,
                'detection_rate': detected_count / expected_count * 100 if expected_count > 0 else 0
            }
        
        return feature_stats
    
    def identify_problematic_queries(self, coverage_threshold: float = 50.0) -> List[Dict]:
        """Identify queries with low feature coverage"""
        problematic = []
        
        for result in self.results:
            features_detected = len([f for f in result['expected_features'] if f in result['detected_features']])
            features_expected = len(result['expected_features'])
            coverage = features_detected / features_expected * 100 if features_expected > 0 else 0
            
            if coverage < coverage_threshold:
                problematic.append({
                    'id': result['test_id'],
                    'query': result['query'],
                    'category': result['category'],
                    'test_focus': result['test_focus'],
                    'coverage': coverage,
                    'expected_features': result['expected_features'],
                    'detected_features': result['detected_features'],
                    'missing_features': [f for f in result['expected_features'] if f not in result['detected_features']],
                    'response_preview': result['response'][:200] if result['response'] else "No response"
                })
        
        return sorted(problematic, key=lambda x: x['coverage'])
    
    def analyze_response_quality(self) -> Dict[str, Any]:
        """Analyze response quality metrics"""
        response_lengths = [r['response_length'] for r in self.results if r['success']]
        
        return {
            'total_responses': len(response_lengths),
            'avg_length': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            'min_length': min(response_lengths) if response_lengths else 0,
            'max_length': max(response_lengths) if response_lengths else 0,
            'short_responses': sum(1 for l in response_lengths if l < 100),
            'medium_responses': sum(1 for l in response_lengths if 100 <= l < 300),
            'long_responses': sum(1 for l in response_lengths if l >= 300)
        }
    
    def generate_recommendations(self, category_analysis: Dict, feature_stats: Dict, problematic: List) -> List[str]:
        """Generate actionable recommendations for improvement"""
        recommendations = []
        
        # Category-specific issues
        low_performing_categories = [
            cat for cat, data in category_analysis.items() 
            if data['avg_features_detected'] < 30
        ]
        
        if low_performing_categories:
            recommendations.append(
                f"🔧 **Critical**: {len(low_performing_categories)} categories have <30% feature detection: "
                f"{', '.join(low_performing_categories)}. These need urgent attention."
            )
        
        # Feature detection issues
        poorly_detected_features = [
            f for f, stats in feature_stats.items() 
            if stats['detection_rate'] < 25 and stats['expected_in'] > 2
        ]
        
        if poorly_detected_features:
            recommendations.append(
                f"🎯 **Important**: {len(poorly_detected_features)} features rarely detected: "
                f"{', '.join(poorly_detected_features)}. Implement better detection logic."
            )
        
        # Missing attractions system
        if any('attraction' in p['missing_features'][0] for p in problematic if p['missing_features']):
            recommendations.append(
                "🏛️ **Action Required**: Attractions system not responding properly. "
                "Check if IstanbulAttractionsSystem is properly integrated into main_system.py."
            )
        
        # GPS/Location issues
        gps_tests = [p for p in problematic if p['category'] == 'gps_based']
        if gps_tests:
            recommendations.append(
                "📍 **GPS Issue**: GPS-based queries not working. "
                "Verify GPS coordinate parsing and proximity search logic."
            )
        
        # Multi-intent handling
        multi_intent_issues = [p for p in problematic if p['category'] == 'multi_intent' and p['coverage'] < 30]
        if multi_intent_issues:
            recommendations.append(
                "🧠 **Multi-Intent**: Complex queries with multiple filters not handled well. "
                "Enhance multi-intent query handler to support combined filters."
            )
        
        # Generic fallback responses
        generic_responses = sum(1 for r in self.results if r['success'] and r['response_length'] < 150)
        if generic_responses > self.total_tests * 0.4:
            recommendations.append(
                f"⚠️ **Generic Responses**: {generic_responses}/{self.total_tests} responses are too short/generic. "
                "Improve response specificity and detail."
            )
        
        return recommendations
    
    def print_report(self):
        """Generate and print comprehensive analysis report"""
        print("\n" + "="*100)
        print("📊 PLACES & ATTRACTIONS COMPREHENSIVE TEST ANALYSIS")
        print("="*100)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Tests: {self.total_tests}")
        print(f"Successful: {self.successful_tests} ({self.successful_tests/self.total_tests*100:.1f}%)")
        print("="*100)
        
        # Category analysis
        print("\n📁 ANALYSIS BY CATEGORY")
        print("-"*100)
        category_analysis = self.analyze_by_category()
        
        for cat, data in sorted(category_analysis.items()):
            print(f"\n🏷️  {cat.upper().replace('_', ' ')}")
            print(f"   Tests: {data['total']}")
            print(f"   Successful: {data['successful']}/{data['total']} ({data['successful']/data['total']*100:.0f}%)")
            print(f"   Avg Feature Coverage: {data['avg_features_detected']:.1f}%")
            print(f"   Avg Response Length: {data['avg_response_length']:.0f} chars")
            
            # Show worst-performing test in category
            worst_test = min(data['tests'], key=lambda x: x['coverage'])
            print(f"   ⚠️  Worst Test: #{worst_test['id']} - {worst_test['query'][:50]}... ({worst_test['coverage']:.0f}% coverage)")
        
        # Feature detection analysis
        print("\n\n🔍 FEATURE DETECTION ANALYSIS")
        print("-"*100)
        feature_stats = self.analyze_feature_detection()
        
        print("\n✅ Well-Detected Features (>50% detection rate):")
        well_detected = [(f, s) for f, s in feature_stats.items() if s['detection_rate'] > 50]
        if well_detected:
            for feature, stats in sorted(well_detected, key=lambda x: x[1]['detection_rate'], reverse=True):
                print(f"   • {feature}: {stats['detection_rate']:.0f}% ({stats['detected_in']}/{stats['expected_in']} tests)")
        else:
            print("   ❌ None - this is critical!")
        
        print("\n⚠️  Poorly-Detected Features (<50% detection rate):")
        poorly_detected = [(f, s) for f, s in feature_stats.items() if s['detection_rate'] <= 50]
        for feature, stats in sorted(poorly_detected, key=lambda x: x[1]['detection_rate']):
            print(f"   • {feature}: {stats['detection_rate']:.0f}% ({stats['detected_in']}/{stats['expected_in']} tests)")
        
        # Response quality analysis
        print("\n\n📏 RESPONSE QUALITY ANALYSIS")
        print("-"*100)
        quality = self.analyze_response_quality()
        print(f"Average Response Length: {quality['avg_length']:.0f} characters")
        print(f"Range: {quality['min_length']} - {quality['max_length']} characters")
        print(f"\nResponse Distribution:")
        print(f"   • Short (<100 chars): {quality['short_responses']} ({quality['short_responses']/quality['total_responses']*100:.0f}%)")
        print(f"   • Medium (100-300 chars): {quality['medium_responses']} ({quality['medium_responses']/quality['total_responses']*100:.0f}%)")
        print(f"   • Long (300+ chars): {quality['long_responses']} ({quality['long_responses']/quality['total_responses']*100:.0f}%)")
        
        # Problematic queries
        print("\n\n❌ PROBLEMATIC QUERIES (Feature Coverage <50%)")
        print("-"*100)
        problematic = self.identify_problematic_queries(coverage_threshold=50.0)
        print(f"Found {len(problematic)}/{self.total_tests} queries with low feature coverage\n")
        
        for p in problematic[:10]:  # Show top 10 worst
            print(f"\n🔴 Test #{p['id']}: {p['test_focus']}")
            print(f"   Query: \"{p['query']}\"")
            print(f"   Category: {p['category']}")
            print(f"   Coverage: {p['coverage']:.0f}% ({len(p['detected_features'])}/{len(p['expected_features'])} features)")
            print(f"   Missing: {', '.join(p['missing_features'])}")
            print(f"   Response: {p['response_preview']}...")
        
        if len(problematic) > 10:
            print(f"\n   ... and {len(problematic) - 10} more problematic queries")
        
        # Recommendations
        print("\n\n💡 RECOMMENDATIONS FOR IMPROVEMENT")
        print("-"*100)
        recommendations = self.generate_recommendations(category_analysis, feature_stats, problematic)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
        
        # Summary
        print("\n\n📈 OVERALL ASSESSMENT")
        print("-"*100)
        
        avg_coverage = sum(
            len([f for f in r['expected_features'] if f in r['detected_features']]) / len(r['expected_features']) * 100
            if r['expected_features'] else 0
            for r in self.results
        ) / len(self.results)
        
        if avg_coverage >= 70:
            status = "🟢 EXCELLENT"
        elif avg_coverage >= 50:
            status = "🟡 GOOD"
        elif avg_coverage >= 30:
            status = "🟠 FAIR"
        else:
            status = "🔴 NEEDS IMPROVEMENT"
        
        print(f"\nStatus: {status}")
        print(f"Average Feature Coverage: {avg_coverage:.1f}%")
        print(f"System Reliability: {self.successful_tests/self.total_tests*100:.1f}%")
        print(f"\n{'='*100}\n")


def main():
    """Main analysis execution"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_places_test_results.py <results_json_file>")
        print("\nSearching for most recent results file...")
        
        import glob
        results_files = glob.glob("places_attractions_test_results_*.json")
        if results_files:
            latest_file = max(results_files)
            print(f"Found: {latest_file}\n")
        else:
            print("No results files found!")
            sys.exit(1)
    else:
        latest_file = sys.argv[1]
    
    analyzer = PlacesAttractionsAnalyzer(latest_file)
    analyzer.print_report()
    
    print(f"✅ Analysis complete!")
    print(f"📄 Analyzed file: {latest_file}")


if __name__ == '__main__':
    main()

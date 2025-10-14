#!/usr/bin/env python3
"""
Enhanced Response Analyzer with Method Patching
Evaluates AI system responses after applying missing method patches
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def apply_patches():
    """Apply patches before importing the main system"""
    print("🔧 Applying system patches...")
    try:
        from patch_missing_methods import patch_istanbul_daily_talk_ai
        success = patch_istanbul_daily_talk_ai()
        if success:
            print("✅ Patches applied successfully!")
            return True
        else:
            print("❌ Failed to apply patches")
            return False
    except Exception as e:
        print(f"❌ Error applying patches: {e}")
        return False

def main():
    """Main analysis function"""
    print("🚀 Enhanced Istanbul AI Response Analysis")
    print("=" * 70)
    
    # Apply patches first
    if not apply_patches():
        print("❌ Cannot proceed without patches. Exiting...")
        return False
    
    # Now import the system
    print("🔄 Initializing AI system...")
    try:
        from istanbul_daily_talk_system import IstanbulDailyTalkAI
        
        # Initialize the AI system
        ai_system = IstanbulDailyTalkAI()
        print("✅ AI system ready!")
        
    except Exception as e:
        print(f"❌ Failed to initialize AI system: {e}")
        return False
    
    # Test cases to evaluate various aspects of the system
    test_cases = [
        {
            'name': 'FALLBACK_HANDLING',
            'query': 'xyz random unclear text',
            'expected_features': ['helpful_fallback', 'user_guidance']
        },
        {
            'name': 'MULTI_INTENT',
            'query': 'I want to visit museums, find restaurants, and get transportation info',
            'expected_features': ['multiple_topics', 'comprehensive_response']
        },
        {
            'name': 'CULTURAL_SITES',
            'query': 'Show me the most important cultural and historical sites',
            'expected_features': ['cultural_info', 'historical_context']
        },
        {
            'name': 'FAMILY_FRIENDLY',
            'query': 'We have children, suggest family-friendly attractions',
            'expected_features': ['family_considerations', 'practical_info']
        },
        {
            'name': 'FOOD_CULTURE',
            'query': 'I love Turkish cuisine and culture, where should I go?',
            'expected_features': ['food_recommendations', 'cultural_context']
        },
        {
            'name': 'ROMANTIC_SPOTS',
            'query': 'Romantic places for couples in Istanbul',
            'expected_features': ['romantic_atmosphere', 'scenic_locations']
        },
        {
            'name': 'BUDGET_TRAVEL',
            'query': 'Free or cheap attractions for budget travelers',
            'expected_features': ['budget_conscious', 'cost_information']
        },
        {
            'name': 'WEATHER_SPECIFIC',
            'query': 'It\'s raining, what indoor activities do you recommend?',
            'expected_features': ['weather_consideration', 'indoor_options']
        }
    ]
    
    print(f"🚀 Starting Enhanced Response Analysis")
    print("=" * 70)
    print()
    
    results = []
    successful_tests = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔍 Test {i}: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print("-" * 50)
        
        try:
            # Generate response with timing
            start_time = time.time()
            response = ai_system.process_message(test_case['query'], f"analysis_user_{i}")
            response_time = time.time() - start_time
            
            # Analyze response quality
            quality_score = analyze_response_quality(response, test_case)
            
            print(f"✅ Response generated in {response_time:.1f}s")
            print(f"📊 Quality Score: {quality_score}/100")
            print(f"📝 Length: {len(response.split())} words")
            print(f"📖 Preview: {response[:100]}...")
            print()
            
            results.append({
                'test_name': test_case['name'],
                'query': test_case['query'],
                'response': response,
                'quality_score': quality_score,
                'response_time': response_time,
                'word_count': len(response.split()),
                'success': True
            })
            
            successful_tests += 1
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
            
            results.append({
                'test_name': test_case['name'],
                'query': test_case['query'],
                'error': str(e),
                'quality_score': 0,
                'success': False
            })
    
    # Generate comprehensive report
    print("=" * 70)
    print("📊 ENHANCED ANALYSIS REPORT")
    print("=" * 70)
    print()
    
    success_rate = (successful_tests / len(test_cases)) * 100
    avg_quality = sum(r['quality_score'] for r in results if r['success']) / max(successful_tests, 1)
    
    print(f"🎯 OVERALL PERFORMANCE:")
    print(f"   • Total Tests: {len(test_cases)}")
    print(f"   • Successful: {successful_tests}")
    print(f"   • Failed: {len(test_cases) - successful_tests}")
    print(f"   • Success Rate: {success_rate:.1f}%")
    print()
    
    print(f"⭐ QUALITY METRICS:")
    print(f"   • Average Quality Score: {avg_quality:.1f}/100")
    
    if successful_tests > 0:
        best_result = max([r for r in results if r['success']], key=lambda x: x['quality_score'])
        worst_result = min([r for r in results if r['success']], key=lambda x: x['quality_score'])
        
        print(f"   • Best Score: {best_result['quality_score']}/100 ({best_result['test_name']})")
        print(f"   • Worst Score: {worst_result['quality_score']}/100 ({worst_result['test_name']})")
    
    print()
    
    # Provide improvement recommendations
    print("💡 SYSTEM STATUS:")
    if success_rate == 100:
        print("   ✅ All missing method errors resolved!")
        print("   ✅ System is stable and functional")
    elif success_rate >= 75:
        print("   ⚠️ Most tests passing, some issues remain")
    else:
        print("   ❌ Significant issues still present")
    
    print()
    
    if avg_quality >= 70:
        print("   ✅ Response quality is good")
    elif avg_quality >= 50:
        print("   ⚠️ Response quality needs improvement")
    else:
        print("   ❌ Response quality requires significant work")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"enhanced_analysis_results_{timestamp}.json"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'total_tests': len(test_cases),
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'average_quality': avg_quality,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced analysis completed!")
    print(f"📁 Detailed results saved to: {results_file}")
    print()
    print("🎉 Analysis completed with method patches applied!")
    
    return success_rate == 100

def analyze_response_quality(response: str, test_case: Dict[str, Any]) -> int:
    """Analyze the quality of a response based on various criteria"""
    
    score = 0
    words = response.lower().split()
    response_lower = response.lower()
    
    # Basic response quality (30 points)
    if len(words) >= 20:
        score += 15  # Adequate length
    elif len(words) >= 10:
        score += 10
    elif len(words) >= 5:
        score += 5
    
    if any(word in response_lower for word in ['istanbul', 'turkey', 'turkish']):
        score += 10  # Relevant to Istanbul
    
    if response.strip():
        score += 5  # Non-empty response
    
    # Content structure (20 points)
    if '**' in response or '*' in response:
        score += 5  # Formatted content
    
    if any(word in response_lower for word in ['recommend', 'suggest', 'visit', 'try']):
        score += 5  # Actionable advice
    
    if any(word in response_lower for word in ['help', 'assist', 'guide']):
        score += 5  # Helpful tone
    
    if response.count('.') >= 2:
        score += 5  # Multiple sentences
    
    # Context awareness (25 points)
    test_name = test_case['name'].lower()
    
    if 'fallback' in test_name:
        if any(word in response_lower for word in ['help', 'assist', 'tell me', 'what would you like']):
            score += 15
    elif 'multi_intent' in test_name:
        if len([word for word in ['museum', 'restaurant', 'transport', 'attraction'] if word in response_lower]) >= 2:
            score += 15
    elif 'cultural' in test_name:
        if any(word in response_lower for word in ['history', 'culture', 'historical', 'heritage']):
            score += 15
    elif 'family' in test_name:
        if any(word in response_lower for word in ['family', 'children', 'kids', 'child-friendly']):
            score += 15
    elif 'food' in test_name:
        if any(word in response_lower for word in ['food', 'restaurant', 'cuisine', 'eat', 'taste']):
            score += 15
    elif 'romantic' in test_name:
        if any(word in response_lower for word in ['romantic', 'couple', 'sunset', 'beautiful', 'scenic']):
            score += 15
    elif 'budget' in test_name:
        if any(word in response_lower for word in ['free', 'budget', 'cheap', 'affordable', 'cost']):
            score += 15
    elif 'weather' in test_name:
        if any(word in response_lower for word in ['indoor', 'inside', 'covered', 'weather', 'rain']):
            score += 15
    
    # Practical information (15 points)
    if any(word in response_lower for word in ['hour', 'time', 'open', 'close']):
        score += 5  # Time information
    
    if any(word in response_lower for word in ['address', 'location', 'district', 'area', 'near']):
        score += 5  # Location information
    
    if any(word in response_lower for word in ['metro', 'bus', 'tram', 'transport', 'walk']):
        score += 5  # Transportation info
    
    # User engagement (10 points)
    if '?' in response:
        score += 5  # Asks follow-up questions
    
    if any(word in response_lower for word in ['let me know', 'tell me', 'would you like', 'need help']):
        score += 5  # Encourages interaction
    
    return min(score, 100)  # Cap at 100

if __name__ == "__main__":
    main()

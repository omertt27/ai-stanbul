#!/usr/bin/env python3
"""
Simple Istanbul AI Attractions Test Suite
Tests the integrated attractions system with 20 diverse queries
"""

import sys
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.append('/Users/omer/Desktop/ai-stanbul')

from istanbul_daily_talk_system import IstanbulDailyTalkAI

def test_attractions_queries():
    """Test the Istanbul AI system with 20 attraction-focused queries"""
    
    print("🏛️ Istanbul AI Attractions Test Suite")
    print("Testing 78+ curated attractions with advanced filtering...")
    print("=" * 60)
    
    # Initialize the AI system
    print("🚀 Initializing Istanbul Daily Talk AI System...")
    ai = IstanbulDailyTalkAI()
    print("✅ System initialized successfully!")
    
    # Test queries covering all the features mentioned
    test_queries = [
        # 1. Museums (Category filtering)
        "Show me the best museums in Istanbul",
        
        # 2. Monuments (Category filtering)  
        "What historical monuments should I visit?",
        
        # 3. Parks (Category filtering)
        "Where are the beautiful parks and gardens?",
        
        # 4. Religious sites (Category filtering)
        "Recommend important mosques and religious places",
        
        # 5. District-based - Sultanahmet
        "What attractions are in Sultanahmet district?",
        
        # 6. District-based - Beyoğlu
        "Show me places to visit in Beyoğlu",
        
        # 7. Weather-appropriate - Indoor (Rainy day)
        "It's raining, suggest indoor attractions",
        
        # 8. Weather-appropriate - Outdoor (Sunny day)
        "Perfect weather! What outdoor places should I visit?",
        
        # 9. Family-friendly activities
        "Best attractions for families with children",
        
        # 10. Romantic spots for couples
        "Romantic places for a date in Istanbul",
        
        # 11. Budget-friendly (Free activities)
        "What free attractions can I visit?",
        
        # 12. Budget-friendly (Cheap activities)
        "Affordable places to visit on a budget",
        
        # 13. Photography spots
        "Most Instagram-worthy places for photos",
        
        # 14. Cultural experiences  
        "Authentic cultural sites and experiences",
        
        # 15. Architecture focus
        "Most impressive architecture in Istanbul",
        
        # 16. Bosphorus views
        "Places with the best Bosphorus views",
        
        # 17. Ottoman history
        "Ottoman historical sites to visit",
        
        # 18. Byzantine history
        "Byzantine historical places in Istanbul",
        
        # 19. Art and galleries
        "Art museums and galleries to visit",
        
        # 20. Complete tourist itinerary
        "Plan a perfect day visiting Istanbul's must-see attractions"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*60}")
        print(f"🧪 Test {i}/20: {query}")
        
        try:
            start_time = time.time()
            response = ai.process_message(query, f'test_user_{i}')
            processing_time = time.time() - start_time
            
            # Analyze response for attraction keywords
            response_lower = response.lower()
            
            # Key attraction keywords to look for
            attraction_keywords = [
                'hagia sophia', 'blue mosque', 'topkapi', 'galata tower', 
                'sultanahmet', 'beyoğlu', 'grand bazaar', 'bosphorus',
                'dolmabahçe', 'basilica cistern', 'spice bazaar', 'taksim',
                'museum', 'palace', 'mosque', 'park', 'tower', 'bridge',
                'attraction', 'visit', 'historic', 'cultural', 'gallery'
            ]
            
            found_keywords = [kw for kw in attraction_keywords if kw in response_lower]
            
            # Check for specific features
            features_found = []
            if 'free' in response_lower or 'no cost' in response_lower:
                features_found.append('budget-friendly')
            if 'family' in response_lower or 'children' in response_lower:
                features_found.append('family-friendly')  
            if 'romantic' in response_lower or 'couple' in response_lower:
                features_found.append('romantic')
            if 'photo' in response_lower or 'instagram' in response_lower:
                features_found.append('photography')
            if 'indoor' in response_lower:
                features_found.append('weather-appropriate')
            if 'outdoor' in response_lower:
                features_found.append('weather-appropriate')
                
            result = {
                'query': query,
                'response_length': len(response),
                'processing_time': round(processing_time, 3),
                'keywords_found': found_keywords,
                'features_found': features_found,
                'success': len(found_keywords) > 0 or len(features_found) > 0,
                'response_preview': response[:200] + '...' if len(response) > 200 else response
            }
            
            results.append(result)
            
            # Display results
            if result['success']:
                print(f"✅ SUCCESS")
            else:
                print(f"⚠️  LIMITED SUCCESS")
                
            print(f"📊 Response: {len(response)} chars, {processing_time:.3f}s")
            print(f"🎯 Keywords: {len(found_keywords)} found ({', '.join(found_keywords[:3])})")
            print(f"🏷️  Features: {', '.join(features_found) if features_found else 'None detected'}")
            print(f"📝 Preview: {result['response_preview']}")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            results.append({
                'query': query,
                'error': str(e),
                'success': False,
                'response_length': 0,
                'processing_time': 0
            })
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("🏛️ ATTRACTIONS TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in results if r.get('success', False)]
    total_keywords = sum(len(r.get('keywords_found', [])) for r in results)
    avg_response_length = sum(r.get('response_length', 0) for r in results) / len(results) if results else 0
    avg_processing_time = sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0
    
    print(f"📊 Total tests: {len(test_queries)}")
    print(f"✅ Successful: {len(successful_tests)}")
    print(f"⚠️  Limited/Failed: {len(test_queries) - len(successful_tests)}")
    print(f"🎯 Success rate: {len(successful_tests)/len(test_queries)*100:.1f}%")
    print(f"🏛️ Total attraction keywords found: {total_keywords}")
    print(f"📏 Average response length: {avg_response_length:.0f} chars")
    print(f"⏱️  Average processing time: {avg_processing_time:.3f}s")
    
    # Feature analysis
    all_features = []
    for r in results:
        all_features.extend(r.get('features_found', []))
    
    if all_features:
        from collections import Counter
        feature_counts = Counter(all_features)
        print(f"\n🏷️  Feature Coverage:")
        for feature, count in feature_counts.items():
            print(f"   {feature}: {count} times")
    
    # Most mentioned attractions
    all_keywords = []
    for r in results:
        all_keywords.extend(r.get('keywords_found', []))
    
    if all_keywords:
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        print(f"\n🏛️ Top Attractions Mentioned:")
        for keyword, count in keyword_counts.most_common(10):
            print(f"   {keyword}: {count} times")
    
    # Performance assessment
    print(f"\n💡 Assessment:")
    success_rate = len(successful_tests)/len(test_queries)*100
    
    if success_rate >= 90:
        print("   🌟 EXCELLENT: System handles attractions queries exceptionally well!")
    elif success_rate >= 80:
        print("   ✅ VERY GOOD: Strong performance with minor areas for improvement")
    elif success_rate >= 70:
        print("   👍 GOOD: Solid performance, some optimization opportunities")
    else:
        print("   ⚠️  NEEDS IMPROVEMENT: Consider enhancing attraction detection")
    
    if avg_processing_time < 1.0:
        print("   ⚡ FAST: Excellent response times!")
    elif avg_processing_time < 2.0:
        print("   🕐 ACCEPTABLE: Good response times")
    else:
        print("   🐌 SLOW: Consider optimizing for better performance")
    
    print(f"\n✨ Test completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    try:
        print("Starting Istanbul AI Attractions Test...")
        results = test_attractions_queries()
        print("\n🎉 Test suite completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()

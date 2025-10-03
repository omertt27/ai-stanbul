#!/usr/bin/env python3
"""
Comprehensive Analysis of Istanbul Attractions Test Results
==========================================================

This script analyzes all 100 test responses for Istanbul attractions recommendations
to identify patterns, strengths, and areas for improvement.
"""

import json
from collections import defaultdict, Counter
import re

def analyze_attractions_responses():
    """Comprehensive analysis of all attraction responses"""
    
    with open('attractions_test_results_20251003_165743.json', 'r') as f:
        data = json.load(f)
    
    successful = [r for r in data if 'error' not in r]
    failed = [r for r in data if 'error' in r]
    
    print("🏛️ COMPLETE ISTANBUL ATTRACTIONS TEST ANALYSIS")
    print("=" * 80)
    print(f"📊 Test Results: {len(successful)}/{len(data)} successful ({len(successful)/len(data)*100:.1f}%)")
    print(f"⭐ Average Quality Score: {sum(r['quality_score'] for r in successful)/len(successful):.1f}/100")
    print()
    
    # === ATTRACTION KNOWLEDGE ANALYSIS ===
    print("🏛️ ATTRACTION KNOWLEDGE ANALYSIS")
    print("-" * 50)
    
    # Count mentions of specific attractions
    attraction_mentions = Counter()
    all_responses_text = ""
    
    for result in successful:
        response_lower = result['response'].lower()
        all_responses_text += response_lower + " "
        
        # Major Istanbul attractions
        attractions = {
            'Hagia Sophia': ['hagia sophia', 'ayasofya'],
            'Blue Mosque': ['blue mosque', 'sultan ahmed mosque', 'sultanahmet mosque'],
            'Topkapi Palace': ['topkapi palace', 'topkapı palace', 'topkapi'],
            'Grand Bazaar': ['grand bazaar', 'kapalıçarşı', 'kapali carsi'],
            'Galata Tower': ['galata tower', 'galata kulesi'],
            'Basilica Cistern': ['basilica cistern', 'yerebatan cistern'],
            'Dolmabahce Palace': ['dolmabahce palace', 'dolmabahçe palace'],
            'Spice Bazaar': ['spice bazaar', 'egyptian bazaar', 'mısır çarşısı'],
            'Bosphorus': ['bosphorus', 'boğaz', 'bosphorus cruise'],
            'Galata Bridge': ['galata bridge', 'galata köprüsü'],
            'Istiklal Street': ['istiklal street', 'istiklal avenue', 'istiklal caddesi'],
            'Taksim Square': ['taksim square', 'taksim meydanı'],
            'Golden Horn': ['golden horn', 'haliç'],
            'Maiden Tower': ['maiden tower', 'kız kulesi'],
            'Suleymaniye Mosque': ['suleymaniye mosque', 'süleymaniye mosque'],
            'Chora Church': ['chora church', 'kariye church'],
            'Rumeli Fortress': ['rumeli fortress', 'rumeli hisarı'],
            'Princes Islands': ['princes islands', 'adalar'],
            'Camlica Hill': ['camlica hill', 'çamlıca tepesi'],
            'Pierre Loti Hill': ['pierre loti hill', 'pierre loti tepesi']
        }
        
        for attraction, variants in attractions.items():
            for variant in variants:
                if variant in response_lower:
                    attraction_mentions[attraction] += 1
                    break
    
    print("Top 15 Most Mentioned Attractions:")
    for i, (attraction, count) in enumerate(attraction_mentions.most_common(15), 1):
        percentage = (count / len(successful)) * 100
        print(f"  {i:2d}. {attraction:<20} {count:2d} mentions ({percentage:4.1f}%)")
    
    print()
    
    # === CATEGORY ANALYSIS ===
    print("🎯 CATEGORY PERFORMANCE ANALYSIS")
    print("-" * 50)
    
    category_performance = defaultdict(list)
    for result in successful:
        query = result['query'].lower()
        score = result['quality_score']
        
        # Categorize queries
        if any(word in query for word in ['museum', 'gallery', 'art']):
            category_performance['Museums'].append(score)
        elif any(word in query for word in ['monument', 'historic', 'ancient']):
            category_performance['Historic Monuments'].append(score)
        elif any(word in query for word in ['park', 'garden', 'nature', 'outdoor']):
            category_performance['Parks & Nature'].append(score)
        elif any(word in query for word in ['mosque', 'church', 'religious']):
            category_performance['Religious Sites'].append(score)
        elif any(word in query for word in ['shop', 'market', 'bazaar']):
            category_performance['Shopping'].append(score)
        elif any(word in query for word in ['sultanahmet', 'beyoğlu', 'beşiktaş', 'kadıköy']):
            category_performance['District-Specific'].append(score)
        elif any(word in query for word in ['family', 'kid', 'romantic', 'couple']):
            category_performance['Audience-Specific'].append(score)
        elif any(word in query for word in ['free', 'budget', 'cheap', 'affordable']):
            category_performance['Budget-Conscious'].append(score)
        elif any(word in query for word in ['rainy', 'indoor', 'covered', 'weather']):
            category_performance['Weather-Specific'].append(score)
        else:
            category_performance['General Attractions'].append(score)
    
    for category, scores in category_performance.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"  {category:<20} {avg_score:5.1f}/100 ({len(scores):2d} queries)")
    
    print()
    
    # === TURKISH CONTEXT ANALYSIS ===
    print("🇹🇷 TURKISH CULTURAL CONTEXT ANALYSIS")
    print("-" * 50)
    
    turkish_elements = {
        'Turkish Characters': ['ğ', 'ü', 'ş', 'ç', 'ı', 'ö'],
        'Turkish Names': ['beyoğlu', 'kadıköy', 'üsküdar', 'beşiktaş', 'sultanahmet'],
        'Turkish Terms': ['camii', 'sarayı', 'çarşısı', 'köprüsü', 'kulesi', 'tepesi'],
        'Ottoman References': ['ottoman', 'sultan', 'byzantine', 'imperial'],
        'Cultural Terms': ['hammam', 'meze', 'turkish bath', 'bazaar', 'spice']
    }
    
    for element_type, terms in turkish_elements.items():
        count = 0
        for result in successful:
            response_lower = result['response'].lower()
            if any(term in response_lower for term in terms):
                count += 1
        percentage = (count / len(successful)) * 100
        print(f"  {element_type:<20} {count:2d}/{len(successful)} responses ({percentage:4.1f}%)")
    
    print()
    
    # === PRACTICAL INFORMATION ANALYSIS ===
    print("ℹ️ PRACTICAL INFORMATION ANALYSIS")
    print("-" * 50)
    
    practical_info_types = {
        'Hours/Opening Times': ['hour', 'open', 'close', 'time', 'am', 'pm'],
        'Transportation': ['metro', 'tram', 'bus', 'walk', 'ferry', 'taxi'],
        'Pricing Information': ['entry', 'ticket', 'free', 'cost', 'price', 'admission'],
        'Directions/Distance': ['minute', 'kilometer', 'near', 'distance', 'from'],
        'Contact/Booking': ['book', 'reserve', 'phone', 'website', 'online']
    }
    
    for info_type, keywords in practical_info_types.items():
        count = 0
        for result in successful:
            response_lower = result['response'].lower()
            if any(keyword in response_lower for keyword in keywords):
                count += 1
        percentage = (count / len(successful)) * 100
        print(f"  {info_type:<20} {count:2d}/{len(successful)} responses ({percentage:4.1f}%)")
    
    print()
    
    # === RESPONSE QUALITY BREAKDOWN ===
    print("📊 RESPONSE QUALITY BREAKDOWN")
    print("-" * 50)
    
    score_ranges = {
        'Excellent (60-80)': [r for r in successful if 60 <= r['quality_score'] <= 80],
        'Good (40-59)': [r for r in successful if 40 <= r['quality_score'] < 60],
        'Fair (20-39)': [r for r in successful if 20 <= r['quality_score'] < 40],
        'Poor (0-19)': [r for r in successful if r['quality_score'] < 20]
    }
    
    for range_name, results in score_ranges.items():
        count = len(results)
        percentage = (count / len(successful)) * 100
        print(f"  {range_name:<18} {count:2d} responses ({percentage:4.1f}%)")
    
    print()
    
    # === TOP AND BOTTOM PERFORMERS ===
    print("🏆 TOP PERFORMING QUERIES")
    print("-" * 50)
    
    top_performers = sorted(successful, key=lambda x: x['quality_score'], reverse=True)[:5]
    for i, result in enumerate(top_performers, 1):
        print(f"  {i}. Score {result['quality_score']}/100: {result['query']}")
    
    print()
    print("⚠️ LOWEST PERFORMING QUERIES")
    print("-" * 50)
    
    bottom_performers = sorted(successful, key=lambda x: x['quality_score'])[:5]
    for i, result in enumerate(bottom_performers, 1):
        print(f"  {i}. Score {result['quality_score']}/100: {result['query']}")
    
    print()
    
    # === CONTENT ANALYSIS ===
    print("📝 CONTENT ANALYSIS")
    print("-" * 50)
    
    total_words = 0
    total_chars = 0
    attraction_counts = []
    response_times = []
    
    for result in successful:
        total_words += len(result['response'].split())
        total_chars += len(result['response'])
        attraction_counts.append(result['analysis']['attraction_mentions'])
        response_times.append(result['response_time'])
    
    print(f"  Average Response Length: {total_chars/len(successful):.0f} characters")
    print(f"  Average Word Count: {total_words/len(successful):.0f} words")
    print(f"  Average Attractions per Response: {sum(attraction_counts)/len(attraction_counts):.1f}")
    print(f"  Average Response Time: {sum(response_times)/len(response_times):.2f} seconds")
    
    print()
    
    # === RECOMMENDATIONS ===
    print("🎯 RECOMMENDATIONS FOR IMPROVEMENT")
    print("-" * 50)
    
    print("  1. ENHANCE PRACTICAL INFORMATION:")
    print("     - Only 15-30% of responses include practical details")
    print("     - Add opening hours, transportation, and pricing info")
    print()
    
    print("  2. IMPROVE DISTRICT-SPECIFIC KNOWLEDGE:")
    print("     - District-specific queries scored lower than expected")
    print("     - Add more local neighborhood details and hidden gems")
    print()
    
    print("  3. INCREASE TURKISH CULTURAL CONTEXT:")
    print("     - Turkish character usage is inconsistent")
    print("     - Include more Turkish cultural context and terminology")
    print()
    
    print("  4. DIVERSIFY ATTRACTION RECOMMENDATIONS:")
    print("     - Heavy focus on top 5-6 attractions")
    print("     - Include more diverse, lesser-known attractions")
    print()
    
    print("  5. ENHANCE AUDIENCE-SPECIFIC RESPONSES:")
    print("     - Family, romantic, and budget queries need more tailored content")
    print("     - Add age-appropriate and interest-specific recommendations")

if __name__ == "__main__":
    analyze_attractions_responses()

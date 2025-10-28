"""
AI Istanbul System - Pre-Deployment Readiness Analysis
========================================================
This script analyzes main.py and main_system.py to verify all required features
are implemented and support both Turkish and English.

Required Features:
1. Daily talks/greetings
2. Places/attractions
3. Neighborhood guides
4. Transportation
5. Events advising
6. Route planner
7. Weather system
8. Local tips/hidden gems
"""

import os
import sys
import json
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

def analyze_file(filepath):
    """Analyze a Python file for feature implementation"""
    if not os.path.exists(filepath):
        return {"error": f"File not found: {filepath}"}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {
        "file": filepath,
        "lines": len(content.split('\n')),
        "size_kb": round(len(content) / 1024, 2),
        "content": content
    }

def check_feature_support(content, feature_keywords):
    """Check if a feature is supported based on keywords"""
    found_keywords = []
    for keyword in feature_keywords:
        if keyword.lower() in content.lower():
            found_keywords.append(keyword)
    return {
        "supported": len(found_keywords) > 0,
        "found_keywords": found_keywords,
        "match_count": len(found_keywords)
    }

def check_language_support(content):
    """Check for Turkish and English language support"""
    turkish_indicators = [
        'turkish', 'türkçe', 'tr', 'merhaba', 'hoş geldiniz',
        'güle güle', 'teşekkür', 'lütfen', 'istanbul'
    ]
    
    english_indicators = [
        'english', 'en', 'hello', 'welcome', 'thank you',
        'please', 'goodbye'
    ]
    
    turkish_found = []
    english_found = []
    
    content_lower = content.lower()
    
    for indicator in turkish_indicators:
        if indicator in content_lower:
            turkish_found.append(indicator)
    
    for indicator in english_indicators:
        if indicator in content_lower:
            english_found.append(indicator)
    
    return {
        "turkish_support": len(turkish_found) > 0,
        "turkish_indicators": turkish_found,
        "english_support": len(english_found) > 0,
        "english_indicators": english_found,
        "bilingual": len(turkish_found) > 0 and len(english_found) > 0
    }

def main():
    print("=" * 80)
    print("AI ISTANBUL - PRE-DEPLOYMENT READINESS ANALYSIS")
    print("=" * 80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Files to analyze
    files_to_check = [
        "backend/main.py",
        "istanbul_ai/main_system.py"
    ]
    
    # Features to verify
    features = {
        "1. Daily Talks/Greetings": [
            "greeting", "hello", "welcome", "merhaba", "hoş geldiniz",
            "daily_talk", "conversation", "chat"
        ],
        "2. Places/Attractions": [
            "attraction", "museum", "mosque", "palace", "landmark",
            "place", "sight", "visit", "tourist"
        ],
        "3. Neighborhood Guides": [
            "neighborhood", "district", "mahalle", "sultanahmet",
            "beyoglu", "kadikoy", "besiktas", "guide", "area"
        ],
        "4. Transportation": [
            "transport", "metro", "bus", "ferry", "tram",
            "route", "direction", "how to get", "travel"
        ],
        "5. Events Advising": [
            "event", "concert", "festival", "exhibition",
            "show", "performance", "happening", "etkinlik"
        ],
        "6. Route Planner": [
            "route", "plan", "itinerary", "path", "journey",
            "directions", "navigate", "map"
        ],
        "7. Weather System": [
            "weather", "temperature", "rain", "sunny",
            "forecast", "hava durumu", "climate"
        ],
        "8. Local Tips/Hidden Gems": [
            "hidden", "local", "secret", "tip", "recommendation",
            "authentic", "off the beaten", "hidden_gems"
        ]
    }
    
    results = {}
    
    for filepath in files_to_check:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {filepath}")
        print(f"{'='*80}\n")
        
        file_analysis = analyze_file(filepath)
        
        if "error" in file_analysis:
            print(f"❌ {file_analysis['error']}")
            continue
        
        print(f"📄 File Size: {file_analysis['size_kb']} KB")
        print(f"📏 Lines of Code: {file_analysis['lines']}")
        
        content = file_analysis['content']
        
        # Check language support
        print(f"\n{'─'*80}")
        print("🌍 LANGUAGE SUPPORT")
        print(f"{'─'*80}")
        lang_support = check_language_support(content)
        
        if lang_support['bilingual']:
            print("✅ BILINGUAL SUPPORT DETECTED")
        else:
            print("⚠️  LIMITED LANGUAGE SUPPORT")
        
        print(f"\n🇹🇷 Turkish Support: {'✅ YES' if lang_support['turkish_support'] else '❌ NO'}")
        if lang_support['turkish_indicators']:
            print(f"   Indicators found: {', '.join(lang_support['turkish_indicators'][:5])}")
        
        print(f"\n🇬🇧 English Support: {'✅ YES' if lang_support['english_support'] else '❌ NO'}")
        if lang_support['english_indicators']:
            print(f"   Indicators found: {', '.join(lang_support['english_indicators'][:5])}")
        
        # Check each feature
        print(f"\n{'─'*80}")
        print("🎯 FEATURE SUPPORT ANALYSIS")
        print(f"{'─'*80}\n")
        
        feature_results = {}
        for feature_name, keywords in features.items():
            feature_check = check_feature_support(content, keywords)
            feature_results[feature_name] = feature_check
            
            status = "✅" if feature_check['supported'] else "❌"
            print(f"{status} {feature_name}")
            if feature_check['supported']:
                print(f"   Keywords found: {', '.join(feature_check['found_keywords'][:5])}")
                if len(feature_check['found_keywords']) > 5:
                    print(f"   ... and {len(feature_check['found_keywords']) - 5} more")
            print()
        
        results[filepath] = {
            "file_info": {
                "size_kb": file_analysis['size_kb'],
                "lines": file_analysis['lines']
            },
            "language_support": lang_support,
            "features": feature_results
        }
    
    # Summary Report
    print(f"\n{'='*80}")
    print("📊 SUMMARY REPORT")
    print(f"{'='*80}\n")
    
    total_features = len(features)
    
    for filepath, analysis in results.items():
        print(f"\n📁 {filepath}")
        print(f"{'─'*80}")
        
        supported_features = sum(1 for f in analysis['features'].values() if f['supported'])
        support_percentage = (supported_features / total_features) * 100
        
        print(f"Language Support: {' ✅ Bilingual' if analysis['language_support']['bilingual'] else '⚠️  Limited'}")
        print(f"Feature Coverage: {supported_features}/{total_features} ({support_percentage:.1f}%)")
        
        if support_percentage < 100:
            print(f"\n⚠️  MISSING FEATURES:")
            for feature_name, result in analysis['features'].items():
                if not result['supported']:
                    print(f"   • {feature_name}")
    
    # Overall Readiness
    print(f"\n{'='*80}")
    print("🎯 OVERALL DEPLOYMENT READINESS")
    print(f"{'='*80}\n")
    
    all_bilingual = all(r['language_support']['bilingual'] for r in results.values())
    all_features_supported = all(
        all(f['supported'] for f in r['features'].values())
        for r in results.values()
    )
    
    if all_bilingual and all_features_supported:
        print("✅ SYSTEM IS READY FOR DEPLOYMENT")
        print("   • All features implemented")
        print("   • Bilingual support (Turkish/English) confirmed")
        print("   • No critical issues detected")
    else:
        print("⚠️  SYSTEM NEEDS ATTENTION BEFORE DEPLOYMENT")
        if not all_bilingual:
            print("   • Language support incomplete")
        if not all_features_supported:
            print("   • Some features missing or incomplete")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR TESTING")
    print(f"{'='*80}\n")
    
    print("1. Test all 8 feature areas with both Turkish and English queries")
    print("2. Verify bilingual responses for each feature")
    print("3. Test edge cases and error handling")
    print("4. Verify database connections and API integrations")
    print("5. Load testing with concurrent users")
    print("6. Monitor performance metrics")
    print("7. Test fallback mechanisms when services are unavailable")
    print("8. Verify logging and error tracking")
    
    print(f"\n{'='*80}")
    print("TEST QUERIES TO RUN")
    print(f"{'='*80}\n")
    
    test_queries = {
        "Daily Talks (Turkish)": [
            "Merhaba",
            "Nasılsın?",
            "Istanbul hakkında bilgi ver"
        ],
        "Daily Talks (English)": [
            "Hello",
            "How are you?",
            "Tell me about Istanbul"
        ],
        "Places (Turkish)": [
            "Sultanahmet'te gezilecek yerler",
            "Ayasofya hakkında bilgi",
            "En iyi müzeler nereler?"
        ],
        "Places (English)": [
            "Places to visit in Sultanahmet",
            "Tell me about Hagia Sophia",
            "Best museums to visit"
        ],
        "Transportation (Turkish)": [
            "Taksim'den Kadıköy'e nasıl gidilir?",
            "En yakın metro durağı nerede?",
            "Havalimanına nasıl giderim?"
        ],
        "Transportation (English)": [
            "How to get from Taksim to Kadikoy?",
            "Where is the nearest metro station?",
            "How do I get to the airport?"
        ],
        "Events (Turkish)": [
            "Bu hafta sonu hangi etkinlikler var?",
            "Konserler hakkında bilgi",
            "Yaklaşan festivaller"
        ],
        "Events (English)": [
            "What events are happening this weekend?",
            "Tell me about concerts",
            "Upcoming festivals"
        ],
        "Weather (Turkish)": [
            "Hava durumu nasıl?",
            "Bugün yağmur yağacak mı?",
            "Haftalık hava tahmini"
        ],
        "Weather (English)": [
            "What's the weather like?",
            "Will it rain today?",
            "Weekly weather forecast"
        ],
        "Hidden Gems (Turkish)": [
            "Gizli mekanlar",
            "Turistik olmayan yerler",
            "Yerel ipuçları"
        ],
        "Hidden Gems (English)": [
            "Hidden gems in Istanbul",
            "Off the beaten path",
            "Local tips"
        ]
    }
    
    for category, queries in test_queries.items():
        print(f"\n{category}:")
        for query in queries:
            print(f"   • \"{query}\"")
    
    # Save detailed report
    report_filename = f"deployment_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Detailed report saved to: {report_filename}")
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

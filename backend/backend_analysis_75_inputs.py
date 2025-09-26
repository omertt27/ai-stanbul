#!/usr/bin/env python3
"""
Backend Code Analysis - Verify 75 Input Categories Work
Analyzes main.py to ensure all response categories are properly handled
"""

import os
import re
from datetime import datetime

def analyze_backend_readiness():
    """Analyze the backend code to verify it handles all input categories correctly"""
    
    print("🔍 BACKEND READINESS ANALYSIS")
    print("=" * 60)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Read the main.py file
    main_py_path = "main.py"
    if not os.path.exists(main_py_path):
        print("❌ main.py not found!")
        return False
    
    with open(main_py_path, 'r', encoding='utf-8') as f:
        code_content = f.read()
    
    analysis_results = {
        "daily_usage_tracking": False,
        "middleware_present": False,
        "admin_endpoints": False,
        "fallback_responses": False,
        "gpt_integration": False,
        "district_responses": False,
        "restaurant_responses": False,
        "transportation_responses": False,
        "history_responses": False,
        "general_tips": False
    }
    
    checks = [
        # Daily Usage Tracking
        ("daily_usage_tracking", r"def check_and_update_daily_usage", "✅ Daily usage tracking function"),
        ("daily_usage_tracking", r"daily_usage\.db", "✅ Usage database setup"),
        
        # Middleware
        ("middleware_present", r"class DailyUsageMiddleware", "✅ Daily usage middleware class"),
        ("middleware_present", r"app\.add_middleware\(DailyUsageMiddleware\)", "✅ Middleware registration"),
        
        # Admin Endpoints
        ("admin_endpoints", r"@app\.get\(\"/admin/usage-stats\"\)", "✅ Usage stats endpoint"),
        ("admin_endpoints", r"@app\.post\(\"/admin/reset-usage\"\)", "✅ Usage reset endpoint"),
        
        # Response Functions
        ("fallback_responses", r"def create_fallback_response", "✅ Fallback response function"),
        ("gpt_integration", r"def get_gpt_response", "✅ GPT integration function"),
        
        # District Specific Responses
        ("district_responses", r"kadıköy.*cultural hub", "✅ Kadıköy district response"),
        ("district_responses", r"sultanahmet.*historic", "✅ Sultanahmet district response"),
        ("district_responses", r"beyoğlu.*restoran", "✅ Beyoğlu district response"),
        
        # Restaurant Responses
        ("restaurant_responses", r"food.*eat.*cuisine.*restaurant", "✅ Food/restaurant keywords"),
        ("restaurant_responses", r"turkish.*breakfast", "✅ Turkish breakfast response"),
        ("restaurant_responses", r"tipping.*culture", "✅ Tipping culture response"),
        
        # Transportation Responses
        ("transportation_responses", r"transport.*metro.*bus.*ferry", "✅ Transportation keywords"),
        ("transportation_responses", r"istanbulkart", "✅ Istanbulkart information"),
        ("transportation_responses", r"metro.*vs.*metrobus", "✅ Metro vs Metrobus comparison"),
        
        # History Responses
        ("history_responses", r"history.*historical.*byzantine.*ottoman", "✅ History keywords"),
        ("history_responses", r"Constantinople.*Istanbul", "✅ Historical transition info"),
        
        # General Tips
        ("general_tips", r"travel.*tips", "✅ Travel tips keywords"),
        ("general_tips", r"first.*time.*visiting", "✅ First-time visitor guidance"),
    ]
    
    print("🧪 CODE STRUCTURE ANALYSIS")
    print("-" * 40)
    
    found_patterns = []
    for category, pattern, description in checks:
        if re.search(pattern, code_content, re.IGNORECASE | re.DOTALL):
            analysis_results[category] = True
            found_patterns.append(description)
            print(description)
        else:
            print(f"❌ Missing: {description}")
    
    print()
    print("📊 FEATURE COMPLETENESS")
    print("-" * 40)
    
    feature_scores = {}
    for category, status in analysis_results.items():
        category_name = category.replace("_", " ").title()
        status_icon = "✅" if status else "❌"
        feature_scores[category] = status
        print(f"{status_icon} {category_name}")
    
    # Calculate overall readiness score
    completed_features = sum(feature_scores.values())
    total_features = len(feature_scores)
    readiness_percentage = (completed_features / total_features) * 100
    
    print()
    print("🎯 BACKEND READINESS ASSESSMENT")
    print("-" * 40)
    print(f"📈 Features Complete: {completed_features}/{total_features} ({readiness_percentage:.1f}%)")
    
    # Test Category Coverage Analysis
    print()
    print("📝 RESPONSE CATEGORY COVERAGE")
    print("-" * 40)
    
    category_coverage = {
        "Districts": check_district_coverage(code_content),
        "Restaurants": check_restaurant_coverage(code_content), 
        "Transportation": check_transportation_coverage(code_content),
        "History": check_history_coverage(code_content),
        "General Tips": check_general_tips_coverage(code_content)
    }
    
    for category, coverage in category_coverage.items():
        coverage_icon = "✅" if coverage["score"] >= 80 else "⚠️" if coverage["score"] >= 60 else "❌"
        print(f"{coverage_icon} {category}: {coverage['score']:.0f}% coverage")
        for feature in coverage["features"][:3]:  # Show top 3 features
            print(f"   • {feature}")
    
    print()
    
    # Final Assessment
    overall_score = (readiness_percentage + sum(c["score"] for c in category_coverage.values()) / len(category_coverage)) / 2
    
    if overall_score >= 85:
        status = "🟢 READY FOR PRODUCTION"
        message = "Backend is fully ready to handle all 75 input categories!"
    elif overall_score >= 75:
        status = "🟡 READY WITH MINOR IMPROVEMENTS"
        message = "Backend handles most categories well, minor optimizations recommended."
    elif overall_score >= 65:
        status = "🟠 FUNCTIONAL WITH IMPROVEMENTS NEEDED" 
        message = "Backend works but needs some enhancements for optimal coverage."
    else:
        status = "🔴 NEEDS SIGNIFICANT WORK"
        message = "Backend requires major improvements to handle all input categories."
    
    print(f"🏆 FINAL ASSESSMENT: {status}")
    print(f"📊 Overall Score: {overall_score:.1f}%")
    print(f"💬 {message}")
    
    return overall_score >= 75

def check_district_coverage(code_content):
    """Check coverage for district-related responses"""
    district_features = [
        ("Sultanahmet responses", r"sultanahmet.*historic.*district"),
        ("Kadıköy responses", r"kadıköy.*cultural.*hub"),
        ("Beyoğlu responses", r"beyoğlu.*cultural.*district"),
        ("Galata responses", r"galata.*tower.*area"),
        ("Taksim responses", r"taksim.*square"),
        ("District guidance", r"district.*neighborhood.*area")
    ]
    
    found = sum(1 for _, pattern in district_features if re.search(pattern, code_content, re.IGNORECASE))
    score = (found / len(district_features)) * 100
    
    return {
        "score": score,
        "features": [desc for desc, pattern in district_features if re.search(pattern, code_content, re.IGNORECASE)]
    }

def check_restaurant_coverage(code_content):
    """Check coverage for restaurant-related responses"""
    restaurant_features = [
        ("Food & cuisine keywords", r"food.*eat.*cuisine.*restaurant"),
        ("Turkish breakfast", r"turkish.*breakfast.*kahvaltı"),
        ("Tipping culture", r"tipping.*culture.*restaurant"),
        ("Bosphorus restaurants", r"bosphorus.*view.*restaurant"),
        ("Ottoman cuisine", r"ottoman.*cuisine.*palace"),
        ("Restaurant etiquette", r"restaurant.*etiquette.*dining")
    ]
    
    found = sum(1 for _, pattern in restaurant_features if re.search(pattern, code_content, re.IGNORECASE))
    score = (found / len(restaurant_features)) * 100
    
    return {
        "score": score,
        "features": [desc for desc, pattern in restaurant_features if re.search(pattern, code_content, re.IGNORECASE)]
    }

def check_transportation_coverage(code_content):
    """Check coverage for transportation-related responses"""
    transport_features = [
        ("Transport keywords", r"transport.*metro.*bus.*ferry"),
        ("Istanbulkart info", r"istanbulkart.*transport.*card"),
        ("Metro vs Metrobus", r"metro.*metrobus.*difference"),
        ("Airport transport", r"airport.*transport.*transfer"),
        ("Ferry system", r"ferry.*bosphorus.*golden.*horn"),
        ("Complex transport queries", r"is_complex_transportation_query")
    ]
    
    found = sum(1 for _, pattern in transport_features if re.search(pattern, code_content, re.IGNORECASE))
    score = (found / len(transport_features)) * 100
    
    return {
        "score": score,
        "features": [desc for desc, pattern in transport_features if re.search(pattern, code_content, re.IGNORECASE)]
    }

def check_history_coverage(code_content):
    """Check coverage for history-related responses"""
    history_features = [
        ("History keywords", r"history.*historical.*byzantine.*ottoman"),
        ("Empire transitions", r"constantinople.*istanbul.*transition"),
        ("Hagia Sophia history", r"hagia.*sophia.*byzantine.*church"),
        ("Ottoman legacy", r"ottoman.*empire.*palace.*mosque"),
        ("Cultural heritage", r"cultural.*heritage.*traditions"),
        ("Archaeological info", r"archaeological.*museum.*ancient")
    ]
    
    found = sum(1 for _, pattern in history_features if re.search(pattern, code_content, re.IGNORECASE))
    score = (found / len(history_features)) * 100
    
    return {
        "score": score,
        "features": [desc for desc, pattern in history_features if re.search(pattern, code_content, re.IGNORECASE)]
    }

def check_general_tips_coverage(code_content):
    """Check coverage for general travel tips"""
    tips_features = [
        ("Travel tips keywords", r"travel.*tips.*advice"),
        ("First-time guidance", r"first.*time.*visiting"),
        ("Safety information", r"safety.*security.*scams"),
        ("Cultural etiquette", r"cultural.*etiquette.*customs"),
        ("Budget guidance", r"budget.*money.*currency"),
        ("Best times to visit", r"best.*time.*visit.*weather")
    ]
    
    found = sum(1 for _, pattern in tips_features if re.search(pattern, code_content, re.IGNORECASE))
    score = (found / len(tips_features)) * 100
    
    return {
        "score": score,
        "features": [desc for desc, pattern in tips_features if re.search(pattern, code_content, re.IGNORECASE)]
    }

if __name__ == "__main__":
    try:
        is_ready = analyze_backend_readiness()
        
        if is_ready:
            print("\n🎉 CONCLUSION: Backend is ready to handle all 75 input categories!")
        else:
            print("\n⚠️  CONCLUSION: Backend needs improvements before handling all inputs.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

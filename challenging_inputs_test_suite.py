#!/usr/bin/env python3
"""
Challenging AI Istanbul Test Suite - 50 Tricky Inputs
Tests edge cases, ambiguous questions, and scenarios prone to wrong answers
Focus: Inputs that could mislead the AI or result in incorrect/incomplete responses
"""

import requests
import json
import time
from datetime import datetime
import sys

# Backend URL
BASE_URL = "http://localhost:8000"

# 50 challenging test cases designed to potentially get wrong answers
challenging_test_cases = [
    # ========== AMBIGUOUS LOCATION REFERENCES (10 tests) ==========
    {
        "category": "ambiguous_location",
        "input": "Where is the bridge?",
        "challenge": "Multiple bridges exist - Golden Horn, Bosphorus, Galata Bridge",
        "potential_wrong_answer": "Might assume one specific bridge without asking for clarification"
    },
    {
        "category": "ambiguous_location", 
        "input": "How do I get to the palace?",
        "challenge": "Multiple palaces - Topkapi, Dolmabahce, Beylerbeyi",
        "potential_wrong_answer": "Might default to Topkapi without asking which palace"
    },
    {
        "category": "ambiguous_location",
        "input": "Tell me about the mosque near the square",
        "challenge": "Many mosques near various squares",
        "potential_wrong_answer": "Might assume Blue Mosque near Sultanahmet Square"
    },
    {
        "category": "ambiguous_location",
        "input": "Where's the best fish restaurant by the water?",
        "challenge": "Istanbul has water on multiple sides - Bosphorus, Golden Horn, Marmara",
        "potential_wrong_answer": "Might not ask which waterfront area"
    },
    {
        "category": "ambiguous_location",
        "input": "I want to visit the tower in Istanbul",
        "challenge": "Multiple towers - Galata Tower, Maiden's Tower, Beyazit Tower",
        "potential_wrong_answer": "Likely to assume Galata Tower only"
    },
    {
        "category": "ambiguous_location",
        "input": "How far is the airport from downtown?",
        "challenge": "Istanbul has two airports - IST and SAW, unclear which downtown area",
        "potential_wrong_answer": "Might assume one airport and one central location"
    },
    {
        "category": "ambiguous_location",
        "input": "What's near the university?",
        "challenge": "Many universities in Istanbul - Bogazici, Istanbul University, ITU",
        "potential_wrong_answer": "Might assume one specific university"
    },
    {
        "category": "ambiguous_location",
        "input": "Take me to the old city",
        "challenge": "Could refer to Sultanahmet, Fatih district, or historic peninsula generally",
        "potential_wrong_answer": "Might be too specific or too vague"
    },
    {
        "category": "ambiguous_location",
        "input": "Where can I see the sunset over the water?",
        "challenge": "Multiple waterfront locations with sunset views",
        "potential_wrong_answer": "Might suggest east-facing locations incorrectly"
    },
    {
        "category": "ambiguous_location",
        "input": "I'm looking for the market with spices",
        "challenge": "Could be Spice Bazaar, Grand Bazaar spice section, or local markets",
        "potential_wrong_answer": "Might not mention all relevant options"
    },

    # ========== MISLEADING TRANSPORTATION QUERIES (10 tests) ==========
    {
        "category": "misleading_transport",
        "input": "How do I take the subway to Taksim?",
        "challenge": "Istanbul calls it 'metro' not 'subway', might confuse systems",
        "potential_wrong_answer": "Might not clarify metro system or give wrong line info"
    },
    {
        "category": "misleading_transport",
        "input": "Is there an underground train to the European side?",
        "challenge": "Marmaray goes underground but connects sides, metro doesn't cross",
        "potential_wrong_answer": "Might confuse Marmaray with metro system"
    },
    {
        "category": "misleading_transport",
        "input": "Can I use Uber to get around Istanbul easily?",
        "challenge": "Uber exists but BiTaksi, local taxis are more common",
        "potential_wrong_answer": "Might overstate Uber availability or not mention alternatives"
    },
    {
        "category": "misleading_transport",
        "input": "What's the fastest way to cross the Bosphorus by car?",
        "challenge": "Multiple bridges, traffic varies by time, might suggest wrong bridge",
        "potential_wrong_answer": "Might not consider traffic patterns or time of day"
    },
    {
        "category": "misleading_transport",
        "input": "Do trains run all night in Istanbul?",
        "challenge": "Some limited night service but not all lines, complex schedule",
        "potential_wrong_answer": "Might give overly simple yes/no without details"
    },
    {
        "category": "misleading_transport",
        "input": "How much is a taxi from the old town to Galata?",
        "challenge": "Distance is short but traffic/route affects price significantly",
        "potential_wrong_answer": "Might give price without considering traffic or route options"
    },
    {
        "category": "misleading_transport",
        "input": "Is the ferry faster than the metro to cross the city?",
        "challenge": "Depends on specific locations, ferry doesn't serve all areas",
        "potential_wrong_answer": "Might make general comparison without considering specific routes"
    },
    {
        "category": "misleading_transport",
        "input": "Can I walk from Sultanahmet to Taksim Square?",
        "challenge": "Technically possible but very long walk, crosses Golden Horn",
        "potential_wrong_answer": "Might say yes without mentioning distance/difficulty"
    },
    {
        "category": "misleading_transport",
        "input": "Which metro line goes to the Grand Bazaar?",
        "challenge": "No direct metro to Grand Bazaar, closest is Vezneciler or tram",
        "potential_wrong_answer": "Might incorrectly suggest metro line or miss tram option"
    },
    {
        "category": "misleading_transport",
        "input": "How late do buses run to the Asian side?",
        "challenge": "Various bus lines, different schedules, some night buses exist",
        "potential_wrong_answer": "Might give general answer without specifying which buses"
    },

    # ========== CONFUSING FOOD/RESTAURANT QUERIES (10 tests) ==========
    {
        "category": "confusing_food",
        "input": "Where can I find good Turkish pizza?",
        "challenge": "Turkish pizza is 'pide' or 'lahmacun', not like Italian pizza",
        "potential_wrong_answer": "Might suggest regular pizza places instead of pide/lahmacun"
    },
    {
        "category": "confusing_food",
        "input": "What's the best kebab shop near me?",
        "challenge": "No location context provided, 'near me' is ambiguous",
        "potential_wrong_answer": "Can't determine location but might give general answers"
    },
    {
        "category": "confusing_food",
        "input": "Do Turkish restaurants serve alcohol?",
        "challenge": "Some do, some don't, depends on type of restaurant and location",
        "potential_wrong_answer": "Might give overly general yes/no answer"
    },
    {
        "category": "confusing_food",
        "input": "Where's the cheapest place to eat in the tourist area?",
        "challenge": "Tourist areas are generally expensive, might suggest overpriced places",
        "potential_wrong_answer": "Might not emphasize walking to less touristy streets"
    },
    {
        "category": "confusing_food",
        "input": "Can I find vegetarian food easily in Istanbul?",
        "challenge": "Traditional Turkish cuisine has many vegetarian options but not always labeled",
        "potential_wrong_answer": "Might understate availability or overstate difficulty"
    },
    {
        "category": "confusing_food",
        "input": "What time do restaurants close for dinner?",
        "challenge": "Varies widely - some close early, some very late, different types",
        "potential_wrong_answer": "Might give single time without mentioning variation"
    },
    {
        "category": "confusing_food",
        "input": "Where can I eat like a local, not touristy?",
        "challenge": "Vague request, depends on budget, location, preferences",
        "potential_wrong_answer": "Might suggest places that are actually touristy or too specific"
    },
    {
        "category": "confusing_food",
        "input": "Is tipping required in Turkish restaurants?",
        "challenge": "Not required but expected, amount varies by establishment type",
        "potential_wrong_answer": "Might give unclear guidance on amounts or situations"
    },
    {
        "category": "confusing_food",
        "input": "What's the difference between döner and şawarma?",
        "challenge": "Similar concepts, cultural/preparation differences, might confuse",
        "potential_wrong_answer": "Might not explain regional/cultural differences clearly"
    },
    {
        "category": "confusing_food",
        "input": "Can I eat street food safely as a tourist?",
        "challenge": "Generally safe but depends on vendor, hygiene varies",
        "potential_wrong_answer": "Might be overly cautious or overly permissive"
    },

    # ========== MISLEADING CULTURAL/HISTORICAL QUERIES (10 tests) ==========
    {
        "category": "misleading_culture",
        "input": "Is the Hagia Sophia a church or mosque?",
        "challenge": "Complex history - was church, then mosque, then museum, now mosque again",
        "potential_wrong_answer": "Might oversimplify current status or historical changes"
    },
    {
        "category": "misleading_culture",
        "input": "What's the dress code for visiting mosques?",
        "challenge": "Different requirements for men/women, varies by mosque",
        "potential_wrong_answer": "Might not specify gender differences or mosque variations"
    },
    {
        "category": "misleading_culture",
        "input": "Can non-Muslims visit all the mosques in Istanbul?",
        "challenge": "Most yes, but some restrictions during prayer times, special occasions",
        "potential_wrong_answer": "Might give blanket yes without mentioning restrictions"
    },
    {
        "category": "misleading_culture",
        "input": "What language do people speak in Istanbul?",
        "challenge": "Primarily Turkish, but Kurdish, Arabic, English varies by area",
        "potential_wrong_answer": "Might oversimplify linguistic diversity"
    },
    {
        "category": "misleading_culture",
        "input": "Is Istanbul in Europe or Asia?",
        "challenge": "City spans both continents, European and Asian sides",
        "potential_wrong_answer": "Might choose one or not explain the division clearly"
    },
    {
        "category": "misleading_culture",
        "input": "What's the best way to bargain in the Grand Bazaar?",
        "challenge": "Bargaining customs are nuanced, depends on item and vendor",
        "potential_wrong_answer": "Might give overly aggressive or passive bargaining advice"
    },
    {
        "category": "misleading_culture",
        "input": "Are there any areas I should avoid in Istanbul?",
        "challenge": "Safety varies by time, context, and personal factors",
        "potential_wrong_answer": "Might be overly alarming or dismissive of real concerns"
    },
    {
        "category": "misleading_culture",
        "input": "What's the currency in Turkey and can I use dollars?",
        "challenge": "Turkish Lira is currency, USD sometimes accepted but not recommended",
        "potential_wrong_answer": "Might overstate USD acceptance or not mention exchange rate issues"
    },
    {
        "category": "misleading_culture",
        "input": "How do I greet people politely in Turkish culture?",
        "challenge": "Varies by formality, age, gender, context",
        "potential_wrong_answer": "Might give one greeting without explaining contextual variations"
    },
    {
        "category": "misleading_culture",
        "input": "What should I know about Turkish coffee culture?",
        "challenge": "Rich traditions, specific preparation, social customs",
        "potential_wrong_answer": "Might focus only on coffee without cultural context"
    },

    # ========== TRICKY TIMING/SEASONAL QUERIES (10 tests) ==========
    {
        "category": "tricky_timing",
        "input": "What's open during Ramadan in Istanbul?",
        "challenge": "Complex - many places adjust hours, some close during day",
        "potential_wrong_answer": "Might oversimplify or not mention schedule changes"
    },
    {
        "category": "tricky_timing",
        "input": "Is it worth visiting Istanbul in winter?",
        "challenge": "Weather is mild but rainy, fewer crowds, mixed pros/cons",
        "potential_wrong_answer": "Might be too positive or negative without balanced view"
    },
    {
        "category": "tricky_timing",
        "input": "What time do museums close on weekends?",
        "challenge": "Different museums have different weekend schedules",
        "potential_wrong_answer": "Might give single time or assume all museums same"
    },
    {
        "category": "tricky_timing",
        "input": "How crowded is Istanbul during Turkish holidays?",
        "challenge": "Depends on specific holiday, local vs national holidays differ",
        "potential_wrong_answer": "Might not distinguish between different types of holidays"
    },
    {
        "category": "tricky_timing",
        "input": "What's the best time of day to visit the Grand Bazaar?",
        "challenge": "Depends on goals - fewer crowds vs full energy vs better prices",
        "potential_wrong_answer": "Might suggest one time without considering different priorities"
    },
    {
        "category": "tricky_timing",
        "input": "Do shops close for prayer time in Istanbul?",
        "challenge": "Some do, some don't, varies by neighborhood and business type",
        "potential_wrong_answer": "Might give blanket yes/no without explaining variation"
    },
    {
        "category": "tricky_timing",
        "input": "How early should I arrive at the airport for domestic flights?",
        "challenge": "Different from international, varies by airport IST vs SAW",
        "potential_wrong_answer": "Might give international timing or not distinguish airports"
    },
    {
        "category": "tricky_timing",
        "input": "What's the rush hour like in Istanbul?",
        "challenge": "Multiple rush hours, varies by route and transportation method",
        "potential_wrong_answer": "Might oversimplify or give only car traffic info"
    },
    {
        "category": "tricky_timing",
        "input": "Is Friday a good day to visit mosques?",
        "challenge": "Friday prayer is important, crowds and restrictions vary",
        "potential_wrong_answer": "Might not explain Friday prayer significance and timing"
    },
    {
        "category": "tricky_timing",
        "input": "How long does it take to see all the major sights?",
        "challenge": "Depends on travel style, interests, depth of visits",
        "potential_wrong_answer": "Might give single number without considering variables"
    }
]

def make_chat_request(message):
    """Send chat request to the backend"""
    try:
        response = requests.post(
            f"{BASE_URL}/ai",
            json={"user_input": message},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}

def analyze_response_quality(test_case, response_data):
    """Analyze if the response properly handles the challenging aspect"""
    if "error" in response_data:
        return {
            "status": "ERROR",
            "score": 0,
            "analysis": f"Request failed: {response_data['error']}"
        }
    
    response_text = response_data.get("response", "").lower()
    challenge = test_case["challenge"].lower()
    
    # Quality indicators
    quality_indicators = {
        "asks_for_clarification": any(phrase in response_text for phrase in [
            "which", "what do you mean", "could you specify", "can you tell me more",
            "there are several", "multiple options", "clarify", "more specific"
        ]),
        "acknowledges_complexity": any(phrase in response_text for phrase in [
            "depends", "varies", "several", "different", "multiple", "various",
            "it depends on", "there are many", "complexity", "complicated"
        ]),
        "provides_multiple_options": any(phrase in response_text for phrase in [
            "options", "choices", "alternatively", "or", "either", "both",
            "several ways", "different types", "various locations"
        ]),
        "shows_cultural_awareness": any(phrase in response_text for phrase in [
            "turkish", "local", "culture", "tradition", "custom", "etiquette",
            "locally", "in turkey", "istanbul culture"
        ]),
        "mentions_context_importance": any(phrase in response_text for phrase in [
            "time of day", "season", "budget", "preference", "depending on",
            "context", "situation", "circumstances"
        ])
    }
    
    # Score based on quality indicators
    score = 0
    analysis_points = []
    
    # Check if response handles the specific challenge
    if "ambiguous" in challenge and quality_indicators["asks_for_clarification"]:
        score += 30
        analysis_points.append("✓ Asks for clarification on ambiguous input")
    
    if "multiple" in challenge and quality_indicators["provides_multiple_options"]:
        score += 25
        analysis_points.append("✓ Provides multiple relevant options")
    
    if quality_indicators["acknowledges_complexity"]:
        score += 20
        analysis_points.append("✓ Acknowledges complexity/nuance")
    
    if quality_indicators["shows_cultural_awareness"]:
        score += 15
        analysis_points.append("✓ Shows cultural awareness")
    
    if quality_indicators["mentions_context_importance"]:
        score += 10
        analysis_points.append("✓ Mentions context importance")
    
    # Response length check (very short might be inadequate)
    if len(response_text) < 50:
        score -= 20
        analysis_points.append("⚠ Response seems too short")
    
    # Determine overall status
    if score >= 80:
        status = "EXCELLENT"
    elif score >= 60:
        status = "GOOD" 
    elif score >= 40:
        status = "ADEQUATE"
    elif score >= 20:
        status = "POOR"
    else:
        status = "VERY_POOR"
    
    return {
        "status": status,
        "score": score,
        "analysis": "; ".join(analysis_points) if analysis_points else "No quality indicators found"
    }

def run_challenging_test_suite():
    """Run the challenging test suite"""
    print("🧪 Starting AI Istanbul Challenging Test Suite")
    print(f"📊 Testing {len(challenging_test_cases)} challenging inputs")
    print("=" * 80)
    
    results = {
        "total_tests": len(challenging_test_cases),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "categories": {},
        "detailed_results": [],
        "timestamp": datetime.now().isoformat()
    }
    
    for i, test_case in enumerate(challenging_test_cases, 1):
        category = test_case["category"]
        if category not in results["categories"]:
            results["categories"][category] = {"total": 0, "passed": 0, "scores": []}
        
        print(f"\n[{i:2d}/50] Testing: {category}")
        print(f"Input: {test_case['input']}")
        print(f"Challenge: {test_case['challenge']}")
        
        # Make request
        response_data = make_chat_request(test_case["input"])
        
        # Analyze response
        analysis = analyze_response_quality(test_case, response_data)
        
        # Store results
        test_result = {
            "test_number": i,
            "category": category,
            "input": test_case["input"],
            "challenge": test_case["challenge"],
            "potential_wrong_answer": test_case["potential_wrong_answer"],
            "response": response_data.get("response", ""),
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        results["detailed_results"].append(test_result)
        
        # Update counters
        results["categories"][category]["total"] += 1
        results["categories"][category]["scores"].append(analysis["score"])
        
        if analysis["status"] == "ERROR":
            results["errors"] += 1
            print(f"❌ ERROR: {analysis['analysis']}")
        elif analysis["score"] >= 60:
            results["passed"] += 1
            results["categories"][category]["passed"] += 1
            print(f"✅ {analysis['status']} (Score: {analysis['score']}/100)")
            print(f"Analysis: {analysis['analysis']}")
        else:
            results["failed"] += 1
            print(f"❌ {analysis['status']} (Score: {analysis['score']}/100)")
            print(f"Analysis: {analysis['analysis']}")
        
        # Show response preview
        response_preview = (response_data.get("response", "")[:100] + "...") if len(response_data.get("response", "")) > 100 else response_data.get("response", "")
        print(f"Response preview: {response_preview}")
        
        # Brief pause between requests
        time.sleep(0.5)
    
    # Calculate summary statistics
    print("\n" + "=" * 80)
    print("📊 CHALLENGING TEST SUITE RESULTS SUMMARY")
    print("=" * 80)
    
    total_score = sum(r["analysis"]["score"] for r in results["detailed_results"] if r["analysis"]["status"] != "ERROR")
    avg_score = total_score / max(1, results["total_tests"] - results["errors"])
    
    print(f"Total Tests: {results['total_tests']}")
    print(f"Passed (Score ≥60): {results['passed']} ({results['passed']/results['total_tests']*100:.1f}%)")
    print(f"Failed (Score <60): {results['failed']} ({results['failed']/results['total_tests']*100:.1f}%)")
    print(f"Errors: {results['errors']} ({results['errors']/results['total_tests']*100:.1f}%)")
    print(f"Average Score: {avg_score:.1f}/100")
    
    # Category breakdown
    print(f"\n📈 CATEGORY BREAKDOWN:")
    for category, stats in results["categories"].items():
        if stats["scores"]:
            cat_avg = sum(stats["scores"]) / len(stats["scores"])
            pass_rate = (stats["passed"] / stats["total"]) * 100
            print(f"  {category:20} | {stats['passed']:2d}/{stats['total']:2d} passed ({pass_rate:4.1f}%) | Avg: {cat_avg:4.1f}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"challenging_inputs_test_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    
    # Show some examples of responses that handled challenges well
    print(f"\n🏆 TOP PERFORMING RESPONSES:")
    top_responses = sorted(results["detailed_results"], 
                         key=lambda x: x["analysis"]["score"], reverse=True)[:3]
    
    for i, result in enumerate(top_responses, 1):
        if result["analysis"]["status"] != "ERROR":
            print(f"\n{i}. Input: {result['input'][:60]}...")
            print(f"   Challenge: {result['challenge'][:80]}...")
            print(f"   Score: {result['analysis']['score']}/100 ({result['analysis']['status']})")
            print(f"   Why it scored well: {result['analysis']['analysis']}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_challenging_test_suite()
        
        # Exit with appropriate code
        if results["errors"] > results["total_tests"] * 0.1:  # More than 10% errors
            sys.exit(2)
        elif results["passed"] / results["total_tests"] < 0.6:  # Less than 60% pass rate
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        sys.exit(1)

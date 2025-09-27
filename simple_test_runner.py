#!/usr/bin/env python3
"""
AI Istanbul Simple Test Runner
==============================

A simplified version that tests your AI system with the 80 inputs
and provides basic analysis of response quality.
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List

# Configuration
API_BASE_URL = "http://localhost:8001"  # Updated to match your backend
CHAT_ENDPOINT = f"{API_BASE_URL}/ai/chat"

def test_api_connection():
    """Test if the API is accessible."""
    try:
        health_endpoint = f"{API_BASE_URL}/health"
        response = requests.get(health_endpoint, timeout=5)
        return response.status_code == 200
    except:
        return False

def send_chat_message(message: str, session_id: str = "test_session") -> Dict:
    """Send a message to the AI chat API."""
    try:
        payload = {
            "message": message,
            "session_id": session_id
        }
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            return {
                "success": True,
                "response": response.json().get("response", ""),
                "status": response.status_code
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "status": response.status_code
            }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "status": 0
        }

def validate_factual_accuracy(response: str, test_input: str, category: str) -> Dict:
    """Validate factual accuracy of the AI response."""
    accuracy_score = 0
    accuracy_issues = []
    factual_errors = []
    correct_facts = []
    
    response_lower = response.lower()
    
    # Transportation Facts Validation
    if category == "Transportation":
        # Istanbul Airport to Sultanahmet
        if "istanbul airport" in test_input.lower() and "sultanahmet" in test_input.lower():
            if "havabus" in response_lower or "havaist" in response_lower:
                correct_facts.append("Mentions HAVABUS/HAVAIST airport shuttle")
                accuracy_score += 15
            else:
                factual_errors.append("Missing HAVABUS/HAVAIST mention for airport transfer")
                
            if "kabataş" in response_lower and "tram" in response_lower:
                correct_facts.append("Correct Kabataş-Sultanahmet tram connection")
                accuracy_score += 15
            else:
                factual_errors.append("Missing Kabataş tram connection info")
                
            if "m1a" in response_lower or "zeytinburnu" in response_lower:
                correct_facts.append("Mentions M1A metro line")
                accuracy_score += 10
                
        # Ferry information
        if "ferry" in test_input.lower():
            if "şehir hatları" in response_lower or "sehir hatlari" in response_lower:
                correct_facts.append("Mentions Şehir Hatları ferry company")
                accuracy_score += 10
            if "eminönü" in response_lower and "üsküdar" in response_lower:
                correct_facts.append("Correct ferry route mentioned")
                accuracy_score += 10
                
        # Metro system
        if "metro" in test_input.lower():
            if "european" in test_input.lower() and "asian" in test_input.lower():
                if "no direct" in response_lower or "marmaray" in response_lower:
                    correct_facts.append("Correctly states no direct metro between sides")
                    accuracy_score += 15
                elif "direct metro" in response_lower and "marmaray" not in response_lower:
                    factual_errors.append("Incorrectly suggests direct metro between continents")
                    
        # Istanbul Kart
        if "istanbul kart" in test_input.lower() or "istanbulkart" in test_input.lower():
            if "metro station" in response_lower and ("kiosk" in response_lower or "büfe" in response_lower):
                correct_facts.append("Correct Istanbul Kart purchase locations")
                accuracy_score += 10
                
        # Night transportation
        if "night" in test_input.lower() and "transport" in test_input.lower():
            if "limited" in response_lower or "reduced" in response_lower:
                correct_facts.append("Correctly mentions limited night service")
                accuracy_score += 10
            elif "24 hour" in response_lower or "all night" in response_lower:
                factual_errors.append("Incorrectly suggests extensive night service")
    
    # Restaurant & Food Facts Validation
    elif category == "Restaurant & Food":
        # Turkish breakfast
        if "turkish breakfast" in test_input.lower():
            if "kahvaltı" in response_lower:
                correct_facts.append("Uses correct Turkish term 'kahvaltı'")
                accuracy_score += 10
            if any(item in response_lower for item in ["cheese", "olives", "tomatoes", "cucumber", "honey", "jam"]):
                correct_facts.append("Mentions typical breakfast items")
                accuracy_score += 10
                
        # Vegetarian Turkish food
        if "vegetarian" in test_input.lower():
            if "zeytinyağlı" in response_lower:
                correct_facts.append("Mentions zeytinyağlı (olive oil) dishes")
                accuracy_score += 15
            if "dolma" in response_lower or "sarma" in response_lower:
                correct_facts.append("Mentions vegetarian dolma/sarma")
                accuracy_score += 10
            if "meat" in response_lower and ("avoid" in response_lower or "careful" in response_lower):
                correct_facts.append("Warns about meat in Turkish cuisine")
                accuracy_score += 5
                
        # Turkish desserts
        if "dessert" in test_input.lower():
            if "baklava" in response_lower:
                correct_facts.append("Mentions baklava")
                accuracy_score += 10
            if "künefe" in response_lower or "kunefe" in response_lower:
                correct_facts.append("Mentions künefe")
                accuracy_score += 10
            if "lokum" in response_lower or "turkish delight" in response_lower:
                correct_facts.append("Mentions Turkish delight/lokum")
                accuracy_score += 10
                
        # Water safety
        if "tap water" in test_input.lower():
            if "safe" in response_lower and "istanbul" in response_lower:
                correct_facts.append("Correctly states tap water is generally safe")
                accuracy_score += 15
            elif "not safe" in response_lower or "avoid" in response_lower:
                factual_errors.append("Incorrectly suggests tap water is unsafe")
                
        # Tipping
        if "tip" in test_input.lower() and "restaurant" in test_input.lower():
            if any(percent in response_lower for percent in ["10%", "10 percent", "5-10"]):
                correct_facts.append("Correct tipping percentage (5-10%)")
                accuracy_score += 15
            elif any(percent in response_lower for percent in ["15%", "20%", "25%"]):
                factual_errors.append("Suggests too high tipping (Western standards)")
    
    # Museums & Cultural Sites Facts Validation
    elif category == "Museums & Cultural Sites":
        # Hagia Sophia
        if "hagia sophia" in test_input.lower():
            if "mosque" in response_lower and ("free" in response_lower or "no ticket" in response_lower):
                correct_facts.append("Correctly states Hagia Sophia is now a mosque (free entry)")
                accuracy_score += 20
            elif "ticket" in response_lower and "price" in response_lower:
                factual_errors.append("Incorrectly suggests Hagia Sophia requires tickets")
            if "prayer time" in response_lower or "namaz" in response_lower:
                correct_facts.append("Mentions prayer times restriction")
                accuracy_score += 10
                
        # Blue Mosque
        if "blue mosque" in test_input.lower():
            if "six minarets" in response_lower or "6 minarets" in response_lower:
                correct_facts.append("Correctly mentions six minarets")
                accuracy_score += 15
            if "blue tiles" in response_lower or "iznik tiles" in response_lower:
                correct_facts.append("Mentions blue tiles/Iznik ceramics")
                accuracy_score += 10
                
        # Topkapi Palace
        if "topkapi" in test_input.lower():
            if "ottoman" in response_lower and ("palace" in response_lower or "sultan" in response_lower):
                correct_facts.append("Correctly identifies as Ottoman palace")
                accuracy_score += 10
            if "harem" in response_lower and "treasury" in response_lower:
                correct_facts.append("Mentions key sections (harem, treasury)")
                accuracy_score += 15
                
        # Basilica Cistern
        if "basilica cistern" in test_input.lower() or "yerebatan" in test_input.lower():
            if "medusa" in response_lower:
                correct_facts.append("Mentions Medusa heads")
                accuracy_score += 10
            if "underground" in response_lower and "column" in response_lower:
                correct_facts.append("Describes underground structure with columns")
                accuracy_score += 10
    
    # Districts & Neighborhoods Facts Validation
    elif category == "Districts & Neighborhoods":
        # Sultanahmet
        if "sultanahmet" in test_input.lower():
            if "historical" in response_lower and ("hagia sophia" in response_lower or "blue mosque" in response_lower):
                correct_facts.append("Correctly identifies historical significance")
                accuracy_score += 15
                
        # European vs Asian sides
        if "european" in test_input.lower() and "asian" in test_input.lower():
            if "bosphorus" in response_lower:
                correct_facts.append("Mentions Bosphorus as divider")
                accuracy_score += 10
            if "ferry" in response_lower or "bridge" in response_lower:
                correct_facts.append("Mentions crossing methods")
                accuracy_score += 10
                
        # Kadıköy
        if "kadıköy" in test_input.lower() or "kadikoy" in test_input.lower():
            if "asian side" in response_lower:
                correct_facts.append("Correctly places Kadıköy on Asian side")
                accuracy_score += 10
            if "ferry" in response_lower and "eminönü" in response_lower:
                correct_facts.append("Mentions ferry connection to Eminönü")
                accuracy_score += 10
    
    # General Tips & Practical Facts Validation
    elif category == "General Tips & Practical":
        # Weather in March
        if "march" in test_input.lower() and "weather" in test_input.lower():
            if any(temp in response_lower for temp in ["mild", "cool", "10-15", "15-20", "moderate"]):
                correct_facts.append("Reasonable March temperature description")
                accuracy_score += 10
            if "rain" in response_lower or "variable" in response_lower:
                correct_facts.append("Mentions spring weather variability")
                accuracy_score += 10
                
        # Solo female travel safety
        if "solo female" in test_input.lower():
            if "generally safe" in response_lower or "relatively safe" in response_lower:
                correct_facts.append("Balanced safety assessment")
                accuracy_score += 10
            if "precaution" in response_lower and "conservative dress" in response_lower:
                correct_facts.append("Mentions cultural considerations")
                accuracy_score += 10
                
        # Emergency numbers
        if "emergency" in test_input.lower() and "number" in test_input.lower():
            if "112" in response_lower:
                correct_facts.append("Correct emergency number (112)")
                accuracy_score += 20
            elif "911" in response_lower or "999" in response_lower:
                factual_errors.append("Incorrect emergency number (not Turkish)")
                
        # Money exchange
        if "money" in test_input.lower() and "exchange" in test_input.lower():
            if "official" in response_lower and ("bank" in response_lower or "exchange office" in response_lower):
                correct_facts.append("Recommends official exchange sources")
                accuracy_score += 10
            if "street" in response_lower and ("avoid" in response_lower or "scam" in response_lower):
                correct_facts.append("Warns against street exchanges")
                accuracy_score += 10
    
    # Calculate final accuracy score
    max_possible_score = 100
    accuracy_score = min(accuracy_score, max_possible_score)
    
    if factual_errors:
        accuracy_score -= len(factual_errors) * 10  # Penalty for factual errors
        accuracy_score = max(0, accuracy_score)  # Don't go below 0
    
    return {
        "accuracy_score": accuracy_score,
        "correct_facts": correct_facts,
        "factual_errors": factual_errors,
        "accuracy_issues": accuracy_issues
    }

def analyze_response_quality(response: str, test_input: str, category: str) -> Dict:
    """Comprehensive analysis of response quality including factual accuracy."""
    if not response:
        return {
            "score": 0, 
            "issues": ["Empty response"],
            "accuracy_score": 0,
            "correct_facts": [],
            "factual_errors": ["No response provided"]
        }
    
    # Basic quality metrics
    score = 30  # Reduced base score to account for accuracy
    issues = []
    
    # Length check (15 points)
    word_count = len(response.split())
    if word_count < 10:
        issues.append("Response too short")
        score -= 15
    elif word_count > 500:
        issues.append("Response too long - may be verbose")
        score -= 5
    else:
        score += 15
    
    # Cultural sensitivity (10 points)
    cultural_words = ["respect", "cultural", "traditional", "local", "Turkish", "Islamic", "etiquette", "custom", "polite"]
    cultural_score = sum(1 for word in cultural_words if word.lower() in response.lower())
    if cultural_score >= 2:
        score += 10
    elif cultural_score >= 1:
        score += 5
    else:
        issues.append("Limited cultural sensitivity shown")
    
    # Practical usefulness (15 points)
    practical_words = ["recommend", "suggest", "located", "address", "time", "cost", "price", "how to", "where", "when"]
    practical_score = sum(1 for word in practical_words if word.lower() in response.lower())
    if practical_score >= 3:
        score += 15
    elif practical_score >= 1:
        score += 8
    else:
        issues.append("Limited practical information provided")
        score -= 5
    
    # Validate factual accuracy (30 points)
    accuracy_analysis = validate_factual_accuracy(response, test_input, category)
    factual_accuracy_score = min(30, accuracy_analysis["accuracy_score"] * 30 // 100)
    score += factual_accuracy_score
    
    # Structure and helpfulness (10 points)
    if any(indicator in response.lower() for indicator in ["1.", "2.", "•", "-", "first", "second", "also", "additionally"]):
        score += 5  # Well-structured response
    if any(tone in response.lower() for tone in ["help", "assist", "guide", "recommend", "suggest"]):
        score += 5  # Helpful tone
    
    # Penalty for obvious errors or problematic content
    error_indicators = ["i don't know", "i'm not sure", "cannot provide", "sorry, i can't", "no information"]
    if any(error in response.lower() for error in error_indicators):
        score -= 20
        issues.append("Response indicates lack of knowledge")
    
    # Cap score at 100
    score = min(100, max(0, score))
    
    # Combine all analysis
    return {
        "score": score,
        "word_count": word_count,
        "issues": issues,
        "accuracy_score": accuracy_analysis["accuracy_score"],
        "correct_facts": accuracy_analysis["correct_facts"],
        "factual_errors": accuracy_analysis["factual_errors"],
        "cultural_sensitivity": cultural_score,
        "practical_usefulness": practical_score,
        "factual_accuracy_contribution": factual_accuracy_score
    }

def run_test_suite():
    """Run the complete test suite."""
    
    # All 80 test inputs organized by category
    test_inputs = {
        "Transportation": [
            "How do I get from Istanbul Airport to Sultanahmet?",
            "What's the best way to get from Taksim to Grand Bazaar?",
            "I want to take a ferry from Eminönü to Üsküdar. What's the schedule and cost?",
            "How can I get to Büyükada (Prince Islands) and what transport is available on the island?",
            "Is there a direct metro connection between European and Asian sides?",
            "What's the cheapest way to travel around Istanbul for a week?",
            "How do I get to Sabiha Gökçen Airport from Kadıköy at 4 AM?",
            "Can you explain the difference between dolmuş, minibüs, and regular buses?",
            "How does the tram system work in Istanbul?",
            "I need to get from Atatürk Airport area to Asian side during rush hour. Best route?",
            "What are the main ferry routes and which one is most scenic?",
            "How do I use Istanbul Kart and where can I buy one?",
            "What's the night transportation situation in Istanbul?",
            "How do I get to Belgrade Forest from city center?",
            "Are taxis expensive in Istanbul and how do I avoid scams?",
            "What's the best way to do a Bosphorus tour including both continents?"
        ],
        "Restaurant & Food": [
            "Where can I find the best Turkish breakfast in Sultanahmet?",
            "I'm vegetarian. What traditional Turkish dishes can I eat?",
            "Can you recommend high-end Ottoman cuisine restaurants with historical ambiance?",
            "What street foods should I try and where are they safe to eat?",
            "I have celiac disease. Can you suggest gluten-free restaurants in Beyoğlu?",
            "What's the best Turkish dessert and where can I find it?",
            "I want to experience a traditional Turkish cooking class. Where can I find authentic ones?",
            "What's the difference between Turkish coffee houses in different districts?",
            "Can you recommend good seafood restaurants near the Bosphorus?",
            "I'm interested in the cultural significance of Turkish tea culture. Where can I experience it authentically?",
            "What are the best food markets in Istanbul and what should I buy?",
            "Is tap water safe to drink in Istanbul restaurants?",
            "Can you explain the etiquette and customs around dining in Turkish homes vs restaurants?",
            "What are the best budget-friendly local food spots that tourists usually miss?",
            "How much should I tip in Turkish restaurants?",
            "I want to understand regional Turkish cuisine differences. What should I look for in Istanbul?"
        ],
        "Museums & Cultural Sites": [
            "What are the opening hours and ticket prices for Hagia Sophia?",
            "Can you explain the historical significance of Topkapi Palace and what to prioritize during a visit?",
            "I'm interested in Byzantine history. Beyond Hagia Sophia, what lesser-known sites should I visit?",
            "What's the difference between the Blue Mosque and other mosques in Istanbul?",
            "Are there any good art museums showcasing contemporary Turkish art?",
            "Can you recommend a cultural itinerary that shows Istanbul's evolution from Byzantine to Ottoman to modern?",
            "What should I know before visiting the Grand Bazaar?",
            "How can I learn about Ottoman architecture while exploring Istanbul?",
            "What are some hidden architectural gems that showcase Istanbul's multicultural past?",
            "Is the Basilica Cistern worth visiting and what should I expect?",
            "Can you suggest museums that are good for families with children?",
            "I'm researching Islamic calligraphy and ceramics. Which museums have the best collections?",
            "What's the best way to avoid crowds at popular tourist sites?",
            "Are there any archaeological sites within Istanbul city limits?",
            "How has Istanbul's cultural landscape changed in the past decade?",
            "What are the must-see cultural sites for a first-time visitor with only 2 days?"
        ],
        "Districts & Neighborhoods": [
            "What's special about Sultanahmet district and what can I find there?",
            "I want to experience local life away from tourists. Which neighborhoods should I explore?",
            "Can you explain the character differences between Beyoğlu, Beşiktaş, and Şişli?",
            "What can I do in Kadıköy on the Asian side?",
            "Is Galata area worth staying in and what's the neighborhood like?",
            "I'm interested in Istanbul's gentrification process. Which areas are currently changing?",
            "What's the best area for nightlife and entertainment?",
            "Can you recommend family-friendly neighborhoods to explore with children?",
            "What's the socioeconomic profile of different Istanbul districts?",
            "Which area has the best shopping opportunities?",
            "What's unique about the Bosphorus waterfront neighborhoods?",
            "How do the European and Asian sides of Istanbul differ culturally and socially?",
            "Is it safe to walk around different neighborhoods at night?",
            "Which neighborhoods are best for street photography and why?",
            "How has neighborhood character in Istanbul changed due to Syrian refugee influx?",
            "What are the main characteristics of Ortaköy district?"
        ],
        "General Tips & Practical": [
            "What's the weather like in Istanbul in March and what should I pack?",
            "What are the most important cultural etiquette rules I should follow?",
            "How do I navigate bureaucracy if I need to extend my visa or handle official matters?",
            "Is it safe for solo female travelers in Istanbul?",
            "What are the key Turkish phrases I should learn for daily interactions?",
            "How do I understand and respect Islamic customs during Ramadan?",
            "What's the best way to exchange money and avoid scams?",
            "How widespread is English and how can I communicate effectively?",
            "What should I know about Turkish business culture if I'm here for work?",
            "Are there any cultural taboos or things I should definitely avoid doing?",
            "What's the healthcare system like and how can I access medical care as a tourist?",
            "How do I understand and navigate Turkish social hierarchies and respect systems?",
            "What are the emergency numbers and basic safety information I should know?",
            "How do I handle haggling and price negotiations in markets?",
            "What are the environmental and sustainability challenges facing Istanbul?",
            "What should I do if I lose my passport or have other travel document emergencies?"
        ]
    }
    
    print("🇹🇷 AI Istanbul Test Suite - 80 Inputs")
    print("=" * 60)
    
    # Check API connection
    if not test_api_connection():
        print("❌ Cannot connect to API. Please check:")
        print(f"   1. Backend is running on {API_BASE_URL}")
        print("   2. Network connectivity")
        print("   3. API endpoints are accessible")
        return
    
    print("✅ API connection successful")
    print(f"🔗 Testing endpoint: {CHAT_ENDPOINT}")
    
    start_time = datetime.now()
    results = {
        "test_info": {
            "start_time": start_time.isoformat(),
            "total_tests": 80,
            "api_endpoint": CHAT_ENDPOINT
        },
        "category_results": {},
        "detailed_results": [],
        "summary": {}
    }
    
    test_number = 1
    
    for category, inputs in test_inputs.items():
        print(f"\n🧪 Testing {category} ({len(inputs)} tests)")
        category_scores = []
        category_results = []
        
        for i, test_input in enumerate(inputs, 1):
            print(f"   Test {test_number}: ", end="", flush=True)
            
            # Send message to AI
            response_data = send_chat_message(test_input, f"test_session_{test_number}")
            
            if response_data["success"]:
                ai_response = response_data["response"]
                analysis = analyze_response_quality(ai_response, test_input, category)
                
                test_result = {
                    "test_number": test_number,
                    "category": category,
                    "input": test_input,
                    "ai_response": ai_response,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
                
                category_results.append(test_result)
                category_scores.append(analysis["score"])
                results["detailed_results"].append(test_result)
                
                print(f"✅ Score: {analysis['score']}/100")
                
                if analysis["issues"]:
                    print(f"      Issues: {', '.join(analysis['issues'])}")
            else:
                print(f"❌ Error: {response_data['error']}")
                test_result = {
                    "test_number": test_number,
                    "category": category,
                    "input": test_input,
                    "error": response_data["error"],
                    "analysis": {"score": 0, "issues": ["API Error"]},
                    "timestamp": datetime.now().isoformat()
                }
                category_results.append(test_result)
                category_scores.append(0)
                results["detailed_results"].append(test_result)
            
            test_number += 1
            time.sleep(1)  # Brief pause between requests
        
        # Calculate category statistics
        if category_scores:
            avg_score = sum(category_scores) / len(category_scores)
            passed_tests = len([s for s in category_scores if s >= 70])
            pass_rate = (passed_tests / len(category_scores)) * 100
        else:
            avg_score = 0
            pass_rate = 0
        
        results["category_results"][category] = {
            "average_score": avg_score,
            "pass_rate": pass_rate,
            "total_tests": len(inputs),
            "scores": category_scores,
            "tests": category_results
        }
        
        print(f"   📊 Category Average: {avg_score:.1f}/100 ({pass_rate:.1f}% pass rate)")
    
    # Generate overall summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    all_scores = [r["analysis"]["score"] for r in results["detailed_results"] 
                  if "analysis" in r and "score" in r["analysis"]]
    
    if all_scores:
        overall_avg = sum(all_scores) / len(all_scores)
        overall_passed = len([s for s in all_scores if s >= 70])
        overall_pass_rate = (overall_passed / len(all_scores)) * 100
    else:
        overall_avg = 0
        overall_pass_rate = 0
    
    results["summary"] = {
        "end_time": end_time.isoformat(),
        "duration": str(duration),
        "overall_average": overall_avg,
        "overall_pass_rate": overall_pass_rate,
        "total_passed": overall_passed if all_scores else 0,
        "total_failed": len(all_scores) - overall_passed if all_scores else 0
    }
    
    # Print final summary
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"⏱️  Duration: {duration}")
    print(f"🎯 Overall Average: {overall_avg:.1f}/100")
    print(f"✅ Overall Pass Rate: {overall_pass_rate:.1f}%")
    print(f"📈 Tests Passed: {overall_passed if all_scores else 0}/80")
    
    print(f"\n📋 CATEGORY BREAKDOWN:")
    for category, stats in results["category_results"].items():
        print(f"   {category}: {stats['average_score']:.1f}/100 ({stats['pass_rate']:.1f}% pass)")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"ai_istanbul_test_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if overall_avg < 60:
        print("   🚨 Critical: Overall performance below acceptable threshold")
        print("   📈 Focus on improving response accuracy and completeness")
    elif overall_avg < 70:
        print("   ⚠️  Performance needs improvement in several areas")
    elif overall_avg < 80:
        print("   ✅ Good performance with room for optimization")
    else:
        print("   🏆 Excellent performance across all categories")
    
    # Category-specific recommendations
    for category, stats in results["category_results"].items():
        if stats["average_score"] < 60:
            print(f"   🔧 Priority improvement needed: {category}")
        elif stats["average_score"] < 70:
            print(f"   📝 Minor improvements suggested: {category}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_test_suite()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running test suite: {e}")
        print("Please check your backend configuration and try again.")

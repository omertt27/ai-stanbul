#!/usr/bin/env python3
"""
AI Istanbul 80-Input Test Suite - Manual Testing Version
=======================================================

This script provides 80 comprehensive test inputs organized by category.
Use this for manual testing of your AI Istanbul chatbot.

Categories:
- Transportation (16 tests)
- Restaurants & Food (16 tests) 
- Museums & Cultural Sites (16 tests)
- Districts & Neighborhoods (16 tests)
- General Tips & Practical (16 tests)

Usage:
1. Run the script to see all test inputs
2. Copy individual test inputs to your chatbot
3. Evaluate responses based on the provided criteria
4. Use the validation checklist for each response
"""

import json
from datetime import datetime

class ManualTestSuite:
    
    # Transportation Tests (16)
    TRANSPORTATION_INPUTS = [
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
    ]

    # Restaurant & Food Tests (16)
    RESTAURANT_FOOD_INPUTS = [
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
    ]

    # Museums & Cultural Sites Tests (16)
    MUSEUM_CULTURAL_INPUTS = [
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
    ]

    # Districts & Neighborhoods Tests (16)  
    DISTRICTS_INPUTS = [
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
    ]

    # General Tips & Practical Tests (16)
    GENERAL_PRACTICAL_INPUTS = [
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

    def __init__(self):
        self.manual_results = []

    def print_category_tests(self, category_name: str, inputs: list):
        """Print all test inputs for a category."""
        print(f"\n{'='*20} {category_name.upper()} TESTS {'='*20}")
        for i, test_input in enumerate(inputs, 1):
            print(f"\n🧪 TEST #{i}: {test_input}")
            print("📝 Instructions:")
            print("   1. Copy the above question to your AI chat")
            print("   2. Analyze the response for accuracy and completeness")
            print("   3. Rate the response (1-10)")
            print("   4. Note any missing information or errors")
            print("-" * 60)

    def print_all_tests(self):
        """Print all 80 test inputs organized by category."""
        print("🇹🇷 AI ISTANBUL COMPREHENSIVE TEST SUITE - 80 INPUTS 🇹🇷")
        print("=" * 80)
        print("📋 MANUAL TESTING INSTRUCTIONS:")
        print("1. Test each input with your AI Istanbul chatbot")
        print("2. Rate responses on a scale of 1-10")
        print("3. Note specific issues or missing information")
        print("4. Focus on accuracy, completeness, and cultural sensitivity")
        print("5. Check for practical usefulness and actionable advice")
        
        categories = [
            ("Transportation", self.TRANSPORTATION_INPUTS),
            ("Restaurants & Food", self.RESTAURANT_FOOD_INPUTS), 
            ("Museums & Cultural Sites", self.MUSEUM_CULTURAL_INPUTS),
            ("Districts & Neighborhoods", self.DISTRICTS_INPUTS),
            ("General Tips & Practical", self.GENERAL_PRACTICAL_INPUTS)
        ]
        
        test_number = 1
        for category_name, inputs in categories:
            print(f"\n{'🚌' if 'Transportation' in category_name else '🍽️' if 'Restaurant' in category_name else '🏛️' if 'Museums' in category_name else '🏘️' if 'Districts' in category_name else '💡'} {category_name.upper()}")
            print("=" * 50)
            
            for test_input in inputs:
                print(f"\n📝 TEST #{test_number}: {test_input}")
                print("─" * 40)
                test_number += 1
        
        print(f"\n✅ Total Tests: {test_number - 1}")

    def create_evaluation_checklist(self):
        """Create a checklist for evaluating AI responses."""
        checklist = {
            "response_evaluation_criteria": {
                "accuracy": {
                    "description": "Information is factually correct",
                    "score_range": "1-3 points",
                    "questions": [
                        "Are locations, prices, and times accurate?",
                        "Is cultural information correct?",
                        "Are transportation details up-to-date?"
                    ]
                },
                "completeness": {
                    "description": "Response addresses all aspects of the question",
                    "score_range": "1-3 points", 
                    "questions": [
                        "Does it answer all parts of the question?",
                        "Are important details included?",
                        "Is context provided when needed?"
                    ]
                },
                "cultural_sensitivity": {
                    "description": "Shows awareness and respect for Turkish culture",
                    "score_range": "1-2 points",
                    "questions": [
                        "Is cultural context appropriately included?",
                        "Are local customs respected?",
                        "Is language culturally appropriate?"
                    ]
                },
                "practical_usefulness": {
                    "description": "Provides actionable, helpful information",
                    "score_range": "1-2 points",
                    "questions": [
                        "Can the user act on this information?",
                        "Are specific recommendations provided?",
                        "Is the advice practical for tourists?"
                    ]
                }
            },
            "scoring_guide": {
                "9-10": "Excellent - Comprehensive, accurate, culturally aware",
                "7-8": "Very Good - Mostly accurate with minor gaps",
                "5-6": "Good - Adequate but missing some important elements", 
                "3-4": "Fair - Basic response with significant gaps",
                "1-2": "Poor - Inaccurate or inadequate response"
            },
            "common_issues_to_watch": [
                "Outdated information (prices, schedules, closures)",
                "Generic responses not specific to Istanbul",
                "Missing cultural context or sensitivity",
                "Impractical or vague recommendations", 
                "Failure to address safety or etiquette concerns",
                "Not considering seasonal variations",
                "Ignoring budget or accessibility concerns"
            ]
        }
        
        return checklist

    def save_manual_test_template(self):
        """Save a template for manual test recording."""
        template = {
            "test_session_info": {
                "date": datetime.now().isoformat(),
                "tester_name": "YOUR_NAME",
                "ai_system_version": "VERSION_INFO",
                "test_environment": "TESTING_CONDITIONS"
            },
            "category_results": {
                "transportation": {"tests": [], "category_score": 0, "notes": ""},
                "restaurants_food": {"tests": [], "category_score": 0, "notes": ""},
                "museums_cultural": {"tests": [], "category_score": 0, "notes": ""},
                "districts": {"tests": [], "category_score": 0, "notes": ""},
                "general_practical": {"tests": [], "category_score": 0, "notes": ""}
            },
            "test_template": {
                "test_number": 0,
                "category": "",
                "input": "",
                "ai_response": "",
                "score": 0,
                "accuracy": 0,
                "completeness": 0, 
                "cultural_sensitivity": 0,
                "practical_usefulness": 0,
                "notes": "",
                "issues_found": [],
                "recommendations": ""
            },
            "evaluation_checklist": self.create_evaluation_checklist()
        }
        
        filename = f"ai_istanbul_manual_test_template_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        return filename

    def get_test_by_number(self, test_num: int) -> str:
        """Get a specific test input by number (1-80)."""
        all_inputs = (self.TRANSPORTATION_INPUTS + self.RESTAURANT_FOOD_INPUTS + 
                     self.MUSEUM_CULTURAL_INPUTS + self.DISTRICTS_INPUTS + 
                     self.GENERAL_PRACTICAL_INPUTS)
        
        if 1 <= test_num <= len(all_inputs):
            return all_inputs[test_num - 1]
        else:
            return f"Invalid test number. Please choose between 1 and {len(all_inputs)}"

    def get_category_tests(self, category: str) -> list:
        """Get all tests for a specific category."""
        categories = {
            "transportation": self.TRANSPORTATION_INPUTS,
            "restaurants": self.RESTAURANT_FOOD_INPUTS,
            "food": self.RESTAURANT_FOOD_INPUTS,
            "museums": self.MUSEUM_CULTURAL_INPUTS,
            "cultural": self.MUSEUM_CULTURAL_INPUTS,
            "districts": self.DISTRICTS_INPUTS,
            "neighborhoods": self.DISTRICTS_INPUTS,
            "general": self.GENERAL_PRACTICAL_INPUTS,
            "practical": self.GENERAL_PRACTICAL_INPUTS,
            "tips": self.GENERAL_PRACTICAL_INPUTS
        }
        
        return categories.get(category.lower(), [])

def main():
    """Main function for manual testing."""
    test_suite = ManualTestSuite()
    
    print("🚀 AI Istanbul Manual Test Suite")
    print("Choose an option:")
    print("1. View all 80 tests")
    print("2. View tests by category") 
    print("3. Get specific test by number")
    print("4. Create evaluation template")
    print("5. Show evaluation criteria")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            test_suite.print_all_tests()
        
        elif choice == "2":
            print("\nAvailable categories:")
            print("- transportation")
            print("- restaurants/food") 
            print("- museums/cultural")
            print("- districts/neighborhoods")
            print("- general/practical/tips")
            
            category = input("Enter category: ").strip()
            tests = test_suite.get_category_tests(category)
            if tests:
                test_suite.print_category_tests(category, tests)
            else:
                print("Category not found!")
        
        elif choice == "3":
            test_num = int(input("Enter test number (1-80): "))
            test_input = test_suite.get_test_by_number(test_num)
            print(f"\n🧪 TEST #{test_num}: {test_input}")
        
        elif choice == "4":
            filename = test_suite.save_manual_test_template()
            print(f"📄 Template saved as: {filename}")
            print("Use this template to record your manual test results.")
        
        elif choice == "5":
            checklist = test_suite.create_evaluation_checklist()
            print("\n📋 RESPONSE EVALUATION CRITERIA")
            print("=" * 50)
            for criteria, details in checklist["response_evaluation_criteria"].items():
                print(f"\n{criteria.upper().replace('_', ' ')}: {details['description']}")
                print(f"Score Range: {details['score_range']}")
                for question in details['questions']:
                    print(f"  • {question}")
            
            print(f"\n🎯 SCORING GUIDE")
            for score, description in checklist["scoring_guide"].items():
                print(f"{score}: {description}")
        
        else:
            print("Invalid choice!")
            
    except KeyboardInterrupt:
        print("\n\nTesting session ended.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

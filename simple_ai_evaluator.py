#!/usr/bin/env python3
"""
Simple AI Istanbul Evaluator - Run all 80 tests and analyze responses
"""

import json
import datetime

# All 80 test questions
TEST_QUESTIONS = [
    # Transportation (1-16)
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
    "What's the best way to do a Bosphorus tour including both continents?",
    
    # Restaurants & Food (17-32)
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
    "I want to understand regional Turkish cuisine differences. What should I look for in Istanbul?",
    
    # Museums & Cultural Sites (33-48)
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
    "What are the must-see cultural sites for a first-time visitor with only 2 days?",
    
    # Districts & Neighborhoods (49-64)
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
    "What are the main characteristics of Ortaköy district?",
    
    # General Tips & Practical (65-80)
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

def get_category(test_num):
    """Get category name based on test number"""
    if 1 <= test_num <= 16:
        return "Transportation"
    elif 17 <= test_num <= 32:
        return "Restaurants & Food"
    elif 33 <= test_num <= 48:
        return "Museums & Culture"
    elif 49 <= test_num <= 64:
        return "Districts & Neighborhoods"
    else:
        return "General & Practical"

def display_test(test_num):
    """Display a single test question"""
    if 1 <= test_num <= 80:
        question = TEST_QUESTIONS[test_num - 1]
        category = get_category(test_num)
        
        print(f"\n{'='*80}")
        print(f"🧪 TEST #{test_num}/80 - {category.upper()}")
        print(f"{'='*80}")
        print(f"❓ QUESTION: {question}")
        print(f"{'='*80}")
        print("\n📋 EVALUATION INSTRUCTIONS:")
        print("1. Copy this question to your AI Istanbul chatbot")
        print("2. Paste the full AI response below")
        print("3. Score the response using these criteria:")
        print("   • Accuracy (0-3): Is the information factually correct?")
        print("   • Completeness (0-3): Does it fully answer the question?")
        print("   • Cultural Sensitivity (0-2): Shows Turkish cultural awareness?")
        print("   • Practical Usefulness (0-2): Provides actionable advice?")
        print("4. Note any specific issues or missing information")
        print(f"{'='*80}")
        return question
    else:
        print("❌ Invalid test number. Please choose 1-80.")
        return None

def run_all_tests():
    """Display all 80 test questions for manual evaluation"""
    print("🇹🇷 AI ISTANBUL COMPREHENSIVE EVALUATION - ALL 80 TESTS")
    print("=" * 80)
    print("📅 Date:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("🎯 Goal: Evaluate AI responses for accuracy, completeness, cultural sensitivity, and usefulness")
    print("📊 Scoring: Each response scored 0-10 points total (3+3+2+2)")
    print("=" * 80)
    
    # Create results log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ai_istanbul_evaluation_results_{timestamp}.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("AI Istanbul Comprehensive Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total Tests: 80\n")
        f.write("Scoring: Accuracy(0-3) + Completeness(0-3) + Cultural(0-2) + Practical(0-2) = Total(0-10)\n\n")
    
    for i in range(1, 81):
        question = display_test(i)
        
        print("\n🤖 PASTE AI RESPONSE HERE:")
        print("(Press Enter when ready to continue)")
        input()
        
        print("📊 QUICK SCORING:")
        print("Rate this response:")
        print("• Overall Quality (1-10): [  ]")
        print("• Major Issues Found: [  ]")
        print("• Recommendations: [  ]")
        
        # Log to file
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"TEST #{i} - {get_category(i)}\n")
            f.write(f"Q: {question}\n")
            f.write("A: [PASTE AI RESPONSE HERE]\n")
            f.write("Score: ___/10\n")
            f.write("Issues: _____\n")
            f.write("Notes: _____\n")
            f.write("-" * 50 + "\n")
        
        if i % 5 == 0:
            print(f"\n✅ Progress: {i}/80 tests completed ({i/80*100:.1f}%)")
            
        if i < 80:
            print("\n" + "-"*50)
            continue_choice = input("Press Enter for next test, 'q' to quit, 's' to skip to test number: ").strip()
            if continue_choice.lower() == 'q':
                break
            elif continue_choice.lower() == 's':
                try:
                    skip_to = int(input("Skip to test number (1-80): "))
                    if 1 <= skip_to <= 80:
                        i = skip_to - 1
                except:
                    pass
    
    print(f"\n📄 Results template saved to: {results_file}")
    print("🎉 Evaluation session completed!")

def run_specific_test():
    """Run a specific test by number"""
    try:
        test_num = int(input("Enter test number (1-80): "))
        display_test(test_num)
    except ValueError:
        print("❌ Please enter a valid number")

def run_category_tests():
    """Run tests for a specific category"""
    print("\nAvailable categories:")
    print("1. Transportation (Tests 1-16)")
    print("2. Restaurants & Food (Tests 17-32)")
    print("3. Museums & Culture (Tests 33-48)")
    print("4. Districts & Neighborhoods (Tests 49-64)")
    print("5. General & Practical (Tests 65-80)")
    
    try:
        choice = int(input("Choose category (1-5): "))
        if choice == 1:
            start, end = 1, 16
        elif choice == 2:
            start, end = 17, 32
        elif choice == 3:
            start, end = 33, 48
        elif choice == 4:
            start, end = 49, 64
        elif choice == 5:
            start, end = 65, 80
        else:
            print("❌ Invalid choice")
            return
        
        for i in range(start, end + 1):
            display_test(i)
            input("\nPress Enter for next test...")
            if i < end:
                print("-" * 50)
                
    except ValueError:
        print("❌ Please enter a valid number")

def main():
    """Main menu"""
    while True:
        print("\n🚀 AI ISTANBUL EVALUATION SYSTEM")
        print("=" * 40)
        print("1. Run ALL 80 tests (Full Evaluation)")
        print("2. Run specific test by number")
        print("3. Run tests by category")
        print("4. Exit")
        
        try:
            choice = int(input("\nChoose option (1-4): "))
            
            if choice == 1:
                print("\n🎯 Starting comprehensive evaluation of all 80 tests...")
                run_all_tests()
                break
            elif choice == 2:
                run_specific_test()
            elif choice == 3:
                run_category_tests()
            elif choice == 4:
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please choose 1-4.")
                
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\n👋 Evaluation cancelled by user.")
            break

if __name__ == "__main__":
    main()

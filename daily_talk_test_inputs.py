#!/usr/bin/env python3
"""
Daily Talk Test Inputs for AI Istanbul Chatbot
==============================================

This file contains 20 diverse test inputs to comprehensively test the AI Istanbul chatbot.
These inputs cover various scenarios a tourist might encounter in Istanbul.

Usage:
- Run individual tests manually through the web interface
- Use this as a reference for systematic testing
- Copy-paste inputs to test different response variations

Categories:
1. Restaurant & Food Queries (5 inputs)
2. Transportation & Navigation (4 inputs) 
3. Cultural Sites & Museums (4 inputs)
4. Shopping & Local Markets (3 inputs)
5. General Travel & Practical (4 inputs)
"""

# Restaurant & Food Queries
RESTAURANT_FOOD_QUERIES = [
    "I'm staying in Sultanahmet and want authentic Turkish breakfast. What are the best traditional places nearby?",
    
    "Can you recommend good seafood restaurants in Karaköy with Bosphorus views? I'm looking for something romantic for dinner tonight.",
    
    "I'm vegetarian and staying in Beyoğlu. Where can I find good vegetarian Turkish food? Also, what traditional dishes should I try?",
    
    "What are the must-try street foods in Istanbul? I want to experience local flavors but I'm concerned about food safety.",
    
    "I have a gluten allergy. Can you suggest restaurants in Kadıköy that cater to gluten-free diets and understand Turkish cuisine restrictions?"
]

# Transportation & Navigation
TRANSPORTATION_QUERIES = [
    "How do I get from Istanbul Airport to Sultanahmet using public transportation? What's the cheapest and fastest way?",
    
    "I want to take a ferry from Eminönü to Üsküdar. What's the schedule and how much does it cost? Any scenic spots to see during the ride?",
    
    "What's the best way to travel between the European and Asian sides of Istanbul? I'll be doing this daily for a week.",
    
    "I'm at Taksim Square and need to get to the Grand Bazaar. Should I take the metro, bus, or taxi? What about walking routes?"
]

# Cultural Sites & Museums
CULTURAL_SITES_QUERIES = [
    "I have 2 days in Istanbul. What are the absolute must-see historical sites? Can you suggest an itinerary with timing?",
    
    "Tell me about Hagia Sophia. What's the best time to visit to avoid crowds? What should I know about the history and current status?",
    
    "I'm interested in Ottoman history. Beyond the obvious sites, what lesser-known museums or historical places would you recommend?",
    
    "Can you explain the difference between the Blue Mosque and other mosques in Istanbul? What's the proper etiquette for visiting?"
]

# Shopping & Local Markets
SHOPPING_QUERIES = [
    "I want to buy authentic Turkish carpets and ceramics. Where should I go and how do I avoid tourist traps? Any negotiation tips?",
    
    "What's the difference between the Grand Bazaar and the Spice Bazaar? Which one is better for souvenirs and local products?",
    
    "I'm looking for modern Turkish design and fashion. Can you recommend contemporary shopping areas beyond the traditional markets?"
]

# General Travel & Practical
GENERAL_TRAVEL_QUERIES = [
    "What's the weather like in Istanbul in December? What should I pack and are there any seasonal activities or closures to know about?",
    
    "I have a 12-hour layover in Istanbul. Can I leave the airport and see some sights? What's possible in that timeframe?",
    
    "What are some common cultural mistakes tourists make in Istanbul? I want to be respectful to local customs and traditions.",
    
    "Can you help me with basic Turkish phrases for dining and shopping? Also, how widely is English spoken in tourist areas?"
]

def print_all_test_inputs():
    """Print all test inputs in a formatted way for easy copying."""
    
    print("🇹🇷 AI ISTANBUL CHATBOT - DAILY TALK TEST INPUTS 🇹🇷")
    print("=" * 60)
    
    all_categories = [
        ("🍽️  RESTAURANT & FOOD QUERIES", RESTAURANT_FOOD_QUERIES),
        ("🚌 TRANSPORTATION & NAVIGATION", TRANSPORTATION_QUERIES),
        ("🏛️  CULTURAL SITES & MUSEUMS", CULTURAL_SITES_QUERIES),
        ("🛒 SHOPPING & LOCAL MARKETS", SHOPPING_QUERIES),
        ("✈️  GENERAL TRAVEL & PRACTICAL", GENERAL_TRAVEL_QUERIES)
    ]
    
    test_number = 1
    
    for category_name, queries in all_categories:
        print(f"\n{category_name}")
        print("-" * 50)
        
        for query in queries:
            print(f"\n🤖 TEST INPUT #{test_number}:")
            print(f"📝 Query: {query}")
            print("─" * 40)
            test_number += 1
    
    print(f"\n✅ Total Test Inputs: {test_number - 1}")
    print("\n📋 TESTING INSTRUCTIONS:")
    print("1. Copy each query to the AI Istanbul chat interface")
    print("2. Evaluate response quality, accuracy, and helpfulness")
    print("3. Check for appropriate local knowledge and cultural awareness")
    print("4. Verify that practical information (times, prices, locations) is current")
    print("5. Assess whether responses encourage further exploration")

def get_test_input(number):
    """Get a specific test input by number (1-20)."""
    all_queries = (RESTAURANT_FOOD_QUERIES + TRANSPORTATION_QUERIES + 
                  CULTURAL_SITES_QUERIES + SHOPPING_QUERIES + 
                  GENERAL_TRAVEL_QUERIES)
    
    if 1 <= number <= len(all_queries):
        return all_queries[number - 1]
    else:
        return f"Invalid test number. Please choose between 1 and {len(all_queries)}"

def get_random_test_input():
    """Get a random test input for quick testing."""
    import random
    all_queries = (RESTAURANT_FOOD_QUERIES + TRANSPORTATION_QUERIES + 
                  CULTURAL_SITES_QUERIES + SHOPPING_QUERIES + 
                  GENERAL_TRAVEL_QUERIES)
    return random.choice(all_queries)

if __name__ == "__main__":
    print_all_test_inputs()
    
    # Example usage
    print(f"\n🎲 RANDOM TEST: {get_random_test_input()}")
    print(f"\n🔍 TEST #5: {get_test_input(5)}")

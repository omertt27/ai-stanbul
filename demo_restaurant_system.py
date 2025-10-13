#!/usr/bin/env python3
"""
Demo script for testing the comprehensive restaurant advising system
Tests 20 diverse restaurant queries to demonstrate AI capabilities
"""

import sys
from datetime import datetime

def demo_restaurant_system():
    """Demonstrate the full restaurant advising system capabilities"""
    print("🍽️ Istanbul Restaurant AI System Demo")
    print("=" * 60)
    
    try:
        # Import the main AI system (now using improved modular architecture)
        from istanbul_daily_talk_system_modular import IstanbulDailyTalkAI
        
        # Create AI instance
        ai = IstanbulDailyTalkAI()
        print("✅ AI system with restaurant integration ready!\n")
        
        # 20 comprehensive restaurant test queries
        restaurant_queries = [
            {
                "query": "I want authentic Turkish breakfast in Sultanahmet",
                "description": "Location + cuisine + meal type"
            },
            {
                "query": "Show me budget-friendly vegetarian restaurants near Taksim",
                "description": "Budget + dietary + location"
            },
            {
                "query": "Where can I find the best seafood in Kadıköy?",
                "description": "Cuisine + quality + district"
            },
            {
                "query": "I need halal restaurants open late night in Beyoğlu",
                "description": "Dietary + timing + location"
            },
            {
                "query": "Recommend romantic dinner places with Bosphorus view",
                "description": "Atmosphere + view + meal type"
            },
            {
                "query": "I'm looking for gluten-free options in Beşiktaş",
                "description": "Dietary restriction + location"
            },
            {
                "query": "What are some good street food spots in Eminönü?",
                "description": "Food type + location"
            },
            {
                "query": "I want expensive fine dining restaurants in Nişantaşı",
                "description": "Price level + cuisine style + location"
            },
            {
                "query": "Show me family-friendly restaurants with kids menu",
                "description": "Family + specific service"
            },
            {
                "query": "I need vegan restaurants that deliver to Şişli",
                "description": "Dietary + service + location"
            },
            {
                "query": "Where can I eat traditional Ottoman cuisine?",
                "description": "Historical cuisine type"
            },
            {
                "query": "I'm looking for kosher restaurants in Istanbul",
                "description": "Specific dietary requirement"
            },
            {
                "query": "Show me rooftop restaurants with city views",
                "description": "Setting + view type"
            },
            {
                "query": "I want cheap but good Turkish pide places",
                "description": "Budget + quality + specific dish"
            },
            {
                "query": "Recommend restaurants near Galata Tower for lunch",
                "description": "Landmark proximity + meal time"
            },
            {
                "query": "I need restaurants open for breakfast at 7 AM",
                "description": "Timing + meal type"
            },
            {
                "query": "Show me international cuisine options in Levent",
                "description": "Cuisine variety + business district"
            },
            {
                "query": "I want authentic meze restaurants with live music",
                "description": "Food type + entertainment"
            },
            {
                "query": "Where can I find the best Turkish coffee and desserts?",
                "description": "Beverage + dessert focus"
            },
            {
                "query": "I need restaurants with outdoor seating for large groups",
                "description": "Seating + group size"
            }
        ]
        
        user_id = "restaurant_demo_user"
        
        for i, demo in enumerate(restaurant_queries, 1):
            print(f"🍽️ Test {i:2d}: {demo['description']}")
            print(f"👤 User: \"{demo['query']}\"")
            print("🤖 AI Response:")
            print("-" * 70)
            
            try:
                response = ai.process_message(demo['query'], user_id)
                print(response)
                print()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
        
        print("🎉 Restaurant system demo completed!")
        print("\n💡 Key restaurant features tested:")
        print("• Location-specific searches (districts & landmarks)")
        print("• Cuisine filtering (Turkish, seafood, international)")
        print("• Dietary restrictions (vegetarian, vegan, halal, kosher, gluten-free)")
        print("• Price level preferences (budget to fine dining)")
        print("• Timing requirements (breakfast, lunch, dinner, late night)")
        print("• Atmosphere preferences (romantic, family-friendly, rooftop)")
        print("• Service types (delivery, outdoor seating, group dining)")
        print("• Special features (live music, city views, kids menu)")
        print("• Traditional vs international cuisine options")
        print("• Smart query understanding and context awareness")
        
    except Exception as e:
        print(f"❌ Restaurant demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = demo_restaurant_system()
    sys.exit(0 if success else 1)

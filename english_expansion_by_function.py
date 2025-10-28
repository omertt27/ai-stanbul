#!/usr/bin/env python3
"""
English Training Data Expansion System
Aligned with 8 core functions of AI Istanbul system

Core Functions:
1. Daily talks - Greetings, small talk, basic conversations
2. Places/attractions - Tourist sites, landmarks, museums
3. Neighborhood guides - District information, area guides
4. Transportation - Metro, bus, ferry, directions
5. Events advising - Concerts, festivals, cultural events
6. Route planner - Navigation, directions, itineraries
7. Weather system - Forecasts, what to wear
8. Local tips/hidden gems - Insider advice, off-beaten-path
"""

import json
import random
from collections import Counter, defaultdict
from typing import Dict, List

# Comprehensive English training data mapped to core functions
ENGLISH_TRAINING_DATA_BY_FUNCTION = {
    
    # ============================================================
    # 1. DAILY TALKS
    # ============================================================
    "daily_talks": {
        "greeting": [
            "Hello", "Hi there", "Good morning", "Good afternoon", "Good evening",
            "Hey", "Hi", "Hello there", "Greetings", "Welcome",
            "Nice to meet you", "How are you?", "How's it going?",
            "What's up?", "How do you do?", "Pleased to meet you",
            "Hi friend", "Hello friend", "Welcome to Istanbul",
            "Good to see you", "Nice day", "Beautiful day",
            "Morning", "Evening", "Afternoon", "Hey there",
        ],
        
        "farewell": [
            "Goodbye", "Bye", "See you", "Farewell", "Take care",
            "See you later", "Catch you later", "Until next time",
            "Have a good day", "Have a great day", "Safe travels",
            "Bye bye", "Good night", "See you soon", "Talk to you later",
            "Thanks for everything", "It was nice talking", "Enjoy your stay",
            "Have a wonderful trip", "Come back soon", "Visit again",
        ],
        
        "thanks": [
            "Thank you", "Thanks", "Thanks a lot", "Thank you very much",
            "Much appreciated", "I appreciate it", "Thanks so much",
            "Many thanks", "Thanks for your help", "That's helpful",
            "Great help", "Perfect", "Excellent", "Amazing",
            "You're awesome", "Really helpful", "This helps a lot",
            "Appreciate your time", "Thanks for the info",
        ],
        
        "help": [
            "Can you help me?", "I need help", "Help please",
            "Can you assist me?", "I need assistance", "I'm lost",
            "Where do I go?", "What should I do?", "I'm confused",
            "Can you explain?", "I don't understand", "Tell me more",
            "How does this work?", "What does this mean?",
            "I need information", "Please help", "Assist me",
            "Guide me", "Show me", "Explain please",
        ],
        
        "general_info": [
            "Tell me about Istanbul", "Information about Istanbul",
            "What is Istanbul like?", "Istanbul guide", "City information",
            "Things to know", "Basic information", "General info",
            "What should I know?", "Istanbul overview", "City guide",
            "Tourist information", "Visitor guide", "Travel tips",
            "Istanbul facts", "City basics", "Getting around",
        ],
    },
    
    # ============================================================
    # 2. PLACES/ATTRACTIONS
    # ============================================================
    "places_attractions": {
        "attraction": [
            "What to see in Istanbul?", "Tourist attractions", "Must-see places",
            "Famous landmarks", "Top attractions", "Best places to visit",
            "Popular tourist spots", "Sightseeing recommendations",
            "Where should I visit?", "Main attractions", "Tourist sites",
            "Historical sites", "Cultural sites", "Heritage sites",
            "Photo spots", "Instagram locations", "Scenic views",
            "Iconic places", "Landmarks", "Monuments",
            "Visit Hagia Sophia", "See Blue Mosque", "Topkapi Palace",
            "Galata Tower", "Basilica Cistern", "Grand Bazaar visit",
            "Sultanahmet Square", "Bosphorus view", "Golden Horn",
            "Dolmabahce Palace", "Maiden's Tower", "Taksim Square",
            "Istiklal Street", "Spice Bazaar", "Ortakoy Mosque",
        ],
        
        "museum": [
            "Art museums", "History museums", "Museums in Istanbul",
            "Museum recommendations", "Best museums", "Which museum to visit?",
            "Museum hours", "Museum tickets", "Museum entrance fee",
            "Free museums", "Museum pass", "Istanbul Museum Pass",
            "Archaeological museums", "Modern art museum", "Contemporary art",
            "Islamic art museum", "Turkish art", "Ottoman museum",
            "Museum tour", "Guided museum tour", "Audio guide",
            "Museum collections", "Special exhibitions", "Museum cafe",
            "Museum shop", "Museum photography", "Child-friendly museums",
            "Interactive museums", "Science museum", "Technology museum",
        ],
        
        "hidden_gems": [
            "Hidden gems in Istanbul", "Off the beaten path", "Secret places",
            "Local favorites", "Undiscovered spots", "Hidden attractions",
            "Less touristy places", "Authentic Istanbul", "Local spots",
            "Secret gardens", "Hidden cafes", "Unknown places",
            "Non-touristy areas", "Insider spots", "Hidden treasures",
            "Undiscovered neighborhoods", "Secret viewpoints",
            "Hidden historical sites", "Secret beaches", "Hidden parks",
            "Local hangouts", "Authentic experiences", "Real Istanbul",
        ],
    },
    
    # ============================================================
    # 3. NEIGHBORHOOD GUIDES
    # ============================================================
    "neighborhood_guides": {
        "neighborhoods": [
            "Best neighborhoods", "Which area to stay?", "District guide",
            "Neighborhood recommendations", "Where to stay in Istanbul?",
            "Safe neighborhoods", "Popular districts", "Best areas",
            "Kadikoy neighborhood", "Beyoglu district", "Besiktas area",
            "Sultanahmet area", "Taksim neighborhood", "Nisantasi district",
            "Cihangir area", "Ortakoy neighborhood", "Bebek area",
            "Karakoy district", "Balat neighborhood", "Fener area",
            "Uskudar neighborhood", "Moda district", "Galata area",
            "What's Kadikoy like?", "Tell me about Beyoglu",
            "Sultanahmet information", "Besiktas guide", "Ortakoy overview",
            "Family-friendly neighborhoods", "Nightlife districts",
            "Shopping areas", "Residential areas", "Business districts",
        ],
        
        "local_tips": [
            "Local tips", "Insider advice", "Local recommendations",
            "What locals do", "Local secrets", "Insider tips",
            "Best kept secrets", "Local knowledge", "Native advice",
            "Where do locals eat?", "Where do locals shop?",
            "Local favorites", "Authentic local experience",
            "How locals travel", "Local customs", "Cultural tips",
            "Etiquette in Istanbul", "Do's and don'ts", "Cultural norms",
            "Local traditions", "Istanbul lifestyle", "Daily life",
            "Local shopping tips", "Bargaining tips", "Market tips",
            "How to avoid tourist traps", "Save money tips",
        ],
    },
    
    # ============================================================
    # 4. TRANSPORTATION
    # ============================================================
    "transportation": {
        "transportation": [
            "How to get around?", "Public transportation", "Metro system",
            "Bus routes", "Tram lines", "Ferry schedule",
            "Istanbul transport", "Getting around Istanbul",
            "Transport options", "Metro map", "Bus map", "Tram map",
            "Istanbul card", "Transport card", "How to buy ticket",
            "Metro ticket", "Bus fare", "Ferry ticket",
            "Transport prices", "Daily ticket", "Weekly pass",
            "Airport to city", "Airport transfer", "Airport shuttle",
            "Taxi in Istanbul", "Uber available?", "Taxi fare",
            "Metro hours", "Last metro", "First bus", "Night buses",
            "Marmaray line", "Metrobus route", "Dolmus service",
            "How does metro work?", "How to use Istanbul card?",
            "Where to buy transport card?", "Top up card",
        ],
        
        "gps_navigation": [
            "Directions to Taksim", "How to reach Sultanahmet?",
            "Navigate to Blue Mosque", "GPS to Galata Tower",
            "Location of Grand Bazaar", "Find Hagia Sophia",
            "Where is Topkapi Palace?", "Directions please",
            "Show me the way", "How do I get there?",
            "Walking directions", "Driving directions", "Route to",
            "Best route", "Fastest way", "Shortest route",
            "Navigate me", "GPS navigation", "Turn by turn directions",
            "Street directions", "Map to location", "Find address",
        ],
        
        "route_planning": [
            "Plan my route", "Create itinerary", "Plan my day",
            "Route planner", "Trip planner", "Journey planner",
            "Multi-stop route", "Optimize route", "Best itinerary",
            "Day trip plan", "Two day itinerary", "Three day plan",
            "Weekend itinerary", "One week plan", "Custom route",
            "Plan visits", "Schedule my day", "Time my visits",
            "Efficient route", "Optimal path", "Smart routing",
            "Route with stops", "Include these places", "Visit these sites",
        ],
    },
    
    # ============================================================
    # 5. EVENTS ADVISING
    # ============================================================
    "events_advising": {
        "events": [
            "Events in Istanbul", "What's happening today?", "Tonight's events",
            "This weekend events", "Concerts in Istanbul", "Live music",
            "Shows and performances", "Theater shows", "Opera performances",
            "Ballet shows", "Dance performances", "Music concerts",
            "Rock concerts", "Classical music", "Jazz concerts",
            "Festivals in Istanbul", "Cultural festivals", "Music festivals",
            "Food festivals", "Art festivals", "Film festivals",
            "Events this week", "Upcoming events", "Event calendar",
            "What to do tonight?", "Evening events", "Daytime events",
            "Free events", "Outdoor events", "Indoor events",
            "Family events", "Kids events", "Art exhibitions",
            "Gallery openings", "Special events", "Seasonal events",
        ],
        
        "cultural_info": [
            "Turkish culture", "Ottoman culture", "Byzantine history",
            "Cultural information", "Traditions", "Customs",
            "Turkish traditions", "Local customs", "Cultural norms",
            "Turkish tea culture", "Coffee culture", "Food culture",
            "Turkish music", "Traditional music", "Folk music",
            "Turkish dance", "Traditional dance", "Folk dance",
            "Religious customs", "Mosque etiquette", "Prayer times",
            "Ramadan in Istanbul", "Eid celebrations", "Cultural events",
            "Traditional crafts", "Carpet making", "Calligraphy",
            "Turkish baths", "Hamam culture", "Spa traditions",
        ],
    },
    
    # ============================================================
    # 6. ROUTE PLANNER (Covered above in transportation)
    # ============================================================
    
    # ============================================================
    # 7. WEATHER SYSTEM
    # ============================================================
    "weather_system": {
        "weather": [
            "Weather in Istanbul", "What's the weather like?", "Temperature today",
            "Weather forecast", "Today's weather", "Tomorrow's weather",
            "Weekly forecast", "10 day forecast", "Weather this week",
            "Will it rain?", "Is it sunny?", "Is it cloudy?",
            "Temperature", "How hot is it?", "How cold is it?",
            "What to wear?", "Do I need umbrella?", "Do I need jacket?",
            "Weather conditions", "Current weather", "Real-time weather",
            "Humidity", "Wind speed", "Precipitation",
            "Sunrise time", "Sunset time", "UV index",
            "Weather warning", "Storm alert", "Snow forecast",
            "Spring weather", "Summer weather", "Autumn weather", "Winter weather",
            "Best time to visit", "Weather in June", "Weather in December",
            "Climate information", "Average temperature", "Rainfall",
        ],
    },
    
    # ============================================================
    # 8. LOCAL TIPS/HIDDEN GEMS (Covered above)
    # ============================================================
    
    # ============================================================
    # SUPPORTING INTENTS
    # ============================================================
    "supporting_intents": {
        "restaurant": [
            "Where to eat?", "Best restaurants", "Food recommendations",
            "Dining options", "Good restaurants", "Restaurant near me",
            "Turkish restaurant", "Seafood restaurant", "Vegetarian restaurant",
            "Vegan options", "Halal food", "Kosher food",
            "Fine dining", "Casual dining", "Street food",
            "Cheap eats", "Budget restaurants", "Expensive restaurants",
            "Romantic restaurant", "Family restaurant", "Kids menu",
            "Restaurant with view", "Rooftop restaurant", "Bosphorus restaurant",
            "Traditional food", "Modern Turkish cuisine", "International food",
            "Italian restaurant", "Asian food", "Fast food",
            "Breakfast place", "Brunch spot", "Lunch restaurant",
            "Dinner reservation", "Late night food", "24 hour restaurant",
            "Restaurant hours", "Book table", "Reserve table",
        ],
        
        "accommodation": [
            "Where to stay?", "Hotel recommendations", "Best hotels",
            "Cheap hotels", "Budget accommodation", "Luxury hotels",
            "5 star hotels", "4 star hotels", "Boutique hotels",
            "Hostel options", "Guest houses", "Airbnb",
            "Hotel near Sultanahmet", "Hotel in Taksim", "Hotel in Beyoglu",
            "Hotel with sea view", "Hotel with breakfast", "Pet friendly hotel",
            "Family hotel", "Business hotel", "Airport hotel",
            "Book hotel", "Hotel reservation", "Check availability",
            "Hotel prices", "Room rates", "Hotel deals",
            "Hotel amenities", "Hotel facilities", "Swimming pool",
            "Gym", "Spa", "Room service",
        ],
        
        "shopping": [
            "Where to shop?", "Shopping areas", "Best shopping",
            "Shopping malls", "Shopping streets", "Markets",
            "Grand Bazaar", "Spice Bazaar", "Egyptian Bazaar",
            "Istiklal Street shopping", "Nisantasi shopping", "Kadikoy shops",
            "Souvenir shopping", "Gift shops", "Carpet shopping",
            "Turkish delight", "Spices", "Tea shopping",
            "Antique shops", "Art galleries", "Craft shops",
            "Luxury shopping", "Designer stores", "Outlet malls",
            "Local markets", "Flea markets", "Weekend markets",
            "Shopping hours", "When shops open?", "Tax free shopping",
        ],
        
        "booking": [
            "Book tickets", "Make reservation", "Reserve table",
            "Buy tickets online", "Advance booking", "Last minute booking",
            "Group booking", "Cancel booking", "Modify reservation",
            "Booking confirmation", "Ticket prices", "Discount tickets",
            "Student tickets", "Senior discount", "Family package",
            "Skip the line", "Fast track", "Priority entrance",
            "Book tour", "Book guide", "Book transfer",
        ],
        
        "budget": [
            "Cheap options", "Budget travel", "Save money",
            "Free things to do", "Free attractions", "Free museums",
            "Budget restaurants", "Cheap eats", "Affordable hotels",
            "Student discounts", "Free walking tours", "Budget tips",
            "How to save money?", "Economical options", "Low cost",
            "Best deals", "Discount cards", "City pass",
            "Free entrance days", "Happy hour", "Lunch specials",
        ],
        
        "price_info": [
            "How much is it?", "What's the price?", "Cost information",
            "Entrance fee", "Ticket price", "Menu prices",
            "Transport costs", "Hotel prices", "Tour prices",
            "Average costs", "Budget estimate", "Daily costs",
            "Expensive or cheap?", "Price range", "Cost breakdown",
        ],
        
        "emergency": [
            "Emergency", "Call police", "Hospital", "Ambulance",
            "I need help urgently", "Medical emergency", "Accident",
            "Police station", "Pharmacy", "Doctor",
            "I'm lost", "Lost passport", "Stolen wallet",
            "Embassy", "Consulate", "Emergency number",
            "Tourist police", "Fire department", "Urgent care",
            "24 hour pharmacy", "Emergency room", "Urgent medical help",
        ],
        
        "nightlife": [
            "Nightlife in Istanbul", "Bars", "Clubs", "Night clubs",
            "Live music venues", "Jazz bars", "Rock bars",
            "Rooftop bars", "Beach clubs", "Dance clubs",
            "DJ events", "Party tonight", "Where to party?",
            "Late night venues", "After hours", "Night entertainment",
            "Drinks", "Cocktail bars", "Wine bars", "Beer bars",
            "LGBTQ+ bars", "Gay clubs", "Friendly nightlife",
        ],
        
        "romantic": [
            "Romantic places", "Romantic spots", "Date ideas",
            "Romantic restaurant", "Couples activities", "Anniversary dinner",
            "Proposal spots", "Honeymoon activities", "Romantic walk",
            "Sunset spots", "Romantic views", "Couples spa",
            "Private dinner", "Candlelit dinner", "Romantic cruise",
            "Special occasions", "Romantic experience", "Love locks",
            "Couples massage", "Romantic hotel", "Private tour",
        ],
        
        "family_activities": [
            "Family activities", "Things to do with kids", "Kid-friendly places",
            "Children attractions", "Family fun", "Kids activities",
            "Playground", "Children's museum", "Aquarium",
            "Zoo", "Theme park", "Water park",
            "Family restaurant", "Kids menu", "Stroller friendly",
            "Baby changing rooms", "Family tours", "Child tickets",
            "Educational activities", "Interactive museums", "Science center",
            "Family beach", "Safe for kids", "Supervised activities",
        ],
        
        "luxury": [
            "Luxury experiences", "High-end restaurants", "5 star hotels",
            "Premium services", "VIP treatment", "Exclusive access",
            "Private tours", "Luxury shopping", "Designer stores",
            "Michelin restaurants", "Fine dining", "Gourmet food",
            "Luxury spa", "Premium yacht", "Private yacht",
            "Helicopter tour", "Luxury transport", "Chauffeur service",
        ],
        
        "history": [
            "Historical information", "Ottoman history", "Byzantine history",
            "Istanbul history", "Historical sites", "Ancient history",
            "Roman history", "Greek history", "Historical facts",
            "Historical tours", "History museum", "Historical buildings",
            "Archaeological sites", "Historical walls", "Old city",
        ],
        
        "recommendation": [
            "What do you recommend?", "Recommendations please",
            "Suggest something", "What should I do?", "Best options",
            "Your recommendation", "Advice needed", "Suggestions",
            "What's good here?", "Popular choices", "Best picks",
            "Top recommendations", "Must-do activities", "Don't miss",
        ],
        
        "food": [
            "Turkish food", "Traditional dishes", "Local cuisine",
            "Turkish breakfast", "Kebab", "Baklava", "Turkish delight",
            "Meze", "Pide", "Lahmacun", "Borek",
            "Turkish coffee", "Turkish tea", "Raki",
            "Street food", "Food tour", "Cooking class",
            "Food markets", "Fresh produce", "Spices",
        ],
    },
}


def expand_english_training_data():
    """Expand English training data for all core functions"""
    print("=" * 80)
    print("ENGLISH TRAINING DATA EXPANSION")
    print("Aligned with 8 Core Functions")
    print("=" * 80)
    print()
    
    # Flatten all data
    all_examples = []
    function_stats = {}
    intent_stats = Counter()
    
    for function, intents_data in ENGLISH_TRAINING_DATA_BY_FUNCTION.items():
        function_count = 0
        for intent, queries in intents_data.items():
            for query in queries:
                all_examples.append({
                    'text': query,
                    'intent': intent,
                    'language': 'en',
                    'user_function': function,
                    'source': 'english_expansion'
                })
                intent_stats[intent] += 1
                function_count += 1
        
        function_stats[function] = function_count
    
    # Shuffle
    random.shuffle(all_examples)
    
    print(f"📊 Generated {len(all_examples)} English training examples")
    print()
    
    print("📈 By Core Function:")
    for function, count in sorted(function_stats.items(), key=lambda x: -x[1]):
        print(f"   {function:30s}: {count:4d} examples")
    print()
    
    print("📈 By Intent (Top 15):")
    for intent, count in intent_stats.most_common(15):
        print(f"   {intent:20s}: {count:4d} examples")
    print()
    
    # Load existing training data if available
    try:
        with open('augmented_intent_training_data.json', 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            if isinstance(existing_data, dict) and 'training_data' in existing_data:
                existing_examples = existing_data['training_data']
            else:
                existing_examples = existing_data
    except FileNotFoundError:
        print("⚠️ No existing training data found, starting fresh")
        existing_examples = []
    
    # Combine with existing data
    print(f"📥 Existing data: {len(existing_examples)} examples")
    
    # Add new English examples
    combined_data = existing_examples + all_examples
    random.shuffle(combined_data)
    
    print(f"📊 Combined data: {len(combined_data)} examples")
    print(f"   Turkish/existing: {len(existing_examples)}")
    print(f"   New English: {len(all_examples)}")
    print()
    
    # Save enhanced dataset
    output_file = 'english_expanded_training_data.json'
    output_data = {
        'training_data': combined_data,
        'metadata': {
            'total_examples': len(combined_data),
            'english_examples': len(all_examples),
            'existing_examples': len(existing_examples),
            'num_intents': len(set(item['intent'] for item in combined_data)),
            'core_functions': list(ENGLISH_TRAINING_DATA_BY_FUNCTION.keys()),
            'generated_at': '2025-10-28'
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved to: {output_file}")
    print()
    print("=" * 80)
    print("✅ ENGLISH EXPANSION COMPLETE!")
    print("=" * 80)
    print()
    print("🎯 Next Steps:")
    print(f"   1. Review the data: cat {output_file} | less")
    print(f"   2. Train model: python train_intent_classifier.py --data-file {output_file}")
    print("   3. Test English performance: python distilbert_intent_inference.py")
    print()


if __name__ == "__main__":
    expand_english_training_data()

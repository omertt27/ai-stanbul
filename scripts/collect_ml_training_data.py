"""
ML Training Data Collection Script for AI Istanbul
Collects training data from test results and creates a dataset for fine-tuning
"""

import json
from pathlib import Path
from typing import List, Dict
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_training_dataset_from_tests() -> Path:
    """
    Convert test cases into training data for intent classification
    Based on comprehensive test results and common Istanbul queries
    """
    
    training_examples = []
    
    # ========== RESTAURANT QUERIES ==========
    restaurant_search = [
        "Best seafood restaurants in Istanbul",
        "Restaurants in Beyoğlu",
        "Street food in Istanbul",
        "Cheap eats in Istanbul",
        "Fine dining restaurants",
        "Ottoman cuisine restaurants",
        "Where to eat in Sultanahmet",
        "Kebab restaurants near Taksim",
        "Vegetarian restaurants in Kadıköy",
        "Breakfast places in Beşiktaş",
        "Restaurants with Bosphorus view",
        "Best meyhane in Beyoğlu",
        "Traditional Turkish restaurants",
        "Seafood restaurants in Kumkapı",
        "Budget-friendly restaurants",
        "Rooftop restaurants Istanbul",
        "Late night food options",
        "Halal restaurants in Istanbul",
        "Where can I find good börek",
        "Best kebab in old city",
        "Restaurants open on Sunday",
        "Turkish breakfast in Karaköy",
        "Street food markets",
        "Where to try baklava",
        "Restaurants near Blue Mosque",
    ]
    
    restaurant_info = [
        "Tell me about Mikla restaurant",
        "What's special about Nusr-Et",
        "Karaköy Lokantası menu",
        "Is Çiya Sofrası expensive",
        "Opening hours for Hamdi Restaurant",
    ]
    
    # ========== ATTRACTION QUERIES ==========
    attraction_search = [
        "Museums in Istanbul",
        "Historical monuments to visit",
        "Famous mosques in Istanbul",
        "What to see in Sultanahmet",
        "Tourist attractions in Beyoğlu",
        "Best palaces to visit",
        "Parks and gardens in Istanbul",
        "Art galleries in Karaköy",
        "Ancient sites in Istanbul",
        "Top 10 places to visit",
        "Byzantine landmarks",
        "Ottoman architecture",
        "Hidden gems in Istanbul",
        "Photography spots",
        "Bosphorus viewpoints",
        "Free attractions in Istanbul",
        "Religious sites",
        "Archaeological sites",
        "Churches in Istanbul",
        "Bazaars and markets",
        "What to visit in Asian side",
        "Historical hammams",
        "Towers and monuments",
        "City walls and gates",
        "Cultural centers",
    ]
    
    attraction_info = [
        "Tell me about Hagia Sophia",
        "Blue Mosque opening hours",
        "Topkapı Palace ticket prices",
        "Is Basilica Cistern worth it",
        "How long to visit Dolmabahçe Palace",
    ]
    
    # ========== TRANSPORTATION QUERIES ==========
    transport_route = [
        "How to get from Taksim to Sultanahmet",
        "Metro from airport to city center",
        "Ferry routes in Istanbul",
        "Best way from Kadıköy to Topkapı Palace",
        "How to reach Grand Bazaar by metro",
        "Transportation to Princes' Islands",
        "Bus from Beşiktaş to Ortaköy",
        "Tram route to Blue Mosque",
        "How to cross to Asian side",
        "Metro to Istiklal Street",
        "Ferry from Eminönü to Üsküdar",
        "How to get to Galata Tower",
        "Transportation from hotel to airport",
        "Marmaray route and stops",
        "Funicular to Taksim",
    ]
    
    transport_info = [
        "How to use Istanbul metro",
        "Istanbulkart information",
        "Public transport ticket prices",
        "Metro operating hours",
        "Ferry schedules",
        "Transportation app recommendations",
        "Is taxi expensive in Istanbul",
        "How does the tram system work",
        "Public transport at night",
        "Airport shuttle services",
    ]
    
    # ========== WEATHER QUERIES ==========
    weather_queries = [
        "What's the weather like today?",
        "Best places to cool down in summer",
        "Winter activities in Istanbul",
        "What to do on a rainy day in Istanbul",
        "Is it hot in August",
        "Best time to visit Istanbul",
        "Weather forecast this week",
        "What to wear in Istanbul in October",
        "Does it snow in Istanbul",
        "Indoor activities when it rains",
        "Summer festivals Istanbul",
        "Beach recommendations near Istanbul",
        "Best season to visit",
        "Temperature in December",
    ]
    
    # ========== EVENT QUERIES ==========
    event_search = [
        "Cultural events and festivals",
        "What's happening this weekend?",
        "Concerts in Istanbul",
        "Events in Istanbul this month",
        "Music festivals",
        "Art exhibitions",
        "Theater performances",
        "Nightlife in Beyoğlu",
        "Live music venues",
        "Traditional dance shows",
        "Food festivals",
        "Cultural nights",
        "Film festivals Istanbul",
        "Ramadan events",
        "New Year celebrations",
    ]
    
    # ========== DAILY TALKS ==========
    daily_greetings = [
        "Merhaba!",
        "Hello! I'm visiting Istanbul",
        "Hi there",
        "Good morning",
        "Selam",
        "Hey, can you help me?",
    ]
    
    daily_gratitude = [
        "Thanks for the recommendations",
        "Thank you so much!",
        "That was helpful",
        "Teşekkürler",
        "Appreciate your help",
    ]
    
    daily_help = [
        "How many days do I need in Istanbul?",
        "I'm planning a trip to Istanbul",
        "What should I know before visiting?",
        "First time in Istanbul",
        "Travel tips for Istanbul",
        "Is Istanbul safe for tourists?",
        "Do I need a visa?",
        "What's the currency?",
        "Can I use credit cards everywhere?",
        "How much money should I bring?",
    ]
    
    daily_farewell = [
        "Goodbye",
        "Thanks, bye!",
        "See you later",
        "Hoşça kal",
        "That's all I needed",
    ]
    
    # ========== NEIGHBORHOOD QUERIES ==========
    neighborhood_info = [
        "Tell me about Beyoğlu neighborhood",
        "What's Kadıköy like?",
        "Sultanahmet area information",
        "Beşiktaş neighborhood guide",
        "What's special about Balat",
        "Ortaköy area attractions",
        "Cihangir neighborhood vibe",
        "Fatih district information",
    ]
    
    neighborhood_search = [
        "Hipster neighborhoods in Istanbul",
        "Best neighborhoods for first-time visitors",
        "Trendy areas in Istanbul",
        "Safe neighborhoods to stay",
        "Nightlife districts",
        "Shopping districts",
        "Historical neighborhoods",
        "Local neighborhoods off tourist path",
    ]
    
    # ========== HOTEL QUERIES ==========
    hotel_search = [
        "Hotels in Sultanahmet",
        "Boutique hotels in Beyoğlu",
        "Budget hotels near Taksim",
        "Hotels with Bosphorus view",
        "Where to stay in Istanbul",
        "Best area to book hotel",
        "Family-friendly hotels",
        "Luxury hotels Istanbul",
    ]
    
    # ========== PRICE & PRACTICAL QUERIES ==========
    price_inquiries = [
        "How much does a meal cost",
        "Average hotel prices",
        "Metro ticket price",
        "Museum entrance fees",
        "Is Istanbul expensive",
        "Budget for 3 days in Istanbul",
        "Cheap vs expensive neighborhoods",
    ]
    
    practical_info = [
        "Where to exchange money",
        "ATM locations",
        "WiFi availability",
        "SIM card for tourists",
        "Emergency numbers",
        "Tourist information centers",
        "Pharmacy locations",
        "Where to charge phone",
    ]
    
    # ========== RECOMMENDATION QUERIES ==========
    recommendations = [
        "What do you recommend for today",
        "Best things to do in 2 days",
        "Hidden gems in Istanbul",
        "Local favorites",
        "Must-try foods",
        "Photography spots",
        "Romantic places",
        "Family-friendly activities",
        "Shopping recommendations",
        "Sunset viewing spots",
    ]
    
    # ========== COMPARISON QUERIES ==========
    comparisons = [
        "Sultanahmet vs Beyoğlu for staying",
        "Topkapı or Dolmabahçe Palace",
        "Metro vs taxi",
        "Grand Bazaar or Spice Bazaar",
        "Asian side vs European side",
    ]
    
    # ========== BUILD TRAINING DATA ==========
    
    # Add all examples with their intents
    for query in restaurant_search:
        training_examples.append({"text": query, "intent": "restaurant_search"})
    
    for query in restaurant_info:
        training_examples.append({"text": query, "intent": "restaurant_info"})
    
    for query in attraction_search:
        training_examples.append({"text": query, "intent": "attraction_search"})
    
    for query in attraction_info:
        training_examples.append({"text": query, "intent": "attraction_info"})
    
    for query in transport_route:
        training_examples.append({"text": query, "intent": "transport_route"})
    
    for query in transport_info:
        training_examples.append({"text": query, "intent": "transport_info"})
    
    for query in weather_queries:
        training_examples.append({"text": query, "intent": "weather_query"})
    
    for query in event_search:
        training_examples.append({"text": query, "intent": "event_search"})
    
    for query in daily_greetings:
        training_examples.append({"text": query, "intent": "daily_greeting"})
    
    for query in daily_gratitude:
        training_examples.append({"text": query, "intent": "daily_gratitude"})
    
    for query in daily_help:
        training_examples.append({"text": query, "intent": "daily_help"})
    
    for query in daily_farewell:
        training_examples.append({"text": query, "intent": "daily_farewell"})
    
    for query in neighborhood_info:
        training_examples.append({"text": query, "intent": "neighborhood_info"})
    
    for query in neighborhood_search:
        training_examples.append({"text": query, "intent": "neighborhood_search"})
    
    for query in hotel_search:
        training_examples.append({"text": query, "intent": "hotel_search"})
    
    for query in price_inquiries:
        training_examples.append({"text": query, "intent": "price_inquiry"})
    
    for query in practical_info:
        training_examples.append({"text": query, "intent": "practical_info"})
    
    for query in recommendations:
        training_examples.append({"text": query, "intent": "recommendation_request"})
    
    for query in comparisons:
        training_examples.append({"text": query, "intent": "comparison_request"})
    
    # Create output directory
    output_file = Path("data/intent_training_data.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("✅ TRAINING DATA COLLECTION COMPLETE")
    print("="*60)
    print(f"📊 Total Examples: {len(training_examples)}")
    print(f"📁 Saved to: {output_file}")
    print("\n📋 Intent Distribution:")
    
    # Count intents
    intent_counts = {}
    for example in training_examples:
        intent = example['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    # Sort by count
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {intent:30s}: {count:3d} examples")
    
    print("\n🎯 Next Steps:")
    print("   1. Review the training data in: data/intent_training_data.json")
    print("   2. Run fine-tuning script: python scripts/finetune_intent_classifier.py")
    print("   3. Test the fine-tuned model")
    print("="*60 + "\n")
    
    return output_file


def augment_training_data(input_file: Path) -> Path:
    """
    Augment training data with variations (optional)
    - Add typos
    - Add Turkish variations
    - Add shortened forms
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    augmented = data.copy()
    
    # Add Turkish variations for common phrases
    turkish_variations = {
        "Best": "En iyi",
        "Where": "Nerede",
        "How to get": "Nasıl giderim",
        "What's": "Nedir",
        "Tell me about": "Anlat",
        "restaurants": "restoranlar",
        "hotels": "oteller",
    }
    
    # Add variations (simple example)
    for example in data[:20]:  # Augment first 20 as example
        text = example['text']
        for eng, tur in turkish_variations.items():
            if eng in text:
                augmented_text = text.replace(eng, tur)
                augmented.append({
                    "text": augmented_text,
                    "intent": example['intent']
                })
    
    # Save augmented data
    output_file = input_file.parent / "intent_training_data_augmented.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Augmented data saved: {output_file}")
    print(f"📊 Original: {len(data)} → Augmented: {len(augmented)}")
    
    return output_file


if __name__ == "__main__":
    print("\n🚀 Starting ML Training Data Collection...")
    print("="*60)
    
    # Create base training dataset
    output_file = create_training_dataset_from_tests()
    
    # Optional: Augment data
    # augment_training_data(output_file)
    
    print("✅ ML-P0.1 Complete!")
    print("\n📚 Dataset ready for fine-tuning!")

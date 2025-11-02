#!/usr/bin/env python3
"""
Create Balanced Bilingual Training Data
========================================

Balances Turkish and English training examples for equal performance
on both languages. Target: 50% Turkish, 50% English.
"""

import json
from collections import Counter
from datetime import datetime

def load_training_data(filepath):
    """Load existing training data"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def detect_language(text):
    """Simple language detection"""
    english_keywords = ['what', 'where', 'how', 'when', 'can', 'is', 'are', 'the', 'best', 'nearest', 'find', 'get', 'show', 'tell', 'want']
    turkish_chars = set('çğıöşüÇĞİÖŞÜ')
    
    # Check for Turkish characters
    if any(char in text for char in turkish_chars):
        return 'turkish'
    
    # Check for English keywords
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in english_keywords):
        return 'english'
    
    # Default to Turkish (since most short queries are Turkish)
    return 'turkish'

def analyze_language_distribution(data):
    """Analyze language distribution by intent"""
    intent_lang_counts = {}
    
    for query, intent in data:
        lang = detect_language(query)
        
        if intent not in intent_lang_counts:
            intent_lang_counts[intent] = {'turkish': 0, 'english': 0, 'total': 0}
        
        intent_lang_counts[intent][lang] += 1
        intent_lang_counts[intent]['total'] += 1
    
    return intent_lang_counts

def generate_english_translations(turkish_examples):
    """
    Generate English equivalents for Turkish queries
    This is a comprehensive mapping of Turkish to English patterns
    """
    translations = {
        # Transportation patterns
        "nasıl gidebilirim": "how can I get",
        "nasıl giderim": "how do I get",
        "ulaşım": "transportation",
        "metro": "metro",
        "otobüs": "bus",
        "tramvay": "tram",
        "taksi": "taxi",
        "vapur": "ferry",
        "havalimanına": "to the airport",
        "en yakın": "nearest",
        "hangi saatte": "what time",
        "kaçta": "when",
        "nereden": "from where",
        
        # Restaurant patterns
        "restoran": "restaurant",
        "yemek": "food",
        "en iyi": "best",
        "nerede yenir": "where to eat",
        "kahvaltı": "breakfast",
        "önerir misin": "do you recommend",
        "balık": "fish",
        "kebap": "kebab",
        "vejetaryen": "vegetarian",
        
        # Route planning
        "günlük rota": "daily route",
        "tur planı": "tour plan",
        "gezi rotası": "trip itinerary",
        "hafta sonu": "weekend",
        "iki günlük": "two day",
        
        # Weather
        "hava durumu": "weather",
        "bugün hava": "today's weather",
        "yağmur yağacak mı": "will it rain",
        "sıcaklık": "temperature",
        "güneşli": "sunny",
        
        # Attractions
        "gezmek": "visit",
        "görmek": "see",
        "tarihi yerler": "historical places",
        "müze": "museum",
        "giriş saatleri": "opening hours",
        
        # Hidden gems
        "gizli": "hidden",
        "turistik olmayan": "non-touristy",
        "yerel": "local",
        "keşfedilmemiş": "undiscovered",
        
        # Neighborhoods
        "semt": "neighborhood",
        "mahalle": "district",
        "bölge": "area",
        "nasıl bir": "what kind of",
    }
    
    english_examples = []
    
    for turkish_query, intent in turkish_examples:
        # Try to translate
        english_query = turkish_query.lower()
        
        # Apply translation patterns
        for tr_pattern, en_pattern in translations.items():
            if tr_pattern in english_query:
                english_query = english_query.replace(tr_pattern, en_pattern)
        
        # Only add if it's different and contains English
        if english_query != turkish_query.lower() and any(c.isascii() and c.isalpha() for c in english_query):
            # Clean up and capitalize
            english_query = ' '.join(english_query.split())
            english_query = english_query.capitalize()
            english_examples.append([english_query, intent])
    
    return english_examples

# Comprehensive English training examples for each intent
ENGLISH_EXAMPLES = {
    'transportation': [
        ["How do I get to Taksim?", "transportation"],
        ["What's the best way to reach the airport?", "transportation"],
        ["Where is the nearest metro station?", "transportation"],
        ["How can I get to Sultanahmet?", "transportation"],
        ["Is there a bus to Kadıköy?", "transportation"],
        ["What time does the tram run?", "transportation"],
        ["How much is a taxi to the airport?", "transportation"],
        ["Where can I buy an Istanbul card?", "transportation"],
        ["Ferry to the Princes' Islands?", "transportation"],
        ["Best transportation to Galata Tower?", "transportation"],
        ["How long does it take by metro?", "transportation"],
        ["Is there a direct bus?", "transportation"],
        ["Where do I catch the ferry?", "transportation"],
        ["Metro line to Taksim?", "transportation"],
        ["Can I walk there?", "transportation"],
        ["Night bus schedule", "transportation"],
        ["Marmaray route", "transportation"],
        ["Metrobus stops", "transportation"],
        ["Funicular to Taksim", "transportation"],
        ["Cable car times", "transportation"],
        ["How to use Istanbul Kart?", "transportation"],
        ["Transfer between lines", "transportation"],
        ["Closest tram stop", "transportation"],
        ["Bus number to Ortaköy", "transportation"],
        ["Ferry schedule Kadıköy", "transportation"],
    ],
    'restaurant': [
        ["Best fish restaurant in Istanbul", "restaurant"],
        ["Where can I find good kebab?", "restaurant"],
        ["Recommend a romantic restaurant", "restaurant"],
        ["Cheap places to eat", "restaurant"],
        ["Vegetarian restaurants near me", "restaurant"],
        ["Best breakfast spot in Kadıköy", "restaurant"],
        ["Rooftop restaurant with Bosphorus view", "restaurant"],
        ["Traditional Turkish food", "restaurant"],
        ["Where to eat mezze?", "restaurant"],
        ["Good seafood restaurants", "restaurant"],
        ["Family-friendly restaurants", "restaurant"],
        ["Late night food options", "restaurant"],
        ["Best baklava in Istanbul", "restaurant"],
        ["Turkish coffee shops", "restaurant"],
        ["Local food recommendations", "restaurant"],
        ["Budget-friendly dining", "restaurant"],
        ["Fine dining restaurants", "restaurant"],
        ["Street food locations", "restaurant"],
        ["Authentic Turkish cuisine", "restaurant"],
        ["Best brunch spots", "restaurant"],
        ["Halal restaurants", "restaurant"],
        ["Vegan options", "restaurant"],
        ["Where locals eat", "restaurant"],
        ["Hidden gem restaurants", "restaurant"],
        ["Waterfront dining", "restaurant"],
    ],
    'route_planning': [
        ["Plan a one day itinerary", "route_planning"],
        ["Three day Istanbul tour", "route_planning"],
        ["Weekend trip plan", "route_planning"],
        ["What should I see in 2 days?", "route_planning"],
        ["Create a morning route", "route_planning"],
        ["Full day sightseeing plan", "route_planning"],
        ["Historical sites tour", "route_planning"],
        ["Best order to visit attractions", "route_planning"],
        ["Efficient day plan", "route_planning"],
        ["Cultural route suggestions", "route_planning"],
        ["First day in Istanbul plan", "route_planning"],
        ["Photography tour route", "route_planning"],
        ["Walking tour itinerary", "route_planning"],
        ["Family-friendly day plan", "route_planning"],
        ["Romantic day itinerary", "route_planning"],
        ["Museum tour plan", "route_planning"],
        ["Food tour route", "route_planning"],
        ["Shopping day itinerary", "route_planning"],
        ["Bosphorus tour plan", "route_planning"],
        ["Old city route", "route_planning"],
        ["Modern Istanbul tour", "route_planning"],
        ["Half day suggestions", "route_planning"],
        ["Evening walk route", "route_planning"],
        ["Weekend getaway plan", "route_planning"],
        ["Quick visit itinerary", "route_planning"],
    ],
    'weather': [
        ["What's the weather like today?", "weather"],
        ["Will it rain tomorrow?", "weather"],
        ["How hot is it?", "weather"],
        ["Weather forecast for the weekend", "weather"],
        ["Should I bring an umbrella?", "weather"],
        ["What's the temperature?", "weather"],
        ["Is it sunny today?", "weather"],
        ["Weather this week", "weather"],
        ["Will it be cold tonight?", "weather"],
        ["Good weather for sightseeing?", "weather"],
        ["Chance of rain today?", "weather"],
        ["What's the climate like in Istanbul?", "weather"],
        ["Best time to visit weather-wise?", "weather"],
        ["Is it windy?", "weather"],
        ["Do I need a jacket?", "weather"],
        ["Current temperature", "weather"],
        ["Will there be sun tomorrow?", "weather"],
        ["Weather conditions now", "weather"],
        ["5 day forecast", "weather"],
        ["Is it humid?", "weather"],
        ["Good beach weather?", "weather"],
        ["Outdoor activity weather", "weather"],
        ["Storm warning?", "weather"],
        ["Air quality today", "weather"],
        ["Visibility conditions", "weather"],
    ],
    'attraction': [
        ["Where is Hagia Sophia?", "attraction"],
        ["Topkapi Palace visiting hours", "attraction"],
        ["Blue Mosque entrance fee", "attraction"],
        ["How to get to Galata Tower?", "attraction"],
        ["Basilica Cistern tickets", "attraction"],
        ["Best museums in Istanbul", "attraction"],
        ["Historical sites to visit", "attraction"],
        ["Grand Bazaar opening times", "attraction"],
        ["Dolmabahçe Palace tour", "attraction"],
        ["Things to see in Sultanahmet", "attraction"],
        ["Where is the Spice Bazaar?", "attraction"],
        ["Maiden's Tower boat schedule", "attraction"],
        ["Bosphorus cruise options", "attraction"],
        ["Princes' Islands ferry", "attraction"],
        ["Best viewpoints in Istanbul", "attraction"],
        ["Rumeli Fortress entrance", "attraction"],
        ["Chora Church mosaics", "attraction"],
        ["Istanbul Archaeology Museum", "attraction"],
        ["Süleymaniye Mosque visit", "attraction"],
        ["Pierre Loti Hill cable car", "attraction"],
        ["Miniaturk park info", "attraction"],
        ["Istanbul Aquarium tickets", "attraction"],
        ["Bosphorus bridge view", "attraction"],
        ["Must-see attractions", "attraction"],
        ["Famous landmarks", "attraction"],
    ],
    'hidden_gems': [
        ["Off the beaten path places", "hidden_gems"],
        ["Local secret spots", "hidden_gems"],
        ["Non-touristy areas", "hidden_gems"],
        ["Hidden neighborhoods", "hidden_gems"],
        ["Undiscovered Istanbul", "hidden_gems"],
        ["Where locals hang out", "hidden_gems"],
        ["Secret gardens", "hidden_gems"],
        ["Hidden cafes", "hidden_gems"],
        ["Underground cisterns", "hidden_gems"],
        ["Lesser-known museums", "hidden_gems"],
        ["Hidden viewpoints", "hidden_gems"],
        ["Local markets", "hidden_gems"],
        ["Secret beaches", "hidden_gems"],
        ["Authentic neighborhoods", "hidden_gems"],
        ["Hidden historical sites", "hidden_gems"],
        ["Local hangout spots", "hidden_gems"],
        ["Away from tourists", "hidden_gems"],
        ["Insider tips", "hidden_gems"],
        ["Hidden restaurants", "hidden_gems"],
        ["Secret parks", "hidden_gems"],
        ["Undiscovered gems", "hidden_gems"],
        ["Local favorites", "hidden_gems"],
        ["Hidden terraces", "hidden_gems"],
        ["Secret passages", "hidden_gems"],
        ["Authentic experiences", "hidden_gems"],
    ],
    'neighborhoods': [
        ["Tell me about Beyoğlu", "neighborhoods"],
        ["What's Kadıköy like?", "neighborhoods"],
        ["Sultanahmet area info", "neighborhoods"],
        ["Beşiktaş neighborhood", "neighborhoods"],
        ["Where is Balat?", "neighborhoods"],
        ["Üsküdar district guide", "neighborhoods"],
        ["Ortaköy characteristics", "neighborhoods"],
        ["Cihangir neighborhood vibe", "neighborhoods"],
        ["Moda area description", "neighborhoods"],
        ["What to do in Karaköy?", "neighborhoods"],
        ["Nişantaşı shopping district", "neighborhoods"],
        ["Galata neighborhood", "neighborhoods"],
        ["Asian side areas", "neighborhoods"],
        ["Trendy neighborhoods", "neighborhoods"],
        ["Historic districts", "neighborhoods"],
        ["Bosphorus neighborhoods", "neighborhoods"],
        ["Where to stay in Istanbul?", "neighborhoods"],
        ["Best area for tourists", "neighborhoods"],
        ["Local neighborhoods", "neighborhoods"],
        ["Nightlife districts", "neighborhoods"],
        ["Family-friendly areas", "neighborhoods"],
        ["Waterfront neighborhoods", "neighborhoods"],
        ["Cultural districts", "neighborhoods"],
        ["Shopping areas", "neighborhoods"],
        ["Residential neighborhoods", "neighborhoods"],
    ],
}

def create_balanced_dataset():
    """Create balanced bilingual training dataset"""
    print("=" * 70)
    print("Creating Balanced Bilingual Training Dataset")
    print("=" * 70)
    
    # Load existing data
    print("\n📥 Loading existing training data...")
    existing_data = load_training_data('comprehensive_training_data.json')
    
    # Analyze current distribution
    print("\n📊 Analyzing current language distribution...")
    lang_dist = analyze_language_distribution(existing_data)
    
    # Separate Turkish and English
    turkish_examples = []
    english_examples = []
    
    for query, intent in existing_data:
        if detect_language(query) == 'turkish':
            turkish_examples.append([query, intent])
        else:
            english_examples.append([query, intent])
    
    print(f"   Current Turkish examples: {len(turkish_examples)}")
    print(f"   Current English examples: {len(english_examples)}")
    
    # Add comprehensive English examples
    print("\n📝 Adding comprehensive English examples...")
    new_english_count = 0
    
    for intent, examples in ENGLISH_EXAMPLES.items():
        for example in examples:
            # Check if not already in dataset
            if example not in english_examples and example not in existing_data:
                english_examples.append(example)
                new_english_count += 1
    
    print(f"   Added {new_english_count} new English examples")
    print(f"   Total English examples: {len(english_examples)}")
    
    # Balance the dataset
    print("\n⚖️  Balancing dataset...")
    target_size = max(len(turkish_examples), len(english_examples))
    
    # If we need more samples, we keep all
    # Otherwise we can sample to balance
    balanced_data = turkish_examples + english_examples
    
    print(f"   Final Turkish examples: {len(turkish_examples)}")
    print(f"   Final English examples: {len(english_examples)}")
    print(f"   Total balanced examples: {len(balanced_data)}")
    
    turkish_pct = len(turkish_examples) / len(balanced_data) * 100
    english_pct = len(english_examples) / len(balanced_data) * 100
    
    print(f"   Turkish: {turkish_pct:.1f}%")
    print(f"   English: {english_pct:.1f}%")
    
    # Save balanced dataset
    print("\n💾 Saving balanced dataset...")
    output_file = 'comprehensive_training_data_balanced.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)
    
    print(f"   Saved to: {output_file}")
    
    # Intent distribution
    print("\n📊 Intent distribution in balanced dataset:")
    intent_counts = Counter([intent for _, intent in balanced_data])
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        lang_info = lang_dist.get(intent, {'turkish': 0, 'english': 0, 'total': 0})
        tr_count = sum(1 for q, i in balanced_data if i == intent and detect_language(q) == 'turkish')
        en_count = sum(1 for q, i in balanced_data if i == intent and detect_language(q) == 'english')
        print(f"   {intent:25s}: {count:4d} total (TR: {tr_count:3d}, EN: {en_count:3d})")
    
    print("\n" + "=" * 70)
    print("✅ Balanced dataset created successfully!")
    print("=" * 70)
    print(f"\n📁 Next step: Train with balanced dataset")
    print(f"   Update training script to use: {output_file}")

if __name__ == "__main__":
    create_balanced_dataset()

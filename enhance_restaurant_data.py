#!/usr/bin/env python3
"""
Add High-Quality Bilingual Restaurant Training Data
Focuses on clear restaurant intent signals for both English and Turkish
"""

import json
from pathlib import Path


def get_restaurant_training_data():
    """Get comprehensive bilingual restaurant training data"""
    return [
        # English - Direct restaurant queries
        {"text": "Where can I eat?", "intent": "restaurant"},
        {"text": "I need a restaurant", "intent": "restaurant"},
        {"text": "Looking for a place to eat", "intent": "restaurant"},
        {"text": "Good restaurants nearby", "intent": "restaurant"},
        {"text": "Best place to eat lunch", "intent": "restaurant"},
        {"text": "Where should I have dinner?", "intent": "restaurant"},
        {"text": "Restaurant recommendations", "intent": "restaurant"},
        {"text": "I'm hungry, where can I eat?", "intent": "restaurant"},
        {"text": "Find me a restaurant", "intent": "restaurant"},
        {"text": "Good food places", "intent": "restaurant"},
        
        # English - Specific cuisines
        {"text": "Turkish restaurant recommendations", "intent": "restaurant"},
        {"text": "Best kebab place", "intent": "restaurant"},
        {"text": "Where to eat seafood", "intent": "restaurant"},
        {"text": "Good fish restaurant", "intent": "restaurant"},
        {"text": "Traditional Turkish food", "intent": "restaurant"},
        {"text": "Authentic local cuisine", "intent": "restaurant"},
        {"text": "Best meze places", "intent": "restaurant"},
        {"text": "Where to try baklava", "intent": "restaurant"},
        {"text": "Turkish breakfast spots", "intent": "restaurant"},
        {"text": "Best döner place", "intent": "restaurant"},
        
        # English - Meal times
        {"text": "Where to have breakfast", "intent": "restaurant"},
        {"text": "Good lunch spots", "intent": "restaurant"},
        {"text": "Dinner restaurant suggestions", "intent": "restaurant"},
        {"text": "Late night food options", "intent": "restaurant"},
        {"text": "Brunch places in Istanbul", "intent": "restaurant"},
        
        # English - With location
        {"text": "Restaurants in Sultanahmet", "intent": "restaurant"},
        {"text": "Where to eat in Beyoğlu", "intent": "restaurant"},
        {"text": "Good restaurants near Taksim", "intent": "restaurant"},
        {"text": "Kadıköy food scene", "intent": "restaurant"},
        {"text": "Best restaurants in Beşiktaş", "intent": "restaurant"},
        {"text": "Eating options in Karaköy", "intent": "restaurant"},
        
        # English - Price/budget
        {"text": "Cheap restaurants", "intent": "restaurant"},
        {"text": "Budget-friendly eating places", "intent": "restaurant"},
        {"text": "Affordable restaurants", "intent": "restaurant"},
        {"text": "Best value restaurants", "intent": "restaurant"},
        {"text": "Expensive fine dining", "intent": "restaurant"},
        {"text": "Luxury restaurant recommendations", "intent": "restaurant"},
        
        # English - Special requirements
        {"text": "Vegetarian restaurants", "intent": "restaurant"},
        {"text": "Vegan food options", "intent": "restaurant"},
        {"text": "Halal restaurants", "intent": "restaurant"},
        {"text": "Gluten-free dining", "intent": "restaurant"},
        {"text": "Family-friendly restaurants", "intent": "restaurant"},
        {"text": "Restaurants with kids menu", "intent": "restaurant"},
        {"text": "Romantic dinner spots", "intent": "restaurant"},
        {"text": "Bosphorus view restaurants", "intent": "restaurant"},
        
        # Turkish - Direct queries
        {"text": "Nerede yemek yiyebilirim?", "intent": "restaurant"},
        {"text": "Bir restoran lazım", "intent": "restaurant"},
        {"text": "Yemek yemek için yer arıyorum", "intent": "restaurant"},
        {"text": "Yakınımda iyi restoranlar", "intent": "restaurant"},
        {"text": "Öğle yemeği için en iyi yer", "intent": "restaurant"},
        {"text": "Akşam yemeğini nerede yesem?", "intent": "restaurant"},
        {"text": "Restoran önerileri", "intent": "restaurant"},
        {"text": "Acıktım, nerede yemek yiyebilirim?", "intent": "restaurant"},
        {"text": "Bana bir restoran bul", "intent": "restaurant"},
        {"text": "İyi yemek yerleri", "intent": "restaurant"},
        
        # Turkish - Specific cuisines
        {"text": "Türk restoranı önerileri", "intent": "restaurant"},
        {"text": "En iyi kebapçı", "intent": "restaurant"},
        {"text": "Nerede deniz mahsulleri yenir", "intent": "restaurant"},
        {"text": "İyi balık lokantası", "intent": "restaurant"},
        {"text": "Geleneksel Türk yemekleri", "intent": "restaurant"},
        {"text": "Otantik yerel mutfak", "intent": "restaurant"},
        {"text": "En iyi meze yerleri", "intent": "restaurant"},
        {"text": "Nerede baklava denerim", "intent": "restaurant"},
        {"text": "Türk kahvaltısı mekanları", "intent": "restaurant"},
        {"text": "En iyi dönerci", "intent": "restaurant"},
        
        # Turkish - Meal times
        {"text": "Kahvaltı nerede yapabilirim", "intent": "restaurant"},
        {"text": "İyi öğle yemeği mekanları", "intent": "restaurant"},
        {"text": "Akşam yemeği restoran önerileri", "intent": "restaurant"},
        {"text": "Geç saatte yemek seçenekleri", "intent": "restaurant"},
        {"text": "İstanbul'da brunch yerleri", "intent": "restaurant"},
        
        # Turkish - With location
        {"text": "Sultanahmet'te restoranlar", "intent": "restaurant"},
        {"text": "Beyoğlu'nda nerede yemek yenir", "intent": "restaurant"},
        {"text": "Taksim yakınında iyi restoranlar", "intent": "restaurant"},
        {"text": "Kadıköy yemek sahnesi", "intent": "restaurant"},
        {"text": "Beşiktaş'ta en iyi restoranlar", "intent": "restaurant"},
        {"text": "Karaköy'de yemek seçenekleri", "intent": "restaurant"},
        
        # Turkish - Price/budget
        {"text": "Ucuz restoranlar", "intent": "restaurant"},
        {"text": "Bütçeye uygun yemek yerleri", "intent": "restaurant"},
        {"text": "Uygun fiyatlı restoranlar", "intent": "restaurant"},
        {"text": "En iyi değer restoranlar", "intent": "restaurant"},
        {"text": "Pahalı lüks yemek", "intent": "restaurant"},
        {"text": "Lüks restoran önerileri", "intent": "restaurant"},
        
        # Turkish - Special requirements
        {"text": "Vejeteryan restoranlar", "intent": "restaurant"},
        {"text": "Vegan yemek seçenekleri", "intent": "restaurant"},
        {"text": "Helal restoranlar", "intent": "restaurant"},
        {"text": "Glutensiz yemek", "intent": "restaurant"},
        {"text": "Aile dostu restoranlar", "intent": "restaurant"},
        {"text": "Çocuk menüsü olan restoranlar", "intent": "restaurant"},
        {"text": "Romantik akşam yemeği mekanları", "intent": "restaurant"},
        {"text": "Boğaz manzaralı restoranlar", "intent": "restaurant"},
        
        # More explicit restaurant phrases (English)
        {"text": "I want to eat at a restaurant", "intent": "restaurant"},
        {"text": "Take me to a restaurant", "intent": "restaurant"},
        {"text": "Restaurant near me", "intent": "restaurant"},
        {"text": "Food recommendations", "intent": "restaurant"},
        {"text": "Where's a good restaurant", "intent": "restaurant"},
        {"text": "Dining options", "intent": "restaurant"},
        {"text": "Place to grab food", "intent": "restaurant"},
        {"text": "Eating establishment", "intent": "restaurant"},
        {"text": "Food venue suggestions", "intent": "restaurant"},
        {"text": "Cuisine recommendations", "intent": "restaurant"},
        
        # More explicit restaurant phrases (Turkish)
        {"text": "Bir restoranda yemek yemek istiyorum", "intent": "restaurant"},
        {"text": "Beni bir restorana götür", "intent": "restaurant"},
        {"text": "Yakınımda restoran", "intent": "restaurant"},
        {"text": "Yemek önerileri", "intent": "restaurant"},
        {"text": "İyi bir restoran nerede", "intent": "restaurant"},
        {"text": "Yemek seçenekleri", "intent": "restaurant"},
        {"text": "Yemek yiyebileceğim yer", "intent": "restaurant"},
        {"text": "Yemek mekanı", "intent": "restaurant"},
        {"text": "Yemek yeri önerileri", "intent": "restaurant"},
        {"text": "Mutfak önerileri", "intent": "restaurant"},
        
        # Strong restaurant signals (English)
        {"text": "Recommend a restaurant", "intent": "restaurant"},
        {"text": "Suggest a place to eat", "intent": "restaurant"},
        {"text": "Need food recommendations", "intent": "restaurant"},
        {"text": "Best place to dine", "intent": "restaurant"},
        {"text": "Top rated restaurants", "intent": "restaurant"},
        {"text": "Highly rated dining", "intent": "restaurant"},
        {"text": "Popular restaurants", "intent": "restaurant"},
        {"text": "Famous food places", "intent": "restaurant"},
        {"text": "Must-try restaurants", "intent": "restaurant"},
        {"text": "Award-winning restaurants", "intent": "restaurant"},
        
        # Strong restaurant signals (Turkish)
        {"text": "Bir restoran öner", "intent": "restaurant"},
        {"text": "Yemek yiyeceğim yer öner", "intent": "restaurant"},
        {"text": "Yemek önerilerine ihtiyacım var", "intent": "restaurant"},
        {"text": "Yemek için en iyi yer", "intent": "restaurant"},
        {"text": "En yüksek puanlı restoranlar", "intent": "restaurant"},
        {"text": "Yüksek puanlı yemek yerleri", "intent": "restaurant"},
        {"text": "Popüler restoranlar", "intent": "restaurant"},
        {"text": "Ünlü yemek yerleri", "intent": "restaurant"},
        {"text": "Mutlaka denenmesi gereken restoranlar", "intent": "restaurant"},
        {"text": "Ödüllü restoranlar", "intent": "restaurant"},
    ]


def add_restaurant_data_to_training_set():
    """Add restaurant data to existing training set"""
    input_file = Path("comprehensive_training_data_10_intents.json")
    output_file = Path("comprehensive_training_data_10_intents_enhanced.json")
    
    # Load existing data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    training_data = data['training_data'] if isinstance(data, dict) else data
    
    # Get new restaurant data
    new_restaurant_data = get_restaurant_training_data()
    
    # Count current restaurant samples
    current_restaurant_count = sum(1 for item in training_data if item['intent'] == 'restaurant')
    
    print(f"Current dataset: {len(training_data)} samples")
    print(f"Current restaurant samples: {current_restaurant_count} ({current_restaurant_count/len(training_data)*100:.1f}%)")
    print(f"Adding {len(new_restaurant_data)} new restaurant samples...")
    
    # Add new data
    training_data.extend(new_restaurant_data)
    
    # Calculate new stats
    new_restaurant_count = sum(1 for item in training_data if item['intent'] == 'restaurant')
    
    print(f"\nNew dataset: {len(training_data)} samples")
    print(f"New restaurant samples: {new_restaurant_count} ({new_restaurant_count/len(training_data)*100:.1f}%)")
    
    # Save enhanced dataset
    output_data = {'training_data': training_data} if isinstance(data, dict) else training_data
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Enhanced dataset saved to: {output_file}")
    
    # Print intent distribution
    intent_counts = {}
    for item in training_data:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\n📊 New Intent Distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {intent}: {count} ({count/len(training_data)*100:.1f}%)")


if __name__ == "__main__":
    add_restaurant_data_to_training_set()

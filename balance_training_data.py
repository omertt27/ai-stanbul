#!/usr/bin/env name="python3
"""
Balance Training Data for All 10 Intents
Adds targeted data for underperforming intents: route_planning, attraction, neighborhood
"""

import json
from pathlib import Path


def get_route_planning_data():
    """Route planning specific queries"""
    return [
        # English
        {"text": "Plan my Istanbul trip", "intent": "route_planning"},
        {"text": "Create an itinerary", "intent": "route_planning"},
        {"text": "Help me plan my day", "intent": "route_planning"},
        {"text": "3-day Istanbul itinerary", "intent": "route_planning"},
        {"text": "One day in Istanbul", "intent": "route_planning"},
        {"text": "How to spend a week in Istanbul", "intent": "route_planning"},
        {"text": "Plan a sightseeing route", "intent": "route_planning"},
        {"text": "Optimize my travel plan", "intent": "route_planning"},
        {"text": "Create a walking tour", "intent": "route_planning"},
        {"text": "Best route for tourists", "intent": "route_planning"},
        {"text": "Plan a day trip", "intent": "route_planning"},
        {"text": "Organize my visit", "intent": "route_planning"},
        {"text": "Schedule my Istanbul tour", "intent": "route_planning"},
        {"text": "Map out my trip", "intent": "route_planning"},
        {"text": "Design my itinerary", "intent": "route_planning"},
        {"text": "Put together a travel plan", "intent": "route_planning"},
        {"text": "Arrange my sightseeing", "intent": "route_planning"},
        {"text": "Structure my visit", "intent": "route_planning"},
        {"text": "Build an itinerary", "intent": "route_planning"},
        {"text": "Help plan my vacation", "intent": "route_planning"},
        
        # Turkish
        {"text": "İstanbul gezimi planla", "intent": "route_planning"},
        {"text": "Bir gezi programı oluştur", "intent": "route_planning"},
        {"text": "Günümü planlamama yardım et", "intent": "route_planning"},
        {"text": "3 günlük İstanbul programı", "intent": "route_planning"},
        {"text": "İstanbul'da bir gün", "intent": "route_planning"},
        {"text": "İstanbul'da bir hafta nasıl geçirilir", "intent": "route_planning"},
        {"text": "Gezilecek yer rotası planla", "intent": "route_planning"},
        {"text": "Seyahat planımı optimize et", "intent": "route_planning"},
        {"text": "Yürüyüş turu oluştur", "intent": "route_planning"},
        {"text": "Turistler için en iyi rota", "intent": "route_planning"},
        {"text": "Günlük gezi planla", "intent": "route_planning"},
        {"text": "Ziyaretimi organize et", "intent": "route_planning"},
        {"text": "İstanbul turumu programla", "intent": "route_planning"},
        {"text": "Gezimi haritala", "intent": "route_planning"},
        {"text": "Programımı tasarla", "intent": "route_planning"},
        {"text": "Seyahat planı hazırla", "intent": "route_planning"},
        {"text": "Gezilecekyerleri düzenle", "intent": "route_planning"},
        {"text": "Ziyaretimi yapılandır", "intent": "route_planning"},
        {"text": "Program oluştur", "intent": "route_planning"},
        {"text": "Tatilimi planlamaya yardım et", "intent": "route_planning"},
        
        # More specific route planning
        {"text": "What order should I visit places", "intent": "route_planning"},
        {"text": "How to organize my sightseeing", "intent": "route_planning"},
        {"text": "Best order to see attractions", "intent": "route_planning"},
        {"text": "Efficient Istanbul tour route", "intent": "route_planning"},
        {"text": "Minimize travel time between sites", "intent": "route_planning"},
        
        {"text": "Yerleri hangi sırayla gezmeliyim", "intent": "route_planning"},
        {"text": "Gezilecekyerleri nasıl organize edeyim", "intent": "route_planning"},
        {"text": "Gezilecek yerlerin en iyi sırası", "intent": "route_planning"},
        {"text": "Verimli İstanbul tur rotası", "intent": "route_planning"},
        {"text": "Yerler arası seyahat süresini azalt", "intent": "route_planning"},
    ]


def get_attraction_data():
    """Clearer attraction queries"""
    return [
        # English - Explicit attraction keywords
        {"text": "What attractions should I visit", "intent": "attraction"},
        {"text": "Tourist attractions in Istanbul", "intent": "attraction"},
        {"text": "Main sights to see", "intent": "attraction"},
        {"text": "Popular tourist spots", "intent": "attraction"},
        {"text": "Famous landmarks", "intent": "attraction"},
        {"text": "Historical monuments", "intent": "attraction"},
        {"text": "Must-see sights", "intent": "attraction"},
        {"text": "Top sights in Istanbul", "intent": "attraction"},
        {"text": "What sights are worth visiting", "intent": "attraction"},
        {"text": "Best tourist destinations", "intent": "attraction"},
        {"text": "Where are the main attractions", "intent": "attraction"},
        {"text": "Show me the landmarks", "intent": "attraction"},
        {"text": "Famous monuments to visit", "intent": "attraction"},
        {"text": "Important historical sites", "intent": "attraction"},
        {"text": "Top rated attractions", "intent": "attraction"},
        
        # Turkish - Explicit attraction keywords
        {"text": "Hangi turistik yerleri gezmeliyim", "intent": "attraction"},
        {"text": "İstanbul'da turistik yerler", "intent": "attraction"},
        {"text": "Görülmesi gereken başlıca yerler", "intent": "attraction"},
        {"text": "Popüler turist noktaları", "intent": "attraction"},
        {"text": "Ünlü simgesel yapılar", "intent": "attraction"},
        {"text": "Tarihi anıtlar", "intent": "attraction"},
        {"text": "Mutlaka görülmesi gerekenler", "intent": "attraction"},
        {"text": "İstanbul'un en iyi gezilecek yerleri", "intent": "attraction"},
        {"text": "Hangi yerler gezmeye değer", "intent": "attraction"},
        {"text": "En iyi turist destinasyonları", "intent": "attraction"},
        {"text": "Ana turistik yerler nerede", "intent": "attraction"},
        {"text": "Simgesel yapıları göster", "intent": "attraction"},
        {"text": "Gezilmesi gereken ünlü anıtlar", "intent": "attraction"},
        {"text": "Önemli tarihi alanlar", "intent": "attraction"},
        {"text": "En çok puan alan turistik yerler", "intent": "attraction"},
        
        # Specific attractions
        {"text": "Is Topkapi Palace worth it", "intent": "attraction"},
        {"text": "Tell me about Hagia Sophia", "intent": "attraction"},
        {"text": "Blue Mosque visiting hours", "intent": "attraction"},
        {"text": "Basilica Cistern information", "intent": "attraction"},
        {"text": "Grand Bazaar tour", "intent": "attraction"},
        
        {"text": "Topkapı Sarayı'na değer mi", "intent": "attraction"},
        {"text": "Ayasofya hakkında bilgi ver", "intent": "attraction"},
        {"text": "Sultanahmet Camii ziyaret saatleri", "intent": "attraction"},
        {"text": "Yerebatan Sarnıcı bilgileri", "intent": "attraction"},
        {"text": "Kapalıçarşı turu", "intent": "attraction"},
    ]


def get_neighborhood_data():
    """Clearer neighborhood queries"""
    return [
        # English
        {"text": "Tell me about Sultanahmet district", "intent": "neighborhood"},
        {"text": "What's Beyoğlu neighborhood like", "intent": "neighborhood"},
        {"text": "Describe Kadıköy area", "intent": "neighborhood"},
        {"text": "Information about Beşiktaş", "intent": "neighborhood"},
        {"text": "What is Karaköy known for", "intent": "neighborhood"},
        {"text": "Tell me about Balat", "intent": "neighborhood"},
        {"text": "Which neighborhood for nightlife", "intent": "neighborhood"},
        {"text": "Best area to stay", "intent": "neighborhood"},
        {"text": "Trendy neighborhoods", "intent": "neighborhood"},
        {"text": "Local neighborhoods to explore", "intent": "neighborhood"},
        {"text": "Hipster areas in Istanbul", "intent": "neighborhood"},
        {"text": "Traditional neighborhoods", "intent": "neighborhood"},
        {"text": "Modern districts", "intent": "neighborhood"},
        {"text": "Family-friendly areas", "intent": "neighborhood"},
        {"text": "Which district has best cafes", "intent": "neighborhood"},
        
        # Turkish  
        {"text": "Sultanahmet ilçesini anlat", "intent": "neighborhood"},
        {"text": "Beyoğlu semti nasıl", "intent": "neighborhood"},
        {"text": "Kadıköy bölgesini tanımla", "intent": "neighborhood"},
        {"text": "Beşiktaş hakkında bilgi", "intent": "neighborhood"},
        {"text": "Karaköy neyle ünlü", "intent": "neighborhood"},
        {"text": "Balat'ı anlat", "intent": "neighborhood"},
        {"text": "Gece hayatı için hangi semt", "intent": "neighborhood"},
        {"text": "Kalmak için en iyi bölge", "intent": "neighborhood"},
        {"text": "Moda semtler", "intent": "neighborhood"},
        {"text": "Keşfedilecek yerel semtler", "intent": "neighborhood"},
        {"text": "Hipster bölgeler", "intent": "neighborhood"},
        {"text": "Geleneksel mahalleler", "intent": "neighborhood"},
        {"text": "Modern ilçeler", "intent": "neighborhood"},
        {"text": "Aile dostu bölgeler", "intent": "neighborhood"},
        {"text": "Hangi ilçede en iyi kafeler var", "intent": "neighborhood"},
    ]


def balance_training_data():
    """Add balanced data to training set"""
    input_file = Path("comprehensive_training_data_10_intents_enhanced.json")
    output_file = Path("comprehensive_training_data_10_intents_balanced.json")
    
    # Load existing data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    training_data = data['training_data'] if isinstance(data, dict) else data
    
    print(f"Current dataset: {len(training_data)} samples")
    
    # Get new data
    route_data = get_route_planning_data()
    attraction_data = get_attraction_data()
    neighborhood_data = get_neighborhood_data()
    
    print(f"Adding {len(route_data)} route_planning samples")
    print(f"Adding {len(attraction_data)} attraction samples")
    print(f"Adding {len(neighborhood_data)} neighborhood samples")
    
    # Add new data
    training_data.extend(route_data)
    training_data.extend(attraction_data)
    training_data.extend(neighborhood_data)
    
    print(f"\nNew dataset: {len(training_data)} samples")
    
    # Save balanced dataset
    output_data = {'training_data': training_data} if isinstance(data, dict) else training_data
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Balanced dataset saved to: {output_file}")
    
    # Print intent distribution
    intent_counts = {}
    for item in training_data:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\n📊 New Intent Distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {intent}: {count} ({count/len(training_data)*100:.1f}%)")


if __name__ == "__main__":
    balance_training_data()

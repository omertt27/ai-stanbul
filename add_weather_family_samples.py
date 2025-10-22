#!/usr/bin/env python3
"""
Add targeted samples for weather, family_activities, and accommodation
Focus on the specific weak patterns we're seeing
"""

import json


def add_targeted_samples():
    """Add samples for weak intents"""
    
    # Load current dataset
    with open('final_bilingual_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Current dataset: {len(data)} samples")
    
    # Weather samples - both languages
    weather_samples = [
        # Turkish weather
        {"text": "Hava durumu nasıl?", "intent": "weather"},
        {"text": "Bugün hava nasıl olacak?", "intent": "weather"},
        {"text": "Yarın yağmur yağacak mı?", "intent": "weather"},
        {"text": "Hava sıcaklığı kaç derece?", "intent": "weather"},
        {"text": "Bu hafta hava nasıl?", "intent": "weather"},
        {"text": "Hava tahmini", "intent": "weather"},
        {"text": "Bugünkü hava durumu", "intent": "weather"},
        {"text": "Yağmur yağıyor mu?", "intent": "weather"},
        {"text": "Sıcaklık ne kadar?", "intent": "weather"},
        {"text": "Hava şartları", "intent": "weather"},
        {"text": "İstanbul'da hava nasıl?", "intent": "weather"},
        {"text": "Meteoroloji", "intent": "weather"},
        {"text": "Kar yağacak mı?", "intent": "weather"},
        {"text": "Rüzgar var mı?", "intent": "weather"},
        {"text": "Güneşli mi?", "intent": "weather"},
        
        # English weather
        {"text": "What's the weather?", "intent": "weather"},
        {"text": "How's the weather today?", "intent": "weather"},
        {"text": "Will it rain tomorrow?", "intent": "weather"},
        {"text": "What's the temperature?", "intent": "weather"},
        {"text": "Weather forecast", "intent": "weather"},
        {"text": "Is it raining?", "intent": "weather"},
        {"text": "How hot is it?", "intent": "weather"},
        {"text": "Weather conditions", "intent": "weather"},
        {"text": "Istanbul weather", "intent": "weather"},
        {"text": "Will it snow?", "intent": "weather"},
        {"text": "Is it sunny?", "intent": "weather"},
        {"text": "What's the weather like?", "intent": "weather"},
        {"text": "Temperature today", "intent": "weather"},
        {"text": "Weather report", "intent": "weather"},
        {"text": "Is it windy?", "intent": "weather"},
    ]
    
    # Family activities - more English samples
    family_samples = [
        # English family
        {"text": "Where to go with kids?", "intent": "family_activities"},
        {"text": "Kid-friendly places", "intent": "family_activities"},
        {"text": "Family activities", "intent": "family_activities"},
        {"text": "Things to do with children", "intent": "family_activities"},
        {"text": "Best for kids", "intent": "family_activities"},
        {"text": "Child-friendly restaurants", "intent": "family_activities"},
        {"text": "Places for families", "intent": "family_activities"},
        {"text": "What can we do with toddlers?", "intent": "family_activities"},
        {"text": "Kids playground", "intent": "family_activities"},
        {"text": "Family-friendly attractions", "intent": "family_activities"},
        {"text": "Where can children play?", "intent": "family_activities"},
        {"text": "Activities for young kids", "intent": "family_activities"},
        {"text": "Best parks for children", "intent": "family_activities"},
        {"text": "Family entertainment", "intent": "family_activities"},
        {"text": "Kid activities in Istanbul", "intent": "family_activities"},
        {"text": "Things for kids to do", "intent": "family_activities"},
        {"text": "Children's museum", "intent": "family_activities"},
        {"text": "Where to take the kids?", "intent": "family_activities"},
        {"text": "Family fun", "intent": "family_activities"},
        {"text": "Kid-friendly tours", "intent": "family_activities"},
        
        # Turkish family
        {"text": "Çocuklarla nereye gidilir?", "intent": "family_activities"},
        {"text": "Çocuk dostu yerler", "intent": "family_activities"},
        {"text": "Aile aktiviteleri", "intent": "family_activities"},
        {"text": "Çocuklarla yapılacak şeyler", "intent": "family_activities"},
        {"text": "Çocuk oyun alanı", "intent": "family_activities"},
        {"text": "Çocuk parkı", "intent": "family_activities"},
        {"text": "Aile için yerler", "intent": "family_activities"},
        {"text": "Çocuk etkinlikleri", "intent": "family_activities"},
        {"text": "Küçük çocuklar için", "intent": "family_activities"},
        {"text": "Çocuk müzesi", "intent": "family_activities"},
    ]
    
    # Accommodation - more English samples with "cheap/budget" context
    accommodation_samples = [
        # English accommodation
        {"text": "Looking for cheap hotel", "intent": "accommodation"},
        {"text": "Budget hotel", "intent": "accommodation"},
        {"text": "Affordable accommodation", "intent": "accommodation"},
        {"text": "Cheap place to stay", "intent": "accommodation"},
        {"text": "Inexpensive hotel", "intent": "accommodation"},
        {"text": "Budget-friendly hotel", "intent": "accommodation"},
        {"text": "Low-cost hotel", "intent": "accommodation"},
        {"text": "Economical accommodation", "intent": "accommodation"},
        {"text": "Cheap hostel", "intent": "accommodation"},
        {"text": "Budget guesthouse", "intent": "accommodation"},
        {"text": "Where to stay cheap?", "intent": "accommodation"},
        {"text": "Affordable hotel near", "intent": "accommodation"},
        {"text": "Budget stay", "intent": "accommodation"},
        {"text": "Cheap hotels in Istanbul", "intent": "accommodation"},
        {"text": "Find cheap accommodation", "intent": "accommodation"},
        
        # Turkish accommodation
        {"text": "Ucuz otel arıyorum", "intent": "accommodation"},
        {"text": "Uygun fiyatlı konaklama", "intent": "accommodation"},
        {"text": "Ucuz kalacak yer", "intent": "accommodation"},
        {"text": "Bütçe dostu otel", "intent": "accommodation"},
        {"text": "Ekonomik otel", "intent": "accommodation"},
    ]
    
    # Add all samples
    new_samples = weather_samples + family_samples + accommodation_samples
    data.extend(new_samples)
    
    print(f"Added {len(new_samples)} samples:")
    print(f"  - Weather: {len(weather_samples)}")
    print(f"  - Family: {len(family_samples)}")
    print(f"  - Accommodation: {len(accommodation_samples)}")
    print(f"New total: {len(data)} samples")
    
    # Save updated dataset
    with open('final_bilingual_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("✅ Dataset updated!")
    

if __name__ == "__main__":
    add_targeted_samples()

#!/usr/bin/env python3
"""
Targeted Training Data for Low-Performing Intents
Focus on: emergency (Turkish), family_activities (English), attraction (context)
"""

import json
import random

# Targeted samples for problematic intents
TARGETED_IMPROVEMENTS = {
    # Turkish emergency - "Acil durum" was misclassified
    "emergency": {
        "turkish": [
            "Acil durum!", "Acil yardım lazım", "Acil!", "Acil durum var",
            "Polise ihtiyacım var", "Polis çağırın", "Polis lazım",
            "Hastaneye gitmem gerekiyor", "Hastane nerede acil", "Acil hastane",
            "Ambulans çağırın", "Ambulans lazım", "Ambulans gerekli",
            "Kayboldum yardım", "Kayboldum ne yapmalıyım", "Kayıp durumdayım",
            "Pasaportumu kaybettim acil", "Pasaport kayboldu", "Evraklarımı kaybettim",
            "Çantam çalındı", "Hırsızlık oldu", "Soyuldum",
            "Kaza geçirdim", "Kaza oldu", "Yaralandım",
            "Tehlikedeyim", "Güvende değilim", "Korku içindeyim",
        ],
        "english": [
            "Emergency!", "Urgent help needed", "Emergency situation",
            "I need police", "Call the police now", "Police emergency",
            "Need hospital urgently", "Emergency hospital", "Medical emergency",
            "Call ambulance", "Need ambulance", "Ambulance required",
            "I'm lost help", "Lost and need help", "I'm in trouble",
            "Lost my passport urgently", "Passport stolen", "Lost documents",
            "Bag stolen", "I was robbed", "Theft happened",
            "Had an accident", "Accident occurred", "I'm injured",
            "I'm in danger", "Not safe", "Feeling threatened",
        ]
    },
    
    # Family activities - English low confidence
    "family_activities": {
        "english": [
            "Kids activities", "What to do with children", "Child friendly",
            "Family fun", "Activities for kids", "Children entertainment",
            "Where to take kids", "Kids attractions", "Family places",
            "Kid friendly restaurants", "Children's menu", "Family dining",
            "Playground nearby", "Parks for kids", "Children park",
            "Zoo", "Aquarium", "Theme park for kids",
            "Indoor activities kids", "Outdoor kids", "Educational activities",
            "Baby friendly", "Toddler activities", "Teen activities",
            "Family tour", "Kids tour", "Children's museum",
            "Entertainment for children", "Fun for kids", "Kids events",
            "Where can kids play", "Safe for children", "Age appropriate",
            "Stroller accessible", "High chair available", "Kids facilities",
            "Family package", "Kids discount", "Children free",
        ],
        "turkish": [
            "Çocuk aktiviteleri", "Çocuklarla ne yapılır", "Çocuk dostu",
            "Aile eğlencesi", "Çocuklar için aktiviteler", "Çocuk eğlencesi",
            "Çocukları nereye götüreyim", "Çocuk cazibe merkezi", "Aile yerleri",
            "Çocuk dostu restoranlar", "Çocuk menüsü", "Aile yemeği",
            "Yakında oyun parkı", "Çocuk parkları", "Çocuk bahçesi",
            "Hayvanat bahçesi", "Akvaryum", "Çocuklar için tema parkı",
        ]
    },
    
    # Attraction - Add context for "Boğaz turu" type queries
    "attraction": {
        "turkish": [
            "Boğaz turu", "Boğaz gezisi", "Boğazda gezi",
            "Boğaz turları ne zaman", "Boğaz turu fiyatları", "Boğaz vapuru",
            "Adalar turu", "Adalar gezisi", "Prens Adaları turu",
            "Tarihi yarımada turu", "Sultanahmet turu", "Eski şehir turu",
            "Mimari turlar", "Tarihi yapılar turu", "Kültür turu",
            "Gezi teknesi", "Tekne turu", "Vapur gezisi",
            "Turistik yerler", "Görülmesi gereken", "Mutlaka görülmeli",
        ],
        "english": [
            "Bosphorus tour", "Bosphorus cruise", "Boat tour Bosphorus",
            "Bosphorus trip", "Strait tour", "Ferry tour",
            "Islands tour", "Princes Islands tour", "Island hopping",
            "Historical peninsula tour", "Sultanahmet tour", "Old city tour",
            "Architectural tours", "Historical buildings tour", "Cultural tour",
            "Cruise boat", "Boat trip", "Ferry cruise",
            "Tourist spots", "Must see", "Top attractions",
        ]
    },
    
    # Additional improvements for confusing intents
    "gps_navigation": {
        "turkish": [
            "Konum göster", "Neredeyim", "GPS'te göster",
            "Haritada göster", "Konum bilgisi", "GPS koordinatları",
            "Navigasyon başlat", "Yol göster", "Nereden giderim",
        ],
        "english": [
            "Show location", "Where am I", "Show on GPS",
            "Show on map", "Location info", "GPS coordinates",
            "Start navigation", "Show directions", "How to get there",
        ]
    },
    
    "restaurant": {
        "turkish": [
            "Yemek yeri", "Yemek nerede yenir", "Restoran öner",
            "Balık nerede yenir", "Kebap yerleri", "Meze mekanları",
            "Boğaz kenarı restoran", "Deniz manzaralı restoran", "Manzaralı yemek",
        ],
        "english": [
            "Where to eat", "Food places", "Suggest restaurant",
            "Where to eat fish", "Kebab places", "Meze restaurants",
            "Bosphorus restaurant", "Sea view restaurant", "Restaurant with view",
        ]
    }
}


def create_targeted_dataset():
    """Create targeted improvement dataset"""
    print("=" * 80)
    print("CREATING TARGETED IMPROVEMENT DATASET")
    print("=" * 80)
    print()
    
    # Load existing enhanced dataset
    with open('enhanced_bilingual_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    original_count = len(dataset)
    
    # Add targeted samples
    added_by_intent = {}
    
    for intent, languages in TARGETED_IMPROVEMENTS.items():
        added_count = 0
        
        for lang, queries in languages.items():
            for query in queries:
                dataset.append({
                    "text": query,
                    "intent": intent
                })
                added_count += 1
        
        added_by_intent[intent] = added_count
    
    # Shuffle
    random.shuffle(dataset)
    
    new_count = len(dataset)
    total_added = new_count - original_count
    
    print(f"📊 Dataset Enhancement:")
    print(f"   Original: {original_count} samples")
    print(f"   Added: {total_added} targeted samples")
    print(f"   Total: {new_count} samples")
    print()
    
    print("📈 Samples added per intent:")
    for intent, count in sorted(added_by_intent.items()):
        print(f"   {intent:20s}: +{count:2d} samples")
    print()
    
    # Count final samples per intent
    intent_counts = {}
    for item in dataset:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("📊 Final samples per intent:")
    for intent in sorted(intent_counts.keys()):
        count = intent_counts[intent]
        marker = "🎯" if intent in added_by_intent else "  "
        print(f"   {marker} {intent:20s}: {count:3d} samples")
    print()
    
    # Save final dataset
    output_file = "final_bilingual_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved to: {output_file}")
    print()
    print("=" * 80)
    print("✅ TARGETED IMPROVEMENTS READY!")
    print("=" * 80)
    print()
    print("🎯 Improvements focus:")
    print("   • Emergency (Turkish) - Acil durum detection")
    print("   • Family activities (English) - Kids queries")
    print("   • Attraction - Boğaz turu context")
    print("   • GPS navigation - Clear distinction")
    print("   • Restaurant - Better context")
    print()
    print("Next: python3 quick_retrain.py")
    print()


if __name__ == "__main__":
    create_targeted_dataset()

#!/usr/bin/env python3
"""
Create massive bilingual training dataset with augmentation
Target: 3000+ samples for better 25-class classification
"""

import json
import random

# All 25 intents
INTENTS = [
    "accommodation", "attraction", "booking", "budget", "cultural_info",
    "emergency", "events", "family_activities", "food", "general_info",
    "gps_navigation", "hidden_gems", "history", "local_tips", "luxury",
    "museum", "nightlife", "price_info", "recommendation", "restaurant",
    "romantic", "route_planning", "shopping", "transportation", "weather"
]

# Comprehensive bilingual templates with variations
TEMPLATES = {
    "emergency": {
        "turkish": [
            "Acil", "Acil durum", "Yardım", "Yardım edin", "Kayboldum",
            "Polis lazım", "Polis çağırın", "Hastane", "Hastane nerede",
            "Ambulans", "Ambulans çağırın", "Tehlike", "Tehlikede", "Çantam çalındı",
            "Kayboldum yardım edin", "Acil yardım lazım", "Polis istiyorum",
            "En yakın hastane", "Acil sağlık yardımı", "Kaza oldu",
            "Çocuğum kayboldu", "Yardım edin lütfen", "Acil tıbbi yardım gerekli"
        ],
        "english": [
            "Emergency", "Help", "Help me", "I'm lost", "I am lost",
            "Need police", "Call police", "Hospital", "Where is hospital",
            "Ambulance", "Call ambulance", "Danger", "In danger", "Stolen bag",
            "Lost please help", "Emergency help", "I need police",
            "Nearest hospital", "Medical emergency", "Accident happened",
            "My child is lost", "Please help me", "Need immediate help",
            "Emergency assistance", "Urgent help needed", "Police station nearby"
        ]
    },
    "attraction": {
        "turkish": [
            "Ayasofya", "Topkapı", "Topkapı Sarayı", "Galata", "Galata Kulesi",
            "Sultanahmet", "Sultanahmet Camii", "Mavi Cami", "Yerebatan",
            "Yerebatan Sarnıcı", "Gezilecek yerler", "Görülecek yerler",
            "Turistik yerler", "Müze", "Boğaz", "Boğaz turu", "Tarihi yerler",
            "Ayasofya'yı görmek istiyorum", "Topkapı'yı gezmek istiyorum",
            "En güzel yerler", "İstanbul'da neler gezilir", "Önemli yerler",
            "Turistik mekanlar"
        ],
        "english": [
            "Hagia Sophia", "Topkapi", "Topkapi Palace", "Galata", "Galata Tower",
            "Sultanahmet", "Blue Mosque", "Basilica Cistern", "Grand Bazaar",
            "Places to visit", "Places to see", "Tourist attractions",
            "Sightseeing", "Bosphorus", "Bosphorus tour", "Bosphorus cruise",
            "Historical sites", "I want to visit Hagia Sophia",
            "I want to see Topkapi", "Best places", "What to see in Istanbul",
            "Important places", "Tourist spots", "Main attractions",
            "Must-see places", "Famous landmarks", "Top sights"
        ]
    },
    "restaurant": {
        "turkish": [
            "Restoran", "Yemek", "Lokanta", "Balık", "Kebap", "Meze",
            "Güzel restoran", "Restoran öner", "Nerede yemek yenir",
            "Balık restoranı", "Kebap nerede", "Türk mutfağı",
            "Deniz ürünleri", "Ucuz restoran", "Pahalı restoran",
            "İyi restoran", "Lezzetli yemek", "Yemek yenecek yer"
        ],
        "english": [
            "Restaurant", "Food", "Eat", "Dining", "Fish", "Kebab", "Meze",
            "Good restaurant", "Recommend restaurant", "Where to eat",
            "Fish restaurant", "Kebab place", "Turkish cuisine",
            "Seafood", "Cheap restaurant", "Expensive restaurant",
            "Best restaurant", "Delicious food", "Place to eat",
            "Restaurant recommendation", "Best places to eat", "Food options",
            "Dining options", "Local restaurants", "Turkish food"
        ]
    },
    "transportation": {
        "turkish": [
            "Metro", "Tramvay", "Otobüs", "Taksi", "Ulaşım", "Ferry",
            "Vapur", "Toplu taşıma", "İstanbulkart", "Marmaray",
            "Metro nasıl kullanılır", "Tramvay saatleri", "Otobüs hatları",
            "Nasıl giderim", "Nasıl ulaşırım"
        ],
        "english": [
            "Metro", "Tram", "Bus", "Taxi", "Transport", "Ferry",
            "Public transport", "Istanbulkart", "Marmaray",
            "How to use metro", "Tram schedule", "Bus routes",
            "How do I get", "How to reach", "Transportation",
            "Getting around", "Transit", "How to travel"
        ]
    },
    "weather": {
        "turkish": [
            "Hava", "Hava durumu", "Yağmur", "Güneş", "Sıcaklık", "Derece",
            "Yarın hava", "Bugün hava", "Yağmur yağar mı", "Soğuk",
            "Sıcak", "Hava nasıl", "Kaç derece"
        ],
        "english": [
            "Weather", "Rain", "Sunny", "Temperature", "Degrees",
            "Tomorrow weather", "Today weather", "Will it rain", "Cold",
            "Hot", "What's the weather", "How many degrees",
            "Weather forecast", "Climate", "Weather conditions"
        ]
    },
    "family_activities": {
        "turkish": [
            "Çocukla", "Çocuklarla", "Aile", "Çocuk dostu", "Oyun parkı",
            "Nereye gidebilirim çocuklarla", "Aile için", "Çocuk aktiviteleri",
            "Eğlence parkı", "Çocuk müzesi", "Aile gezisi"
        ],
        "english": [
            "With kids", "With children", "Family", "Child friendly",
            "Kids friendly", "Playground", "Where to go with kids",
            "For family", "Children activities", "Amusement park",
            "Kids museum", "Family trip", "Family activities",
            "What can children do", "Kid-friendly", "Family fun"
        ]
    },
    "accommodation": {
        "turkish": [
            "Otel", "Hostel", "Konaklama", "Nerede kalabilirim", "Pansiyon",
            "Ucuz otel", "Pahalı otel", "Lüks otel", "Butik otel",
            "Otel rezervasyonu", "Apart", "Oda", "Kalacak yer"
        ],
        "english": [
            "Hotel", "Hostel", "Accommodation", "Where to stay", "Inn",
            "Cheap hotel", "Expensive hotel", "Luxury hotel", "Boutique hotel",
            "Hotel reservation", "Apartment", "Room", "Place to stay",
            "Looking for hotel", "Budget hotel", "Affordable hotel",
            "Where should I stay", "Lodging", "Best hotels"
        ]
    },
    "gps_navigation": {
        "turkish": [
            "Konum", "GPS", "Navigasyon", "Harita", "Yol tarifi",
            "Konumum", "Neredeyim", "Yönlendir", "Rota", "Bul",
            "Konumumu göster", "Haritada göster"
        ],
        "english": [
            "Location", "GPS", "Navigation", "Map", "Directions",
            "My location", "Where am I", "Navigate", "Route", "Find",
            "Show location", "Show on map", "Get directions",
            "Navigate me", "Find route"
        ]
    },
    "museum": {
        "turkish": [
            "Müze", "Sanat galerisi", "Galeri", "Sergi", "İstanbul Modern",
            "Arkeoloji müzesi", "Pera Müzesi", "Hangi müzeleri gezmeliyim",
            "Müze giriş ücreti", "Müze saatleri"
        ],
        "english": [
            "Museum", "Art gallery", "Gallery", "Exhibition", "Istanbul Modern",
            "Archeology museum", "Pera Museum", "Which museums should I visit",
            "Museum entrance fee", "Museum hours", "Museum tickets",
            "Best museums", "Art museums", "Museum recommendations"
        ]
    },
    "shopping": {
        "turkish": [
            "Alışveriş", "Çarşı", "Pazar", "Market", "Mağaza", "AVM",
            "Kapalıçarşı", "Mısır Çarşısı", "Alışveriş merkezi",
            "Nerede alışveriş yapabilirim", "Butik"
        ],
        "english": [
            "Shopping", "Bazaar", "Market", "Store", "Shop", "Mall",
            "Grand Bazaar", "Spice Bazaar", "Shopping mall",
            "Where can I shop", "Boutique", "Shopping center",
            "Best shopping", "Shopping areas", "Where to shop"
        ]
    },
    "route_planning": {
        "turkish": [
            "Rota", "Güzergah", "İtinerare", "Plan", "Gezi planı",
            "Nasıl giderim", "En iyi rota", "Yol", "Güzergah planı",
            "Günlük plan"
        ],
        "english": [
            "Route", "Itinerary", "Plan", "Trip plan", "Travel plan",
            "How do I get to", "Best route", "Path", "Route plan",
            "Daily plan", "Journey plan", "Travel route"
        ]
    },
    "romantic": {
        "turkish": [
            "Romantik", "Çift", "Balayı", "Gün batımı", "Romantik yemek",
            "Romantik restoran", "Çiftler için", "Romantik yer",
            "Romantik gezinti"
        ],
        "english": [
            "Romantic", "Couple", "Honeymoon", "Sunset", "Romantic dinner",
            "Romantic restaurant", "For couples", "Romantic place",
            "Romantic walk", "Date night", "Romantic spots"
        ]
    },
    "nightlife": {
        "turkish": [
            "Gece hayatı", "Bar", "Kulüp", "Eğlence", "Canlı müzik",
            "DJ", "Dans", "Gece mekanları", "Müzik"
        ],
        "english": [
            "Nightlife", "Bar", "Club", "Entertainment", "Live music",
            "DJ", "Dancing", "Night venues", "Music", "Night clubs",
            "Party", "Night out"
        ]
    },
    "booking": {
        "turkish": [
            "Rezervasyon", "Ayırt", "Bilet", "Online rezervasyon",
            "Masa ayırt", "Randevu", "Yer ayırt"
        ],
        "english": [
            "Reservation", "Book", "Ticket", "Online booking",
            "Reserve table", "Appointment", "Reserve place",
            "Make reservation", "Book online"
        ]
    },
    "price_info": {
        "turkish": [
            "Fiyat", "Ücret", "Ne kadar", "Kaç para", "Maliyet",
            "Giriş ücreti", "Bilet fiyatı", "Ücretli mi", "Fiyat listesi",
            "Tram schedule"  # Common confusion
        ],
        "english": [
            "Price", "Cost", "How much", "Fee", "Entrance fee",
            "Ticket price", "Is it paid", "Price list", "Prices",
            "Cost information", "Fees", "How much does it cost"
        ]
    },
    "food": {
        "turkish": [
            "Yemek", "Türk mutfağı", "Kahvaltı", "Tatlı", "İçecek",
            "Sokak lezzetleri", "Yerel yemekler", "Geleneksel yemek"
        ],
        "english": [
            "Food", "Turkish cuisine", "Breakfast", "Dessert", "Drink",
            "Street food", "Local food", "Traditional food",
            "Best food", "Food tour", "Culinary"
        ]
    },
    "budget": {
        "turkish": [
            "Ucuz", "Bütçe", "Ekonomik", "Ücretsiz", "Düşük bütçe",
            "Pahalı mı", "Fiyatlı", "Hesaplı"
        ],
        "english": [
            "Cheap", "Budget", "Affordable", "Free", "Low budget",
            "Is it expensive", "Budget-friendly", "Inexpensive",
            "Cost-effective", "Budget options", "What can children do"  # Common confusion
        ]
    },
    "events": {
        "turkish": [
            "Etkinlik", "Festival", "Konser", "Gösteri", "Aktivite",
            "Ne yapılır", "Bugün ne var", "Etkinlik takvimi"
        ],
        "english": [
            "Event", "Festival", "Concert", "Show", "Activity",
            "What to do", "What's on", "Event calendar",
            "Things to do", "Activities today"
        ]
    },
    "hidden_gems": {
        "turkish": [
            "Gizli yerler", "Saklı yerler", "Yerel mekanlar", "Turistik olmayan",
            "Bilinmeyen yerler", "Yerel favoriler"
        ],
        "english": [
            "Hidden gems", "Secret places", "Local spots", "Off beaten path",
            "Non-touristy", "Local favorites", "Hidden places"
        ]
    },
    "history": {
        "turkish": [
            "Tarih", "Tarihi", "Geçmiş", "Hikaye", "Osmanlı", "Bizans",
            "Tarihçe", "Eski", "Tarihi bilgi"
        ],
        "english": [
            "History", "Historical", "Past", "Story", "Ottoman", "Byzantine",
            "Historic", "Ancient", "Historical info", "Historical background"
        ]
    },
    "cultural_info": {
        "turkish": [
            "Kültür", "Gelenek", "Örf", "Adet", "Kültürel", "Yerel kültür",
            "Geleneksel", "Kültürel bilgi"
        ],
        "english": [
            "Culture", "Tradition", "Custom", "Cultural", "Local culture",
            "Traditional", "Cultural info", "Cultural information"
        ]
    },
    "local_tips": {
        "turkish": [
            "İpucu", "Tavsiye", "Öneri", "Yerel tavsiyeleri", "Bilgi",
            "İpuçları", "Yerel bilgi"
        ],
        "english": [
            "Tip", "Advice", "Suggestion", "Local tips", "Information",
            "Tips", "Local advice", "Insider tips"
        ]
    },
    "luxury": {
        "turkish": [
            "Lüks", "Pahalı", "Lüks otel", "VIP", "Premium", "Üst düzey",
            "Lüks restoran", "Yüksek kalite"
        ],
        "english": [
            "Luxury", "Expensive", "Luxury hotel", "VIP", "Premium",
            "High-end", "Upscale", "Luxurious", "Luxury restaurant",
            "High quality"
        ]
    },
    "recommendation": {
        "turkish": [
            "Öneri", "Tavsiye", "Ne önerirsiniz", "Öner", "En iyisi",
            "Tavsiye eder misiniz", "Önerir misiniz"
        ],
        "english": [
            "Recommendation", "Suggest", "What do you recommend", "Recommend",
            "Best", "Suggestions", "Advice", "Your recommendation"
        ]
    },
    "general_info": {
        "turkish": [
            "Bilgi", "Hakkında", "Genel bilgi", "Nedir", "Ne", "Nasıl",
            "Anlat", "Genel", "İnfo"
        ],
        "english": [
            "Information", "About", "General info", "What is", "What",
            "How", "Tell me", "General", "Info"
        ]
    }
}

def create_massive_dataset():
    """Create large dataset with variations and augmentations"""
    samples = []
    
    for intent, languages in TEMPLATES.items():
        # Turkish samples
        for text in languages.get("turkish", []):
            samples.append({
                "text": text,
                "intent": intent,
                "language": "tr"
            })
            # Add variations with punctuation
            samples.append({
                "text": text + "?",
                "intent": intent,
                "language": "tr"
            })
            samples.append({
                "text": text.capitalize(),
                "intent": intent,
                "language": "tr"
            })
        
        # English samples
        for text in languages.get("english", []):
            samples.append({
                "text": text,
                "intent": intent,
                "language": "en"
            })
            # Add variations
            samples.append({
                "text": text + "?",
                "intent": intent,
                "language": "en"
            })
            samples.append({
                "text": text.lower(),
                "intent": intent,
                "language": "en"
            })
    
    return samples

def main():
    samples = create_massive_dataset()
    
    # Remove duplicates while preserving order
    seen = set()
    unique_samples = []
    for sample in samples:
        key = (sample['text'].lower(), sample['intent'])
        if key not in seen:
            seen.add(key)
            unique_samples.append(sample)
    
    dataset = {
        "metadata": {
            "version": "3.0",
            "created": "2024-10-22",
            "description": "Massive bilingual dataset with augmentation",
            "total_samples": len(unique_samples),
            "languages": ["Turkish", "English"],
            "intents": len(INTENTS)
        },
        "samples": unique_samples
    }
    
    # Save
    output_file = "final_bilingual_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Massive dataset created: {output_file}")
    print(f"   Total samples: {len(unique_samples)}")
    print(f"   Intents: {len(INTENTS)}")
    
    # Count per intent
    from collections import Counter
    intent_counts = Counter(s['intent'] for s in unique_samples)
    print(f"\n📊 Samples per intent:")
    for intent in sorted(INTENTS):
        count = intent_counts[intent]
        print(f"   {intent:20s}: {count:3d} samples")

if __name__ == "__main__":
    main()

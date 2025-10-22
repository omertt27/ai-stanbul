#!/usr/bin/env python3
"""
Create comprehensive bilingual training dataset
Focus on improving English performance and fixing known issues
"""

import json

# All 25 intents
INTENTS = [
    "accommodation", "attraction", "booking", "budget", "cultural_info",
    "emergency", "events", "family_activities", "food", "general_info",
    "gps_navigation", "hidden_gems", "history", "local_tips", "luxury",
    "museum", "nightlife", "price_info", "recommendation", "restaurant",
    "romantic", "route_planning", "shopping", "transportation", "weather"
]

# Comprehensive training data with heavy English focus
training_data = {
    "emergency": {
        "turkish": [
            "Acil yardım edin", "Kayboldum", "Polis lazım", "Hastane nerede",
            "Ambulans çağırın", "Tehlike var", "Yardım!", "Acil durum",
            "Kayboldum yardım edin", "Acil polis lazım", "Hastaneye gitmem gerekiyor",
            "Çantamı çaldılar", "Kayıp çocuk", "Acil tıbbi yardım"
        ],
        "english": [
            "Emergency help", "I'm lost", "Need police", "Where is hospital",
            "Call ambulance", "Danger", "Help!", "Emergency",
            "I'm lost please help", "Need police urgently", "Where is nearest hospital",
            "My bag was stolen", "Lost child", "Medical emergency", "Need help now",
            "Police station location", "Hospital emergency", "I need immediate help"
        ]
    },
    "attraction": {
        "turkish": [
            "Ayasofya", "Topkapı Sarayı", "Galata Kulesi", "Sultanahmet",
            "Ayasofya'yı gezmek istiyorum", "Topkapı Sarayı nerede",
            "Sultanahmet Camii", "Yerebatan Sarnıcı", "Kapalıçarşı",
            "Gezilecek yerler", "Görülecek yerler", "Turistik yerler",
            "Boğaz turu", "Tarihi yerler", "Önemli mekanlar"
        ],
        "english": [
            "Hagia Sophia", "Topkapi Palace", "Galata Tower", "Sultanahmet",
            "I want to visit Hagia Sophia", "Where is Topkapi Palace",
            "Blue Mosque", "Basilica Cistern", "Grand Bazaar",
            "Places to visit", "Tourist attractions", "Sightseeing",
            "Bosphorus tour", "Historical places", "Must-see places",
            "What to see in Istanbul", "Top attractions", "Famous landmarks",
            "Tourist spots", "Bosphorus cruise", "Main sights",
            "Tourist places", "Sightseeing tours"
        ]
    },
    "restaurant": {
        "turkish": [
            "Restoran", "Yemek yemek", "Balık lokantası", "Kebap nerede",
            "Güzel bir restoran öner", "Balık yemek için nereye gidebilirim",
            "Kebap nerede yenir", "Türk mutfağı", "Lezzetli yemek",
            "Ucuz restoran", "Pahalı restoran", "Deniz mahsulleri"
        ],
        "english": [
            "Restaurant", "Where to eat", "Fish restaurant", "Kebab place",
            "Recommend a good restaurant", "Where can I eat fish",
            "Best kebab place", "Turkish cuisine", "Delicious food",
            "Cheap restaurant", "Fine dining", "Seafood restaurant",
            "Good restaurant", "Best places to eat", "Food recommendations",
            "Where to dine", "Restaurant suggestions", "Eating options",
            "Best food", "Local restaurants", "Turkish food"
        ]
    },
    "transportation": {
        "turkish": [
            "Metro", "Tramvay", "Otobüs", "Taksi", "Ulaşım",
            "Metro nasıl kullanılır", "Tramvay saatleri",
            "Otobüs hatları", "İstanbulkart", "Marmaray",
            "Toplu taşıma", "Vapur", "Ferry"
        ],
        "english": [
            "Metro", "Tram", "Bus", "Taxi", "Transport",
            "How to use metro", "Tram schedule",
            "Bus routes", "Istanbulkart", "Marmaray",
            "Public transport", "Ferry", "Transportation options",
            "Getting around", "Transit", "How to travel",
            "Metro map", "Bus schedule", "Transport info"
        ]
    },
    "weather": {
        "turkish": [
            "Hava durumu", "Yağmur", "Güneş", "Sıcaklık", "Soğuk", "Sıcak",
            "Yarın hava nasıl olacak", "Bugün yağmur yağar mı",
            "Kaç derece", "Hava tahmini", "Meteoroloji"
        ],
        "english": [
            "Weather", "Rain", "Sunny", "Temperature", "Cold", "Hot",
            "What's the weather tomorrow", "Will it rain today",
            "Temperature today", "Weather forecast", "Climate",
            "Weather conditions", "Is it raining", "How hot is it",
            "Weather report", "Forecast", "Weather today"
        ]
    },
    "family_activities": {
        "turkish": [
            "Çocuklarla", "Aile için", "Çocuk dostu", "Oyun parkı",
            "Çocuklarla nereye gidebilirim", "Aile için aktiviteler",
            "Çocuklar için", "Eğlence parkı", "Çocuk etkinlikleri",
            "Aile gezisi", "Çocuk müzesi"
        ],
        "english": [
            "With kids", "Family friendly", "Children activities", "Playground",
            "Where to go with kids", "Family friendly activities",
            "For children", "Amusement park", "Kids events",
            "Family trip", "Children's museum", "What can children do",
            "Family fun", "Kid-friendly places", "Activities for kids",
            "Family attractions", "Child-friendly", "Kids entertainment",
            "Things to do with family", "Children's activities"
        ]
    },
    "accommodation": {
        "turkish": [
            "Otel", "Hostel", "Konaklama", "Nerede kalabilirim", "Pansiyon",
            "Ucuz otel arıyorum", "Otel rezervasyonu",
            "Lüks otel", "Butik otel", "Apart otel"
        ],
        "english": [
            "Hotel", "Hostel", "Accommodation", "Where to stay", "Inn",
            "Looking for cheap hotel", "Hotel reservation",
            "Luxury hotel", "Boutique hotel", "Apart hotel",
            "Budget hotel", "Where should I stay", "Budget hostel recommendations",
            "Place to stay", "Hotel suggestions", "Affordable accommodation",
            "Best hotels", "Cheap places to stay", "Lodging options"
        ]
    },
    "gps_navigation": {
        "turkish": [
            "Konum", "GPS", "Navigasyon", "Harita", "Yol tarifi",
            "Konumumu göster", "Buradan Galata'ya rota",
            "Neredeyim", "Beni yönlendir", "Rota bul"
        ],
        "english": [
            "Location", "GPS", "Navigation", "Map", "Directions",
            "Show my location", "Navigate to Galata Tower",
            "Where am I", "Guide me", "Find route",
            "Show location", "Get directions", "Navigate me",
            "Location map", "Route to", "How to reach"
        ]
    },
    "museum": {
        "turkish": [
            "Müze", "Sanat galerisi", "Sergi", "Müze giriş ücreti",
            "Hangi müzeleri gezmeliyim", "İstanbul Modern",
            "Arkeoloji müzesi", "Pera Müzesi", "Müze saatleri"
        ],
        "english": [
            "Museum", "Art gallery", "Exhibition", "Museum entrance fee",
            "Which museums should I visit", "Istanbul Modern",
            "Archeology museum", "Pera Museum", "Museum hours",
            "Museum entrance fees", "Best museums", "Art museums",
            "Museum recommendations", "Museum tickets", "Gallery"
        ]
    },
    "shopping": {
        "turkish": [
            "Alışveriş", "Çarşı", "Market", "Mağaza", "AVM",
            "Kapalıçarşı nerede", "Alışveriş yapılacak yerler",
            "Mısır Çarşısı", "Alışveriş merkezi", "Butik"
        ],
        "english": [
            "Shopping", "Bazaar", "Market", "Store", "Mall",
            "Where is Grand Bazaar", "Best places for shopping",
            "Spice Bazaar", "Shopping mall", "Boutique",
            "Shopping center", "Where to shop", "Shopping areas",
            "Best shopping", "Markets", "Shopping districts"
        ]
    },
    "route_planning": {
        "turkish": [
            "Rota", "Güzergah", "İtinerare", "Plan", "Gezinti rotası",
            "Taksim'e nasıl giderim", "En iyi rota",
            "Gezi planı", "Rota önerisi", "Günlük plan"
        ],
        "english": [
            "Route", "Itinerary", "Plan", "Travel route", "Journey plan",
            "How do I get to Taksim", "Best route",
            "Trip plan", "Route suggestion", "Daily itinerary",
            "Travel plan", "Route planning", "Journey route"
        ]
    },
    "romantic": {
        "turkish": [
            "Romantik", "Çift için", "Balayı", "Gün batımı", "Romantik yemek",
            "Istanbul'da romantic dinner nerede", "Çiftler için mekan",
            "Romantik gezinti", "Romantik restoran"
        ],
        "english": [
            "Romantic", "For couples", "Honeymoon", "Sunset", "Romantic dinner",
            "Romantic restaurant", "Couple activities",
            "Romantic walk", "Date night", "Romantic places",
            "Couple-friendly", "Romantic spots", "Love"
        ]
    },
    "nightlife": {
        "turkish": [
            "Gece hayatı", "Bar", "Kulüp", "Canlı müzik", "Müzik",
            "Eğlence", "Gece mekanları", "DJ", "Dans"
        ],
        "english": [
            "Nightlife", "Bar", "Club", "Live music", "Music",
            "Entertainment", "Night venues", "DJ", "Dancing",
            "Night clubs", "Bars", "Party", "Night out"
        ]
    },
    "booking": {
        "turkish": [
            "Rezervasyon", "Ayırt", "Bilet al", "Online rezervasyon",
            "Masa ayır", "Randevu", "Bilet rezervasyonu"
        ],
        "english": [
            "Reservation", "Book", "Buy ticket", "Online booking",
            "Reserve table", "Appointment", "Ticket reservation",
            "Make reservation", "Book online", "Reserve"
        ]
    },
    "price_info": {
        "turkish": [
            "Fiyat", "Ücret", "Ne kadar", "Maliyet", "Bilet fiyatı",
            "Giriş ücreti", "Ücretli mi", "Fiyat listesi",
            "Tram schedule"  # This was being confused
        ],
        "english": [
            "Price", "Cost", "How much", "Fee", "Ticket price",
            "Entrance fee", "Is it paid", "Price list",
            "Prices", "Cost information", "Fees"
        ]
    },
    "food": {
        "turkish": [
            "Yemek", "Türk mutfağı", "Kahvaltı", "Tatlı", "İçecek",
            "Sokak lezzetleri", "Yerel yemekler"
        ],
        "english": [
            "Food", "Turkish cuisine", "Breakfast", "Dessert", "Drink",
            "Street food", "Local food", "Traditional food",
            "Best food", "Food tour", "Culinary"
        ]
    },
    "budget": {
        "turkish": [
            "Ucuz", "Bütçe", "Ekonomik", "Ücretsiz", "Pahalı mı",
            "Budget", "Düşük bütçe"
        ],
        "english": [
            "Cheap", "Budget", "Affordable", "Free", "Is it expensive",
            "Low budget", "Budget-friendly", "Inexpensive",
            "Budget options", "Cost-effective", "What can children do"  # This was being confused
        ]
    },
    "events": {
        "turkish": [
            "Etkinlik", "Festival", "Konser", "Gösteri", "Aktivite",
            "Ne yapılır", "Bugün ne var", "Etkinlik takvimi"
        ],
        "english": [
            "Event", "Festival", "Concert", "Show", "Activity",
            "What to do", "What's on today", "Event calendar",
            "Events", "Things to do", "Activities"
        ]
    },
    "hidden_gems": {
        "turkish": [
            "Gizli yerler", "Bilinmeyen yerler", "Yerel mekanlar", "Saklı cennetler",
            "Turistik olmayan", "Yerel favoriler"
        ],
        "english": [
            "Hidden gems", "Secret places", "Local spots", "Off the beaten path",
            "Non-touristy", "Local favorites", "Hidden places",
            "Where to go with kids"  # This was being confused - needs better separation
        ]
    },
    "history": {
        "turkish": [
            "Tarih", "Tarihi", "Geçmiş", "Hikaye", "Osmanlı",
            "Bizans", "Tarihçe", "Eski"
        ],
        "english": [
            "History", "Historical", "Past", "Story", "Ottoman",
            "Byzantine", "Historic", "Ancient", "Historical background"
        ]
    },
    "cultural_info": {
        "turkish": [
            "Kültür", "Gelenek", "Örf", "Adet", "Kültürel",
            "Yerel kültür", "Geleneksel"
        ],
        "english": [
            "Culture", "Tradition", "Custom", "Cultural",
            "Local culture", "Traditional", "Cultural information"
        ]
    },
    "local_tips": {
        "turkish": [
            "İpucu", "Tavsiye", "Öneri", "Yerel tavsiyeleri",
            "Bilgi", "İpuçları"
        ],
        "english": [
            "Tip", "Advice", "Suggestion", "Local tips",
            "Information", "Tips", "Recommendations", "Insider tips"
        ]
    },
    "luxury": {
        "turkish": [
            "Lüks", "Pahalı", "Lüks otel", "VIP", "Nerede kalabilirim",  # This was being confused
            "Premium", "Üst düzey"
        ],
        "english": [
            "Luxury", "Expensive", "Luxury hotel", "VIP",
            "Premium", "High-end", "Upscale", "Luxurious"
        ]
    },
    "recommendation": {
        "turkish": [
            "Öneri", "Tavsiye", "Ne önerirsiniz", "Öner",
            "En iyisi", "Tavsiye eder misiniz"
        ],
        "english": [
            "Recommendation", "Suggest", "What do you recommend", "Recommend",
            "Best", "Suggestions", "Advice"
        ]
    },
    "general_info": {
        "turkish": [
            "Bilgi", "Hakkında", "Genel bilgi", "Nedir", "Ne",
            "Nasıl", "Anlat"
        ],
        "english": [
            "Information", "About", "General info", "What is", "What",
            "How", "Tell me", "Info"
        ]
    }
}

def create_training_samples():
    """Create training samples in the required format"""
    samples = []
    
    for intent, languages in training_data.items():
        # Add Turkish samples
        for text in languages.get("turkish", []):
            samples.append({
                "text": text,
                "intent": intent,
                "language": "tr"
            })
        
        # Add English samples  
        for text in languages.get("english", []):
            samples.append({
                "text": text,
                "intent": intent,
                "language": "en"
            })
    
    return samples

def main():
    samples = create_training_samples()
    
    # Create dataset
    dataset = {
        "metadata": {
            "version": "2.0",
            "created": "2024-10-22",
            "description": "Comprehensive bilingual dataset with heavy English focus",
            "total_samples": len(samples),
            "languages": ["Turkish", "English"],
            "intents": len(INTENTS)
        },
        "samples": samples
    }
    
    # Save to file
    output_file = "final_bilingual_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Dataset created: {output_file}")
    print(f"   Total samples: {len(samples)}")
    print(f"   Intents: {len(INTENTS)}")
    print(f"\n📊 Samples per intent:")
    
    # Count samples per intent
    from collections import Counter
    intent_counts = Counter(s['intent'] for s in samples)
    for intent in sorted(INTENTS):
        count = intent_counts[intent]
        print(f"   {intent:20s}: {count:3d} samples")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create Bilingual (Turkish + English) Training Dataset
FREE solution - no translation API costs!
"""

import json
import random

# Bilingual training data for all 25 intents
BILINGUAL_TRAINING_DATA = {
    "accommodation": [
        # Turkish
        "Otel önerisi istiyorum", "Nerede kalabilirim?", "Ucuz hostel arıyorum",
        "5 yıldızlı otel var mı?", "Sultanahmet'te konaklama", "Butik otel önerisi",
        "Aile için otel", "Deniz manzaralı otel", "Şehir merkezinde konaklama",
        "Havuzlu otel arıyorum", "Spa'lı otel", "Kahvaltı dahil otel",
        # English
        "Hotel recommendations please", "Where can I stay?", "Looking for cheap hostel",
        "Any 5 star hotels?", "Accommodation in Sultanahmet", "Boutique hotel suggestions",
        "Hotel for family", "Hotel with sea view", "Stay in city center",
        "Looking for hotel with pool", "Hotel with spa", "Hotel with breakfast included",
    ],
    
    "attraction": [
        # Turkish
        "Ayasofya'yı görmek istiyorum", "Topkapı Sarayı nerede?", "Görülecek yerler",
        "En güzel yerler neresi?", "Galata Kulesi'ni ziyaret etmek istiyorum",
        "Tarihi yerler", "Müze tavsiyeleri", "Boğaz turu", "Kız Kulesi",
        "Yerebatan Sarnıcı", "Sultanahmet Meydanı", "Çamlıca Tepesi",
        # English
        "I want to see Hagia Sophia", "Where is Topkapi Palace?", "Places to visit",
        "What are the best places?", "I want to visit Galata Tower",
        "Historical places", "Museum recommendations", "Bosphorus tour", "Maiden's Tower",
        "Basilica Cistern", "Sultanahmet Square", "Camlica Hill",
    ],
    
    "booking": [
        # Turkish
        "Rezervasyon yapmak istiyorum", "Tur nasıl rezerve ederim?", "Bilet almak istiyorum",
        "Online rezervasyon", "Masa ayırtmak istiyorum", "Rezervasyon iptal",
        "Bilet fiyatları", "Grup rezervasyonu", "Son dakika rezervasyon",
        "Ön ödeme gerekli mi?", "Rezervasyon değişikliği", "Booking yapmak istiyorum",
        # English
        "I want to make a reservation", "How do I book a tour?", "I want to buy tickets",
        "Online booking", "I want to reserve a table", "Cancel reservation",
        "Ticket prices", "Group booking", "Last minute reservation",
        "Do I need prepayment?", "Change reservation", "I want to make a booking",
    ],
    
    "budget": [
        # Turkish
        "Ucuz seçenekler", "Bütçe dostu yerler", "Ekonomik tatil",
        "Para tasarrufu nasıl yapılır?", "En uygun fiyatlar", "Ücretsiz aktiviteler",
        "Ucuz yemek yerleri", "Student discounts", "Backpacker önerileri",
        "Bedava müze günleri", "Pahalı değil", "Hesaplı tatil",
        # English
        "Cheap options", "Budget friendly places", "Economical vacation",
        "How to save money?", "Best prices", "Free activities",
        "Cheap food places", "Student discounts", "Backpacker recommendations",
        "Free museum days", "Not expensive", "Affordable vacation",
    ],
    
    "cultural_info": [
        # Turkish
        "Türk kültürü hakkında bilgi", "Gelenekler nelerdir?", "Ramazan'da nelere dikkat etmeliyim?",
        "Kültürel özellikler", "Görgü kuralları", "Türk misafirperverliği",
        "Geleneksel törenler", "Dini günler", "Kültür hakkında",
        "Yerel adetler", "Türk kahvesi kültürü", "Hamam kültürü",
        # English
        "Information about Turkish culture", "What are the traditions?", "What should I know during Ramadan?",
        "Cultural features", "Etiquette rules", "Turkish hospitality",
        "Traditional ceremonies", "Religious holidays", "About culture",
        "Local customs", "Turkish coffee culture", "Hammam culture",
    ],
    
    "emergency": [
        # Turkish
        "Acil durum!", "Polisi arayın", "Hastaneye gitmem lazım",
        "Kayboldum yardım edin", "Pasaportumu kaybettim", "Ambulans çağırın",
        "Hırsızlık", "Kaza geçirdim", "İlaç lazım", "Doktor çağırın",
        "Güvende değilim", "Yardım!",
        # English
        "Emergency!", "Call the police", "I need to go to hospital",
        "I'm lost help me", "I lost my passport", "Call ambulance",
        "Theft", "I had an accident", "Need medicine", "Call a doctor",
        "I'm not safe", "Help!",
    ],
    
    "events": [
        # Turkish
        "Bu hafta hangi etkinlikler var?", "Festival ne zaman?", "Konser programı",
        "Etkinlik takvimi", "Ne yapabilirim bu akşam?", "Canlı müzik",
        "Açık hava konseri", "Sanat etkinlikleri", "Tiyatro gösterileri",
        "Spor etkinlikleri", "Yerel festivaller", "Kültür etkinlikleri",
        # English
        "What events this week?", "When is the festival?", "Concert schedule",
        "Event calendar", "What can I do tonight?", "Live music",
        "Outdoor concert", "Art events", "Theater shows",
        "Sports events", "Local festivals", "Cultural events",
    ],
    
    "family_activities": [
        # Turkish
        "Çocuklarla nereye gidilir?", "Aile dostu yerler", "Çocuk parkları",
        "Ailece yapılabilecek aktiviteler", "Çocuklar için etkinlikler",
        "Bebek arabası girebilir mi?", "Oyun alanları", "Aile restoranları",
        "Çocuk menüsü var mı?", "Lunapark", "Akvaryum", "Hayvanat bahçesi",
        # English
        "Where to go with kids?", "Family friendly places", "Children's parks",
        "Activities for families", "Events for children",
        "Is stroller accessible?", "Playgrounds", "Family restaurants",
        "Do you have kids menu?", "Amusement park", "Aquarium", "Zoo",
    ],
    
    "food": [
        # Turkish
        "Türk mutfağı hakkında", "Geleneksel yemekler", "Kahvaltı kültürü",
        "Baklava nerede yenir?", "Meze çeşitleri", "Kebap türleri",
        "Sokak lezzetleri", "Tatlılar", "İçecekler", "Balık ekmek",
        "Simit", "Börek çeşitleri",
        # English
        "About Turkish cuisine", "Traditional dishes", "Breakfast culture",
        "Where to eat baklava?", "Types of meze", "Types of kebab",
        "Street food", "Desserts", "Beverages", "Fish sandwich",
        "Simit", "Types of borek",
    ],
    
    "general_info": [
        # Turkish
        "İstanbul hakkında bilgi", "Şehir rehberi", "Genel bilgi",
        "İstanbul nasıl bir yer?", "Temel bilgiler", "Nüfus kaç?",
        "Tarih hakkında", "Coğrafya", "İklim", "Bölgeler",
        "İstanbul'un tarihi", "Şehir hakkında",
        # English
        "Information about Istanbul", "City guide", "General information",
        "What is Istanbul like?", "Basic information", "What's the population?",
        "About history", "Geography", "Climate", "Districts",
        "History of Istanbul", "About the city",
    ],
    
    "gps_navigation": [
        # Turkish
        "Konumumu göster", "GPS koordinatları", "Haritada göster",
        "En yakın yol neresi?", "Navigasyon başlat", "Yol tarifi",
        "Buraya nasıl giderim?", "Yönlendirme", "Rota bul",
        "Neredeyim?", "Konum paylaş", "Harita ver",
        # English
        "Show my location", "GPS coordinates", "Show on map",
        "What's the nearest route?", "Start navigation", "Directions",
        "How do I get here?", "Guidance", "Find route",
        "Where am I?", "Share location", "Give me map",
    ],
    
    "hidden_gems": [
        # Turkish
        "Gizli yerler", "Turistik olmayan yerler", "Yerel mekanlar",
        "Bilinmeyen güzellikler", "Keşfedilmemiş yerler", "Saklı cennetler",
        "Yerel halkın gittiği yerler", "Turist tuzaklarından uzak",
        "Özgün mekanlar", "Az bilinen yerler", "Off the beaten path",
        # English
        "Hidden places", "Non-touristy places", "Local spots",
        "Unknown beauties", "Undiscovered places", "Hidden gems",
        "Where locals go", "Away from tourist traps",
        "Authentic places", "Lesser known places", "Off the beaten path",
    ],
    
    "history": [
        # Turkish
        "İstanbul'un tarihi", "Bizans dönemi", "Osmanlı İmparatorluğu",
        "Tarihi bilgi", "Constantinopolis", "Fetih hikayesi",
        "Tarihi yapılar", "Antik şehir", "Geçmiş hakkında",
        "Tarih dersi", "Eski İstanbul", "Roma dönemi",
        # English
        "History of Istanbul", "Byzantine period", "Ottoman Empire",
        "Historical information", "Constantinople", "Conquest story",
        "Historical buildings", "Ancient city", "About the past",
        "History lesson", "Old Istanbul", "Roman period",
    ],
    
    "local_tips": [
        # Turkish
        "Yerel ipuçları", "İçeriden bilgiler", "Yerel halk ne yapıyor?",
        "Tavsiyelerin var mı?", "İpucu verin", "Insider tips",
        "Yerel sırları", "Püf noktaları", "Bilmen gerekenler",
        "Uzmanların önerileri", "Pro tips", "Pratik bilgiler",
        # English
        "Local tips", "Insider information", "What do locals do?",
        "Any recommendations?", "Give me tips", "Insider tips",
        "Local secrets", "Tricks", "Things you should know",
        "Expert recommendations", "Pro tips", "Practical information",
    ],
    
    "luxury": [
        # Turkish
        "Lüks restoranlar", "Pahalı oteller", "VIP deneyimler",
        "Premium hizmetler", "Özel turlar", "En iyi lukslar",
        "5 yıldızlı", "High-end", "Exclusive", "Prestijli mekanlar",
        "Michelin yıldızlı", "First class",
        # English
        "Luxury restaurants", "Expensive hotels", "VIP experiences",
        "Premium services", "Private tours", "Best luxury",
        "5 star", "High-end", "Exclusive", "Prestigious places",
        "Michelin starred", "First class",
    ],
    
    "museum": [
        # Turkish
        "Müze önerileri", "Hangi müzeleri gezmeliyim?", "Arkeoloji Müzesi nerede?",
        "Müze biletleri", "Sanat galerileri", "Modern sanat müzesi",
        "Tarih müzeleri", "Ücretsiz müze günleri", "Müze saatleri",
        "İstanbul Modern", "Pera Müzesi", "Sakıp Sabancı Müzesi",
        # English
        "Museum recommendations", "Which museums should I visit?", "Where is Archaeological Museum?",
        "Museum tickets", "Art galleries", "Modern art museum",
        "History museums", "Free museum days", "Museum hours",
        "Istanbul Modern", "Pera Museum", "Sakip Sabanci Museum",
    ],
    
    "nightlife": [
        # Turkish
        "Gece hayatı", "Bar önerileri", "Gece kulüpleri",
        "Eğlence mekanları", "Canlı müzik nerede?", "Dans edebileceğim yerler",
        "Rooftop barlar", "Pub crawl", "DJ performansları",
        "Kokteyl barları", "Beyoğlu gece hayatı", "Ortaköy clubları",
        # English
        "Nightlife", "Bar recommendations", "Night clubs",
        "Entertainment venues", "Where's live music?", "Places to dance",
        "Rooftop bars", "Pub crawl", "DJ performances",
        "Cocktail bars", "Beyoglu nightlife", "Ortakoy clubs",
    ],
    
    "price_info": [
        # Turkish
        "Fiyatlar ne kadar?", "Giriş ücreti", "Ne kadar para gerekir?",
        "Maliyet", "Ücret bilgisi", "Bilet fiyatı", "Ortalama fiyat",
        "Pahalı mı?", "Ücretsiz mi?", "Discount var mı?",
        "Fiyat listesi", "Ne kadar tutar?",
        # English
        "How much does it cost?", "Entrance fee", "How much money needed?",
        "Cost", "Price information", "Ticket price", "Average price",
        "Is it expensive?", "Is it free?", "Any discounts?",
        "Price list", "How much is it?",
    ],
    
    "recommendation": [
        # Turkish
        "Öneri istiyorum", "Ne önerirsiniz?", "En iyisi hangisi?",
        "Tavsiyeniz nedir?", "Yardım edin", "Önerilerin var mı?",
        "En iyi seçenek", "Ne yapmalıyım?", "Neresi daha iyi?",
        "Hangisini tercih etmeliyim?", "Fikrin nedir?", "Suggest something",
        # English
        "I want recommendations", "What do you recommend?", "Which is the best?",
        "What's your recommendation?", "Help me", "Any suggestions?",
        "Best option", "What should I do?", "Which is better?",
        "Which should I choose?", "What's your opinion?", "Suggest something",
    ],
    
    "restaurant": [
        # Turkish
        "Restoran önerisi", "En yakın restoran nerede?", "Balık restoranı",
        "Meze mekanı", "Yemek yiyebileceğim yer", "İyi bir restoran",
        "Kebapçı", "Vegetaryan restoran", "Deniz mahsulleri",
        "Manzaralı restoran", "Aile restoranı", "Romantik restoran",
        # English
        "Restaurant recommendation", "Where's the nearest restaurant?", "Fish restaurant",
        "Meze place", "Where can I eat?", "Good restaurant",
        "Kebab place", "Vegetarian restaurant", "Seafood",
        "Restaurant with view", "Family restaurant", "Romantic restaurant",
    ],
    
    "romantic": [
        # Turkish
        "Romantik yerler", "Çiftler için aktiviteler", "Balayı önerileri",
        "Gün batımı nerede izlenir?", "Romantik akşam yemeği",
        "Sevgiliye sürpriz", "Özel anlar için", "Evlenme teklifi yerleri",
        "Yıldönümü kutlaması", "Couple activities", "Honeymoon",
        # English
        "Romantic places", "Activities for couples", "Honeymoon recommendations",
        "Where to watch sunset?", "Romantic dinner",
        "Surprise for girlfriend", "For special moments", "Proposal places",
        "Anniversary celebration", "Couple activities", "Honeymoon",
    ],
    
    "route_planning": [
        # Turkish
        "Rota planla", "En iyi güzergah", "Nereden başlamalıyım?",
        "Gezilecek yerler sırası", "İtinerimi oluştur", "Günlük plan",
        "3 günlük gezi planı", "Optimum rota", "Hangi sırayla gezeyim?",
        "Zaman planlaması", "Tur programı", "Gezilecek yerler listesi",
        # English
        "Plan route", "Best itinerary", "Where should I start?",
        "Order of places to visit", "Create my itinerary", "Daily plan",
        "3 day trip plan", "Optimal route", "In which order should I visit?",
        "Time planning", "Tour program", "List of places to visit",
    ],
    
    "shopping": [
        # Turkish
        "Alışveriş merkezleri", "Kapalıçarşı nerede?", "Hediyelik eşya",
        "Marka mağazaları", "Pazar yerleri", "Outlet", "İndirimler",
        "Antika dükkanları", "Zanaat ürünleri", "Halı alışverişi",
        "Tekstil", "Mücevher dükkanları",
        # English
        "Shopping malls", "Where is Grand Bazaar?", "Souvenirs",
        "Brand stores", "Markets", "Outlet", "Discounts",
        "Antique shops", "Handicrafts", "Carpet shopping",
        "Textile", "Jewelry shops",
    ],
    
    "transportation": [
        # Turkish
        "Ulaşım nasıl?", "Metro hattı", "Otobüs saatleri",
        "Tramvay güzergahı", "Taksi bulmak", "İstanbulkart",
        "Havalimanına nasıl giderim?", "Vapur saatleri", "Marmaray",
        "Metrobüs", "Dolmuş", "Toplu taşıma",
        # English
        "How's transportation?", "Metro line", "Bus schedule",
        "Tram route", "Find taxi", "Istanbul card",
        "How to get to airport?", "Ferry schedule", "Marmaray",
        "Metrobus", "Dolmus", "Public transport",
    ],
    
    "weather": [
        # Turkish
        "Hava durumu nasıl?", "Yarın yağmur yağar mı?", "Sıcaklık kaç derece?",
        "Hava tahmini", "Bugün hava nasıl?", "Şemsiye lazım mı?",
        "Soğuk mu?", "Güneşli mi?", "Kar yağacak mı?",
        "Haftalık hava durumu", "Nem oranı", "Rüzgar var mı?",
        # English
        "How's the weather?", "Will it rain tomorrow?", "What's the temperature?",
        "Weather forecast", "How's weather today?", "Do I need umbrella?",
        "Is it cold?", "Is it sunny?", "Will it snow?",
        "Weekly weather", "Humidity", "Is it windy?",
    ],
}


def create_bilingual_dataset():
    """Create bilingual training dataset"""
    print("=" * 80)
    print("CREATING BILINGUAL TRAINING DATASET (Turkish + English)")
    print("=" * 80)
    print()
    
    dataset = []
    
    for intent, queries in BILINGUAL_TRAINING_DATA.items():
        for query in queries:
            dataset.append({
                "text": query,
                "intent": intent
            })
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Statistics
    total_samples = len(dataset)
    turkish_samples = sum(1 for item in dataset if any(c in 'çğıöşüÇĞİÖŞÜ' for c in item['text']))
    english_samples = total_samples - turkish_samples
    
    print(f"✅ Created bilingual dataset:")
    print(f"   Total samples: {total_samples}")
    print(f"   Turkish: ~{turkish_samples} ({turkish_samples/total_samples*100:.1f}%)")
    print(f"   English: ~{english_samples} ({english_samples/total_samples*100:.1f}%)")
    print(f"   Intents: {len(BILINGUAL_TRAINING_DATA)}")
    print()
    
    # Save dataset
    output_file = "bilingual_training_dataset.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved to: {output_file}")
    print()
    
    # Show samples per intent
    print("📊 Samples per intent:")
    intent_counts = {}
    for item in dataset:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    for intent in sorted(intent_counts.keys()):
        count = intent_counts[intent]
        print(f"   {intent:20s}: {count:3d} samples")
    
    print()
    print("=" * 80)
    print("✅ BILINGUAL DATASET READY FOR TRAINING!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Train model: python3 phase2_bilingual_training.py")
    print("2. Test model: python3 integration_test.py")
    print("3. Compare Turkish vs English accuracy")
    print()
    
    return dataset


if __name__ == "__main__":
    create_bilingual_dataset()

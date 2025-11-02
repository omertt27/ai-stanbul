#!/usr/bin/env python3
"""
Enhance training data for weak intents with high-quality bilingual examples
Focuses on: restaurant, attraction, route_planning, daily_talks, general_info
"""

import json
from typing import List, Dict

def generate_enhanced_data() -> Dict[str, List[str]]:
    """Generate high-quality, specific training examples"""
    
    return {
        "restaurant": {
            "english": [
                # Explicit restaurant requests
                "I want to eat at a restaurant",
                "Looking for a place to dine",
                "Recommend restaurants for dinner",
                "Where's a good restaurant nearby?",
                "Best place to eat lunch",
                "Restaurant recommendations please",
                "Need a restaurant suggestion",
                "Good restaurants around here?",
                "Where should I eat tonight?",
                "Restaurant with good food",
                "Find me a restaurant",
                "Dining options in this area",
                
                # Cuisine-specific
                "Turkish restaurant recommendations",
                "Seafood restaurant near Karaköy",
                "Vegetarian restaurants in Beyoğlu",
                "Kebab restaurants in Sultanahmet",
                "Best meze restaurants",
                "Italian restaurants in Nişantaşı",
                "Chinese food in Kadıköy",
                "Sushi restaurants Istanbul",
                
                # Context-specific
                "Restaurant for romantic dinner",
                "Family-friendly restaurants",
                "Budget restaurant recommendations",
                "Fancy restaurant for special occasion",
                "Restaurant with Bosphorus view",
                "Halal restaurants near me",
                "Late night restaurants open now",
                "Breakfast restaurants in Taksim",
                
                # Location + restaurant
                "Restaurants in Sultanahmet area",
                "Beyoğlu restaurants list",
                "Where to eat in Kadıköy?",
                "Karaköy restaurant suggestions",
                "Restaurants near Galata Tower",
                "Beşiktaş dining options",
            ],
            "turkish": [
                # Explicit restaurant requests
                "Restoran önerisi istiyorum",
                "Yemek yiyebileceğim yer arıyorum",
                "Akşam yemeği için restoran öner",
                "Yakında iyi restoran var mı?",
                "Öğle yemeği için en iyi yer",
                "Restoran önerileri lütfen",
                "Restoran önerisi lazım",
                "Buralarda iyi restoran var mı?",
                "Bu akşam nerede yemek yesem?",
                "İyi yemek yapan restoran",
                "Bana restoran bul",
                "Bu bölgede yemek yerleri",
                
                # Cuisine-specific
                "Türk restoranı önerisi",
                "Karaköy'de balık restoranı",
                "Beyoğlu'nda vejetaryen restoranlar",
                "Sultanahmet'te kebapçı",
                "En iyi meze restoranları",
                "Nişantaşı'nda İtalyan restoranı",
                "Kadıköy'de Çin yemeği",
                "İstanbul'da suşi restoranı",
                
                # Context-specific
                "Romantik akşam yemeği restoranı",
                "Aile dostu restoranlar",
                "Ucuz restoran önerileri",
                "Özel gün için lüks restoran",
                "Boğaz manzaralı restoran",
                "Yakınımda helal restoranlar",
                "Gece geç saatte açık restoran",
                "Taksim'de kahvaltı restoranı",
                
                # Location + restaurant
                "Sultanahmet bölgesinde restoranlar",
                "Beyoğlu restoran listesi",
                "Kadıköy'de nerede yenir?",
                "Karaköy restoran önerileri",
                "Galata Kulesi yakınında restoranlar",
                "Beşiktaş yemek yerleri",
            ]
        },
        
        "attraction": {
            "english": [
                # Explicit sightseeing
                "What attractions should I visit?",
                "Show me tourist sites",
                "Places to see in Istanbul",
                "Sightseeing recommendations",
                "What landmarks are must-see?",
                "Tourist attractions list",
                "Famous places to visit",
                "Must-visit attractions",
                "Historical sites to see",
                
                # Specific attractions
                "Tell me about Hagia Sophia",
                "Blue Mosque visiting info",
                "Topkapi Palace tour",
                "Galata Tower information",
                "Dolmabahçe Palace details",
                "Basilica Cistern visit",
                "Grand Bazaar tour",
                "Spice Market information",
                
                # Activity-based
                "Museums to visit in Istanbul",
                "Historical monuments list",
                "Religious sites to see",
                "Palaces in Istanbul",
                "Parks and gardens to visit",
                "Cultural sites recommendations",
                "UNESCO sites Istanbul",
                
                # Compound queries
                "Attractions near Sultanahmet",
                "What to see in Beyoğlu?",
                "Kadıköy tourist spots",
                "Beşiktaş landmarks",
                "Must-see places around Taksim",
            ],
            "turkish": [
                # Explicit sightseeing
                "Hangi yerleri gezmeli miyim?",
                "Turistik yerleri göster",
                "İstanbul'da gezilecek yerler",
                "Gezi önerileri",
                "Hangi anıtlar mutlaka görülmeli?",
                "Turistik yerler listesi",
                "Ünlü gezilecek yerler",
                "Mutlaka gezilmesi gereken yerler",
                "Tarihi yerler",
                
                # Specific attractions
                "Ayasofya hakkında bilgi",
                "Sultanahmet Camii ziyaret",
                "Topkapı Sarayı turu",
                "Galata Kulesi bilgileri",
                "Dolmabahçe Sarayı detayları",
                "Yerebatan Sarnıcı ziyareti",
                "Kapalıçarşı turu",
                "Mısır Çarşısı bilgisi",
                
                # Activity-based
                "İstanbul'da gezilecek müzeler",
                "Tarihi anıtlar listesi",
                "Dini yerler",
                "İstanbul'daki saraylar",
                "Gezilecek parklar ve bahçeler",
                "Kültürel mekanlar önerileri",
                "UNESCO bölgeleri İstanbul",
                
                # Compound queries
                "Sultanahmet'te gezilecek yerler",
                "Beyoğlu'nda ne gezilir?",
                "Kadıköy turistik noktaları",
                "Beşiktaş'ta gezilecek yerler",
                "Taksim çevresinde görülmesi gerekenler",
            ]
        },
        
        "route_planning": {
            "english": [
                # Explicit itinerary
                "Plan my day in Istanbul",
                "Create a 3-day itinerary",
                "Help me plan my trip",
                "What's the best route to see everything?",
                "Organize my Istanbul visit",
                "Daily tour plan needed",
                "Plan my sightseeing route",
                "Create travel itinerary",
                "Schedule my day of sightseeing",
                
                # Multi-stop routes
                "Route from Sultanahmet to Taksim via Galata",
                "Best route to visit 3 museums today",
                "Walking tour from Blue Mosque to Galata Tower",
                "Plan route: Hagia Sophia, Topkapi, then lunch",
                "Optimal path to see 5 attractions",
                "Day trip itinerary to Asian side",
                
                # Time-based planning
                "One day itinerary for Istanbul",
                "Morning to evening sightseeing plan",
                "Weekend trip plan",
                "Half-day tour suggestions",
                "2-day Istanbul itinerary",
                "Full week travel plan",
                
                # Specific planning
                "Historical sites tour route",
                "Museum route with lunch break",
                "Bosphorus day trip plan",
                "Old city walking route",
                "Asian side day plan",
            ],
            "turkish": [
                # Explicit itinerary
                "İstanbul'da günümü planla",
                "3 günlük gezi programı oluştur",
                "Gezimi planlamama yardım et",
                "Her şeyi görmek için en iyi rota nedir?",
                "İstanbul ziyaretimi organize et",
                "Günlük tur planı lazım",
                "Gezi rotamı planla",
                "Seyahat programı oluştur",
                "Gün boyu gezi programımı düzenle",
                
                # Multi-stop routes
                "Sultanahmet'ten Taksim'e Galata üzerinden rota",
                "Bugün 3 müze gezmek için en iyi rota",
                "Sultanahmet Camii'nden Galata Kulesi'ne yürüyüş turu",
                "Plan: Ayasofya, Topkapı, sonra öğle yemeği",
                "5 yeri görmek için optimal yol",
                "Anadolu yakasına günlük gezi planı",
                
                # Time-based planning
                "İstanbul için bir günlük plan",
                "Sabahtan akşama gezi programı",
                "Hafta sonu gezisi planı",
                "Yarım günlük tur önerileri",
                "2 günlük İstanbul programı",
                "Tam haftalık seyahat planı",
                
                # Specific planning
                "Tarihi yerler tur rotası",
                "Öğle yemeği molası ile müze rotası",
                "Boğaz gezisi günlük planı",
                "Eski şehir yürüyüş rotası",
                "Anadolu yakası gün planı",
            ]
        },
        
        "daily_talks": {
            "english": [
                # Greetings
                "Hello", "Hi there", "Hey", "Good morning",
                "Good afternoon", "Good evening", "Greetings",
                "Hi!", "Hello there", "Hey there",
                
                # Thanks
                "Thank you", "Thanks", "Thanks a lot",
                "Thank you so much", "Much appreciated",
                "Thanks for your help", "I appreciate it",
                "That's helpful, thanks", "Great, thank you",
                
                # Farewell
                "Goodbye", "Bye", "See you", "Take care",
                "Have a nice day", "Talk to you later",
                "Farewell", "Catch you later", "See you soon",
                
                # Help
                "Help me", "I need help", "Can you help?",
                "Assist me please", "Help", "I need assistance",
                
                # Small talk
                "How are you?", "What's up?", "How's it going?",
                "Nice to meet you", "Pleasure talking to you",
                "You're helpful", "This is great",
            ],
            "turkish": [
                # Greetings
                "Merhaba", "Selam", "Selamlar", "Günaydın",
                "İyi günler", "İyi akşamlar", "Tünaydın",
                "Merhaba!", "Selam!", "Selamlar!",
                
                # Thanks
                "Teşekkürler", "Teşekkür ederim", "Sağol",
                "Çok teşekkürler", "Çok sağol", "Minnettarım",
                "Yardımın için teşekkürler", "Takdir ediyorum",
                "Faydalı oldu, teşekkürler", "Harika, teşekkürler",
                
                # Farewell
                "Güle güle", "Hoşça kal", "Görüşürüz",
                "Kendine iyi bak", "İyi günler dilerim",
                "Sonra görüşürüz", "Elveda", "Yakında görüşürüz",
                
                # Help
                "Yardım et", "Yardıma ihtiyacım var",
                "Yardım edebilir misin?", "Lütfen yardım et",
                "Yardım", "Desteğe ihtiyacım var",
                
                # Small talk
                "Nasılsın?", "Ne haber?", "Naber?",
                "Tanıştığımıza memnun oldum", "Seninle konuşmak güzel",
                "Çok yardımcı oluyorsun", "Bu harika",
            ]
        },
        
        "general_info": {
            "english": [
                # Facts and info
                "Tell me about Istanbul",
                "Istanbul information",
                "Facts about Istanbul",
                "Istanbul city guide",
                "What is Istanbul known for?",
                "Istanbul history",
                "About Istanbul city",
                "Istanbul overview",
                "Istanbul details",
                
                # Demographics
                "Population of Istanbul",
                "How many people live in Istanbul?",
                "Istanbul city size",
                "Istanbul area facts",
                
                # Culture
                "Turkish culture information",
                "Istanbul traditions",
                "Local customs in Istanbul",
                "Istanbul lifestyle",
                "What's Istanbul like?",
                
                # Practical
                "Emergency numbers Istanbul",
                "Istanbul phone codes",
                "Currency in Turkey",
                "Visa requirements Turkey",
                "Safety in Istanbul",
                
                # General questions
                "What should I know about Istanbul?",
                "Istanbul travel tips",
                "Important Istanbul information",
                "Istanbul visitor guide",
            ],
            "turkish": [
                # Facts and info
                "İstanbul hakkında bilgi ver",
                "İstanbul bilgileri",
                "İstanbul hakkında gerçekler",
                "İstanbul şehir rehberi",
                "İstanbul neyle ünlü?",
                "İstanbul tarihi",
                "İstanbul şehri hakkında",
                "İstanbul genel bakış",
                "İstanbul detayları",
                
                # Demographics
                "İstanbul'un nüfusu",
                "İstanbul'da kaç kişi yaşıyor?",
                "İstanbul şehir büyüklüğü",
                "İstanbul alan bilgileri",
                
                # Culture
                "Türk kültürü bilgileri",
                "İstanbul gelenekleri",
                "İstanbul'da yerel görenekler",
                "İstanbul yaşam tarzı",
                "İstanbul nasıl bir yer?",
                
                # Practical
                "İstanbul acil numaraları",
                "İstanbul telefon kodları",
                "Türkiye'de para birimi",
                "Türkiye vize gereksinimleri",
                "İstanbul'da güvenlik",
                
                # General questions
                "İstanbul hakkında ne bilmeliyim?",
                "İstanbul seyahat ipuçları",
                "Önemli İstanbul bilgileri",
                "İstanbul ziyaretçi rehberi",
            ]
        }
    }

def create_training_samples(enhanced_data: Dict) -> List[Dict]:
    """Convert enhanced data to training sample format"""
    samples = []
    
    for intent, language_data in enhanced_data.items():
        # English samples
        for text in language_data["english"]:
            samples.append({
                "text": text,
                "intent": intent
            })
        
        # Turkish samples
        for text in language_data["turkish"]:
            samples.append({
                "text": text,
                "intent": intent
            })
    
    return samples

def main():
    """Generate and save enhanced training data"""
    print("🔧 Generating enhanced training data for weak intents...")
    
    # Generate data
    enhanced_data = generate_enhanced_data()
    training_samples = create_training_samples(enhanced_data)
    
    # Count samples per intent
    print("\n📊 Sample counts:")
    from collections import Counter
    intent_counts = Counter(s["intent"] for s in training_samples)
    for intent, count in sorted(intent_counts.items()):
        english_count = sum(1 for s in training_samples if s["intent"] == intent and all(ord(c) < 128 for c in s["text"]))
        turkish_count = count - english_count
        print(f"  {intent}: {count} total (EN: {english_count}, TR: {turkish_count})")
    
    # Save to file
    output_file = "enhanced_weak_intents_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "training_data": training_samples,
            "metadata": {
                "total_samples": len(training_samples),
                "intents": list(intent_counts.keys()),
                "purpose": "Enhanced training data for weak intents",
                "languages": ["english", "turkish"]
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(training_samples)} samples to {output_file}")
    print("\n📋 Next step: Merge with existing training data and retrain")

if __name__ == "__main__":
    main()

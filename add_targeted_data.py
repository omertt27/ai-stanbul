#!/usr/bin/env python3
"""
Phase 2: Targeted Data Addition
Add 150 specific examples to fix confused intents and push accuracy to 85%+
"""

import json

def add_targeted_training_data():
    """Add focused examples for problematic intent pairs"""
    
    # Load existing data
    with open("comprehensive_training_data.json", "r", encoding="utf-8") as f:
        existing_data = json.load(f)
    
    print(f"📂 Current training data: {len(existing_data)} samples")
    
    # New targeted samples to fix specific confusions
    targeted_samples = []
    
    # Problem 1: Restaurant vs Recommendation (40 samples)
    print("\n🎯 Adding samples for: Restaurant vs Recommendation")
    restaurant_samples = [
        # Explicit restaurant queries
        ("En iyi kebap nerede", "restaurant"),
        ("En iyi kebap yeri", "restaurant"),
        ("En güzel kebap nerede", "restaurant"),
        ("En lezzetli kebap nerede", "restaurant"),
        ("Kebap restoranı nerede", "restaurant"),
        ("Kebap yenecek yer", "restaurant"),
        ("Kebap yiyebileceğim restoran", "restaurant"),
        ("Balık restoranı nerede", "restaurant"),
        ("Balık yenecek yer", "restaurant"),
        ("Balık lokantası", "restaurant"),
        ("Balık mekanı", "restaurant"),
        ("Deniz mahsülleri restoranı", "restaurant"),
        ("En iyi balık restoranı", "restaurant"),
        ("Güzel balık restoranı", "restaurant"),
        ("Meze restoranı", "restaurant"),
        ("Türk mutfağı restoranı", "restaurant"),
        ("Döner yenecek yer", "restaurant"),
        ("Lahmacun yenecek yer", "restaurant"),
        ("Pide salonu", "restaurant"),
        ("Köfte restoranı", "restaurant"),
        ("Mangal restoranı", "restaurant"),
        ("Et restoranı", "restaurant"),
        ("Tavuk restoranı", "restaurant"),
        ("Izgara restoranı", "restaurant"),
        ("Balık ekmek nerede", "restaurant"),
        ("Kokoreç yeri", "restaurant"),
        ("Kumpir nerede", "restaurant"),
        ("Çiğ köfte nerede", "restaurant"),
        ("Mantı restoranı", "restaurant"),
        ("Tantuni nerede", "restaurant"),
        ("Yemek nerede yenir", "restaurant"),
        ("Yemek yenecek mekan", "restaurant"),
        ("Açık restoran", "restaurant"),
        ("24 saat açık restoran", "restaurant"),
        ("Gece açık restoran", "restaurant"),
        ("Sabah kahvaltısı nerede", "restaurant"),
        ("Kahvaltı mekanı", "restaurant"),
        ("Serpme kahvaltı", "restaurant"),
        ("Brunch nerede", "restaurant"),
        ("Açık büfe restoran", "restaurant"),
    ]
    targeted_samples.extend(restaurant_samples)
    
    # Clear recommendation samples (not about food)
    recommendation_samples = [
        ("Ne görmeliyim", "recommendation"),
        ("Önerileriniz neler", "recommendation"),
        ("Tavsiyeleriniz", "recommendation"),
        ("Nereleri gezmeliyim", "recommendation"),
        ("Ne yapmamı önerirsiniz", "recommendation"),
        ("En iyi yerler", "recommendation"),
        ("Top 10 yer", "recommendation"),
        ("Gezilecek yerler", "recommendation"),
        ("Görülecek yerler", "recommendation"),
        ("Ziyaret edilmesi gereken yerler", "recommendation"),
    ]
    targeted_samples.extend(recommendation_samples)
    
    # Problem 2: GPS Navigation vs others (30 samples)
    print("🎯 Adding samples for: GPS Navigation")
    gps_samples = [
        ("Taksim'e nasıl gidilir", "gps_navigation"),
        ("Taksim'e git", "gps_navigation"),
        ("Taksim'e yol tarifi", "gps_navigation"),
        ("Taksim yolu", "gps_navigation"),
        ("Taksim'e götür", "gps_navigation"),
        ("Sultanahmet'e nasıl gidilir", "gps_navigation"),
        ("Sultanahmet'e git", "gps_navigation"),
        ("Sultanahmet yolu", "gps_navigation"),
        ("Beşiktaş'a nasıl gidilir", "gps_navigation"),
        ("Beşiktaş'a git", "gps_navigation"),
        ("Kadıköy'e nasıl gidilir", "gps_navigation"),
        ("Kadıköy'e git", "gps_navigation"),
        ("Üsküdar'a nasıl gidilir", "gps_navigation"),
        ("Ortaköy'e nasıl gidilir", "gps_navigation"),
        ("Bebek'e nasıl gidilir", "gps_navigation"),
        ("Eminönü'ne nasıl gidilir", "gps_navigation"),
        ("Fatih'e nasıl gidilir", "gps_navigation"),
        ("Şişli'ye nasıl gidilir", "gps_navigation"),
        ("Mecidiyeköy'e nasıl gidilir", "gps_navigation"),
        ("Levent'e nasıl gidilir", "gps_navigation"),
        ("Etiler'e nasıl gidilir", "gps_navigation"),
        ("Nişantaşı'na nasıl gidilir", "gps_navigation"),
        ("Karaköy'e nasıl gidilir", "gps_navigation"),
        ("Galata'ya nasıl gidilir", "gps_navigation"),
        ("Beyoğlu'na nasıl gidilir", "gps_navigation"),
        ("İstiklal'e nasıl gidilir", "gps_navigation"),
        ("Beni buraya götür", "gps_navigation"),
        ("Şuraya git", "gps_navigation"),
        ("Yol göster", "gps_navigation"),
        ("Navigasyon", "gps_navigation"),
    ]
    targeted_samples.extend(gps_samples)
    
    # Problem 3: Cultural Info vs Weather/Events (25 samples)
    print("🎯 Adding samples for: Cultural Info")
    cultural_samples = [
        ("Boğaz turu", "cultural_info"),
        ("Boğaz gezisi", "cultural_info"),
        ("Boğaz turu bilgileri", "cultural_info"),
        ("Boğaz turu fiyatları", "cultural_info"),
        ("Boğaz turu nereden", "cultural_info"),
        ("Boğaz'da gezi", "cultural_info"),
        ("Cami ziyareti", "cultural_info"),
        ("Cami kuralları", "cultural_info"),
        ("Camiye nasıl girilir", "cultural_info"),
        ("Namaz vakitleri", "cultural_info"),
        ("Türk kültürü", "cultural_info"),
        ("Türk gelenekleri", "cultural_info"),
        ("Yerel gelenek", "cultural_info"),
        ("Türk görenekleri", "cultural_info"),
        ("Kültürel etkinlikler", "cultural_info"),
        ("Geleneksel tören", "cultural_info"),
        ("Ramazan", "cultural_info"),
        ("Bayram etkinlikleri", "cultural_info"),
        ("Mevlevi töreni", "cultural_info"),
        ("Semazen gösterisi", "cultural_info"),
        ("Türk hamamı", "cultural_info"),
        ("Hamam deneyimi", "cultural_info"),
        ("Çay kültürü", "cultural_info"),
        ("Kahve kültürü", "cultural_info"),
        ("Nazar boncuğu", "cultural_info"),
    ]
    targeted_samples.extend(cultural_samples)
    
    # Problem 4: Museum vs History/Attraction (25 samples)
    print("🎯 Adding samples for: Museum")
    museum_samples = [
        ("İstanbul Modern nerede", "museum"),
        ("İstanbul Modern saatleri", "museum"),
        ("İstanbul Modern bilgileri", "museum"),
        ("İstanbul Modern giriş", "museum"),
        ("Pera Müzesi", "museum"),
        ("Pera Müzesi nerede", "museum"),
        ("Arkeoloji Müzesi", "museum"),
        ("Arkeoloji Müzesi saatleri", "museum"),
        ("Sakıp Sabancı Müzesi", "museum"),
        ("Rahmi Koç Müzesi", "museum"),
        ("Sanat müzesi", "museum"),
        ("Modern sanat müzesi", "museum"),
        ("Çağdaş sanat", "museum"),
        ("Sergi nerede", "museum"),
        ("Hangi müzeyi görmeliyim", "museum"),
        ("Müze tavsiyeleri", "museum"),
        ("En iyi müzeler", "museum"),
        ("Ücretsiz müzeler", "museum"),
        ("Müze pazartesi açık mı", "museum"),
        ("Müze giriş ücreti", "museum"),
        ("Tarih müzesi", "museum"),
        ("Bilim müzesi", "museum"),
        ("Teknoloji müzesi", "museum"),
        ("Oyuncak müzesi", "museum"),
        ("Müze kaça kadar açık", "museum"),
    ]
    targeted_samples.extend(museum_samples)
    
    # Problem 5: Shopping vs Route Planning (20 samples)
    print("🎯 Adding samples for: Shopping")
    shopping_samples = [
        ("Kapalıçarşı nerede", "shopping"),
        ("Kapalıçarşı kaçta açılır", "shopping"),
        ("Kapalıçarşı kaçta kapanır", "shopping"),
        ("Kapalıçarşı alışveriş", "shopping"),
        ("Kapalıçarşı mağazaları", "shopping"),
        ("Grand Bazaar", "shopping"),
        ("Mısır Çarşısı", "shopping"),
        ("Mısır Çarşısı alışveriş", "shopping"),
        ("Baharat çarşısı", "shopping"),
        ("Alışveriş nerede yapılır", "shopping"),
        ("Mağaza önerileri", "shopping"),
        ("AVM nerede", "shopping"),
        ("Alışveriş merkezi yakın", "shopping"),
        ("Outlet", "shopping"),
        ("İndirimli mağaza", "shopping"),
        ("Marka mağazaları", "shopping"),
        ("İstiklal Caddesi alışveriş", "shopping"),
        ("Hediyelik eşya", "shopping"),
        ("Suvenir", "shopping"),
        ("Antika dükkanı", "shopping"),
    ]
    targeted_samples.extend(shopping_samples)
    
    # Problem 6: Events vs Weather (15 samples)
    print("🎯 Adding samples for: Events")
    events_samples = [
        ("Konser var mı", "events"),
        ("Konser bileti", "events"),
        ("Bu akşam konser", "events"),
        ("Müzik etkinliği", "events"),
        ("Canlı müzik", "events"),
        ("Festival var mı", "events"),
        ("Bu hafta festival", "events"),
        ("Etkinlik takvimi", "events"),
        ("Bugün etkinlik", "events"),
        ("Bu akşam ne var", "events"),
        ("Hafta sonu etkinlik", "events"),
        ("Tiyatro gösterisi", "events"),
        ("Sergi var mı", "events"),
        ("Açık hava konseri", "events"),
        ("Organizasyon", "events"),
    ]
    targeted_samples.extend(events_samples)
    
    print(f"\n📊 Added {len(targeted_samples)} targeted samples")
    
    # Combine with existing
    all_data = existing_data + targeted_samples
    
    # Remove duplicates
    all_data = list(set(all_data))
    
    print(f"✅ Total after deduplication: {len(all_data)} samples")
    print(f"📈 Growth: {len(existing_data)} → {len(all_data)} (+{len(all_data) - len(existing_data)})")
    
    # Save
    with open("comprehensive_training_data_v2.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved to comprehensive_training_data_v2.json")
    
    # Show intent distribution
    intent_counts = {}
    for _, intent in all_data:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\n📊 Top intents by sample count:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {intent}: {count}")
    
    return len(all_data)


if __name__ == "__main__":
    total = add_targeted_training_data()
    print(f"\n✅ Ready to train with {total} samples!")
    print("   Run: python3 phase2_final_push.py")

#!/usr/bin/env python3
"""
Turkish Neural Training Data Enhancement - Phase 2
Adds comprehensive Turkish examples for improved neural intent classification

Focus Areas:
1. Transportation (from 40 to 120+ samples)
2. Restaurant (from 60 to 120+ samples)
3. Hidden Gems (from 120 to 180+ samples)
4. Route Planning (from 25 to 100+ samples)
5. Neighborhood (from 100 to 150+ samples)
6. Weather (from 25 to 80+ samples)
7. Attraction (from 50 to 100+ samples)

New samples include:
- Verb conjugations and tenses
- Question patterns (how, where, when, which)
- Colloquial expressions
- Tourist scenarios
- Natural conversational queries
"""

import json
from pathlib import Path
from datetime import datetime

# Enhanced Turkish training data for key intents
ENHANCED_TURKISH_DATA = [
    # ============ TRANSPORTATION (80+ new samples) ============
    # Question patterns
    ["Taksim'e nasıl gidebilirim", "transportation"],
    ["Havalimanına ulaşım nasıl olur", "transportation"],
    ["En yakın metro istasyonu nerede", "transportation"],
    ["Tramvay hangi saatte kalkıyor", "transportation"],
    ["Marmaray nereden binilebilir", "transportation"],
    ["Otobüs kaç dakikada bir gelir", "transportation"],
    ["İstanbulkart nerede alabilirim", "transportation"],
    ["Taksi çağırmak istiyorum", "transportation"],
    ["Vapur saatleri nedir", "transportation"],
    ["Boğaz'ı geçmek için ne kullanmalıyım", "transportation"],
    
    # Verb conjugations
    ["Metro ile gitmek istiyorum", "transportation"],
    ["Tramvaya bineceğiz", "transportation"],
    ["Taksi tutabiliyor muyum", "transportation"],
    ["Otobüsle gidelim", "transportation"],
    ["Vapura binmeli miyim", "transportation"],
    ["Marmaray'a geçeyim", "transportation"],
    ["Metrobüse binmem gerekiyor", "transportation"],
    ["Taksim'den metro var mı", "transportation"],
    ["Karaköy'e tramvayla gidilir mi", "transportation"],
    ["Üsküdar'a vapur kalkar mı", "transportation"],
    
    # Colloquial/Natural
    ["Oraya nasıl giderim", "transportation"],
    ["Buradan metro bulabilir miyim", "transportation"],
    ["Tramvay buraları geçer mi", "transportation"],
    ["Taksi bulmak zor mu", "transportation"],
    ["Vapurla karşıya geçelim", "transportation"],
    ["Metro hattı var mı", "transportation"],
    ["Otobüs duraklıyor mu", "transportation"],
    ["Toplu taşıma kullanacağım", "transportation"],
    ["Kart yüklemem lazım", "transportation"],
    ["Tramvay bekliyoruz", "transportation"],
    
    # Specific routes & stations
    ["Kadıköy'e nasıl giderim", "transportation"],
    ["Beşiktaş'tan Taksim'e ulaşım", "transportation"],
    ["Fatih'e metro var mı", "transportation"],
    ["Eminönü'nden vapur", "transportation"],
    ["Mecidiyeköy metrobüs durağı", "transportation"],
    ["Kabataş'a füniküler", "transportation"],
    ["Üsküdar iskelesi nerede", "transportation"],
    ["Sultanahmet'e tramvay", "transportation"],
    ["Avrupa yakası ulaşım", "transportation"],
    ["Anadolu yakasına geçiş", "transportation"],
    
    # Time-related
    ["Sabah kaçta metro açılıyor", "transportation"],
    ["Gece otobüsü var mı", "transportation"],
    ["Son vapur saat kaçta", "transportation"],
    ["Tramvay kaça kadar çalışıyor", "transportation"],
    ["Hafta sonu metro saatleri", "transportation"],
    ["İlk sefer ne zaman", "transportation"],
    ["Akşam vapuru var mı", "transportation"],
    ["Gece geç saatte ulaşım", "transportation"],
    
    # Practical questions
    ["İstanbulkart doldurmak", "transportation"],
    ["Bilet fiyatları ne kadar", "transportation"],
    ["Aktarma yapmam gerekir mi", "transportation"],
    ["Kaç durak sonra inmeliyim", "transportation"],
    ["Doğru hatta mıyım", "transportation"],
    ["Hangi yönden binmeliyim", "transportation"],
    ["Transfer noktası nerede", "transportation"],
    ["Durak ismi nedir", "transportation"],
    
    # Alternative transportation
    ["Minibüs bulunur mu", "transportation"],
    ["Dolmuş hattı var mı", "transportation"],
    ["Taksi durağı yakınlarda mı", "transportation"],
    ["Uber çağırabilir miyim", "transportation"],
    ["Bisiklet kiralama", "transportation"],
    ["Elektrikli scooter var mı", "transportation"],
    ["Yürüyerek gidebilir miyim", "transportation"],
    ["Teknelerden biri gider mi", "transportation"],
    
    # Mixed/Complex
    ["Havalimanından otele ulaşım seçenekleri", "transportation"],
    ["En ucuz ulaşım yolu", "transportation"],
    ["En hızlı nasıl giderim", "transportation"],
    ["Trafiksiz ulaşım", "transportation"],
    ["Bagajla metro kullanılır mı", "transportation"],
    ["Çocuk arabasıyla otobüs", "transportation"],
    ["Engelli erişimi var mı", "transportation"],
    ["İstanbulkart bakiye sorgulama", "transportation"],
    
    # ============ RESTAURANT (60+ new samples) ============
    # Question patterns
    ["En iyi balık restoranı nerede", "restaurant"],
    ["Kebap nerede yenir", "restaurant"],
    ["Meyhane önerebilir misin", "restaurant"],
    ["Meze için mekan", "restaurant"],
    ["Boğaz manzaralı restoran", "restaurant"],
    ["Kahvaltı yapılacak yerler", "restaurant"],
    ["Vejetaryen restoran var mı", "restaurant"],
    ["Lezzetli döner nerede", "restaurant"],
    ["Pide salonu arıyorum", "restaurant"],
    ["Mantı nerede bulunur", "restaurant"],
    
    # Verb forms
    ["Yemek yemek istiyorum", "restaurant"],
    ["Akşam yemeği yiyeceğiz", "restaurant"],
    ["Kahvaltı edelim", "restaurant"],
    ["Öğle yemeği yiyelim", "restaurant"],
    ["Balık yemeğe gidelim", "restaurant"],
    ["Lokanta arayalım", "restaurant"],
    ["Meze seçelim", "restaurant"],
    ["Rezervasyon yaptırmam lazım", "restaurant"],
    
    # Cuisine types
    ["Osmanlı mutfağı", "restaurant"],
    ["Deniz ürünleri restoranı", "restaurant"],
    ["Türk mutfağı önerileri", "restaurant"],
    ["Etli yemekler", "restaurant"],
    ["Çorba içmek istiyorum", "restaurant"],
    ["Tatlı nerede yenir", "restaurant"],
    ["Baklava arıyorum", "restaurant"],
    ["Künefe yiyebileceğim yer", "restaurant"],
    ["Dondurma dükkanı", "restaurant"],
    ["Kahve içeceğim", "restaurant"],
    
    # Location-based
    ["Sultanahmet'te restoran", "restaurant"],
    ["Beyoğlu'nda meyhane", "restaurant"],
    ["Kadıköy'de kahvaltı", "restaurant"],
    ["Beşiktaş'ta balık lokantası", "restaurant"],
    ["Ortaköy'de kumpir", "restaurant"],
    ["Eminönü'nde balık ekmek", "restaurant"],
    ["Karaköy'de brunch", "restaurant"],
    ["Nişantaşı'nda fine dining", "restaurant"],
    ["Moda'da cafe", "restaurant"],
    
    # Price/Budget
    ["Ucuz lokanta", "restaurant"],
    ["Bütçe dostu restoran", "restaurant"],
    ["Lüks yemek yeri", "restaurant"],
    ["Öğrenci dostudur", "restaurant"],
    ["Ekonomik kahvaltı", "restaurant"],
    ["Fiyatları uygun mu", "restaurant"],
    
    # Specific dishes
    ["İskender kebap nerede", "restaurant"],
    ["Midye dolma satılan yer", "restaurant"],
    ["Simit nereden alınır", "restaurant"],
    ["Kokoreç mekanı", "restaurant"],
    ["Lahmacun pide", "restaurant"],
    ["Çiğ köfte dükkanı", "restaurant"],
    ["Börek satılan yer", "restaurant"],
    ["Gözleme yapan mekan", "restaurant"],
    
    # Time/Occasion
    ["Geç saate kadar açık restoran", "restaurant"],
    ["24 saat açık lokanta", "restaurant"],
    ["Sabah erken kahvaltı", "restaurant"],
    ["Gece yemeği için mekan", "restaurant"],
    ["Romantik akşam yemeği", "restaurant"],
    ["Aile restoranı", "restaurant"],
    
    # Features
    ["Canlı müzikli restoran", "restaurant"],
    ["Çocuk menüsü olan yer", "restaurant"],
    ["Açık havada yemek", "restaurant"],
    ["Teraslı restoran", "restaurant"],
    ["Manzaralı kahvaltı mekanı", "restaurant"],
    
    # ============ HIDDEN GEMS (60+ new samples) ============
    ["Turistik olmayan yerler", "hidden_gems"],
    ["Gizli güzellikler", "hidden_gems"],
    ["Yerel halkın gittiği mekanlar", "hidden_gems"],
    ["Keşfedilmemiş semtler", "hidden_gems"],
    ["Az bilinen yerler", "hidden_gems"],
    ["Saklı kalmış camiler", "hidden_gems"],
    ["Gizli bahçeler", "hidden_gems"],
    ["Yeraltı sarnıçları", "hidden_gems"],
    ["Eski İstanbul sokakları", "hidden_gems"],
    ["Mahalle kahveleri", "hidden_gems"],
    ["Köhne mekanlar", "hidden_gems"],
    ["Klasik pastaneler", "hidden_gems"],
    ["Unutulmuş tarihi yapılar", "hidden_gems"],
    ["Antik kalıntılar", "hidden_gems"],
    ["Bilinmeyen müzeler", "hidden_gems"],
    ["Gizli teraslar", "hidden_gems"],
    ["Yerel pazarlar", "hidden_gems"],
    ["Mahalle fırınları", "hidden_gems"],
    ["Eski hanlar", "hidden_gems"],
    ["Tarihi çeşmeler", "hidden_gems"],
    ["Otantik sokaklar", "hidden_gems"],
    ["Sessiz mahalleler", "hidden_gems"],
    ["Kalabalık olmayan mekanlar", "hidden_gems"],
    ["Yerli tavsiyesi", "hidden_gems"],
    ["Turistlerin gitmediği yerler", "hidden_gems"],
    ["Özel mekanlar", "hidden_gems"],
    ["Gizli köşeler", "hidden_gems"],
    ["Az keşfedilmiş bölgeler", "hidden_gems"],
    ["Yerel lezzetler nerede", "hidden_gems"],
    ["Gizli cafe'ler", "hidden_gems"],
    ["Mahalle içi mekanlar", "hidden_gems"],
    ["İçeriden tavsiyeler", "hidden_gems"],
    ["Sıradışı yerler", "hidden_gems"],
    ["Farklı mekanlar", "hidden_gems"],
    ["Alışılmadık yerler", "hidden_gems"],
    ["Uğrak mekanlar", "hidden_gems"],
    ["Mahallenin incileri", "hidden_gems"],
    ["Yerel çarşılar", "hidden_gems"],
    ["Esnaf lokantaları", "hidden_gems"],
    ["Muhit kahveleri", "hidden_gems"],
    ["Sahilden yerler", "hidden_gems"],
    ["Kıyı kenarı mekanlar", "hidden_gems"],
    ["Arka sokaklar", "hidden_gems"],
    ["Girintiler", "hidden_gems"],
    ["Küçük meydanlar", "hidden_gems"],
    ["Tarihi çarşılar", "hidden_gems"],
    ["Eski dükkânlar", "hidden_gems"],
    ["Antika mekanlar", "hidden_gems"],
    ["Nostaljik yerler", "hidden_gems"],
    ["Geleneksel atölyeler", "hidden_gems"],
    ["El sanatları dükkanları", "hidden_gems"],
    ["Özgün mekanlar", "hidden_gems"],
    ["Benzersiz deneyimler", "hidden_gems"],
    ["Gizemli yerler", "hidden_gems"],
    ["Sır mekanlar", "hidden_gems"],
    ["Kendi halkımızın gittiği yerler", "hidden_gems"],
    ["İstanbullunun uğrak yerleri", "hidden_gems"],
    ["Mahalle abilerinin mekanı", "hidden_gems"],
    ["Cennet köşeler", "hidden_gems"],
    ["Şahane manzaralar", "hidden_gems"],
    
    # ============ ROUTE PLANNING (75+ new samples) ============
    ["Günlük rota planla", "route_planning"],
    ["İki günlük tur programı", "route_planning"],
    ["Üç gün İstanbul gezisi", "route_planning"],
    ["Hafta sonu rotası", "route_planning"],
    ["Bir günde ne gezilir", "route_planning"],
    ["Sabah gezilecek yerler", "route_planning"],
    ["Öğleden sonra programı", "route_planning"],
    ["Akşam için plan", "route_planning"],
    ["Tam gün tur", "route_planning"],
    ["Yarım günlük gezi", "route_planning"],
    ["Hızlı tur planı", "route_planning"],
    ["Detaylı gezi rotası", "route_planning"],
    ["Optimum rota", "route_planning"],
    ["En iyi sıralama", "route_planning"],
    ["Verimli plan", "route_planning"],
    ["Yakın yerler birlikte", "route_planning"],
    ["Bölge bazında gezinti", "route_planning"],
    ["Tema bazlı rota", "route_planning"],
    ["Tarihi mekanlar rotası", "route_planning"],
    ["Müze turu planı", "route_planning"],
    ["Yeme-içme rotası", "route_planning"],
    ["Alışveriş günü planı", "route_planning"],
    ["Fotoğraf turu rotası", "route_planning"],
    ["Romantik gezi planı", "route_planning"],
    ["Aile gezisi rotası", "route_planning"],
    ["Çocuklu gezi programı", "route_planning"],
    ["Genç gezgin rotası", "route_planning"],
    ["Yaşlı dostu plan", "route_planning"],
    ["Engelsiz rota", "route_planning"],
    ["Yürüyerek gezi rotası", "route_planning"],
    ["Toplu taşıma ile plan", "route_planning"],
    ["Arabayla tur programı", "route_planning"],
    ["Boğaz turu rotası", "route_planning"],
    ["Avrupa yakası planı", "route_planning"],
    ["Anadolu yakası turu", "route_planning"],
    ["İki yakayı birleştiren rota", "route_planning"],
    ["Sultanahmet bölgesi gezisi", "route_planning"],
    ["Beyoğlu rotası", "route_planning"],
    ["Kadıköy gezintisi", "route_planning"],
    ["Üsküdar planı", "route_planning"],
    ["Boğaziçi turu", "route_planning"],
    ["Adalar gezisi", "route_planning"],
    ["Kıyı rotası", "route_planning"],
    ["Tepe manzaraları turu", "route_planning"],
    ["Gün batımı rotası", "route_planning"],
    ["Gece gezisi planı", "route_planning"],
    ["Gündüz programı", "route_planning"],
    ["Dolu dolu gün", "route_planning"],
    ["Rahat tempolu tur", "route_planning"],
    ["Yoğun program", "route_planning"],
    ["Sakin gezi", "route_planning"],
    ["İlk gün önerisi", "route_planning"],
    ["Son gün rotası", "route_planning"],
    ["Ara gün planı", "route_planning"],
    ["Bütçe dostu rota", "route_planning"],
    ["Ücretsiz mekanlar turu", "route_planning"],
    ["Premium deneyim planı", "route_planning"],
    ["Kültür turu rotası", "route_planning"],
    ["Doğa gezisi planı", "route_planning"],
    ["Mimari keşif rotası", "route_planning"],
    ["Gastronomi turu", "route_planning"],
    ["Alışveriş merkezi rotası", "route_planning"],
    ["Çarşı-pazar gezisi", "route_planning"],
    ["Antika mekanlar turu", "route_planning"],
    ["Modern İstanbul rotası", "route_planning"],
    ["Eski İstanbul gezisi", "route_planning"],
    ["Bizans İstanbul'u", "route_planning"],
    ["Osmanlı izleri turu", "route_planning"],
    ["Dini mekanlar rotası", "route_planning"],
    ["Saray gezisi planı", "route_planning"],
    ["Kale ve surlar turu", "route_planning"],
    ["Park ve bahçeler gezisi", "route_planning"],
    ["Deniz kenarı rotası", "route_planning"],
    
    # ============ NEIGHBORHOOD (50+ new samples) ============
    ["Sultanahmet nasıl bir semt", "neighborhoods"],
    ["Beyoğlu hakkında bilgi", "neighborhoods"],
    ["Kadıköy'ü anlat", "neighborhoods"],
    ["Beşiktaş semti", "neighborhoods"],
    ["Üsküdar'da neler var", "neighborhoods"],
    ["Fatih mahallesi", "neighborhoods"],
    ["Ortaköy özellikleri", "neighborhoods"],
    ["Balat semti nasıl", "neighborhoods"],
    ["Fener mahallesi", "neighborhoods"],
    ["Karaköy'de gezilecek yerler", "neighborhoods"],
    ["Galata bölgesi", "neighborhoods"],
    ["Taksim çevresi", "neighborhoods"],
    ["Cihangir mahallesi", "neighborhoods"],
    ["Moda semti", "neighborhoods"],
    ["Nişantaşı bölgesi", "neighborhoods"],
    ["Şişli merkez", "neighborhoods"],
    ["Mecidiyeköy çevresi", "neighborhoods"],
    ["Etiler semti", "neighborhoods"],
    ["Bebek mahallesi", "neighborhoods"],
    ["Arnavutköy bölgesi", "neighborhoods"],
    ["Rumelihisarı semti", "neighborhoods"],
    ["Emirgan mahallesi", "neighborhoods"],
    ["İstinye bölgesi", "neighborhoods"],
    ["Sarıyer semti", "neighborhoods"],
    ["Tarabya mahallesi", "neighborhoods"],
    ["Yeşilköy bölgesi", "neighborhoods"],
    ["Bakırköy semti", "neighborhoods"],
    ["Ataköy mahallesi", "neighborhoods"],
    ["Florya bölgesi", "neighborhoods"],
    ["Kuzguncuk semti", "neighborhoods"],
    ["Çengelköy mahallesi", "neighborhoods"],
    ["Beylerbeyi bölgesi", "neighborhoods"],
    ["Çamlıca semti", "neighborhoods"],
    ["Kısıklı mahallesi", "neighborhoods"],
    ["Bağlarbaşı bölgesi", "neighborhoods"],
    ["Validebağ semti", "neighborhoods"],
    ["Acıbadem mahallesi", "neighborhoods"],
    ["Göztepe bölgesi", "neighborhoods"],
    ["Fenerbahçe semti", "neighborhoods"],
    ["Suadiye mahallesi", "neighborhoods"],
    ["Bostancı bölgesi", "neighborhoods"],
    ["Maltepe semti", "neighborhoods"],
    ["Kartal mahallesi", "neighborhoods"],
    ["Pendik bölgesi", "neighborhoods"],
    ["Şile semti", "neighborhoods"],
    ["Ağva mahallesi", "neighborhoods"],
    ["Polonezköy bölgesi", "neighborhoods"],
    ["Çekmeköy semti", "neighborhoods"],
    ["Ümraniye mahallesi", "neighborhoods"],
    ["Maslak bölgesi", "neighborhoods"],
    
    # ============ WEATHER (55+ new samples) ============
    ["Hava durumu nasıl", "weather"],
    ["Bugün hava nasıl", "weather"],
    ["Yarın hava nasıl olacak", "weather"],
    ["Hafta sonu hava durumu", "weather"],
    ["Bu akşam hava", "weather"],
    ["Sabah hava nasıl", "weather"],
    ["Öğleden sonra hava durumu", "weather"],
    ["Yağmur yağacak mı", "weather"],
    ["Güneşli mi olacak", "weather"],
    ["Bulutlu mu", "weather"],
    ["Sıcaklık kaç derece", "weather"],
    ["Ne kadar sıcak", "weather"],
    ["Soğuk mu", "weather"],
    ["Rüzgar var mı", "weather"],
    ["Fırtına olacak mı", "weather"],
    ["Kar yağar mı", "weather"],
    ["Dolu yağacak mı", "weather"],
    ["Sisli mi", "weather"],
    ["Nemli mi", "weather"],
    ["Kuru hava", "weather"],
    ["İklim nasıl", "weather"],
    ["Mevsim özellikleri", "weather"],
    ["Şu an hava", "weather"],
    ["Canlı hava durumu", "weather"],
    ["Güncel sıcaklık", "weather"],
    ["Hissedilen sıcaklık", "weather"],
    ["Minimum sıcaklık", "weather"],
    ["Maksimum derece", "weather"],
    ["Gece hava nasıl", "weather"],
    ["Gündüz sıcaklık", "weather"],
    ["Haftalık tahmin", "weather"],
    ["5 günlük hava durumu", "weather"],
    ["On günlük tahmin", "weather"],
    ["Bu ay hava nasıl", "weather"],
    ["Sezon hava durumu", "weather"],
    ["Yaz ayları sıcaklık", "weather"],
    ["Kış aylarında hava", "weather"],
    ["İlkbahar iklimi", "weather"],
    ["Sonbahar havası", "weather"],
    ["Şemsiye almalı mıyım", "weather"],
    ["Mont gerekir mi", "weather"],
    ["Hafif giyinebilir miyim", "weather"],
    ["Kalın giyinmeliyim", "weather"],
    ["Güneş kremi gerekli mi", "weather"],
    ["Güneş gözlüğü", "weather"],
    ["Yağmurluk lazım mı", "weather"],
    ["Dışarı çıkılır mı", "weather"],
    ["Piknik havası", "weather"],
    ["Denize girilir mi", "weather"],
    ["Boğaz turu için uygun mu", "weather"],
    ["Açık hava etkinliği yapılır mı", "weather"],
    ["Gezi için hava uygun mu", "weather"],
    ["Fotoğraf çekimi havası", "weather"],
    ["Gün batımı görülür mü", "weather"],
    ["Görüş mesafesi", "weather"],
    
    # ============ ATTRACTION (50+ new samples) ============
    ["Ayasofya'yı gezmek istiyorum", "attraction"],
    ["Topkapı Sarayı giriş saatleri", "attraction"],
    ["Kapalıçarşı açık mı", "attraction"],
    ["Galata Kulesi'ne çıkmak", "attraction"],
    ["Yerebatan Sarnıcı bilgi", "attraction"],
    ["Süleymaniye Camii ziyaret", "attraction"],
    ["Sultanahmet Camii giriş", "attraction"],
    ["Dolmabahçe Sarayı tur", "attraction"],
    ["Çırağan Sarayı görmek", "attraction"],
    ["Beylerbeyi Sarayı ziyaret", "attraction"],
    ["Rumeli Hisarı gezisi", "attraction"],
    ["Anadolu Hisarı bakış", "attraction"],
    ["Kız Kulesi'ne nasıl gidilir", "attraction"],
    ["Boğaz turu nereden", "attraction"],
    ["Adalar'a vapur", "attraction"],
    ["Büyükada gezisi", "attraction"],
    ["Heybeliada tur", "attraction"],
    ["Burgazada ziyaret", "attraction"],
    ["Kınalıada'ya gitmek", "attraction"],
    ["Miniatürk park", "attraction"],
    ["İstanbul Aquarium ziyaret", "attraction"],
    ["Turkcell Platinum gibi müze", "attraction"],
    ["Rahmi M. Koç Müzesi tur", "attraction"],
    ["İstanbul Modern görme", "attraction"],
    ["Pera Müzesi ziyaret", "attraction"],
    ["Sakıp Sabancı Müzesi", "attraction"],
    ["Arkeoloji Müzesi giriş", "attraction"],
    ["Türk İslam Eserleri Müzesi", "attraction"],
    ["Kariye Müzesi görmek", "attraction"],
    ["Fener Rum Patrikhanesi", "attraction"],
    ["Patrikhane ziyareti", "attraction"],
    ["Pierre Loti tepesi", "attraction"],
    ["Çamlıca tepesine çıkmak", "attraction"],
    ["Emirgan Korusu gezisi", "attraction"],
    ["Belgrad Ormanı piknik", "attraction"],
    ["Gülhane Parkı gezintisi", "attraction"],
    ["Yıldız Parkı ziyareti", "attraction"],
    ["Fethi Paşa Korusu", "attraction"],
    ["Maçka Parkı", "attraction"],
    ["Aşiyan Müzesi", "attraction"],
    ["Sabiha Gökçen Müzesi", "attraction"],
    ["Vialand tema parkı", "attraction"],
    ["Moipark ziyaret", "attraction"],
    ["Türk Telekom Stadyumu tur", "attraction"],
    ["Vodafone Arena gezme", "attraction"],
    ["Göztepe parkı", "attraction"],
    ["Fenerbahçe parkı", "attraction"],
    ["Validebağ korusu", "attraction"],
    ["Beykoz korusu", "attraction"],
    ["Polonezköy doğa parkı", "attraction"],
]

def load_existing_data(filepath):
    """Load existing training data"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def analyze_additions(existing_data, new_data):
    """Analyze what will be added"""
    from collections import Counter
    
    existing_intents = Counter([item[1] for item in existing_data])
    new_intents = Counter([item[1] for item in new_data])
    
    print("\n=== ADDITION ANALYSIS ===")
    print(f"Existing samples: {len(existing_data)}")
    print(f"New samples to add: {len(new_data)}")
    print(f"Total after merge: {len(existing_data) + len(new_data)}")
    
    print("\n=== New Samples by Intent ===")
    for intent, count in sorted(new_intents.items(), key=lambda x: x[1], reverse=True):
        old_count = existing_intents.get(intent, 0)
        new_count = old_count + count
        increase_pct = (count / old_count * 100) if old_count > 0 else float('inf')
        print(f"{intent:25s}: {old_count:4d} → {new_count:4d} (+{count:3d}, +{increase_pct:.0f}%)")

def merge_and_save(existing_data, new_data, output_path):
    """Merge and save training data"""
    # Check for duplicates
    existing_queries = {item[0].lower().strip() for item in existing_data}
    unique_new_data = []
    duplicates = 0
    
    for query, intent in new_data:
        if query.lower().strip() not in existing_queries:
            unique_new_data.append([query, intent])
            existing_queries.add(query.lower().strip())
        else:
            duplicates += 1
    
    if duplicates > 0:
        print(f"\n⚠️  Skipped {duplicates} duplicate queries")
    
    # Merge
    merged_data = existing_data + unique_new_data
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved {len(merged_data)} total samples to {output_path}")
    return merged_data

def create_backup(filepath):
    """Create backup of original file"""
    from shutil import copy2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.replace('.json', f'_backup_{timestamp}.json')
    copy2(filepath, backup_path)
    print(f"📦 Created backup: {backup_path}")
    return backup_path

def main():
    """Main execution"""
    input_file = "comprehensive_training_data.json"
    output_file = "comprehensive_training_data.json"
    
    print("=" * 60)
    print("Turkish Neural Training Data Enhancement - Phase 2")
    print("=" * 60)
    
    # Load existing data
    print(f"\n📂 Loading existing training data from {input_file}...")
    existing_data = load_existing_data(input_file)
    
    if not existing_data:
        print("❌ Could not load existing data. Aborting.")
        return
    
    # Analyze additions
    analyze_additions(existing_data, ENHANCED_TURKISH_DATA)
    
    # Create backup
    print("\n📦 Creating backup...")
    create_backup(input_file)
    
    # Merge and save
    print("\n💾 Merging and saving enhanced training data...")
    merged_data = merge_and_save(existing_data, ENHANCED_TURKISH_DATA, output_file)
    
    # Final analysis
    from collections import Counter
    final_intents = Counter([item[1] for item in merged_data])
    
    print("\n=== FINAL DISTRIBUTION ===")
    total = len(merged_data)
    for intent, count in sorted(final_intents.items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100
        print(f"{intent:25s}: {count:4d} samples ({pct:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ Turkish Training Data Enhancement Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print("1. Review the enhanced dataset")
    print("2. Retrain the neural classifier")
    print("3. Validate performance improvements")
    print("4. Update documentation")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 2 FINAL: Comprehensive Balanced Training Dataset
500+ samples with 20+ samples per intent for >90% accuracy
"""

def create_final_comprehensive_dataset():
    """Create balanced dataset with all 25 intents well-represented"""
    
    training_data = []
    
    # Each intent should have 20-30 samples minimum
    
    # ===== ATTRACTION (50 samples) =====
    attractions = ["Sultanahmet", "Ayasofya", "Topkapı Sarayı", "Galata Kulesi", 
                   "Dolmabahçe", "Yerebatan Sarnıcı", "Kız Kulesi", "Süleymaniye"]
    
    for place in attractions:
        training_data.extend([
            (f"{place} nerede", "attraction"),
            (f"{place} görmek istiyorum", "attraction"),
            (f"{place} hakkında bilgi", "attraction"),
            (f"{place} kaça kadar açık", "attraction"),
            (f"{place}'yi ziyaret etmek istiyorum", "attraction"),
            (f"{place} giriş saatleri", "attraction"),
        ])
    
    training_data.extend([
        ("Görülecek yerler", "attraction"),
        ("En güzel yerler", "attraction"),
    ])
    
    # ===== MUSEUM (30 samples) =====
    museums = ["İstanbul Modern", "Pera Müzesi", "Arkeoloji Müzesi", "Sakıp Sabancı", "Rahmi Koç"]
    
    for museum in museums:
        training_data.extend([
            (f"{museum} nerede", "museum"),
            (f"{museum} kaça kadar açık", "museum"),
            (f"{museum} bilgileri", "museum"),
        ])
    
    training_data.extend([
        ("Müze önerisi", "museum"),
        ("Hangi müzeleri görmeliyim", "museum"),
        ("Sanat müzesi", "museum"),
        ("Tarih müzesi", "museum"),
        ("Müze gezisi", "museum"),
        ("Çağdaş sanat müzesi", "museum"),
        ("Bilim müzesi", "museum"),
        ("Müze tavsiyeleri", "museum"),
        ("En iyi müzeler", "museum"),
        ("Modern sanat", "museum"),
        ("Arkeoloji", "museum"),
    ])
    
    # ===== RESTAURANT (60 samples) =====
    foods = ["kebap", "balık", "meze", "künefe", "lahmacun", "pide", "köfte", "baklava", "döner", "iskender"]
    
    for food in foods:
        training_data.extend([
            (f"En iyi {food} nerede", "restaurant"),
            (f"{food} yemek istiyorum", "restaurant"),
            (f"{food} restoranı öner", "restaurant"),
            (f"{food} için mekan", "restaurant"),
            (f"Kaliteli {food} nerede", "restaurant"),
        ])
    
    training_data.extend([
        ("Nerede yemek yiyebilirim", "restaurant"),
        ("Yemek için restoran", "restaurant"),
        ("Akşam yemeği mekan", "restaurant"),
        ("Öğle yemeği nerede", "restaurant"),
        ("Deniz mahsülleri restoranı", "restaurant"),
        ("Vejeteryan restoran", "restaurant"),
        ("Vegan yemek", "restaurant"),
        ("Türk mutfağı", "restaurant"),
        ("Yerel yemek", "restaurant"),
        ("Geleneksel restoran", "restaurant"),
    ])
    
    # ===== TRANSPORTATION (40 samples) =====
    training_data.extend([
        ("Havalimanından şehre nasıl gidilir", "transportation"),
        ("Metro saatleri", "transportation"),
        ("Otobüs saatleri", "transportation"),
        ("Tramvay güzergahı", "transportation"),
        ("Marmaray bilgileri", "transportation"),
        ("Taksi çağır", "transportation"),
        ("Vapur saatleri", "transportation"),
        ("Toplu taşıma kartı", "transportation"),
        ("İstanbulkart nereden alınır", "transportation"),
        ("Havalimanı otobüsü", "transportation"),
        ("Metro hattı", "transportation"),
        ("Otobüs güzergahı", "transportation"),
        ("Vapur hareket saatleri", "transportation"),
        ("Tünel bilgileri", "transportation"),
        ("Füniküler", "transportation"),
        ("Teleferik", "transportation"),
        ("Deniz otobüsü", "transportation"),
        ("IDO seferleri", "transportation"),
        ("Metrobüs güzergahı", "transportation"),
        ("Akbil", "transportation"),
        ("Toplu taşıma ücreti", "transportation"),
        ("En yakın metro", "transportation"),
        ("En yakın durak", "transportation"),
        ("Otobüs numarası", "transportation"),
        ("Ulaşım bilgileri", "transportation"),
        ("Taksi ücreti", "transportation"),
        ("Uber", "transportation"),
        ("Bisiklet kiralama", "transportation"),
        ("Scooter kiralama", "transportation"),
        ("Araç kiralama", "transportation"),
        ("Otopark", "transportation"),
        ("Park yeri", "transportation"),
        ("Havalimanı transfer", "transportation"),
        ("Servis aracı", "transportation"),
        ("Minibüs", "transportation"),
        ("Dolmuş", "transportation"),
        ("Toplu taşıma haritası", "transportation"),
        ("Ulaşım ağı", "transportation"),
        ("Kaç durak", "transportation"),
        ("Aktarma noktası", "transportation"),
    ])
    
    # ===== GPS_NAVIGATION (40 samples) =====
    places = ["Taksim", "Sultanahmet", "Beşiktaş", "Kadıköy", "Üsküdar", "Eminönü", "Ortaköy", "Bebek"]
    
    for place in places:
        training_data.extend([
            (f"{place}'e nasıl gidilir", "gps_navigation"),
            (f"{place}'e git", "gps_navigation"),
            (f"{place}'ye yol tarifi", "gps_navigation"),
            (f"Beni {place}'e götür", "gps_navigation"),
            (f"{place} yolu", "gps_navigation"),
        ])
    
    # ===== ACCOMMODATION (30 samples) =====
    training_data.extend([
        ("Ucuz otel önerisi", "accommodation"),
        ("Ekonomik konaklama", "accommodation"),
        ("Butik otel", "accommodation"),
        ("Taksim'de otel", "accommodation"),
        ("Boğaz manzaralı otel", "accommodation"),
        ("Sultanahmet'da otel", "accommodation"),
        ("Beşiktaş'ta konaklama", "accommodation"),
        ("Kadıköy'de otel", "accommodation"),
        ("Otel önerisi", "accommodation"),
        ("Nerede kalmalıyım", "accommodation"),
        ("Kalacak yer", "accommodation"),
        ("Konaklama önerileri", "accommodation"),
        ("Otel fiyatları", "accommodation"),
        ("4 yıldızlı otel", "accommodation"),
        ("Apart otel", "accommodation"),
        ("Aile oteli", "accommodation"),
        ("İş oteli", "accommodation"),
        ("Spa oteli", "accommodation"),
        ("Tarihi otel", "accommodation"),
        ("Modern otel", "accommodation"),
        ("Deniz manzaralı otel", "accommodation"),
        ("Şehir merkezi otel", "accommodation"),
        ("Havalimanı yakını otel", "accommodation"),
        ("Otel tavsiyeleri", "accommodation"),
        ("En iyi oteller", "accommodation"),
        ("Otel araştırması", "accommodation"),
        ("Pansyon", "accommodation"),
        ("Guest house", "accommodation"),
        ("Airbnb", "accommodation"),
        ("Kiralık daire", "accommodation"),
    ])
    
    # ===== SHOPPING (30 samples) =====
    training_data.extend([
        ("Kapalıçarşı kaça kadar açık", "shopping"),
        ("Alışveriş merkezi", "shopping"),
        ("Hediyelik eşya nereden alınır", "shopping"),
        ("Moda mağazaları", "shopping"),
        ("İstiklal Caddesi mağazaları", "shopping"),
        ("Grand Bazaar", "shopping"),
        ("Outlet mağaza", "shopping"),
        ("Alışveriş için öneriler", "shopping"),
        ("Yerel ürünler nerede", "shopping"),
        ("Antika mağazaları", "shopping"),
        ("Mısır Çarşısı", "shopping"),
        ("Baharat çarşısı", "shopping"),
        ("Çarşı gezisi", "shopping"),
        ("El sanatları", "shopping"),
        ("Halı mağazası", "shopping"),
        ("Tekstil", "shopping"),
        ("Deri ürünleri", "shopping"),
        ("Takı mağazası", "shopping"),
        ("Gümüş", "shopping"),
        ("Altın", "shopping"),
        ("Pazar", "shopping"),
        ("Bit pazarı", "shopping"),
        ("Antik eşya", "shopping"),
        ("Kitap mağazası", "shopping"),
        ("Müzik dükkanı", "shopping"),
        ("Elektronik", "shopping"),
        ("Mall", "shopping"),
        ("AVM", "shopping"),
        ("Alışveriş caddesi", "shopping"),
        ("Butik", "shopping"),
    ])
    
    # ===== NIGHTLIFE (30 samples) =====
    training_data.extend([
        ("Gece hayatı önerileri", "nightlife"),
        ("Bar tavsiyesi", "nightlife"),
        ("Canlı müzik mekanı", "nightlife"),
        ("Kulüp önerisi", "nightlife"),
        ("Rooftop bar", "nightlife"),
        ("Gece eğlence mekanı", "nightlife"),
        ("Müzikli mekan", "nightlife"),
        ("Dans edebileceğim yer", "nightlife"),
        ("Bira barı", "nightlife"),
        ("Kokteyl barı", "nightlife"),
        ("Lounge bar", "nightlife"),
        ("Jazz bar", "nightlife"),
        ("Rock bar", "nightlife"),
        ("DJ mekanı", "nightlife"),
        ("Gece kulübü", "nightlife"),
        ("Disco", "nightlife"),
        ("Pub", "nightlife"),
        ("Irish pub", "nightlife"),
        ("Sports bar", "nightlife"),
        ("Karaoke", "nightlife"),
        ("Nargilecafe", "nightlife"),
        ("Gece manzarası", "nightlife"),
        ("Gece gezisi", "nightlife"),
        ("Akşam eğlencesi", "nightlife"),
        ("Party mekanı", "nightlife"),
        ("Eğlence yerleri", "nightlife"),
        ("Gece hayatı rehberi", "nightlife"),
        ("Ortaköy gece", "nightlife"),
        ("Beyoğlu gece", "nightlife"),
        ("Kadıköy gece hayatı", "nightlife"),
    ])
    
    # ===== EVENTS (25 samples) =====
    training_data.extend([
        ("Bu hafta sonu etkinlikler", "events"),
        ("Konser takvimi", "events"),
        ("Festival", "events"),
        ("Müzik etkinlikleri", "events"),
        ("Sergi", "events"),
        ("Bugün ne var", "events"),
        ("Bu akşam etkinlik", "events"),
        ("Tiyatro gösterileri", "events"),
        ("Açık hava konseri", "events"),
        ("Caz konseri", "events"),
        ("Rock konseri", "events"),
        ("Klasik müzik", "events"),
        ("Opera", "events"),
        ("Bale", "events"),
        ("Stand-up", "events"),
        ("Komedi gösterisi", "events"),
        ("Sanat sergisi", "events"),
        ("Fuar", "events"),
        ("Kültür etkinlikleri", "events"),
        ("Spor etkinlikleri", "events"),
        ("Maç bileti", "events"),
        ("Sinema", "events"),
        ("Film gösterimi", "events"),
        ("Etkinlik takvimi", "events"),
        ("Bu hafta programı", "events"),
    ])
    
    # ===== WEATHER (25 samples) =====
    training_data.extend([
        ("Hava durumu nasıl", "weather"),
        ("Yarın hava nasıl olacak", "weather"),
        ("Yağmur yağacak mı", "weather"),
        ("Sıcaklık kaç derece", "weather"),
        ("Hava sıcak mı", "weather"),
        ("Bu hafta hava", "weather"),
        ("Hava tahmini", "weather"),
        ("Kar yağacak mı", "weather"),
        ("Güneşli mi", "weather"),
        ("Hava soğuk mu", "weather"),
        ("Rüzgar var mı", "weather"),
        ("Nem oranı", "weather"),
        ("Hava raporu", "weather"),
        ("Meteoroloji", "weather"),
        ("Bugün hava", "weather"),
        ("Yağış", "weather"),
        ("Fırtına", "weather"),
        ("Hava durumu bilgisi", "weather"),
        ("Sis var mı", "weather"),
        ("Bulutlu mu", "weather"),
        ("Açık hava", "weather"),
        ("Hava koşulları", "weather"),
        ("Haftalık tahmin", "weather"),
        ("5 günlük", "weather"),
        ("Mevsim tahmini", "weather"),
    ])
    
    # ===== EMERGENCY (20 samples) =====
    training_data.extend([
        ("En yakın hastane", "emergency"),
        ("Polis çağır", "emergency"),
        ("Acil yardım", "emergency"),
        ("Eczane nerede", "emergency"),
        ("112", "emergency"),
        ("Ambulans", "emergency"),
        ("İtfaiye", "emergency"),
        ("Kayboldum", "emergency"),
        ("Yardım edin", "emergency"),
        ("Acil durum", "emergency"),
        ("Doktor", "emergency"),
        ("Sağlık ocağı", "emergency"),
        ("Nöbetçi eczane", "emergency"),
        ("Kan kaybı", "emergency"),
        ("Kaza", "emergency"),
        ("Yangın", "emergency"),
        ("Hırsızlık", "emergency"),
        ("Kayıp eşya", "emergency"),
        ("Konsolosluk", "emergency"),
        ("Elçilik", "emergency"),
    ])
    
    # ===== GENERAL_INFO (20 samples) =====
    training_data.extend([
        ("İstanbul hakkında bilgi", "general_info"),
        ("Genel bilgiler", "general_info"),
        ("İstanbul'da ne yapabilirim", "general_info"),
        ("Turistik bilgiler", "general_info"),
        ("Gezi rehberi", "general_info"),
        ("Şehir bilgileri", "general_info"),
        ("İstanbul'u tanıt", "general_info"),
        ("İstanbul tarihi", "general_info"),
        ("Nüfus", "general_info"),
        ("Coğrafya", "general_info"),
        ("İlçeler", "general_info"),
        ("Mahalleler", "general_info"),
        ("Genel öneriler", "general_info"),
        ("İpuçları", "general_info"),
        ("Bilmem gerekenler", "general_info"),
        ("Temel bilgiler", "general_info"),
        ("Şehir rehberi", "general_info"),
        ("Turist rehberi", "general_info"),
        ("Seyahat bilgileri", "general_info"),
        ("İstanbul hakkında", "general_info"),
    ])
    
    # ===== RECOMMENDATION (20 samples) =====
    training_data.extend([
        ("Ne görmeliyim", "recommendation"),
        ("Önerileriniz neler", "recommendation"),
        ("Tavsiye", "recommendation"),
        ("Ne yapmamı önerirsiniz", "recommendation"),
        ("Mutlaka görülmesi gerekenler", "recommendation"),
        ("En iyiler", "recommendation"),
        ("Top 10", "recommendation"),
        ("Favoriler", "recommendation"),
        ("Popüler yerler", "recommendation"),
        ("Trend mekanlar", "recommendation"),
        ("En çok beğenilen", "recommendation"),
        ("Puanı yüksek", "recommendation"),
        ("Tavsiye edilen", "recommendation"),
        ("Öneri listesi", "recommendation"),
        ("Gidilecek yerler", "recommendation"),
        ("Yapılacak şeyler", "recommendation"),
        ("Deneyim önerileri", "recommendation"),
        ("Ne ziyaret edilmeli", "recommendation"),
        ("Rehber önerileri", "recommendation"),
        ("Kişisel öneriler", "recommendation"),
    ])
    
    # ===== ROUTE_PLANNING (25 samples) =====
    training_data.extend([
        ("Rota planla", "route_planning"),
        ("Gezi planı", "route_planning"),
        ("İtineraryoluştur", "route_planning"),
        ("Günlük program", "route_planning"),
        ("Gezi rotası", "route_planning"),
        ("Tur planı", "route_planning"),
        ("3 günlük plan", "route_planning"),
        ("Hafta sonu planı", "route_planning"),
        ("Nasıl bir rota izlemeliyim", "route_planning"),
        ("Hangi sırayla gezilmeli", "route_planning"),
        ("Optimum rota", "route_planning"),
        ("Gezi programı", "route_planning"),
        ("Günlük rotasyon", "route_planning"),
        ("Sıralı gezinti", "route_planning"),
        ("Plan yap", "route_planning"),
        ("Gezi planı oluştur", "route_planning"),
        ("İstanbul gezisi planı", "route_planning"),
        ("Tarih sırasına göre", "route_planning"),
        ("Coğrafi yakınlık", "route_planning"),
        ("En verimli rota", "route_planning"),
        ("Gün programı", "route_planning"),
        ("Sabah akşam planı", "route_planning"),
        ("Zaman optimizasyonu", "route_planning"),
        ("Rotayı planla", "route_planning"),
        ("Gezi haritası", "route_planning"),
    ])
    
    # ===== PRICE_INFO (20 samples) =====
    training_data.extend([
        ("Giriş ücreti ne kadar", "price_info"),
        ("Müze ücreti", "price_info"),
        ("Tur fiyatları", "price_info"),
        ("Bilet fiyatı", "price_info"),
        ("Kaça mal olur", "price_info"),
        ("Fiyat bilgisi", "price_info"),
        ("Ne kadar öderim", "price_info"),
        ("Ücret ne kadar", "price_info"),
        ("Maliyet", "price_info"),
        ("Fiyat aralığı", "price_info"),
        ("Ücretli mi", "price_info"),
        ("Ücretsiz mi", "price_info"),
        ("İndirim var mı", "price_info"),
        ("Öğrenci indirimi", "price_info"),
        ("Kombine bilet", "price_info"),
        ("Günlük bilet", "price_info"),
        ("Haftalık bilet", "price_info"),
        ("Aylık ücret", "price_info"),
        ("Sezon fiyatı", "price_info"),
        ("Promosyon", "price_info"),
    ])
    
    # ===== BOOKING (20 samples) =====
    training_data.extend([
        ("Rezervasyon yapmak istiyorum", "booking"),
        ("Tur rezervasyonu", "booking"),
        ("Otel rezervasyon", "booking"),
        ("Masa ayırtmak istiyorum", "booking"),
        ("Bilet al", "booking"),
        ("Online rezervasyon", "booking"),
        ("Booking", "booking"),
        ("Rezerve et", "booking"),
        ("Yer ayır", "booking"),
        ("Kayıt yaptır", "booking"),
        ("Ön kayıt", "booking"),
        ("Bilet satın al", "booking"),
        ("Yer ayırtma", "booking"),
        ("Randevu al", "booking"),
        ("Rezervasyon bilgileri", "booking"),
        ("Nasıl rezervasyon yapılır", "booking"),
        ("Yer rezervasyonu", "booking"),
        ("Ticket", "booking"),
        ("Online bilet", "booking"),
        ("İptal ve iade", "booking"),
    ])
    
    # ===== CULTURAL_INFO (25 samples) =====
    training_data.extend([
        ("Boğaz turu hakkında bilgi", "cultural_info"),
        ("Cami ziyareti kuralları", "cultural_info"),
        ("Ramazan etkinlikleri", "cultural_info"),
        ("Yerel gelenek ve görenekler", "cultural_info"),
        ("Türk kültürü", "cultural_info"),
        ("Geleneksel etkinlikler", "cultural_info"),
        ("Kültürel özellikler", "cultural_info"),
        ("Dini yerler", "cultural_info"),
        ("Kutsal mekanlar", "cultural_info"),
        ("Gelenek", "cultural_info"),
        ("Görenek", "cultural_info"),
        ("Adet", "cultural_info"),
        ("Örf", "cultural_info"),
        ("Kültür gezisi", "cultural_info"),
        ("Mevlevi", "cultural_info"),
        ("Semazen", "cultural_info"),
        ("Türk hamamı", "cultural_info"),
        ("Hamam kültürü", "cultural_info"),
        ("Çay kültürü", "cultural_info"),
        ("Kahve falı", "cultural_info"),
        ("Nazar boncuğu", "cultural_info"),
        ("El sanatları", "cultural_info"),
        ("Hat sanatı", "cultural_info"),
        ("Ebru sanatı", "cultural_info"),
        ("Geleneksel müzik", "cultural_info"),
    ])
    
    # ===== FOOD (25 samples) =====
    training_data.extend([
        ("Türk mutfağı", "food"),
        ("Yemek kültürü", "food"),
        ("Geleneksel yemekler", "food"),
        ("Sokak lezzetleri", "food"),
        ("Street food", "food"),
        ("Tatlı önerileri", "food"),
        ("Kahvaltı nerede", "food"),
        ("Brunch", "food"),
        ("Deniz ürünleri", "food"),
        ("Meyhane", "food"),
        ("Rakı sofrası", "food"),
        ("Simit", "food"),
        ("Midye dolma", "food"),
        ("Balık ekmek", "food"),
        ("Kokoreç", "food"),
        ("Kumpir", "food"),
        ("Çiğ köfte", "food"),
        ("Gözleme", "food"),
        ("Börek", "food"),
        ("Mantı", "food"),
        ("İmam bayıldı", "food"),
        ("Hünkar beğendi", "food"),
        ("Deniz mahsülleri", "food"),
        ("Mevsim yemekleri", "food"),
        ("Organik yemek", "food"),
    ])
    
    # ===== HISTORY (25 samples) =====
    training_data.extend([
        ("Osmanlı tarihi", "history"),
        ("Bizans dönemi", "history"),
        ("Tarihi yerler", "history"),
        ("Konstantinopolis", "history"),
        ("Fatih Sultan Mehmet", "history"),
        ("İstanbul'un fethi", "history"),
        ("Osmanlı İmparatorluğu", "history"),
        ("Bizans İmparatorluğu", "history"),
        ("Roma dönemi", "history"),
        ("Antik dönem", "history"),
        ("Tarihi yapılar", "history"),
        ("Eski İstanbul", "history"),
        ("Tarihi semtler", "history"),
        ("Osmanlı mimarisi", "history"),
        ("Bizans mimarisi", "history"),
        ("Tarihi kalıntılar", "history"),
        ("Arkeolojik alanlar", "history"),
        ("Tarih dersi", "history"),
        ("Kronoloji", "history"),
        ("Tarihsel önem", "history"),
        ("Kültürel miras", "history"),
        ("UNESCO", "history"),
        ("Dünya mirası", "history"),
        ("Tarih bilgisi", "history"),
        ("Geçmiş", "history"),
    ])
    
    # ===== LOCAL_TIPS (20 samples) =====
    training_data.extend([
        ("Yerel önerileri", "local_tips"),
        ("Yerli gibi gez", "local_tips"),
        ("Yerel ipuçları", "local_tips"),
        ("İçeriden bilgiler", "local_tips"),
        ("Yerel sırları", "local_tips"),
        ("Mahalle önerileri", "local_tips"),
        ("Yerli tavsiyeleri", "local_tips"),
        ("Yerel restoran", "local_tips"),
        ("Mahalle kahvesi", "local_tips"),
        ("Semt pazarı", "local_tips"),
        ("Yerel deneyim", "local_tips"),
        ("Otantik yerler", "local_tips"),
        ("Gerçek İstanbul", "local_tips"),
        ("Yerel yaşam", "local_tips"),
        ("Gündelik hayat", "local_tips"),
        ("Mahalle kültürü", "local_tips"),
        ("Semt önerileri", "local_tips"),
        ("Yerli mekanları", "local_tips"),
        ("Yerel lezzetler", "local_tips"),
        ("Semt tavsiyeleri", "local_tips"),
    ])
    
    # ===== HIDDEN_GEMS (20 samples) =====
    training_data.extend([
        ("Turistik olmayan yerler", "hidden_gems"),
        ("Gizli mekanlar", "hidden_gems"),
        ("Az bilinen yerler", "hidden_gems"),
        ("Turistlerin gitmediği yerler", "hidden_gems"),
        ("Keşfedilmemiş yerler", "hidden_gems"),
        ("Gizli cennetler", "hidden_gems"),
        ("Bilinmeyen yerler", "hidden_gems"),
        ("Saklı kalmış", "hidden_gems"),
        ("Keşfedilecek yerler", "hidden_gems"),
        ("Farklı yerler", "hidden_gems"),
        ("Alternatif mekanlar", "hidden_gems"),
        ("Sıra dışı", "hidden_gems"),
        ("Underground", "hidden_gems"),
        ("İndependent", "hidden_gems"),
        ("Bağımsız mekanlar", "hidden_gems"),
        ("Küçük yerler", "hidden_gems"),
        ("Samimi mekanlar", "hidden_gems"),
        ("Butik yerler", "hidden_gems"),
        ("Özel yerler", "hidden_gems"),
        ("Gizli bahçeler", "hidden_gems"),
    ])
    
    # ===== FAMILY_ACTIVITIES (20 samples) =====
    training_data.extend([
        ("Çocuklu gezilecek yerler", "family_activities"),
        ("Aile için restoran", "family_activities"),
        ("Çocuk parkı", "family_activities"),
        ("Çocuk dostu mekan", "family_activities"),
        ("Oyun alanı", "family_activities"),
        ("Çocuklarla ne yapabilirim", "family_activities"),
        ("Aile etkinlikleri", "family_activities"),
        ("Çocuk müzesi", "family_activities"),
        ("Aquarium", "family_activities"),
        ("Akvaryum", "family_activities"),
        ("Hayvanat bahçesi", "family_activities"),
        ("Lunapark", "family_activities"),
        ("Tema parkı", "family_activities"),
        ("Çocuk tiyatrosu", "family_activities"),
        ("Çocuk atölyeleri", "family_activities"),
        ("Bilim merkezi", "family_activities"),
        ("Oyuncak müzesi", "family_activities"),
        ("Bebek değiştirme", "family_activities"),
        ("Aile paketi", "family_activities"),
        ("Çocuk menüsü", "family_activities"),
    ])
    
    # ===== ROMANTIC (20 samples) =====
    training_data.extend([
        ("Romantik restoran önerisi", "romantic"),
        ("Romantik yerler", "romantic"),
        ("Çiftler için", "romantic"),
        ("Balayı", "romantic"),
        ("Romantik gezi", "romantic"),
        ("Sevgiliye sürpriz", "romantic"),
        ("Romantik akşam", "romantic"),
        ("Mum ışığında yemek", "romantic"),
        ("Boğaz manzarası romantik", "romantic"),
        ("Gün batımı izleme", "romantic"),
        ("Romantik otel", "romantic"),
        ("Romantik deneyim", "romantic"),
        ("Evlilik teklifi", "romantic"),
        ("Yıldönümü", "romantic"),
        ("Sevgililer günü", "romantic"),
        ("Romantik kaçamak", "romantic"),
        ("İki kişilik", "romantic"),
        ("Couple aktiviteleri", "romantic"),
        ("Romantik mekan", "romantic"),
        ("Aşk temalı", "romantic"),
    ])
    
    # ===== BUDGET (20 samples) =====
    training_data.extend([
        ("Ucuz yerler", "budget"),
        ("Ekonomik gezinti", "budget"),
        ("Bütçeye uygun", "budget"),
        ("Hesaplı mekanlar", "budget"),
        ("Ucuz konaklama", "budget"),
        ("Hostel tavsiyesi", "budget"),
        ("Backpacker", "budget"),
        ("Sırt çantalı gezi", "budget"),
        ("Ücretsiz etkinlikler", "budget"),
        ("Bedava gezilecek", "budget"),
        ("Para harcamadan", "budget"),
        ("Ekonomik rehber", "budget"),
        ("Ucuz yemek", "budget"),
        ("Student discount", "budget"),
        ("Öğrenci indirimi", "budget"),
        ("Bütçe dostu", "budget"),
        ("Tasarruflu gezi", "budget"),
        ("Düşük bütçe", "budget"),
        ("Minimal harcama", "budget"),
        ("Ucuza tatil", "budget"),
    ])
    
    # ===== LUXURY (20 samples) =====
    training_data.extend([
        ("5 yıldızlı otel", "luxury"),
        ("Lüks otel", "luxury"),
        ("VIP hizmet", "luxury"),
        ("Prestijli mekanlar", "luxury"),
        ("Lüks restoran", "luxury"),
        ("Fine dining", "luxury"),
        ("Michelin yıldızlı", "luxury"),
        ("Premium", "luxury"),
        ("Exclusive", "luxury"),
        ("High-end", "luxury"),
        ("Lüks alışveriş", "luxury"),
        ("Designer mağazalar", "luxury"),
        ("Butik", "luxury"),
        ("Spa otel", "luxury"),
        ("Wellness", "luxury"),
        ("Lüks tur", "luxury"),
        ("Private tour", "luxury"),
        ("Özel rehber", "luxury"),
        ("Helikopter turu", "luxury"),
        ("Yat turu", "luxury"),
    ])
    
    return training_data


if __name__ == "__main__":
    data = create_final_comprehensive_dataset()
    
    # Remove duplicates
    data = list(set(data))
    
    print(f"Total samples: {len(data)}")
    
    # Count per intent
    intent_counts = {}
    for _, intent in data:
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\nSamples per intent:")
    for intent, count in sorted(intent_counts.items()):
        status = "✅" if count >= 20 else "⚠️ "
        print(f"{status} {intent}: {count}")
    
    # Save to file
    import json
    with open("comprehensive_training_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved to comprehensive_training_data.json")

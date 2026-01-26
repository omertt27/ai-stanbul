"""
Station Aliases - Multilingual Location Mappings
================================================

Comprehensive alias mappings for Istanbul locations:
- Station names in multiple languages (EN, TR, RU, DE, FR, AR)
- Tourist landmarks mapped to nearest stations
- Neighborhoods mapped to transit hubs

Author: AI Istanbul Team
Date: December 2024
"""

from typing import Dict, List


def build_station_aliases() -> Dict[str, List[str]]:
    """Build comprehensive alias mappings for popular locations."""
    return {
        # ====== TAKSIM AREA ======
        "taksim": ["M2-Taksim"],
        "taksim square": ["M2-Taksim"],
        "taksim meydani": ["M2-Taksim"],
        
        # ====== KADIKOY AREA ======
        "kadikoy": ["M4-Kadıköy", "FERRY-Kadıköy"],
        "kadıköy": ["M4-Kadıköy", "FERRY-Kadıköy"],
        "ayrilik cesmesi": ["M4-Ayrılık Çeşmesi", "MARMARAY-Ayrılık Çeşmesi"],
        "ayrılık çeşmesi": ["M4-Ayrılık Çeşmesi", "MARMARAY-Ayrılık Çeşmesi"],
        
        # ====== BESIKTAS AREA ======
        "besiktas": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "beşiktas": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "beşiktaş": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        
        # ====== SULTANAHMET/FATIH AREA ======
        "sultanahmet": ["T1-Sultanahmet"],
        "sultanahmet square": ["T1-Sultanahmet"],
        "blue mosque": ["T1-Sultanahmet"],
        "hagia sophia": ["T1-Sultanahmet"],
        "ayasofya": ["T1-Sultanahmet"],
        
        # ====== GALATA/KARAKOY AREA ======
        "galata": ["T1-Karaköy"],
        "galata tower": ["T1-Karaköy"],
        "karakoy": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
        "karaköy": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
        
        # ====== USKUDAR AREA ======
        "uskudar": ["M5-Üsküdar", "MARMARAY-Üsküdar", "FERRY-Üsküdar"],
        "üsküdar": ["M5-Üsküdar", "MARMARAY-Üsküdar", "FERRY-Üsküdar"],
        "uskudar square": ["M5-Üsküdar", "MARMARAY-Üsküdar", "FERRY-Üsküdar"],
        
        # ====== ISTIKLAL/BEYOGLU AREA ======
        "istiklal": ["M2-Taksim"],
        "istiklal street": ["M2-Taksim"],
        "istiklal caddesi": ["M2-Taksim"],
        "beyoglu": ["M2-Taksim", "T1-Karaköy"],
        "beyoğlu": ["M2-Taksim", "T1-Karaköy"],
        
        # ====== AIRPORTS ======
        "airport": ["M11-İstanbul Havalimanı"],
        "istanbul airport": ["M11-İstanbul Havalimanı"],
        "new airport": ["M11-İstanbul Havalimanı"],
        "ist airport": ["M11-İstanbul Havalimanı"],
        "yeni havalimani": ["M11-İstanbul Havalimanı"],
        "havalimani": ["M11-İstanbul Havalimanı"],
        "havalimanı": ["M11-İstanbul Havalimanı"],
        "havalimanından": ["M11-İstanbul Havalimanı"],
        "havalimanindan": ["M11-İstanbul Havalimanı"],
        "havalimanına": ["M11-İstanbul Havalimanı"],
        "havalimanina": ["M11-İstanbul Havalimanı"],
        "ataturk airport": ["M1A-Atatürk Havalimanı"],
        "atatürk airport": ["M1A-Atatürk Havalimanı"],
        "atatürk havalimani": ["M1A-Atatürk Havalimanı"],
        "sabiha gokcen": ["M4-Sabiha Gökçen Havalimanı"],
        "sabiha gökçen": ["M4-Sabiha Gökçen Havalimanı"],
        "saw": ["M4-Sabiha Gökçen Havalimanı"],
        "sabiha gokcen airport": ["M4-Sabiha Gökçen Havalimanı"],
        "sabiha gökçen airport": ["M4-Sabiha Gökçen Havalimanı"],
        
        # ====== EMINONU AREA ======
        "eminonu": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü"],
        "eminönü": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü"],
        "spice bazaar": ["T1-Eminönü"],
        "misir carsisi": ["T1-Eminönü"],
        
        # ====== SIRKECI ======
        "sirkeci": ["MARMARAY-Sirkeci", "T1-Sirkeci"],
        
        # ====== LEVENT AREA ======
        "levent": ["M2-Levent", "M6-Levent"],
        "4.levent": ["M2-4. Levent"],
        "4 levent": ["M2-4. Levent"],
        
        # ====== SISLI/MECIDIYEKOY AREA ======
        "sisli": ["M2-Şişli-Mecidiyeköy"],
        "şişli": ["M2-Şişli-Mecidiyeköy"],
        "mecidiyekoy": ["M7-Mecidiyeköy", "M2-Şişli-Mecidiyeköy"],
        "mecidiyeköy": ["M7-Mecidiyeköy", "M2-Şişli-Mecidiyeköy"],
        
        # ====== OLIMPIYAT/IKITELLI AREA ======
        "olimpiyat": ["M3-Olimpiyat", "M9-Olimpiyat"],
        "ikitelli": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
        "İkitelli": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
        "ikitelli sanayi": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
        
        # ====== BOSTANCI AREA ======
        "bostanci": ["M4-Bostancı", "MARMARAY-Bostancı"],
        "bostancı": ["M4-Bostancı", "MARMARAY-Bostancı"],
        
        # ====== PENDIK ======
        "pendik": ["MARMARAY-Pendik"],
        
        # ====== LANDMARKS & TOURIST ATTRACTIONS ======
        # Palaces
        "dolmabahce": ["T1-Kabataş"],
        "dolmabahce palace": ["T1-Kabataş"],
        "dolmabahçe": ["T1-Kabataş"],
        "dolmabahçe palace": ["T1-Kabataş"],
        "dolmabahçe sarayı": ["T1-Kabataş"],
        "topkapi": ["T1-Gülhane", "T1-Sultanahmet"],
        "topkapi palace": ["T1-Gülhane", "T1-Sultanahmet"],
        "topkapı": ["T1-Gülhane", "T1-Sultanahmet"],
        "topkapı palace": ["T1-Gülhane", "T1-Sultanahmet"],
        "topkapı sarayı": ["T1-Gülhane", "T1-Sultanahmet"],
        
        # Mosques
        "sultan ahmed mosque": ["T1-Sultanahmet"],
        "sultanahmet camii": ["T1-Sultanahmet"],
        "suleymaniye": ["T1-Beyazıt-Kapalıçarşı"],
        "suleymaniye mosque": ["T1-Beyazıt-Kapalıçarşı"],
        "süleymaniye": ["T1-Beyazıt-Kapalıçarşı"],
        "süleymaniye camii": ["T1-Beyazıt-Kapalıçarşı"],
        
        # Museums
        "aya sofya": ["T1-Sultanahmet"],
        "archaeological museum": ["T1-Gülhane"],
        "arkeoloji muzesi": ["T1-Gülhane"],
        
        # Markets
        "grand bazaar": ["T1-Beyazıt-Kapalıçarşı"],
        "kapali carsi": ["T1-Beyazıt-Kapalıçarşı"],
        "kapalıçarşı": ["T1-Beyazıt-Kapalıçarşı"],
        "egyptian bazaar": ["T1-Eminönü"],
        
        # Galata
        "galata bridge": ["T1-Karaköy", "T1-Eminönü"],
        "galata köprüsü": ["T1-Karaköy", "T1-Eminönü"],
        
        # Parks
        "gulhane": ["T1-Gülhane"],
        "gülhane": ["T1-Gülhane"],
        "gulhane park": ["T1-Gülhane"],
        "gülhane parkı": ["T1-Gülhane"],
        
        # ====== NEIGHBORHOODS ======
        "ortakoy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "ortaköy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "balat": ["T4-Fener", "T4-Balat"],
        "fener": ["T4-Fener"],
        "fatih": ["T4-Fatih", "T1-Aksaray"],
        "aksaray": ["T1-Aksaray", "M1A-Aksaray"],
        "cihangir": ["M2-Taksim"],
        "galatasaray": ["M2-Taksim"],
        "nisantasi": ["M2-Osmanbey"],
        "nişantaşı": ["M2-Osmanbey"],
        "osmanbey": ["M2-Osmanbey"],
        "bebek": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        
        # ====== GENERIC DESTINATIONS ======
        "asian side": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
        "asia": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
        "anadolu yakasi": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
        "anadolu yakası": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
        "european side": ["M2-Taksim"],
        "europe": ["M2-Taksim"],
        "avrupa yakasi": ["M2-Taksim"],
        "avrupa yakası": ["M2-Taksim"],
        "city center": ["M2-Taksim", "T1-Sultanahmet"],
        "city centre": ["M2-Taksim", "T1-Sultanahmet"],
        "center": ["M2-Taksim", "T1-Sultanahmet"],
        "centre": ["M2-Taksim", "T1-Sultanahmet"],
        "downtown": ["M2-Taksim"],
        "sehir merkezi": ["M2-Taksim", "T1-Sultanahmet"],
        "şehir merkezi": ["M2-Taksim", "T1-Sultanahmet"],
        "old city": ["T1-Sultanahmet"],
        "historic peninsula": ["T1-Sultanahmet"],
        "tarihi yarimada": ["T1-Sultanahmet"],
        "tarihi yarımada": ["T1-Sultanahmet"],
        
        # Islands
        "princes islands": ["FERRY-Kadıköy", "FERRY-Eminönü"],
        "adalar": ["FERRY-Kadıköy", "FERRY-Eminönü"],
        "buyukada": ["FERRY-Kadıköy", "FERRY-Eminönü"],
        "büyükada": ["FERRY-Kadıköy", "FERRY-Eminönü"],
        "heybeliada": ["FERRY-Kadıköy", "FERRY-Eminönü"],
        
        # ====== RUSSIAN (Cyrillic) ======
        "таксим": ["M2-Taksim"],
        "кадыкёй": ["M4-Kadıköy", "FERRY-Kadıköy"],
        "кадикой": ["M4-Kadıköy", "FERRY-Kadıköy"],
        "ускюдар": ["M5-Üsküdar", "MARMARAY-Üsküdar", "FERRY-Üsküdar"],
        "султанахмет": ["T1-Sultanahmet"],
        "эминёню": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü"],
        "бешикташ": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "галата": ["T1-Karaköy"],
        "каракёй": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
        "аэропорт": ["M11-İstanbul Havalimanı"],
        "стамбульский аэропорт": ["M11-İstanbul Havalimanı"],
        "истикляль": ["M2-Taksim"],
        "гранд базар": ["T1-Beyazıt-Kapalıçarşı"],
        "айя софия": ["T1-Sultanahmet"],
        "голубая мечеть": ["T1-Sultanahmet"],
        "топкапы": ["T1-Gülhane", "T1-Sultanahmet"],
        "дворец долмабахче": ["T1-Kabataş"],
        "долмабахче": ["T1-Kabataş"],
        
        # ====== GERMAN ======
        "flughafen": ["M11-İstanbul Havalimanı"],
        "istanbul flughafen": ["M11-İstanbul Havalimanı"],
        "blaue moschee": ["T1-Sultanahmet"],
        "großer basar": ["T1-Beyazıt-Kapalıçarşı"],
        "gewürzbasar": ["T1-Eminönü"],
        "topkapi palast": ["T1-Gülhane", "T1-Sultanahmet"],
        "dolmabahce palast": ["T1-Kabataş"],
        "galata turm": ["T1-Karaköy"],
        "asiatische seite": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
        "europäische seite": ["M2-Taksim"],
        "altstadt": ["T1-Sultanahmet"],
        "stadtzentrum": ["M2-Taksim", "T1-Sultanahmet"],
        
        # ====== FRENCH ======
        "aéroport": ["M11-İstanbul Havalimanı"],
        "aéroport d'istanbul": ["M11-İstanbul Havalimanı"],
        "mosquée bleue": ["T1-Sultanahmet"],
        "grand bazar": ["T1-Beyazıt-Kapalıçarşı"],
        "bazar aux épices": ["T1-Eminönü"],
        "sainte sophie": ["T1-Sultanahmet"],
        "palais de topkapi": ["T1-Gülhane", "T1-Sultanahmet"],
        "palais de dolmabahce": ["T1-Kabataş"],
        "tour de galata": ["T1-Karaköy"],
        "côté asiatique": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
        "côté européen": ["M2-Taksim"],
        "vieille ville": ["T1-Sultanahmet"],
        "centre-ville": ["M2-Taksim", "T1-Sultanahmet"],
        
        # ====== ARABIC ======
        "تقسيم": ["M2-Taksim"],
        "تكسيم": ["M2-Taksim"],
        "كاديكوي": ["M4-Kadıköy", "FERRY-Kadıköy"],
        "كادي كوي": ["M4-Kadıköy", "FERRY-Kadıköy"],
        "اسكودار": ["M5-Üsküdar", "MARMARAY-Üsküdar", "FERRY-Üsküdar"],
        "السلطان أحمد": ["T1-Sultanahmet"],
        "سلطان احمد": ["T1-Sultanahmet"],
        "امينونو": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü"],
        "بشيكتاش": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "غلطة": ["T1-Karaköy"],
        "كاراكوي": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
        "المطار": ["M11-İstanbul Havalimanı"],
        "مطار اسطنبول": ["M11-İstanbul Havalimanı"],
        "شارع الاستقلال": ["M2-Taksim"],
        "البازار الكبير": ["T1-Beyazıt-Kapalıçarşı"],
        "آيا صوفيا": ["T1-Sultanahmet"],
        "المسجد الازرق": ["T1-Sultanahmet"],
        "قصر توبكابي": ["T1-Gülhane", "T1-Sultanahmet"],
        "قصر دولما بهجة": ["T1-Kabataş"],
        "الجانب الآسيوي": ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"],
        "الجانب الأوروبي": ["M2-Taksim"],
        "المدينة القديمة": ["T1-Sultanahmet"],
        "وسط المدينة": ["M2-Taksim", "T1-Sultanahmet"],
    }


def build_neighborhood_stations() -> Dict[str, List[str]]:
    """Map neighborhoods to their nearest major transit stations."""
    return {
        # ASIAN SIDE
        "kadıköy": ["M4-Kadıköy", "M4-Ayrılık Çeşmesi"],
        "kadikoy": ["M4-Kadıköy", "M4-Ayrılık Çeşmesi"],
        "üsküdar": ["MARMARAY-Üsküdar", "M5-Üsküdar"],
        "uskudar": ["MARMARAY-Üsküdar", "M5-Üsküdar"],
        "bostancı": ["MARMARAY-Bostancı", "M4-Bostancı"],
        "bostanci": ["MARMARAY-Bostancı", "M4-Bostancı"],
        "pendik": ["MARMARAY-Pendik", "M4-Pendik"],
        "kartal": ["MARMARAY-Kartal", "M4-Kartal"],
        "maltepe": ["M4-Maltepe"],
        "ataşehir": ["M4-Ünalan", "M4-Kozyatağı"],
        "atasehir": ["M4-Ünalan", "M4-Kozyatağı"],
        
        # EUROPEAN SIDE - Historic/Tourist
        "taksim": ["M2-Taksim"],
        "beyoğlu": ["M2-Taksim", "T1-Karaköy"],
        "beyoglu": ["M2-Taksim", "T1-Karaköy"],
        "sultanahmet": ["T1-Sultanahmet"],
        "eminönü": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü", "MARMARAY-Sirkeci"],
        "eminonu": ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü", "MARMARAY-Sirkeci"],
        "karaköy": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
        "karakoy": ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"],
        "kabataş": ["T1-Kabataş", "T4-Kabataş", "FERRY-Kabataş"],
        "kabatas": ["T1-Kabataş", "T4-Kabataş", "FERRY-Kabataş"],
        "beşiktaş": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "besiktas": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "şişli": ["M2-Şişli-Mecidiyeköy"],
        "sisli": ["M2-Şişli-Mecidiyeköy"],
        "levent": ["M2-Levent", "M2-4. Levent", "M6-Levent"],
        "mecidiyeköy": ["M2-Şişli-Mecidiyeköy", "M7-Mecidiyeköy"],
        "mecidiyekoy": ["M2-Şişli-Mecidiyeköy", "M7-Mecidiyeköy"],
        "zeytinburnu": ["T1-Zeytinburnu", "MARMARAY-Zeytinburnu"],
        "bakırköy": ["MARMARAY-Bakırköy"],
        "bakirkoy": ["MARMARAY-Bakırköy"],
        "yeşilköy": ["MARMARAY-Yeşilköy"],
        "yesilkoy": ["MARMARAY-Yeşilköy"],
        
        # Additional Neighborhoods
        "fatih": ["T4-Fatih", "T1-Aksaray"],
        "aksaray": ["T1-Aksaray", "M1A-Aksaray"],
        "balat": ["T4-Fener", "T4-Balat"],
        "fener": ["T4-Fener"],
        "ortaköy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "ortakoy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "nişantaşı": ["M2-Osmanbey"],
        "nisantasi": ["M2-Osmanbey"],
        "osmanbey": ["M2-Osmanbey"],
        "cihangir": ["M2-Taksim"],
        "galata": ["T1-Karaköy"],
        "bebek": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "etiler": ["M2-4. Levent"],
        "maslak": ["M2-Hacıosman"],
        "sariyer": ["M2-Hacıosman"],
        "sarıyer": ["M2-Hacıosman"],
        
        # Airports
        "atatürk airport": ["M1A-Atatürk Havalimanı"],
        "ataturk airport": ["M1A-Atatürk Havalimanı"],
        "istanbul airport": ["M11-İstanbul Havalimanı"],
        "new airport": ["M11-İstanbul Havalimanı"],
        "sabiha gökçen": ["M4-Sabiha Gökçen Havalimanı"],
        "sabiha gokcen": ["M4-Sabiha Gökçen Havalimanı"],
        
        # Transfer Hubs
        "yenikapı": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M2-Yenikapı"],
        "yenikapi": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M2-Yenikapı"],
        "sirkeci": ["MARMARAY-Sirkeci", "T1-Sirkeci"],
    }


def build_route_patterns() -> Dict[str, List[str]]:
    """Build common route patterns for major destinations."""
    return {
        "kadıköy_to_taksim": ["M4", "MARMARAY", "M2", "F1"],
        "kadıköy_to_sultanahmet": ["M4", "MARMARAY", "T1"],
        "kadıköy_to_beyoğlu": ["M4", "MARMARAY", "M2"],
        "kadıköy_to_beşiktaş": ["FERRY", "M4", "MARMARAY", "M2"],
        "taksim_to_kadıköy": ["F1", "MARMARAY", "M4"],
        "taksim_to_sultanahmet": ["M2", "T1"],
        "taksim_to_airport": ["M2", "M1A", "M1B"],
        "sultanahmet_to_kadıköy": ["T1", "MARMARAY", "M4"],
        "sultanahmet_to_taksim": ["T1", "F1", "M2"],
        "sultanahmet_to_airport": ["T1", "M1A", "M1B"],
        "european_to_asian": ["MARMARAY", "FERRY"],
        "asian_to_european": ["MARMARAY", "FERRY"],
    }

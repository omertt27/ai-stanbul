"""
Station Aliases - Comprehensive Multilingual Location Mappings
==============================================================

Comprehensive alias mappings for Istanbul locations supporting 6 languages:
- English (EN)
- Turkish (TR) - with grammatical variations (ablative -dan/-den, dative -a/-e)
- Russian (RU) - Cyrillic script
- German (DE)
- French (FR)
- Arabic (AR)

Features:
- Station names in multiple languages
- Tourist landmarks mapped to nearest stations
- Neighborhoods mapped to transit hubs
- Common misspellings and typos
- Informal/colloquial names

Author: AI Istanbul Team
Date: January 2026
"""

from typing import Dict, List


def build_station_aliases() -> Dict[str, List[str]]:
    """
    Build comprehensive alias mappings for popular locations.
    
    Supports:
    - 6 languages (EN, TR, RU, DE, FR, AR)
    - Turkish grammatical cases (ablative -dan/-den, dative -a/-e/-ya/-ye)
    - Common misspellings and typos
    - Informal names tourists use
    """
    
    aliases = {}
    
    # ============================================================
    # TAKSIM AREA - Major Tourist Hub
    # ============================================================
    taksim_stations = ["M2-Taksim"]
    
    # English
    aliases.update({
        "taksim": taksim_stations,
        "taksim square": taksim_stations,
        "taksim metro": taksim_stations,
        "taksim station": taksim_stations,
    })
    
    # Turkish (with grammatical variations)
    aliases.update({
        "taksim meydanı": taksim_stations,
        "taksim meydani": taksim_stations,
        # Ablative case (-dan/-den = "from")
        "taksimden": taksim_stations,
        "taksim'den": taksim_stations,
        "taksimdan": taksim_stations,  # common misspelling
        # Dative case (-a/-e = "to")
        "taksime": taksim_stations,
        "taksim'e": taksim_stations,
        "taksıme": taksim_stations,  # common typo
        # Locative case (-da/-de = "at/in")
        "taksimde": taksim_stations,
        "taksim'de": taksim_stations,
    })
    
    # Russian (Cyrillic)
    aliases.update({
        "таксим": taksim_stations,
        "площадь таксим": taksim_stations,
        "таксим сквер": taksim_stations,
    })
    
    # German
    aliases.update({
        "taksim platz": taksim_stations,
        "taksimplatz": taksim_stations,
    })
    
    # French
    aliases.update({
        "place taksim": taksim_stations,
        "place de taksim": taksim_stations,
    })
    
    # Arabic
    aliases.update({
        "تقسيم": taksim_stations,
        "تكسيم": taksim_stations,
        "ميدان تقسيم": taksim_stations,
        "ساحة تقسيم": taksim_stations,
    })
    
    # ============================================================
    # KADIKOY AREA - Asian Side Hub
    # ============================================================
    kadikoy_stations = ["M4-Kadıköy", "FERRY-Kadıköy"]
    
    # English
    aliases.update({
        "kadikoy": kadikoy_stations,
        "kadikoy square": kadikoy_stations,
        "kadikoy pier": kadikoy_stations,
        "kadikoy ferry": kadikoy_stations,
    })
    
    # Turkish (with grammatical variations)
    aliases.update({
        "kadıköy": kadikoy_stations,
        "kadıkoy": kadikoy_stations,  # common typo
        # Ablative case
        "kadıköyden": kadikoy_stations,
        "kadıköy'den": kadikoy_stations,
        "kadikoyden": kadikoy_stations,
        # Dative case
        "kadıköye": kadikoy_stations,
        "kadıköy'e": kadikoy_stations,
        "kadikoye": kadikoy_stations,
        # Locative case
        "kadıköyde": kadikoy_stations,
        "kadıköy'de": kadikoy_stations,
    })
    
    # Russian
    aliases.update({
        "кадыкёй": kadikoy_stations,
        "кадикой": kadikoy_stations,
        "кадыкей": kadikoy_stations,
    })
    
    # German
    aliases.update({
        "kadıköy fähre": kadikoy_stations,
        "kadikoy fähre": kadikoy_stations,
    })
    
    # French
    aliases.update({
        "kadıköy ferry": kadikoy_stations,
        "port de kadıköy": kadikoy_stations,
    })
    
    # Arabic
    aliases.update({
        "كاديكوي": kadikoy_stations,
        "كادي كوي": kadikoy_stations,
        "قاضي كوي": kadikoy_stations,
    })
    
    # ============================================================
    # BESIKTAS AREA
    # ============================================================
    besiktas_stations = ["T4-Beşiktaş", "FERRY-Beşiktaş"]
    
    # English & Turkish variations
    aliases.update({
        "besiktas": besiktas_stations,
        "beşiktas": besiktas_stations,
        "beşiktaş": besiktas_stations,
        "besiktas pier": besiktas_stations,
        "besiktas ferry": besiktas_stations,
        # Turkish grammatical cases
        "beşiktaştan": besiktas_stations,
        "beşiktaş'tan": besiktas_stations,
        "besiktastan": besiktas_stations,
        "beşiktaşa": besiktas_stations,
        "beşiktaş'a": besiktas_stations,
        "besiktasa": besiktas_stations,
        "beşiktaşta": besiktas_stations,
    })
    
    # Russian
    aliases.update({
        "бешикташ": besiktas_stations,
        "бешикташ порт": besiktas_stations,
    })
    
    # German
    aliases.update({
        "besiktas hafen": besiktas_stations,
    })
    
    # French
    aliases.update({
        "besiktas port": besiktas_stations,
    })
    
    # Arabic
    aliases.update({
        "بشيكتاش": besiktas_stations,
        "بيشكتاش": besiktas_stations,
    })
    
    # ============================================================
    # SULTANAHMET / FATIH AREA - Historic Peninsula
    # ============================================================
    sultanahmet_stations = ["T1-Sultanahmet"]
    
    # English
    aliases.update({
        "sultanahmet": sultanahmet_stations,
        "sultanahmet square": sultanahmet_stations,
        "sultan ahmet": sultanahmet_stations,
        "blue mosque": sultanahmet_stations,
        "the blue mosque": sultanahmet_stations,
        "hagia sophia": sultanahmet_stations,
        "aya sophia": sultanahmet_stations,
        "ayasofya": sultanahmet_stations,
        "aya sofya": sultanahmet_stations,
        "hippodrome": sultanahmet_stations,
        "at meydani": sultanahmet_stations,
    })
    
    # Turkish
    aliases.update({
        "sultanahmet meydanı": sultanahmet_stations,
        "sultanahmet meydani": sultanahmet_stations,
        "sultanahmet camii": sultanahmet_stations,
        "sultan ahmed camii": sultanahmet_stations,
        "mavi cami": sultanahmet_stations,  # "blue mosque" in Turkish
        # Grammatical cases
        "sultanahmetten": sultanahmet_stations,
        "sultanahmet'ten": sultanahmet_stations,
        "sultanahmete": sultanahmet_stations,
        "sultanahmet'e": sultanahmet_stations,
        "sultanahmetle": sultanahmet_stations,
    })
    
    # Russian
    aliases.update({
        "султанахмет": sultanahmet_stations,
        "голубая мечеть": sultanahmet_stations,
        "синяя мечеть": sultanahmet_stations,
        "айя софия": sultanahmet_stations,
        "айя-софия": sultanahmet_stations,
        "святая софия": sultanahmet_stations,
        "ипподром": sultanahmet_stations,
    })
    
    # German
    aliases.update({
        "sultanahmet platz": sultanahmet_stations,
        "blaue moschee": sultanahmet_stations,
        "hagia sophia museum": sultanahmet_stations,
    })
    
    # French
    aliases.update({
        "place sultanahmet": sultanahmet_stations,
        "mosquée bleue": sultanahmet_stations,
        "la mosquée bleue": sultanahmet_stations,
        "sainte sophie": sultanahmet_stations,
        "sainte-sophie": sultanahmet_stations,
    })
    
    # Arabic
    aliases.update({
        "السلطان أحمد": sultanahmet_stations,
        "سلطان احمد": sultanahmet_stations,
        "الجامع الأزرق": sultanahmet_stations,
        "المسجد الازرق": sultanahmet_stations,
        "آيا صوفيا": sultanahmet_stations,
        "ايا صوفيا": sultanahmet_stations,
    })
    
    # ============================================================
    # GALATA / KARAKOY AREA
    # ============================================================
    karakoy_stations = ["T1-Karaköy", "T4-Karaköy", "FERRY-Karaköy"]
    galata_stations = ["T1-Karaköy"]  # Galata Tower area
    
    # English - Galata
    aliases.update({
        "galata": galata_stations,
        "galata tower": galata_stations,
        "the galata tower": galata_stations,
        "galata district": galata_stations,
    })
    
    # Turkish - Galata
    aliases.update({
        "galata kulesi": galata_stations,
        "galata kulesine": galata_stations,  # dative
        "galata kulesinden": galata_stations,  # ablative
        "galatada": galata_stations,
        "galataya": galata_stations,
        "galatadan": galata_stations,
    })
    
    # Russian - Galata
    aliases.update({
        "галата": galata_stations,
        "галатская башня": galata_stations,
        "башня галата": galata_stations,
    })
    
    # German - Galata
    aliases.update({
        "galata turm": galata_stations,
        "galaturm": galata_stations,
        "galataturm": galata_stations,
    })
    
    # French - Galata
    aliases.update({
        "tour de galata": galata_stations,
        "la tour de galata": galata_stations,
        "tour galata": galata_stations,
    })
    
    # Arabic - Galata
    aliases.update({
        "غلطة": galata_stations,
        "برج غلطة": galata_stations,
        "برج جالاتا": galata_stations,
    })
    
    # English - Karakoy
    aliases.update({
        "karakoy": karakoy_stations,
        "karakoy pier": karakoy_stations,
        "karakoy ferry": karakoy_stations,
        "karakoy port": karakoy_stations,
    })
    
    # Turkish - Karakoy
    aliases.update({
        "karaköy": karakoy_stations,
        "karakoyden": karakoy_stations,
        "karaköy'den": karakoy_stations,
        "karakoye": karakoy_stations,
        "karaköy'e": karakoy_stations,
        "karaköyde": karakoy_stations,
    })
    
    # Russian - Karakoy
    aliases.update({
        "каракёй": karakoy_stations,
        "каракой": karakoy_stations,
    })
    
    # Arabic - Karakoy
    aliases.update({
        "كاراكوي": karakoy_stations,
        "كاراكوى": karakoy_stations,
    })
    
    # Galata Bridge
    galata_bridge_stations = ["T1-Karaköy", "T1-Eminönü"]
    aliases.update({
        "galata bridge": galata_bridge_stations,
        "galata köprüsü": galata_bridge_stations,
        "galata koprusu": galata_bridge_stations,
        "галатский мост": galata_bridge_stations,
        "galatabrücke": galata_bridge_stations,
        "pont de galata": galata_bridge_stations,
        "جسر غلطة": galata_bridge_stations,
    })
    
    # Galatasaray (different location - near Taksim)
    aliases.update({
        "galatasaray": ["M2-Taksim"],
        "galatasaray lisesi": ["M2-Taksim"],
        "galatasaray high school": ["M2-Taksim"],
    })
    
    # ============================================================
    # USKUDAR AREA - Asian Side
    # ============================================================
    uskudar_stations = ["M5-Üsküdar", "MARMARAY-Üsküdar", "FERRY-Üsküdar"]
    
    # English
    aliases.update({
        "uskudar": uskudar_stations,
        "uskudar square": uskudar_stations,
        "uskudar pier": uskudar_stations,
        "uskudar ferry": uskudar_stations,
        "scutari": uskudar_stations,  # historical name
    })
    
    # Turkish
    aliases.update({
        "üsküdar": uskudar_stations,
        "uskudardan": uskudar_stations,
        "üsküdar'dan": uskudar_stations,
        "uskudara": uskudar_stations,
        "üsküdar'a": uskudar_stations,
        "uskudarda": uskudar_stations,
    })
    
    # Russian
    aliases.update({
        "ускюдар": uskudar_stations,
        "юскюдар": uskudar_stations,
        "ускудар": uskudar_stations,
    })
    
    # German
    aliases.update({
        "üsküdar hafen": uskudar_stations,
    })
    
    # French
    aliases.update({
        "üsküdار port": uskudar_stations,
        "scutari": uskudar_stations,
    })
    
    # Arabic
    aliases.update({
        "اسكودار": uskudar_stations,
        "أسكدار": uskudar_stations,
        "سكوتاري": uskudar_stations,
    })
    
    # ============================================================
    # ISTIKLAL / BEYOGLU AREA
    # ============================================================
    istiklal_stations = ["M2-Taksim"]
    beyoglu_stations = ["M2-Taksim", "T1-Karaköy"]
    
    # English
    aliases.update({
        "istiklal": istiklal_stations,
        "istiklal street": istiklal_stations,
        "istiklal avenue": istiklal_stations,
        "independence avenue": istiklal_stations,
    })
    
    # Turkish
    aliases.update({
        "istiklal caddesi": istiklal_stations,
        "istiklâl caddesi": istiklal_stations,
        "istiklalden": istiklal_stations,
        "istiklale": istiklal_stations,
    })
    
    # Russian
    aliases.update({
        "истикляль": istiklal_stations,
        "улица истикляль": istiklal_stations,
        "проспект независимости": istiklal_stations,
    })
    
    # German
    aliases.update({
        "istiklal straße": istiklal_stations,
        "istiklalstraße": istiklal_stations,
    })
    
    # French
    aliases.update({
        "avenue istiklal": istiklal_stations,
        "rue istiklal": istiklal_stations,
    })
    
    # Arabic
    aliases.update({
        "شارع الاستقلال": istiklal_stations,
        "شارع استقلال": istiklal_stations,
    })
    
    # Beyoglu
    aliases.update({
        "beyoglu": beyoglu_stations,
        "beyoğlu": beyoglu_stations,
        "pera": beyoglu_stations,  # historical name
        "бейоглу": beyoglu_stations,
        "بيوغلو": beyoglu_stations,
    })
    
    # ============================================================
    # AIRPORTS
    # ============================================================
    istanbul_airport = ["M11-İstanbul Havalimanı"]
    ataturk_airport = ["M1A-Atatürk Havalimanı"]
    sabiha_airport = ["M4-Sabiha Gökçen Havalimanı"]
    
    # Istanbul Airport (New) - All languages
    aliases.update({
        # English
        "airport": istanbul_airport,
        "istanbul airport": istanbul_airport,
        "new airport": istanbul_airport,
        "ist": istanbul_airport,
        "ist airport": istanbul_airport,
        "isl": istanbul_airport,
        # Turkish (with grammatical cases)
        "havalimanı": istanbul_airport,
        "havalimani": istanbul_airport,
        "istanbul havalimanı": istanbul_airport,
        "istanbul havalimani": istanbul_airport,
        "yeni havalimanı": istanbul_airport,
        "yeni havalimani": istanbul_airport,
        # From airport (ablative)
        "havalimanından": istanbul_airport,
        "havalimanindan": istanbul_airport,
        "havalimanı'ndan": istanbul_airport,
        "havalimanından nasıl gidilir": istanbul_airport,
        # To airport (dative)
        "havalimanına": istanbul_airport,
        "havalimanina": istanbul_airport,
        "havalimanı'na": istanbul_airport,
        # Russian
        "аэропорт": istanbul_airport,
        "аэропорт стамбула": istanbul_airport,
        "стамбульский аэропорт": istanbul_airport,
        "новый аэропорт": istanbul_airport,
        # German
        "flughafen": istanbul_airport,
        "flughafen istanbul": istanbul_airport,
        "istanbul flughafen": istanbul_airport,
        "neuer flughafen": istanbul_airport,
        # French
        "aéroport": istanbul_airport,
        "aéroport d'istanbul": istanbul_airport,
        "aeroport istanbul": istanbul_airport,
        "nouvel aéroport": istanbul_airport,
        # Arabic
        "المطار": istanbul_airport,
        "مطار اسطنبول": istanbul_airport,
        "مطار إسطنبول": istanbul_airport,
        "مطار استانبول": istanbul_airport,
        "المطار الجديد": istanbul_airport,
    })
    
    # Ataturk Airport (Old)
    aliases.update({
        "ataturk": ataturk_airport,
        "ataturk airport": ataturk_airport,
        "atatürk": ataturk_airport,
        "atatürk airport": ataturk_airport,
        "atatürk havalimanı": ataturk_airport,
        "atatürk havalimani": ataturk_airport,
        "ataturk havalimani": ataturk_airport,
        "old airport": ataturk_airport,
        "eski havalimanı": ataturk_airport,
        "аэропорт ататюрк": ataturk_airport,
        "flughafen atatürk": ataturk_airport,
        "aéroport atatürk": ataturk_airport,
        "مطار أتاتورك": ataturk_airport,
    })
    
    # Sabiha Gokcen Airport
    aliases.update({
        "sabiha": sabiha_airport,
        "sabiha gokcen": sabiha_airport,
        "sabiha gökçen": sabiha_airport,
        "sabiha gokcen airport": sabiha_airport,
        "sabiha gökçen airport": sabiha_airport,
        "sabiha gökçen havalimanı": sabiha_airport,
        "saw": sabiha_airport,
        "asian airport": sabiha_airport,
        "аэропорт сабиха гёкчен": sabiha_airport,
        "сабиха гёкчен": sabiha_airport,
        "flughafen sabiha gökçen": sabiha_airport,
        "aéroport sabiha gökçen": sabiha_airport,
        "مطار صبيحة": sabiha_airport,
        "مطار صبيحة جوكشن": sabiha_airport,
    })
    
    # ============================================================
    # EMINONU AREA
    # ============================================================
    eminonu_stations = ["T1-Eminönü", "T4-Eminönü", "FERRY-Eminönü"]
    
    aliases.update({
        # English
        "eminonu": eminonu_stations,
        "eminonu pier": eminonu_stations,
        "eminonu ferry": eminonu_stations,
        "spice bazaar": eminonu_stations,
        "spice market": eminonu_stations,
        "egyptian bazaar": eminonu_stations,
        # Turkish
        "eminönü": eminonu_stations,
        "eminönü'nden": eminonu_stations,
        "eminonuden": eminonu_stations,
        "eminönü'ne": eminonu_stations,
        "eminonuye": eminonu_stations,
        "mısır çarşısı": eminonu_stations,
        "misir carsisi": eminonu_stations,
        # Russian
        "эминёню": eminonu_stations,
        "эминоню": eminonu_stations,
        "рынок специй": eminonu_stations,
        "египетский базар": eminonu_stations,
        # German
        "gewürzbasar": eminonu_stations,
        "gewurzbasar": eminonu_stations,
        "ägyptischer basar": eminonu_stations,
        # French
        "bazar aux épices": eminonu_stations,
        "marché aux épices": eminonu_stations,
        "bazar égyptien": eminonu_stations,
        # Arabic
        "امينونو": eminonu_stations,
        "أمينونو": eminonu_stations,
        "سوق التوابل": eminonu_stations,
        "البازار المصري": eminonu_stations,
    })
    
    # ============================================================
    # SIRKECI
    # ============================================================
    sirkeci_stations = ["MARMARAY-Sirkeci", "T1-Sirkeci"]
    
    aliases.update({
        "sirkeci": sirkeci_stations,
        "sirkeci station": sirkeci_stations,
        "sirkeci train station": sirkeci_stations,
        "sirkeci garı": sirkeci_stations,
        "sirkeci gari": sirkeci_stations,
        "sirkeciden": sirkeci_stations,
        "sirkeci'den": sirkeci_stations,
        "sirkeciye": sirkeci_stations,
        "сиркеджи": sirkeci_stations,
        "вокзал сиркеджи": sirkeci_stations,
        "bahnhof sirkeci": sirkeci_stations,
        "gare de sirkeci": sirkeci_stations,
        "محطة سيركجي": sirkeci_stations,
    })
    
    # ============================================================
    # LEVENT AREA - Business District
    # ============================================================
    levent_stations = ["M2-Levent", "M6-Levent"]
    levent4_stations = ["M2-4. Levent"]
    
    aliases.update({
        "levent": levent_stations,
        "levent metro": levent_stations,
        "левент": levent_stations,
        "4.levent": levent4_stations,
        "4 levent": levent4_stations,
        "4. levent": levent4_stations,
        "dördüncü levent": levent4_stations,
        "dorduncu levent": levent4_stations,
        "4-й левент": levent4_stations,
        "4ый левент": levent4_stations,
    })
    
    # ============================================================
    # SISLI / MECIDIYEKOY AREA
    # ============================================================
    sisli_stations = ["M2-Şişli-Mecidiyeköy"]
    mecidiyekoy_stations = ["M7-Mecidiyeköy", "M2-Şişli-Mecidiyeköy"]
    
    aliases.update({
        "sisli": sisli_stations,
        "şişli": sisli_stations,
        "sisliden": sisli_stations,
        "şişli'den": sisli_stations,
        "sisliye": sisli_stations,
        "шишли": sisli_stations,
        "mecidiyekoy": mecidiyekoy_stations,
        "mecidiyeköy": mecidiyekoy_stations,
        "mecidiyeköy'den": mecidiyekoy_stations,
        "mecidiyeköy'e": mecidiyekoy_stations,
        "меджидиекёй": mecidiyekoy_stations,
    })
    
    # ============================================================
    # LANDMARKS & TOURIST ATTRACTIONS
    # ============================================================
    
    # Palaces
    dolmabahce_stations = ["T1-Kabataş"]
    topkapi_stations = ["T1-Gülhane", "T1-Sultanahmet"]
    
    aliases.update({
        # Dolmabahce - All languages
        "dolmabahce": dolmabahce_stations,
        "dolmabahce palace": dolmabahce_stations,
        "dolmabahçe": dolmabahce_stations,
        "dolmabahçe palace": dolmabahce_stations,
        "dolmabahçe sarayı": dolmabahce_stations,
        "dolmabahce sarayi": dolmabahce_stations,
        "dolmabahçeden": dolmabahce_stations,
        "dolmabahçeye": dolmabahce_stations,
        "дворец долмабахче": dolmabahce_stations,
        "долмабахче": dolmabahce_stations,
        "dolmabahce palast": dolmabahce_stations,
        "schloss dolmabahce": dolmabahce_stations,
        "palais de dolmabahce": dolmabahce_stations,
        "palais de dolmabahçe": dolmabahce_stations,
        "قصر دولما بهجة": dolmabahce_stations,
        "قصر دولمة باهجة": dolmabahce_stations,
        
        # Topkapi - All languages
        "topkapi": topkapi_stations,
        "topkapi palace": topkapi_stations,
        "topkapı": topkapi_stations,
        "topkapı palace": topkapi_stations,
        "topkapı sarayı": topkapi_stations,
        "topkapi sarayi": topkapi_stations,
        "topkapı müzesi": topkapi_stations,
        "topkapıdan": topkapi_stations,
        "topkapıya": topkapi_stations,
        "топкапы": topkapi_stations,
        "дворец топкапы": topkapi_stations,
        "topkapi palast": topkapi_stations,
        "schloss topkapi": topkapi_stations,
        "palais de topkapi": topkapi_stations,
        "palais de topkapı": topkapi_stations,
        "قصر توبكابي": topkapi_stations,
        "قصر طوب قابي": topkapi_stations,
    })
    
    # Mosques
    suleymaniye_stations = ["T1-Beyazıt-Kapalıçarşı"]
    aliases.update({
        "suleymaniye": suleymaniye_stations,
        "suleymaniye mosque": suleymaniye_stations,
        "süleymaniye": suleymaniye_stations,
        "süleymaniye camii": suleymaniye_stations,
        "suleymaniye camii": suleymaniye_stations,
        "мечеть сулеймание": suleymaniye_stations,
        "сулеймание": suleymaniye_stations,
        "süleymaniye moschee": suleymaniye_stations,
        "mosquée süleymaniye": suleymaniye_stations,
        "mosquée de soliman": suleymaniye_stations,
        "جامع السليمانية": suleymaniye_stations,
        "مسجد السليمانية": suleymaniye_stations,
    })
    
    # Museums
    gulhane_stations = ["T1-Gülhane"]
    aliases.update({
        "archaeological museum": gulhane_stations,
        "archaeology museum": gulhane_stations,
        "istanbul archaeological museums": gulhane_stations,
        "arkeoloji müzesi": gulhane_stations,
        "arkeoloji muzesi": gulhane_stations,
        "археологический музей": gulhane_stations,
        "archäologisches museum": gulhane_stations,
        "musée archéologique": gulhane_stations,
        "المتحف الأثري": gulhane_stations,
    })
    
    # Grand Bazaar
    grand_bazaar_stations = ["T1-Beyazıt-Kapalıçarşı"]
    aliases.update({
        "grand bazaar": grand_bazaar_stations,
        "the grand bazaar": grand_bazaar_stations,
        "covered bazaar": grand_bazaar_stations,
        "kapali carsi": grand_bazaar_stations,
        "kapalıçarşı": grand_bazaar_stations,
        "kapalı çarşı": grand_bazaar_stations,
        "kapalıçarşıdan": grand_bazaar_stations,
        "kapalıçarşıya": grand_bazaar_stations,
        "гранд базар": grand_bazaar_stations,
        "большой базар": grand_bazaar_stations,
        "крытый рынок": grand_bazaar_stations,
        "großer basar": grand_bazaar_stations,
        "gedeckter basar": grand_bazaar_stations,
        "grand bazar": grand_bazaar_stations,
        "le grand bazar": grand_bazaar_stations,
        "البازار الكبير": grand_bazaar_stations,
        "السوق المسقوف": grand_bazaar_stations,
    })
    
    # Parks
    aliases.update({
        "gulhane": gulhane_stations,
        "gülhane": gulhane_stations,
        "gulhane park": gulhane_stations,
        "gülhane parkı": gulhane_stations,
        "gulhane parki": gulhane_stations,
        "парк гюльхане": gulhane_stations,
        "гюльхане": gulhane_stations,
        "gülhane park": gulhane_stations,
        "parc gülhane": gulhane_stations,
        "حديقة جولهانه": gulhane_stations,
    })
    
    # ============================================================
    # GENERIC DESTINATIONS
    # ============================================================
    asian_side = ["MARMARAY-Üsküdar", "M5-Üsküdar", "FERRY-Üsküdar"]
    european_side = ["M2-Taksim"]
    city_center = ["M2-Taksim", "T1-Sultanahmet"]
    old_city = ["T1-Sultanahmet"]
    
    aliases.update({
        # Asian Side - All languages
        "asian side": asian_side,
        "asia": asian_side,
        "asia side": asian_side,
        "anatolian side": asian_side,
        "anadolu yakası": asian_side,
        "anadolu yakasi": asian_side,
        "asya yakası": asian_side,
        "asya yakasi": asian_side,
        "anadolu tarafı": asian_side,
        "азиатская сторона": asian_side,
        "азиатская часть": asian_side,
        "asiatische seite": asian_side,
        "côté asiatique": asian_side,
        "rive asiatique": asian_side,
        "الجانب الآسيوي": asian_side,
        "الطرف الآسيوي": asian_side,
        
        # European Side - All languages
        "european side": european_side,
        "europe": european_side,
        "europe side": european_side,
        "avrupa yakası": european_side,
        "avrupa yakasi": european_side,
        "avrupa tarafı": european_side,
        "европейская сторона": european_side,
        "европейская часть": european_side,
        "europäische seite": european_side,
        "côté européen": european_side,
        "rive européenne": european_side,
        "الجانب الأوروبي": european_side,
        "الطرف الأوروبي": european_side,
        
        # City Center - All languages
        "city center": city_center,
        "city centre": city_center,
        "center": city_center,
        "centre": city_center,
        "downtown": city_center,
        "şehir merkezi": city_center,
        "sehir merkezi": city_center,
        "merkez": city_center,
        "центр города": city_center,
        "центр": city_center,
        "stadtzentrum": city_center,
        "stadtmitte": city_center,
        "centre-ville": city_center,
        "وسط المدينة": city_center,
        "مركز المدينة": city_center,
        
        # Old City / Historic Peninsula
        "old city": old_city,
        "old town": old_city,
        "historic peninsula": old_city,
        "historical peninsula": old_city,
        "tarihi yarımada": old_city,
        "tarihi yarimada": old_city,
        "eski şehir": old_city,
        "eski sehir": old_city,
        "старый город": old_city,
        "исторический полуостров": old_city,
        "altstadt": old_city,
        "historische halbinsel": old_city,
        "vieille ville": old_city,
        "péninsule historique": old_city,
        "المدينة القديمة": old_city,
        "شبه الجزيرة التاريخية": old_city,
    })
    
    # Islands
    islands_stations = ["FERRY-Kadıköy", "FERRY-Eminönü"]
    aliases.update({
        "princes islands": islands_stations,
        "prince islands": islands_stations,
        "islands": islands_stations,
        "adalar": islands_stations,
        "prens adaları": islands_stations,
        "büyükada": islands_stations,
        "buyukada": islands_stations,
        "heybeliada": islands_stations,
        "kınalıada": islands_stations,
        "burgazada": islands_stations,
        "принцевы острова": islands_stations,
        "острова": islands_stations,
        "prinzeninseln": islands_stations,
        "îles des princes": islands_stations,
        "جزر الأمراء": islands_stations,
        "الجزر": islands_stations,
    })
    
    # ============================================================
    # ADDITIONAL NEIGHBORHOODS
    # ============================================================
    aliases.update({
        # Ortakoy
        "ortakoy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "ortaköy": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "ортакёй": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        
        # Balat/Fener
        "balat": ["T4-Fener", "T4-Balat"],
        "fener": ["T4-Fener"],
        "балат": ["T4-Fener", "T4-Balat"],
        "фенер": ["T4-Fener"],
        
        # Fatih
        "fatih": ["T4-Fatih", "T1-Aksaray"],
        "فاتح": ["T4-Fatih", "T1-Aksaray"],
        
        # Aksaray
        "aksaray": ["T1-Aksaray", "M1A-Aksaray"],
        "аксарай": ["T1-Aksaray", "M1A-Aksaray"],
        
        # Cihangir
        "cihangir": ["M2-Taksim"],
        "джихангир": ["M2-Taksim"],
        
        # Nisantasi
        "nisantasi": ["M2-Osmanbey"],
        "nişantaşı": ["M2-Osmanbey"],
        "нишанташи": ["M2-Osmanbey"],
        
        # Osmanbey
        "osmanbey": ["M2-Osmanbey"],
        "османбей": ["M2-Osmanbey"],
        
        # Bebek
        "bebek": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        "бебек": ["T4-Beşiktaş", "FERRY-Beşiktaş"],
        
        # Etiler
        "etiler": ["M2-4. Levent"],
        "этилер": ["M2-4. Levent"],
        
        # Maslak
        "maslak": ["M2-Hacıosman"],
        "маслак": ["M2-Hacıosman"],
        
        # Sariyer
        "sariyer": ["M2-Hacıosman"],
        "sarıyer": ["M2-Hacıosman"],
        "сарыер": ["M2-Hacıosman"],
        
        # Bostanci
        "bostanci": ["M4-Bostancı", "MARMARAY-Bostancı"],
        "bostancı": ["M4-Bostancı", "MARMARAY-Bostancı"],
        "бостанджи": ["M4-Bostancı", "MARMARAY-Bostancı"],
        
        # Pendik
        "pendik": ["MARMARAY-Pendik"],
        "пендик": ["MARMARAY-Pendik"],
        
        # Kabatas
        "kabatas": ["T1-Kabataş", "T4-Kabataş", "FERRY-Kabataş"],
        "kabataş": ["T1-Kabataş", "T4-Kabataş", "FERRY-Kabataş"],
        "kabataştan": ["T1-Kabataş", "T4-Kabataş", "FERRY-Kabataş"],
        "kabataşa": ["T1-Kabataş", "T4-Kabataş", "FERRY-Kabataş"],
        "кабаташ": ["T1-Kabataş", "T4-Kabataş", "FERRY-Kabataş"],
        
        # Yenikapi - Major transfer hub
        "yenikapi": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M2-Yenikapı"],
        "yenikapı": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M2-Yenikapı"],
        "yenikapıdan": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M2-Yenikapı"],
        "yenikapıya": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M2-Yenikapı"],
        "еникапы": ["MARMARAY-Yenikapı", "M1A-Yenikapı", "M2-Yenikapı"],
        
        # Zeytinburnu
        "zeytinburnu": ["T1-Zeytinburnu", "MARMARAY-Zeytinburnu"],
        "зейтинбурну": ["T1-Zeytinburnu", "MARMARAY-Zeytinburnu"],
        
        # Bakirkoy
        "bakirkoy": ["MARMARAY-Bakırköy"],
        "bakırköy": ["MARMARAY-Bakırköy"],
        "бакыркёй": ["MARMARAY-Bakırköy"],
        
        # Ayrilik Cesmesi
        "ayrilik cesmesi": ["M4-Ayrılık Çeşmesi", "MARMARAY-Ayrılık Çeşmesi"],
        "ayrılık çeşmesi": ["M4-Ayrılık Çeşmesi", "MARMARAY-Ayrılık Çeşmesi"],
        
        # Olimpiyat
        "olimpiyat": ["M3-Olimpiyat", "M9-Olimpiyat"],
        "olympic": ["M3-Olimpiyat", "M9-Olimpiyat"],
        "olympic stadium": ["M3-Olimpiyat", "M9-Olimpiyat"],
        "olimpiyat stadı": ["M3-Olimpiyat", "M9-Olimpiyat"],
        
        # Ikitelli
        "ikitelli": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
        "İkitelli": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
        "ikitelli sanayi": ["M3-İkitelli Sanayi", "M9-İkitelli Sanayi"],
    })
    
    return aliases


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

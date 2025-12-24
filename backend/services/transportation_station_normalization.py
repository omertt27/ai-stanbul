#!/usr/bin/env python3
"""
Canonical Station and Line ID Normalization System

Provides:
- Canonical station IDs (e.g., "M2-Taksim", "T1-Sultanahmet")
- Canonical line IDs with metadata
- Multilingual station names (Turkish, English, and common variants)
- Station coordinate lookup by canonical ID
- Line metadata (colors, types, full names)

This ensures consistent IDs across the entire system and supports
frontend features like station-specific links, multilingual UI, etc.

Author: AI Istanbul Team
Date: January 2025
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CanonicalStation:
    """A station with canonical IDs and multilingual names"""
    canonical_id: str  # e.g., "M2-Taksim"
    station_id: str    # Short ID, e.g., "taksim"
    line_id: str       # e.g., "M2"
    name_tr: str       # Turkish name (official)
    name_en: str       # English name
    name_variants: List[str]  # Common variants/aliases
    lat: float
    lon: float
    transfers: List[str]  # List of line IDs for transfers


@dataclass
class CanonicalLine:
    """A transit line with canonical metadata"""
    line_id: str       # e.g., "M2"
    name_tr: str       # Turkish name
    name_en: str       # English name
    line_type: str     # "metro", "tram", "funicular", "ferry", "marmaray", "bus"
    color: str         # Hex color for UI
    full_name_tr: str  # Full descriptive name in Turkish
    full_name_en: str  # Full descriptive name in English


class StationNormalizer:
    """
    Provides canonical station and line ID normalization.
    
    This is the single source of truth for:
    - Station IDs and names
    - Line IDs and metadata
    - Multilingual support
    """
    
    def __init__(self):
        """Initialize canonical station and line databases"""
        self.lines = self._build_line_metadata()
        self.stations = self._build_station_database()
        
        # Build stations by line (preserving physical order)
        self.stations_by_line = self._build_stations_by_line()
        
        # Build reverse lookups
        self.station_by_canonical_id = {s.canonical_id: s for s in self.stations}
        self.station_by_name_tr = {}
        self.station_by_name_en = {}
        
        for station in self.stations:
            # Map Turkish name
            self.station_by_name_tr[station.name_tr.lower()] = station
            # Map English name
            self.station_by_name_en[station.name_en.lower()] = station
            # Map variants
            for variant in station.name_variants:
                self.station_by_name_tr[variant.lower()] = station
        
        logger.info(f"✅ Station normalizer initialized: {len(self.stations)} stations, {len(self.lines)} lines")
    
    def _build_stations_by_line(self) -> Dict[str, List[str]]:
        """
        Build a mapping of line ID to ordered list of station IDs.
        
        This preserves the physical order of stations on each line,
        which is crucial for finding adjacent stations correctly.
        
        Returns:
            Dict mapping line_id to list of canonical station IDs in physical order
        """
        stations_by_line = {}
        
        for station in self.stations:
            line_id = station.line_id
            if line_id not in stations_by_line:
                stations_by_line[line_id] = []
            stations_by_line[line_id].append(station.canonical_id)
        
        return stations_by_line
    
    def get_stations_on_line_in_order(self, line_id: str) -> List[str]:
        """
        Get station IDs on a line in their physical order.
        
        This is critical for pathfinding - stations must be in physical order
        (not alphabetical) to find correct adjacent stations.
        
        Args:
            line_id: Line identifier (e.g., "M4", "MARMARAY")
            
        Returns:
            List of canonical station IDs in physical order along the line
        """
        return self.stations_by_line.get(line_id, [])
    
    def _build_line_metadata(self) -> Dict[str, CanonicalLine]:
        """Build canonical line metadata database"""
        lines = {}
        
        # Metro lines
        lines["M1A"] = CanonicalLine("M1A", "M1A", "M1A Metro", "metro", "#E31C23", "Yenikapı-Atatürk Havalimanı", "Yenikapı-Atatürk Airport")
        lines["M1B"] = CanonicalLine("M1B", "M1B", "M1B Metro", "metro", "#B4277E", "Yenikapı-Kirazlı", "Yenikapı-Kirazlı")
        lines["M2"] = CanonicalLine("M2", "M2", "M2 Metro", "metro", "#00A650", "Yenikapı-Hacıosman", "Yenikapı-Hacıosman")
        lines["M3"] = CanonicalLine("M3", "M3", "M3 Metro", "metro", "#EF4136", "Kirazlı-Başakşehir", "Kirazlı-Başakşehir")
        lines["M4"] = CanonicalLine("M4", "M4", "M4 Metro", "metro", "#FF6E1E", "Kadıköy-Tavşantepe", "Kadıköy-Tavşantepe")
        lines["M5"] = CanonicalLine("M5", "M5", "M5 Metro", "metro", "#8E3994", "Üsküdar-Çekmeköy", "Üsküdar-Çekmeköy")
        lines["M6"] = CanonicalLine("M6", "M6", "M6 Metro", "metro", "#D3A029", "Levent-Boğaziçi Üniversitesi", "Levent-Boğaziçi University")
        lines["M7"] = CanonicalLine("M7", "M7", "M7 Metro", "metro", "#E91E8C", "Mecidiyeköy-Mahmutbey", "Mecidiyeköy-Mahmutbey")
        lines["M9"] = CanonicalLine("M9", "M9", "M9 Metro", "metro", "#8E3994", "Olimpiyat-İkitelli Sanayi", "Olimpiyat-İkitelli Industrial")
        lines["M11"] = CanonicalLine("M11", "M11", "M11 Metro", "metro", "#00A651", "Gayrettepe-İstanbul Havalimanı", "Gayrettepe-Istanbul Airport")
        
        # Tram lines
        lines["T1"] = CanonicalLine("T1", "T1", "T1 Tram", "tram", "#ED1C24", "Kabataş-Bağcılar", "Kabataş-Bağcılar")
        lines["T4"] = CanonicalLine("T4", "T4", "T4 Tram", "tram", "#F7941D", "Topkapı-Mescid-i Selam", "Topkapı-Mescid-i Selam")
        lines["T5"] = CanonicalLine("T5", "T5", "T5 Tram", "tram", "#00AEEF", "Cibali-Alibeyköy", "Cibali-Alibeyköy")
        
        # Funiculars
        lines["F1"] = CanonicalLine("F1", "F1", "F1 Füniküler", "funicular", "#EE2E24", "Kabataş-Taksim", "Kabataş-Taksim")
        lines["F2"] = CanonicalLine("F2", "F2", "F2 Tünel", "funicular", "#00A651", "Karaköy-Tünel", "Karaköy-Tünel")
        
        # Ferry (Şehir Hatları)
        lines["FERRY"] = CanonicalLine("FERRY", "Vapur", "Ferry", "ferry", "#009FE3", "Şehir Hatları", "City Lines Ferry")
        
        # Marmaray
        lines["MARMARAY"] = CanonicalLine("MARMARAY", "Marmaray", "Marmaray", "marmaray", "#E4032E", "Gebze-Halkalı", "Gebze-Halkalı")
        
        return lines
    
    def _build_station_database(self) -> List[CanonicalStation]:
        """
        Build comprehensive station database with canonical IDs and multilingual names.
        
        This is the single source of truth for all station data.
        """
        stations = []
        
        # ==========================================
        # M1A LINE (Yenikapı - Atatürk Havalimanı/Airport) - 18 stations
        # ==========================================
        m1a_stations = [
            ("Yenikapı", "Yenikapi", "yenikapi", ["yenikapi", "yeni kapi"], 41.0042, 28.9519, ["M1B", "M2", "MARMARAY"]),
            ("Aksaray", "Aksaray", "aksaray", ["aksaray"], 41.0164, 28.9450, ["T1", "M1B"]),
            ("Emniyet-Fatih", "Emniyet-Fatih", "emniyet_fatih", ["emniyet", "fatih"], 41.0192, 28.9358, []),
            ("Ulubatlı", "Ulubatli", "ulubatli", ["ulubatli"], 41.0231, 28.9267, []),
            ("Bayrampaşa-Maltepe", "Bayrampasa-Maltepe", "bayrampasa", ["bayrampasa"], 41.0344, 28.9122, []),
            ("Sağmalcılar", "Sagmalcilar", "sagmalcilar", ["sagmalcilar"], 41.0472, 28.8936, []),
            ("Kocatepe", "Kocatepe", "kocatepe", ["kocatepe"], 41.0586, 28.8789, []),
            ("Otogar", "Otogar", "otogar", ["otogar", "bus terminal"], 41.0736, 28.8781, []),
            ("Terazidere", "Terazidere", "terazidere_m1a", ["terazidere"], 41.0833, 28.8567, []),
            ("Davutpaşa-YTÜ", "Davutpasa-YTU", "davutpasa_m1a", ["davutpasa", "ytu"], 41.0614, 28.8331, []),
            ("Merter", "Merter", "merter_m1a", ["merter"], 41.0114, 28.8736, []),
            ("Zeytinburnu", "Zeytinburnu", "zeytinburnu_m1a", ["zeytinburnu"], 41.0072, 28.9053, ["T1", "MARMARAY", "M1B"]),
            ("Bakırköy-İncirli", "Bakirkoy-Incirli", "bakirkoy_m1a", ["bakirkoy", "incirli"], 40.9903, 28.8667, []),
            ("Bahçelievler", "Bahcelievler", "bahcelievler_m1a", ["bahcelievler"], 40.9972, 28.8517, []),
            ("Şirinevler", "Sirinevler", "sirinevler", ["sirinevler"], 41.0167, 28.8364, []),
            ("DTM-İstanbul Fuar Merkezi", "DTM-Istanbul Fair Center", "dtm", ["dtm", "fuar merkezi"], 40.9833, 28.8167, []),
            ("Yenibosna", "Yenibosna", "yenibosna", ["yenibosna"], 40.9756, 28.8278, []),
            ("Atatürk Havalimanı", "Ataturk Airport", "ataturk_airport", ["ataturk airport", "atatürk airport", "airport", "havalimani", "havalimanı"], 40.9767, 28.8092, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m1a_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M1A-{name_tr}",
                station_id=short_id,
                line_id="M1A",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # M1B LINE (Yenikapı - Kirazlı) - 13 stations
        # ==========================================
        m1b_stations = [
            ("Yenikapı", "Yenikapi", "yenikapi_m1b", ["yenikapi", "yeni kapi"], 41.0042, 28.9519, ["M1A", "M2", "MARMARAY"]),
            ("Aksaray", "Aksaray", "aksaray_m1b", ["aksaray"], 41.0164, 28.9450, ["T1", "M1A"]),
            ("Emniyet-Fatih", "Emniyet-Fatih", "emniyet_fatih_m1b", ["emniyet", "fatih"], 41.0192, 28.9358, []),
            ("Ulubatlı", "Ulubatli", "ulubatli_m1b", ["ulubatli"], 41.0231, 28.9267, []),
            ("Bayrampaşa-Maltepe", "Bayrampasa-Maltepe", "bayrampasa_m1b", ["bayrampasa"], 41.0344, 28.9122, []),
            ("Sağmalcılar", "Sagmalcilar", "sagmalcilar_m1b", ["sagmalcilar"], 41.0472, 28.8936, []),
            ("Kocatepe", "Kocatepe", "kocatepe_m1b", ["kocatepe"], 41.0586, 28.8789, []),
            ("Otogar", "Otogar", "otogar_m1b", ["otogar", "bus terminal"], 41.0736, 28.8781, []),
            ("Esenler", "Esenler", "esenler_m1b", ["esenler"], 41.0806, 28.8719, []),
            ("Terazidere", "Terazidere", "terazidere_m1b", ["terazidere"], 41.0833, 28.8567, []),
            ("Davutpaşa-YTÜ", "Davutpasa-YTU", "davutpasa_m1b", ["davutpasa", "ytu"], 41.0614, 28.8331, []),
            ("Merter", "Merter", "merter_m1b", ["merter"], 41.0114, 28.8736, []),
            ("Kirazlı", "Kirazli", "kirazli", ["kirazli"], 41.0194, 28.8194, ["M3"]),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m1b_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M1B-{name_tr}",
                station_id=short_id,
                line_id="M1B",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # M6 LINE (Levent - Boğaziçi Üniversitesi) - 4 stations
        # ==========================================
        m6_stations = [
            ("Levent", "Levent", "levent_m6", ["levent"], 41.0789, 29.0114, ["M2"]),
            ("Nispetiye", "Nispetiye", "nispetiye", ["nispetiye"], 41.0858, 29.0119, []),
            ("Etiler", "Etiler", "etiler", ["etiler"], 41.0906, 29.0156, []),
            ("Boğaziçi Üniversitesi", "Bogazici University", "bogazici", ["bogazici", "bogazici university"], 41.0858, 29.0467, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m6_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M6-{name_tr}",
                station_id=short_id,
                line_id="M6",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # M7 LINE (Mecidiyeköy - Mahmutbey) - 17 stations
        # ==========================================
        m7_stations = [
            ("Mecidiyeköy", "Mecidiyekoy", "mecidiyekoy", ["mecidiyekoy", "sisli"], 41.0644, 28.9989, ["M2"]),
            ("Çağlayan", "Caglayan", "caglayan", ["caglayan"], 41.0628, 28.9831, []),
            ("Kağıthane", "Kagithane", "kagithane", ["kagithane"], 41.0786, 28.9769, []),
            ("Nurtepe", "Nurtepe", "nurtepe", ["nurtepe"], 41.0914, 28.9703, []),
            ("Alibeyköy", "Alibeykoy", "alibeykoy", ["alibeykoy"], 41.1025, 28.9536, []),
            ("Çırçır", "Circir", "circir", ["circir"], 41.1089, 28.9392, []),
            ("Veysel Karani-Akşemsettin", "Veysel Karani-Aksemseddin", "veysel_karani", ["veysel karani", "aksemseddin"], 41.1131, 28.9258, []),
            ("Yenimahalle", "Yenimahalle", "yenimahalle_m7", ["yenimahalle"], 41.1153, 28.9092, []),
            ("Karadeniz Mahallesi", "Karadeniz Mahallesi", "karadeniz", ["karadeniz mahallesi"], 41.1089, 28.8886, []),
            ("Tekstilkent", "Tekstilkent", "tekstilkent", ["tekstilkent"], 41.1006, 28.8678, []),
            ("İstoç", "Istoc", "istoc", ["istoc"], 41.0956, 28.8589, []),
            ("Göztepe Mahallesi", "Goztepe Mahallesi", "goztepe_m7", ["goztepe mahallesi"], 41.0919, 28.8547, []),
            ("Yenibosna", "Yenibosna", "yenibosna_m7", ["yenibosna"], 41.0875, 28.8458, []),
            ("Soğanlı", "Soganli", "soganli", ["soganli"], 41.0825, 28.8369, []),
            ("Kazım Karabekir", "Kazim Karabekir", "kazim_karabekir", ["kazim karabekir"], 41.0792, 28.8286, []),
            ("Yıldırım", "Yildirim", "yildirim", ["yildirim"], 41.0861, 28.8203, []),
            ("Mahmutbey", "Mahmutbey", "mahmutbey", ["mahmutbey"], 41.0847, 28.8408, ["M3"]),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m7_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M7-{name_tr}",
                station_id=short_id,
                line_id="M7",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # M9 LINE (Olimpiyat - İkitelli Sanayi) - 2 stations
        # ==========================================
        # Note: M9 is a short line opened in October 2023
        # Serves the İkitelli industrial zone
        # Both stations are shared with M3 line
        m9_stations = [
            ("Olimpiyat", "Olimpiyat", "olimpiyat", ["olimpiyat"], 41.0744, 28.7644, ["M3"]),
            ("İkitelli Sanayi", "Ikitelli Sanayi", "ikitelli_sanayi", ["ikitelli", "ikitelli sanayi"], 41.0656, 28.7828, ["M3"]),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m9_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M9-{name_tr}",
                station_id=short_id,
                line_id="M9",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # M11 LINE (Gayrettepe - Istanbul Airport) - 9 stations
        # Opened: 2023, connects city center to new Istanbul Airport
        # ==========================================
        m11_stations = [
            ("Gayrettepe", "Gayrettepe", "gayrettepe_m11", ["gayret tepe", "gayret"], 41.0683, 29.0139, ["M2"]),
            ("Kağıthane", "Kagithane", "kagithane_m11", ["kagithane"], 41.0789, 28.9847, []),
            ("Kemerburgaz", "Kemerburgaz", "kemerburgaz", ["kemer burgaz", "kemer"], 41.1469, 28.8142, []),
            ("Göktürk", "Gokturk", "gokturk", ["gok turk"], 41.1711, 28.8500, []),
            ("İhsaniye", "Ihsaniye", "ihsaniye", ["ihsan"], 41.2167, 28.7833, []),
            ("Havalimanı Mahallesi", "Airport District", "airport_district", ["havalimani mahallesi"], 41.2528, 28.7611, []),
            ("Havalimanı İstasyonu-1", "Airport Station-1", "airport_station_1", ["havalimani istasyonu 1"], 41.2639, 28.7444, []),
            ("Havalimanı İstasyonu-2", "Airport Station-2", "airport_station_2", ["havalimani istasyonu 2"], 41.2683, 28.7417, []),
            ("İstanbul Havalimanı", "Istanbul Airport", "istanbul_airport", ["havalimani", "ist airport", "new airport", "yeni havalimani"], 41.2750, 28.7519, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m11_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M11-{name_tr}",
                station_id=short_id,
                line_id="M11",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # M2 LINE (Yenikapı - Hacıosman) - 16 stations
        # ==========================================
        m2_stations = [
            ("Yenikapı", "Yenikapi", "yenikapi_m2", ["yenikapi", "yeni kapi"], 41.0042, 28.9519, ["M1A", "M1B", "MARMARAY"]),
            ("Vezneciler", "Vezneciler", "vezneciler", ["vezneciler"], 41.0133, 28.9539, ["T1"]),
            ("Haliç", "Halic", "halic", ["halic", "golden horn"], 41.0231, 28.9536, []),
            ("Şişhane", "Sishane", "sishane", ["sishane", "sisane"], 41.0256, 28.9750, ["F2"]),
            ("Taksim", "Taksim", "taksim", ["taksim square", "taksim meydani"], 41.0369, 28.9850, ["F1"]),
            ("Osmanbey", "Osmanbey", "osmanbey", ["osman bey"], 41.0483, 28.9867, []),
            ("Şişli-Mecidiyeköy", "Sisli-Mecidiyekoy", "sisli_mecidiyekoy", ["sisli", "mecidiyekoy", "sisli mecidiyekoy"], 41.0644, 28.9989, ["M7"]),
            ("Gayrettepe", "Gayrettepe", "gayrettepe", ["gayret tepe"], 41.0683, 29.0139, ["M11"]),
            ("Levent", "Levent", "levent_m2", ["1 levent"], 41.0789, 29.0114, ["M6"]),
            ("4. Levent", "4. Levent", "4_levent", ["4 levent", "dorduncu levent"], 41.0861, 29.0089, []),
            ("Sanayi Mahallesi", "Sanayi Mahallesi", "sanayi_mahallesi", ["sanayi"], 41.0994, 29.0089, []),
            ("İTÜ-Ayazağa", "ITU-Ayazaga", "itu_ayazaga", ["itu", "ayazaga"], 41.1064, 29.0194, []),
            ("Atatürk Oto Sanayi", "Ataturk Oto Sanayi", "ataturk_oto_sanayi", ["ataturk oto sanayi"], 41.1158, 29.0286, []),
            ("Seyrantepe", "Seyrantepe", "seyrantepe", ["seyrantepe"], 41.1208, 29.0319, []),
            ("Darüşşafaka", "Darussafaka", "darussafaka", ["darussafaka"], 41.1258, 29.0350, []),
            ("Hacıosman", "Haciosman", "haciosman", ["haciost", "haciosman"], 41.1372, 29.0397, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m2_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M2-{name_tr}",
                station_id=short_id,
                line_id="M2",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # M3 LINE (Kirazlı - Kayaşehir) - 19 stations
        # ==========================================
        m3_stations = [
            ("Kirazlı", "Kirazli", "kirazli_m3", ["kirazli"], 41.0194, 28.8194, ["M1B"]),
            ("Bağcılar Meydan", "Bagcilar Meydan", "bagcilar_meydan", ["bagcilar meydan"], 41.0333, 28.8167, []),
            ("Mahmutbey", "Mahmutbey", "mahmutbey_m3", ["mahmutbey"], 41.0417, 28.8056, ["M7"]),
            ("Yeni Mahalle", "Yeni Mahalle", "yeni_mahalle_m3", ["yeni mahalle"], 41.0486, 28.7944, []),
            ("Mimar Sinan", "Mimar Sinan", "mimar_sinan", ["mimar sinan"], 41.0528, 28.7889, []),
            ("İkitelli Sanayi", "Ikitelli Sanayi", "ikitelli_sanayi", ["ikitelli", "ikitelli sanayi"], 41.0656, 28.7828, ["M9"]),
            ("Olimpiyat", "Olimpiyat", "olimpiyat", ["olimpiyat"], 41.0744, 28.7644, ["M9"]),
            ("Ziya Gökalp Mahallesi", "Ziya Gokalp Mahallesi", "ziya_gokalp", ["ziya gokalp"], 41.0792, 28.7556, []),
            ("Başakşehir-Metrokent", "Basaksehir-Metrokent", "basaksehir", ["basaksehir", "metrokent"], 41.0833, 28.7833, []),
            ("Siteler", "Siteler", "siteler", ["siteler"], 41.0917, 28.7708, []),
            ("Turgut Özal", "Turgut Ozal", "turgut_ozal", ["turgut ozal"], 41.0986, 28.7597, []),
            ("Şehir Hastanesi 1", "City Hospital 1", "sehir_hastanesi_1", ["sehir hastanesi 1", "city hospital 1"], 41.1056, 28.7500, []),
            ("Şehir Hastanesi 2", "City Hospital 2", "sehir_hastanesi_2", ["sehir hastanesi 2", "city hospital 2"], 41.1028, 28.7417, []),
            ("Başak Konutları", "Basak Konutlari", "basak_konutlari", ["basak konutlari"], 41.0825, 28.7500, []),
            ("Kayabaşı", "Kayabasi", "kayabasi", ["kayabasi"], 41.0667, 28.7333, []),
            ("Göktürk", "Gokturk", "gokturk", ["gokturk"], 41.0528, 28.7250, []),
            ("Kartal-1", "Kartal-1", "kartal_1", ["kartal 1"], 41.0417, 28.7194, []),
            ("Kartal-2", "Kartal-2", "kartal_2", ["kartal 2"], 41.0361, 28.7167, []),
            ("Kayaşehir Merkez", "Kayasehir Merkez", "kayasehir", ["kayasehir"], 41.0319, 28.7197, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m3_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M3-{name_tr}",
                station_id=short_id,
                line_id="M3",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # M4 LINE (Kadıköy - Tavşantepe) - 23 stations
        # ==========================================
        m4_stations = [
            ("Kadıköy", "Kadikoy", "kadikoy", ["kadikoy"], 40.9903, 29.0275, []),
            ("Ayrılık Çeşmesi", "Ayrilik Cesmesi", "ayrilik_cesmesi", ["ayrilik", "ayrilik cesmesi"], 40.9797, 29.0510, ["MARMARAY"]),
            ("Acıbadem", "Acibadem", "acibadem", ["acibadem"], 41.0028, 29.0194, []),
            ("Ünalan", "Unalan", "unalan", ["unalan"], 41.0092, 29.0247, []),
            ("Göztepe", "Goztepe", "goztepe_m4", ["goztepe"], 41.0164, 29.0381, []),
            ("Yenisahra", "Yenisahra", "yenisahra", ["yeni sahra"], 41.0194, 29.0497, []),
            ("Kozyatağı", "Kozyatagi", "kozyatagi", ["kozyatagi"], 41.0242, 29.0625, []),
            ("Bostancı", "Bostanci", "bostanci_m4", ["bostanci"], 40.9647, 29.0875, ["MARMARAY"]),
            ("Küçükyalı", "Kucukyali", "kucukyali_m4", ["kucukyali"], 40.9486, 29.1050, []),
            ("Maltepe", "Maltepe", "maltepe_m4", ["maltepe"], 40.9367, 29.1306, []),
            ("Huzurevi", "Huzurevi", "huzurevi", ["huzurevi"], 40.9236, 29.1483, []),
            ("Gülsuyu", "Gulsuyu", "gulsuyu", ["gulsuyu"], 40.9125, 29.1578, []),
            ("Esenkent", "Esenkent", "esenkent", ["esenkent"], 40.9069, 29.1653, []),
            ("Hastane-Adliye", "Hastane-Adliye", "hastane_adliye", ["hastane adliye"], 40.9014, 29.1728, []),
            ("Soğanlık", "Soganlik", "soganlik", ["soganlik"], 40.8958, 29.1806, []),
            ("Kartal", "Kartal", "kartal_m4", ["kartal"], 40.9000, 29.1833, ["MARMARAY"]),
            ("Yakacık-Adnan Kahveci", "Yakacik-Adnan Kahveci", "yakacik", ["yakacik", "adnan kahveci"], 40.8917, 29.1944, []),
            ("Pendik", "Pendik", "pendik_m4", ["pendik"], 40.8796, 29.2326, ["MARMARAY"]),
            ("Yunus", "Yunus", "yunus", ["yunus"], 40.8731, 29.2519, []),
            ("Kaynarca", "Kaynarca", "kaynarca", ["kaynarca"], 40.8683, 29.2706, []),
            ("Fevzi Çakmak", "Fevzi Cakmak", "fevzi_cakmak", ["fevzi cakmak"], 40.8639, 29.2883, []),
            ("Güzelyalı", "Guzelyali", "guzelyali_m4", ["guzelyali"], 40.8600, 29.3019, []),
            ("Tavşantepe", "Tavsantepe", "tavsantepe", ["tavsantepe"], 40.8644, 29.3136, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m4_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M4-{name_tr}",
                station_id=short_id,
                line_id="M4",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # M5 LINE (Üsküdar - Çekmeköy) - 20 stations
        # ==========================================
        m5_stations = [
            ("Üsküdar", "Uskudar", "uskudar_m5", ["uskudar"], 41.0256, 29.0100, ["MARMARAY"]),
            ("Fıstıkağacı", "Fistikagaci", "fistikagaci", ["fistikagaci"], 41.0344, 29.0192, []),
            ("Bağlarbaşı", "Baglarbasi", "baglarbasi", ["baglarbasi"], 41.0417, 29.0250, []),
            ("Altunizade", "Altunizade", "altunizade", ["altunizade"], 41.0472, 29.0322, []),
            ("Kısıklı", "Kisikli", "kisikli", ["kisikli"], 41.0525, 29.0408, []),
            ("Bulgurlu", "Bulgurlu", "bulgurlu", ["bulgurlu"], 41.0578, 29.0500, []),
            ("Ümraniye", "Umraniye", "umraniye", ["umraniye"], 41.0650, 29.0583, []),
            ("Çarşı", "Carsi", "carsi_m5", ["carsi"], 41.0717, 29.0667, []),
            ("Yamanevler", "Yamanevler", "yamanevler", ["yamanevler"], 41.0789, 29.0756, []),
            ("Çakmak", "Cakmak", "cakmak", ["cakmak"], 41.0856, 29.0831, []),
            ("Ihlamurkuyu", "Ihlamurkuyu", "ihlamurkuyu", ["ihlamurkuyu"], 41.0919, 29.0903, []),
            ("Altınşehir", "Altinsehir", "altinsehir", ["altinsehir"], 41.0986, 29.0972, []),
            ("Göztepe", "Goztepe", "goztepe_m5", ["goztepe"], 41.1050, 29.1044, []),
            ("Küçükbakkalköy", "Kucukbakkalkoy", "kucukbakkalkoy", ["kucukbakkalkoy"], 41.1111, 29.1114, []),
            ("Soğanlık", "Soganlik", "soganlik", ["soganlik"], 41.1172, 29.1186, []),
            ("Çamçeşme", "Camcesme", "camcesme", ["camcesme"], 41.1231, 29.1256, []),
            ("Hastane", "Hastane", "hastane_m5", ["hastane"], 41.1289, 29.1328, []),
            ("Sancaktepe", "Sancaktepe", "sancaktepe", ["sancaktepe"], 41.1347, 29.1400, []),
            ("Sarıgazi", "Sarigazi", "sarigazi", ["sarigazi"], 41.1403, 29.1472, []),
            ("Çekmeköy", "Cekmekoy", "cekmekoy", ["cekmekoy"], 41.1458, 29.1544, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in m5_stations:
            stations.append(CanonicalStation(
                canonical_id=f"M5-{name_tr}",
                station_id=short_id,
                line_id="M5",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # T1 TRAM LINE (Kabataş - Bağcılar) - 31 stations
        # ==========================================
        t1_stations = [
            ("Kabataş", "Kabatas", "kabatas_t1", ["kabatas"], 41.0383, 29.0069, ["F1"]),
            ("Fındıklı", "Findikli", "findikli", ["findikli"], 41.0344, 29.0019, []),
            ("Tophane", "Tophane", "tophane", ["tophane"], 41.0275, 28.9869, []),
            ("Karaköy", "Karakoy", "karakoy_t1", ["karakoy"], 41.0242, 28.9778, ["F2"]),
            ("Eminönü", "Eminonu", "eminonu", ["eminonu"], 41.0178, 28.9708, []),
            ("Sirkeci", "Sirkeci", "sirkeci_t1", ["sirkeci"], 41.0169, 28.9769, ["MARMARAY"]),
            ("Gülhane", "Gulhane", "gulhane", ["gulhane"], 41.0133, 28.9806, []),
            ("Sultanahmet", "Sultanahmet", "sultanahmet", ["sultan ahmet", "blue mosque"], 41.0058, 28.9769, []),
            ("Beyazıt-Kapalıçarşı", "Beyazit-Kapalicarsi", "beyazit", ["beyazit", "kapalicarsi", "grand bazaar"], 41.0103, 28.9647, []),
            ("Çemberlitaş", "Cemberlitas", "cemberlitas", ["cemberlitas"], 41.0092, 28.9686, []),
            ("Laleli-Üniversite", "Laleli-Universite", "laleli", ["laleli", "universite"], 41.0111, 28.9539, []),
            ("Aksaray", "Aksaray", "aksaray_t1", ["aksaray"], 41.0164, 28.9450, ["M1A", "M1B"]),
            ("Yusufpaşa", "Yusufpasa", "yusufpasa", ["yusufpasa"], 41.0203, 28.9403, []),
            ("Haseki", "Haseki", "haseki", ["haseki"], 41.0217, 28.9333, []),
            ("Pazartekke", "Pazartekke", "pazartekke", ["pazartekke"], 41.0228, 28.9275, []),
            ("Fındıkzade", "Findikzade", "findikzade", ["findikzade"], 41.0175, 28.9219, []),
            ("Çapa-Şehremini", "Capa-Sehremini", "capa", ["capa", "sehremini"], 41.0150, 28.9158, []),
            ("Topkapı", "Topkapi", "topkapi", ["topkapi"], 41.0131, 28.9092, []),
            ("Sütlüce", "Sutluce", "sutluce", ["sutluce"], 41.0114, 28.9025, []),
            ("Edirnekapı", "Edirnekapi", "edirnekapi", ["edirnekapi"], 41.0086, 28.8964, []),
            ("Ulubatlı", "Ulubatli", "ulubatli_t1", ["ulubatli"], 41.0061, 28.8908, []),
            ("Cevizlibağ", "Cevizlibag", "cevizlibag", ["cevizlibag"], 41.0056, 28.8850, []),
            ("Zeytinburnu", "Zeytinburnu", "zeytinburnu_t1", ["zeytinburnu"], 41.0072, 28.9053, ["M1A", "M1B", "MARMARAY"]),
            ("Merkezefendi", "Merkezefendi", "merkezefendi", ["merkezefendi"], 41.0233, 28.8819, []),
            ("Mithatpaşa", "Mithatpasa", "mithatpasa", ["mithatpasa"], 41.0258, 28.8747, []),
            ("Çırpıcı", "Cirpici", "cirpici", ["cirpici"], 41.0294, 28.8683, []),
            ("Sağmalcılar", "Sagmalcilar", "sagmalcilar_t1", ["sagmalcilar"], 41.0331, 28.8619, []),
            ("Soğanlı", "Soganli", "soganli_t1", ["soganli"], 41.0367, 28.8556, []),
            ("Yavuz Selim", "Yavuz Selim", "yavuz_selim", ["yavuz selim"], 41.0394, 28.8506, []),
            ("Bağcılar İdealtepe", "Bagcilar Idealtepe", "bagcilar_idealtepe", ["bagcilar idealtepe"], 41.0417, 28.8464, []),
            ("Bağcılar", "Bagcilar", "bagcilar_t1", ["bagcilar"], 41.0394, 28.8506, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in t1_stations:
            stations.append(CanonicalStation(
                canonical_id=f"T1-{name_tr}",
                station_id=short_id,
                line_id="T1",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # T4 TRAM LINE (Topkapı - Mescid-i Selam) - 22 stations
        # Historical peninsula tram line on European side
        # ==========================================
        t4_stations = [
            ("Topkapı", "Topkapi", "topkapi_t4", ["topkapi"], 41.0144, 28.9194, ["M1A"]),
            ("Pazartekke", "Pazartekke", "pazartekke", ["pazartekke"], 41.0131, 28.9289, []),
            ("Çapa", "Capa", "capa", ["capa"], 41.0139, 28.9386, []),
            ("Fındıkzade", "Findikzade", "findikzade", ["findikzade"], 41.0156, 28.9447, []),
            ("Haseki", "Haseki", "haseki", ["haseki"], 41.0142, 28.9517, []),
            ("Yusufpaşa", "Yusufpasa", "yusufpasa", ["yusufpasa"], 41.0147, 28.9597, []),
            ("Aksaray", "Aksaray", "aksaray_t4", ["aksaray"], 41.0161, 28.9508, []),
            ("Laleli", "Laleli", "laleli", ["laleli"], 41.0139, 28.9594, []),
            ("Beyazıt", "Beyazit", "beyazit", ["beyazit"], 41.0133, 28.9664, []),
            ("Çemberlitaş", "Cemberlitas", "cemberlitas", ["cemberlitas"], 41.0103, 28.9706, []),
            ("Sultanahmet", "Sultanahmet", "sultanahmet_t4", ["sultanahmet"], 41.0058, 28.9781, ["T1"]),
            ("Gülhane", "Gulhane", "gulhane_t4", ["gulhane"], 41.0131, 28.9814, ["T1"]),
            ("Sirkeci", "Sirkeci", "sirkeci_t4", ["sirkeci"], 41.0175, 28.9839, ["T1", "MARMARAY"]),
            ("Eminönü", "Eminonu", "eminonu_t4", ["eminonu"], 41.0178, 28.9706, ["T1"]),
            ("Karaköy", "Karakoy", "karakoy_t4", ["karakoy"], 41.0236, 28.9753, ["T1", "F2"]),
            ("Tophane", "Tophane", "tophane_t4", ["tophane"], 41.0264, 28.9828, ["T1"]),
            ("Fındıklı", "Findikli", "findikli", ["findikli"], 41.0292, 28.9883, []),
            ("Kabataş", "Kabatas", "kabatas_t4", ["kabatas"], 41.0356, 28.9914, ["T1", "F1"]),
            ("Dolmabahçe", "Dolmabahce", "dolmabahce", ["dolmabahce"], 41.0408, 28.9958, []),
            ("Beşiktaş", "Besiktas", "besiktas_t4", ["besiktas"], 41.0425, 29.0019, []),
            ("Barbaros", "Barbaros", "barbaros", ["barbaros"], 41.0467, 29.0081, []),
            ("Mescid-i Selam", "Mescid-i Selam", "mescid_selam", ["mescid selam", "mescidi selam"], 41.0528, 29.0117, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in t4_stations:
            stations.append(CanonicalStation(
                canonical_id=f"T4-{name_tr}",
                station_id=short_id,
                line_id="T4",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # T5 TRAM LINE (Cibali - Alibeyköy) - 11 stations
        # Northern European side tram line
        # ==========================================
        t5_stations = [
            ("Cibali", "Cibali", "cibali", ["cibali"], 41.0325, 28.9508, []),
            ("Fener", "Fener", "fener", ["fener"], 41.0364, 28.9481, []),
            ("Balat", "Balat", "balat", ["balat"], 41.0419, 28.9469, []),
            ("Ayvansaray", "Ayvansaray", "ayvansaray", ["ayvansaray"], 41.0467, 28.9453, []),
            ("Eyüpsultan", "Eyupsultan", "eyupsultan", ["eyup", "eyupsultan"], 41.0508, 28.9397, []),
            ("Silahtar", "Silahtar", "silahtar", ["silahtar"], 41.0558, 28.9392, []),
            ("Defterdar", "Defterdar", "defterdar", ["defterdar"], 41.0603, 28.9397, []),
            ("Halıcıoğlu", "Halicioglu", "halicioglu", ["halicioglu"], 41.0653, 28.9400, []),
            ("Sütlüce", "Sutluce", "sutluce", ["sutluce"], 41.0697, 28.9414, []),
            ("Piripaşa", "Piripasa", "piripasa", ["piripasa"], 41.0739, 28.9436, []),
            ("Alibeyköy", "Alibeykoy", "alibeykoy_t5", ["alibeykoy"], 41.0778, 28.9458, ["M7"]),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in t5_stations:
            stations.append(CanonicalStation(
                canonical_id=f"T5-{name_tr}",
                station_id=short_id,
                line_id="T5",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # MARMARAY (Gebze - Halkalı) - 43 stations
        # The undersea commuter rail tunnel connecting Europe and Asia
        # ==========================================
        marmaray_stations = [
            # ASIAN SIDE (East to West) - Gebze to Üsküdar
            ("Gebze", "Gebze", "gebze_marmaray", ["gebze"], 40.8021, 29.4309, []),
            ("Darica", "Darica", "darica", ["darica"], 40.8189, 29.3792, []),
            ("Osmangazi", "Osmangazi", "osmangazi_marmaray", ["osmangazi"], 40.8356, 29.3264, []),
            ("Çayırova", "Cayirova", "cayirova", ["cayirova"], 40.8472, 29.2858, []),
            ("Tersane", "Tersane", "tersane", ["tersane"], 40.8614, 29.2558, []),
            ("İçmeler", "Icmeler", "icmeler", ["icmeler"], 40.8711, 29.2386, []),
            ("Pendik", "Pendik", "pendik_marmaray", ["pendik"], 40.8796, 29.2326, ["M4"]),
            ("Güzelyalı", "Guzelyali", "guzelyali_marmaray", ["guzelyali"], 40.8919, 29.1894, []),
            ("Aydıntepe", "Aydintepe", "aydintepe", ["aydintepe"], 40.9039, 29.1689, []),
            ("Maltepe", "Maltepe", "maltepe_marmaray", ["maltepe"], 40.9139, 29.1458, []),
            ("Cevizli", "Cevizli", "cevizli", ["cevizli"], 40.9211, 29.1189, []),
            ("Atalar", "Atalar", "atalar", ["atalar"], 40.9325, 29.1042, []),
            ("Kartal", "Kartal", "kartal_marmaray", ["kartal"], 40.9000, 29.1833, ["M4"]),
            ("Küçükyalı", "Kucukyali", "kucukyali_marmaray", ["kucukyali"], 40.9486, 29.1050, []),
            ("Bostancı", "Bostanci", "bostanci_marmaray", ["bostanci"], 40.9647, 29.0875, ["M4"]),
            ("Suadiye", "Suadiye", "suadiye", ["suadiye"], 40.9694, 29.0592, []),
            ("Erenköy", "Erenkoy", "erenkoy", ["erenkoy"], 40.9719, 29.0431, []),
            ("Göztepe", "Goztepe", "goztepe_marmaray", ["goztepe"], 40.9772, 29.0347, []),
            ("Feneryolu", "Feneryolu", "feneryolu", ["feneryolu"], 40.9831, 29.0253, []),
            ("Söğütlüçeşme", "Sogutlucesme", "sogutlucesme", ["sogutlucesme"], 40.9872, 29.0136, []),
            ("Ayrılık Çeşmesi", "Ayrilik Cesmesi", "ayrilik_cesmesi_marmaray", ["ayrilik", "ayrilik cesmesi"], 40.9797, 29.0510, ["M4"]),
            ("Üsküdar", "Uskudar", "uskudar_marmaray", ["uskudar"], 41.0255, 29.0144, ["M5"]),
            
            # UNDER BOSPHORUS - The undersea tunnel
            ("Sirkeci", "Sirkeci", "sirkeci_marmaray", ["sirkeci"], 41.0169, 28.9769, ["T1"]),
            
            # EUROPEAN SIDE (East to West) - Sirkeci to Halkalı
            ("Yenikapı", "Yenikapi", "yenikapi_marmaray", ["yenikapi", "yeni kapi"], 41.0042, 28.9519, ["M1A", "M1B", "M2"]),
            ("Kumkapı", "Kumkapi", "kumkapi", ["kumkapi"], 41.0072, 28.9364, []),
            ("Narli", "Narli", "narli", ["narli"], 41.0089, 28.9281, []),
            ("Kazlıçeşme", "Kazlicesme", "kazlicesme", ["kazlicesme"], 41.0078, 28.9264, []),
            ("Zeytinburnu", "Zeytinburnu", "zeytinburnu_marmaray", ["zeytinburnu"], 41.0072, 28.9053, ["T1", "M1A", "M1B"]),
            ("Yenimahalle", "Yenimahalle", "yenimahalle_marmaray", ["yeni mahalle"], 41.0019, 28.8933, []),
            ("Bakırköy", "Bakirkoy", "bakirkoy_marmaray", ["bakirkoy"], 40.9833, 28.8667, []),
            ("Ataköy", "Atakoy", "atakoy", ["atakoy"], 40.9767, 28.8408, []),
            ("Yeşilyurt", "Yesilyurt", "yesilyurt", ["yesilyurt"], 40.9711, 28.8244, []),
            ("Yeşilköy", "Yesilkoy", "yesilkoy", ["yesilkoy"], 40.9667, 28.8167, []),
            ("Florya Akvaryum", "Florya Aquarium", "florya_akvaryum", ["florya akvaryum", "florya aquarium"], 40.9694, 28.7964, []),
            ("Florya", "Florya", "florya", ["florya"], 40.9722, 28.7889, []),
            ("Küçükçekmece", "Kucukcekmece", "kucukcekmece_marmaray", ["kucukcekmece"], 40.9917, 28.7667, []),
            ("Cennet", "Cennet", "cennet", ["cennet"], 40.9972, 28.7464, []),
            ("Sefaköy", "Sefakoy", "sefakoy", ["sefakoy"], 41.0111, 28.7242, []),
            ("Haznedar", "Haznedar", "haznedar", ["haznedar"], 41.0150, 28.6958, []),
            ("Yenibosna", "Yenibosna", "yenibosna_marmaray", ["yenibosna"], 40.9906, 28.7103, []),
            ("Mustafa Kemal", "Mustafa Kemal", "mustafa_kemal", ["mustafa kemal"], 41.0047, 28.7519, []),
            ("Marmaray", "Marmaray", "marmaray_station", ["marmaray"], 41.0089, 28.6833, []),
            ("Halkalı", "Halkali", "halkali", ["halkali"], 41.0078, 28.6456, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in marmaray_stations:
            stations.append(CanonicalStation(
                canonical_id=f"MARMARAY-{name_tr}",
                station_id=short_id,
                line_id="MARMARAY",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        # ==========================================
        # FERRY TERMINALS (Şehir Hatları) - 15 terminals
        # Ferry terminals are not a continuous line but point-to-point connections
        # ==========================================
        ferry_terminals = [
            # European Side - Bosphorus
            ("Eminönü", "Eminonu", "eminonu_ferry", ["eminonu", "eminönü vapur"], 41.0197, 28.9739, ["T1", "T4"]),
            ("Karaköy", "Karakoy", "karakoy_ferry", ["karakoy", "karaköy vapur"], 41.0236, 28.9753, ["T1", "T4", "F2"]),
            ("Beşiktaş", "Besiktas", "besiktas_ferry", ["besiktas", "beşiktaş vapur"], 41.0425, 29.0019, ["T4"]),
            ("Kabataş", "Kabatas", "kabatas_ferry", ["kabatas", "kabataş vapur"], 41.0356, 28.9914, ["T1", "T4", "F1"]),
            ("Sarıyer", "Sariyer", "sariyer_ferry", ["sariyer", "sarıyer vapur"], 41.1667, 29.0500, []),
            ("Rumeli Kavağı", "Rumeli Kavagi", "rumeli_kavagi_ferry", ["rumeli kavagi", "rumeli kavağı"], 41.1833, 29.0667, []),
            
            # Asian Side - Bosphorus
            ("Kadıköy", "Kadikoy", "kadikoy_ferry", ["kadikoy", "kadıköy vapur"], 40.9833, 29.0283, ["M4"]),
            ("Üsküdar", "Uskudar", "uskudar_ferry", ["uskudar", "üsküdar vapur"], 41.0256, 29.0178, ["M5", "MARMARAY"]),
            ("Beykoz", "Beykoz", "beykoz_ferry", ["beykoz", "beykoz vapur"], 41.1333, 29.0833, []),
            ("Çengelköy", "Cengelkoy", "cengelkoy_ferry", ["cengelkoy", "çengelköy"], 41.0500, 29.0667, []),
            ("Kanlıca", "Kanlica", "kanlica_ferry", ["kanlica", "kanlıca"], 41.0667, 29.0667, []),
            
            # Princes' Islands (Adalar)
            ("Büyükada", "Buyukada", "buyukada_ferry", ["buyukada", "büyükada"], 40.8597, 29.1214, []),
            ("Heybeliada", "Heybeliada", "heybeliada_ferry", ["heybeliada"], 40.8772, 29.0864, []),
            ("Burgazada", "Burgazada", "burgazada_ferry", ["burgazada"], 40.8808, 29.0631, []),
            ("Kınalıada", "Kinaliada", "kinaliada_ferry", ["kinaliada", "kınalıada"], 40.9167, 29.0500, []),
        ]
        
        for name_tr, name_en, short_id, variants, lat, lon, transfers in ferry_terminals:
            stations.append(CanonicalStation(
                canonical_id=f"FERRY-{name_tr}",
                station_id=short_id,
                line_id="FERRY",
                name_tr=name_tr,
                name_en=name_en,
                name_variants=variants,
                lat=lat,
                lon=lon,
                transfers=transfers
            ))
        
        return stations
    
    def normalize_station(self, station_name: str, line_hint: Optional[str] = None) -> Optional[CanonicalStation]:
        """
        Find canonical station by name (fuzzy matching).
        
        Args:
            station_name: Station name in any language/format
            line_hint: Optional line ID to disambiguate (e.g., "M2" vs "F1" for Taksim)
        
        Returns:
            CanonicalStation or None if not found
        """
        station_name_lower = station_name.lower().strip()
        
        # Try Turkish name lookup
        if station_name_lower in self.station_by_name_tr:
            station = self.station_by_name_tr[station_name_lower]
            if line_hint and station.line_id != line_hint:
                # Look for another station with same name but different line
                for s in self.stations:
                    if s.name_tr.lower() == station_name_lower and s.line_id == line_hint:
                        return s
            return station
        
        # Try English name lookup
        if station_name_lower in self.station_by_name_en:
            station = self.station_by_name_en[station_name_lower]
            if line_hint and station.line_id != line_hint:
                for s in self.stations:
                    if s.name_en.lower() == station_name_lower and s.line_id == line_hint:
                        return s
            return station
        
        # Try partial match
        for station in self.stations:
            if (station_name_lower in station.name_tr.lower() or 
                station_name_lower in station.name_en.lower() or
                station.name_tr.lower() in station_name_lower or
                station.name_en.lower() in station_name_lower):
                if line_hint and station.line_id != line_hint:
                    continue
                return station
        
        return None
    
    def get_line_metadata(self, line_id: str) -> Optional[CanonicalLine]:
        """Get canonical line metadata"""
        return self.lines.get(line_id)
    
    def enrich_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a route step with canonical IDs and multilingual names.
        
        Args:
            step: A route step dict (from transportation RAG)
        
        Returns:
            Enriched step with canonical IDs and names
        """
        enriched = step.copy()
        
        # Normalize line
        line_id = step.get('line', '')
        if line_id:
            line_meta = self.get_line_metadata(line_id)
            if line_meta:
                enriched['line_id'] = line_meta.line_id
                enriched['line_name'] = line_meta.name_en
                enriched['line_name_tr'] = line_meta.name_tr
                enriched['line_type'] = line_meta.line_type
                enriched['line_color'] = line_meta.color
        
        # Normalize 'from' station
        from_name = step.get('from', '')
        if from_name:
            station = self.normalize_station(from_name, line_id)
            if station:
                enriched['from_station_id'] = station.canonical_id
                enriched['from_station_name'] = station.name_en
                enriched['from_station_name_tr'] = station.name_tr
                enriched['from_lat'] = station.lat
                enriched['from_lon'] = station.lon
        
        # Normalize 'to' station
        to_name = step.get('to', '')
        if to_name:
            station = self.normalize_station(to_name, line_id)
            if station:
                enriched['to_station_id'] = station.canonical_id
                enriched['to_station_name'] = station.name_en
                enriched['to_station_name_tr'] = station.name_tr
                enriched['to_lat'] = station.lat
                enriched['to_lon'] = station.lon
        
        return enriched
    
    def enrich_route_data(self, route_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich complete route_data with canonical IDs and multilingual names.
        
        Args:
            route_data: Complete route_data dict from transportation RAG
        
        Returns:
            Enriched route_data with canonical IDs
        """
        enriched = route_data.copy()
        
        # Enrich origin
        origin_name = route_data.get('origin', '')
        if origin_name:
            # Try to determine line from first step
            line_hint = None
            if enriched.get('steps') and len(enriched['steps']) > 0:
                line_hint = enriched['steps'][0].get('line')
            
            station = self.normalize_station(origin_name, line_hint)
            if station:
                enriched['origin_station_id'] = station.canonical_id
                enriched['origin_name_en'] = station.name_en
                enriched['origin_name_tr'] = station.name_tr
                enriched['origin_lat'] = station.lat
                enriched['origin_lon'] = station.lon
        
        # Enrich destination
        destination_name = route_data.get('destination', '')
        if destination_name:
            # Try to determine line from last step
            line_hint = None
            if enriched.get('steps') and len(enriched['steps']) > 0:
                line_hint = enriched['steps'][-1].get('line')
            
            station = self.normalize_station(destination_name, line_hint)
            if station:
                enriched['destination_station_id'] = station.canonical_id
                enriched['destination_name_en'] = station.name_en
                enriched['destination_name_tr'] = station.name_tr
                enriched['destination_lat'] = station.lat
                enriched['destination_lon'] = station.lon
        
        # Enrich steps
        if 'steps' in enriched and isinstance(enriched['steps'], list):
            enriched['steps'] = [self.enrich_step(step) for step in enriched['steps']]
        
        # Enrich lines_used
        if 'lines_used' in enriched and isinstance(enriched['lines_used'], list):
            enriched_lines = []
            for line_id in enriched['lines_used']:
                line_meta = self.get_line_metadata(line_id)
                if line_meta:
                    enriched_lines.append({
                        'line_id': line_meta.line_id,
                        'line_name': line_meta.name_en,
                        'line_name_tr': line_meta.name_tr,
                        'line_type': line_meta.line_type,
                        'color': line_meta.color
                    })
                else:
                    # Fallback if no metadata
                    enriched_lines.append({
                        'line_id': line_id,
                        'line_name': line_id,
                        'line_name_tr': line_id,
                        'line_type': 'unknown',
                        'color': '#888888'
                    })
            enriched['lines_used_enriched'] = enriched_lines
        
        return enriched


# Singleton instance
_station_normalizer = None

def get_station_normalizer() -> StationNormalizer:
    """Get singleton station normalizer instance"""
    global _station_normalizer
    if _station_normalizer is None:
        _station_normalizer = StationNormalizer()
    return _station_normalizer

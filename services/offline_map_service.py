#!/usr/bin/env python3
"""
Offline Map Service for Istanbul Public Transport
==================================================

Provides static geographic data for metro, tram, ferry, and bus routes
using İBB (Istanbul Metropolitan Municipality) data structures.

Features:
- Metro line routes with station coordinates
- Tram line routes with stop locations
- Ferry routes with pier locations
- GeoJSON format for easy map integration
- Offline-first design with static data fallback
- Integration with GTFS for schedule data
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GeoCoordinate:
    """Geographic coordinate"""
    lat: float
    lon: float
    
    def to_geojson_point(self) -> Dict[str, Any]:
        """Convert to GeoJSON Point"""
        return {
            "type": "Point",
            "coordinates": [self.lon, self.lat]
        }


@dataclass
class TransportStop:
    """Transport stop/station with location"""
    stop_id: str
    name: str
    name_en: Optional[str]
    location: GeoCoordinate
    stop_type: str  # metro, tram, bus, ferry
    accessible: bool = True
    facilities: List[str] = None
    
    def to_geojson_feature(self) -> Dict[str, Any]:
        """Convert to GeoJSON Feature"""
        return {
            "type": "Feature",
            "geometry": self.location.to_geojson_point(),
            "properties": {
                "stop_id": self.stop_id,
                "name": self.name,
                "name_en": self.name_en,
                "stop_type": self.stop_type,
                "accessible": self.accessible,
                "facilities": self.facilities or []
            }
        }


@dataclass
class TransportRoute:
    """Transport route with geometry"""
    route_id: str
    name: str
    name_en: Optional[str]
    route_type: str  # metro, tram, bus, ferry
    color: str
    stops: List[TransportStop]
    geometry: List[GeoCoordinate]  # Route path
    operational: bool = True
    
    def to_geojson_feature_collection(self) -> Dict[str, Any]:
        """Convert to GeoJSON FeatureCollection with route line and stops"""
        features = []
        
        # Add route line
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[coord.lon, coord.lat] for coord in self.geometry]
            },
            "properties": {
                "route_id": self.route_id,
                "name": self.name,
                "name_en": self.name_en,
                "route_type": self.route_type,
                "color": self.color,
                "operational": self.operational
            }
        })
        
        # Add stops as points
        for stop in self.stops:
            features.append(stop.to_geojson_feature())
        
        return {
            "type": "FeatureCollection",
            "features": features
        }


class OfflineMapService:
    """
    Service for offline map data of Istanbul public transport
    """
    
    def __init__(self):
        self.metro_lines: Dict[str, TransportRoute] = {}
        self.tram_lines: Dict[str, TransportRoute] = {}
        self.ferry_routes: Dict[str, TransportRoute] = {}
        self.bus_routes: Dict[str, TransportRoute] = {}
        
        # Load static data
        self._load_istanbul_metro_lines()
        self._load_istanbul_tram_lines()
        self._load_istanbul_ferry_routes()
        self._load_major_bus_routes()
        
        logger.info(f"✅ Offline map data loaded: {len(self.metro_lines)} metro lines, "
                   f"{len(self.tram_lines)} tram lines, {len(self.ferry_routes)} ferry routes")
    
    def _load_istanbul_metro_lines(self):
        """Load Istanbul metro line data with station coordinates"""
        
        # M1A: Yenikapı - Atatürk Airport (Red Line)
        m1a_stops = [
            TransportStop("M1A_01", "Yenikapı", "Yenikapi", GeoCoordinate(41.0036, 28.9518), "metro"),
            TransportStop("M1A_02", "Aksaray", "Aksaray", GeoCoordinate(41.0017, 28.9574), "metro"),
            TransportStop("M1A_03", "Emniyet-Fatih", "Emniyet-Fatih", GeoCoordinate(41.0039, 28.9651), "metro"),
            TransportStop("M1A_04", "Ulubatlı", "Ulubatli", GeoCoordinate(41.0089, 28.9320), "metro"),
            TransportStop("M1A_05", "Bayrampaşa-Maltepe", "Bayrampasa-Maltepe", GeoCoordinate(41.0386, 28.8997), "metro"),
            TransportStop("M1A_06", "Zeytinburnu", "Zeytinburnu", GeoCoordinate(40.9906, 28.9041), "metro"),
            TransportStop("M1A_07", "Bakırköy-İncirli", "Bakirkoy-Incirli", GeoCoordinate(40.9813, 28.8709), "metro"),
            TransportStop("M1A_08", "Atatürk Havalimanı", "Ataturk Airport", GeoCoordinate(40.9765, 28.8152), "metro"),
        ]
        m1a_geometry = [stop.location for stop in m1a_stops]
        self.metro_lines["M1A"] = TransportRoute(
            "M1A", "Yenikapı - Atatürk Havalimanı", "Yenikapi - Ataturk Airport",
            "metro", "#E53E3E", m1a_stops, m1a_geometry
        )
        
        # M1B: Yenikapı - Kirazlı (Light Red Line)
        m1b_stops = [
            TransportStop("M1B_01", "Yenikapı", "Yenikapi", GeoCoordinate(41.0036, 28.9518), "metro"),
            TransportStop("M1B_02", "Aksaray", "Aksaray", GeoCoordinate(41.0017, 28.9574), "metro"),
            TransportStop("M1B_03", "Emniyet-Fatih", "Emniyet-Fatih", GeoCoordinate(41.0039, 28.9651), "metro"),
            TransportStop("M1B_04", "Topkapı-Ulubatlı", "Topkapi-Ulubatli", GeoCoordinate(41.0148, 28.9226), "metro"),
            TransportStop("M1B_05", "Bayrampaşa", "Bayrampasa", GeoCoordinate(41.0455, 28.8956), "metro"),
            TransportStop("M1B_06", "Esenler", "Esenler", GeoCoordinate(41.0556, 28.8789), "metro"),
            TransportStop("M1B_07", "Kirazlı", "Kirazli", GeoCoordinate(41.0103, 28.7891), "metro"),
        ]
        m1b_geometry = [stop.location for stop in m1b_stops]
        self.metro_lines["M1B"] = TransportRoute(
            "M1B", "Yenikapı - Kirazlı", "Yenikapi - Kirazli",
            "metro", "#FF6B9D", m1b_stops, m1b_geometry
        )
        
        # M2: Vezneciler - Hacıosman (Green Line)
        m2_stops = [
            TransportStop("M2_01", "Vezneciler", "Vezneciler", GeoCoordinate(41.0158, 28.9541), "metro"),
            TransportStop("M2_02", "Haliç", "Halic (Golden Horn)", GeoCoordinate(41.0217, 28.9467), "metro"),
            TransportStop("M2_03", "Şişhane", "Sishane", GeoCoordinate(41.0256, 28.9742), "metro"),
            TransportStop("M2_04", "Taksim", "Taksim", GeoCoordinate(41.0369, 28.9850), "metro", facilities=["transfer"]),
            TransportStop("M2_05", "Osmanbey", "Osmanbey", GeoCoordinate(41.0489, 28.9885), "metro"),
            TransportStop("M2_06", "Şişli-Mecidiyeköy", "Sisli-Mecidiyekoy", GeoCoordinate(41.0634, 29.0084), "metro", facilities=["transfer"]),
            TransportStop("M2_07", "Gayrettepe", "Gayrettepe", GeoCoordinate(41.0679, 29.0159), "metro", facilities=["transfer"]),
            TransportStop("M2_08", "Levent", "Levent", GeoCoordinate(41.0822, 29.0138), "metro"),
            TransportStop("M2_09", "4.Levent", "4.Levent", GeoCoordinate(41.0876, 29.0157), "metro"),
            TransportStop("M2_10", "Sanayi Mahallesi", "Sanayi Mahallesi", GeoCoordinate(41.0928, 29.0185), "metro"),
            TransportStop("M2_11", "İTÜ-Ayazağa", "ITU-Ayazaga", GeoCoordinate(41.1037, 29.0218), "metro"),
            TransportStop("M2_12", "Atatürk Oto Sanayi", "Ataturk Oto Sanayi", GeoCoordinate(41.1078, 29.0256), "metro"),
            TransportStop("M2_13", "Darüşşafaka", "Darussafaka", GeoCoordinate(41.1108, 29.0301), "metro"),
            TransportStop("M2_14", "Hacıosman", "Haciiosman", GeoCoordinate(41.1187, 29.0343), "metro"),
        ]
        m2_geometry = [stop.location for stop in m2_stops]
        self.metro_lines["M2"] = TransportRoute(
            "M2", "Vezneciler - Hacıosman", "Vezneciler - Haciosman",
            "metro", "#00A651", m2_stops, m2_geometry
        )
        
        # M3: Kirazlı - Olimpiyat (Blue Line)
        m3_stops = [
            TransportStop("M3_01", "Kirazlı", "Kirazli", GeoCoordinate(41.0103, 28.7891), "metro", facilities=["transfer"]),
            TransportStop("M3_02", "Başak Konutları", "Basak Konutlari", GeoCoordinate(41.0345, 28.7812), "metro"),
            TransportStop("M3_03", "Siteler", "Siteler", GeoCoordinate(41.0456, 28.7890), "metro"),
            TransportStop("M3_04", "Turgut Özal", "Turgut Ozal", GeoCoordinate(41.0489, 28.7923), "metro"),
            TransportStop("M3_05", "İkitelli Sanayi", "Ikitelli Sanayi", GeoCoordinate(41.0534, 28.7967), "metro"),
            TransportStop("M3_06", "Olimpiyat", "Olimpiyat", GeoCoordinate(41.0678, 28.8012), "metro"),
        ]
        m3_geometry = [stop.location for stop in m3_stops]
        self.metro_lines["M3"] = TransportRoute(
            "M3", "Kirazlı - Olimpiyat", "Kirazli - Olimpiyat",
            "metro", "#0078C8", m3_stops, m3_geometry
        )
        
        # M4: Kadıköy - Sabiha Gökçen Airport (Pink Line)
        m4_stops = [
            TransportStop("M4_01", "Kadıköy", "Kadikoy", GeoCoordinate(40.9907, 29.0265), "metro", facilities=["transfer"]),
            TransportStop("M4_02", "Ayrılık Çeşmesi", "Ayrilik Cesmesi", GeoCoordinate(40.9956, 29.0312), "metro"),
            TransportStop("M4_03", "Acıbadem", "Acibadem", GeoCoordinate(40.9989, 29.0389), "metro"),
            TransportStop("M4_04", "Ünalan", "Unalan", GeoCoordinate(41.0023, 29.0456), "metro"),
            TransportStop("M4_05", "Göztepe", "Goztepe", GeoCoordinate(41.0089, 29.0567), "metro"),
            TransportStop("M4_06", "Bostancı", "Bostanci", GeoCoordinate(40.9645, 29.0878), "metro"),
            TransportStop("M4_07", "Küçükyalı", "Kucukyali", GeoCoordinate(40.9234, 29.1234), "metro"),
            TransportStop("M4_08", "Sabiha Gökçen Havalimanı", "Sabiha Gokcen Airport", GeoCoordinate(40.8989, 29.3092), "metro"),
        ]
        m4_geometry = [stop.location for stop in m4_stops]
        self.metro_lines["M4"] = TransportRoute(
            "M4", "Kadıköy - Sabiha Gökçen", "Kadikoy - Sabiha Gokcen Airport",
            "metro", "#E91E63", m4_stops, m4_geometry
        )
        
        # M5: Üsküdar - Çekmeköy (Purple Line)
        m5_stops = [
            TransportStop("M5_01", "Üsküdar", "Uskudar", GeoCoordinate(41.0240, 29.0152), "metro", facilities=["transfer"]),
            TransportStop("M5_02", "Fıstıkağacı", "Fistikagaci", GeoCoordinate(41.0289, 29.0234), "metro"),
            TransportStop("M5_03", "Bağlarbaşı", "Baglarbasi", GeoCoordinate(41.0345, 29.0345), "metro"),
            TransportStop("M5_04", "Altunizade", "Altunizade", GeoCoordinate(41.0389, 29.0456), "metro"),
            TransportStop("M5_05", "Çamlıca", "Camlica", GeoCoordinate(41.0456, 29.0589), "metro"),
            TransportStop("M5_06", "Çekmeköy", "Cekmekoy", GeoCoordinate(41.0323, 29.1256), "metro"),
        ]
        m5_geometry = [stop.location for stop in m5_stops]
        self.metro_lines["M5"] = TransportRoute(
            "M5", "Üsküdar - Çekmeköy", "Uskudar - Cekmekoy",
            "metro", "#9C27B0", m5_stops, m5_geometry
        )
        
        # M7: Mecidiyeköy - Mahmutbey (Light Pink Line)
        m7_stops = [
            TransportStop("M7_01", "Mecidiyeköy", "Mecidiyekoy", GeoCoordinate(41.0634, 29.0084), "metro", facilities=["transfer"]),
            TransportStop("M7_02", "Çağlayan", "Caglayan", GeoCoordinate(41.0689, 28.9934), "metro"),
            TransportStop("M7_03", "Kağıthane", "Kagithane", GeoCoordinate(41.0756, 28.9823), "metro"),
            TransportStop("M7_04", "Yıldız", "Yildiz", GeoCoordinate(41.0823, 28.9645), "metro"),
            TransportStop("M7_05", "Mahmutbey", "Mahmutbey", GeoCoordinate(41.0556, 28.8234), "metro"),
        ]
        m7_geometry = [stop.location for stop in m7_stops]
        self.metro_lines["M7"] = TransportRoute(
            "M7", "Mecidiyeköy - Mahmutbey", "Mecidiyekoy - Mahmutbey",
            "metro", "#FF69B4", m7_stops, m7_geometry
        )
    
    def _load_istanbul_tram_lines(self):
        """Load Istanbul tram line data"""
        
        # T1: Kabataş - Bağcılar (Historic Tram)
        t1_stops = [
            TransportStop("T1_01", "Kabataş", "Kabatas", GeoCoordinate(41.0363, 28.9889), "tram", facilities=["transfer"]),
            TransportStop("T1_02", "Fındıklı", "Findikli", GeoCoordinate(41.0334, 28.9867), "tram"),
            TransportStop("T1_03", "Tophane", "Tophane", GeoCoordinate(41.0267, 28.9823), "tram"),
            TransportStop("T1_04", "Karaköy", "Karakoy", GeoCoordinate(41.0234, 28.9756), "tram", facilities=["transfer"]),
            TransportStop("T1_05", "Eminönü", "Eminonu", GeoCoordinate(41.0178, 28.9723), "tram", facilities=["ferry_connection"]),
            TransportStop("T1_06", "Gülhane", "Gulhane", GeoCoordinate(41.0134, 28.9801), "tram"),
            TransportStop("T1_07", "Sultanahmet", "Sultanahmet", GeoCoordinate(41.0056, 28.9769), "tram"),
            TransportStop("T1_08", "Beyazıt", "Beyazit", GeoCoordinate(41.0108, 28.9645), "tram"),
            TransportStop("T1_09", "Laleli", "Laleli", GeoCoordinate(41.0089, 28.9589), "tram"),
            TransportStop("T1_10", "Aksaray", "Aksaray", GeoCoordinate(41.0017, 28.9574), "tram", facilities=["transfer"]),
            TransportStop("T1_11", "Yusufpaşa", "Yusufpasa", GeoCoordinate(40.9978, 28.9456), "tram"),
            TransportStop("T1_12", "Zeytinburnu", "Zeytinburnu", GeoCoordinate(40.9906, 28.9041), "tram", facilities=["transfer"]),
            TransportStop("T1_13", "Cevizlibağ", "Cevizlibag", GeoCoordinate(40.9845, 28.8756), "tram"),
            TransportStop("T1_14", "Bağcılar", "Bagcilar", GeoCoordinate(40.9756, 28.8345), "tram"),
        ]
        t1_geometry = [stop.location for stop in t1_stops]
        self.tram_lines["T1"] = TransportRoute(
            "T1", "Kabataş - Bağcılar", "Kabatas - Bagcilar",
            "tram", "#FF9800", t1_stops, t1_geometry
        )
        
        # T4: Topkapı - Mescid-i Selam
        t4_stops = [
            TransportStop("T4_01", "Topkapı", "Topkapi", GeoCoordinate(41.0148, 28.9226), "tram", facilities=["transfer"]),
            TransportStop("T4_02", "Edirnekapı", "Edirnekapi", GeoCoordinate(41.0234, 28.9234), "tram"),
            TransportStop("T4_03", "Sultan Çiftliği", "Sultan Ciftligi", GeoCoordinate(41.0389, 28.9156), "tram"),
            TransportStop("T4_04", "Alibeyköy", "Alibeykoy", GeoCoordinate(41.0489, 28.9078), "tram"),
            TransportStop("T4_05", "Mescid-i Selam", "Mescid-i Selam", GeoCoordinate(41.0567, 28.8967), "tram"),
        ]
        t4_geometry = [stop.location for stop in t4_stops]
        self.tram_lines["T4"] = TransportRoute(
            "T4", "Topkapı - Mescid-i Selam", "Topkapi - Mescid-i Selam",
            "tram", "#FFC107", t4_stops, t4_geometry
        )
        
        # T5: Eminönü - Alibeyköy (Cibali Tram)
        t5_stops = [
            TransportStop("T5_01", "Eminönü", "Eminonu", GeoCoordinate(41.0178, 28.9723), "tram"),
            TransportStop("T5_02", "Karaköy", "Karakoy", GeoCoordinate(41.0234, 28.9756), "tram"),
            TransportStop("T5_03", "Cibali", "Cibali", GeoCoordinate(41.0345, 28.9534), "tram"),
            TransportStop("T5_04", "Alibeyköy", "Alibeykoy", GeoCoordinate(41.0489, 28.9078), "tram"),
        ]
        t5_geometry = [stop.location for stop in t5_stops]
        self.tram_lines["T5"] = TransportRoute(
            "T5", "Eminönü - Alibeyköy", "Eminonu - Alibeykoy",
            "tram", "#FFEB3B", t5_stops, t5_geometry
        )
    
    def _load_istanbul_ferry_routes(self):
        """Load Istanbul ferry route data"""
        
        # Kadıköy - Eminönü Ferry
        kadikoy_eminonu = [
            TransportStop("F_KAD", "Kadıköy İskelesi", "Kadikoy Pier", GeoCoordinate(40.9907, 29.0265), "ferry"),
            TransportStop("F_EMI", "Eminönü İskelesi", "Eminonu Pier", GeoCoordinate(41.0178, 28.9723), "ferry"),
        ]
        self.ferry_routes["F_KAD_EMI"] = TransportRoute(
            "F_KAD_EMI", "Kadıköy - Eminönü", "Kadikoy - Eminonu",
            "ferry", "#2196F3", kadikoy_eminonu, [stop.location for stop in kadikoy_eminonu]
        )
        
        # Kadıköy - Karaköy Ferry
        kadikoy_karakoy = [
            TransportStop("F_KAD", "Kadıköy İskelesi", "Kadikoy Pier", GeoCoordinate(40.9907, 29.0265), "ferry"),
            TransportStop("F_KRY", "Karaköy İskelesi", "Karakoy Pier", GeoCoordinate(41.0234, 28.9756), "ferry"),
        ]
        self.ferry_routes["F_KAD_KRY"] = TransportRoute(
            "F_KAD_KRY", "Kadıköy - Karaköy", "Kadikoy - Karakoy",
            "ferry", "#2196F3", kadikoy_karakoy, [stop.location for stop in kadikoy_karakoy]
        )
        
        # Üsküdar - Eminönü Ferry
        uskudar_eminonu = [
            TransportStop("F_USK", "Üsküdar İskelesi", "Uskudar Pier", GeoCoordinate(41.0240, 29.0152), "ferry"),
            TransportStop("F_EMI", "Eminönü İskelesi", "Eminonu Pier", GeoCoordinate(41.0178, 28.9723), "ferry"),
        ]
        self.ferry_routes["F_USK_EMI"] = TransportRoute(
            "F_USK_EMI", "Üsküdar - Eminönü", "Uskudar - Eminonu",
            "ferry", "#2196F3", uskudar_eminonu, [stop.location for stop in uskudar_eminonu]
        )
        
        # Beşiktaş - Kadıköy Ferry
        besiktas_kadikoy = [
            TransportStop("F_BES", "Beşiktaş İskelesi", "Besiktas Pier", GeoCoordinate(41.0422, 29.0067), "ferry"),
            TransportStop("F_KAD", "Kadıköy İskelesi", "Kadikoy Pier", GeoCoordinate(40.9907, 29.0265), "ferry"),
        ]
        self.ferry_routes["F_BES_KAD"] = TransportRoute(
            "F_BES_KAD", "Beşiktaş - Kadıköy", "Besiktas - Kadikoy",
            "ferry", "#2196F3", besiktas_kadikoy, [stop.location for stop in besiktas_kadikoy]
        )
        
        # Bosphorus Tour (Circular)
        bosphorus_tour = [
            TransportStop("F_EMI", "Eminönü İskelesi", "Eminonu Pier", GeoCoordinate(41.0178, 28.9723), "ferry"),
            TransportStop("F_KRY", "Karaköy İskelesi", "Karakoy Pier", GeoCoordinate(41.0234, 28.9756), "ferry"),
            TransportStop("F_BES", "Beşiktaş İskelesi", "Besiktas Pier", GeoCoordinate(41.0422, 29.0067), "ferry"),
            TransportStop("F_ORT", "Ortaköy İskelesi", "Ortakoy Pier", GeoCoordinate(41.0556, 29.0267), "ferry"),
            TransportStop("F_BEB", "Bebek İskelesi", "Bebek Pier", GeoCoordinate(41.0756, 29.0423), "ferry"),
            TransportStop("F_KAN", "Kanlıca İskelesi", "Kanlica Pier", GeoCoordinate(41.0734, 29.0634), "ferry"),
            TransportStop("F_ANV", "Anadolu Kavağı", "Anadolu Kavagi", GeoCoordinate(41.1856, 29.0823), "ferry"),
        ]
        self.ferry_routes["F_BOSPHORUS"] = TransportRoute(
            "F_BOSPHORUS", "Boğaz Turu", "Bosphorus Tour",
            "ferry", "#03A9F4", bosphorus_tour, [stop.location for stop in bosphorus_tour]
        )
    
    def _load_major_bus_routes(self):
        """Load major bus routes (Metrobus and popular lines)"""
        
        # Metrobus (BRT)
        metrobus_stops = [
            TransportStop("MB_01", "Avcılar", "Avcilar", GeoCoordinate(40.9779, 28.7219), "bus"),
            TransportStop("MB_02", "Beylikdüzü", "Beylikduzu", GeoCoordinate(40.9890, 28.6577), "bus"),
            TransportStop("MB_03", "Metrokent", "Metrokent", GeoCoordinate(41.0074, 28.7875), "bus"),
            TransportStop("MB_04", "Bakırköy", "Bakirkoy", GeoCoordinate(40.9813, 28.8709), "bus"),
            TransportStop("MB_05", "Zeytinburnu", "Zeytinburnu", GeoCoordinate(40.9906, 28.9041), "bus"),
            TransportStop("MB_06", "Topkapı", "Topkapi", GeoCoordinate(41.0148, 28.9226), "bus"),
            TransportStop("MB_07", "Mecidiyeköy", "Mecidiyekoy", GeoCoordinate(41.0634, 29.0084), "bus"),
            TransportStop("MB_08", "Zincirlikuyu", "Zincirlikuyu", GeoCoordinate(41.0739, 29.0176), "bus"),
            TransportStop("MB_09", "Akatlar", "Akatlar", GeoCoordinate(41.0823, 29.0289), "bus"),
            TransportStop("MB_10", "Uzunçayır", "Uzuncayir", GeoCoordinate(41.0345, 29.0789), "bus"),
        ]
        metrobus_geometry = [stop.location for stop in metrobus_stops]
        self.bus_routes["METROBUS"] = TransportRoute(
            "METROBUS", "Metrobüs", "Metrobus BRT",
            "bus", "#FF6600", metrobus_stops, metrobus_geometry
        )
    
    def get_all_routes_geojson(self) -> Dict[str, Any]:
        """Get all routes as a single GeoJSON FeatureCollection"""
        all_features = []
        
        # Add all metro lines
        for route in self.metro_lines.values():
            fc = route.to_geojson_feature_collection()
            all_features.extend(fc['features'])
        
        # Add all tram lines
        for route in self.tram_lines.values():
            fc = route.to_geojson_feature_collection()
            all_features.extend(fc['features'])
        
        # Add all ferry routes
        for route in self.ferry_routes.values():
            fc = route.to_geojson_feature_collection()
            all_features.extend(fc['features'])
        
        # Add all bus routes
        for route in self.bus_routes.values():
            fc = route.to_geojson_feature_collection()
            all_features.extend(fc['features'])
        
        return {
            "type": "FeatureCollection",
            "features": all_features,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source": "Istanbul AI - Offline Map Service",
                "metro_lines": len(self.metro_lines),
                "tram_lines": len(self.tram_lines),
                "ferry_routes": len(self.ferry_routes),
                "bus_routes": len(self.bus_routes)
            }
        }
    
    def get_routes_by_type(self, route_type: str) -> Dict[str, Any]:
        """Get routes filtered by type (metro, tram, ferry, bus)"""
        routes_dict = {
            "metro": self.metro_lines,
            "tram": self.tram_lines,
            "ferry": self.ferry_routes,
            "bus": self.bus_routes
        }
        
        routes = routes_dict.get(route_type.lower(), {})
        features = []
        
        for route in routes.values():
            fc = route.to_geojson_feature_collection()
            features.extend(fc['features'])
        
        return {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "route_type": route_type,
                "count": len(routes),
                "generated_at": datetime.now().isoformat()
            }
        }
    
    def get_route_by_id(self, route_id: str) -> Optional[Dict[str, Any]]:
        """Get specific route by ID"""
        # Search in all route types
        all_routes = {
            **self.metro_lines,
            **self.tram_lines,
            **self.ferry_routes,
            **self.bus_routes
        }
        
        route = all_routes.get(route_id)
        if route:
            return route.to_geojson_feature_collection()
        return None
    
    def find_nearest_stop(self, lat: float, lon: float, max_distance_km: float = 1.0) -> List[Dict[str, Any]]:
        """Find nearest stops to a location"""
        from math import radians, sin, cos, sqrt, atan2
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance between two points using Haversine formula"""
            R = 6371  # Earth radius in kilometers
            
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            
            return R * c
        
        nearby_stops = []
        
        # Collect all stops from all routes
        all_routes = {
            **self.metro_lines,
            **self.tram_lines,
            **self.ferry_routes,
            **self.bus_routes
        }
        
        for route in all_routes.values():
            for stop in route.stops:
                distance = haversine_distance(lat, lon, stop.location.lat, stop.location.lon)
                if distance <= max_distance_km:
                    nearby_stops.append({
                        "stop": stop.to_geojson_feature(),
                        "route_id": route.route_id,
                        "route_name": route.name,
                        "route_type": route.route_type,
                        "distance_km": round(distance, 3),
                        "distance_m": round(distance * 1000)
                    })
        
        # Sort by distance
        nearby_stops.sort(key=lambda x: x['distance_km'])
        
        return nearby_stops
    
    def export_to_file(self, filepath: str, route_type: Optional[str] = None):
        """Export map data to GeoJSON file"""
        if route_type:
            data = self.get_routes_by_type(route_type)
        else:
            data = self.get_all_routes_geojson()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Map data exported to {filepath}")


# Global instance for easy access
_offline_map_service = None

def get_offline_map_service() -> OfflineMapService:
    """Get or create the global offline map service instance"""
    global _offline_map_service
    if _offline_map_service is None:
        _offline_map_service = OfflineMapService()
    return _offline_map_service


if __name__ == "__main__":
    # Example usage and testing
    print("🗺️  Istanbul AI - Offline Map Service\n")
    
    service = OfflineMapService()
    
    # Export all routes
    service.export_to_file("./data/istanbul_transit_map.geojson")
    
    # Export by type
    service.export_to_file("./data/istanbul_metro_map.geojson", "metro")
    service.export_to_file("./data/istanbul_tram_map.geojson", "tram")
    service.export_to_file("./data/istanbul_ferry_map.geojson", "ferry")
    
    # Test nearest stop finder
    print("\n📍 Finding stops near Taksim Square (41.0369, 28.9850):")
    nearby = service.find_nearest_stop(41.0369, 28.9850, max_distance_km=0.5)
    for stop in nearby[:5]:
        print(f"  - {stop['stop']['properties']['name']} ({stop['route_type'].upper()}) - {stop['distance_m']}m")
    
    print(f"\n✅ Offline map service ready with {len(service.metro_lines)} metro lines, "
          f"{len(service.tram_lines)} tram lines, {len(service.ferry_routes)} ferry routes")

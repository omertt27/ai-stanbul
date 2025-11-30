"""
Transportation Directions Service
==================================

Provides detailed, Google Maps-style directions for Istanbul public transportation.
Includes metro, tram, Marmaray, funicular, and ferry routes with step-by-step instructions.

Features:
- Multi-modal transportation (metro, tram, Marmaray, funicular, ferry, walking)
- Detailed step-by-step directions
- Line-specific information
- Transfer instructions
- Real-time estimates
- Integration with OSRM for walking segments
- Note: Bus routes excluded - focuses on rail and ferry transit only
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Import OSRM for walking segments
try:
    from .osrm_routing_service import OSRMRoutingService
    OSRM_AVAILABLE = True
    logger.info("✅ Backend OSRM routing service imported successfully")
except ImportError as e:
    OSRM_AVAILABLE = False
    logger.warning(f"⚠️ OSRM not available - walking segments will be estimated. Error: {e}")
except Exception as e:
    OSRM_AVAILABLE = False
    logger.error(f"⚠️ Error importing OSRM service: {e}")

# Import graph-based routing engine
try:
    from .graph_routing_engine import (
        TransportationGraph,
        GraphRoutingEngine,
        create_istanbul_graph,
        RoutePath
    )
    GRAPH_ROUTING_AVAILABLE = True
    logger.info("✅ Graph-based routing engine imported successfully")
except ImportError as e:
    GRAPH_ROUTING_AVAILABLE = False
    logger.warning(f"⚠️ Graph routing not available - using fallback routing. Error: {e}")
except Exception as e:
    GRAPH_ROUTING_AVAILABLE = False
    logger.error(f"⚠️ Error importing graph routing engine: {e}")


@dataclass
class TransportStep:
    """A single step in a transportation route"""
    mode: str  # 'walk', 'metro', 'tram', 'ferry', 'funicular'
    instruction: str  # Human-readable instruction
    distance: float  # meters
    duration: int  # minutes
    start_location: Tuple[float, float]  # (lat, lng)
    end_location: Tuple[float, float]  # (lat, lng)
    line_name: Optional[str] = None  # e.g., "M2 Metro Line"
    stops_count: Optional[int] = None  # Number of stops
    waypoints: Optional[List[Tuple[float, float]]] = None  # Route polyline


@dataclass
class TransportRoute:
    """Complete transportation route"""
    steps: List[TransportStep]
    total_distance: float  # meters
    total_duration: int  # minutes
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    summary: str = ""
    modes_used: List[str] = None
    estimated_cost: float = 0.0  # Turkish Lira (₺)


class TransportationDirectionsService:
    """Service for generating detailed transportation directions in Istanbul"""
    
    def __init__(self):
        """Initialize the transportation directions service"""
        self.osrm = None
        if OSRM_AVAILABLE:
            try:
                # Initialize with backend OSRM service parameters
                self.osrm = OSRMRoutingService(
                    server='primary',
                    profile='foot',
                    timeout=10,
                    use_fallback=True
                )
                logger.info("✅ OSRM service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OSRM service: {e}")
                self.osrm = None
        
        # Initialize Istanbul metro/tram/bus lines
        self._initialize_transit_lines()
        
        # Initialize graph-based routing engine
        self.graph = None
        self.routing_engine = None
        if GRAPH_ROUTING_AVAILABLE:
            try:
                # Create transit data dictionary from our line data (metro, tram, Marmaray, ferry only)
                transit_data = {
                    'metro_lines': self.metro_lines,
                    'tram_lines': self.tram_lines,
                    'funicular_lines': self.funicular_lines,
                    'ferry_routes': self.ferry_routes
                }
                
                # Build the graph
                self.graph = create_istanbul_graph(transit_data)
                self.routing_engine = GraphRoutingEngine(self.graph)
                logger.info("✅ Graph-based routing engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize graph routing: {e}")
                self.graph = None
                self.routing_engine = None
        
        logger.info("✅ Transportation Directions Service initialized")
    
    def _initialize_transit_lines(self):
        """Initialize Istanbul public transit lines with stations and routes"""
        
        # Metro lines with complete station data (all official Istanbul metro lines)
        self.metro_lines = {
            'M1A': {
                'name': 'M1A Yenikapı - Atatürk Havalimanı',
                'color': 'red',
                'stations': [
                    {'name': 'Yenikapı', 'lat': 41.0035, 'lng': 28.9510},
                    {'name': 'Aksaray', 'lat': 41.0166, 'lng': 28.9548},
                    {'name': 'Emniyet-Fatih', 'lat': 41.0195, 'lng': 28.9419},
                    {'name': 'Ulubatlı', 'lat': 41.0133, 'lng': 28.9256},
                    {'name': 'Sağmalcılar', 'lat': 41.0095, 'lng': 28.9089},
                    {'name': 'Kocatepe', 'lat': 41.0089, 'lng': 28.8978},
                    {'name': 'Otogar', 'lat': 41.0142, 'lng': 28.8831},
                    {'name': 'Terazidere', 'lat': 41.0208, 'lng': 28.8672},
                    {'name': 'Davutpaşa-YTÜ', 'lat': 41.0267, 'lng': 28.8525},
                    {'name': 'Merter', 'lat': 41.0336, 'lng': 28.8414},
                    {'name': 'Zeytinburnu', 'lat': 41.0089, 'lng': 28.9089},
                    {'name': 'Bakırköy-İncirli', 'lat': 40.9856, 'lng': 28.8756},
                    {'name': 'Bahçelievler', 'lat': 40.9989, 'lng': 28.8567},
                    {'name': 'Ataköy-Şirinevler', 'lat': 40.9814, 'lng': 28.8403},
                    {'name': 'Yenibosna', 'lat': 40.9647, 'lng': 28.8236},
                    {'name': 'DTM-İstanbul Fuar Merkezi', 'lat': 40.9567, 'lng': 28.8150},
                    {'name': 'Atatürk Havalimanı', 'lat': 40.9769, 'lng': 28.8150},
                ],
                'notes': 'European side metro line to old Atatürk Airport (now mostly closed)'
            },
            'M1B': {
                'name': 'M1B Yenikapı - Kirazlı',
                'color': 'red',
                'stations': [
                    {'name': 'Yenikapı', 'lat': 41.0035, 'lng': 28.9510},
                    {'name': 'Aksaray', 'lat': 41.0166, 'lng': 28.9548},
                    {'name': 'Emniyet-Fatih', 'lat': 41.0195, 'lng': 28.9419},
                    {'name': 'Ulubatlı', 'lat': 41.0133, 'lng': 28.9256},
                    {'name': 'Sağmalcılar', 'lat': 41.0095, 'lng': 28.9089},
                    {'name': 'Kocatepe', 'lat': 41.0089, 'lng': 28.8978},
                    {'name': 'Otogar', 'lat': 41.0142, 'lng': 28.8831},
                    {'name': 'Terazidere', 'lat': 41.0208, 'lng': 28.8672},
                    {'name': 'Davutpaşa-YTÜ', 'lat': 41.0267, 'lng': 28.8525},
                    {'name': 'Merter', 'lat': 41.0336, 'lng': 28.8414},
                    {'name': 'Zeytinburnu', 'lat': 41.0050, 'lng': 28.9014},
                    {'name': 'Bakırköy-İncirli', 'lat': 40.9856, 'lng': 28.8756},
                    {'name': 'Bahçelievler', 'lat': 40.9989, 'lng': 28.8567},
                    {'name': 'Ataköy-Şirinevler', 'lat': 40.9814, 'lng': 28.8403},
                    {'name': 'Yenibosna', 'lat': 40.9647, 'lng': 28.8236},
                    {'name': 'DTM-İstanbul Fuar Merkezi', 'lat': 40.9567, 'lng': 28.8150},
                    {'name': 'Esenler', 'lat': 41.0458, 'lng': 28.8756},
                    {'name': 'Menderes', 'lat': 41.0389, 'lng': 28.8642},
                    {'name': 'Üçyüzlü', 'lat': 41.0347, 'lng': 28.8558},
                    {'name': 'Bağcılar İdో', 'lat': 41.0306, 'lng': 28.8478},
                    {'name': 'Kirazlı', 'lat': 41.0285, 'lng': 28.8264},
                ],
                'notes': 'Main M1B branch from Yenikapı to Kirazlı, connects to M3 at Kirazlı'
            },
            'M2': {
                'name': 'M2 Yenikapı - Hacıosman',
                'color': 'green',
                'stations': [
                    {'name': 'Yenikapı', 'lat': 41.0035, 'lng': 28.9510},
                    {'name': 'Vezneciler', 'lat': 41.0130, 'lng': 28.9545},
                    {'name': 'Haliç', 'lat': 41.0200, 'lng': 28.9650},
                    {'name': 'Şişhane', 'lat': 41.0268, 'lng': 28.9737},
                    {'name': 'Taksim', 'lat': 41.0370, 'lng': 28.9850},
                    {'name': 'Osmanbey', 'lat': 41.0478, 'lng': 28.9885},
                    {'name': 'Şişli-Mecidiyeköy', 'lat': 41.0642, 'lng': 28.9956},
                    {'name': 'Gayrettepe', 'lat': 41.0689, 'lng': 29.0089},
                    {'name': 'Levent', 'lat': 41.0788, 'lng': 29.0103},
                    {'name': 'Sanayi Mahallesi', 'lat': 41.0856, 'lng': 29.0142},
                    {'name': 'ITÜ-Ayazağa', 'lat': 41.1050, 'lng': 29.0231},
                    {'name': 'Atatürk Oto Sanayi', 'lat': 41.1167, 'lng': 29.0281},
                    {'name': 'Darüşşafaka', 'lat': 41.1250, 'lng': 29.0342},
                    {'name': 'Hacıosman', 'lat': 41.1358, 'lng': 29.0428},
                ],
                'notes': 'Main north-south line through European side, connects to M6 at Levent'
            },
            'M3': {
                'name': 'M3 Kirazlı - Başakşehir/Kayaşehir',
                'color': 'blue',
                'stations': [
                    {'name': 'Kirazlı', 'lat': 41.0285, 'lng': 28.8264},
                    {'name': 'Mahmutbey', 'lat': 41.0456, 'lng': 28.8089},
                    {'name': 'İstoç', 'lat': 41.0567, 'lng': 28.7956},
                    {'name': 'İkitelli Sanayi', 'lat': 41.0678, 'lng': 28.7825},
                    {'name': 'Turgut Özal', 'lat': 41.0756, 'lng': 28.7714},
                    {'name': 'Siteler', 'lat': 41.0828, 'lng': 28.7603},
                    {'name': 'Başak Konutları', 'lat': 41.0892, 'lng': 28.7492},
                    {'name': 'Başakşehir', 'lat': 41.0956, 'lng': 28.7381},
                    {'name': 'Olimpiyat', 'lat': 41.1028, 'lng': 28.7270},
                    {'name': 'Kayaşehir Merkez', 'lat': 41.1089, 'lng': 28.7156},
                ],
                'notes': 'Connects to M1B at Kirazlı, serves western suburbs'
            },
            'M4': {
                'name': 'M4 Kadıköy - Sabiha Gökçen Havalimanı',
                'color': 'pink',
                'stations': [
                    {'name': 'Kadıköy', 'lat': 40.9900, 'lng': 29.0250},
                    {'name': 'Ayrılık Çeşmesi', 'lat': 40.9850, 'lng': 29.0350},
                    {'name': 'Acıbadem', 'lat': 40.9806, 'lng': 29.0489},
                    {'name': 'Ünalan', 'lat': 40.9778, 'lng': 29.0625},
                    {'name': 'Göztepe', 'lat': 40.9750, 'lng': 29.0750},
                    {'name': 'Yenisahra', 'lat': 40.9729, 'lng': 29.0892},
                    {'name': 'Kozyatağı', 'lat': 40.9689, 'lng': 29.1000},
                    {'name': 'Bostancı', 'lat': 40.9536, 'lng': 29.1086},
                    {'name': 'Küçükyalı', 'lat': 40.9289, 'lng': 29.1231},
                    {'name': 'Maltepe', 'lat': 40.9125, 'lng': 29.1369},
                    {'name': 'Huzurevi', 'lat': 40.9028, 'lng': 29.1453},
                    {'name': 'Gülsuyu', 'lat': 40.8989, 'lng': 29.1594},
                    {'name': 'Esenkent', 'lat': 40.8978, 'lng': 29.1711},
                    {'name': 'Hastane-Adliye', 'lat': 40.8961, 'lng': 29.1811},
                    {'name': 'Kartal', 'lat': 40.8956, 'lng': 29.1850},
                    {'name': 'Yakacık-Adnan Kahveci', 'lat': 40.8864, 'lng': 29.2019},
                    {'name': 'Pendik', 'lat': 40.8786, 'lng': 29.2389},
                    {'name': 'Tavşantepe', 'lat': 40.8967, 'lng': 29.3133},
                    {'name': 'Sabiha Gökçen Havalimanı', 'lat': 40.8986, 'lng': 29.3092},
                ],
                'notes': 'Asian side metro, connects to Marmaray at Ayrılık Çeşmesi and M8 at Bostancı'
            },
            'M5': {
                'name': 'M5 Üsküdar - Çekmeköy/Samandıra',
                'color': 'purple',
                'stations': [
                    {'name': 'Üsküdar', 'lat': 41.0226, 'lng': 29.0150},
                    {'name': 'Fistikağacı', 'lat': 41.0289, 'lng': 29.0325},
                    {'name': 'Bağlarbaşı', 'lat': 41.0336, 'lng': 29.0456},
                    {'name': 'Altunizade', 'lat': 41.0389, 'lng': 29.0578},
                    {'name': 'Kısıklı', 'lat': 41.0456, 'lng': 29.0725},
                    {'name': 'Bulgurlu', 'lat': 41.0525, 'lng': 29.0872},
                    {'name': 'Ümraniye', 'lat': 41.0200, 'lng': 29.1100},
                    {'name': 'Çarşı', 'lat': 41.0267, 'lng': 29.1225},
                    {'name': 'Yamanevler', 'lat': 41.0336, 'lng': 29.1356},
                    {'name': 'Çakmak', 'lat': 41.0403, 'lng': 29.1489},
                    {'name': 'Küçükbakkalköy', 'lat': 41.0478, 'lng': 29.1625},
                    {'name': 'İnönü', 'lat': 41.0550, 'lng': 29.1758},
                    {'name': 'Çekmeköy', 'lat': 41.0628, 'lng': 29.1892},
                    {'name': 'Sultançiftliği', 'lat': 41.0700, 'lng': 29.2028},
                    {'name': 'Samandıra', 'lat': 41.0778, 'lng': 29.2164},
                ],
                'notes': 'Asian side metro, connects Üsküdar to eastern suburbs'
            },
            'M6': {
                'name': 'M6 Levent - Boğaziçi Üniversitesi/Hisarüstü',
                'color': 'brown',
                'stations': [
                    {'name': 'Levent', 'lat': 41.0788, 'lng': 29.0103},
                    {'name': 'Nispetiye', 'lat': 41.0856, 'lng': 29.0225},
                    {'name': 'Etiler', 'lat': 41.0925, 'lng': 29.0347},
                    {'name': 'Boğaziçi Üniversitesi/Hisarüstü', 'lat': 41.0994, 'lng': 29.0469},
                ],
                'notes': 'Short line, connects M2 at Levent to Boğaziçi University'
            },
            'M7': {
                'name': 'M7 Mecidiyeköy - Mahmutbey',
                'color': 'pink',
                'stations': [
                    {'name': 'Mecidiyeköy', 'lat': 41.0642, 'lng': 28.9956},
                    {'name': 'Çağlayan', 'lat': 41.0589, 'lng': 28.9814},
                    {'name': 'Kağıthane', 'lat': 41.0536, 'lng': 28.9672},
                    {'name': 'Nurtepe', 'lat': 41.0483, 'lng': 28.9530},
                    {'name': 'Alibeyköy', 'lat': 41.0430, 'lng': 28.9388},
                    {'name': 'Çırçır', 'lat': 41.0377, 'lng': 28.9246},
                    {'name': 'Veysel Karani-Akşemsettin', 'lat': 41.0336, 'lng': 28.9128},
                    {'name': 'Yeşilpınar', 'lat': 41.0324, 'lng': 28.9104},
                    {'name': 'Kazım Karabekir', 'lat': 41.0375, 'lng': 28.8756},
                    {'name': 'Yenimahalle', 'lat': 41.0425, 'lng': 28.8614},
                    {'name': 'Karadeniz Mahallesi', 'lat': 41.0475, 'lng': 28.8472},
                    {'name': 'Tekstilkent', 'lat': 41.0525, 'lng': 28.8330},
                    {'name': 'Göztepe Mahallesi', 'lat': 41.0575, 'lng': 28.8188},
                    {'name': 'İstoç', 'lat': 41.0567, 'lng': 28.7956},
                    {'name': 'Mahmutbey', 'lat': 41.0456, 'lng': 28.8089},
                ],
                'notes': 'Connects M2 at Mecidiyeköy to M3 at Mahmutbey and İstoç'
            },
            'M8': {
                'name': 'M8 Bostancı - Parseller',
                'color': 'cyan',
                'stations': [
                    {'name': 'Bostancı', 'lat': 40.9536, 'lng': 29.1086},
                    {'name': 'Kozyatağı', 'lat': 40.9689, 'lng': 29.1000},
                    {'name': 'Küçükbakkalköy', 'lat': 40.9736, 'lng': 29.1142},
                    {'name': 'İdealtepe', 'lat': 40.9764, 'lng': 29.1214},
                    {'name': 'Soğanlık', 'lat': 40.9783, 'lng': 29.1284},
                    {'name': 'Ferhafpaşa', 'lat': 40.9806, 'lng': 29.1356},
                    {'name': 'Parseller', 'lat': 40.9830, 'lng': 29.1426},
                ],
                'notes': 'Asian side line connecting M4 at Bostancı to inner suburbs'
            },
            'M9': {
                'name': 'M9 Ataköy - İkitelli',
                'color': 'yellow',
                'stations': [
                    {'name': 'Ataköy', 'lat': 40.9814, 'lng': 28.8403},
                    {'name': 'Şirinevler', 'lat': 40.9900, 'lng': 28.8350},
                    {'name': 'Yenibosna', 'lat': 40.9647, 'lng': 28.8236},
                    {'name': 'Mimar Sinan', 'lat': 40.9950, 'lng': 28.8489},
                    {'name': 'Bahariye', 'lat': 41.0000, 'lng': 28.8567},
                    {'name': 'Bahçelievler', 'lat': 40.9989, 'lng': 28.8567},
                    {'name': 'Mahmutbey', 'lat': 41.0456, 'lng': 28.8089},
                    {'name': 'İstoç', 'lat': 41.0567, 'lng': 28.7956},
                    {'name': 'İkitelli Sanayi', 'lat': 41.0678, 'lng': 28.7825},
                ],
                'notes': 'Connects Ataköy to M3/M7 at Mahmutbey and İkitelli'
            },
            'Marmaray': {
                'name': 'Marmaray (Halkalı - Gebze)',
                'color': 'red',
                'stations': [
                    {'name': 'Halkalı', 'lat': 41.0089, 'lng': 28.6092},
                    {'name': 'Mustafa Kemal', 'lat': 41.0056, 'lng': 28.6325},
                    {'name': 'Küçükçekmece', 'lat': 41.0023, 'lng': 28.6558},
                    {'name': 'Yenimahalle', 'lat': 40.9990, 'lng': 28.6791},
                    {'name': 'Florya', 'lat': 40.9957, 'lng': 28.7024},
                    {'name': 'Yeşilköy', 'lat': 40.9924, 'lng': 28.7257},
                    {'name': 'Ataköy', 'lat': 40.9891, 'lng': 28.7490},
                    {'name': 'Bakırköy', 'lat': 40.9858, 'lng': 28.7723},
                    {'name': 'Yenikapı', 'lat': 41.0035, 'lng': 28.9510},
                    {'name': 'Sirkeci', 'lat': 41.0176, 'lng': 28.9765},
                    {'name': 'Üsküdar', 'lat': 41.0226, 'lng': 29.0150},
                    {'name': 'Ayrılık Çeşmesi', 'lat': 40.9850, 'lng': 29.0350},
                    {'name': 'Söğütlüçeşme', 'lat': 40.9750, 'lng': 29.0550},
                    {'name': 'Feneryolu', 'lat': 40.9650, 'lng': 29.0750},
                    {'name': 'Göztepe', 'lat': 40.9550, 'lng': 29.0950},
                    {'name': 'Erenköy', 'lat': 40.9450, 'lng': 29.1150},
                    {'name': 'Suadiye', 'lat': 40.9350, 'lng': 29.1350},
                    {'name': 'Bostancı', 'lat': 40.9250, 'lng': 29.1550},
                    {'name': 'Küçükyalı', 'lat': 40.9150, 'lng': 29.1750},
                    {'name': 'Maltepe', 'lat': 40.9050, 'lng': 29.1950},
                    {'name': 'Cevizli', 'lat': 40.8950, 'lng': 29.2150},
                    {'name': 'Pendik', 'lat': 40.8786, 'lng': 29.2389},
                    {'name': 'Kartal', 'lat': 40.8650, 'lng': 29.2750},
                    {'name': 'Gebze', 'lat': 40.8028, 'lng': 29.4308},
                ],
                'type': 'underground_rail',
                'notes': 'Cross-continental rail line connecting European and Asian sides via underwater Bosphorus tunnel'
            },
        }
        
        # Tram lines (all official Istanbul tram lines with complete stations)
        self.tram_lines = {
            'T1': {
                'name': 'T1 Kabataş - Bağcılar',
                'color': 'blue',
                'stations': [
                    {'name': 'Kabataş', 'lat': 41.0311, 'lng': 29.0097},
                    {'name': 'Karaköy', 'lat': 41.0242, 'lng': 28.9742},
                    {'name': 'Eminönü', 'lat': 41.0177, 'lng': 28.9742},
                    {'name': 'Sirkeci', 'lat': 41.0176, 'lng': 28.9765},
                    {'name': 'Gülhane', 'lat': 41.0128, 'lng': 28.9806},
                    {'name': 'Sultanahmet', 'lat': 41.0059, 'lng': 28.9769},
                    {'name': 'Beyazıt-Kapalıçarşı', 'lat': 41.0106, 'lng': 28.9680},
                    {'name': 'Çemberlitaş', 'lat': 41.0089, 'lng': 28.9712},
                    {'name': 'Laleli-Üniversite', 'lat': 41.0122, 'lng': 28.9583},
                    {'name': 'Aksaray', 'lat': 41.0166, 'lng': 28.9548},
                    {'name': 'Yusufpaşa', 'lat': 41.0200, 'lng': 28.9450},
                    {'name': 'Haseki', 'lat': 41.0233, 'lng': 28.9350},
                    {'name': 'Fındıkzade', 'lat': 41.0144, 'lng': 28.9311},
                    {'name': 'Pazartekke', 'lat': 41.0089, 'lng': 28.9189},
                    {'name': 'Çapa-Şehremini', 'lat': 41.0117, 'lng': 28.9217},
                    {'name': 'Cevizlibağ', 'lat': 41.0056, 'lng': 28.9056},
                    {'name': 'Topkapı-Ulubatlı', 'lat': 41.0133, 'lng': 28.9256},
                    {'name': 'Sağmalcılar', 'lat': 41.0095, 'lng': 28.9089},
                    {'name': 'Bayrampaşa-Maltepe', 'lat': 41.0314, 'lng': 28.9011},
                    {'name': 'Zeytinburnu', 'lat': 41.0050, 'lng': 28.9014},
                    {'name': 'Bağcılar-Kirazlı', 'lat': 41.0285, 'lng': 28.8264},
                ],
                'type': 'tram',
                'notes': 'Main heritage tram line through historic peninsula, connects to M1A/M1B at Aksaray and Zeytinburnu'
            },
            'T3': {
                'name': 'T3 Kadıköy - Moda Nostaljik Tramvay',
                'color': 'red',
                'stations': [
                    {'name': 'Kadıköy', 'lat': 40.9900, 'lng': 29.0250},
                    {'name': 'Rıhtım', 'lat': 40.9878, 'lng': 29.0267},
                    {'name': 'Moda', 'lat': 40.9842, 'lng': 29.0261},
                ],
                'type': 'nostalgic',
                'duration': 10,  # minutes
                'notes': 'Historic nostalgic tram, short scenic route in Kadıköy along the waterfront'
            },
            'T4': {
                'name': 'T4 Topkapı - Mescid-i Selam',
                'color': 'green',
                'stations': [
                    {'name': 'Topkapı', 'lat': 41.0133, 'lng': 28.9256},
                    {'name': 'Sağmalcılar', 'lat': 41.0095, 'lng': 28.9089},
                    {'name': 'Maltepe-Cevizlibağ', 'lat': 41.0056, 'lng': 28.9056},
                    {'name': 'Mezitabya', 'lat': 41.0183, 'lng': 28.8950},
                    {'name': 'Edirnekapı', 'lat': 41.0269, 'lng': 28.9328},
                    {'name': 'Sultançiftliği', 'lat': 41.0314, 'lng': 28.9011},
                    {'name': 'Arnavutköy', 'lat': 41.0356, 'lng': 28.8897},
                    {'name': 'Mescid-i Selam', 'lat': 41.0278, 'lng': 28.8825},
                ],
                'type': 'tram',
                'notes': 'Connects Topkapı area (M1A/T1 connection) to western suburbs along city walls'
            },
            'T5': {
                'name': 'T5 Cibali - Alibeyköy',
                'color': 'orange',
                'stations': [
                    {'name': 'Cibali', 'lat': 41.0328, 'lng': 28.9492},
                    {'name': 'Fener', 'lat': 41.0336, 'lng': 28.9489},
                    {'name': 'Balat', 'lat': 41.0344, 'lng': 28.9486},
                    {'name': 'Ayvansaray', 'lat': 41.0361, 'lng': 28.9450},
                    {'name': 'Eyüpsultan', 'lat': 41.0472, 'lng': 28.9339},
                    {'name': 'Şişhane', 'lat': 41.0503, 'lng': 28.9403},
                    {'name': 'Halıcıoğlu', 'lat': 41.0528, 'lng': 28.9428},
                    {'name': 'Sütlüce', 'lat': 41.0539, 'lng': 28.9442},
                    {'name': 'Alibeyköy', 'lat': 41.0430, 'lng': 28.9388},
                ],
                'type': 'tram',
                'notes': 'Historic Golden Horn route, connects old city to northern shore, meets M7 at Alibeyköy'
            },
            'T6': {
                'name': 'T6 İstiklal Caddesi Nostaljik Tramvay',
                'color': 'red',
                'stations': [
                    {'name': 'Taksim', 'lat': 41.0370, 'lng': 28.9850},
                    {'name': 'Tünel', 'lat': 41.0294, 'lng': 28.9745},
                ],
                'type': 'nostalgic',
                'duration': 8,  # minutes
                'notes': 'Historic nostalgic red tram along İstiklal Avenue, connects to F1 at Taksim and F2 at Tünel'
            },
        }
        
        # Funicular lines (cable cars)
        self.funicular_lines = {
            'F1': {
                'name': 'F1 Taksim - Kabataş Funicular',
                'color': 'orange',
                'stations': [
                    {'name': 'Taksim', 'lat': 41.0370, 'lng': 28.9850},
                    {'name': 'Kabataş', 'lat': 41.0311, 'lng': 29.0097},
                ],
                'type': 'funicular',
                'duration': 3,  # minutes
                'frequency': '5 minutes',
                'notes': 'Quick connection between Kabataş (sea level) and Taksim (hill top)'
            },
            'F2': {
                'name': 'F2 Karaköy - Tünel Funicular',
                'color': 'orange', 
                'stations': [
                    {'name': 'Karaköy', 'lat': 41.0242, 'lng': 28.9742},
                    {'name': 'Tünel (Beyoğlu)', 'lat': 41.0294, 'lng': 28.9745},
                ],
                'type': 'funicular',
                'duration': 2,  # minutes
                'frequency': '5 minutes',
                'notes': 'Historic funicular (1875), connects Karaköy to İstiklal Avenue'
            },
        }
        
        # Major ferry routes
        self.ferry_routes = {
            'eminonu_kadikoy': {
                'name': 'Eminönü - Kadıköy Ferry',
                'color': 'cyan',
                'stops': [
                    {'name': 'Eminönü', 'lat': 41.0177, 'lng': 28.9742},
                    {'name': 'Kadıköy', 'lat': 40.9900, 'lng': 29.0250},
                ],
                'duration': 20,  # minutes
            },
            'kabatas_uskudar': {
                'name': 'Kabataş - Üsküdar Ferry',
                'color': 'cyan',
                'stops': [
                    {'name': 'Kabataş', 'lat': 41.0311, 'lng': 29.0097},
                    {'name': 'Üsküdar', 'lat': 41.0226, 'lng': 29.0150},
                ],
                'duration': 15,  # minutes
            },
            'besiktas_kadikoy': {
                'name': 'Beşiktaş - Kadıköy Ferry',
                'color': 'cyan',
                'stops': [
                    {'name': 'Beşiktaş', 'lat': 41.0426, 'lng': 29.0050},
                    {'name': 'Kadıköy', 'lat': 40.9900, 'lng': 29.0250},
                ],
                'duration': 25,  # minutes
            },
        }
        
        # Bus routes removed - focusing on metro, tram, Marmaray, and ferry only
        # For areas not covered by rail transit, users can be advised to use taxi or other alternatives
        self.bus_routes = {}
    
    def get_directions(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str = "Start",
        end_name: str = "Destination",
        preferred_modes: Optional[List[str]] = None
    ) -> Optional[TransportRoute]:
        """
        Get detailed transportation directions using graph-based routing
        
        Args:
            start: Start coordinates (lat, lng)
            end: End coordinates (lat, lng)
            start_name: Name of start location
            end_name: Name of end location
            preferred_modes: Preferred transportation modes
            
        Returns:
            TransportRoute with detailed steps
        """
        logger.info(f"🚇 Getting directions from {start_name} to {end_name}")
        
        # Edge case: same location (within 100 meters)
        distance = self._calculate_distance(start, end)
        if distance < 0.1:  # Less than 100 meters
            logger.warning("⚠️ Start and end locations are the same or very close")
            return None
        
        # Try graph-based routing first (BEST OPTION)
        # Skip walking-only check and let the graph routing decide the best route
        if self.routing_engine:
            try:
                logger.info("🎯 Using graph-based routing engine")
                route_path = self.routing_engine.find_route(
                    start[0], start[1],
                    end[0], end[1],
                    max_transfers=3
                )
                
                if route_path:
                    logger.info(f"✅ Graph routing found route: {route_path.summary}")
                    return self._convert_graph_route_to_transport_route(
                        route_path, start, end, start_name, end_name
                    )
                else:
                    logger.warning("Graph routing found no route, trying fallback")
            except Exception as e:
                logger.error(f"Graph routing failed: {e}")
        else:
            logger.warning("Graph routing engine not available, using fallback")
        
        # If graph routing failed or not available, check if walking is feasible
        distance = self._calculate_distance(start, end)
        if distance < 2.0:  # Less than 2km, suggest walking as fallback
            logger.info(f"📏 Distance is only {distance:.2f}km, suggesting walking")
            return self._create_walking_route(start, end, start_name, end_name, distance)
        
        # Fallback to legacy routing
        transit_route = self._find_transit_route(start, end, start_name, end_name, preferred_modes)
        
        if transit_route:
            return transit_route
        
        # Final fallback to combined walking + transit
        return self._create_mixed_mode_route(start, end, start_name, end_name)
    
    def _convert_graph_route_to_transport_route(
        self,
        route_path: 'RoutePath',
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str,
        end_name: str
    ) -> TransportRoute:
        """Convert graph RoutePath to TransportRoute with detailed steps"""
        
        # Log input path characteristics
        logger.debug(f"🔄 Converting graph path: {len(route_path.nodes)} nodes, {len(route_path.edges)} edges")
        logger.debug(f"   Path nodes: {[n.name for n in route_path.nodes[:5]]}{'...' if len(route_path.nodes) > 5 else ''}")
        logger.debug(f"   Path edges: {[(e.from_node.name, e.to_node.name, e.line_id, e.mode) for e in route_path.edges[:3]]}{'...' if len(route_path.edges) > 3 else ''}")
        
        steps = []
        
        # Add initial walking to first station if needed
        first_node = route_path.nodes[0]
        start_distance = self._calculate_distance(start, (first_node.lat, first_node.lng))
        
        if start_distance > 0.05:  # More than 50 meters
            walk_duration = max(1, int((start_distance / 5.0) * 60))  # 5 km/h walking
            steps.append(
                TransportStep(
                    mode='walk',
                    instruction=f"Walk to {first_node.name}",
                    distance=start_distance * 1000,
                    duration=walk_duration,
                    start_location=start,
                    end_location=(first_node.lat, first_node.lng)
                )
            )
        
        # Process edges to create transit/transfer steps
        current_line = None
        segment_start_node = None
        segment_nodes = []
        segment_duration = 0
        segment_distance = 0
        
        logger.debug(f"   Processing {len(route_path.edges)} edges to create transit steps...")
        
        for i, edge in enumerate(route_path.edges):
            from_node = edge.from_node
            to_node = edge.to_node
            
            # Start of a new segment or first edge
            if segment_start_node is None:
                segment_start_node = from_node
                current_line = edge.line_id
                segment_nodes = [from_node]
                logger.debug(f"   🆕 Starting new segment at {from_node.name} on line {current_line}")
            
            segment_nodes.append(to_node)
            segment_duration += edge.duration
            segment_distance += edge.distance
            
            # Check if we need to create a step (line change or transfer)
            is_last_edge = (i == len(route_path.edges) - 1)
            next_edge = route_path.edges[i + 1] if not is_last_edge else None
            
            should_create_step = False
            if is_last_edge:
                should_create_step = True
                logger.debug(f"   ✅ Last edge reached at {to_node.name}")
            elif edge.edge_type == 'transfer':
                should_create_step = True
                logger.debug(f"   🔄 Transfer edge detected: {from_node.name} → {to_node.name}")
            elif next_edge and next_edge.line_id != current_line:
                should_create_step = True
                logger.debug(f"   🔄 Line change detected: {current_line} → {next_edge.line_id} at {to_node.name}")

            
            if should_create_step and segment_start_node:
                # Create appropriate step
                if edge.edge_type == 'transfer':
                    # Transfer/walking step
                    logger.debug(f"   ➕ Creating transfer step: {from_node.name} → {to_node.name} ({edge.duration}min)")
                    steps.append(
                        TransportStep(
                            mode='walk',
                            instruction=f"Transfer to {to_node.name}",
                            distance=edge.distance,
                            duration=edge.duration,
                            start_location=(from_node.lat, from_node.lng),
                            end_location=(to_node.lat, to_node.lng),
                            line_name=None  # Explicitly set to None for clarity
                        )
                    )
                else:
                    # Transit step
                    mode_emoji = {
                        'metro': '🚇',
                        'tram': '🚊',
                        'ferry': '⛴️',
                        'funicular': '🚡'
                    }.get(edge.mode, '🚉')
                    
                    line_name = self._get_line_name(current_line, edge.mode)
                    stops_count = len(segment_nodes) - 1
                    
                    instruction = f"Take {line_name} from {segment_start_node.name} to {to_node.name}"
                    if stops_count > 1:
                        instruction += f" ({stops_count} stops)"
                    
                    logger.debug(f"   ➕ Creating transit step: {mode_emoji} {line_name}")
                    logger.debug(f"      From: {segment_start_node.name} → To: {to_node.name}")
                    logger.debug(f"      Stops: {stops_count}, Duration: {segment_duration}min, Distance: {segment_distance:.0f}m")
                    logger.debug(f"      Segment nodes: {[n.name for n in segment_nodes]}")
                    
                    steps.append(
                        TransportStep(
                            mode=edge.mode,
                            instruction=instruction,
                            distance=segment_distance,
                            duration=segment_duration,
                            start_location=(segment_start_node.lat, segment_start_node.lng),
                            end_location=(to_node.lat, to_node.lng),
                            line_name=line_name,
                            stops_count=stops_count,
                            waypoints=[(node.lat, node.lng) for node in segment_nodes]
                        )
                    )

                
                # Reset for next segment
                segment_start_node = None if edge.edge_type == 'transfer' else to_node
                current_line = None if edge.edge_type == 'transfer' else (next_edge.line_id if next_edge else None)
                segment_nodes = [] if edge.edge_type == 'transfer' else [to_node]
                segment_duration = 0
                segment_distance = 0
        
        # Add final walking from last station if needed
        last_node = route_path.nodes[-1]
        end_distance = self._calculate_distance((last_node.lat, last_node.lng), end)
        
        if end_distance > 0.05:  # More than 50 meters
            walk_duration = max(1, int((end_distance / 5.0) * 60))
            steps.append(
                TransportStep(
                    mode='walk',
                    instruction=f"Walk to {end_name}",
                    distance=end_distance * 1000,
                    duration=walk_duration,
                    start_location=(last_node.lat, last_node.lng),
                    end_location=end
                )
            )
        
        # Build enhanced summary
        total_duration = sum(step.duration for step in steps)
        total_distance = sum(step.distance for step in steps)
        modes_used = list(dict.fromkeys([step.mode for step in steps]))
        
        # Count steps and walking distance
        total_steps = len(steps)
        walking_distance = sum(step.distance for step in steps if step.mode == 'walk')
        transit_steps = [s for s in steps if s.mode != 'walk']
        transfers = len(transit_steps) - 1 if len(transit_steps) > 1 else 0
        
        # Build main summary
        summary = f"{total_duration} min via "
        mode_names = []
        for mode in modes_used:
            if mode != 'walk':
                mode_names.append(mode.capitalize())
        summary += " → ".join(mode_names) if mode_names else "walking"
        
        # Add details
        if transfers > 0:
            summary += f" ({transfers} transfer{'s' if transfers > 1 else ''})"
        if walking_distance > 100:  # More than 100 meters of walking
            summary += f", {walking_distance/1000:.1f}km walking"
        
        # Create route object
        route = TransportRoute(
            steps=steps,
            total_distance=total_distance,
            total_duration=total_duration,
            summary=summary,
            modes_used=modes_used
        )
        
        # Calculate and add cost
        route.estimated_cost = self._calculate_fare(route)
        
        # Log final route characteristics for debugging
        transit_sequence = []
        for step in steps:
            if step.mode != 'walk':
                transit_sequence.append(f"{step.mode}:{step.line_name}")
        
        route_signature = "_".join(transit_sequence) if transit_sequence else "walking_only"
        logger.debug(f"✅ Route converted: {total_duration}min, {len(steps)} steps, ₺{route.estimated_cost:.2f}")
        logger.debug(f"   Transit sequence: {transit_sequence}")
        logger.debug(f"   Route signature: {route_signature}")
        logger.debug(f"   Summary: {summary}")
        
        return route
    
    def _get_line_name(self, line_id: Optional[str], mode: str) -> str:
        """Get full line name from line ID"""
        if not line_id:
            return mode.capitalize()
        
        # Check all line types (metro, tram, Marmaray, funicular, ferry only)
        all_lines = {
            **self.metro_lines,
            **self.tram_lines,
            **self.funicular_lines,
            **self.ferry_routes
        }
        
        if line_id in all_lines:
            return all_lines[line_id].get('name', line_id)
        
        return line_id
    
    def _create_walking_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str,
        end_name: str,
        distance: float
    ) -> TransportRoute:
        """Create a walking-only route"""
        
        # Get OSRM walking route if available
        waypoints = [start, end]
        if self.osrm:
            try:
                osrm_route = self.osrm.get_route(start, end)
                if osrm_route and osrm_route.waypoints:
                    waypoints = osrm_route.waypoints
                    distance = osrm_route.total_distance / 1000.0  # Convert to km
            except Exception as e:
                logger.warning(f"OSRM fallback failed: {e}")
        
        # Estimate duration (average walking speed 5 km/h)
        duration = int((distance / 5.0) * 60)  # minutes
        
        steps = [
            TransportStep(
                mode='walk',
                instruction=f"Walk from {start_name} to {end_name}",
                distance=distance * 1000,  # meters
                duration=duration,
                start_location=start,
                end_location=end,
                waypoints=waypoints
            )
        ]
        
        route = TransportRoute(
            steps=steps,
            total_distance=distance * 1000,
            total_duration=duration,
            summary=f"Walk to {end_name} ({distance:.1f} km, {duration} min)",
            modes_used=['walk'],
            estimated_cost=0.0  # Walking is free
        )
        
        return route
    
    def _find_transit_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str,
        end_name: str,
        preferred_modes: Optional[List[str]] = None
    ) -> Optional[TransportRoute]:
        """Find transit route using metro/tram/ferry"""
        
        # Find nearest stations to start and end
        start_station = self._find_nearest_station(start)
        end_station = self._find_nearest_station(end)
        
        if not start_station or not end_station:
            return None
        
        # Check if they're on the same line
        route = self._find_direct_line(start_station, end_station)
        if route:
            # Add walking segments to/from stations
            return self._build_complete_route(
                start, end, start_name, end_name,
                start_station, end_station, route
            )
        
        # Check for transfer routes
        transfer_route = self._find_transfer_route(start_station, end_station)
        if transfer_route:
            return self._build_complete_route(
                start, end, start_name, end_name,
                start_station, end_station, transfer_route
            )
        
        return None
    
    def _create_mixed_mode_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str,
        end_name: str
    ) -> TransportRoute:
        """Create a route combining walking and basic transit info"""
        
        distance = self._calculate_distance(start, end)
        
        # Provide general guidance
        steps = []
        
        # Walking to nearest station
        nearest_station = self._find_nearest_station(start)
        if nearest_station:
            walk_dist = self._calculate_distance(start, (nearest_station['lat'], nearest_station['lng']))
            walk_time = int((walk_dist / 5.0) * 60)
            
            steps.append(
                TransportStep(
                    mode='walk',
                    instruction=f"Walk to {nearest_station['name']} station",
                    distance=walk_dist * 1000,
                    duration=walk_time,
                    start_location=start,
                    end_location=(nearest_station['lat'], nearest_station['lng'])
                )
            )
            
            # Transit suggestion
            steps.append(
                TransportStep(
                    mode='metro',
                    instruction=f"Take {nearest_station.get('line', 'metro')} towards your destination",
                    distance=distance * 1000 * 0.7,  # Estimate
                    duration=int(distance * 4),  # Rough estimate
                    start_location=(nearest_station['lat'], nearest_station['lng']),
                    end_location=end,
                    line_name=nearest_station.get('line', 'Metro')
                )
            )
        
        total_duration = sum(step.duration for step in steps)
        
        route = TransportRoute(
            steps=steps,
            total_distance=distance * 1000,
            total_duration=total_duration,
            summary=f"Combined route: {total_duration} min",
            modes_used=['walk', 'metro']
        )
        
        # Calculate cost for mixed routes too
        route.estimated_cost = self._calculate_fare(route)
        
        return route
    
    def _find_nearest_station(self, location: Tuple[float, float]) -> Optional[Dict]:
        """Find nearest metro/tram station to a location"""
        nearest = None
        min_distance = float('inf')
        
        # Check metro stations
        for line_id, line_data in self.metro_lines.items():
            for station in line_data['stations']:
                dist = self._calculate_distance(
                    location,
                    (station['lat'], station['lng'])
                )
                if dist < min_distance:
                    min_distance = dist
                    nearest = {**station, 'line': f"{line_id} {line_data['name']}", 'type': 'metro'}
        
        # Check tram stations
        for line_id, line_data in self.tram_lines.items():
            for station in line_data['stations']:
                dist = self._calculate_distance(
                    location,
                    (station['lat'], station['lng'])
                )
                if dist < min_distance:
                    min_distance = dist
                    nearest = {**station, 'line': f"{line_id} {line_data['name']}", 'type': 'tram'}
        
        return nearest if min_distance < 2.0 else None  # Max 2km to station
    
    def _find_direct_line(self, start_station: Dict, end_station: Dict) -> Optional[Dict]:
        """Check if two stations are on the same line"""
        # Implementation would check if both stations exist on same line
        # For now, return None to use transfer logic
        return None
    
    def _find_transfer_route(self, start_station: Dict, end_station: Dict) -> Optional[Dict]:
        """Find route with transfers between lines"""
        # Simplified implementation
        # Real implementation would use graph search
        return None
    
    def _build_complete_route(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str,
        end_name: str,
        start_station: Dict,
        end_station: Dict,
        transit_info: Dict
    ) -> TransportRoute:
        """Build complete route with walking + transit segments"""
        steps = []
        
        # Walking to start station
        walk_to_station_dist = self._calculate_distance(start, (start_station['lat'], start_station['lng']))
        steps.append(
            TransportStep(
                mode='walk',
                instruction=f"Walk to {start_station['name']} station",
                distance=walk_to_station_dist * 1000,
                duration=int((walk_to_station_dist / 5.0) * 60),
                start_location=start,
                end_location=(start_station['lat'], start_station['lng'])
            )
        )
        
        # Transit segment
        transit_dist = self._calculate_distance(
            (start_station['lat'], start_station['lng']),
            (end_station['lat'], end_station['lng'])
        )
        steps.append(
            TransportStep(
                mode=start_station['type'],
                instruction=f"Take {start_station['line']} to {end_station['name']}",
                distance=transit_dist * 1000,
                duration=int(transit_dist * 3),  # Rough estimate
                start_location=(start_station['lat'], start_station['lng']),
                end_location=(end_station['lat'], end_station['lng']),
                line_name=start_station['line']
            )
        )
        
        # Walking from end station
        walk_from_station_dist = self._calculate_distance((end_station['lat'], end_station['lng']), end)
        steps.append(
            TransportStep(
                mode='walk',
                instruction=f"Walk to {end_name}",
                distance=walk_from_station_dist * 1000,
                duration=int((walk_from_station_dist / 5.0) * 60),
                start_location=(end_station['lat'], end_station['lng']),
                end_location=end
            )
        )
        
        total_duration = sum(step.duration for step in steps)
        
        return TransportRoute(
            steps=steps,
            total_distance=sum(step.distance for step in steps),
            total_duration=total_duration,
            summary=f"Via {start_station['line']}: {total_duration} min",
            modes_used=[step.mode for step in steps]
        )
    
    def _calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates in kilometers (Haversine formula)"""
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1 = radians(loc1[0]), radians(loc1[1])
        lat2, lon2 = radians(loc2[0]), radians(loc2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return 6371.0 * c  # Earth radius in kilometers
    
    def format_directions_text(self, route: TransportRoute) -> str:
        """Format route as detailed text directions (Google Maps style)"""
        
        if not route or not route.steps:
            return "No route available"
        
        lines = []
        lines.append(f"📍 **{route.summary}**")
        lines.append(f"⏱️ Total time: {route.total_duration} min")
        lines.append(f"📏 Total distance: {route.total_distance/1000:.1f} km")
        lines.append("")
        
        for i, step in enumerate(route.steps, 1):
            icon = {
                'walk': '🚶',
                'metro': '🚇',
                'tram': '🚊',
                'ferry': '⛴️',
                'funicular': '🚡'
            }.get(step.mode, '➡️')
            
            lines.append(f"{i}. {icon} **{step.instruction}**")
            lines.append(f"   📏 {step.distance/1000:.1f} km • ⏱️ {step.duration} min")
            
            if step.line_name:
                lines.append(f"   🚇 Line: {step.line_name}")
            if step.stops_count:
                lines.append(f"   🛑 {step.stops_count} stops")
            lines.append("")
        
        return "\n".join(lines)
    
    def _calculate_fare(self, route: TransportRoute) -> float:
        """
        Calculate fare for a route based on Istanbul transit pricing
        
        Istanbul transit fare structure (2025):
        - Base fare: ₺15.00 (first boarding)
        - Transfer within 2 hours: ₺7.50 per transfer
        - Students: 50% discount
        - Walking is free
        
        Args:
            route: TransportRoute object
            
        Returns:
            Total fare in Turkish Lira (₺)
        """
        base_fare = 15.00
        transfer_fare = 7.50
        
        # Count non-walking transit segments
        transit_segments = [step for step in route.steps if step.mode != 'walk']
        
        if len(transit_segments) == 0:
            return 0.0  # Walking only
        
        # Base fare + transfer fares
        transfers = len(transit_segments) - 1
        total_fare = base_fare + (transfers * transfer_fare)
        
        return total_fare
    
# Singleton instance
_service_instance = None

def get_transportation_service() -> TransportationDirectionsService:
    """Get singleton instance of transportation service"""
    global _service_instance
    if _service_instance is None:
        _service_instance = TransportationDirectionsService()
    return _service_instance
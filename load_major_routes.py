"""
Option B: Manual Quick Start - Major Istanbul Routes Loader
Priority: Marmaray + Metro Lines (most important for routing advice)
Then: Major bus routes, ferries, trams

This creates a working routing system TODAY with the most critical routes.
"""

import asyncio
import sys
import logging
from typing import List, Tuple
from services.route_network_builder import (
    TransportationNetwork, TransportStop, TransportLine, RouteNetworkBuilder
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MajorRoutesLoader:
    """Load major Istanbul routes manually for quick routing capability"""
    
    def __init__(self):
        self.network = TransportationNetwork()
        self.stats = {
            'stops_added': 0,
            'lines_added': 0,
            'marmaray_lines': 0,
            'metro_lines': 0,
            'bus_lines': 0,
            'ferry_lines': 0,
            'tram_lines': 0
        }
    
    def add_marmaray_routes(self):
        """
        PRIORITY 1: Marmaray - The most important cross-continental rail line
        Connects Europe and Asia through the Bosphorus undersea tunnel
        """
        logger.info("\n🚇 PRIORITY 1: Loading MARMARAY (Cross-Continental Rail)")
        logger.info("   Most important route for Europe-Asia connections!")
        
        # Marmaray stations (Europe to Asia via undersea tunnel)
        marmaray_stations = [
            # European Side
            ("MAR_HAL", "Halkalı", 41.0090, 28.6450),
            ("MAR_KUC", "Küçükçekmece", 41.0130, 28.7800),
            ("MAR_ATA", "Ataköy", 40.9800, 28.8500),
            ("MAR_BAK", "Bakırköy", 40.9820, 28.8700),
            ("MAR_YEN", "Yenikapı", 41.0054, 28.9518),
            ("MAR_SIR", "Sirkeci", 41.0175, 28.9760),
            
            # UNDERSEA TUNNEL (Bosphorus Crossing!)
            
            # Asian Side
            ("MAR_USK", "Üsküdar", 41.0226, 29.0078),
            ("MAR_AYR", "Ayrılık Çeşmesi", 41.0150, 29.0300),
            ("MAR_BOG", "Bostancı", 40.9640, 29.0870),
            ("MAR_SUA", "Suadiye", 40.9580, 29.1020),
            ("MAR_PEN", "Pendik", 40.8760, 29.2370),
            ("MAR_GEB", "Gebze", 40.8030, 29.4300),
        ]
        
        # Add Marmaray stops
        for stop_id, name, lat, lon in marmaray_stations:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type='metro'  # Marmaray is part of metro system
            )
            self.network.add_stop(stop)
            self.stats['stops_added'] += 1
        
        # Create Marmaray line (full route)
        marmaray_line = TransportLine(
            line_id="MARMARAY",
            name="Marmaray (Halkalı-Gebze)",
            transport_type='metro',
            stops=[s[0] for s in marmaray_stations],
            color='#E60000'
        )
        self.network.add_line(marmaray_line)
        self.stats['lines_added'] += 1
        self.stats['marmaray_lines'] += 1
        
        logger.info(f"   ✓ Loaded Marmaray: {len(marmaray_stations)} stations")
        logger.info(f"   ✓ Europe-Asia connection via Bosphorus tunnel!")
    
    def add_metro_routes(self):
        """
        PRIORITY 2: Metro Lines - Istanbul's backbone rapid transit
        Essential for routing throughout the city
        """
        logger.info("\n🚇 PRIORITY 2: Loading METRO LINES")
        
        # M1A: Yenikapı - Atatürk Havalimanı (Airport Line - CRITICAL!)
        m1a_stations = [
            ("M1A_YEN", "Yenikapı", 41.0054, 28.9518),
            ("M1A_AKS", "Aksaray", 41.0160, 28.9550),
            ("M1A_EMI", "Emniyet-Fatih", 41.0180, 28.9480),
            ("M1A_TOP", "Topkapı-Ulubatlı", 41.0140, 28.9220),
            ("M1A_BAY", "Bayrampaşa-Maltepe", 41.0350, 28.8990),
            ("M1A_SAG", "Sağmalcılar", 41.0470, 28.8800),
            ("M1A_OTO", "Otogar", 41.0190, 28.8750),
            ("M1A_TEV", "Terazidere", 41.0150, 28.8600),
            ("M1A_DAV", "Davutpaşa-YTÜ", 41.0170, 28.8450),
            ("M1A_MER", "Merter", 41.0150, 28.8320),
            ("M1A_ZEY", "Zeytinburnu", 41.0080, 28.9050),
            ("M1A_BAK", "Bakırköy-İncirli", 40.9950, 28.8720),
            ("M1A_BAH", "Bahçelievler", 41.0020, 28.8550),
            ("M1A_ATA", "Atatürk Havalimanı", 40.9910, 28.8170),
        ]
        
        # M1B: Yenikapı - Kirazlı (Extension)
        m1b_stations = [
            ("M1B_YEN", "Yenikapı", 41.0054, 28.9518),
            ("M1B_AKS", "Aksaray", 41.0160, 28.9550),
            ("M1B_EMI", "Emniyet-Fatih", 41.0180, 28.9480),
            ("M1B_TOP", "Topkapı-Ulubatlı", 41.0140, 28.9220),
            ("M1B_BAY", "Bayrampaşa-Maltepe", 41.0350, 28.8990),
            ("M1B_SAG", "Sağmalcılar", 41.0470, 28.8800),
            ("M1B_KIR", "Kirazlı", 41.0243, 28.8328),
        ]
        
        # M2: Yenikapı - Hacıosman (CRITICAL - Connects Taksim!)
        m2_stations = [
            ("M2_YEN", "Yenikapı", 41.0054, 28.9518),
            ("M2_VEZ", "Vezneciler", 41.0130, 28.9585),
            ("M2_HAL", "Haliç", 41.0240, 28.9650),
            ("M2_SIS", "Şişhane", 41.0287, 28.9740),
            ("M2_TAK", "Taksim", 41.0370, 28.9857),
            ("M2_OSM", "Osmanbey", 41.0486, 28.9868),
            ("M2_SIS2", "Şişli-Mecidiyeköy", 41.0602, 28.9879),
            ("M2_GAY", "Gayrettepe", 41.0680, 28.9920),
            ("M2_LEV", "Levent", 41.0782, 29.0070),
            ("M2_4LE", "4.Levent", 41.0860, 29.0120),
            ("M2_SAR", "Sariyer", 41.1020, 29.0210),
            ("M2_ITU", "İTÜ-Ayazağa", 41.1060, 29.0240),
            ("M2_HAC", "Hacıosman", 41.1095, 29.0272),
        ]
        
        # M3: Kirazlı - Başakşehir-Metrokent-Olimpiyat
        m3_stations = [
            ("M3_KIR", "Kirazlı", 41.0243, 28.8328),
            ("M3_MET", "Metrokent", 41.0410, 28.8100),
            ("M3_BAS", "Başakşehir", 41.0680, 28.8000),
            ("M3_OLI", "Olimpiyat", 41.0750, 28.8090),
        ]
        
        # M4: Kadıköy - Tavşantepe (ASIAN SIDE - CRITICAL!)
        m4_stations = [
            ("M4_KAD", "Kadıköy", 40.9905, 29.0250),
            ("M4_AYR", "Ayrılık Çeşmesi", 40.9950, 29.0320),
            ("M4_ACI", "Acıbadem", 41.0020, 29.0420),
            ("M4_UNU", "Ünalan", 41.0120, 29.0580),
            ("M4_GÖZ", "Göztepe", 41.0180, 29.0680),
            ("M4_YEN2", "Yenisahra", 41.0250, 29.0850),
            ("M4_KOZ", "Kozyatağı", 41.0330, 29.0950),
            ("M4_BOŞ", "Bostancı", 40.9640, 29.0870),
            ("M4_KÜÇ", "Küçükyalı", 40.9380, 29.1120),
            ("M4_MAL", "Maltepe", 40.9220, 29.1350),
            ("M4_HUZ", "Huzurevi", 40.9080, 29.1520),
            ("M4_BÜY", "Bülent Ecevit Üniversitesi", 40.8950, 29.1680),
            ("M4_KAR", "Kartal", 40.8910, 29.1890),
            ("M4_YAK", "Yakacık-Adnan Kahveci", 40.8830, 29.2150),
            ("M4_PEN", "Pendik", 40.8760, 29.2370),
            ("M4_TAV", "Tavşantepe", 40.9898, 29.3254),
        ]
        
        # M5: Üsküdar - Çekmeköy (ASIAN SIDE)
        m5_stations = [
            ("M5_USK", "Üsküdar", 41.0226, 29.0078),
            ("M5_FIH", "Fıstıkağacı", 41.0280, 29.0220),
            ("M5_BAĞ", "Bağlarbaşı", 41.0350, 29.0380),
            ("M5_ALT", "Altunizade", 41.0420, 29.0520),
            ("M5_KIS", "Kısıklı", 41.0520, 29.0680),
            ("M5_ÇAV", "Çamlık", 41.0620, 29.0850),
            ("M5_ÜMR", "Ümraniye", 41.0180, 29.1180),
            ("M5_ÇEK", "Çekmeköy", 41.0330, 29.1420),
        ]
        
        # M6: Levent - Boğaziçi Üniversitesi/Hisarüstü
        m6_stations = [
            ("M6_LEV", "Levent", 41.0782, 29.0070),
            ("M6_NIS", "Nispetiye", 41.0820, 29.0180),
            ("M6_ETI", "Etiler", 41.0850, 29.0270),
            ("M6_BOĞ", "Boğaziçi Üniversitesi", 41.0880, 29.0420),
        ]
        
        # M7: Mecidiyeköy - Mahmutbey
        m7_stations = [
            ("M7_MEC", "Mecidiyeköy", 41.0602, 28.9879),
            ("M7_ÇAĞ", "Çağlayan", 41.0680, 28.9750),
            ("M7_KAĞ", "Kağıthane", 41.0780, 28.9720),
            ("M7_YEN3", "Yenimahalle", 41.0850, 28.9650),
            ("M7_MAH", "Mahmutbey", 41.0920, 28.8420),
        ]
        
        # M9: Ataköy - İkitelli (Under construction, but important)
        m9_stations = [
            ("M9_ATA", "Ataköy", 40.9800, 28.8500),
            ("M9_ŞEN", "Şenlikköy", 40.9750, 28.8320),
            ("M9_BAK2", "Bakırköy", 40.9820, 28.8700),
            ("M9_IKI", "İkitelli", 41.0680, 28.7820),
        ]
        
        # Add all metro lines
        metro_lines = [
            ("M1A", "M1A: Yenikapı-Atatürk Havalimanı (Airport)", m1a_stations, '#FF0000'),
            ("M1B", "M1B: Yenikapı-Kirazlı", m1b_stations, '#FF0000'),
            ("M2", "M2: Yenikapı-Hacıosman (via Taksim)", m2_stations, '#00FF00'),
            ("M3", "M3: Kirazlı-Olimpiyat", m3_stations, '#0000FF'),
            ("M4", "M4: Kadıköy-Tavşantepe (Asian Side)", m4_stations, '#FF69B4'),
            ("M5", "M5: Üsküdar-Çekmeköy", m5_stations, '#800080'),
            ("M6", "M6: Levent-Boğaziçi Üniversitesi", m6_stations, '#8B4513'),
            ("M7", "M7: Mecidiyeköy-Mahmutbey", m7_stations, '#FFA500'),
            ("M9", "M9: Ataköy-İkitelli", m9_stations, '#20B2AA'),
        ]
        
        for line_id, line_name, stations, color in metro_lines:
            # Add stops
            for stop_id, name, lat, lon in stations:
                if stop_id not in self.network.stops:
                    stop = TransportStop(
                        stop_id=stop_id,
                        name=name,
                        lat=lat,
                        lon=lon,
                        transport_type='metro'
                    )
                    self.network.add_stop(stop)
                    self.stats['stops_added'] += 1
            
            # Create line
            line = TransportLine(
                line_id=line_id,
                name=line_name,
                transport_type='metro',
                stops=[s[0] for s in stations],
                color=color
            )
            self.network.add_line(line)
            self.stats['lines_added'] += 1
            self.stats['metro_lines'] += 1
            
            logger.info(f"   ✓ Loaded {line_name}: {len(stations)} stations")
    
    def add_major_bus_routes(self):
        """Add major high-frequency bus routes"""
        logger.info("\n🚌 Loading Major Bus Routes (High-frequency)")
        
        # Major bus routes connecting key areas
        # Note: These would ideally come from İBB data, but we'll add key routes
        logger.info("   ⚠️ Bus routes require stop-level data from İBB")
        logger.info("   ℹ️ For now, buses will connect to metro/Marmaray stops")
        logger.info("   ℹ️ Full bus network will be loaded from İBB in Phase 2")
    
    def add_ferry_routes(self):
        """Add major ferry routes (Bosphorus crossings)"""
        logger.info("\n⛴️ Loading Ferry Routes (Bosphorus Crossings)")
        
        # Major ferry piers
        ferry_piers = [
            ("FER_EMI", "Eminönü", 41.0175, 28.9710),
            ("FER_KAR", "Karaköy", 41.0245, 28.9785),
            ("FER_BES", "Beşiktaş", 41.0420, 29.0070),
            ("FER_KAD", "Kadıköy", 40.9905, 29.0250),
            ("FER_USK", "Üsküdar", 41.0226, 29.0078),
            ("FER_BOS", "Bostancı", 40.9640, 29.0870),
        ]
        
        # Add ferry stops
        for stop_id, name, lat, lon in ferry_piers:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type='ferry'
            )
            self.network.add_stop(stop)
            self.stats['stops_added'] += 1
        
        # Ferry lines
        ferry_lines = [
            ("F1", "Eminönü-Kadıköy", ['FER_EMI', 'FER_KAD']),
            ("F2", "Karaköy-Kadıköy", ['FER_KAR', 'FER_KAD']),
            ("F3", "Beşiktaş-Üsküdar", ['FER_BES', 'FER_USK']),
            ("F4", "Eminönü-Üsküdar", ['FER_EMI', 'FER_USK']),
        ]
        
        for line_id, line_name, stops in ferry_lines:
            line = TransportLine(
                line_id=line_id,
                name=line_name,
                transport_type='ferry',
                stops=stops
            )
            self.network.add_line(line)
            self.stats['lines_added'] += 1
            self.stats['ferry_lines'] += 1
        
        logger.info(f"   ✓ Loaded {len(ferry_lines)} ferry routes")
    
    def add_tram_routes(self):
        """Add tram routes (T1, T4, T5)"""
        logger.info("\n🚊 Loading Tram Routes")
        
        # T1: Kabataş - Bağcılar (Historic and modern tram)
        t1_stations = [
            ("T1_KAB", "Kabataş", 41.0380, 29.0075),
            ("T1_FIN", "Fındıklı", 41.0330, 28.9920),
            ("T1_TOP2", "Tophane", 41.0280, 28.9870),
            ("T1_KAR2", "Karaköy", 41.0245, 28.9785),
            ("T1_EMI2", "Eminönü", 41.0175, 28.9710),
            ("T1_GÜL", "Gülhane", 41.0135, 28.9810),
            ("T1_SUL", "Sultanahmet", 41.0058, 28.9769),
            ("T1_BEY", "Beyazıt", 41.0105, 28.9645),
            ("T1_LAL", "Laleli-Üniversite", 41.0135, 28.9550),
            ("T1_AKS2", "Aksaray", 41.0160, 28.9550),
            ("T1_YUS", "Yusufpaşa", 41.0185, 28.9350),
            ("T1_TOP3", "Topkapı", 41.0140, 28.9220),
            ("T1_BAG", "Bağcılar", 41.0380, 28.8580),
        ]
        
        # T4: Topkapı - Mescid-i Selam
        t4_stations = [
            ("T4_TOP", "Topkapı", 41.0140, 28.9220),
            ("T4_MES", "Mescid-i Selam", 41.0080, 28.8920),
        ]
        
        # T5: Cibali - Alibeyköy
        t5_stations = [
            ("T5_CIB", "Cibali", 41.0320, 28.9560),
            ("T5_ALI", "Alibeyköy", 41.0780, 28.9420),
        ]
        
        # Add tram lines
        tram_lines = [
            ("T1", "T1: Kabataş-Bağcılar (Historic Tram)", t1_stations, '#FF4500'),
            ("T4", "T4: Topkapı-Mescid-i Selam", t4_stations, '#4B0082'),
            ("T5", "T5: Cibali-Alibeyköy", t5_stations, '#00CED1'),
        ]
        
        for line_id, line_name, stations, color in tram_lines:
            # Add stops
            for stop_id, name, lat, lon in stations:
                if stop_id not in self.network.stops:
                    stop = TransportStop(
                        stop_id=stop_id,
                        name=name,
                        lat=lat,
                        lon=lon,
                        transport_type='tram'
                    )
                    self.network.add_stop(stop)
                    self.stats['stops_added'] += 1
            
            # Create line
            line = TransportLine(
                line_id=line_id,
                name=line_name,
                transport_type='tram',
                stops=[s[0] for s in stations],
                color=color
            )
            self.network.add_line(line)
            self.stats['lines_added'] += 1
            self.stats['tram_lines'] += 1
            
            logger.info(f"   ✓ Loaded {line_name}: {len(stations)} stations")
    
    def add_transfer_connections(self):
        """
        Add critical transfer connections between different lines
        These are the KEY connections that make multi-modal routing work!
        """
        logger.info("\n🔄 Adding Transfer Connections at Major Hubs...")
        
        # CRITICAL TRANSFERS - These enable Marmaray/Metro integration!
        transfers = [
            # YENIKAPY - THE BIGGEST HUB (Marmaray + M1A + M1B + M2)
            ("MAR_YEN", "M1A_YEN", "same_station", 50, 2),
            ("MAR_YEN", "M1B_YEN", "same_station", 50, 2),
            ("MAR_YEN", "M2_YEN", "same_station", 100, 3),
            ("M1A_YEN", "M1B_YEN", "same_station", 30, 1),
            ("M1A_YEN", "M2_YEN", "same_station", 80, 3),
            ("M1B_YEN", "M2_YEN", "same_station", 80, 3),
            
            # ÜSKÜDAR - Major Asian hub (Marmaray + M5)
            ("MAR_USK", "M5_USK", "same_station", 80, 3),
            ("MAR_USK", "FER_USK", "walking", 200, 5),
            ("M5_USK", "FER_USK", "walking", 180, 4),
            
            # KADIKÖY - Asian hub (M4 + Ferry)
            ("M4_KAD", "FER_KAD", "walking", 150, 4),
            
            # AKSARAY - Metro + Tram hub (M1A + M1B + T1)
            ("M1A_AKS", "M1B_AKS", "same_station", 30, 1),
            ("M1A_AKS", "T1_AKS2", "walking", 120, 3),
            ("M1B_AKS", "T1_AKS2", "walking", 120, 3),
            
            # KIRAZLI - Metro transfer (M1B + M3)
            ("M1B_KIR", "M3_KIR", "same_station", 80, 3),
            
            # LEVENT - Metro transfer (M2 + M6)
            ("M2_LEV", "M6_LEV", "same_station", 100, 3),
            
            # MECİDİYEKÖY - Metro transfer (M2 + M7)
            ("M2_SIS2", "M7_MEC", "same_station", 100, 3),
            
            # AYRILIK ÇEŞMESİ - Marmaray + M4 (Asian side)
            ("MAR_AYR", "M4_AYR", "walking", 150, 4),
            
            # BOSTANCI - Marmaray + M4 + Ferry
            ("MAR_BOG", "M4_BOŞ", "walking", 200, 5),
            ("M4_BOŞ", "FER_BOS", "walking", 300, 7),
            ("MAR_BOG", "FER_BOS", "walking", 350, 8),
            
            # PENDİK - Marmaray + M4
            ("MAR_PEN", "M4_PEN", "walking", 180, 5),
            
            # TOPKAPI - Multiple connections (M1A + M1B + T1 + T4)
            ("M1A_TOP", "M1B_TOP", "same_station", 30, 1),
            ("M1A_TOP", "T1_TOP3", "walking", 150, 4),
            ("M1B_TOP", "T1_TOP3", "walking", 150, 4),
            ("T1_TOP3", "T4_TOP", "same_station", 50, 2),
            
            # KARAKÖY - Ferry + Tram
            ("FER_KAR", "T1_KAR2", "walking", 100, 3),
            
            # EMİNÖNÜ - Ferry + Tram
            ("FER_EMI", "T1_EMI2", "walking", 80, 2),
            
            # KABATAŞ - Tram + Ferry (near Beşiktaş)
            ("T1_KAB", "FER_BES", "walking", 800, 15),
            
            # BAKIRKÖY - Marmaray + M1A + M9
            ("MAR_BAK", "M1A_BAK", "walking", 150, 4),
            ("M1A_BAK", "M9_BAK2", "walking", 200, 5),
            ("MAR_BAK", "M9_BAK2", "walking", 250, 6),
            
            # ATAKÖY - Marmaray + M9
            ("MAR_ATA", "M9_ATA", "walking", 150, 4),
            
            # BAYRAMPAŞA - Metro connections (M1A + M1B)
            ("M1A_BAY", "M1B_BAY", "same_station", 30, 1),
            
            # SAĞMALCILAR - Metro connections (M1A + M1B)
            ("M1A_SAG", "M1B_SAG", "same_station", 30, 1),
            
            # EMNİYET-FATİH - Metro connections (M1A + M1B)
            ("M1A_EMI", "M1B_EMI", "same_station", 30, 1),
        ]
        
        transfer_count = 0
        for from_stop, to_stop, transfer_type, walking_meters, duration_minutes in transfers:
            # Check if both stops exist
            if from_stop in self.network.stops and to_stop in self.network.stops:
                self.network.add_transfer(
                    from_stop_id=from_stop,
                    to_stop_id=to_stop,
                    transfer_type=transfer_type,
                    walking_meters=walking_meters,
                    duration_minutes=duration_minutes
                )
                transfer_count += 1
            else:
                missing = []
                if from_stop not in self.network.stops:
                    missing.append(from_stop)
                if to_stop not in self.network.stops:
                    missing.append(to_stop)
                logger.warning(f"   ⚠️ Skipped transfer {from_stop}-{to_stop}: Missing stops {missing}")
        
        logger.info(f"   ✓ Added {transfer_count} transfer connections")
        logger.info(f"   ✓ Key hubs connected: Yenikapı, Üsküdar, Kadıköy, Aksaray, etc.")
    
    def build_network(self):
        """Build network edges from loaded lines"""
        logger.info("\n🔗 Building Network Connections...")
        self.network.build_network()
        logger.info(f"   ✓ Created {len(self.network.edges)} edges")
    
    def save_network(self, filename='major_routes_network.json'):
        """Save network for quick loading"""
        import json
        
        network_data = {
            'stops': {sid: {
                'name': stop.name,
                'lat': stop.lat,
                'lon': stop.lon,
                'type': stop.transport_type
            } for sid, stop in self.network.stops.items()},
            'lines': {lid: {
                'name': line.name,
                'type': line.transport_type,
                'stops': line.stops,
                'color': line.color
            } for lid, line in self.network.lines.items()},
            'transfers': [{
                'from_stop_id': t.from_stop_id,
                'to_stop_id': t.to_stop_id,
                'walking_time': t.walking_time,
                'walking_distance': t.walking_distance,
                'is_step_free': t.is_step_free
            } for t in self.network.transfers]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(network_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n💾 Network saved to: {filename}")
        logger.info(f"   - Stops: {len(network_data['stops'])}")
        logger.info(f"   - Lines: {len(network_data['lines'])}")
        logger.info(f"   - Transfers: {len(network_data['transfers'])}")
    
    def print_summary(self):
        """Print loading summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 MAJOR ROUTES LOADING SUMMARY")
        logger.info("="*80)
        logger.info(f"  Stops Added:       {self.stats['stops_added']}")
        logger.info(f"  Lines Added:       {self.stats['lines_added']}")
        logger.info(f"    - Marmaray:      {self.stats['marmaray_lines']} ⭐ PRIORITY")
        logger.info(f"    - Metro:         {self.stats['metro_lines']} ⭐ PRIORITY")
        logger.info(f"    - Ferry:         {self.stats['ferry_lines']}")
        logger.info(f"    - Tram:          {self.stats['tram_lines']}")
        logger.info(f"  Network Edges:     {len(self.network.edges)}")
        logger.info(f"  Network Nodes:     {len(self.network.stops)}")
        logger.info("="*80)
        logger.info("\n✅ ROUTING PRIORITY:")
        logger.info("   1. Marmaray (Europe-Asia connection)")
        logger.info("   2. Metro Lines (M1-M9 for city-wide coverage)")
        logger.info("   3. Ferries (Bosphorus crossings)")
        logger.info("   4. Trams (Historic and modern lines)")
        logger.info("\n🎯 System is now ready for routing advice!")
        logger.info("   Users can plan journeys using Marmaray and Metro primarily")
        logger.info("="*80)


async def main():
    """Load major routes and create working routing system"""
    
    logger.info("="*80)
    logger.info("🚀 OPTION B: QUICK START - MAJOR ROUTES LOADER")
    logger.info("="*80)
    logger.info("Loading Istanbul's most important transportation routes...")
    logger.info("PRIORITY: Marmaray + Metro (for best routing advice)\n")
    
    loader = MajorRoutesLoader()
    
    # Load routes in priority order
    loader.add_marmaray_routes()   # PRIORITY 1
    loader.add_metro_routes()       # PRIORITY 2
    loader.add_ferry_routes()       # PRIORITY 3
    loader.add_tram_routes()        # PRIORITY 4
    loader.add_major_bus_routes()   # PRIORITY 5 (stub for now)
    
    # Build network (creates edges between stops on same line)
    loader.build_network()
    
    # Add transfer connections (CRITICAL for multi-modal routing!)
    loader.add_transfer_connections()
    
    # Save network
    loader.save_network()
    
    # Print summary
    loader.print_summary()
    
    logger.info("\n🎉 SUCCESS! Major routes loaded with transfer connections.")
    logger.info("📋 NEXT STEP: Test routing with: python3 test_real_ibb_routing.py")
    logger.info("="*80)
    
    return loader.network


if __name__ == '__main__':
    network = asyncio.run(main())
    sys.exit(0)

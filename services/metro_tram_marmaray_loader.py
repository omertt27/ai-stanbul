"""
Metro, Tram & Marmaray Network Loader
Loads comprehensive rail transit network (Metro, Tram, Marmaray)
No API key required - uses hardcoded reliable data
"""

import logging
from typing import Dict, List
from services.route_network_builder import (
    TransportationNetwork, TransportStop, TransportLine, NetworkEdge
)

logger = logging.getLogger(__name__)


class MetroTramMarmarayLoader:
    """
    Production-grade loader for Istanbul's rail transit system
    Focuses on Metro, Tram, and Marmaray only for reliability
    """
    
    def __init__(self):
        self.network = TransportationNetwork()
        
    def load_full_network(self) -> TransportationNetwork:
        """Load complete Metro, Tram, and Marmaray network"""
        logger.info("=" * 70)
        logger.info("LOADING METRO, TRAM & MARMARAY NETWORK")
        logger.info("=" * 70)
        
        # Load all lines
        self._load_m1_metro()
        self._load_m2_metro()
        self._load_m3_metro()
        self._load_m4_metro()
        self._load_m5_metro()
        self._load_m6_metro()
        self._load_m7_metro()
        self._load_m9_metro()
        self._load_m11_metro()
        self._load_t1_tram()
        self._load_t4_tram()
        self._load_t5_tram()
        self._load_marmaray()
        self._load_funiculars()
        
        # Build network graph
        logger.info("Building network graph...")
        self.network.build_network()
        
        # Create transfer connections
        self._create_transfer_hubs()
        
        # Statistics
        logger.info("=" * 70)
        logger.info("NETWORK LOADED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Total Stops: {len(self.network.stops)}")
        logger.info(f"Total Lines: {len(self.network.lines)}")
        logger.info(f"Total Edges: {len(self.network.edges)}")
        logger.info("=" * 70)
        
        return self.network
    
    def _load_m1_metro(self):
        """Load M1 Metro Line (Yenikapı - Atatürk Havalimanı / Kirazlı)"""
        logger.info("Loading M1 Metro...")
        
        stops_data = [
            ("m1_yenikapi", "Yenikapı", 41.0066, 28.9569),
            ("m1_aksaray", "Aksaray", 41.0175, 28.9518),
            ("m1_emniyet_fatih", "Emniyet-Fatih", 41.0235, 28.9445),
            ("m1_topkapi_ulubatli", "Topkapı-Ulubatlı", 41.0266, 28.9327),
            ("m1_bayrampa", "Bayrampaşa", 41.0366, 28.9185),
            ("m1_sagmalcilar", "Sağmalcılar", 41.0438, 28.9053),
            ("m1_kocatepe", "Kocatepe", 41.0481, 28.8956),
            ("m1_otogar", "Otogar", 41.0546, 28.8834),
            ("m1_terazidere", "Terazidere", 41.0623, 28.8721),
            ("m1_davutpasa_y", "Davutpaşa-YTÜ", 41.0682, 28.8556),
            ("m1_merter", "Merter", 41.0711, 28.8435),
            ("m1_zeytinburnu", "Zeytinburnu", 41.0699, 28.8290),
            ("m1_bakirkoy_i", "Bakırköy-İncirli", 41.0653, 28.8146),
            ("m1_bahcelievler", "Bahçelievler", 41.0609, 28.8045),
            ("m1_atakoy_sirinev", "Ataköy-Şirinevler", 41.0547, 28.7941),
            ("m1_yenibosna", "Yenibosna", 41.0487, 28.7863),
            ("m1_dtm_istanbul_fuar", "DTM-İstanbul Fuar Merkezi", 41.0439, 28.7758),
            ("m1_havalimani", "Atatürk Havalimanı", 40.9902, 28.8109),
        ]
        
        line = TransportLine(
            line_id="m1",
            name="M1 Yenikapı-Atatürk Havalimanı/Kirazlı",
            transport_type="metro",
            stops=[s[0] for s in stops_data],
            color="#FF0000"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="metro",
                lines=["M1"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_m2_metro(self):
        """Load M2 Metro Line (Yenikapı - Hacıosman)"""
        logger.info("Loading M2 Metro...")
        
        stops_data = [
            ("m2_yenikapi", "Yenikapı", 41.0066, 28.9569),
            ("m2_vezneciler", "Vezneciler", 41.0131, 28.9552),
            ("m2_haliç", "Haliç", 41.0217, 28.9546),
            ("m2_sishanemeydan", "Şişhane", 41.0271, 28.9744),
            ("m2_taksim", "Taksim", 41.0370, 28.9857),
            ("m2_osmanbey", "Osmanbey", 41.0480, 28.9880),
            ("m2_sisli_mecidiyekoy", "Şişli-Mecidiyeköy", 41.0644, 28.9960),
            ("m2_gayrettepe", "Gayrettepe", 41.0708, 29.0052),
            ("m2_levent", "Levent", 41.0782, 29.0070),
            ("m2_4levent", "4. Levent", 41.0852, 29.0092),
            ("m2_sanayi_mahallesi", "Sanayi Mahallesi", 41.0959, 29.0154),
            ("m2_1tl_seyrantepe", "ITÜ-Ayazağa", 41.1057, 29.0237),
            ("m2_atatork_otosans", "Atatürk Oto Sanayi", 41.1119, 29.0308),
            ("m2_darussafaka", "Darüşşafaka", 41.1176, 29.0417),
            ("m2_hacisamanlı", "Hacıosman", 41.1232, 29.0508),
        ]
        
        line = TransportLine(
            line_id="m2",
            name="M2 Yenikapı-Hacıosman",
            transport_type="metro",
            stops=[s[0] for s in stops_data],
            color="#00A651"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="metro",
                lines=["M2"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_m3_metro(self):
        """Load M3 Metro Line (Kirazlı - Olimpiyat/Başakşehir)"""
        logger.info("Loading M3 Metro...")
        
        stops_data = [
            ("m3_kirazli", "Kirazlı", 41.0321, 28.7918),
            ("m3_metro_sitesi", "Metrokent", 41.0416, 28.7834),
            ("m3_turgut_ozal", "Turgut Özal", 41.0518, 28.7769),
            ("m3_ikitelli_sanayi", "İkitelli Sanayi", 41.0631, 28.7701),
            ("m3_ispartakule", "Ispartakule", 41.0720, 28.7572),
            ("m3_mahmutbey", "Mahmutbey", 41.0781, 28.7391),
            ("m3_istanbul_fuar", "İstanbul Fuar Merkezi", 41.0886, 28.7233),
            ("m3_basaksehir", "Başakşehir", 41.0971, 28.7089),
            ("m3_siteler", "Siteler", 41.1039, 28.6982),
            ("m3_mezitli", "Metrokent", 41.1142, 28.6865),
            ("m3_olimpiyat", "Olimpiyat", 41.1221, 28.6761),
        ]
        
        line = TransportLine(
            line_id="m3",
            name="M3 Kirazlı-Olimpiyat/Başakşehir",
            transport_type="metro",
            stops=[s[0] for s in stops_data],
            color="#0000FF"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="metro",
                lines=["M3"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_m4_metro(self):
        """Load M4 Metro Line (Kadıköy - Tavşantepe)"""
        logger.info("Loading M4 Metro...")
        
        stops_data = [
            ("m4_kadikoy", "Kadıköy", 40.9905, 29.0250),
            ("m4_ayrilikcesme", "Ayrılık Çeşmesi", 40.9889, 29.0317),
            ("m4_acıbadem", "Acıbadem", 40.9879, 29.0395),
            ("m4_unalan", "Ünalan", 40.9883, 29.0504),
            ("m4_goztepe", "Göztepe", 40.9878, 29.0587),
            ("m4_yenisahra", "Yenisahra", 40.9866, 29.0686),
            ("m4_kozyatagi", "Kozyatağı", 40.9847, 29.0815),
            ("m4_bostanci", "Bostancı", 40.9792, 29.0916),
            ("m4_kucukyali", "Küçükyalı", 40.9724, 29.1037),
            ("m4_maltepe", "Maltepe", 40.9639, 29.1241),
            ("m4_huzurevi", "Huzurevi", 40.9547, 29.1426),
            ("m4_gulsuyu", "Gülsuyu", 40.9486, 29.1538),
            ("m4_esenkent", "Esenkent", 40.9421, 29.1669),
            ("m4_hastane", "Hastane-Adliye", 40.9348, 29.1755),
            ("m4_soganlik", "Soğanlık", 40.9278, 29.1856),
            ("m4_kartal", "Kartal", 40.9187, 29.1973),
            ("m4_yakaciik", "Yakacık-Çarşı", 40.9094, 29.2108),
            ("m4_pendik", "Pendik", 40.8961, 29.2344),
            ("m4_tavşantepe", "Tavşantepe", 40.8846, 29.2557),
        ]
        
        line = TransportLine(
            line_id="m4",
            name="M4 Kadıköy-Tavşantepe",
            transport_type="metro",
            stops=[s[0] for s in stops_data],
            color="#FF69B4"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="metro",
                lines=["M4"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_m5_metro(self):
        """Load M5 Metro Line (Üsküdar - Çekmeköy)"""
        logger.info("Loading M5 Metro...")
        
        stops_data = [
            ("m5_uskudar", "Üsküdar", 41.0226, 29.0078),
            ("m5_fistikagaci", "Fıstıkağacı", 41.0259, 29.0189),
            ("m5_bağlarbaşı", "Bağlarbaşı", 41.0311, 29.0325),
            ("m5_altunizade", "Altunizade", 41.0344, 29.0456),
            ("m5_kısıklı", "Kısıklı", 41.0392, 29.0608),
            ("m5_cavusbasi", "Çavuşbaşı", 41.0449, 29.0741),
            ("m5_umraniye", "Ümraniye", 41.0518, 29.0894),
            ("m5_yamanevler", "Yamanevler", 41.0586, 29.1056),
            ("m5_carsı", "Çarşı", 41.0659, 29.1209),
            ("m5_ihsaniye", "İhsaniye", 41.0732, 29.1374),
            ("m5_alemdag", "Alemdağ", 41.0791, 29.1528),
            ("m5_sultanbeyli", "Sultanbeyli", 40.9637, 29.2614),
            ("m5_cekmekoy", "Çekmeköy", 41.0303, 29.1836),
        ]
        
        line = TransportLine(
            line_id="m5",
            name="M5 Üsküdar-Çekmeköy/Sultanbeyli",
            transport_type="metro",
            stops=[s[0] for s in stops_data],
            color="#8B00FF"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="metro",
                lines=["M5"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_m6_metro(self):
        """Load M6 Metro Line (Levent - Boğaziçi Üniversitesi/Hisarüstü)"""
        logger.info("Loading M6 Metro...")
        
        stops_data = [
            ("m6_levent", "Levent", 41.0782, 29.0070),
            ("m6_nispetiye", "Nispetiye", 41.0811, 29.0171),
            ("m6_etiler", "Etiler", 41.0840, 29.0254),
            ("m6_bogazici_uni", "Boğaziçi Üniversitesi", 41.0869, 29.0321),
        ]
        
        line = TransportLine(
            line_id="m6",
            name="M6 Levent-Boğaziçi Üniversitesi/Hisarüstü",
            transport_type="metro",
            stops=[s[0] for s in stops_data],
            color="#804000"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="metro",
                lines=["M6"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_m7_metro(self):
        """Load M7 Metro Line (Mecidiyeköy - Mahmutbey)"""
        logger.info("Loading M7 Metro...")
        
        stops_data = [
            ("m7_mecidiyekoy", "Mecidiyeköy", 41.0644, 28.9960),
            ("m7_ceglayan", "Çağlayan", 41.0672, 28.9783),
            ("m7_kagithane", "Kağıthane", 41.0734, 28.9645),
            ("m7_nurtepe", "Nurtepe", 41.0817, 28.9516),
            ("m7_alibeyköy", "Alibeyköy", 41.0891, 28.9397),
            ("m7_cebeci", "Cebeci", 41.0968, 28.9268),
            ("m7_yenimahalle", "Yenimahalle", 41.1052, 28.9144),
            ("m7_karadeniz_mah", "Karadeniz Mahallesi", 41.1131, 28.8976),
            ("m7_tekstilkent", "Tekstilkent", 41.1202, 28.8816),
            ("m7_iett_deposu", "İETT Deposu", 41.1266, 28.8657),
            ("m7_giyim_sanatkar", "Giyim Sanatkârları", 41.1322, 28.8484),
            ("m7_yesilpinar", "Yeşilpınar", 41.1368, 28.8324),
            ("m7_kazim_karabekir", "Kâzım Karabekir", 41.1412, 28.8168),
            ("m7_yıldırım", "Yıldırım", 41.1441, 28.7981),
            ("m7_mahmutbey", "Mahmutbey", 41.0781, 28.7391),
        ]
        
        line = TransportLine(
            line_id="m7",
            name="M7 Mecidiyeköy-Mahmutbey",
            transport_type="metro",
            stops=[s[0] for s in stops_data],
            color="#FF1493"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="metro",
                lines=["M7"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_m9_metro(self):
        """Load M9 Metro Line (Ataköy - İkitelli)"""
        logger.info("Loading M9 Metro...")
        
        stops_data = [
            ("m9_atakoy", "Ataköy", 40.9795, 28.8431),
            ("m9_sahil", "Sahil", 40.9867, 28.8246),
            ("m9_halkalı", "Halkalı", 41.0038, 28.7941),
            ("m9_ikitelli", "İkitelli", 41.0631, 28.7701),
        ]
        
        line = TransportLine(
            line_id="m9",
            name="M9 Ataköy-İkitelli",
            transport_type="metro",
            stops=[s[0] for s in stops_data],
            color="#6A0DAD"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="metro",
                lines=["M9"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_m11_metro(self):
        """Load M11 Metro Line (Gayrettepe - Istanbul Airport)"""
        logger.info("Loading M11 Metro...")
        
        stops_data = [
            ("m11_gayrettepe", "Gayrettepe", 41.0708, 29.0052),
            ("m11_kagithane", "Kağıthane", 41.0734, 28.9645),
            ("m11_havalimani", "İstanbul Havalimanı", 41.2625, 28.7413),
        ]
        
        line = TransportLine(
            line_id="m11",
            name="M11 Gayrettepe-İstanbul Havalimanı",
            transport_type="metro",
            stops=[s[0] for s in stops_data],
            color="#00CED1"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="metro",
                lines=["M11"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_t1_tram(self):
        """Load T1 Tram Line (Kabataş - Bağcılar)"""
        logger.info("Loading T1 Tram...")
        
        stops_data = [
            ("t1_kabatas", "Kabataş", 41.0385, 29.0079),
            ("t1_findikli", "Fındıklı", 41.0311, 28.9933),
            ("t1_tophane", "Tophane", 41.0259, 28.9847),
            ("t1_karakoy", "Karaköy", 41.0242, 28.9753),
            ("t1_eminonu", "Eminönü", 41.0167, 28.9727),
            ("t1_sultanahmet", "Sultanahmet", 41.0082, 28.9784),
            ("t1_beyazit_kapalic", "Beyazıt-Kapalıçarşı", 41.0108, 28.9682),
            ("t1_laleli_uni", "Laleli-Üniversite", 41.0145, 28.9598),
            ("t1_aksaray", "Aksaray", 41.0175, 28.9518),
            ("t1_yusufpasa", "Yusufpaşa", 41.0208, 28.9427),
            ("t1_haseki", "Haseki", 41.0241, 28.9346),
            ("t1_cerrahpasa", "Cerrahpaşa", 41.0271, 28.9258),
            ("t1_mevlanakapı", "Mevlanakapı", 41.0312, 28.9158),
            ("t1_topkapi", "Topkapı", 41.0266, 28.9327),
            ("t1_pazartekke", "Pazartekke", 41.0391, 28.8976),
            ("t1_capa", "Capa", 41.0428, 28.8892),
            ("t1_fetihkapi", "Fethkapı", 41.0461, 28.8813),
            ("t1_ulubatli", "Ulubatlı", 41.0498, 28.8725),
            ("t1_bagcilar", "Bağcılar", 41.0534, 28.8647),
        ]
        
        line = TransportLine(
            line_id="t1",
            name="T1 Kabataş-Bağcılar",
            transport_type="tram",
            stops=[s[0] for s in stops_data],
            color="#00A651"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="tram",
                lines=["T1"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_t4_tram(self):
        """Load T4 Tram Line (Topkapı - Mescid-i Selam)"""
        logger.info("Loading T4 Tram...")
        
        stops_data = [
            ("t4_topkapi", "Topkapı", 41.0266, 28.9327),
            ("t4_davutpasa", "Davutpaşa", 41.0682, 28.8556),
            ("t4_habibler", "Habibler", 41.0321, 28.8894),
            ("t4_mescid_i_selam", "Mescid-i Selam", 41.0389, 28.8742),
        ]
        
        line = TransportLine(
            line_id="t4",
            name="T4 Topkapı-Mescid-i Selam",
            transport_type="tram",
            stops=[s[0] for s in stops_data],
            color="#FFA500"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="tram",
                lines=["T4"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_t5_tram(self):
        """Load T5 Tram Line (Cibali - Alibeyköy)"""
        logger.info("Loading T5 Tram...")
        
        stops_data = [
            ("t5_cibali", "Cibali", 41.0335, 28.9512),
            ("t5_fener", "Fener", 41.0421, 28.9485),
            ("t5_balat", "Balat", 41.0511, 28.9462),
            ("t5_ayvansaray", "Ayvansaray", 41.0604, 28.9441),
            ("t5_eyupsultan", "Eyüpsultan", 41.0698, 28.9337),
            ("t5_silahtaraga", "Silahtarağa", 41.0781, 28.9271),
            ("t5_alibeykoy", "Alibeyköy", 41.0891, 28.9397),
        ]
        
        line = TransportLine(
            line_id="t5",
            name="T5 Cibali-Alibeyköy",
            transport_type="tram",
            stops=[s[0] for s in stops_data],
            color="#0000CD"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="tram",
                lines=["T5"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_marmaray(self):
        """Load Marmaray Line (Halkalı - Gebze)"""
        logger.info("Loading Marmaray...")
        
        stops_data = [
            ("mr_halkali", "Halkalı", 41.0038, 28.7941),
            ("mr_kucukcekmece", "Küçükçekmece", 41.0095, 28.7624),
            ("mr_florya", "Florya", 40.9867, 28.7891),
            ("mr_yesilkoy", "Yeşilköy", 40.9768, 28.8145),
            ("mr_bakirkoy", "Bakırköy", 40.9856, 28.8533),
            ("mr_yenimahalle", "Yenimahalle", 40.9921, 28.8678),
            ("mr_zeytinburnu", "Zeytinburnu", 41.0699, 28.8290),
            ("mr_kazlicesme", "Kazlıçeşme", 41.0034, 28.9048),
            ("mr_yenikapi", "Yenikapı", 41.0066, 28.9569),
            ("mr_sirkeci", "Sirkeci", 41.0175, 28.9764),
            ("mr_uskudar", "Üsküdar", 41.0226, 29.0078),
            ("mr_ayrilikcesmesi", "Ayrılıkçeşmesi", 40.9889, 29.0317),
            ("mr_soğutluçeşme", "Söğütlüçeşme", 40.9783, 29.0524),
            ("mr_feneryolu", "Feneryolu", 40.9687, 29.0736),
            ("mr_goztepe", "Göztepe", 40.9878, 29.0587),
            ("mr_erenkoy", "Erenköy", 40.9745, 29.0814),
            ("mr_suadiye", "Suadiye", 40.9621, 29.1038),
            ("mr_maltepe", "Maltepe", 40.9639, 29.1241),
            ("mr_cevizli", "Cevizli", 40.9567, 29.1462),
            ("mr_pendik", "Pendik", 40.8961, 29.2344),
            ("mr_kartal", "Kartal", 40.9187, 29.1973),
            ("mr_gebze", "Gebze", 40.8028, 29.4318),
        ]
        
        line = TransportLine(
            line_id="marmaray",
            name="Marmaray Halkalı-Gebze",
            transport_type="rail",
            stops=[s[0] for s in stops_data],
            color="#E2001A"
        )
        self.network.add_line(line)
        
        for stop_id, name, lat, lon in stops_data:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="rail",
                lines=["Marmaray"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _load_funiculars(self):
        """Load Funicular Lines"""
        logger.info("Loading Funiculars...")
        
        # F1: Kabataş-Taksim
        f1_stops = [
            ("f1_kabatas", "Kabataş", 41.0385, 29.0079),
            ("f1_taksim", "Taksim", 41.0370, 28.9857),
        ]
        
        f1_line = TransportLine(
            line_id="f1",
            name="F1 Kabataş-Taksim",
            transport_type="funicular",
            stops=[s[0] for s in f1_stops],
            color="#FF0000"
        )
        self.network.add_line(f1_line)
        
        for stop_id, name, lat, lon in f1_stops:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="funicular",
                lines=["F1"],
                accessibility=True
            )
            self.network.add_stop(stop)
        
        # F2: Tünel (Karaköy-Beyoğlu)
        f2_stops = [
            ("f2_karakoy", "Karaköy", 41.0242, 28.9753),
            ("f2_beyoglu", "Beyoğlu", 41.0329, 28.9779),
        ]
        
        f2_line = TransportLine(
            line_id="f2",
            name="F2 Tünel",
            transport_type="funicular",
            stops=[s[0] for s in f2_stops],
            color="#000000"
        )
        self.network.add_line(f2_line)
        
        for stop_id, name, lat, lon in f2_stops:
            stop = TransportStop(
                stop_id=stop_id,
                name=name,
                lat=lat,
                lon=lon,
                transport_type="funicular",
                lines=["F2"],
                accessibility=True
            )
            self.network.add_stop(stop)
    
    def _create_transfer_hubs(self):
        """Create transfer connections between lines at major hubs"""
        logger.info("Creating transfer connections...")
        
        transfer_hubs = [
            # Yenikapı: M1, M2, Marmaray
            ("m1_yenikapi", "m2_yenikapi", 2),
            ("m1_yenikapi", "mr_yenikapi", 3),
            ("m2_yenikapi", "mr_yenikapi", 3),
            
            # Taksim: M2, F1
            ("m2_taksim", "f1_taksim", 2),
            
            # Kabataş: T1, F1
            ("t1_kabatas", "f1_kabatas", 1),
            
            # Karaköy: T1, F2
            ("t1_karakoy", "f2_karakoy", 2),
            
            # Levent: M2, M6
            ("m2_levent", "m6_levent", 3),
            
            # Mecidiyeköy: M2, M7
            ("m2_sisli_mecidiyekoy", "m7_mecidiyekoy", 2),
            
            # Gayrettepe: M2, M11
            ("m2_gayrettepe", "m11_gayrettepe", 2),
            
            # Üsküdar: M5, Marmaray
            ("m5_uskudar", "mr_uskudar", 3),
            
            # Kadıköy: M4, Marmaray (nearby)
            ("m4_kadikoy", "mr_ayrilikcesmesi", 5),
            
            # Topkapı: M1, T1, T4
            ("m1_topkapi_ulubatli", "t1_topkapi", 3),
            ("t1_topkapi", "t4_topkapi", 2),
        ]
        
        transfers_created = 0
        for from_stop, to_stop, duration in transfer_hubs:
            if from_stop in self.network.stops and to_stop in self.network.stops:
                stop_from = self.network.stops[from_stop]
                stop_to = self.network.stops[to_stop]
                distance = stop_from.distance_to(stop_to)
                
                # Create bidirectional transfer edges
                edge1 = NetworkEdge(
                    from_stop_id=from_stop,
                    to_stop_id=to_stop,
                    line_id="transfer",
                    transport_type="transfer",
                    travel_time=duration,
                    distance=distance,
                    edge_type="transfer"
                )
                edge2 = NetworkEdge(
                    from_stop_id=to_stop,
                    to_stop_id=from_stop,
                    line_id="transfer",
                    transport_type="transfer",
                    travel_time=duration,
                    distance=distance,
                    edge_type="transfer"
                )
                
                self.network.add_edge(edge1)
                self.network.add_edge(edge2)
                transfers_created += 2
        
        logger.info(f"✓ Created {transfers_created} transfer connections")


def load_metro_tram_marmaray_network() -> TransportationNetwork:
    """
    Convenience function to load the complete Metro, Tram & Marmaray network
    """
    loader = MetroTramMarmarayLoader()
    return loader.load_full_network()

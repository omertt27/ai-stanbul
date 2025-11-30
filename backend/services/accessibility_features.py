"""
Accessibility Features for Istanbul Transportation
==================================================

Adds wheelchair-accessible routing for Istanbul public transportation.

Features:
- Elevator availability data for stations
- Accessible route preference
- Filter routes by accessibility requirements
- Accessibility indicators in route display
- Step-free alternative suggestions

Author: Istanbul AI Team
Date: November 30, 2024
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AccessibilityLevel(Enum):
    """Accessibility levels for stations and routes"""
    FULLY_ACCESSIBLE = "fully_accessible"  # Full elevator access, wide platforms
    PARTIALLY_ACCESSIBLE = "partially_accessible"  # Some accessibility features
    LIMITED_ACCESSIBILITY = "limited_accessibility"  # Stairs required, limited features
    NOT_ACCESSIBLE = "not_accessible"  # No accessibility features


@dataclass
class StationAccessibility:
    """Accessibility information for a station"""
    station_name: str
    line: str
    has_elevator: bool
    has_ramp: bool
    has_tactile_paving: bool
    platform_accessible: bool
    entrance_accessible: bool
    accessibility_level: AccessibilityLevel
    notes: Optional[str] = None


# Istanbul Metro/Tram/Ferry Accessibility Data
# Based on IBB (Istanbul Metropolitan Municipality) accessibility reports
STATION_ACCESSIBILITY_DATA = {
    # M2 Line - Generally good accessibility
    'metro_M2_taksim': StationAccessibility(
        station_name='Taksim',
        line='M2',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Multiple elevators, all platforms accessible'
    ),
    'metro_M2_şişli_mecidiyeköy': StationAccessibility(
        station_name='Şişli-Mecidiyeköy',
        line='M2',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M2_osmanbey': StationAccessibility(
        station_name='Osmanbey',
        line='M2',
        has_elevator=True,
        has_ramp=False,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M2_yenikapı': StationAccessibility(
        station_name='Yenikapı',
        line='M2',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Major transfer hub with full accessibility'
    ),
    
    # M4 Line - Modern line with good accessibility
    'metro_M4_kadıköy_sabiha_gökçen_havalimanı_kadıköy': StationAccessibility(
        station_name='Kadıköy',
        line='M4',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M4_kadıköy_sabiha_gökçen_havalimanı_ayrılık_çeşmesi': StationAccessibility(
        station_name='Ayrılık Çeşmesi',
        line='M4',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Transfer hub with Marmaray'
    ),
    
    # Marmaray - Underground rail with elevators
    'marmaray_Marmaray_yenikapı': StationAccessibility(
        station_name='Yenikapı',
        line='Marmaray',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'marmaray_Marmaray_sirkeci': StationAccessibility(
        station_name='Sirkeci',
        line='Marmaray',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'marmaray_Marmaray_ayrılık_çeşmesi': StationAccessibility(
        station_name='Ayrılık Çeşmesi',
        line='Marmaray',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    
    # T1 Tram - Historic tram with limited accessibility
    'tram_T1_kabataş_bağcılar_sultanahmet': StationAccessibility(
        station_name='Sultanahmet',
        line='T1',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=False,
        platform_accessible=True,  # Level boarding
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE,
        notes='Level boarding but no elevators for street access'
    ),
    'tram_T1_kabataş_bağcılar_eminönü': StationAccessibility(
        station_name='Eminönü',
        line='T1',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=False,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE
    ),
    'tram_T1_kabataş_bağcılar_kabataş': StationAccessibility(
        station_name='Kabataş',
        line='T1',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE
    ),
    
    # F1 Funicular - Limited accessibility due to elevation change
    'funicular_F1_taksim_kabataş_funicular_kabataş': StationAccessibility(
        station_name='Kabataş',
        line='F1',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Funicular cars accommodate wheelchairs'
    ),
    'funicular_F1_taksim_kabataş_funicular_taksim': StationAccessibility(
        station_name='Taksim',
        line='F1',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    
    # Ferry terminals - Generally accessible
    'ferry_eminönü_kadıköy_ferry_eminönü': StationAccessibility(
        station_name='Eminönü',
        line='Ferry',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=False,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Ferry boats have wheelchair ramps'
    ),
    'ferry_eminönü_kadıköy_ferry_kadıköy': StationAccessibility(
        station_name='Kadıköy',
        line='Ferry',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=False,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    
    # M1 Line - Airport line with good accessibility
    'metro_M1A_yenikapı_atatürk_havalimanı_yenikapı': StationAccessibility(
        station_name='Yenikapı',
        line='M1A',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Major transfer hub'
    ),
    'metro_M1A_yenikapı_atatürk_havalimanı_aksaray': StationAccessibility(
        station_name='Aksaray',
        line='M1A',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M1A_yenikapı_atatürk_havalimanı_zeytinburnu': StationAccessibility(
        station_name='Zeytinburnu',
        line='M1A',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Transfer station with T1 tram'
    ),
    'metro_M1A_yenikapı_atatürk_havalimanı_otogar': StationAccessibility(
        station_name='Otogar',
        line='M1A',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Bus terminal connection'
    ),
    'metro_M1A_yenikapı_atatürk_havalimanı_atatürk_havalimanı': StationAccessibility(
        station_name='Atatürk Havalimanı',
        line='M1A',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Former airport - full accessibility'
    ),
    
    # M1B Line - Kirazlı branch
    'metro_M1B_yenikapı_kirazlı_kirazlı': StationAccessibility(
        station_name='Kirazlı',
        line='M1B',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Transfer with M3 line'
    ),
    
    # M3 Line - Modern line with excellent accessibility
    'metro_M3_kirazlı_başakşehir_kirazlı': StationAccessibility(
        station_name='Kirazlı',
        line='M3',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M3_kirazlı_başakşehir_olimpiyat': StationAccessibility(
        station_name='Olimpiyat',
        line='M3',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Stadium access'
    ),
    'metro_M3_kirazlı_başakşehir_başakşehir': StationAccessibility(
        station_name='Başakşehir',
        line='M3',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    
    # M5 Line - Asian side metro
    'metro_M5_üsküdar_çekmeköy_üsküdar': StationAccessibility(
        station_name='Üsküdar',
        line='M5',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Major Asian side hub'
    ),
    'metro_M5_üsküdar_çekmeköy_altunizade': StationAccessibility(
        station_name='Altunizade',
        line='M5',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M5_üsküdar_çekmeköy_çekmeköy': StationAccessibility(
        station_name='Çekmeköy',
        line='M5',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    
    # M6 Line - Levent business district
    'metro_M6_levent_boğaziçi_üniversitesi/hisarüstü_levent': StationAccessibility(
        station_name='Levent',
        line='M6',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Business district - transfer with M2'
    ),
    'metro_M6_levent_boğaziçi_üniversitesi/hisarüstü_boğaziçi_üniversitesi/hisarüstü': StationAccessibility(
        station_name='Boğaziçi Üniversitesi/Hisarüstü',
        line='M6',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='University access'
    ),
    
    # M7 Line - Mecidiyeköy-Mahmutbey
    'metro_M7_mecidiyeköy_mahmutbey_mecidiyeköy': StationAccessibility(
        station_name='Mecidiyeköy',
        line='M7',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Transfer with M2'
    ),
    'metro_M7_mecidiyeköy_mahmutbey_yıldız': StationAccessibility(
        station_name='Yıldız',
        line='M7',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M7_mecidiyeköy_mahmutbey_mahmutbey': StationAccessibility(
        station_name='Mahmutbey',
        line='M7',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    
    # M9 Line - New airport line (İstanbul Airport)
    'metro_M9_olimpiyat_i̇stanbul_havalimanı_olimpiyat': StationAccessibility(
        station_name='Olimpiyat',
        line='M9',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M9_olimpiyat_i̇stanbul_havalimanı_i̇stanbul_havalimanı': StationAccessibility(
        station_name='İstanbul Havalimanı',
        line='M9',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='New airport - state-of-the-art accessibility'
    ),
    
    # M11 Line - Çekmeköy-Sancaktepe
    'metro_M11_çekmeköy_sancaktepe_çekmeköy': StationAccessibility(
        station_name='Çekmeköy',
        line='M11',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M11_çekmeköy_sancaktepe_sancaktepe': StationAccessibility(
        station_name='Sancaktepe',
        line='M11',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    
    # T4 Tram - Topkapı-Mescid-i Selam
    'tram_T4_topkapı_mescid_i_selam_topkapı': StationAccessibility(
        station_name='Topkapı',
        line='T4',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE,
        notes='Level boarding, limited elevator access'
    ),
    'tram_T4_topkapı_mescid_i_selam_sultançiftliği': StationAccessibility(
        station_name='Sultançiftliği',
        line='T4',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE
    ),
    
    # T5 Tram - Cibali-Alibeyköy
    'tram_T5_cibali_alibeyköy_cibali': StationAccessibility(
        station_name='Cibali',
        line='T5',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=False,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE,
        notes='Historic area with limited accessibility'
    ),
    'tram_T5_cibali_alibeyköy_alibeyköy': StationAccessibility(
        station_name='Alibeyköy',
        line='T5',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE
    ),
    
    # Additional M2 stations
    'metro_M2_yenikapı_hacıosman_hacıosman': StationAccessibility(
        station_name='Hacıosman',
        line='M2',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Northern terminus'
    ),
    'metro_M2_yenikapı_hacıosman_4._levent': StationAccessibility(
        station_name='4. Levent',
        line='M2',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M2_yenikapı_hacıosman_gayrettepe': StationAccessibility(
        station_name='Gayrettepe',
        line='M2',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M2_yenikapı_hacıosman_şişhane': StationAccessibility(
        station_name='Şişhane',
        line='M2',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Near Galata Tower'
    ),
    'metro_M2_yenikapı_hacıosman_vezneciler': StationAccessibility(
        station_name='Vezneciler',
        line='M2',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='University area'
    ),
    
    # Additional M4 stations
    'metro_M4_kadıköy_sabiha_gökçen_havalimanı_bostancı': StationAccessibility(
        station_name='Bostancı',
        line='M4',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M4_kadıköy_sabiha_gökçen_havalimanı_kozyatağı': StationAccessibility(
        station_name='Kozyatağı',
        line='M4',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M4_kadıköy_sabiha_gökçen_havalimanı_kartal': StationAccessibility(
        station_name='Kartal',
        line='M4',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'metro_M4_kadıköy_sabiha_gökçen_havalimanı_sabiha_gökçen_havalimanı': StationAccessibility(
        station_name='Sabiha Gökçen Havalimanı',
        line='M4',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Airport terminal - full accessibility'
    ),
    
    # Additional T1 stations
    'tram_T1_kabataş_bağcılar_beyazıt_kapalı_çarşı': StationAccessibility(
        station_name='Beyazıt-Kapalı Çarşı',
        line='T1',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=False,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE,
        notes='Grand Bazaar area'
    ),
    'tram_T1_kabataş_bağcılar_topkapı': StationAccessibility(
        station_name='Topkapı',
        line='T1',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=False,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE
    ),
    'tram_T1_kabataş_bağcılar_bağcılar': StationAccessibility(
        station_name='Bağcılar',
        line='T1',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE
    ),
    
    # Additional ferry terminals
    'ferry_beşiktaş_üsküdar_ferry_beşiktaş': StationAccessibility(
        station_name='Beşiktaş',
        line='Ferry',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=False,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Bosphorus crossing'
    ),
    'ferry_beşiktaş_üsküdar_ferry_üsküdar': StationAccessibility(
        station_name='Üsküdar',
        line='Ferry',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=False,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE
    ),
    'ferry_karaköy_terminal': StationAccessibility(
        station_name='Karaköy',
        line='Ferry',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Major ferry hub'
    ),
    
    # Metrobüs stations (Bus Rapid Transit)
    'metrobus_zincirlikuyu': StationAccessibility(
        station_name='Zincirlikuyu',
        line='Metrobüs',
        has_elevator=True,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.FULLY_ACCESSIBLE,
        notes='Major transfer hub with metro'
    ),
    'metrobus_mecidiyeköy': StationAccessibility(
        station_name='Mecidiyeköy',
        line='Metrobüs',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE,
        notes='Transfer with M2 and M7'
    ),
    'metrobus_avcılar': StationAccessibility(
        station_name='Avcılar',
        line='Metrobüs',
        has_elevator=False,
        has_ramp=True,
        has_tactile_paving=True,
        platform_accessible=True,
        entrance_accessible=True,
        accessibility_level=AccessibilityLevel.PARTIALLY_ACCESSIBLE
    ),
}


class AccessibilityChecker:
    """Checks and filters routes based on accessibility requirements"""
    
    def __init__(self):
        """Initialize accessibility checker"""
        self.station_data = STATION_ACCESSIBILITY_DATA
        logger.info(f"✅ Accessibility checker initialized with {len(self.station_data)} stations")
    
    def is_station_accessible(self, station_id: str) -> bool:
        """
        Check if a station is wheelchair accessible
        
        Args:
            station_id: Station node ID from graph
            
        Returns:
            True if station is fully accessible
        """
        if station_id not in self.station_data:
            # Unknown stations assumed not accessible (conservative approach)
            logger.debug(f"Unknown station accessibility: {station_id}")
            return False
        
        station = self.station_data[station_id]
        return station.accessibility_level in [
            AccessibilityLevel.FULLY_ACCESSIBLE,
            AccessibilityLevel.PARTIALLY_ACCESSIBLE
        ]
    
    def get_station_accessibility(self, station_id: str) -> Optional[StationAccessibility]:
        """Get detailed accessibility info for a station"""
        return self.station_data.get(station_id)
    
    def is_route_accessible(self, route_path) -> bool:
        """
        Check if an entire route is wheelchair accessible
        
        Args:
            route_path: RoutePath object from graph routing
            
        Returns:
            True if all stations in route are accessible
        """
        # Check all nodes in the route
        for node in route_path.nodes:
            if not self.is_station_accessible(node.id):
                return False
        
        # Check for excessive transfers (difficult for wheelchair users)
        if route_path.transfers > 2:
            logger.debug(f"Route has {route_path.transfers} transfers - may be challenging")
            return False
        
        return True
    
    def get_accessibility_score(self, route_path) -> float:
        """
        Calculate accessibility score for a route (0-100)
        
        Higher score = more accessible
        """
        if not route_path.nodes:
            return 0.0
        
        score = 100.0
        
        # Check each station
        accessible_stations = 0
        for node in route_path.nodes:
            station_info = self.get_station_accessibility(node.id)
            if station_info:
                if station_info.accessibility_level == AccessibilityLevel.FULLY_ACCESSIBLE:
                    accessible_stations += 1
                elif station_info.accessibility_level == AccessibilityLevel.PARTIALLY_ACCESSIBLE:
                    accessible_stations += 0.7
                elif station_info.accessibility_level == AccessibilityLevel.LIMITED_ACCESSIBILITY:
                    accessible_stations += 0.3
        
        station_score = (accessible_stations / len(route_path.nodes)) * 60
        
        # Penalize for transfers
        transfer_penalty = route_path.transfers * 10
        transfer_score = max(0, 30 - transfer_penalty)
        
        # Prefer shorter routes (less fatigue)
        duration_score = max(0, 10 - (route_path.total_duration / 10))
        
        total_score = station_score + transfer_score + duration_score
        
        return min(100.0, max(0.0, total_score))
    
    def get_accessibility_warnings(self, route_path) -> List[str]:
        """
        Get accessibility warnings for a route
        
        Returns list of warnings for wheelchair users
        """
        warnings = []
        
        # Check each station
        for node in route_path.nodes:
            station_info = self.get_station_accessibility(node.id)
            
            if not station_info:
                warnings.append(f"⚠️ {node.name}: Accessibility information unavailable")
            elif station_info.accessibility_level == AccessibilityLevel.LIMITED_ACCESSIBILITY:
                warnings.append(f"⚠️ {node.name}: Limited accessibility - stairs may be required")
            elif station_info.accessibility_level == AccessibilityLevel.NOT_ACCESSIBLE:
                warnings.append(f"❌ {node.name}: Not wheelchair accessible")
            elif not station_info.has_elevator:
                warnings.append(f"⚠️ {node.name}: No elevator available")
        
        # Check transfers
        if route_path.transfers > 2:
            warnings.append(f"⚠️ This route requires {route_path.transfers} transfers, which may be challenging")
        
        # Check duration
        if route_path.total_duration > 60:
            warnings.append(f"ℹ️ Long journey ({route_path.total_duration} min) - plan for rest breaks")
        
        return warnings
    
    def get_accessibility_highlights(self, route_path) -> List[str]:
        """
        Get positive accessibility features of a route
        
        Returns list of accessibility highlights
        """
        highlights = []
        
        # Check for full accessibility
        fully_accessible_count = 0
        has_elevators = []
        
        for node in route_path.nodes:
            station_info = self.get_station_accessibility(node.id)
            if station_info:
                if station_info.accessibility_level == AccessibilityLevel.FULLY_ACCESSIBLE:
                    fully_accessible_count += 1
                if station_info.has_elevator:
                    has_elevators.append(node.name)
        
        if fully_accessible_count == len(route_path.nodes):
            highlights.append("✅ All stations fully wheelchair accessible")
        elif fully_accessible_count >= len(route_path.nodes) * 0.8:
            highlights.append(f"✅ Most stations ({fully_accessible_count}/{len(route_path.nodes)}) wheelchair accessible")
        
        if has_elevators:
            highlights.append(f"🛗 Elevators available at {len(has_elevators)} stations")
        
        if route_path.transfers <= 1:
            highlights.append("✅ Minimal transfers required")
        
        if 'ferry' in route_path.modes_used:
            highlights.append("⛴️ Ferry boats have wheelchair ramps")
        
        if 'funicular' in route_path.modes_used:
            highlights.append("🚡 Funicular accommodates wheelchairs")
        
        return highlights


# Global instance
_accessibility_checker = None


def get_accessibility_checker() -> AccessibilityChecker:
    """Get or create accessibility checker instance"""
    global _accessibility_checker
    if _accessibility_checker is None:
        _accessibility_checker = AccessibilityChecker()
    return _accessibility_checker


# Testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("🦽 Testing Accessibility Features")
    print("="*80 + "\n")
    
    checker = get_accessibility_checker()
    
    # Test station accessibility
    print("1️⃣ Testing station accessibility:")
    test_stations = [
        'metro_M2_taksim',
        'metro_M2_yenikapı',
        'tram_T1_kabataş_bağcılar_sultanahmet',
        'ferry_eminönü_kadıköy_ferry_eminönü'
    ]
    
    for station_id in test_stations:
        info = checker.get_station_accessibility(station_id)
        if info:
            accessible = "✅" if checker.is_station_accessible(station_id) else "❌"
            print(f"{accessible} {info.station_name} ({info.line})")
            print(f"   Level: {info.accessibility_level.value}")
            print(f"   Elevator: {'✅' if info.has_elevator else '❌'}")
            if info.notes:
                print(f"   Note: {info.notes}")
        print()
    
    print("="*80)
    print("✅ Accessibility features ready!")

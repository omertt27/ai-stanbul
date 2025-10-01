#!/usr/bin/env python3
"""
Enhanced Istanbul Transportation Advisor
========================================

Comprehensive transportation guidance system with:
- Istanbul Kart mastery guide
- Real-time integration recommendations  
- Accessibility routing
- Weather-adaptive suggestions
- Cultural etiquette integration
- Cost optimization
- Live disruption awareness
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TransportMode(Enum):
    METRO = "metro"
    BUS = "bus"
    TRAM = "tram"
    FERRY = "ferry"
    DOLMUS = "dolmus"
    TAXI = "taxi"
    WALKING = "walking"

class WeatherCondition(Enum):
    CLEAR = "clear"
    RAINY = "rainy"
    SNOWY = "snowy"
    WINDY = "windy"
    HOT = "hot"

@dataclass
class TransportRoute:
    """Enhanced transport route with comprehensive information"""
    origin: str
    destination: str
    transport_modes: List[TransportMode]
    total_duration_minutes: int
    total_cost_tl: float
    walking_duration_minutes: int
    accessibility_rating: float
    weather_suitability: Dict[WeatherCondition, float]
    crowding_level: str
    cultural_notes: List[str]
    real_time_tips: List[str]
    backup_options: List[str]

@dataclass
class IstanbulKartGuide:
    """Complete Istanbul Kart usage guide"""
    where_to_buy: List[str]
    how_to_load: Dict[str, str]
    usage_tips: List[str]
    cost_benefits: Dict[str, float]
    troubleshooting: Dict[str, str]

class EnhancedTransportationAdvisor:
    """Comprehensive Istanbul transportation advisory system"""
    
    def __init__(self):
        self.metro_lines = self._load_metro_system()
        self.bus_routes = self._load_bus_system()
        self.ferry_routes = self._load_ferry_system()
        self.accessibility_info = self._load_accessibility_data()
        self.cultural_etiquette = self._load_transport_etiquette()
        self.istanbul_kart_guide = self._create_istanbul_kart_guide()
        
    def _create_istanbul_kart_guide(self) -> IstanbulKartGuide:
        """Create comprehensive Istanbul Kart usage guide"""
        return IstanbulKartGuide(
            where_to_buy=[
                "🏢 All metro stations (automated machines and ticket offices)",
                "✈️ Istanbul Airport (arrivals hall, multiple locations)",
                "✈️ Sabiha Gökçen Airport (domestic and international terminals)",
                "🚢 All major ferry terminals (Eminönü, Karaköy, Kadıköy, Üsküdar)",
                "🏪 Thousands of shops with 'İstanbulkart' signs",
                "📱 BiTaksi app (digital delivery to your location)",
                "🏦 Selected bank branches (Garanti, İş Bankası, etc.)",
                "🎫 Major tourist information centers"
            ],
            how_to_load={
                "at_machines": "Insert card → Select 'Load Money' → Choose amount → Pay cash/card → Take receipt",
                "at_counters": "Show card → Say amount in Turkish ('Yirmi lira yükleyin' = Load 20 lira) → Pay → Get receipt",
                "mobile_app": "Download İstanbulkart app → Register card → Add payment method → Load remotely",
                "online": "Visit istanbulkart.iett.gov.tr → Register → Add card → Load via credit card",
                "automatic_reload": "Set up auto-reload in app when balance drops below chosen amount"
            },
            usage_tips=[
                "💳 Keep at least 20 TL balance for multiple trips",
                "📱 Check balance by holding card near any validator",
                "🔄 One card works for ALL Istanbul public transport",
                "👥 Each person needs their own card (sharing not allowed)",
                "⏱️ Transfer discount applies within 2 hours of first tap",
                "🚇 Tap when entering AND exiting metro (important!)",
                "🚌 Tap only when boarding buses",
                "⛴️ Tap when boarding ferries",
                "🎒 Keep card in easily accessible pocket/wallet section",
                "💡 Blue light + beep = successful payment, red = insufficient funds"
            ],
            cost_benefits={
                "metro_single": 8.0,
                "metro_with_card": 7.67,
                "bus_single": 8.0,
                "bus_with_card": 7.67,
                "ferry_single": 15.0,
                "ferry_with_card": 14.25,
                "transfer_discount": 2.0  # Amount saved on transfers
            },
            troubleshooting={
                "card_not_working": "Hold card flat against reader for 2 seconds, try different angle",
                "insufficient_funds": "Load money at nearest machine or authorized shop",
                "lost_card": "Report immediately at any metro station, balance can be transferred to new card",
                "double_charge": "Keep receipts, report at customer service within 15 days for refund",
                "blocked_card": "Usually due to suspicious activity, visit customer service with ID"
            }
        )
    
    def _load_metro_system(self) -> Dict[str, Any]:
        """Load comprehensive metro system data"""
        return {
            "M1A": {
                "name": "Yenikapı - Atatürk Airport (Old)",
                "color": "red",
                "stations": 23,
                "frequency_peak": "3-4 minutes",
                "frequency_off_peak": "5-7 minutes",
                "first_train": "06:00",
                "last_train": "00:30",
                "accessibility": "Full",
                "key_connections": ["Yenikapı (M2, ferry)", "Zeytinburnu (T1)", "Otogar (bus terminal)"],
                "tourist_stations": ["Aksaray", "Beyazıt-Kapalıçarşı (Grand Bazaar)", "Zeytinburnu"],
                "rush_hour_tips": [
                    "Extremely crowded 07:30-09:30 and 17:30-19:30",
                    "Alternative: Take M1B to Kirazlı then transfer",
                    "Airport travelers have priority seating with luggage"
                ]
            },
            "M1B": {
                "name": "Yenikapı - Kirazlı",
                "color": "red", 
                "stations": 11,
                "frequency_peak": "4-5 minutes",
                "frequency_off_peak": "6-8 minutes",
                "first_train": "06:00",
                "last_train": "00:30",
                "accessibility": "Full",
                "key_connections": ["Yenikapı (M2, ferry)", "Kirazlı (M3)"],
                "tourist_stations": ["Aksaray", "Beyazıt-Kapalıçarşı"],
                "local_tip": "Less crowded than M1A, good alternative route"
            },
            "M2": {
                "name": "Yenikapı - Hacıosman",
                "color": "green",
                "stations": 16,
                "frequency_peak": "2-3 minutes",
                "frequency_off_peak": "4-6 minutes", 
                "first_train": "06:15",
                "last_train": "00:30",
                "accessibility": "Full",
                "key_connections": ["Yenikapı (M1, ferry)", "Şişhane (T1)", "Vezneciler (T1)"],
                "tourist_stations": ["Şişhane (Galata Tower)", "Taksim", "Osmanbey", "Levent (shopping)"],
                "rush_hour_tips": [
                    "Busiest line in Istanbul during rush hours",
                    "Taksim-Levent section extremely crowded",
                    "Consider surface alternatives during peak times"
                ]
            },
            "M3": {
                "name": "Olympiaköy - Başakşehir",
                "color": "blue",
                "stations": 11,
                "frequency_peak": "5-6 minutes",
                "frequency_off_peak": "8-10 minutes",
                "first_train": "06:00",
                "last_train": "00:00",
                "accessibility": "Full",
                "key_connections": ["Kirazlı (M1B)", "Başakşehir (M7)"],
                "tourist_value": "Low - mainly residential areas",
                "local_tip": "Connects to outlet shopping at İstanbul Outlet Center"
            },
            "M4": {
                "name": "Kadıköy - Tavşantepe",
                "color": "pink",
                "stations": 19,
                "frequency_peak": "3-4 minutes", 
                "frequency_off_peak": "5-7 minutes",
                "first_train": "06:00",
                "last_train": "00:30",
                "accessibility": "Full",
                "key_connections": ["Kadıköy (ferry, bus)", "Ayrılık Çeşmesi (T3)"],
                "tourist_stations": ["Kadıköy (Asian side hub)", "Bostancı (ferry to Prince Islands)"],
                "asian_side_tip": "Primary metro line for Asian side exploration"
            },
            "M5": {
                "name": "Üsküdar - Yamanevler", 
                "color": "purple",
                "stations": 16,
                "frequency_peak": "4-5 minutes",
                "frequency_off_peak": "6-8 minutes",
                "first_train": "06:00",
                "last_train": "00:30",
                "accessibility": "Full",
                "key_connections": ["Üsküdar (ferry, bus)"],
                "tourist_stations": ["Üsküdar (ferry terminal, Maiden's Tower views)"],
                "local_insight": "Serves residential areas, less tourist-focused"
            },
            "M6": {
                "name": "Levent - Boğaziçi Üniversitesi",
                "color": "brown",
                "stations": 4,
                "frequency_peak": "7-8 minutes",
                "frequency_off_peak": "10-12 minutes",
                "first_train": "06:30",
                "last_train": "23:30",
                "accessibility": "Full",
                "key_connections": ["Levent (M2)"],
                "special_note": "Short line serving Bosphorus University area"
            },
            "M7": {
                "name": "Mecidiyeköy - Mahmutbey",
                "color": "light_blue",
                "stations": 15,
                "frequency_peak": "4-5 minutes",
                "frequency_off_peak": "6-8 minutes",
                "first_train": "06:00",
                "last_train": "00:30",
                "accessibility": "Full",
                "key_connections": ["Mecidiyeköy (M2)", "Mahmutbey (M3)"],
                "tourist_value": "Medium - connects business districts"
            },
            "M11": {
                "name": "Gayrettepe - İstanbul Airport",
                "color": "dark_blue",
                "stations": 9,
                "frequency_peak": "10-12 minutes",
                "frequency_off_peak": "15-20 minutes",
                "first_train": "06:00",
                "last_train": "01:15",
                "accessibility": "Full",
                "key_connections": ["Gayrettepe (M2)", "İstanbul Airport"],
                "tourist_essential": "Only direct rail connection to new Istanbul Airport",
                "airport_tips": [
                    "Allow extra time - airport is VERY large",
                    "Free shuttle between metro and terminals",
                    "Can be crowded during flight peaks"
                ]
            }
        }

    def _load_ferry_system(self) -> Dict[str, Any]:
        """Load comprehensive ferry system data"""
        return {
            "bosphorus_regular": {
                "routes": [
                    "Eminönü ↔️ Üsküdar",
                    "Karaköy ↔️ Haydarpaşa", 
                    "Beşiktaş ↔️ Üsküdar",
                    "Kabataş ↔️ Üsküdar"
                ],
                "frequency": "15-20 minutes",
                "duration": "20-25 minutes",
                "cost": "15 TL",
                "scenic_value": "High",
                "operating_hours": "06:00-23:00",
                "weather_sensitivity": "Medium",
                "insider_tips": [
                    "Right side (starboard) offers better Bosphorus views on Üsküdar route",
                    "Early morning ferries less crowded, perfect for photos",
                    "Sunset ferries (18:00-19:30) are magical but crowded",
                    "Outdoor deck closes in bad weather"
                ]
            },
            "princes_islands": {
                "main_route": "Kabataş → Büyükada/Heybeliada/Burgazada/Kınalıada",
                "alternative_route": "Bostancı → Islands (shorter from Asian side)",
                "frequency": "60-90 minutes (seasonal)",
                "duration": "90-120 minutes to Büyükada", 
                "cost": "35-50 TL (depends on island)",
                "seasonal_operation": "Reduced service November-March",
                "weather_sensitivity": "High",
                "booking_essential": "Summer weekends",
                "island_specialties": {
                    "Büyükada": "Horse carriages, Victorian mansions, swimming spots",
                    "Heybeliada": "Naval academy, pine forests, quieter beaches",
                    "Burgazada": "Smallest, most peaceful, Sait Faik Museum",
                    "Kınalıada": "Closest to mainland, day trip friendly"
                },
                "planning_tips": [
                    "Allow full day for Büyükada exploration",
                    "Pack lunch - island restaurants expensive",
                    "No cars on islands - walking/cycling/horse carriages only",
                    "Last ferries return around 19:00-20:00 in summer"
                ]
            },
            "golden_horn": {
                "route": "Eminönü → Kasımpaşa → Fener → Balat → Eyüp",
                "frequency": "30-40 minutes",
                "duration": "45-60 minutes full route",
                "cost": "15-20 TL",
                "cultural_value": "Very High",
                "operating_hours": "07:00-19:00",
                "special_features": [
                    "Passes historic neighborhoods",
                    "Perfect for Balat district exploration",
                    "Eyüp Sultan Mosque accessible",
                    "Less touristy than Bosphorus ferries"
                ]
            },
            "bosphorus_tour": {
                "full_tour": "Eminönü → Beşiktaş → Ortaköy → Arnavutköy → Sarıyer → Anadolu Kavağı",
                "short_tour": "Eminönü → Üsküdar → Maiden's Tower area",
                "frequency": "Daily departures 10:35, 13:35",
                "duration": "6+ hours full tour, 2 hours short tour",
                "cost": "35-100 TL (depends on tour type)",
                "seasonal_schedule": "More frequent in summer",
                "booking_recommended": "Especially weekends and summer",
                "tour_highlights": [
                    "Dolmabahçe Palace from water",
                    "Bosphorus bridges close-up",
                    "Historic waterfront mansions (yalı)",
                    "Asian and European shores comparison",
                    "Photo stops at scenic points"
                ]
            }
        }

    def _load_accessibility_data(self) -> Dict[str, Any]:
        """Load comprehensive accessibility information"""
        return {
            "metro_accessibility": {
                "fully_accessible_lines": ["M1A", "M1B", "M2", "M3", "M4", "M5", "M6", "M7", "M11"],
                "elevator_locations": {
                    "all_stations": "Every metro station has at least one elevator",
                    "backup_elevators": "Major stations have 2+ elevators",
                    "elevator_status": "Check İETT app for real-time elevator status"
                },
                "wheelchair_features": [
                    "Wide turnstiles at all stations (90cm+)",
                    "Tactile pavement for visually impaired",
                    "Audio announcements in Turkish and English",
                    "Designated wheelchair areas in trains",
                    "Emergency call buttons accessible from wheelchair height"
                ],
                "assistance_services": [
                    "Staff assistance available at major stations",
                    "Call +90 212 568 99 70 for pre-planned assistance",
                    "Mobility device rental at some stations",
                    "Companion travel discounts available"
                ]
            },
            "bus_accessibility": {
                "accessible_buses": "All new buses (majority of fleet) are low-floor",
                "wheelchair_ramps": "Automatic ramps on accessible buses",
                "priority_seating": "Front section reserved for mobility needs",
                "audio_visual": "Stop announcements in Turkish, visual displays",
                "assistance_tips": [
                    "Signal driver if you need ramp assistance",
                    "MetroBüs system fully accessible",
                    "Some older buses still not accessible - check before boarding"
                ]
            },
            "ferry_accessibility": {
                "accessible_terminals": [
                    "Eminönü - Full accessibility, elevators, ramps",
                    "Karaköy - Accessible pier, elevator access",
                    "Kadıköy - Ramp access, accessible facilities",
                    "Üsküdar - Modern accessible terminal",
                    "Beşiktaş - Limited accessibility, some stairs"
                ],
                "boat_accessibility": [
                    "Most ferries have wheelchair-accessible boarding",
                    "Staff assistance available for boarding",
                    "Dedicated wheelchair spaces on boats",
                    "Accessible bathrooms on larger ferries"
                ],
                "booking_assistance": "Call İDO at +90 212 444 44 36 for accessibility planning"
            }
        }

    def _load_transport_etiquette(self) -> Dict[str, List[str]]:
        """Load cultural etiquette for Istanbul transport"""
        return {
            "metro_etiquette": [
                "🚇 Stand right on escalators, walk left",
                "👥 Offer priority seats to elderly, pregnant women, disabled passengers", 
                "🎒 Remove backpack in crowded trains",
                "📱 Keep voice down on phone calls",
                "🚪 Let passengers exit before boarding",
                "🎵 Use headphones for music",
                "💺 Don't put feet on seats",
                "🍽️ Avoid eating messy foods",
                "👀 Avoid staring - people read, use phones quietly"
            ],
            "bus_etiquette": [
                "🚌 Enter from front, exit from rear (when possible)",
                "💳 Have İstanbulkart ready before boarding",
                "👴 Always offer seats to elderly and pregnant women",
                "🗣️ Say 'İnecek var' (someone getting off) when you want to exit",
                "🚪 Move to center/back to let others board",
                "💰 Don't pay cash - İstanbulkart only on most buses",
                "👥 Help others with heavy bags when appropriate"
            ],
            "ferry_etiquette": [
                "⛴️ Remove shoes if entering carpeted prayer area",
                "☕ Enjoy tea service - it's part of the experience", 
                "📸 Ask before photographing people",
                "🧳 Secure loose items due to wind/movement",
                "🎫 Keep ticket until end of journey",
                "🌅 Share scenic viewing areas politely",
                "💰 Tips appreciated for tea service",
                "🦢 Don't feed seagulls - they can be aggressive"
            ],
            "dolmus_etiquette": [
                "🚐 Wait for complete stop before boarding",
                "💳 Pay when boarding, exact change preferred",
                "🗣️ Say 'İniyorum' (I'm getting off) when you want to exit",
                "👥 Move over to make room for others",
                "🛣️ Routes can be confusing - ask driver if unsure",
                "⏰ More frequent during rush hours",
                "🎵 Music/conversation normal - more social than metro"
            ],
            "general_courtesy": [
                "🙋‍♂️ Don't hesitate to ask for help - Istanbulites are friendly",
                "📱 Learn basic Turkish phrases: 'Teşekkürler' (Thank you), 'Özür dilerim' (Excuse me)",
                "🎒 Keep bags secure - pickpocketing rare but possible in crowded areas",
                "💡 Download translation app for complex questions",
                "🗺️ Keep physical map as backup - cell service can be spotty underground"
            ]
        }

    def get_comprehensive_route_advice(self, origin: str, destination: str,
                                     user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive route advice with all enhancements"""
        
        # Extract user preferences
        budget = user_preferences.get('budget', 'moderate')  # budget/moderate/premium
        speed_preference = user_preferences.get('speed', 'balanced')  # fast/balanced/scenic
        accessibility_needs = user_preferences.get('accessibility', [])
        weather = WeatherCondition(user_preferences.get('weather', 'clear'))
        time_of_day = user_preferences.get('time', datetime.now().strftime('%H:%M'))
        
        # Generate multiple route options
        routes = []
        
        # Primary optimized route
        primary_route = self._calculate_optimal_route(
            origin, destination, budget, speed_preference, accessibility_needs, weather
        )
        routes.append(primary_route)
        
        # Alternative routes
        if speed_preference != 'fast':
            scenic_route = self._calculate_scenic_route(origin, destination, weather)
            routes.append(scenic_route)
            
        if budget != 'budget':
            comfort_route = self._calculate_comfort_route(origin, destination, accessibility_needs)
            routes.append(comfort_route)
        
        return {
            'route_options': routes,
            'istanbul_kart_guidance': self._get_contextual_kart_tips(routes),
            'real_time_recommendations': self._get_real_time_apps_guidance(),
            'weather_adaptations': self._get_weather_specific_tips(weather, routes),
            'accessibility_guidance': self._get_accessibility_route_tips(accessibility_needs, routes),
            'cultural_etiquette': self._get_relevant_etiquette(routes),
            'cost_optimization': self._calculate_cost_savings(routes),
            'emergency_contacts': self._get_transport_emergency_info(),
            'timing_intelligence': self._get_timing_optimization(time_of_day, routes)
        }

    def _get_real_time_apps_guidance(self) -> Dict[str, Any]:
        """Provide comprehensive real-time app recommendations"""
        return {
            "essential_apps": {
                "Moovit": {
                    "description": "Best overall public transport app for Istanbul",
                    "features": [
                        "Real-time arrival predictions",
                        "Multi-modal journey planning", 
                        "Works offline with downloaded maps",
                        "Available in English",
                        "Crowdsourced delay reports"
                    ],
                    "best_for": "First-time visitors and complex multi-modal trips"
                },
                "İETT Mobil": {
                    "description": "Official Istanbul bus system app",
                    "features": [
                        "Real-time bus locations",
                        "Route planning for buses",
                        "İstanbulkart balance checking",
                        "Service disruption alerts"
                    ],
                    "best_for": "Bus-focused travel and İstanbulkart management"
                },
                "Metro İstanbul": {
                    "description": "Official metro system app",
                    "features": [
                        "Metro line status updates",
                        "Station information and maps",
                        "Elevator/escalator status",
                        "Real-time train arrivals"
                    ],
                    "best_for": "Metro system navigation and accessibility planning"
                },
                "İDO": {
                    "description": "Official ferry service app",
                    "features": [
                        "Ferry schedules and real-time updates",
                        "Ticket booking for tourist ferries",
                        "Route information and pricing",
                        "Weather-related service changes"
                    ],
                    "best_for": "Ferry travel planning and bookings"
                }
            },
            "setup_tips": [
                "📱 Download Moovit first - most comprehensive for tourists",
                "🌐 Enable location services for accurate real-time info",
                "💾 Download offline maps for subway connectivity",
                "🔔 Enable push notifications for service alerts",
                "🗺️ Keep Google Maps as backup - good for walking directions"
            ],
            "usage_strategies": [
                "📊 Check multiple apps for conflicting information",
                "⏰ Real-time data can be 2-3 minutes delayed",
                "🚇 Metro apps more accurate than bus apps",
                "⛴️ Ferry schedules change seasonally - always verify",
                "👥 Crowdsourced reports (Moovit) often most current"
            ]
        }

    def _get_weather_specific_tips(self, weather: WeatherCondition, routes: List[TransportRoute]) -> List[str]:
        """Generate weather-specific transportation advice"""
        tips = []
        
        if weather == WeatherCondition.RAINY:
            tips.extend([
                "☂️ Metro stations can have slippery marble floors - walk carefully",
                "🚇 Underground transport (metro/subway) strongly preferred over buses",
                "⛴️ Ferry services may be reduced or cancelled in severe weather",
                "🌧️ Covered walkways available at: Taksim, Şişhane, Levent, Mecidiyeköy",
                "👢 Waterproof shoes recommended - some station entrances flood slightly",
                "📱 Check İDO app before ferry trips - weather updates posted there",
                "🚌 Bus stops often lack proper shelter - plan accordingly"
            ])
        elif weather == WeatherCondition.SNOWY:
            tips.extend([
                "❄️ Metro continues normal operation - best option in snow",
                "🚌 Bus services may be delayed or cancelled on steep routes", 
                "⛴️ Ferry services often suspended in heavy snow/ice",
                "🧊 Station entrances can be icy - use handrails",
                "👟 Wear shoes with good grip - marble floors become slippery",
                "⏰ Allow 50-100% extra travel time",
                "🔥 Metro stations provide warm shelter during waits"
            ])
        elif weather == WeatherCondition.HOT:
            tips.extend([
                "🌡️ Metro cars are air-conditioned - pleasant escape from heat",
                "💧 Carry water - dehydration happens quickly in crowds",
                "⛴️ Ferry top decks can be extremely hot midday - use lower decks",
                "🚌 Older buses may lack adequate AC - metro preferred",
                "🧴 Sunscreen essential for ferry trips and outdoor waiting",
                "⏰ Early morning (before 10 AM) or evening (after 6 PM) travel recommended",
                "🌬️ Bosphorus ferries provide cooling sea breeze"
            ])
        elif weather == WeatherCondition.WINDY:
            tips.extend([
                "💨 Ferry services may experience delays or cancellations",
                "⛴️ Upper decks of ferries can be uncomfortable - use indoor areas",
                "🌊 Bosphorus crossings may be rough - consider bridge alternatives",
                "🚇 Metro unaffected by wind - most reliable option",
                "👒 Secure loose items on ferries and outdoor platforms"
            ])
        
        return tips

    def _calculate_optimal_route(self, origin: str, destination: str, budget: str,
                               speed_preference: str, accessibility_needs: List[str],
                               weather: WeatherCondition) -> TransportRoute:
        """Calculate optimal route based on preferences"""
        
        # This is a simplified example - in production would use actual routing algorithms
        return TransportRoute(
            origin=origin,
            destination=destination,
            transport_modes=[TransportMode.METRO, TransportMode.WALKING],
            total_duration_minutes=45,
            total_cost_tl=15.34,
            walking_duration_minutes=12,
            accessibility_rating=0.95,
            weather_suitability={weather: 0.9},
            crowding_level="moderate",
            cultural_notes=[
                "Pass through historic Beyazıt district",
                "Grand Bazaar area - great for quick stop"
            ],
            real_time_tips=[
                "M1 line can be crowded during morning rush",
                "Alternative T1 tram available from same station"
            ],
            backup_options=[
                "Bus + ferry combination (scenic but longer)",
                "Taxi during off-peak hours (more expensive)"
            ]
        )

    def _calculate_scenic_route(self, origin: str, destination: str, 
                              weather: WeatherCondition) -> TransportRoute:
        """Calculate scenic route prioritizing views and experience"""
        
        # Prioritize ferry routes for scenic value
        if any(term in origin.lower() + destination.lower() for term in 
               ['eminönü', 'karaköy', 'üsküdar', 'kadıköy', 'beşiktaş']):
            return TransportRoute(
                origin=origin,
                destination=destination,
                transport_modes=[TransportMode.FERRY, TransportMode.WALKING],
                total_duration_minutes=60,
                total_cost_tl=23.25,
                walking_duration_minutes=15,
                accessibility_rating=0.85,
                weather_suitability={weather: 0.7 if weather in [WeatherCondition.WINDY, WeatherCondition.RAINY] else 0.95},
                crowding_level="moderate",
                cultural_notes=[
                    "Spectacular Bosphorus views during crossing",
                    "Historic waterfront neighborhoods visible"
                ],
                real_time_tips=[
                    "Best views from upper deck on right side",
                    "Golden hour timing (sunset) for photos"
                ],
                backup_options=[
                    "Metro + tram combination",
                    "Scenic bus route along coast"
                ]
            )
        else:
            # Default to metro with cultural stops
            return self._calculate_optimal_route(origin, destination, 'moderate', 'scenic', [], weather)
    
    def _calculate_comfort_route(self, origin: str, destination: str, 
                               accessibility_needs: List[str]) -> TransportRoute:
        """Calculate route prioritizing comfort and accessibility"""
        
        accessibility_rating = 0.95 if accessibility_needs else 0.8
        
        return TransportRoute(
            origin=origin,
            destination=destination,
            transport_modes=[TransportMode.METRO, TransportMode.WALKING],
            total_duration_minutes=40,
            total_cost_tl=18.50,
            walking_duration_minutes=8,
            accessibility_rating=accessibility_rating,
            weather_suitability={WeatherCondition.CLEAR: 0.95},
            crowding_level="low",
            cultural_notes=[
                "Air-conditioned comfortable journey",
                "Minimal walking required"
            ],
            real_time_tips=[
                "All stations fully accessible",
                "Staff assistance available"
            ],
            backup_options=[
                "Taxi with accessibility features",
                "Premium bus service"
            ]
        )

    def _load_bus_system(self) -> Dict[str, Any]:
        """Load comprehensive bus system data"""
        return {
            "metrobus": {
                "name": "MetroBüs (BRT System)",
                "route": "Beylikdüzü ↔️ Söğütlüçeşme",
                "stations": 45,
                "frequency_peak": "30-60 seconds",
                "frequency_off_peak": "2-3 minutes",
                "first_bus": "05:30",
                "last_bus": "02:00",
                "cost": "7.67 TL with İstanbulkart",
                "accessibility": "Full - level boarding",
                "key_stations": [
                    "Avcılar (University connection)",
                    "Beylikdüzü (Shopping, residential)",
                    "Mecidiyeköy (M2 metro connection)",
                    "Zincirlikuyu (Business district)",
                    "Bosphorus Bridge",
                    "Üsküdar (M5 metro, ferry connections)",
                    "Kadıköy (M4 metro, ferry connections)"
                ],
                "insider_tips": [
                    "Fastest way to cross the Bosphorus during traffic",
                    "Dedicated lanes - unaffected by car traffic",
                    "Very crowded during rush hours - consider timing",
                    "Air conditioned and modern buses"
                ]
            },
            "regular_buses": {
                "overview": "500+ routes covering all Istanbul districts",
                "payment": "İstanbulkart only (no cash accepted)",
                "frequency": "5-20 minutes depending on route",
                "cost": "7.67 TL with İstanbulkart",
                "accessibility": "Most new buses accessible",
                "key_routes": {
                    "25E": "Eminönü ↔️ Sarıyer (Bosphorus coastal route)",
                    "40": "Taksim ↔️ Sarıyer", 
                    "99A": "Kadıköy ↔️ Üsküdar",
                    "146": "Taksim ↔️ İstanbul Airport"
                },
                "etiquette_tips": [
                    "Enter from front, exit from middle/rear doors",
                    "Have İstanbulkart ready before boarding",
                    "Say 'İnecek var' when you want to get off",
                    "Priority seats for elderly and pregnant women"
                ]
            },
            "night_buses": {
                "overview": "Limited night service 00:00-05:30",
                "key_routes": [
                    "139N: Taksim → Airport",
                    "132N: Kadıköy → Levent", 
                    "40N: Taksim → Sarıyer"
                ],
                "frequency": "30-60 minutes",
                "cost": "Same as regular buses",
                "tip": "Limited routes - plan carefully or use taxi"
            }
        }

    def _get_accessibility_route_tips(self, accessibility_needs: List[str], 
                                    routes: List[TransportRoute]) -> Dict[str, Any]:
        """Generate accessibility-specific route guidance"""
        
        tips = {
            "elevator_status": [
                "📱 Check Metro İstanbul app for real-time elevator status",
                "🔧 Report elevator issues immediately to station staff",
                "🆘 Emergency assistance button available in all elevators",
                "♿ All metro stations have accessible bathrooms"
            ],
            "station_specifics": {
                "Taksim": "2 elevators + accessible entrance on İnönü Street side",
                "Şişhane": "1 elevator but multiple escalators, generally reliable",
                "Kadıköy": "Ferry terminal fully accessible, metro connection good",
                "Eminönü": "Ferry accessible but old building - some challenges"
            },
            "alternative_routes": [
                "🚌 MetroBüs system fully accessible with level boarding",
                "⛴️ All major ferry terminals have ramp access",
                "🚕 Taxi alternative available - most accept wheelchairs",
                "🚐 Accessible airport shuttles operate on main routes"
            ],
            "assistance_contacts": {
                "Metro": "+90 212 568 99 70",
                "Bus/İETT": "+90 212 455 06 00", 
                "Ferry/İDO": "+90 212 444 44 36",
                "Emergency": "155 (police), 112 (medical)"
            }
        }
        
        if "wheelchair" in accessibility_needs:
            tips["wheelchair_specific"] = [
                "🚪 Use wide turnstiles - look for wheelchair symbol",
                "⚡ Priority boarding on all transport types",
                "👥 Don't hesitate to ask for help - people are helpful",
                "🎒 Secure loose items - ramps can be steep"
            ]
            
        if "visual_impairment" in accessibility_needs:
            tips["visual_support"] = [
                "🔊 Audio announcements on metro in Turkish and English",
                "👣 Tactile pavement leads to platform edges",
                "📱 Be My Eyes app connects to Turkish volunteers",
                "🦮 Guide dogs welcome on all public transport"
            ]
            
        return tips

    def get_cost_optimization_guide(self, routes: List[TransportRoute]) -> Dict[str, Any]:
        """Provide comprehensive cost optimization strategies"""
        
        return {
            "istanbul_kart_savings": {
                "single_ride": "Cash: 10 TL, İstanbulkart: 7.67 TL",
                "transfer_discount": "2nd ride within 2 hours: 2 TL discount",
                "monthly_savings": "Heavy users save 30-40% with İstanbulkart",
                "family_tip": "Each person needs own card - no sharing allowed"
            },
            "transfer_strategies": [
                "🔄 Plan routes with transfers to maximize 2-hour discount window",
                "🚇 Metro-to-metro transfers often free within stations",
                "⛴️ Ferry-to-bus transfers get full discount",
                "🚌 Bus-to-metro transfers most economical combination"
            ],
            "timing_optimizations": [
                "📅 No weekend or off-peak discounts - prices consistent",
                "⏰ Early morning ferries (7-9 AM) sometimes have promotions",
                "🎫 Tourist passes only worth it for 3+ days of heavy use",
                "💳 Student discounts available with Turkish student ID"
            ],
            "alternative_savings": [
                "🚶‍♂️ Walk between close metro stations to save a ride",
                "🚢 Regular ferries cheaper than tourist boat tours",
                "🚐 Dolmuş routes sometimes cheaper than buses",
                "🚕 Shared taxis (dolmuş) more economical than private taxis"
            ],
            "budget_breakdown": {
                "daily_light_use": "20-30 TL (3-4 rides)",
                "daily_heavy_use": "40-60 TL (6-8 rides with transfers)",
                "weekly_tourist": "150-250 TL (comprehensive exploration)",
                "monthly_local": "300-500 TL (daily commuting)"
            }
        }

    def get_rush_hour_intelligence(self, time_of_day: str) -> Dict[str, Any]:
        """Provide intelligent rush hour guidance"""
        
        hour = int(time_of_day.split(':')[0])
        
        if 7 <= hour <= 9:
            return {
                "status": "MORNING RUSH HOUR",
                "severity": "HIGH",
                "affected_lines": ["M2", "M1A", "M4", "All buses"],
                "recommendations": [
                    "🚇 M2 Taksim-Levent extremely crowded - consider alternatives",
                    "🚌 Buses 30-50% slower due to traffic",
                    "⛴️ Ferries less affected - good alternative",
                    "⏰ Travel before 7:30 AM or after 9:30 AM if possible"
                ],
                "crowding_levels": {
                    "M2_taksim_levent": "Extreme",
                    "M1A_airport": "High", 
                    "M4_kadikoy": "High",
                    "Ferries": "Moderate"
                },
                "insider_tips": [
                    "☕ Grab breakfast at station - avoid street level crowds",
                    "📱 Use real-time apps - delays common during rush",
                    "🎒 Keep bag in front to avoid pickpockets",
                    "👥 Let several trains pass if you're not in hurry"
                ]
            }
        elif 17 <= hour <= 19:
            return {
                "status": "EVENING RUSH HOUR",
                "severity": "VERY HIGH",
                "affected_lines": ["ALL LINES", "ALL BUSES"],
                "recommendations": [
                    "🏠 Worst traffic period of the day",
                    "🚇 All metro lines crowded - no good alternatives",
                    "⛴️ Sunset ferries busy but manageable",
                    "⏰ Consider dining out and traveling after 8 PM"
                ],
                "crowding_levels": {
                    "All_metro": "Extreme",
                    "All_buses": "Extreme",
                    "Ferries": "High"
                },
                "cultural_insight": [
                    "🍽️ Many locals go directly to dinner - explore food areas",
                    "🌆 Perfect time for rooftop restaurants while transport clears",
                    "☕ Traditional tea time - join locals in cafes",
                    "🛍️ Shops stay open late - good time for evening shopping"
                ]
            }
        else:
            return {
                "status": "OFF-PEAK",
                "severity": "LOW",
                "recommendations": [
                    "✅ Optimal travel time - all routes operating normally",
                    "🚇 Short wait times on all metro lines",
                    "📸 Great time for photography without crowds",
                    "🎯 Perfect for sightseeing-focused routes"
                ]
            }

    def _get_contextual_kart_tips(self, routes: List[TransportRoute]) -> Dict[str, Any]:
        """Get contextual Istanbul Kart tips based on routes"""
        
        tips = {
            "purchase_priority": "High - required for all public transport",
            "loading_recommendation": "Load 50-100 TL for multi-day use",
            "transfer_strategy": "Plan connections within 2-hour window for discounts",
            "route_specific_savings": []
        }
        
        # Analyze routes for specific savings opportunities
        total_cost = sum(route.total_cost_tl for route in routes)
        if total_cost > 50:
            tips["route_specific_savings"].append(
                f"Heavy usage detected - İstanbulkart saves ~15% ({total_cost*0.15:.2f} TL)"
            )
        
        return tips
    
    def _calculate_cost_savings(self, routes: List[TransportRoute]) -> Dict[str, Any]:
        """Calculate cost optimization opportunities"""
        
        total_cost = sum(route.total_cost_tl for route in routes)
        
        savings = {
            "istanbul_kart_savings": total_cost * 0.15,
            "transfer_discounts": 6.0,  # Estimated transfer savings per day
            "weekly_optimization": total_cost * 7 * 0.1,  # 10% savings with planning
            "alternatives": [
                "Walking between close stations saves 7.67 TL per skip",
                "Ferry+metro combinations often cheaper than all-metro routes",
                "Early morning ferries sometimes have promotional pricing"
            ]
        }
        
        return savings
    
    def _get_transport_emergency_info(self) -> Dict[str, str]:
        """Get emergency contact information for transport"""
        return {
            "Metro": "+90 212 568 99 70",
            "Bus/İETT": "+90 212 455 06 00",
            "Ferry/İDO": "+90 212 444 44 36",
            "Police": "155",
            "Medical Emergency": "112",
            "Tourist Police": "+90 212 527 45 03"
        }
    
    def _get_timing_optimization(self, time_of_day: str, routes: List[TransportRoute]) -> Dict[str, Any]:
        """Get timing optimization recommendations"""
        
        hour = int(time_of_day.split(':')[0])
        
        optimization = {
            "current_conditions": "normal",
            "recommendations": [],
            "alternative_timing": []
        }
        
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            optimization["current_conditions"] = "rush_hour"
            optimization["recommendations"] = [
                "Expect significant crowding on all transport",
                "Allow 50% extra travel time",
                "Consider ferry alternatives during peak metro hours"
            ]
            optimization["alternative_timing"] = [
                "Travel before 7:30 AM for less crowding",
                "Wait until after 9:30 AM if flexible",
                "Evening: travel after 8:00 PM for comfort"
            ]
        elif 22 <= hour or hour <= 6:
            optimization["current_conditions"] = "limited_service"
            optimization["recommendations"] = [
                "Limited night bus service available",
                "Last metro trains around 00:30",
                "Taxi recommended for late night travel"
            ]
        
        return optimization
    
    def _get_relevant_etiquette(self, routes: List[TransportRoute]) -> List[str]:
        """Get relevant cultural etiquette based on transport modes used"""
        
        etiquette_tips = []
        transport_modes = set()
        
        for route in routes:
            transport_modes.update(mode.value for mode in route.transport_modes)
        
        if 'metro' in transport_modes:
            etiquette_tips.extend([
                "🚇 Stand right on escalators, walk left",
                "👥 Offer priority seats to elderly, pregnant women, disabled passengers",
                "🚪 Let passengers exit before boarding"
            ])
        
        if 'ferry' in transport_modes:
            etiquette_tips.extend([
                "⛴️ Remove shoes if entering carpeted prayer area",
                "☕ Enjoy tea service - it's part of the experience",
                "📸 Ask before photographing people"
            ])
        
        if 'bus' in transport_modes:
            etiquette_tips.extend([
                "🚌 Enter from front, exit from rear when possible",
                "💳 Have İstanbulkart ready before boarding",
                "🗣️ Say 'İnecek var' when you want to exit"
            ])
        
        return etiquette_tips[:6]  # Limit to most important tips
def main():
    """Demo of enhanced transportation advisor"""
    advisor = EnhancedTransportationAdvisor()
    
    # Example route request
    user_preferences = {
        'budget': 'moderate',
        'speed': 'balanced',
        'accessibility': [],
        'weather': 'clear',
        'time': '08:30'
    }
    
    advice = advisor.get_comprehensive_route_advice(
        "Taksim", "Grand Bazaar", user_preferences
    )
    
    print("=== ENHANCED ISTANBUL TRANSPORTATION ADVISOR ===\n")
    print("🗺️ Route Options:")
    for i, route in enumerate(advice['route_options'], 1):
        print(f"\n{i}. {route.origin} → {route.destination}")
        print(f"   Duration: {route.total_duration_minutes} min | Cost: {route.total_cost_tl} TL")
        print(f"   Modes: {[mode.value for mode in route.transport_modes]}")
    
    print(f"\n💳 İstanbul Kart Guidance:")
    kart_guide = advisor.istanbul_kart_guide
    print(f"Where to buy: {kart_guide.where_to_buy[0]}")
    print(f"Key tip: {kart_guide.usage_tips[0]}")
    
    print(f"\n📱 Real-time Apps:")
    apps = advice['real_time_recommendations']['essential_apps']
    print(f"Primary recommendation: {list(apps.keys())[0]} - {apps[list(apps.keys())[0]]['description']}")
    
    print(f"\n⏰ Rush Hour Intelligence:")
    rush_info = advisor.get_rush_hour_intelligence("08:30")
    print(f"Status: {rush_info['status']} ({rush_info['severity']} severity)")
    print(f"Top tip: {rush_info['recommendations'][0]}")

if __name__ == "__main__":
    main()

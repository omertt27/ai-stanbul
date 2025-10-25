"""
Weather-Based Recommendations Service
Provides weather-appropriate activity recommendations for Istanbul
"""

from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WeatherRecommendationsService:
    """Weather-aware activity recommendations for Istanbul"""
    
    def __init__(self):
        self.hot_weather_activities = self._load_hot_weather_activities()
        self.cold_weather_activities = self._load_cold_weather_activities()
        self.rainy_day_activities = self._load_rainy_day_activities()
        self.mild_weather_activities = self._load_mild_weather_activities()
        
    def _load_hot_weather_activities(self) -> List[Dict]:
        """Load activities for hot weather (>28°C)"""
        return [
            {
                'name': 'Princes\' Islands Day Trip',
                'type': 'escape',
                'description': 'Cooler coastal temperatures, sea breeze, and beach clubs',
                'temp_difference': '-3 to -5°C cooler',
                'transport': 'Ferry from Kabataş (45 min) or Kadıköy (30 min)',
                'cost': 'Ferry: 15 TL, Bike rental: 50 TL/day',
                'duration': 'Full day (6-8 hours)',
                'best_islands': ['Büyükada', 'Heybeliada']
            },
            {
                'name': 'Şile Black Sea Beach',
                'type': 'beach',
                'description': 'Black Sea beaches with refreshing breeze, much cooler than city',
                'temp_difference': '-5 to -7°C cooler',
                'transport': 'Bus 139 from Üsküdar (90 min)',
                'cost': 'Bus: 15 TL, Beach clubs: 50-200 TL',
                'duration': 'Full day',
                'note': 'Waves can be strong, great for surfing'
            },
            {
                'name': 'Istanbul Modern Art Museum',
                'type': 'indoor',
                'description': 'Air-conditioned modern art museum with stunning Bosphorus views',
                'transport': 'Tophane tram station (5 min walk)',
                'cost': '200 TL, Free on Thursdays after 18:00',
                'duration': '2-3 hours',
                'bonus': 'Rooftop café with AC and panoramic views'
            },
            {
                'name': 'Basilica Cistern',
                'type': 'indoor',
                'description': 'Ancient underground water reservoir, naturally cool and mystical',
                'temp_difference': 'Naturally 15-20°C cooler inside',
                'transport': 'Sultanahmet tram station (2 min walk)',
                'cost': '450 TL',
                'duration': '45 min - 1 hour',
                'note': 'Famous Medusa head columns'
            },
            {
                'name': 'Bosphorus Evening Cruise',
                'type': 'water',
                'description': 'Wait for sunset, cooler evening breeze on the water',
                'best_time': 'After 18:00 when temperature drops',
                'transport': 'Multiple departure points (Eminönü, Kabataş)',
                'cost': 'Public ferry: 15 TL, Private cruise: 150-300 TL',
                'duration': '1.5-2 hours'
            },
            {
                'name': 'Emirgan Park',
                'type': 'outdoor',
                'description': 'Shaded park with large trees, cooler than city center',
                'transport': 'Bus from Kabataş or Taksim',
                'cost': 'Free',
                'duration': '2-3 hours',
                'activities': 'Walking, picnic, tea gardens'
            },
            {
                'name': 'Sapphire Observation Deck',
                'type': 'indoor',
                'description': 'Air-conditioned shopping mall with panoramic city views',
                'temp_difference': 'Fully air-conditioned',
                'transport': 'Metro M2 to 4. Levent',
                'cost': 'Mall free, Observation deck: 130 TL',
                'duration': '2-3 hours',
                'bonus': 'Shopping, restaurants, cinema'
            },
            {
                'name': 'Aquarium and Sea Life',
                'type': 'indoor',
                'description': 'Istanbul Aquarium or Türkiye Aquarium, cool indoor activity',
                'transport': 'Various locations (Florya, Beylikdüzü)',
                'cost': '200-300 TL',
                'duration': '2-3 hours',
                'good_for': 'Families with children'
            },
            {
                'name': 'Grand Bazaar Indoor Shopping',
                'type': 'indoor',
                'description': 'Historic covered market, naturally cooler indoors',
                'transport': 'Beyazıt tram station',
                'cost': 'Free to enter',
                'duration': '2-4 hours',
                'note': 'Over 4,000 shops, bring bargaining skills'
            },
            {
                'name': 'Kadıköy Moda Seaside',
                'type': 'outdoor',
                'description': 'Asian side coastal walk with sea breeze and cafes',
                'transport': 'Ferry to Kadıköy, walk to Moda',
                'cost': 'Free walk, cafe prices vary',
                'duration': '2-4 hours',
                'activities': 'Walking, cafes, ice cream shops'
            },
            {
                'name': 'Movie Theater (Cinemaximum/Cineplex)',
                'type': 'indoor',
                'description': 'Air-conditioned cinemas in major malls',
                'locations': ['Zorlu Center', 'Cevahir', 'Kanyon'],
                'cost': '80-150 TL per ticket',
                'duration': '2-3 hours'
            },
            {
                'name': 'Dolmabahçe Palace Garden',
                'type': 'outdoor',
                'description': 'Waterfront palace gardens with shade and Bosphorus breeze',
                'transport': 'Kabataş tram/ferry terminal',
                'cost': 'Garden: Free, Palace: 500 TL',
                'duration': '1-2 hours gardens, +1 hour palace',
                'note': 'Palace is air-conditioned'
            },
            {
                'name': 'Turkish Bath (Hamam)',
                'type': 'wellness',
                'description': 'Traditional spa experience, actually cooling after heat exposure',
                'recommendations': ['Çemberlitaş Hamamı', 'Kılıç Ali Paşa Hamamı'],
                'cost': '300-800 TL',
                'duration': '1.5-2 hours',
                'note': 'Cool marble and proper cooling room'
            },
            {
                'name': 'Bebek Bay Cafes',
                'type': 'outdoor',
                'description': 'Trendy Bosphorus-side cafes with sea views and breeze',
                'transport': 'Bus from Kabataş or Taksim',
                'cost': 'Cafe prices: 100-300 TL per person',
                'duration': '2-3 hours',
                'vibe': 'Upscale, relaxed'
            },
            {
                'name': 'Belgrade Forest',
                'type': 'nature',
                'description': 'Large forest area significantly cooler than city',
                'temp_difference': '-3 to -5°C cooler',
                'transport': 'Bus from various points',
                'cost': 'Free',
                'duration': '3-5 hours',
                'activities': 'Hiking, picnic, nature walks'
            }
        ]
    
    def _load_cold_weather_activities(self) -> List[Dict]:
        """Load activities for cold weather (<10°C)"""
        return [
            {
                'name': 'Traditional Turkish Bath (Hamam)',
                'type': 'wellness',
                'description': 'Warm up with centuries-old bathing tradition',
                'recommendations': [
                    'Çemberlitaş Hamamı (historic, 1584)',
                    'Kılıç Ali Paşa Hamamı (architectural beauty)',
                    'Süleymaniye Hamamı (authentic, less touristy)'
                ],
                'transport': 'Various locations in Sultanahmet, Beyoğlu',
                'cost': '300-800 TL (includes scrub and massage)',
                'duration': '1.5-2 hours',
                'tip': 'Book in advance, bring cash for tips'
            },
            {
                'name': 'Grand Bazaar Shopping',
                'type': 'indoor',
                'description': 'Warm covered market with 4,000+ shops',
                'transport': 'Beyazıt tram station',
                'cost': 'Free entry',
                'duration': '2-4 hours',
                'what_to_buy': 'Carpets, ceramics, jewelry, Turkish delight',
                'note': 'Closed Sundays'
            },
            {
                'name': 'Spice Bazaar (Egyptian Bazaar)',
                'type': 'indoor',
                'description': 'Aromatic covered market, warm and cozy',
                'transport': 'Eminönü tram/ferry station',
                'cost': 'Free entry',
                'duration': '1-2 hours',
                'purchases': 'Spices, tea, Turkish delight, dried fruits'
            },
            {
                'name': 'Museum Hopping',
                'type': 'indoor',
                'description': 'World-class museums with heating',
                'recommendations': [
                    'Topkapı Palace (Ottoman treasures)',
                    'Hagia Sophia (architectural marvel)',
                    'Istanbul Modern (contemporary art)',
                    'Pera Museum (Orientalist paintings)',
                    'Istanbul Archaeology Museums'
                ],
                'cost': '200-500 TL per museum',
                'duration': '2-3 hours per museum',
                'tip': 'Museum Pass saves money: 850 TL for 5 days'
            },
            {
                'name': 'Cozy Tea Houses & Cafes',
                'type': 'relaxation',
                'description': 'Warm up with Turkish tea or coffee',
                'recommendations': [
                    'Pierre Loti Café (historic, Golden Horn views)',
                    'Mandabatmaz (best Turkish coffee)',
                    'Fazıl Bey Turkish Coffee',
                    'Çorlulu Ali Paşa Medresesi (hidden gem)'
                ],
                'cost': '50-150 TL',
                'duration': '1-2 hours',
                'order': 'Turkish tea (çay), Turkish coffee, salep (winter drink)'
            },
            {
                'name': 'Indoor Food Markets',
                'type': 'food',
                'description': 'Heated markets with street food and local delicacies',
                'locations': [
                    'Kadıköy Market (Asian side)',
                    'Beşiktaş Market',
                    'Eminönü Food Stalls'
                ],
                'cost': '50-200 TL',
                'try': 'Midye dolma, balık ekmek, roasted chestnuts'
            },
            {
                'name': 'Traditional Turkish Restaurant Dinner',
                'type': 'food',
                'description': 'Cozy restaurants with warm Ottoman cuisine',
                'recommendations': [
                    'Hünkar (home-style Turkish)',
                    'Pandeli (above Spice Bazaar)',
                    'Asitane (Ottoman palace recipes)'
                ],
                'cost': '200-500 TL per person',
                'must_try': 'Lamb dishes, soups, künefe dessert'
            },
            {
                'name': 'Shopping Malls',
                'type': 'indoor',
                'description': 'Modern heated malls with entertainment',
                'recommendations': [
                    'Zorlu Center (upscale)',
                    'Cevahir (largest in Europe)',
                    'Istinye Park (luxury brands)',
                    'Kanyon (architectural landmark)'
                ],
                'activities': 'Shopping, cinema, dining, arcade',
                'duration': '3-5 hours'
            },
            {
                'name': 'Whirling Dervishes Ceremony',
                'type': 'cultural',
                'description': 'Indoor spiritual performance, heated venue',
                'venues': [
                    'Hodjapasha Cultural Center',
                    'Galata Mevlevi Museum',
                    'Sirkeci Train Station'
                ],
                'cost': '250-400 TL',
                'duration': '1 hour',
                'note': 'Sacred ceremony, dress respectfully'
            },
            {
                'name': 'Turkish Cooking Class',
                'type': 'experience',
                'description': 'Learn to cook in warm kitchen, eat what you make',
                'locations': ['Sultanahmet', 'Beyoğlu'],
                'cost': '400-800 TL',
                'duration': '3-4 hours',
                'learn': 'Meze, börek, baklava, köfte'
            },
            {
                'name': 'Miniaturk Park (if not too cold)',
                'type': 'outdoor',
                'description': 'Miniature models of Turkish landmarks',
                'transport': 'Golden Horn area',
                'cost': '150 TL',
                'duration': '2-3 hours',
                'note': 'Some indoor sections, dress warm'
            },
            {
                'name': 'Roasted Chestnut Walk',
                'type': 'outdoor',
                'description': 'Walk Istiklal Street with roasted chestnuts to warm up',
                'location': 'Istiklal Street, Taksim',
                'cost': 'Chestnuts: 50-100 TL per bag',
                'duration': '1-2 hours',
                'vibe': 'Winter tradition, very cozy'
            },
            {
                'name': 'Beyoğlu Cinema & Theater',
                'type': 'cultural',
                'description': 'Historic cinemas and theaters in heated buildings',
                'venues': ['Atlas Cinema', 'Pera Museum Cinema'],
                'cost': '80-200 TL',
                'note': 'Some international films, some Turkish with subtitles'
            },
            {
                'name': 'Bosphorus Cruise (Short Route)',
                'type': 'outdoor',
                'description': 'Short ferry rides with indoor heated cabins',
                'route': 'Eminönü → Kadıköy (20 min)',
                'cost': '15 TL',
                'tip': 'Sit indoors, get tea from vendor (çaycı)'
            },
            {
                'name': 'Indoor Swimming & Spa',
                'type': 'wellness',
                'description': 'Heated pools and saunas in luxury hotels',
                'locations': ['Çırağan Palace', 'Four Seasons', 'Swissôtel'],
                'cost': '500-1500 TL',
                'duration': '2-4 hours',
                'note': 'Day passes available at some hotels'
            }
        ]
    
    def _load_rainy_day_activities(self) -> List[Dict]:
        """Load activities for rainy weather"""
        return [
            {
                'name': 'Museum Route (Connected)',
                'type': 'indoor',
                'description': 'Multiple museums connected by short covered walks',
                'route': [
                    'Hagia Sophia',
                    'Basilica Cistern (underground)',
                    'Istanbul Archaeology Museums',
                    'Topkapı Palace (mostly covered)'
                ],
                'transport': 'All in Sultanahmet, walkable',
                'cost': '850 TL Museum Pass or individual tickets',
                'duration': 'Full day',
                'covered_connections': True
            },
            {
                'name': 'Grand Bazaar + Spice Bazaar Route',
                'type': 'indoor',
                'description': 'Covered markets connected by tram',
                'route': 'Grand Bazaar → Tram → Spice Bazaar → Ferry terminal covered',
                'transport': 'T1 tram line',
                'duration': '4-6 hours',
                'note': 'Mostly indoor, minimal rain exposure'
            },
            {
                'name': 'Istiklal Street Covered Walk',
                'type': 'mixed',
                'description': 'Pedestrian street with overhangs and arcade entrances',
                'route': 'Taksim → Istiklal → Galata Tower',
                'covered_areas': 'Many arcades, galleries, and covered passages',
                'duration': '2-3 hours',
                'activities': 'Shopping, cafes, bookstores'
            },
            {
                'name': 'Shopping Mall Day',
                'type': 'indoor',
                'description': 'Fully covered shopping and entertainment',
                'recommendations': [
                    'Zorlu Center (luxury + PSM theater)',
                    'Kanyon (architectural beauty)',
                    'Cevahir (largest, has ski slope!)',
                    'İstinyePark (upscale)'
                ],
                'duration': 'Full day possible',
                'activities': 'Shopping, cinema, dining, arcade'
            },
            {
                'name': 'Turkish Bath Marathon',
                'type': 'wellness',
                'description': 'Perfect rainy day activity, fully indoors',
                'experience': 'Traditional hamam + spa package',
                'cost': '500-1200 TL',
                'duration': '2-4 hours',
                'ultimate_relaxation': True
            },
            {
                'name': 'Underground City Exploration',
                'type': 'indoor',
                'description': 'Basilica Cistern and other underground sites',
                'sites': [
                    'Basilica Cistern',
                    'Theodosius Cistern',
                    'Şerefiye Cistern'
                ],
                'cost': '450-600 TL',
                'duration': '3-4 hours',
                'note': 'Cool underground, bring light jacket'
            },
            {
                'name': 'Cooking Class Experience',
                'type': 'indoor',
                'description': 'Learn Turkish cooking in covered kitchen',
                'cost': '400-800 TL',
                'duration': '3-4 hours',
                'outcome': 'Warm meal you cooked yourself'
            },
            {
                'name': 'Art Gallery Hopping',
                'type': 'indoor',
                'description': 'Contemporary art galleries in Karaköy/Beyoğlu',
                'area': 'Karaköy and Beyoğlu districts',
                'galleries': [
                    'Istanbul Modern',
                    'Pera Museum',
                    'SALT Galata',
                    'Arter'
                ],
                'cost': 'Mostly free or 100-200 TL',
                'duration': 'Half day'
            },
            {
                'name': 'Covered Passage (Pasaj) Exploration',
                'type': 'indoor',
                'description': 'Historic covered shopping passages',
                'passages': [
                    'Çiçek Pasajı (Flower Passage)',
                    'Atlas Pasajı',
                    'Aznavur Pasajı',
                    'Avrupa Pasajı'
                ],
                'location': 'Beyoğlu, off Istiklal Street',
                'cost': 'Free to explore',
                'activities': 'Cafes, restaurants, vintage shops'
            },
            {
                'name': 'Cinema Day',
                'type': 'indoor',
                'description': 'Modern cinemas or historic movie theaters',
                'options': [
                    'Cinemaximum (modern, multiple locations)',
                    'Atlas Cinema (historic)',
                    'Beyoğlu Cinema'
                ],
                'cost': '80-150 TL per movie',
                'duration': '2-3 hours per movie'
            }
        ]
    
    def _load_mild_weather_activities(self) -> List[Dict]:
        """Load activities for mild/perfect weather (15-25°C)"""
        return [
            {
                'name': 'Bosphorus Walking Tour',
                'type': 'outdoor',
                'description': 'Perfect weather for coastal walks',
                'routes': [
                    'Ortaköy → Bebek → Arnavutköy (2-3 hours)',
                    'Kadıköy → Moda → Caddebostan (2 hours)',
                    'Karaköy → Galata → Cihangir (1.5 hours)'
                ],
                'cost': 'Free',
                'best_for': 'Scenic views, photos, cafes along the way'
            },
            {
                'name': 'Princes\' Islands Bike Tour',
                'type': 'outdoor',
                'description': 'Perfect temperature for cycling',
                'islands': 'Büyükada or Heybeliada',
                'transport': 'Ferry from Kabataş or Kadıköy',
                'cost': 'Ferry 15 TL + Bike rental 50 TL',
                'duration': 'Full day'
            },
            {
                'name': 'Historic Peninsula Walking Tour',
                'type': 'outdoor',
                'description': 'Comfortable weather for exploration',
                'sites': 'Sultanahmet → Blue Mosque → Hagia Sophia → Grand Bazaar',
                'duration': '4-6 hours',
                'cost': 'Museum entries',
                'note': 'Best weather for outdoor sightseeing'
            }
        ]
    
    def get_weather_appropriate_activities(self, temperature: float, 
                                          weather_condition: str = 'clear',
                                          preferences: List[str] = None) -> Dict[str, any]:
        """
        Get weather-appropriate activity recommendations
        
        Args:
            temperature: Current temperature in Celsius
            weather_condition: 'clear', 'rain', 'snow', 'cloudy', 'drizzle'
            preferences: User preferences list (e.g., ['indoor', 'cultural', 'food'])
            
        Returns:
            Dictionary with recommendations and context
        """
        activities = []
        weather_note = ""
        
        # Rainy weather takes priority
        if weather_condition in ['rain', 'drizzle', 'thunderstorm', 'snow']:
            activities = self.rainy_day_activities
            weather_note = "☔ Rainy day activities - Stay dry and enjoy Istanbul indoors!"
        
        # Temperature-based recommendations
        elif temperature > 28:
            activities = self.hot_weather_activities
            weather_note = f"🌡️ Hot weather ({temperature}°C) - Cool down with these activities!"
        
        elif temperature < 10:
            activities = self.cold_weather_activities
            weather_note = f"❄️ Cold weather ({temperature}°C) - Warm up with these cozy activities!"
        
        else:
            activities = self.mild_weather_activities
            weather_note = f"🌤️ Perfect weather ({temperature}°C) - Great day to explore Istanbul!"
        
        # Filter by preferences if provided
        if preferences:
            filtered_activities = []
            for activity in activities:
                activity_type = activity.get('type', '').lower()
                if any(pref.lower() in activity_type for pref in preferences):
                    filtered_activities.append(activity)
            
            if filtered_activities:
                activities = filtered_activities
        
        return {
            'weather_note': weather_note,
            'temperature': temperature,
            'condition': weather_condition,
            'activities': activities[:10],  # Return top 10
            'total_available': len(activities)
        }
    
    def format_weather_activities_response(self, temperature: float, 
                                          weather_condition: str = 'clear',
                                          limit: int = 5) -> str:
        """Format weather activities into a readable response"""
        
        recommendations = self.get_weather_appropriate_activities(temperature, weather_condition)
        
        response = f"{recommendations['weather_note']}\n\n"
        
        activities = recommendations['activities'][:limit]
        
        for i, activity in enumerate(activities, 1):
            response += f"**{i}. {activity['name']}** ({activity['type'].title()})\n"
            response += f"   {activity['description']}\n"
            
            if 'temp_difference' in activity:
                response += f"   🌡️ Temperature: {activity['temp_difference']}\n"
            
            if 'transport' in activity:
                response += f"   🚇 How to get there: {activity['transport']}\n"
            
            if 'cost' in activity:
                response += f"   💰 Cost: {activity['cost']}\n"
            
            if 'duration' in activity:
                response += f"   ⏱️ Duration: {activity['duration']}\n"
            
            if 'note' in activity or 'tip' in activity:
                note_text = activity.get('note') or activity.get('tip')
                response += f"   💡 Tip: {note_text}\n"
            
            response += "\n"
        
        if len(recommendations['activities']) > limit:
            response += f"💡 **{len(recommendations['activities']) - limit} more activities available!** Ask for more weather-appropriate suggestions.\n"
        
        return response


# Singleton instance
_weather_recommendations_service = None

def get_weather_recommendations_service() -> WeatherRecommendationsService:
    """Get or create weather recommendations service instance"""
    global _weather_recommendations_service
    if _weather_recommendations_service is None:
        _weather_recommendations_service = WeatherRecommendationsService()
    return _weather_recommendations_service

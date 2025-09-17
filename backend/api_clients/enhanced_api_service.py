"""
Enhanced API Integration Service
Combines all real API clients for the AI Istanbul project.
"""

from typing import Dict, List, Optional
import logging
from .enhanced_google_places import EnhancedGooglePlacesClient
from .google_weather import GoogleWeatherClient
from .istanbul_transport import IstanbulTransportClient

logger = logging.getLogger(__name__)

class EnhancedAPIService:
    """Unified service for all enhanced APIs with real data integration."""
    
    def __init__(self):
        self.places_client = EnhancedGooglePlacesClient()
        self.weather_client = GoogleWeatherClient()
        self.transport_client = IstanbulTransportClient()
        
        logger.info("Enhanced API Service initialized with all clients")
    
    def search_restaurants_enhanced(self, location: str, keyword: Optional[str] = None, 
                                  min_rating: Optional[float] = None) -> Dict:
        """Search for restaurants with enhanced real data."""
        try:
            results = self.places_client.search_restaurants(
                location=location,
                keyword=keyword,
                min_rating=min_rating
            )
            
            # Add weather context to restaurant recommendations
            weather = self.weather_client.get_current_weather()
            results['weather_context'] = {
                'current_temp': weather['main']['temp'],
                'condition': weather['weather'][0]['description'],
                'recommendations': weather.get('activity_recommendations', [])
            }
            
            return results
        except Exception as e:
            logger.error(f"Error in enhanced restaurant search: {e}")
            return {"error": str(e), "results": []}
    
    def get_complete_weather_info(self, city: str = "Istanbul") -> Dict:
        """Get comprehensive weather information with activity suggestions."""
        try:
            current = self.weather_client.get_current_weather(city)
            forecast = self.weather_client.get_forecast(city, days=3)
            
            return {
                'current': current,
                'forecast': forecast,
                'travel_advice': self._generate_weather_travel_advice(current)
            }
        except Exception as e:
            logger.error(f"Error getting weather info: {e}")
            return {"error": str(e)}
    
    def get_transport_recommendations(self, from_location: str, to_location: str) -> Dict:
        """Get comprehensive transport recommendations."""
        try:
            routes = self.transport_client.get_route_info(from_location, to_location)
            card_info = self.transport_client.get_istanbul_card_info()
            
            # Add current weather to transport recommendations
            weather = self.weather_client.get_current_weather()
            weather_impact = self._assess_weather_transport_impact(weather)
            
            return {
                'routes': routes,
                'payment_info': card_info,
                'weather_impact': weather_impact
            }
        except Exception as e:
            logger.error(f"Error getting transport recommendations: {e}")
            return {"error": str(e)}
    
    def get_contextual_recommendations(self, user_query: str, location: Optional[str] = None) -> Dict:
        """Get contextual recommendations combining all APIs."""
        try:
            # Get weather context
            weather = self.weather_client.get_current_weather()
            
            # Determine if query is about restaurants, transport, or general
            query_lower = user_query.lower()
            
            if any(word in query_lower for word in ['restaurant', 'food', 'eat', 'dining']):
                # Restaurant query
                restaurants = self.places_client.search_restaurants(
                    location=location or "Istanbul",
                    keyword=self._extract_cuisine_keyword(user_query)
                )
                
                return {
                    'type': 'restaurant_recommendations',
                    'results': restaurants,
                    'weather_context': weather,
                    'suggestions': self._weather_based_dining_suggestions(weather)
                }
            
            elif any(word in query_lower for word in ['transport', 'bus', 'metro', 'get to', 'how to']):
                # Transport query
                locations = self._extract_locations(user_query)
                if len(locations) >= 2:
                    transport = self.transport_client.get_route_info(locations[0], locations[1])
                else:
                    transport = self.transport_client.get_metro_status()
                
                return {
                    'type': 'transport_recommendations',
                    'results': transport,
                    'weather_context': weather
                }
            
            else:
                # General query - provide comprehensive info
                return {
                    'type': 'general_recommendations',
                    'weather': weather,
                    'transport_tips': self.transport_client.get_istanbul_card_info(),
                    'activity_suggestions': weather.get('activity_recommendations', [])
                }
        
        except Exception as e:
            logger.error(f"Error getting contextual recommendations: {e}")
            return {"error": str(e)}
    
    def _generate_weather_travel_advice(self, weather: Dict) -> List[str]:
        """Generate travel advice based on weather."""
        advice = []
        temp = weather['main']['temp']
        condition = weather['weather'][0]['main'].lower()
        
        if temp > 25:
            advice.extend([
                "Stay hydrated - carry water bottle",
                "Seek shade during midday hours",
                "Light, breathable clothing recommended"
            ])
        elif temp < 10:
            advice.extend([
                "Dress warmly in layers",
                "Indoor activities recommended",
                "Hot beverages available at many cafes"
            ])
        
        if condition == 'rain':
            advice.extend([
                "Carry umbrella or waterproof jacket",
                "Consider covered attractions",
                "Public transport may be crowded"
            ])
        
        return advice
    
    def _assess_weather_transport_impact(self, weather: Dict) -> Dict:
        """Assess how weather impacts transportation."""
        condition = weather['weather'][0]['main'].lower()
        temp = weather['main']['temp']
        
        impact = {
            'overall': 'normal',
            'recommendations': [],
            'delays_expected': False
        }
        
        if condition == 'rain':
            impact['overall'] = 'affected'
            impact['recommendations'].extend([
                "Allow extra travel time",
                "Metro less affected than buses",
                "Ferry services may be delayed"
            ])
            impact['delays_expected'] = True
        
        elif temp > 35 or temp < 0:
            impact['overall'] = 'challenging'
            impact['recommendations'].extend([
                "Use air-conditioned metro when possible",
                "Plan for platform waiting times"
            ])
        
        return impact
    
    def _extract_cuisine_keyword(self, query: str) -> Optional[str]:
        """Extract cuisine type from user query."""
        cuisines = ['turkish', 'ottoman', 'kebab', 'seafood', 'international', 'local']
        query_lower = query.lower()
        
        for cuisine in cuisines:
            if cuisine in query_lower:
                return cuisine
        return None
    
    def _extract_locations(self, query: str) -> List[str]:
        """Extract location names from query."""
        locations = ['taksim', 'sultanahmet', 'galata', 'kadikoy', 'besiktas', 'eminonu', 'airport']
        found_locations = []
        query_lower = query.lower()
        
        for location in locations:
            if location in query_lower:
                found_locations.append(location.title())
        
        return found_locations
    
    def _weather_based_dining_suggestions(self, weather: Dict) -> List[str]:
        """Generate dining suggestions based on weather."""
        temp = weather['main']['temp']
        condition = weather['weather'][0]['main'].lower()
        
        suggestions = []
        
        if temp > 25 and condition != 'rain':
            suggestions.extend([
                "Perfect weather for rooftop dining with Bosphorus views",
                "Outdoor terraces in Galata area are ideal",
                "Consider light, refreshing meals"
            ])
        elif condition == 'rain':
            suggestions.extend([
                "Cozy indoor restaurants with traditional atmosphere",
                "Perfect time for hot Turkish soup and tea",
                "Covered bazaar restaurants are great options"
            ])
        elif temp < 15:
            suggestions.extend([
                "Warm, hearty Turkish stews and kebabs recommended",
                "Indoor dining with traditional heating",
                "Hot beverages to complement your meal"
            ])
        
        return suggestions
    
    def get_all_api_status(self) -> Dict:
        """Get status of all API clients."""
        return {
            'google_places': self.places_client.get_api_status(),
            'weather': self.weather_client.get_api_status(),
            'transport': self.transport_client.get_api_status(),
            'integration_status': 'operational'
        }

#!/usr/bin/env python3
"""
Non-LLM Istanbul Assistant
===========================

Ultra-specialized rule-based system that provides natural, conversational responses
without relying on GPT or any language models. Uses pattern matching, templates,
and comprehensive local knowledge database.
"""

import re
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

class IstanbulKnowledgeBase:
    """Comprehensive Istanbul knowledge database"""
    
    def __init__(self):
        self.restaurants = self._load_restaurant_data()
        self.districts = self._load_district_data()
        self.attractions = self._load_attraction_data()
        self.transportation = self._load_transportation_data()
    
    def _load_restaurant_data(self) -> Dict:
        """Load comprehensive restaurant database"""
        return {
            'sultanahmet': [
                {
                    'name': 'Pandeli Restaurant',
                    'cuisine': 'Ottoman',
                    'budget': 'premium',
                    'description': 'Historic Ottoman cuisine restaurant with traditional recipes and elegant atmosphere since 1901',
                    'location': 'Sultanahmet, near Spice Bazaar',
                    'specialties': ['lamb stew', 'Ottoman palace dishes', 'traditional desserts']
                },
                {
                    'name': 'Sultanahmet Fish House',
                    'cuisine': 'Seafood',
                    'budget': 'premium',
                    'description': 'Traditional fish restaurant near Blue Mosque with fresh Bosphorus seafood',
                    'location': 'Sultanahmet Square area',
                    'specialties': ['grilled sea bass', 'meze selection', 'fresh catch of the day']
                },
                {
                    'name': 'Aya Sofya Kebab House',
                    'cuisine': 'Turkish',
                    'budget': 'budget',
                    'description': 'Family-run kebab restaurant serving authentic grilled meats near Hagia Sophia',
                    'location': 'Walking distance from Hagia Sophia',
                    'specialties': ['Adana kebab', 'lamb shish', 'Turkish breakfast']
                },
                {
                    'name': 'Ottoman Kitchen',
                    'cuisine': 'Ottoman',
                    'budget': 'luxury',
                    'description': 'Upscale Ottoman cuisine restaurant with historical recipes and royal ambiance',
                    'location': 'Historic Sultanahmet district',
                    'specialties': ['palace cuisine', 'imperial recipes', 'fine dining experience']
                },
                {
                    'name': 'Spice Bazaar Cafe',
                    'cuisine': 'Turkish',
                    'budget': 'budget',
                    'description': 'Small cafe inside Spice Bazaar serving Turkish coffee and traditional sweets',
                    'location': 'Inside Egyptian Bazaar',
                    'specialties': ['Turkish coffee', 'baklava', 'Turkish delight']
                }
            ],
            'beyoğlu': [
                {
                    'name': 'Galata House Restaurant',
                    'cuisine': 'Turkish',
                    'budget': 'premium',
                    'description': 'Traditional Turkish restaurant in historic Galata building with live music',
                    'location': 'Galata district, Beyoğlu',
                    'specialties': ['meze platters', 'traditional music', 'historic atmosphere']
                },
                {
                    'name': 'Cafe Central',
                    'cuisine': 'International',
                    'budget': 'mid-range',
                    'description': 'Historic European-style cafe serving light meals and excellent coffee since 1950s',
                    'location': 'İstiklal Street area',
                    'specialties': ['European pastries', 'specialty coffee', 'vintage atmosphere']
                },
                {
                    'name': 'Helvetia Lokantası',
                    'cuisine': 'Turkish',
                    'budget': 'mid-range',
                    'description': 'Traditional meyhane serving authentic Turkish meze and rakı in historic setting',
                    'location': 'Near Galata Tower',
                    'specialties': ['meze selection', 'rakı service', 'traditional atmosphere']
                },
                {
                    'name': 'Teppanyaki Ginza',
                    'cuisine': 'Japanese',
                    'budget': 'luxury',
                    'description': 'Premium Japanese teppanyaki restaurant with live cooking and fresh ingredients',
                    'location': 'Upscale Beyoğlu area',
                    'specialties': ['teppanyaki', 'fresh sushi', 'premium ingredients']
                },
                {
                    'name': 'Meyhane Asmalımescit',
                    'cuisine': 'Turkish',
                    'budget': 'budget',
                    'description': 'Lively traditional meyhane with live Turkish folk music and authentic meze',
                    'location': 'Asmalımescit neighborhood',
                    'specialties': ['live music', 'traditional meze', 'authentic atmosphere']
                }
            ],
            'kadıköy': [
                {
                    'name': 'Çiya Sofrası',
                    'cuisine': 'Turkish',
                    'budget': 'mid-range',
                    'description': 'Famous restaurant preserving Anatolian cuisine traditions with regional specialties',
                    'location': 'Kadıköy market area',
                    'specialties': ['Anatolian dishes', 'regional recipes', 'authentic experience']
                },
                {
                    'name': 'Kadıköy Fish Market Restaurant',
                    'cuisine': 'Seafood',
                    'budget': 'mid-range',
                    'description': 'Fresh seafood restaurant in heart of Kadıköy fish market with daily catches',
                    'location': 'Kadıköy fish market',
                    'specialties': ['daily fresh catch', 'seafood meze', 'market experience']
                },
                {
                    'name': 'Moda Pier Restaurant',
                    'cuisine': 'Seafood',
                    'budget': 'premium',
                    'description': 'Seaside restaurant on Moda Pier with panoramic sea views and fresh fish',
                    'location': 'Moda waterfront',
                    'specialties': ['sea views', 'fresh fish', 'waterfront dining']
                },
                {
                    'name': 'Sade Kahve',
                    'cuisine': 'Turkish',
                    'budget': 'mid-range',
                    'description': 'Specialty coffee roastery and cafe with artisanal Turkish coffee and light meals',
                    'location': 'Kadıköy center',
                    'specialties': ['specialty coffee', 'Turkish coffee', 'light meals']
                }
            ]
        }
    
    def _load_district_data(self) -> Dict:
        """Load district information database"""
        return {
            'sultanahmet': {
                'character': 'the heart of historic Istanbul',
                'vibe': 'ancient stones meeting modern life',
                'best_time': 'early morning before crowds arrive',
                'highlights': ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace'],
                'hidden_gems': ['Byzantine Cistern', 'traditional Turkish baths'],
                'transport': 'Sultanahmet tram station or short walk from Eminönü ferry'
            },
            'beyoğlu': {
                'character': "Istanbul's vibrant cultural hub",
                'vibe': 'art galleries, rooftop bars, and endless energy',
                'best_time': 'late afternoon into the night',
                'highlights': ['Galata Tower', 'İstiklal Street', 'Pera Museum'],
                'hidden_gems': ['historic Pera Palace Hotel', 'rooftop terraces'],
                'transport': 'metro to Şişhane or historic tünel funicular'
            },
            'kadıköy': {
                'character': 'the creative soul of the Asian side',
                'vibe': 'bohemian cafés, street art, and local authenticity',
                'best_time': 'weekend mornings for the fish market',
                'highlights': ['Kadıköy Market', 'Moda waterfront', 'Çiya restaurants'],
                'hidden_gems': ['street art tours', 'local design shops'],
                'transport': 'ferry from Eminönü - the scenic route locals love'
            },
            'beşiktaş': {
                'character': 'where Ottoman elegance meets Bosphorus beauty',
                'vibe': 'waterfront palaces and passionate football culture',
                'best_time': 'sunset for the best Bosphorus views',
                'highlights': ['Dolmabahçe Palace', 'Beşiktaş Square', 'Maritime Museum'],
                'hidden_gems': ['traditional fish restaurants', 'local football culture'],
                'transport': 'ferry or metro - both offer great approaches'
            }
        }
    
    def _load_attraction_data(self) -> Dict:
        """Load attractions database"""
        return {
            'general': [
                {
                    'name': 'Hagia Sophia',
                    'description': 'Magnificent 6th-century architectural wonder, now an active mosque with stunning Byzantine mosaics',
                    'location': 'Sultanahmet',
                    'visit_duration': 'about an hour',
                    'best_time': 'early morning or late afternoon'
                },
                {
                    'name': 'Blue Mosque',
                    'description': 'Stunning 17th-century mosque famous for its six minarets and beautiful blue Iznik tiles',
                    'location': 'Sultanahmet Square',
                    'visit_duration': 'about 30 minutes',
                    'best_time': 'between prayer times'
                },
                {
                    'name': 'Grand Bazaar',
                    'description': 'Historic covered market with 4,000 shops offering carpets, jewelry, spices, and traditional crafts',
                    'location': 'Beyazıt',
                    'visit_duration': 'about 2 hours',
                    'best_time': 'weekday mornings'
                },
                {
                    'name': 'Topkapi Palace',
                    'description': 'Former Ottoman imperial palace with treasury, harem quarters, and stunning Bosphorus views',
                    'location': 'Sultanahmet',
                    'visit_duration': 'about 3 hours',
                    'best_time': 'early morning'
                },
                {
                    'name': 'Galata Tower',
                    'description': 'Medieval Genoese tower offering 360-degree panoramic views of Istanbul and the Bosphorus',
                    'location': 'Galata, Beyoğlu',
                    'visit_duration': 'about 45 minutes',
                    'best_time': 'sunset'
                }
            ]
        }
    
    def _load_transportation_data(self) -> Dict:
        """Load transportation information"""
        return {
            'metro_lines': {
                'm2': {
                    'name': 'M2 Green Line',
                    'route': 'Vezneciler ↔ Hacıosman',
                    'key_stations': ['Taksim', 'Şişhane', 'Vezneciler', 'Levent']
                },
                't1': {
                    'name': 'T1 Red Tram',
                    'route': 'Kabataş ↔ Bağcılar',
                    'key_stations': ['Sultanahmet', 'Eminönü', 'Karaköy', 'Kabataş']
                }
            },
            'common_routes': {
                'taksim_to_sultanahmet': {
                    'best_route': 'M2 Metro to Vezneciler + 5-minute walk',
                    'duration': '15 minutes total',
                    'alternative': 'F1 Funicular + T1 Tram (scenic route)'
                },
                'airport_to_city': {
                    'best_route': 'M11 Metro from Gayrettepe',
                    'duration': '37 minutes',
                    'alternative': 'HAVAIST bus or taxi'
                }
            }
        }

class NonLLMIstanbulAssistant:
    """Rule-based Istanbul assistant without LLM dependency"""
    
    def __init__(self):
        self.knowledge_base = IstanbulKnowledgeBase()
        self.response_patterns = self._build_response_patterns()
    
    def _build_response_patterns(self) -> Dict:
        """Build response pattern templates"""
        return {
            'restaurant_opening': {
                'general': "Here are {count} restaurants in {location}:",
                'budget': "Here are {count} {budget_type} restaurants in {location}:",
                'cuisine': "Here are {count} {cuisine} restaurants in {location}:",
                'coffee': "Here are {count} great coffee spots in {location}:"
            },
            'district_opening': {
                'general': "**{district}** is {character} — think {vibe}.",
                'detailed': "**{district}** is {character} — think {vibe}. It's best to visit around {best_time}."
            },
            'attraction_opening': {
                'single': "You shouldn't miss **{name}** — {description}.",
                'multiple': "Here are the top attractions in {location}:"
            }
        }
    
    def generate_response(self, user_input: str, context: Dict = None) -> str:
        """Generate response using rule-based system"""
        # Normalize input
        normalized_input = user_input.lower().strip()
        
        # Pattern matching for query type
        if self._is_restaurant_query(normalized_input):
            return self._handle_restaurant_query(normalized_input, user_input)
        elif self._is_district_query(normalized_input):
            return self._handle_district_query(normalized_input, user_input)
        elif self._is_attraction_query(normalized_input):
            return self._handle_attraction_query(normalized_input, user_input)
        elif self._is_transportation_query(normalized_input):
            return self._handle_transportation_query(normalized_input, user_input)
        else:
            return self._handle_general_query(user_input)
    
    def _is_restaurant_query(self, query: str) -> bool:
        """Check if query is about restaurants"""
        restaurant_keywords = [
            'restaurant', 'food', 'eat', 'dining', 'meal', 'breakfast', 'lunch', 'dinner',
            'kebab', 'turkish cuisine', 'coffee', 'cafe', 'where to eat', 'best food'
        ]
        return any(keyword in query for keyword in restaurant_keywords)
    
    def _is_district_query(self, query: str) -> bool:
        """Check if query is about districts"""
        district_patterns = [
            'tell me about', 'what is', 'describe', 'neighborhood', 'district', 'area'
        ]
        return any(pattern in query for pattern in district_patterns)
    
    def _is_attraction_query(self, query: str) -> bool:
        """Check if query is about attractions"""
        attraction_keywords = [
            'attractions', 'things to see', 'places to visit', 'sightseeing', 'tourist spots'
        ]
        return any(keyword in query for keyword in attraction_keywords)
    
    def _is_transportation_query(self, query: str) -> bool:
        """Check if query is about transportation"""
        transport_keywords = [
            'how to get', 'transport', 'metro', 'tram', 'bus', 'taxi', 'getting to'
        ]
        return any(keyword in query for keyword in transport_keywords)
    
    def _handle_restaurant_query(self, normalized_query: str, original_query: str) -> str:
        """Handle restaurant-related queries with enhanced local guide templates"""
        # Extract location and query context
        location = self._extract_location(normalized_query)
        if not location:
            location = 'Istanbul'
        
        # Detect query type for context-aware responses
        query_context = self._detect_restaurant_context(normalized_query)
        
        # Get restaurants for location
        restaurants = self.knowledge_base.restaurants.get(location.lower(), [])
        
        if not restaurants:
            return self._generate_no_restaurants_fallback(location, query_context)
        
        # Apply context-specific filtering
        filtered_restaurants = self._filter_restaurants_by_context(restaurants, query_context)
        
        # If no restaurants match the filter, return fallback
        if not filtered_restaurants:
            return self._generate_no_restaurants_fallback(location, query_context)
        
        # Generate context-aware opening
        count = min(len(filtered_restaurants), 5)
        opening = self._generate_restaurant_opening(location, count, query_context)
        
        # Format each restaurant with local guide personality
        restaurant_descriptions = []
        for i, restaurant in enumerate(filtered_restaurants[:5]):
            description = self._format_restaurant_description(restaurant, i == 0)
            restaurant_descriptions.append(description)
        
        # Join with natural connectors
        restaurants_text = self._join_restaurant_descriptions(restaurant_descriptions)
        
        # Add contextual closing
        closing = self._generate_restaurant_closing(location, query_context)
        
        return f"{opening}\n\n{restaurants_text}\n\n{closing}"
    
    def _handle_district_query(self, normalized_query: str, original_query: str) -> str:
        """Handle district-related queries"""
        # Extract district name
        district = self._extract_location(normalized_query)
        
        if district and district.lower() in self.knowledge_base.districts:
            district_info = self.knowledge_base.districts[district.lower()]
            
            response = f"**{district.title()}** is {district_info['character']} — think {district_info['vibe']}. "
            response += f"It's best to visit around {district_info['best_time']}. "
            response += f"You'll find {', '.join(district_info['highlights'][:2])}, plus hidden gems like {district_info['hidden_gems'][0]}. "
            response += f"It's easy to reach via {district_info['transport']}. Want me to build you a short walking route?"
            
            return response
        else:
            return "Istanbul has many fascinating districts, each with its own character. Are you interested in historic Sultanahmet, vibrant Beyoğlu, creative Kadıköy, or elegant Beşiktaş?"
    
    def _handle_attraction_query(self, normalized_query: str, original_query: str) -> str:
        """Handle attraction-related queries"""
        location = self._extract_location(normalized_query)
        attractions = self.knowledge_base.attractions['general']
        
        if location:
            # Filter attractions by location if specified
            location_attractions = [a for a in attractions if location.lower() in a['location'].lower()]
            if location_attractions:
                attractions = location_attractions
        
        main_attraction = attractions[0]
        response = f"You shouldn't miss **{main_attraction['name']}** — {main_attraction['description']}. "
        response += f"Perfect for morning exploration, and usually takes {main_attraction['visit_duration']} to explore. "
        
        if len(attractions) > 1:
            response += f"If you have time, nearby you can also stop by **{attractions[1]['name']}**."
        
        return response
    
    def _handle_transportation_query(self, normalized_query: str, original_query: str) -> str:
        """Handle transportation-related queries"""
        # Check for airport queries first
        if 'airport' in normalized_query:
            if 'istanbul airport' in normalized_query or 'new airport' in normalized_query:
                return "The easiest route from **Istanbul Airport** to the city is by **M11 Metro from Gayrettepe**. It usually takes around 37 minutes. If you'd rather avoid transfers, you can also try HAVAIST bus or taxi."
            elif 'sabiha' in normalized_query:
                return "The easiest route from **Sabiha Gökçen Airport** to the city is by **M4 Metro from Kadıköy**. It usually takes around 35 minutes. Alternatively, you can take the HAVABUS or taxi."
        
        # Extract from/to locations
        from_match = re.search(r'from\s+(\w+)', normalized_query)
        to_match = re.search(r'to\s+(\w+)', normalized_query)
        
        if from_match and to_match:
            from_loc = from_match.group(1)
            to_loc = to_match.group(1)
            
            # Check for common routes
            route_key = f"{from_loc}_to_{to_loc}"
            if route_key in self.knowledge_base.transportation['common_routes']:
                route_info = self.knowledge_base.transportation['common_routes'][route_key]
                response = f"The easiest route from **{from_loc.title()}** to **{to_loc.title()}** is by **{route_info['best_route']}**. "
                response += f"It usually takes around {route_info['duration']}. "
                if 'alternative' in route_info:
                    response += f"If you'd rather take the scenic route, you can also try {route_info['alternative']}."
                return response
            else:
                # Generic route response
                return f"The easiest route from **{from_loc.title()}** to **{to_loc.title()}** is typically by metro or tram. The M2 Green Line and T1 Red Tram connect most major areas. Want me to check the specific route for you?"
        
        # Generic transportation response
        return "The easiest way to get around Istanbul is by metro and tram using an Istanbulkart. The M2 Green Line connects Taksim to the historic peninsula, while the T1 Red Tram runs through all major tourist areas. Want specific route directions?"
    
    def _handle_general_query(self, query: str) -> str:
        """Handle general queries"""
        general_responses = [
            "I'm here to help you explore Istanbul! Are you looking for restaurants, attractions, or transportation advice?",
            "Istanbul has so much to offer! Tell me what you'd like to explore — food, sightseeing, or getting around the city?",
            "Let me help you discover the best of Istanbul. What specific area or experience interests you?",
        ]
        
        import random
        return random.choice(general_responses)
    
    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location from query"""
        locations = ['sultanahmet', 'beyoğlu', 'beyoglu', 'kadıköy', 'kadikoy', 'beşiktaş', 'besiktas', 'taksim', 'galata']
        
        for location in locations:
            if location in query:
                # Return normalized location name
                location_map = {
                    'beyoglu': 'beyoğlu',
                    'kadikoy': 'kadıköy', 
                    'besiktas': 'beşiktaş'
                }
                return location_map.get(location, location)
        
        return None
    
    def _detect_restaurant_context(self, query: str) -> str:
        """Detect specific restaurant query context"""
        if any(word in query for word in ['expensive', 'luxury', 'upscale', 'fine dining']):
            return 'luxury'
        elif any(word in query for word in ['cheap', 'budget', 'affordable', 'inexpensive']):
            return 'budget'
        elif any(word in query for word in ['coffee', 'cafe', 'espresso', 'cappuccino']):
            return 'coffee'
        elif any(word in query for word in ['vegan', 'vegetarian', 'plant-based']):
            return 'vegan'
        elif any(word in query for word in ['seafood', 'fish', 'sushi', 'ocean']):
            return 'seafood'
        elif any(word in query for word in ['turkish', 'ottoman', 'traditional', 'local']):
            return 'turkish'
        else:
            return 'general'
    
    def _filter_restaurants_by_context(self, restaurants: List[Dict], context: str) -> List[Dict]:
        """Filter restaurants based on query context"""
        if context == 'luxury':
            return [r for r in restaurants if r['budget'] in ['luxury', 'premium']]
        elif context == 'budget':
            return [r for r in restaurants if r['budget'] in ['budget', 'mid-range']]
        elif context == 'coffee':
            return [r for r in restaurants if 'coffee' in r['description'].lower() or 'cafe' in r['name'].lower()]
        elif context == 'vegan':
            return [r for r in restaurants if any(word in r['description'].lower() for word in ['vegan', 'vegetarian', 'plant'])]
        elif context == 'seafood':
            return [r for r in restaurants if r['cuisine'].lower() == 'seafood' or 'fish' in r['description'].lower()]
        elif context == 'turkish':
            return [r for r in restaurants if r['cuisine'].lower() in ['turkish', 'ottoman']]
        else:
            return restaurants
    
    def _generate_restaurant_opening(self, location: str, count: int, context: str) -> str:
        """Generate context-aware restaurant opening"""
        location_title = location.title()
        
        if context == 'luxury':
            return f"Here are {count} premium dining spots in {location_title}:"
        elif context == 'budget':
            return f"Here are {count} budget-friendly restaurants in {location_title}:"
        elif context == 'coffee':
            return f"Here are {count} great coffee spots in {location_title}:"
        elif context == 'vegan':
            return f"Here are {count} restaurants with excellent vegan options in {location_title}:"
        elif context == 'seafood':
            return f"Here are {count} fantastic seafood restaurants in {location_title}:"
        elif context == 'turkish':
            return f"Here are {count} authentic Turkish restaurants in {location_title}:"
        else:
            return f"Here are {count} restaurants in {location_title}:"
    
    def _format_restaurant_description(self, restaurant: Dict, is_first: bool) -> str:
        """Format single restaurant with local guide personality"""
        name = restaurant['name']
        description = restaurant['description']
        budget = restaurant['budget']
        cuisine = restaurant['cuisine']
        
        # Budget context mapping
        budget_context = {
            'budget': 'wallet-friendly',
            'mid-range': 'reasonably priced',
            'premium': 'upscale',
            'luxury': 'high-end'
        }.get(budget, 'great value')
        
        # Use different bullet styles for variety
        bullet = "• **" if is_first else "• **"
        
        return f"{bullet}{name}** — {description} This {budget_context} {cuisine.lower()} spot is perfect for authentic local dining."
    
    def _join_restaurant_descriptions(self, descriptions: List[str]) -> str:
        """Join restaurant descriptions with natural flow"""
        if len(descriptions) == 1:
            return descriptions[0]
        
        # Add natural variety in transitions
        connectors = [
            "\n\n", "\n\nAlso worth trying: ", "\n\nDon't miss: ", 
            "\n\nLocals absolutely love: ", "\n\nAnother gem: "
        ]
        
        result = descriptions[0]
        for i, description in enumerate(descriptions[1:], 1):
            if i < len(connectors):
                connector = connectors[i] if i > 0 else "\n\n"
                # Remove bullet point when using special connectors
                clean_desc = description.replace("• **", "**") if "worth trying:" in connector or "miss:" in connector or "love:" in connector or "gem:" in connector else description
                result += connector + clean_desc
            else:
                result += "\n\n" + description
        
        return result
    
    def _generate_restaurant_closing(self, location: str, context: str) -> str:
        """Generate contextual closing for restaurant recommendations"""
        closings = {
            'luxury': "Want reservations info or directions to any of these upscale spots?",
            'budget': "Need directions or opening hours for any of these affordable gems?",
            'coffee': "Looking for the perfect coffee spot timing or wifi info?",
            'vegan': "Want to know about other dietary accommodations at these places?",
            'seafood': "Interested in the freshest catch timing or seaside views?",
            'turkish': "Want to know about traditional dining customs at these spots?",
            'general': "Want specific directions or opening hours for any of these restaurants?"
        }
        
        return closings.get(context, closings['general'])
    
    def _generate_no_restaurants_fallback(self, location: str, context: str) -> str:
        """Enhanced fallback when no restaurants found"""
        alternatives = {
            'sultanahmet': 'Beyoğlu has incredible dining diversity',
            'beyoğlu': 'Kadıköy offers amazing authentic experiences', 
            'kadıköy': 'Sultanahmet has traditional Ottoman cuisine',
            'beşiktaş': 'Beyoğlu and Sultanahmet both have fantastic options'
        }
        
        context_note = {
            'luxury': 'upscale dining',
            'budget': 'affordable eats',
            'coffee': 'coffee culture',
            'vegan': 'plant-based options',
            'seafood': 'fresh seafood',
            'turkish': 'traditional Turkish cuisine'
        }.get(context, 'great dining')
        
        alt_location = alternatives.get(location.lower(), 'nearby districts have excellent options')
        return f"I don't have specific {context_note} recommendations for {location} right now, but {alt_location}. Want me to explore those areas instead?"

# Global instance
istanbul_assistant = NonLLMIstanbulAssistant()

def get_response(user_input: str, context: Dict = None) -> str:
    """Main interface for getting responses"""
    return istanbul_assistant.generate_response(user_input, context)

#!/usr/bin/env python3
"""
Ultra-Specialized Istanbul Response System
==========================================

Pure rule-based response system without GPT/LLM dependency.
Uses pattern matching, templates, and local knowledge database
to provide natural, conversational responses that compete with AI systems.
"""

import re
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enhanced_response_templates import IstanbulResponseTemplates

class UltraSpecializedResponseSystem:
    """Rule-based response system using pattern matching and templates"""
    
    def __init__(self):
        self.templates = IstanbulResponseTemplates()
        self.patterns = self._build_pattern_database()
        self.response_cache = {}
        
    def _build_pattern_database(self) -> Dict[str, Any]:
        """Build comprehensive pattern matching database"""
        return {
            'restaurant_patterns': {
                'location_based': [
                    r'restaurants?\s+in\s+(\w+)',
                    r'where\s+to\s+eat\s+in\s+(\w+)',
                    r'food\s+in\s+(\w+)',
                    r'dining\s+in\s+(\w+)',
                    r'best\s+places?\s+to\s+eat\s+in\s+(\w+)'
                ],
                'cuisine_based': [
                    r'(turkish|seafood|italian|french|japanese|korean|thai)\s+restaurants?',
                    r'(kebab|fish|pizza|sushi)\s+places?',
                    r'traditional\s+(turkish|ottoman)\s+food',
                    r'authentic\s+(\w+)\s+cuisine'
                ],
                'budget_based': [
                    r'(cheap|budget|affordable|inexpensive)\s+restaurants?',
                    r'(expensive|luxury|upscale|fine\s+dining)\s+restaurants?',
                    r'(mid-range|moderate)\s+dining'
                ],
                'dietary_based': [
                    r'(vegan|vegetarian|halal|kosher|gluten-free)\s+restaurants?',
                    r'restaurants?\s+with\s+(vegan|vegetarian|halal)\s+options'
                ],
                'coffee_based': [
                    r'coffee\s+shops?',
                    r'(cafe|cafes)',
                    r'where\s+to\s+get\s+coffee',
                    r'best\s+coffee\s+in'
                ]
            },
            'district_patterns': [
                r'tell\s+me\s+about\s+(\w+)',
                r'what.+like\s+in\s+(\w+)',
                r'(\w+)\s+neighborhood',
                r'visiting\s+(\w+)',
                r'exploring\s+(\w+)'
            ],
            'attraction_patterns': [
                r'things\s+to\s+see\s+in\s+(\w+)',
                r'attractions\s+in\s+(\w+)',
                r'best\s+places\s+to\s+visit',
                r'what\s+to\s+do\s+in\s+(\w+)',
                r'sightseeing\s+in\s+(\w+)'
            ],
            'transportation_patterns': [
                r'how\s+to\s+get\s+from\s+(\w+)\s+to\s+(\w+)',
                r'transport\s+from\s+(\w+)\s+to\s+(\w+)',
                r'metro\s+from\s+(\w+)\s+to\s+(\w+)',
                r'getting\s+to\s+(\w+)\s+from\s+(\w+)'
            ]
        }
    
    def generate_response(self, user_input: str, context: Dict = None) -> Dict[str, Any]:
        """Generate specialized response using pattern matching and templates"""
        try:
            # Normalize input
            normalized_input = self._normalize_input(user_input)
            
            # Detect query type and extract parameters
            query_type, parameters = self._analyze_query(normalized_input)
            
            # Generate response based on query type
            if query_type == 'restaurant':
                response = self._generate_restaurant_response(parameters, user_input)
            elif query_type == 'district':
                response = self._generate_district_response(parameters, user_input)
            elif query_type == 'attraction':
                response = self._generate_attraction_response(parameters, user_input)
            elif query_type == 'transportation':
                response = self._generate_transportation_response(parameters, user_input)
            else:
                response = self._generate_fallback_response(user_input)
            
            # Apply context-aware enhancements
            if context:
                response = self._apply_context_enhancements(response, context)
            
            return {
                'success': True,
                'response': response,
                'query_type': query_type,
                'parameters': parameters
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': "I'd be happy to help you explore Istanbul! Could you tell me more about what you're looking for?"
            }
    
    def _normalize_input(self, user_input: str) -> str:
        """Normalize user input for better pattern matching"""
        # Convert to lowercase
        normalized = user_input.lower().strip()
        
        # Fix common Turkish location spellings
        turkish_fixes = {
            'sultanahemt': 'sultanahmet',
            'sultanahmed': 'sultanahmet',
            'beyoglu': 'beyoğlu',
            'kadikoy': 'kadıköy',
            'besiktas': 'beşiktaş',
            'uskudar': 'üsküdar',
            'sisli': 'şişli',
            'bakirkoy': 'bakırköy'
        }
        
        for wrong, correct in turkish_fixes.items():
            normalized = normalized.replace(wrong, correct)
        
        return normalized
    
    def _analyze_query(self, normalized_input: str) -> Tuple[str, Dict]:
        """Analyze query to determine type and extract parameters"""
        
        # Check restaurant patterns first (most specific)
        for pattern_type, patterns in self.patterns['restaurant_patterns'].items():
            for pattern in patterns:
                match = re.search(pattern, normalized_input)
                if match:
                    params = {'pattern_type': pattern_type}
                    if pattern_type == 'location_based':
                        params['location'] = match.group(1)
                    elif pattern_type in ['cuisine_based', 'dietary_based']:
                        params['cuisine_type'] = match.group(1)
                    elif pattern_type == 'budget_based':
                        params['budget_level'] = match.group(1)
                    return 'restaurant', params
        
        # Check district patterns
        for pattern in self.patterns['district_patterns']:
            match = re.search(pattern, normalized_input)
            if match:
                return 'district', {'location': match.group(1)}
        
        # Check attraction patterns
        for pattern in self.patterns['attraction_patterns']:
            match = re.search(pattern, normalized_input)
            if match:
                location = match.group(1) if match.groups() else None
                return 'attraction', {'location': location}
        
        # Check transportation patterns
        for pattern in self.patterns['transportation_patterns']:
            match = re.search(pattern, normalized_input)
            if match:
                return 'transportation', {
                    'from_location': match.group(1),
                    'to_location': match.group(2)
                }
        
        # Default fallback
        return 'general', {}
    
    def _generate_restaurant_response(self, parameters: Dict, original_query: str) -> str:
        """Generate restaurant response using mock data and templates"""
        try:
            # Import restaurant data
            from api_clients.google_places import GooglePlacesClient
            
            # Create client and get mock data
            places_client = GooglePlacesClient()
            
            # Extract search parameters
            location = parameters.get('location', 'Istanbul')
            keyword = None
            
            if parameters.get('pattern_type') == 'cuisine_based':
                keyword = parameters.get('cuisine_type')
            elif parameters.get('pattern_type') == 'budget_based':
                keyword = parameters.get('budget_level')
            elif parameters.get('pattern_type') == 'dietary_based':
                keyword = parameters.get('cuisine_type')
            
            # Get restaurant data
            mock_data = places_client._get_mock_restaurant_data(location, keyword)
            restaurants = mock_data.get('results', [])
            
            # Use enhanced templates for formatting
            if restaurants:
                return self.templates.format_restaurant_response(restaurants, location, original_query)
            else:
                return self.templates._get_no_restaurants_fallback(location, original_query)
                
        except Exception as e:
            return f"I'd love to help you find great restaurants in Istanbul! Let me know what specific area or type of cuisine you're interested in."
    
    def _generate_district_response(self, parameters: Dict, original_query: str) -> str:
        """Generate district guide response"""
        location = parameters.get('location', 'Istanbul')
        
        # Mock attraction data for district
        mock_attractions = [
            {'name': f'{location.title()} Historical Sites', 'description': 'Rich historical heritage'},
            {'name': f'{location.title()} Local Markets', 'description': 'Authentic local experience'}
        ]
        
        return self.templates.format_district_guide(location, mock_attractions)
    
    def _generate_attraction_response(self, parameters: Dict, original_query: str) -> str:
        """Generate attraction recommendations"""
        location = parameters.get('location', 'Istanbul')
        
        # Mock attraction data
        mock_attractions = [
            {
                'name': f'Main Attraction in {location.title()}',
                'description': 'Must-see historical landmark with rich cultural significance'
            },
            {
                'name': f'Hidden Gem in {location.title()}',
                'description': 'Local favorite spot off the beaten path'
            }
        ]
        
        return self.templates.format_attraction_response(mock_attractions, location)
    
    def _generate_transportation_response(self, parameters: Dict, original_query: str) -> str:
        """Generate transportation guidance"""
        from_loc = parameters.get('from_location', 'point A')
        to_loc = parameters.get('to_location', 'point B')
        
        # Mock route data
        mock_route = {
            'mode': 'metro',
            'time': '20-25 minutes',
            'line': 'M2 Green Line',
            'direction': to_loc,
            'alternative': 'tram and ferry combination'
        }
        
        return self.templates.format_transportation_response(from_loc, to_loc, mock_route)
    
    def _generate_fallback_response(self, user_input: str) -> str:
        """Generate helpful fallback response"""
        fallback_responses = [
            "I'm here to help you explore Istanbul! Are you looking for restaurants, attractions, or transportation advice?",
            "Let me help you discover the best of Istanbul. What specific area or experience interests you?",
            "Istanbul has so much to offer! Tell me what you'd like to explore — food, sightseeing, or getting around the city?",
            "I'd love to share local insights about Istanbul with you. What would you like to know more about?"
        ]
        
        return random.choice(fallback_responses)
    
    def _apply_context_enhancements(self, response: str, context: Dict) -> str:
        """Apply contextual enhancements to response"""
        enhanced_response = response
        
        # Add memory references if available
        if context.get('previous_locations'):
            last_location = context['previous_locations'][-1]
            if last_location not in response.lower():
                memory_additions = [
                    f"Since you were interested in {last_location}, you might also enjoy",
                    f"Building on your {last_location} exploration,",
                    f"Following up on {last_location},"
                ]
                if not any(phrase in response for phrase in ["Since you", "Building on", "Following up"]):
                    memory_intro = random.choice(memory_additions)
                    enhanced_response = response.replace("Here are", f"{memory_intro} here are", 1)
        
        # Add time context if available
        current_hour = datetime.now().hour
        if current_hour < 12:
            time_context = "this morning"
        elif current_hour < 18:
            time_context = "this afternoon"
        else:
            time_context = "this evening"
        
        # Occasionally add time references
        if random.random() < 0.3 and time_context not in enhanced_response:
            enhanced_response = enhanced_response.replace("Perfect", f"Perfect for {time_context}", 1)
        
        return enhanced_response

# Global instance for easy import
ultra_specialized_system = UltraSpecializedResponseSystem()

def get_specialized_response(user_input: str, context: Dict = None) -> Dict[str, Any]:
    """Main interface for getting specialized responses"""
    return ultra_specialized_system.generate_response(user_input, context)

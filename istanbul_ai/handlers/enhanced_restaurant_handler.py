#!/usr/bin/env python3
"""
Enhanced Restaurant Query Handler for Istanbul AI
Provides sophisticated restaurant recommendations with advanced matching
"""

import logging
from typing import Dict, List, Optional, Any
from ..services.advanced_restaurant_service import AdvancedRestaurantService, PriceRange, CuisineType, DietaryRequirement
from ..core.user_profile import UserProfile
from ..core.conversation_context import ConversationContext

logger = logging.getLogger(__name__)

class EnhancedRestaurantHandler:
    """
    Enhanced restaurant query handler with sophisticated recommendation logic
    """
    
    def __init__(self):
        self.restaurant_service = AdvancedRestaurantService()
        self.logger = logger
        
        # Query type patterns
        self.query_patterns = {
            'specific_dish': ['kebab', 'baklava', 'meze', 'pide', 'lahmacun', 'doner', 'turkish coffee'],
            'cuisine_type': ['turkish', 'ottoman', 'seafood', 'italian', 'french', 'asian', 'mediterranean'],
            'dietary': ['vegetarian', 'vegan', 'gluten free', 'lactose free', 'halal', 'diabetic'],
            'occasion': ['date', 'business', 'family', 'celebration', 'romantic', 'anniversary'],
            'atmosphere': ['quiet', 'lively', 'cozy', 'trendy', 'traditional', 'modern'],
            'price': ['cheap', 'expensive', 'budget', 'luxury', 'affordable', 'upscale'],
            'time': ['breakfast', 'lunch', 'dinner', 'late night', 'brunch'],
            'location_specific': ['waterfront', 'rooftop', 'view', 'historic', 'bosphorus']
        }

    def handle_restaurant_query(
        self,
        user_input: str,
        user_profile: UserProfile,
        detected_location: Optional[str] = None,
        context: Optional[ConversationContext] = None
    ) -> str:
        """
        Handle restaurant queries with enhanced recommendations
        """
        
        # Analyze query complexity and type
        query_analysis = self._analyze_query(user_input)
        
        # Get recommendations from advanced service
        recommendations = self.restaurant_service.get_recommendations(
            user_input=user_input,
            user_profile=user_profile,
            location=detected_location,
            max_recommendations=5
        )
        
        if not recommendations:
            return self._handle_no_recommendations(user_input, detected_location)
        
        # Format response based on query type and recommendations
        return self._format_enhanced_response(
            recommendations=recommendations,
            query_analysis=query_analysis,
            user_input=user_input,
            detected_location=detected_location,
            context=context
        )

    def _analyze_query(self, user_input: str) -> Dict[str, Any]:
        """Analyze user query to understand intent and complexity"""
        user_input_lower = user_input.lower()
        
        analysis = {
            'query_types': [],
            'complexity': 'simple',
            'specific_requests': [],
            'flexibility_indicators': [],
            'urgency_indicators': []
        }
        
        # Detect query types
        for pattern_type, keywords in self.query_patterns.items():
            if any(keyword in user_input_lower for keyword in keywords):
                analysis['query_types'].append(pattern_type)
        
        # Assess complexity
        if len(analysis['query_types']) >= 3:
            analysis['complexity'] = 'complex'
        elif len(analysis['query_types']) >= 2:
            analysis['complexity'] = 'moderate'
        
        # Detect specific requests
        if any(word in user_input_lower for word in ['must', 'need', 'require', 'essential']):
            analysis['specific_requests'].append('strict_requirements')
        
        if any(word in user_input_lower for word in ['best', 'top', 'recommend', 'suggest']):
            analysis['specific_requests'].append('seeking_recommendations')
        
        # Detect flexibility indicators
        if any(word in user_input_lower for word in ['maybe', 'perhaps', 'could', 'might']):
            analysis['flexibility_indicators'].append('flexible')
        
        if any(word in user_input_lower for word in ['or', 'alternative', 'else', 'other']):
            analysis['flexibility_indicators'].append('open_to_alternatives')
        
        # Detect urgency
        if any(word in user_input_lower for word in ['now', 'tonight', 'today', 'asap']):
            analysis['urgency_indicators'].append('immediate')
        
        if any(word in user_input_lower for word in ['planning', 'future', 'next week']):
            analysis['urgency_indicators'].append('future_planning')
        
        return analysis

    def _format_enhanced_response(
        self,
        recommendations: List,
        query_analysis: Dict[str, Any],
        user_input: str,
        detected_location: Optional[str],
        context: Optional[ConversationContext]
    ) -> str:
        """Format response with enhanced personalization and context"""
        
        # Dynamic response header based on query analysis
        response = self._generate_dynamic_header(query_analysis, detected_location, len(recommendations))
        
        # Add location context explanation if applicable
        if detected_location and context:
            detection_method = context.get_context('location_detection_method', 'unknown')
            if detection_method == 'proximity_inference':
                response += f"🗺️ **Nearby restaurants in {detected_location} area:**\n\n"
            elif detection_method == 'user_profile':
                response += f"🗺️ **Restaurants near your location in {detected_location}:**\n\n"
            elif detection_method == 'explicit_query':
                response += f"🗺️ **Best restaurants in {detected_location} (as requested):**\n\n"
            else:
                response += f"🗺️ **Top restaurants in {detected_location}:**\n\n"
        elif detected_location:
            response += f"🗺️ **Restaurant recommendations for {detected_location}:**\n\n"
        else:
            response += "🗺️ **Restaurant recommendations across Istanbul:**\n\n"
        
        # Format each recommendation
        for i, rec in enumerate(recommendations, 1):
            response += f"**{i}. {self.restaurant_service.format_recommendation(rec)}**\n\n"
        
        # Add contextual advice based on query analysis
        response += self._add_contextual_advice(query_analysis, recommendations, detected_location)
        
        # Add follow-up suggestions
        response += self._add_follow_up_suggestions(query_analysis, recommendations)
        
        return response

    def _generate_dynamic_header(self, query_analysis: Dict[str, Any], location: Optional[str], rec_count: int) -> str:
        """Generate dynamic response header based on query analysis"""
        
        headers = {
            'simple_general': "🍽️ **Restaurant Recommendations for Istanbul**\n\n",
            'simple_specific': "🍽️ **Perfect Restaurant Matches Found**\n\n",
            'moderate_dietary': "🍽️ **Restaurants Matching Your Dietary Needs**\n\n",
            'moderate_occasion': "🍽️ **Perfect Restaurants for Your Occasion**\n\n",
            'complex_multi': "🍽️ **Curated Restaurant Selection**\n\n",
            'location_focused': f"🍽️ **{location} Restaurant Guide**\n\n" if location else "🍽️ **Local Restaurant Recommendations**\n\n"
        }
        
        # Select appropriate header
        if query_analysis['complexity'] == 'complex':
            return headers['complex_multi']
        elif 'dietary' in query_analysis['query_types']:
            return headers['moderate_dietary']
        elif 'occasion' in query_analysis['query_types']:
            return headers['moderate_occasion']
        elif 'location_specific' in query_analysis['query_types'] and location:
            return headers['location_focused']
        elif query_analysis['query_types']:
            return headers['simple_specific']
        else:
            return headers['simple_general']

    def _add_contextual_advice(self, query_analysis: Dict[str, Any], recommendations: List, location: Optional[str]) -> str:
        """Add contextual advice based on query and recommendations"""
        advice = "💡 **Helpful Tips:**\n"
        
        # Time-based advice
        if 'time' in query_analysis['query_types']:
            if 'breakfast' in str(query_analysis).lower():
                advice += "• Most Turkish restaurants serve breakfast until 11 AM\n"
            elif 'dinner' in str(query_analysis).lower():
                advice += "• Dinner typically starts after 7 PM in Istanbul\n"
                advice += "• Make reservations for popular restaurants\n"
        
        # Dietary advice
        if 'dietary' in query_analysis['query_types']:
            advice += "• Always confirm dietary accommodations when making reservations\n"
            advice += "• Turkish cuisine has many naturally vegetarian options\n"
        
        # Occasion advice
        if 'occasion' in query_analysis['query_types']:
            if 'business' in str(query_analysis).lower():
                advice += "• Business lunch typically lasts 1-2 hours in Turkish culture\n"
            elif 'romantic' in str(query_analysis).lower():
                advice += "• Evening reservations recommended for romantic dinners\n"
        
        # Location-specific advice
        if location:
            location_tips = {
                'Sultanahmet': "• Tourist area - restaurants may be pricier but often have English menus\n• Try traditional Ottoman cuisine in historic settings\n",
                'Beyoğlu': "• Vibrant nightlife area with diverse dining options\n• Great for international cuisine and trendy spots\n",
                'Kadıköy': "• Local favorite with authentic, affordable options\n• More frequented by locals than tourists\n",
                'Beşiktaş': "• Excellent seafood restaurants along the Bosphorus\n• Mix of upscale and casual dining\n"
            }
            
            if location in location_tips:
                advice += location_tips[location]
        
        # General Istanbul dining advice
        advice += "• Tipping: 10-15% is standard for good service\n"
        advice += "• Many restaurants don't accept reservations - arrive early for popular spots\n\n"
        
        return advice

    def _add_follow_up_suggestions(self, query_analysis: Dict[str, Any], recommendations: List) -> str:
        """Add relevant follow-up suggestions"""
        suggestions = "🤔 **Need More Help?**\n"
        
        # Based on query complexity
        if query_analysis['complexity'] == 'simple':
            suggestions += "• Ask me about specific cuisines: 'Show me Turkish restaurants'\n"
            suggestions += "• Specify your budget: 'Affordable restaurants in Taksim'\n"
        
        # Based on recommendations
        if len(recommendations) >= 3:
            suggestions += "• Want alternatives? Ask: 'What about Italian restaurants?'\n"
        
        # Location-based suggestions
        suggestions += "• Need directions? Ask: 'How do I get to [restaurant name]?'\n"
        suggestions += "• Want similar places? Ask: 'More restaurants like [restaurant name]'\n"
        
        return suggestions

    def _handle_no_recommendations(self, user_input: str, location: Optional[str]) -> str:
        """Handle cases where no recommendations are found"""
        response = "🍽️ **Restaurant Search Results**\n\n"
        response += "I couldn't find restaurants that exactly match your specific requirements"
        
        if location:
            response += f" in {location}"
        
        response += ".\n\n"
        
        # Provide alternatives
        response += "💡 **Let me help you differently:**\n\n"
        response += "**1. Broaden Your Search:**\n"
        response += "• Try a nearby district\n"
        response += "• Consider similar cuisine types\n"
        response += "• Adjust your budget range\n\n"
        
        response += "**2. Popular Options:**\n"
        
        # Get popular restaurants for the location
        if location:
            popular = self.restaurant_service._filter_by_location(location)[:3]
        else:
            popular = self.restaurant_service.restaurants[:3]
        
        for i, restaurant in enumerate(popular, 1):
            response += f"**{i}. {restaurant.name}** ({restaurant.district})\n"
            response += f"   {restaurant.description[:100]}...\n\n"
        
        response += "🤔 **Try asking differently:**\n"
        response += "• 'Best Turkish restaurants in Sultanahmet'\n"
        response += "• 'Vegetarian-friendly places in Beyoğlu'\n"
        response += "• 'Romantic restaurants with Bosphorus view'\n"
        
        return response

    def get_restaurant_details(self, restaurant_name: str) -> Optional[str]:
        """Get detailed information about a specific restaurant"""
        for restaurant in self.restaurant_service.restaurants:
            if restaurant.name.lower() == restaurant_name.lower():
                details = f"🍽️ **{restaurant.name}** - Detailed Information\n\n"
                details += f"📍 **Location:** {restaurant.address}, {restaurant.district}\n"
                details += f"🍜 **Cuisine:** {', '.join([c.value.replace('_', ' ').title() for c in restaurant.cuisine_types])}\n"
                details += f"💰 **Price Range:** {restaurant.price_range.value}\n"
                details += f"🎭 **Atmosphere:** {restaurant.ambient_type.value.replace('_', ' ').title()}\n\n"
                
                details += f"**Description:**\n{restaurant.description}\n\n"
                
                if restaurant.signature_dishes:
                    details += f"**Signature Dishes:**\n"
                    for dish in restaurant.signature_dishes:
                        details += f"• {dish}\n"
                    details += "\n"
                
                # Features
                features = []
                if restaurant.features.outdoor_seating:
                    features.append("Outdoor seating")
                if restaurant.features.wheelchair_accessible:
                    features.append("Wheelchair accessible")
                if restaurant.features.live_music:
                    features.append("Live music")
                if restaurant.features.english_menu:
                    features.append("English menu available")
                if restaurant.features.view:
                    features.append(f"{restaurant.features.view.replace('_', ' ').title()} view")
                
                if features:
                    details += f"**Features:**\n"
                    for feature in features:
                        details += f"• {feature}\n"
                    details += "\n"
                
                # Dietary options
                if restaurant.dietary_options:
                    details += f"**Dietary Options:**\n"
                    for option in restaurant.dietary_options:
                        details += f"• {option.value.replace('_', ' ').title()}\n"
                    details += "\n"
                
                if restaurant.special_notes:
                    details += f"**Special Notes:**\n"
                    for note in restaurant.special_notes:
                        details += f"• {note}\n"
                
                return details
        
        return None

"""
Response Enhancer Service
Ensures responses match query features and include relevant keywords
"""

from typing import Dict, List, Set, Optional
import re
import logging

logger = logging.getLogger(__name__)


class ResponseEnhancer:
    """Ensure responses match query features and improve personalization"""
    
    def __init__(self):
        self.stop_words = self._load_stop_words()
        self.price_indicators = self._load_price_indicators()
    
    def _load_stop_words(self) -> Set[str]:
        """Load common stop words to filter out"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'about', 'as', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'can', 'what',
            'where', 'when', 'why', 'how', 'which', 'who', 'whom', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our',
            'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
        }
    
    def _load_price_indicators(self) -> Dict[str, str]:
        """Load price level indicators"""
        return {
            'free': '🆓 Free Options',
            'budget': '💰 Budget-Friendly Options',
            'cheap': '💰 Affordable Choices',
            'moderate': '💰💰 Mid-Range Selections',
            'mid-range': '💰💰 Mid-Range Options',
            'upscale': '💰💰💰 Upscale Choices',
            'expensive': '💰💰💰 Premium Options',
            'luxury': '💰💰💰💰 Luxury Experiences',
            'fine': '💰💰💰💰 Fine Dining'
        }
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Convert to lowercase and split
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter out stop words and short words
        key_terms = [
            word for word in words 
            if word not in self.stop_words and len(word) > 2
        ]
        
        return key_terms
    
    def calculate_feature_match(self, query: str, response: str) -> float:
        """Calculate percentage of query keywords present in response"""
        key_terms = self.extract_key_terms(query)
        
        if not key_terms:
            return 1.0  # If no key terms, consider it a match
        
        response_lower = response.lower()
        matches = sum(1 for term in key_terms if term in response_lower)
        
        return matches / len(key_terms)
    
    def enhance_response(self, query: str, base_response: str, entities: Dict = None) -> str:
        """Add query-specific elements to response"""
        if entities is None:
            entities = {}
        
        enhanced = base_response
        prefix_parts = []
        
        # Add location-specific intro if location in query
        if entities.get('location') or entities.get('locations'):
            location = entities.get('location') or (
                entities['locations'][0] if isinstance(entities.get('locations'), list) else entities.get('locations')
            )
            if location:
                prefix_parts.append(f"📍 **{location.title()}** - Great choice!")
        
        # Add price context if budget mentioned in query
        query_lower = query.lower()
        for price_keyword, indicator in self.price_indicators.items():
            if price_keyword in query_lower:
                prefix_parts.append(indicator)
                break
        
        # Add temporal context if time mentioned
        temporal_keywords = {
            'today': '📅 For Today',
            'tonight': '🌙 For Tonight',
            'tomorrow': '📅 For Tomorrow',
            'weekend': '📅 For This Weekend',
            'week': '📅 For This Week',
            'month': '📅 For This Month'
        }
        
        for keyword, context in temporal_keywords.items():
            if keyword in query_lower:
                prefix_parts.append(context)
                break
        
        # Add cuisine/food type if mentioned
        if entities.get('cuisine'):
            cuisine = entities['cuisine']
            if isinstance(cuisine, list):
                cuisine = cuisine[0]
            prefix_parts.append(f"🍽️ **{cuisine.title()} Cuisine**")
        
        # Build prefix
        if prefix_parts:
            prefix = '\n\n'.join(prefix_parts) + '\n\n'
            enhanced = prefix + enhanced
        
        # Check feature match and add keywords if needed
        feature_match = self.calculate_feature_match(query, enhanced)
        
        if feature_match < 0.5:  # Less than 50% match
            enhanced = self._inject_missing_keywords(query, enhanced)
        
        return enhanced
    
    def _inject_missing_keywords(self, query: str, response: str) -> str:
        """Inject missing important keywords into response naturally"""
        key_terms = self.extract_key_terms(query)
        response_lower = response.lower()
        
        missing_terms = [term for term in key_terms if term not in response_lower]
        
        if not missing_terms:
            return response
        
        # Add a context line mentioning missing terms
        missing_context = f"\n\n💡 **Related to your search:** {', '.join(missing_terms)}\n"
        
        # Insert before the last paragraph or at the end
        lines = response.split('\n')
        if len(lines) > 3:
            # Insert before last 2 lines
            insert_pos = len(lines) - 2
            lines.insert(insert_pos, missing_context)
            return '\n'.join(lines)
        else:
            return response + missing_context
    
    def add_context_specific_header(self, query: str, response: str, intent: str = None) -> str:
        """Add query-aware header to response"""
        query_lower = query.lower()
        
        # Intent-specific headers
        if intent == 'restaurant':
            if 'breakfast' in query_lower:
                header = "🍳 **Breakfast Recommendations**\n\n"
            elif 'lunch' in query_lower:
                header = "🍽️ **Lunch Options**\n\n"
            elif 'dinner' in query_lower:
                header = "🌙 **Dinner Recommendations**\n\n"
            elif 'cheap' in query_lower or 'budget' in query_lower:
                header = "💰 **Budget-Friendly Restaurants**\n\n"
            else:
                return response
        elif intent == 'attraction':
            if 'free' in query_lower:
                header = "🆓 **Free Attractions**\n\n"
            elif 'museum' in query_lower:
                header = "🏛️ **Museum Recommendations**\n\n"
            elif 'hidden' in query_lower or 'secret' in query_lower:
                header = "💎 **Hidden Gems**\n\n"
            else:
                return response
        else:
            return response
        
        # Add header if not already present
        if not response.startswith(header.strip()[:20]):
            return header + response
        
        return response
    
    def validate_actionability(self, response: str) -> Dict[str, any]:
        """Check if response has actionable information"""
        checks = {
            'has_location_info': bool(re.search(r'📍|location:|address:', response.lower())),
            'has_hours': bool(re.search(r'⏰|hours:|open:|closes?:', response.lower())),
            'has_price_info': bool(re.search(r'💰|price:|cost:|₺|free', response.lower())),
            'has_directions': bool(re.search(r'how to get|transport|metro|tram|bus|ferry', response.lower())),
            'has_tips': bool(re.search(r'💡|tip:|pro tip|note:', response.lower())),
        }
        
        actionability_score = sum(checks.values()) / len(checks)
        
        return {
            'score': actionability_score,
            'checks': checks,
            'is_actionable': actionability_score >= 0.6  # 60% threshold
        }
    
    def enrich_response_with_details(self, response: str, additional_info: Dict = None) -> str:
        """Add missing actionable details to response"""
        if additional_info is None:
            return response
        
        enrichments = []
        
        # Add transportation if available
        if additional_info.get('transportation') and '🚇' not in response:
            enrichments.append(f"\n🚇 **How to Get There:** {additional_info['transportation']}")
        
        # Add hours if available
        if additional_info.get('hours') and '⏰' not in response:
            enrichments.append(f"\n⏰ **Hours:** {additional_info['hours']}")
        
        # Add price if available
        if additional_info.get('price') and '💰' not in response:
            enrichments.append(f"\n💰 **Price:** {additional_info['price']}")
        
        # Add tip if available
        if additional_info.get('tip') and '💡' not in response:
            enrichments.append(f"\n💡 **Pro Tip:** {additional_info['tip']}")
        
        if enrichments:
            return response + '\n' + ''.join(enrichments)
        
        return response
    
    def format_with_emojis(self, response: str) -> str:
        """Ensure consistent emoji usage for better readability"""
        # Patterns to enhance
        replacements = [
            (r'\b(location|address):', '📍 Location:'),
            (r'\b(hours?|open|opening):', '⏰ Hours:'),
            (r'\b(price|cost):', '💰 Price:'),
            (r'\b(phone|tel|contact):', '📞 Contact:'),
            (r'\b(tip|advice|note):', '💡 Tip:'),
            (r'\b(website|url):', '🌐 Website:'),
            (r'\b(food|dish|menu):', '🍽️ Food:'),
            (r'\b(drinks?|beverage):', '🍹 Drinks:'),
        ]
        
        enhanced = response
        for pattern, replacement in replacements:
            enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)
        
        return enhanced
    
    def improve_feature_matching(self, query: str, response: str, entities: Dict = None, intent: str = None) -> str:
        """Main method to improve feature matching of response"""
        # Step 1: Add context-specific headers
        enhanced = self.add_context_specific_header(query, response, intent)
        
        # Step 2: Enhance with entity-based context
        enhanced = self.enhance_response(query, enhanced, entities)
        
        # Step 3: Format with emojis for consistency
        enhanced = self.format_with_emojis(enhanced)
        
        # Step 4: Check feature match
        feature_match = self.calculate_feature_match(query, enhanced)
        
        logger.info(f"Feature match score: {feature_match:.2%}")
        
        return enhanced
    
    def generate_summary_line(self, query: str, intent: str = None) -> str:
        """Generate a query-aware summary line"""
        query_lower = query.lower()
        
        # Extract main concepts
        concepts = []
        
        # Location
        location_keywords = ['in', 'at', 'near', 'around']
        for keyword in location_keywords:
            if keyword in query_lower:
                parts = query_lower.split(keyword)
                if len(parts) > 1:
                    location = parts[1].strip().split()[0]
                    concepts.append(f"in {location.title()}")
                    break
        
        # Budget
        if any(word in query_lower for word in ['cheap', 'budget', 'affordable']):
            concepts.append('budget-friendly')
        elif any(word in query_lower for word in ['expensive', 'luxury', 'fine']):
            concepts.append('upscale')
        
        # Time
        if 'breakfast' in query_lower:
            concepts.append('for breakfast')
        elif 'lunch' in query_lower:
            concepts.append('for lunch')
        elif 'dinner' in query_lower:
            concepts.append('for dinner')
        
        if concepts:
            return f"Here are recommendations {' '.join(concepts)}:"
        
        return "Here are personalized recommendations based on your query:"


# Singleton instance
_response_enhancer = None

def get_response_enhancer() -> ResponseEnhancer:
    """Get or create response enhancer instance"""
    global _response_enhancer
    if _response_enhancer is None:
        _response_enhancer = ResponseEnhancer()
    return _response_enhancer

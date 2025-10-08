#!/usr/bin/env python3
"""
Location Intent Detection System for AI Istanbul
Detects when users want location-based recommendations for museums, restaurants, etc.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from thefuzz import fuzz, process

class LocationIntentType(Enum):
    RESTAURANTS = "restaurants"
    MUSEUMS = "museums"
    ATTRACTIONS = "attractions"
    SHOPPING = "shopping"
    TRANSPORTATION = "transportation"
    GENERAL_NEARBY = "general_nearby"
    ROUTE_PLANNING = "route_planning"

@dataclass
class LocationIntent:
    """Represents a detected location-based intent"""
    intent_type: LocationIntentType
    confidence: float
    keywords_matched: List[str]
    location_context: Optional[str] = None
    specific_requirements: Dict[str, any] = None
    distance_preference: Optional[str] = None

class LocationIntentDetector:
    """Detects user intents for location-based queries"""
    
    def __init__(self):
        # Restaurant-related patterns
        self.restaurant_patterns = {
            'direct': [
                r'\b(restaurant|restaurants|restoran|lokanta|café|cafe|bistro|diner|eatery|eateries)\b',
                r'\b(eat|eating|food|yemek|meal|lunch|dinner|breakfast|dining|dine)\b',
                r'\b(hungry|starving|famished)\b',
                r'\b(where to eat|places to eat|good food|best food|food places)\b',
                r'\b(dining recommendations|dining options|food recommendations)\b',
                r'\b(local eateries|local restaurants|food scene)\b'
            ],
            'cuisine': [
                r'\b(turkish|ottoman|kebab|döner|baklava|turkish food|traditional turkish)\b',
                r'\b(seafood|fish|balık|italian|chinese|pizza|burger|meze|pide)\b',
                r'\b(vegetarian|vegan|halal|kosher|gluten.free|gluten free)\b',
                r'\b(street food|börek|balık ekmek|döner kebab)\b',
                r'\b(authentic turkish|turkish cuisine|ottoman food|turkish breakfast)\b',
                r'\b(fresh seafood|fish restaurants|black sea fish|maritime cuisine)\b',
                r'\b(turkish street food|street food vendors|pastry shops)\b'
            ],
            'dining_context': [
                r'\b(romantic|family|casual|fine dining|cheap|expensive|budget|luxury)\b',
                r'\b(rooftop|view|terrace|garden|outdoor|bosphorus view)\b',
                r'\b(traditional|authentic|local|hip|trendy|popular)\b'
            ],
            'dietary_religious': [
                r'\b(religious|religiously|compliant|muslim|islamic|christian|jewish)\b',
                r'\b(dietary|diet|dietary restrictions|food restrictions)\b',
                r'\b(halal certified|kosher certified|religious dining)\b',
                r'\b(halal|kosher|muslim-friendly|jewish-friendly)\b',
                r'\b(religiously compliant|religious compliance)\b',
                r'\b(celiac|coeliac|celiac-friendly|coeliac-friendly)\b',
                r'\b(gluten.free|gluten-free|gluten free|wheat.free|wheat-free|wheat free)\b',
                r'\b(allergy|allergies|food allergy|food allergies|allergy-friendly)\b',
                r'\b(special diet|special dietary|dietary needs|dietary requirements)\b',
                r'\b(friendly|accommodating|suitable|compliant|specialized)\b'
            ],
            'budget_terms': [
                r'\b(cheap|budget|affordable|inexpensive|low.cost|economical)\b',
                r'\b(cheap eats|budget.friendly|value|good value|bargain)\b',
                r'\b(student.friendly|backpacker|budget dining)\b'
            ]
        }
        
        # Museum-related patterns
        self.museum_patterns = {
            'direct': [
                r'\b(museum|museums|müze|gallery|galeri|exhibition|sergi)\b',
                r'\b(palace|saray|mosque|cami|church)\b',
                r'\b(historical sites|cultural sites|archaeological sites)\b'
            ],
            'specific': [
                r'\b(topkapi|hagia sophia|ayasofya|dolmabahçe|basilica cistern)\b',
                r'\b(byzantine|ottoman|archaeology|arkeoloji)\b',
                r'\b(turkish museums|museum pass|museumpass)\b',
                r'\b(galata tower|chora church|istanbul modern)\b'
            ],
            'context': [
                r'\b(art museum|history museum|archaeological museum)\b',
                r'\b(visit museum|see museum|museum tour|museum ticket)\b'
            ]
        }
        
        # Location context patterns
        self.location_patterns = {
            'proximity': [
                r'\b(near|nearby|close|around|within|walking distance)\b',
                r'\b(yakın|yakında|civar|etraf|yürüme mesafesi)\b',
                r'\b(\d+\s*(km|meter|mile|minute|dakika))\b'
            ],
            'direction': [
                r'\b(from here|current location|where i am)\b',
                r'\b(buradan|bulunduğum yer|şu anki konum)\b'
            ],
            'districts': [
                r'\b(sultanahmet|taksim|galata|karaköy|beşiktaş|kadıköy)\b',
                r'\b(fatih|beyoğlu|üsküdar|sarıyer|şişli)\b',
                r'\b(asian side|anatolian side|europe|european side)\b',
                r'\b(moda|cihangir|nişantaşı|ortaköy|bebek)\b',
                r'\b(istiklal|istiklal avenue|istiklal caddesi)\b',
                r'\b(old city|historic peninsula|new city)\b',
                r'\b(fenerbahçe|kumkapı|eminönü|topkapı)\b'
            ],
            'specific_places': [
                r'\b(taksim square|istiklal avenue|galata tower)\b',
                r'\b(blue mosque|hagia sophia|topkapi palace)\b',
                r'\b(galata bridge|bosphorus|golden horn)\b',
                r'\b(grand bazaar|spice bazaar|egyptian bazaar)\b',
                r'\b(basilica cistern|chora church|dolmabahçe palace)\b'
            ],
            'neighborhoods': [
                r'\b(neighborhood|neighbourhood|area|district|quarter)\b',
                r'\b(mahalle|semt|bölge|çevre)\b'
            ]
        }
        
        # Route planning patterns
        self.route_patterns = [
            r'\b(how to get|directions|route|way to|path to|how do i get)\b',
            r'\b(nasıl gidilir|yol tarifi|güzergah)\b',
            r'\b(metro|bus|ferry|taxi|walk|drive)\b',
            r'\b(get to|go to|travel to)\b'
        ]

        # Common Turkish and English typos and corrections
        self.typo_corrections = {
            # Restaurant typos
            'resturant': 'restaurant',
            'resturants': 'restaurants',
            'restraunt': 'restaurant',
            'restaurnt': 'restaurant',
            'restaraunt': 'restaurant',
            'restoran': 'restaurant',
            'restorant': 'restaurant',
            'restuaran': 'restaurant',
            
            # District typos  
            'beyoglu': 'beyoğlu',
            'beyolu': 'beyoğlu',
            'beyolu': 'beyoğlu',
            'taksim': 'taksim',
            'taksi': 'taksim',
            'sultanhamet': 'sultanahmet',
            'sultanamet': 'sultanahmet',
            'sultanahme': 'sultanahmet',
            'kadikoy': 'kadıköy',
            'kadikoy': 'kadıköy',
            'kadikoy': 'kadıköy',
            'besiktas': 'beşiktaş',
            'besikta': 'beşiktaş',
            
            # Food type typos
            'turkis': 'turkish',
            'turiksh': 'turkish',
            'turksh': 'turkish',
            'seafod': 'seafood',
            'seefood': 'seafood',
            'vegitarian': 'vegetarian',
            'vegeterian': 'vegetarian',
            'vegetrarian': 'vegetarian',
            'halall': 'halal',
            'hala': 'halal',
            
            # Common words
            'recomendations': 'recommendations',
            'recomendation': 'recommendation',
            'recomend': 'recommend',
            'rcommend': 'recommend',
            'rcommendation': 'recommendation',
            'options': 'options',
            'optins': 'options',
            'good': 'good',
            'goo': 'good',
            'near': 'near',
            'nea': 'near',
            'beste': 'best',
            'bst': 'best'
        }

    def correct_typos(self, text: str) -> str:
        """
        Correct common typos in user input
        
        Args:
            text: Input text with potential typos
            
        Returns:
            Corrected text
        """
        corrected_text = text.lower()
        
        # Apply direct corrections
        for typo, correction in self.typo_corrections.items():
            corrected_text = re.sub(r'\b' + re.escape(typo) + r'\b', correction, corrected_text)
        
        # For words not in direct corrections, use fuzzy matching for key terms
        words = corrected_text.split()
        corrected_words = []
        
        key_terms = [
            'restaurant', 'restaurants', 'beyoğlu', 'sultanahmet', 'kadıköy', 'beşiktaş',
            'turkish', 'seafood', 'vegetarian', 'halal', 'recommendations', 'options',
            'museum', 'museums', 'galata', 'taksim'
        ]
        
        for word in words:
            if len(word) > 3:  # Only correct longer words
                best_match = process.extractOne(word, key_terms, scorer=fuzz.ratio)
                if best_match and best_match[1] >= 80:  # 80% similarity threshold
                    corrected_words.append(best_match[0])
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)

    def detect_intent(self, user_message: str, user_location: Optional[Dict] = None) -> List[LocationIntent]:
        """
        Detect location-based intents in user message
        
        Args:
            user_message: The user's query
            user_location: User's current location (lat, lng, district)
            
        Returns:
            List of detected intents with confidence scores
        """
        # Apply typo correction first
        corrected_message = self.correct_typos(user_message)
        user_message = corrected_message.lower()
        detected_intents = []
        
        # Check for restaurant intent
        restaurant_intent = self._detect_restaurant_intent(user_message, user_location)
        if restaurant_intent:
            detected_intents.append(restaurant_intent)
        
        # Check for museum intent
        museum_intent = self._detect_museum_intent(user_message, user_location)
        if museum_intent:
            detected_intents.append(museum_intent)
        
        # Check for route planning intent
        route_intent = self._detect_route_intent(user_message, user_location)
        if route_intent:
            detected_intents.append(route_intent)
        
        # Check for general nearby intent
        if not detected_intents:
            general_intent = self._detect_general_nearby_intent(user_message, user_location)
            if general_intent:
                detected_intents.append(general_intent)
        
        return sorted(detected_intents, key=lambda x: x.confidence, reverse=True)

    def _detect_restaurant_intent(self, message: str, location: Optional[Dict]) -> Optional[LocationIntent]:
        """Detect restaurant-related intents"""
        score = 0
        matched_keywords = []
        requirements = {}
        
        # Check direct restaurant mentions (higher weight)
        for pattern in self.restaurant_patterns['direct']:
            if re.search(pattern, message):
                score += 0.6  # Increased from 0.4
                matched_keywords.extend(re.findall(pattern, message))
        
        # Check cuisine preferences
        for pattern in self.restaurant_patterns['cuisine']:
            matches = re.findall(pattern, message)
            if matches:
                score += 0.4  # Increased from 0.3
                matched_keywords.extend(matches)
                requirements['cuisine'] = matches
        
        # Check dining context
        for pattern in self.restaurant_patterns['dining_context']:
            matches = re.findall(pattern, message)
            if matches:
                score += 0.3  # Increased from 0.2
                matched_keywords.extend(matches)
                requirements['dining_style'] = matches
        
        # Check budget terms (high weight for budget queries)
        for pattern in self.restaurant_patterns['budget_terms']:
            matches = re.findall(pattern, message)
            if matches:
                score += 0.5  # High score for budget-related queries
                matched_keywords.extend(matches)
                if 'dining_style' not in requirements:
                    requirements['dining_style'] = []
                requirements['dining_style'].extend(matches)
        
        # Check dietary/religious terms (high weight for dietary queries)
        for pattern in self.restaurant_patterns['dietary_religious']:
            matches = re.findall(pattern, message)
            if matches:
                score += 0.6  # Very high score for dietary/religious queries
                matched_keywords.extend(matches)
                if 'dietary_requirements' not in requirements:
                    requirements['dietary_requirements'] = []
                requirements['dietary_requirements'].extend(matches)
        
        # Check for location context
        location_score, distance_pref = self._detect_location_context(message)
        score += location_score
        
        if score >= 0.3:  # Minimum threshold
            return LocationIntent(
                intent_type=LocationIntentType.RESTAURANTS,
                confidence=min(score, 1.0),
                keywords_matched=matched_keywords,
                location_context=location.get('district') if location else None,
                specific_requirements=requirements,
                distance_preference=distance_pref
            )
        
        return None

    def _detect_museum_intent(self, message: str, location: Optional[Dict]) -> Optional[LocationIntent]:
        """Detect museum/cultural site intents"""
        score = 0
        matched_keywords = []
        requirements = {}
        
        # Check direct museum mentions
        for pattern in self.museum_patterns['direct']:
            if re.search(pattern, message):
                score += 0.5
                matched_keywords.extend(re.findall(pattern, message))
        
        # Check specific museum mentions
        for pattern in self.museum_patterns['specific']:
            matches = re.findall(pattern, message)
            if matches:
                score += 0.4
                matched_keywords.extend(matches)
                requirements['specific_sites'] = matches
        
        # Check museum context patterns
        for pattern in self.museum_patterns['context']:
            if re.search(pattern, message):
                score += 0.3
                matched_keywords.extend(re.findall(pattern, message))
        
        # Check for location context
        location_score, distance_pref = self._detect_location_context(message)
        score += location_score
        
        if score >= 0.3:
            return LocationIntent(
                intent_type=LocationIntentType.MUSEUMS,
                confidence=min(score, 1.0),
                keywords_matched=matched_keywords,
                location_context=location.get('district') if location else None,
                specific_requirements=requirements,
                distance_preference=distance_pref
            )
        
        return None

    def _detect_route_intent(self, message: str, location: Optional[Dict]) -> Optional[LocationIntent]:
        """Detect route planning intents"""
        score = 0
        matched_keywords = []
        
        for pattern in self.route_patterns:
            if re.search(pattern, message):
                score += 0.5
                matched_keywords.extend(re.findall(pattern, message))
        
        if score >= 0.4:
            return LocationIntent(
                intent_type=LocationIntentType.ROUTE_PLANNING,
                confidence=min(score, 1.0),
                keywords_matched=matched_keywords,
                location_context=location.get('district') if location else None
            )
        
        return None

    def _detect_general_nearby_intent(self, message: str, location: Optional[Dict]) -> Optional[LocationIntent]:
        """Detect general 'what's nearby' intents"""
        score = 0
        matched_keywords = []
        
        # Generic nearby patterns
        nearby_patterns = [
            r'\b(what\'s near|what is near|around here|nearby)\b',
            r'\b(yakında ne var|civar|etraf)\b',
            r'\b(recommend|suggest|öneri)\b'
        ]
        
        for pattern in nearby_patterns:
            if re.search(pattern, message):
                score += 0.3
                matched_keywords.extend(re.findall(pattern, message))
        
        # Check for location context
        location_score, distance_pref = self._detect_location_context(message)
        score += location_score
        
        if score >= 0.3:
            return LocationIntent(
                intent_type=LocationIntentType.GENERAL_NEARBY,
                confidence=min(score, 1.0),
                keywords_matched=matched_keywords,
                location_context=location.get('district') if location else None,
                distance_preference=distance_pref
            )
        
        return None

    def _detect_location_context(self, message: str) -> Tuple[float, Optional[str]]:
        """Detect location context and distance preferences"""
        score = 0
        distance_pref = None
        
        # Check proximity patterns
        for pattern in self.location_patterns['proximity']:
            if re.search(pattern, message):
                score += 0.2
                # Extract distance if mentioned
                distance_match = re.search(r'(\d+\s*(km|meter|mile|minute|dakika))', message)
                if distance_match:
                    distance_pref = distance_match.group(1)
        
        # Check direction patterns
        for pattern in self.location_patterns['direction']:
            if re.search(pattern, message):
                score += 0.3
        
        # Check district mentions (high weight for clear location references)
        for pattern in self.location_patterns['districts']:
            if re.search(pattern, message):
                score += 0.3  # Increased from 0.1
        
        # Check specific places (high weight for landmark mentions)
        for pattern in self.location_patterns['specific_places']:
            if re.search(pattern, message):
                score += 0.4  # High score for specific landmarks
        
        # Check neighborhood indicators
        for pattern in self.location_patterns['neighborhoods']:
            if re.search(pattern, message):
                score += 0.2  # Medium score for area indicators
        
        return score, distance_pref

    def generate_location_response(self, intent: LocationIntent, user_location: Optional[Dict] = None) -> Dict:
        """
        Generate appropriate response based on detected intent
        
        Returns:
            Response configuration for the AI system
        """
        response_config = {
            'intent_type': intent.intent_type.value,
            'confidence': intent.confidence,
            'location_aware': True,
            'user_location': user_location,
            'search_params': {}
        }
        
        if intent.intent_type == LocationIntentType.RESTAURANTS:
            response_config.update({
                'search_type': 'restaurants',
                'search_params': {
                    'category': 'restaurant',
                    'radius': self._parse_distance(intent.distance_preference) or 2000,  # 2km default
                    'cuisine': intent.specific_requirements.get('cuisine', []),
                    'dining_style': intent.specific_requirements.get('dining_style', [])
                },
                'response_template': 'location_restaurants'
            })
        
        elif intent.intent_type == LocationIntentType.MUSEUMS:
            response_config.update({
                'search_type': 'museums',
                'search_params': {
                    'category': 'museum',
                    'radius': self._parse_distance(intent.distance_preference) or 3000,  # 3km default
                    'specific_sites': intent.specific_requirements.get('specific_sites', [])
                },
                'response_template': 'location_museums'
            })
        
        elif intent.intent_type == LocationIntentType.ROUTE_PLANNING:
            response_config.update({
                'search_type': 'routes',
                'search_params': {
                    'from_location': user_location,
                    'transport_preferences': intent.specific_requirements.get('transport', ['walking', 'metro'])
                },
                'response_template': 'location_routes'
            })
        
        else:  # GENERAL_NEARBY
            response_config.update({
                'search_type': 'general',
                'search_params': {
                    'radius': self._parse_distance(intent.distance_preference) or 1500,  # 1.5km default
                    'categories': ['restaurant', 'museum', 'attraction']
                },
                'response_template': 'location_general'
            })
        
        return response_config

    def _parse_distance(self, distance_str: Optional[str]) -> Optional[int]:
        """Parse distance string to meters"""
        if not distance_str:
            return None
        
        # Extract number and unit
        match = re.search(r'(\d+(?:\.\d+)?)\s*(km|kilometer|m|meter|mile|minute|dakika)', distance_str.lower())
        if not match:
            return None
        
        value = float(match.group(1))
        unit = match.group(2)
        
        # Convert to meters
        if unit in ['km', 'kilometer']:
            return int(value * 1000)
        elif unit in ['m', 'meter']:
            return int(value)
        elif unit == 'mile':
            return int(value * 1609)
        elif unit in ['minute', 'dakika']:
            # Assume walking speed of 5 km/h
            return int(value * 83.33)  # meters per minute walking
        
        return None


# Example usage and integration
def example_usage():
    """Example of how to use the LocationIntentDetector"""
    
    detector = LocationIntentDetector()
    
    # Example user location (Sultanahmet)
    user_location = {
        'latitude': 41.0082,
        'longitude': 28.9784,
        'district': 'Sultanahmet',
        'accuracy': 50
    }
    
    # Test queries
    test_queries = [
        "What restaurants are near me?",
        "I'm hungry, any good Turkish food around here?",
        "Show me museums within walking distance",
        "How do I get to Galata Tower from here?",
        "What's around Taksim Square?",
        "Any romantic restaurants with a view nearby?",
        "Museums about Ottoman history close by",
        "Vegetarian places within 1km"
    ]
    
    print("=== Location Intent Detection Examples ===\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        intents = detector.detect_intent(query, user_location)
        
        if intents:
            for i, intent in enumerate(intents, 1):
                print(f"  Intent {i}: {intent.intent_type.value}")
                print(f"    Confidence: {intent.confidence:.2f}")
                print(f"    Keywords: {intent.keywords_matched}")
                if intent.specific_requirements:
                    print(f"    Requirements: {intent.specific_requirements}")
                if intent.distance_preference:
                    print(f"    Distance: {intent.distance_preference}")
                
                # Generate response config
                response_config = detector.generate_location_response(intent, user_location)
                print(f"    Response Config: {response_config['search_type']} with {response_config['search_params']}")
        else:
            print("  No location intent detected")
        
        print()

if __name__ == "__main__":
    example_usage()

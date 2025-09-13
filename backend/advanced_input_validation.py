"""
Advanced Input Validation and Processing System for AI Istanbul Chatbot
Handles the challenge of processing billions of possible user inputs
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class InputCategory(Enum):
    """Categories for input classification"""
    TOURISM = "tourism"
    FOOD = "food"
    TRANSPORT = "transport" 
    ACCOMMODATION = "accommodation"
    SHOPPING = "shopping"
    ENTERTAINMENT = "entertainment"
    WEATHER = "weather"
    CULTURE = "culture"
    PRACTICAL = "practical"
    GIBBERISH = "gibberish"
    SPAM = "spam"
    INAPPROPRIATE = "inappropriate"
    UNKNOWN = "unknown"

@dataclass
class ValidationResult:
    """Result of input validation"""
    is_valid: bool
    category: InputCategory
    confidence: float
    sanitized_input: str
    detected_entities: Dict[str, Any]
    suggestions: List[str]
    error_message: Optional[str] = None

class InputValidator:
    """Advanced input validation and processing system"""
    
    def __init__(self):
        self.max_input_length = 500
        self.min_input_length = 2
        self.spam_threshold = 0.8
        self.gibberish_threshold = 0.7
        
        # Load common patterns and dictionaries
        self._load_patterns()
        self._load_entity_dictionaries()
    
    def _load_patterns(self):
        """Load regex patterns for different types of validation"""
        self.patterns = {
            'harmful_chars': r'[<>"\';{}()\\]',
            'repeated_chars': r'(.)\1{4,}',
            'only_special': r'^[^a-zA-Z0-9\s]+$',
            'sql_injection': r'(?i)(drop|delete|insert|update|select|union|exec|script)',
            'spam_words': r'(?i)(buy now|click here|free money|urgent|!!!!!)',
            'email_phone': r'[\w\.-]+@[\w\.-]+\.\w+|\+?1?\d{9,15}',
            'urls': r'https?://[^\s]+',
            'emoji_only': r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\s]+$'
        }
    
    def _load_entity_dictionaries(self):
        """Load dictionaries for entity recognition"""
        self.entities = {
            'districts': [
                'sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'uskudar', 'besiktas',
                'sisli', 'taksim', 'eminonu', 'fatih', 'bakirkoy', 'zeytinburnu',
                'cihangir', 'ortakoy', 'bebek', 'arnavutkoy', 'balat', 'fener'
            ],
            'food_types': [
                'turkish', 'kebab', 'meze', 'baklava', 'turkish breakfast', 'fish',
                'vegetarian', 'vegan', 'halal', 'seafood', 'street food', 'dessert',
                'coffee', 'tea', 'ottoman', 'mediterranean', 'middle eastern'
            ],
            'attractions': [
                'hagia sophia', 'blue mosque', 'topkapi palace', 'grand bazaar',
                'galata tower', 'basilica cistern', 'dolmabahce palace', 'taksim square',
                'bosphorus', 'maiden tower', 'spice bazaar', 'chora church'
            ],
            'transport': [
                'metro', 'bus', 'tram', 'ferry', 'taxi', 'dolmus', 'marmaray',
                'funicular', 'metrobus', 'airport', 'istanbulkart'
            ],
            'time_expressions': [
                'morning', 'afternoon', 'evening', 'night', 'weekend', 'weekday',
                'summer', 'winter', 'spring', 'autumn', 'ramadan', 'holiday'
            ]
        }
    
    def sanitize_input(self, raw_input: str) -> str:
        """Clean and sanitize user input"""
        if not raw_input:
            return ""
        
        # Basic cleaning
        text = raw_input.strip()
        
        # Length limits
        if len(text) > self.max_input_length:
            text = text[:self.max_input_length]
        
        # Remove harmful characters but preserve meaningful punctuation
        text = re.sub(self.patterns['harmful_chars'], '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs (privacy/security)
        text = re.sub(self.patterns['urls'], '', text)
        
        # Remove email/phone (privacy)
        text = re.sub(self.patterns['email_phone'], '', text)
        
        return text.strip()
    
    def detect_spam_gibberish(self, text: str) -> Tuple[bool, bool, float]:
        """Detect spam and gibberish content"""
        if not text:
            return True, True, 0.0
        
        # Check for repeated characters (gibberish indicator)
        repeated_match = re.search(self.patterns['repeated_chars'], text)
        has_repeated = bool(repeated_match)
        
        # Check for only special characters
        only_special = bool(re.match(self.patterns['only_special'], text))
        
        # Check for spam patterns
        spam_match = re.search(self.patterns['spam_words'], text)
        has_spam = bool(spam_match)
        
        # Check for only emojis
        only_emoji = bool(re.match(self.patterns['emoji_only'], text))
        
        # Calculate gibberish score
        words = text.split()
        if not words:
            return True, True, 0.0
        
        # Check for reasonable word structure
        valid_words = 0
        for word in words:
            # Simple heuristic: word should have vowels and reasonable length
            if (len(word) >= 2 and 
                any(c in 'aeiouAEIOU' for c in word) and
                not re.match(r'^(.)\1+$', word)):  # Not just repeated characters
                valid_words += 1
        
        valid_ratio = valid_words / len(words) if words else 0
        
        is_gibberish = (has_repeated or only_special or only_emoji or 
                       valid_ratio < self.gibberish_threshold)
        is_spam = has_spam
        
        confidence = valid_ratio
        
        return is_spam, is_gibberish, confidence
    
    def categorize_input(self, text: str) -> Tuple[InputCategory, float]:
        """Categorize the input based on content analysis"""
        if not text:
            return InputCategory.UNKNOWN, 0.0
        
        text_lower = text.lower()
        
        # Define category keywords with weights
        category_keywords = {
            InputCategory.FOOD: {
                'keywords': ['restaurant', 'food', 'eat', 'meal', 'kitchen', 'cafe', 'breakfast', 'lunch', 'dinner'] + self.entities['food_types'],
                'weight': 1.0
            },
            InputCategory.TOURISM: {
                'keywords': ['visit', 'see', 'tour', 'attraction', 'museum', 'palace', 'mosque', 'church'] + self.entities['attractions'],
                'weight': 1.0
            },
            InputCategory.TRANSPORT: {
                'keywords': ['transport', 'travel', 'go', 'get', 'move', 'way'] + self.entities['transport'],
                'weight': 1.0
            },
            InputCategory.ACCOMMODATION: {
                'keywords': ['hotel', 'stay', 'sleep', 'accommodation', 'hostel', 'apartment', 'room'],
                'weight': 1.0
            },
            InputCategory.SHOPPING: {
                'keywords': ['shop', 'buy', 'market', 'bazaar', 'store', 'souvenir', 'clothes', 'gift'],
                'weight': 1.0
            },
            InputCategory.ENTERTAINMENT: {
                'keywords': ['fun', 'entertainment', 'nightlife', 'bar', 'club', 'music', 'dance', 'party'],
                'weight': 1.0
            },
            InputCategory.WEATHER: {
                'keywords': ['weather', 'temperature', 'rain', 'sun', 'climate', 'season', 'hot', 'cold'],
                'weight': 1.0
            },
            InputCategory.CULTURE: {
                'keywords': ['culture', 'history', 'tradition', 'custom', 'language', 'people', 'local'],
                'weight': 1.0
            },
            InputCategory.PRACTICAL: {
                'keywords': ['money', 'currency', 'safe', 'tip', 'advice', 'help', 'information', 'guide'],
                'weight': 1.0
            }
        }
        
        # Calculate scores for each category
        category_scores = {}
        total_words = len(text_lower.split())
        
        for category, data in category_keywords.items():
            score = 0
            keywords = data['keywords']
            weight = data['weight']
            
            for keyword in keywords:
                if keyword in text_lower:
                    # Give higher score for exact matches
                    if f" {keyword} " in f" {text_lower} ":
                        score += 2 * weight
                    else:
                        score += weight
            
            # Normalize by total words
            if total_words > 0:
                category_scores[category] = score / total_words
            else:
                category_scores[category] = 0
        
        # Find best category
        if category_scores:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            confidence = category_scores[best_category]
            
            if confidence > 0.1:  # Minimum confidence threshold
                return best_category, confidence
        
        return InputCategory.UNKNOWN, 0.0
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from the input text"""
        entities = {
            'districts': [],
            'food_types': [],
            'attractions': [],
            'transport': [],
            'time_expressions': []
        }
        
        text_lower = text.lower()
        
        for entity_type, entity_list in self.entities.items():
            for entity in entity_list:
                if entity in text_lower:
                    entities[entity_type].append(entity)
        
        return entities
    
    def generate_suggestions(self, category: InputCategory, entities: Dict[str, List[str]]) -> List[str]:
        """Generate helpful suggestions based on category and entities"""
        suggestions = []
        
        if category == InputCategory.FOOD:
            suggestions = [
                "restaurants in Sultanahmet",
                "Turkish breakfast places",
                "vegetarian restaurants",
                "halal food options"
            ]
        elif category == InputCategory.TOURISM:
            suggestions = [
                "top attractions in Istanbul",
                "museums to visit",
                "historical sites",
                "Bosphorus cruise"
            ]
        elif category == InputCategory.TRANSPORT:
            suggestions = [
                "how to use Istanbul metro",
                "airport to city center",
                "public transport cards",
                "taxi vs metro costs"
            ]
        elif category == InputCategory.UNKNOWN:
            suggestions = [
                "places to visit in Istanbul",
                "best Turkish restaurants",
                "how to get around the city",
                "things to do in Istanbul"
            ]
        
        # Add location-specific suggestions if district detected
        if entities.get('districts'):
            district = entities['districts'][0]
            suggestions.append(f"things to do in {district}")
            suggestions.append(f"restaurants in {district}")
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def validate_input(self, raw_input: str) -> ValidationResult:
        """Main validation function that processes input through all checks"""
        try:
            # Step 1: Basic sanitization
            sanitized = self.sanitize_input(raw_input)
            
            # Step 2: Length validation
            if len(sanitized) < self.min_input_length:
                return ValidationResult(
                    is_valid=False,
                    category=InputCategory.UNKNOWN,
                    confidence=0.0,
                    sanitized_input=sanitized,
                    detected_entities={},
                    suggestions=["Please enter a more detailed question"],
                    error_message="Input too short"
                )
            
            # Step 3: Spam and gibberish detection
            is_spam, is_gibberish, quality_score = self.detect_spam_gibberish(sanitized)
            
            if is_spam:
                return ValidationResult(
                    is_valid=False,
                    category=InputCategory.SPAM,
                    confidence=0.0,
                    sanitized_input=sanitized,
                    detected_entities={},
                    suggestions=["Please ask about Istanbul travel information"],
                    error_message="Spam content detected"
                )
            
            if is_gibberish:
                return ValidationResult(
                    is_valid=False,
                    category=InputCategory.GIBBERISH,
                    confidence=quality_score,
                    sanitized_input=sanitized,
                    detected_entities={},
                    suggestions=[
                        "restaurants in Istanbul",
                        "places to visit",
                        "transportation help",
                        "tourist attractions"
                    ],
                    error_message="Please enter a clear question about Istanbul"
                )
            
            # Step 4: Category classification
            category, confidence = self.categorize_input(sanitized)
            
            # Step 5: Entity extraction
            entities = self.extract_entities(sanitized)
            
            # Step 6: Generate suggestions
            suggestions = self.generate_suggestions(category, entities)
            
            return ValidationResult(
                is_valid=True,
                category=category,
                confidence=confidence,
                sanitized_input=sanitized,
                detected_entities=entities,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error in input validation: {e}")
            return ValidationResult(
                is_valid=False,
                category=InputCategory.UNKNOWN,
                confidence=0.0,
                sanitized_input=raw_input[:100],  # Truncated safe version
                detected_entities={},
                suggestions=["Please try asking about Istanbul travel information"],
                error_message="Processing error occurred"
            )

# Global validator instance
input_validator = InputValidator()

def validate_user_input(raw_input: str) -> ValidationResult:
    """Convenience function for validating user input"""
    return input_validator.validate_input(raw_input)

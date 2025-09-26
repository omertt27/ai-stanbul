"""
Advanced Language Processing Module for AI Istanbul Travel Guide

This module provides sophisticated natural language processing capabilities
including intent recognition, entity extraction, sentiment analysis, and
contextual understanding specifically tailored for travel-related queries.
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from thefuzz import fuzz, process


@dataclass
class IntentResult:
    """Result of intent recognition"""
    intent: str = ""
    confidence: float = 0.0
    entities: Dict[str, List[str]] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityExtractionResult:
    """Result of entity extraction"""
    locations: List[str] = field(default_factory=list)
    cuisines: List[str] = field(default_factory=list)
    time_references: List[str] = field(default_factory=list)
    budget_indicators: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    numbers: List[int] = field(default_factory=list)
    sentiment: str = "neutral"  # positive, negative, neutral


class AdvancedLanguageProcessor:
    """Advanced language processing for travel queries"""
    
    def __init__(self):
        self.load_knowledge_base()
        
    def load_knowledge_base(self):
        """Load Istanbul-specific knowledge base"""
        self.istanbul_districts = [
            'sultanahmet', 'beyoğlu', 'kadıköy', 'beşiktaş', 'şişli', 'fatih',
            'eminönü', 'galata', 'taksim', 'ortaköy', 'bebek', 'üsküdar',
            'karaköy', 'cihangir', 'balat', 'fener', 'eyüp', 'sarıyer'
        ]
        
        self.landmarks = [
            'hagia sophia', 'blue mosque', 'topkapi palace', 'grand bazaar',
            'galata tower', 'bosphorus bridge', 'maiden\'s tower', 'basilica cistern',
            'spice bazaar', 'dolmabahçe palace', 'çamlıca tower', 'rumeli fortress'
        ]
        
        self.cuisine_types = [
            'turkish', 'ottoman', 'kebab', 'seafood', 'mediterranean', 'asian',
            'italian', 'french', 'international', 'vegetarian', 'vegan', 'halal',
            'street food', 'fine dining', 'traditional', 'modern'
        ]
        
        self.time_indicators = {
            'morning': ['morning', 'breakfast', 'early', 'sunrise', 'am'],
            'afternoon': ['afternoon', 'lunch', 'midday', 'noon', 'pm'],
            'evening': ['evening', 'dinner', 'sunset', 'dusk'],
            'night': ['night', 'late', 'nightlife', 'club', 'bar']
        }
        
        self.budget_indicators = {
            'budget': ['cheap', 'budget', 'affordable', 'inexpensive', 'economical'],
            'mid-range': ['moderate', 'average', 'reasonable', 'medium'],
            'luxury': ['expensive', 'luxury', 'high-end', 'premium', 'exclusive', 'upscale']
        }
        
        self.intent_patterns = {
            'restaurant_search': [
                r'\b(restaurant|food|eat|dining|cuisine)\b',
                r'\b(hungry|meal|dinner|lunch|breakfast)\b',
                r'\b(taste|flavor|dish|menu)\b'
            ],
            'attraction_search': [
                r'\b(visit|see|attraction|museum|palace|mosque)\b',
                r'\b(sightseeing|tour|explore|discover)\b',
                r'\b(historic|cultural|architecture)\b'
            ],
            'transportation': [
                r'\b(how to get|transport|metro|bus|taxi|ferry)\b',
                r'\b(travel to|go to|reach|arrive)\b',
                r'\b(direction|route|way)\b'
            ],
            'accommodation': [
                r'\b(hotel|stay|accommodation|lodge)\b',
                r'\b(room|booking|reservation)\b',
                r'\b(sleep|overnight)\b'
            ],
            'shopping': [
                r'\b(shop|shopping|buy|purchase|market|bazaar)\b',
                r'\b(souvenir|gift|store)\b',
                r'\b(price|cost|bargain)\b'
            ],
            'nightlife': [
                r'\b(nightlife|bar|club|drink|party)\b',
                r'\b(entertainment|music|dance)\b',
                r'\b(evening|night)\b'
            ],
            'weather_clothing': [
                r'\b(weather|temperature|rain|sunny|cold|warm)\b',
                r'\b(what to wear|clothing|dress)\b',
                r'\b(climate|season)\b'
            ],
            'general_info': [
                r'\b(information|help|guide|advice)\b',
                r'\b(tell me|what is|explain)\b',
                r'\b(about|istanbul|turkey)\b'
            ]
        }
        
        self.sentiment_patterns = {
            'positive': [
                r'\b(love|like|enjoy|amazing|beautiful|wonderful|great|excellent|fantastic)\b',
                r'\b(recommend|suggest|prefer|interested|excited)\b',
                r'\b(good|nice|awesome|perfect|brilliant)\b'
            ],
            'negative': [
                r'\b(hate|dislike|avoid|terrible|awful|bad|horrible|disappointing)\b',
                r'\b(not interested|don\'t like|can\'t stand)\b',
                r'\b(boring|expensive|crowded|tourist trap)\b'
            ]
        }

    def recognize_intent(self, text: str, context: Optional[Dict] = None) -> IntentResult:
        """Recognize intent from user input with context awareness"""
        text_lower = text.lower()
        intent_scores = {}
        
        # Score each intent based on pattern matching
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 10  # Base score for each match
            
            # Boost score based on context
            if context:
                if intent == context.get('last_intent') and score > 0:
                    score *= 1.3  # Boost if continuing same topic
                
                if intent in context.get('conversation_history', []):
                    score *= 1.1  # Small boost for previously discussed topics
            
            intent_scores[intent] = score
        
        # Find best intent
        if not intent_scores or max(intent_scores.values()) == 0:
            best_intent = 'general_info'
            confidence = 0.1
        else:
            best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
            max_score = intent_scores[best_intent]
            total_score = sum(intent_scores.values())
            confidence = min(max_score / (total_score + 10), 1.0)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Build context
        result_context = {
            'intent_scores': intent_scores,
            'text_length': len(text),
            'question_words': self._extract_question_words(text),
            'has_location': len(entities.locations) > 0,
            'has_time': len(entities.time_references) > 0,
            'sentiment': entities.sentiment
        }
        
        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            entities=entities.__dict__,
            context=result_context
        )

    def extract_entities(self, text: str) -> EntityExtractionResult:
        """Extract various entities from text"""
        text_lower = text.lower()
        
        # Extract locations (districts and landmarks)
        locations = []
        for location in self.istanbul_districts + self.landmarks:
            if location.lower() in text_lower:
                locations.append(location)
        
        # Use fuzzy matching for partial matches
        words = text_lower.split()
        for word in words:
            if len(word) > 3:  # Only check longer words
                # Check districts
                best_district = process.extractOne(word, self.istanbul_districts, score_cutoff=80)
                if best_district and best_district[0] not in locations:
                    locations.append(best_district[0])
                
                # Check landmarks
                best_landmark = process.extractOne(word, self.landmarks, score_cutoff=80)
                if best_landmark and best_landmark[0] not in locations:
                    locations.append(best_landmark[0])
        
        # Extract cuisines
        cuisines = []
        for cuisine in self.cuisine_types:
            if cuisine.lower() in text_lower:
                cuisines.append(cuisine)
        
        # Extract time references
        time_references = []
        for time_period, indicators in self.time_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    time_references.append(time_period)
                    break
        
        # Extract budget indicators
        budget_indicators = []
        for budget_level, indicators in self.budget_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    budget_indicators.append(budget_level)
                    break
        
        # Extract interests (based on keywords)
        interests = []
        interest_keywords = {
            'history': ['history', 'historic', 'ancient', 'old', 'traditional', 'heritage'],
            'culture': ['culture', 'cultural', 'art', 'museum', 'gallery', 'exhibition'],
            'architecture': ['architecture', 'building', 'structure', 'design', 'palace', 'mosque'],
            'food': ['food', 'cuisine', 'restaurant', 'eat', 'taste', 'culinary'],
            'shopping': ['shopping', 'shop', 'market', 'bazaar', 'buy', 'purchase'],
            'nightlife': ['nightlife', 'bar', 'club', 'entertainment', 'party', 'drink'],
            'nature': ['nature', 'park', 'garden', 'outdoor', 'scenic', 'view'],
            'photography': ['photo', 'photography', 'picture', 'instagram', 'scenic']
        }
        
        for interest, keywords in interest_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                interests.append(interest)
        
        # Extract numbers
        numbers = []
        number_pattern = r'\b\d+\b'
        number_matches = re.findall(number_pattern, text)
        numbers = [int(match) for match in number_matches]
        
        # Determine sentiment
        sentiment = self._analyze_sentiment(text_lower)
        
        return EntityExtractionResult(
            locations=list(set(locations)),
            cuisines=list(set(cuisines)),
            time_references=list(set(time_references)),
            budget_indicators=list(set(budget_indicators)),
            interests=list(set(interests)),
            numbers=numbers,
            sentiment=sentiment
        )

    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of the text"""
        positive_score = 0
        negative_score = 0
        
        for patterns in self.sentiment_patterns['positive']:
            positive_score += len(re.findall(patterns, text))
        
        for patterns in self.sentiment_patterns['negative']:
            negative_score += len(re.findall(patterns, text))
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'

    def _extract_question_words(self, text: str) -> List[str]:
        """Extract question words from text"""
        question_words = ['what', 'where', 'when', 'why', 'how', 'which', 'who']
        found_words = []
        text_lower = text.lower()
        
        for word in question_words:
            if word in text_lower:
                found_words.append(word)
        
        return found_words

    def is_followup_question(self, text: str, previous_context: Optional[Dict] = None) -> bool:
        """Determine if this is a follow-up question"""
        followup_indicators = [
            r'\bwhat about\b', r'\bany others?\b', r'\bmore\b', r'\balso\b',
            r'\badditionally\b', r'\bbesides\b', r'\belse\b', r'\bthere\b',
            r'\bnear there\b', r'\bin that area\b', r'\baround there\b',
            r'\bsimilar\b', r'\blike that\b', r'\band\b.*\?'
        ]
        
        text_lower = text.lower()
        has_followup_indicator = any(re.search(pattern, text_lower) for pattern in followup_indicators)
        
        # Check if it's a short query (likely followup)
        is_short = len(text.split()) <= 3
        
        # Check if it references previous context
        has_reference = False
        if previous_context and 'locations' in previous_context:
            for location in previous_context['locations']:
                if 'there' in text_lower or 'that area' in text_lower:
                    has_reference = True
                    break
        
        return has_followup_indicator or (is_short and has_reference)

    def generate_clarification_questions(self, intent: str, entities: Dict) -> List[str]:
        """Generate clarification questions based on missing information"""
        questions = []
        
        if intent == 'restaurant_search':
            if not entities.get('cuisines'):
                questions.append("What type of cuisine are you interested in?")
            if not entities.get('budget_indicators'):
                questions.append("What's your budget preference - budget-friendly, mid-range, or luxury?")
            if not entities.get('locations'):
                questions.append("Which area of Istanbul would you like to dine in?")
        
        elif intent == 'attraction_search':
            if not entities.get('interests'):
                questions.append("What type of attractions interest you most - historic sites, museums, or cultural experiences?")
            if not entities.get('time_references'):
                questions.append("What time of day are you planning to visit?")
            if not entities.get('locations'):
                questions.append("Which district would you like to explore?")
        
        elif intent == 'transportation':
            if not entities.get('locations') or len(entities.get('locations', [])) < 2:
                questions.append("Where would you like to travel from and to?")
        
        return questions[:2]  # Limit to 2 questions to avoid overwhelming

    def enhance_query_understanding(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Comprehensive query understanding"""
        intent_result = self.recognize_intent(text, context)
        entities = self.extract_entities(text)
        
        # Check for ambiguity
        ambiguity_score = self._calculate_ambiguity(text, intent_result)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(intent_result.intent, entities)
        
        # Generate clarification questions if needed
        clarifications = []
        if intent_result.confidence < 0.7 or ambiguity_score > 0.5:
            clarifications = self.generate_clarification_questions(intent_result.intent, entities.__dict__)
        
        return {
            'intent': intent_result.intent,
            'confidence': intent_result.confidence,
            'entities': entities.__dict__,
            'ambiguity_score': ambiguity_score,
            'suggestions': suggestions,
            'clarifications': clarifications,
            'is_followup': self.is_followup_question(text, context),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }

    def _calculate_ambiguity(self, text: str, intent_result: IntentResult) -> float:
        """Calculate how ambiguous the query is"""
        # Multiple high-scoring intents indicate ambiguity
        intent_scores = intent_result.context.get('intent_scores', {})
        if not intent_scores:
            return 1.0
        
        sorted_scores = sorted(intent_scores.values(), reverse=True)
        if len(sorted_scores) < 2:
            return 0.0
        
        # If top two scores are close, it's ambiguous
        top_score = sorted_scores[0]
        second_score = sorted_scores[1]
        
        if top_score == 0:
            return 1.0
        
        ambiguity = second_score / top_score
        return min(ambiguity, 1.0)

    def _generate_suggestions(self, intent: str, entities: EntityExtractionResult) -> List[str]:
        """Generate helpful suggestions based on intent and entities"""
        suggestions = []
        
        if intent == 'restaurant_search':
            if entities.locations:
                suggestions.append(f"Popular restaurants in {entities.locations[0]}")
            if entities.cuisines:
                suggestions.append(f"Best {entities.cuisines[0]} restaurants")
            suggestions.append("Restaurants with outdoor seating")
            suggestions.append("Traditional Turkish cuisine recommendations")
        
        elif intent == 'attraction_search':
            suggestions.append("Must-see historic attractions")
            suggestions.append("Free attractions in Istanbul")
            suggestions.append("Family-friendly activities")
            if entities.locations:
                suggestions.append(f"Hidden gems in {entities.locations[0]}")
        
        elif intent == 'transportation':
            suggestions.append("Metro and public transport guide")
            suggestions.append("Best taxi apps in Istanbul")
            suggestions.append("Ferry routes and schedules")
        
        return suggestions[:3]  # Limit suggestions


# Global instance for easy access
language_processor = AdvancedLanguageProcessor()


def process_user_query(text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience function to process user queries"""
    return language_processor.enhance_query_understanding(text, context)


def extract_intent_and_entities(text: str) -> Tuple[str, Dict]:
    """Quick function to get intent and entities"""
    result = language_processor.recognize_intent(text)
    return result.intent, result.entities


def is_followup(text: str, context: Optional[Dict] = None) -> bool:
    """Quick function to check if query is a followup"""
    return language_processor.is_followup_question(text, context)

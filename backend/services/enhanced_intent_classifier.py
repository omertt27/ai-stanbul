"""
Enhanced Intent Classification System for AI Istanbul
Advanced intent detection with confidence thresholds and session context
Improves query routing accuracy and reduces GPT dependency through better understanding
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import numpy as np

# ML libraries for advanced classification
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Enhanced intent types for Istanbul tourism"""
    # Primary intents
    TRANSPORTATION = "transportation"
    FOOD_DINING = "food_dining" 
    ATTRACTIONS = "attractions"
    PRACTICAL_INFO = "practical_info"
    SHOPPING = "shopping"
    ACCOMMODATION = "accommodation"
    
    # Secondary intents
    AREA_EXPLORATION = "area_exploration"
    CULTURAL_ACTIVITIES = "cultural_activities"
    NIGHTLIFE = "nightlife"
    OUTDOOR_ACTIVITIES = "outdoor_activities"
    PHOTOGRAPHY = "photography"
    
    # Meta intents
    PLANNING = "planning"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    COMPLAINT = "complaint"
    GREETING = "greeting"
    GENERAL = "general"

@dataclass
class IntentPrediction:
    """Intent prediction with confidence and context"""
    intent: IntentType
    confidence: float
    sub_intents: List[Tuple[str, float]] = field(default_factory=list)
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
    context_factors: Dict[str, Any] = field(default_factory=dict)
    fallback_reason: Optional[str] = None

@dataclass
class SessionContext:
    """Session context for intent classification"""
    user_id: str
    session_id: str
    conversation_history: List[Dict] = field(default_factory=list)
    current_topic: Optional[str] = None
    user_location: Optional[str] = None
    time_of_day: Optional[str] = None
    session_duration: timedelta = field(default_factory=lambda: timedelta(0))
    interaction_count: int = 0
    last_intent: Optional[IntentType] = None
    context_entities: Set[str] = field(default_factory=set)

@dataclass
class IntentResult:
    """Intent classification result for production integration"""
    primary_intent: IntentType
    secondary_intents: List[IntentType] = field(default_factory=list)
    confidence: float = 0.0
    entities: Dict[str, List[str]] = field(default_factory=dict)
    context_factors: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    
    @classmethod
    def from_prediction(cls, prediction: IntentPrediction):
        """Create IntentResult from IntentPrediction"""
        return cls(
            primary_intent=prediction.intent,
            confidence=prediction.confidence,
            entities=prediction.extracted_entities,
            context_factors=prediction.context_factors
        )

class EnhancedIntentClassifier:
    """
    Advanced intent classification system with confidence thresholds
    and context awareness for improved query routing
    """
    
    def __init__(self, confidence_threshold: float = 0.75):
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = 0.4
        
        # Intent patterns with weights
        self.intent_patterns = self._initialize_intent_patterns()
        
        # Entity extraction patterns
        self.entity_patterns = self._initialize_entity_patterns()
        
        # Context keywords for better understanding
        self.context_keywords = self._initialize_context_keywords()
        
        # ML classifier (if available)
        self.ml_classifier = None
        self.tfidf_vectorizer = None
        
        # Training data for ML classifier
        self.training_data = self._load_training_data()
        
        # Session management
        self.active_sessions: Dict[str, SessionContext] = {}
        
        # Performance tracking
        self.classification_stats = {
            'total_classifications': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'fallback_used': 0,
            'context_enhanced': 0
        }
        
        # Initialize ML components
        if SKLEARN_AVAILABLE:
            self._train_ml_classifier()
    
    def _initialize_intent_patterns(self) -> Dict[IntentType, Dict[str, List[Tuple[str, float]]]]:
        """Initialize intent detection patterns with confidence weights"""
        return {
            IntentType.TRANSPORTATION: {
                'primary': [
                    (r'\b(how to get|how do i get|way to get|directions to)\b', 0.9),
                    (r'\b(metro|subway|tram|bus|ferry|taxi|uber|dolmus)\b', 0.8),
                    (r'\b(transport|transportation|travel to|go to)\b', 0.8),
                    (r'\b(from .* to|between .* and)\b', 0.7),
                    (r'\b(M1|M2|M3|M4|T1|T2|Line)\b', 0.9)
                ],
                'secondary': [
                    (r'\b(distance|duration|time|cost|price)\b', 0.4),
                    (r'\b(fastest|cheapest|quickest)\b', 0.5),
                    (r'\b(schedule|timetable|frequency)\b', 0.6)
                ]
            },
            
            IntentType.FOOD_DINING: {
                'primary': [
                    (r'\b(restaurant|eat|food|dining|meal)\b', 0.8),
                    (r'\b(breakfast|lunch|dinner|brunch)\b', 0.8),
                    (r'\b(turkish food|local cuisine|traditional food)\b', 0.9),
                    (r'\b(kebab|baklava|meze|dolma|turkish breakfast)\b', 0.9),
                    (r'\b(where to eat|good food|best restaurant)\b', 0.8)
                ],
                'secondary': [
                    (r'\b(vegetarian|vegan|halal|kosher)\b', 0.6),
                    (r'\b(budget|expensive|cheap|fine dining)\b', 0.5),
                    (r'\b(reservation|booking|table)\b', 0.7),
                    (r'\b(menu|price|cost)\b', 0.4)
                ]
            },
            
            IntentType.ATTRACTIONS: {
                'primary': [
                    (r'\b(hagia sophia|blue mosque|topkapi|galata tower|grand bazaar)\b', 0.9),
                    (r'\b(museum|palace|mosque|church|tower)\b', 0.8),
                    (r'\b(visit|see|attraction|tourist|sightseeing)\b', 0.7),
                    (r'\b(historical|ancient|byzantine|ottoman)\b', 0.8)
                ],
                'secondary': [
                    (r'\b(photo|picture|instagram|photography)\b', 0.5),
                    (r'\b(free|paid|entrance fee)\b', 0.4),
                    (r'\b(guided tour|audio guide)\b', 0.6)
                ]
            },
            
            IntentType.PRACTICAL_INFO: {
                'primary': [
                    (r'\b(opening hours|open|close|schedule)\b', 0.8),
                    (r'\b(price|cost|ticket|fee|admission)\b', 0.8),
                    (r'\b(when|what time|hours)\b', 0.6),
                    (r'\b(how much|how long|duration)\b', 0.7),
                    (r'\b(closed|holiday|weekend)\b', 0.7)
                ],
                'secondary': [
                    (r'\b(contact|phone|website|address)\b', 0.6),
                    (r'\b(book|reserve|advance|online)\b', 0.5),
                    (r'\b(discount|student|senior)\b', 0.5)
                ]
            },
            
            IntentType.SHOPPING: {
                'primary': [
                    (r'\b(shop|shopping|buy|purchase|store)\b', 0.8),
                    (r'\b(souvenir|gift|memento)\b', 0.9),
                    (r'\b(market|bazaar|mall|shopping center)\b', 0.8),
                    (r'\b(carpet|jewelry|ceramic|turkish delight)\b', 0.9)
                ],
                'secondary': [
                    (r'\b(bargain|negotiate|price|haggle)\b', 0.6),
                    (r'\b(authentic|genuine|fake|replica)\b', 0.5),
                    (r'\b(tax free|duty free)\b', 0.7)
                ]
            },
            
            IntentType.AREA_EXPLORATION: {
                'primary': [
                    (r'\b(sultanahmet|beyoglu|galata|karakoy|besiktas|kadikoy|uskudar|taksim)\b', 0.8),
                    (r'\b(district|area|neighborhood|quarter)\b', 0.7),
                    (r'\b(explore|walk around|wander|stroll)\b', 0.8),
                    (r'\b(what to see|things to do|activities)\b', 0.7)
                ],
                'secondary': [
                    (r'\b(safe|dangerous|avoid|recommended)\b', 0.5),
                    (r'\b(local|authentic|tourist|crowded)\b', 0.4)
                ]
            },
            
            IntentType.PLANNING: {
                'primary': [
                    (r'\b(plan|planning|itinerary|schedule)\b', 0.8),
                    (r'\b(day trip|half day|full day|weekend)\b', 0.8),
                    (r'\b(first time|visit|trip|vacation)\b', 0.6),
                    (r'\b(suggest|recommend|advice|help)\b', 0.6)
                ],
                'secondary': [
                    (r'\b(priority|must see|important|essential)\b', 0.5),
                    (r'\b(time|duration|hours|days)\b', 0.4)
                ]
            }
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, List[Tuple[str, float]]]:
        """Initialize entity extraction patterns"""
        return {
            'locations': [
                (r'\b(hagia sophia|ayasofya)\b', 0.9),
                (r'\b(blue mosque|sultanahmet mosque|sultanahmet camii)\b', 0.9),
                (r'\b(topkapi palace|topkapi sarayi)\b', 0.9),
                (r'\b(galata tower|galata kulesi)\b', 0.9),
                (r'\b(grand bazaar|kapali carsi|covered bazaar)\b', 0.9),
                (r'\b(bosphorus|bosphorus bridge|bogaz)\b', 0.9),
                (r'\b(taksim square|taksim)\b', 0.8),
                (r'\b(istiklal street|istiklal caddesi)\b', 0.8),
                (r'\b(sultanahmet|eminonu|beyoglu|galata|karakoy|besiktas|kadikoy|uskudar)\b', 0.8)
            ],
            'transport': [
                (r'\b(M1|M2|M3|M4|M5|M6|M7)\b', 0.9),
                (r'\b(T1|T2|T3|T4|T5)\b', 0.9),
                (r'\b(metro|tram|bus|ferry|funicular|taxi|uber|dolmus)\b', 0.8),
                (r'\b(istanbulkart|akbil|bilet)\b', 0.8)
            ],
            'time': [
                (r'\b(\d{1,2}:\d{2}|\d{1,2} ?(am|pm|AM|PM))\b', 0.8),
                (r'\b(morning|afternoon|evening|night|dawn|dusk)\b', 0.6),
                (r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 0.7),
                (r'\b(weekend|weekday|today|tomorrow|yesterday)\b', 0.7)
            ],
            'food': [
                (r'\b(kebab|döner|iskender|adana|urfa)\b', 0.9),
                (r'\b(baklava|turkish delight|lokum|halva)\b', 0.9),
                (r'\b(turkish breakfast|kahvalti|meze|dolma|sarma)\b', 0.9),
                (r'\b(turkish tea|cay|turkish coffee|kahve)\b', 0.8),
                (r'\b(vegetarian|vegan|halal|gluten.free)\b', 0.7)
            ],
            'budget': [
                (r'\b(budget|cheap|inexpensive|affordable)\b', 0.8),
                (r'\b(expensive|luxury|high.end|premium)\b', 0.8),
                (r'\b(\d+\s?(tl|lira|turkish lira|euro|dollar|usd))\b', 0.9)
            ]
        }
    
    def _initialize_context_keywords(self) -> Dict[str, List[str]]:
        """Initialize context keywords for better understanding"""
        return {
            'urgency': ['urgent', 'asap', 'quickly', 'fast', 'hurry', 'immediate'],
            'preference': ['prefer', 'like', 'love', 'enjoy', 'interested', 'favorite'],
            'comparison': ['better', 'best', 'compare', 'versus', 'vs', 'difference', 'which'],
            'planning': ['plan', 'organize', 'schedule', 'arrange', 'prepare', 'book'],
            'experience': ['experience', 'feeling', 'atmosphere', 'vibe', 'ambiance'],
            'group': ['family', 'kids', 'children', 'couple', 'friends', 'group', 'solo', 'alone'],
            'accessibility': ['wheelchair', 'disabled', 'accessible', 'mobility', 'elevator'],
            'weather': ['rain', 'sunny', 'cold', 'hot', 'weather', 'indoor', 'outdoor']
        }
    
    def _load_training_data(self) -> List[Tuple[str, str]]:
        """Load training data for ML classifier"""
        return [
            # Transportation examples
            ("How do I get to Hagia Sophia from Taksim?", "transportation"),
            ("What's the best way to reach Blue Mosque?", "transportation"),
            ("Metro route to Galata Tower", "transportation"),
            ("Bus from Sultanahmet to Beyoglu", "transportation"),
            ("Taxi cost from airport to city center", "transportation"),
            
            # Food examples
            ("Best Turkish restaurants in Sultanahmet", "food_dining"),
            ("Where can I find good kebab?", "food_dining"),
            ("Traditional Turkish breakfast places", "food_dining"),
            ("Vegetarian restaurants near Galata Tower", "food_dining"),
            ("Turkish coffee shops recommendation", "food_dining"),
            
            # Attractions examples
            ("What time does Hagia Sophia open?", "practical_info"),
            ("Blue Mosque entrance fee", "practical_info"),
            ("Topkapi Palace visiting hours", "practical_info"),
            ("Is Grand Bazaar open on Sunday?", "practical_info"),
            
            # Shopping examples
            ("Where to buy Turkish carpets?", "shopping"),
            ("Best souvenir shops in Istanbul", "shopping"),
            ("Grand Bazaar shopping guide", "shopping"),
            ("Authentic Turkish delight stores", "shopping"),
            
            # Area exploration examples
            ("What to see in Beyoglu district?", "area_exploration"),
            ("Things to do in Sultanahmet", "area_exploration"),
            ("Explore Karakoy neighborhood", "area_exploration"),
            ("Walking tour of Galata area", "area_exploration"),
            
            # Planning examples
            ("Plan a day in Istanbul", "planning"),
            ("First time visitor itinerary", "planning"),
            ("What should I prioritize in Istanbul?", "planning"),
            ("Help me organize my trip", "planning")
        ]
    
    def _train_ml_classifier(self):
        """Train ML classifier if sklearn is available"""
        if not SKLEARN_AVAILABLE or not self.training_data:
            return
        
        try:
            texts, labels = zip(*self.training_data)
            
            # Create pipeline with TF-IDF and classifier
            self.ml_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 3),
                    stop_words='english',
                    lowercase=True
                )),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
            
            # Train the classifier
            self.ml_classifier.fit(texts, labels)
            
            logger.info("✅ ML intent classifier trained successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not train ML classifier: {e}")
            self.ml_classifier = None
    
    def classify_intent(self, query: str, session_context: SessionContext = None,
                       user_context: Dict = None) -> IntentPrediction:
        """
        Main intent classification method with confidence thresholds
        """
        self.classification_stats['total_classifications'] += 1
        
        # Normalize query
        query_normalized = self._normalize_query(query)
        
        # Rule-based classification
        rule_prediction = self._classify_with_rules(query_normalized)
        
        # ML-based classification (if available)
        ml_prediction = self._classify_with_ml(query_normalized) if self.ml_classifier else None
        
        # Combine predictions
        combined_prediction = self._combine_predictions(rule_prediction, ml_prediction)
        
        # Apply session context
        if session_context:
            combined_prediction = self._apply_session_context(combined_prediction, session_context, query)
            self.classification_stats['context_enhanced'] += 1
        
        # Apply user context
        if user_context:
            combined_prediction = self._apply_user_context(combined_prediction, user_context, query)
        
        # Extract entities
        combined_prediction.extracted_entities = self._extract_entities(query_normalized)
        
        # Determine final confidence and fallback
        final_prediction = self._apply_confidence_thresholds(combined_prediction)
        
        # Update statistics
        self._update_classification_stats(final_prediction)
        
        return final_prediction
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for better pattern matching"""
        normalized = query.lower().strip()
        
        # Turkish character replacements
        replacements = {
            'ç': 'c', 'ğ': 'g', 'ı': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
            'ayasofya': 'hagia sophia',
            'sultanahmet camii': 'blue mosque',
            'kapalı çarşı': 'grand bazaar',
            'topkapı sarayı': 'topkapi palace',
            'galata kulesi': 'galata tower'
        }
        
        for turkish, english in replacements.items():
            normalized = normalized.replace(turkish, english)
        
        return normalized
    
    def _classify_with_rules(self, query: str) -> IntentPrediction:
        """Rule-based intent classification"""
        intent_scores = defaultdict(float)
        matched_patterns = []
        
        for intent_type, pattern_groups in self.intent_patterns.items():
            total_score = 0
            
            for group, patterns in pattern_groups.items():
                group_score = 0
                weight = 1.0 if group == 'primary' else 0.5
                
                for pattern, confidence in patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        group_score = max(group_score, confidence)
                        matched_patterns.append((pattern, confidence, intent_type))
                
                total_score += group_score * weight
            
            if total_score > 0:
                intent_scores[intent_type] = min(1.0, total_score)
        
        # Find best intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            best_confidence = intent_scores[best_intent]
        else:
            best_intent = IntentType.GENERAL
            best_confidence = 0.3
        
        # Create sub-intents list
        sub_intents = [(intent.value, score) for intent, score in intent_scores.items() if score > 0.2]
        sub_intents.sort(key=lambda x: x[1], reverse=True)
        
        return IntentPrediction(
            intent=best_intent,
            confidence=best_confidence,
            sub_intents=sub_intents,
            context_factors={'method': 'rule_based', 'matched_patterns': len(matched_patterns)}
        )
    
    def _classify_with_ml(self, query: str) -> Optional[IntentPrediction]:
        """ML-based intent classification"""
        if not self.ml_classifier:
            return None
        
        try:
            # Get prediction probabilities
            probabilities = self.ml_classifier.predict_proba([query])[0]
            classes = self.ml_classifier.classes_
            
            # Find best prediction
            best_idx = np.argmax(probabilities)
            best_intent_str = classes[best_idx]
            best_confidence = probabilities[best_idx]
            
            # Convert string to IntentType
            try:
                best_intent = IntentType(best_intent_str)
            except ValueError:
                best_intent = IntentType.GENERAL
            
            # Create sub-intents
            sub_intents = [(classes[i], prob) for i, prob in enumerate(probabilities) if prob > 0.1]
            sub_intents.sort(key=lambda x: x[1], reverse=True)
            
            return IntentPrediction(
                intent=best_intent,
                confidence=best_confidence,
                sub_intents=sub_intents,
                context_factors={'method': 'ml_based', 'model_confidence': best_confidence}
            )
            
        except Exception as e:
            logger.warning(f"ML classification failed: {e}")
            return None
    
    def _combine_predictions(self, rule_pred: IntentPrediction, 
                           ml_pred: Optional[IntentPrediction]) -> IntentPrediction:
        """Combine rule-based and ML predictions"""
        if not ml_pred:
            return rule_pred
        
        # Weight the predictions (rule-based gets slight preference for Istanbul tourism)
        rule_weight = 0.6
        ml_weight = 0.4
        
        # If both predict the same intent, boost confidence
        if rule_pred.intent == ml_pred.intent:
            combined_confidence = min(1.0, (rule_pred.confidence * rule_weight + 
                                          ml_pred.confidence * ml_weight) * 1.2)
            final_intent = rule_pred.intent
        else:
            # Use the prediction with higher weighted confidence
            rule_weighted = rule_pred.confidence * rule_weight
            ml_weighted = ml_pred.confidence * ml_weight
            
            if rule_weighted >= ml_weighted:
                final_intent = rule_pred.intent
                combined_confidence = rule_pred.confidence
            else:
                final_intent = ml_pred.intent
                combined_confidence = ml_pred.confidence
        
        # Combine sub-intents
        combined_sub_intents = list(rule_pred.sub_intents)
        for intent, score in ml_pred.sub_intents:
            combined_sub_intents.append((intent, score * ml_weight))
        
        # Remove duplicates and sort
        intent_scores = defaultdict(float)
        for intent, score in combined_sub_intents:
            intent_scores[intent] = max(intent_scores[intent], score)
        
        final_sub_intents = list(intent_scores.items())
        final_sub_intents.sort(key=lambda x: x[1], reverse=True)
        
        return IntentPrediction(
            intent=final_intent,
            confidence=combined_confidence,
            sub_intents=final_sub_intents[:5],
            context_factors={
                'method': 'combined',
                'rule_confidence': rule_pred.confidence,
                'ml_confidence': ml_pred.confidence if ml_pred else 0.0
            }
        )
    
    def _apply_session_context(self, prediction: IntentPrediction, 
                             session_context: SessionContext, query: str) -> IntentPrediction:
        """Apply session context to enhance prediction"""
        
        # Context continuity boost
        if session_context.last_intent and session_context.last_intent == prediction.intent:
            prediction.confidence = min(1.0, prediction.confidence + 0.1)
            prediction.context_factors['context_continuity'] = True
        
        # Topic consistency check
        if session_context.current_topic:
            topic_related = self._is_topic_related(query, session_context.current_topic)
            if topic_related:
                prediction.confidence = min(1.0, prediction.confidence + 0.05)
                prediction.context_factors['topic_consistency'] = True
        
        # Entity context enhancement
        query_entities = set(self._extract_entities(query).get('locations', []))
        context_overlap = len(query_entities.intersection(session_context.context_entities))
        if context_overlap > 0:
            prediction.confidence = min(1.0, prediction.confidence + (context_overlap * 0.05))
            prediction.context_factors['entity_overlap'] = context_overlap
        
        # Update session context
        session_context.context_entities.update(query_entities)
        session_context.last_intent = prediction.intent
        session_context.interaction_count += 1
        
        return prediction
    
    def _apply_user_context(self, prediction: IntentPrediction, 
                          user_context: Dict, query: str) -> IntentPrediction:
        """Apply user context for personalization"""
        
        # User preference alignment
        user_preferences = user_context.get('user_preferences', {})
        
        # Interest-based confidence boost
        if 'interests' in user_preferences:
            for interest_pref in user_preferences['interests']:
                interest = interest_pref['value']
                pref_confidence = interest_pref['confidence']
                
                if self._intent_matches_interest(prediction.intent, interest):
                    boost = pref_confidence * 0.1
                    prediction.confidence = min(1.0, prediction.confidence + boost)
                    prediction.context_factors['interest_alignment'] = interest
        
        # Behavioral pattern consideration
        behavioral_hints = user_context.get('behavioral_hints', {})
        if behavioral_hints.get('is_frequent_user', False):
            # Frequent users get slight confidence boost for complex queries
            if len(query.split()) > 10:
                prediction.confidence = min(1.0, prediction.confidence + 0.05)
                prediction.context_factors['frequent_user_boost'] = True
        
        return prediction
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from query"""
        entities = defaultdict(list)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern, confidence in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] if match[0] else match[1]
                    if match and match not in entities[entity_type]:
                        entities[entity_type].append(match)
        
        return dict(entities)
    
    def _apply_confidence_thresholds(self, prediction: IntentPrediction) -> IntentPrediction:
        """Apply confidence thresholds and fallback logic"""
        
        if prediction.confidence >= self.confidence_threshold:
            # High confidence - use as is
            pass
        elif prediction.confidence >= self.low_confidence_threshold:
            # Medium confidence - add fallback suggestion
            prediction.fallback_reason = f"Medium confidence ({prediction.confidence:.2f}). Consider template fallback."
        else:
            # Low confidence - suggest fallback
            prediction.fallback_reason = f"Low confidence ({prediction.confidence:.2f}). Recommend smart fallback."
            prediction.intent = IntentType.GENERAL
        
        return prediction
    
    def _is_topic_related(self, query: str, topic: str) -> bool:
        """Check if query is related to current topic"""
        topic_keywords = {
            'transportation': ['get', 'go', 'travel', 'move', 'transport'],
            'food': ['eat', 'food', 'restaurant', 'meal', 'dining'],
            'attractions': ['see', 'visit', 'attraction', 'place', 'site'],
            'shopping': ['buy', 'shop', 'purchase', 'market', 'store']
        }
        
        query_lower = query.lower()
        if topic in topic_keywords:
            return any(keyword in query_lower for keyword in topic_keywords[topic])
        
        return False
    
    def _intent_matches_interest(self, intent: IntentType, interest: str) -> bool:
        """Check if intent matches user interest"""
        interest_intent_mapping = {
            'historical': [IntentType.ATTRACTIONS, IntentType.CULTURAL_ACTIVITIES],
            'food': [IntentType.FOOD_DINING],
            'art': [IntentType.ATTRACTIONS, IntentType.CULTURAL_ACTIVITIES],
            'shopping': [IntentType.SHOPPING],
            'nightlife': [IntentType.NIGHTLIFE],
            'nature': [IntentType.OUTDOOR_ACTIVITIES],
            'photography': [IntentType.PHOTOGRAPHY, IntentType.ATTRACTIONS]
        }
        
        return intent in interest_intent_mapping.get(interest, [])
    
    def _update_classification_stats(self, prediction: IntentPrediction):
        """Update classification statistics"""
        if prediction.confidence >= self.confidence_threshold:
            self.classification_stats['high_confidence'] += 1
        elif prediction.confidence >= self.low_confidence_threshold:
            self.classification_stats['medium_confidence'] += 1
        else:
            self.classification_stats['low_confidence'] += 1
        
        if prediction.fallback_reason:
            self.classification_stats['fallback_used'] += 1
    
    def create_session_context(self, user_id: str, session_id: str = None) -> SessionContext:
        """Create or get session context"""
        if not session_id:
            session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = SessionContext(
                user_id=user_id,
                session_id=session_id
            )
        
        return self.active_sessions[session_id]
    
    def update_session_context(self, session_context: SessionContext, 
                             query: str, intent_prediction: IntentPrediction):
        """Update session context with new interaction"""
        session_context.conversation_history.append({
            'query': query,
            'intent': intent_prediction.intent.value,
            'confidence': intent_prediction.confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(session_context.conversation_history) > 10:
            session_context.conversation_history = session_context.conversation_history[-10:]
        
        # Update current topic
        if intent_prediction.confidence > 0.7:
            session_context.current_topic = intent_prediction.intent.value
        
        # Update entities
        entities = intent_prediction.extracted_entities
        for entity_list in entities.values():
            session_context.context_entities.update(entity_list)
    
    def get_classification_statistics(self) -> Dict:
        """Get classification system statistics"""
        total = self.classification_stats['total_classifications']
        
        if total == 0:
            return self.classification_stats
        
        stats = dict(self.classification_stats)
        stats['high_confidence_rate'] = (stats['high_confidence'] / total) * 100
        stats['medium_confidence_rate'] = (stats['medium_confidence'] / total) * 100
        stats['low_confidence_rate'] = (stats['low_confidence'] / total) * 100
        stats['fallback_rate'] = (stats['fallback_used'] / total) * 100
        stats['context_enhancement_rate'] = (stats['context_enhanced'] / total) * 100
        
        return stats
    
    def should_use_fallback(self, prediction: IntentPrediction) -> Tuple[bool, str]:
        """Determine if fallback should be used"""
        if prediction.confidence < self.low_confidence_threshold:
            return True, "low_confidence"
        
        if prediction.fallback_reason:
            return True, "threshold_based"
        
        if prediction.intent == IntentType.GENERAL and prediction.confidence < 0.6:
            return True, "general_intent_low_confidence"
        
        return False, ""
    
    def get_intent_routing_decision(self, prediction: IntentPrediction) -> Dict[str, Any]:
        """Get routing decision based on intent prediction"""
        use_fallback, fallback_reason = self.should_use_fallback(prediction)
        
        routing_decision = {
            'intent': prediction.intent.value,
            'confidence': prediction.confidence,
            'use_fallback': use_fallback,
            'fallback_reason': fallback_reason,
            'extracted_entities': prediction.extracted_entities,
            'routing_priority': self._get_routing_priority(prediction),
            'recommended_handler': self._get_recommended_handler(prediction)
        }
        
        return routing_decision
    
    def _get_routing_priority(self, prediction: IntentPrediction) -> str:
        """Get routing priority based on confidence and intent"""
        if prediction.confidence >= self.confidence_threshold:
            return "high"
        elif prediction.confidence >= self.low_confidence_threshold:
            return "medium"
        else:
            return "low"
    
    def _get_recommended_handler(self, prediction: IntentPrediction) -> str:
        """Get recommended handler for the intent"""
        handler_mapping = {
            IntentType.TRANSPORTATION: "route_planner",
            IntentType.FOOD_DINING: "restaurant_recommender",
            IntentType.ATTRACTIONS: "attraction_info",
            IntentType.PRACTICAL_INFO: "info_provider",
            IntentType.SHOPPING: "shopping_guide",
            IntentType.AREA_EXPLORATION: "area_explorer",
            IntentType.PLANNING: "trip_planner",
            IntentType.GENERAL: "general_assistant"
        }
        
        return handler_mapping.get(prediction.intent, "general_assistant")

#!/usr/bin/env python3
"""
Fixed Enhanced Neural Intent Classifier
======================================

This version handles missing ML dependencies gracefully and provides
fallback mechanisms for intent classification.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class IntentPrediction:
    """Enhanced intent prediction result"""
    intent: str
    confidence: float
    probability_distribution: Dict[str, float]
    context_influence: Dict[str, float]
    sub_intents: List[str] = field(default_factory=list)
    uncertainty_score: float = 0.0
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class FixedEnhancedNeuralIntentClassifier:
    """Enhanced neural intent classifier with robust fallback mechanisms"""
    
    def __init__(self):
        self.ml_available = False
        self.model = None
        self.tokenizer = None
        
        # Try to import ML dependencies
        self._check_ml_dependencies()
        
        # Intent categories for Istanbul tourism
        self.intent_categories = [
            'restaurant_search',
            'restaurant_info', 
            'attraction_search',
            'attraction_info',
            'transport_route',
            'transport_info',
            'hotel_search',
            'hotel_info',
            'general_info',
            'practical_info',
            'recommendation_request',
            'comparison_request',
            'reservation_request',
            'navigation_help',
            'cultural_inquiry',
            'historical_question',
            'weather_inquiry',
            'event_search',
            'shopping_inquiry',
            'nightlife_search',
            'emergency_help',
            'language_help',
            'photo_request',
            'price_inquiry',
            'time_schedule',
            'greeting',
            'goodbye',
            'emotional_expression',
            'unknown'
        ]
        
        # Pattern-based intent classification rules
        self.intent_patterns = {
            'restaurant_search': [
                r'restaurant|dining|food|eat|cuisine|kebab|turkish food',
                r'where.*eat|find.*restaurant|good.*food|best.*restaurant',
                r'hungry|dinner|lunch|breakfast|meal'
            ],
            'attraction_search': [
                r'attraction|tourist|visit|see|sightseeing|monument',
                r'hagia sophia|blue mosque|topkapi|galata tower|bosphorus',
                r'museum|palace|church|mosque|historical'
            ],
            'transport_route': [
                r'how to get|transport|metro|bus|taxi|tram',
                r'route|direction|travel|go to|from.*to',
                r'airport|ferry|subway|train'
            ],
            'hotel_search': [
                r'hotel|accommodation|stay|room|booking',
                r'where to stay|place to sleep|hostel|guest house'
            ],
            'price_inquiry': [
                r'price|cost|expensive|cheap|how much|budget',
                r'fee|ticket|entrance|fare'
            ],
            'weather_inquiry': [
                r'weather|temperature|rain|sunny|cloudy|forecast',
                r'what.*weather|how.*weather'
            ],
            'greeting': [
                r'hello|hi|hey|good morning|good afternoon|good evening',
                r'merhaba|selam'
            ],
            'emotional_expression': [
                r'amazing|wonderful|terrible|disappointed|excited|love',
                r'happy|sad|angry|frustrated|pleased'
            ]
        }
        
        if self.ml_available:
            self._initialize_ml_model()
    
    def _check_ml_dependencies(self):
        """Check if ML dependencies are available"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from sklearn.preprocessing import LabelEncoder
            self.ml_available = True
            logger.info("✅ ML dependencies available")
        except ImportError as e:
            self.ml_available = False
            logger.warning(f"⚠️  ML dependencies not available: {e}")
    
    def _initialize_ml_model(self):
        """Initialize ML model if dependencies are available"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from sklearn.preprocessing import LabelEncoder
            
            # Use a lighter model for better compatibility
            model_name = "distilbert-base-uncased"
            
            logger.info(f"🧠 Initializing neural model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.intent_categories),
                problem_type="single_label_classification"
            )
            
            # Set up label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.intent_categories)
            
            logger.info("✅ Neural model initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML model: {e}")
            self.ml_available = False
    
    def predict_intent(self, query: str, context: Optional[Dict[str, Any]] = None) -> IntentPrediction:
        """Predict intent using available methods"""
        if self.ml_available and self.model:
            return self._predict_with_ml(query, context)
        else:
            return self._predict_with_patterns(query, context)
    
    def _predict_with_ml(self, query: str, context: Optional[Dict[str, Any]] = None) -> IntentPrediction:
        """Predict intent using ML model"""
        try:
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                max_length=512,
                padding=True,
                truncation=True
            )
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get predictions
            predicted_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = float(probabilities[0][predicted_idx])
            
            # Decode prediction
            predicted_intent = self.label_encoder.inverse_transform([predicted_idx])[0]
            
            # Create probability distribution
            probability_dist = {}
            for i, intent in enumerate(self.intent_categories):
                probability_dist[intent] = float(probabilities[0][i])
            
            return IntentPrediction(
                intent=predicted_intent,
                confidence=confidence,
                probability_distribution=probability_dist,
                context_influence={},
                explanation=f"ML-based prediction with {confidence:.1%} confidence",
                metadata={"method": "neural_network", "model_available": True}
            )
            
        except Exception as e:
            logger.error(f"❌ ML prediction failed: {e}")
            return self._predict_with_patterns(query, context)
    
    def _predict_with_patterns(self, query: str, context: Optional[Dict[str, Any]] = None) -> IntentPrediction:
        """Predict intent using pattern matching as fallback"""
        query_lower = query.lower()
        intent_scores = {}
        
        # Score each intent based on pattern matches
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            
            if score > 0:
                intent_scores[intent] = score / len(patterns)
        
        # If no patterns match, default to unknown
        if not intent_scores:
            predicted_intent = "unknown"
            confidence = 0.3
        else:
            # Get highest scoring intent
            predicted_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            confidence = min(intent_scores[predicted_intent] * 2, 1.0)  # Scale up confidence
        
        # Create probability distribution
        probability_dist = {intent: score for intent, score in intent_scores.items()}
        
        # Add zero scores for missing intents
        for intent in self.intent_categories:
            if intent not in probability_dist:
                probability_dist[intent] = 0.0
        
        return IntentPrediction(
            intent=predicted_intent,
            confidence=confidence,
            probability_distribution=probability_dist,
            context_influence={},
            explanation=f"Pattern-based prediction with {confidence:.1%} confidence",
            metadata={"method": "pattern_matching", "model_available": self.ml_available}
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "ml_available": self.ml_available,
            "model_loaded": self.model is not None,
            "intent_categories": self.intent_categories,
            "num_categories": len(self.intent_categories),
            "prediction_method": "neural_network" if (self.ml_available and self.model) else "pattern_matching"
        }

# Global instance
_fixed_intent_classifier = None

def get_fixed_intent_classifier() -> FixedEnhancedNeuralIntentClassifier:
    """Get the global fixed intent classifier instance"""
    global _fixed_intent_classifier
    if _fixed_intent_classifier is None:
        _fixed_intent_classifier = FixedEnhancedNeuralIntentClassifier()
    return _fixed_intent_classifier

# Test the classifier
if __name__ == "__main__":
    print("🎯 Fixed Enhanced Neural Intent Classifier")
    print("=" * 50)
    
    classifier = FixedEnhancedNeuralIntentClassifier()
    
    test_queries = [
        "Best Turkish restaurant in Sultanahmet",
        "How to get to Galata Tower?",
        "What time does Hagia Sophia open?",
        "I'm looking for a romantic dinner place",
        "Where can I find good kebab?",
        "Hello, I need help with directions",
        "This place is amazing!"
    ]
    
    print(f"\n🧪 Testing with {classifier.get_model_info()['prediction_method']} method:")
    for query in test_queries:
        prediction = classifier.predict_intent(query)
        print(f"  '{query}' -> {prediction.intent} ({prediction.confidence:.2%})")

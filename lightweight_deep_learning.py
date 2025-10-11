#!/usr/bin/env python3
"""
Lightweight Deep Learning System for Istanbul AI
Integrated with Multi-Intent Handler for Enhanced Query Processing
Step-by-step integration with the main AI system
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import advanced ML libraries with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using numpy-based fallback")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available, using basic similarity")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available, using basic data structures")

class LearningMode(Enum):
    """Different learning modes for the system"""
    ACTIVE = "active"           # Full learning and adaptation
    PASSIVE = "passive"         # Read-only, no updates
    INFERENCE = "inference"     # Fast inference only
    TRAINING = "training"       # Model training mode

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"           # Single intent, clear meaning
    MODERATE = "moderate"       # Multiple intents or some ambiguity
    COMPLEX = "complex"         # Multiple intents with relationships
    AMBIGUOUS = "ambiguous"     # Unclear intent or context needed

@dataclass
class LearningContext:
    """Context for deep learning operations"""
    user_id: str
    session_id: str
    conversation_history: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    location_context: Optional[Tuple[float, float]] = None
    temporal_context: datetime = field(default_factory=datetime.now)
    emotional_state: str = "neutral"
    interaction_count: int = 0

@dataclass
class MLPrediction:
    """Machine learning prediction result"""
    prediction: Any
    confidence: float
    reasoning: List[str] = field(default_factory=list)
    alternatives: List[Tuple[Any, float]] = field(default_factory=list)
    processing_time: float = 0.0

class LightweightNeuralNetwork:
    """Lightweight neural network implementation"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        if TORCH_AVAILABLE:
            self._init_torch_model()
        else:
            self._init_numpy_model()
    
    def _init_torch_model(self):
        """Initialize PyTorch-based model"""
        class SimpleNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(SimpleNN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return F.softmax(x, dim=1)
        
        self.model = SimpleNN(self.input_size, self.hidden_size, self.output_size)
        self.is_torch = True
        logger.info("🧠 PyTorch neural network initialized")
    
    def _init_numpy_model(self):
        """Initialize numpy-based fallback model"""
        # Simple feedforward network with numpy
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.bias2 = np.zeros((1, self.output_size))
        self.is_torch = False
        logger.info("📊 NumPy neural network initialized")
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.is_torch and TORCH_AVAILABLE:
            return self._torch_predict(inputs)
        else:
            return self._numpy_predict(inputs)
    
    def _torch_predict(self, inputs: np.ndarray) -> np.ndarray:
        """PyTorch prediction"""
        with torch.no_grad():
            x = torch.FloatTensor(inputs)
            output = self.model(x)
            return output.numpy()
    
    def _numpy_predict(self, inputs: np.ndarray) -> np.ndarray:
        """NumPy prediction"""
        # Forward pass
        z1 = np.dot(inputs, self.weights1) + self.bias1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = np.dot(a1, self.weights2) + self.bias2
        # Softmax activation
        exp_z2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        return exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

class IntentClassifier:
    """Intent classification using lightweight ML"""
    
    def __init__(self):
        # Initialize feature extraction
        if SKLEARN_AVAILABLE:
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.use_tfidf = True
        else:
            self.use_tfidf = False
            logger.info("Using basic keyword-based intent classification")
        
        # Intent patterns for fallback
        self.intent_keywords = {
            'restaurant': ['restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal'],
            'museum': ['museum', 'gallery', 'exhibition', 'art', 'history', 'cultural'],
            'district': ['district', 'neighborhood', 'area', 'quarter', 'region'],
            'transportation': ['metro', 'bus', 'transport', 'travel', 'get to', 'how to'],
            'attraction': ['attraction', 'visit', 'see', 'tourist', 'landmark', 'monument'],
            'recommendation': ['recommend', 'suggest', 'best', 'good', 'popular'],
            'information': ['what', 'how', 'when', 'where', 'why', 'tell me', 'info']
        }
        
        # Initialize neural network for intent classification
        self.neural_network = LightweightNeuralNetwork(
            input_size=100,  # Feature vector size
            hidden_size=50,
            output_size=len(self.intent_keywords)
        )
        
        logger.info("🎯 Intent classifier initialized")
    
    def classify_intent(self, query: str, context: LearningContext) -> MLPrediction:
        """Classify user intent with confidence"""
        start_time = datetime.now()
        
        if self.use_tfidf and SKLEARN_AVAILABLE:
            prediction = self._classify_with_tfidf(query, context)
        else:
            prediction = self._classify_with_keywords(query, context)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        prediction.processing_time = processing_time
        
        return prediction
    
    def _classify_with_tfidf(self, query: str, context: LearningContext) -> MLPrediction:
        """TF-IDF based classification"""
        # Use keyword-based approach first (more reliable)
        keyword_result = self._classify_with_keywords(query, context)
        
        # If keyword approach has high confidence, use it
        if keyword_result.confidence > 0.7:
            return keyword_result
        
        # Otherwise, enhance with neural network
        query_lower = query.lower()
        
        # Create feature vector with better feature engineering
        features = np.zeros(100)
        
        # Enhanced keyword scoring with weights
        for i, (intent, keywords) in enumerate(self.intent_keywords.items()):
            keyword_score = 0
            for keyword in keywords:
                if keyword in query_lower:
                    # Weight longer keywords more heavily
                    weight = len(keyword.split()) * 2
                    keyword_score += weight
            
            if i < len(self.intent_keywords):
                features[i] = min(keyword_score / 10.0, 1.0)  # Normalize
        
        # Add specific pattern matching
        patterns = {
            'restaurant': [r'\b(eat|food|restaurant|dining|cuisine|meal)\b'],
            'museum': [r'\b(museum|gallery|exhibition|art|history)\b'],
            'district': [r'\b(district|neighborhood|area|quarter)\b'],
            'transportation': [r'\b(metro|bus|transport|get\s+to|how\s+to)\b'],
            'recommendation': [r'\b(recommend|suggest|best|good|show\s+me)\b']
        }
        
        for i, (intent, pattern_list) in enumerate(patterns.items()):
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    if i < 50:  # Add to second half of feature vector
                        features[50 + i] = 1.0
        
        # Add context features
        if context.location_context:
            features[60] = 1.0
        
        features[61] = min(context.interaction_count / 10.0, 1.0)
        
        # Neural network prediction
        try:
            prediction_scores = self.neural_network.predict(features.reshape(1, -1))[0]
            
            # Combine with keyword result
            intent_names = list(self.intent_keywords.keys())
            
            # Find the intent in our keywords
            keyword_intent = keyword_result.prediction
            if keyword_intent in intent_names:
                keyword_idx = intent_names.index(keyword_intent)
                # Boost the keyword prediction
                prediction_scores[keyword_idx] *= 1.5
            
            # Normalize scores
            prediction_scores = prediction_scores / np.sum(prediction_scores)
            
            # Get top prediction
            top_intent_idx = np.argmax(prediction_scores)
            top_intent = intent_names[top_intent_idx] if top_intent_idx < len(intent_names) else 'general'
            confidence = float(prediction_scores[top_intent_idx])
            
            # Generate alternatives
            alternatives = []
            for i, score in enumerate(prediction_scores):
                if i != top_intent_idx and i < len(intent_names):
                    alternatives.append((intent_names[i], float(score)))
            alternatives.sort(key=lambda x: x[1], reverse=True)
            
            return MLPrediction(
                prediction=top_intent,
                confidence=confidence,
                reasoning=[f"Enhanced neural network prediction with keyword boost"],
                alternatives=alternatives[:3]
            )
            
        except Exception as e:
            logger.warning(f"Neural network prediction failed: {e}, falling back to keywords")
            return keyword_result
    
    def _classify_with_keywords(self, query: str, context: LearningContext) -> MLPrediction:
        """Keyword-based fallback classification"""
        query_lower = query.lower()
        scores = {}
        
        # Calculate keyword scores
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[intent] = score / len(keywords)
        
        if not scores:
            return MLPrediction(
                prediction='general',
                confidence=0.5,
                reasoning=["No specific intent detected, using general classification"]
            )
        
        # Get top intent
        top_intent = max(scores.items(), key=lambda x: x[1])
        confidence = min(top_intent[1] * 2, 1.0)  # Boost confidence but cap at 1.0
        
        # Generate alternatives
        alternatives = [(intent, score) for intent, score in scores.items() 
                       if intent != top_intent[0]]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return MLPrediction(
            prediction=top_intent[0],
            confidence=confidence,
            reasoning=[f"Keyword matching with {len(scores)} intent matches"],
            alternatives=alternatives[:3]
        )

class PersonalizationEngine:
    """Lightweight personalization engine"""
    
    def __init__(self):
        self.user_embeddings = {}  # Store user preference embeddings
        self.item_embeddings = {}  # Store item/location embeddings
        
        if SKLEARN_AVAILABLE:
            self.clustering_model = KMeans(n_clusters=5, random_state=42)
            self.use_clustering = True
        else:
            self.use_clustering = False
        
        logger.info("🎨 Personalization engine initialized")
    
    def learn_user_preferences(self, user_id: str, interactions: List[Dict]) -> None:
        """Learn user preferences from interactions"""
        if not interactions:
            return
        
        # Extract preference features
        features = self._extract_preference_features(interactions)
        
        # Update user embedding
        if user_id in self.user_embeddings:
            # Exponential moving average for continuous learning
            alpha = 0.3
            old_embedding = self.user_embeddings[user_id]
            new_embedding = alpha * features + (1 - alpha) * old_embedding
            self.user_embeddings[user_id] = new_embedding
        else:
            self.user_embeddings[user_id] = features
        
        logger.debug(f"📊 Updated preferences for user {user_id}")
    
    def _extract_preference_features(self, interactions: List[Dict]) -> np.ndarray:
        """Extract preference features from user interactions"""
        features = np.zeros(20)  # 20-dimensional preference vector
        
        for interaction in interactions:
            # Category preferences
            if interaction.get('category') == 'restaurant':
                features[0] += 1
            elif interaction.get('category') == 'museum':
                features[1] += 1
            elif interaction.get('category') == 'district':
                features[2] += 1
            
            # Budget preferences
            budget = interaction.get('budget', 'moderate')
            if budget == 'budget':
                features[3] += 1
            elif budget == 'luxury':
                features[4] += 1
            
            # Time preferences
            hour = interaction.get('hour', 12)
            if hour < 12:
                features[5] += 1  # Morning preference
            elif hour < 18:
                features[6] += 1  # Afternoon preference
            else:
                features[7] += 1  # Evening preference
            
            # Rating feedback
            rating = interaction.get('rating', 0)
            if rating >= 4:
                features[8] += 1  # Likes high-quality places
            
            # Location preferences
            district = interaction.get('district', '').lower()
            if 'sultanahmet' in district:
                features[10] += 1
            elif 'beyoglu' in district:
                features[11] += 1
            elif 'kadikoy' in district:
                features[12] += 1
        
        # Normalize features
        total_interactions = len(interactions)
        if total_interactions > 0:
            features = features / total_interactions
        
        return features
    
    def get_personalized_recommendations(self, user_id: str, items: List[Dict], 
                                       context: LearningContext) -> List[Tuple[Dict, float]]:
        """Get personalized recommendations"""
        if user_id not in self.user_embeddings:
            # Return items with default scoring
            return [(item, 0.5) for item in items]
        
        user_embedding = self.user_embeddings[user_id]
        recommendations = []
        
        for item in items:
            # Calculate compatibility score
            item_features = self._extract_item_features(item, context)
            
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity([user_embedding], [item_features])[0][0]
            else:
                # Simple dot product similarity
                similarity = np.dot(user_embedding, item_features) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(item_features) + 1e-8
                )
            
            # Boost score based on context
            context_boost = self._calculate_context_boost(item, context)
            final_score = similarity * 0.7 + context_boost * 0.3
            
            recommendations.append((item, float(final_score)))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def _extract_item_features(self, item: Dict, context: LearningContext) -> np.ndarray:
        """Extract features from an item for similarity comparison"""
        features = np.zeros(20)
        
        # Category features
        category = item.get('category', '').lower()
        if 'restaurant' in category:
            features[0] = 1
        elif 'museum' in category:
            features[1] = 1
        elif 'district' in category:
            features[2] = 1
        
        # Budget features
        price = item.get('price_level', 'moderate').lower()
        if price in ['budget', 'cheap']:
            features[3] = 1
        elif price in ['luxury', 'expensive']:
            features[4] = 1
        
        # Time suitability
        current_hour = context.temporal_context.hour
        suitable_times = item.get('suitable_times', [])
        if current_hour < 12 and 'morning' in suitable_times:
            features[5] = 1
        elif current_hour < 18 and 'afternoon' in suitable_times:
            features[6] = 1
        elif 'evening' in suitable_times:
            features[7] = 1
        
        # Quality indicators
        rating = item.get('rating', 0)
        if rating >= 4:
            features[8] = 1
        
        # Location features
        location = item.get('location', '').lower()
        if 'sultanahmet' in location:
            features[10] = 1
        elif 'beyoglu' in location:
            features[11] = 1
        elif 'kadikoy' in location:
            features[12] = 1
        
        return features
    
    def _calculate_context_boost(self, item: Dict, context: LearningContext) -> float:
        """Calculate context-based score boost"""
        boost = 0.0
        
        # Location proximity boost
        if context.location_context and item.get('coordinates'):
            # Simple distance calculation (placeholder)
            boost += 0.2
        
        # Time appropriateness boost
        current_hour = context.temporal_context.hour
        suitable_times = item.get('suitable_times', [])
        
        if current_hour < 12 and 'morning' in suitable_times:
            boost += 0.3
        elif current_hour < 18 and 'afternoon' in suitable_times:
            boost += 0.3
        elif current_hour >= 18 and 'evening' in suitable_times:
            boost += 0.3
        
        # Accessibility boost if needed
        if context.user_preferences.get('accessibility_needs') and item.get('accessible'):
            boost += 0.2
        
        return min(boost, 1.0)

class DeepLearningMultiIntentIntegration:
    """
    Integration layer between Deep Learning system and Multi-Intent Handler
    """
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.personalization_engine = PersonalizationEngine()
        self.learning_mode = LearningMode.ACTIVE
        
        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'avg_confidence': 0.0,
            'avg_processing_time': 0.0
        }
        
        logger.info("🤖 Deep Learning Multi-Intent Integration initialized")
    
    def enhance_multi_intent_analysis(self, query: str, base_result: Any, 
                                    context: LearningContext) -> Dict[str, Any]:
        """Enhance multi-intent analysis with deep learning"""
        
        # Step 1: Classify intent with ML
        ml_intent_prediction = self.intent_classifier.classify_intent(query, context)
        
        # Step 2: Get personalized insights
        if hasattr(base_result, 'primary_intent'):
            # Extract items from base result for personalization
            items = self._extract_items_from_result(base_result)
            personalized_items = self.personalization_engine.get_personalized_recommendations(
                context.user_id, items, context
            )
        else:
            personalized_items = []
        
        # Step 3: Generate enhanced response
        enhanced_result = {
            'original_result': base_result,
            'ml_intent_prediction': ml_intent_prediction,
            'personalized_recommendations': personalized_items,
            'confidence_boost': self._calculate_confidence_boost(ml_intent_prediction, base_result),
            'learning_insights': self._generate_learning_insights(query, context),
            'processing_metadata': {
                'deep_learning_used': True,
                'personalization_applied': len(personalized_items) > 0,
                'confidence_score': ml_intent_prediction.confidence,
                'processing_time': ml_intent_prediction.processing_time
            }
        }
        
        # Update statistics
        self._update_prediction_stats(ml_intent_prediction)
        
        return enhanced_result
    
    def learn_from_interaction(self, user_id: str, query: str, response: str, 
                             feedback: Optional[Dict] = None) -> None:
        """Learn from user interactions"""
        if self.learning_mode != LearningMode.ACTIVE:
            return
        
        # Extract interaction data
        interaction_data = {
            'query': query,
            'response': response,
            'timestamp': datetime.now(),
            'feedback': feedback or {}
        }
        
        # Update personalization engine
        if feedback:
            interactions = [self._convert_feedback_to_interaction(feedback)]
            self.personalization_engine.learn_user_preferences(user_id, interactions)
        
        logger.debug(f"🎓 Learning from interaction for user {user_id}")
    
    def _extract_items_from_result(self, result: Any) -> List[Dict]:
        """Extract items/recommendations from multi-intent result"""
        items = []
        
        if hasattr(result, 'execution_plan'):
            for step in result.execution_plan:
                if 'items' in step:
                    items.extend(step['items'])
        
        # Fallback: create dummy items for testing
        if not items:
            items = [
                {'name': 'Sample Item', 'category': 'general', 'rating': 4.0}
            ]
        
        return items
    
    def _calculate_confidence_boost(self, ml_prediction: MLPrediction, base_result: Any) -> float:
        """Calculate confidence boost from ML prediction"""
        base_confidence = getattr(base_result, 'confidence_score', 0.5)
        ml_confidence = ml_prediction.confidence
        
        # Weighted combination
        boosted_confidence = 0.6 * base_confidence + 0.4 * ml_confidence
        return min(boosted_confidence, 1.0)
    
    def _generate_learning_insights(self, query: str, context: LearningContext) -> Dict[str, Any]:
        """Generate insights for continuous learning"""
        insights = {
            'query_complexity': self._assess_query_complexity(query),
            'user_journey_stage': self._assess_user_journey_stage(context),
            'personalization_opportunities': self._identify_personalization_opportunities(query, context),
            'improvement_suggestions': []
        }
        
        # Add improvement suggestions
        if context.interaction_count < 5:
            insights['improvement_suggestions'].append("Collect more user preferences")
        
        if not context.location_context:
            insights['improvement_suggestions'].append("Request location for better recommendations")
        
        return insights
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess the complexity of the user query"""
        word_count = len(query.split())
        question_marks = query.count('?')
        conjunctions = len(re.findall(r'\b(and|or|but|also|with|plus)\b', query.lower()))
        
        if word_count <= 5 and question_marks <= 1 and conjunctions == 0:
            return QueryComplexity.SIMPLE.value
        elif word_count <= 15 and conjunctions <= 2:
            return QueryComplexity.MODERATE.value
        elif conjunctions > 2 or word_count > 15:
            return QueryComplexity.COMPLEX.value
        else:
            return QueryComplexity.AMBIGUOUS.value
    
    def _assess_user_journey_stage(self, context: LearningContext) -> str:
        """Assess what stage the user is at in their journey"""
        interaction_count = context.interaction_count
        
        if interaction_count <= 2:
            return "discovery"
        elif interaction_count <= 10:
            return "exploration"
        elif interaction_count <= 20:
            return "decision_making"
        else:
            return "experienced_user"
    
    def _identify_personalization_opportunities(self, query: str, context: LearningContext) -> List[str]:
        """Identify opportunities for better personalization"""
        opportunities = []
        
        if not context.user_preferences:
            opportunities.append("collect_basic_preferences")
        
        if not context.location_context:
            opportunities.append("request_location_access")
        
        if context.interaction_count > 5 and context.emotional_state == "neutral":
            opportunities.append("enhance_emotional_understanding")
        
        # Check for repeated similar queries
        similar_queries = sum(1 for hist_query in context.conversation_history 
                            if self._calculate_query_similarity(query, hist_query) > 0.7)
        if similar_queries > 1:
            opportunities.append("diversify_recommendations")
        
        return opportunities
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate similarity between two queries"""
        # Simple word overlap similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _convert_feedback_to_interaction(self, feedback: Dict) -> Dict:
        """Convert user feedback to interaction data for learning"""
        return {
            'category': feedback.get('category', 'general'),
            'rating': feedback.get('rating', 3),
            'budget': feedback.get('budget', 'moderate'),
            'hour': datetime.now().hour,
            'district': feedback.get('location', ''),
            'liked': feedback.get('rating', 3) >= 4
        }
    
    def _update_prediction_stats(self, prediction: MLPrediction) -> None:
        """Update prediction statistics"""
        self.prediction_stats['total_predictions'] += 1
        total = self.prediction_stats['total_predictions']
        
        # Update running averages
        self.prediction_stats['avg_confidence'] = (
            (self.prediction_stats['avg_confidence'] * (total - 1) + prediction.confidence) / total
        )
        self.prediction_stats['avg_processing_time'] = (
            (self.prediction_stats['avg_processing_time'] * (total - 1) + prediction.processing_time) / total
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and performance metrics"""
        return {
            'learning_mode': self.learning_mode.value,
            'torch_available': TORCH_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'pandas_available': PANDAS_AVAILABLE,
            'prediction_stats': self.prediction_stats.copy(),
            'models_loaded': {
                'intent_classifier': True,
                'personalization_engine': True,
                'neural_network': self.intent_classifier.neural_network is not None
            }
        }

# Factory function for easy integration
def create_lightweight_deep_learning_system() -> DeepLearningMultiIntentIntegration:
    """Factory function to create the deep learning system"""
    return DeepLearningMultiIntentIntegration()

# Test function
def test_deep_learning_integration():
    """Test the deep learning integration"""
    print("🧪 Testing Lightweight Deep Learning Integration")
    
    # Create system
    dl_system = create_lightweight_deep_learning_system()
    
    # Create test context
    context = LearningContext(
        user_id="test_user",
        session_id="test_session",
        conversation_history=["Hello", "I want food"],
        user_preferences={'budget': 'moderate'},
        location_context=(41.0082, 28.9784)
    )
    
    # Test intent classification
    test_queries = [
        "I want Turkish food in Sultanahmet",
        "Show me museums near me",
        "What can I do in Beyoglu?",
        "How do I get to Taksim?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Testing: '{query}'")
        result = dl_system.intent_classifier.classify_intent(query, context)
        print(f"   Predicted intent: {result.prediction}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Processing time: {result.processing_time:.3f}s")
    
    # Test system status
    status = dl_system.get_system_status()
    print(f"\n📊 System Status: {status}")
    
    print("✅ Deep Learning Integration test completed!")

if __name__ == "__main__":
    test_deep_learning_integration()

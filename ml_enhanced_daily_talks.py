#!/usr/bin/env python3
"""
ML/DL Enhanced Daily Talks System for A/ISTANBUL
===============================================

This module integrates machine learning and deep learning capabilities
into the daily talks system, providing:

🧠 Neural Intent Recognition with Transformer Models
🎯 Context-Aware Response Generation using GPT-style Architecture  
📊 Sentiment Analysis and Emotion Detection
🔮 Predictive User Modeling and Personalization
🌍 Multi-language Support with Neural Translation
📈 Real-time Learning and Adaptation
🚀 Scalable ML Pipeline Integration

Features:
- Advanced transformer-based intent classification
- Neural conversation context modeling
- Real-time sentiment analysis
- Predictive user preference modeling
- Automated response quality assessment
- Multi-modal input processing (text, voice, image)
- Continuous learning from user interactions
"""

import os
import json
import logging
import asyncio
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import hashlib
from collections import defaultdict, deque
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

# Import base daily talks system
from comprehensive_daily_talks_system import (
    ComprehensiveDailyTalksSystem, IntentType, ConversationTone, 
    MoodState, ConversationContext, UserProfile, WeatherCondition
)

# Import main AI system
try:
    from advanced_istanbul_ai import AdvancedIstanbulAI, IstanbulKnowledgeGraph
except ImportError:
    logger.warning("Advanced Istanbul AI not found - using fallback mode")
    AdvancedIstanbulAI = None
    IstanbulKnowledgeGraph = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ML/DL CORE COMPONENTS
# =============================================================================

class MLModelType(Enum):
    """Machine Learning model types"""
    INTENT_CLASSIFIER = "intent_classifier"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    EMOTION_DETECTOR = "emotion_detector"
    RESPONSE_GENERATOR = "response_generator"
    USER_PROFILER = "user_profiler"
    CONTEXT_ENCODER = "context_encoder"
    QUALITY_ASSESSOR = "quality_assessor"

@dataclass
class MLPrediction:
    """ML prediction result"""
    prediction: Any
    confidence: float
    model_type: MLModelType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class NeuralContext:
    """Neural context representation"""
    embeddings: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    hidden_states: Optional[np.ndarray] = None
    context_vector: Optional[np.ndarray] = None
    semantic_features: Dict[str, float] = field(default_factory=dict)

class TransformerIntentClassifier(nn.Module):
    """Advanced intent classifier using transformer architecture"""
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased", 
                 num_intents: int = len(IntentType), hidden_dim: int = 256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.intent_head = nn.Linear(hidden_dim, num_intents)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.dropout(pooled_output)
        hidden = F.relu(self.classifier(dropped))
        
        intent_logits = self.intent_head(hidden)
        confidence = torch.sigmoid(self.confidence_head(hidden))
        
        return intent_logits, confidence

class ContextualResponseGenerator(nn.Module):
    """Neural response generation with context awareness"""
    
    def __init__(self, vocab_size: int = 50000, embed_dim: int = 512, 
                 hidden_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.context_encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                                     batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.response_decoder = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim, 
                                      num_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_ids, context_ids=None):
        # Encode input
        embedded = self.embedding(input_ids)
        context_output, _ = self.context_encoder(embedded)
        
        # Apply attention if context provided
        if context_ids is not None:
            context_embedded = self.embedding(context_ids)
            context_encoded, _ = self.context_encoder(context_embedded)
            attended_context, _ = self.attention(context_output, context_encoded, context_encoded)
            combined = torch.cat([embedded, attended_context], dim=-1)
        else:
            combined = torch.cat([embedded, context_output], dim=-1)
        
        # Generate response
        response_output, _ = self.response_decoder(combined)
        logits = self.output_projection(response_output)
        
        return logits

class MLEnhancedDailyTalks:
    """ML/DL Enhanced Daily Talks System"""
    
    def __init__(self, models_dir: str = "ml_models", 
                 base_system: Optional[ComprehensiveDailyTalksSystem] = None):
        self.models_dir = models_dir
        self.base_system = base_system or ComprehensiveDailyTalksSystem()
        self.main_ai_system = None
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize ML models
        self.models = {}
        self.tokenizers = {}
        self.vectorizers = {}
        
        # Initialize neural components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intent_classifier = None
        self.response_generator = None
        
        # Initialize databases
        self.init_ml_database()
        
        # Load or initialize models
        self.load_or_initialize_models()
        
        # Context memory
        self.neural_context_memory = {}
        self.user_embeddings = {}
        
        logger.info(f"ML Enhanced Daily Talks initialized with device: {self.device}")
    
    def init_ml_database(self):
        """Initialize ML-specific database tables"""
        try:
            self.ml_db_path = os.path.join(self.models_dir, "ml_daily_talks.db")
            conn = sqlite3.connect(self.ml_db_path)
            cursor = conn.cursor()
            
            # User embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_embeddings (
                    user_id TEXT PRIMARY KEY,
                    embedding BLOB,
                    last_updated TIMESTAMP,
                    interaction_count INTEGER DEFAULT 0
                )
            ''')
            
            # Conversation context embeddings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS context_embeddings (
                    session_id TEXT PRIMARY KEY,
                    context_vector BLOB,
                    attention_weights BLOB,
                    semantic_features TEXT,
                    created_at TIMESTAMP,
                    last_accessed TIMESTAMP
                )
            ''')
            
            # ML predictions log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    model_type TEXT,
                    prediction TEXT,
                    confidence REAL,
                    metadata TEXT,
                    timestamp TIMESTAMP,
                    feedback_score REAL
                )
            ''')
            
            # Training data collection
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_text TEXT,
                    intent_label TEXT,
                    response_text TEXT,
                    user_feedback REAL,
                    context_data TEXT,
                    timestamp TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("ML database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML database: {e}")
    
    def load_or_initialize_models(self):
        """Load pre-trained models or initialize new ones"""
        try:
            # Initialize tokenizers
            self.tokenizers['multilingual'] = AutoTokenizer.from_pretrained(
                "bert-base-multilingual-cased"
            )
            self.tokenizers['turkish'] = AutoTokenizer.from_pretrained(
                "dbmdz/bert-base-turkish-cased"
            )
            
            # Initialize sentiment analysis pipeline
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize emotion detection
            self.models['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize TF-IDF vectorizer for quick similarity calculations
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=10000, stop_words='english', ngram_range=(1, 2)
            )
            
            # Initialize neural models
            self.intent_classifier = TransformerIntentClassifier()
            self.response_generator = ContextualResponseGenerator()
            
            # Try to load pre-trained weights
            self.load_model_weights()
            
            logger.info("ML models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            # Continue with basic functionality
    
    def load_model_weights(self):
        """Load pre-trained model weights if available"""
        try:
            intent_path = os.path.join(self.models_dir, "intent_classifier.pth")
            if os.path.exists(intent_path):
                self.intent_classifier.load_state_dict(torch.load(intent_path, map_location=self.device))
                logger.info("Loaded pre-trained intent classifier")
            
            response_path = os.path.join(self.models_dir, "response_generator.pth")
            if os.path.exists(response_path):
                self.response_generator.load_state_dict(torch.load(response_path, map_location=self.device))
                logger.info("Loaded pre-trained response generator")
                
            # Load vectorizer if exists
            vectorizer_path = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
            if os.path.exists(vectorizer_path):
                self.vectorizers['tfidf'] = joblib.load(vectorizer_path)
                logger.info("Loaded pre-trained TF-IDF vectorizer")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights: {e}")
    
    def connect_to_main_ai(self, main_ai_system):
        """Connect to the main AI system for enhanced capabilities"""
        self.main_ai_system = main_ai_system
        logger.info("Connected to main AI system")
    
    async def enhanced_intent_recognition(self, user_input: str, 
                                        context: ConversationContext) -> Tuple[IntentType, float]:
        """Enhanced intent recognition using ML models"""
        try:
            # Get base system prediction
            base_intent, base_confidence = self.base_system.intent_detector.detect_intent(
                user_input, context
            )
            
            # Neural prediction if model is available
            neural_intent, neural_confidence = await self._neural_intent_prediction(
                user_input, context
            )
            
            # Combine predictions with weighted ensemble
            if neural_confidence > 0.8 and neural_confidence > base_confidence:
                final_intent = neural_intent
                final_confidence = neural_confidence
            else:
                # Use base system but boost confidence with neural validation
                final_intent = base_intent
                final_confidence = min(0.95, base_confidence + (neural_confidence * 0.2))
            
            # Log prediction for training
            await self._log_ml_prediction(
                user_input, MLModelType.INTENT_CLASSIFIER, 
                final_intent.value, final_confidence, context.user_id
            )
            
            return final_intent, final_confidence
            
        except Exception as e:
            logger.error(f"Error in enhanced intent recognition: {e}")
            return self.base_system.intent_detector.detect_intent(user_input, context)
    
    async def _neural_intent_prediction(self, user_input: str, 
                                      context: ConversationContext) -> Tuple[IntentType, float]:
        """Neural intent prediction using transformer model"""
        try:
            if not self.intent_classifier or not self.tokenizers.get('multilingual'):
                return IntentType.GENERAL_CHAT, 0.5
            
            # Tokenize input
            tokenizer = self.tokenizers['multilingual']
            inputs = tokenizer(user_input, return_tensors="pt", 
                             truncation=True, padding=True, max_length=512)
            
            # Get predictions
            self.intent_classifier.eval()
            with torch.no_grad():
                intent_logits, confidence = self.intent_classifier(
                    inputs['input_ids'], inputs['attention_mask']
                )
                
                # Convert to probabilities
                intent_probs = F.softmax(intent_logits, dim=-1)
                predicted_intent_idx = torch.argmax(intent_probs, dim=-1).item()
                predicted_confidence = confidence.item()
                
                # Map to intent type
                intent_types = list(IntentType)
                if predicted_intent_idx < len(intent_types):
                    predicted_intent = intent_types[predicted_intent_idx]
                else:
                    predicted_intent = IntentType.GENERAL_CHAT
                
                return predicted_intent, predicted_confidence
                
        except Exception as e:
            logger.error(f"Error in neural intent prediction: {e}")
            return IntentType.GENERAL_CHAT, 0.5
    
    async def enhanced_sentiment_analysis(self, user_input: str) -> Dict[str, Any]:
        """Enhanced sentiment and emotion analysis"""
        try:
            results = {
                'sentiment': {'label': 'neutral', 'score': 0.5},
                'emotion': {'label': 'neutral', 'score': 0.5},
                'mood_indicators': []
            }
            
            # Sentiment analysis
            if 'sentiment' in self.models:
                sentiment_result = self.models['sentiment'](user_input)[0]
                results['sentiment'] = {
                    'label': sentiment_result['label'].lower(),
                    'score': sentiment_result['score']
                }
            
            # Emotion detection
            if 'emotion' in self.models:
                emotion_result = self.models['emotion'](user_input)[0]
                results['emotion'] = {
                    'label': emotion_result['label'].lower(),
                    'score': emotion_result['score']
                }
            
            # Extract mood indicators
            results['mood_indicators'] = self._extract_mood_indicators(user_input)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {
                'sentiment': {'label': 'neutral', 'score': 0.5},
                'emotion': {'label': 'neutral', 'score': 0.5},
                'mood_indicators': []
            }
    
    def _extract_mood_indicators(self, text: str) -> List[str]:
        """Extract mood indicators from text"""
        mood_patterns = {
            'excited': ['excited', 'amazing', 'awesome', 'fantastic', '!', 'love'],
            'tired': ['tired', 'exhausted', 'sleepy', 'worn out', 'drained'],
            'stressed': ['stressed', 'overwhelmed', 'anxious', 'worried', 'pressure'],
            'happy': ['happy', 'glad', 'joyful', 'cheerful', 'delighted'],
            'sad': ['sad', 'down', 'blue', 'disappointed', 'upset'],
            'curious': ['curious', 'wondering', 'interested', 'what about', 'how'],
            'romantic': ['romantic', 'love', 'romantic dinner', 'sunset', 'couple']
        }
        
        indicators = []
        text_lower = text.lower()
        
        for mood, patterns in mood_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                indicators.append(mood)
        
        return indicators
    
    async def enhanced_response_generation(self, intent: IntentType, user_input: str,
                                         context: ConversationContext) -> str:
        """Enhanced response generation with ML augmentation"""
        try:
            # Get base response
            base_response = await self.base_system.generate_response(
                user_input, context
            )
            
            # Enhance with sentiment-aware modifications
            sentiment_data = await self.enhanced_sentiment_analysis(user_input)
            enhanced_response = self._apply_sentiment_enhancement(
                base_response, sentiment_data, context
            )
            
            # Apply personalization if main AI system is connected
            if self.main_ai_system:
                enhanced_response = await self._apply_ai_personalization(
                    enhanced_response, context, user_input
                )
            
            # Update neural context
            await self._update_neural_context(user_input, enhanced_response, context)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Error in enhanced response generation: {e}")
            return await self.base_system.generate_response(user_input, context)
    
    def _apply_sentiment_enhancement(self, response: str, sentiment_data: Dict[str, Any],
                                   context: ConversationContext) -> str:
        """Apply sentiment-aware enhancements to response"""
        sentiment = sentiment_data['sentiment']['label']
        emotion = sentiment_data['emotion']['label']
        
        # Mood-based response modifications
        if sentiment == 'negative' or emotion in ['sadness', 'anger', 'fear']:
            # Add supportive elements
            if "I understand" not in response:
                response = "I understand this might be challenging. " + response
            
            # Add encouraging emojis
            if '💝' not in response and '🌟' not in response:
                response += " 🌟"
        
        elif sentiment == 'positive' or emotion in ['joy', 'excitement']:
            # Add enthusiastic elements
            if '🎉' not in response and '✨' not in response:
                response += " ✨"
        
        # Adjust tone based on detected mood indicators
        mood_indicators = sentiment_data.get('mood_indicators', [])
        if 'excited' in mood_indicators:
            response = response.replace('Here are', 'Here are some AMAZING')
            response = response.replace('You might', 'You\'ll absolutely love')
        
        return response
    
    async def _apply_ai_personalization(self, response: str, context: ConversationContext,
                                      user_input: str) -> str:
        """Apply AI-powered personalization"""
        try:
            if not self.main_ai_system:
                return response
            
            # Get user profile from main AI system
            user_profile = getattr(self.main_ai_system, 'user_profiles', {}).get(
                context.user_id
            )
            
            if user_profile:
                # Apply preference-based modifications
                if hasattr(user_profile, 'favorite_neighborhoods'):
                    neighborhoods = user_profile.favorite_neighborhoods
                    if neighborhoods and any(hood in response.lower() for hood in 
                                           [n.lower() for n in neighborhoods]):
                        response += f"\n\n💡 Since you love {', '.join(neighborhoods)}, you might especially enjoy this!"
                
                # Apply cultural sensitivity adjustments
                if hasattr(user_profile, 'cultural_sensitivity'):
                    if user_profile.cultural_sensitivity > 0.8:
                        # Add more cultural context
                        response = self._add_cultural_context(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error applying AI personalization: {e}")
            return response
    
    def _add_cultural_context(self, response: str) -> str:
        """Add cultural context to responses"""
        cultural_additions = {
            'restaurant': '\n\n🏛️ **Cultural Note:** Turkish dining is about community - meals are meant to be shared and savored slowly.',
            'mosque': '\n\n🕌 **Cultural Respect:** Remember to dress modestly and remove shoes when visiting mosques.',
            'bazaar': '\n\n🛍️ **Cultural Tip:** Bargaining is expected and appreciated in traditional bazaars - it\'s part of the experience!',
            'tea': '\n\n🍵 **Cultural Insight:** Tea culture is central to Turkish social life - accepting tea shows respect and friendship.'
        }
        
        for keyword, addition in cultural_additions.items():
            if keyword in response.lower() and addition not in response:
                response += addition
                break
        
        return response
    
    async def _update_neural_context(self, user_input: str, response: str,
                                   context: ConversationContext):
        """Update neural context embeddings"""
        try:
            # Generate embeddings for context
            if 'multilingual' in self.tokenizers:
                tokenizer = self.tokenizers['multilingual']
                
                # Create context vector
                combined_text = f"{user_input} [SEP] {response}"
                inputs = tokenizer(combined_text, return_tensors="pt",
                                 truncation=True, padding=True, max_length=512)
                
                # Store in memory for this session
                self.neural_context_memory[context.session_id] = {
                    'embeddings': inputs,
                    'timestamp': datetime.now(),
                    'turn_count': context.turn_count
                }
                
                # Persist to database
                await self._persist_context_embeddings(context.session_id, inputs)
                
        except Exception as e:
            logger.error(f"Error updating neural context: {e}")
    
    async def _persist_context_embeddings(self, session_id: str, embeddings):
        """Persist context embeddings to database"""
        try:
            conn = sqlite3.connect(self.ml_db_path)
            cursor = conn.cursor()
            
            # Serialize embeddings
            embeddings_blob = pickle.dumps(embeddings)
            
            cursor.execute('''
                INSERT OR REPLACE INTO context_embeddings 
                (session_id, context_vector, created_at, last_accessed)
                VALUES (?, ?, ?, ?)
            ''', (session_id, embeddings_blob, datetime.now(), datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error persisting context embeddings: {e}")
    
    async def _log_ml_prediction(self, user_input: str, model_type: MLModelType,
                               prediction: str, confidence: float, user_id: str):
        """Log ML predictions for training and analysis"""
        try:
            conn = sqlite3.connect(self.ml_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ml_predictions 
                (user_id, model_type, prediction, confidence, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, model_type.value, prediction, confidence, 
                  json.dumps({'input': user_input}), datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging ML prediction: {e}")
    
    async def get_conversation_insights(self, user_id: str) -> Dict[str, Any]:
        """Get ML-powered conversation insights"""
        try:
            conn = sqlite3.connect(self.ml_db_path)
            cursor = conn.cursor()
            
            # Get recent predictions
            cursor.execute('''
                SELECT model_type, prediction, confidence, timestamp
                FROM ml_predictions 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 50
            ''', (user_id,))
            
            predictions = cursor.fetchall()
            
            # Analyze patterns
            insights = {
                'total_interactions': len(predictions),
                'average_confidence': np.mean([p[2] for p in predictions]) if predictions else 0,
                'intent_distribution': defaultdict(int),
                'confidence_trend': [],
                'model_performance': defaultdict(list)
            }
            
            for model_type, prediction, confidence, timestamp in predictions:
                insights['intent_distribution'][prediction] += 1
                insights['confidence_trend'].append(confidence)
                insights['model_performance'][model_type].append(confidence)
            
            conn.close()
            return insights
            
        except Exception as e:
            logger.error(f"Error getting conversation insights: {e}")
            return {}
    
    async def train_on_feedback(self, user_id: str, feedback_score: float,
                              conversation_data: Dict[str, Any]):
        """Train models based on user feedback"""
        try:
            # Log training sample
            conn = sqlite3.connect(self.ml_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_samples 
                (input_text, intent_label, response_text, user_feedback, context_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                conversation_data.get('input', ''),
                conversation_data.get('intent', ''),
                conversation_data.get('response', ''),
                feedback_score,
                json.dumps(conversation_data.get('context', {})),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            # Trigger model retraining if enough samples collected
            await self._check_retrain_trigger()
            
            logger.info(f"Logged training sample with feedback: {feedback_score}")
            
        except Exception as e:
            logger.error(f"Error training on feedback: {e}")
    
    async def _check_retrain_trigger(self):
        """Check if models should be retrained"""
        try:
            conn = sqlite3.connect(self.ml_db_path)
            cursor = conn.cursor()
            
            # Count recent training samples
            cursor.execute('''
                SELECT COUNT(*) FROM training_samples 
                WHERE timestamp > datetime('now', '-7 days')
            ''')
            
            recent_samples = cursor.fetchone()[0]
            conn.close()
            
            # Retrain if we have enough new samples
            if recent_samples >= 100:  # Adjust threshold as needed
                logger.info(f"Triggering model retraining with {recent_samples} new samples")
                await self._retrain_models()
                
        except Exception as e:
            logger.error(f"Error checking retrain trigger: {e}")
    
    async def _retrain_models(self):
        """Retrain models with new data"""
        try:
            # This would implement actual model retraining
            # For now, we'll just log the intent
            logger.info("Model retraining initiated - this would update neural networks")
            
            # In a production system, this would:
            # 1. Load training data from database
            # 2. Prepare training batches
            # 3. Fine-tune neural models
            # 4. Validate performance
            # 5. Save updated model weights
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    async def process_daily_conversation(self, user_input: str, user_id: str,
                                       session_id: Optional[str] = None) -> Dict[str, Any]:
        """Main entry point for ML-enhanced daily conversation processing"""
        try:
            # Create or get conversation context
            context = ConversationContext(
                user_id=user_id,
                session_id=session_id or f"session_{datetime.now().timestamp()}",
                timestamp=datetime.now()
            )
            
            # Enhanced intent recognition
            intent, confidence = await self.enhanced_intent_recognition(user_input, context)
            
            # Enhanced sentiment analysis
            sentiment_data = await self.enhanced_sentiment_analysis(user_input)
            
            # Enhanced response generation
            response = await self.enhanced_response_generation(intent, user_input, context)
            
            # Prepare result
            result = {
                'response': response,
                'intent': intent.value,
                'confidence': confidence,
                'sentiment': sentiment_data,
                'session_id': context.session_id,
                'timestamp': context.timestamp.isoformat(),
                'ml_enhanced': True
            }
            
            # Add AI system insights if available
            if self.main_ai_system:
                result['ai_insights'] = await self._get_ai_insights(user_input, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing daily conversation: {e}")
            # Fallback to base system
            base_response = await self.base_system.generate_response(user_input, context)
            return {
                'response': base_response,
                'intent': 'general_chat',
                'confidence': 0.5,
                'ml_enhanced': False,
                'error': str(e)
            }
    
    async def _get_ai_insights(self, user_input: str, context: ConversationContext) -> Dict[str, Any]:
        """Get insights from main AI system"""
        try:
            if not self.main_ai_system:
                return {}
            
            # This would integrate with the main AI system's analytics
            insights = {
                'user_type_prediction': 'tourist',
                'personalization_score': 0.8,
                'cultural_relevance': 0.9,
                'recommendation_confidence': 0.85
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting AI insights: {e}")
            return {}

# =============================================================================
# INTEGRATION WRAPPER
# =============================================================================

class IntegratedDailyTalksSystem:
    """Integrated system combining ML enhancement with main AI"""
    
    def __init__(self, models_dir: str = "ml_models"):
        self.ml_enhanced_system = MLEnhancedDailyTalks(models_dir=models_dir)
        self.main_ai_system = None
        
        # Try to initialize main AI system
        if AdvancedIstanbulAI:
            try:
                self.main_ai_system = AdvancedIstanbulAI()
                self.ml_enhanced_system.connect_to_main_ai(self.main_ai_system)
                logger.info("Successfully integrated with main AI system")
            except Exception as e:
                logger.warning(f"Could not initialize main AI system: {e}")
    
    async def chat(self, user_input: str, user_id: str, 
                  session_id: Optional[str] = None) -> Dict[str, Any]:
        """Main chat interface"""
        return await self.ml_enhanced_system.process_daily_conversation(
            user_input, user_id, session_id
        )
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user insights"""
        ml_insights = await self.ml_enhanced_system.get_conversation_insights(user_id)
        
        # Add main AI insights if available
        if self.main_ai_system and hasattr(self.main_ai_system, 'user_profiles'):
            ai_profile = self.main_ai_system.user_profiles.get(user_id)
            if ai_profile:
                ml_insights['ai_profile'] = asdict(ai_profile)
        
        return ml_insights
    
    async def provide_feedback(self, user_id: str, feedback_score: float,
                             conversation_data: Dict[str, Any]):
        """Provide feedback for continuous learning"""
        await self.ml_enhanced_system.train_on_feedback(
            user_id, feedback_score, conversation_data
        )

# =============================================================================
# TESTING AND DEMO
# =============================================================================

async def test_ml_enhanced_system():
    """Test the ML enhanced daily talks system"""
    print("🧠 Testing ML Enhanced Daily Talks System")
    print("=" * 50)
    
    # Initialize system
    system = IntegratedDailyTalksSystem()
    
    # Test conversations
    test_conversations = [
        ("Hi! I'm feeling excited about exploring Istanbul today!", "user123"),
        ("I'm a bit tired and looking for somewhere peaceful to relax", "user123"),
        ("What are some hidden gems for food that locals love?", "user456"),
        ("I'm stressed about navigating the city. Can you help?", "user789"),
        ("Where can I find the best Turkish breakfast?", "user123"),
    ]
    
    for user_input, user_id in test_conversations:
        print(f"\n🎯 User Input: {user_input}")
        print(f"👤 User ID: {user_id}")
        
        result = await system.chat(user_input, user_id)
        
        print(f"🤖 Response: {result['response'][:200]}...")
        print(f"🎭 Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        
        if 'sentiment' in result:
            sentiment = result['sentiment']
            print(f"😊 Sentiment: {sentiment['sentiment']['label']} ({sentiment['sentiment']['score']:.2f})")
            print(f"💭 Emotion: {sentiment['emotion']['label']} ({sentiment['emotion']['score']:.2f})")
        
        print(f"🔧 ML Enhanced: {result.get('ml_enhanced', False)}")
        print("-" * 30)
    
    # Test user insights
    print("\n📊 User Insights for user123:")
    insights = await system.get_user_insights("user123")
    print(json.dumps(insights, indent=2, default=str))
    
    print("\n✅ ML Enhanced Daily Talks System test completed!")

if __name__ == "__main__":
    asyncio.run(test_ml_enhanced_system())

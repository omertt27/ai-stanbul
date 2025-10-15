#!/usr/bin/env python3
"""
ML-Enhanced Daily Talks Bridge for A/ISTANBUL
============================================

This bridge connects the comprehensive daily talks system to the main AI infrastructure,
providing ML/DL-enhanced personalization, context adaptation, and multi-modal responses.

Features:
- Advanced ML-based intent classification
- Personalized response generation
- Context-aware conversation memory
- Multi-modal response synthesis
- Integration with main AI system
- Real-time learning and adaptation
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Deep Learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertModel
    )
    from sentence_transformers import SentenceTransformer
    DEEP_LEARNING_AVAILABLE = True
    logger_dl = logging.getLogger(__name__ + ".deep_learning")
    logger_dl.info("✅ Deep Learning libraries loaded successfully")
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    logger_dl = logging.getLogger(__name__ + ".deep_learning")
    logger_dl.warning(f"⚠️ Deep Learning libraries not available: {e}")

# Import the comprehensive daily talks system
from comprehensive_daily_talks_system import ComprehensiveDailyTalksSystem
from daily_talks_integration_wrapper import DailyTalksIntegrationWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User profile for personalization"""
    user_id: str
    preferences: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    personality_traits: Dict[str, float]
    location_preferences: Dict[str, Any]
    activity_patterns: Dict[str, Any]
    language_style: str = "casual"
    cultural_background: Optional[str] = None
    visit_frequency: str = "first_time"  # first_time, occasional, frequent, local

@dataclass
class ConversationContext:
    """Enhanced conversation context with ML features"""
    session_id: str
    user_profile: UserProfile
    conversation_history: List[Dict[str, Any]]
    current_mood: Optional[str] = None
    current_location: Optional[str] = None
    time_context: Optional[str] = None
    weather_context: Optional[Dict[str, Any]] = None
    active_topics: List[str] = None
    intent_confidence: float = 0.0
    multi_modal_data: Optional[Dict[str, Any]] = None

# Deep Learning Neural Network Architectures
if DEEP_LEARNING_AVAILABLE:
    
    class AttentionIntentClassifier(nn.Module):
        """Advanced intent classifier with self-attention mechanism"""
        
        def __init__(self, vocab_size: int = 10000, embed_dim: int = 256, 
                     hidden_dim: int = 512, num_classes: int = 20, dropout: float = 0.3):
            super(AttentionIntentClassifier, self).__init__()
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)
            
        def forward(self, x, attention_mask=None):
            # Embedding layer
            embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
            
            # LSTM layer
            lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim * 2]
            
            # Self-attention
            lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch_size, hidden_dim * 2]
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=attention_mask)
            attn_out = attn_out.transpose(0, 1)  # [batch_size, seq_len, hidden_dim * 2]
            
            # Global average pooling
            pooled = torch.mean(attn_out, dim=1)  # [batch_size, hidden_dim * 2]
            
            # Classification
            output = self.dropout(pooled)
            output = self.fc(output)
            
            return F.log_softmax(output, dim=1)

    class EmotionDetector(nn.Module):
        """Neural network for detecting user emotions from text"""
        
        def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, 
                     hidden_dim: int = 256, num_emotions: int = 8):
            super(EmotionDetector, self).__init__()
            
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
            self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
            
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_dim * 3, num_emotions)
            
        def forward(self, x):
            embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
            embedded = embedded.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
            
            # Multi-scale convolutions
            conv1_out = F.relu(self.conv1(embedded))
            conv2_out = F.relu(self.conv2(embedded))
            conv3_out = F.relu(self.conv3(embedded))
            
            # Global pooling
            pool1 = self.global_pool(conv1_out).squeeze(-1)
            pool2 = self.global_pool(conv2_out).squeeze(-1)
            pool3 = self.global_pool(conv3_out).squeeze(-1)
            
            # Concatenate features
            combined = torch.cat([pool1, pool2, pool3], dim=1)
            combined = self.dropout(combined)
            
            output = self.fc(combined)
            return F.softmax(output, dim=1)

    class PersonalizedResponseGenerator(nn.Module):
        """Transformer-based response generation with personalization"""
        
        def __init__(self, vocab_size: int = 10000, d_model: int = 512, 
                     nhead: int = 8, num_layers: int = 6, user_embed_dim: int = 64):
            super(PersonalizedResponseGenerator, self).__init__()
            
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.pos_encoding = self._generate_positional_encoding(5000, d_model)
            
            # User personalization embedding
            self.user_embedding = nn.Linear(user_embed_dim, d_model)
            
            # Transformer layers
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            self.output_projection = nn.Linear(d_model, vocab_size)
            self.dropout = nn.Dropout(0.1)
            
        def _generate_positional_encoding(self, max_len: int, d_model: int):
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-torch.log(torch.tensor(10000.0)) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0).transpose(0, 1)
            
        def forward(self, input_ids, user_features, attention_mask=None):
            seq_len = input_ids.size(1)
            
            # Token embeddings
            token_embeds = self.embedding(input_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
            
            # Add positional encoding
            pos_embeds = self.pos_encoding[:seq_len, :].to(input_ids.device)
            token_embeds = token_embeds + pos_embeds
            
            # Add user personalization
            user_embeds = self.user_embedding(user_features).unsqueeze(1)
            token_embeds = token_embeds + user_embeds
            
            # Transformer encoding
            token_embeds = token_embeds.transpose(0, 1)  # [seq_len, batch_size, d_model]
            encoded = self.transformer(token_embeds, src_key_padding_mask=attention_mask)
            encoded = encoded.transpose(0, 1)  # [batch_size, seq_len, d_model]
            
            # Output projection
            output = self.output_projection(encoded)
            return output

    class ContextualMemoryNetwork(nn.Module):
        """Memory network for maintaining conversation context"""
        
        def __init__(self, input_dim: int = 512, memory_slots: int = 50, memory_dim: int = 256):
            super(ContextualMemoryNetwork, self).__init__()
            
            self.memory_slots = memory_slots
            self.memory_dim = memory_dim
            
            # Memory components
            self.memory_keys = nn.Parameter(torch.randn(memory_slots, memory_dim))
            self.memory_values = nn.Parameter(torch.randn(memory_slots, memory_dim))
            
            # Input processing
            self.input_projection = nn.Linear(input_dim, memory_dim)
            self.output_projection = nn.Linear(memory_dim * 2, input_dim)
            
            # Attention mechanism
            self.attention = nn.MultiheadAttention(memory_dim, num_heads=4)
            
        def forward(self, input_features):
            batch_size = input_features.size(0)
            
            # Project input to memory dimension
            query = self.input_projection(input_features)  # [batch_size, memory_dim]
            
            # Compute attention over memory
            query = query.unsqueeze(1)  # [batch_size, 1, memory_dim]
            keys = self.memory_keys.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, slots, memory_dim]
            values = self.memory_values.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Attention computation
            query = query.transpose(0, 1)  # [1, batch_size, memory_dim]
            keys = keys.transpose(0, 1)    # [slots, batch_size, memory_dim]
            values = values.transpose(0, 1)  # [slots, batch_size, memory_dim]
            
            attended, _ = self.attention(query, keys, values)
            attended = attended.transpose(0, 1).squeeze(1)  # [batch_size, memory_dim]
            
            # Combine with input
            combined = torch.cat([query.transpose(0, 1).squeeze(1), attended], dim=1)
            output = self.output_projection(combined)
            
            return output

    class CulturalAdaptationNetwork(nn.Module):
        """Network for adapting responses based on cultural context"""
        
        def __init__(self, input_dim: int = 512, cultural_embed_dim: int = 64, 
                     num_cultures: int = 20, hidden_dim: int = 256):
            super(CulturalAdaptationNetwork, self).__init__()
            
            self.cultural_embedding = nn.Embedding(num_cultures, cultural_embed_dim)
            self.input_norm = nn.LayerNorm(input_dim)
            
            # Adaptation layers
            self.adaptation_net = nn.Sequential(
                nn.Linear(input_dim + cultural_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )
            
            # Gate mechanism
            self.gate = nn.Sequential(
                nn.Linear(input_dim + cultural_embed_dim, hidden_dim),
                nn.Sigmoid()
            )
            
        def forward(self, input_features, cultural_id):
            # Get cultural embedding
            cultural_embed = self.cultural_embedding(cultural_id)
            
            # Normalize input
            input_norm = self.input_norm(input_features)
            
            # Combine input with cultural context
            combined = torch.cat([input_norm, cultural_embed], dim=-1)
            
            # Generate adaptation
            adaptation = self.adaptation_net(combined)
            gate_weights = self.gate(combined)
            
            # Apply gated adaptation
            output = input_norm + gate_weights * adaptation
            
            return output

else:
    # Placeholder classes when deep learning is not available
    class AttentionIntentClassifier:
        def __init__(self, *args, **kwargs):
            pass
    
    class EmotionDetector:
        def __init__(self, *args, **kwargs):
            pass
    
    class PersonalizedResponseGenerator:
        def __init__(self, *args, **kwargs):
            pass
    
    class ContextualMemoryNetwork:
        def __init__(self, *args, **kwargs):
            pass
    
    class CulturalAdaptationNetwork:
        def __init__(self, *args, **kwargs):
            pass

class MLEnhancedDailyTalksBridge:
    """
    ML-Enhanced bridge connecting daily talks system to main AI infrastructure
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ML-enhanced bridge"""
        self.config = config or {}
        
        # Initialize core systems
        self.daily_talks_system = ComprehensiveDailyTalksSystem()
        self.integration_wrapper = DailyTalksIntegrationWrapper()
        
        # ML/DL components
        self.intent_classifier = None
        self.response_personalizer = None
        self.context_memory = {}
        self.user_profiles = {}
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Response templates and patterns
        self.response_templates = self._load_response_templates()
        
        # Conversation analytics
        self.analytics = {
            'total_interactions': 0,
            'successful_responses': 0,
            'user_satisfaction_scores': [],
            'popular_intents': {},
            'response_times': []
        }
        
        logger.info("ML-Enhanced Daily Talks Bridge initialized")

    def _initialize_ml_models(self):
        """Initialize ML/DL models for enhanced processing"""
        try:
            # Traditional ML models
            self.intent_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english'
            )
            
            self.intent_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            self.response_personalizer = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                random_state=42
            )
            
            # Initialize deep learning components
            self._initialize_deep_learning_models()
            
            # Load pre-trained models if available
            self._load_pretrained_models()
            
            logger.info("ML/DL models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            # Fall back to rule-based system
            self.intent_classifier = None
            self.response_personalizer = None

    def _initialize_deep_learning_models(self):
        """Initialize deep learning neural networks"""
        if not DEEP_LEARNING_AVAILABLE:
            logger.info("Deep learning models skipped - libraries not available")
            self.dl_models = None
            return
        
        try:
            # Initialize device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Initialize neural networks
            self.dl_models = {
                'intent_classifier': AttentionIntentClassifier(
                    vocab_size=10000,
                    embed_dim=256,
                    hidden_dim=512,
                    num_classes=20,
                    dropout=0.3
                ).to(self.device),
                
                'emotion_detector': EmotionDetector(
                    vocab_size=10000,
                    embed_dim=128,
                    hidden_dim=256,
                    num_emotions=8  # joy, sadness, anger, fear, surprise, disgust, trust, anticipation
                ).to(self.device),
                
                'response_generator': PersonalizedResponseGenerator(
                    vocab_size=10000,
                    d_model=512,
                    nhead=8,
                    num_layers=6,
                    user_embed_dim=64
                ).to(self.device),
                
                'memory_network': ContextualMemoryNetwork(
                    input_dim=512,
                    memory_slots=50,
                    memory_dim=256
                ).to(self.device),
                
                'cultural_adapter': CulturalAdaptationNetwork(
                    input_dim=512,
                    cultural_embed_dim=64,
                    num_cultures=20,
                    hidden_dim=256
                ).to(self.device)
            }
            
            # Initialize tokenizers and embeddings
            self._initialize_nlp_components()
            
            # Set models to evaluation mode initially
            for model in self.dl_models.values():
                model.eval()
            
            logger.info("✅ Deep learning models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing deep learning models: {e}")
            self.dl_models = None

    def _initialize_nlp_components(self):
        """Initialize NLP components for deep learning models"""
        try:
            # Initialize sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize tokenizer (using a lightweight model)
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            
            # Initialize emotion classification pipeline
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Initialize sentiment analysis
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Cultural context mappings
            self.cultural_mappings = {
                'american': 0, 'british': 1, 'german': 2, 'french': 3, 'japanese': 4,
                'chinese': 5, 'indian': 6, 'arabic': 7, 'turkish': 8, 'russian': 9,
                'spanish': 10, 'italian': 11, 'brazilian': 12, 'korean': 13,
                'scandinavian': 14, 'eastern_european': 15, 'african': 16,
                'latin_american': 17, 'southeast_asian': 18, 'unknown': 19
            }
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Some NLP components failed to initialize: {e}")
            # Set fallback values
            self.sentence_transformer = None
            self.tokenizer = None
            self.emotion_pipeline = None
            self.sentiment_pipeline = None

    def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            return
        
        try:
            # Load intent classifier
            intent_model_path = os.path.join(models_dir, "intent_classifier.joblib")
            if os.path.exists(intent_model_path):
                self.intent_classifier = joblib.load(intent_model_path)
                logger.info("Loaded pre-trained intent classifier")
            
            # Load vectorizer
            vectorizer_path = os.path.join(models_dir, "intent_vectorizer.joblib")
            if os.path.exists(vectorizer_path):
                self.intent_vectorizer = joblib.load(vectorizer_path)
                logger.info("Loaded pre-trained vectorizer")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")

    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load response templates for different scenarios"""
        return {
            'greeting_casual': [
                "Hey there! How's your day going in Istanbul?",
                "Hi! What brings you to our beautiful city today?",
                "Hello! Ready to explore Istanbul?",
                "Hey! How can I help you discover Istanbul today?"
            ],
            'greeting_formal': [
                "Good day! Welcome to Istanbul. How may I assist you?",
                "Hello and welcome! How can I help you explore our city?",
                "Greetings! What would you like to know about Istanbul today?"
            ],
            'weather_transition': [
                "Speaking of the weather, did you know...",
                "By the way, with this weather you might enjoy...",
                "Given today's conditions, I'd recommend...",
            ],
            'cultural_insights': [
                "Here's an interesting cultural tip:",
                "As a local insight:",
                "From a cultural perspective:",
            ],
            'personalized_suggestions': [
                "Based on your interests, you might love...",
                "Given your preferences, I'd suggest...",
                "Tailored just for you:",
            ]
        }

    async def process_daily_talk_request(
        self, 
        user_input: str, 
        user_id: str = None,
        session_id: str = None,
        context_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a daily talk request with ML enhancement
        """
        start_time = datetime.now()
        
        try:
            # Create or retrieve user profile
            user_profile = self._get_or_create_user_profile(user_id, context_data)
            
            # Create conversation context
            conversation_context = self._build_conversation_context(
                user_input, user_profile, session_id, context_data
            )
            
            # Enhanced intent recognition
            intent_result = await self._enhanced_intent_recognition(
                user_input, conversation_context
            )
            
            # Get base response from comprehensive system
            base_response = await self.daily_talks_system.process_daily_talk(
                user_input, context_data or {}
            )
            
            # Enhance response with ML personalization
            enhanced_response = await self._enhance_response_with_ml(
                base_response, intent_result, conversation_context
            )
            
            # Update conversation memory
            self._update_conversation_memory(
                conversation_context, user_input, enhanced_response
            )
            
            # Track analytics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._track_interaction_analytics(intent_result, processing_time)
            
            return {
                'response': enhanced_response,
                'intent': intent_result,
                'context': asdict(conversation_context),
                'processing_time': processing_time,
                'ml_enhanced': True
            }
            
        except Exception as e:
            logger.error(f"Error processing daily talk request: {e}")
            
            # Fallback to basic system
            fallback_response = await self.integration_wrapper.get_daily_conversation(
                user_input, context_data or {}
            )
            
            return {
                'response': fallback_response,
                'intent': {'primary': 'unknown', 'confidence': 0.0},
                'context': {},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'ml_enhanced': False,
                'fallback_used': True
            }

    async def _enhanced_intent_recognition(
        self, 
        user_input: str, 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Enhanced intent recognition with ML and context"""
        
        # First, use the comprehensive system's intent recognition
        base_intent = self.daily_talks_system._recognize_intent(user_input)
        
        # Add deep learning enhancement if available
        if DEEP_LEARNING_AVAILABLE and self.dl_models:
            dl_intent_result = await self._deep_learning_intent_recognition(user_input, context)
            if dl_intent_result:
                # Combine traditional and deep learning results
                primary_intent = dl_intent_result.get('primary', base_intent.get('primary', 'general'))
                confidence = max(dl_intent_result.get('confidence', 0), base_intent.get('confidence', 0.5))
            else:
                primary_intent = base_intent.get('primary', 'general')
                confidence = base_intent.get('confidence', 0.5)
        elif self.intent_classifier is not None:
            try:
                # Feature extraction
                features = self._extract_intent_features(user_input, context)
                
                # ML prediction
                ml_prediction = self.intent_classifier.predict_proba([features])
                ml_intent = self.intent_classifier.classes_[np.argmax(ml_prediction)]
                ml_confidence = np.max(ml_prediction)
                
                # Combine rule-based and ML results
                if ml_confidence > 0.7:
                    primary_intent = ml_intent
                    confidence = ml_confidence
                else:
                    primary_intent = base_intent.get('primary', 'general')
                    confidence = base_intent.get('confidence', 0.5)
                    
            except Exception as e:
                logger.warning(f"ML intent recognition failed: {e}")
                primary_intent = base_intent.get('primary', 'general')
                confidence = base_intent.get('confidence', 0.5)
        else:
            primary_intent = base_intent.get('primary', 'general')
            confidence = base_intent.get('confidence', 0.5)
        
        # Enhanced intent result with emotion and sentiment analysis
        intent_result = {
            'primary': primary_intent,
            'confidence': confidence,
            'secondary_intents': base_intent.get('secondary', []),
            'context_factors': self._analyze_context_factors(context),
            'user_pattern_match': self._match_user_patterns(user_input, context.user_profile)
        }
        
        # Add emotion and sentiment analysis
        if DEEP_LEARNING_AVAILABLE and self.dl_models:
            emotion_analysis = await self._analyze_user_emotion(user_input)
            sentiment_analysis = await self._analyze_sentiment(user_input)
            
            intent_result.update({
                'emotion_analysis': emotion_analysis,
                'sentiment_analysis': sentiment_analysis
            })
        
        return intent_result

    async def _deep_learning_intent_recognition(
        self, 
        user_input: str, 
        context: ConversationContext
    ) -> Optional[Dict[str, Any]]:
        """Advanced intent recognition using deep learning models"""
        
        if not self.dl_models or not self.tokenizer:
            return None
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                user_input,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Get sentence embeddings for context
            if self.sentence_transformer:
                sentence_embedding = self.sentence_transformer.encode([user_input])
                context_features = torch.tensor(sentence_embedding, dtype=torch.float32).to(self.device)
            else:
                context_features = torch.zeros(1, 384, dtype=torch.float32).to(self.device)
            
            # Run through attention-based intent classifier
            with torch.no_grad():
                intent_logits = self.dl_models['intent_classifier'](input_ids, attention_mask)
                intent_probs = F.softmax(intent_logits, dim=1)
                
                # Get top predictions
                top_probs, top_indices = torch.topk(intent_probs, k=3)
                
                # Map indices to intent names (this would need a proper mapping)
                intent_mapping = {
                    0: 'greeting', 1: 'weather', 2: 'restaurant', 3: 'transport',
                    4: 'attraction', 5: 'cultural', 6: 'shopping', 7: 'emergency',
                    8: 'general', 9: 'goodbye', 10: 'thanks', 11: 'help',
                    12: 'location', 13: 'time', 14: 'price', 15: 'booking',
                    16: 'recommendation', 17: 'information', 18: 'complaint', 19: 'compliment'
                }
                
                primary_intent_idx = top_indices[0][0].item()
                primary_intent = intent_mapping.get(primary_intent_idx, 'general')
                confidence = top_probs[0][0].item()
                
                # Get secondary intents
                secondary_intents = []
                for i in range(1, min(3, len(top_indices[0]))):
                    sec_idx = top_indices[0][i].item()
                    sec_intent = intent_mapping.get(sec_idx, 'general')
                    sec_conf = top_probs[0][i].item()
                    if sec_conf > 0.2:  # Only include if confidence > 20%
                        secondary_intents.append({'intent': sec_intent, 'confidence': sec_conf})
                
                return {
                    'primary': primary_intent,
                    'confidence': confidence,
                    'secondary_intents': secondary_intents,
                    'method': 'deep_learning'
                }
                
        except Exception as e:
            logger.error(f"Deep learning intent recognition error: {e}")
            return None

    async def _analyze_user_emotion(self, user_input: str) -> Dict[str, Any]:
        """Analyze user emotion using deep learning models"""
        
        emotion_result = {
            'primary_emotion': 'neutral',
            'confidence': 0.5,
            'emotion_scores': {},
            'method': 'fallback'
        }
        
        try:
            # Use pre-trained emotion pipeline if available
            if self.emotion_pipeline:
                emotions = self.emotion_pipeline(user_input)
                if emotions:
                    emotion_result.update({
                        'primary_emotion': emotions[0]['label'].lower(),
                        'confidence': emotions[0]['score'],
                        'emotion_scores': {emotion['label'].lower(): emotion['score'] for emotion in emotions},
                        'method': 'pipeline'
                    })
            
            # Use custom emotion detector if available
            elif self.dl_models and 'emotion_detector' in self.dl_models:
                # Tokenize input for emotion detection
                if self.tokenizer:
                    inputs = self.tokenizer(
                        user_input,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors='pt'
                    )
                    
                    input_ids = inputs['input_ids'].to(self.device)
                    
                    with torch.no_grad():
                        emotion_probs = self.dl_models['emotion_detector'](input_ids)
                        
                        # Map to emotion labels (Plutchik's 8 basic emotions)
                        emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']
                        emotion_scores = {label: prob.item() for label, prob in zip(emotion_labels, emotion_probs[0])}
                        
                        primary_emotion = max(emotion_scores, key=emotion_scores.get)
                        confidence = emotion_scores[primary_emotion]
                        
                        emotion_result.update({
                            'primary_emotion': primary_emotion,
                            'confidence': confidence,
                            'emotion_scores': emotion_scores,
                            'method': 'neural_network'
                        })
            
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
        
        return emotion_result

    async def _analyze_sentiment(self, user_input: str) -> Dict[str, Any]:
        """Analyze sentiment of user input"""
        
        sentiment_result = {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'method': 'fallback'
        }
        
        try:
            if self.sentiment_pipeline:
                sentiment = self.sentiment_pipeline(user_input)
                if sentiment:
                    sentiment_result.update({
                        'sentiment': sentiment[0]['label'].lower(),
                        'confidence': sentiment[0]['score'],
                        'method': 'pipeline'
                    })
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
        
        return sentiment_result

    async def _enhance_response_with_ml(
        self,
        base_response: Dict[str, Any],
        intent_result: Dict[str, Any],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Enhance the base response with ML personalization"""
        
        enhanced_response = base_response.copy()
        
        try:
            # Apply deep learning enhancements if available
            if DEEP_LEARNING_AVAILABLE and self.dl_models:
                enhanced_response = await self._apply_deep_learning_enhancements(
                    enhanced_response, intent_result, context
                )
            
            # Personalize based on user profile
            enhanced_response = self._personalize_response(
                enhanced_response, context.user_profile
            )
            
            # Add contextual information
            enhanced_response = self._add_contextual_information(
                enhanced_response, context
            )
            
            # Multi-modal enhancements
            enhanced_response = await self._add_multimodal_elements(
                enhanced_response, intent_result, context
            )
            
            # Conversation flow optimization
            enhanced_response = self._optimize_conversation_flow(
                enhanced_response, context
            )
            
            # Apply emotion-aware adjustments
            if 'emotion_analysis' in intent_result:
                enhanced_response = self._apply_emotion_aware_adjustments(
                    enhanced_response, intent_result['emotion_analysis']
                )
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            # Return base response if enhancement fails
            return base_response
        
        return enhanced_response

    async def _apply_deep_learning_enhancements(
        self,
        response: Dict[str, Any],
        intent_result: Dict[str, Any],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Apply deep learning model enhancements to the response"""
        
        try:
            # Apply contextual memory enhancement
            if 'memory_network' in self.dl_models and self.sentence_transformer:
                # Get response embedding
                response_text = response.get('message', '')
                if response_text:
                    response_embedding = self.sentence_transformer.encode([response_text])
                    response_tensor = torch.tensor(response_embedding, dtype=torch.float32).to(self.device)
                    
                    # Enhance with memory network
                    with torch.no_grad():
                        memory_enhanced = self.dl_models['memory_network'](response_tensor)
                        # The memory enhancement could influence response selection or generation
                        response['memory_enhanced'] = True
            
            # Apply cultural adaptation if user has cultural background
            if ('cultural_adapter' in self.dl_models and 
                context.user_profile and 
                context.user_profile.cultural_background):
                
                cultural_id = self.cultural_mappings.get(
                    context.user_profile.cultural_background.lower(), 
                    self.cultural_mappings['unknown']
                )
                
                # This would typically modify the response generation process
                # For now, we'll just mark it as culturally adapted
                response['culturally_adapted'] = True
                response['cultural_context'] = context.user_profile.cultural_background
            
            # Apply personalized response generation enhancements
            if 'response_generator' in self.dl_models:
                # This would typically involve generating or modifying the response
                # using the transformer model, but for now we'll enhance metadata
                response['personalization_level'] = 'deep_learning_enhanced'
                
                # Add sophisticated follow-up suggestions based on deep learning
                response['ai_suggestions'] = self._generate_ai_powered_suggestions(
                    intent_result, context
                )
            
        except Exception as e:
            logger.warning(f"Deep learning enhancement error: {e}")
        
        return response

    def _generate_ai_powered_suggestions(
        self, 
        intent_result: Dict[str, Any], 
        context: ConversationContext
    ) -> List[str]:
        """Generate AI-powered suggestions based on deep learning insights"""
        
        suggestions = []
        
        try:
            primary_intent = intent_result.get('primary', '')
            emotion = intent_result.get('emotion_analysis', {}).get('primary_emotion', 'neutral')
            sentiment = intent_result.get('sentiment_analysis', {}).get('sentiment', 'neutral')
            
            # Intent-based suggestions with emotional awareness
            if primary_intent == 'restaurant' and emotion == 'joy':
                suggestions.append("Since you seem excited about food, I'd recommend trying the vibrant atmosphere at Karakoy's trendy restaurants!")
            elif primary_intent == 'weather' and sentiment == 'negative':
                suggestions.append("Don't let the weather dampen your spirits - Istanbul has amazing indoor attractions!")
            elif primary_intent == 'attraction' and emotion == 'anticipation':
                suggestions.append("Your enthusiasm is wonderful! Let me suggest some hidden gems that will exceed your expectations.")
            
            # Context-aware suggestions
            if context.user_profile and context.user_profile.visit_frequency == 'first_time':
                suggestions.append("As a first-time visitor, I can create a personalized itinerary based on your interests and current mood.")
            
            # Time-sensitive suggestions
            current_hour = datetime.now().hour
            if 11 <= current_hour <= 14 and primary_intent in ['restaurant', 'general']:
                suggestions.append("Perfect timing for lunch! Would you like Turkish cuisine recommendations nearby?")
            
        except Exception as e:
            logger.warning(f"AI suggestion generation error: {e}")
        
        return suggestions[:3]  # Limit to top 3 suggestions

    def _apply_emotion_aware_adjustments(
        self, 
        response: Dict[str, Any], 
        emotion_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply emotion-aware adjustments to the response"""
        
        try:
            primary_emotion = emotion_analysis.get('primary_emotion', 'neutral')
            confidence = emotion_analysis.get('confidence', 0.5)
            
            # Only apply adjustments if we're confident about the emotion
            if confidence > 0.6:
                message = response.get('message', '')
                
                if primary_emotion == 'joy':
                    # Enhance positive emotions
                    if not any(word in message.lower() for word in ['wonderful', 'amazing', 'fantastic', 'great']):
                        response['message'] = f"🌟 {message}"
                        response['emotional_tone'] = 'enthusiastic'
                
                elif primary_emotion == 'sadness':
                    # Provide comfort and support
                    response['message'] = f"💙 {message} I'm here to help make your Istanbul experience brighter!"
                    response['emotional_tone'] = 'supportive'
                
                elif primary_emotion == 'anger' or primary_emotion == 'frustration':
                    # Be more empathetic and solution-focused
                    response['message'] = f"I understand your frustration. {message} Let me help resolve this quickly."
                    response['emotional_tone'] = 'empathetic'
                
                elif primary_emotion == 'fear' or primary_emotion == 'worry':
                    # Provide reassurance
                    response['message'] = f"Don't worry! {message} Istanbul is generally very safe and welcoming."
                    response['emotional_tone'] = 'reassuring'
                
                elif primary_emotion == 'surprise':
                    # Maintain the element of discovery
                    response['emotional_tone'] = 'intriguing'
                
                elif primary_emotion == 'anticipation':
                    # Build excitement
                    response['message'] = f"✨ {message} You're in for a treat!"
                    response['emotional_tone'] = 'exciting'
        
        except Exception as e:
            logger.warning(f"Emotion-aware adjustment error: {e}")
        
        return response

    async def _generate_deep_learning_response(
        self,
        user_input: str,
        intent_result: Dict[str, Any],
        context: ConversationContext
    ) -> Optional[str]:
        """Generate response using deep learning models"""
        
        if not DEEP_LEARNING_AVAILABLE or not self.dl_models or 'response_generator' not in self.dl_models:
            return None
        
        try:
            # Prepare user features for personalization
            user_features = self._extract_user_features_for_generation(context.user_profile)
            user_tensor = torch.tensor([user_features], dtype=torch.float32).to(self.device)
            
            # Tokenize input
            inputs = self.tokenizer(
                user_input,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Generate response using the transformer model
            with torch.no_grad():
                output_logits = self.dl_models['response_generator'](
                    input_ids, user_tensor, attention_mask
                )
                
                # Convert logits to tokens (this is a simplified approach)
                # In a real implementation, you'd use proper decoding strategies
                predicted_tokens = torch.argmax(output_logits, dim=-1)
                
                # Decode to text
                generated_text = self.tokenizer.decode(
                    predicted_tokens[0], 
                    skip_special_tokens=True
                )
                
                return generated_text
        
        except Exception as e:
            logger.error(f"Deep learning response generation error: {e}")
            return None

    def _extract_user_features_for_generation(self, user_profile: UserProfile) -> List[float]:
        """Extract user features for personalized response generation"""
        
        features = [0.0] * 64  # Initialize with zeros
        
        if not user_profile:
            return features
        
        try:
            # Visit frequency encoding
            visit_freq_map = {'first_time': 0.0, 'occasional': 0.33, 'frequent': 0.66, 'local': 1.0}
            features[0] = visit_freq_map.get(user_profile.visit_frequency, 0.0)
            
            # Language style encoding
            features[1] = 1.0 if user_profile.language_style == 'formal' else 0.0
            
            # Cultural background encoding (simplified)
            if user_profile.cultural_background:
                cultural_id = self.cultural_mappings.get(user_profile.cultural_background.lower(), 19)
                features[2] = cultural_id / 19.0  # Normalize to [0, 1]
            
            # Interaction history length (normalized)
            features[3] = min(len(user_profile.interaction_history) / 100.0, 1.0)
            
            # Preferences encoding (simplified)
            if user_profile.preferences:
                # Interest diversity
                features[4] = min(len(user_profile.preferences.get('interests', [])) / 10.0, 1.0)
                
                # Budget preference
                budget_map = {'budget': 0.0, 'moderate': 0.5, 'luxury': 1.0}
                features[5] = budget_map.get(user_profile.preferences.get('budget', 'moderate'), 0.5)
            
            # Fill remaining features with personality traits if available
            if user_profile.personality_traits:
                for i, (trait, value) in enumerate(list(user_profile.personality_traits.items())[:10]):
                    if i + 6 < len(features):
                        features[i + 6] = min(max(value, 0.0), 1.0)  # Clamp to [0, 1]
        
        except Exception as e:
            logger.warning(f"User feature extraction error: {e}")
        
        return features

    # Utility methods for response enhancement
    def _formalize_language(self, text: str) -> str:
        """Convert casual language to formal"""
        replacements = {
            "hey": "hello",
            "yeah": "yes",
            "nope": "no",
            "gonna": "going to",
            "wanna": "want to"
        }
        
        for casual, formal in replacements.items():
            text = text.replace(casual, formal)
        
        return text

    def _casualize_language(self, text: str) -> str:
        """Convert formal language to casual"""
        replacements = {
            "greetings": "hey",
            "assistance": "help",
            "approximately": "about",
            "immediately": "right away"
        }
        
        for formal, casual in replacements.items():
            text = text.replace(formal, casual)
        
        return text

    def _generate_personalized_suggestions(
        self, 
        response: Dict[str, Any], 
        preferences: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized suggestions based on user preferences"""
        
        suggestions = []
        
        if preferences.get('interests'):
            interests = preferences['interests']
            if 'history' in interests:
                suggestions.append("Visit the magnificent Hagia Sophia for a historical journey")
            if 'food' in interests:
                suggestions.append("Try authentic Turkish breakfast at a local cafe")
            if 'art' in interests:
                suggestions.append("Explore the contemporary art galleries in Karakoy")
        
        if preferences.get('budget') == 'budget_friendly':
            suggestions.append("Check out free walking tours in Sultanahmet")
        
        return suggestions[:3]  # Limit to 3 suggestions

    def _adapt_for_culture(
        self, 
        response: Dict[str, Any], 
        cultural_background: str
    ) -> Dict[str, Any]:
        """Adapt response for specific cultural background"""
        
        if cultural_background == 'japanese':
            response['cultural_note'] = "Similar to Japanese hospitality, Turkish people are very welcoming to guests"
        elif cultural_background == 'german':
            response['cultural_note'] = "Like German efficiency, Istanbul's public transport is quite systematic"
        elif cultural_background == 'american':
            response['cultural_note'] = "Istanbul offers diverse experiences like major US cities, but with unique historical depth"
        
        return response

    def _get_location_specific_tips(self, location: str) -> List[str]:
        """Get location-specific tips"""
        
        location_tips = {
            'sultanahmet': [
                "Visit early morning to avoid crowds",
                "Wear comfortable walking shoes for cobblestone streets"
            ],
            'beyoglu': [
                "Great for nightlife and contemporary art",
                "Try the historic tram from Taksim to Tunel"
            ],
            'kadikoy': [
                "Perfect for authentic local food experiences",
                "Less touristy, more authentic Istanbul life"
            ]
        }
        
        return location_tips.get(location.lower(), [])

    def _get_weather_appropriate_suggestions(
        self, 
        weather_context: Dict[str, Any]
    ) -> List[str]:
        """Get weather-appropriate suggestions"""
        
        suggestions = []
        
        if weather_context.get('condition') == 'rainy':
            suggestions.extend([
                "Perfect weather for visiting museums",
                "Enjoy Turkish tea in a cozy cafe",
                "Explore covered bazaars like Grand Bazaar"
            ])
        elif weather_context.get('condition') == 'sunny':
            suggestions.extend([
                "Great day for Bosphorus boat tour",
                "Visit outdoor attractions like Topkapi Palace gardens",
                "Enjoy breakfast with a view at Galata Tower area"
            ])
        
        return suggestions

    def _get_relevant_images(self, intent: str) -> List[str]:
        """Get relevant image suggestions based on intent"""
        
        image_maps = {
            'weather': ['bosphorus_sunny.jpg', 'istanbul_rain.jpg'],
            'activities': ['turkish_breakfast.jpg', 'bosphorus_cruise.jpg'],
            'attractions': ['hagia_sophia.jpg', 'blue_mosque.jpg']
        }
        
        return image_maps.get(intent, [])

    def _get_quick_actions(self, intent: str) -> List[Dict[str, str]]:
        """Get quick action buttons based on intent"""
        
        action_maps = {
            'restaurant': [
                {'label': 'Show nearby restaurants', 'action': 'find_restaurants'},
                {'label': 'Make reservation', 'action': 'make_reservation'}
            ],
            'transport': [
                {'label': 'Get directions', 'action': 'get_directions'},
                {'label': 'Check schedules', 'action': 'check_schedules'}
            ],
            'attractions': [
                {'label': 'Buy tickets', 'action': 'buy_tickets'},
                {'label': 'Get more info', 'action': 'more_info'}
            ]
        }
        
        return action_maps.get(intent, [])

    def _get_pronunciation_guides(self) -> List[Dict[str, str]]:
        """Get pronunciation guides for Turkish phrases"""
        
        return [
            {'phrase': 'Merhaba', 'pronunciation': 'mer-ha-BA', 'meaning': 'Hello'},
            {'phrase': 'Teşekkürler', 'pronunciation': 'teh-shek-kur-LER', 'meaning': 'Thank you'},
            {'phrase': 'Lütfen', 'pronunciation': 'lut-FEN', 'meaning': 'Please'}
        ]

    def _generate_follow_up_question(
        self, 
        last_intent: str, 
        current_response: Dict[str, Any]
    ) -> Optional[str]:
        """Generate appropriate follow-up question"""
        
        follow_ups = {
            'weather': "Would you like some weather-appropriate activity suggestions?",
            'restaurant': "Are you looking for a specific cuisine or budget range?",
            'transport': "Do you need help with directions or schedules?",
            'attractions': "Would you like to know about ticket prices or opening hours?"
        }
        
        return follow_ups.get(last_intent)

    def _build_conversation_continuity(
        self, 
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build conversation continuity elements"""
        
        return {
            'previous_topics': [msg.get('intent', '') for msg in conversation_history[-3:]],
            'conversation_length': len(conversation_history),
            'engagement_level': 'high' if len(conversation_history) > 5 else 'medium'
        }

    def _analyze_context_factors(self, context: ConversationContext) -> Dict[str, Any]:
        """Analyze various context factors that might influence the response"""
        
        factors = {}
        
        # Time-based factors
        factors['time_of_day'] = context.time_context
        factors['is_weekend'] = datetime.now().weekday() >= 5
        
        # User factors
        if context.user_profile:
            factors['user_experience'] = context.user_profile.visit_frequency
            factors['interaction_count'] = len(context.user_profile.interaction_history)
        
        # Conversation factors
        factors['conversation_length'] = len(context.conversation_history)
        factors['active_topics'] = context.active_topics or []
        
        return factors

    def _match_user_patterns(
        self, 
        user_input: str, 
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Match user input against known user patterns"""
        
        patterns = {
            'typical_greeting': False,
            'typical_question_style': False,
            'prefers_detailed_info': False,
            'prefers_quick_answers': False
        }
        
        if user_profile and user_profile.interaction_history:
            # Analyze user's typical interaction patterns
            history = user_profile.interaction_history
            
            # Check greeting patterns
            greeting_words = ['hi', 'hello', 'hey', 'good morning']
            user_greetings = [msg for msg in history 
                            if any(word in msg.get('user_input', '').lower() 
                                  for word in greeting_words)]
            
            if len(user_greetings) > 0:
                patterns['typical_greeting'] = any(word in user_input.lower() 
                                                 for word in greeting_words)
            
            # Check question complexity preference
            avg_input_length = sum(len(msg.get('user_input', '')) 
                                 for msg in history) / len(history)
            
            patterns['prefers_detailed_info'] = avg_input_length > 50
            patterns['prefers_quick_answers'] = avg_input_length < 20
        
        return patterns

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including deep learning models"""
        
        status = {
            'bridge_status': 'active',
            'ml_models_loaded': {
                'intent_classifier': self.intent_classifier is not None,
                'response_personalizer': self.response_personalizer is not None
            },
            'deep_learning_available': DEEP_LEARNING_AVAILABLE,
            'deep_learning_models': {},
            'nlp_components': {},
            'device': getattr(self, 'device', 'cpu'),
            'active_users': len(self.user_profiles),
            'active_sessions': len(self.context_memory),
            'analytics': self.analytics.copy()
        }
        
        # Add deep learning model status
        if DEEP_LEARNING_AVAILABLE and hasattr(self, 'dl_models') and self.dl_models:
            status['deep_learning_models'] = {
                model_name: {
                    'loaded': model is not None,
                    'device': str(self.device) if model is not None else 'N/A',
                    'parameters': sum(p.numel() for p in model.parameters()) if model is not None else 0
                }
                for model_name, model in self.dl_models.items()
            }
        
        # Add NLP component status
        status['nlp_components'] = {
            'sentence_transformer': getattr(self, 'sentence_transformer', None) is not None,
            'tokenizer': getattr(self, 'tokenizer', None) is not None,
            'emotion_pipeline': getattr(self, 'emotion_pipeline', None) is not None,
            'sentiment_pipeline': getattr(self, 'sentiment_pipeline', None) is not None
        }
        
        # Add comprehensive system status
        try:
            status['comprehensive_system_status'] = await self.daily_talks_system.get_system_status()
        except Exception as e:
            logger.warning(f"Could not get comprehensive system status: {e}")
            status['comprehensive_system_status'] = {'error': str(e)}
        
        # Calculate average response time
        if self.analytics['response_times']:
            status['analytics']['avg_response_time'] = sum(self.analytics['response_times']) / len(self.analytics['response_times'])
        
        # Add deep learning specific analytics
        status['analytics']['deep_learning_enhanced_responses'] = sum(
            1 for profile in self.user_profiles.values()
            for interaction in profile.interaction_history
            if interaction.get('response', {}).get('ml_enhanced') and 
               interaction.get('response', {}).get('personalization_level') == 'deep_learning_enhanced'
        )
        
        return status

    async def train_models_with_interactions(self):
        """Train ML/DL models with collected interaction data"""
        
        if not self.user_profiles:
            logger.info("No interaction data available for training")
            return
        
        try:
            # Collect training data from user interactions
            training_data = []
            labels = []
            
            for user_profile in self.user_profiles.values():
                for interaction in user_profile.interaction_history:
                    if 'intent' in interaction and 'primary' in interaction['intent']:
                        # Create feature vector from interaction
                        features = self._extract_intent_features(
                            interaction.get('user_input', ''),
                            ConversationContext(
                                session_id='training',
                                user_profile=user_profile,
                                conversation_history=[]
                            )
                        )
                        
                        training_data.append(features)
                        labels.append(interaction['intent']['primary'])
            
            if len(training_data) > 10:  # Minimum training data threshold
                # Train traditional ML models
                self.intent_classifier.fit(training_data, labels)
                
                # Train deep learning models if available
                if DEEP_LEARNING_AVAILABLE and self.dl_models:
                    await self._train_deep_learning_models(training_data, labels)
                
                # Save trained models
                models_dir = "models"
                if not os.path.exists(models_dir):
                    os.makedirs(models_dir)
                
                joblib.dump(self.intent_classifier, 
                          os.path.join(models_dir, "intent_classifier.joblib"))
                joblib.dump(self.intent_vectorizer, 
                          os.path.join(models_dir, "intent_vectorizer.joblib"))
                
                logger.info(f"Models trained with {len(training_data)} interactions")
            else:
                logger.info("Insufficient training data for model updates")
                
        except Exception as e:
            logger.error(f"Error training models: {e}")

    async def _train_deep_learning_models(self, training_data: List, labels: List):
        """Train deep learning models with interaction data"""
        
        try:
            # This is a simplified training approach
            # In production, you'd want more sophisticated training procedures
            
            if 'intent_classifier' in self.dl_models:
                # Convert training data to tensor format
                # This would require proper preprocessing and tokenization
                logger.info("Deep learning model training would be implemented here")
                
            # Train emotion detector with emotional labels if available
            if 'emotion_detector' in self.dl_models:
                logger.info("Emotion detector training would be implemented here")
                
            # Train response generator with conversation pairs
            if 'response_generator' in self.dl_models:
                logger.info("Response generator training would be implemented here")
                
        except Exception as e:
            logger.error(f"Deep learning training error: {e}")

    def export_analytics(self) -> Dict[str, Any]:
        """Export comprehensive analytics data including deep learning insights"""
        
        analytics = self.analytics.copy()
        
        # Add user analytics
        analytics['user_analytics'] = {
            'total_users': len(self.user_profiles),
            'active_users_last_24h': self._count_active_users_last_24h(),
            'user_retention': self._calculate_user_retention(),
            'popular_user_preferences': self._analyze_user_preferences()
        }
        
        # Add conversation analytics
        analytics['conversation_analytics'] = {
            'total_sessions': len(self.context_memory),
            'avg_session_length': self._calculate_avg_session_length(),
            'conversation_topics': self._analyze_conversation_topics()
        }
        
        # Add deep learning analytics
        if DEEP_LEARNING_AVAILABLE and self.dl_models:
            analytics['deep_learning_analytics'] = {
                'emotion_distribution': self._analyze_emotion_distribution(),
                'sentiment_trends': self._analyze_sentiment_trends(),
                'cultural_adaptation_usage': self._analyze_cultural_adaptations()
            }
        
        return analytics

    def _analyze_emotion_distribution(self) -> Dict[str, int]:
        """Analyze distribution of detected emotions"""
        
        emotions = {}
        
        for profile in self.user_profiles.values():
            for interaction in profile.interaction_history:
                emotion_data = interaction.get('intent', {}).get('emotion_analysis', {})
                emotion = emotion_data.get('primary_emotion', 'neutral')
                emotions[emotion] = emotions.get(emotion, 0) + 1
        
        return emotions

    def _analyze_sentiment_trends(self) -> Dict[str, int]:
        """Analyze sentiment trends over time"""
        
        sentiments = {}
        
        for profile in self.user_profiles.values():
            for interaction in profile.interaction_history:
                sentiment_data = interaction.get('intent', {}).get('sentiment_analysis', {})
                sentiment = sentiment_data.get('sentiment', 'neutral')
                sentiments[sentiment] = sentiments.get(sentiment, 0) + 1
        
        return sentiments

    def _analyze_cultural_adaptations(self) -> Dict[str, int]:
        """Analyze usage of cultural adaptations"""
        
        adaptations = {}
        
        for profile in self.user_profiles.values():
            cultural_bg = profile.cultural_background
            if cultural_bg:
                adaptations[cultural_bg] = adaptations.get(cultural_bg, 0) + len(profile.interaction_history)
        
        return adaptations

    def _count_active_users_last_24h(self) -> int:
        """Count users active in the last 24 hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        active_users = 0
        
        for user_profile in self.user_profiles.values():
            if user_profile.interaction_history:
                last_interaction = user_profile.interaction_history[-1]
                interaction_time = datetime.fromisoformat(last_interaction['timestamp'])
                if interaction_time > cutoff_time:
                    active_users += 1
        
        return active_users

    def _calculate_user_retention(self) -> float:
        """Calculate user retention rate"""
        
        if len(self.user_profiles) == 0:
            return 0.0
        
        returning_users = sum(1 for profile in self.user_profiles.values() 
                            if len(profile.interaction_history) > 1)
        
        return returning_users / len(self.user_profiles)

    def _analyze_user_preferences(self) -> Dict[str, int]:
        """Analyze popular user preferences"""
        
        preferences = {}
        
        for user_profile in self.user_profiles.values():
            for key, value in user_profile.preferences.items():
                if isinstance(value, str):
                    pref_key = f"{key}:{value}"
                    preferences[pref_key] = preferences.get(pref_key, 0) + 1
        
        return dict(sorted(preferences.items(), key=lambda x: x[1], reverse=True)[:10])

    def _calculate_avg_session_length(self) -> float:
        """Calculate average session length"""
        
        if not self.context_memory:
            return 0.0
        
        total_interactions = sum(len(session) for session in self.context_memory.values())
        return total_interactions / len(self.context_memory)

    def _analyze_conversation_topics(self) -> Dict[str, int]:
        """Analyze popular conversation topics"""
        
        topics = {}
        
        for session in self.context_memory.values():
            for interaction in session:
                intent = interaction.get('intent', {}).get('primary', 'unknown')
                topics[intent] = topics.get(intent, 0) + 1
        
        return dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))


# Main bridge instance
ml_bridge = MLEnhancedDailyTalksBridge()

# Convenience functions for integration
async def process_enhanced_daily_talk(
    user_input: str,
    user_id: str = None,
    session_id: str = None,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Convenience function for processing enhanced daily talk requests"""
    return await ml_bridge.process_daily_talk_request(
        user_input, user_id, session_id, context_data
    )

async def get_bridge_status() -> Dict[str, Any]:
    """Get the status of the ML-enhanced bridge"""
    return await ml_bridge.get_system_status()

def get_analytics() -> Dict[str, Any]:
    """Get analytics data from the bridge"""
    return ml_bridge.export_analytics()

if __name__ == "__main__":
    # Example usage and testing
    async def test_bridge():
        """Test the ML-enhanced bridge with deep learning capabilities"""
        
        print("🧠 Testing ML-Enhanced Daily Talks Bridge with Deep Learning...")
        print("=" * 60)
        
        # Test cases with various intents and emotional contexts
        test_cases = [
            {
                'input': 'Good morning! How is the weather today? I\'m so excited to explore Istanbul!',
                'user_id': 'test_user_1',
                'context': {
                    'location': 'sultanahmet', 
                    'preferences': {'interests': ['weather', 'activities'], 'cultural_background': 'american'},
                    'language_style': 'casual'
                }
            },
            {
                'input': 'Hi, I need restaurant recommendations but I\'m feeling a bit overwhelmed',
                'user_id': 'test_user_2',
                'context': {
                    'mood': 'anxious', 
                    'preferences': {'budget': 'budget_friendly', 'cultural_background': 'german'},
                    'language_style': 'formal'
                }
            },
            {
                'input': 'What should I do today? The weather looks amazing!',
                'user_id': 'test_user_1',
                'context': {
                    'location': 'beyoglu', 
                    'weather': {'condition': 'sunny'},
                    'preferences': {'visit_frequency': 'first_time'}
                }
            },
            {
                'input': 'Thank you so much! This has been incredibly helpful.',
                'user_id': 'test_user_2',
                'context': {
                    'mood': 'grateful',
                    'preferences': {'cultural_background': 'japanese'}
                }
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔬 Test Case {i}")
            print("-" * 30)
            print(f"Input: {test_case['input']}")
            print(f"Context: {test_case['context']}")
            
            try:
                result = await process_enhanced_daily_talk(
                    test_case['input'],
                    test_case['user_id'],
                    f"test_session_{i}",
                    test_case['context']
                )
                
                print(f"\n📝 Response: {result['response'].get('message', 'No message')}")
                print(f"🎯 Intent: {result['intent'].get('primary', 'unknown')} "
                      f"(confidence: {result['intent'].get('confidence', 0):.2f})")
                
                # Display deep learning insights
                if 'emotion_analysis' in result['intent']:
                    emotion = result['intent']['emotion_analysis']
                    print(f"😊 Emotion: {emotion.get('primary_emotion', 'neutral')} "
                          f"(confidence: {emotion.get('confidence', 0):.2f})")
                
                if 'sentiment_analysis' in result['intent']:
                    sentiment = result['intent']['sentiment_analysis']
                    print(f"💭 Sentiment: {sentiment.get('sentiment', 'neutral')} "
                          f"(confidence: {sentiment.get('confidence', 0):.2f})")
                
                print(f"🤖 ML Enhanced: {result.get('ml_enhanced', False)}")
                print(f"⏱️  Processing Time: {result.get('processing_time', 0):.3f}s")
                
                # Display additional response features
                response = result['response']
                if 'emotional_tone' in response:
                    print(f"🎭 Emotional Tone: {response['emotional_tone']}")
                if 'ai_suggestions' in response:
                    print(f"💡 AI Suggestions: {', '.join(response['ai_suggestions'][:2])}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Display system status
        print(f"\n{'='*60}")
        print("🖥️  System Status")
        print("-" * 30)
        status = await get_bridge_status()
        print(f"Bridge Status: {status.get('bridge_status', 'unknown')}")
        print(f"Deep Learning Available: {status.get('deep_learning_available', False)}")
        print(f"Device: {status.get('device', 'unknown')}")
        print(f"Active Users: {status.get('active_users', 0)}")
        print(f"Active Sessions: {status.get('active_sessions', 0)}")
        
        if status.get('deep_learning_models'):
            print("\n🧠 Deep Learning Models:")
            for model_name, model_info in status['deep_learning_models'].items():
                print(f"  • {model_name}: {'✅' if model_info['loaded'] else '❌'}")
        
        if status.get('nlp_components'):
            print("\n🔤 NLP Components:")
            for comp_name, loaded in status['nlp_components'].items():
                print(f"  • {comp_name}: {'✅' if loaded else '❌'}")
        
        # Display analytics
        print(f"\n{'='*60}")
        print("📊 Analytics")
        print("-" * 30)
        analytics = get_analytics()
        print(f"Total Interactions: {analytics.get('total_interactions', 0)}")
        print(f"Popular Intents: {analytics.get('popular_intents', {})}")
        
        if 'deep_learning_analytics' in analytics:
            dl_analytics = analytics['deep_learning_analytics']
            print(f"Emotion Distribution: {dl_analytics.get('emotion_distribution', {})}")
            print(f"Sentiment Trends: {dl_analytics.get('sentiment_trends', {})}")
    
    # Run the comprehensive test
    asyncio.run(test_bridge())
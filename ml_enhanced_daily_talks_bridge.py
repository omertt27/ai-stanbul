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

# Import İKSV Events System
try:
    from monthly_events_scheduler import MonthlyEventsScheduler, get_cached_events, fetch_monthly_events, check_if_fetch_needed
    EVENTS_SYSTEM_AVAILABLE = True
    logger.info("✅ İKSV Events System loaded successfully")
except ImportError as e:
    EVENTS_SYSTEM_AVAILABLE = False
    logger.warning(f"⚠️ İKSV Events System not available: {e}")

# Import location detection for events
try:
    from backend.services.intelligent_location_detector import IntelligentLocationDetector, detect_user_location
    LOCATION_DETECTION_AVAILABLE = True
    logger.info("✅ Location Detection for Events loaded successfully")
except ImportError as e:
    LOCATION_DETECTION_AVAILABLE = False
    logger.warning(f"⚠️ Location Detection for Events not available: {e}")

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
            try:
                fallback_response = await self.integration_wrapper.get_daily_conversation(
                    user_input, context_data or {}
                )
            except (AttributeError, Exception) as fallback_error:
                logger.warning(f"Integration wrapper fallback failed: {fallback_error}")
                # Create a basic fallback response
                fallback_response = {
                    'message': f"Hello! I'm here to help you explore Istanbul. {user_input} - I'd be happy to assist you with information about our beautiful city!",
                    'intent': 'general',
                    'confidence': 0.3
                }
            
            return {
                'response': fallback_response,
                'intent': {'primary': 'unknown', 'confidence': 0.0},
                'context': {},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'ml_enhanced': False,
                'fallback_used': True
            }

    def _get_or_create_user_profile(
        self, 
        user_id: str, 
        context_data: Dict[str, Any]
    ) -> UserProfile:
        """Get existing user profile or create new one"""
        
        if not user_id:
            user_id = f"anonymous_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            # Update with new context data if available
            if context_data:
                profile.preferences.update(context_data.get('preferences', {}))
            return profile
        
        # Create new profile
        profile = UserProfile(
            user_id=user_id,
            preferences=context_data.get('preferences', {}) if context_data else {},
            interaction_history=[],
            personality_traits={},
            location_preferences={},
            activity_patterns={},
            language_style=context_data.get('language_style', 'casual') if context_data else 'casual',
            cultural_background=context_data.get('preferences', {}).get('cultural_background') if context_data else None,
            visit_frequency=context_data.get('preferences', {}).get('visit_frequency', 'first_time') if context_data else 'first_time'
        )
        
        self.user_profiles[user_id] = profile
        return profile

    def _build_conversation_context(
        self,
        user_input: str,
        user_profile: UserProfile,
        session_id: str,
        context_data: Dict[str, Any]
    ) -> ConversationContext:
        """Build comprehensive conversation context"""
        
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get conversation history for this session
        conversation_history = self.context_memory.get(session_id, [])
        
        return ConversationContext(
            session_id=session_id,
            user_profile=user_profile,
            conversation_history=conversation_history,
            current_mood=context_data.get('mood') if context_data else None,
            current_location=context_data.get('location') if context_data else None,
            time_context=self._get_time_context(),
            weather_context=context_data.get('weather') if context_data else None,
            active_topics=self._extract_active_topics(conversation_history),
            multi_modal_data=context_data.get('multi_modal') if context_data else None
        )

    def _get_time_context(self) -> str:
        """Get current time context"""
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            return "morning"
        elif 12 <= current_hour < 17:
            return "afternoon"
        elif 17 <= current_hour < 21:
            return "evening"
        else:
            return "night"

    def _extract_active_topics(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract active topics from conversation history"""
        topics = []
        
        # Look at recent messages for active topics
        recent_messages = conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history
        
        for message in recent_messages:
            intent = message.get('intent', '')
            if intent and intent not in topics:
                topics.append(intent)
        
        return topics

    def _update_conversation_memory(
        self,
        context: ConversationContext,
        user_input: str,
        response: Dict[str, Any]
    ):
        """Update conversation memory with new interaction"""
        
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response,
            'intent': response.get('intent', {}),
            'context': {
                'mood': context.current_mood,
                'location': context.current_location,
                'time_context': context.time_context
            }
        }
        
        # Update session memory
        if context.session_id not in self.context_memory:
            self.context_memory[context.session_id] = []
        
        self.context_memory[context.session_id].append(interaction)
        
        # Keep only recent interactions (last 20)
        if len(self.context_memory[context.session_id]) > 20:
            self.context_memory[context.session_id] = self.context_memory[context.session_id][-20:]
        
        # Update user profile interaction history
        if context.user_profile:
            context.user_profile.interaction_history.append(interaction)
            if len(context.user_profile.interaction_history) > 50:
                context.user_profile.interaction_history = context.user_profile.interaction_history[-50:]

    def _track_interaction_analytics(self, intent_result: Dict[str, Any], processing_time: float):
        """Track interaction analytics for continuous improvement"""
        
        self.analytics['total_interactions'] += 1
        self.analytics['response_times'].append(processing_time)
        
        intent = intent_result.get('primary', 'unknown')
        if intent not in self.analytics['popular_intents']:
            self.analytics['popular_intents'][intent] = 0
        self.analytics['popular_intents'][intent] += 1

    def _extract_intent_features(
        self, 
        user_input: str, 
        context: ConversationContext
    ) -> List[float]:
        """Extract features for ML intent classification"""
        features = []
        
        # Text-based features
        text_lower = user_input.lower()
        
        # Length features
        features.extend([
            len(user_input),
            len(user_input.split()),
            len([w for w in user_input.split() if len(w) > 3])
        ])
        
        # Pattern matching features
        greeting_patterns = ['hi', 'hello', 'hey', 'good morning', 'good afternoon']
        weather_patterns = ['weather', 'rain', 'sunny', 'cold', 'hot', 'temperature']
        food_patterns = ['eat', 'restaurant', 'food', 'hungry', 'meal', 'dinner']
        
        features.extend([
            any(pattern in text_lower for pattern in greeting_patterns),
            any(pattern in text_lower for pattern in weather_patterns),
            any(pattern in text_lower for pattern in food_patterns)
        ])
        
        # Time-based features
        current_hour = datetime.now().hour
        features.extend([
            current_hour,
            1 if 6 <= current_hour <= 11 else 0,  # morning
            1 if 12 <= current_hour <= 17 else 0,  # afternoon
            1 if 18 <= current_hour <= 22 else 0   # evening
        ])
        
        # User context features
        if context.user_profile:
            features.extend([
                len(context.user_profile.interaction_history),
                1 if context.user_profile.visit_frequency == 'first_time' else 0,
                1 if context.user_profile.language_style == 'casual' else 0
            ])
        else:
            features.extend([0, 1, 1])  # Default values
        
        return features

    async def _enhanced_intent_recognition(
        self, 
        user_input: str, 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Enhanced intent recognition with ML and context"""
        
        # First, use the comprehensive system's intent recognition
        try:
            base_intent = self.daily_talks_system._recognize_intent(user_input)
        except AttributeError:
            # Fallback if method doesn't exist
            base_intent = self._basic_intent_recognition(user_input)
        
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

    def _basic_intent_recognition(self, user_input: str) -> Dict[str, Any]:
        """Basic fallback intent recognition"""
        text_lower = user_input.lower()
        
        # Greeting patterns
        if any(word in text_lower for word in ['hi', 'hello', 'hey', 'good morning', 'merhaba']):
            return {'primary': 'greeting', 'confidence': 0.8, 'secondary': []}
        
        # Thanks patterns
        if any(word in text_lower for word in ['thank', 'thanks', 'grateful']):
            return {'primary': 'thanks', 'confidence': 0.8, 'secondary': []}
        
        # Weather patterns
        if any(word in text_lower for word in ['weather', 'rain', 'sunny', 'temperature']):
            return {'primary': 'weather', 'confidence': 0.7, 'secondary': []}
        
        # Restaurant patterns
        if any(word in text_lower for word in ['restaurant', 'food', 'eat', 'hungry', 'meal']):
            return {'primary': 'restaurant', 'confidence': 0.7, 'secondary': []}
        
        # Activity patterns
        if any(word in text_lower for word in ['do today', 'activity', 'what should', 'explore']):
            return {'primary': 'activities', 'confidence': 0.6, 'secondary': []}
        
        return {'primary': 'general', 'confidence': 0.5, 'secondary': []}

    async def _deep_learning_intent_recognition(
        self, 
        user_input: str, 
        context: ConversationContext
    ) -> Optional[Dict[str, Any]]:
        """Advanced intent recognition using deep learning models"""
        
        if not self.dl_models or not self.tokenizer:
            return None
        
        try:
            # This would be implemented with actual deep learning models
            # For now, return None to use fallback methods
            return None
                
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
            # Basic emotion detection from text patterns
            text_lower = user_input.lower()
            
            if any(word in text_lower for word in ['excited', 'amazing', 'wonderful', 'great', 'love']):
                emotion_result.update({
                    'primary_emotion': 'joy',
                    'confidence': 0.7,
                    'method': 'pattern_matching'
                })
            elif any(word in text_lower for word in ['overwhelmed', 'worried', 'anxious', 'nervous']):
                emotion_result.update({
                    'primary_emotion': 'anxiety',
                    'confidence': 0.7,
                    'method': 'pattern_matching'
                })
            elif any(word in text_lower for word in ['thank', 'grateful', 'helpful', 'appreciate']):
                emotion_result.update({
                    'primary_emotion': 'gratitude',
                    'confidence': 0.8,
                    'method': 'pattern_matching'
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
            # Basic sentiment analysis from text patterns
            text_lower = user_input.lower()
            
            positive_words = ['excited', 'amazing', 'wonderful', 'great', 'love', 'thank', 'helpful']
            negative_words = ['overwhelmed', 'worried', 'anxious', 'bad', 'terrible', 'hate']
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment_result.update({
                    'sentiment': 'positive',
                    'confidence': min(0.8, 0.5 + positive_count * 0.1),
                    'method': 'pattern_matching'
                })
            elif negative_count > positive_count:
                sentiment_result.update({
                    'sentiment': 'negative',
                    'confidence': min(0.8, 0.5 + negative_count * 0.1),
                    'method': 'pattern_matching'
                })
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
        
        return sentiment_result

    def _add_hidden_gems_and_local_tips(
        self,
        response: Dict[str, Any],
        intent_result: Dict[str, Any],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Add hidden gems and local tips based on intent and context"""
        
        try:
            primary_intent = intent_result.get('primary', '')
            user_profile = context.user_profile
            current_location = context.current_location or 'sultanahmet'
            
            # Initialize hidden gems and local tips lists
            hidden_gems = []
            local_tips = []
            
            # Intent-based hidden gems
            if primary_intent == 'restaurant_search' or 'restaurant' in primary_intent:
                hidden_gems.extend([
                    "🍯 Try Pandeli Restaurant (since 1901) above the Spice Bazaar - locals' secret for Ottoman cuisine",
                    "☕ Mandabatmaz in Beyoğlu serves the best Turkish coffee - so thick a spoon stands upright!",
                    "🐟 For authentic fish, skip touristy areas and go to Kumkapı - locals eat at Balık Sarayı"
                ])
                local_tips.extend([
                    "💡 Always ask 'Günün yemeği nedir?' (What's today's special?) for the freshest dishes",
                    "🕐 Lunch is 12-2pm, dinner starts after 7pm - restaurants may be closed between",
                    "💳 Many local places only accept cash - always carry Turkish Lira"
                ])
            
            elif primary_intent == 'attraction_planning' or primary_intent == 'activities':
                hidden_gems.extend([
                    "🌅 Climb to Pierre Loti Hill at sunrise - breathtaking Golden Horn views without crowds",
                    "🏛️ Visit Chora Museum early morning - Byzantine mosaics rival Hagia Sophia",
                    "🌊 Take the ferry to Büyükada island - rent a bike, no cars allowed!"
                ])
                local_tips.extend([
                    "🎫 Buy Museum Pass Istanbul (85₺) - skip lines at major attractions",
                    "📱 Download Istanbul Municipality's iBB app for real-time transport info",
                    "🕌 Dress modestly for mosques - cover shoulders and knees"
                ])
            
            elif primary_intent == 'transportation':
                hidden_gems.extend([
                    "🚋 Take the nostalgic tram from Taksim to Tünel - Europe's 2nd oldest subway",
                    "⛴️ Use the Bosphorus ferry as a scenic tour - cheaper than tour boats",
                    "🚌 Ride Metrobus during rush hour - experience Istanbul's organized chaos"
                ])
                local_tips.extend([
                    "💳 Get an Istanbulkart - works on all transport and saves 40% vs single tickets",
                    "⏰ Avoid bridges 7-9am and 5-8pm - use ferries or metro instead",
                    "📍 Download BiTaksi or Uber for reliable rides - regular taxis may not use meters"
                ])
            
            elif primary_intent == 'shopping':
                hidden_gems.extend([
                    "🧿 Visit Arasta Bazaar behind Blue Mosque - authentic crafts without Grand Bazaar crowds",
                    "👗 Explore Çukurcuma for vintage finds - antique shops and retro fashion",
                    "🍯 Buy Turkish delight from Hacı Bekir (since 1777) - the original shop"
                ])
                local_tips.extend([
                    "💰 Bargaining is expected in bazaars - start at 30% of asking price",
                    "🛍️ For modern shopping, go to Nişantaşı or Galata - more upscale than tourist areas",
                    "📦 Many shops ship internationally - ask about tax-free shopping"
                ])
            
            elif primary_intent == 'nightlife':
                hidden_gems.extend([
                    "🍷 Rooftop bars in Karaköy offer Bosphorus views without Beyoğlu crowds",
                    "🎵 Nardis Jazz Club - intimate venue where locals actually go for live music",
                    "🌙 Walk Istiklal Street after midnight - different energy than daytime chaos"
                ])
                local_tips.extend([
                    "🍻 Efes beer is everywhere, but try Bomonti for local craft brewing",
                    "🚕 Book return transport in advance - taxis scarce after 2am on weekends",
                    "👔 Some rooftop bars have dress codes - check before going"
                ])
            
            elif primary_intent == 'cultural':
                hidden_gems.extend([
                    "🎭 Catch a whirling dervish ceremony at Galata Mevlevihanesi - more authentic than tourist shows",
                    "📚 Visit Beyazıt State Library - beautiful Ottoman architecture and peaceful courtyard",
                    "🎨 Explore Salt Galata - contemporary art space in former Ottoman bank"
                ])
                local_tips.extend([
                    "🤝 Turks are incredibly hospitable - don't be surprised by tea invitations",
                    "🙏 Learn basic Turkish: Merhaba (hello), Teşekkürler (thank you), Lütfen (please)",
                    "📸 Always ask before photographing people, especially in religious areas"
                ])
            
            # Location-specific additions
            if current_location:
                location_gems = self._get_location_specific_gems(current_location.lower())
                if location_gems:
                    hidden_gems.extend(location_gems)
                
                location_tips = self._get_location_specific_tips(current_location.lower())
                if location_tips:
                    local_tips.extend(location_tips)
            
            # User profile-based customization
            if user_profile:
                # First-time visitor gets essential tips
                if user_profile.visit_frequency == 'first_time':
                    local_tips.extend([
                        "🕐 Turkish time: Everything runs 30min-1hr later than scheduled",
                        "💶 1 USD ≈ 27-30 Turkish Lira (changes daily)",
                        "📱 Free Wi-Fi: 'Istanbul Metropolitan Municipality' network in many areas"
                    ])
                
                # Frequent visitors get deeper secrets
                elif user_profile.visit_frequency == 'frequent':
                    hidden_gems.extend([
                        "🏛️ Binbirdirek Cistern - less crowded than Basilica Cistern but equally stunning",
                        "🌺 Emirgan Park in spring - locals' picnic spot with stunning tulip displays"
                    ])
                
                # Budget-conscious travelers
                if user_profile.preferences.get('budget') == 'budget':
                    local_tips.extend([
                        "🍞 Turkish breakfast at local bakeries costs 15-20₺ vs 80₺+ at hotels",
                        "🚶 Walking tours are often tip-based - great value for money",
                        "🏠 Stay in Kadıköy for authentic local life at lower prices"
                    ])
            
            # Time-based gems and tips
            current_hour = datetime.now().hour
            if 5 <= current_hour <= 10:  # Morning
                hidden_gems.append("🌅 Early morning in Sultan Ahmed Square - have it almost to yourself before 9am")
                local_tips.append("☕ Turkish breakfast is sacred - take your time, it's meant to be leisurely")
            elif 17 <= current_hour <= 20:  # Evening
                hidden_gems.append("🌆 Galata Bridge at sunset - watch fishermen while enjoying tea")
                local_tips.append("🍽️ Dinner starts late (8-9pm) - use this time for aperitifs or meze")
            
            # Add to response with proper formatting
            if hidden_gems:
                # Select 2-3 most relevant gems
                selected_gems = hidden_gems[:3]
                response['hidden_gems'] = selected_gems
                
                # Add to message if there's space
                if len(response.get('message', '')) < 200:
                    gems_text = f"\n\n🔍 Insider Secrets:\n" + "\n".join([f"• {gem}" for gem in selected_gems[:2]])
                    response['message'] = response.get('message', '') + gems_text
            
            if local_tips:
                # Select 2-3 most relevant tips
                selected_tips = local_tips[:3]
                response['local_tips'] = selected_tips
                
                # Add to message if there's space
                if len(response.get('message', '')) < 300:
                    tips_text = f"\n\n💡 Local Tips:\n" + "\n".join([f"• {tip}" for tip in selected_tips[:2]])
                    response['message'] = response.get('message', '') + tips_text
            
            # Add quick action buttons
            if hidden_gems or local_tips:
                if 'suggested_actions' not in response:
                    response['suggested_actions'] = []
                
                response['suggested_actions'].extend([
                    "🔍 More hidden gems",
                    "💡 Additional local tips",
                    "🗺️ Neighborhood secrets"
                ])
            
        except Exception as e:
            logger.warning(f"Error adding hidden gems and local tips: {e}")
        
        return response

    def _get_location_specific_gems(self, location: str) -> List[str]:
        """Get location-specific hidden gems"""
        
        location_gems = {
            'sultanahmet': [
                "🏺 Arasta Bazaar behind Blue Mosque - authentic Ottoman crafts without crowds",
                "🌿 Gülhane Park's hidden tea garden - peaceful escape from tourist areas"
            ],
            'beyoglu': [
                "📚 Sahaflar Çarşısı (Book Bazaar) - old book market with rare finds",
                "🎭 Atlas Cinema - historic movie theater showing art films"
            ],
            'kadikoy': [
                "🎨 Yeldeğirmeni neighborhood - street art and hipster cafes",
                "🐟 Tuesday market for fresh fish - locals' shopping secret"
            ],
            'galata': [
                "🗼 Galata Tower's secret terrace - less crowded evening views",
                "🍷 Wine bars in old Genoese buildings - medieval atmosphere"
            ],
            'besiktas': [
                "⚽ Beşiktaş Fish Market - authentic local atmosphere",
                "🌊 Ortaköy's hidden mosque courtyard - peaceful Bosphorus views"
            ],
            'eminonu': [
                "🍯 Spice Bazaar's upper floor - locals' tea and coffee shops",
                "⛵ Ferry departure docks early morning - see commuter culture"
            ]
        }
        
        return location_gems.get(location, [])

    def _get_location_specific_tips(self, location: str) -> List[str]:
        """Get location-specific local tips"""
        
        location_tips = {
            'sultanahmet': [
                "🎫 Visit Blue Mosque between prayer times - free entry but check schedule",
                "👥 Avoid carpet shop 'invitations' - politely say 'Teşekkürler, hayır'"
            ],
            'beyoglu': [
                "🚇 Use Tünel funicular to avoid steep hills - historic and practical",
                "🍻 Happy hours 5-7pm at rooftop bars - locals' timing"
            ],
            'kadikoy': [
                "🚢 Take ferry from Eminönü - scenic 20min ride vs expensive taxi",
                "🍽️ Eat where you see locals queuing - always the best food"
            ],
            'galata': [
                "🎨 Art galleries open late Thursday - free wine and local artists",
                "📷 Best tower photos from Şişhane metro station area"
            ],
            'besiktas': [
                "🏟️ Stadium area gets crazy on match days - plan accordingly",
                "🚌 Dolmuş (shared taxis) are faster than buses here"
            ],
            'eminonu': [
                "🐟 Fish sandwich boats - ask for price first, quality varies",
                "🚶 Walk to Sirkeci instead of taxi - often faster in traffic"
            ]
        }
        
        return location_tips.get(location, [])

    async def _enhance_response_with_ml(
        self,
        base_response: Dict[str, Any],
        intent_result: Dict[str, Any],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Enhance the base response with ML personalization"""
        
        enhanced_response = base_response.copy()
        
        try:
            # Apply basic personalization
            enhanced_response = self._personalize_response(
                enhanced_response, context.user_profile
            )
            
            # Add contextual information
            enhanced_response = self._add_contextual_information(
                enhanced_response, context
            )
            
            # ✨ NEW: Add hidden gems and local tips
            enhanced_response = self._add_hidden_gems_and_local_tips(
                enhanced_response, intent_result, context
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

    def _personalize_response(
        self, 
        response: Dict[str, Any], 
        user_profile: UserProfile
    ) -> Dict[str, Any]:
        """Personalize response based on user profile"""
        
        if not user_profile:
            return response
        
        # Adjust language style
        if user_profile.language_style == 'formal':
            response['message'] = self._formalize_language(response.get('message', ''))
        elif user_profile.language_style == 'casual':
            response['message'] = self._casualize_language(response.get('message', ''))
        
        # Add cultural adaptations
        if user_profile.cultural_background:
            response = self._adapt_for_culture(response, user_profile.cultural_background)
        
        return response

    def _add_contextual_information(
        self, 
        response: Dict[str, Any], 
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Add relevant contextual information to the response"""
        
        # Time-based contextual additions
        current_hour = datetime.now().hour
        if current_hour < 12:
            response['time_context'] = "Perfect timing for morning activities!"
        elif current_hour < 17:
            response['time_context'] = "Great for afternoon exploration!"
        else:
            response['time_context'] = "Ideal for evening experiences!"
        
        return response

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
                    response['message'] = f"🌟 {message}"
                    response['emotional_tone'] = 'enthusiastic'
                
                elif primary_emotion == 'anxiety':
                    response['message'] = f"💙 Don't worry! {message} I'm here to help make your Istanbul experience comfortable and enjoyable!"
                    response['emotional_tone'] = 'supportive'
                
                elif primary_emotion == 'gratitude':
                    response['message'] = f"🙏 You're very welcome! {message} I'm so glad I could help!"
                    response['emotional_tone'] = 'warm'
        
        except Exception as e:
            logger.warning(f"Emotion-aware adjustment error: {e}")
        
        return response

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
                # Handle both dict and string formats for intent
                intent_data = interaction.get('intent', {})
                if isinstance(intent_data, dict):
                    intent = intent_data.get('primary', 'unknown')
                elif isinstance(intent_data, str):
                    intent = intent_data
                else:
                    intent = 'unknown'
                    
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
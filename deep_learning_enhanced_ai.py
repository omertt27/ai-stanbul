#!/usr/bin/env python3
"""
Deep Learning Enhanced Istanbul AI System
Advanced neural networks for superior conversational experience
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, pipeline
import numpy as np
import pandas as pd
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import spacy
import networkx as nx
from collections import defaultdict, deque
import sqlite3
import aiofiles
import regex as re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationTone(Enum):
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    ENTHUSIASTIC = "enthusiastic"
    CULTURAL = "cultural"

class UserType(Enum):
    FIRST_TIME_VISITOR = "first_time_visitor"
    REGULAR_VISITOR = "regular_visitor"
    LOCAL_RESIDENT = "local_resident"
    BUSINESS_TRAVELER = "business_traveler"
    CULTURAL_ENTHUSIAST = "cultural_enthusiast"

@dataclass
class ConversationMemory:
    """Advanced conversation memory with temporal awareness"""
    user_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    mentioned_entities: Dict[str, int] = field(default_factory=dict)  # entity -> frequency
    temporal_context: Dict[str, datetime] = field(default_factory=dict)
    emotional_state: str = "neutral"
    satisfaction_score: float = 0.0
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)

class AdvancedNeuralIntentClassifier(nn.Module):
    """Deep learning model for intent classification with attention mechanism"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512, 
                 num_intents: int = 15, dropout: float = 0.3):
        super().__init__()
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_encoding = self._create_position_encoding(512, embedding_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads=8, dropout=dropout)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, dropout=dropout, batch_first=True)
        
        # Attention mechanism for importance weighting
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_intents)
        )
        
        # Context integration layer
        self.context_layer = nn.Linear(embedding_dim, hidden_dim)
        
    def _create_position_encoding(self, max_length: int, d_model: int):
        """Create positional encoding for transformer-like attention"""
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                context_vector: Optional[torch.Tensor] = None):
        """Forward pass with attention mechanism"""
        batch_size, seq_len = input_ids.shape
        
        # Embedding with positional encoding
        embedded = self.embedding(input_ids)
        if seq_len <= self.position_encoding.size(1):
            embedded += self.position_encoding[:, :seq_len, :]
        
        # Multi-head attention
        embedded_t = embedded.transpose(0, 1)  # (seq_len, batch, embed_dim)
        attn_output, _ = self.multihead_attn(embedded_t, embedded_t, embedded_t)
        attn_output = attn_output.transpose(0, 1)  # (batch, seq_len, embed_dim)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(attn_output)
        
        # Attention mechanism for sequence weighting
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        attended_output = (lstm_out * attention_weights).sum(dim=1)
        
        # Context integration
        if context_vector is not None:
            context_features = self.context_layer(context_vector)
            attended_output = attended_output + context_features
        
        # Classification
        logits = self.classifier(attended_output)
        
        return logits, attention_weights

class EntityRecognitionNetwork(nn.Module):
    """Neural network for Named Entity Recognition specific to Istanbul"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 256, num_entities: int = 25):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM-CRF architecture
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           bidirectional=True, dropout=0.3, batch_first=True)
        
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_entities)
        
        # CRF layer for sequence labeling
        self.crf_transitions = nn.Parameter(torch.randn(num_entities, num_entities))
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass for entity recognition"""
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        tag_scores = self.hidden2tag(lstm_out)
        
        return tag_scores

class ContextualEmbeddingGenerator(nn.Module):
    """Generate contextual embeddings for conversation state"""
    
    def __init__(self, input_dim: int = 768, context_dim: int = 256):
        super().__init__()
        
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, context_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(context_dim * 2, context_dim),
            nn.Tanh()
        )
        
        # Attention for temporal context
        self.temporal_attention = nn.MultiheadAttention(context_dim, num_heads=4)
        
    def forward(self, conversation_embeddings: torch.Tensor):
        """Generate contextual representation of conversation"""
        context_repr = self.context_encoder(conversation_embeddings)
        
        # Apply temporal attention
        context_t = context_repr.transpose(0, 1)
        attended_context, _ = self.temporal_attention(context_t, context_t, context_t)
        
        return attended_context.transpose(0, 1)

class PersonalizationEngine(nn.Module):
    """Neural network for user personalization and preference learning"""
    
    def __init__(self, user_feature_dim: int = 128, preference_dim: int = 64):
        super().__init__()
        
        # User embedding
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, preference_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(preference_dim * 2, preference_dim)
        )
        
        # Preference prediction
        self.preference_predictor = nn.Sequential(
            nn.Linear(preference_dim + 256, 128),  # +256 for context
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Predict various preferences
            nn.Sigmoid()
        )
        
    def forward(self, user_features: torch.Tensor, context_features: torch.Tensor):
        """Predict user preferences based on features and context"""
        user_embedding = self.user_encoder(user_features)
        combined_features = torch.cat([user_embedding, context_features], dim=-1)
        preferences = self.preference_predictor(combined_features)
        
        return preferences, user_embedding

class ResponseGenerationNetwork(nn.Module):
    """Advanced neural response generation with style control"""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 512, num_styles: int = 5):
        super().__init__()
        
        # Style embedding
        self.style_embedding = nn.Embedding(num_styles, hidden_dim)
        
        # Context-aware decoder
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=3, 
                              dropout=0.3, batch_first=True)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, context_vector: torch.Tensor, style_id: torch.Tensor, 
                max_length: int = 100):
        """Generate response with specified style"""
        batch_size = context_vector.size(0)
        
        # Style conditioning
        style_embed = self.style_embedding(style_id)
        
        # Initialize generation
        hidden = (context_vector.unsqueeze(0).repeat(3, 1, 1),
                 torch.zeros_like(context_vector.unsqueeze(0).repeat(3, 1, 1)))
        
        generated_tokens = []
        input_token = style_embed.unsqueeze(1)
        
        for _ in range(max_length):
            output, hidden = self.decoder(input_token, hidden)
            token_logits = self.output_projection(output)
            
            # Apply attention
            attended_output, _ = self.attention(output.transpose(0, 1), 
                                              output.transpose(0, 1), 
                                              output.transpose(0, 1))
            
            generated_tokens.append(token_logits)
            input_token = output  # Use output as next input
        
        return torch.stack(generated_tokens, dim=1)

class IstanbulKnowledgeGraph:
    """Enhanced knowledge graph with deep learning embeddings"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        self.embedding_dim = 256
        
        # Initialize with Istanbul-specific knowledge
        self._initialize_knowledge_base()
        self._train_graph_embeddings()
    
    def _initialize_knowledge_base(self):
        """Initialize comprehensive Istanbul knowledge"""
        
        # Neighborhoods with rich attributes
        neighborhoods = {
            "Beyoğlu": {
                "type": "district",
                "atmosphere": "vibrant_nightlife",
                "best_for": ["dining", "nightlife", "shopping", "culture"],
                "transport": ["metro", "tram", "bus", "ferry"],
                "cultural_significance": 0.9,
                "tourist_density": 0.8,
                "local_tips": ["Visit Istiklal Street", "Try meyhanes in Nevizade"],
                "price_level": "moderate_to_high",
                "cuisine_specialties": ["international", "traditional_meyhane", "street_food"]
            },
            "Sultanahmet": {
                "type": "historic_district",
                "atmosphere": "historic_traditional",
                "best_for": ["sightseeing", "history", "culture", "traditional_dining"],
                "transport": ["tram", "bus", "walking"],
                "cultural_significance": 1.0,
                "tourist_density": 1.0,
                "local_tips": ["Visit early morning", "Book Hagia Sophia in advance"],
                "price_level": "tourist_premium",
                "cuisine_specialties": ["ottoman", "traditional_turkish", "tourist_friendly"]
            },
            "Kadıköy": {
                "type": "asian_side_district",
                "atmosphere": "local_authentic",
                "best_for": ["local_culture", "authentic_food", "markets", "alternative_scene"],
                "transport": ["ferry", "metro", "bus", "dolmuş"],
                "cultural_significance": 0.7,
                "tourist_density": 0.4,
                "local_tips": ["Explore Moda", "Visit Tuesday market"],
                "price_level": "budget_friendly",
                "cuisine_specialties": ["street_food", "local_eateries", "fish_sandwich"]
            }
        }
        
        # Add neighborhoods to graph
        for name, attrs in neighborhoods.items():
            self.graph.add_node(name, **attrs)
        
        # Restaurants with detailed information
        self._add_restaurant_knowledge()
        
        # Transportation network
        self._add_transportation_knowledge()
        
        # Cultural sites
        self._add_cultural_knowledge()
    
    def _add_restaurant_knowledge(self):
        """Add comprehensive restaurant information"""
        restaurants = {
            "Pandeli": {
                "location": "Eminönü",
                "cuisine": "Ottoman",
                "price_range": "expensive",
                "specialties": ["lamb_stew", "ottoman_classics"],
                "atmosphere": "historic_elegant",
                "reservation_needed": True,
                "cultural_significance": 0.9
            },
            "Çiya Sofrası": {
                "location": "Kadıköy",
                "cuisine": "Regional_Turkish",
                "price_range": "moderate",
                "specialties": ["forgotten_recipes", "anatolian_dishes"],
                "atmosphere": "authentic_traditional",
                "reservation_needed": False,
                "cultural_significance": 0.8
            },
            "Mikla": {
                "location": "Beyoğlu",
                "cuisine": "Modern_Turkish",
                "price_range": "very_expensive",
                "specialties": ["modern_interpretations", "tasting_menu"],
                "atmosphere": "fine_dining_contemporary",
                "reservation_needed": True,
                "cultural_significance": 0.7
            }
        }
        
        for name, attrs in restaurants.items():
            self.graph.add_node(name, type="restaurant", **attrs)
            # Connect to neighborhood
            if attrs["location"] in self.graph.nodes():
                self.graph.add_edge(attrs["location"], name, relation="contains")
    
    def _add_transportation_knowledge(self):
        """Add transportation network information"""
        transport_lines = {
            "M2_Metro": {
                "type": "metro",
                "stations": ["Taksim", "Şişhane", "Vezneciler", "Haliç"],
                "frequency": "3-5_minutes",
                "operating_hours": "06:00-00:30"
            },
            "T1_Tram": {
                "type": "tram",
                "stations": ["Kabataş", "Karaköy", "Eminönü", "Sultanahmet", "Beyazıt"],
                "frequency": "5-7_minutes",
                "operating_hours": "06:00-00:30"
            }
        }
        
        for line, attrs in transport_lines.items():
            self.graph.add_node(line, **attrs)
            for station in attrs["stations"]:
                self.graph.add_node(station, type="station")
                self.graph.add_edge(line, station, relation="serves")
    
    def _add_cultural_knowledge(self):
        """Add cultural sites and events"""
        cultural_sites = {
            "Hagia_Sophia": {
                "type": "museum",
                "significance": "world_heritage",
                "best_time": "early_morning",
                "duration": "2-3_hours",
                "ticket_required": True,
                "cultural_context": "Byzantine_Ottoman_history"
            },
            "Blue_Mosque": {
                "type": "mosque",
                "significance": "active_worship",
                "best_time": "outside_prayer_times",
                "duration": "1_hour",
                "ticket_required": False,
                "cultural_context": "islamic_architecture"
            }
        }
        
        for site, attrs in cultural_sites.items():
            self.graph.add_node(site, **attrs)
    
    def _train_graph_embeddings(self):
        """Train graph embeddings using node2vec approach"""
        try:
            # Simple random walk based embeddings
            nodes = list(self.graph.nodes())
            for node in nodes:
                # Create simple embedding based on node attributes
                embedding = np.random.normal(0, 0.1, self.embedding_dim)
                
                # Encode attributes into embedding
                if 'cultural_significance' in self.graph.nodes[node]:
                    cultural_weight = self.graph.nodes[node]['cultural_significance']
                    embedding[:10] *= cultural_weight
                
                self.entity_embeddings[node] = embedding
                
        except Exception as e:
            logger.warning(f"Graph embedding training failed: {e}")
    
    def find_related_entities(self, entity: str, relation_type: str = None, 
                            limit: int = 5) -> List[Tuple[str, float]]:
        """Find entities related to given entity using embeddings"""
        if entity not in self.entity_embeddings:
            return []
        
        entity_embedding = self.entity_embeddings[entity]
        similarities = []
        
        for other_entity, other_embedding in self.entity_embeddings.items():
            if other_entity != entity:
                similarity = cosine_similarity(
                    entity_embedding.reshape(1, -1),
                    other_embedding.reshape(1, -1)
                )[0][0]
                similarities.append((other_entity, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]

class DeepLearningEnhancedAI:
    """Main AI system with deep learning capabilities"""
    
    def __init__(self):
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        
        # Initialize neural networks
        self._initialize_neural_networks()
        
        # Knowledge graph
        self.knowledge_graph = IstanbulKnowledgeGraph()
        
        # Conversation memory
        self.conversation_memories: Dict[str, ConversationMemory] = {}
        
        # User analytics
        self.user_analytics = defaultdict(dict)
        
        # Cultural intelligence
        self.cultural_patterns = self._load_cultural_patterns()
        
        # Response templates with personality
        self.response_templates = self._initialize_response_templates()
        
        logger.info("🧠 Deep Learning Enhanced AI System initialized successfully!")
    
    def _initialize_neural_networks(self):
        """Initialize all neural network components"""
        vocab_size = self.tokenizer.vocab_size
        
        # Intent classifier
        self.intent_classifier = AdvancedNeuralIntentClassifier(
            vocab_size=vocab_size,
            num_intents=15
        ).to(self.device)
        
        # Entity recognizer
        self.entity_recognizer = EntityRecognitionNetwork(
            vocab_size=vocab_size,
            num_entities=25
        ).to(self.device)
        
        # Context generator
        self.context_generator = ContextualEmbeddingGenerator().to(self.device)
        
        # Personalization engine
        self.personalization_engine = PersonalizationEngine().to(self.device)
        
        # Response generator
        self.response_generator = ResponseGenerationNetwork(
            vocab_size=vocab_size
        ).to(self.device)
        
        # Load pre-trained weights if available
        self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        """Load pre-trained model weights"""
        try:
            # In production, load from saved checkpoints
            logger.info("Pre-trained weights would be loaded here")
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights: {e}")
    
    def _load_cultural_patterns(self) -> Dict[str, Any]:
        """Load cultural intelligence patterns"""
        return {
            "greetings": {
                "turkish": ["Merhaba", "Selam", "İyi günler"],
                "cultural_context": "Turkish people appreciate attempts at local language",
                "response_style": "warm_appreciative"
            },
            "local_customs": {
                "dining": {
                    "tea_culture": "Turkish tea is central to social interaction",
                    "tipping": "10-15% is standard in restaurants",
                    "meal_times": "Dinner is typically late (8-10 PM)"
                },
                "religious_sites": {
                    "dress_code": "Modest clothing required",
                    "prayer_times": "Respect prayer times at mosques",
                    "behavior": "Quiet and respectful demeanor expected"
                }
            },
            "communication_style": {
                "warmth": "Turkish culture values warmth and hospitality",
                "storytelling": "Stories and personal anecdotes are appreciated",
                "humor": "Light humor about daily life is welcome"
            }
        }
    
    def _initialize_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize culturally-aware response templates"""
        return {
            "restaurant_recommendation": {
                "enthusiastic": [
                    "🍽️ Oh, I'm so excited to share some incredible dining spots with you! Istanbul's food scene is absolutely magical!",
                    "✨ You're in for such a treat! Let me tell you about some phenomenal restaurants that will make your taste buds dance!",
                    "🌟 Istanbul's culinary world is waiting for you! I have some absolutely divine recommendations!"
                ],
                "cultural": [
                    "🏛️ Let me share some restaurants that truly capture Istanbul's rich culinary heritage...",
                    "🎭 These dining spots will give you an authentic taste of our beautiful city's culture...",
                    "📚 Each of these restaurants has a story to tell about Istanbul's fascinating food history..."
                ],
                "friendly": [
                    "😊 I'd love to help you find some amazing places to eat! Here are my personal favorites...",
                    "🤗 Great question! I know some wonderful spots that I think you'll really enjoy...",
                    "💫 Let me recommend some fantastic restaurants that never disappoint..."
                ]
            },
            "cultural_information": {
                "educational": [
                    "📖 Here's something fascinating about Istanbul's culture...",
                    "🎨 Let me share the rich history behind this...",
                    "🏛️ This cultural aspect has deep roots in Istanbul's heritage..."
                ],
                "storytelling": [
                    "✨ There's a beautiful story behind this tradition...",
                    "🌙 Legend has it that this custom began centuries ago...",
                    "🌸 Local families have been celebrating this way for generations..."
                ]
            }
        }
    
    async def process_message(self, message: str, user_id: str, 
                            session_id: Optional[str] = None) -> str:
        """Process user message with deep learning enhancement"""
        try:
            start_time = datetime.now()
            
            # Get or create conversation memory
            if user_id not in self.conversation_memories:
                self.conversation_memories[user_id] = ConversationMemory(user_id=user_id)
            
            memory = self.conversation_memories[user_id]
            
            # Tokenize input
            inputs = self.tokenizer(
                message,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get BERT embeddings
            with torch.no_grad():
                bert_outputs = self.bert_model(**inputs)
                sentence_embedding = bert_outputs.last_hidden_state.mean(dim=1)
            
            # Intent classification with deep learning
            intent_logits, attention_weights = self.intent_classifier(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
            predicted_intent_id = torch.argmax(intent_logits, dim=-1).item()
            intent_confidence = torch.softmax(intent_logits, dim=-1).max().item()
            
            # Map intent ID to intent name
            intent_names = [
                "restaurant_recommendation", "location_search", "cultural_information",
                "transportation", "entertainment", "shopping", "accommodation",
                "food_specific", "neighborhood_info", "historical_sites",
                "practical_info", "emergency", "weather", "currency", "language_help"
            ]
            
            predicted_intent = intent_names[min(predicted_intent_id, len(intent_names)-1)]
            
            # Entity recognition
            entity_scores = self.entity_recognizer(inputs['input_ids'], inputs['attention_mask'])
            
            # Extract entities (simplified)
            entities = self._extract_entities(message, entity_scores)
            
            # Generate contextual embedding
            context_embedding = self.context_generator(sentence_embedding.unsqueeze(0))
            
            # User personalization
            user_features = self._get_user_features(user_id, memory)
            preferences, user_embedding = self.personalization_engine(
                user_features.to(self.device),
                context_embedding.squeeze(0)
            )
            
            # Update conversation memory
            self._update_conversation_memory(memory, message, predicted_intent, entities)
            
            # Generate response
            response = await self._generate_intelligent_response(
                message, predicted_intent, entities, memory, 
                intent_confidence, context_embedding
            )
            
            # Update analytics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_user_analytics(user_id, predicted_intent, intent_confidence, 
                                      processing_time, entities)
            
            logger.info(f"Processed message for {user_id}: intent={predicted_intent}, "
                       f"confidence={intent_confidence:.3f}, time={processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I apologize, but I'm having some technical difficulties. Please try again in a moment. 🤖"
    
    def _extract_entities(self, message: str, entity_scores: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract entities using neural network output"""
        entities = []
        
        # Entity types
        entity_types = [
            "NEIGHBORHOOD", "RESTAURANT", "CUISINE", "TRANSPORT", "ATTRACTION",
            "PRICE", "TIME", "DIETARY", "ATMOSPHERE", "ACTIVITY"
        ]
        
        # Simple entity extraction (in production, use proper NER)
        message_lower = message.lower()
        
        # Neighborhoods
        neighborhoods = ["beyoğlu", "sultanahmet", "kadıköy", "taksim", "galata", 
                        "beşiktaş", "üsküdar", "fatih", "şişli", "bakırköy"]
        for neighborhood in neighborhoods:
            if neighborhood in message_lower:
                entities.append({
                    "text": neighborhood.title(),
                    "type": "NEIGHBORHOOD",
                    "confidence": 0.9
                })
        
        # Cuisines
        cuisines = ["turkish", "ottoman", "seafood", "italian", "japanese", 
                   "mediterranean", "street food", "vegetarian", "vegan"]
        for cuisine in cuisines:
            if cuisine in message_lower:
                entities.append({
                    "text": cuisine.title(),
                    "type": "CUISINE",
                    "confidence": 0.8
                })
        
        return entities
    
    def _get_user_features(self, user_id: str, memory: ConversationMemory) -> torch.Tensor:
        """Generate user feature vector"""
        features = []
        
        # Interaction history features
        features.append(len(memory.conversation_history))  # Number of interactions
        features.append(memory.satisfaction_score)  # Satisfaction score
        
        # Preference features
        preferred_neighborhoods = len(memory.user_preferences.get('neighborhoods', []))
        preferred_cuisines = len(memory.user_preferences.get('cuisines', []))
        features.extend([preferred_neighborhoods, preferred_cuisines])
        
        # Temporal features
        now = datetime.now()
        if memory.conversation_history:
            last_interaction = memory.conversation_history[-1].get('timestamp', now)
            if isinstance(last_interaction, str):
                last_interaction = datetime.fromisoformat(last_interaction)
            time_since_last = (now - last_interaction).total_seconds() / 3600  # hours
            features.append(time_since_last)
        else:
            features.append(0)
        
        # Pad to required dimension
        while len(features) < 128:
            features.append(0.0)
        
        return torch.tensor(features[:128], dtype=torch.float32)
    
    def _update_conversation_memory(self, memory: ConversationMemory, message: str,
                                  intent: str, entities: List[Dict[str, Any]]):
        """Update conversation memory with new interaction"""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "intent": intent,
            "entities": entities
        }
        
        memory.conversation_history.append(interaction)
        
        # Keep only recent history (last 50 interactions)
        if len(memory.conversation_history) > 50:
            memory.conversation_history = memory.conversation_history[-50:]
        
        # Update entity mentions
        for entity in entities:
            entity_text = entity["text"].lower()
            memory.mentioned_entities[entity_text] = memory.mentioned_entities.get(entity_text, 0) + 1
        
        # Update preferences based on entities
        for entity in entities:
            if entity["type"] == "NEIGHBORHOOD":
                neighborhoods = memory.user_preferences.setdefault('neighborhoods', [])
                if entity["text"] not in neighborhoods:
                    neighborhoods.append(entity["text"])
            elif entity["type"] == "CUISINE":
                cuisines = memory.user_preferences.setdefault('cuisines', [])
                if entity["text"] not in cuisines:
                    cuisines.append(entity["text"])
    
    async def _generate_intelligent_response(self, message: str, intent: str, 
                                           entities: List[Dict[str, Any]],
                                           memory: ConversationMemory,
                                           confidence: float,
                                           context_embedding: torch.Tensor) -> str:
        """Generate intelligent, personalized response"""
        
        # Determine conversation tone based on user history and preferences
        tone = self._determine_conversation_tone(memory, intent)
        
        # Get base response template
        if intent in self.response_templates:
            template_category = self.response_templates[intent]
            tone_key = tone.value if hasattr(tone, 'value') else 'friendly'
            templates = template_category.get(tone_key, template_category.get('friendly', []))
            
            if templates:
                base_response = np.random.choice(templates)
            else:
                base_response = "I'd be happy to help you with that!"
        else:
            base_response = "That's an interesting question! Let me help you with that."
        
        # Generate specific content based on intent and entities
        specific_content = await self._generate_specific_content(intent, entities, memory)
        
        # Add cultural intelligence
        cultural_addition = self._add_cultural_intelligence(message, intent, entities)
        
        # Combine response parts
        response_parts = [base_response]
        
        if specific_content:
            response_parts.append(specific_content)
        
        if cultural_addition:
            response_parts.append(cultural_addition)
        
        # Add personalized suggestions
        suggestions = self._generate_personalized_suggestions(memory, intent, entities)
        if suggestions:
            response_parts.append(f"\n\n💡 {suggestions}")
        
        final_response = "\n\n".join(response_parts)
        
        return final_response
    
    def _determine_conversation_tone(self, memory: ConversationMemory, intent: str) -> ConversationTone:
        """Determine appropriate conversation tone"""
        # Default tone
        tone = ConversationTone.FRIENDLY
        
        # Adjust based on user history
        if len(memory.conversation_history) < 3:
            tone = ConversationTone.ENTHUSIASTIC  # New users get enthusiastic welcome
        elif memory.satisfaction_score > 0.8:
            tone = ConversationTone.FRIENDLY
        elif intent in ["cultural_information", "historical_sites"]:
            tone = ConversationTone.CULTURAL
        
        return tone
    
    async def _generate_specific_content(self, intent: str, entities: List[Dict[str, Any]],
                                       memory: ConversationMemory) -> str:
        """Generate specific content based on intent and entities"""
        
        if intent == "restaurant_recommendation":
            return await self._generate_restaurant_recommendations(entities, memory)
        elif intent == "neighborhood_info":
            return self._generate_neighborhood_info(entities)
        elif intent == "transportation":
            return self._generate_transportation_info(entities)
        elif intent == "cultural_information":
            return self._generate_cultural_info(entities)
        else:
            return self._generate_general_istanbul_info(intent, entities)
    
    async def _generate_restaurant_recommendations(self, entities: List[Dict[str, Any]],
                                                 memory: ConversationMemory) -> str:
        """Generate personalized restaurant recommendations"""
        recommendations = []
        
        # Extract preferences
        neighborhood = None
        cuisine = None
        
        for entity in entities:
            if entity["type"] == "NEIGHBORHOOD":
                neighborhood = entity["text"].lower()
            elif entity["type"] == "CUISINE":
                cuisine = entity["text"].lower()
        
        # Use knowledge graph to find restaurants
        if neighborhood and neighborhood in self.knowledge_graph.graph.nodes():
            related_entities = self.knowledge_graph.find_related_entities(
                neighborhood, limit=3
            )
            
            for entity_name, similarity in related_entities:
                node_data = self.knowledge_graph.graph.nodes().get(entity_name, {})
                if node_data.get('type') == 'restaurant':
                    recommendations.append({
                        'name': entity_name,
                        'similarity': similarity,
                        **node_data
                    })
        
        # Generate response
        if recommendations:
            response = "Here are my top restaurant recommendations:\n\n"
            for i, rest in enumerate(recommendations[:3], 1):
                response += f"🍽️ **{i}. {rest['name']}**\n"
                response += f"   📍 Located in {rest.get('location', 'Istanbul')}\n"
                response += f"   🍜 Cuisine: {rest.get('cuisine', 'Turkish')}\n"
                response += f"   💰 Price: {rest.get('price_range', 'Moderate')}\n"
                if rest.get('specialties'):
                    response += f"   ⭐ Famous for: {', '.join(rest['specialties'])}\n"
                response += "\n"
        else:
            # Fallback recommendations
            response = self._get_fallback_restaurant_recommendations(neighborhood, cuisine)
        
        return response
    
    def _get_fallback_restaurant_recommendations(self, neighborhood: str = None, 
                                               cuisine: str = None) -> str:
        """Fallback restaurant recommendations"""
        recommendations = [
            {
                "name": "Pandeli",
                "location": "Eminönü",
                "cuisine": "Ottoman",
                "specialty": "Historic Ottoman cuisine in a beautiful setting",
                "price": "High-end"
            },
            {
                "name": "Çiya Sofrası", 
                "location": "Kadıköy",
                "cuisine": "Regional Turkish",
                "specialty": "Authentic regional dishes you won't find elsewhere",
                "price": "Moderate"
            },
            {
                "name": "Hamdi Restaurant",
                "location": "Eminönü", 
                "cuisine": "Turkish/Kebab",
                "specialty": "Famous for lamb dishes and Bosphorus views",
                "price": "Moderate"
            }
        ]
        
        response = "Here are some exceptional restaurants I highly recommend:\n\n"
        for i, rest in enumerate(recommendations, 1):
            response += f"🍽️ **{i}. {rest['name']}**\n"
            response += f"   📍 {rest['location']} • 🍜 {rest['cuisine']} • 💰 {rest['price']}\n"
            response += f"   ⭐ {rest['specialty']}\n\n"
        
        return response
    
    def _generate_neighborhood_info(self, entities: List[Dict[str, Any]]) -> str:
        """Generate neighborhood information"""
        neighborhood_info = {
            "beyoğlu": {
                "description": "The heart of modern Istanbul with vibrant nightlife, excellent dining, and cultural attractions.",
                "highlights": ["İstiklal Street", "Galata Tower", "Pera Museum", "Nevizade Street"],
                "best_for": "Nightlife, dining, shopping, and cultural experiences",
                "atmosphere": "Cosmopolitan and energetic"
            },
            "sultanahmet": {
                "description": "The historic peninsula where Byzantine and Ottoman empires left their magnificent marks.",
                "highlights": ["Hagia Sophia", "Blue Mosque", "Topkapi Palace", "Grand Bazaar"],
                "best_for": "History, architecture, and traditional Turkish culture",
                "atmosphere": "Historic and touristy"
            },
            "kadıköy": {
                "description": "Istanbul's Asian side gem, beloved by locals for authentic culture and great food.",
                "highlights": ["Moda neighborhood", "Kadıköy Market", "Ferry terminal", "Alternative art scene"],
                "best_for": "Local culture, affordable dining, and authentic experiences",
                "atmosphere": "Local and laid-back"
            }
        }
        
        # Find mentioned neighborhood
        for entity in entities:
            if entity["type"] == "NEIGHBORHOOD":
                neighborhood = entity["text"].lower()
                if neighborhood in neighborhood_info:
                    info = neighborhood_info[neighborhood]
                    response = f"🏛️ **{entity['text']} District**\n\n"
                    response += f"{info['description']}\n\n"
                    response += f"🌟 **Top Highlights:**\n"
                    response += "\n".join([f"   • {highlight}" for highlight in info['highlights']])
                    response += f"\n\n🎯 **Best for:** {info['best_for']}\n"
                    response += f"🌆 **Atmosphere:** {info['atmosphere']}"
                    return response
        
        return "I'd be happy to tell you about any Istanbul neighborhood! Which area interests you?"
    
    def _generate_transportation_info(self, entities: List[Dict[str, Any]]) -> str:
        """Generate transportation information"""
        return ("🚇 **Getting Around Istanbul:**\n\n"
                "**Metro & Tram:** Clean, efficient, and covers major areas\n"
                "**Ferry:** Scenic and practical for crossing the Bosphorus\n"
                "**Bus:** Extensive network, can be crowded during rush hour\n"
                "**Taxi/Uber:** Convenient but traffic can be heavy\n"
                "**Walking:** Many attractions are walkable in historic areas\n\n"
                "💡 **Pro tip:** Get an İstanbulkart for easy payment on all public transport!")
    
    def _generate_cultural_info(self, entities: List[Dict[str, Any]]) -> str:
        """Generate cultural information"""
        cultural_topics = {
            "general": ("🎭 **Turkish Culture Highlights:**\n\n"
                       "**Hospitality:** Turkish people are incredibly welcoming\n"
                       "**Tea Culture:** Çay (tea) is central to social life\n"
                       "**Family Values:** Family is extremely important in Turkish culture\n"
                       "**Respect:** Show respect at mosques and religious sites\n"
                       "**Tipping:** 10-15% is standard in restaurants\n\n"
                       "🌟 Turkish culture beautifully blends European and Asian influences!")
        }
        
        return cultural_topics.get("general", "Turkish culture is rich and fascinating!")
    
    def _generate_general_istanbul_info(self, intent: str, entities: List[Dict[str, Any]]) -> str:
        """Generate general Istanbul information"""
        info_map = {
            "practical_info": ("📱 **Practical Istanbul Info:**\n\n"
                              "**Currency:** Turkish Lira (TL)\n"
                              "**Language:** Turkish (English widely spoken in tourist areas)\n"
                              "**Emergency:** 112 (general emergency)\n"
                              "**Electricity:** 230V, European plugs\n"
                              "**WiFi:** Available in most cafes and hotels"),
            
            "weather": ("🌤️ **Istanbul Weather:**\n\n"
                       "**Spring (Mar-May):** Mild and pleasant, perfect for walking\n"
                       "**Summer (Jun-Aug):** Warm and humid, great for outdoor dining\n"
                       "**Autumn (Sep-Nov):** Cool and comfortable, ideal visiting time\n"
                       "**Winter (Dec-Feb):** Cool and rainy, cozy indoor activities"),
        }
        
        return info_map.get(intent, "I'm here to help you discover the best of Istanbul! What would you like to know?")
    
    def _add_cultural_intelligence(self, message: str, intent: str, 
                                 entities: List[Dict[str, Any]]) -> str:
        """Add cultural intelligence to responses"""
        cultural_additions = []
        
        # Check for cultural patterns
        message_lower = message.lower()
        
        # Detect attempt at Turkish language
        turkish_words = ["merhaba", "teşekkür", "selam", "günaydın"]
        if any(word in message_lower for word in turkish_words):
            cultural_additions.append("🇹🇷 I love that you're trying some Turkish! That's wonderful!")
        
        # Add cultural context for neighborhoods
        neighborhood_contexts = {
            "sultanahmet": "This historic area was the heart of both Byzantine and Ottoman empires!",
            "beyoğlu": "This district represents Istanbul's cosmopolitan side with European influences!",
            "kadıköy": "The Asian side offers a more authentic, local Istanbul experience!"
        }
        
        for entity in entities:
            if entity["type"] == "NEIGHBORHOOD":
                neighborhood = entity["text"].lower()
                if neighborhood in neighborhood_contexts:
                    cultural_additions.append(f"🏛️ Cultural note: {neighborhood_contexts[neighborhood]}")
        
        return " ".join(cultural_additions) if cultural_additions else None
    
    def _generate_personalized_suggestions(self, memory: ConversationMemory, 
                                         intent: str, entities: List[Dict[str, Any]]) -> str:
        """Generate personalized follow-up suggestions"""
        suggestions = []
        
        # Based on previous interests
        if 'neighborhoods' in memory.user_preferences:
            favorite_neighborhoods = memory.user_preferences['neighborhoods']
            if len(favorite_neighborhoods) > 0:
                suggestions.append(f"Since you're interested in {favorite_neighborhoods[0]}, "
                                 f"you might also love exploring nearby areas!")
        
        # Intent-based suggestions
        if intent == "restaurant_recommendation":
            suggestions.append("Would you like directions to any of these restaurants?")
            suggestions.append("I can also suggest the best times to visit to avoid crowds!")
        elif intent == "neighborhood_info":
            suggestions.append("I can recommend specific restaurants or attractions in this area!")
        
        return " ".join(suggestions[:2]) if suggestions else None
    
    def _update_user_analytics(self, user_id: str, intent: str, confidence: float,
                             processing_time: float, entities: List[Dict[str, Any]]):
        """Update user analytics and learning"""
        analytics = self.user_analytics[user_id]
        
        # Update counters
        analytics['total_interactions'] = analytics.get('total_interactions', 0) + 1
        analytics['total_processing_time'] = analytics.get('total_processing_time', 0) + processing_time
        
        # Intent tracking
        intent_counts = analytics.setdefault('intent_counts', {})
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Confidence tracking
        confidences = analytics.setdefault('confidences', [])
        confidences.append(confidence)
        
        # Keep only recent confidences
        if len(confidences) > 100:
            analytics['confidences'] = confidences[-100:]
        
        # Entity tracking
        entity_counts = analytics.setdefault('entity_counts', {})
        for entity in entities:
            entity_type = entity['type']
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        # Update timestamp
        analytics['last_interaction'] = datetime.now().isoformat()
    
    def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user analytics"""
        if user_id not in self.user_analytics:
            return {"message": "No analytics data available for this user"}
        
        analytics = self.user_analytics[user_id]
        memory = self.conversation_memories.get(user_id)
        
        # Calculate derived metrics
        avg_confidence = np.mean(analytics.get('confidences', [0.5]))
        avg_processing_time = (analytics.get('total_processing_time', 0) / 
                             max(analytics.get('total_interactions', 1), 1))
        
        # Most common intent
        intent_counts = analytics.get('intent_counts', {})
        most_common_intent = max(intent_counts.items(), key=lambda x: x[1])[0] if intent_counts else None
        
        return {
            "user_id": user_id,
            "total_interactions": analytics.get('total_interactions', 0),
            "average_confidence": round(avg_confidence, 3),
            "average_processing_time": round(avg_processing_time, 3),
            "most_common_intent": most_common_intent,
            "intent_distribution": intent_counts,
            "favorite_neighborhoods": memory.user_preferences.get('neighborhoods', []) if memory else [],
            "favorite_cuisines": memory.user_preferences.get('cuisines', []) if memory else [],
            "satisfaction_score": memory.satisfaction_score if memory else 0.0,
            "last_interaction": analytics.get('last_interaction')
        }

# Export main class
__all__ = ['DeepLearningEnhancedAI', 'ConversationMemory', 'ConversationTone', 'UserType']

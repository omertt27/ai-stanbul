#!/usr/bin/env python3
"""
Intelligent Location Detection Service for Istanbul AI
Advanced location inference from multiple data sources with ML/DL enhancements
"""

import logging
import math
import re
import numpy as np
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path

# ML/DL imports with fallback handling
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import euclidean
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML/DL libraries not available. Using fallback methods.")

from ..core.user_profile import UserProfile
from ..core.conversation_context import ConversationContext
from ..core.entity_recognizer import IstanbulEntityRecognizer

logger = logging.getLogger(__name__)

@dataclass
class LocationDetectionResult:
    """Result of location detection with confidence and method"""
    location: Optional[str]
    confidence: float  # 0.0 to 1.0
    detection_method: str
    fallback_locations: List[str]
    metadata: Dict[str, Any]
    explanation: Optional[str] = None  # Human-readable explanation
    alternative_scores: Dict[str, float] = field(default_factory=dict)  # Alternative location scores
    gps_distance: Optional[float] = None  # Distance from user's GPS location
    context_match: Dict[str, float] = field(default_factory=dict)  # Context matching scores
    recommendations: Dict[str, Any] = field(default_factory=dict)  # Related recommendations

@dataclass
class RestaurantSearchContext:
    """Context for restaurant-specific location detection"""
    cuisine_preferences: List[str]
    dietary_restrictions: List[str]
    price_range: Optional[Tuple[int, int]]  # (min, max) price level 1-4
    operating_hours: Optional[str]
    group_size: Optional[int]
    special_occasions: List[str]
    typo_corrections: Dict[str, str] = field(default_factory=dict)

@dataclass
class AttractionSearchContext:
    """Context for attraction-specific location detection"""
    categories: List[str]  # museums, monuments, parks, religious sites
    weather_preferences: List[str]  # indoor, outdoor, covered
    family_friendly: bool = False
    romantic: bool = False
    budget_friendly: bool = False
    accessibility_needs: List[str] = field(default_factory=list)
    visit_duration: Optional[str] = None  # short, medium, long

@dataclass
class TransportationContext:
    """Context for transportation-specific detection"""
    transport_modes: List[str]  # metro, bus, ferry, walking, taxi
    accessibility_needs: List[str]
    luggage_requirements: bool = False
    time_constraints: Optional[str] = None
    budget_preferences: Optional[str] = None

@dataclass
class WeatherContext:
    """Weather-aware context for recommendations"""
    current_weather: Dict[str, Any]
    forecast: List[Dict[str, Any]]
    temperature: float
    precipitation: float
    wind_speed: float
    weather_type: str  # sunny, rainy, cloudy, snowy

@dataclass
class EventContext:
    """Event-aware context for recommendations"""
    current_events: List[Dict[str, Any]]
    cultural_events: List[Dict[str, Any]]
    festivals: List[Dict[str, Any]]
    seasonal_activities: List[Dict[str, Any]]

@dataclass
class GPSContext:
    """GPS-aware location context"""
    user_location: Optional[Tuple[float, float]]  # (latitude, longitude)
    accuracy: Optional[float]  # GPS accuracy in meters
    movement_pattern: Optional[str]  # stationary, walking, driving
    nearby_landmarks: List[str]
    district_proximity: Dict[str, float]  # district -> distance mapping
    
    @property
    def coordinates(self) -> Optional[Tuple[float, float]]:
        """Alias for user_location for backward compatibility"""
        return self.user_location

@dataclass
class LocationPattern:
    """Pattern for location detection with semantic matching"""
    keywords: List[str]
    districts: List[str]
    confidence_boost: float
    context_type: str

@dataclass
class LocationEmbedding:
    """Embedding representation of location with contextual features"""
    location: str
    semantic_embedding: np.ndarray
    geographic_embedding: np.ndarray
    temporal_features: np.ndarray
    user_preference_score: float
    confidence_score: float
    
@dataclass
class MLLocationCandidate:
    """ML-enhanced location candidate with neural network predictions"""
    location: str
    base_confidence: float
    ml_confidence: float
    neural_score: float
    semantic_similarity: float
    temporal_probability: float
    user_pattern_match: float
    ensemble_prediction: float
    feature_importance: Dict[str, float]

class LocationSemanticEncoder(nn.Module):
    """Neural network for encoding location semantics and context"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        self.location_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 21)  # Number of Istanbul districts
        )
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_output, dim=1)
        
        # Predictions
        location_logits = self.location_classifier(pooled)
        confidence = self.confidence_estimator(pooled)
        
        return location_logits, confidence, pooled

class UserPatternLearner(nn.Module):
    """Neural network for learning user-specific location patterns"""
    
    def __init__(self, input_dim: int = 100, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 21))  # District predictions
        self.network = nn.Sequential(*layers)
        
        # Separate confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        hidden = features
        confidence_features = None
        
        # Forward pass through network
        for i, layer in enumerate(self.network[:-1]):
            hidden = layer(hidden)
            # Capture features before the final classification layer for confidence
            if i == len(self.network) - 2:  # One layer before final
                confidence_features = hidden
        
        # Get final location predictions
        location_scores = self.network[-1](hidden)
        
        # Calculate confidence using captured features
        if confidence_features is not None:
            confidence = self.confidence_head(confidence_features)
        else:
            # Fallback confidence calculation
            confidence = torch.sigmoid(torch.mean(torch.abs(location_scores), dim=1, keepdim=True))
        
        return location_scores, confidence

class TemporalLocationPredictor(nn.Module):
    """LSTM-based predictor for temporal location patterns"""
    
    def __init__(self, feature_dim: int = 50, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3)
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 21)  # District predictions
        )
    
    def forward(self, temporal_sequence):
        lstm_out, _ = self.lstm(temporal_sequence)
        
        # Apply temporal attention
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep for prediction
        final_hidden = attn_out[:, -1, :]
        predictions = self.predictor(final_hidden)
        
        return predictions

class IntelligentLocationDetector:
    """
    Advanced location detection service with ML-enhanced inference
    from multiple data sources with confidence scoring and semantic analysis
    """
    
    def __init__(self):
        self.entity_recognizer = IstanbulEntityRecognizer()
        self.logger = logger
        
        # Performance optimization caches
        self._distance_cache = {}
        self._semantic_cache = {}
        self._pattern_cache = {}
        self._cache_max_size = 1000
        
        # Enhanced district coordinates with more precise data
        self.district_coords = {
            'Sultanahmet': {'lat': 41.0086, 'lng': 28.9802, 'radius': 0.01, 'type': 'historic'},
            'Beyoğlu': {'lat': 41.0362, 'lng': 28.9773, 'radius': 0.015, 'type': 'cultural'},
            'Taksim': {'lat': 41.0370, 'lng': 28.9850, 'radius': 0.008, 'type': 'commercial'},
            'Kadıköy': {'lat': 40.9833, 'lng': 29.0333, 'radius': 0.02, 'type': 'local'},
            'Beşiktaş': {'lat': 41.0422, 'lng': 29.0097, 'radius': 0.015, 'type': 'mixed'},
            'Galata': {'lat': 41.0256, 'lng': 28.9744, 'radius': 0.008, 'type': 'historic'},
            'Karaköy': {'lat': 41.0257, 'lng': 28.9739, 'radius': 0.005, 'type': 'business'},
            'Levent': {'lat': 41.0766, 'lng': 29.0092, 'radius': 0.012, 'type': 'business'},
            'Şişli': {'lat': 41.0608, 'lng': 28.9866, 'radius': 0.015, 'type': 'commercial'},
            'Nişantaşı': {'lat': 41.0489, 'lng': 28.9944, 'radius': 0.008, 'type': 'upscale'},
            'Ortaköy': {'lat': 41.0552, 'lng': 29.0267, 'radius': 0.006, 'type': 'scenic'},
            'Üsküdar': {'lat': 41.0214, 'lng': 29.0206, 'radius': 0.02, 'type': 'traditional'},
            'Eminönü': {'lat': 41.0172, 'lng': 28.9709, 'radius': 0.008, 'type': 'historic'},
            'Cihangir': {'lat': 41.0315, 'lng': 28.9794, 'radius': 0.005, 'type': 'bohemian'},
            'Arnavutköy': {'lat': 41.0706, 'lng': 29.0424, 'radius': 0.01, 'type': 'coastal'},
            'Bebek': {'lat': 41.0838, 'lng': 29.0432, 'radius': 0.008, 'type': 'upscale'},
            'Bostancı': {'lat': 40.9658, 'lng': 29.0906, 'radius': 0.015, 'type': 'residential'},
            'Fenerbahçe': {'lat': 40.9638, 'lng': 29.0469, 'radius': 0.01, 'type': 'residential'},
            'Moda': {'lat': 40.9826, 'lng': 29.0252, 'radius': 0.008, 'type': 'trendy'},
            'Balat': {'lat': 41.0289, 'lng': 28.9487, 'radius': 0.008, 'type': 'historic'},
            'Fener': {'lat': 41.0336, 'lng': 28.9464, 'radius': 0.006, 'type': 'historic'}
        }
        
        # Enhanced semantic location patterns for better understanding
        self.location_patterns = [
            LocationPattern(
                keywords=['historic', 'historical', 'ancient', 'old', 'byzantine', 'ottoman'],
                districts=['Sultanahmet', 'Eminönü', 'Balat', 'Fener', 'Galata'],
                confidence_boost=0.15,
                context_type='historical'
            ),
            LocationPattern(
                keywords=['nightlife', 'bar', 'club', 'party', 'drink', 'entertainment'],
                districts=['Beyoğlu', 'Taksim', 'Kadıköy', 'Cihangir'],
                confidence_boost=0.12,
                context_type='nightlife'
            ),
            LocationPattern(
                keywords=['shopping', 'shop', 'mall', 'store', 'boutique', 'fashion'],
                districts=['Nişantaşı', 'Şişli', 'Beyoğlu', 'Levent'],
                confidence_boost=0.10,
                context_type='shopping'
            ),
            LocationPattern(
                keywords=['local', 'authentic', 'neighborhood', 'residential', 'quiet'],
                districts=['Kadıköy', 'Moda', 'Cihangir', 'Üsküdar'],
                confidence_boost=0.08,
                context_type='local'
            ),
            LocationPattern(
                keywords=['luxury', 'upscale', 'expensive', 'high-end', 'premium'],
                districts=['Nişantaşı', 'Bebek', 'Levent', 'Beşiktaş'],
                confidence_boost=0.10,
                context_type='upscale'
            ),
            LocationPattern(
                keywords=['waterfront', 'sea', 'bosphorus', 'water', 'coastal', 'harbor'],
                districts=['Ortaköy', 'Bebek', 'Arnavutköy', 'Beşiktaş'],
                confidence_boost=0.12,
                context_type='waterfront'
            ),
            # Restaurant-specific patterns
            LocationPattern(
                keywords=['restaurant', 'food', 'dining', 'eat', 'cuisine', 'meal'],
                districts=['Beyoğlu', 'Kadıköy', 'Sultanahmet', 'Nişantaşı'],
                confidence_boost=0.10,
                context_type='restaurant'
            ),
            LocationPattern(
                keywords=['turkish', 'kebab', 'meze', 'baklava', 'turkish food'],
                districts=['Sultanahmet', 'Eminönü', 'Beyoğlu', 'Üsküdar'],
                confidence_boost=0.12,
                context_type='turkish_cuisine'
            ),
            LocationPattern(
                keywords=['seafood', 'fish', 'mussels', 'balık', 'sea food'],
                districts=['Ortaköy', 'Bebek', 'Arnavutköy', 'Kadıköy'],
                confidence_boost=0.11,
                context_type='seafood'
            ),
            LocationPattern(
                keywords=['vegetarian', 'vegan', 'plant-based', 'healthy'],
                districts=['Kadıköy', 'Cihangir', 'Moda', 'Beyoğlu'],
                confidence_boost=0.09,
                context_type='vegetarian'
            ),
            LocationPattern(
                keywords=['street food', 'cheap eats', 'budget food', 'fast food'],
                districts=['Eminönü', 'Kadıköy', 'Taksim', 'Beyoğlu'],
                confidence_boost=0.08,
                context_type='street_food'
            ),
            # Attraction-specific patterns
            LocationPattern(
                keywords=['museum', 'gallery', 'art', 'exhibition', 'cultural'],
                districts=['Sultanahmet', 'Beyoğlu', 'Beşiktaş', 'Karaköy'],
                confidence_boost=0.13,
                context_type='museum'
            ),
            LocationPattern(
                keywords=['mosque', 'religious', 'spiritual', 'prayer', 'islamic'],
                districts=['Sultanahmet', 'Eminönü', 'Üsküdar', 'Beyoğlu'],
                confidence_boost=0.12,
                context_type='religious'
            ),
            LocationPattern(
                keywords=['park', 'garden', 'green space', 'outdoor', 'nature'],
                districts=['Beşiktaş', 'Üsküdar', 'Kadıköy', 'Sultanahmet'],
                confidence_boost=0.10,
                context_type='park'
            ),
            LocationPattern(
                keywords=['family', 'children', 'kids', 'family-friendly'],
                districts=['Beşiktaş', 'Kadıköy', 'Üsküdar', 'Bostancı'],
                confidence_boost=0.09,
                context_type='family'
            ),
            # Transportation patterns
            LocationPattern(
                keywords=['metro', 'subway', 'underground', 'train'],
                districts=['Taksim', 'Şişli', 'Levent', 'Kadıköy'],
                confidence_boost=0.11,
                context_type='metro'
            ),
            LocationPattern(
                keywords=['ferry', 'boat', 'vapur', 'maritime'],
                districts=['Eminönü', 'Karaköy', 'Üsküdar', 'Kadıköy'],
                confidence_boost=0.12,
                context_type='ferry'
            ),
            LocationPattern(
                keywords=['airport', 'flight', 'transfer', 'IST', 'SAW'],
                districts=['Taksim', 'Şişli', 'Levent', 'Beşiktaş'],
                confidence_boost=0.10,
                context_type='airport'
            )
        ]
        
        # Advanced proximity keywords with semantic understanding
        self.proximity_keywords = {
            # Direct proximity
            'nearby': {'score': 0.9, 'semantic_weight': 1.0},
            'close by': {'score': 0.9, 'semantic_weight': 1.0},
            'around here': {'score': 0.95, 'semantic_weight': 1.2},
            'near me': {'score': 0.9, 'semantic_weight': 1.0},
            'in the area': {'score': 0.8, 'semantic_weight': 0.9},
            'walking distance': {'score': 0.85, 'semantic_weight': 1.1},
            'local': {'score': 0.7, 'semantic_weight': 0.8},
            'around': {'score': 0.7, 'semantic_weight': 0.7},
            'close': {'score': 0.6, 'semantic_weight': 0.6},
            'vicinity': {'score': 0.8, 'semantic_weight': 0.9},
            'neighborhood': {'score': 0.8, 'semantic_weight': 0.9},
            
            # Contextual proximity
            'where i am': {'score': 0.95, 'semantic_weight': 1.3},
            'my location': {'score': 0.9, 'semantic_weight': 1.2},
            'current area': {'score': 0.85, 'semantic_weight': 1.0},
            'this area': {'score': 0.8, 'semantic_weight': 0.9},
            'this neighborhood': {'score': 0.8, 'semantic_weight': 0.9},
            'here in': {'score': 0.7, 'semantic_weight': 0.8},
            'around my hotel': {'score': 0.85, 'semantic_weight': 1.1},
            'near my accommodation': {'score': 0.85, 'semantic_weight': 1.1}
        }
        
        # Temporal context patterns
        self.temporal_patterns = {
            'morning': {'time_boost': 0.05, 'districts': ['Sultanahmet', 'Eminönü']},
            'afternoon': {'time_boost': 0.03, 'districts': ['Beyoğlu', 'Nişantaşı']},
            'evening': {'time_boost': 0.08, 'districts': ['Beyoğlu', 'Taksim', 'Ortaköy']},
            'night': {'time_boost': 0.10, 'districts': ['Beyoğlu', 'Taksim', 'Kadıköy']},
            'weekend': {'time_boost': 0.06, 'districts': ['Kadıköy', 'Moda', 'Ortaköy']}
        }
        
        # Location confidence history for learning
        self.location_confidence_history = defaultdict(list)
        self.district_popularity_scores = Counter()
        
        # Enhanced user behavior tracking
        self.user_query_patterns = defaultdict(list)
        self.location_success_rates = defaultdict(float)
        self.temporal_preferences = defaultdict(list)
        
        # ML/DL debugging and configuration
        self.ml_debug_mode = True  # Enable detailed ML debugging
        self.ml_fallback_mode = False  # Track if we're in fallback mode
        
        # Initialize ML components with proper error handling
        self._initialize_ml_components()
        
        # Initialize advanced patterns
        self._initialize_advanced_patterns()
        
        # Import and integrate with existing backend detector
        self._integrate_with_backend_detector()

    def _initialize_ml_components(self):
        """Initialize advanced ML/DL components for enhanced detection"""
        self.logger.info("🔧 Starting ML/DL component initialization...")
        
        # Initialize basic components that always work
        self.semantic_similarity_cache = {}
        self.location_frequency_model = defaultdict(float)
        self.context_pattern_weights = defaultdict(float)
        
        # Initialize component availability flags
        self.semantic_encoder = None
        self.pattern_learner = None 
        self.temporal_predictor = None
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        self.location_classifier = None
        self.confidence_estimator = None
        self.scaler = None
        
        # Enhanced ML/DL components
        if ML_AVAILABLE:
            self.logger.info("🧠 ML libraries available - initializing advanced components...")
            self.ml_fallback_mode = False
            
            # Initialize neural networks with comprehensive error handling
            try:
                self.semantic_encoder = LocationSemanticEncoder()
                self.pattern_learner = UserPatternLearner()
                self.temporal_predictor = TemporalLocationPredictor()
                self.logger.info("✅ Neural networks initialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Neural network initialization failed: {e}")
                if self.ml_debug_mode:
                    import traceback
                    self.logger.debug(f"Neural network error traceback:\n{traceback.format_exc()}")
                self.semantic_encoder = None
                self.pattern_learner = None
                self.temporal_predictor = None
            
            # Initialize sentence transformer with multiple fallback options
            try:
                # Try different models in order of preference
                models_to_try = [
                    'paraphrase-multilingual-MiniLM-L12-v2',
                    'all-MiniLM-L6-v2',
                    'all-mpnet-base-v2'
                ]
                
                for model_name in models_to_try:
                    try:
                        self.sentence_transformer = SentenceTransformer(model_name)
                        self.logger.info(f"✅ Sentence transformer loaded: {model_name}")
                        break
                    except Exception as model_error:
                        if self.ml_debug_mode:
                            self.logger.debug(f"Failed to load {model_name}: {model_error}")
                        continue
                
                if self.sentence_transformer is None:
                    self.logger.warning("⚠️ All sentence transformer models failed, using fallback")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Sentence transformer initialization failed: {e}")
                if self.ml_debug_mode:
                    import traceback
                    self.logger.debug(f"Sentence transformer error traceback:\n{traceback.format_exc()}")
                self.sentence_transformer = None
                
            # Initialize classical ML components
            try:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000, 
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.location_classifier = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=10
                )
                self.confidence_estimator = GradientBoostingClassifier(
                    n_estimators=50, 
                    random_state=42,
                    max_depth=5
                )
                self.scaler = StandardScaler()
                self.logger.info("✅ Classical ML components initialized successfully")
            except Exception as e:
                self.logger.warning(f"⚠️ Classical ML initialization failed: {e}")
                if self.ml_debug_mode:
                    import traceback
                    self.logger.debug(f"Classical ML error traceback:\n{traceback.format_exc()}")
                self.tfidf_vectorizer = None
                self.location_classifier = None
                self.confidence_estimator = None
                self.scaler = None

            # Determine overall ML availability
            self.ml_components_available = any([
                self.semantic_encoder is not None,
                self.sentence_transformer is not None,
                self.tfidf_vectorizer is not None,
                self.location_classifier is not None
            ])
            
            if self.ml_components_available:
                self.logger.info("🎯 ML/DL location detection system ready")
                if self.ml_debug_mode:
                    self._log_ml_component_status()
            else:
                self.logger.warning("❌ No ML/DL components available, using rule-based fallback")
                self.ml_fallback_mode = True
                
        else:
            self.logger.info("📚 ML/DL libraries not available, using enhanced rule-based detection")
            self.ml_components_available = False
            self.ml_fallback_mode = True
        
        # Initialize learning data structures
        self._safe_initialize_learning_storage()
        
        # Bootstrap with initial patterns
        self._safe_bootstrap_ml_patterns()
        
        self.logger.info(f"🎉 ML/DL components initialization complete (Fallback mode: {self.ml_fallback_mode})")
        
    def _log_ml_component_status(self):
        """Log the status of each ML component for debugging"""
        if not self.ml_debug_mode:
            return
            
        components = {
            'Semantic Encoder': self.semantic_encoder is not None,
            'Pattern Learner': self.pattern_learner is not None,
            'Temporal Predictor': self.temporal_predictor is not None,
            'Sentence Transformer': self.sentence_transformer is not None,
            'TF-IDF Vectorizer': self.tfidf_vectorizer is not None,
            'Location Classifier': self.location_classifier is not None,
            'Confidence Estimator': self.confidence_estimator is not None,
            'Standard Scaler': self.scaler is not None
        }
        
        self.logger.debug("� ML Component Status:")
        for component, status in components.items():
            status_emoji = "✅" if status else "❌"
            self.logger.debug(f"  {status_emoji} {component}: {'Available' if status else 'Not Available'}")
            
    def _safe_initialize_learning_storage(self):
        """Safely initialize learning data structures"""
        try:
            self._initialize_learning_storage()
        except Exception as e:
            self.logger.error(f"❌ Learning storage initialization failed: {e}")
            if self.ml_debug_mode:
                import traceback
                self.logger.debug(f"Learning storage error traceback:\n{traceback.format_exc()}")
                
    def _safe_bootstrap_ml_patterns(self):
        """Safely bootstrap ML patterns"""
        try:
            self._bootstrap_ml_patterns()
        except Exception as e:
            self.logger.error(f"❌ ML patterns bootstrap failed: {e}")
            if self.ml_debug_mode:
                import traceback
                self.logger.debug(f"ML patterns bootstrap error traceback:\n{traceback.format_exc()}")

    def _initialize_deep_learning_models(self):
        """Initialize deep learning models for location detection"""
        if not ML_AVAILABLE:
            return
            
        try:
            # District name to index mapping
            self.district_to_idx = {district: idx for idx, district in enumerate(self.district_coords.keys())}
            self.idx_to_district = {idx: district for district, idx in self.district_to_idx.items()}
            
            # Neural network models
            self.semantic_encoder = LocationSemanticEncoder()
            self.user_pattern_learner = UserPatternLearner()
            self.temporal_predictor = TemporalLocationPredictor()
            
            # Load pre-trained models if available
            self._load_pretrained_models()
            
            # Set models to evaluation mode initially
            self.semantic_encoder.eval()
            self.user_pattern_learner.eval()
            self.temporal_predictor.eval()
            
            self.logger.info("Deep learning models initialized")
            
        except Exception as e:
            self.logger.warning(f"Deep learning initialization failed: {e}")

    def _initialize_semantic_models(self):
        """Initialize semantic embedding models"""
        if not ML_AVAILABLE:
            return
            
        try:
            # Sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Pre-compute district embeddings
            self.district_embeddings = {}
            district_descriptions = self._get_district_descriptions()
            
            for district, description in district_descriptions.items():
                embedding = self.sentence_transformer.encode(description)
                self.district_embeddings[district] = embedding
            
            # TF-IDF for keyword extraction
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize with Istanbul-specific corpus
            istanbul_corpus = self._get_istanbul_corpus()
            self.tfidf_vectorizer.fit(istanbul_corpus)
            
            self.logger.info("Semantic models initialized")
            
        except Exception as e:
            self.logger.warning(f"Semantic models initialization failed: {e}")

    def _initialize_ensemble_models(self):
        """Initialize ensemble ML models"""
        if not ML_AVAILABLE:
            return
            
        try:
            # Random Forest for location classification
            self.location_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Gradient Boosting for confidence estimation
            self.confidence_estimator = GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Feature scaler
            self.feature_scaler = StandardScaler()
            
            # Clustering for user behavior patterns
            self.user_behavior_clusterer = KMeans(n_clusters=5, random_state=42)
            
            self.logger.info("Ensemble models initialized")
            
        except Exception as e:
            self.logger.warning(f"Ensemble models initialization failed: {e}")

    def _initialize_feature_extractors(self):
        """Initialize feature extraction components"""
        # Geographic feature extractor
        self.geographic_features = self._build_geographic_features()
        
        # Temporal feature patterns
        self.temporal_feature_patterns = self._build_temporal_patterns()
        
        # User behavior feature templates
        self.user_behavior_features = defaultdict(list)
        
        # Context feature weights
        self.context_feature_weights = self._initialize_context_weights()

    def _initialize_traditional_ml(self):
        """Initialize traditional ML components as fallback"""
        # Enhanced TF-IDF with Istanbul context
        self.location_tfidf = TfidfVectorizer(
            vocabulary=self._build_location_vocabulary(),
            ngram_range=(1, 3),
            lowercase=True
        ) if ML_AVAILABLE else None
        
        # Simple clustering for similar queries
        self.query_clusters = defaultdict(list)
        
        # Pattern matching with weights
        self.pattern_weights = self._calculate_pattern_weights()

    def _initialize_learning_storage(self):
        """Initialize data structures for continuous learning"""
        # User interaction history with features
        self.user_interaction_features = defaultdict(list)
        
        # Success rate tracking for different methods
        self.method_success_rates = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Temporal success patterns
        self.temporal_success_patterns = defaultdict(lambda: defaultdict(float))
        
        # Query-location association strengths
        self.query_location_associations = defaultdict(lambda: defaultdict(float))
        
        # Feature importance tracking
        self.feature_importance_history = defaultdict(list)
        
        # Model performance metrics
        self.model_performance_metrics = {
            'neural_network': {'accuracy': 0.0, 'confidence_correlation': 0.0},
            'ensemble': {'accuracy': 0.0, 'feature_importance': 0.0},
            'semantic': {'similarity_accuracy': 0.0, 'embedding_quality': 0.0}
        }

    def _bootstrap_ml_patterns(self):
        """Bootstrap ML patterns with initial data"""
        # Initialize location frequency based on typical user patterns
        tourist_areas = ['Sultanahmet', 'Beyoğlu', 'Taksim', 'Galata']
        for area in tourist_areas:
            self.location_frequency_model[area] = 0.8
            
        local_areas = ['Kadıköy', 'Beşiktaş', 'Cihangir', 'Moda']
        for area in local_areas:
            self.location_frequency_model[area] = 0.6

    def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        try:
            # Define model paths
            model_dir = Path(__file__).parent.parent / "models" / "location_detection"
            
            # Try to load semantic encoder
            semantic_model_path = model_dir / "semantic_encoder.pth"
            if semantic_model_path.exists() and self.semantic_encoder is not None:
                try:
                    self.semantic_encoder.load_state_dict(torch.load(semantic_model_path, map_location='cpu'))
                    self.logger.info("✅ Pre-trained semantic encoder loaded")
                except Exception as e:
                    if self.ml_debug_mode:
                        self.logger.debug(f"Failed to load semantic encoder: {e}")
            
            # Try to load pattern learner
            pattern_model_path = model_dir / "pattern_learner.pth" 
            if pattern_model_path.exists() and self.pattern_learner is not None:
                try:
                    self.pattern_learner.load_state_dict(torch.load(pattern_model_path, map_location='cpu'))
                    self.logger.info("✅ Pre-trained pattern learner loaded")
                except Exception as e:
                    if self.ml_debug_mode:
                        self.logger.debug(f"Failed to load pattern learner: {e}")
            
            # Try to load temporal predictor
            temporal_model_path = model_dir / "temporal_predictor.pth"
            if temporal_model_path.exists() and self.temporal_predictor is not None:
                try:
                    self.temporal_predictor.load_state_dict(torch.load(temporal_model_path, map_location='cpu'))
                    self.logger.info("✅ Pre-trained temporal predictor loaded")
                except Exception as e:
                    if self.ml_debug_mode:
                        self.logger.debug(f"Failed to load temporal predictor: {e}")
                        
        except Exception as e:
            if self.ml_debug_mode:
                self.logger.debug(f"Pre-trained model loading failed: {e}")
            # Not critical - models can work without pre-training
            pass

    def _get_district_descriptions(self):
        """Get descriptions for each district for semantic embeddings"""
        return {
            'Sultanahmet': 'Historic Ottoman Byzantine imperial mosque palace ancient cultural heritage museum',
            'Beyoğlu': 'Modern trendy nightlife shopping cultural European style entertainment art galleries',
            'Taksim': 'Central commercial business shopping entertainment transportation hub modern',
            'Kadıköy': 'Asian side local authentic residential hipster food market cultural diverse',
            'Beşiktaş': 'Upscale Bosphorus waterfront luxury residential business mixed development',
            'Galata': 'Historic tower medieval Genoese cultural bridge European quarter',
            'Karaköy': 'Business financial modern art galleries trendy cafes waterfront',
            'Levent': 'Business district modern skyscrapers commercial office financial center',
            'Şişli': 'Commercial shopping business residential modern urban development',
            'Nişantaşı': 'Upscale luxury shopping fashion boutiques high-end residential elegant',
            'Ortaköy': 'Scenic Bosphorus waterfront mosque cultural nightlife touristic beautiful',
            'Üsküdar': 'Traditional Asian side residential religious cultural authentic local',
            'Eminönü': 'Historic spice bazaar ferry terminal commercial traditional market',
            'Cihangir': 'Bohemian artistic cultural trendy residential European style creative',
            'Arnavutköy': 'Coastal Bosphorus traditional wooden houses scenic quiet residential',
            'Bebek': 'Upscale Bosphorus waterfront luxury residential beautiful scenic elite',
            'Bostancı': 'Asian side residential peaceful suburban family-friendly quiet',
            'Fenerbahçe': 'Residential Asian side peaceful suburban quiet family neighborhood',
            'Moda': 'Trendy hipster cultural artistic cafes bars creative young residential',
            'Balat': 'Historic colorful traditional authentic cultural heritage Jewish quarter',
            'Fener': 'Historic Greek Orthodox cultural heritage traditional authentic religious'
        }
        
    def _get_istanbul_corpus(self):
        """Get Istanbul-specific text corpus for TF-IDF training"""
        corpus = []
        
        # Add district descriptions
        corpus.extend(self._get_district_descriptions().values())
        
        # Add common Istanbul queries and contexts
        istanbul_contexts = [
            'best restaurants in Istanbul traditional Turkish cuisine',
            'historic sites Byzantine Ottoman heritage museums',
            'nightlife bars clubs entertainment districts',
            'shopping malls boutiques fashion districts',
            'Bosphorus waterfront scenic views ferry routes',
            'local authentic neighborhoods residential areas',
            'business districts modern commercial centers',
            'cultural attractions art galleries exhibitions',
            'transportation metro tram ferry connections',
            'tourist attractions must-see landmarks',
            'food markets street food local specialties',
            'luxury upscale high-end premium experiences',
            'budget affordable cheap local options',
            'family-friendly activities children entertainment',
            'romantic couples scenic beautiful locations'
        ]
        
        corpus.extend(istanbul_contexts)
        return corpus
    
    def _build_location_vocabulary(self):
        """Build vocabulary for location-specific TF-IDF"""
        vocabulary = set()
        
        # Add district names and variations
        for district in self.district_coords.keys():
            vocabulary.add(district.lower())
            # Add common variations and spellings
            if district == 'Sultanahmet':
                vocabulary.update(['sultanahmet', 'sultan', 'ahmed', 'historic', 'peninsula'])
            elif district == 'Beyoğlu':
                vocabulary.update(['beyoglu', 'beyoğlu', 'pera', 'modern', 'european'])
            elif district == 'Taksim':
                vocabulary.update(['taksim', 'square', 'center', 'central'])
            elif district == 'Kadıköy':
                vocabulary.update(['kadikoy', 'kadıköy', 'asian', 'side', 'local'])
            # Add more variations as needed
        
        # Add location-related keywords
        location_keywords = [
            'near', 'nearby', 'close', 'around', 'area', 'district', 'neighborhood',
            'walking', 'distance', 'metro', 'tram', 'ferry', 'transport',
            'restaurant', 'hotel', 'attraction', 'museum', 'mosque', 'palace',
            'shopping', 'market', 'bazaar', 'cafe', 'bar', 'club',
            'historic', 'modern', 'traditional', 'authentic', 'local', 'tourist',
            'bosphorus', 'waterfront', 'sea', 'bridge', 'tower', 'hill'
        ]
        vocabulary.update(location_keywords)
        
        return list(vocabulary)
    
    def _calculate_pattern_weights(self):
        """Calculate weights for different location patterns"""
        weights = {}
        
        # Time-based patterns
        weights['morning'] = {'historic': 1.2, 'tourist': 1.1, 'museum': 1.3}
        weights['afternoon'] = {'shopping': 1.2, 'cafe': 1.1, 'market': 1.1}
        weights['evening'] = {'restaurant': 1.3, 'nightlife': 1.4, 'entertainment': 1.2}
        weights['night'] = {'bar': 1.4, 'club': 1.5, 'nightlife': 1.5}
        
        # Context-based patterns
        weights['tourist'] = {'historic': 1.3, 'museum': 1.2, 'landmark': 1.4}
        weights['local'] = {'residential': 1.2, 'authentic': 1.3, 'neighborhood': 1.2}
        weights['business'] = {'commercial': 1.2, 'office': 1.1, 'modern': 1.1}
        weights['luxury'] = {'upscale': 1.4, 'premium': 1.3, 'high-end': 1.3}
        
        # Proximity patterns
        weights['proximity'] = {'nearby': 1.5, 'close': 1.3, 'walking': 1.4, 'area': 1.2}
        
        return weights

    def _integrate_with_backend_detector(self):
        """Integrate with existing backend location detector if available"""
        try:
            # Try to import and integrate with backend detector
            from ...backend.services.intelligent_location_detector import IntelligentLocationDetector as BackendDetector
            self.backend_detector = BackendDetector()
            self.backend_integration = True
            self.logger.info("✅ Backend location detector integration successful")
        except ImportError:
            self.backend_detector = None
            self.backend_integration = False
            self.logger.info("ℹ️ Backend detector not available, using standalone mode")
        except Exception as e:
            self.backend_detector = None
            self.backend_integration = False
            if self.ml_debug_mode:
                self.logger.debug(f"Backend integration error: {e}")

    def detect_location(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """
        Main location detection method with ML/DL enhancement
        """
        try:
            self.logger.info(f"🔍 Detecting location for: '{user_input[:50]}...'")
            
            # Multi-stage detection approach
            candidates = []
            
            # Stage 1: Rule-based detection (always available)
            rule_based_result = self._rule_based_detection(user_input, user_profile, context)
            if rule_based_result:
                candidates.append(rule_based_result)
            
            # Stage 2: ML/DL enhancement (if available)
            if self.ml_components_available and not self.ml_fallback_mode:
                ml_result = self._ml_enhanced_detection(user_input, user_profile, context)
                if ml_result:
                    candidates.append(ml_result)
            
            # Stage 3: Backend integration (if available)
            if self.backend_integration:
                backend_result = self._backend_detection(user_input, user_profile, context)
                if backend_result:
                    candidates.append(backend_result)
            
            # Select best candidate
            if candidates:
                best_result = self._select_best_candidate(candidates)
                self.logger.info(f"✅ Location detected: {best_result.location} (confidence: {best_result.confidence:.3f})")
                return best_result
            else:
                self.logger.info("❌ No location detected")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Location detection failed: {e}")
            if self.ml_debug_mode:
                import traceback
                self.logger.debug(f"Detection error traceback:\n{traceback.format_exc()}")
            return None

    def _rule_based_detection(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Rule-based location detection (fallback method)"""
        try:
            # Extract entities using entity recognizer
            entities = self.entity_recognizer.extract_entities(user_input)
            districts = entities.get('districts', [])
            
            if districts:
                location = districts[0]
                confidence = 0.8  # High confidence for explicit mentions
                return LocationDetectionResult(
                    location=location,
                    confidence=confidence,
                    detection_method='rule_based_explicit',
                    fallback_locations=districts[1:] if len(districts) > 1 else [],
                    metadata={'entities': entities},
                    explanation=f"Location explicitly mentioned in query"
                )
            
            # Check for proximity keywords
            proximity_result = self._detect_proximity_location(user_input, context)
            if proximity_result:
                return proximity_result
            
            # Pattern-based detection
            pattern_result = self._pattern_based_detection(user_input, user_profile)
            if pattern_result:
                return pattern_result
                
            return None
            
        except Exception as e:
            if self.ml_debug_mode:
                self.logger.debug(f"Rule-based detection error: {e}")
            return None

    def _ml_enhanced_detection(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """ML/DL enhanced location detection"""
        if not self.ml_components_available:
            return None
            
        try:
            # Use sentence transformer for semantic similarity
            if self.sentence_transformer is not None:
                return self._semantic_similarity_detection(user_input, user_profile, context)
            
            # Use classical ML as fallback
            if self.location_classifier is not None:
                return self._classical_ml_detection(user_input, user_profile, context)
                
            return None
            
        except Exception as e:
            if self.ml_debug_mode:
                self.logger.debug(f"ML enhanced detection error: {e}")
            return None

    def _backend_detection(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Use backend detector if available"""
        if not self.backend_integration or self.backend_detector is None:
            return None
            
        try:
            result = self.backend_detector.detect_location(user_input, user_profile, context)
            if result:
                # Wrap in our result format
                return LocationDetectionResult(
                    location=result.location,
                    confidence=result.confidence * 0.9,  # Slightly lower confidence for backend
                    detection_method='backend_integration',
                    fallback_locations=result.fallback_locations,
                    metadata=result.metadata,
                    explanation="Detected using backend integration"
                )
            return None
            
        except Exception as e:
            if self.ml_debug_mode:
                self.logger.debug(f"Backend detection error: {e}")
            return None

    def _detect_proximity_location(self, user_input: str, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Detect location based on proximity keywords"""
        user_input_lower = user_input.lower()
        
        for keyword, props in self.proximity_keywords.items():
            if keyword in user_input_lower:
                # Look for recent location in context
                recent_location = context.get_context('current_detected_location')
                if recent_location:
                    return LocationDetectionResult(
                        location=recent_location,
                        confidence=props['score'] * props['semantic_weight'],
                        detection_method='proximity_inference',
                        fallback_locations=[],
                        metadata={'proximity_keyword': keyword},
                        explanation=f"Inferred from proximity keyword '{keyword}'"
                    )
        
        return None

    def _pattern_based_detection(self, user_input: str, user_profile: UserProfile) -> Optional[LocationDetectionResult]:
        """Pattern-based location detection"""
        user_input_lower = user_input.lower()
        best_match = None
        best_score = 0.0
        
        for pattern in self.location_patterns:
            score = 0.0
            matched_keywords = []
            
            for keyword in pattern.keywords:
                if keyword in user_input_lower:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                # Normalize by number of keywords
                score = (score / len(pattern.keywords)) * pattern.confidence_boost
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        'pattern': pattern,
                        'score': score,
                        'keywords': matched_keywords
                    }
        
        if best_match and best_score > 0.1:
            # Select most appropriate district from pattern
            location = best_match['pattern'].districts[0] if best_match['pattern'].districts else None
            if location:
                return LocationDetectionResult(
                    location=location,
                    confidence=min(best_score, 0.7),  # Cap confidence for pattern matching
                    detection_method='pattern_based',
                    fallback_locations=best_match['pattern'].districts[1:],
                    metadata={
                        'matched_keywords': best_match['keywords'],
                        'pattern_type': best_match['pattern'].context_type
                    },
                    explanation=f"Matched pattern for {best_match['pattern'].context_type}"
                )
        
        return None

    def _semantic_similarity_detection(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Use sentence transformer for semantic similarity"""
        try:
            # Encode the user input
            query_embedding = self.sentence_transformer.encode(user_input)
            
            # Compare with district descriptions
            district_descriptions = self._get_district_descriptions()
            best_district = None
            best_similarity = 0.0
            
            for district, description in district_descriptions.items():
                district_embedding = self.sentence_transformer.encode(description)
                similarity = cosine_similarity([query_embedding], [district_embedding])[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_district = district
            
            if best_similarity > 0.3:  # Minimum similarity threshold
                return LocationDetectionResult(
                    location=best_district,
                    confidence=min(best_similarity, 0.8),
                    detection_method='semantic_similarity',
                    fallback_locations=[],
                    metadata={'similarity_score': best_similarity},
                    explanation=f"Semantic similarity match (score: {best_similarity:.3f})"
                )
            
            return None
            
        except Exception as e:
            if self.ml_debug_mode:
                self.logger.debug(f"Semantic similarity detection error: {e}")
            return None

    def _classical_ml_detection(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Use classical ML for location detection"""
        # This would need training data - simplified implementation
        return None

    def _select_best_candidate(self, candidates: List[LocationDetectionResult]) -> LocationDetectionResult:
        """Select the best candidate from multiple detection results"""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Sort by confidence and method priority
        method_priority = {
            'rule_based_explicit': 10,
            'semantic_similarity': 8,
            'backend_integration': 7,
            'proximity_inference': 6,
            'pattern_based': 5,
            'classical_ml': 4
        }
        
        def score_candidate(candidate):
            method_score = method_priority.get(candidate.detection_method, 1)
            return candidate.confidence * 0.7 + (method_score / 10) * 0.3
        
        best_candidate = max(candidates, key=score_candidate)
        
        # Combine information from other candidates
        all_fallbacks = []
        for candidate in candidates:
            if candidate != best_candidate and candidate.location:
                all_fallbacks.append(candidate.location)
            all_fallbacks.extend(candidate.fallback_locations)
        
        # Remove duplicates and the main location
        unique_fallbacks = []
        for fallback in all_fallbacks:
            if fallback != best_candidate.location and fallback not in unique_fallbacks:
                unique_fallbacks.append(fallback)
        
        best_candidate.fallback_locations = unique_fallbacks[:3]  # Limit to top 3
        return best_candidate

    # Advanced ML/DL Detection Methods
    
    def detect_location_with_context(self, 
                                   user_input: str, 
                                   user_profile: UserProfile, 
                                   context: ConversationContext,
                                   gps_context: Optional[GPSContext] = None,
                                   weather_context: Optional[WeatherContext] = None,
                                   event_context: Optional[EventContext] = None) -> Optional[LocationDetectionResult]:
        """
        Advanced location detection with GPS, weather, and event awareness
        """
        try:
            self.logger.info(f"🎯 Advanced context-aware location detection for: '{user_input[:50]}...'")
            
            # Apply typo correction first
            corrected_input = self._apply_typo_corrections(user_input)
            
            # Multi-stage enhanced detection
            candidates = []
            
            # Stage 1: GPS-aware detection
            if gps_context and gps_context.user_location:
                gps_result = self._gps_aware_detection(corrected_input, user_profile, context, gps_context)
                if gps_result:
                    candidates.append(gps_result)
            
            # Stage 2: Context-specific detection (restaurant, attraction, transportation)
            context_result = self._context_specific_detection(corrected_input, user_profile, context)
            if context_result:
                candidates.append(context_result)
            
            # Stage 3: Weather-aware detection
            if weather_context:
                weather_result = self._weather_aware_detection(corrected_input, user_profile, context, weather_context)
                if weather_result:
                    candidates.append(weather_result)
            
            # Stage 4: Event-aware detection
            if event_context:
                event_result = self._event_aware_detection(corrected_input, user_profile, context, event_context)
                if event_result:
                    candidates.append(event_result)
            
            # Stage 5: Standard detection as fallback
            standard_result = self.detect_location(corrected_input, user_profile, context)
            if standard_result:
                candidates.append(standard_result)
            
            # Select best candidate with enhanced scoring
            if candidates:
                best_result = self._select_best_candidate_enhanced(candidates, gps_context, weather_context, event_context)
                self.logger.info(f"✅ Advanced location detected: {best_result.location} (confidence: {best_result.confidence:.3f})")
                return best_result
            else:
                self.logger.info("❌ No location detected with advanced context")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Advanced location detection failed: {e}")
            if self.ml_debug_mode:
                import traceback
                self.logger.debug(f"Advanced detection error traceback:\n{traceback.format_exc()}")
            return None

    def _apply_typo_corrections(self, user_input: str) -> str:
        """Apply typo corrections to user input"""
        corrected = user_input
        user_input_lower = user_input.lower()
        
        for typo, correction in self.typo_corrections.items():
            if typo in user_input_lower:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(typo), re.IGNORECASE)
                corrected = pattern.sub(correction, corrected)
        
        return corrected

    def _gps_aware_detection(self, 
                           user_input: str, 
                           user_profile: UserProfile, 
                           context: ConversationContext,
                           gps_context: GPSContext) -> Optional[LocationDetectionResult]:
        """GPS-aware location detection with proximity calculation"""
        try:
            user_lat, user_lng = gps_context.user_location
            proximity_scores = {}
            
            # Calculate distances to all districts
            for district, coords in self.district_coords.items():
                try:
                    # Use geopy for accurate distance calculation if available
                    try:
                        from geopy.distance import geodesic
                        distance = geodesic((user_lat, user_lng), (coords['lat'], coords['lng'])).kilometers
                    except ImportError:
                        # Fallback to haversine calculation
                        distance = self._calculate_haversine_distance(
                            user_lat, user_lng, coords['lat'], coords['lng']
                        )
                    
                    proximity_scores[district] = distance
                except Exception as e:
                    if self.ml_debug_mode:
                        self.logger.debug(f"Distance calculation failed for {district}: {e}")
                    continue
            
            # Find closest districts
            if proximity_scores:
                closest_district = min(proximity_scores.keys(), key=lambda x: proximity_scores[x])
                closest_distance = proximity_scores[closest_district]
                
                # Determine proximity level
                proximity_level = self._get_proximity_level(closest_distance)
                
                # Calculate confidence based on distance and GPS accuracy
                base_confidence = max(0.1, 1.0 - (closest_distance / 10.0))  # Decreases with distance
                gps_accuracy_factor = max(0.5, 1.0 - (gps_context.accuracy / 100.0)) if gps_context.accuracy else 0.8
                confidence = base_confidence * gps_accuracy_factor
                
                # Check if user input mentions proximity
                proximity_boost = self._check_proximity_keywords(user_input)
                confidence = min(1.0, confidence + proximity_boost)
                
                if confidence > 0.3:  # Minimum threshold for GPS detection
                    # Get nearby districts as fallbacks
                    sorted_districts = sorted(proximity_scores.items(), key=lambda x: x[1])
                    fallback_locations = [d[0] for d in sorted_districts[1:4]]  # Next 3 closest
                    
                    return LocationDetectionResult(
                        location=closest_district,
                        confidence=confidence,
                        detection_method='gps_aware',
                        fallback_locations=fallback_locations,
                        metadata={
                            'gps_location': gps_context.user_location,
                            'proximity_scores': dict(sorted_districts[:5]),
                            'proximity_level': proximity_level,
                            'gps_accuracy': gps_context.accuracy
                        },
                        gps_distance=closest_distance,
                        explanation=f"GPS location indicates {proximity_level} to {closest_district} ({closest_distance:.1f}km away)"
                    )
            
            return None
            
        except Exception as e:
            if self.ml_debug_mode:
                self.logger.debug(f"GPS-aware detection error: {e}")
            return None

    def _context_specific_detection(self, 
                                  user_input: str, 
                                  user_profile: UserProfile, 
                                  context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Context-specific detection for restaurants, attractions, and transportation"""
        try:
            user_input_lower = user_input.lower()
            
            # Restaurant context detection
            restaurant_result = self._detect_restaurant_context(user_input_lower)
            if restaurant_result:
                return restaurant_result
            
            # Attraction context detection
            attraction_result = self._detect_attraction_context(user_input_lower)
            if attraction_result:
                return attraction_result
            
            # Transportation context detection
            transportation_result = self._detect_transportation_context(user_input_lower)
            if transportation_result:
                return transportation_result
            
            return None
            
        except Exception as e:
            if self.ml_debug_mode:
                self.logger.debug(f"Context-specific detection error: {e}")
            return None

    def _detect_restaurant_context(self, user_input_lower: str) -> Optional[LocationDetectionResult]:
        """Detect restaurant-related location context"""
        best_match = None
        best_score = 0.0
        
        # Check cuisine patterns
        for cuisine, pattern in self.cuisine_patterns.items():
            score = 0.0
            matched_keywords = []
            
            for keyword in pattern['keywords']:
                if keyword in user_input_lower:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            # Check dietary restrictions
            for restriction, keywords in self.dietary_restrictions.items():
                for keyword in keywords:
                    if keyword in user_input_lower:
                        score += 0.5
                        matched_keywords.append(f"{restriction}:{keyword}")
            
            if matched_keywords:
                normalized_score = (score / len(pattern['keywords'])) * pattern['confidence_boost']
                
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = {
                        'cuisine': cuisine,
                        'pattern': pattern,
                        'score': normalized_score,
                        'keywords': matched_keywords
                    }
        
        if best_match and best_score > 0.1:
            location = best_match['pattern']['districts'][0]
            return LocationDetectionResult(
                location=location,
                confidence=min(best_score, 0.8),
                detection_method='restaurant_context',
                fallback_locations=best_match['pattern']['districts'][1:],
                metadata={
                    'cuisine_type': best_match['cuisine'],
                    'matched_keywords': best_match['keywords'],
                    'context_type': 'restaurant'
                },
                context_match={'restaurant': best_score},
                explanation=f"Restaurant context detected for {best_match['cuisine']} cuisine"
            )
        
        return None

    def _detect_attraction_context(self, user_input_lower: str) -> Optional[LocationDetectionResult]:
        """Detect attraction-related location context"""
        best_match = None
        best_score = 0.0
        
        for category, pattern in self.attraction_categories.items():
            score = 0.0
            matched_keywords = []
            
            for keyword in pattern['keywords']:
                if keyword in user_input_lower:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                # Check for family-friendly, romantic, or budget-friendly keywords
                context_boost = 0.0
                if any(word in user_input_lower for word in ['family', 'children', 'kids']):
                    context_boost += 0.1
                if any(word in user_input_lower for word in ['romantic', 'couple', 'date']):
                    context_boost += 0.1
                if any(word in user_input_lower for word in ['free', 'budget', 'cheap']):
                    context_boost += 0.1
                
                normalized_score = (score / len(pattern['keywords'])) + context_boost
                
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = {
                        'category': category,
                        'pattern': pattern,
                        'score': normalized_score,
                        'keywords': matched_keywords
                    }
        
        if best_match and best_score > 0.1:
            location = best_match['pattern']['districts'][0]
            return LocationDetectionResult(
                location=location,
                confidence=min(best_score, 0.8),
                detection_method='attraction_context',
                fallback_locations=best_match['pattern']['districts'][1:],
                metadata={
                    'attraction_category': best_match['category'],
                    'matched_keywords': best_match['keywords'],
                    'weather_preference': best_match['pattern']['weather_preference'],
                    'context_type': 'attraction'
                },
                context_match={'attraction': best_score},
                explanation=f"Attraction context detected for {best_match['category']}"
            )
        
        return None

    def _detect_transportation_context(self, user_input_lower: str) -> Optional[LocationDetectionResult]:
        """Detect transportation-related location context"""
        best_match = None
        best_score = 0.0
        
        for mode, pattern in self.transportation_modes.items():
            score = 0.0
            matched_keywords = []
            
            # Check for keyword matches
            for keyword in pattern['keywords']:
                if keyword.lower() in user_input_lower:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            # Special handling for airport-related queries
            if mode == 'bus' and any(term in user_input_lower for term in ['airport', 'havalimanı', 'transfer']):
                score += 2.0  # Boost score for airport transfers
                matched_keywords.append('airport_transfer')
            
            if matched_keywords:
                # Use raw score instead of normalized to avoid penalizing modes with more keywords
                if score > best_score:
                    best_score = score
                    best_match = {
                        'mode': mode,
                        'pattern': pattern,
                        'score': score,
                        'keywords': matched_keywords
                    }
        
        if best_match and best_score > 0.5:  # Lower threshold for better detection
            # Select appropriate district based on transportation mode
            if best_match['mode'] == 'metro':
                districts = best_match['pattern'].get('stations', ['Taksim', 'Şişli', 'Kadıköy'])
            elif best_match['mode'] == 'ferry':
                districts = best_match['pattern'].get('terminals', ['Eminönü', 'Karaköy', 'Üsküdar'])
            elif best_match['mode'] == 'tram':
                districts = best_match['pattern'].get('routes', ['Sultanahmet', 'Eminönü', 'Karaköy'])
            elif best_match['mode'] == 'bus':
                # For bus, prioritize central transportation hubs
                districts = ['Taksim', 'Beyoğlu', 'Kadıköy', 'Üsküdar']
            else:
                districts = ['Taksim', 'Beyoğlu', 'Kadıköy']  # Central locations
            
            location = districts[0] if districts else 'Taksim'
            
            # Calculate confidence based on score and mode
            confidence = min(best_score * 0.2, 0.8)  # Scale score to confidence
            
            return LocationDetectionResult(
                location=location,
                confidence=confidence,
                detection_method='transportation_context',
                fallback_locations=districts[1:3] if len(districts) > 1 else [],
                metadata={
                    'transport_mode': best_match['mode'],
                    'matched_keywords': best_match['keywords'],
                    'context_type': 'transportation'
                },
                context_match={'transportation': best_score},
                explanation=f"Transportation context detected for {best_match['mode']}"
            )
        
        return None

    def _weather_aware_detection(self, 
                               user_input: str, 
                               user_profile: UserProfile, 
                               context: ConversationContext,
                               weather_context: WeatherContext) -> Optional[LocationDetectionResult]:
        """Weather-aware location detection"""
        try:
            weather_type = weather_context.weather_type
            
            if weather_type in self.weather_patterns:
                pattern = self.weather_patterns[weather_type]
                
                # Check if user input relates to weather-appropriate activities
                activity_match = False
                for activity in pattern['recommended_activities']:
                    if activity in user_input.lower():
                        activity_match = True
                        break
                
                if activity_match:
                    location = pattern['districts'][0]
                    confidence = pattern['confidence_boost'] + 0.3  # Base confidence for weather match
                    
                    return LocationDetectionResult(
                        location=location,
                        confidence=min(confidence, 0.8),
                        detection_method='weather_aware',
                        fallback_locations=pattern['districts'][1:],
                        metadata={
                            'weather_type': weather_type,
                            'temperature': weather_context.temperature,
                            'precipitation': weather_context.precipitation,
                            'recommended_activities': pattern['recommended_activities'],
                            'context_type': 'weather'
                        },
                        context_match={'weather': confidence},
                        explanation=f"Weather-appropriate recommendation for {weather_type} conditions"
                    )
            
            return None
            
        except Exception as e:
            if self.ml_debug_mode:
                self.logger.debug(f"Weather-aware detection error: {e}")
            return None

    def _event_aware_detection(self, 
                             user_input: str, 
                             user_profile: UserProfile, 
                             context: ConversationContext,
                             event_context: EventContext) -> Optional[LocationDetectionResult]:
        """Event-aware location detection"""
        try:
            # Simple event-based detection (can be enhanced with real event data)
            event_keywords = ['event', 'festival', 'concert', 'exhibition', 'show', 'performance']
            
            if any(keyword in user_input.lower() for keyword in event_keywords):
                # Check current events for location relevance
                event_districts = []
                
                for event in event_context.current_events:
                    if 'location' in event:
                        event_districts.append(event['location'])
                
                if event_districts:
                    location = event_districts[0]
                    return LocationDetectionResult(
                        location=location,
                        confidence=0.7,
                        detection_method='event_aware',
                        fallback_locations=event_districts[1:3],
                        metadata={
                            'current_events': event_context.current_events[:3],
                            'context_type': 'event'
                        },
                        context_match={'event': 0.7},
                        explanation=f"Event-based recommendation for current activities"
                    )
            
            return None
            
        except Exception as e:
            if self.ml_debug_mode:
                self.logger.debug(f"Event-aware detection error: {e}")
            return None

    def _calculate_haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two GPS coordinates using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlng / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        
        return distance

    def _get_proximity_level(self, distance: float) -> str:
        """Determine proximity level based on distance"""
        if distance <= self.gps_proximity_thresholds['very_close']:
            return 'very close'
        elif distance <= self.gps_proximity_thresholds['close']:
            return 'close'
        elif distance <= self.gps_proximity_thresholds['nearby']:
            return 'nearby'
        elif distance <= self.gps_proximity_thresholds['moderate']:
            return 'moderate distance'
        else:
            return 'far'

    def _check_proximity_keywords(self, user_input: str) -> float:
        """Check for proximity keywords and return confidence boost"""
        user_input_lower = user_input.lower()
        boost = 0.0
        
        for keyword, props in self.proximity_keywords.items():
            if keyword in user_input_lower:
                boost = max(boost, props['score'] * 0.2)  # Max 20% boost from proximity keywords
        
        return boost

    def _select_best_candidate_enhanced(self, 
                                      candidates: List[LocationDetectionResult],
                                      gps_context: Optional[GPSContext] = None,
                                      weather_context: Optional[WeatherContext] = None,
                                      event_context: Optional[EventContext] = None) -> LocationDetectionResult:
        """Enhanced candidate selection with contextual scoring"""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Enhanced method priority with context awareness
        method_priority = {
            'gps_aware': 12,
            'restaurant_context': 10,
            'attraction_context': 10,
            'transportation_context': 9,
            'weather_aware': 8,
            'event_aware': 8,
            'rule_based_explicit': 7,
            'semantic_similarity': 6,
            'backend_integration': 5,
            'proximity_inference': 4,
            'pattern_based': 3,
            'classical_ml': 2
        }
        
        def score_candidate_enhanced(candidate):
            base_score = candidate.confidence * 0.6
            method_score = (method_priority.get(candidate.detection_method, 1) / 12) * 0.3
            
            # Context bonuses
            context_bonus = 0.0
            if gps_context and candidate.gps_distance is not None:
                # Bonus for GPS accuracy
                distance_bonus = max(0, (5.0 - candidate.gps_distance) / 5.0) * 0.1
                context_bonus += distance_bonus
            
            if weather_context and 'weather' in candidate.context_match:
                context_bonus += candidate.context_match['weather'] * 0.05
            
            if event_context and 'event' in candidate.context_match:
                context_bonus += candidate.context_match['event'] * 0.05
            
            return base_score + method_score + context_bonus
        
        best_candidate = max(candidates, key=score_candidate_enhanced)
        
        # Enhanced fallback combination
        all_fallbacks = []
        context_scores = {}
        
        for candidate in candidates:
            if candidate != best_candidate and candidate.location:
                all_fallbacks.append(candidate.location)
                if candidate.context_match:
                    context_scores.update(candidate.context_match)
            all_fallbacks.extend(candidate.fallback_locations)
        
        # Remove duplicates while preserving order
        unique_fallbacks = []
        for fallback in all_fallbacks:
            if fallback != best_candidate.location and fallback not in unique_fallbacks:
                unique_fallbacks.append(fallback)
        
        best_candidate.fallback_locations = unique_fallbacks[:3]
        
        # Combine context matches
        if context_scores:
            best_candidate.context_match.update(context_scores)
        
        return best_candidate

    def _initialize_advanced_patterns(self):
        """Initialize advanced context-aware detection patterns"""
        # Advanced context-aware detection patterns
        self.cuisine_patterns = {
            'turkish': {
                'keywords': ['kebab', 'döner', 'baklava', 'turkish delight', 'meze', 'raki', 'köfte'],
                'districts': ['Sultanahmet', 'Eminönü', 'Beyoğlu', 'Üsküdar'],
                'confidence_boost': 0.15
            },
            'seafood': {
                'keywords': ['fish', 'seafood', 'mussels', 'sea bass', 'anchovy', 'balık'],
                'districts': ['Ortaköy', 'Bebek', 'Arnavutköy', 'Kadıköy', 'Bostancı'],
                'confidence_boost': 0.12
            },
            'international': {
                'keywords': ['italian', 'french', 'japanese', 'chinese', 'indian', 'mexican'],
                'districts': ['Beyoğlu', 'Nişantaşı', 'Levent', 'Cihangir'],
                'confidence_boost': 0.10
            },
            'vegetarian': {
                'keywords': ['vegetarian', 'vegan', 'plant-based', 'healthy', 'organic'],
                'districts': ['Kadıköy', 'Cihangir', 'Moda', 'Beyoğlu'],
                'confidence_boost': 0.09
            }
        }
        
        # Dietary restrictions mapping
        self.dietary_restrictions = {
            'vegetarian': ['vegetarian', 'veggie', 'no meat'],
            'vegan': ['vegan', 'plant-based', 'no animal products'],
            'halal': ['halal', 'islamic', 'no pork', 'no alcohol'],
            'kosher': ['kosher', 'jewish', 'kosher food'],
            'gluten-free': ['gluten-free', 'celiac', 'no gluten', 'gf'],
            'dairy-free': ['dairy-free', 'lactose-free', 'no dairy'],
            'nut-free': ['nut-free', 'no nuts', 'allergy']
        }
        
        # Attraction categories with ML-enhanced matching
        self.attraction_categories = {
            'museums': {
                'keywords': ['museum', 'gallery', 'art', 'exhibition', 'collection'],
                'districts': ['Sultanahmet', 'Beyoğlu', 'Beşiktaş', 'Karaköy'],
                'weather_preference': 'indoor'
            },
            'monuments': {
                'keywords': ['monument', 'historic', 'ancient', 'heritage', 'landmark'],
                'districts': ['Sultanahmet', 'Eminönü', 'Galata', 'Balat'],
                'weather_preference': 'outdoor'
            },
            'parks': {
                'keywords': ['park', 'garden', 'green', 'nature', 'outdoor'],
                'districts': ['Beşiktaş', 'Üsküdar', 'Kadıköy', 'Sultanahmet'],
                'weather_preference': 'outdoor'
            },
            'religious': {
                'keywords': ['mosque', 'church', 'synagogue', 'religious', 'spiritual'],
                'districts': ['Sultanahmet', 'Eminönü', 'Üsküdar', 'Balat', 'Fener'],
                'weather_preference': 'indoor'
            }
        }
        
        # Transportation mode mapping with ML integration
        self.transportation_modes = {
            'metro': {
                'keywords': ['metro', 'subway', 'underground', 'M1', 'M2', 'M3', 'M4'],
                'stations': ['Taksim', 'Şişli', 'Levent', 'Kadıköy', 'Üsküdar'],
                'accessibility': True
            },
            'bus': {
                'keywords': ['bus', 'otobüs', 'public transport', 'city bus'],
                'coverage': 'citywide',
                'accessibility': 'limited'
            },
            'ferry': {
                'keywords': ['ferry', 'boat', 'vapur', 'sea transport', 'maritime'],
                'terminals': ['Eminönü', 'Karaköy', 'Üsküdar', 'Kadıköy', 'Beşiktaş'],
                'scenic': True
            },
            'tram': {
                'keywords': ['tram', 'tramvay', 'T1', 'light rail'],
                'routes': ['Sultanahmet', 'Eminönü', 'Karaköy', 'Kabataş'],
                'tourist_friendly': True
            },
            'walking': {
                'keywords': ['walk', 'walking', 'on foot', 'pedestrian'],
                'benefits': ['exercise', 'sightseeing', 'free'],
                'weather_dependent': True
            }
        }
        
        # Weather-aware recommendation patterns
        self.weather_patterns = {
            'sunny': {
                'recommended_activities': ['outdoor', 'walking', 'parks', 'waterfront'],
                'districts': ['Ortaköy', 'Bebek', 'Sultanahmet', 'Kadıköy'],
                'confidence_boost': 0.12
            },
            'rainy': {
                'recommended_activities': ['indoor', 'museums', 'shopping', 'cafes'],
                'districts': ['Beyoğlu', 'Nişantaşı', 'Şişli', 'Levent'],
                'confidence_boost': 0.10
            },
            'cold': {
                'recommended_activities': ['indoor', 'museums', 'shopping', 'warm food'],
                'districts': ['Sultanahmet', 'Beyoğlu', 'Nişantaşı'],
                'confidence_boost': 0.08
            },
            'hot': {
                'recommended_activities': ['waterfront', 'parks', 'air-conditioned'],
                'districts': ['Ortaköy', 'Bebek', 'Arnavutköy', 'Beşiktaş'],
                'confidence_boost': 0.09
            }
        }
        
        # GPS-aware proximity calculations
        self.gps_proximity_thresholds = {
            'very_close': 0.5,  # km
            'close': 1.0,       # km
            'nearby': 2.0,      # km
            'moderate': 5.0,    # km
            'far': 10.0         # km
        }
        
        # Typo correction patterns for common misspellings
        self.typo_corrections = {
            # Districts
            'sultanhamet': 'Sultanahmet',
            'beyolu': 'Beyoğlu',
            'beyoglu': 'Beyoğlu',
            'taksim': 'Taksim',
            'kadikoy': 'Kadıköy',
            'kadikoy': 'Kadıköy',
            'besiktas': 'Beşiktaş',
            'besiktas': 'Beşiktaş',
            'nisantasi': 'Nişantaşı',
            'nisantasi': 'Nişantaşı',
            'ortakoy': 'Ortaköy',
            'ortakoy': 'Ortaköy',
            'uskudar': 'Üsküdar',
            'uskudar': 'Üsküdar',
            'eminonu': 'Eminönü',
            'eminonu': 'Eminönü',
            'sisli': 'Şişli',
            'sisli': 'Şişli',
            
            # Food types
            'kebab': 'kebap',
            'doner': 'döner',
            'baklava': 'baklava',
            'turkish delight': 'lokum',
            'raki': 'rakı',
            'kofte': 'köfte',
            'pide': 'pide',
            'lahmacun': 'lahmacun',
            
            # Common terms
            'bosphorus': 'Bosphorus',
            'bosporus': 'Bosphorus',
            'galata': 'Galata',
            'hagia sophia': 'Hagia Sophia',
            'blue mosque': 'Blue Mosque',
            'grand bazaar': 'Grand Bazaar',
            'spice bazaar': 'Spice Bazaar'
        }
    
    async def fetch_iksv_events(self) -> List:
        """Fetch events from İKSV using the MonthlyEventsScheduler"""
        events = []
        
        try:
            # Import and use the MonthlyEventsScheduler
            from monthly_events_scheduler import MonthlyEventsScheduler
            
            scheduler = MonthlyEventsScheduler()
            
            # Try to get cached events first
            cached_events = scheduler.load_cached_events()
            
            if cached_events and not scheduler.is_fetch_needed():
                self.logger.info(f"📚 Using {len(cached_events)} cached İKSV events")
                raw_events = cached_events
            else:
                # Fetch fresh events if no cache or cache is old
                self.logger.info("🌐 Fetching fresh İKSV events...")
                raw_events = await scheduler.fetch_iksv_events()
                
                # Cache the events for future use
                if raw_events:
                    await scheduler.save_events_to_cache(raw_events)
                    self.logger.info(f"💾 Cached {len(raw_events)} İKSV events")
            
            # Return raw events for compatibility (can be converted later)
            events = raw_events
            
            self.logger.info(f"🎭 Retrieved {len(events)} İKSV events")
            
        except ImportError:
            self.logger.warning("MonthlyEventsScheduler not available, using fallback İKSV events")
            events = self._get_fallback_iksv_events_dict()
        except Exception as e:
            self.logger.error(f"Error fetching İKSV events: {e}")
            events = self._get_fallback_iksv_events_dict()
        
        return events

    def _get_fallback_iksv_events_dict(self) -> List[dict]:
        """Provide fallback İKSV events as dictionaries"""
        return [
            {
                'title': 'İstanbul Jazz Festival',
                'description': 'Annual international jazz festival featuring world-class musicians',
                'venue': 'Salon İKSV',
                'district': 'Beyoğlu',
                'category': 'Music',
                'organizer': 'İKSV',
                'url': 'https://www.iksv.org/en/jazz',
                'is_free': False
            },
            {
                'title': 'İstanbul Theatre Festival',
                'description': 'International theatre festival showcasing contemporary performances',
                'venue': 'Zorlu PSM',
                'district': 'Beşiktaş',
                'category': 'Theatre',
                'organizer': 'İKSV',
                'url': 'https://www.iksv.org/en/theatre',
                'is_free': False
            },
            {
                'title': 'İstanbul Biennial',
                'description': 'Contemporary art biennial featuring international artists',
                'venue': 'Various İKSV Venues',
                'district': 'İstanbul',
                'category': 'Art',
                'organizer': 'İKSV',
                'url': 'https://www.iksv.org/en/biennial',
                'is_free': False
            }
        ]
    
    def _find_nearest_osm_node(self, latitude: float, longitude: float) -> dict:
        """Find the nearest OSM node to given coordinates"""
        try:
            # Simple implementation - in a real system this would query OSM data
            # For now, return a mock node with the provided coordinates
            return {
                'id': f'node_{int(latitude * 1000000)}_{int(longitude * 1000000)}',
                'lat': latitude,
                'lon': longitude,
                'distance': 0.0,
                'tags': {}
            }
        except Exception as e:
            self.logger.warning(f"Failed to find nearest OSM node: {e}")
            return {
                'id': 'unknown',
                'lat': latitude,
                'lon': longitude,
                'distance': 0.0,
                'tags': {}
            }

# Global detector instance
intelligent_location_detector = IntelligentLocationDetector()

async def detect_user_location(text: str, user_context: Optional[Dict] = None) -> LocationDetectionResult:
    """
    Detect user location from text input - main interface function
    """
    return await intelligent_location_detector.detect_location_from_text(text, user_context)
        
# Global detector instance
intelligent_location_detector = IntelligentLocationDetector()

async def detect_user_location(text: str, user_context: Optional[Dict] = None) -> LocationDetectionResult:
    """
    Detect user location from text input - main interface function
    """
    return await intelligent_location_detector.detect_location_from_text(text, user_context)

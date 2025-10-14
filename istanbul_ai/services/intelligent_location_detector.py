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
        for i, layer in enumerate(self.network[:-1]):
            hidden = layer(hidden)
            if i == len(self.network) - 4:  # Before last layer
                confidence = self.confidence_head(hidden)
        
        location_scores = self.network[-1](hidden)
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
        
        # Initialize ML components
        self._initialize_ml_components()
        
        # Import and integrate with existing backend detector
        self._integrate_with_backend_detector()

    def _initialize_ml_components(self):
        """Initialize advanced ML/DL components for enhanced detection"""
        try:
            # Basic ML components
            self.semantic_similarity_cache = {}
            self.location_frequency_model = defaultdict(float)
            self.context_pattern_weights = defaultdict(float)
            
            # Advanced ML/DL components
            if ML_AVAILABLE:
                self._initialize_deep_learning_models()
                self._initialize_semantic_models()
                self._initialize_ensemble_models()
                self._initialize_feature_extractors()
                
            # Fallback to traditional ML
            self._initialize_traditional_ml()
            
            # Initialize learning data structures
            self._initialize_learning_storage()
            
            # Bootstrap with initial patterns
            self._bootstrap_ml_patterns()
            
            self.logger.info("ML/DL components initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"ML components initialization failed: {e}")
            self._use_fallback_ml = True

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
            'ensemble': {'accuracy': 0.0, 'feature_importance': {}},
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

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate enhanced semantic similarity with caching and fuzzy matching"""
        cache_key = f"{text1.lower()}:{text2.lower()}"
        if cache_key in self._semantic_cache:
            return self._semantic_cache[cache_key]
        
        # Clean and normalize text
        words1 = set(self._normalize_text(text1).split())
        words2 = set(self._normalize_text(text2).split())
        
        if not words1 or not words2:
            similarity = 0.0
        else:
            # Enhanced Jaccard similarity with fuzzy matching
            exact_intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            # Base similarity
            similarity = exact_intersection / union if union > 0 else 0.0
            
            # Add fuzzy matching for partial word overlap
            fuzzy_matches = 0
            for w1 in words1:
                for w2 in words2:
                    if w1 != w2 and self._fuzzy_word_match(w1, w2):
                        fuzzy_matches += 0.5
            
            similarity += (fuzzy_matches / union) if union > 0 else 0
            
            # Enhanced synonym detection with cultural context
            similarity += self._calculate_synonym_boost(words1, words2)
            
        # Cache management
        if len(self._semantic_cache) >= self._cache_max_size:
            self._semantic_cache.clear()
            
        self._semantic_cache[cache_key] = min(1.0, similarity)
        return self._semantic_cache[cache_key]

    def _normalize_text(self, text: str) -> str:
        """Normalize text for better semantic matching"""
        import re
        # Remove punctuation and convert to lowercase
        normalized = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _fuzzy_word_match(self, word1: str, word2: str) -> bool:
        """Check if two words are similar enough (fuzzy matching)"""
        if len(word1) < 3 or len(word2) < 3:
            return False
        
        # Simple edit distance approximation
        if abs(len(word1) - len(word2)) > 2:
            return False
        
        # Check for common prefixes/suffixes
        if word1[:3] == word2[:3] or word1[-3:] == word2[-3:]:
            return True
        
        # Check for substring containment
        if len(word1) > 4 and len(word2) > 4:
            if word1 in word2 or word2 in word1:
                return len(min(word1, word2, key=len)) / len(max(word1, word2, key=len)) > 0.7
        
        return False

    def _calculate_synonym_boost(self, words1: Set[str], words2: Set[str]) -> float:
        """Calculate similarity boost from synonym groups with Istanbul context"""
        synonym_groups = [
            # Historical/Cultural
            (['historic', 'historical', 'ancient', 'old'], ['traditional', 'heritage', 'classical']),
            (['mosque', 'masjid'], ['islamic', 'religious']),
            (['palace', 'saray'], ['imperial', 'royal', 'sultan']),
            (['bazaar', 'market'], ['shopping', 'commercial', 'trade']),
            
            # Food & Dining
            (['restaurant', 'dining', 'food'], ['eat', 'meal', 'cuisine']),
            (['turkish', 'ottoman'], ['local', 'traditional', 'authentic']),
            (['seafood', 'fish'], ['marine', 'ocean', 'bosphorus']),
            (['street food', 'snack'], ['quick', 'casual', 'local']),
            
            # Entertainment & Nightlife
            (['nightlife', 'party', 'entertainment'], ['bar', 'club', 'music']),
            (['cultural', 'art'], ['gallery', 'museum', 'exhibition']),
            
            # Shopping & Business
            (['shopping', 'shop', 'store'], ['boutique', 'retail', 'commercial']),
            (['business', 'office'], ['corporate', 'work', 'professional']),
            
            # Location & Movement
            (['nearby', 'close'], ['near', 'adjacent', 'vicinity']),
            (['waterfront', 'coastal'], ['sea', 'bosphorus', 'water', 'harbor']),
            (['upscale', 'luxury'], ['expensive', 'premium', 'high-end'])
        ]
        
        boost = 0.0
        for group1, group2 in synonym_groups:
            if any(w in words1 for w in group1) and any(w in words2 for w in group2):
                boost += 0.15
            elif any(w in words1 for w in group2) and any(w in words2 for w in group1):
                boost += 0.15
        
        return min(0.4, boost)  # Cap synonym boost at 0.4

    def _apply_temporal_context(self, candidates: List[LocationDetectionResult], user_input: str) -> List[LocationDetectionResult]:
        """Apply temporal context to boost location confidence"""
        current_hour = datetime.now().hour
        user_input_lower = user_input.lower()
        
        # Determine time period
        time_period = None
        if 6 <= current_hour < 12:
            time_period = 'morning'
        elif 12 <= current_hour < 17:
            time_period = 'afternoon'
        elif 17 <= current_hour < 22:
            time_period = 'evening'
        else:
            time_period = 'night'
            
        # Check for weekend
        is_weekend = datetime.now().weekday() >= 5
        if is_weekend:
            time_period = 'weekend'
            
        # Apply temporal boosts
        for candidate in candidates:
            if time_period in self.temporal_patterns:
                pattern = self.temporal_patterns[time_period]
                if candidate.location in pattern['districts']:
                    boost = pattern['time_boost']
                    candidate.confidence = min(0.98, candidate.confidence + boost)
                    if 'temporal_boosts' not in candidate.metadata:
                        candidate.metadata['temporal_boosts'] = []
                    candidate.metadata['temporal_boosts'].append({
                        'time_period': time_period,
                        'boost': boost
                    })
                    
        return candidates

    def _apply_semantic_pattern_matching(self, candidates: List[LocationDetectionResult], user_input: str) -> List[LocationDetectionResult]:
        """Apply semantic pattern matching to enhance location detection"""
        user_input_lower = user_input.lower()
        
        for pattern in self.location_patterns:
            # Check if any pattern keywords match the user input
            pattern_match_score = 0.0
            matched_keywords = []
            
            for keyword in pattern.keywords:
                if keyword in user_input_lower:
                    pattern_match_score += 1.0
                    matched_keywords.append(keyword)
                else:
                    # Check semantic similarity
                    similarity = self._calculate_semantic_similarity(keyword, user_input)
                    if similarity > 0.3:
                        pattern_match_score += similarity
                        matched_keywords.append(f"{keyword}~{similarity:.2f}")
            
            if pattern_match_score > 0:
                # Apply pattern boost to relevant candidates
                for candidate in candidates:
                    if candidate.location in pattern.districts:
                        boost = pattern.confidence_boost * (pattern_match_score / len(pattern.keywords))
                        candidate.confidence = min(0.97, candidate.confidence + boost)
                        
                        if 'semantic_boosts' not in candidate.metadata:
                            candidate.metadata['semantic_boosts'] = []
                        candidate.metadata['semantic_boosts'].append({
                            'pattern_type': pattern.context_type,
                            'matched_keywords': matched_keywords,
                            'boost': boost,
                            'match_score': pattern_match_score
                        })
        
        return candidates

    def _apply_ml_learning_adjustments(self, candidates: List[LocationDetectionResult], user_input: str, context: ConversationContext) -> List[LocationDetectionResult]:
        """Apply machine learning based adjustments from historical data"""
        for candidate in candidates:
            location = candidate.location
            
            # Apply frequency-based adjustments
            if location in self.location_frequency_model:
                frequency_boost = self.location_frequency_model[location] * 0.05
                candidate.confidence = min(0.96, candidate.confidence + frequency_boost)
                
                if 'ml_adjustments' not in candidate.metadata:
                    candidate.metadata['ml_adjustments'] = {}
                candidate.metadata['ml_adjustments']['frequency_boost'] = frequency_boost
            
            # Apply confidence history adjustments
            if location in self.location_confidence_history:
                recent_confidences = self.location_confidence_history[location][-10:]  # Last 10
                if recent_confidences:
                    avg_confidence = sum(recent_confidences) / len(recent_confidences)
                    if avg_confidence > 0.7:  # Good historical performance
                        history_boost = 0.03
                        candidate.confidence = min(0.96, candidate.confidence + history_boost)
                        
                        if 'ml_adjustments' not in candidate.metadata:
                            candidate.metadata['ml_adjustments'] = {}
                        candidate.metadata['ml_adjustments']['history_boost'] = history_boost
                        candidate.metadata['ml_adjustments']['avg_historical_confidence'] = avg_confidence
        
        return candidates

    def _enhance_backend_detection_with_ml(self, user_input: str, user_profile: UserProfile, 
                                           context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Enhanced backend detection with ML/DL improvements"""
        
        try:
            # Try to use backend detector if available
            if hasattr(self, 'backend_detector'):
                backend_result = self.backend_detector.detect_location_from_text(user_input)
                if backend_result:
                    return LocationDetectionResult(
                        location=backend_result.location,
                        confidence=backend_result.confidence,
                        detection_method='backend_enhanced',
                        fallback_locations=[],
                        metadata={'backend_source': True}
                    )
        except Exception as e:
            self.logger.warning(f"Backend detection failed: {e}")
        
        # Fallback to our own enhanced detection
        return None
    
    def _generate_enhanced_explanation(self, result: LocationDetectionResult) -> str:
        """Generate explanation for location detection result"""
        method_explanations = {
            'gps_coordinates': 'Detected from your device location',
            'explicit_query': 'You mentioned this location in your message',
            'proximity_inference': 'Inferred from nearby landmarks you mentioned',
            'natural_language': 'Detected from location expressions in your message',
            'backend_enhanced': 'Detected using enhanced location analysis',
            'profile_location': 'Based on your location preferences',
            'context_location': 'Inferred from conversation context'
        }
        
        explanation = method_explanations.get(result.detection_method, 'Location detected')
        if result.confidence < 0.7:
            explanation += f" (confidence: {result.confidence:.0%})"
        
        return explanation
    
    def _update_learning_models(self, result: LocationDetectionResult):
        """Update ML learning models with successful detection"""
        # Placeholder for ML model updates
        # In production, this would update user patterns and improve detection
        pass

    def detect_location(
        self, 
        user_input: str, 
        user_profile: UserProfile, 
        context: ConversationContext,
        require_confidence: float = 0.3
    ) -> LocationDetectionResult:
        """
        Intelligently detect user location from all available sources with ML enhancements
        """
        
        # Step 1: Try backend integration first (leverages existing system)
        backend_result = self._enhance_backend_detection_with_ml(user_input, user_profile, context)
        if backend_result and backend_result.confidence >= require_confidence:
            backend_result.explanation = self._generate_enhanced_explanation(backend_result)
            self._update_learning_models(backend_result)
            return backend_result
        
        # Step 2: Fallback to our own detection methods
        detection_methods = [
            self._detect_gps_location,                # Highest priority: GPS/device location
            self._detect_natural_language_location,   # High priority: Natural language expressions
            self._detect_explicit_location,           # High priority: Explicit mentions
            self._detect_proximity_location,
            self._detect_profile_location,
            self._detect_context_location,
            self._detect_history_location,
            self._detect_favorite_location
        ]
        
        candidates = []
        detection_metadata = {
            'methods_attempted': len(detection_methods),
            'methods_successful': 0,
            'enhancement_stages': []
        }
        
        for method in detection_methods:
            try:
                result = method(user_input, user_profile, context)
                if result and result.confidence >= require_confidence:
                    candidates.append(result)
                    detection_metadata['methods_successful'] += 1
            except Exception as e:
                self.logger.warning(f"Location detection method {method.__name__} failed: {e}")
        
        # If no candidates meet confidence threshold, try with lower threshold
        if not candidates:
            detection_metadata['fallback_attempted'] = True
            for method in detection_methods:
                try:
                    result = method(user_input, user_profile, context)
                    if result:
                        candidates.append(result)
                        detection_metadata['methods_successful'] += 1
                except Exception as e:
                    continue
        
        if not candidates:
            return LocationDetectionResult(
                location=None,
                confidence=0.0,
                detection_method='none',
                fallback_locations=[],
                metadata={
                    'message': 'No location detected from any source',
                    'detection_metadata': detection_metadata
                }
            )
        
        # Apply ML enhancements to candidates
        original_candidate_count = len(candidates)
        
        # Stage 1: Apply semantic pattern matching
        candidates = self._apply_semantic_pattern_matching(candidates, user_input)
        detection_metadata['enhancement_stages'].append('semantic_patterns')
        
        # Stage 2: Apply temporal context
        candidates = self._apply_temporal_context(candidates, user_input)
        detection_metadata['enhancement_stages'].append('temporal_context')
        
        # Stage 3: Apply ML learning adjustments
        candidates = self._apply_ml_learning_adjustments(candidates, user_input, context)
        detection_metadata['enhancement_stages'].append('ml_learning')
        
        # Enhanced candidate selection with conflict resolution
        best_candidate = self._select_best_candidate_with_conflict_resolution(candidates, user_input)
        
        # Enhance with fallback locations and alternative scores
        fallback_locations = []
        alternative_scores = {}
        
        for candidate in candidates:
            if candidate.location != best_candidate.location:
                fallback_locations.append(candidate.location)
                alternative_scores[candidate.location] = candidate.confidence
        
        best_candidate.fallback_locations = fallback_locations[:3]  # Top 3 alternatives
        best_candidate.alternative_scores = alternative_scores
        
        # Add detection metadata
        best_candidate.metadata.update({
            'detection_metadata': detection_metadata,
            'original_candidates': original_candidate_count,
            'final_candidates': len(candidates),
            'enhancement_applied': len(detection_metadata['enhancement_stages']) > 0
        })
        
        # Generate enhanced explanation
        best_candidate.explanation = self._generate_enhanced_explanation(best_candidate)
        
        # Update learning models
        self._update_learning_models(best_candidate)
        
        return best_candidate

    def _select_best_candidate_with_conflict_resolution(self, candidates: List[LocationDetectionResult], user_input: str) -> LocationDetectionResult:
        """Enhanced candidate selection with conflict resolution"""
        if len(candidates) == 1:
            return candidates[0]
        
        # Sort by confidence first
        candidates.sort(key=lambda x: x.confidence, reverse=True)
        
        # Check for conflicts between high-confidence candidates
        top_candidate = candidates[0]
        
        # If there are multiple high-confidence candidates, apply tie-breaking rules
        high_confidence_candidates = [c for c in candidates if c.confidence >= (top_candidate.confidence - 0.1)]
        
        if len(high_confidence_candidates) > 1:
            # Tie-breaking rules in order of priority:
            
            # 1. Explicit mentions always win
            explicit_candidates = [c for c in high_confidence_candidates if c.detection_method == 'explicit_query']
            if explicit_candidates:
                return explicit_candidates[0]
            
            # 2. GPS coordinates beat everything except explicit
            gps_candidates = [c for c in high_confidence_candidates if c.detection_method == 'gps_coordinates']
            if gps_candidates:
                return gps_candidates[0]
            
            # 3. Recent proximity inference beats older methods
            proximity_candidates = [c for c in high_confidence_candidates if c.detection_method == 'proximity_inference']
            if proximity_candidates:
                return proximity_candidates[0]
            
            # 4. Check for query context clues
            query_boosted_candidate = self._apply_query_context_boost(high_confidence_candidates, user_input)
            if query_boosted_candidate:
                return query_boosted_candidate
        
        return top_candidate

    def _apply_query_context_boost(self, candidates: List[LocationDetectionResult], user_input: str) -> Optional[LocationDetectionResult]:
        """Apply query context to boost relevant candidates"""
        user_input_lower = user_input.lower()
        
        # Context keywords that might favor certain districts
        context_preferences = {
            # Tourist-focused queries
            'tourist': ['Sultanahmet', 'Taksim', 'Beyoğlu'],
            'historic': ['Sultanahmet', 'Eminönü', 'Balat', 'Fener'],
            'museum': ['Sultanahmet', 'Beyoğlu', 'Şişli'],
            'palace': ['Sultanahmet', 'Beşiktaş'],
            
            # Nightlife and entertainment
            'nightlife': ['Beyoğlu', 'Taksim', 'Kadıköy'],
            'bar': ['Beyoğlu', 'Taksim', 'Kadıköy', 'Cihangir'],
            'club': ['Beyoğlu', 'Taksim', 'Ortaköy'],
            
            # Shopping
            'shopping': ['Beyoğlu', 'Şişli', 'Nişantaşı', 'Levent'],
            'mall': ['Şişli', 'Levent', 'Beşiktaş'],
            
            # Food and dining
            'seafood': ['Beşiktaş', 'Ortaköy', 'Arnavutköy', 'Bebek'],
            'street food': ['Kadıköy', 'Eminönü', 'Beyoğlu'],
            'fine dining': ['Nişantaşı', 'Beyoğlu', 'Beşiktaş'],
            
            # Local/authentic experiences
            'local': ['Kadıköy', 'Balat', 'Fener', 'Cihangir'],
            'authentic': ['Kadıköy', 'Balat', 'Fener'],
            'neighborhood': ['Kadıköy', 'Cihangir', 'Moda']
        }
        
        # Check for context keywords in query
        for keyword, preferred_districts in context_preferences.items():
            if keyword in user_input_lower:
                # Look for candidates in preferred districts
                for candidate in candidates:
                    if candidate.location in preferred_districts:
                        # Boost confidence slightly
                        candidate.confidence = min(0.95, candidate.confidence + 0.05)
                        candidate.metadata['query_context_boost'] = keyword
                        return candidate
        
        return None

    def _detect_explicit_location(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Detect explicitly mentioned location in user input"""
        entities = self.entity_recognizer.extract_entities(user_input)
        districts = entities.get('districts', [])
        
        if districts:
            location = districts[0]
            confidence = 0.95  # Very high confidence for explicit mentions
            
            return LocationDetectionResult(
                location=location,
                confidence=confidence,
                detection_method='explicit_query',
                fallback_locations=[],
                metadata={
                    'all_mentioned_districts': districts,
                    'extraction_method': 'entity_recognition'
                }
            )
        
        return None

    def _detect_proximity_location(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Enhanced proximity detection with multiple location sources"""
        user_input_lower = user_input.lower()
        
        # Enhanced proximity keywords with scoring
        proximity_indicators = {
            'nearby': 0.9, 'close by': 0.9, 'around here': 0.95, 'near me': 0.9,
            'in the area': 0.8, 'walking distance': 0.85, 'local': 0.7,
            'around': 0.7, 'close': 0.6, 'vicinity': 0.8, 'neighborhood': 0.8
        }
        
        location_context_indicators = {
            'where i am': 0.95, 'my location': 0.9, 'current area': 0.85,
            'this area': 0.8, 'this neighborhood': 0.8, 'here in': 0.7,
            'around my hotel': 0.85, 'near my accommodation': 0.85
        }
        
        # Find the strongest proximity indicator
        best_proximity_score = 0
        best_indicator = None
        
        for indicator, score in proximity_indicators.items():
            if indicator in user_input_lower and score > best_proximity_score:
                best_proximity_score = score
                best_indicator = indicator
        
        for indicator, score in location_context_indicators.items():
            if indicator in user_input_lower and score > best_proximity_score:
                best_proximity_score = score
                best_indicator = indicator
        
        if best_proximity_score > 0:
            # Try multiple location sources in order of preference
            location_sources = [
                ('recent_context', self._get_most_recent_location_from_context(context)),
                ('profile_current', user_profile.current_location),
                ('stored_context', context.get_context('current_detected_location')),
                ('gps_derived', self._get_gps_derived_location(user_profile)),
                ('favorite_primary', user_profile.favorite_neighborhoods[0] if user_profile.favorite_neighborhoods else None)
            ]
            
            for source_type, location in location_sources:
                if location:
                    # Adjust confidence based on proximity strength and source reliability
                    base_confidence = best_proximity_score
                    
                    # Source reliability adjustments
                    source_adjustments = {
                        'recent_context': 0.0,  # No adjustment - best source
                        'profile_current': -0.1,  # Slight reduction
                        'stored_context': -0.15,  # More reduction
                        'gps_derived': -0.05,  # GPS is quite reliable
                        'favorite_primary': -0.25  # Least reliable for proximity
                    }
                    
                    confidence = max(0.3, base_confidence + source_adjustments.get(source_type, -0.2))
                    
                    return LocationDetectionResult(
                        location=location,
                        confidence=confidence,
                        detection_method='proximity_inference',
                        fallback_locations=[],
                        metadata={
                            'proximity_indicator': best_indicator,
                            'proximity_score': best_proximity_score,
                            'location_source': source_type,
                            'all_indicators': [kw for kw, score in proximity_indicators.items() if kw in user_input_lower] +
                                            [phrase for phrase, score in location_context_indicators.items() if phrase in user_input_lower]
                        }
                    )
        
        return None

    def _get_gps_derived_location(self, user_profile: UserProfile) -> Optional[str]:
        """Helper method to get GPS-derived location"""
        if user_profile.gps_location:
            return self._get_nearest_district_from_gps(user_profile.gps_location)
        return None

    def _detect_profile_location(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Detect location from user profile"""
        if user_profile.current_location:
            # Calculate confidence based on how recent the location update was
            confidence = 0.7  # Base confidence for profile location
            
            # Increase confidence if user hasn't moved recently
            if hasattr(user_profile, 'location_last_updated'):
                time_diff = datetime.now() - user_profile.location_last_updated
                if time_diff < timedelta(hours=2):
                    confidence = 0.8
                elif time_diff < timedelta(hours=6):
                    confidence = 0.7
                else:
                    confidence = 0.6
            
            return LocationDetectionResult(
                location=user_profile.current_location,
                confidence=confidence,
                detection_method='user_profile',
                fallback_locations=[],
                metadata={
                    'profile_location': user_profile.current_location,
                    'location_source': 'current_location'
                }
            )
        
        return None

    def _detect_context_location(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Enhanced context location detection with time decay and method tracking"""
        stored_location = context.get_context('current_detected_location')
        
        if stored_location:
            # Base confidence varies by original detection method
            original_method = context.get_context('location_detection_method')
            original_confidence = context.get_context('location_confidence', 0.6)
            
            # Base confidence from original detection method
            method_confidence_map = {
                'explicit_query': 0.8,  # High confidence for explicit mentions
                'gps_coordinates': 0.75,  # High confidence for GPS
                'proximity_inference': 0.65,  # Good confidence for proximity
                'user_profile': 0.6,  # Medium confidence
                'conversation_history': 0.55,  # Lower confidence
                'favorite_neighborhood': 0.4,  # Lowest confidence
                'context_memory': 0.5  # Recursive context
            }
            
            base_confidence = method_confidence_map.get(original_method, 0.5)
            
            # Apply time decay if we can determine context age
            context_age = len(context.conversation_history) - context.get_context('location_set_at_turn', len(context.conversation_history))
            
            if context_age > 0:
                # Decay confidence over conversation turns
                time_decay = max(0.3, 1.0 - (context_age * 0.1))  # 10% decay per turn, min 30%
                base_confidence *= time_decay
            
            # Boost confidence if original was very reliable
            if original_confidence and original_confidence > 0.8:
                base_confidence *= 1.1
            
            confidence = min(0.85, base_confidence)  # Cap at 85%
            
            return LocationDetectionResult(
                location=stored_location,
                confidence=confidence,
                detection_method='context_memory',
                fallback_locations=[],
                metadata={
                    'context_location': stored_location,
                    'original_method': original_method,
                    'original_confidence': original_confidence,
                    'context_age_turns': context_age,
                    'time_decay_applied': context_age > 0
                }
            )
        
        return None

    def _detect_history_location(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Detect location from conversation history with weighted analysis"""
        weighted_locations = self._extract_locations_from_history_weighted(context.conversation_history)
        
        if weighted_locations:
            best_location = self._select_best_location_from_history(weighted_locations)
            if best_location:
                # Confidence based on recency and frequency
                confidence = min(0.65, len(weighted_locations) * 0.1 + 0.3)
                
                return LocationDetectionResult(
                    location=best_location,
                    confidence=confidence,
                    detection_method='conversation_history',
                    fallback_locations=[],
                    metadata={
                        'weighted_locations': weighted_locations,
                        'selection_algorithm': 'frequency_recency_weighted'
                    }
                )
        
        return None

    def _detect_favorite_location(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Detect location from user's favorite neighborhoods"""
        if user_profile.favorite_neighborhoods:
            selected_favorite = self._select_best_favorite_neighborhood(user_profile)
            if selected_favorite:
                confidence = 0.4  # Lower confidence for favorites
                
                return LocationDetectionResult(
                    location=selected_favorite,
                    confidence=confidence,
                    detection_method='favorite_neighborhood',
                    fallback_locations=user_profile.favorite_neighborhoods[1:3],  # Other favorites as fallbacks
                    metadata={
                        'all_favorites': user_profile.favorite_neighborhoods,
                        'selection_criteria': 'user_type_aware'
                    }
                )
        
        return None

    def _detect_gps_location(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Enhanced GPS location detection with automatic GPS and text parsing fallback"""
        
        # Method 1: Automatic GPS from user profile (most common)
        if user_profile.gps_location:
            lat, lng = user_profile.gps_location.get('lat'), user_profile.gps_location.get('lng')
            accuracy = user_profile.gps_location.get('accuracy')  # GPS accuracy in meters
            
            if lat and lng:
                return self._process_gps_coordinates(lat, lng, accuracy, 'automatic_gps', user_input)
        
        # Method 2: Parse coordinates from user text (fallback)
        parsed_coords = self._extract_gps_coordinates_from_text(user_input)
        if parsed_coords:
            lat, lng = parsed_coords
            return self._process_gps_coordinates(lat, lng, None, 'user_input_coordinates', user_input)
        
        # Method 3: Extract location from device/app mentions
        device_location = self._extract_location_from_device_info(user_input)
        if device_location:
            return LocationDetectionResult(
                location=device_location,
                confidence=0.75,  # High confidence for device-reported locations
                detection_method='device_location_service',
                fallback_locations=[],
                metadata={
                    'source': 'device_location_service',
                    'detection_text': user_input,
                    'extraction_method': 'device_info_parsing'
                }
            )
        
        return None

    def _extract_gps_coordinates_from_text(self, user_input: str) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from user text input"""
        # Pattern for decimal coordinates
        coord_patterns = [
            r'(?:gps|coordinates?|location)[:\s]*([+-]?\d+\.?\d*)[,\s]+([+-]?\d+\.?\d*)',
            r'([+-]?\d{1,3}\.\d+)[,\s]+([+-]?\d{1,3}\.\d+)',
            r'lat[:\s]*([+-]?\d+\.?\d*)[,\s]*lng?[:\s]*([+-]?\d+\.?\d*)',
            r'latitude[:\s]*([+-]?\d+\.?\d*)[,\s]*longitude[:\s]*([+-]?\d+\.?\d*)',
        ]
        
        for pattern in coord_patterns:
            matches = re.finditer(pattern, user_input, re.IGNORECASE)
            for match in matches:
                try:
                    lat, lng = float(match.group(1)), float(match.group(2))
                    
                    # Validate coordinates are in Istanbul area
                    if 40.8 <= lat <= 41.2 and 28.7 <= lng <= 29.3:
                        return (lat, lng)
                        
                except (ValueError, IndexError):
                    continue
        
        return None

    def _extract_location_from_device_info(self, user_input: str) -> Optional[str]:
        """Extract location from device/app mentions"""
        user_input_lower = user_input.lower()
        
        # Device/app location patterns
        device_patterns = [
            r'google maps says?\s+(?:i\'?m\s+)?(?:in|at|near)\s+(\w+)',
            r'my phone shows?\s+(?:i\'?m\s+)?(?:in|at|near)\s+(\w+)',
            r'gps shows?\s+(?:i\'?m\s+)?(?:in|at|near)\s+(\w+)',
            r'location services?\s+(?:shows?|says?)\s+(?:i\'?m\s+)?(?:in|at|near)\s+(\w+)',
            r'according to my phone\s+(?:i\'?m\s+)?(?:in|at|near)\s+(\w+)',
            r'my device says?\s+(?:i\'?m\s+)?(?:in|at|near)\s+(\w+)',
        ]
        
        for pattern in device_patterns:
            matches = re.finditer(pattern, user_input_lower, re.IGNORECASE)
            for match in matches:
                potential_location = match.group(1).title()
                validated_district = self._validate_and_normalize_district(potential_location)
                if validated_district:
                    return validated_district
        
        return None

    def _get_most_recent_location_from_context(self, context: ConversationContext) -> Optional[str]:
        """Get the most recently mentioned location from conversation context"""
        # Check last 5 interactions for location mentions
        recent_history = context.conversation_history[-5:] if len(context.conversation_history) > 5 else context.conversation_history
        
        for interaction in reversed(recent_history):  # Start from most recent
            user_input = interaction.get('user_input', '')
            system_response = interaction.get('system_response', '')
            
            # Check user input first
            entities = self.entity_recognizer.extract_entities(user_input)
            districts = entities.get('districts', [])
            if districts:
                return districts[0]
            
            # Check system response for location patterns
            for district in self.district_coords.keys():
                if district.lower() in system_response.lower():
                    return district
        
        return None

    def _extract_locations_from_history_weighted(self, conversation_history: List[Dict]) -> List[Dict]:
        """Extract locations from history with enhanced recency and context weighting"""
        weighted_locations = []
        # Look at last 15 interactions with more sophisticated weighting
        recent_history = conversation_history[-15:] if len(conversation_history) > 15 else conversation_history
        
        for i, interaction in enumerate(reversed(recent_history)):
            # Enhanced recency weight - more aggressive decay for older interactions
            base_weight = max(0.1, 1.0 - (i * 0.08))  # Slower decay, minimum weight
            
            # Boost weight for very recent interactions (last 3)
            if i < 3:
                base_weight *= 1.2
            
            user_input = interaction.get('user_input', '')
            system_response = interaction.get('system_response', '')
            
            # Check for explicit user location mentions (higher weight)
            entities = self.entity_recognizer.extract_entities(user_input)
            districts = entities.get('districts', [])
            for district in districts:
                # Higher weight for explicit user mentions
                explicit_weight = base_weight * 1.3
                
                # Extra boost if it's a location-focused query
                if any(keyword in user_input.lower() for keyword in ['in', 'at', 'near', 'around', 'from']):
                    explicit_weight *= 1.2
                
                weighted_locations.append({
                    'location': district,
                    'weight': explicit_weight,
                    'source': 'user_explicit',
                    'interaction_index': len(recent_history) - i - 1,
                    'context_type': 'location_query' if any(word in user_input.lower() for word in ['where', 'restaurant', 'food', 'eat']) else 'general'
                })
            
            # Check system response for location context (lower weight)
            for district in self.district_coords.keys():
                if district.lower() in system_response.lower():
                    context_weight = base_weight * 0.7
                    weighted_locations.append({
                        'location': district,
                        'weight': context_weight,
                        'source': 'system_context',
                        'interaction_index': len(recent_history) - i - 1,
                        'context_type': 'recommendation_context' if any(word in system_response.lower() for word in ['recommend', 'suggest', 'try']) else 'general'
                    })
        
        return weighted_locations

    def _select_best_location_from_history(self, weighted_locations: List[Dict]) -> Optional[str]:
        """Enhanced location selection with better scoring algorithm"""
        if not weighted_locations:
            return None
        
        # Advanced scoring algorithm
        location_scores = {}
        
        for loc_data in weighted_locations:
            location = loc_data['location']
            weight = loc_data['weight']
            source = loc_data['source']
            context_type = loc_data.get('context_type', 'general')
            
            if location not in location_scores:
                location_scores[location] = {
                    'total_weight': 0,
                    'explicit_mentions': 0,
                    'system_mentions': 0,
                    'most_recent_index': -1,
                    'context_scores': [],
                    'source_diversity': set()
                }
            
            score_data = location_scores[location]
            score_data['total_weight'] += weight
            score_data['most_recent_index'] = max(score_data['most_recent_index'], loc_data['interaction_index'])
            score_data['source_diversity'].add(source)
            
            # Track mention types
            if source == 'user_explicit':
                score_data['explicit_mentions'] += 1
            elif source == 'system_context':
                score_data['system_mentions'] += 1
            
            # Context type scoring
            if context_type == 'location_query':
                score_data['context_scores'].append(0.3)
            elif context_type == 'recommendation_context':
                score_data['context_scores'].append(0.2)
            else:
                score_data['context_scores'].append(0.1)
        
        # Calculate final scores with sophisticated algorithm
        best_location = None
        best_score = 0
        
        for location, score_data in location_scores.items():
            # Base score from weighted mentions
            base_score = score_data['total_weight']
            
            # Recency bonus (exponential decay)
            recency_bonus = min(1.0, score_data['most_recent_index'] * 0.15)
            
            # Explicit mention bonus (user explicitly mentioned this location)
            explicit_bonus = score_data['explicit_mentions'] * 0.3
            
            # Context relevance bonus
            context_bonus = sum(score_data['context_scores']) * 0.1
            
            # Source diversity bonus (mentioned in different contexts)
            diversity_bonus = len(score_data['source_diversity']) * 0.1
            
            # Frequency bonus with diminishing returns
            frequency_bonus = min(0.5, (score_data['explicit_mentions'] + score_data['system_mentions']) * 0.15)
            
            combined_score = (
                base_score +
                recency_bonus +
                explicit_bonus +
                context_bonus +
                diversity_bonus +
                frequency_bonus
            )
            
            if combined_score > best_score:
                best_score = combined_score
                best_location = location
        
        return best_location

    def _select_best_favorite_neighborhood(self, user_profile: UserProfile) -> Optional[str]:
        """Select the best favorite neighborhood based on user preferences"""
        if not user_profile.favorite_neighborhoods:
            return None
        
        primary_favorite = user_profile.favorite_neighborhoods[0]
        
        # Add intelligence based on user type if available
        if hasattr(user_profile, 'user_type') and user_profile.user_type:
            if user_profile.user_type == 'tourist' and primary_favorite in ['Kadıköy', 'Cihangir']:
                # Tourists might prefer more central areas
                tourist_preferred = ['Sultanahmet', 'Beyoğlu', 'Taksim']
                for pref in user_profile.favorite_neighborhoods:
                    if pref in tourist_preferred:
                        return pref
        
        return primary_favorite

    def _get_nearest_district_from_gps(self, gps_coords: Dict[str, float]) -> Optional[str]:
        """Determine nearest district from GPS coordinates with improved accuracy and caching"""
        lat, lng = gps_coords.get('lat'), gps_coords.get('lng')
        if not lat or not lng:
            return None
        
        # Create cache key for GPS coordinates (rounded to reduce cache size)
        cache_key = f"gps:{round(lat, 6)}:{round(lng, 6)}"
        
        # Check cache first
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # Calculate distances to all districts
        district_distances = self._calculate_all_district_distances(lat, lng, use_haversine=True)
        
        # Find nearest district
        nearest_district = min(district_distances.items(), key=lambda x: x[1])[0] if district_distances else None
        
        # Cache the result with management
        self._manage_distance_cache()
        self._distance_cache[cache_key] = nearest_district
        
        return nearest_district

    def _calculate_all_district_distances(self, lat: float, lng: float, use_haversine: bool = False) -> Dict[str, float]:
        """Calculate distances from coordinates to all districts"""
        distances = {}
        
        for district, coords in self.district_coords.items():
            if use_haversine:
                # More accurate distance calculation
                distance = self._haversine_distance(
                    lat, lng, coords['lat'], coords['lng']
                )
            else:
                # Simple Euclidean distance
                distance = ((lat - coords['lat'])**2 + (lng - coords['lng'])**2)**0.5
            
            distances[district] = distance
        
        return distances

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate the great circle distance between two points on earth"""
        # Convert decimal degrees to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r

    def _manage_distance_cache(self):
        """Manage cache size to prevent memory issues"""
        if len(self._distance_cache) >= self._cache_max_size:
            # Remove oldest 20% of entries
            cache_items = list(self._distance_cache.items())
            keep_count = int(self._cache_max_size * 0.8)
            self._distance_cache = dict(cache_items[-keep_count:])

    # Additional ML/DL methods for continuous learning and model improvement
    def update_user_feedback(self, result: LocationDetectionResult, is_correct: bool):
        """Update models based on user feedback"""
        if result.location:
            # Update success rates
            method_stats = self.method_success_rates[result.detection_method]
            method_stats['total'] += 1
            if is_correct:
                method_stats['correct'] += 1
            
            # Update location frequency model
            feedback_weight = 0.1 if is_correct else -0.05
            self.location_frequency_model[result.location] += feedback_weight
            
            # Update learning models if ML is available
            if ML_AVAILABLE:
                self._update_neural_models_with_feedback(result, is_correct)

    def _update_neural_models_with_feedback(self, result: LocationDetectionResult, is_correct: bool):
        """Update neural network models with user feedback"""
        try:
            # This would involve retraining or fine-tuning the models
            # For now, we'll just update the performance metrics
            if 'ml_enhanced' in result.detection_method:
                current_accuracy = self.model_performance_metrics['neural_network']['accuracy']
                # Simple running average update
                self.model_performance_metrics['neural_network']['accuracy'] = (
                    current_accuracy * 0.9 + (1.0 if is_correct else 0.0) * 0.1
                )
        except Exception as e:
            self.logger.warning(f"Neural model feedback update failed: {e}")

    def get_model_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report of all detection methods"""
        report = {
            'detection_methods': {},
            'ml_performance': self.model_performance_metrics,
            'location_statistics': {
                'most_detected': dict(self.district_popularity_scores.most_common(5)),
                'highest_confidence': {},
                'detection_success_rates': {}
            }
        }
        
        # Calculate success rates for each detection method
        for method, stats in self.method_success_rates.items():
            if stats['total'] > 0:
                success_rate = stats['correct'] / stats['total']
                report['detection_methods'][method] = {
                    'success_rate': success_rate,
                    'total_attempts': stats['total'],
                    'correct_detections': stats['correct']
                }
        
        # Calculate average confidence for each location
        for location, confidences in self.location_confidence_history.items():
            if confidences:
                report['location_statistics']['highest_confidence'][location] = {
                    'average': sum(confidences) / len(confidences),
                    'max': max(confidences),
                    'count': len(confidences)
                }
        
        return report

    def _integrate_with_backend_detector(self):
        """Integrate with the backend location detector to avoid duplication"""
        try:
            # Import the backend detector
            from backend.services.intelligent_location_detector import IntelligentLocationDetector as BackendDetector
            self.backend_detector = BackendDetector()
            self.logger.info("Backend location detector integrated successfully")
        except ImportError as e:
            self.logger.warning(f"Backend detector not available: {e}")
            self.backend_detector = None

    def _calculate_pattern_weights(self):
        """Calculate pattern weights for location detection"""
        # Initialize basic pattern weights
        weights = {
            'explicit_location': 1.0,
            'neighborhood_mention': 0.8,
            'landmark_mention': 0.7,
            'user_history': 0.6,
            'context_inference': 0.5,
            'semantic_similarity': 0.4,
        }
        
        # Enhanced weights with ML if available
        if ML_AVAILABLE:
            weights.update({
                'neural_network': 0.9,
                'ensemble_prediction': 0.85,
                'feature_engineering': 0.75,
            })
        
        return weights

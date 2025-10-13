#!/usr/bin/env python3
"""
Intelligent Location Detection Service for Istanbul AI
Advanced location inference from multiple data sources with ML enhancements
"""

import logging
import math
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
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

    def _initialize_ml_components(self):
        """Initialize machine learning components for enhanced detection"""
        try:
            # Simple similarity computation for semantic matching
            self.semantic_similarity_cache = {}
            self.location_frequency_model = defaultdict(float)
            self.context_pattern_weights = defaultdict(float)
            
            # Initialize with some base patterns
            self._bootstrap_ml_patterns()
            
        except Exception as e:
            self.logger.warning(f"ML components initialization failed: {e}")

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

    def _update_learning_models(self, result: LocationDetectionResult, user_feedback: Optional[bool] = None):
        """Update learning models based on detection results and feedback"""
        if result.location:
            # Update frequency model
            self.location_frequency_model[result.location] += 0.01
            
            # Update confidence history
            self.location_confidence_history[result.location].append(result.confidence)
            
            # Keep only recent history
            if len(self.location_confidence_history[result.location]) > 50:
                self.location_confidence_history[result.location] = self.location_confidence_history[result.location][-50:]
            
            # Update popularity scores
            self.district_popularity_scores[result.location] += 1
            
            # Apply feedback if provided
            if user_feedback is not None:
                feedback_weight = 0.1 if user_feedback else -0.05
                self.location_frequency_model[result.location] += feedback_weight

    def _generate_enhanced_explanation(self, result: LocationDetectionResult) -> str:
        """Generate enhanced human-readable explanation"""
        if not result.location:
            return "No location could be determined from the available information."
        
        base_explanations = {
            'explicit_query': f"You explicitly mentioned '{result.location}' in your request",
            'gps_coordinates': f"Based on your GPS coordinates, you appear to be in {result.location}",
            'proximity_inference': f"You mentioned being nearby, and {result.location} was determined from context",
            'user_profile': f"Using your saved location preference for {result.location}",
            'conversation_history': f"Based on recent conversation, {result.location} seems to be your focus area",
            'context_memory': f"Continuing from our previous discussion about {result.location}",
            'favorite_neighborhood': f"Using {result.location} as one of your favorite neighborhoods"
        }
        
        explanation = base_explanations.get(result.detection_method, f"Detected {result.location} through advanced analysis")
        
        # Add confidence qualifier with more nuance
        if result.confidence >= 0.9:
            confidence_qualifier = "with very high confidence"
        elif result.confidence >= 0.8:
            confidence_qualifier = "with high confidence"
        elif result.confidence >= 0.6:
            confidence_qualifier = "with good confidence"
        elif result.confidence >= 0.4:
            confidence_qualifier = "with moderate confidence"
        else:
            confidence_qualifier = "with low confidence"
        
        full_explanation = f"{explanation} {confidence_qualifier} ({result.confidence:.1%})"
        
        # Add enhancement details
        enhancements = []
        if 'semantic_boosts' in result.metadata:
            semantic_count = len(result.metadata['semantic_boosts'])
            enhancements.append(f"{semantic_count} semantic pattern{'s' if semantic_count > 1 else ''}")
        
        if 'temporal_boosts' in result.metadata:
            temporal_count = len(result.metadata['temporal_boosts'])
            enhancements.append(f"{temporal_count} temporal factor{'s' if temporal_count > 1 else ''}")
        
        if 'ml_adjustments' in result.metadata:
            enhancements.append("ML learning adjustments")
        
        if enhancements:
            full_explanation += f" (Enhanced by: {', '.join(enhancements)})"
        
        # Add alternatives if available
        if result.fallback_locations:
            full_explanation += f". Alternatives: {', '.join(result.fallback_locations[:2])}"
        
        return full_explanation

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
        
        # Collect all possible location detections (ordered by priority)
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

    def _detect_natural_language_location(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> Optional[LocationDetectionResult]:
        """Extract location from natural language expressions like 'I am in Beyoğlu'"""
        user_input_lower = user_input.lower()
        
        # Enhanced natural language location patterns
        location_patterns = [
            # Direct location statements
            r'\bi\s+am\s+in\s+(\w+)',  # "i am in beyoglu"
            r'\bi\'m\s+in\s+(\w+)',    # "i'm in beyoglu"
            r'\bcurrently\s+in\s+(\w+)',  # "currently in beyoglu"
            r'\bstaying\s+in\s+(\w+)',    # "staying in beyoglu"
            r'\blocated\s+in\s+(\w+)',    # "located in beyoglu"
            r'\bvisiting\s+(\w+)',        # "visiting beyoglu"
            r'\bat\s+(\w+)',              # "at beyoglu"
            r'\bhere\s+in\s+(\w+)',       # "here in beyoglu"
            
            # Movement/arrival patterns
            r'\bjust\s+arrived\s+in\s+(\w+)',    # "just arrived in beyoglu"
            r'\bgot\s+to\s+(\w+)',               # "got to beyoglu"
            r'\breached\s+(\w+)',                # "reached beyoglu"
            r'\bmade\s+it\s+to\s+(\w+)',         # "made it to beyoglu"
            r'\bnow\s+in\s+(\w+)',               # "now in beyoglu"
            
            # Position/direction patterns
            r'\bstanding\s+in\s+(\w+)',          # "standing in beyoglu"
            r'\bwalking\s+through\s+(\w+)',      # "walking through beyoglu"
            r'\bexploring\s+(\w+)',              # "exploring beyoglu"
            r'\bwandering\s+around\s+(\w+)',     # "wandering around beyoglu"
            
            # Accommodation patterns
            r'\bhotel\s+is\s+in\s+(\w+)',        # "hotel is in beyoglu"
            r'\bstaying\s+at.*?in\s+(\w+)',      # "staying at hotel in beyoglu"
            r'\baccommodation\s+in\s+(\w+)',     # "accommodation in beyoglu"
            
            # Contextual patterns with area/neighborhood
            r'\bin\s+the\s+(\w+)\s+area',        # "in the beyoglu area"
            r'\bin\s+(\w+)\s+district',          # "in beyoglu district"
            r'\bin\s+(\w+)\s+neighborhood',      # "in beyoglu neighborhood"
            r'\baround\s+(\w+)\s+area',          # "around beyoglu area"
        ]
        
        detected_locations = []
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, user_input_lower, re.IGNORECASE)
            for match in matches:
                potential_location = match.group(1).title()
                
                # Validate against known districts with fuzzy matching
                matched_district = self._validate_and_normalize_district(potential_location)
                if matched_district:
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_natural_language_confidence(pattern, user_input_lower, matched_district)
                    
                    detected_locations.append({
                        'location': matched_district,
                        'confidence': confidence,
                        'pattern': pattern,
                        'original_text': match.group(0),
                        'extracted_name': potential_location
                    })
        
        if detected_locations:
            # Select best detection based on confidence and pattern specificity
            best_detection = max(detected_locations, key=lambda x: x['confidence'])
            
            return LocationDetectionResult(
                location=best_detection['location'],
                confidence=best_detection['confidence'],
                detection_method='natural_language',
                fallback_locations=[d['location'] for d in detected_locations if d['location'] != best_detection['location']][:3],
                metadata={
                    'pattern_matched': best_detection['pattern'],
                    'original_text': best_detection['original_text'],
                    'extracted_name': best_detection['extracted_name'],
                    'all_detections': len(detected_locations),
                    'extraction_method': 'natural_language_parsing'
                }
            )
        
        return None

    def _validate_and_normalize_district(self, potential_location: str) -> Optional[str]:
        """Validate and normalize a potential district name with fuzzy matching"""
        if not potential_location:
            return None
        
        potential_lower = potential_location.lower()
        
        # Direct exact match (case insensitive)
        for district in self.district_coords.keys():
            if district.lower() == potential_lower:
                return district
        
        # Fuzzy matching for common variations and typos
        district_variations = {
            # Common English variations
            'sultanahmet': 'Sultanahmet',
            'beyoglu': 'Beyoğlu',
            'beyolu': 'Beyoğlu',
            'taksim': 'Taksim',
            'kadikoy': 'Kadıköy',
            'kadiköy': 'Kadıköy',
            'besiktas': 'Beşiktaş',
            'besiktaş': 'Beşiktaş',
            'galata': 'Galata',
            'karakoy': 'Karaköy',
            'karaköy': 'Karaköy',
            'levent': 'Levent',
            'sisli': 'Şişli',
            'şişli': 'Şişli',
            'nisantasi': 'Nişantaşı',
            'nişantaşı': 'Nişantaşı',
            'ortakoy': 'Ortaköy',
            'ortaköy': 'Ortaköy',
            'uskudar': 'Üsküdar',
            'üsküdar': 'Üsküdar',
            'eminonu': 'Eminönü',
            'eminönü': 'Eminönü',
            'cihangir': 'Cihangir',
            'arnavutkoy': 'Arnavutköy',
            'arnavutköy': 'Arnavutköy',
            'bebek': 'Bebek',
            'bostanci': 'Bostancı',
            'bostancı': 'Bostancı',
            'fenerbahce': 'Fenerbahçe',
            'fenerbahçe': 'Fenerbahçe',
            'moda': 'Moda',
            'balat': 'Balat',
            'fener': 'Fener'
        }
        
        # Check variations map
        if potential_lower in district_variations:
            return district_variations[potential_lower]
        
        # Partial matching for longer names
        for district in self.district_coords.keys():
            district_lower = district.lower()
            # Check if potential location is a substring of a known district
            if len(potential_lower) >= 4 and potential_lower in district_lower:
                return district
            # Check if known district is a substring of potential location
            if len(district_lower) >= 4 and district_lower in potential_lower:
                return district
        
        # Fuzzy matching with edit distance for typos
        for district in self.district_coords.keys():
            if self._calculate_string_similarity(potential_lower, district.lower()) > 0.8:
                return district
        
        return None

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using a simple edit distance approach"""
        if not str1 or not str2:
            return 0.0
        
        # Simple approach: check character overlap and length similarity
        if abs(len(str1) - len(str2)) > 3:
            return 0.0
        
        # Count matching characters in order
        matches = 0
        max_len = max(len(str1), len(str2))
        min_len = min(len(str1), len(str2))
        
        for i in range(min_len):
            if i < len(str1) and i < len(str2) and str1[i] == str2[i]:
                matches += 1
        
        # Calculate similarity ratio
        similarity = matches / max_len if max_len > 0 else 0.0
        
        # Boost for very similar length
        if abs(len(str1) - len(str2)) <= 1:
            similarity += 0.1
        
        return min(1.0, similarity)

    def _calculate_natural_language_confidence(self, pattern: str, user_input: str, district: str) -> float:
        """Calculate confidence score for natural language location detection"""
        base_confidence = 0.8  # High confidence for explicit natural language statements
        
        # Pattern-specific confidence adjustments
        high_confidence_patterns = [
            r'\bi\s+am\s+in\s+(\w+)',
            r'\bi\'m\s+in\s+(\w+)',
            r'\bcurrently\s+in\s+(\w+)',
            r'\bstaying\s+in\s+(\w+)',
            r'\blocated\s+in\s+(\w+)'
        ]
        
        medium_confidence_patterns = [
            r'\bvisiting\s+(\w+)',
            r'\bat\s+(\w+)',
            r'\bhere\s+in\s+(\w+)',
            r'\bjust\s+arrived\s+in\s+(\w+)'
        ]
        
        if any(re.match(p, pattern) for p in high_confidence_patterns):
            base_confidence = 0.9
        elif any(re.match(p, pattern) for p in medium_confidence_patterns):
            base_confidence = 0.75
        else:
            base_confidence = 0.65
        
        # Boost confidence for clear, unambiguous statements
        if len(user_input.split()) <= 6:  # Short, clear statements
            base_confidence += 0.05
        
        # Boost for present tense indicators
        present_indicators = ['am', 'currently', 'now', 'here']
        if any(indicator in user_input.lower() for indicator in present_indicators):
            base_confidence += 0.05
        
        # Slight reduction for longer, more complex sentences
        if len(user_input.split()) > 15:
            base_confidence -= 0.1
        
        # District popularity adjustment (slight boost for well-known areas)
        popular_districts = ['Sultanahmet', 'Beyoğlu', 'Taksim', 'Kadıköy', 'Beşiktaş']
        if district in popular_districts:
            base_confidence += 0.02
        
        return min(0.95, max(0.4, base_confidence))
    
    def _process_gps_coordinates(self, lat: float, lng: float, accuracy: Optional[float], source_method: str, user_input: str) -> Optional[LocationDetectionResult]:
        """Process GPS coordinates and return location detection result"""
        if not lat or not lng:
            return None
            
            # Find nearest districts with distance analysis
            district_distances = []
            for district, coords in self.district_coords.items():
                distance = ((lat - coords['lat'])**2 + (lng - coords['lng'])**2)**0.5
                distance_km = distance * 111.32  # Convert to km
                within_radius = distance <= coords.get('radius', 0.01)
                
                district_distances.append({
                    'district': district,
                    'distance': distance,
                    'distance_km': distance_km,
                    'within_radius': within_radius,
                    'coords': coords
                })
            
            # Sort by distance
            district_distances.sort(key=lambda x: x['distance'])
            closest = district_distances[0]
            second_closest = district_distances[1] if len(district_distances) > 1 else None
            
            # Calculate sophisticated confidence
            base_confidence = 0.85
            
            # GPS accuracy adjustments
            if accuracy:
                if accuracy > 200:  # Very poor accuracy
                    base_confidence *= 0.5
                elif accuracy > 100:  # Poor accuracy
                    base_confidence *= 0.6
                elif accuracy > 50:  # Moderate accuracy
                    base_confidence *= 0.8
                # else: good accuracy, no reduction
            
            # Distance-based confidence
            if not closest['within_radius']:
                # Outside district boundary - reduce confidence based on distance
                distance_penalty = min(0.4, closest['distance_km'] * 0.1)
                base_confidence *= (1.0 - distance_penalty)
            else:
                # Within boundary - boost confidence if very close to center
                if closest['distance'] < closest['coords'].get('radius', 0.01) * 0.3:
                    base_confidence = min(0.95, base_confidence * 1.1)
            
            # Confidence reduction if another district is very close
            if second_closest and closest['distance'] > 0 and second_closest['distance'] / closest['distance'] < 1.3:
                base_confidence *= 0.85
            
            # Find alternative districts within reasonable distance
            alternatives = [d['district'] for d in district_distances[1:4] if d['distance_km'] < 5.0]
            
            return LocationDetectionResult(
                location=closest['district'],
                confidence=max(0.3, base_confidence),
                detection_method='gps_coordinates',
                fallback_locations=alternatives,
                metadata={
                    'gps_coords': user_profile.gps_location,
                    'gps_accuracy_meters': accuracy,
                    'distance_from_center_km': closest['distance_km'],
                    'within_district_boundary': closest['within_radius'],
                    'nearby_districts': [d['district'] for d in district_distances[1:3]],
                    'confidence_factors': {
                        'gps_accuracy_good': not accuracy or accuracy <= 50,
                        'within_boundary': closest['within_radius'],
                        'clear_closest': not second_closest or second_closest['distance'] / closest['distance'] >= 1.3
                    }
                }
            )
        
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
        
        # Calculate time-based recency if timestamps available
        current_time = datetime.now()
        
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
                    # Context weight for system mentions
                    context_weight = base_weight * 0.6
                    
                    # Check if system was providing location-specific recommendations
                    if any(phrase in system_response.lower() for phrase in ['in ' + district.lower(), 'near ' + district.lower(), district.lower() + ' area']):
                        context_weight *= 1.1
                    
                    weighted_locations.append({
                        'location': district,
                        'weight': context_weight,
                        'source': 'system_context',
                        'interaction_index': len(recent_history) - i - 1,
                        'context_type': 'recommendation_context'
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
                score_data['context_scores'].append(1.5)
            elif context_type == 'recommendation_context':
                score_data['context_scores'].append(1.0)
            else:
                score_data['context_scores'].append(0.8)
        
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
            
            self.logger.debug(f"Location '{location}' score: {combined_score:.3f} (base:{base_score:.2f}, recency:{recency_bonus:.2f}, explicit:{explicit_bonus:.2f})");
            
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
            from ..utils.constants import UserType
            
            if user_profile.user_type == UserType.FIRST_TIME_VISITOR:
                # First-time visitors might prefer tourist-friendly areas
                tourist_friendly = ['Sultanahmet', 'Taksim', 'Beyoğlu', 'Galata']
                for neighborhood in user_profile.favorite_neighborhoods:
                    if neighborhood in tourist_friendly:
                        return neighborhood
            elif user_profile.user_type == UserType.LOCAL_RESIDENT:
                # Locals might prefer authentic neighborhoods
                authentic_areas = ['Kadıköy', 'Beşiktaş', 'Balat', 'Fener', 'Cihangir']
                for neighborhood in user_profile.favorite_neighborhoods:
                    if neighborhood in authentic_areas:
                        return neighborhood
        
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
            cached_result = self._distance_cache[cache_key]
            self.logger.debug(f"Using cached GPS result for {cache_key}: {cached_result}")
            return cached_result
        
        # Calculate distances to all districts
        district_distances = self._calculate_all_district_distances(lat, lng, use_haversine=True)
        
        # Find nearest district
        nearest_district = min(district_distances.items(), key=lambda x: x[1])[0] if district_distances else None
        
        # Cache the result with management
        self._manage_distance_cache()
        self._distance_cache[cache_key] = nearest_district
        
        return nearest_district

    def get_location_suggestions(self, user_input: str, user_profile: UserProfile, context: ConversationContext) -> List[str]:
        """Get location suggestions when no definitive location is detected"""
        suggestions = []
        
        # Try to detect with lower confidence threshold
        result = self.detect_location(user_input, user_profile, context, require_confidence=0.1)
        
        if result.location:
            suggestions.append(result.location)
        
        suggestions.extend(result.fallback_locations)
        
        # Add popular areas if no suggestions
        if not suggestions:
            popular_areas = ['Sultanahmet', 'Taksim', 'Beyoğlu', 'Kadıköy', 'Beşiktaş']
            suggestions.extend(popular_areas[:3])
        
        return list(dict.fromkeys(suggestions))  # Remove duplicates while preserving order

    def update_location_context(self, context: ConversationContext, location: str, method: str, confidence: float):
        """Update location context with enhanced tracking"""
        context.set_context('current_detected_location', location)
        context.set_context('location_detection_method', method)
        context.set_context('location_confidence', confidence)
        context.set_context('location_set_at_turn', len(context.conversation_history))
        context.set_context('location_updated_at', datetime.now().isoformat())
        
        # Track location history for pattern analysis
        location_history = context.get_context('location_history', [])
        location_history.append({
            'location': location,
            'method': method,
            'confidence': confidence,
            'turn': len(context.conversation_history),
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 location updates
        if len(location_history) > 10:
            location_history = location_history[-10:]
        
        context.set_context('location_history', location_history)

    def get_location_confidence_explanation(self, result: LocationDetectionResult) -> str:
        """Generate human-readable explanation of location detection confidence"""
        if not result.location:
            return "No location could be determined from the available information."
        
        method_explanations = {
            'explicit_query': f"You explicitly mentioned '{result.location}' in your request",
            'gps_coordinates': f"Based on your GPS coordinates, you appear to be in {result.location}",
            'proximity_inference': f"You mentioned being 'nearby' or 'around here', and {result.location} was your most recent location",
            'user_profile': f"Using your saved location preference for {result.location}",
            'conversation_history': f"Based on recent conversation, {result.location} seems to be your area of interest",
            'context_memory': f"Continuing from our previous discussion about {result.location}",
            'favorite_neighborhood': f"Using {result.location} as one of your favorite neighborhoods"
        }
        
        explanation = method_explanations.get(result.detection_method, f"Detected {result.location} through context analysis")
        
        # Add confidence qualifier
        if result.confidence >= 0.8:
            confidence_qualifier = "with high confidence"
        elif result.confidence >= 0.6:
            confidence_qualifier = "with good confidence"
        elif result.confidence >= 0.4:
            confidence_qualifier = "with moderate confidence"
        else:
            confidence_qualifier = "with low confidence"
        
        full_explanation = f"{explanation} {confidence_qualifier} ({result.confidence:.1%})"
        
        # Add fallback information if available
        if result.fallback_locations:
            full_explanation += f". Other possible locations: {', '.join(result.fallback_locations[:2])}"
        
        return full_explanation
    
    def _calculate_all_district_distances(self, lat: float, lng: float, use_haversine: bool = True) -> Dict[str, float]:
        """Calculate distances from GPS coordinates to all known Istanbul districts"""
        distances = {}
        
        # Istanbul district coordinates (approximate centers)
        district_coordinates = {
            'Sultanahmet': (41.0055, 28.9769),
            'Beyoğlu': (41.0396, 28.9784),
            'Taksim': (41.0369, 28.9840),
            'Galata': (41.0257, 28.9740),
            'Karaköy': (41.0257, 28.9740),
            'Kadıköy': (40.9804, 29.0295),
            'Üsküdar': (41.0214, 29.0128),
            'Beşiktaş': (41.0429, 29.0094),
            'Ortaköy': (41.0555, 29.0263),
            'Bebek': (41.0837, 29.0430),
            'Emirgan': (41.1087, 29.0531),
            'Sarıyer': (41.1732, 29.0532),
            'Etiler': (41.0774, 29.0247),
            'Levent': (41.0843, 28.9953),
            'Maslak': (41.1121, 29.0155),
            'Şişli': (41.0602, 28.9816),
            'Nişantaşı': (41.0478, 28.9905),
            'Mecidiyeköy': (41.0733, 28.9849),
            'Bosphorus': (41.0839, 29.0436),
            'Asian Side': (40.9804, 29.0295),
            'European Side': (41.0055, 28.9769),
            'Golden Horn': (41.0257, 28.9740),
            'Fatih': (41.0186, 28.9350),
            'Eminönü': (41.0176, 28.9706),
            'Sirkeci': (41.0137, 28.9784),
            'Laleli': (41.0097, 28.9560),
            'Aksaray': (41.0104, 28.9475),
            'Beyazıt': (41.0107, 28.9640),
            'Fener': (41.0297, 28.9487),
            'Balat': (41.0297, 28.9487),
            'Eyüp': (41.0467, 28.9344),
            'Kağıthane': (41.0847, 28.9711),
            'Bakırköy': (40.9669, 28.8735),
            'Ataköy': (40.9669, 28.8735),
            'Yeşilköy': (40.9669, 28.8135),
            'Florya': (40.9669, 28.7935),
            'Zeytinburnu': (41.0047, 28.9089),
            'Güngören': (41.0156, 28.8751),
            'Merter': (41.0278, 28.8878),
            'Topkapı': (41.0192, 28.9289),
            'Edirnekapı': (41.0423, 28.9289),
            'Avcılar': (41.0267, 28.7210),
            'Küçükçekmece': (41.0267, 28.7610)
        }
        
        for district, (d_lat, d_lng) in district_coordinates.items():
            if use_haversine:
                distance = self._haversine_distance(lat, lng, d_lat, d_lng)
            else:
                # Simple Euclidean distance for faster calculation
                distance = ((lat - d_lat) ** 2 + (lng - d_lng) ** 2) ** 0.5
            
            distances[district] = distance
        
        return distances

    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate the great circle distance between two points on earth (in kilometers)"""
        import math
        
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
        """Manage distance cache size and cleanup old entries"""
        max_cache_size = 1000  # Maximum number of cached distance calculations
        
        if len(self._distance_cache) >= max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self._distance_cache) // 5
            oldest_keys = list(self._distance_cache.keys())[:items_to_remove]
            
            for key in oldest_keys:
                del self._distance_cache[key]
            
            self.logger.debug(f"Distance cache cleaned up: removed {items_to_remove} entries")

    def _manage_semantic_cache(self):
        """Manage semantic similarity cache size and cleanup old entries"""
        max_cache_size = 500  # Maximum number of cached semantic calculations
        
        if len(self._semantic_cache) >= max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self._semantic_cache) // 5
            oldest_keys = list(self._semantic_cache.keys())[:items_to_remove]
            
            for key in oldest_keys:
                del self._semantic_cache[key]
            
            self.logger.debug(f"Semantic cache cleaned up: removed {items_to_remove} entries")

    def _manage_pattern_cache(self):
        """Manage pattern matching cache size and cleanup old entries"""
        max_cache_size = 300  # Maximum number of cached pattern calculations
        
        if len(self._pattern_cache) >= max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self._pattern_cache) // 5
            oldest_keys = list(self._pattern_cache.keys())[:items_to_remove]
            
            for key in oldest_keys:
                del self._pattern_cache[key]
            
            self.logger.debug(f"Pattern cache cleaned up: removed {items_to_remove} entries")

    def clear_all_caches(self):
        """Clear all performance caches"""
        self._distance_cache.clear()
        self._semantic_cache.clear()
        self._pattern_cache.clear()
        self.logger.info("All location detection caches cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about cache usage"""
        return {
            'distance_cache_size': len(self._distance_cache),
            'semantic_cache_size': len(self._semantic_cache),
            'pattern_cache_size': len(self._pattern_cache),
            'total_cached_items': len(self._distance_cache) + len(self._semantic_cache) + len(self._pattern_cache)
        }

"""
signals.py - Signal Detection System

Multi-intent signal detection with semantic matching and language-aware thresholds.

Supported Signals:
- needs_restaurant: Restaurant recommendations
- needs_attraction: Attractions and museums
- needs_transportation: Directions and transit
- needs_neighborhood: Neighborhood information
- needs_events: Events and activities
- needs_weather: Weather-aware recommendations
- needs_hidden_gems: Off-the-beaten-path locations
- needs_map: Visual map generation
- needs_gps_routing: GPS-based routing
- needs_translation: Translation requests

Author: AI Istanbul Team
Date: November 2025
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SignalDetector:
    """
    Multi-intent signal detection system with semantic matching.
    
    Features:
    - Keyword-based detection (fast)
    - Semantic similarity detection (accurate)
    - Language-aware thresholds
    - A/B testing integration
    - Confidence scoring
    """
    
    def __init__(
        self,
        embedding_model=None,
        language_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize signal detector.
        
        Args:
            embedding_model: Embedding model for semantic matching
            language_thresholds: Language-specific detection thresholds
        """
        self.embedding_model = embedding_model
        self.language_thresholds = language_thresholds or self._default_thresholds()
        
        # Initialize signal patterns
        self._init_signal_patterns()
        
        # Statistics
        self.stats = defaultdict(int)
        
        logger.info("✅ Signal Detector initialized")
    
    def _default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Default detection thresholds for each language."""
        return {
            'default': {
                'needs_restaurant': 0.35,
                'needs_attraction': 0.35,
                'needs_transportation': 0.40,
                'needs_neighborhood': 0.35,
                'needs_events': 0.35,
                'needs_weather': 0.30,
                'needs_hidden_gems': 0.40,
                'needs_map': 0.45,
                'needs_gps_routing': 0.50,
                'needs_translation': 0.35
            },
            'en': {
                'needs_restaurant': 0.30,
                'needs_attraction': 0.30,
                'needs_transportation': 0.35,
                'needs_neighborhood': 0.30,
                'needs_events': 0.30,
                'needs_weather': 0.25,
                'needs_hidden_gems': 0.35,
                'needs_map': 0.40,
                'needs_gps_routing': 0.45,
                'needs_translation': 0.30
            },
            'tr': {
                'needs_restaurant': 0.35,
                'needs_attraction': 0.35,
                'needs_transportation': 0.40,
                'needs_neighborhood': 0.35,
                'needs_events': 0.35,
                'needs_weather': 0.30,
                'needs_hidden_gems': 0.40,
                'needs_map': 0.45,
                'needs_gps_routing': 0.50,
                'needs_translation': 0.35
            }
        }
    
    def _init_signal_patterns(self):
        """Initialize keyword patterns for each signal."""
        self.signal_patterns = {
            'needs_restaurant': {
                'en': [
                    r'\b(restaurant|cafe|food|eat|dining|lunch|dinner|breakfast|cuisine)\b',
                    r'\b(where\s+to\s+eat|place\s+to\s+eat|grab\s+a\s+bite)\b',
                    r'\b(hungry|meal|dish|menu|reservation)\b'
                ],
                'tr': [
                    r'\b(restoran|kafe|yemek|lokanta|meze|kahvaltı)\b',
                    r'\b(nerede\s+yenir|nerede\s+yemek)\b',
                    r'\b(açım|öğle|akşam\s+yemeği)\b'
                ]
            },
            'needs_attraction': {
                'en': [
                    r'\b(museum|attraction|palace|mosque|church|tower|sight|landmark)\b',
                    r'\b(visit|see|tour|explore|historical|culture)\b',
                    r'\b(what\s+to\s+see|what\s+to\s+visit|things\s+to\s+do)\b'
                ],
                'tr': [
                    r'\b(müze|saray|cami|kilise|kule|anıt|tarihi)\b',
                    r'\b(gezilecek|görülecek|ziyaret|tur)\b',
                    r'\b(ne\s+gezilir|nereye\s+gidilir)\b'
                ]
            },
            'needs_transportation': {
                'en': [
                    r'\b(how\s+to\s+get|how\s+do\s+i\s+get|directions?|route|way\s+to)\b',
                    r'\b(metro|bus|tram|ferry|taxi|transport|travel)\b',
                    r'\b(from.*to|navigate|reach)\b'
                ],
                'tr': [
                    r'\b(nasıl\s+gidilir|nasıl\s+giderim|yol\s+tarifi)\b',
                    r'\b(metro|otobüs|tramvay|vapur|taksi|ulaşım)\b',
                    r'\b(gidiş|ulaşmak)\b'
                ]
            },
            'needs_neighborhood': {
                'en': [
                    r'\b(neighborhood|district|area|quarter|region)\b',
                    r'\b(beyoglu|sultanahmet|kadikoy|besiktas|taksim)\b',
                    r'\b(what.*like|atmosphere|vibe|character)\b'
                ],
                'tr': [
                    r'\b(semt|mahalle|bölge|ilçe)\b',
                    r'\b(beyoğlu|sultanahmet|kadıköy|beşiktaş|taksim)\b',
                    r'\b(nasıl\s+bir\s+yer|atmosfer)\b'
                ]
            },
            'needs_events': {
                'en': [
                    r'\b(event|festival|concert|exhibition|show|performance)\b',
                    r'\b(what.*happening|what.*on|activities)\b',
                    r'\b(tonight|today|weekend|this\s+week)\b'
                ],
                'tr': [
                    r'\b(etkinlik|festival|konser|sergi|gösteri)\b',
                    r'\b(ne\s+var|neler\s+oluyor|aktivite)\b',
                    r'\b(bu\s+gece|bugün|hafta\s+sonu)\b'
                ]
            },
            'needs_weather': {
                'en': [
                    r'\b(weather|rain|sunny|temperature|forecast|cold|hot)\b',
                    r'\b(should\s+i\s+bring|what\s+to\s+wear)\b',
                    r'\b(umbrella|jacket|outdoor|indoor)\b'
                ],
                'tr': [
                    r'\b(hava\s+durumu|yağmur|güneşli|sıcaklık|tahmin)\b',
                    r'\b(ne\s+giysem|şemsiye|mont)\b',
                    r'\b(dışarı|içeri|açık\s+hava)\b'
                ]
            },
            'needs_hidden_gems': {
                'en': [
                    r'\b(hidden\s+gem|off.*beaten.*path|local.*secret|authentic)\b',
                    r'\b(less\s+touristy|not\s+many\s+tourist|unknown|secret)\b',
                    r'\b(locals?\s+go|locals?\s+favorite)\b'
                ],
                'tr': [
                    r'\b(gizli\s+cennet|turistik\s+olmayan|yerel\s+sır)\b',
                    r'\b(az\s+bilinen|bilinmeyen|saklı)\b',
                    r'\b(yerel.*gider|yerel.*favori)\b'
                ]
            },
            'needs_map': {
                'en': [
                    r'\b(map|show.*map|visual|locate|location)\b',
                    r'\b(where.*is|where.*are|find.*on.*map)\b'
                ],
                'tr': [
                    r'\b(harita|haritada\s+göster|konum|yer)\b',
                    r'\b(nerede|haritada\s+bul)\b'
                ]
            },
            'needs_gps_routing': {
                'en': [
                    r'\b(take\s+me|guide\s+me|navigate|gps|turn.*by.*turn)\b',
                    r'\b(from\s+here|my\s+location|current\s+location)\b'
                ],
                'tr': [
                    r'\b(beni\s+götür|yol\s+göster|navigasyon|gps)\b',
                    r'\b(buradan|konumum|bulunduğum)\b'
                ]
            },
            'needs_translation': {
                'en': [
                    r'\b(translate|translation|how\s+do\s+you\s+say|what.*mean)\b',
                    r'\b(in\s+turkish|in\s+english|language)\b'
                ],
                'tr': [
                    r'\b(çevir|çeviri|nasıl\s+denir|ne\s+demek)\b',
                    r'\b(türkçe|ingilizce|dil)\b'
                ]
            }
        }
    
    async def detect_signals(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en",
        user_id: Optional[str] = None,
        experimentation_manager=None
    ) -> Dict[str, Any]:
        """
        Detect all applicable signals from a query.
        
        Args:
            query: User query
            user_location: User GPS location
            language: Query language
            user_id: User identifier (for A/B testing)
            experimentation_manager: Experimentation manager for A/B testing
            
        Returns:
            Dict with:
            - signals: Dict of signal_name -> bool
            - confidence_scores: Dict of signal_name -> float
            - detection_method: Dict of signal_name -> 'keyword' | 'semantic'
        """
        self.stats['total_detections'] += 1
        
        query_lower = query.lower()
        
        # Get thresholds for this language
        thresholds = self.language_thresholds.get(
            language,
            self.language_thresholds['default']
        )
        
        # Initialize results
        signals = {}
        confidence_scores = {}
        detection_method = {}
        
        # Detect each signal
        for signal_name in thresholds.keys():
            # Get threshold (may be from A/B test)
            threshold = self._get_threshold(
                signal_name=signal_name,
                language=language,
                user_id=user_id,
                experimentation_manager=experimentation_manager
            )
            
            # Try keyword detection first (fast)
            keyword_match, keyword_confidence = self._keyword_detection(
                query_lower=query_lower,
                signal_name=signal_name,
                language=language
            )
            
            if keyword_match:
                signals[signal_name] = True
                confidence_scores[signal_name] = keyword_confidence
                detection_method[signal_name] = 'keyword'
                self.stats[f'{signal_name}_keyword'] += 1
                continue
            
            # Try semantic detection (slower but more accurate)
            if self.embedding_model:
                semantic_match, semantic_confidence = await self._semantic_detection(
                    query=query,
                    signal_name=signal_name,
                    threshold=threshold
                )
                
                if semantic_match:
                    signals[signal_name] = True
                    confidence_scores[signal_name] = semantic_confidence
                    detection_method[signal_name] = 'semantic'
                    self.stats[f'{signal_name}_semantic'] += 1
                    continue
            
            # No match
            signals[signal_name] = False
            confidence_scores[signal_name] = 0.0
            detection_method[signal_name] = 'none'
        
        # Special case: GPS routing requires user location
        if signals.get('needs_gps_routing') and not user_location:
            signals['needs_gps_routing'] = False
            logger.debug("GPS routing signal disabled (no user location)")
        
        # Track multi-signal queries
        active_count = sum(1 for v in signals.values() if v)
        if active_count > 2:
            self.stats['multi_signal_queries'] += 1
        
        return {
            'signals': signals,
            'confidence_scores': confidence_scores,
            'detection_method': detection_method,
            'active_count': active_count
        }
    
    def _get_threshold(
        self,
        signal_name: str,
        language: str,
        user_id: Optional[str],
        experimentation_manager
    ) -> float:
        """
        Get threshold for a signal (may be from A/B test).
        
        Args:
            signal_name: Signal name
            language: Language code
            user_id: User identifier
            experimentation_manager: Experimentation manager
            
        Returns:
            Threshold value
        """
        # Check if there's an active A/B test
        if experimentation_manager and user_id:
            try:
                threshold = experimentation_manager.get_threshold_for_experiment(
                    signal_name=signal_name,
                    language=language,
                    user_id=user_id
                )
                if threshold:
                    return threshold
            except Exception as e:
                logger.warning(f"Failed to get experimental threshold: {e}")
        
        # Default threshold
        thresholds = self.language_thresholds.get(
            language,
            self.language_thresholds['default']
        )
        return thresholds.get(signal_name, 0.35)
    
    def _keyword_detection(
        self,
        query_lower: str,
        signal_name: str,
        language: str
    ) -> Tuple[bool, float]:
        """
        Keyword-based detection.
        
        Args:
            query_lower: Lowercase query
            signal_name: Signal name
            language: Language code
            
        Returns:
            Tuple of (matched, confidence)
        """
        patterns = self.signal_patterns.get(signal_name, {}).get(language, [])
        
        if not patterns:
            return False, 0.0
        
        # Check each pattern
        matches = 0
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                matches += 1
        
        if matches > 0:
            # Confidence based on number of matches
            confidence = min(1.0, 0.5 + (matches * 0.25))
            return True, confidence
        
        return False, 0.0
    
    async def _semantic_detection(
        self,
        query: str,
        signal_name: str,
        threshold: float
    ) -> Tuple[bool, float]:
        """
        Semantic similarity detection.
        
        Args:
            query: User query
            signal_name: Signal name
            threshold: Detection threshold
            
        Returns:
            Tuple of (matched, confidence)
        """
        if not self.embedding_model:
            return False, 0.0
        
        try:
            # Get query embedding
            query_embedding = await self.embedding_model.encode(query)
            
            # Get signal embedding (cached)
            signal_embedding = await self._get_signal_embedding(signal_name)
            
            # Calculate similarity
            similarity = self._cosine_similarity(query_embedding, signal_embedding)
            
            # Check against threshold
            matched = similarity >= threshold
            
            return matched, similarity
            
        except Exception as e:
            logger.warning(f"Semantic detection failed for {signal_name}: {e}")
            return False, 0.0
    
    async def _get_signal_embedding(self, signal_name: str):
        """Get or compute embedding for a signal (cached)."""
        # TODO: Implement caching
        # For now, use signal name as proxy
        signal_text = signal_name.replace('needs_', '').replace('_', ' ')
        return await self.embedding_model.encode(signal_text)
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return dict(self.stats)

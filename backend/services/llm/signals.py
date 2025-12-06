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
- needs_airport: Airport transport information
- needs_daily_life: Practical living tips (NEW - Phase 2)
- needs_shopping: Shopping recommendations (PHASE 3)
- needs_nightlife: Nightlife and entertainment (PHASE 3)
- needs_family_friendly: Family-friendly activities (PHASE 3)

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
                'needs_gps_routing': 0.35,  # Lowered from 0.50
                'needs_translation': 0.35,
                'needs_airport': 0.30,  # Airport transport queries
                'needs_daily_life': 0.30,  # Practical living tips
                'needs_shopping': 0.35,  # PHASE 3
                'needs_nightlife': 0.35,  # PHASE 3
                'needs_family_friendly': 0.35  # PHASE 3
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
                'needs_gps_routing': 0.30,  # Lowered from 0.45
                'needs_translation': 0.30,
                'needs_airport': 0.25,  # Airport transport queries
                'needs_daily_life': 0.25,  # Practical living tips
                'needs_shopping': 0.30,  # PHASE 3
                'needs_nightlife': 0.30,  # PHASE 3
                'needs_family_friendly': 0.30  # PHASE 3
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
                'needs_gps_routing': 0.35,  # Lowered from 0.50
                'needs_translation': 0.35,
                'needs_daily_life': 0.30,  # Practical living tips
                'needs_shopping': 0.35,  # PHASE 3
                'needs_nightlife': 0.35,  # PHASE 3
                'needs_family_friendly': 0.35  # PHASE 3
            }
        }
    
    def _init_signal_patterns(self):
        """Initialize keyword patterns for each signal."""
        self.signal_patterns = {
            'needs_restaurant': {
                'en': [
                    r'\b(restaurants?|cafes?|food|eat|eating|dining|lunch|dinner|breakfast|brunch|cuisine|eatery|eateries)\b',
                    r'\b(where\s+to\s+eat|where\s+can\s+i\s+eat|place\s+to\s+eat|grab\s+a\s+bite|places?\s+to\s+dine)\b',
                    r'\b(hungry|meals?|dishes?|menus?|reservations?|food\s+options)\b',
                    r'\b(nearby|near\s+me|near\s+by|close\s+to\s+me|close\s+by|around\s+me|around\s+here|in\s+the\s+area)\b'  # Nearby patterns
                ],
                'tr': [
                    r'\b(restoranlar?|kafeler?|yemek|lokanta|meze|kahvaltı)\b',
                    r'\b(nerede\s+yenir|nerede\s+yemek|yemek\s+yerleri)\b',
                    r'\b(açım|öğle|akşam\s+yemeği)\b',
                    r'\b(yakın|yakında|yakınımda|burada|çevrede|civarda)\b'  # Turkish nearby
                ]
            },
            'needs_attraction': {
                'en': [
                    r'\b(museums?|attractions?|palaces?|mosques?|churches?|towers?|sights?|landmarks?|monuments?)\b',
                    r'\b(visit|visiting|see|seeing|tour|tours|explore|exploring|historical|historic|culture|cultural)\b',
                    r'\b(what\s+to\s+see|what\s+to\s+visit|things\s+to\s+do|places\s+to\s+visit)\b',
                    r'\b(nearby|near\s+me|near\s+by|close\s+to\s+me|close\s+by|around\s+me|around\s+here|in\s+the\s+area)\b'  # Nearby patterns
                ],
                'tr': [
                    r'\b(müzeler?|saraylar?|camiler?|kiliseler?|kuleler?|anıtlar?|tarihi|yerler)\b',
                    r'\b(gezilecek|görülecek|ziyaret|tur|gezmek|görmek)\b',
                    r'\b(ne\s+gezilir|nereye\s+gidilir|neler\s+var)\b',
                    r'\b(yakın|yakında|yakınımda|burada|çevrede|civarda)\b'  # Turkish nearby patterns
                ]
            },
            'needs_transportation': {
                'en': [
                    r'\b(how\s+to\s+get|how\s+do\s+i\s+get|how\s+can\s+i\s+go|how.*go\s+to|directions?|route|way\s+to)\b',
                    r'\b(metro|bus|tram|ferry|taxi|transport|travel|transit)\b',
                    r'\b(from.*to|navigate|reach|get\s+to)\b'
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
                    r'\b(tonight|today|weekend|this\s+week)\b',
                    r'\b(nearby|near\s+me|close\s+to\s+me|around\s+me|around\s+here)\b'  # Nearby patterns
                ],
                'tr': [
                    r'\b(etkinlik|festival|konser|sergi|gösteri)\b',
                    r'\b(ne\s+var|neler\s+oluyor|aktivite)\b',
                    r'\b(bu\s+gece|bugün|hafta\s+sonu)\b',
                    r'\b(yakın|yakında|burada|çevrede|yakınımda)\b'  # Turkish nearby
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
                    r'\b(locals?\s+go|locals?\s+favorite)\b',
                    r'\b(nearby|near\s+me|close\s+to\s+me|around\s+me|around\s+here)\b'  # Nearby patterns
                ],
                'tr': [
                    r'\b(gizli\s+cennet|turistik\s+olmayan|yerel\s+sır)\b',
                    r'\b(az\s+bilinen|bilinmeyen|saklı)\b',
                    r'\b(yerel.*gider|yerel.*favori)\b',
                    r'\b(yakın|yakında|burada|çevrede|yakınımda)\b'  # Turkish nearby
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
                    r'\b(from\s+here|my\s+location|current\s+location)\b',
                    r'\b(how\s+to\s+get|how\s+do\s+i\s+get|directions?|route|way\s+to)\b',
                    r'\b(get\s+to|go\s+to|reach|travel\s+to)\b'
                ],
                'tr': [
                    r'\b(beni\s+götür|yol\s+göster|navigasyon|gps)\b',
                    r'\b(buradan|konumum|bulunduğum)\b',
                    r'\b(nasıl\s+gidilir|nasıl\s+giderim|yol\s+tarifi)\b',
                    r'\b(ulaşmak|varmak|gitmek)\b'
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
            },
            'needs_airport': {
                'en': [
                    r'\b(airport|flight|terminal|arrival|departure|IST|SAW)\b',
                    r'\b(istanbul\s+airport|sabiha\s+gokcen|atatürk\s+airport)\b',
                    r'\b(to\s+airport|from\s+airport|airport\s+transport|airport\s+shuttle)\b',
                    r'\b(how\s+to\s+get.*airport|reach.*airport|go.*airport)\b'
                ],
                'tr': [
                    r'\b(havalimanı|uçuş|terminal|varış|kalkış|IST|SAW)\b',
                    r'\b(istanbul\s+havalimanı|sabiha\s+gökçen|atatürk\s+havalimanı)\b',
                    r'\b(havalimanına|havalimanından|havalimanı\s+ulaşım)\b',
                    r'\b(nasıl\s+gidilir.*havalimanı|havalimanına\s+ulaş)\b'
                ]
            },
            'needs_daily_life': {
                'en': [
                    r'\b(where\s+to\s+buy|where\s+can\s+i\s+buy|where\s+to\s+get)\b',
                    r'\b(pharmacy|drugstore|medicine|prescription)\b',
                    r'\b(bank|atm|exchange|money|currency)\b',
                    r'\b(grocery|supermarket|market|shopping|convenience)\b',
                    r'\b(post\s+office|mail|package|send)\b',
                    r'\b(hospital|doctor|clinic|medical|dentist)\b',
                    r'\b(sim\s+card|phone|mobile|internet|wifi)\b',
                    r'\b(practical|daily\s+life|living|expat|local\s+life)\b',
                    r'\b(nearby|near\s+me|close\s+to\s+me|around\s+me|around\s+here)\b'  # Nearby patterns
                ],
                'tr': [
                    r'\b(nerede\s+bulabilirim|nerede\s+alabilirim|nereden\s+alınır)\b',
                    r'\b(eczane|ilaç|reçete)\b',
                    r'\b(banka|atm|döviz|para|kur)\b',
                    r'\b(market|süpermarket|bakkal|manav)\b',
                    r'\b(ptt|kargo|posta|gönderi)\b',
                    r'\b(hastane|doktor|klinik|sağlık|diş)\b',
                    r'\b(sim\s+kart|telefon|mobil|internet)\b',
                    r'\b(pratik|günlük\s+hayat|yaşam|yerel)\b',
                    r'\b(yakın|yakında|burada|çevrede|yakınımda)\b'  # Turkish nearby
                ],
                'ru': [
                    r'\b(где\s+купить|где\s+можно\s+купить|где\s+найти)\b',
                    r'\b(аптека|лекарство|рецепт|медикаменты)\b',
                    r'\b(банк|банкомат|обмен|деньги|валюта)\b',
                    r'\b(магазин|супермаркет|продукты|рынок)\b',
                    r'\b(почта|посылка|отправить)\b',
                    r'\b(больница|врач|клиника|медицинский|стоматолог)\b',
                    r'\b(сим[\s-]?карта|телефон|мобильный|интернет)\b',
                    r'\b(практический|повседневная\s+жизнь|жизнь|местный)\b',
                    r'\b(рядом|рядом\s+со\s+мной|около|близко)\b'  # Russian nearby
                ],
                'de': [
                    r'\b(wo\s+kann\s+ich\s+kaufen|wo\s+finde\s+ich|wo\s+bekomme\s+ich)\b',
                    r'\b(apotheke|medizin|medikamente|rezept)\b',
                    r'\b(bank|geldautomat|wechsel|geld|währung)\b',
                    r'\b(supermarkt|geschäft|markt|einkaufen)\b',
                    r'\b(post|paket|senden|versenden)\b',
                    r'\b(krankenhaus|arzt|klinik|medizinisch|zahnarzt)\b',
                    r'\b(sim[\s-]?karte|telefon|handy|internet|wifi)\b',
                    r'\b(praktisch|alltag|leben|lokal)\b',
                    r'\b(in\s+der\s+nähe|nahe|in\s+meiner\s+nähe|um\s+mich\s+herum)\b'  # German nearby
                ],
                'fr': [
                    r'\b(où\s+acheter|où\s+puis[\s-]?je\s+acheter|où\s+trouver)\b',
                    r'\b(pharmacie|médicament|ordonnance|médecine)\b',
                    r'\b(banque|distributeur|guichet|argent|devise|change)\b',
                    r'\b(supermarché|magasin|marché|épicerie|courses)\b',
                    r'\b(poste|colis|envoyer|courrier)\b',
                    r'\b(hôpital|médecin|docteur|clinique|dentiste)\b',
                    r'\b(carte\s+sim|téléphone|mobile|internet|wifi)\b',
                    r'\b(pratique|vie\s+quotidienne|vivre|local)\b',
                    r'\b(à\s+proximité|près\s+de\s+moi|proche|autour\s+de\s+moi)\b'  # French nearby
                ]
            },
            # PHASE 3: New Signals
            'needs_shopping': {
                'en': [
                    r'\b(shop|shopping|mall|market|store|boutique|buy|purchase)\b',
                    r'\b(grand\s+bazaar|spice\s+market|istiklal|shopping\s+street)\b',
                    r'\b(souvenir|gift|clothes|fashion|retail)\b',
                    r'\b(nearby|near\s+me|close\s+to\s+me|around\s+me|around\s+here)\b'  # Nearby patterns
                ],
                'tr': [
                    r'\b(alışveriş|mağaza|market|çarşı|pazar|dükkan|satın\s+al)\b',
                    r'\b(kapalı\s+çarşı|mısır\s+çarşısı|istiklal)\b',
                    r'\b(hediyelik|hediye|kıyafet|moda)\b',
                    r'\b(yakın|yakında|burada|çevrede|yakınımda)\b'  # Turkish nearby
                ]
            },
            'needs_nightlife': {
                'en': [
                    r'\b(nightlife|bar|club|pub|party|drink|cocktail)\b',
                    r'\b(night\s+out|going\s+out|evening|late\s+night)\b',
                    r'\b(live\s+music|dj|dance|entertainment)\b',
                    r'\b(nearby|near\s+me|close\s+to\s+me|around\s+me|around\s+here)\b'  # Nearby patterns
                ],
                'tr': [
                    r'\b(gece\s+hayatı|bar|kulüp|pub|parti|içki|kokteyl)\b',
                    r'\b(gece\s+çıkma|eğlence|akşam)\b',
                    r'\b(canlı\s+müzik|dj|dans)\b',
                    r'\b(yakın|yakında|burada|çevrede|yakınımda)\b'  # Turkish nearby
                ]
            },
            'needs_family_friendly': {
                'en': [
                    r'\b(family|kid|child|children|baby|toddler)\b',
                    r'\b(family.*friendly|kid.*friendly|with\s+kids|with\s+children)\b',
                    r'\b(playground|aquarium|zoo|park|activity\s+for\s+kids)\b',
                    r'\b(nearby|near\s+me|close\s+to\s+me|around\s+me|around\s+here)\b'  # Nearby patterns
                ],
                'tr': [
                    r'\b(aile|çocuk|bebek|küçük)\b',
                    r'\b(aile\s+dostu|çocuklu|çocuklarla)\b',
                    r'\b(oyun\s+alanı|akvaryum|hayvanat\s+bahçesi|park)\b',
                    r'\b(yakın|yakında|burada|çevrede|yakınımda)\b'  # Turkish nearby
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
        
        # Priority order: Check transportation/directions signals first
        # This prevents "Kadikoy" from being detected as neighborhood when query is "how to go to Kadikoy"
        priority_signals = ['needs_transportation', 'needs_gps_routing']
        other_signals = [s for s in thresholds.keys() if s not in priority_signals]
        signal_order = priority_signals + other_signals
        
        # Detect each signal in priority order
        for signal_name in signal_order:
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

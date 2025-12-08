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
import time
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict

# Import numpy with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    # NumPy is optional - using Python fallback implementations

# Import fuzzy matching and enhanced patterns (Phase 1 improvements)
try:
    from .fuzzy_matcher import get_fuzzy_matcher, FuzzyMatcher
    from .enhanced_patterns import (
        get_enhanced_patterns,
        get_fuzzy_keywords,
        CONTEXT_NEGATIVE,
        CONTEXT_INTENSIFIERS
    )
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logging.warning("⚠️ Fuzzy matching not available. Install requirements_phase1.txt")

# Import embedding service (Phase 4 - Priority 1)
try:
    from .embedding_service import get_embedding_service, EmbeddingService
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("⚠️ Embedding service not available. Install sentence-transformers")

# Import Istanbul knowledge (Phase 4 - Priority 2)
try:
    from .istanbul_knowledge import get_istanbul_knowledge, IstanbulKnowledge
    ISTANBUL_KNOWLEDGE_AVAILABLE = True
except ImportError:
    ISTANBUL_KNOWLEDGE_AVAILABLE = False
    logging.warning("⚠️ Istanbul knowledge not available")

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
        language_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        enable_fuzzy_matching: bool = True,
        fuzzy_threshold: int = 85,
        enable_context_awareness: bool = True,
        enable_semantic_embeddings: bool = True,
        enable_istanbul_intelligence: bool = True
    ):
        """
        Initialize signal detector.
        
        Args:
            embedding_model: Embedding model for semantic matching (deprecated, use embedding_service)
            language_thresholds: Language-specific detection thresholds
            enable_fuzzy_matching: Enable fuzzy matching for misspellings (Phase 1)
            fuzzy_threshold: Minimum similarity for fuzzy match (0-100)
            enable_context_awareness: Enable context-aware disambiguation (Phase 1)
            enable_semantic_embeddings: Enable real semantic embeddings (Phase 4.1)
            enable_istanbul_intelligence: Enable Istanbul-specific knowledge (Phase 4.2)
        """
        self.embedding_model = embedding_model  # Legacy support
        self.language_thresholds = language_thresholds or self._default_thresholds()
        self.enable_fuzzy_matching = enable_fuzzy_matching and FUZZY_AVAILABLE
        self.enable_context_awareness = enable_context_awareness
        self.enable_semantic_embeddings = enable_semantic_embeddings and EMBEDDINGS_AVAILABLE
        self.enable_istanbul_intelligence = enable_istanbul_intelligence and ISTANBUL_KNOWLEDGE_AVAILABLE
        
        # Initialize fuzzy matcher (Phase 1)
        self.fuzzy_matcher = None
        if self.enable_fuzzy_matching:
            try:
                self.fuzzy_matcher = get_fuzzy_matcher(
                    fuzzy_threshold=fuzzy_threshold,
                    enable_phonetic=True
                )
                logger.info("✅ Fuzzy matching enabled (phonetic search available)")
            except Exception as e:
                logger.info(f"ℹ️  Fuzzy matcher not available - using exact matching")
                self.enable_fuzzy_matching = False
        
        # Initialize embedding service (Phase 4.1)
        self.embedding_service = None
        if self.enable_semantic_embeddings:
            try:
                self.embedding_service = get_embedding_service(model_name='lightweight')
                health = self.embedding_service.health_check()
                logger.info(f"✅ Semantic embeddings enabled: {health['status']}")
            except Exception as e:
                logger.info(f"ℹ️  Embedding service not available - using keyword matching")
                self.enable_semantic_embeddings = False
        
        # Initialize Istanbul knowledge (Phase 4.2)
        self.istanbul_knowledge = None
        if self.enable_istanbul_intelligence:
            try:
                self.istanbul_knowledge = get_istanbul_knowledge()
                logger.info(
                    f"✅ Istanbul intelligence enabled: "
                    f"{len(self.istanbul_knowledge.landmarks)} landmarks"
                )
            except Exception as e:
                logger.info(f"ℹ️  Istanbul knowledge not available - using core features")
                self.enable_istanbul_intelligence = False
        
        # Initialize signal patterns
        self._init_signal_patterns()
        
        # Statistics
        self.stats = defaultdict(int)
        
        logger.info(
            f"✅ Signal Detector initialized "
            f"(fuzzy={self.enable_fuzzy_matching}, "
            f"context={self.enable_context_awareness}, "
            f"embeddings={self.enable_semantic_embeddings}, "
            f"istanbul={self.enable_istanbul_intelligence})"
        )
    
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
                    r'\b(nearby|near\s+me|near\s+by|close\s+to\s+me|close\s+by|around\s+me|around\s+here|in\s+the\s+area)\b'
                ],
                'tr': [
                    r'\b(restoranlar?|kafeler?|yemek|lokanta|meze|kahvaltı|restoran)\b',
                    r'\b(nerede\s+yenir|nerede\s+yemek|yemek\s+yerleri)\b',
                    r'\b(açım|öğle|akşam\s+yemeği)\b',
                    r'\b(yakın|yakında|yakınımda|burada|çevrede|civarda)\b'
                ],
                'de': [
                    r'\b(restaurants?|cafés?|essen|essensplatz|küche|gastronomie)\b',
                    r'\b(wo\s+kann\s+ich\s+essen|wo\s+essen|essen\s+gehen|restaurant\s+finden)\b',
                    r'\b(hungrig|mahlzeit|speisen|frühstück|mittagessen|abendessen)\b',
                    r'\b(in\s+der\s+nähe|nahe|nah\s+bei\s+mir|um\s+mich\s+herum|hier\s+in\s+der\s+gegend)\b'
                ],
                'fr': [
                    r'\b(restaurants?|cafés?|nourriture|manger|repas|cuisine|gastronomie)\b',
                    r'\b(où\s+manger|où\s+puis[\s-]?je\s+manger|restaurant\s+près|endroit\s+pour\s+manger)\b',
                    r'\b(faim|petit[\s-]?déjeuner|déjeuner|dîner|plats?)\b',
                    r'\b(à\s+proximité|près\s+de\s+moi|proche|autour\s+de\s+moi|dans\s+le\s+coin)\b'
                ],
                'ru': [
                    r'\b(ресторан|кафе|еда|поесть|питание|кухня)\b',
                    r'\b(где\s+поесть|где\s+можно\s+поесть|где\s+покушать|ресторан\s+рядом)\b',
                    r'\b(голоден|завтрак|обед|ужин|блюда)\b',
                    r'\b(рядом|рядом\s+со\s+мной|поблизости|около|близко|в\s+районе)\b'
                ],
                'ar': [
                    r'\b(مطعم|مطاعم|مقهى|طعام|أكل|وجبة)\b',
                    r'\b(أين\s+آكل|أين\s+أجد\s+مطعم|مكان\s+للأكل)\b',
                    r'\b(جائع|فطور|غداء|عشاء|وجبات)\b',
                    r'\b(قريب|قريب\s+مني|بالقرب|حولي|في\s+المنطقة)\b'
                ]
            },
            'needs_attraction': {
                'en': [
                    r'\b(museums?|attractions?|palaces?|mosques?|churches?|towers?|sights?|landmarks?|monuments?)\b',
                    r'\b(visit|visiting|see|seeing|tour|tours|explore|exploring|historical|historic|culture|cultural)\b',
                    r'\b(what\s+to\s+see|what\s+to\s+visit|things\s+to\s+do|places\s+to\s+visit)\b',
                    r'\b(nearby|near\s+me|near\s+by|close\s+to\s+me|close\s+by|around\s+me|around\s+here|in\s+the\s+area)\b'
                ],
                'tr': [
                    r'\b(müzeler?|saraylar?|camiler?|kiliseler?|kuleler?|anıtlar?|tarihi|yerler)\b',
                    r'\b(gezilecek|görülecek|ziyaret|tur|gezmek|görmek)\b',
                    r'\b(ne\s+gezilir|nereye\s+gidilir|neler\s+var)\b',
                    r'\b(yakın|yakında|yakınımda|burada|çevrede|civarda)\b'
                ],
                'de': [
                    r'\b(museen?|sehenswürdigkeiten?|paläste?|moscheen?|kirchen?|türme?|denkmäler?)\b',
                    r'\b(besuchen|besichtigen|sehen|tour|erkunden|historisch|kultur)\b',
                    r'\b(was\s+zu\s+sehen|was\s+besuchen|sehenswert)\b',
                    r'\b(in\s+der\s+nähe|nahe|nah\s+bei\s+mir|um\s+mich\s+herum|hier\s+in\s+der\s+gegend)\b'
                ],
                'fr': [
                    r'\b(musées?|attractions?|palais|mosquées?|églises?|tours?|monuments?|sites?)\b',
                    r'\b(visiter|voir|excursion|explorer|historique|culture|culturel)\b',
                    r'\b(que\s+voir|que\s+visiter|choses\s+à\s+faire|à\s+voir)\b',
                    r'\b(à\s+proximité|près\s+de\s+moi|proche|autour\s+de\s+moi|dans\s+le\s+coin)\b'
                ],
                'ru': [
                    r'\b(музей|музеи|достопримечательност|дворец|мечеть|церков|башн|памятник)\b',
                    r'\b(посетить|увидеть|экскурсия|исследовать|историческ|культур)\b',
                    r'\b(что\s+посмотреть|куда\s+пойти|что\s+посетить)\b',
                    r'\b(рядом|рядом\s+со\s+мной|поблизости|около|близко|в\s+районе)\b'
                ],
                'ar': [
                    r'\b(متحف|متاحف|معالم|قصر|مسجد|كنيسة|برج|نصب\s+تذكاري)\b',
                    r'\b(زيارة|مشاهدة|جولة|استكشاف|تاريخي|ثقافي)\b',
                    r'\b(ماذا\s+أرى|ماذا\s+أزور|أماكن\s+للزيارة)\b',
                    r'\b(قريب|قريب\s+مني|بالقرب|حولي|في\s+المنطقة)\b'
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
                    r'\b(nearby|near\s+me|near\s+by|close\s+to\s+me|close\s+by|around\s+me|around\s+here|in\s+the\s+area)\b'
                ],
                'tr': [
                    r'\b(alışveriş|mağaza|market|çarşı|pazar|dükkan|satın\s+al)\b',
                    r'\b(kapalı\s+çarşı|mısır\s+çarşısı|istiklal)\b',
                    r'\b(hediyelik|hediye|kıyafet|moda)\b',
                    r'\b(yakın|yakında|burada|çevrede|yakınımda|civarda)\b'
                ],
                'de': [
                    r'\b(einkaufen|shopping|geschäft|markt|laden|kaufen)\b',
                    r'\b(großer\s+basar|gewürzmarkt|einkaufsstraße)\b',
                    r'\b(souvenir|geschenk|kleidung|mode)\b',
                    r'\b(in\s+der\s+nähe|nahe|nah\s+bei\s+mir|um\s+mich\s+herum|hier\s+in\s+der\s+gegend)\b'
                ],
                'fr': [
                    r'\b(shopping|magasin|marché|boutique|acheter|achats)\b',
                    r'\b(grand\s+bazar|marché\s+aux\s+épices|rue\s+commerçante)\b',
                    r'\b(souvenir|cadeau|vêtements|mode)\b',
                    r'\b(à\s+proximité|près\s+de\s+moi|proche|autour\s+de\s+moi|dans\s+le\s+coin)\b'
                ],
                'ru': [
                    r'\b(магазин|торговый\s+центр|рынок|покупки|купить)\b',
                    r'\b(гранд[\s-]?базар|рынок\s+специй|торговая\s+улица)\b',
                    r'\b(сувенир|подарок|одежда|мода)\b',
                    r'\b(рядом|рядом\s+со\s+мной|поблизости|около|близко|в\s+районе)\b'
                ],
                'ar': [
                    r'\b(تسوق|محل|سوق|متجر|شراء)\b',
                    r'\b(البازار\s+الكبير|سوق\s+التوابل|شارع\s+التسوق)\b',
                    r'\b(هدية\s+تذكارية|هدية|ملابس|موضة)\b',
                    r'\b(قريب|قريب\s+مني|بالقرب|حولي|في\s+المنطقة)\b'
                ]
            },
            'needs_nightlife': {
                'en': [
                    r'\b(nightlife|bar|club|pub|party|drink|cocktail)\b',
                    r'\b(night\s+out|going\s+out|evening|late\s+night)\b',
                    r'\b(live\s+music|dj|dance|entertainment)\b',
                    r'\b(nearby|near\s+me|near\s+by|close\s+to\s+me|close\s+by|around\s+me|around\s+here|in\s+the\s+area)\b'
                ],
                'tr': [
                    r'\b(gece\s+hayatı|bar|kulüp|pub|parti|içki|kokteyl)\b',
                    r'\b(gece\s+çıkma|eğlence|akşam)\b',
                    r'\b(canlı\s+müzik|dj|dans)\b',
                    r'\b(yakın|yakında|burada|çevrede|yakınımda|civarda)\b'
                ],
                'de': [
                    r'\b(nachtleben|bar|club|kneipe|party|trinken|cocktail)\b',
                    r'\b(ausgehen|abend|nacht)\b',
                    r'\b(live[\s-]?musik|dj|tanzen|unterhaltung)\b',
                    r'\b(in\s+der\s+nähe|nahe|nah\s+bei\s+mir|um\s+mich\s+herum|hier\s+in\s+der\s+gegend)\b'
                ],
                'fr': [
                    r'\b(vie\s+nocturne|bar|club|pub|fête|boisson|cocktail)\b',
                    r'\b(sortir\s+le\s+soir|soirée|nuit)\b',
                    r'\b(musique\s+live|dj|danse|divertissement)\b',
                    r'\b(à\s+proximité|près\s+de\s+moi|proche|autour\s+de\s+moi|dans\s+le\s+coin)\b'
                ],
                'ru': [
                    r'\b(ночная\s+жизнь|бар|клуб|паб|вечеринка|напиток|коктейль)\b',
                    r'\b(пойти\s+вечером|вечер|ночь)\b',
                    r'\b(живая\s+музыка|диджей|танц|развлечени)\b',
                    r'\b(рядом|рядом\s+со\s+мной|поблизости|около|близко|в\s+районе)\b'
                ],
                'ar': [
                    r'\b(حياة\s+ليلية|بار|نادي|حفلة|مشروب|كوكتيل)\b',
                    r'\b(خروج\s+ليلي|سهرة|ليل)\b',
                    r'\b(موسيقى\s+حية|دي\s+جي|رقص|ترفيه)\b',
                    r'\b(قريب|قريب\s+مني|بالقرب|حولي|في\s+المنطقة)\b'
                ]
            },
            'needs_family_friendly': {
                'en': [
                    r'\b(family|kid|child|children|baby|toddler)\b',
                    r'\b(family.*friendly|kid.*friendly|with\s+kids|with\s+children)\b',
                    r'\b(playground|aquarium|zoo|park|activity\s+for\s+kids)\b',
                    r'\b(nearby|near\s+me|near\s+by|close\s+to\s+me|close\s+by|around\s+me|around\s+here|in\s+the\s+area)\b'
                ],
                'tr': [
                    r'\b(aile|çocuk|bebek|küçük)\b',
                    r'\b(aile\s+dostu|çocuklu|çocuklarla)\b',
                    r'\b(oyun\s+alanı|akvaryum|hayvanat\s+bahçesi|park)\b',
                    r'\b(yakın|yakında|burada|çevrede|yakınımda|civarda)\b'
                ],
                'de': [
                    r'\b(familie|kind|kinder|baby|kleinkind)\b',
                    r'\b(familienfreundlich|kinderfreundlich|mit\s+kindern)\b',
                    r'\b(spielplatz|aquarium|zoo|park|kinderaktivität)\b',
                    r'\b(in\s+der\s+nähe|nahe|nah\s+bei\s+mir|um\s+mich\s+herum|hier\s+in\s+der\s+gegend)\b'
                ],
                'fr': [
                    r'\b(famille|enfant|enfants|bébé|tout[\s-]?petit)\b',
                    r'\b(familial|adapté\s+aux\s+enfants|avec\s+des\s+enfants)\b',
                    r'\b(aire\s+de\s+jeux|aquarium|zoo|parc|activité\s+pour\s+enfants)\b',
                    r'\b(à\s+proximité|près\s+de\s+moi|proche|autour\s+de\s+moi|dans\s+le\s+coin)\b'
                ],
                'ru': [
                    r'\b(семья|ребенок|дети|младенец|малыш)\b',
                    r'\b(семейн|для\s+детей|с\s+детьми)\b',
                    r'\b(площадка|аквариум|зоопарк|парк|активност.*детей)\b',
                    r'\b(рядом|рядом\s+со\s+мной|поблизости|около|близко|в\s+районе)\b'
                ],
                'ar': [
                    r'\b(عائلة|طفل|أطفال|رضيع)\b',
                    r'\b(عائلي|مناسب\s+للأطفال|مع\s+الأطفال)\b',
                    r'\b(ملعب|حوض\s+سمك|حديقة\s+حيوان|حديقة|نشاط\s+للأطفال)\b',
                    r'\b(قريب|قريب\s+مني|بالقرب|حولي|في\s+المنطقة)\b'
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
        
        # Calculate confidence scores for detected signals
        for signal_name, detected in signals.items():
            if detected and signal_name in detection_method:
                confidence_scores[signal_name] = self._calculate_signal_confidence(
                    query=query,
                    signal_name=signal_name,
                    language=language,
                    detection_method=detection_method[signal_name]
                )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(confidence_scores)
        
        return {
            'signals': signals,
            'confidence_scores': confidence_scores,
            'overall_confidence': overall_confidence,
            'detection_method': detection_method,
            'active_count': active_count
        }
    
    async def detect_signals_multipass(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None,
        language: str = 'en',
        user_id: Optional[str] = None,
        experimentation_manager=None,
        enable_multipass: bool = True
    ) -> Dict[str, Any]:
        """
        Multi-pass signal detection with progressive enhancement (PHASE 3 - PRIORITY 4).
        
        Tries multiple detection strategies in sequence:
        1. Pass 1: Fast regex detection (always)
        2. Pass 2: Fuzzy matching (if confidence < 0.6)
        3. Pass 3: Semantic embeddings (if confidence < 0.7)
        4. Pass 4: Query expansion (if confidence < 0.75)
        
        Args:
            query: User query
            user_location: User's GPS location
            language: Language code
            user_id: User identifier
            experimentation_manager: For A/B testing
            enable_multipass: Enable multi-pass detection
            
        Returns:
            Dict with signals, confidence scores, and metadata
        """
        # Start timer
        start_time = time.time()
        
        # Pass 1: Standard regex detection (fast path)
        logger.debug(f"🔍 Pass 1: Regex detection for '{query}'")
        result = await self.detect_signals(
            query=query,
            user_location=user_location,
            language=language,
            user_id=user_id,
            experimentation_manager=experimentation_manager
        )
        
        pass_info = [{
            'pass': 1,
            'method': 'regex',
            'overall_confidence': result['overall_confidence'],
            'signals_detected': result['active_count']
        }]
        
        # Check if multipass is needed
        if not enable_multipass or result['overall_confidence'] >= 0.75:
            logger.debug(f"✅ Pass 1 sufficient (confidence: {result['overall_confidence']:.2f})")
            result['multipass_info'] = {
                'passes_attempted': 1,
                'passes_info': pass_info,
                'total_time': time.time() - start_time
            }
            return result
        
        # Pass 2: Fuzzy matching (if confidence low)
        if result['overall_confidence'] < 0.6:
            logger.debug(f"🔍 Pass 2: Fuzzy matching (confidence: {result['overall_confidence']:.2f})")
            fuzzy_signals = await self._detect_signals_fuzzy_pass(
                query, language, result['signals']
            )
            
            if fuzzy_signals:
                # Merge fuzzy signals
                for signal_name, fuzzy_detected in fuzzy_signals.items():
                    if fuzzy_detected and not result['signals'].get(signal_name):
                        result['signals'][signal_name] = True
                        result['confidence_scores'][signal_name] = 0.65  # Fuzzy confidence
                        result['detection_method'][signal_name] = 'fuzzy'
                        result['active_count'] += 1
                        logger.debug(f"  ✨ Fuzzy detected: {signal_name}")
                
                # Recalculate overall confidence
                result['overall_confidence'] = self._calculate_overall_confidence(
                    result['confidence_scores']
                )
                
                pass_info.append({
                    'pass': 2,
                    'method': 'fuzzy',
                    'overall_confidence': result['overall_confidence'],
                    'signals_detected': result['active_count'],
                    'new_signals': list(fuzzy_signals.keys())
                })
        
        # Pass 3: Semantic embeddings (if still low confidence)
        if result['overall_confidence'] < 0.7 and self.embedding_model:
            logger.debug(f"🔍 Pass 3: Semantic embeddings (confidence: {result['overall_confidence']:.2f})")
            semantic_signals = await self._detect_signals_semantic_pass(
                query, language, result['signals']
            )
            
            if semantic_signals:
                # Merge semantic signals
                for signal_name, (detected, confidence) in semantic_signals.items():
                    if detected and not result['signals'].get(signal_name):
                        result['signals'][signal_name] = True
                        result['confidence_scores'][signal_name] = confidence
                        result['detection_method'][signal_name] = 'semantic_multipass'
                        result['active_count'] += 1
                        logger.debug(f"  ✨ Semantic detected: {signal_name} ({confidence:.2f})")
                
                # Recalculate overall confidence
                result['overall_confidence'] = self._calculate_overall_confidence(
                    result['confidence_scores']
                )
                
                pass_info.append({
                    'pass': 3,
                    'method': 'semantic',
                    'overall_confidence': result['overall_confidence'],
                    'signals_detected': result['active_count'],
                    'new_signals': list(semantic_signals.keys())
                })
        
        # Pass 4: Query expansion (if still uncertain)
        if result['overall_confidence'] < 0.75:
            logger.debug(f"🔍 Pass 4: Query expansion (confidence: {result['overall_confidence']:.2f})")
            expanded_signals = await self._detect_signals_expansion_pass(
                query, language, result['signals']
            )
            
            if expanded_signals:
                # Merge expanded signals
                for signal_name, expanded_detected in expanded_signals.items():
                    if expanded_detected and not result['signals'].get(signal_name):
                        result['signals'][signal_name] = True
                        result['confidence_scores'][signal_name] = 0.60  # Expansion confidence
                        result['detection_method'][signal_name] = 'expansion'
                        result['active_count'] += 1
                        logger.debug(f"  ✨ Expansion detected: {signal_name}")
                
                # Recalculate overall confidence
                result['overall_confidence'] = self._calculate_overall_confidence(
                    result['confidence_scores']
                )
                
                pass_info.append({
                    'pass': 4,
                    'method': 'expansion',
                    'overall_confidence': result['overall_confidence'],
                    'signals_detected': result['active_count'],
                    'new_signals': list(expanded_signals.keys())
                })
        
        # Pass 5: Istanbul Intelligence (Phase 4.2 - NEW!)
        if self.enable_istanbul_intelligence and self.istanbul_knowledge:
            logger.debug(f"🗺️  Pass 5: Istanbul intelligence")
            istanbul_signals = self._detect_signals_istanbul_pass(
                query, language, result['signals']
            )
            
            if istanbul_signals:
                # Merge Istanbul-specific signals
                for signal_name, confidence in istanbul_signals.items():
                    if not result['signals'].get(signal_name):
                        result['signals'][signal_name] = True
                        result['confidence_scores'][signal_name] = confidence
                        result['detection_method'][signal_name] = 'istanbul_intelligence'
                        result['active_count'] += 1
                        logger.debug(f"  🗺️  Istanbul detected: {signal_name} ({confidence:.2f})")
                
                # Recalculate overall confidence
                result['overall_confidence'] = self._calculate_overall_confidence(
                    result['confidence_scores']
                )
                
                pass_info.append({
                    'pass': 5,
                    'method': 'istanbul_intelligence',
                    'overall_confidence': result['overall_confidence'],
                    'signals_detected': result['active_count'],
                    'new_signals': list(istanbul_signals.keys())
                })
        
        # Add multipass metadata
        result['multipass_info'] = {
            'passes_attempted': len(pass_info),
            'passes_info': pass_info,
            'total_time': time.time() - start_time,
            'improvement': result['overall_confidence'] - pass_info[0]['overall_confidence']
        }
        
        logger.info(
            f"🎯 Multi-pass complete: {len(pass_info)} passes, "
            f"confidence: {pass_info[0]['overall_confidence']:.2f} → {result['overall_confidence']:.2f}"
        )
        
        return result
    
    async def _detect_signals_fuzzy_pass(
        self,
        query: str,
        language: str,
        existing_signals: Dict[str, bool]
    ) -> Dict[str, bool]:
        """
        Fuzzy matching pass for typos and misspellings.
        
        Args:
            query: User query
            language: Language code
            existing_signals: Already detected signals
            
        Returns:
            Dict of newly detected signals
        """
        if not self.enable_fuzzy_matching or not self.fuzzy_matcher:
            return {}
        
        new_signals = {}
        query_words = query.lower().split()
        
        # Check each signal type
        for signal_name, patterns_by_lang in self.signal_patterns.items():
            # Skip already detected
            if existing_signals.get(signal_name):
                continue
            
            # Get patterns for language
            patterns = patterns_by_lang.get(language, [])
            if not patterns:
                continue
            
            # Extract keywords from patterns
            keywords = self._extract_keywords_from_patterns(patterns)
            
            # Fuzzy match query words against keywords
            for word in query_words:
                if len(word) < 3:  # Skip very short words
                    continue
                
                for keyword in keywords:
                    similarity = self.fuzzy_matcher.fuzzy_match_word(word, keyword)
                    if similarity >= 80:  # 80% similarity threshold
                        new_signals[signal_name] = True
                        logger.debug(f"Fuzzy match: '{word}' ≈ '{keyword}' ({similarity}%)")
                        break
                
                if new_signals.get(signal_name):
                    break
        
        return new_signals
    
    async def _detect_signals_semantic_pass(
        self,
        query: str,
        language: str,
        existing_signals: Dict[str, bool]
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Semantic embedding pass for implicit intents (Phase 4 Enhanced).
        
        Uses real sentence embeddings instead of templates for better accuracy.
        
        Args:
            query: User query
            language: Language code
            existing_signals: Already detected signals
            
        Returns:
            Dict of signal_name -> (detected, confidence)
        """
        new_signals = {}
        
        # Phase 4: Use real embedding service if available
        if self.enable_semantic_embeddings and self.embedding_service:
            try:
                # Classify intent using real embeddings
                intent_results = self.embedding_service.classify_intent(
                    query=query,
                    threshold=0.65,
                    top_k=3
                )
                
                # Map intent names to signal names
                for intent_name, (detected, confidence) in intent_results.items():
                    # Skip if already detected
                    if existing_signals.get(intent_name):
                        continue
                    
                    if detected:
                        new_signals[intent_name] = (True, confidence)
                        logger.debug(
                            f"Semantic (embeddings) match: {intent_name} "
                            f"(confidence: {confidence:.3f})"
                        )
                
                return new_signals
                
            except Exception as e:
                logger.warning(f"Embedding-based semantic pass failed: {e}")
                # Fall through to legacy method
        
        # Legacy: Template-based semantic matching (Phase 3)
        if not self.embedding_model:
            return {}
        
        try:
            # Get query embedding (legacy)
            query_embedding = await self.embedding_model.encode(query)
            
            # Intent templates for semantic matching
            intent_templates = {
                'needs_restaurant': [
                    "place to eat", "find food", "hungry", "dining",
                    "restaurant nearby", "where to eat"
                ],
                'needs_attraction': [
                    "what to see", "tourist sites", "visit places", "sightseeing",
                    "historical places", "museums"
                ],
                'needs_transportation': [
                    "how to get there", "directions", "travel route", "go to",
                    "reach location", "navigate"
                ],
                'needs_events': [
                    "things happening", "activities today", "events nearby",
                    "what to do", "entertainment"
                ]
            }
            
            # Check each signal type
            for signal_name, templates in intent_templates.items():
                # Skip already detected
                if existing_signals.get(signal_name):
                    continue
                
                # Calculate semantic similarity with templates
                max_similarity = 0.0
                for template in templates:
                    template_embedding = await self.embedding_model.encode(template)
                    similarity = self._cosine_similarity(query_embedding, template_embedding)
                    max_similarity = max(max_similarity, similarity)
                
                # Threshold for semantic detection
                if max_similarity >= 0.7:
                    new_signals[signal_name] = (True, max_similarity)
                    logger.debug(f"Semantic (legacy) match: {signal_name} ({max_similarity:.2f})")
        
        except Exception as e:
            logger.warning(f"Legacy semantic pass failed: {e}")
        
        return new_signals
    
    async def _detect_signals_expansion_pass(
        self,
        query: str,
        language: str,
        existing_signals: Dict[str, bool]
    ) -> Dict[str, bool]:
        """
        Query expansion pass using synonyms and related terms.
        
        Args:
            query: User query
            language: Language code
            existing_signals: Already detected signals
            
        Returns:
            Dict of newly detected signals
        """
        # Expansion dictionaries (could be loaded from file)
        expansions = {
            'en': {
                'eat': ['dine', 'meal', 'food', 'lunch', 'dinner', 'breakfast'],
                'nearby': ['close', 'near', 'around', 'vicinity', 'local'],
                'good': ['best', 'great', 'top', 'recommended', 'popular'],
                'place': ['spot', 'location', 'venue', 'site'],
                'go': ['get', 'reach', 'travel', 'navigate', 'head'],
                'see': ['visit', 'explore', 'tour', 'view', 'check out']
            },
            'tr': {
                'yemek': ['yiyecek', 'gıda', 'öğün', 'meal'],
                'yakın': ['yakında', 'civar', 'etraf', 'bölge'],
                'iyi': ['güzel', 'harika', 'muhteşem', 'tavsiye'],
                'yer': ['mekan', 'lokasyon', 'alan', 'bölge']
            }
        }
        
        expansion_dict = expansions.get(language, {})
        if not expansion_dict:
            return {}
        
        # Expand query
        query_lower = query.lower()
        expanded_terms = set()
        
        for base_term, related_terms in expansion_dict.items():
            if base_term in query_lower:
                expanded_terms.update(related_terms)
        
        if not expanded_terms:
            return {}
        
        # Build expanded query
        expanded_query = query + " " + " ".join(expanded_terms)
        
        # Run regex detection on expanded query
        new_signals = {}
        
        for signal_name, patterns_by_lang in self.signal_patterns.items():
            # Skip already detected
            if existing_signals.get(signal_name):
                continue
            
            # Get patterns for language
            patterns = patterns_by_lang.get(language, [])
            if not patterns:
                continue
            
            # Check if any pattern matches expanded query
            for pattern in patterns:
                if re.search(pattern, expanded_query, re.IGNORECASE):
                    new_signals[signal_name] = True
                    logger.debug(f"Expansion match: {signal_name} via {expanded_terms}")
                    break
        
        return new_signals
    
    def _detect_signals_istanbul_pass(
        self,
        query: str,
        language: str,
        existing_signals: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Istanbul-specific domain knowledge pass (Phase 4 Priority 2).
        
        Detects signals using Istanbul-specific knowledge:
        - Landmarks and neighborhoods
        - Local transportation terms
        - Istanbul slang and colloquialisms
        - Cultural context
        
        Args:
            query: User query
            language: Language code
            existing_signals: Already detected signals
            
        Returns:
            Dict of signal_name -> confidence
        """
        if not self.istanbul_knowledge:
            return {}
        
        new_signals = {}
        query_lower = query.lower()
        
        # 1. Check for landmark mentions
        for landmark in self.istanbul_knowledge.landmarks:
            if landmark.lower() in query_lower:
                # Landmark mention implies attraction/tourism interest
                if not existing_signals.get('needs_attraction'):
                    new_signals['needs_attraction'] = 0.85
                    logger.debug(f"Istanbul pass: Landmark '{landmark}' → needs_attraction")
                break
        
        # 2. Check for neighborhood mentions
        for neighborhood in self.istanbul_knowledge.neighborhoods:
            if neighborhood.lower() in query_lower:
                # Neighborhood mention often implies neighborhood info
                if not existing_signals.get('needs_neighborhood'):
                    new_signals['needs_neighborhood'] = 0.80
                    logger.debug(f"Istanbul pass: Neighborhood '{neighborhood}' → needs_neighborhood")
                
                # May also imply restaurant search in that area
                if not existing_signals.get('needs_restaurant') and any(
                    food_term in query_lower for food_term in ['yemek', 'eat', 'food', 'restoran', 'lokanta']
                ):
                    new_signals['needs_restaurant'] = 0.75
                    logger.debug(f"Istanbul pass: Neighborhood + food term → needs_restaurant")
                break
        
        # 3. Check for transport mentions (Metrobüs, Marmaray, etc.)
        for transport_term in self.istanbul_knowledge.transport_terms:
            if transport_term.lower() in query_lower:
                if not existing_signals.get('needs_transportation'):
                    new_signals['needs_transportation'] = 0.88
                    logger.debug(f"Istanbul pass: Transport term '{transport_term}' → needs_transportation")
                break
        
        # 4. Check for Istanbul slang/colloquialisms
        for slang_term in self.istanbul_knowledge.slang_terms:
            if slang_term.lower() in query_lower:
                # Some slang terms are very casual, boost local context
                if not existing_signals.get('needs_hidden_gems'):
                    # Slang usage often indicates interest in local/authentic experiences
                    new_signals['needs_hidden_gems'] = 0.70
                    logger.debug(f"Istanbul pass: Slang '{slang_term}' → needs_hidden_gems")
                break
        
        # 5. Cross-Bosphorus travel detection
        bosphorus_keywords = [
            'avrupa', 'asya', 'karşı yakası', 'boğaz', 'karaköy', 'kadıköy',
            'europe', 'asia', 'bosphorus', 'cross', 'ferry', 'vapur'
        ]
        bosphorus_count = sum(1 for kw in bosphorus_keywords if kw in query_lower)
        if bosphorus_count >= 2:  # Need multiple indicators
            if not existing_signals.get('needs_transportation'):
                new_signals['needs_transportation'] = 0.82
                logger.debug("Istanbul pass: Cross-Bosphorus travel detected → needs_transportation")
        
        # 6. Food/cuisine context (Istanbul-specific)
        istanbul_food_terms = [
            'balık ekmek', 'simit', 'midye', 'kokoreç', 'ıslak burger',
            'menemen', 'kahvaltı', 'çay', 'türk kahvesi', 'baklava', 'kebap'
        ]
        if any(food in query_lower for food in istanbul_food_terms):
            if not existing_signals.get('needs_restaurant'):
                new_signals['needs_restaurant'] = 0.87
                logger.debug("Istanbul pass: Local food term detected → needs_restaurant")
        
        # 7. Weather and timing context (Istanbul-specific)
        # Istanbul's unpredictable weather often part of travel planning
        weather_keywords = ['hava', 'yağmur', 'weather', 'rain', 'soğuk', 'sıcak']
        if any(kw in query_lower for kw in weather_keywords):
            if not existing_signals.get('needs_weather'):
                new_signals['needs_weather'] = 0.78
                logger.debug("Istanbul pass: Weather context detected → needs_weather")
        
        # 8. Shopping districts detection
        shopping_areas = ['istiklal', 'nişantaşı', 'bağdat caddesi', 'zorlu', 'kanyon', 'cevahir']
        if any(area in query_lower for area in shopping_areas):
            if not existing_signals.get('needs_shopping'):
                new_signals['needs_shopping'] = 0.80
                logger.debug("Istanbul pass: Shopping district detected → needs_shopping")
        
        # 9. Historical/cultural context
        cultural_keywords = [
            'osmanlı', 'bizans', 'tarih', 'müze', 'saray', 'cami', 'kilise',
            'ottoman', 'byzantine', 'history', 'museum', 'palace', 'mosque', 'church'
        ]
        if any(kw in query_lower for kw in cultural_keywords):
            if not existing_signals.get('needs_attraction'):
                new_signals['needs_attraction'] = 0.85
                logger.debug("Istanbul pass: Cultural/historical context → needs_attraction")
        
        # 10. Nightlife areas (Istanbul-specific)
        nightlife_areas = ['beyoğlu', 'taksim', 'ortaköy', 'kadıköy moda', 'bebek']
        nightlife_keywords = ['bar', 'club', 'nightlife', 'gece hayatı', 'eğlence']
        if any(area in query_lower for area in nightlife_areas) and any(kw in query_lower for kw in nightlife_keywords):
            if not existing_signals.get('needs_nightlife'):
                new_signals['needs_nightlife'] = 0.83
                logger.debug("Istanbul pass: Nightlife area + keyword → needs_nightlife")
        
        return new_signals
    
    def _extract_keywords_from_patterns(self, patterns: List[str]) -> Set[str]:
        """Extract individual keywords from regex patterns."""
        keywords = set()
        
        for pattern in patterns:
            # Remove regex special characters and extract words
            # Pattern like r'\b(restaurant|cafe|food)\b' → ['restaurant', 'cafe', 'food']
            cleaned = pattern.replace(r'\b', '').replace('\\b', '')
            cleaned = cleaned.replace('(', '').replace(')', '')
            cleaned = cleaned.replace('[', '').replace(']', '')
            cleaned = cleaned.replace('?', '').replace('*', '').replace('+', '')
            cleaned = cleaned.replace('\\s', ' ').replace('\\', '')
            
            # Split by |
            parts = cleaned.split('|')
            for part in parts:
                # Extract alphanumeric words
                words = re.findall(r'[a-zA-Zığüşöç]+', part)
                keywords.update(w.lower() for w in words if len(w) > 2)
        
        return keywords
    
    def _keyword_detection(
        self,
        query_lower: str,
        signal_name: str,
        language: str
    ) -> Tuple[bool, float]:
        """
        Fast keyword-based detection using regex patterns.
        
        Args:
            query_lower: Lowercased query
            signal_name: Signal to detect
            language: Language code
            
        Returns:
            (match_found, confidence_score)
        """
        patterns = self.signal_patterns.get(signal_name, {}).get(language, [])
        if not patterns:
            return False, 0.0
        
        # Check each pattern
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return True, 0.9  # High confidence for keyword match
        
        return False, 0.0
    
    async def _semantic_detection(
        self,
        query: str,
        signal_name: str,
        threshold: float
    ) -> Tuple[bool, float]:
        """
        Semantic similarity-based detection using embeddings.
        
        Args:
            query: User query
            signal_name: Signal to detect
            threshold: Similarity threshold
            
        Returns:
            (match_found, confidence_score)
        """
        if not self.embedding_model:
            return False, 0.0
        
        try:
            # This would use actual semantic templates in production
            # For now, return False to rely on keyword detection
            return False, 0.0
        except Exception as e:
            logger.warning(f"Semantic detection failed: {e}")
            return False, 0.0
    
    def _get_threshold(
        self,
        signal_name: str,
        language: str,
        user_id: Optional[str] = None,
        experimentation_manager=None
    ) -> float:
        """
        Get detection threshold (may be from A/B test).
        
        Args:
            signal_name: Signal name
            language: Language code
            user_id: User identifier
            experimentation_manager: Experimentation manager
            
        Returns:
            Detection threshold
        """
        # Get default threshold
        thresholds = self.language_thresholds.get(
            language,
            self.language_thresholds['default']
        )
        return thresholds.get(signal_name, 0.35)
    
    def _calculate_signal_confidence(
        self,
        query: str,
        signal_name: str,
        language: str,
        detection_method: str
    ) -> float:
        """
        Calculate confidence score for a detected signal.
        
        Args:
            query: User query
            signal_name: Signal name
            language: Language code
            detection_method: How it was detected
            
        Returns:
            Confidence score (0-1)
        """
        if detection_method == 'keyword':
            return 0.9
        elif detection_method == 'semantic':
            return 0.8
        elif detection_method == 'fuzzy':
            return 0.65
        elif detection_method == 'expansion':
            return 0.60
        elif detection_method == 'semantic_multipass':
            return 0.75
        else:
            return 0.5
    
    def _calculate_overall_confidence(
        self,
        confidence_scores: Dict[str, float]
    ) -> float:
        """
        Calculate overall confidence from individual signal scores.
        
        Args:
            confidence_scores: Dict of signal -> confidence
            
        Returns:
            Overall confidence (0-1)
        """
        if not confidence_scores:
            return 0.0
        
        # Get scores > 0
        active_scores = [s for s in confidence_scores.values() if s > 0]
        if not active_scores:
            return 0.0
        
        # Return average of active signals
        return sum(active_scores) / len(active_scores)
    
    def _cosine_similarity(
        self,
        embedding1,
        embedding2
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (list or numpy array)
            embedding2: Second embedding (list or numpy array)
            
        Returns:
            Similarity score (0-1)
        """
        try:
            if NUMPY_AVAILABLE and np is not None:
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
            else:
                # Fallback implementation using pure Python
                dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                norm1 = sum(x ** 2 for x in embedding1) ** 0.5
                norm2 = sum(x ** 2 for x in embedding2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.warning(f"Cosine similarity failed: {e}")
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return dict(self.stats)

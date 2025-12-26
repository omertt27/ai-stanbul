"""
Hybrid Transportation Intent Classifier

Combines multiple signals to determine if a query is transportation-related:
1. Regex patterns (fast, catches obvious cases)
2. Semantic similarity (catches human phrasing)
3. Location mentions (contextual signal)
4. LLM fallback (only when uncertain)

This prevents fragile keyword-matching and handles:
- "I'm near the ferry, how do I reach the square?"
- "What's the best way to cross to Europe side?"
- "I'm lost, can you help?"
- "Need to be in Taksim by 9am"

Author: AI Istanbul Team
Date: December 16, 2025
"""

import logging
import re
from typing import Dict, Tuple, Optional, List, Pattern
import numpy as np

logger = logging.getLogger(__name__)


class TransportationIntentClassifier:
    """
    Hybrid intent classifier that combines:
    - Pattern matching (fast, pre-compiled)
    - Semantic embeddings (robust, cached)
    - LLM clarification (uncertain cases)
    - **Negative patterns** (false positive reduction)
    
    Returns confidence score (0.0-1.0) instead of boolean.
    
    Performance optimizations:
    - Pre-compiled regex patterns
    - Cached semantic example embeddings
    - Priority-based negative pattern checking (check negatives FIRST for efficiency)
    """
    
    # Confidence thresholds
    TRANSPORT_THRESHOLD = 0.5  # Above this = transport intent
    UNCERTAIN_LOW = 0.4       # Below this = definitely not transport
    UNCERTAIN_HIGH = 0.6      # Above this = definitely transport
    
    def __init__(self, warm_up: bool = False):
        """
        Initialize the classifier with pre-compiled patterns and cached embeddings.
        
        Args:
            warm_up: If True, pre-compute all embeddings during initialization.
                     Adds ~1-2s startup time but makes first query faster.
        """
        # Build and PRE-COMPILE regex patterns for performance
        high_conf_raw, normal_raw = self._build_patterns()
        negative_raw = self._build_negative_patterns()
        
        self.high_confidence_patterns: List[Pattern] = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in high_conf_raw
        ]
        self.normal_patterns: List[Pattern] = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in normal_raw
        ]
        self.negative_patterns: List[Pattern] = [
            re.compile(p, re.IGNORECASE | re.UNICODE) for p in negative_raw
        ]
        
        self.semantic_examples = self._build_semantic_examples()
        self.location_keywords = self._build_location_keywords()
        
        # Cached embeddings for semantic examples (computed lazily or on warm-up)
        self._example_embeddings: Optional[List[np.ndarray]] = None
        
        # Initialize embedding service (if available)
        self.embedding_service = None
        try:
            from services.llm.embedding_service import get_embedding_service
            self.embedding_service = get_embedding_service()
            logger.info("✅ Hybrid Intent Classifier initialized with embeddings")
            
            # Optional: warm up embeddings cache
            if warm_up and self.embedding_service and not self.embedding_service.offline_mode:
                self._get_example_embeddings()
                
        except Exception as e:
            logger.warning(f"⚠️ Embeddings not available, using patterns only: {e}")
    
    def warm_up(self) -> bool:
        """
        Pre-compute all cached values for faster first query.
        Call this during app startup if you want deterministic latency.
        
        Returns:
            True if warm-up successful, False otherwise
        """
        try:
            embeddings = self._get_example_embeddings()
            if embeddings:
                logger.info(f"🔥 Classifier warmed up: {len(embeddings)} embeddings cached")
                return True
            return False
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")
            return False
    
    def _get_example_embeddings(self) -> Optional[List[np.ndarray]]:
        """
        Lazily compute and cache embeddings for semantic examples.
        Only computed once, then reused for all subsequent calls.
        """
        if self._example_embeddings is not None:
            return self._example_embeddings
        
        if not self.embedding_service or self.embedding_service.offline_mode:
            return None
        
        try:
            embeddings = []
            for example in self.semantic_examples:
                emb = self.embedding_service.encode(example)
                if emb is not None:
                    embeddings.append(emb)
            
            if embeddings:
                self._example_embeddings = embeddings
                logger.info(f"✅ Cached {len(embeddings)} semantic example embeddings")
                return self._example_embeddings
        except Exception as e:
            logger.warning(f"Failed to cache example embeddings: {e}")
        
        return None
    
    def _build_patterns(self) -> Tuple[List[str], List[str]]:
        """
        Build regex patterns for transportation queries.
        Returns (high_confidence_patterns, normal_patterns)
        
        Supports: English, Turkish, German, French, Arabic
        """
        # =====================================================================
        # HIGH CONFIDENCE PATTERNS (all languages)
        # =====================================================================
        high_confidence = [
            # === ENGLISH ===
            r'\b(directions?|route|routing)\s+(from|to)\b',
            r'\bhow\s+(do|can|could|to)\s+(i|we|you)\s+(get|go|travel|reach|navigate)\s+(from|to)\b',
            r'\bhow\s+(to|can\s+i|do\s+i)\s+(get|go|travel|reach|cross)\b',
            r'\b(show|tell|give|find)\s+(me\s+)?(the\s+)?(way|route|directions?|path)\s+(from|to|between)\b',
            r'\b(need|want)\s+.*\b(directions?|route|way|navigation)\b',
            r'\bi\s+need\s+to\s+(get|go|reach|arrive)\b',
            r'\b(which|what)\s+(metro|tram|bus|ferry|line)\s+(goes|takes|to|from)\b',
            r'\b(does|is)\s+(the\s+)?(metro|tram|bus|ferry|marmaray)\s+(go|run)\b',
            r'\b(airport|ist|saw|sabiha)\s+(to|from|transfer)\b',
            r'\b(to|from)\s+(the\s+)?airport\b',
            r'\b(ferry|vapur)\s+(to|from|schedule)\b',
            r'\bhow\s+long\s+(does\s+it\s+take|will\s+it\s+take|to\s+get)\b',
            r'\b(cross|crossing)\s+(the\s+)?(bosphorus|bosporus|boğaz)\b',  # Cross Bosphorus
            r'\b(asian|european)\s+side\b',  # Asian/European side
            
            # === TURKISH ===
            r'\b(nasıl|nasil)\s+(gid|git|ulaş|var|geç)\w*\b',  # nasıl gidilir, nasıl geçerim
            r'\b(nereden|nereye)\s+(gid|git)\w*\b',  # nereden gidilir
            r"['\u2019]?den\s+.+['\u2019]?[eyae]\s+(nasıl|nasil)",  # X'den Y'e nasıl
            r'\b(yol|rota|güzergah|güzergâh)\s+(tarifi|göster)\b',  # yol tarifi
            r'\b(hangi)\s+(metro|tramvay|otobüs|vapur|hat)\b',  # hangi metro
            r'\b(metro|tramvay|otobüs|vapur|marmaray)\s+(ile|la|le)\s+(nasıl|git|gid)\b',
            r'\b(havalimanı|havaalanı|havalimani|Havalimanı)\w*\b',  # Havalimanı'na
            r'\b(ne\s+kadar)\s+(sürer|süre|zaman)\b',  # ne kadar sürer
            r'\b(ulaşım|ulasim|ulaşmak|ulasmak)\b',
            r'\b(marmaray|metro|tramvay|otobüs|vapur)\b.*(geçiyor|geciyor|uğruyor|ugruyor|duruyor|gidiyor)\s*(mu|mı|mi)?\b',  # X geçiyor mu? / gidiyor mu?
            r'\b(avrupa|asya)\s+(yakası|yakasına|yakasından)\b',  # Avrupa/Asya yakası
            r'\b(beni|bizi)\b.*(götür|getir)\w*',  # Beni X'e götür (Take me to X) - more flexible
            r'\b(aktarmasız|direkt|direk)\b.*(gid|git|ulaş)\w*',  # aktarmasız gidebilir (direct route)
            r'\b(gitmem|gitmek)\s+(lazım|gerek|gerekiyor|istiyorum)\b',  # gitmem lazım (I need to go)
            r'\b(lazım|gerek)\b.*(git|gid|ulaş|var)\b',  # X lazım... gitmek (need to go)
            r'\b(marmaray)\b.*\b(gid|git|var|ulaş|geç)\w*\b',  # Marmaray... gidiyor
            
            # === GERMAN ===
            r'\bwie\s+komme?\s+ich\s+(von|nach|zu|zum|zur)\b',  # wie komme ich
            r'\bwie\s+(erreiche|gelange|überquere)\s+ich\b',  # wie erreiche/überquere ich
            r'\b(weg|route|strecke)\s+(von|nach|zu)\b',
            r'\bvon\s+.+\s+nach\b',  # von X nach Y
            r'\b(welche|welcher)\s+(metro|u-?bahn|s-?bahn|bus|fähre|linie)\b',
            r'\b(fährt|geht)\s+(die|der|das|nach)\s*(metro|u-?bahn|bus|fähre|marmaray)?\b',  # fährt nach, fährt die Metro
            r'\bzum\s+(flughafen|airport)\b',
            r'\bwie\s+lange\s+(dauert|braucht)\b',  # wie lange dauert
            r'\b(bosporus|bosphorus)\s+(überqueren|kreuzen)\b',  # Cross Bosphorus
            r'\bwie\s+weit\s+(ist|sind)\b',  # Wie weit ist es
            r'\bvom\s+.+\s+zum\b',  # vom Hotel zum Hafen
            r'\bwie\s+nehme\s+ich\b',  # wie nehme ich (die Fähre)
            r'\b(welche[sr]?)\s+(linie|verbindung)\s+(fährt|geht|nach)\b',  # welche Linie fährt nach
            r'\b(marmaray|metro|u-?bahn)\s+(nach|fährt|geht)\b',  # Marmaray nach X, Metro fährt
            r'\b(metrolinie|u-?bahnlinie|buslinie)\b',  # compound words
            
            # === FRENCH ===
            r'\bcomment\s+(aller|arriver|me\s+rendre|se\s+rendre|traverser)\s+(à|au|de|du|en|le)\b',
            r'\bcomment\s+(puis-?je|peut-?on)\s+(aller|arriver|rejoindre|traverser)\b',
            r'\b(itinéraire|trajet|chemin)\s+(de|pour|vers)\b',
            r'\bde\s+.+\s+à\b',  # de X à Y
            r'\b(quel|quelle)\s+(métro|metro|tram|bus|ligne|ferry)\b',
            r'\b(est-?ce\s+que|le)\s+(métro|metro|tram|bus)\s+(va|passe)\b',
            r"\b(à|vers)\s+l['\u2019]?aéroport\b",
            r'\bcombien\s+de\s+temps\s+(faut|prend|met)\b',  # combien de temps
            r'\b(traverser|croiser)\s+(le\s+)?(bosphore|bosporus)\b',  # Cross Bosphorus
            r'\b(je\s+dois|il\s+faut)\s+(aller|me\s+rendre|arriver)\b',  # je dois aller (I need to go)
            r'\b(pour\s+aller|pour\s+arriver|pour\s+rejoindre)\s+(à|au)\b',  # pour aller à
            
            # === ARABIC ===
            r'\bكيف\s+(أصل|اصل|أذهب|اذهب|أوصل|اوصل|أعبر|اعبر|آخذ|اخذ)\b',  # كيف أصل/أعبر/آخذ
            r'\b(من|إلى|الى)\s+.+\s+(إلى|الى|من)\b',  # من X إلى Y
            r'\b(طريق|طريقة|مسار)\s+(إلى|الى|من)\b',
            r'\b(أي|اي|ما)\s+(مترو|حافلة|باص|عبارة|خط)\b',  # أي مترو
            r'\b(هل)\s+(المترو|الحافلة|الباص|العبارة)\s+(يذهب|يصل|يمر|تذهب)\b',
            r'\b(مطار|المطار)\b',  # airport
            r'\b(كم|كم من الوقت)\s+(يستغرق|تستغرق|يأخذ)\b',  # كم يستغرق
            r'\b(البوسفور|البسفور)\b',  # Bosphorus
            r'\b(أريد|أود)\s+(الذهاب|أن\s+أذهب|أذهب)\b',  # أريد الذهاب (I want to go)
            r'\b(خذني|خذنا)\s+(إلى|الى)\b',  # خذني إلى (take me to)
            r'\b(العبارة|عبارة)\s+(إلى|الى|من|ل)\b',  # العبارة إلى (ferry to)
            r'\b(كيف)\s+(آخذ|أركب|استخدم)\s+(العبارة|الفيري|المترو)\b',  # كيف آخذ العبارة
        ]
        
        # =====================================================================
        # NORMAL PATTERNS (all languages)
        # =====================================================================
        normal = [
            # === ENGLISH ===
            r'\b(go|get|reach|travel|arrive)\b.*\b(to|from)\b',
            r'\b(how|what).*(way|route|get|go|reach)\b',
            r'\b(directions?|route|routing|way|path)\b',
            r'\b(from .+ to |to .+ from )\b',
            r'\bfrom\b.*\bto\b',
            r'\b(navigate|find my way|get there)\b',
            r'\b(metro|tram|bus|ferry|marmaray|funicular|train)\b.*\b(to|from|station)\b',
            r'\b(best way|quickest way|fastest way|easiest way)\b',
            r'\b(going|headed|heading)\s+to\b',
            r'\b(need|trying|want)\s+to\s+(get|go|reach)\b',
            r'\b(istanbulkart|istanbul\s+card)\b',
            
            # === TURKISH ===
            r'\b(git|gid|var|ulaş)\w*\s+(için|ile)\b',
            r'\b(buradan|şuradan|oradan)\b',
            r'\b(metro|tramvay|otobüs|vapur|feribot|dolmuş|minibüs)\b',
            r'\b(durak|istasyon|iskele|terminal)\b',
            r'\b(aktarma|aktarmalı|bağlantı)\b',
            r'\b(en\s+iyi|en\s+hızlı|en\s+kolay)\s+(yol|rota)\b',
            r'\b(lazım|gerek|gerekiyor)\b.*\b(git|gid|var)\b',
            r'\b(toplu\s+taşıma|toplu\s+ulaşım)\b',
            
            # === GERMAN ===
            r'\b(fahren|gehen|kommen)\s+(nach|zu|von)\b',
            r'\b(metro|u-?bahn|s-?bahn|bus|tram|straßenbahn|fähre)\b',
            r'\b(haltestelle|bahnhof|station|anleger)\b',
            r'\b(umsteigen|umstieg|verbindung|anschluss)\b',
            r'\b(schnellste|beste|einfachste)\s+(weg|route|verbindung)\b',
            r'\b(muss|will|möchte)\s+(nach|zu|zum|zur)\b',
            r'\böffentliche[nr]?\s+verkehrsmittel[n]?\b',  # öffentliche/öffentlichen Verkehrsmittel/Verkehrsmitteln
            r'\bmit\s+(dem\s+)?(bus|bahn|zug|metro|tram)\b',  # mit dem Bus, mit der Bahn
            
            # === FRENCH ===
            r'\b(aller|arriver|partir|venir)\s+(à|au|de|du|en)\b',
            r'\b(métro|metro|tram|tramway|bus|ferry|bateau)\b',
            r'\b(arrêt|station|gare|terminal|quai)\b',
            r'\b(correspondance|changement|connexion)\b',
            r'\b(meilleur|plus\s+rapide|plus\s+facile)\s+(moyen|chemin|trajet)\b',
            r'\b(dois|veux|voudrais)\s+aller\b',
            r'\btransport\s+(en\s+commun|public)\b',
            
            # === ARABIC ===
            r'\b(اذهب|أذهب|اروح|أروح|امشي|أمشي)\b',
            r'\b(مترو|المترو|حافلة|الحافلة|باص|الباص|عبارة|العبارة|قطار|القطار|ترام|الترام)\b',  # Added tram
            r'\b(محطة|موقف|مرسى|مرفأ)\b',
            r'\b(تحويل|تغيير|ربط|انتقال|أقوم\s+بالتحويل|بالتحويل)\b',  # transfer patterns
            r'\b(أفضل|أسرع|أسهل)\s+(طريق|طريقة)\b',
            r'\b(أحتاج|أريد|أود)\s+(أن\s+)?(أصل|أذهب|أوصل)\b',
            r'\b(نقل\s+عام|مواصلات\s+عامة|وسائل\s+النقل\s+العام)\b',  # Added وسائل النقل العام
            r'\b(يتوقف|توقف|يقف)\b',  # stops/stopping
            r'\b(استخدام|استخدم|أستخدم)\b',  # use/using
            r'\b(للوصول|الوصول)\s+(إلى|الى)\b',  # to reach/to arrive at
        ]
        
        # =====================================================================
        # NORMAL PATTERNS (all languages)
        # =====================================================================
        normal = [
            # === ENGLISH ===
            r'\b(go|get|reach|travel|arrive)\b.*\b(to|from)\b',
            r'\b(how|what).*(way|route|get|go|reach)\b',
            r'\b(directions?|route|routing|way|path)\b',
            r'\b(from .+ to |to .+ from )\b',
            r'\bfrom\b.*\bto\b',
            r'\b(navigate|find my way|get there)\b',
            r'\b(metro|tram|bus|ferry|marmaray|funicular|train)\b.*\b(to|from|station)\b',
            r'\b(best way|quickest way|fastest way|easiest way)\b',
            r'\b(going|headed|heading)\s+to\b',
            r'\b(need|trying|want)\s+to\s+(get|go|reach)\b',
            r'\b(istanbulkart|istanbul\s+card)\b',
            
            # === TURKISH ===
            r'\b(git|gid|var|ulaş)\w*\s+(için|ile)\b',
            r'\b(buradan|şuradan|oradan)\b',
            r'\b(metro|tramvay|otobüs|vapur|feribot|dolmuş|minibüs)\b',
            r'\b(durak|istasyon|iskele|terminal)\b',
            r'\b(aktarma|aktarmalı|bağlantı)\b',
            r'\b(en\s+iyi|en\s+hızlı|en\s+kolay)\s+(yol|rota)\b',
            r'\b(lazım|gerek|gerekiyor)\b.*\b(git|gid|var)\b',
            r'\b(toplu\s+taşıma|toplu\s+ulaşım)\b',
            
            # === GERMAN ===
            r'\b(fahren|gehen|kommen)\s+(nach|zu|von)\b',
            r'\b(metro|u-?bahn|s-?bahn|bus|tram|straßenbahn|fähre)\b',
            r'\b(haltestelle|bahnhof|station|anleger)\b',
            r'\b(umsteigen|umstieg|verbindung|anschluss)\b',
            r'\b(schnellste|beste|einfachste)\s+(weg|route|verbindung)\b',
            r'\b(muss|will|möchte)\s+(nach|zu|zum|zur)\b',
            r'\böffentliche[nr]?\s+verkehrsmittel[n]?\b',  # öffentliche/öffentlichen Verkehrsmittel/Verkehrsmitteln
            r'\bmit\s+(dem\s+)?(bus|bahn|zug|metro|tram)\b',  # mit dem Bus, mit der Bahn
            
            # === FRENCH ===
            r'\b(aller|arriver|partir|venir)\s+(à|au|de|du|en)\b',
            r'\b(métro|metro|tram|tramway|bus|ferry|bateau)\b',
            r'\b(arrêt|station|gare|terminal|quai)\b',
            r'\b(correspondance|changement|connexion)\b',
            r'\b(meilleur|plus\s+rapide|plus\s+facile)\s+(moyen|chemin|trajet)\b',
            r'\b(dois|veux|voudrais)\s+aller\b',
            r'\btransport\s+(en\s+commun|public)\b',
            
            # === ARABIC ===
            r'\b(اذهب|أذهب|اروح|أروح|امشي|أمشي)\b',
            r'\b(مترو|المترو|حافلة|الحافلة|باص|الباص|عبارة|العبارة|قطار|القطار|ترام|الترام)\b',  # Added tram
            r'\b(محطة|موقف|مرسى|مرفأ)\b',
            r'\b(تحويل|تغيير|ربط|انتقال|أقوم\s+بالتحويل|بالتحويل)\b',  # transfer patterns
            r'\b(أفضل|أسرع|أسهل)\s+(طريق|طريقة)\b',
            r'\b(أحتاج|أريد|أود)\s+(أن\s+)?(أصل|أذهب|أوصل)\b',
            r'\b(نقل\s+عام|مواصلات\s+عامة|وسائل\s+النقل\s+العام)\b',  # Added وسائل النقل العام
            r'\b(يتوقف|توقف|يقف)\b',  # stops/stopping
            r'\b(استخدام|استخدم|أستخدم)\b',  # use/using
            r'\b(للوصول|الوصول)\s+(إلى|الى)\b',  # to reach/to arrive at
        ]
        
        return high_confidence, normal
    
    def _build_negative_patterns(self) -> list:
        """
        Build regex patterns that indicate non-transportation queries.
        These patterns REDUCE confidence to prevent false positives.
        
        Key categories:
        1. Price/cost queries about transit (not directions)
        2. Info queries about transit (hours, safety, etc.)
        3. "What to see/do" queries even if near transport hubs
        4. Restaurant/food queries near transport locations
        """
        return [
            # === PRICE/COST QUERIES (not directions) ===
            r'\b(price|prices|cost|costs|fee|fees|fare|fares|ticket)\b',
            r'\b(how\s+much|what.*cost|what.*price)\b',
            r'\b(expensive|cheap|affordable|budget)\b.*\b(metro|tram|bus|ferry|taxi|istanbulkart)\b',
            r'\b(istanbulkart|istanbul\s*card)\s*(price|cost|fee|buy|purchase|where)\b',
            r'\b(buy|purchase|get|refill|top\s*up)\s*(istanbulkart|istanbul\s*card|ticket)\b',
            
            # === INFO QUERIES ABOUT TRANSIT (not directions) ===
            r'\b(what\s+time|when)\s+(does|do|is)\s+(the\s+)?(metro|tram|bus|ferry)\s+(open|close|start|stop|run)\b',
            r'\b(opening|closing)\s*(hours|time)\b.*\b(metro|tram|bus|ferry|station)\b',
            r'\b(metro|tram|bus|ferry)\s*(opening|closing|hours)\b',  # Removed "schedule" - schedule with from/to is routing
            r'\b(is|are)\s+(the\s+)?(metro|tram|bus|ferry|ferries)\s+(safe|crowded|busy|reliable|clean)\b',
            r'\b(how\s+)(safe|crowded|busy|reliable)\s+(is|are)\b',
            
            # === "WHAT TO SEE/DO" QUERIES (sightseeing, not directions) ===
            r'\b(what\s+to\s+)(see|do|visit|explore|try|eat|drink)\b',
            r'\b(things\s+to\s+)(see|do|visit|explore|try|eat)\b',
            r'\b(places\s+to\s+)(see|visit|explore|eat|drink)\b',
            r'\b(attractions?|sights?|landmarks?|museums?)\s+(near|around|at|in)\b',
            r'\b(restaurants?|cafes?|bars?|food|eat|dining)\s+(near|around|at|in)\b',
            
            # === GENERAL INFO QUERIES ===
            r'\b(tell\s+me\s+about|what\s+is|history\s+of|famous\s+for)\b',
            r'\b(shopping|nightlife|culture|traditions?)\s+(in|at|near)\b',
            
            # === EXPLICIT NON-TRANSPORT INTENTS ===
            r'\b(best|good|popular|recommended)\s+(restaurants?|cafes?|hotels?|shops?)\b',
            r'\b(where\s+can\s+i\s+)(eat|drink|shop|stay|sleep)\b',
            r'\b(recommend|suggest)\s+(a|an)?\s*(hotel|restaurant|cafe|bar|place\s+to\s+stay)\b',
            r'\b(hotel|hostel|accommodation)\s+(near|around|in|at|recommendation)\b',
            r'\b(is|are)\s+.*\s+(safe|dangerous|secure)\s+(for|to)\b',  # Is X safe for tourists
            r'\bsafe\s+for\s+(tourists?|travelers?|visitors?)\b',
            
            # === RESTAURANT/FOOD QUERIES ===
            r'\b(restaurant|restaurants|cafe|cafes|bar|bars)\s+(in|near|at|around)\b',
            r'\b(best|good|top|rated)\s+.*(restaurant|cafe|food|kebab|fish|breakfast)\b',
            r'\b(kebab|fish|seafood|breakfast|lunch|dinner)\s+(restaurant|place|spot)\b',
            r'\b(where\s+is|where\s+are)\s+(the\s+)?(best|good)\b',  # "Where is the best X"
            r'\b(top-?rated|highest-?rated)\s+.*(places?|restaurants?|cafes?)\b',
            r'\b(looking\s+for)\s+(a|an)?\s*(restaurant|cafe|place\s+to\s+eat)\b',
            r'\bplace\s+to\s+(eat|dine|drink)\b',
            
            # === TURKISH FOOD QUERIES ===
            r'\b(lokanta|restoran|kafe|kahvalti|yemek)\b.*\b(nerede|nereye|onerir)\b',
            r'\b(en\s+iyi|guzel)\s+(lokanta|restoran|balik|kahvalti)\b',
            r'\b(yemek\s+yiyebil|kahvalti\s+icin)\b',
            
            # === GERMAN FOOD/RESTAURANT QUERIES ===
            r'\b(restaurant|restaurants|café|cafés|lokal|gaststätte)\s+(in|bei|nahe|um)\b',
            r'\b(beste[sr]?|gute[sr]?)\s+.*(restaurant|essen|küche|lokal)\b',
            r'\b(fischrestaurant|türkisches\s+essen|kebab|döner)\b',
            r'\b(wo\s+kann\s+ich\s+)(essen|frühstücken|speisen)\b',
            r'\b(empfehlen|empfehlung)\s+.*(restaurant|lokal|café)\b',
            r'\b(frühstück|mittagessen|abendessen)\s+(in|bei|nahe)\b',
            
            # === FRENCH FOOD/RESTAURANT QUERIES ===
            r'\b(restaurant|restaurants|café|cafés|bistrot|brasserie)\s+(à|au|près|dans)\b',
            r'\b(meilleur|bon|bonne)\s+.*(restaurant|cuisine|repas)\b',
            r'\b(où\s+(manger|déjeuner|dîner|trouver.*restaurant))\b',
            r'\b(recommander|suggestion)\s+.*(restaurant|café)\b',
            r'\b(petit[\s-]*déjeuner|déjeuner|عشاء)\s+(à|au|près)\b',
            r'\b(fruits?\s+de\s+mer|poisson|kebab)\b',
            
            # === ARABIC FOOD/RESTAURANT QUERIES ===
            r'\b(مطعم|مطاعم|مقهى|كافيه)\s+(في|بالقرب|حول)\b',
            r'\b(أفضل|أحسن|جيد)\s+.*(مطعم|طعام|أكل|مأكولات)\b',
            r'\b(أين\s+(آكل|أتناول|أجد.*مطعم))\b',
            r'\b(توصية|اقتراح)\s+.*(مطعم|مقهى)\b',
            r'\b(فطور|غداء|عشاء)\s+(في|بالقرب)\b',
            r'\b(كباب|سمك|مأكولات\s+بحرية)\b',
        ]
    
    def _build_semantic_examples(self) -> list:
        """
        Build example queries that represent transportation intent.
        Used for semantic similarity matching.
        """
        return [
            # Explicit routing queries
            "Give me directions from place A to place B",
            "I need directions to a location",
            "Show me the route from one place to another",
            "How do I get from here to there",
            "What's the way from point A to point B",
            
            # General navigation
            "I want to get somewhere",
            "How do I reach a place",
            "Help me find my way",
            "I need to go somewhere",
            "What's the best route",
            "How can I travel from one place to another",
            "I'm trying to get to a destination",
            "Show me how to reach my destination",
            "I need navigation help",
            "How do I get there",
            "Help me cross to another area",
            "I'm lost and need directions",
            "Need to arrive at a place",
            
            # === ENHANCED: Transit-specific ===
            "Which metro line should I take",
            "What bus goes to this place",
            "Does the ferry go there",
            "How to take the tram",
            "Which line goes to the airport",
            "Is there a direct metro",
            
            # === ENHANCED: Time-based ===
            "How long does it take to get there",
            "What's the fastest way",
            "Night bus options",
            "Early morning transport",
            "Last ferry schedule",
            
            # === ENHANCED: Walking ===
            "Can I walk there",
            "Is it walking distance",
            "Walking directions please",
            "How far is it by foot",
            
            # === ENHANCED: Airport ===
            "How to get to the airport",
            "Airport transfer options",
            "Transport from airport to city",
            "Fastest way to airport",
            
            # === ENHANCED: Ferry/Islands ===
            "How to reach the islands",
            "Ferry to the princes islands",
            "Best way to get to Buyukada",
            "Boat schedule to islands",
            
            # === ENHANCED: Informal/conversational ===
            "Need to get to Kadikoy",
            "Going to the Blue Mosque, how should I travel",
            "What's the easiest way to reach Ortakoy",
            "Trying to find my way to Taksim",
            
            # === ENHANCED: Istanbulkart ===
            "Where can I buy Istanbulkart",
            "How to use the transit card",
            "Is the card valid on ferries",
        ]
    
    def _build_location_keywords(self) -> set:
        """Keywords that suggest geographic/location context"""
        return {
            # Istanbul neighborhoods (expanded)
            'taksim', 'kadıköy', 'kadikoy', 'beşiktaş', 'besiktas', 'sultanahmet',
            'eminönü', 'eminonu', 'üsküdar', 'uskudar', 'beyoğlu', 'beyoglu',
            'galata', 'karaköy', 'karakoy', 'levent', 'şişli', 'sisli',
            'ortaköy', 'ortakoy', 'balat', 'fatih', 'mecidiyeköy', 'mecidiyekoy',
            'şişhane', 'sishane', 'kabataş', 'kabatas', 'beykoz', 'sarıyer',
            
            # Major landmarks
            'hagia sophia', 'ayasofya', 'blue mosque', 'topkapi', 'topkapı',
            'dolmabahçe', 'dolmabahce', 'grand bazaar', 'kapalıçarşı', 'spice market',
            'galata tower', 'maiden tower', 'istiklal', 'bosphorus bridge',
            
            # Transport modes
            'metro', 'tram', 'bus', 'ferry', 'marmaray', 'funicular', 'train',
            'station', 'stop', 'terminal', 'pier', 'line', 'vapur', 'dolmus',
            'istanbulkart', 'minibus', 'taxi',
            
            # Airports
            'airport', 'ist', 'saw', 'sabiha', 'havalimanı', 'havalimani',
            
            # Geographic terms (expanded)
            'side', 'shore', 'coast', 'european', 'asian', 'bosphorus',
            'square', 'street', 'district', 'area', 'neighborhood',
            'near', 'close', 'far', 'distance', 'meydan', 'caddesi',
            
            # Islands
            'princes', 'islands', 'adalar', 'buyukada', 'büyükada',
            'heybeliada', 'kinaliada', 'burgazada',
            
            # Piers
            'eminonu pier', 'kabatas pier', 'karakoy pier', 'besiktas pier',
        }
    
    def classify_intent(
        self,
        query: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, any]]:
        """
        Classify transportation intent with confidence score.
        
        Args:
            query: User's query text
            user_location: Optional GPS coordinates
            
        Returns:
            Tuple of (confidence_score, debug_info)
            
        Confidence breakdown:
            0.0-0.4: Unlikely transportation query
            0.4-0.6: Uncertain, may need clarification
            0.6-0.8: Likely transportation query
            0.8-1.0: Highly confident transportation query
        """
        query_lower = query.lower().strip()
        
        # Track confidence contributions
        confidence = 0.0
        debug_info = {
            'query': query,
            'regex_hit': False,
            'semantic_score': 0.0,
            'location_mentions': 0,
            'has_gps': user_location is not None,
            'signals': []
        }
        
        # Signal 1: Regex pattern matching (using pre-compiled patterns)
        regex_hit = False
        high_confidence_hit = False
        
        # Check high-confidence patterns first (+0.6)
        for pattern in self.high_confidence_patterns:
            if pattern.search(query_lower):
                regex_hit = True
                high_confidence_hit = True
                confidence += 0.6
                debug_info['regex_hit'] = True
                debug_info['high_confidence_pattern'] = True
                debug_info['signals'].append('high_confidence_regex')
                logger.debug(f"✅ High-confidence regex hit: {pattern.pattern[:50]}...")
                break
        
        # If no high-confidence hit, check normal patterns (+0.4)
        if not high_confidence_hit:
            for pattern in self.normal_patterns:
                if pattern.search(query_lower):
                    regex_hit = True
                    confidence += 0.4
                    debug_info['regex_hit'] = True
                    debug_info['signals'].append('regex_pattern_match')
                    logger.debug(f"✅ Normal regex hit: {pattern.pattern[:50]}...")
                    break
        
        # Signal 2: Semantic similarity (+0.4 if high similarity)
        semantic_score = 0.0
        used_embeddings = False
        
        if self.embedding_service and not self.embedding_service.offline_mode:
            try:
                # Get query embedding
                query_embedding = self.embedding_service.encode(query)
                
                if query_embedding is not None:
                    used_embeddings = True
                    max_similarity = 0.0
                    
                    # Use cached embeddings if available (much faster)
                    if self._example_embeddings is not None:
                        for example, example_embedding in zip(self.semantic_examples, self._example_embeddings):
                            similarity = self._cosine_similarity(query_embedding, example_embedding)
                            max_similarity = max(max_similarity, similarity)
                    else:
                        # Fallback: compute on-the-fly (slower)
                        for example in self.semantic_examples:
                            example_embedding = self.embedding_service.encode(example)
                            if example_embedding is not None:
                                similarity = self._cosine_similarity(query_embedding, example_embedding)
                                max_similarity = max(max_similarity, similarity)
                    
                    semantic_score = max_similarity
                    debug_info['semantic_score'] = round(semantic_score, 3)
                    
                    # Add confidence based on semantic score (more lenient thresholds)
                    if semantic_score > 0.60:  # Lowered from 0.75
                        confidence += 0.4
                        debug_info['signals'].append('high_semantic_similarity')
                    elif semantic_score > 0.50:  # Lowered from 0.65
                        confidence += 0.3
                        debug_info['signals'].append('medium_semantic_similarity')
                    elif semantic_score > 0.40:  # Lowered from 0.55
                        confidence += 0.2
                        debug_info['signals'].append('low_semantic_similarity')
                    
                    logger.debug(f"🔍 Semantic score: {semantic_score:.3f}")
                
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {e}")
        
        # Fallback: keyword-based semantic matching (if embeddings not used)
        if not used_embeddings:
            transport_keywords = {
                'how', 'way', 'route', 'get', 'go', 'reach', 'travel', 'arrive',
                'directions', 'navigate', 'from', 'to', 'take', 'guide', 'show'
            }
            query_words = set(query_lower.split())
            overlap = len(query_words.intersection(transport_keywords))
            
            if overlap >= 2:
                confidence += 0.3
                debug_info['signals'].append('keyword_overlap_fallback')
                debug_info['keyword_overlap'] = overlap
                logger.debug(f"🔍 Keyword overlap: {overlap} words")
        
        # Signal 3: Location mentions (+0.2)
        location_mentions = 0
        for keyword in self.location_keywords:
            if keyword in query_lower:
                location_mentions += 1
        
        if location_mentions > 0:
            confidence += min(0.2, location_mentions * 0.1)
            debug_info['location_mentions'] = location_mentions
            debug_info['signals'].append(f'{location_mentions}_location_keywords')
            logger.debug(f"📍 Found {location_mentions} location keywords")
        
        # Signal 4: GPS presence (minor boost +0.1)
        if user_location:
            confidence += 0.1
            debug_info['signals'].append('gps_available')
            logger.debug(f"📍 GPS coordinates available")
        
        # === NEGATIVE SIGNAL: Check for false positive patterns ===
        # This REDUCES confidence if query looks like non-transport intent
        negative_hit = False
        for pattern in self.negative_patterns:
            if pattern.search(query_lower):
                negative_hit = True
                # Strong penalty for clear non-transport queries
                confidence_penalty = 0.5
                confidence = max(0.0, confidence - confidence_penalty)
                debug_info['negative_pattern_hit'] = True
                debug_info['signals'].append('negative_pattern_penalty')
                logger.debug(f"❌ Negative pattern hit: {pattern.pattern[:50]}... (penalty: -{confidence_penalty})")
                break
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        debug_info['final_confidence'] = round(confidence, 3)
        
        logger.info(
            f"🎯 Intent classification: confidence={confidence:.3f}, "
            f"regex={regex_hit}, semantic={semantic_score:.3f}, "
            f"locations={location_mentions}, gps={user_location is not None}"
        )
        
        return confidence, debug_info
    
    async def clarify_with_llm(
        self,
        query: str,
        llm_service
    ) -> bool:
        """
        Use LLM to clarify intent when confidence is uncertain (0.4-0.6).
        
        This is CHEAP and FAST:
        - max_tokens: 3
        - temperature: 0
        - Expected latency: ~300ms
        
        Only called when needed, not for every query.
        """
        try:
            prompt = f"""Does this user want directions or navigation help?
Query: "{query}"

Answer only YES or NO."""

            response = await llm_service.generate(
                prompt=prompt,
                max_tokens=3,
                temperature=0
            )
            
            answer = response.strip().upper()
            is_transportation = 'YES' in answer
            
            logger.info(f"🤖 LLM clarification: '{query[:50]}...' → {answer}")
            
            return is_transportation
            
        except Exception as e:
            logger.error(f"LLM clarification failed: {e}")
            # Fail safe: assume not transportation
            return False
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def analyze_query(self, query: str) -> Dict[str, any]:
        """
        Provide detailed analysis of why a query was classified a certain way.
        Useful for debugging and improving patterns.
        
        Args:
            query: The query to analyze
            
        Returns:
            Detailed breakdown including:
            - All matching patterns (positive and negative)
            - Top semantic matches
            - Location keywords found
            - Confidence breakdown
        """
        query_lower = query.lower().strip()
        
        analysis = {
            'query': query,
            'matching_high_confidence': [],
            'matching_normal': [],
            'matching_negative': [],
            'location_keywords_found': [],
            'semantic_top_3': [],
            'recommendation': '',
        }
        
        # Find all matching high-confidence patterns
        for pattern in self.high_confidence_patterns:
            match = pattern.search(query_lower)
            if match:
                analysis['matching_high_confidence'].append({
                    'pattern': pattern.pattern[:80],
                    'matched_text': match.group()
                })
        
        # Find all matching normal patterns
        for pattern in self.normal_patterns:
            match = pattern.search(query_lower)
            if match:
                analysis['matching_normal'].append({
                    'pattern': pattern.pattern[:80],
                    'matched_text': match.group()
                })
        
        # Find all matching negative patterns
        for pattern in self.negative_patterns:
            match = pattern.search(query_lower)
            if match:
                analysis['matching_negative'].append({
                    'pattern': pattern.pattern[:80],
                    'matched_text': match.group()
                })
        
        # Find location keywords
        for keyword in self.location_keywords:
            if keyword in query_lower:
                analysis['location_keywords_found'].append(keyword)
        
        # Get semantic similarities if available
        if self.embedding_service and not self.embedding_service.offline_mode:
            try:
                query_embedding = self.embedding_service.encode(query)
                if query_embedding is not None:
                    similarities = []
                    example_embeddings = self._get_example_embeddings()
                    if example_embeddings:
                        for example, emb in zip(self.semantic_examples, example_embeddings):
                            sim = self._cosine_similarity(query_embedding, emb)
                            similarities.append((example, sim))
                        
                        # Get top 3
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        analysis['semantic_top_3'] = [
                            {'example': ex, 'similarity': round(sim, 3)}
                            for ex, sim in similarities[:3]
                        ]
            except Exception as e:
                analysis['semantic_error'] = str(e)
        
        # Run classification and add result
        confidence, debug_info = self.classify_intent(query)
        analysis['confidence'] = confidence
        analysis['debug_info'] = debug_info
        
        # Provide recommendation
        if confidence < self.UNCERTAIN_LOW:
            analysis['recommendation'] = 'NOT transportation intent'
        elif confidence < self.TRANSPORT_THRESHOLD:
            analysis['recommendation'] = 'UNCERTAIN - may need LLM clarification'
        elif confidence < self.UNCERTAIN_HIGH:
            analysis['recommendation'] = 'LIKELY transportation intent'
        else:
            analysis['recommendation'] = 'HIGH CONFIDENCE transportation intent'
        
        return analysis
    
    def is_transport_intent(self, query: str, threshold: float = None) -> bool:
        """
        Simple boolean check for transport intent.
        
        Args:
            query: The query to check
            threshold: Custom threshold (default uses TRANSPORT_THRESHOLD)
            
        Returns:
            True if query is classified as transportation intent
        """
        threshold = threshold or self.TRANSPORT_THRESHOLD
        confidence, _ = self.classify_intent(query)
        return confidence >= threshold


# Singleton instance
_intent_classifier = None

def get_transportation_intent_classifier() -> TransportationIntentClassifier:
    """Get or create intent classifier singleton"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = TransportationIntentClassifier()
    return _intent_classifier

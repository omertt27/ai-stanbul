#!/usr/bin/env python3
"""
Enhanced Query Understanding System for AI Istanbul
==================================================

Improved query preprocessing, intent understanding, and semantic expansion
without GPT dependencies. Uses rule-based NLP, fuzzy matching, and embeddings.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import difflib
from datetime import datetime

# Turkish character normalization
TURKISH_CHAR_MAP = {
    'ğ': 'g', 'Ğ': 'G', 'ü': 'u', 'Ü': 'U', 
    'ş': 's', 'Ş': 'S', 'ı': 'i', 'İ': 'I',
    'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C'
}

@dataclass
class ParsedQuery:
    """Structured query representation"""
    original_query: str
    normalized_query: str
    intent: str
    entities: Dict[str, Any]
    confidence: float
    corrections: List[str]
    temporal_context: Optional[str] = None
    vibe_tags: List[str] = None

class TurkishSpellCorrector:
    """Turkish-aware spell correction for Istanbul locations and terms"""
    
    def __init__(self):
        self.istanbul_terms = {
            # Districts (with common misspellings)
            'sultanahmet': ['sultanahemt', 'sultanahmed', 'sultanamet'],
            'beyoğlu': ['beyoglu', 'beyogul', 'beyoğul'],
            'kadıköy': ['kadikoy', 'kadıköyy', 'kadiköy', 'kadikköy'],
            'beşiktaş': ['besiktas', 'beşiktas', 'besiktaş'],
            'üsküdar': ['uskudar', 'üskudar', 'uskküdar'],
            'fatih': ['fatıh', 'fatiih'],
            'sarıyer': ['sariyer', 'sarıyyer'],
            'şişli': ['sisli', 'şişlı', 'sislli'],
            'bakırköy': ['bakirkoy', 'bakırköyy'],
            'eminönü': ['eminonu', 'eminönüü'],
            
            # Landmarks & Attractions
            'galata': ['gallata', 'galatha'],
            'bosphorus': ['bosforus', 'bosporus', 'bosfor'],
            'hagia': ['aya', 'ayya', 'hağia'],
            'sophia': ['sofya', 'sofıa', 'soffia'],
            'topkapi': ['topkapı', 'topkapi', 'topkapii'],
            'basilica': ['basilika', 'basılıca'],
            'cistern': ['sarnıç', 'sarnic'],
            'dolmabahce': ['dolmabahçe', 'dolmabahche'],
            'taksim': ['takksim', 'taksimm'],
            'istiklal': ['istıklal', 'istiklall'],
            
            # Food terms
            'kebab': ['kebap', 'kebabb', 'kebabı'],
            'döner': ['doner', 'donner', 'dönerr'],
            'meze': ['mezze', 'mezee'],
            'baklava': ['baklawa', 'bakllava'],
            'turkish': ['turkısh', 'türkish', 'turkissh'],
            
            # Common terms
            'restaurant': ['restoran', 'restorant', 'restaurrant'],
            'museum': ['müze', 'musem', 'museumm'],
            'coffee': ['kahve', 'coffe', 'cofee'],
            'seafood': ['seefood', 'sea food', 'seafood'],
            
            # Transport terms
            'metro': ['metroo', 'mettro'],
            'metrobus': ['metrobüs', 'metrobus'],
            'ferry': ['fery', 'ferri', 'vapur'],
            'dolmus': ['dolmuş', 'dolmush'],
            'tram': ['tramm', 'tramway'],
            
            # Shopping & Entertainment
            'bazaar': ['bazar', 'pazar', 'çarşı'],
            'shopping': ['shooping', 'alışveriş'],
            'nightlife': ['nightlıfe', 'gece hayatı'],
            'hamam': ['hammam', 'turkish bath'],
            
            # Nature & Areas
            'neighborhood': ['neighbourhood', 'mahalle'],
            'district': ['bölge', 'semt'],
            'garden': ['bahçe', 'gardin'],
            'park': ['parkk', 'parq'],
        }
        
        # Build reverse lookup
        self.corrections = {}
        for correct, variants in self.istanbul_terms.items():
            for variant in variants:
                self.corrections[variant.lower()] = correct
    
    def correct_text(self, text: str) -> Tuple[str, List[str]]:
        """Correct Turkish/Istanbul-specific terms"""
        corrected = text.lower()
        corrections = []
        
        # Apply character normalization
        for tr_char, en_char in TURKISH_CHAR_MAP.items():
            if tr_char in corrected:
                corrected = corrected.replace(tr_char, en_char)
        
        # Apply Istanbul-specific corrections
        words = corrected.split()
        corrected_words = []
        
        for word in words:
            if word in self.corrections:
                corrected_word = self.corrections[word]
                corrected_words.append(corrected_word)
                corrections.append(f"{word} → {corrected_word}")
            else:
                # Try fuzzy matching for close misspellings
                matches = difflib.get_close_matches(word, self.corrections.keys(), n=1, cutoff=0.8)
                if matches:
                    corrected_word = self.corrections[matches[0]]
                    corrected_words.append(corrected_word)
                    corrections.append(f"{word} → {corrected_word}")
                else:
                    corrected_words.append(word)
        
        return ' '.join(corrected_words), corrections

class EntityExtractor:
    """Extract entities from queries without ML"""
    
    def __init__(self):
        self.districts = [
            'sultanahmet', 'beyoğlu', 'kadıköy', 'beşiktaş', 'üsküdar',
            'fatih', 'sarıyer', 'şişli', 'bakırköy', 'eminönü'
        ]
        
        self.categories = {
            'restaurant': ['restaurant', 'restoran', 'food', 'dining', 'eat', 'meal', 'dinner', 'lunch'],
            'cafe': ['cafe', 'kahve', 'coffee', 'coffeehouse', 'tea', 'çay'],
            'museum': ['museum', 'müze', 'gallery', 'art', 'exhibition', 'sergi'],
            'attraction': ['attraction', 'place', 'site', 'landmark', 'tourist', 'visit', 'see'],
            'transport': ['transport', 'metro', 'bus', 'ferry', 'taxi', 'dolmuş', 'tram'],
            'neighborhood': ['neighborhood', 'district', 'area', 'mahalle', 'semt', 'quarter'],
            'hotel': ['hotel', 'accommodation', 'stay', 'konaklama', 'otel', 'lodge'],
            'shopping': ['shopping', 'shop', 'store', 'market', 'bazaar', 'pazar', 'alışveriş'],
            'entertainment': ['entertainment', 'fun', 'nightlife', 'bar', 'club', 'eğlence'],
            'cultural': ['cultural', 'culture', 'traditional', 'heritage', 'kültür', 'tarihi'],
            'nature': ['nature', 'park', 'garden', 'outdoor', 'doğa', 'yeşil alan'],
            'wellness': ['spa', 'hamam', 'wellness', 'massage', 'health', 'sağlık'],
            'business': ['business', 'work', 'meeting', 'conference', 'iş', 'toplantı']
        }
        
        self.cuisines = [
            'turkish', 'ottoman', 'italian', 'japanese', 'seafood', 
            'vegetarian', 'vegan', 'kebab', 'meze', 'mediterranean',
            'chinese', 'indian', 'french', 'american', 'korean',
            'mexican', 'thai', 'lebanese', 'greek', 'persian'
        ]
        
        self.transport_modes = [
            'metro', 'metrobus', 'bus', 'ferry', 'tram', 'dolmuş', 
            'taxi', 'uber', 'walking', 'bicycle', 'car'
        ]
        
        self.attraction_types = [
            'mosque', 'museum', 'palace', 'tower', 'bridge', 'park',
            'market', 'bazaar', 'gallery', 'monument', 'square',
            'waterfront', 'viewpoint', 'historic site'
        ]
        
        self.temporal_terms = {
            'now': ['now', 'currently', 'şimdi'],
            'today': ['today', 'bugün'],
            'tonight': ['tonight', 'this evening', 'bu akşam'],
            'tomorrow': ['tomorrow', 'yarın'],
            'weekend': ['weekend', 'hafta sonu'],
            'morning': ['morning', 'sabah'],
            'afternoon': ['afternoon', 'öğleden sonra'],
            'evening': ['evening', 'akşam']
        }
        
        self.vibe_terms = {
            'romantic': ['romantic', 'romantik', 'intimate', 'cozy'],
            'quiet': ['quiet', 'peaceful', 'sakin', 'huzurlu'],
            'busy': ['busy', 'crowded', 'lively', 'hareketli'],
            'historic': ['historic', 'historical', 'old', 'ancient', 'tarihi'],
            'modern': ['modern', 'contemporary', 'new', 'çağdaş'],
            'authentic': ['authentic', 'traditional', 'local', 'gerçek'],
            'hidden': ['hidden', 'secret', 'gizli', 'off-beaten'],
            'luxury': ['luxury', 'upscale', 'expensive', 'lüks'],
            'budget': ['cheap', 'budget', 'affordable', 'ucuz']
        }
    
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract structured entities from query"""
        query_lower = query.lower()
        entities = {
            'districts': [],
            'categories': [],
            'cuisines': [],
            'transport_modes': [],
            'attraction_types': [],
            'temporal': None,
            'vibes': [],
            'budget': None,
            'group_size': None,
            'duration': None
        }
        
        # Extract districts
        for district in self.districts:
            if district in query_lower:
                entities['districts'].append(district)
        
        # Extract categories
        for category, keywords in self.categories.items():
            if any(keyword in query_lower for keyword in keywords):
                entities['categories'].append(category)
        
        # Extract cuisines
        for cuisine in self.cuisines:
            if cuisine in query_lower:
                entities['cuisines'].append(cuisine)
        
        # Extract transport modes
        for transport in self.transport_modes:
            if transport in query_lower:
                entities['transport_modes'].append(transport)
        
        # Extract attraction types
        for attraction_type in self.attraction_types:
            if attraction_type in query_lower:
                entities['attraction_types'].append(attraction_type)
        
        # Extract temporal context
        for time_type, keywords in self.temporal_terms.items():
            if any(keyword in query_lower for keyword in keywords):
                entities['temporal'] = time_type
                break
        
        # Extract vibes
        for vibe, keywords in self.vibe_terms.items():
            if any(keyword in query_lower for keyword in keywords):
                entities['vibes'].append(vibe)
        
        # Extract budget hints
        if any(word in query_lower for word in ['cheap', 'budget', 'affordable', 'ucuz']):
            entities['budget'] = 'budget'
        elif any(word in query_lower for word in ['expensive', 'luxury', 'upscale', 'lüks']):
            entities['budget'] = 'luxury'
        elif any(word in query_lower for word in ['moderate', 'normal', 'orta']):
            entities['budget'] = 'moderate'
        
        # Extract group size
        import re
        group_patterns = [
            (r'(\d+)\s*(people|person|kişi)', 'specific'),
            (r'(couple|two|iki)', 'couple'),
            (r'(family|children|kids|çocuk)', 'family'),
            (r'(group|friends|arkadaş)', 'group'),
            (r'(solo|alone|tek)', 'solo')
        ]
        
        for pattern, group_type in group_patterns:
            if re.search(pattern, query_lower):
                entities['group_size'] = group_type
                break
        
        # Extract duration
        duration_patterns = [
            (r'(\d+)\s*(hour|hours|saat)', 'hours'),
            (r'(\d+)\s*(day|days|gün)', 'days'),
            (r'(quick|fast|hızlı)', 'short'),
            (r'(long|extended|uzun)', 'long'),
            (r'(half.*day|yarım.*gün)', 'half_day'),
            (r'(full.*day|tam.*gün)', 'full_day')
        ]
        
        for pattern, duration_type in duration_patterns:
            if re.search(pattern, query_lower):
                entities['duration'] = duration_type
                break
        
        return entities

class IntentClassifier:
    """Rule-based intent classification for Istanbul queries"""
    
    def __init__(self):
        self.intent_patterns = {
            # FOOD & DINING
            'find_restaurant': {
                'keywords': ['restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal', 'dinner', 'lunch'],
                'patterns': [
                    r'(find|show|recommend|suggest).*(restaurant|food|place to eat)',
                    r'(where|what).*(eat|food|restaurant)',
                    r'(best|good|great).*(restaurant|food|cuisine)',
                    r'(hungry|want to eat)'
                ]
            },
            'find_cafe': {
                'keywords': ['cafe', 'coffee', 'kahve', 'tea', 'çay'],
                'patterns': [
                    r'(coffee|cafe|kahve|tea|çay)',
                    r'(find|show).*(coffee|cafe)',
                    r'(need|want).*(coffee|caffeine|tea)'
                ]
            },
            
            # ATTRACTIONS & CULTURE
            'find_attraction': {
                'keywords': ['museum', 'attraction', 'place', 'visit', 'see', 'landmark', 'monument'],
                'patterns': [
                    r'(what|where).*(see|visit|go)',
                    r'(show|find).*(museum|attraction|place)',
                    r'(tourist|sightseeing)',
                    r'(historical|historic|cultural)'
                ]
            },
            'find_museum': {
                'keywords': ['museum', 'müze', 'gallery', 'art', 'exhibition', 'sergi'],
                'patterns': [
                    r'(museum|müze|gallery|art)',
                    r'(find|show).*(museum|gallery)',
                    r'(exhibition|art|painting|sculpture)'
                ]
            },
            'cultural_experience': {
                'keywords': ['cultural', 'traditional', 'authentic', 'heritage', 'kültür', 'tarihi'],
                'patterns': [
                    r'(cultural|traditional|authentic)',
                    r'(heritage|history|tarihi)',
                    r'(local.*experience|authentic.*life)'
                ]
            },
            
            # NEIGHBORHOODS & AREAS
            'explore_neighborhood': {
                'keywords': ['neighborhood', 'district', 'area', 'mahalle', 'semt', 'quarter', 'explore'],
                'patterns': [
                    r'(explore|walk|wander)',
                    r'(area|neighborhood|district|mahalle)',
                    r'(what.*around|near)',
                    r'(vibe|atmosphere|feel)'
                ]
            },
            'compare_areas': {
                'keywords': ['compare', 'difference', 'vs', 'versus', 'better'],
                'patterns': [
                    r'(compare|difference|vs|versus)',
                    r'(better|prefer|choose)',
                    r'(what.*like|how.*different)'
                ]
            },
            
            # TRANSPORTATION & ROUTES
            'get_directions': {
                'keywords': ['how', 'go', 'get', 'transport', 'metro', 'bus', 'directions', 'route'],
                'patterns': [
                    r'how.*(get|go|reach)',
                    r'(transport|metro|bus|ferry|taxi)',
                    r'(from|to).*(from|to)',
                    r'(directions|route|way)'
                ]
            },
            'plan_route': {
                'keywords': ['route', 'plan', 'itinerary', 'journey', 'trip', 'travel'],
                'patterns': [
                    r'(plan|create|make).*(route|itinerary|trip)',
                    r'(best.*way|optimal.*route)',
                    r'(day.*plan|visit.*order)'
                ]
            },
            'transport_info': {
                'keywords': ['metro', 'bus', 'ferry', 'tram', 'dolmuş', 'taxi', 'transport'],
                'patterns': [
                    r'(metro|bus|ferry|tram|dolmuş|taxi)',
                    r'(public.*transport|transportation)',
                    r'(schedule|time|frequency)'
                ]
            },
            
            # DAILY ACTIVITIES & TALKS
            'daily_conversation': {
                'keywords': ['hello', 'hi', 'good', 'weather', 'how', 'today', 'chat', 'talk'],
                'patterns': [
                    r'(hello|hi|hey|merhaba)',
                    r'(good.*morning|good.*evening)',
                    r'(how.*you|how.*day)',
                    r'(weather|hava)',
                    r'(just.*chat|want.*talk)'
                ]
            },
            'local_tips': {
                'keywords': ['tip', 'advice', 'local', 'secret', 'hidden', 'know', 'insider'],
                'patterns': [
                    r'(tip|advice|suggestion)',
                    r'(local.*know|insider.*info)',
                    r'(secret|hidden|off.*beaten)',
                    r'(should.*know|need.*know)'
                ]
            },
            
            # SHOPPING & ENTERTAINMENT
            'find_shopping': {
                'keywords': ['shopping', 'shop', 'store', 'market', 'bazaar', 'pazar', 'buy'],
                'patterns': [
                    r'(shopping|shop|store|market)',
                    r'(buy|purchase|get)',
                    r'(bazaar|pazar|çarşı)'
                ]
            },
            'find_entertainment': {
                'keywords': ['entertainment', 'fun', 'nightlife', 'bar', 'club', 'music', 'show'],
                'patterns': [
                    r'(entertainment|fun|nightlife)',
                    r'(bar|club|music|show)',
                    r'(party|dance|drink)'
                ]
            },
            
            # ACCOMMODATION & WELLNESS
            'find_accommodation': {
                'keywords': ['hotel', 'accommodation', 'stay', 'sleep', 'lodge', 'hostel'],
                'patterns': [
                    r'(hotel|accommodation|stay)',
                    r'(sleep|lodge|hostel)',
                    r'(where.*stay|place.*sleep)'
                ]
            },
            'wellness_spa': {
                'keywords': ['spa', 'hamam', 'massage', 'wellness', 'relax', 'health'],
                'patterns': [
                    r'(spa|hamam|massage)',
                    r'(wellness|relax|health)',
                    r'(turkish.*bath|traditional.*spa)'
                ]
            },
            
            # NATURE & OUTDOOR
            'find_nature': {
                'keywords': ['nature', 'park', 'garden', 'outdoor', 'green', 'forest'],
                'patterns': [
                    r'(nature|park|garden)',
                    r'(outdoor|green|forest)',
                    r'(walk.*nature|fresh.*air)'
                ]
            },
            
            # BUSINESS & WORK
            'business_info': {
                'keywords': ['business', 'work', 'meeting', 'conference', 'office', 'professional'],
                'patterns': [
                    r'(business|work|meeting)',
                    r'(conference|office|professional)',
                    r'(business.*district|financial.*center)'
                ]
            },
            
            # EMERGENCY & PRACTICAL
            'emergency_help': {
                'keywords': ['emergency', 'help', 'urgent', 'hospital', 'police', 'problem'],
                'patterns': [
                    r'(emergency|urgent|help)',
                    r'(hospital|police|ambulance)',
                    r'(problem|trouble|lost)'
                ]
            },
            'practical_info': {
                'keywords': ['currency', 'money', 'atm', 'bank', 'exchange', 'visa', 'passport'],
                'patterns': [
                    r'(currency|money|atm|bank)',
                    r'(exchange|visa|passport)',
                    r'(practical|useful|need.*know)'
                ]
            },
            
            # GENERAL & CONVERSATIONAL
            'general_info': {
                'keywords': ['help', 'about', 'istanbul', 'information', 'what', 'explain'],
                'patterns': [
                    r'(help|info|about)',
                    r'(what.*istanbul)',
                    r'(tell me|explain)',
                    r'(general.*info|basic.*info)'
                ]
            }
        }
    
    def classify_intent(self, query: str, entities: Dict[str, Any]) -> Tuple[str, float]:
        """Classify intent based on patterns and entities"""
        query_lower = query.lower()
        scores = defaultdict(float)
        
        # Score based on keywords
        for intent, config in self.intent_patterns.items():
            keyword_score = sum(1 for keyword in config['keywords'] if keyword in query_lower)
            scores[intent] += keyword_score * 0.3
            
            # Score based on regex patterns
            pattern_score = sum(1 for pattern in config['patterns'] 
                              if re.search(pattern, query_lower))
            scores[intent] += pattern_score * 0.4
        
        # Boost scores based on entities
        if entities['categories']:
            # Food & Dining
            if 'restaurant' in entities['categories']:
                scores['find_restaurant'] += 0.5
            if 'cafe' in entities['categories']:
                scores['find_cafe'] += 0.5
            
            # Attractions & Culture
            if 'museum' in entities['categories']:
                scores['find_museum'] += 0.5
            if 'attraction' in entities['categories']:
                scores['find_attraction'] += 0.5
            if 'cultural' in entities['categories']:
                scores['cultural_experience'] += 0.5
            
            # Transportation
            if 'transport' in entities['categories']:
                scores['get_directions'] += 0.5
                scores['transport_info'] += 0.3
            
            # Areas & Neighborhoods
            if 'neighborhood' in entities['categories']:
                scores['explore_neighborhood'] += 0.5
            
            # Shopping & Entertainment
            if 'shopping' in entities['categories']:
                scores['find_shopping'] += 0.5
            if 'entertainment' in entities['categories']:
                scores['find_entertainment'] += 0.5
            
            # Accommodation & Wellness
            if 'hotel' in entities['categories']:
                scores['find_accommodation'] += 0.5
            if 'wellness' in entities['categories']:
                scores['wellness_spa'] += 0.5
            
            # Nature & Business
            if 'nature' in entities['categories']:
                scores['find_nature'] += 0.5
            if 'business' in entities['categories']:
                scores['business_info'] += 0.5
        
        # Boost based on transport modes mentioned
        if entities['transport_modes']:
            scores['get_directions'] += 0.4
            scores['transport_info'] += 0.3
        
        # Boost based on attraction types
        if entities['attraction_types']:
            scores['find_attraction'] += 0.4
            if any(atype in ['mosque', 'palace', 'monument'] for atype in entities['attraction_types']):
                scores['cultural_experience'] += 0.3
        
        # Handle direction queries
        direction_words = ['from', 'to', 'how', 'get', 'go']
        if sum(1 for word in direction_words if word in query_lower) >= 2:
            scores['get_directions'] += 0.6
        
        # Get best intent
        if scores:
            best_intent = max(scores.items(), key=lambda x: x[1])
            return best_intent[0], min(best_intent[1], 1.0)
        else:
            return 'general_info', 0.3

class SemanticExpander:
    """Expand queries with semantic understanding without ML models"""
    
    def __init__(self):
        self.semantic_groups = {
            'food_quality': {
                'good': ['best', 'great', 'excellent', 'top', 'amazing', 'fantastic'],
                'bad': ['worst', 'terrible', 'awful', 'poor'],
                'authentic': ['traditional', 'local', 'genuine', 'real', 'original']
            },
            'atmosphere': {
                'quiet': ['peaceful', 'calm', 'serene', 'tranquil', 'silent'],
                'busy': ['crowded', 'lively', 'vibrant', 'energetic', 'bustling'],
                'romantic': ['intimate', 'cozy', 'charming', 'lovely'],
                'casual': ['relaxed', 'informal', 'laid-back', 'easy-going']
            },
            'location': {
                'near': ['close', 'nearby', 'around', 'next to', 'beside'],
                'far': ['distant', 'away', 'remote'],
                'center': ['central', 'downtown', 'heart', 'middle'],
                'waterfront': ['seaside', 'coastal', 'by the water', 'bosphorus']
            },
            'price': {
                'cheap': ['budget', 'affordable', 'inexpensive', 'economical'],
                'expensive': ['costly', 'pricey', 'upscale', 'luxury', 'premium']
            }
        }
    
    def expand_query(self, query: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Expand query with semantic alternatives"""
        query_lower = query.lower()
        expansions = {
            'synonyms': [],
            'related_terms': [],
            'expanded_vibes': []
        }
        
        # Find semantic expansions
        for category, groups in self.semantic_groups.items():
            for main_term, synonyms in groups.items():
                if main_term in query_lower:
                    expansions['synonyms'].extend(synonyms)
                elif any(syn in query_lower for syn in synonyms):
                    expansions['related_terms'].append(main_term)
        
        # Expand vibes based on context
        if entities['vibes']:
            for vibe in entities['vibes']:
                if vibe in ['quiet', 'peaceful']:
                    expansions['expanded_vibes'].extend(['serene', 'calm', 'relaxing'])
                elif vibe in ['romantic']:
                    expansions['expanded_vibes'].extend(['intimate', 'cozy', 'charming'])
                elif vibe in ['authentic']:
                    expansions['expanded_vibes'].extend(['traditional', 'local', 'genuine'])
        
        return expansions

class EnhancedQueryProcessor:
    """Main query processing pipeline"""
    
    def __init__(self):
        self.spell_corrector = TurkishSpellCorrector()
        self.entity_extractor = EntityExtractor()
        self.intent_classifier = IntentClassifier()
        self.semantic_expander = SemanticExpander()
        
        # Conversation context (simple in-memory for now)
        self.conversation_context = {}
    
    def process_query(self, query: str, session_id: str = None) -> ParsedQuery:
        """Process query through the full pipeline"""
        
        # 1. Spell correction and normalization
        corrected_query, corrections = self.spell_corrector.correct_text(query)
        
        # 2. Entity extraction
        entities = self.entity_extractor.extract_entities(corrected_query)
        
        # 3. Intent classification
        intent, confidence = self.intent_classifier.classify_intent(corrected_query, entities)
        
        # 4. Semantic expansion
        semantic_expansion = self.semantic_expander.expand_query(corrected_query, entities)
        
        # 5. Apply conversation context if available
        if session_id and session_id in self.conversation_context:
            entities = self._apply_context(entities, session_id)
        
        # 6. Store context for next turn
        if session_id:
            self._update_context(session_id, intent, entities)
        
        return ParsedQuery(
            original_query=query,
            normalized_query=corrected_query,
            intent=intent,
            entities=entities,
            confidence=confidence,
            corrections=corrections,
            temporal_context=entities.get('temporal'),
            vibe_tags=entities.get('vibes', [])
        )
    
    def _apply_context(self, entities: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Apply conversation context to current entities"""
        context = self.conversation_context.get(session_id, {})
        
        # If no location specified, use previous location
        if not entities['districts'] and context.get('last_districts'):
            entities['districts'] = context['last_districts']
        
        # If no category specified, use previous category
        if not entities['categories'] and context.get('last_categories'):
            entities['categories'] = context['last_categories']
        
        return entities
    
    def _update_context(self, session_id: str, intent: str, entities: Dict[str, Any]):
        """Update conversation context"""
        if session_id not in self.conversation_context:
            self.conversation_context[session_id] = {}
        
        context = self.conversation_context[session_id]
        context['last_intent'] = intent
        context['last_districts'] = entities.get('districts', [])
        context['last_categories'] = entities.get('categories', [])
        context['timestamp'] = datetime.now().isoformat()

# Export main processor
enhanced_query_processor = EnhancedQueryProcessor()

def process_enhanced_query(query: str, session_id: str = None) -> Dict[str, Any]:
    """Main function to process queries with enhanced understanding"""
    parsed = enhanced_query_processor.process_query(query, session_id)
    
    return {
        'original_query': parsed.original_query,
        'normalized_query': parsed.normalized_query,
        'intent': parsed.intent,
        'confidence': parsed.confidence,
        'entities': parsed.entities,
        'corrections': parsed.corrections,
        'temporal_context': parsed.temporal_context,
        'vibe_tags': parsed.vibe_tags,
        'success': True
    }

if __name__ == "__main__":
    # Test the enhanced query processor
    test_queries = [
        "best restaraunts in kadıköyy",  # Spelling errors
        "show me romantic places near bosphorus tonight",
        "how do I go from sultanahmet to beyoglu by metro?",
        "authentic turkish food in quiet area",
        "coffee shops with cats",
        "what about near galata tower?",  # Context-dependent
        "find museums in sultanahmet for 2 hours",  # Museum with duration
        "shopping in grand bazaar for family",  # Shopping with group
        "nightlife in beyoglu for young couple",  # Entertainment
        "hamam experience in historic area",  # Wellness/spa
        "compare beyoglu vs kadikoy neighborhoods",  # Area comparison
        "plan 3 day route visiting major attractions",  # Route planning
        "hello, how's the weather today?",  # Daily conversation
        "secret local tips for authentic experience",  # Local tips
        "business district with good hotels",  # Business + accommodation
        "nature parks for walking with children",  # Nature + family
        "emergency hospital near taksim",  # Emergency
        "currency exchange and atm locations"  # Practical info
    ]
    
    print("🧠 Enhanced Query Understanding System Test")
    print("=" * 50)
    
    session_id = "test_session"
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        result = process_enhanced_query(query, session_id)
        
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Normalized: {result['normalized_query']}")
        
        if result['corrections']:
            print(f"Corrections: {', '.join(result['corrections'])}")
        
        entities = result['entities']
        if entities['districts']:
            print(f"Districts: {', '.join(entities['districts'])}")
        if entities['categories']:
            print(f"Categories: {', '.join(entities['categories'])}")
        if entities['cuisines']:
            print(f"Cuisines: {', '.join(entities['cuisines'])}")
        if entities['transport_modes']:
            print(f"Transport: {', '.join(entities['transport_modes'])}")
        if entities['attraction_types']:
            print(f"Attraction Types: {', '.join(entities['attraction_types'])}")
        if entities['vibes']:
            print(f"Vibes: {', '.join(entities['vibes'])}")
        if entities['temporal']:
            print(f"Temporal: {entities['temporal']}")
        if entities['budget']:
            print(f"Budget: {entities['budget']}")
        if entities['group_size']:
            print(f"Group Size: {entities['group_size']}")
        if entities['duration']:
            print(f"Duration: {entities['duration']}")

"""
Enhanced GPT-Free Query Processing System
Integrates ML semantic caching, query clustering, and existing AI Istanbul components
Targets 95%+ GPT-free operation with intelligent fallback layers
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from dataclasses import dataclass
import json

# Import our components
try:
    from ml_semantic_cache import MLSemanticCache, CachedResponse
    from query_clustering_system import QueryClusteringSystem, GPTFreeQueryProcessor
except ImportError:
    from .ml_semantic_cache import MLSemanticCache, CachedResponse
    from .query_clustering_system import QueryClusteringSystem, GPTFreeQueryProcessor

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Result from query processing"""
    response: str
    source: str  # 'cache', 'clustering', 'templates', 'fallback', 'error'
    confidence: float
    metadata: Dict[str, Any]
    processing_time_ms: float
    user_satisfaction_predicted: float = 0.8

class EnhancedGPTFreeSystem:
    """
    Production-grade GPT-free system for AI Istanbul
    Combines multiple intelligence layers for maximum coverage
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize components
        self.semantic_cache = MLSemanticCache(
            cache_dir=self.config.get('cache_dir', 'cache_data'),
            embedding_model=self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        )
        
        self.clustering_system = QueryClusteringSystem(
            data_dir=self.config.get('clustering_dir', 'clustering_data')
        )
        
        self.gpt_free_processor = GPTFreeQueryProcessor(
            clustering_system=self.clustering_system,
            semantic_cache=self.semantic_cache
        )
        
        # Enhanced fallback system
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Location data for quick access - ensure it's a dict
        if isinstance(self.knowledge_base, dict):
            self.location_data = self.knowledge_base.get('attractions', {})
        else:
            logger.error(f"Knowledge base is not a dict: {type(self.knowledge_base)}")
            self.location_data = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'clustering_hits': 0,
            'fallback_hits': 0,
            'avg_response_time': 0.0,
            'satisfaction_scores': [],
            'coverage_by_source': {
                'cache': 0,
                'clustering': 0,
                'knowledge_base': 0,
                'fallback': 0
            }
        }
        
        # Initialize with pre-trained data
        self._bootstrap_system()
    
    def _initialize_knowledge_base(self) -> Dict:
        """Initialize comprehensive Istanbul knowledge base"""
        return {
            'attractions': {
                'hagia_sophia': {
                    'name': 'Hagia Sophia',
                    'type': 'historical_site',
                    'district': 'Sultanahmet',
                    'opening_hours': '9:00-19:00 (Apr-Oct), 9:00-17:00 (Nov-Mar)',
                    'ticket_price': '200 TL',
                    'visit_duration': '1-2 hours',
                    'highlights': ['Byzantine architecture', 'Ottoman additions', 'Magnificent dome'],
                    'tips': ['Visit early morning to avoid crowds', 'Audio guide recommended', 'Combine with Blue Mosque visit'],
                    'transport': {
                        'metro': 'M1 to Vezneciler, then 10 min walk',
                        'tram': 'T1 to Sultanahmet',
                        'bus': 'Multiple lines to Eminönü'
                    }
                },
                'blue_mosque': {
                    'name': 'Blue Mosque (Sultan Ahmed Mosque)',
                    'type': 'mosque',
                    'district': 'Sultanahmet',
                    'opening_hours': 'Open except during prayer times',
                    'ticket_price': 'Free',
                    'visit_duration': '30-45 minutes',
                    'highlights': ['Six minarets', 'Blue Iznik tiles', 'Grand interior'],
                    'tips': ['Dress modestly', 'Remove shoes', 'Respect prayer times'],
                    'transport': {
                        'tram': 'T1 to Sultanahmet',
                        'metro': 'M1 to Vezneciler, then walk',
                        'walking': '5 minutes from Hagia Sophia'
                    }
                },
                'galata_tower': {
                    'name': 'Galata Tower',
                    'type': 'tower',
                    'district': 'Beyoğlu',
                    'opening_hours': '9:00-20:00',
                    'ticket_price': '150 TL',
                    'visit_duration': '45 minutes',
                    'highlights': ['360° city views', 'Bosphorus panorama', 'Historic architecture'],
                    'tips': ['Buy tickets online', 'Best views at sunset', 'Elevator to top'],
                    'transport': {
                        'metro': 'M2 to Şişhane, then 5 min walk',
                        'funicular': 'Tünel from Galata Bridge',
                        'ferry': 'To Karaköy, then 10 min walk'
                    }
                }
            },
            'districts': {
                'sultanahmet': {
                    'name': 'Sultanahmet',
                    'character': 'Historic heart of Istanbul',
                    'main_attractions': ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace', 'Grand Bazaar'],
                    'best_for': ['History lovers', 'First-time visitors', 'Architecture enthusiasts'],
                    'dining': ['Traditional Ottoman cuisine', 'Tourist-friendly restaurants', 'Rooftop terraces'],
                    'transport_hub': 'T1 Tram line, multiple bus routes'
                },
                'beyoglu': {
                    'name': 'Beyoğlu',
                    'character': 'Modern cultural district',
                    'main_attractions': ['Galata Tower', 'Istiklal Street', 'Pera Museum'],
                    'best_for': ['Nightlife', 'Shopping', 'Art galleries', 'Modern dining'],
                    'dining': ['International cuisine', 'Trendy cafes', 'Rooftop bars'],
                    'transport_hub': 'M2 Metro, Tünel funicular'
                }
            },
            'transport': {
                'metro_lines': {
                    'M1': 'Yenikapi - Airport/Kirazli',
                    'M2': 'Vezneciler - Haciosman',
                    'M3': 'Kirazli - Başakşehir',
                    'M4': 'Kadikoy - Sabiha Gokcen Airport',
                    'M5': 'Üsküdar - Çekmeköy',
                    'M6': 'Levent - Boğaziçi Üniversitesi'
                },
                'tram_lines': {
                    'T1': 'Bağcılar - Kabataş (main tourist line)',
                    'T4': 'Topkapı - Mescid-i Selam'
                },
                'ferry_routes': {
                    'bosphorus': 'European and Asian side connections',
                    'golden_horn': 'Eminönü - Eyüp - Sütlüce'
                },
                'cards': {
                    'istanbulkart': 'Main transport card - buy at stations',
                    'tourist_card': 'Special tourist versions available'
                }
            },
            'food': {
                'traditional_dishes': {
                    'kebab': 'Grilled meat dishes - try Adana, Urfa, or İskender',
                    'meze': 'Small appetizer plates - perfect for sharing',
                    'baklava': 'Sweet pastry with nuts and honey',
                    'turkish_breakfast': 'Extensive spread with cheese, olives, bread',
                    'döner': 'Rotating grilled meat in bread or over rice',
                    'manti': 'Turkish dumplings with yogurt and garlic'
                },
                'dining_areas': {
                    'sultanahmet': 'Tourist-friendly, traditional Ottoman cuisine',
                    'karakoy': 'Trendy restaurants and cafes',
                    'besiktas': 'Local favorites and seafood',
                    'kadikoy': 'Asian side dining and street food'
                }
            }
        }
    
    def _bootstrap_system(self):
        """Bootstrap system with essential queries and responses"""
        bootstrap_queries = [
            {
                'query': 'How to get to Hagia Sophia',
                'response': self._generate_transport_response('hagia_sophia'),
                'intent': 'transportation',
                'context': {'destination': 'hagia_sophia'}
            },
            {
                'query': 'Blue Mosque opening hours',
                'response': self._generate_info_response('blue_mosque'),
                'intent': 'practical_info',
                'context': {'attraction': 'blue_mosque'}
            },
            {
                'query': 'Best restaurants in Sultanahmet',
                'response': self._generate_food_response('sultanahmet'),
                'intent': 'food',
                'context': {'area': 'sultanahmet'}
            },
            {
                'query': 'What to see in Beyoglu',
                'response': self._generate_exploration_response('beyoglu'),
                'intent': 'exploration',
                'context': {'area': 'beyoglu'}
            }
        ]
        
        # Add to cache and clustering system
        for item in bootstrap_queries:
            # Add to semantic cache
            self.semantic_cache.add_to_cache(
                item['query'], item['response'], item['intent'], 'en', item['context']
            )
            
            # Add to clustering system for pattern learning
            self.clustering_system.add_query_sample(
                item['query'], item['response'], item['intent'], item['context']
            )
        
        logger.info(f"✅ Bootstrapped system with {len(bootstrap_queries)} essential queries")
    
    def process_query(self, query: str, context: Dict = None, user_id: str = None, 
                     user_preferences: Dict = None) -> ProcessingResult:
        """
        Main query processing method - uses all available intelligence layers
        """
        start_time = time.time()
        self.performance_stats['total_queries'] += 1
        
        # Enhance context with user preferences
        enhanced_context = dict(context or {})
        if user_preferences:
            enhanced_context['user_preferences'] = user_preferences
        
        try:
            # Layer 1: Semantic Cache (Highest Priority)
            cache_result = self.semantic_cache.get_cached_response(
                query, enhanced_context, user_id
            )
            
            if cache_result:
                response, confidence, metadata = cache_result
                processing_time = (time.time() - start_time) * 1000
                
                self.performance_stats['cache_hits'] += 1
                self.performance_stats['coverage_by_source']['cache'] += 1
                
                return ProcessingResult(
                    response=response,
                    source='semantic_cache',
                    confidence=confidence,
                    metadata=metadata,
                    processing_time_ms=processing_time,
                    user_satisfaction_predicted=0.85
                )
            
            # Layer 2: Query Clustering System
            cluster_match = self.clustering_system.match_query_to_cluster(
                query, enhanced_context
            )
            
            if cluster_match:
                cluster, confidence = cluster_match
                template_response = self.clustering_system.get_template_response(
                    cluster, query, enhanced_context
                )
                
                if template_response:
                    # Enhance with knowledge base
                    enhanced_response = self._enhance_with_knowledge(
                        template_response, query, enhanced_context
                    )
                    
                    # Add to cache for future use
                    self.semantic_cache.add_to_cache(
                        query, enhanced_response, cluster.intent_type, 
                        'en', enhanced_context, cluster.template_id
                    )
                    
                    processing_time = (time.time() - start_time) * 1000
                    self.performance_stats['clustering_hits'] += 1
                    self.performance_stats['coverage_by_source']['clustering'] += 1
                    
                    return ProcessingResult(
                        response=enhanced_response,
                        source='query_clustering',
                        confidence=confidence,
                        metadata={
                            'cluster_id': cluster.cluster_id,
                            'cluster_name': cluster.name,
                            'template_id': cluster.template_id,
                            'intent_type': cluster.intent_type
                        },
                        processing_time_ms=processing_time,
                        user_satisfaction_predicted=0.80
                    )
            
            # Layer 3: Direct Knowledge Base Lookup
            kb_response = self._query_knowledge_base(query, enhanced_context)
            if kb_response:
                processing_time = (time.time() - start_time) * 1000
                self.performance_stats['coverage_by_source']['knowledge_base'] += 1
                
                # Add successful KB response to cache
                intent_type = self._detect_intent(query)
                self.semantic_cache.add_to_cache(
                    query, kb_response, intent_type, 'en', enhanced_context
                )
                
                return ProcessingResult(
                    response=kb_response,
                    source='knowledge_base',
                    confidence=0.7,
                    metadata={'method': 'direct_lookup'},
                    processing_time_ms=processing_time,
                    user_satisfaction_predicted=0.75
                )
            
            # Layer 4: Smart Fallback
            fallback_response = self._generate_smart_fallback(query, enhanced_context)
            processing_time = (time.time() - start_time) * 1000
            self.performance_stats['fallback_hits'] += 1
            self.performance_stats['coverage_by_source']['fallback'] += 1
            
            return ProcessingResult(
                response=fallback_response,
                source='smart_fallback',
                confidence=0.4,
                metadata={'detected_intent': self._detect_intent(query)},
                processing_time_ms=processing_time,
                user_satisfaction_predicted=0.6
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing query '{query}': {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                response="I apologize, but I'm experiencing technical difficulties. Please try again or contact support.",
                source='error',
                confidence=0.0,
                metadata={'error': str(e)},
                processing_time_ms=processing_time,
                user_satisfaction_predicted=0.2
            )
        
        finally:
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_stats(processing_time)
    
    def _enhance_with_knowledge(self, response: str, query: str, context: Dict) -> str:
        """Enhance template response with knowledge base information"""
        try:
            # Extract key entities from query
            query_lower = query.lower()
            
            # Location enhancement
            for location_key, location_data in self.knowledge_base['attractions'].items():
                if location_data['name'].lower() in query_lower:
                    # Add specific information
                    if 'hours' in query_lower or 'open' in query_lower:
                        response += f"\n\n🕒 **Current Hours:** {location_data['opening_hours']}"
                    
                    if 'price' in query_lower or 'ticket' in query_lower:
                        response += f"\n\n💰 **Ticket Price:** {location_data['ticket_price']}"
                    
                    if 'transport' in query_lower or 'get to' in query_lower:
                        transport_info = location_data.get('transport', {})
                        if transport_info:
                            response += f"\n\n🚇 **Transport Options:**"
                            for mode, info in transport_info.items():
                                response += f"\n• **{mode.title()}:** {info}"
            
            # Add current context enhancements
            current_time = datetime.now()
            if current_time.hour < 9:
                response += "\n\n⏰ *Note: Most attractions open at 9:00 AM*"
            elif current_time.hour > 17:
                response += "\n\n⏰ *Note: Many attractions close at 17:00-18:00*"
            
            return response
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}")
            return response
    
    def _query_knowledge_base(self, query: str, context: Dict) -> Optional[str]:
        """Query knowledge base directly for information"""
        query_lower = query.lower()
        
        # Attraction-specific queries
        for location_key, location_data in self.knowledge_base['attractions'].items():
            if location_data['name'].lower() in query_lower:
                if 'hours' in query_lower or 'open' in query_lower:
                    return self._generate_hours_response(location_data)
                elif 'price' in query_lower or 'ticket' in query_lower:
                    return self._generate_price_response(location_data)
                elif 'transport' in query_lower or 'get to' in query_lower:
                    return self._generate_transport_response(location_key)
                elif 'about' in query_lower or 'tell me' in query_lower:
                    return self._generate_about_response(location_data)
        
        # District-specific queries
        for district_key, district_data in self.knowledge_base['districts'].items():
            if district_data['name'].lower() in query_lower:
                if 'what to see' in query_lower or 'attractions' in query_lower:
                    return self._generate_district_attractions_response(district_data)
                elif 'food' in query_lower or 'restaurant' in query_lower:
                    return self._generate_district_food_response(district_data)
        
        return None
    
    def _generate_info_response(self, location_key: str) -> str:
        """Generate information response for a location"""
        location_data = self.location_data.get(location_key, {})
        
        if not location_data:
            return f"I don't have specific information about {location_key}, but I can help you with general tourism information about Istanbul."
        
        return f"""ℹ️ **{location_data.get('name', location_key.title())} Information:**

**About:** {location_data.get('description', 'A popular tourist destination in Istanbul.')}

**Opening Hours:** {location_data.get('opening_hours', 'Please check current hours')}
**Ticket Price:** {location_data.get('ticket_price', 'Please check current pricing')}
**Visit Duration:** {location_data.get('visit_duration', '1-2 hours')}

**Location:** {location_data.get('district', 'Istanbul')}

💡 **Tips:**
{chr(10).join(f"• {tip}" for tip in location_data.get('tips', ['Check the weather before visiting', 'Arrive early to avoid crowds']))}

📍 **How to Get There:**
{location_data.get('transport', 'Accessible by public transportation')}"""
    
    def _generate_hours_response(self, location_data: Dict) -> str:
        return f"""🕒 **{location_data['name']} Opening Hours:**

**Hours:** {location_data['opening_hours']}
**Duration:** {location_data['visit_duration']}

💡 **Tips:**
{chr(10).join(f"• {tip}" for tip in location_data.get('tips', []))}"""
    
    def _generate_price_response(self, location_data: Dict) -> str:
        return f"""💰 **{location_data['name']} Pricing:**

**Ticket Price:** {location_data['ticket_price']}
**Visit Duration:** {location_data['visit_duration']}

💡 **Money-saving tips:**
• Book online for potential discounts
• Consider Istanbul Museum Pass for multiple attractions
• Student discounts available with valid ID"""
    
    def _generate_transport_response(self, location_key: str) -> str:
        location_data = self.knowledge_base['attractions'][location_key]
        transport_info = location_data.get('transport', {})
        
        response = f"""🚇 **Getting to {location_data['name']}:**

"""
        
        for mode, info in transport_info.items():
            response += f"**{mode.title()}:** {info}\n"
        
        response += """
💳 **Payment:** Use İstanbulkart for all public transport
⏱️ **Peak Hours:** Avoid 8-9 AM and 5-7 PM for less crowding"""
        
        return response
    
    def _generate_about_response(self, location_data: Dict) -> str:
        return f"""🏛️ **About {location_data['name']}:**

{location_data.get('description', f"One of Istanbul's most significant {location_data['type']}s")}

**Highlights:**
{chr(10).join(f"• {highlight}" for highlight in location_data.get('highlights', []))}

**Visit Info:**
• **Duration:** {location_data['visit_duration']}
• **Best Time:** {location_data.get('best_time', 'Early morning or late afternoon')}
• **District:** {location_data['district']}

💡 **Pro Tips:**
{chr(10).join(f"• {tip}" for tip in location_data.get('tips', []))}"""
    
    def _generate_food_response(self, area: str) -> str:
        # Safely get area data, ensuring it's a dictionary
        try:
            food_section = self.knowledge_base.get('food', {}) if isinstance(self.knowledge_base, dict) else {}
            dining_areas = food_section.get('dining_areas', {}) if isinstance(food_section, dict) else {}
            area_data = dining_areas.get(area, {}) if isinstance(dining_areas, dict) else {}
            
            # Safely get dishes
            traditional_dishes = food_section.get('traditional_dishes', {}) if isinstance(food_section, dict) else {}
            dishes = list(traditional_dishes.items())[:5] if isinstance(traditional_dishes, dict) else []
        except:
            area_data = {}
            dishes = []
        
        return f"""🍽️ **Food in {area.title()}:**

**Area Character:** {area_data.get('description', 'Great dining options available') if isinstance(area_data, dict) else 'Great dining options available'}

**Must-Try Dishes:**
{chr(10).join(f"• **{dish}:** {desc}" for dish, desc in dishes) if dishes else "• Turkish cuisine offers many delicious options"}

**Dining Tips:**
• Try traditional Turkish breakfast
• Look for places popular with locals
• Many restaurants don't serve alcohol - check first
• Tipping 10-15% is standard

🌟 **Local Recommendation:** Ask locals for their favorite "lokanta" (casual restaurant)"""
    
    def _generate_exploration_response(self, area: str) -> str:
        area_data = self.knowledge_base['districts'].get(area, {})
        
        return f"""🗺️ **Exploring {area_data.get('name', area.title())}:**

**Character:** {area_data.get('character', 'Unique Istanbul neighborhood')}

**Main Attractions:**
{chr(10).join(f"• {attraction}" for attraction in area_data.get('main_attractions', ['Various local sites']))}

**Perfect For:**
{chr(10).join(f"• {visitor_type}" for visitor_type in area_data.get('best_for', ['All visitors']))}

**Getting Around:**
{area_data.get('transport_hub', 'Well connected by public transport')}

⏰ **Time Needed:** Half day to full day depending on interests"""
    
    def _generate_district_attractions_response(self, district_data: Dict) -> str:
        return f"""🏛️ **Top Attractions in {district_data['name']}:**

**Main Highlights:**
{chr(10).join(f"• {attraction}" for attraction in district_data.get('main_attractions', []))}

**Best For:**
{chr(10).join(f"• {visitor_type}" for visitor_type in district_data.get('best_for', []))}

**Character:** {district_data.get('character', 'Distinctive Istanbul district')}

🚇 **Transport:** {district_data.get('transport_hub', 'Accessible by public transport')}"""
    
    def _generate_district_food_response(self, district_data: Dict) -> str:
        return f"""🍽️ **Dining in {district_data['name']}:**

**Food Scene:** {district_data.get('dining', ['Various dining options available'])[0]}

**Specialties:**
{chr(10).join(f"• {specialty}" for specialty in district_data.get('dining', ['Local and international cuisine']))}

**Atmosphere:** {district_data.get('character', 'Authentic Istanbul dining experience')}

💡 **Tip:** Explore side streets for hidden gems and local favorites!"""
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent for categorization"""
        query_lower = query.lower()
        
        intent_keywords = {
            'transportation': ['get to', 'how to go', 'metro', 'bus', 'transport', 'way to'],
            'practical_info': ['hours', 'open', 'close', 'price', 'ticket', 'cost', 'when'],
            'food': ['restaurant', 'eat', 'food', 'dinner', 'lunch', 'breakfast', 'cuisine'],
            'exploration': ['see', 'visit', 'do', 'attractions', 'places', 'explore'],
            'shopping': ['buy', 'shop', 'market', 'bazaar', 'souvenir', 'store']
        }
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return 'general'
    
    def _generate_smart_fallback(self, query: str, context: Dict) -> str:
        """Generate intelligent fallback response"""
        intent = self._detect_intent(query)
        
        fallback_responses = {
            'transportation': """🚇 **Getting Around Istanbul:**

I'd be happy to help you navigate Istanbul! For the best transport advice, please let me know:
• Your starting point
• Your destination
• Preferred transport mode (metro, bus, taxi, walking)

💡 **Quick Tips:**
• Use İstanbulkart for all public transport
• Metro and tram are fastest for long distances
• Ferries offer scenic routes across the Bosphorus""",
            
            'practical_info': """ℹ️ **Istanbul Attraction Information:**

I can help with practical details! Please specify:
• Which attraction you're interested in
• What information you need (hours, prices, tickets)

💡 **General Tips:**
• Most museums: 9:00-17:00, closed Mondays  
• Mosques: Open all day, avoid prayer times
• Book popular attractions online to skip queues""",
            
            'food': """🍽️ **Istanbul Dining Guide:**

Istanbul has incredible food! Let me know:
• Which area you're in or visiting
• Type of cuisine you prefer
• Budget range (budget/mid-range/fine dining)

🌟 **Must-Try:**
• Turkish breakfast
• Kebabs and döner
• Baklava and Turkish delight
• Turkish tea and coffee""",
            
            'exploration': """🗺️ **Exploring Istanbul:**

I'd love to help you discover Istanbul! Please tell me:
• Which area interests you
• Your travel style (history, culture, nightlife, etc.)
• How much time you have

🎯 **Top Districts:**
• Sultanahmet: Historic sites
• Beyoğlu: Modern culture & nightlife
• Karaköy: Trendy restaurants & galleries""",
            
            'shopping': """🛍️ **Shopping in Istanbul:**

Istanbul is a shopper's paradise! Let me know:
• What you're looking to buy
• Your preferred shopping style (markets vs malls)
• Budget range

🏪 **Popular Options:**
• Grand Bazaar: Traditional crafts & souvenirs
• İstiklal Street: Modern shopping
• Local markets: Authentic experiences"""
        }
        
        return fallback_responses.get(intent, 
            """👋 **Welcome to Istanbul!**

I'm here to help you explore this amazing city! I can assist with:

🗺️ **Getting Around:** Transport, directions, metro/bus info
🏛️ **Attractions:** Hours, prices, recommendations  
🍽️ **Food:** Restaurant recommendations, local dishes
🛍️ **Shopping:** Markets, souvenirs, local products
📍 **Areas:** What to see and do in different districts

What would you like to know about Istanbul?""")
    
    def _update_performance_stats(self, processing_time: float):
        """Update system performance statistics"""
        # Update average response time
        current_avg = self.performance_stats['avg_response_time']
        total_queries = self.performance_stats['total_queries']
        
        self.performance_stats['avg_response_time'] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        cache_stats = self.semantic_cache.stats
        clustering_stats = self.clustering_system.get_cluster_statistics()
        
        # Calculate coverage percentages
        total_queries = self.performance_stats['total_queries']
        coverage_percentages = {}
        
        if total_queries > 0:
            for source, count in self.performance_stats['coverage_by_source'].items():
                coverage_percentages[source] = (count / total_queries) * 100
        
        # Calculate cache hit rate
        cache_hit_rate = (self.performance_stats['cache_hits'] / total_queries * 100) if total_queries > 0 else 0
        
        return {
            'overall_performance': {
                'total_queries_processed': total_queries,
                'avg_response_time_ms': round(self.performance_stats['avg_response_time'], 2),
                'cache_hit_rate_percent': round(cache_hit_rate, 1),
                'gpt_free_coverage_percent': round(sum(coverage_percentages.values()), 1)
            },
            'coverage_by_source': coverage_percentages,
            'semantic_cache': {
                'total_cached_responses': len(cache_stats),
                'cache_hits': cache_stats.get('cache_hits', 0),
                'cache_misses': cache_stats.get('cache_misses', 0),
                'embeddings_generated': cache_stats.get('embeddings_generated', 0)
            },
            'clustering_system': clustering_stats,
            'knowledge_base': {
                'attractions_count': len(self.knowledge_base['attractions']),
                'districts_count': len(self.knowledge_base['districts']),
                'transport_options': len(self.knowledge_base['transport'])
            }
        }
    
    def learn_from_feedback(self, query: str, response: str, user_satisfaction: float, 
                          metadata: Dict = None):
        """Learn from user feedback to improve system"""
        try:
            # Update satisfaction tracking
            self.performance_stats['satisfaction_scores'].append(user_satisfaction)
            
            # If response was good, add to cache
            if user_satisfaction >= 0.7:
                intent_type = self._detect_intent(query)
                self.semantic_cache.add_to_cache(
                    query, response, intent_type, 'en', 
                    metadata or {}, user_satisfaction=user_satisfaction
                )
                
                # Also add to clustering system for pattern learning
                self.clustering_system.add_query_sample(
                    query, response, intent_type, metadata, user_satisfaction
                )
                
                logger.info(f"✅ Learned from positive feedback: {user_satisfaction:.2f}")
            
            # If response was poor, log for system improvement
            elif user_satisfaction < 0.4:
                logger.warning(f"⚠️ Poor response feedback: {user_satisfaction:.2f} for '{query[:50]}...'")
                # TODO: Add to improvement queue
            
        except Exception as e:
            logger.error(f"❌ Error learning from feedback: {e}")
    
    def export_system_state(self, filepath: str = None):
        """Export current system state for backup/analysis"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"gpt_free_system_state_{timestamp}.json"
        
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'performance_stats': self.performance_stats,
            'system_statistics': self.get_system_statistics(),
            'cache_size': len(self.semantic_cache.cached_responses),
            'cluster_count': len(self.clustering_system.clusters),
            'template_count': len(self.clustering_system.templates)
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"💾 System state exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error exporting system state: {e}")
            return None

# Factory function for easy integration
def create_gpt_free_system(config: Dict = None) -> EnhancedGPTFreeSystem:
    """Factory function to create and initialize GPT-free system"""
    return EnhancedGPTFreeSystem(config or {})

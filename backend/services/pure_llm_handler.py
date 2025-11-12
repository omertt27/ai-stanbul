"""
Pure LLM Query Handler
Routes ALL queries through RunPod LLM with database context injection
No rule-based fallback - LLM handles everything

Architecture:
- Single entry point for all queries
- Context injection from database
- RAG for similar queries
- Intent-aware system prompts
- Redis caching for responses

Author: Istanbul AI Team
Date: November 12, 2025
"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from datetime import datetime

logger = logging.getLogger(__name__)


class PureLLMHandler:
    """
    Pure LLM architecture - no rule-based processing
    All queries go through RunPod LLM with context injection
    """
    
    def __init__(
        self,
        runpod_client,
        db_session: Session,
        redis_client=None,
        context_builder=None,
        rag_service=None
    ):
        """
        Initialize Pure LLM Handler
        
        Args:
            runpod_client: RunPod LLM client instance
            db_session: SQLAlchemy database session
            redis_client: Redis client for caching (optional)
            context_builder: ML context builder (optional)
            rag_service: RAG vector service (optional)
        """
        self.llm = runpod_client
        self.db = db_session
        self.redis = redis_client
        self.context_builder = context_builder
        self.rag = rag_service
        
        # Load system prompts
        self._load_prompts()
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallback_calls": 0
        }
        
        logger.info("✅ Pure LLM Handler initialized")
        logger.info(f"   RunPod LLM: {'✅ Enabled' if self.llm.enabled else '❌ Disabled'}")
        logger.info(f"   Redis Cache: {'✅ Enabled' if self.redis else '❌ Disabled'}")
        logger.info(f"   RAG Service: {'✅ Enabled' if self.rag else '❌ Disabled'}")
    
    def _load_prompts(self):
        """Load Istanbul-specific system prompts"""
        
        self.base_prompt = """You are AI Istanbul, an expert travel assistant for Istanbul, Turkey.

You have deep knowledge of:
🏛️ Attractions: Museums, mosques, palaces, historical sites
🍽️ Restaurants: Authentic Turkish cuisine, international options
🚇 Transportation: Metro, bus, ferry, tram routes
🏘️ Neighborhoods: Districts, areas, local culture
🎭 Events: Concerts, festivals, cultural activities
💎 Hidden Gems: Local favorites, off-the-beaten-path spots

Guidelines:
1. Provide specific names, locations, and details
2. Use provided database context
3. Include practical info (hours, prices, directions)
4. Be enthusiastic about Istanbul
5. Respond in the same language as the query
6. Never make up information - use context only

Format:
- Start with direct answer
- List 3-5 specific recommendations
- Include practical details
- Add a local tip or insight"""

        self.intent_prompts = {
            'restaurant': """
Focus on restaurants from the provided database context.
Include: name, location, cuisine, price range, rating.
Mention dietary options if relevant.""",

            'attraction': """
Focus on attractions and museums from the provided context.
Include: name, district, description, opening hours, ticket price.
Prioritize based on location and interests.""",

            'transportation': """
Provide clear transportation directions.
Include: metro lines, bus numbers, ferry routes.
Mention transfer points and approximate times.""",

            'neighborhood': """
Describe the neighborhood character and highlights.
Include: atmosphere, best areas, local tips.
Mention nearby attractions and dining.""",

            'events': """
Focus on current and upcoming events.
Include: event name, date, location, price.
Prioritize cultural and authentic experiences.""",

            'general': """
Provide helpful Istanbul travel information.
Draw from all available context.
Be comprehensive but concise."""
        }
    
    async def process_query(
        self,
        query: str,
        user_id: str = "anonymous",
        language: str = "en",
        max_tokens: int = 250
    ) -> Dict[str, Any]:
        """
        Process query using ONLY LLM (no rules)
        
        Pipeline:
        1. Check cache
        2. Detect intent
        3. Build database context
        4. Get RAG embeddings (if available)
        5. Construct prompt
        6. Call RunPod LLM
        7. Cache and return
        
        Args:
            query: User query string
            user_id: User identifier
            language: Response language (en/tr)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with response and metadata
        """
        
        self.stats["total_queries"] += 1
        start_time = datetime.now()
        
        logger.info(f"🔍 Processing query: {query[:50]}...")
        
        # Step 1: Check cache
        cache_key = self._get_cache_key(query, language)
        cached_response = await self._get_cached_response(cache_key)
        
        if cached_response:
            self.stats["cache_hits"] += 1
            logger.info("✅ Cache hit!")
            return cached_response
        
        # Step 2: Detect intent
        intent = self._detect_intent(query)
        logger.info(f"   Intent: {intent}")
        
        # Step 3: Build context from database
        db_context = await self._build_database_context(query, intent)
        logger.info(f"   DB Context: {len(db_context)} chars")
        
        # Step 4: Get RAG context (if available)
        rag_context = await self._get_rag_context(query)
        logger.info(f"   RAG Context: {len(rag_context)} chars")
        
        # Step 5: Build full prompt
        full_prompt = self._build_prompt(
            query=query,
            intent=intent,
            db_context=db_context,
            rag_context=rag_context,
            language=language
        )
        
        # Step 6: Call RunPod LLM
        try:
            self.stats["llm_calls"] += 1
            
            response_data = await self.llm.generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            if not response_data or "generated_text" not in response_data:
                raise Exception("Invalid LLM response")
            
            response_text = response_data["generated_text"]
            
            # Build result
            result = {
                "status": "success",
                "response": response_text,
                "metadata": {
                    "intent": intent,
                    "context_used": bool(db_context),
                    "rag_used": bool(rag_context),
                    "source": "runpod_llm",
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "cached": False
                }
            }
            
            # Step 7: Cache response
            await self._cache_response(cache_key, result)
            
            logger.info(f"✅ Query processed in {result['metadata']['processing_time']:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            self.stats["fallback_calls"] += 1
            
            # Fallback to RAG-only or database context
            return await self._fallback_response(
                query=query,
                intent=intent,
                db_context=db_context,
                rag_context=rag_context
            )
    
    def _detect_intent(self, query: str) -> str:
        """
        Simple keyword-based intent detection
        
        Categories:
        - restaurant: Food, dining, eat
        - attraction: Visit, see, museum
        - transportation: Metro, bus, get to
        - neighborhood: District, area, stay
        - events: Concert, festival, show
        - general: Everything else
        """
        q = query.lower()
        
        # Restaurant keywords
        if any(w in q for w in [
            'restaurant', 'eat', 'food', 'dinner', 'lunch', 'breakfast',
            'cafe', 'coffee', 'cuisine', 'yemek', 'lokanta'
        ]):
            return 'restaurant'
        
        # Attraction keywords
        elif any(w in q for w in [
            'visit', 'see', 'attraction', 'place', 'museum', 'mosque',
            'palace', 'tower', 'church', 'ziyaret', 'gör'
        ]):
            return 'attraction'
        
        # Transportation keywords
        elif any(w in q for w in [
            'metro', 'bus', 'ferry', 'transport', 'get to', 'how to reach',
            'tram', 'istanbulkart', 'otobüs', 'vapur'
        ]):
            return 'transportation'
        
        # Neighborhood keywords
        elif any(w in q for w in [
            'neighborhood', 'district', 'area', 'where to stay', 'region',
            'semt', 'mahalle', 'bölge'
        ]):
            return 'neighborhood'
        
        # Events keywords
        elif any(w in q for w in [
            'event', 'concert', 'show', 'festival', 'activity',
            'etkinlik', 'konser', 'festival'
        ]):
            return 'events'
        
        else:
            return 'general'
    
    async def _build_database_context(
        self,
        query: str,
        intent: str
    ) -> str:
        """Build context from database based on intent"""
        
        context_parts = []
        
        try:
            if intent == 'restaurant':
                context_parts.append(await self._get_restaurant_context(query))
            
            elif intent == 'attraction':
                context_parts.append(await self._get_attraction_context(query))
            
            elif intent == 'transportation':
                context_parts.append(await self._get_transportation_context())
            
            elif intent == 'neighborhood':
                context_parts.append(await self._get_neighborhood_context(query))
            
            elif intent == 'events':
                context_parts.append(await self._get_events_context())
            
            else:
                # General: Include mix of everything
                context_parts.append(await self._get_restaurant_context(query, limit=3))
                context_parts.append(await self._get_attraction_context(query, limit=3))
        
        except Exception as e:
            logger.error(f"Error building database context: {e}")
        
        return "\n\n".join([c for c in context_parts if c])
    
    async def _get_restaurant_context(self, query: str, limit: int = 10) -> str:
        """Get restaurant data from database"""
        try:
            from backend.models import Restaurant
            
            restaurants = self.db.query(Restaurant).limit(limit).all()
            
            if not restaurants:
                return ""
            
            lines = ["**Available Restaurants:**\n"]
            for r in restaurants:
                price_symbols = '₺' * (r.price_level if hasattr(r, 'price_level') else 2)
                lines.append(
                    f"- **{r.name}** ({r.location}): {r.cuisine}, "
                    f"{price_symbols}, Rating: {r.rating}/5"
                )
            
            return "\n".join(lines)
        
        except Exception as e:
            logger.error(f"Error getting restaurant context: {e}")
            return ""
    
    async def _get_attraction_context(self, query: str, limit: int = 10) -> str:
        """Get attraction data from database"""
        try:
            from backend.models import Place
            
            places = self.db.query(Place).limit(limit).all()
            
            if not places:
                return ""
            
            lines = ["**Top Attractions:**\n"]
            for p in places:
                lines.append(
                    f"- **{p.name}** ({p.district}): {p.category}"
                )
            
            return "\n".join(lines)
        
        except Exception as e:
            logger.error(f"Error getting attraction context: {e}")
            return ""
    
    async def _get_transportation_context(self) -> str:
        """Get transportation information"""
        return """**Istanbul Transportation:**

🚇 **Metro Lines:**
- M1: Yenikapı - Atatürk Airport / Kirazlı
- M2: Yenikapı - Hacıosman
- M3: Kirazlı - Olimpiyat / Başakşehir
- M4: Kadıköy - Tavşantepe
- M5: Üsküdar - Çekmeköy
- M7: Mecidiyeköy - Mahmutbey

🚌 **Buses:** Extensive network across all districts
🚢 **Ferries:** Bosphorus crossings every 15-20 minutes
🚊 **Trams:** T1 (Kabataş-Bağcılar), T5 (Cibali-Alibeyköy)
💳 **İstanbulkart:** Required for all public transport"""
    
    async def _get_neighborhood_context(self, query: str) -> str:
        """Get neighborhood information"""
        return """**Popular Istanbul Neighborhoods:**

- **Sultanahmet**: Historic center, major attractions
- **Beyoğlu**: Modern culture, nightlife, İstiklal Street
- **Kadıköy**: Asian side, trendy cafes, local life
- **Beşiktaş**: Waterfront, palaces, markets
- **Nişantaşı**: Upscale shopping, dining
- **Karaköy**: Hipster cafes, art galleries"""
    
    async def _get_events_context(self) -> str:
        """Get events information"""
        try:
            from backend.models import FeedbackEvent
            
            events = self.db.query(FeedbackEvent).limit(5).all()
            
            if not events:
                return "**No upcoming events found in database**"
            
            lines = ["**Upcoming Events:**\n"]
            for e in events:
                lines.append(f"- Event #{e.id}: {e.event_type}")
            
            return "\n".join(lines)
        
        except Exception as e:
            return "**Events information temporarily unavailable**"
    
    async def _get_rag_context(self, query: str) -> str:
        """Get similar queries from RAG system"""
        if not self.rag:
            return ""
        
        try:
            similar_docs = self.rag.search(query, top_k=3)
            if similar_docs:
                return "\n**Similar Queries:**\n" + "\n".join(
                    f"- {doc[:100]}..." for doc in similar_docs
                )
        except Exception as e:
            logger.warning(f"RAG search failed: {e}")
        
        return ""
    
    def _build_prompt(
        self,
        query: str,
        intent: str,
        db_context: str,
        rag_context: str,
        language: str
    ) -> str:
        """Combine all context into final prompt"""
        
        parts = [
            self.base_prompt,
            "",
            self.intent_prompts.get(intent, self.intent_prompts['general'])
        ]
        
        if db_context:
            parts.append(f"\n{db_context}")
        
        if rag_context:
            parts.append(f"\n{rag_context}")
        
        parts.append(f"\n**User Query ({language}):** {query}")
        parts.append("\n**Your Response:**")
        
        return "\n".join(parts)
    
    def _get_cache_key(self, query: str, language: str) -> str:
        """Generate cache key from query"""
        key_string = f"{query.lower().strip()}_{language}"
        return f"llm_response:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached response from Redis"""
        if not self.redis:
            return None
        
        try:
            import json
            cached = self.redis.get(cache_key)
            if cached:
                result = json.loads(cached)
                result["metadata"]["cached"] = True
                return result
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
        
        return None
    
    async def _cache_response(self, cache_key: str, result: Dict[str, Any]):
        """Cache response in Redis"""
        if not self.redis:
            return
        
        try:
            import json
            # Cache for 1 hour
            self.redis.setex(
                cache_key,
                3600,
                json.dumps(result)
            )
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
    
    async def _fallback_response(
        self,
        query: str,
        intent: str,
        db_context: str,
        rag_context: str
    ) -> Dict[str, Any]:
        """Fallback when LLM fails"""
        
        # Use RAG context if available
        if rag_context:
            response = f"Based on similar queries:\n{rag_context}"
        # Use database context
        elif db_context:
            response = f"Here's what I found:\n{db_context[:500]}"
        # Generic fallback
        else:
            response = "I'm currently experiencing technical difficulties. Please try again in a moment or rephrase your question."
        
        return {
            "status": "fallback",
            "response": response,
            "metadata": {
                "intent": intent,
                "context_used": bool(db_context),
                "rag_used": bool(rag_context),
                "source": "fallback",
                "processing_time": 0,
                "cached": False
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        cache_hit_rate = (
            self.stats["cache_hits"] / self.stats["total_queries"] * 100
            if self.stats["total_queries"] > 0
            else 0
        )
        
        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%"
        }

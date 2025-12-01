"""
Priority 3.5: Query Explanation System
Explains detected signals and query understanding to users

Features:
- LLM-based natural language explanations
- Signal confidence scores
- Multi-intent detection explanation
- Multilingual support (EN, TR, etc.)
- Redis caching for repeated explanations
- Debugging and transparency

Updated: December 2024 - Using improved standardized prompt templates

Author: AI Istanbul Team
Date: November 14, 2025
"""

import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

# Import improved prompt templates
from IMPROVED_PROMPT_TEMPLATES import IMPROVED_EXPLANATION_PROMPT

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis not available. Caching disabled.")


class QueryExplainer:
    """
    Explains query understanding and detected signals to users.
    Provides transparency in how the system interprets queries.
    
    Features:
    - Natural language explanations
    - Signal detection breakdown
    - Confidence score interpretation
    - Multi-intent handling
    - Multilingual explanations
    - Caching for performance
    """
    
    def __init__(
        self,
        llm_client: Any,
        redis_client: Optional[redis.Redis] = None,
        cache_ttl: int = 3600,  # 1 hour
        language: str = "en"
    ):
        """
        Initialize Query Explainer.
        
        Args:
            llm_client: LLM client for generating explanations
            redis_client: Optional Redis client for caching
            cache_ttl: Cache TTL in seconds (default: 1 hour)
            language: Default language for explanations (en, tr)
        """
        self.llm_client = llm_client
        self.redis_client = redis_client if REDIS_AVAILABLE else None
        self.cache_ttl = cache_ttl
        self.default_language = language
        
        # Statistics
        self.stats = {
            "total_explanations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0,
            "errors": 0
        }
        
        logger.info("✅ QueryExplainer initialized")
        if self.redis_client:
            logger.info("✅ Redis caching enabled (TTL: %ds)", cache_ttl)
        else:
            logger.warning("⚠️ Redis caching disabled")
    
    def _get_cache_key(self, query: str, signals: Dict[str, Any], language: str) -> str:
        """Generate cache key for explanation."""
        # Include query, signals, and language in cache key
        signals_str = json.dumps(signals, sort_keys=True)
        key_data = f"{query}:{signals_str}:{language}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"query_explanation:{key_hash}"
    
    def _get_cached_explanation(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached explanation from Redis."""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                self.stats["cache_hits"] += 1
                logger.info("✅ Cache hit for explanation")
                explanation = json.loads(cached)
                # Mark as cached
                if "metadata" in explanation:
                    explanation["metadata"]["cached"] = True
                return explanation
            else:
                self.stats["cache_misses"] += 1
                return None
        except Exception as e:
            logger.warning("⚠️ Cache retrieval error: %s", str(e))
            return None
    
    def _cache_explanation(self, cache_key: str, explanation: Dict[str, Any]):
        """Cache explanation in Redis."""
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(explanation)
            )
            logger.info("✅ Cached explanation")
        except Exception as e:
            logger.warning("⚠️ Cache storage error: %s", str(e))
    
    def _build_explanation_prompt(
        self,
        query: str,
        signals: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        language: str
    ) -> str:
        """Build prompt for LLM to generate explanation using improved template."""
        
        # Extract signal information
        detected_signals = signals.get("detected_signals", [])
        confidence_scores = signals.get("confidence_scores", {})
        primary_intent = signals.get("primary_intent", "general")
        primary_confidence = confidence_scores.get(primary_intent, 0.5)
        
        # Build signals summary
        signal_details = []
        for signal in detected_signals:
            confidence = confidence_scores.get(signal, 0.0)
            signal_details.append(f"{signal}: {confidence:.2f}")
        
        signals_summary = ", ".join(signal_details) if signal_details else "no specific signals"
        
        # Use improved explanation prompt template
        prompt = IMPROVED_EXPLANATION_PROMPT.format(
            query=query,
            primary_intent=primary_intent,
            confidence=primary_confidence,
            signals_summary=signals_summary
        )
        
        return prompt
    
    async def explain_query(
        self,
        query: str,
        signals: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for how the query was understood.
        
        Args:
            query: Original user query
            signals: Detected signals and confidence scores
            context: Optional context (conversation history, etc.)
            language: Language for explanation (default: self.default_language)
        
        Returns:
            Dict containing explanation details
        """
        self.stats["total_explanations"] += 1
        lang = language or self.default_language
        
        # Check cache first
        cache_key = self._get_cache_key(query, signals, lang)
        cached = self._get_cached_explanation(cache_key)
        if cached:
            return cached
        
        try:
            # Build prompt
            prompt = self._build_explanation_prompt(query, signals, context, lang)
            
            # Call LLM
            self.stats["llm_calls"] += 1
            start_time = time.time()
            
            response = await self.llm_client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for consistent explanations
                max_tokens=500
            )
            
            llm_time = time.time() - start_time
            
            # Parse response
            explanation_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON
            try:
                # Find JSON in response (handle markdown code blocks)
                if "```json" in explanation_text:
                    json_start = explanation_text.find("```json") + 7
                    json_end = explanation_text.find("```", json_start)
                    explanation_text = explanation_text[json_start:json_end].strip()
                elif "```" in explanation_text:
                    json_start = explanation_text.find("```") + 3
                    json_end = explanation_text.find("```", json_start)
                    explanation_text = explanation_text[json_start:json_end].strip()
                
                explanation = json.loads(explanation_text)
                
                # Add metadata
                explanation["metadata"] = {
                    "language": lang,
                    "llm_time_ms": int(llm_time * 1000),
                    "cached": False,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Cache for future use
                self._cache_explanation(cache_key, explanation)
                
                logger.info(
                    "✅ Generated explanation in %.2fs (lang: %s)",
                    llm_time,
                    lang
                )
                
                return explanation
                
            except json.JSONDecodeError as e:
                logger.error("❌ Failed to parse LLM response as JSON: %s", str(e))
                # Return fallback explanation
                return self._create_fallback_explanation(query, signals, lang, explanation_text)
        
        except Exception as e:
            logger.error("❌ Error generating explanation: %s", str(e))
            self.stats["errors"] += 1
            return self._create_fallback_explanation(query, signals, lang)
    
    def _create_fallback_explanation(
        self,
        query: str,
        signals: Dict[str, Any],
        language: str,
        llm_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create fallback explanation when LLM fails."""
        detected_signals = signals.get("detected_signals", [])
        primary_intent = signals.get("primary_intent", "general")
        
        if language.lower() == "tr":
            summary = f"'{query}' sorgunuzu anladım."
            explanation = "Sorgunuzu işliyorum ve en iyi yanıtı sağlamaya çalışıyorum."
            what_ill_do = "İstanbul hakkında bilgi sağlayacağım."
            confidence = "orta"
        else:
            summary = f"I understood your query about '{query}'."
            explanation = "I'm processing your query and will provide the best answer I can."
            what_ill_do = "I'll provide information about Istanbul."
            confidence = "medium"
        
        return {
            "summary": summary,
            "detected_intents": detected_signals if detected_signals else ["general"],
            "confidence": confidence,
            "explanation": explanation,
            "signals_breakdown": {s: "detected" for s in detected_signals},
            "what_ill_do": what_ill_do,
            "metadata": {
                "language": language,
                "llm_time_ms": 0,
                "cached": False,
                "fallback": True,
                "timestamp": datetime.utcnow().isoformat(),
                "llm_response": llm_response
            }
        }
    
    def explain_signals_simple(
        self,
        signals: Dict[str, Any],
        language: str = "en"
    ) -> str:
        """
        Create simple text explanation of detected signals.
        Non-LLM fallback for quick explanations.
        
        Args:
            signals: Detected signals and confidence scores
            language: Language for explanation
        
        Returns:
            Simple text explanation
        """
        detected_signals = signals.get("detected_signals", [])
        confidence_scores = signals.get("confidence_scores", {})
        primary_intent = signals.get("primary_intent", "general")
        
        if not detected_signals:
            if language.lower() == "tr":
                return "Genel bir soru olarak algıladım."
            else:
                return "I detected this as a general question."
        
        if language.lower() == "tr":
            parts = ["Algılanan amaçlar:"]
            for signal in detected_signals:
                confidence = confidence_scores.get(signal, 0.0)
                conf_level = "yüksek" if confidence > 0.7 else "orta" if confidence > 0.5 else "düşük"
                parts.append(f"  • {signal} ({conf_level} güven)")
            parts.append(f"Ana amaç: {primary_intent}")
        else:
            parts = ["Detected intents:"]
            for signal in detected_signals:
                confidence = confidence_scores.get(signal, 0.0)
                conf_level = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
                parts.append(f"  • {signal} ({conf_level} confidence)")
            parts.append(f"Primary intent: {primary_intent}")
        
        return "\n".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get explanation statistics."""
        total = self.stats["total_explanations"]
        if total > 0:
            cache_hit_rate = (self.stats["cache_hits"] / total) * 100
        else:
            cache_hit_rate = 0.0
        
        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_enabled": self.redis_client is not None
        }
    
    def clear_cache(self) -> int:
        """Clear all cached explanations. Returns count of cleared items."""
        if not self.redis_client:
            return 0
        
        try:
            # Find all explanation cache keys
            keys = []
            cursor = 0
            while True:
                cursor, partial_keys = self.redis_client.scan(
                    cursor,
                    match="query_explanation:*",
                    count=100
                )
                keys.extend(partial_keys)
                if cursor == 0:
                    break
            
            # Delete all keys
            if keys:
                count = self.redis_client.delete(*keys)
                logger.info("✅ Cleared %d cached explanations", count)
                return count
            else:
                logger.info("✅ No cached explanations to clear")
                return 0
        
        except Exception as e:
            logger.error("❌ Error clearing cache: %s", str(e))
            return 0


# Convenience function for simple usage
def create_query_explainer(
    llm_client: Any,
    redis_url: Optional[str] = None,
    language: str = "en"
) -> QueryExplainer:
    """
    Create QueryExplainer with default settings.
    
    Args:
        llm_client: LLM client for generating explanations
        redis_url: Optional Redis URL for caching
        language: Default language (en, tr)
    
    Returns:
        Configured QueryExplainer instance
    """
    redis_client = None
    if redis_url and REDIS_AVAILABLE:
        try:
            redis_client = redis.from_url(redis_url)
            redis_client.ping()
            logger.info("✅ Connected to Redis for query explanations")
        except Exception as e:
            logger.warning("⚠️ Failed to connect to Redis: %s", str(e))
            redis_client = None
    
    return QueryExplainer(
        llm_client=llm_client,
        redis_client=redis_client,
        language=language
    )

"""
Phase 4.4: Suggestion Generator

LLM-powered generation of proactive suggestions based on conversation context.
This module gives the LLM 95% control over creating relevant, helpful suggestions.

The generator handles:
1. Generating contextually relevant suggestions
2. Ranking suggestions by usefulness
3. Ensuring diversity across suggestion types
4. Formatting suggestions naturally

Author: AI Istanbul Team
Date: December 3, 2025
"""

import json
import logging
import time
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import (
    SuggestionContext,
    ProactiveSuggestion,
    ProactiveSuggestionResponse
)
from .llm_response_parser import parse_llm_json_response

logger = logging.getLogger(__name__)


class SuggestionGenerator:
    """
    LLM-powered suggestion generator.
    
    Generates relevant, actionable suggestions based on conversation context.
    The LLM creates natural, contextually appropriate suggestions that feel
    like helpful recommendations from a knowledgeable travel guide.
    
    LLM Responsibility: 95%
    - Suggestion generation
    - Relevance scoring
    - Natural language phrasing
    - Context adaptation
    
    Fallback: Template-based suggestions for common scenarios
    """
    
    def __init__(self, llm_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the suggestion generator.
        
        Args:
            llm_client: LLM API client (RunPod, OpenAI, etc.)
            config: Optional configuration overrides
        """
        self.llm_client = llm_client
        
        # Configuration
        self.config = {
            'use_llm': True,  # NEW: Enable/disable LLM for suggestions
            'timeout_seconds': 4,
            'temperature': 0.8,  # Higher for creative suggestions
            'max_tokens': 800,
            'max_suggestions': 5,
            'min_diversity_score': 0.6,
            'fallback_enabled': True,
            **(config or {})
        }
        
        logger.info(f"SuggestionGenerator initialized (LLM: {self.config['use_llm']})")
    
    async def generate_suggestions(
        self,
        context: SuggestionContext,
        max_suggestions: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> List[ProactiveSuggestion]:
        """
        Generate proactive suggestions based on context.
        
        Args:
            context: Analyzed conversation context
            max_suggestions: Maximum number of suggestions to generate
            categories: Preferred suggestion categories (if any)
            
        Returns:
            List of ProactiveSuggestion objects, ranked by relevance
        """
        start_time = time.time()
        max_count = max_suggestions or self.config['max_suggestions']
        
        try:
            # Check if LLM is disabled - use templates only
            if not self.config.get('use_llm', True):
                logger.info("Using template-based suggestions only (LLM disabled)")
                suggestions_data = self._generate_template_suggestions(context, max_count)
            else:
                # Generate suggestions using LLM with timeout protection
                try:
                    suggestions_data = await asyncio.wait_for(
                        self._generate_suggestions_llm(context, max_count, categories),
                        timeout=self.config['timeout_seconds']
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"LLM suggestion generation timed out after {self.config['timeout_seconds']}s, using templates")
                    suggestions_data = self._generate_template_suggestions(context, max_count)
            
            # Convert to ProactiveSuggestion objects
            suggestions = []
            for i, data in enumerate(suggestions_data):
                suggestion = ProactiveSuggestion(
                    suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                    suggestion_text=data['text'],
                    suggestion_type=data['type'],
                    intent_type=data['intent'],
                    entities=data.get('entities', {}),
                    relevance_score=data.get('relevance', 0.8),
                    priority=max_count - i,  # Higher priority = shown first
                    reasoning=data.get('reasoning'),
                    icon=data.get('icon'),
                    action_type=data.get('action_type', 'query')
                )
                suggestions.append(suggestion)
            
            # Rank and filter
            suggestions = await self.rank_suggestions(suggestions, context)
            suggestions = suggestions[:max_count]
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Generated {len(suggestions)} suggestions "
                f"({processing_time:.0f}ms)"
            )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}, using template fallback")
            return self._template_suggestions(context, max_count, start_time)
    
    async def rank_suggestions(
        self,
        suggestions: List[ProactiveSuggestion],
        context: SuggestionContext
    ) -> List[ProactiveSuggestion]:
        """
        Rank suggestions by relevance and diversity.
        
        Args:
            suggestions: List of suggestions to rank
            context: Conversation context
            
        Returns:
            Ranked list of suggestions
        """
        if not suggestions:
            return []
        
        # Check if LLM ranking is disabled
        if not self.config.get('use_llm', True):
            logger.debug("Skipping LLM ranking (use_llm=False), sorting by relevance")
            # Just sort by relevance score
            suggestions.sort(key=lambda s: s.relevance_score, reverse=True)
            return suggestions
        
        try:
            # Use LLM for intelligent ranking
            ranked_data = await self._rank_suggestions_llm(suggestions, context)
            
            # Update priorities based on LLM ranking
            for i, sugg_id in enumerate(ranked_data['ranked_ids']):
                for suggestion in suggestions:
                    if suggestion.suggestion_id == sugg_id:
                        suggestion.priority = len(suggestions) - i
                        break
            
            # Sort by priority
            suggestions.sort(key=lambda s: s.priority, reverse=True)
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"LLM ranking failed: {e}, using relevance scores")
            # Fallback: sort by relevance score
            suggestions.sort(key=lambda s: s.relevance_score, reverse=True)
            return suggestions
    
    async def generate_with_response(
        self,
        context: SuggestionContext,
        max_suggestions: Optional[int] = None
    ) -> ProactiveSuggestionResponse:
        """
        Generate suggestions and return complete response.
        
        Args:
            context: Conversation context
            max_suggestions: Maximum suggestions
            
        Returns:
            ProactiveSuggestionResponse with metadata
        """
        start_time = time.time()
        
        suggestions = await self.generate_suggestions(context, max_suggestions)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate diversity score
        diversity = self._calculate_diversity(suggestions)
        
        return ProactiveSuggestionResponse(
            suggestions=suggestions,
            context=context,
            generation_method="llm" if suggestions and suggestions[0].reasoning else "template",
            generation_time_ms=processing_time,
            total_suggestions_considered=len(suggestions) * 2,  # Estimate
            confidence=context.trigger_confidence,
            diversity_score=diversity,
            timestamp=datetime.now(),
            llm_used=bool(suggestions and suggestions[0].reasoning)
        )
    
    async def _generate_suggestions_llm(
        self,
        context: SuggestionContext,
        max_count: int,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Call LLM to generate suggestions (wrapped)
        If LLM fails, fall back to templates.
        """
        try:
            # Call to LLM client (async)
            prompt = self._build_llm_prompt(context, max_count, categories)
            llm_response = await self.llm_client.completions.create(
                model=self.llm_client.default_model,
                prompt=prompt,
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature']
            )
            parsed = parse_llm_json_response(llm_response)
            return parsed.get('suggestions', [])
        except Exception as e:
            logger.error(f"LLM generation failed in _generate_suggestions_llm: {e}")
            return self._generate_template_suggestions(context, max_count)

    def _generate_template_suggestions(self, context: SuggestionContext, max_count: int) -> List[Dict[str, Any]]:
        """
        Context-aware suggestion generator (fallback)
        Returns list of dicts with keys: text, type, intent, entities, relevance
        
        Suggestions are highly relevant to:
        1. The current query topic (restaurants → more restaurant suggestions)
        2. Detected intents (transportation → route-related follow-ups)
        3. Extracted entities (mentioned places → suggestions about those places)
        4. Language (all 6 languages: EN, TR, RU, DE, AR, FR)
        """
        language = getattr(context, 'language', 'en') or 'en'
        query_lower = context.current_query.lower() if context.current_query else ""
        response_type = context.response_type or "general"
        detected_intents = context.detected_intents or []
        entities = context.extracted_entities or {}
        
        # Extract location/place names from entities or query
        mentioned_place = entities.get('destination') or entities.get('location') or entities.get('neighborhood')
        origin_place = entities.get('origin')
        
        # CONTEXT-AWARE SUGGESTION TEMPLATES BY CATEGORY
        suggestions_by_category = self._get_contextual_suggestions_by_language(language, mentioned_place)
        
        # Determine which categories to prioritize based on context
        selected_suggestions = []
        
        # 🔥 PRIORITY 1: Use detected intent (most reliable)
        # If intent is transportation, always show transportation-related suggestions
        if response_type == 'transportation' or 'transportation' in detected_intents:
            selected_suggestions.extend(suggestions_by_category.get('after_transportation', []))
            # Add destination-specific suggestion if we know the destination
            if mentioned_place and language == 'en':
                selected_suggestions.insert(0, {
                    "text": f"What can I do in {mentioned_place}?",
                    "type": "exploration",
                    "intent": "needs_attraction",
                    "relevance": 0.98
                })
                selected_suggestions.insert(1, {
                    "text": f"Good restaurants near {mentioned_place}",
                    "type": "practical",
                    "intent": "needs_restaurant",
                    "relevance": 0.95
                })
            elif mentioned_place and language == 'tr':
                selected_suggestions.insert(0, {
                    "text": f"{mentioned_place}'da ne yapabilirim?",
                    "type": "exploration",
                    "intent": "needs_attraction",
                    "relevance": 0.98
                })
                selected_suggestions.insert(1, {
                    "text": f"{mentioned_place} yakınında iyi restoranlar",
                    "type": "practical",
                    "intent": "needs_restaurant",
                    "relevance": 0.95
                })
        
        # 2. RESTAURANT CONTEXT - user asked about food/restaurants
        elif any(word in query_lower for word in ['restaurant', 'food', 'eat', 'yemek', 'restoran', 'ресторан', 'еда', 'essen', 'مطعم', 'طعام', 'manger']):
            selected_suggestions.extend(suggestions_by_category.get('after_restaurant', []))
        
        # 3. ATTRACTION CONTEXT - user asked about places to visit
        elif any(word in query_lower for word in ['visit', 'see', 'attraction', 'museum', 'mosque', 'gezilecek', 'müze', 'cami', 'достопримечательност', 'музей', 'sehenswürdigkeit', 'معلم', 'متحف', 'visite', 'musée']):
            selected_suggestions.extend(suggestions_by_category.get('after_attraction', []))
        
        # 4. TRANSPORTATION CONTEXT (keyword fallback) - user asked about routes/directions
        elif any(word in query_lower for word in ['how to get', 'route', 'metro', 'bus', 'tram', 'ferry', 'from', 'to', 'directions', 'nasıl gid', 'ulaşım', 'как добраться', 'метро', 'wie komme', 'كيف أصل', 'مترو', 'comment aller', 'transport']):
            selected_suggestions.extend(suggestions_by_category.get('after_transportation', []))
        
        # 5. NEIGHBORHOOD CONTEXT - user asked about a specific area
        elif any(word in query_lower for word in ['neighborhood', 'area', 'district', 'semt', 'mahalle', 'район', 'viertel', 'حي', 'quartier']) or mentioned_place:
            selected_suggestions.extend(suggestions_by_category.get('after_neighborhood', []))
        
        # 6. HIDDEN GEMS CONTEXT - user asked about local/hidden places
        elif any(word in query_lower for word in ['hidden', 'local', 'secret', 'off the beaten', 'gizli', 'yerel', 'скрыт', 'местн', 'geheim', 'lokal', 'مخفي', 'محلي', 'caché', 'local']):
            selected_suggestions.extend(suggestions_by_category.get('after_hidden_gems', []))
        
        # 7. WEATHER CONTEXT - user asked about weather
        elif any(word in query_lower for word in ['weather', 'hava', 'погода', 'wetter', 'طقس', 'météo', 'rain', 'yağmur', 'дождь', 'regen', 'مطر', 'pluie']):
            selected_suggestions.extend(suggestions_by_category.get('after_weather', []))
        
        # 8. GREETING/GENERAL - user just said hello or general question
        elif any(word in query_lower for word in ['hello', 'hi', 'merhaba', 'selam', 'привет', 'hallo', 'مرحبا', 'bonjour', 'salut']):
            selected_suggestions.extend(suggestions_by_category.get('after_greeting', []))
        
        # 9. DEFAULT - general helpful suggestions
        else:
            selected_suggestions.extend(suggestions_by_category.get('general', []))
        
        # Add variety - mix in one or two from other categories if we have room
        if len(selected_suggestions) < max_count:
            other_categories = ['general', 'after_attraction', 'after_restaurant']
            for cat in other_categories:
                if cat not in [response_type]:
                    extras = suggestions_by_category.get(cat, [])
                    for s in extras:
                        if s not in selected_suggestions and len(selected_suggestions) < max_count + 2:
                            s_copy = s.copy()
                            s_copy['relevance'] = s_copy.get('relevance', 0.7) - 0.1  # Lower priority
                            selected_suggestions.append(s_copy)
        
        # Sort by relevance and return top-k
        selected_suggestions.sort(key=lambda x: x.get('relevance', 0.5), reverse=True)
        return selected_suggestions[:max_count]
    
    def _get_contextual_suggestions_by_language(self, language: str, mentioned_place: str = None) -> Dict[str, List[Dict]]:
        """Get context-aware suggestions for each language"""
        
        place = mentioned_place or "Sultanahmet"
        
        # ENGLISH suggestions
        if language == 'en':
            return {
                'after_restaurant': [
                    {"text": "Show me more restaurants nearby", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": "Find restaurants with a view", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "What's the best street food around here?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Are there any rooftop restaurants?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "Show me cafes for Turkish breakfast", "type": "dining", "intent": "needs_restaurant", "relevance": 0.75},
                ],
                'after_attraction': [
                    {"text": "What else is near this place?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "How do I get there?", "type": "practical", "intent": "needs_transportation", "relevance": 0.9},
                    {"text": "Are there good restaurants nearby?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "What's the best time to visit?", "type": "practical", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Show me hidden gems in this area", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                ],
                'after_transportation': [
                    {"text": "Show me an alternative route", "type": "practical", "intent": "needs_transportation", "relevance": 0.95},
                    {"text": "What can I see along the way?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.92},
                    {"text": "How much will the trip cost?", "type": "practical", "intent": "needs_transportation", "relevance": 0.88},
                    {"text": "Where can I buy an Istanbulkart?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": "Is there a scenic route option?", "type": "exploration", "intent": "needs_transportation", "relevance": 0.82},
                ],
                'after_neighborhood': [
                    {"text": f"Best restaurants in {place}", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": f"Hidden gems in {place}", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.9},
                    {"text": f"How do I get to {place}?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": f"What's {place} known for?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Show me nearby neighborhoods", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_hidden_gems': [
                    {"text": "Show me more hidden gems", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.95},
                    {"text": "Any local-only restaurants?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Secret viewpoints in Istanbul?", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "Off-the-beaten-path neighborhoods", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.8},
                    {"text": "How do I get to this place?", "type": "practical", "intent": "needs_transportation", "relevance": 0.75},
                ],
                'after_weather': [
                    {"text": "What can I do indoors today?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Best museums to visit", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Cozy cafes for rainy days", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "What's the forecast for tomorrow?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Indoor shopping options", "type": "exploration", "intent": "needs_shopping", "relevance": 0.75},
                ],
                'after_greeting': [
                    {"text": "What should I see in Istanbul?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Best restaurants in Istanbul", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Show me hidden gems", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "How's the weather today?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Plan my day in Istanbul", "type": "exploration", "intent": "needs_trip_planning", "relevance": 0.75},
                ],
                'general': [
                    {"text": "Top attractions in Istanbul", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Best local restaurants", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "How do I get around?", "type": "practical", "intent": "needs_transportation", "relevance": 0.8},
                    {"text": "Hidden gems to explore", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                    {"text": "What's happening this weekend?", "type": "cultural", "intent": "needs_events", "relevance": 0.7},
                ],
            }
        
        # TURKISH suggestions
        elif language == 'tr':
            return {
                'after_restaurant': [
                    {"text": "Yakında başka restoranlar göster", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": "Manzaralı restoranlar var mı?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "En iyi sokak yemekleri nerede?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Teras restoranları göster", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "Kahvaltı mekanları öner", "type": "dining", "intent": "needs_restaurant", "relevance": 0.75},
                ],
                'after_attraction': [
                    {"text": "Bu yerin yakınında ne var?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Oraya nasıl giderim?", "type": "practical", "intent": "needs_transportation", "relevance": 0.9},
                    {"text": "Yakında güzel restoranlar var mı?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "En iyi ziyaret zamanı ne?", "type": "practical", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Bu bölgedeki gizli mekanlar", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                ],
                'after_transportation': [
                    {"text": "Alternatif bir rota göster", "type": "practical", "intent": "needs_transportation", "relevance": 0.95},
                    {"text": "Yol üzerinde ne görebilirim?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.92},
                    {"text": "Bu yolculuk kaça mal olur?", "type": "practical", "intent": "needs_transportation", "relevance": 0.88},
                    {"text": "İstanbulkart nereden alırım?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": "Manzaralı bir rota var mı?", "type": "exploration", "intent": "needs_transportation", "relevance": 0.82},
                ],
                'after_neighborhood': [
                    {"text": f"{place} en iyi restoranlar", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": f"{place} gizli mekanlar", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.9},
                    {"text": f"{place}'e nasıl gidilir?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": f"{place} neyle ünlü?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Yakın semtleri göster", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_hidden_gems': [
                    {"text": "Daha fazla gizli mekan göster", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.95},
                    {"text": "Turistlerin bilmediği restoranlar", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Gizli manzara noktaları", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "Keşfedilmemiş semtler", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.8},
                    {"text": "Bu yere nasıl gidilir?", "type": "practical", "intent": "needs_transportation", "relevance": 0.75},
                ],
                'after_weather': [
                    {"text": "Bugün kapalı mekanlarda ne yapabilirim?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "En iyi müzeler hangileri?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Yağmurlu günler için kafeler", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Yarın hava nasıl olacak?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Kapalı alışveriş merkezleri", "type": "exploration", "intent": "needs_shopping", "relevance": 0.75},
                ],
                'after_greeting': [
                    {"text": "İstanbul'da ne görmeliyim?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "En iyi restoranlar nereler?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Gizli mekanları göster", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "Bugün hava nasıl?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Günümü planla", "type": "exploration", "intent": "needs_trip_planning", "relevance": 0.75},
                ],
                'general': [
                    {"text": "İstanbul'un en güzel yerleri", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "En iyi yerel restoranlar", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Şehirde nasıl dolaşırım?", "type": "practical", "intent": "needs_transportation", "relevance": 0.8},
                    {"text": "Keşfedilecek gizli yerler", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                    {"text": "Bu hafta sonu ne var?", "type": "cultural", "intent": "needs_events", "relevance": 0.7},
                ],
            }
        
        # RUSSIAN suggestions
        elif language == 'ru':
            return {
                'after_restaurant': [
                    {"text": "Покажи ещё рестораны рядом", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": "Рестораны с видом на город", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Где лучшая уличная еда?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Есть рестораны на крыше?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "Кафе с турецким завтраком", "type": "dining", "intent": "needs_restaurant", "relevance": 0.75},
                ],
                'after_attraction': [
                    {"text": "Что ещё рядом с этим местом?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Как туда добраться?", "type": "practical", "intent": "needs_transportation", "relevance": 0.9},
                    {"text": "Есть рестораны поблизости?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Лучшее время для посещения?", "type": "practical", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Скрытые места в этом районе", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                ],
                'after_transportation': [
                    {"text": "Есть более быстрый маршрут?", "type": "practical", "intent": "needs_transportation", "relevance": 0.95},
                    {"text": "Что посмотреть по пути?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Где купить Истанбулкарт?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": "Рестораны у места назначения", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "Что стоит посетить рядом?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_neighborhood': [
                    {"text": f"Лучшие рестораны в {place}", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": f"Скрытые места в {place}", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.9},
                    {"text": f"Как добраться до {place}?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": f"Чем известен {place}?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Покажи соседние районы", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_hidden_gems': [
                    {"text": "Покажи ещё скрытые места", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.95},
                    {"text": "Рестораны только для местных", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Секретные смотровые площадки", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "Нетуристические районы", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.8},
                    {"text": "Как добраться до этого места?", "type": "practical", "intent": "needs_transportation", "relevance": 0.75},
                ],
                'after_weather': [
                    {"text": "Что делать в помещении сегодня?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Лучшие музеи для посещения", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Уютные кафе в дождливый день", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Какой прогноз на завтра?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Крытые торговые центры", "type": "exploration", "intent": "needs_shopping", "relevance": 0.75},
                ],
                'after_greeting': [
                    {"text": "Что посмотреть в Стамбуле?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Лучшие рестораны Стамбула", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Покажи скрытые места", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "Какая сегодня погода?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Спланируй мой день", "type": "exploration", "intent": "needs_trip_planning", "relevance": 0.75},
                ],
                'general': [
                    {"text": "Главные достопримечательности", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Лучшие местные рестораны", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Как передвигаться по городу?", "type": "practical", "intent": "needs_transportation", "relevance": 0.8},
                    {"text": "Скрытые жемчужины города", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                    {"text": "Что интересного на выходных?", "type": "cultural", "intent": "needs_events", "relevance": 0.7},
                ],
            }
        
        # GERMAN suggestions  
        elif language == 'de':
            return {
                'after_restaurant': [
                    {"text": "Zeige mehr Restaurants in der Nähe", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": "Restaurants mit Aussicht", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Wo gibt es das beste Street Food?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Gibt es Dachterrassen-Restaurants?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "Cafés für türkisches Frühstück", "type": "dining", "intent": "needs_restaurant", "relevance": 0.75},
                ],
                'after_attraction': [
                    {"text": "Was gibt es noch in der Nähe?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Wie komme ich dorthin?", "type": "practical", "intent": "needs_transportation", "relevance": 0.9},
                    {"text": "Gibt es gute Restaurants in der Nähe?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Beste Besuchszeit?", "type": "practical", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Geheimtipps in dieser Gegend", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                ],
                'after_transportation': [
                    {"text": "Gibt es eine schnellere Route?", "type": "practical", "intent": "needs_transportation", "relevance": 0.95},
                    {"text": "Was kann ich unterwegs sehen?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Wo kaufe ich eine Istanbulkart?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": "Restaurants am Zielort?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "Sehenswürdigkeiten in der Nähe", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_neighborhood': [
                    {"text": f"Beste Restaurants in {place}", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": f"Geheimtipps in {place}", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.9},
                    {"text": f"Wie komme ich nach {place}?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": f"Wofür ist {place} bekannt?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Zeige Nachbarviertel", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_hidden_gems': [
                    {"text": "Zeige mehr Geheimtipps", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.95},
                    {"text": "Restaurants nur für Einheimische", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Geheime Aussichtspunkte", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "Unentdeckte Viertel", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.8},
                    {"text": "Wie komme ich zu diesem Ort?", "type": "practical", "intent": "needs_transportation", "relevance": 0.75},
                ],
                'after_weather': [
                    {"text": "Was kann ich heute drinnen machen?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Beste Museen zum Besuchen", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Gemütliche Cafés für Regentage", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Wie wird das Wetter morgen?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Überdachte Einkaufsmöglichkeiten", "type": "exploration", "intent": "needs_shopping", "relevance": 0.75},
                ],
                'after_greeting': [
                    {"text": "Was sollte ich in Istanbul sehen?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Beste Restaurants in Istanbul", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Zeige mir Geheimtipps", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "Wie ist das Wetter heute?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Plane meinen Tag", "type": "exploration", "intent": "needs_trip_planning", "relevance": 0.75},
                ],
                'general': [
                    {"text": "Top-Sehenswürdigkeiten in Istanbul", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Beste lokale Restaurants", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Wie komme ich herum?", "type": "practical", "intent": "needs_transportation", "relevance": 0.8},
                    {"text": "Geheimtipps zum Entdecken", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                    {"text": "Was passiert am Wochenende?", "type": "cultural", "intent": "needs_events", "relevance": 0.7},
                ],
            }
        
        # ARABIC suggestions
        elif language == 'ar':
            return {
                'after_restaurant': [
                    {"text": "أرني المزيد من المطاعم القريبة", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": "مطاعم بإطلالة جميلة", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "أين أفضل طعام الشارع؟", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "هل هناك مطاعم على السطح؟", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "مقاهي للفطور التركي", "type": "dining", "intent": "needs_restaurant", "relevance": 0.75},
                ],
                'after_attraction': [
                    {"text": "ماذا يوجد بالقرب من هذا المكان؟", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "كيف أصل إلى هناك؟", "type": "practical", "intent": "needs_transportation", "relevance": 0.9},
                    {"text": "هل توجد مطاعم قريبة؟", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "ما أفضل وقت للزيارة؟", "type": "practical", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "أماكن مخفية في هذه المنطقة", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                ],
                'after_transportation': [
                    {"text": "هل يوجد طريق أسرع؟", "type": "practical", "intent": "needs_transportation", "relevance": 0.95},
                    {"text": "ماذا يمكنني رؤيته في الطريق؟", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "أين أشتري بطاقة إسطنبول؟", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": "مطاعم عند الوجهة", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "ما يستحق الزيارة قريباً", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_neighborhood': [
                    {"text": f"أفضل مطاعم في {place}", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": f"أماكن مخفية في {place}", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.9},
                    {"text": f"كيف أصل إلى {place}؟", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": f"بماذا يشتهر {place}؟", "type": "exploration", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "أرني الأحياء المجاورة", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_hidden_gems': [
                    {"text": "أرني المزيد من الأماكن المخفية", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.95},
                    {"text": "مطاعم للسكان المحليين فقط", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "نقاط مشاهدة سرية", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "أحياء غير سياحية", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.8},
                    {"text": "كيف أصل لهذا المكان؟", "type": "practical", "intent": "needs_transportation", "relevance": 0.75},
                ],
                'after_weather': [
                    {"text": "ماذا أفعل في الداخل اليوم؟", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "أفضل المتاحف للزيارة", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "مقاهي مريحة للأيام الممطرة", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "ما توقعات الطقس غداً؟", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "خيارات التسوق المغطاة", "type": "exploration", "intent": "needs_shopping", "relevance": 0.75},
                ],
                'after_greeting': [
                    {"text": "ماذا يجب أن أزور في اسطنبول؟", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "أفضل مطاعم اسطنبول", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "أرني الأماكن المخفية", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "كيف الطقس اليوم؟", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "خطط يومي في اسطنبول", "type": "exploration", "intent": "needs_trip_planning", "relevance": 0.75},
                ],
                'general': [
                    {"text": "أهم المعالم في اسطنبول", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "أفضل المطاعم المحلية", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "كيف أتنقل في المدينة؟", "type": "practical", "intent": "needs_transportation", "relevance": 0.8},
                    {"text": "جواهر مخفية للاستكشاف", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                    {"text": "ماذا يحدث هذا الأسبوع؟", "type": "cultural", "intent": "needs_events", "relevance": 0.7},
                ],
            }
        
        # FRENCH suggestions
        elif language == 'fr':
            return {
                'after_restaurant': [
                    {"text": "Montre-moi plus de restaurants à proximité", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": "Restaurants avec vue", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Où trouver la meilleure street food?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Y a-t-il des restaurants sur les toits?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "Cafés pour petit-déjeuner turc", "type": "dining", "intent": "needs_restaurant", "relevance": 0.75},
                ],
                'after_attraction': [
                    {"text": "Qu'y a-t-il d'autre près d'ici?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Comment y aller?", "type": "practical", "intent": "needs_transportation", "relevance": 0.9},
                    {"text": "Y a-t-il de bons restaurants à proximité?", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Meilleur moment pour visiter?", "type": "practical", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Trésors cachés dans ce quartier", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                ],
                'after_transportation': [
                    {"text": "Y a-t-il un itinéraire plus rapide?", "type": "practical", "intent": "needs_transportation", "relevance": 0.95},
                    {"text": "Que voir en chemin?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Où acheter une Istanbulkart?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": "Restaurants près de ma destination", "type": "dining", "intent": "needs_restaurant", "relevance": 0.8},
                    {"text": "Que vaut-il la peine de visiter?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_neighborhood': [
                    {"text": f"Meilleurs restaurants à {place}", "type": "dining", "intent": "needs_restaurant", "relevance": 0.95},
                    {"text": f"Trésors cachés à {place}", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.9},
                    {"text": f"Comment aller à {place}?", "type": "practical", "intent": "needs_transportation", "relevance": 0.85},
                    {"text": f"Pour quoi {place} est-il connu?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.8},
                    {"text": "Montre les quartiers voisins", "type": "exploration", "intent": "needs_attraction", "relevance": 0.75},
                ],
                'after_hidden_gems': [
                    {"text": "Montre plus de trésors cachés", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.95},
                    {"text": "Restaurants réservés aux locaux", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Points de vue secrets", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "Quartiers hors des sentiers battus", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.8},
                    {"text": "Comment aller à cet endroit?", "type": "practical", "intent": "needs_transportation", "relevance": 0.75},
                ],
                'after_weather': [
                    {"text": "Que faire à l'intérieur aujourd'hui?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Meilleurs musées à visiter", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Cafés cosy pour les jours de pluie", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Prévisions météo pour demain?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Options shopping couvertes", "type": "exploration", "intent": "needs_shopping", "relevance": 0.75},
                ],
                'after_greeting': [
                    {"text": "Que voir à Istanbul?", "type": "exploration", "intent": "needs_attraction", "relevance": 0.95},
                    {"text": "Meilleurs restaurants d'Istanbul", "type": "dining", "intent": "needs_restaurant", "relevance": 0.9},
                    {"text": "Montre-moi les trésors cachés", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.85},
                    {"text": "Quel temps fait-il aujourd'hui?", "type": "practical", "intent": "needs_weather", "relevance": 0.8},
                    {"text": "Planifie ma journée", "type": "exploration", "intent": "needs_trip_planning", "relevance": 0.75},
                ],
                'general': [
                    {"text": "Meilleures attractions d'Istanbul", "type": "exploration", "intent": "needs_attraction", "relevance": 0.9},
                    {"text": "Meilleurs restaurants locaux", "type": "dining", "intent": "needs_restaurant", "relevance": 0.85},
                    {"text": "Comment se déplacer?", "type": "practical", "intent": "needs_transportation", "relevance": 0.8},
                    {"text": "Trésors cachés à découvrir", "type": "exploration", "intent": "needs_hidden_gems", "relevance": 0.75},
                    {"text": "Que se passe-t-il ce week-end?", "type": "cultural", "intent": "needs_events", "relevance": 0.7},
                ],
            }
        
        # Default to English
        else:
            return self._get_contextual_suggestions_by_language('en', mentioned_place)
    
    async def _rank_suggestions_llm(
        self,
        suggestions: List[ProactiveSuggestion],
        context: SuggestionContext
    ) -> Dict[str, Any]:
        """
        Use LLM to rank suggestions.
        
        Args:
            suggestions: Suggestions to rank
            context: Conversation context
            
        Returns:
            Ranking data with ordered IDs
        """
        prompt = self._build_ranking_prompt(suggestions, context)
        llm_output = await self._call_llm(prompt)
        
        # Parse JSON response using robust parser
        from .llm_response_parser import parse_llm_json_response
        
        data = parse_llm_json_response(llm_output, fallback_value={})
        if not data:
            raise ValueError("LLM returned empty or invalid response")
        
        return data
    
    def _build_generation_prompt(
        self,
        context: SuggestionContext,
        max_count: int,
        categories: Optional[List[str]] = None
    ) -> str:
        """Build prompt for suggestion generation."""
        
        entities_text = ""
        if context.extracted_entities:
            entities_text = f"\nExtracted Entities:\n{json.dumps(context.extracted_entities, indent=2)}"
        
        categories_text = ""
        if categories:
            categories_text = f"\nPrefer these categories: {', '.join(categories)}"
        
        return f"""Generate {max_count} proactive, helpful suggestions for a travel chatbot user:

Current Query: "{context.current_query}"
Response Given: "{context.current_response[:400]}..."
Response Type: {context.response_type}
Detected Intents: {', '.join(context.detected_intents) if context.detected_intents else 'none'}
{entities_text}
{categories_text}

{self._get_generation_instructions()}

Return ONLY valid JSON with exactly {max_count} suggestions."""
    
    def _build_ranking_prompt(
        self,
        suggestions: List[ProactiveSuggestion],
        context: SuggestionContext
    ) -> str:
        """Build prompt for suggestion ranking."""
        
        suggestions_text = "\n".join([
            f"{i+1}. [{s.suggestion_id}] {s.suggestion_text} ({s.suggestion_type})"
            for i, s in enumerate(suggestions)
        ])
        
        return f"""Rank these suggestions by relevance and usefulness:

Query: "{context.current_query}"
Response Type: {context.response_type}

Suggestions:
{suggestions_text}

Rank them from most to least relevant considering:
1. Natural next step for user
2. Relevance to current context
3. Practical value
4. Diversity (don't rank all same type highly)

Respond with JSON:
{{
  "ranked_ids": ["sugg_abc123", "sugg_def456", ...],
  "reasoning": "brief explanation of ranking"
}}"""
    
    def _get_generation_instructions(self) -> str:
        """Get instructions for suggestion generation."""
        return """You are a helpful Istanbul travel guide. Generate suggestions that feel natural and helpful.

SUGGESTION TYPES:
- exploration: "Discover hidden gems nearby", "See other attractions in this area"
- practical: "Get directions", "Check the weather", "Find transportation options"
- cultural: "See cultural events tonight", "Learn about local customs"
- dining: "Find restaurants nearby", "Try local specialties"
- refinement: "Filter by price range", "Show only outdoor activities"

SUGGESTION GUIDELINES:
1. Be specific and actionable (not vague like "learn more")
2. Use natural language ("Get directions to these restaurants" not "directions_query")
3. Include relevant emojis (🗺️ 🍽️ 🎭 🌤️ 💎 🏛️)
4. Ensure diversity - don't repeat similar suggestions
5. Make each suggestion feel valuable
6. Consider what a helpful human guide would suggest

INTENT MAPPING:
Map each suggestion to an intent:
- "Get directions to X" → intent: "get_directions"
- "Find restaurants near X" → intent: "find_restaurant"
- "Check weather for X" → intent: "get_weather"
- "See attractions in X" → intent: "find_attraction"
- "Discover hidden gems in X" → intent: "find_hidden_gems"
- "See events in X" → intent: "find_events"

Respond with JSON:
{
  "suggestions": [
    {
      "text": "Get directions to these restaurants",
      "type": "practical",
      "intent": "get_directions",
      "relevance": 0.95,
      "reasoning": "User asked about restaurants, likely wants to visit them",
      "icon": "🗺️",
      "action_type": "query",
      "entities": {"destination": "Sultanahmet"}
    },
    ...
  ]
}

IMPORTANT:
- Generate exactly the requested number of suggestions
- Ensure high diversity across types
- Make suggestions feel personal and contextual
- Each suggestion should be genuinely helpful"""
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM via existing client.
        
        Args:
            prompt: Prompt to send
            
        Returns:
            LLM response text
        """
        try:
            # Check if client has OpenAI-style interface
            if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                # OpenAI-style client
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert Istanbul travel guide creating helpful suggestions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config['temperature'],
                    timeout=self.config['timeout_seconds']
                )
                return response.choices[0].message.content.strip()
            else:
                # Generic LLM client with generate method
                return await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=self.config['max_tokens'],
                    temperature=self.config['temperature']
                )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _template_suggestions(
        self,
        context: SuggestionContext,
        max_count: int,
        start_time: float
    ) -> List[ProactiveSuggestion]:
        """
        Fallback template-based suggestions.
        
        Uses predefined templates based on response type.
        
        Args:
            context: Conversation context
            max_count: Maximum suggestions
            start_time: Start time for metrics
            
        Returns:
            List of template-based suggestions
        """
        templates = self._get_suggestion_templates(context.response_type)
        
        suggestions = []
        for i, template in enumerate(templates[:max_count]):
            # Fill template with context entities
            text = template['text']
            entities = {}
            
            # Extract location from entities
            if context.extracted_entities:
                location = (
                    context.extracted_entities.get('destination') or
                    context.extracted_entities.get('location') or
                    context.extracted_entities.get('origin')
                )
                if location:
                    text = text.replace("{location}", location)
                    entities['location'] = location
            
            suggestion = ProactiveSuggestion(
                suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                suggestion_text=text,
                suggestion_type=template['type'],
                intent_type=template['intent'],
                entities=entities,
                relevance_score=0.7 - (i * 0.05),  # Decreasing relevance
                priority=max_count - i,
                reasoning=None,  # No reasoning for templates
                icon=template.get('icon'),
                action_type='query'
            )
            suggestions.append(suggestion)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Generated {len(suggestions)} template suggestions ({processing_time:.0f}ms)")
        
        return suggestions
    
    def _get_suggestion_templates(self, response_type: str) -> List[Dict[str, Any]]:
        """
        Get template suggestions for a response type.
        
        Args:
            response_type: Type of response
            
        Returns:
            List of suggestion templates
        """
        templates_map = {
            'restaurant': [
                {'text': 'Get directions to these restaurants', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Check the weather forecast', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
                {'text': 'Find attractions nearby', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
                {'text': 'Discover hidden gems in {location}', 'type': 'exploration', 'intent': 'find_hidden_gems', 'icon': '💎'},
                {'text': 'See cultural events tonight', 'type': 'cultural', 'intent': 'find_events', 'icon': '🎭'},
            ],
            'attraction': [
                {'text': 'Get directions to this attraction', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Find restaurants nearby', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See other attractions in the area', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
                {'text': 'Check opening hours and prices', 'type': 'practical', 'intent': 'get_information', 'icon': '🎫'},
                {'text': 'Discover hidden gems nearby', 'type': 'exploration', 'intent': 'find_hidden_gems', 'icon': '💎'},
            ],
            'route': [
                {'text': 'Find restaurants along the route', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See attractions near your destination', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
                {'text': 'Check the weather for your journey', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
                {'text': 'Find alternative transportation', 'type': 'practical', 'intent': 'get_transport', 'icon': '🚇'},
                {'text': 'Save this route for later', 'type': 'refinement', 'intent': 'save_route', 'icon': '💾'},
            ],
            'museum': [
                {'text': 'Get directions to this museum', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Check current exhibitions', 'type': 'cultural', 'intent': 'get_information', 'icon': '🎨'},
                {'text': 'Find restaurants nearby', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See other museums in the area', 'type': 'exploration', 'intent': 'find_museum', 'icon': '🏛️'},
                {'text': 'Plan a full day museum tour', 'type': 'exploration', 'intent': 'plan_tour', 'icon': '📍'},
            ],
            'event': [
                {'text': 'Get directions to this event', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Find restaurants near the venue', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See other events tonight', 'type': 'cultural', 'intent': 'find_events', 'icon': '🎭'},
                {'text': 'Check the weather forecast', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
                {'text': 'Find nearby parking', 'type': 'practical', 'intent': 'find_parking', 'icon': '🅿️'},
            ],
            'hidden_gems': [
                {'text': 'Get directions to these places', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Find restaurants in the area', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See nearby attractions', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
                {'text': 'Check the weather', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
                {'text': 'Plan a walking tour', 'type': 'exploration', 'intent': 'plan_tour', 'icon': '🚶'},
            ],
        }
        
        # Default suggestions for unknown types
        default = [
            {'text': 'Find restaurants nearby', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
            {'text': 'See nearby attractions', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
            {'text': 'Check the weather', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
            {'text': 'Get directions', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
            {'text': 'Discover hidden gems', 'type': 'exploration', 'intent': 'find_hidden_gems', 'icon': '💎'},
        ]
        
        return templates_map.get(response_type, default)
    
    def _calculate_diversity(self, suggestions: List[ProactiveSuggestion]) -> float:
        """
        Calculate diversity score for suggestions.
        
        Args:
            suggestions: List of suggestions
            
        Returns:
            Diversity score (0-1)
        """
        if not suggestions:
            return 0.0
        
        # Count unique types
        types = set(s.suggestion_type for s in suggestions)
        intents = set(s.intent_type for s in suggestions)
        
        # Diversity based on unique types and intents
        type_diversity = len(types) / len(suggestions)
        intent_diversity = len(intents) / len(suggestions)
        
        return (type_diversity + intent_diversity) / 2


# Singleton instance
_generator_instance: Optional[SuggestionGenerator] = None


def get_suggestion_generator(llm_client, config: Optional[Dict[str, Any]] = None) -> SuggestionGenerator:
    """
    Get or create the singleton SuggestionGenerator instance.
    
    Args:
        llm_client: LLM API client
        config: Optional configuration
        
    Returns:
        SuggestionGenerator instance
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SuggestionGenerator(llm_client=llm_client, config=config)
    return _generator_instance


async def generate_proactive_suggestions(
    context: SuggestionContext,
    max_suggestions: Optional[int] = None,
    llm_client=None
) -> ProactiveSuggestionResponse:
    """
    Convenience function to generate suggestions.
    
    Args:
        context: Conversation context
        max_suggestions: Maximum suggestions
        llm_client: LLM client (uses singleton if provided once)
        
    Returns:
        ProactiveSuggestionResponse with suggestions
    """
    if llm_client:
        generator = get_suggestion_generator(llm_client)
    else:
        generator = _generator_instance
        if generator is None:
            raise ValueError("SuggestionGenerator not initialized. Provide llm_client.")
    
    return await generator.generate_with_response(context, max_suggestions)

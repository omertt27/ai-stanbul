"""
Prompt Builder
System prompts, templates, and prompt assembly

Responsibilities:
- System prompts
- Intent-specific prompts
- Context injection
- Prompt assembly

Extracted from: pure_llm_handler.py (_load_prompts, _build_system_prompt, _build_prompt_with_signals)

Author: Istanbul AI Team
Date: November 14, 2025
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds LLM prompts from templates and context
    
    Features:
    - System prompts
    - Intent-specific prompts
    - Context formatting
    - Multi-language support
    """
    
    def __init__(self):
        """Initialize prompt builder"""
        self._load_prompts()
        
        logger.info("📝 Prompt builder initialized")
    
    def _load_prompts(self):
        """
        Load Istanbul-specific system prompts
        
        Extracted from pure_llm_handler._load_prompts()
        """
        self.base_prompt = """You are AI Istanbul, an expert travel assistant for Istanbul, Turkey.

You have deep knowledge of:
🏛️ Attractions: Museums, mosques, palaces, historical sites
🍽️ Restaurants: Authentic Turkish cuisine, international options
🚇 Transportation: Metro, bus, ferry, tram routes
🏘️ Neighborhoods: Districts, areas, local culture
🎭 Events: Concerts, festivals, cultural activities
💎 Hidden Gems: Local favorites, off-the-beaten-path spots

CRITICAL LANGUAGE RULES:
🔴 NEVER mix languages in your response
🔴 Keep the ENTIRE response in ONE language
🔴 Match the language of the user's query
🔴 If query is Turkish, respond 100% in Turkish
🔴 If query is English, respond 100% in English
🔴 Keep place names in original (e.g., "Sultanahmet", "Beyoğlu")
🔴 Do NOT translate proper nouns (restaurant/place names)

Response Guidelines:
1. Provide specific names, locations, and details
2. Use provided database context
3. Include practical info (hours, prices, directions)
4. Be enthusiastic about Istanbul
5. Respond in the SAME LANGUAGE as the query (100% consistency)
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
Mention transfer points and approximate times.
If available, include a map visualization link.""",

            'neighborhood': """
Describe the neighborhood character and highlights.
Include: atmosphere, best areas, local tips.
Mention nearby attractions and dining.""",

            'events': """
Focus on current and upcoming events.
Include: event name, date, location, price.
Prioritize cultural and authentic experiences.""",

            'weather': """
Provide weather-aware recommendations.
Include current conditions and activity suggestions.
Recommend indoor options for bad weather, outdoor for good weather.""",

            'hidden_gems': """
Focus on local secrets and off-the-beaten-path spots.
Include authentic experiences away from tourist crowds.
Mention accessibility and best times to visit.""",

            'general': """
Provide helpful Istanbul travel information.
Draw from all available context.
Be comprehensive but concise."""
        }
    
    def build_system_prompt(self, signals: Dict[str, bool]) -> str:
        """
        Build signal-aware system prompt
        
        Args:
            signals: Detected service signals
            
        Returns:
            System prompt string with relevant intent prompts
        """
        # Start with base prompt
        prompt = self.base_prompt + "\n\n"
        
        # Add intent-specific prompts based on signals
        if signals.get('likely_restaurant'):
            prompt += self.intent_prompts['restaurant'] + "\n"
        
        if signals.get('likely_attraction'):
            prompt += self.intent_prompts['attraction'] + "\n"
        
        if signals.get('needs_map') or signals.get('needs_gps_routing'):
            prompt += self.intent_prompts['transportation'] + "\n"
        
        if signals.get('needs_weather'):
            prompt += self.intent_prompts['weather'] + "\n"
        
        if signals.get('needs_events'):
            prompt += self.intent_prompts['events'] + "\n"
        
        if signals.get('needs_hidden_gems'):
            prompt += self.intent_prompts['hidden_gems'] + "\n"
        
        # If no specific signals, use general
        if not any(signals.values()):
            prompt += self.intent_prompts['general'] + "\n"
        
        return prompt
    
    def build_prompt_with_signals(
        self,
        query: str,
        signals: Dict[str, bool],
        system_prompt: str,
        db_context: str = "",
        rag_context: str = "",
        weather_context: str = "",
        events_context: str = "",
        hidden_gems_context: str = "",
        language: str = "en"
    ) -> str:
        """
        Build complete prompt with all contexts
        
        Args:
            query: User query
            signals: Detected signals
            system_prompt: System prompt string
            db_context: Database context
            rag_context: RAG context
            weather_context: Weather context
            events_context: Events context
            hidden_gems_context: Hidden gems context
            language: Response language
            
        Returns:
            Complete formatted prompt
        """
        prompt_parts = [system_prompt]
        
        # Add strong language instruction
        lang_names = {
            "en": "English",
            "tr": "Turkish",
            "ar": "Arabic",
            "de": "German",
            "ru": "Russian",
            "fr": "French"
        }
        lang_name = lang_names.get(language, "English")
        
        # Always add explicit language instruction
        prompt_parts.append(f"""
╔════════════════════════════════════════════════════╗
║  CRITICAL: LANGUAGE CONSISTENCY RULE               ║
║  ✅ Respond ONLY in {lang_name}                     ║
║  ❌ Do NOT mix languages                            ║
║  ❌ Do NOT use English words in {lang_name} response║
║  ❌ Do NOT translate names (keep original)         ║
║  ✅ Use {lang_name} throughout entire response     ║
╚════════════════════════════════════════════════════╝
""")
        
        # Add database context
        if db_context:
            prompt_parts.append(f"\n---DATABASE CONTEXT---\n{db_context}\n")
        
        # Add RAG context
        if rag_context:
            prompt_parts.append(f"\n---SIMILAR QUERIES CONTEXT---\n{rag_context}\n")
        
        # Add weather context
        if weather_context:
            prompt_parts.append(f"\n---WEATHER CONTEXT---\n{weather_context}\n")
        
        # Add events context
        if events_context:
            prompt_parts.append(f"\n---EVENTS CONTEXT---\n{events_context}\n")
        
        # Add hidden gems context
        if hidden_gems_context:
            prompt_parts.append(f"\n---HIDDEN GEMS CONTEXT---\n{hidden_gems_context}\n")
        
        # Add user query
        prompt_parts.append(f"\n---USER QUERY---\n{query}\n")
        
        # Add response instruction
        prompt_parts.append("\n---YOUR RESPONSE---\n")
        
        return "\n".join(prompt_parts)

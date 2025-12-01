"""
Prompt Builder
System prompts, templates, and prompt assembly

Responsibilities:
- System prompts
- Intent-specific prompts
- Context injection
- Prompt assembly

Updated: December 2024 - Using improved standardized prompt templates

Author: Istanbul AI Team
Date: November 14, 2025
"""

import logging
from typing import Dict, Any

# Import improved prompt templates
from IMPROVED_PROMPT_TEMPLATES import (
    IMPROVED_BASE_PROMPT,
    INTENT_PROMPTS
)

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
        Load Istanbul-specific system prompts using improved templates
        
        Updated to use IMPROVED_PROMPT_TEMPLATES
        """
        # Use the improved base prompt (will be formatted with language at runtime)
        # Store the template, not a formatted version
        self.base_prompt_template = IMPROVED_BASE_PROMPT
        
        # Use improved intent prompts
        self.intent_prompts = INTENT_PROMPTS
        
        logger.info("📝 Loaded improved prompt templates")
    
    def build_system_prompt(self, signals: Dict[str, bool], language: str = "English") -> str:
        """
        Build signal-aware system prompt using improved templates
        
        Args:
            signals: Detected service signals
            language: Detected language for the response
            
        Returns:
            System prompt string with relevant intent prompts
        """
        # Start with base prompt formatted with detected language
        prompt = self.base_prompt_template.format(detected_language=language)
        prompt += "\n\n"
        
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
            system_prompt: System prompt string (already formatted with language)
            db_context: Database context
            rag_context: RAG context
            weather_context: Weather context
            events_context: Events context
            hidden_gems_context: Hidden gems context
            language: Response language code
            
        Returns:
            Complete formatted prompt
        """
        prompt_parts = [system_prompt]
        
        # Language instruction is now in the base prompt template
        # No need for redundant language enforcement box
        
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
        
        # Add response instruction with language
        lang_names = {
            "en": "English",
            "tr": "Turkish (Türkçe)",
            "ar": "Arabic (العربية)",
            "de": "German (Deutsch)",
            "ru": "Russian (Русский)",
            "fr": "French (Français)"
        }
        lang_name = lang_names.get(language, "English")
        prompt_parts.append(f"\n---YOUR RESPONSE (in {lang_name})---\n")
        
        return "\n".join(prompt_parts)

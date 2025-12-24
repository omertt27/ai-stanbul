"""
Prompt Builder (Legacy Wrapper)
Delegates to services/llm/prompts.py PromptBuilder

This file exists for backwards compatibility with older code.
All new code should use: from services.llm.prompts import PromptBuilder

Author: Istanbul AI Team
Updated: December 2024 - Now delegates to unified PromptBuilder
"""

import logging
from typing import Dict, Any, Optional

# Import the MAIN PromptBuilder from prompts.py
from services.llm.prompts import PromptBuilder as MainPromptBuilder

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Legacy wrapper for PromptBuilder.
    
    Delegates to services.llm.prompts.PromptBuilder for all operations.
    This class exists for backwards compatibility.
    
    For new code, use:
        from services.llm.prompts import PromptBuilder
    """
    
    def __init__(self):
        """Initialize prompt builder by delegating to main PromptBuilder"""
        self._main_builder = MainPromptBuilder()
        logger.info("📝 Legacy PromptBuilder wrapper initialized (delegates to prompts.py)")
    
    def build_system_prompt(self, signals: Dict[str, bool], language: str = "English") -> str:
        """
        Build signal-aware system prompt.
        
        Args:
            signals: Detected service signals
            language: Detected language for the response
            
        Returns:
            System prompt string
        """
        # Map language name to code
        lang_map = {
            "English": "en",
            "Turkish": "tr",
            "German": "de",
            "Russian": "ru",
            "Arabic": "ar"
        }
        lang_code = lang_map.get(language, "en")
        
        # Get system prompt from main builder
        return self._main_builder.system_prompts.get(lang_code, self._main_builder.system_prompts['en'])
    
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
        Build complete prompt with all contexts.
        
        Delegates to main PromptBuilder.build_prompt()
        """
        # Build context dict for main builder
        context = {
            'database': db_context,
            'rag': rag_context,
            'services': {}
        }
        
        if weather_context:
            context['services']['weather'] = weather_context
        if events_context:
            context['services']['events'] = events_context
        if hidden_gems_context:
            context['services']['hidden_gems'] = hidden_gems_context
        
        # Use main builder
        return self._main_builder.build_prompt(
            query=query,
            signals=signals,
            context=context,
            language=language
        )

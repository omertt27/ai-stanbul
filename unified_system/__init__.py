"""
Unified System Package
Central integration point for all LLM operations in Istanbul AI

Combines:
- RunPod LLM Client (backend/services/runpod_llm_client.py)
- Prompt Builder (backend/services/llm/prompts.py)
- Shared caching layer
- Unified metrics collection

Author: Istanbul AI Team
Date: January 17, 2026
"""

from .services.unified_llm_service import UnifiedLLMService, get_unified_llm

__all__ = ['UnifiedLLMService', 'get_unified_llm']
__version__ = '1.0.0'

"""
Handler LLM Mixin - Unified LLM Integration for Handlers

Provides a consistent interface for all Istanbul AI handlers to use UnifiedLLMService
with automatic fallback to legacy llm_service if UnifiedLLMService is not available.

Features:
- Automatic UnifiedLLMService initialization
- Feature flag support (USE_UNIFIED_LLM)
- Backward compatible with existing llm_service
- Component-level metrics tracking
- Transparent caching and circuit breaker protection

Usage:
    class MyHandler(HandlerLLMMixin):
        def __init__(self, llm_service=None, **kwargs):
            self.llm_service = llm_service
            self._init_handler_llm()
            
        def my_method(self):
            response = self._llm_generate(
                prompt="...",
                component="my_handler.my_method",
                max_tokens=200,
                temperature=0.7
            )
"""

import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class HandlerLLMMixin:
    """
    Mixin class to provide UnifiedLLMService integration for handlers.
    
    This mixin provides a consistent _llm_generate() method that automatically
    routes LLM calls through UnifiedLLMService when available, with fallback
    to the legacy llm_service.
    """
    
    def _init_handler_llm(self):
        """
        Initialize UnifiedLLMService for handler use.
        
        Call this in your handler's __init__ after setting self.llm_service.
        Feature flag: USE_UNIFIED_LLM (default: true)
        """
        self.unified_llm = None
        self.use_unified_llm = os.getenv('USE_UNIFIED_LLM', 'true').lower() == 'true'
        
        if not self.use_unified_llm:
            logger.info(f"   UnifiedLLM disabled via feature flag for {self.__class__.__name__}")
            return
        
        try:
            # Import and get singleton instance of UnifiedLLMService
            from unified_system.services.unified_llm_service import get_unified_llm
            
            # Get singleton instance (no parameters needed)
            self.unified_llm = get_unified_llm()
            
            logger.info(f"✅ {self.__class__.__name__}: UnifiedLLMService initialized")
            
        except Exception as e:
            logger.warning(
                f"⚠️ {self.__class__.__name__}: Failed to initialize UnifiedLLMService, "
                f"falling back to legacy llm_service: {e}"
            )
            self.unified_llm = None
    
    def _llm_generate(
        self,
        prompt: str,
        component: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate LLM response with automatic routing.
        
        Routes through UnifiedLLMService if available, falls back to legacy llm_service.
        
        Args:
            prompt: The prompt to send to the LLM
            component: Component name for metrics tracking (e.g., "food_handler.street_food")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters to pass to the LLM
        
        Returns:
            Generated text response
        
        Raises:
            Exception: If both UnifiedLLMService and llm_service fail
        """
        # Try UnifiedLLMService first
        if self.unified_llm is not None:
            try:
                # Use the synchronous wrapper method
                response = self.unified_llm.complete_text_sync(
                    prompt=prompt,
                    component=component,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response
                
            except Exception as e:
                logger.warning(
                    f"⚠️ UnifiedLLMService failed for {component}, "
                    f"falling back to legacy: {e}"
                )
                # Fall through to legacy service
            except Exception as e:
                logger.warning(
                    f"⚠️ UnifiedLLMService failed for {component}, "
                    f"falling back to legacy: {e}"
                )
                # Fall through to legacy service
        
        # Fallback to legacy llm_service
        if hasattr(self, 'llm_service') and self.llm_service is not None:
            try:
                # Legacy service uses .generate() method
                response = self.llm_service.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                return response
                
            except Exception as e:
                logger.error(f"❌ Legacy llm_service failed for {component}: {e}")
                raise
        
        # No LLM service available
        raise RuntimeError(
            f"No LLM service available for {component}. "
            "Both UnifiedLLMService and legacy llm_service are unavailable."
        )
    
    def _llm_generate_with_context(
        self,
        user_query: str,
        rag_context: str,
        system_prompt: str,
        component: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate LLM response with RAG context.
        
        Convenience method for handlers that use RAG context.
        
        Args:
            user_query: User's question
            rag_context: Retrieved context from RAG service
            system_prompt: System instructions for the LLM
            component: Component name for metrics
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional LLM parameters
        
        Returns:
            Generated response
        """
        # Build complete prompt
        prompt_parts = []
        
        if system_prompt:
            prompt_parts.append(system_prompt)
        
        if rag_context:
            prompt_parts.append(f"\nCONTEXT:\n{rag_context}")
        
        prompt_parts.append(f"\nUSER QUESTION: {user_query}")
        prompt_parts.append("\nRESPONSE:")
        
        full_prompt = "\n".join(prompt_parts)
        
        return self._llm_generate(
            prompt=full_prompt,
            component=component,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def _get_llm_health(self) -> Dict[str, Any]:
        """
        Get health status of the LLM service.
        
        Returns:
            Health status dict with availability and service type
        """
        if self.unified_llm is not None:
            try:
                health = self.unified_llm.get_health()
                health['service_type'] = 'UnifiedLLMService'
                return health
            except Exception as e:
                return {
                    'available': False,
                    'service_type': 'UnifiedLLMService',
                    'error': str(e)
                }
        
        if hasattr(self, 'llm_service') and self.llm_service is not None:
            return {
                'available': True,
                'service_type': 'Legacy LLM Service',
                'backend': 'unknown'
            }
        
        return {
            'available': False,
            'service_type': 'None',
            'error': 'No LLM service initialized'
        }
    
    def _get_llm_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get LLM metrics if UnifiedLLMService is available.
        
        Returns:
            Metrics dict or None if UnifiedLLMService not available
        """
        if self.unified_llm is not None:
            try:
                return self.unified_llm.get_metrics()
            except Exception as e:
                logger.warning(f"Failed to get LLM metrics: {e}")
                return None
        
        return None

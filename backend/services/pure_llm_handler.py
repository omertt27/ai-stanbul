"""
LEGACY COMPATIBILITY WRAPPER
This file maintains backward compatibility with the old PureLLMHandler API
while using the new modular system under the hood.

MIGRATION NOTE: This is a thin wrapper around the new modular system.
New code should import from backend.services.llm instead:
    from backend.services.llm import create_pure_llm_core

For existing code using PureLLMHandler, this wrapper provides a drop-in replacement.

Author: Istanbul AI Team
Date: January 2025
"""

import logging
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

# Import the new modular system
from backend.services.llm import (
    create_pure_llm_core,
    PureLLMCore,
    SignalDetector,
    ContextBuilder,
    PromptBuilder,
    AnalyticsManager,
    CacheManager,
    QueryEnhancer,
    ConversationManager,
    ExperimentationManager
)

logger = logging.getLogger(__name__)


class PureLLMHandler:
    """
    LEGACY WRAPPER for backward compatibility.
    
    This class wraps the new modular PureLLMCore system and maintains
    the same API as the original monolithic handler.
    
    New projects should use create_pure_llm_core() directly instead.
    """
    
    def __init__(
        self,
        db: Session,
        rag_service: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        enable_cache: bool = True,
        enable_analytics: bool = True,
        enable_experimentation: bool = False,
        enable_conversation: bool = True,
        enable_query_enhancement: bool = True
    ):
        """
        Initialize the legacy handler wrapper.
        
        Args:
            db: Database session
            rag_service: Optional RAG service for retrieval
            redis_client: Optional Redis client for caching
            enable_cache: Enable response caching
            enable_analytics: Enable analytics tracking
            enable_experimentation: Enable A/B testing
            enable_conversation: Enable conversation management
            enable_query_enhancement: Enable query rewriting
        """
        logger.info("🔄 Initializing PureLLMHandler (legacy wrapper)")
        
        # Store configuration
        self.db = db
        self.rag_service = rag_service
        self.redis_client = redis_client
        
        # Create the modular core system
        self.core = create_pure_llm_core(
            db=db,
            rag_service=rag_service,
            redis_client=redis_client,
            enable_cache=enable_cache,
            enable_analytics=enable_analytics,
            enable_experimentation=enable_experimentation,
            enable_conversation=enable_conversation,
            enable_query_enhancement=enable_query_enhancement
        )
        
        # Expose module instances for direct access (backward compatibility)
        self.signals = self.core.signals
        self.context = self.core.context
        self.prompts = self.core.prompts
        self.analytics = self.core.analytics if enable_analytics else None
        self.cache = self.core.cache if enable_cache else None
        self.query_enhancer = self.core.query_enhancer if enable_query_enhancement else None
        self.conversation = self.core.conversation if enable_conversation else None
        self.experimentation = self.core.experimentation if enable_experimentation else None
        
        logger.info("✅ PureLLMHandler wrapper initialized with modular system")
    
    async def process_query(
        self,
        query: str,
        user_id: Optional[int] = None,
        language: str = "en",
        include_map: bool = False,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user query (main entry point).
        
        This is the primary method that maintains backward compatibility
        with the original PureLLMHandler API.
        
        Args:
            query: User's question/query
            user_id: Optional user ID for personalization
            language: Target language for response
            include_map: Whether to include map visualization
            session_id: Optional session ID for conversation tracking
            **kwargs: Additional parameters for processing
        
        Returns:
            Dictionary containing response, signals, metadata, etc.
        """
        return await self.core.process_query(
            query=query,
            user_id=user_id,
            language=language,
            include_map=include_map,
            session_id=session_id,
            **kwargs
        )
    
    def detect_signals(
        self,
        query: str,
        language: str = "en",
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect signals in a query (legacy method).
        
        Args:
            query: User query text
            language: Query language
            threshold: Minimum confidence threshold
        
        Returns:
            Dictionary with detected signals and metadata
        """
        return self.signals.detect_signals(
            query=query,
            language=language,
            threshold=threshold
        )
    
    def build_context(
        self,
        query: str,
        signals: Dict[str, Any],
        user_id: Optional[int] = None
    ) -> str:
        """
        Build context for LLM from query and signals (legacy method).
        
        Args:
            query: User query
            signals: Detected signals
            user_id: Optional user ID
        
        Returns:
            Formatted context string
        """
        return self.context.build_context(
            query=query,
            signals=signals,
            user_id=user_id,
            db=self.db
        )
    
    def format_prompt(
        self,
        query: str,
        context: str,
        language: str = "en",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format the final prompt for LLM (legacy method).
        
        Args:
            query: User query
            context: Built context
            language: Target language
            conversation_history: Optional conversation history
        
        Returns:
            Formatted prompt string
        """
        return self.prompts.format_prompt(
            query=query,
            context=context,
            language=language,
            conversation_history=conversation_history
        )
    
    def get_analytics(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get analytics data (legacy method).
        
        Args:
            user_id: Optional user ID to filter analytics
        
        Returns:
            Dictionary with analytics metrics
        """
        if not self.analytics:
            return {"error": "Analytics not enabled"}
        
        return self.analytics.get_analytics(user_id=user_id)
    
    def clear_cache(self, pattern: str = "*") -> int:
        """
        Clear cached responses (legacy method).
        
        Args:
            pattern: Redis key pattern to clear
        
        Returns:
            Number of keys cleared
        """
        if not self.cache:
            return 0
        
        return self.cache.clear_cache(pattern=pattern)
    
    def get_conversation_history(
        self,
        session_id: str,
        max_turns: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session (legacy method).
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of turns to return
        
        Returns:
            List of conversation turns
        """
        if not self.conversation:
            return []
        
        return self.conversation.get_history(
            session_id=session_id,
            max_turns=max_turns
        )
    
    def start_experiment(
        self,
        experiment_id: str,
        variants: Dict[str, float]
    ) -> bool:
        """
        Start an A/B test experiment (legacy method).
        
        Args:
            experiment_id: Unique experiment identifier
            variants: Dictionary mapping variant names to traffic percentages
        
        Returns:
            True if experiment started successfully
        """
        if not self.experimentation:
            return False
        
        return self.experimentation.start_experiment(
            experiment_id=experiment_id,
            variants=variants
        )
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get results for an A/B test experiment (legacy method).
        
        Args:
            experiment_id: Experiment identifier
        
        Returns:
            Dictionary with experiment results and metrics
        """
        if not self.experimentation:
            return {"error": "Experimentation not enabled"}
        
        return self.experimentation.get_results(experiment_id=experiment_id)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check health status of all components.
        
        Returns:
            Dictionary with health status of each module
        """
        return {
            "status": "healthy",
            "modules": {
                "core": "initialized",
                "signals": "active" if self.signals else "disabled",
                "context": "active" if self.context else "disabled",
                "prompts": "active" if self.prompts else "disabled",
                "analytics": "active" if self.analytics else "disabled",
                "cache": "active" if self.cache else "disabled",
                "query_enhancer": "active" if self.query_enhancer else "disabled",
                "conversation": "active" if self.conversation else "disabled",
                "experimentation": "active" if self.experimentation else "disabled"
            },
            "database": "connected" if self.db else "disconnected",
            "rag_service": "connected" if self.rag_service else "not configured",
            "redis_client": "connected" if self.redis_client else "not configured"
        }
    
    def __repr__(self) -> str:
        """String representation of the handler."""
        return f"<PureLLMHandler (legacy wrapper) - modular system active>"


# Export for backward compatibility
__all__ = ["PureLLMHandler"]


# Migration helper logging
logger.info("=" * 70)
logger.info("LEGACY WRAPPER LOADED: backend.services.pure_llm_handler")
logger.info("This is a compatibility wrapper around the new modular system.")
logger.info("New code should import from: backend.services.llm")
logger.info("=" * 70)

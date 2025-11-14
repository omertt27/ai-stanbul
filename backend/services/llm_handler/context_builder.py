"""
Context Builder
Database and RAG context retrieval and formatting

Responsibilities:
- Database context retrieval
- RAG context integration
- Smart context selection based on signals
- Context formatting for LLM

Author: Istanbul AI Team
Date: November 14, 2025
"""

import logging
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds intelligent context for LLM prompts
    
    Features:
    - Signal-based context selection
    - Database query optimization
    - RAG integration
    - Context formatting
    """
    
    def __init__(self, db_session: Session, rag_service=None):
        """
        Initialize context builder
        
        Args:
            db_session: SQLAlchemy database session
            rag_service: RAG vector service (optional)
        """
        self.db = db_session
        self.rag = rag_service
        
        # Statistics
        self.stats = {
            "db_queries": 0,
            "rag_queries": 0,
            "context_built": 0,
            "errors": 0
        }
        
        logger.info("✅ Context builder initialized")
        logger.info(f"   RAG service: {'✅ Enabled' if self.rag else '❌ Disabled'}")
    
    async def build_context(
        self,
        query: str,
        signals: Dict[str, bool]
    ) -> str:
        """
        Build smart context based on signals.
        
        Selectively retrieves context from database based on detected intents:
        - Restaurant queries: restaurant data
        - Attraction queries: attraction data
        - Events queries: event data
        - Hidden gems queries: hidden gem data
        - Budget queries: price-filtered data
        
        Args:
            query: User query
            signals: Detected signals
            
        Returns:
            Formatted context string
        """
        try:
            self.stats["context_built"] += 1
            
            context_parts = []
            
            # Restaurant context
            if signals.get('likely_restaurant', False):
                try:
                    from backend.models import Restaurant
                    restaurants = self.db.query(Restaurant).limit(20).all()
                    
                    if restaurants:
                        context_parts.append("=== Available Restaurants ===")
                        for r in restaurants:
                            context_parts.append(
                                f"- {r.name}: {r.cuisine_type} | "
                                f"District: {getattr(r, 'district', 'N/A')} | "
                                f"Price: {getattr(r, 'price_level', 'N/A')}"
                            )
                        
                        self.stats["db_queries"] += 1
                        logger.debug(f"   Added {len(restaurants)} restaurants to context")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to fetch restaurant context: {e}")
                    self.stats["errors"] += 1
            
            # Attraction context
            if signals.get('likely_attraction', False):
                try:
                    from backend.models import Attraction
                    attractions = self.db.query(Attraction).limit(20).all()
                    
                    if attractions:
                        context_parts.append("\n=== Popular Attractions ===")
                        for a in attractions:
                            context_parts.append(
                                f"- {a.name}: {getattr(a, 'category', 'N/A')} | "
                                f"District: {getattr(a, 'district', 'N/A')}"
                            )
                        
                        self.stats["db_queries"] += 1
                        logger.debug(f"   Added {len(attractions)} attractions to context")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to fetch attraction context: {e}")
                    self.stats["errors"] += 1
            
            # Events context
            if signals.get('needs_events', False):
                try:
                    from backend.models import Event
                    events = self.db.query(Event).limit(15).all()
                    
                    if events:
                        context_parts.append("\n=== Current Events ===")
                        for e in events:
                            context_parts.append(
                                f"- {e.title}: {getattr(e, 'date', 'N/A')} | "
                                f"Location: {getattr(e, 'location', 'N/A')}"
                            )
                        
                        self.stats["db_queries"] += 1
                        logger.debug(f"   Added {len(events)} events to context")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to fetch events context: {e}")
                    self.stats["errors"] += 1
            
            # Hidden gems context
            if signals.get('needs_hidden_gems', False):
                try:
                    from backend.models import HiddenGem
                    hidden_gems = self.db.query(HiddenGem).limit(15).all()
                    
                    if hidden_gems:
                        context_parts.append("\n=== Hidden Gems ===")
                        for hg in hidden_gems:
                            context_parts.append(
                                f"- {hg.name}: {getattr(hg, 'description', 'N/A')[:100]}"
                            )
                        
                        self.stats["db_queries"] += 1
                        logger.debug(f"   Added {len(hidden_gems)} hidden gems to context")
                except Exception as e:
                    logger.warning(f"   ⚠️ Failed to fetch hidden gems context: {e}")
                    self.stats["errors"] += 1
            
            # Return combined context
            return "\n".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.error(f"❌ Context building failed: {e}")
            self.stats["errors"] += 1
            return ""
    
    async def get_rag_context(self, query: str) -> str:
        """
        Get RAG vector search context.
        
        Searches for similar queries and responses in vector database.
        
        Args:
            query: User query
            
        Returns:
            RAG context string
        """
        if not self.rag:
            return ""
        
        try:
            self.stats["rag_queries"] += 1
            
            # Search for similar queries
            results = await self.rag.search(query, top_k=5)
            
            if not results:
                return ""
            
            context_parts = ["\n=== Similar Past Queries ==="]
            for i, result in enumerate(results, 1):
                context_parts.append(
                    f"{i}. Q: {result.get('query', 'N/A')}\n"
                    f"   A: {result.get('response', 'N/A')[:200]}..."
                )
            
            logger.debug(f"   Added {len(results)} RAG results to context")
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"   ⚠️ RAG context retrieval failed: {e}")
            self.stats["errors"] += 1
            return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get context builder statistics.
        
        Returns:
            Dict with performance metrics
        """
        return {
            "db_queries": self.stats["db_queries"],
            "rag_queries": self.stats["rag_queries"],
            "context_built": self.stats["context_built"],
            "errors": self.stats["errors"],
            "rag_enabled": self.rag is not None
        }

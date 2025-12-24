"""
Context Assembly Layer

This layer sits between context retrieval and prompt construction.
It provides:
- Context selection (what to include)
- Context ranking (what matters most)
- Context compression/summarization
- Freshness checks (timestamps)

Instead of: Prompt = system + RAG + map + history + rules
We do:     Prompt = system + curated_context(reason-aware)

This reduces hallucinations and removes the "template feel".
"""

import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context that can be assembled"""
    RAG = "rag"                      # Retrieved documents
    DATABASE = "database"            # Structured data
    TRANSPORTATION = "transportation" # Routes, transit info
    WEATHER = "weather"              # Weather data
    RESTAURANT = "restaurant"        # Dining recommendations
    ATTRACTION = "attraction"        # Tourist attractions
    REALTIME = "realtime"            # Live data (traffic, etc.)
    CONVERSATION = "conversation"    # Chat history


@dataclass
class GroundedContext:
    """
    Context with explicit grounding contract.
    
    This ensures the model knows:
    - Where the data came from
    - How confident we are
    - When it expires
    """
    content: str
    source: str
    context_type: ContextType
    confidence: float = 1.0           # 0.0 to 1.0
    valid_until: Optional[datetime] = None
    retrieved_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if this context has expired"""
        if self.valid_until is None:
            return False
        return datetime.now() > self.valid_until
    
    def is_trustworthy(self, threshold: float = 0.7) -> bool:
        """Check if confidence meets threshold"""
        return self.confidence >= threshold and not self.is_expired()
    
    def to_grounding_dict(self) -> Dict[str, Any]:
        """Export grounding contract for prompt inclusion"""
        return {
            "source": self.source,
            "confidence": self.confidence,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "is_fresh": not self.is_expired(),
            "context_type": self.context_type.value
        }


@dataclass
class AssembledContext:
    """Final assembled context ready for prompt construction"""
    primary_context: str              # Main context to include
    supporting_context: str           # Secondary/background context
    grounding_instructions: str       # Instructions about data reliability
    total_tokens_estimate: int        # Estimated token count
    context_items: List[GroundedContext] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ContextAssemblyLayer:
    """
    Intelligent context assembly for LLM prompts.
    
    Key responsibilities:
    1. Select relevant context based on query intent
    2. Rank context by importance and freshness
    3. Compress/summarize to fit token budget
    4. Track grounding contracts for reliability
    """
    
    # Token budget allocation
    MAX_TOTAL_TOKENS = 2000
    PRIMARY_CONTEXT_BUDGET = 1200
    SUPPORTING_CONTEXT_BUDGET = 500
    GROUNDING_BUDGET = 300
    
    # Freshness thresholds by context type
    FRESHNESS_THRESHOLDS = {
        ContextType.WEATHER: timedelta(hours=1),
        ContextType.REALTIME: timedelta(minutes=15),
        ContextType.TRANSPORTATION: timedelta(hours=6),
        ContextType.RESTAURANT: timedelta(days=7),
        ContextType.ATTRACTION: timedelta(days=30),
        ContextType.RAG: timedelta(days=90),
        ContextType.DATABASE: timedelta(days=365),
        ContextType.CONVERSATION: timedelta(hours=24),
    }
    
    # Confidence sources (higher = more trusted)
    SOURCE_CONFIDENCE = {
        "istanbul_transport_api": 0.95,
        "openweathermap": 0.90,
        "google_maps": 0.92,
        "database_verified": 0.95,
        "rag_retrieved": 0.75,
        "user_provided": 0.60,
        "cached": 0.70,
        "fallback": 0.40,
    }
    
    def __init__(self):
        self.context_cache: Dict[str, GroundedContext] = {}
    
    def create_grounded_context(
        self,
        content: str,
        source: str,
        context_type: ContextType,
        confidence: Optional[float] = None,
        ttl_override: Optional[timedelta] = None,
        metadata: Optional[Dict] = None
    ) -> GroundedContext:
        """
        Create a context item with explicit grounding contract.
        
        Args:
            content: The actual context text
            source: Where this data came from
            context_type: Type of context
            confidence: Override confidence (else use source default)
            ttl_override: Override TTL (else use type default)
            metadata: Additional metadata
        """
        # Determine confidence from source if not provided
        if confidence is None:
            confidence = self.SOURCE_CONFIDENCE.get(source, 0.5)
        
        # Determine validity period
        ttl = ttl_override or self.FRESHNESS_THRESHOLDS.get(
            context_type, 
            timedelta(days=1)
        )
        valid_until = datetime.now() + ttl
        
        return GroundedContext(
            content=content,
            source=source,
            context_type=context_type,
            confidence=confidence,
            valid_until=valid_until,
            metadata=metadata or {}
        )
    
    def rank_contexts(
        self,
        contexts: List[GroundedContext],
        query_intent: Optional[str] = None,
        signals: Optional[Dict[str, bool]] = None
    ) -> List[Tuple[GroundedContext, float]]:
        """
        Rank contexts by relevance and quality.
        
        Returns list of (context, score) tuples, sorted by score descending.
        """
        scored = []
        signals = signals or {}
        
        # Intent-to-context-type mapping
        intent_priorities = {
            "transportation": [ContextType.TRANSPORTATION, ContextType.REALTIME],
            "directions": [ContextType.TRANSPORTATION, ContextType.REALTIME],
            "route": [ContextType.TRANSPORTATION, ContextType.REALTIME],
            "weather": [ContextType.WEATHER, ContextType.REALTIME],
            "restaurant": [ContextType.RESTAURANT, ContextType.RAG],
            "food": [ContextType.RESTAURANT, ContextType.RAG],
            "attraction": [ContextType.ATTRACTION, ContextType.RAG],
            "museum": [ContextType.ATTRACTION, ContextType.RAG],
            "place": [ContextType.ATTRACTION, ContextType.RAG, ContextType.DATABASE],
        }
        
        priority_types = intent_priorities.get(query_intent, [])
        
        for ctx in contexts:
            score = 0.0
            
            # Base score from confidence
            score += ctx.confidence * 40
            
            # Freshness bonus (0-30 points)
            if not ctx.is_expired():
                # Calculate how fresh (0-1 scale)
                if ctx.valid_until:
                    time_remaining = (ctx.valid_until - datetime.now()).total_seconds()
                    ttl_total = self.FRESHNESS_THRESHOLDS.get(
                        ctx.context_type, 
                        timedelta(days=1)
                    ).total_seconds()
                    freshness = min(1.0, time_remaining / ttl_total)
                    score += freshness * 30
                else:
                    score += 15  # Unknown freshness = medium
            else:
                score -= 20  # Expired = penalty
            
            # Intent relevance bonus (0-30 points)
            if ctx.context_type in priority_types:
                priority_index = priority_types.index(ctx.context_type)
                score += 30 - (priority_index * 10)
            
            # Signal-based bonuses
            if signals.get('needs_transportation') and ctx.context_type == ContextType.TRANSPORTATION:
                score += 20
            if signals.get('needs_weather') and ctx.context_type == ContextType.WEATHER:
                score += 20
            if signals.get('needs_restaurant') and ctx.context_type == ContextType.RESTAURANT:
                score += 20
            if signals.get('needs_attraction') and ctx.context_type == ContextType.ATTRACTION:
                score += 20
            
            # Content length consideration (prefer substantial but not bloated)
            content_len = len(ctx.content)
            if 100 < content_len < 1000:
                score += 10
            elif content_len > 2000:
                score -= 5  # Too long = potential noise
            
            scored.append((ctx, score))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def compress_context(
        self,
        context: str,
        max_chars: int = 1000,
        preserve_key_info: bool = True
    ) -> str:
        """
        Compress context to fit within budget.
        
        Strategies:
        1. Remove redundant whitespace
        2. Truncate at sentence boundaries
        3. Preserve key information markers
        """
        if len(context) <= max_chars:
            return context
        
        # Clean whitespace
        import re
        context = re.sub(r'\s+', ' ', context).strip()
        
        if len(context) <= max_chars:
            return context
        
        # Try to truncate at sentence boundary
        truncated = context[:max_chars]
        
        # Find last sentence end
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        cut_point = max(last_period, last_newline)
        
        if cut_point > max_chars * 0.7:  # Only if we keep most of it
            return truncated[:cut_point + 1].strip()
        
        # Otherwise just truncate with ellipsis
        return truncated[:max_chars - 3].strip() + "..."
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)"""
        return len(text) // 4
    
    def assemble(
        self,
        contexts: List[GroundedContext],
        query: str,
        query_intent: Optional[str] = None,
        signals: Optional[Dict[str, bool]] = None,
        include_grounding: bool = True
    ) -> AssembledContext:
        """
        Assemble contexts into a coherent, budget-aware prompt context.
        
        Args:
            contexts: List of grounded context items
            query: The user query
            query_intent: Detected intent
            signals: Query signals (needs_transportation, etc.)
            include_grounding: Whether to include grounding instructions
            
        Returns:
            AssembledContext ready for prompt construction
        """
        warnings = []
        
        # Rank contexts
        ranked = self.rank_contexts(contexts, query_intent, signals)
        
        # Separate into primary (high confidence) and supporting (lower)
        primary_items = []
        supporting_items = []
        
        for ctx, score in ranked:
            if ctx.is_expired():
                warnings.append(f"Expired context from {ctx.source} (type: {ctx.context_type.value})")
                continue
            
            if ctx.confidence >= 0.7 and score >= 50:
                primary_items.append(ctx)
            elif ctx.confidence >= 0.4:
                supporting_items.append(ctx)
            else:
                warnings.append(f"Low confidence context dropped: {ctx.source} ({ctx.confidence:.2f})")
        
        # Assemble primary context within budget
        primary_parts = []
        primary_tokens = 0
        
        for ctx in primary_items:
            estimated = self.estimate_tokens(ctx.content)
            if primary_tokens + estimated > self.PRIMARY_CONTEXT_BUDGET:
                # Compress to fit
                remaining_budget = self.PRIMARY_CONTEXT_BUDGET - primary_tokens
                if remaining_budget > 100:  # Worth including compressed
                    compressed = self.compress_context(
                        ctx.content, 
                        max_chars=remaining_budget * 4
                    )
                    primary_parts.append(compressed)
                break
            primary_parts.append(ctx.content)
            primary_tokens += estimated
        
        # Assemble supporting context
        supporting_parts = []
        supporting_tokens = 0
        
        for ctx in supporting_items:
            estimated = self.estimate_tokens(ctx.content)
            if supporting_tokens + estimated > self.SUPPORTING_CONTEXT_BUDGET:
                break
            supporting_parts.append(ctx.content)
            supporting_tokens += estimated
        
        # Build grounding instructions
        grounding_instructions = ""
        if include_grounding:
            grounding_instructions = self._build_grounding_instructions(
                primary_items + supporting_items,
                warnings
            )
        
        # Combine
        primary_context = "\n\n".join(primary_parts) if primary_parts else ""
        supporting_context = "\n\n".join(supporting_parts) if supporting_parts else ""
        
        total_tokens = (
            self.estimate_tokens(primary_context) +
            self.estimate_tokens(supporting_context) +
            self.estimate_tokens(grounding_instructions)
        )
        
        logger.info(f"📦 Context assembled: {len(primary_items)} primary, {len(supporting_items)} supporting")
        logger.info(f"   Tokens estimate: {total_tokens} (budget: {self.MAX_TOTAL_TOKENS})")
        if warnings:
            logger.warning(f"   Warnings: {len(warnings)}")
        
        return AssembledContext(
            primary_context=primary_context,
            supporting_context=supporting_context,
            grounding_instructions=grounding_instructions,
            total_tokens_estimate=total_tokens,
            context_items=primary_items + supporting_items,
            warnings=warnings
        )
    
    def _build_grounding_instructions(
        self,
        contexts: List[GroundedContext],
        warnings: List[str]
    ) -> str:
        """Build instructions about data reliability for the model"""
        
        if not contexts:
            return "No verified context available. Be cautious and disclaim uncertainty."
        
        instructions = []
        
        # Group by confidence level
        high_confidence = [c for c in contexts if c.confidence >= 0.9]
        medium_confidence = [c for c in contexts if 0.7 <= c.confidence < 0.9]
        low_confidence = [c for c in contexts if c.confidence < 0.7]
        
        if high_confidence:
            sources = list(set(c.source for c in high_confidence))
            instructions.append(
                f"HIGH CONFIDENCE data from: {', '.join(sources)}. "
                "You can state this information confidently."
            )
        
        if medium_confidence:
            sources = list(set(c.source for c in medium_confidence))
            instructions.append(
                f"MEDIUM CONFIDENCE data from: {', '.join(sources)}. "
                "Present as reliable but acknowledge if user asks for certainty."
            )
        
        if low_confidence:
            instructions.append(
                "Some context has LOW CONFIDENCE. "
                "Use hedging language like 'based on available information' or 'typically'."
            )
        
        # Check for real-time data freshness
        realtime = [c for c in contexts if c.context_type in [ContextType.WEATHER, ContextType.REALTIME]]
        if realtime:
            oldest = min(c.retrieved_at for c in realtime)
            age_minutes = (datetime.now() - oldest).total_seconds() / 60
            if age_minutes > 30:
                instructions.append(
                    f"Real-time data is {int(age_minutes)} minutes old. "
                    "Mention this if user needs current information."
                )
        
        # Add warning about expired/dropped context
        if warnings:
            instructions.append(
                "Some context was expired or low-quality and was excluded. "
                "Acknowledge limitations if the answer seems incomplete."
            )
        
        return "\n".join(instructions)
    
    def quick_assemble(
        self,
        rag_context: Optional[str] = None,
        database_context: Optional[str] = None,
        service_data: Optional[Dict[str, Any]] = None,
        query_intent: Optional[str] = None,
        signals: Optional[Dict[str, bool]] = None
    ) -> AssembledContext:
        """
        Quick assembly from common context sources.
        
        Convenience method for typical use cases.
        """
        contexts = []
        
        if rag_context:
            contexts.append(self.create_grounded_context(
                content=rag_context,
                source="rag_retrieved",
                context_type=ContextType.RAG,
            ))
        
        if database_context:
            contexts.append(self.create_grounded_context(
                content=database_context,
                source="database_verified",
                context_type=ContextType.DATABASE,
            ))
        
        if service_data:
            # Parse service data into typed contexts
            if 'weather' in service_data:
                contexts.append(self.create_grounded_context(
                    content=str(service_data['weather']),
                    source="openweathermap",
                    context_type=ContextType.WEATHER,
                ))
            
            if 'route' in service_data or 'directions' in service_data:
                route_data = service_data.get('route') or service_data.get('directions')
                contexts.append(self.create_grounded_context(
                    content=str(route_data),
                    source="istanbul_transport_api",
                    context_type=ContextType.TRANSPORTATION,
                ))
            
            if 'restaurants' in service_data:
                contexts.append(self.create_grounded_context(
                    content=str(service_data['restaurants']),
                    source="database_verified",
                    context_type=ContextType.RESTAURANT,
                ))
            
            if 'attractions' in service_data:
                contexts.append(self.create_grounded_context(
                    content=str(service_data['attractions']),
                    source="database_verified",
                    context_type=ContextType.ATTRACTION,
                ))
        
        return self.assemble(
            contexts=contexts,
            query="",  # Not used in quick_assemble
            query_intent=query_intent,
            signals=signals
        )


# Global instance
_context_assembler: Optional[ContextAssemblyLayer] = None


def get_context_assembler() -> ContextAssemblyLayer:
    """Get or create global context assembler instance"""
    global _context_assembler
    if _context_assembler is None:
        _context_assembler = ContextAssemblyLayer()
    return _context_assembler

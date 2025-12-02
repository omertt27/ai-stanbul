"""
LLM Enhancement Models - Structured Response Models

Pydantic models for structured LLM responses in the intent classification,
location resolution, and response enhancement pipeline.

Author: AI Istanbul Team
Date: 2024
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal, Tuple
from datetime import datetime


class IntentClassification(BaseModel):
    """
    Structured output from LLM intent classification.
    
    This model represents the LLM's understanding of what the user wants,
    extracted locations, preferences, and confidence levels.
    """
    
    # Primary intent classification
    primary_intent: Literal[
        "route",           # User wants directions/navigation
        "restaurant",      # User wants restaurant recommendations
        "information",     # User wants information about POIs
        "hidden_gems",     # User wants hidden gems/off-the-beaten-path
        "event",          # User wants event information
        "weather",        # User wants weather information
        "museum",         # User wants museum information
        "transport",      # User wants transportation info
        "general",        # General chat/unclear intent
        "multi_intent"    # Multiple intents detected
    ] = Field(..., description="Primary classified intent")
    
    # Secondary intents for complex queries
    secondary_intents: List[str] = Field(
        default_factory=list,
        description="Additional detected intents for multi-part queries"
    )
    
    # Location extraction
    origin: Optional[str] = Field(
        None,
        description="Extracted origin location (normalized)"
    )
    destination: Optional[str] = Field(
        None,
        description="Extracted destination location (normalized)"
    )
    
    # Entity extraction
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted entities (time, date, numbers, preferences)"
    )
    
    # User preferences extracted from query
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted preferences (cheap, accessible, family-friendly, etc.)"
    )
    
    # Confidence and quality metrics
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0-1)"
    )
    
    ambiguities: List[str] = Field(
        default_factory=list,
        description="List of ambiguous or unclear parts of the query"
    )
    
    # Original query for reference
    original_query: str = Field(
        ...,
        description="Original user query"
    )
    
    # GPS context
    has_gps: bool = Field(
        default=False,
        description="Whether user GPS location is available"
    )
    
    # Metadata
    classification_method: str = Field(
        default="llm",
        description="Method used for classification (llm/hybrid/fallback)"
    )
    
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to classify (milliseconds)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of classification"
    )
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v
    
    def should_use_gps_handler(self) -> bool:
        """
        Determine if this intent should use GPS-based handlers.
        
        Returns:
            bool: True if GPS handler is appropriate
        """
        gps_intents = ["route", "hidden_gems", "transport"]
        return self.primary_intent in gps_intents and self.has_gps
    
    def is_high_confidence(self, threshold: float = 0.75) -> bool:
        """
        Check if classification confidence is high.
        
        Args:
            threshold: Minimum confidence threshold (default 0.75)
            
        Returns:
            bool: True if confidence >= threshold
        """
        return self.confidence >= threshold
    
    def needs_disambiguation(self) -> bool:
        """
        Check if query needs disambiguation.
        
        Returns:
            bool: True if there are ambiguities
        """
        return len(self.ambiguities) > 0


class LocationMatch(BaseModel):
    """
    A single matched location from location resolution.
    
    Represents one location extracted and matched from a user query,
    with confidence scoring and disambiguation information.
    """
    
    name: str = Field(
        ...,
        description="Original location name from query"
    )
    
    matched_name: Optional[str] = Field(
        None,
        description="Matched known location name (normalized)"
    )
    
    coordinates: Optional[Tuple[float, float]] = Field(
        None,
        description="GPS coordinates (latitude, longitude) if found"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this location match (0-1)"
    )
    
    disambiguation_note: Optional[str] = Field(
        None,
        description="Note explaining ambiguity or clarification"
    )
    
    class Config:
        arbitrary_types_allowed = True


class LocationResolution(BaseModel):
    """
    Structured output from LLM location resolution.
    
    Contains all extracted locations from a query with pattern detection,
    confidence scoring, and ambiguity handling.
    """
    
    query: str = Field(
        ...,
        description="Original location query"
    )
    
    locations: List[LocationMatch] = Field(
        default_factory=list,
        description="Extracted and matched locations in order"
    )
    
    pattern: Literal[
        "from_to",           # A to B navigation
        "multi_stop",        # Multiple stops journey
        "destination_only",  # Single destination
        "area_exploration",  # Area/neighborhood exploration
        "unknown"           # Pattern not recognized
    ] = Field(
        ...,
        description="Detected location pattern"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in location resolution (0-1)"
    )
    
    ambiguities: List[str] = Field(
        default_factory=list,
        description="Ambiguous location references that need clarification"
    )
    
    suggestions: List[str] = Field(
        default_factory=list,
        description="Alternative interpretations or suggestions"
    )
    
    used_llm: bool = Field(
        True,
        description="Whether LLM was used for resolution"
    )
    
    fallback_used: bool = Field(
        False,
        description="Whether fallback regex method was used"
    )
    
    class Config:
        arbitrary_types_allowed = True


class EnhancedResponse(BaseModel):
    """
    Structured output from LLM response enhancement.
    
    Represents an enhanced response with personalization and context.
    """
    
    # Original response
    original_response: str = Field(..., description="Original handler response")
    
    # Enhanced response (original + enhancements)
    enhanced_response: str = Field(..., description="LLM-enhanced response with additions")
    
    # Enhancement details
    enhancements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Enhancement details (text, method, etc.)"
    )
    
    # Context used for enhancement
    context_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Context data used for enhancement"
    )
    
    # Metadata
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to enhance (milliseconds)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of enhancement"
    )


class QueryAnalysis(BaseModel):
    """
    Comprehensive query analysis combining intent, location, and context.
    
    This is the main output of the LLM-first pipeline.
    """
    
    # Core analysis
    intent: IntentClassification = Field(..., description="Intent classification")
    
    # Location resolution (if applicable)
    origin_location: Optional[LocationMatch] = Field(
        None,
        description="Resolved origin location"
    )
    destination_location: Optional[LocationMatch] = Field(
        None,
        description="Resolved destination location"
    )
    
    # Context
    user_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="User context (history, preferences, location)"
    )
    
    # Routing recommendation
    recommended_handler: str = Field(
        ...,
        description="Recommended handler based on analysis"
    )
    
    handler_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in handler recommendation"
    )
    
    # Fallback strategy
    fallback_to_llm: bool = Field(
        default=False,
        description="Whether to fallback to full LLM if handler fails"
    )
    
    # Metadata
    total_processing_time_ms: float = Field(
        ...,
        description="Total analysis time (milliseconds)"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of analysis"
    )


class LLMPromptTemplate(BaseModel):
    """
    Template for LLM prompts with variable substitution.
    """
    
    name: str = Field(..., description="Template name")
    template: str = Field(..., description="Prompt template with {variables}")
    variables: List[str] = Field(..., description="Required variable names")
    system_message: Optional[str] = Field(None, description="System message for chat models")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=500, ge=1, le=4000, description="Max response tokens")
    
    def render(self, **kwargs) -> str:
        """
        Render template with provided variables.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            str: Rendered prompt
            
        Raises:
            ValueError: If required variables are missing
        """
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        return self.template.format(**kwargs)


class CachedIntentResult(BaseModel):
    """
    Cached intent classification result with TTL and hit tracking.
    """
    
    query_hash: str = Field(..., description="Hash of the query for cache lookup")
    intent: IntentClassification = Field(..., description="Cached intent classification")
    cache_hit_count: int = Field(default=0, description="Number of cache hits")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Cache creation time")
    last_accessed: datetime = Field(default_factory=datetime.utcnow, description="Last access time")
    ttl_seconds: int = Field(default=3600, description="Time to live in seconds")
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def record_hit(self):
        """Record a cache hit."""
        self.cache_hit_count += 1
        self.last_accessed = datetime.utcnow()


class RoutePreferences(BaseModel):
    """
    Route preference model for Phase 4.1.
    
    Represents user preferences extracted from natural language for route planning.
    This model gives LLM complete control over understanding HOW users want to travel.
    """
    
    # Optimization goal
    optimize_for: Optional[Literal[
        "speed",           # Fastest route
        "cost",            # Cheapest route
        "scenic",          # Nice views/experience
        "accessibility",   # Accessibility needs
        "comfort",         # Comfortable journey
        "ease"             # Easiest/least effort
    ]] = Field(
        None,
        description="What the route should optimize for"
    )
    
    # Transport preferences
    transport_modes: Optional[List[Literal[
        "walk", "metro", "tram", "bus", "ferry", "taxi", "car"
    ]]] = Field(
        None,
        description="Preferred transport modes"
    )
    
    # Avoidance preferences
    avoid: Optional[List[Literal[
        "stairs", "crowds", "hills", "transfers", "walking", "waiting", "heat", "rain"
    ]]] = Field(
        None,
        description="Things to avoid in the route"
    )
    
    # Accessibility requirements
    accessibility: Optional[Literal[
        "wheelchair",  # Wheelchair accessible
        "stroller",    # Baby stroller friendly
        "elderly",     # Elderly-friendly
        "none"         # No special needs
    ]] = Field(
        None,
        description="Accessibility requirements"
    )
    
    # Time constraints
    time_constraint: Optional[Literal[
        "rush",           # In a hurry
        "flexible",       # Flexible time
        "specific_time"   # Must arrive by specific time
    ]] = Field(
        None,
        description="Time constraint type"
    )
    
    # Weather consideration
    weather_consideration: bool = Field(
        default=False,
        description="Should weather be considered in routing?"
    )
    
    # Budget
    budget: Optional[Literal[
        "cheap",      # Cheapest option
        "moderate",   # Normal budget
        "expensive"   # Cost not a concern
    ]] = Field(
        None,
        description="Budget consideration"
    )
    
    # Comfort level
    comfort_level: Optional[Literal[
        "high",    # Comfortable journey
        "medium",  # Normal comfort
        "low"      # Doesn't care about comfort
    ]] = Field(
        None,
        description="Desired comfort level"
    )
    
    # Numerical constraints
    max_walking_distance_km: Optional[float] = Field(
        None,
        ge=0.0,
        le=20.0,
        description="Maximum walking distance in kilometers"
    )
    
    max_transfers: Optional[int] = Field(
        None,
        ge=0,
        le=10,
        description="Maximum number of transfers"
    )
    
    # Boolean preferences
    prefer_walking: bool = Field(
        default=False,
        description="Prefer walking routes?"
    )
    
    prefer_public_transport: bool = Field(
        default=False,
        description="Prefer public transport?"
    )
    
    # Metadata
    source: Literal["llm", "fallback", "user_profile"] = Field(
        default="llm",
        description="Source of preferences"
    )
    
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in extracted preferences"
    )
    
    def to_routing_params(self) -> Dict[str, Any]:
        """
        Convert preferences to routing system parameters.
        
        Returns:
            Dict with routing parameters
        """
        params = {}
        
        # Map optimize_for to routing weights
        if self.optimize_for == "speed":
            params["preference"] = "fastest"
        elif self.optimize_for == "cost":
            params["preference"] = "cheapest"
        elif self.optimize_for == "ease":
            params["preference"] = "least_transfers"
        
        # Map accessibility to routing constraints
        if self.accessibility in ["wheelchair", "stroller"]:
            params["wheelchair"] = True
            params["avoid_stairs"] = True
        
        # Map avoidances
        if self.avoid:
            if "stairs" in self.avoid:
                params["avoid_stairs"] = True
            if "transfers" in self.avoid:
                params["max_transfers"] = 0
        
        # Map constraints
        if self.max_walking_distance_km:
            params["max_walking_distance"] = self.max_walking_distance_km * 1000  # Convert to meters
        
        if self.max_transfers is not None:
            params["max_transfers"] = self.max_transfers
        
        # Map transport modes
        if self.transport_modes:
            params["allowed_modes"] = self.transport_modes
        
        return params
    
    def get_summary(self) -> str:
        """
        Get human-readable summary of preferences.
        
        Returns:
            str: Human-readable summary
        """
        parts = []
        
        if self.optimize_for:
            parts.append(f"optimized for {self.optimize_for}")
        
        if self.accessibility:
            parts.append(f"{self.accessibility}-accessible")
        
        if self.avoid:
            parts.append(f"avoiding {', '.join(self.avoid)}")
        
        if self.transport_modes:
            parts.append(f"using {', '.join(self.transport_modes)}")
        
        if self.time_constraint == "rush":
            parts.append("urgent")
        
        if not parts:
            return "no specific preferences"
        
        return ", ".join(parts)

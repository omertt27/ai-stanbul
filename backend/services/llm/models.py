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
# =============================================================================
# Phase 4.3: Multi-Intent Models
# =============================================================================

class DetectedIntent(BaseModel):
    """
    A single detected intent in a multi-intent query.
    
    Represents one intent extracted from a query that may contain multiple intents.
    """
    
    intent_type: str = Field(
        ...,
        description="Intent type (route, restaurant, weather, etc.)"
    )
    
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted parameters for this intent"
    )
    
    priority: int = Field(
        ...,
        ge=1,
        description="Execution priority (1 = highest)"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this intent detection"
    )
    
    requires_location: bool = Field(
        default=False,
        description="Whether this intent requires location context"
    )
    
    depends_on: Optional[List[int]] = Field(
        None,
        description="Indices of intents this depends on (e.g., [0] means depends on first intent)"
    )
    
    condition: Optional[str] = Field(
        None,
        description="Conditional execution (e.g., 'if_sunny', 'if_not_raining')"
    )


class IntentRelationship(BaseModel):
    """
    Relationship between multiple detected intents.
    """
    
    relationship_type: Literal[
        "sequential",      # Intents execute in order, one after another
        "parallel",        # Intents can execute simultaneously
        "conditional",     # Intent execution depends on condition
        "dependent"        # Intent depends on another's result
    ] = Field(
        ...,
        description="Type of relationship between intents"
    )
    
    intent_indices: List[int] = Field(
        ...,
        description="Indices of intents involved in this relationship"
    )
    
    description: str = Field(
        ...,
        description="Human-readable description of the relationship"
    )


class MultiIntentDetection(BaseModel):
    """
    Result of multi-intent detection from LLM.
    
    Contains all detected intents, their relationships, and execution planning.
    Phase 4.3 gives LLM 95% control over multi-intent understanding.
    """
    
    # Query information
    original_query: str = Field(
        ...,
        description="Original user query"
    )
    
    # Intent detection
    intent_count: int = Field(
        ...,
        ge=1,
        description="Number of intents detected"
    )
    
    intents: List[DetectedIntent] = Field(
        ...,
        description="List of detected intents with parameters"
    )
    
    # Relationships
    relationships: List[IntentRelationship] = Field(
        default_factory=list,
        description="Relationships between intents"
    )
    
    # Execution planning
    execution_strategy: Literal[
        "sequential",      # Execute intents one by one
        "parallel",        # Execute intents simultaneously
        "conditional",     # Execute based on conditions
        "mixed"            # Mix of sequential and parallel
    ] = Field(
        ...,
        description="Overall execution strategy"
    )
    
    # Metadata
    is_multi_intent: bool = Field(
        ...,
        description="Whether this is actually a multi-intent query"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in multi-intent detection"
    )
    
    detection_method: str = Field(
        default="llm",
        description="Method used for detection (llm/fallback)"
    )
    
    processing_time_ms: Optional[float] = Field(
        None,
        description="Time taken to detect (milliseconds)"
    )
    
    def is_simple_query(self) -> bool:
        """Check if this is actually a single-intent query."""
        return self.intent_count == 1 and not self.is_multi_intent
    
    def has_dependencies(self) -> bool:
        """Check if any intents have dependencies."""
        return any(intent.depends_on for intent in self.intents)
    
    def has_conditions(self) -> bool:
        """Check if any intents have conditional execution."""
        return any(intent.condition for intent in self.intents)


class ExecutionStep(BaseModel):
    """
    A single step in the intent execution plan.
    """
    
    step_number: int = Field(
        ...,
        ge=1,
        description="Step number in execution sequence"
    )
    
    intent_indices: List[int] = Field(
        ...,
        description="Indices of intents to execute in this step"
    )
    
    execution_mode: Literal["sequential", "parallel"] = Field(
        ...,
        description="How to execute intents in this step"
    )
    
    requires_results_from: Optional[List[int]] = Field(
        None,
        description="Step numbers whose results are needed"
    )
    
    condition: Optional[str] = Field(
        None,
        description="Condition for executing this step"
    )
    
    timeout_ms: int = Field(
        default=5000,
        description="Timeout for this step in milliseconds"
    )


class ExecutionPlan(BaseModel):
    """
    Complete execution plan for multi-intent query.
    
    Orchestrated by LLM (90% LLM control).
    """
    
    # Plan metadata
    plan_id: str = Field(
        ...,
        description="Unique identifier for this execution plan"
    )
    
    query: str = Field(
        ...,
        description="Original query"
    )
    
    # Execution steps
    steps: List[ExecutionStep] = Field(
        ...,
        description="Ordered list of execution steps"
    )
    
    # Strategy
    total_steps: int = Field(
        ...,
        ge=1,
        description="Total number of steps"
    )
    
    has_parallel_execution: bool = Field(
        default=False,
        description="Whether plan includes parallel execution"
    )
    
    has_conditional_logic: bool = Field(
        default=False,
        description="Whether plan includes conditional execution"
    )
    
    # Fallback
    fallback_strategy: Literal[
        "sequential",       # Fall back to sequential execution
        "skip_failed",      # Skip failed intents
        "stop_on_error"     # Stop if any intent fails
    ] = Field(
        default="skip_failed",
        description="How to handle errors during execution"
    )
    
    # Metadata
    estimated_duration_ms: Optional[int] = Field(
        None,
        description="Estimated execution duration"
    )
    
    planning_method: str = Field(
        default="llm",
        description="Method used for planning (llm/fallback)"
    )
    
    planning_time_ms: Optional[float] = Field(
        None,
        description="Time taken to create plan"
    )


class IntentResult(BaseModel):
    """
    Result from executing a single intent.
    """
    
    intent_index: int = Field(
        ...,
        description="Index of the intent that was executed"
    )
    
    intent_type: str = Field(
        ...,
        description="Type of intent executed"
    )
    
    success: bool = Field(
        ...,
        description="Whether execution was successful"
    )
    
    response: Optional[str] = Field(
        None,
        description="Response text from intent execution"
    )
    
    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Structured data from intent execution"
    )
    
    error: Optional[str] = Field(
        None,
        description="Error message if execution failed"
    )
    
    execution_time_ms: Optional[float] = Field(
        None,
        description="Time taken to execute"
    )


class MultiIntentResponse(BaseModel):
    """
    Synthesized response combining multiple intent results.
    
    Generated by LLM (100% LLM control for synthesis).
    """
    
    # Original query
    original_query: str = Field(
        ...,
        description="Original user query"
    )
    
    # Individual results
    intent_results: List[IntentResult] = Field(
        ...,
        description="Results from each intent execution"
    )
    
    # Synthesized response
    synthesized_response: str = Field(
        ...,
        description="Combined, coherent response from LLM"
    )
    
    # Response metadata
    response_structure: Literal[
        "narrative",        # Flowing narrative combining all results
        "structured",       # Structured sections for each intent
        "comparison",       # Comparison format (for comparing intents)
        "conditional"       # Conditional response based on results
    ] = Field(
        ...,
        description="Structure of the synthesized response"
    )
    
    # Quality metrics
    coherence_score: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="How coherent the combined response is"
    )
    
    completeness: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="How complete the response is (addresses all intents)"
    )
    
    # Metadata
    synthesis_method: str = Field(
        default="llm",
        description="Method used for synthesis (llm/template)"
    )
    
    synthesis_time_ms: Optional[float] = Field(
        None,
        description="Time taken to synthesize"
    )
    
    total_processing_time_ms: Optional[float] = Field(
        None,
        description="Total time from query to response"
    )
    
    # Map visualization data
    map_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Map visualization data for route/location responses"
    )
    
    # Follow-up suggestions
    follow_up_suggestions: Optional[List[str]] = Field(
        None,
        description="Suggested follow-up questions or actions"
    )
# ============================================================================
# Phase 4.4: Proactive Suggestions Models
# ============================================================================

class SuggestionContext(BaseModel):
    """
    Context for generating proactive suggestions.
    
    Contains all information needed to understand the current conversation
    state and generate relevant suggestions.
    """
    
    # Current conversation state
    current_query: str = Field(
        ...,
        description="The user's current query"
    )
    
    current_response: str = Field(
        ...,
        description="The response we just provided"
    )
    
    # Intent and entity information
    detected_intents: List[str] = Field(
        default_factory=list,
        description="All detected intents in current query"
    )
    
    extracted_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entities extracted from query and response"
    )
    
    # Conversation context
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent conversation turns (max 5)"
    )
    
    user_location: Optional[str] = Field(
        None,
        description="User's current location if known"
    )
    
    # Response metadata
    response_type: str = Field(
        ...,
        description="Type of response (restaurant, attraction, route, etc.)"
    )
    
    response_success: bool = Field(
        default=True,
        description="Whether the response was successful"
    )
    
    # Trigger information
    trigger_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that suggestions should be shown"
    )
    
    trigger_reason: Optional[str] = Field(
        None,
        description="Why suggestions were triggered"
    )


class ProactiveSuggestion(BaseModel):
    """
    A single proactive suggestion for the user.
    
    Represents one actionable suggestion that the user can follow up on.
    """
    
    # Identification
    suggestion_id: str = Field(
        ...,
        description="Unique identifier for this suggestion"
    )
    
    # Display text
    suggestion_text: str = Field(
        ...,
        description="Natural language text shown to user"
    )
    
    # Categorization
    suggestion_type: Literal[
        "exploration",   # Discover new places/things
        "practical",     # Practical travel info (directions, weather, etc.)
        "cultural",      # Cultural events, customs, activities
        "dining",        # Food and restaurant related
        "refinement"     # Refine/filter current results
    ] = Field(
        ...,
        description="Category of suggestion"
    )
    
    # Intent mapping
    intent_type: str = Field(
        ...,
        description="The intent that would be triggered (e.g., 'get_directions')"
    )
    
    # Pre-filled entities for executing the suggestion
    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Entities to use if user selects this suggestion"
    )
    
    # Scoring and ranking
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How relevant this suggestion is (0-1)"
    )
    
    priority: int = Field(
        default=0,
        description="Display priority (higher = show first)"
    )
    
    # Metadata
    reasoning: Optional[str] = Field(
        None,
        description="LLM's reasoning for this suggestion"
    )
    
    icon: Optional[str] = Field(
        None,
        description="Icon/emoji to display with suggestion"
    )
    
    action_type: Literal["query", "link", "filter"] = Field(
        default="query",
        description="Type of action when user clicks"
    )


class ProactiveSuggestionResponse(BaseModel):
    """
    Complete response containing all proactive suggestions.
    
    This is what gets added to the chat response.
    """
    
    # Suggestions
    suggestions: List[ProactiveSuggestion] = Field(
        default_factory=list,
        description="List of suggestions (ordered by priority)"
    )
    
    # Context used
    context: SuggestionContext = Field(
        ...,
        description="Context that was analyzed"
    )
    
    # Generation metadata
    generation_method: Literal["llm", "template", "hybrid"] = Field(
        ...,
        description="Method used to generate suggestions"
    )
    
    generation_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken to generate suggestions"
    )
    
    total_suggestions_considered: int = Field(
        ...,
        ge=0,
        description="Total suggestions considered before ranking"
    )
    
    # Quality metrics
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence in suggestion quality"
    )
    
    diversity_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="How diverse the suggestion types are"
    )
    
    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When suggestions were generated"
    )
    
    llm_used: bool = Field(
        default=False,
        description="Whether LLM was used for generation"
    )


class SuggestionAnalysis(BaseModel):
    """
    Analysis of whether suggestions should be shown.
    
    Result of analyzing the conversation context to determine if
    proactive suggestions are appropriate.
    """
    
    should_suggest: bool = Field(
        ...,
        description="Whether suggestions should be shown"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this decision"
    )
    
    reasoning: str = Field(
        ...,
        description="Why suggestions should or shouldn't be shown"
    )
    
    context_summary: str = Field(
        ...,
        description="Brief summary of the conversation context"
    )
    
    suggested_categories: List[str] = Field(
        default_factory=list,
        description="Which suggestion categories would be most relevant"
    )
    
    analysis_method: Literal["llm", "heuristic"] = Field(
        ...,
        description="Method used for analysis"
    )
    
    analysis_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken for analysis"
    )


class SuggestionInteraction(BaseModel):
    """
    Tracks user interaction with a suggestion.
    
    Used for analytics and improving suggestion quality.
    """
    
    suggestion_id: str = Field(
        ...,
        description="ID of the suggestion that was interacted with"
    )
    
    action: Literal["clicked", "ignored", "dismissed", "rated"] = Field(
        ...,
        description="What the user did"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the interaction occurred"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="User's session ID"
    )
    
    query_after: Optional[str] = Field(
        None,
        description="The query user made after clicking (if action=clicked)"
    )
    
    rating: Optional[int] = Field(
        None,
        ge=1,
        le=5,
        description="User rating if action=rated"
    )
    
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context about the interaction"
    )

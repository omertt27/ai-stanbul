"""
Chat Request/Response Schemas Module

Pydantic models for validating all chat-related API endpoints:
- /api/chat/pure-llm
- /api/chat/stream  
- /api/chat/unified

Author: AI Istanbul Team
Date: December 2024
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
import re


# ===========================================
# Constants
# ===========================================

MAX_MESSAGE_LENGTH = 2000
MIN_MESSAGE_LENGTH = 1
MAX_SESSION_ID_LENGTH = 128
MAX_CONVERSATION_HISTORY = 50
SUPPORTED_LANGUAGES = {"en", "tr", "de", "fr", "es", "ar", "ru", "zh", "ja", "ko"}


# ===========================================
# Location Schemas
# ===========================================

class LocationData(BaseModel):
    """User GPS location data"""
    lat: float = Field(..., ge=-90, le=90, description="Latitude (-90 to 90)")
    lon: float = Field(..., ge=-180, le=180, description="Longitude (-180 to 180)")
    accuracy: Optional[float] = Field(None, ge=0, description="Location accuracy in meters")
    timestamp: Optional[datetime] = Field(None, description="Location timestamp")
    
    @field_validator('lat', 'lon')
    @classmethod
    def validate_coordinates(cls, v):
        if v is None:
            return v
        # Ensure reasonable precision (6 decimal places = ~0.1m accuracy)
        return round(v, 6)


class UserLocation(BaseModel):
    """Alternative location format (lat/lng keys)"""
    lat: Optional[float] = Field(None, ge=-90, le=90)
    lng: Optional[float] = Field(None, ge=-180, le=180)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    
    @model_validator(mode='after')
    def normalize_longitude(self):
        # Support both 'lon' and 'lng' keys
        if self.lng is not None and self.lon is None:
            self.lon = self.lng
        return self


# ===========================================
# User Preferences Schema
# ===========================================

class UserPreferences(BaseModel):
    """User preferences for personalization"""
    preferred_transport: Optional[Literal["walking", "driving", "transit", "cycling"]] = None
    accessibility_needs: Optional[bool] = Field(None, description="User has accessibility requirements")
    avoid_crowds: Optional[bool] = Field(None, description="Prefer less crowded places")
    budget_level: Optional[Literal["budget", "moderate", "luxury"]] = None
    cuisine_preferences: Optional[List[str]] = Field(None, max_length=10)
    interests: Optional[List[str]] = Field(None, max_length=20)
    
    @field_validator('cuisine_preferences', 'interests')
    @classmethod
    def validate_list_items(cls, v):
        if v is None:
            return v
        return [item.strip().lower() for item in v if item.strip()]


# ===========================================
# Base Chat Request
# ===========================================

class BaseChatRequest(BaseModel):
    """Base request model with common fields for all chat endpoints"""
    message: str = Field(
        ..., 
        min_length=MIN_MESSAGE_LENGTH, 
        max_length=MAX_MESSAGE_LENGTH,
        description="User message text"
    )
    session_id: Optional[str] = Field(
        None, 
        max_length=MAX_SESSION_ID_LENGTH,
        description="Session identifier for conversation continuity"
    )
    language: Optional[str] = Field(
        "en",
        description="Preferred response language (auto-detected if not specified)"
    )
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty or whitespace only")
        
        # Basic sanitization - remove control characters
        v = ''.join(char for char in v if ord(char) >= 32 or char in '\n\r\t')
        
        return v.strip()
    
    @field_validator('session_id')
    @classmethod  
    def validate_session_id(cls, v):
        if v is None:
            return v
        
        # Allow alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Session ID must contain only alphanumeric characters, hyphens, and underscores")
        
        return v
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        if v is None:
            return "en"
        
        v = v.lower().strip()
        
        # Accept language codes like 'en-US' but normalize to base
        if '-' in v:
            v = v.split('-')[0]
        
        if v not in SUPPORTED_LANGUAGES:
            # Default to English for unsupported languages
            return "en"
        
        return v


# ===========================================
# Pure LLM Chat Request
# ===========================================

class PureLLMChatRequest(BaseChatRequest):
    """Request model for /api/chat/pure-llm endpoint"""
    user_location: Optional[Dict[str, float]] = Field(
        None, 
        description="User GPS location {lat, lon/lng}"
    )
    preferences: Optional[UserPreferences] = Field(
        None,
        description="User preferences for personalization"
    )
    user_id: Optional[str] = Field(
        None,
        max_length=128,
        description="User ID for personalization and history"
    )
    include_suggestions: Optional[bool] = Field(
        True,
        description="Include follow-up suggestions in response"
    )
    
    @field_validator('user_location')
    @classmethod
    def validate_user_location(cls, v):
        if v is None:
            return v
        
        # Validate lat/lon are present and valid
        lat = v.get('lat')
        lon = v.get('lon') or v.get('lng')
        
        if lat is None or lon is None:
            raise ValueError("Location must include 'lat' and 'lon' (or 'lng')")
        
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        
        if not (-180 <= lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        
        return {"lat": lat, "lon": lon}


# ===========================================
# Streaming Chat Request
# ===========================================

class StreamChatRequest(BaseChatRequest):
    """Request model for /api/stream/chat (SSE) endpoint"""
    user_location: Optional[Dict[str, float]] = Field(
        None,
        description="User GPS location {lat, lon}"
    )
    include_context: bool = Field(
        True,
        description="Include conversation context for coherent responses"
    )
    stream_metadata: bool = Field(
        True,
        description="Include metadata events (start, complete) in stream"
    )
    
    @field_validator('user_location')
    @classmethod
    def validate_user_location(cls, v):
        if v is None:
            return v
        
        lat = v.get('lat')
        lon = v.get('lon') or v.get('lng')
        
        if lat is not None and lon is not None:
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                raise ValueError("Invalid coordinates")
            return {"lat": lat, "lon": lon}
        
        return None


# ===========================================
# Unified Chat Request (Legacy)
# ===========================================

class UnifiedChatRequest(BaseChatRequest):
    """Request model for /api/chat/unified endpoint (legacy)"""
    user_location: Optional[Dict[str, float]] = None
    user_id: Optional[str] = Field(None, max_length=128)
    use_rag: Optional[bool] = Field(True, description="Use RAG for context retrieval")
    use_llm: Optional[bool] = Field(True, description="Use LLM for response generation")


# ===========================================
# ML Chat Request
# ===========================================

class MLChatRequest(BaseChatRequest):
    """Request model for ML-powered chat"""
    user_location: Optional[Dict[str, float]] = None
    use_llm: Optional[bool] = Field(
        None, 
        description="Override: Use LLM (None=use default based on confidence)"
    )
    user_id: Optional[str] = Field(None, max_length=128)


# ===========================================
# Chat Response Schemas
# ===========================================

class ChatResponseBase(BaseModel):
    """Base response model for all chat endpoints"""
    response: str = Field(..., description="AI response text")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PureLLMChatResponse(ChatResponseBase):
    """Response model for /api/chat/pure-llm"""
    llm_mode: Literal["explain", "clarify", "general", "error"] = Field(
        default="general",
        description="Response mode"
    )
    intent: Optional[str] = Field(None, description="Detected user intent")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Intent confidence (0-1)")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
    map_data: Optional[Dict[str, Any]] = Field(None, description="Map visualization data")
    route_data: Optional[Dict[str, Any]] = Field(None, description="Transportation route data")
    navigation_active: Optional[bool] = Field(None, description="GPS navigation is active")
    navigation_data: Optional[Dict[str, Any]] = Field(None, description="Navigation state")
    interaction_id: Optional[str] = Field(None, description="Interaction ID for feedback")
    response_time_ms: Optional[int] = Field(None, description="Response generation time in ms")


class StreamChatResponse(BaseModel):
    """Response model for streaming events"""
    event: Literal["start", "token", "complete", "error"] = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")


class MLChatResponse(ChatResponseBase):
    """Response model for ML-powered chat"""
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    method: str = Field(..., description="Response generation method")
    context: List[Dict] = Field(default=[], description="Context items used")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    response_time: float = Field(..., description="Response time in seconds")
    ml_service_used: bool = Field(..., description="Whether ML service was used")


# ===========================================
# Error Response Schema
# ===========================================

class ErrorDetail(BaseModel):
    """Detailed error information"""
    field: Optional[str] = None
    message: str
    type: Optional[str] = None


class ChatErrorResponse(BaseModel):
    """Standardized error response for chat endpoints"""
    error: str = Field(..., description="Error message")
    code: str = Field(..., description="Error code (e.g., VAL_3001)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


# ===========================================
# WebSocket Message Schemas
# ===========================================

class WebSocketMessage(BaseModel):
    """WebSocket message format"""
    type: Literal["message", "ping", "pong", "subscribe", "unsubscribe"] = "message"
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebSocketChatMessage(BaseModel):
    """WebSocket chat message"""
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: Optional[str] = None
    language: Optional[str] = "en"

"""
Schemas Module

Pydantic schemas for request/response validation across all API endpoints.
"""

from .chat import (
    # Request schemas
    BaseChatRequest,
    PureLLMChatRequest,
    StreamChatRequest,
    UnifiedChatRequest,
    MLChatRequest,
    
    # Response schemas
    ChatResponseBase,
    PureLLMChatResponse,
    StreamChatResponse,
    MLChatResponse,
    ChatErrorResponse,
    
    # Supporting schemas
    LocationData,
    UserLocation,
    UserPreferences,
    WebSocketMessage,
    WebSocketChatMessage,
    
    # Constants
    MAX_MESSAGE_LENGTH,
    MIN_MESSAGE_LENGTH,
    SUPPORTED_LANGUAGES,
)

__all__ = [
    # Request schemas
    "BaseChatRequest",
    "PureLLMChatRequest", 
    "StreamChatRequest",
    "UnifiedChatRequest",
    "MLChatRequest",
    
    # Response schemas
    "ChatResponseBase",
    "PureLLMChatResponse",
    "StreamChatResponse",
    "MLChatResponse",
    "ChatErrorResponse",
    
    # Supporting schemas
    "LocationData",
    "UserLocation", 
    "UserPreferences",
    "WebSocketMessage",
    "WebSocketChatMessage",
    
    # Constants
    "MAX_MESSAGE_LENGTH",
    "MIN_MESSAGE_LENGTH",
    "SUPPORTED_LANGUAGES",
]

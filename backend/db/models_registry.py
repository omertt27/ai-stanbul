"""
Central models registry - Import all models once at startup
This ensures SQLAlchemy sees all models and prevents duplicate warnings
"""

# Import Base first
from db.base import Base

# Import all models - this registers them with SQLAlchemy
from models import (
    User,
    Place,
    Museum,
    Restaurant,
    Event,
    ChatHistory,
    BlogPost,
    BlogComment,
    BlogLike,
    FeedbackEvent,
    UserInteractionAggregate,
    ItemFeatureVector,
    OnlineLearningModel,
    LocationHistory,
    NavigationSession,
    RouteHistory,
    NavigationEvent,
    UserPreferences,
    ChatSession,
    ConversationHistory,
)

from specialized_models import (
    TransportRoute,
    UserProfile,
    TurkishPhrases,
    LocalTips,
)

# Export everything
__all__ = [
    "Base",
    "User",
    "Place",
    "Museum",
    "Restaurant",
    "Event",
    "ChatHistory",
    "BlogPost",
    "BlogComment",
    "BlogLike",
    "FeedbackEvent",
    "UserInteractionAggregate",
    "ItemFeatureVector",
    "OnlineLearningModel",
    "LocationHistory",
    "NavigationSession",
    "RouteHistory",
    "NavigationEvent",
    "UserPreferences",
    "ChatSession",
    "ConversationHistory",
    "TransportRoute",
    "UserProfile",
    "TurkishPhrases",
    "LocalTips",
]

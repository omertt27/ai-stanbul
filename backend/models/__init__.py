"""
Backend Models Package
"""

# Import from database module
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database import Base

# Import models from the sibling models.py file
try:
    import importlib.util
    models_path = os.path.join(parent_dir, "models.py")
    spec = importlib.util.spec_from_file_location("models_file", models_path)
    if spec and spec.loader:
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        
        # Import models one by one with fallbacks
        Restaurant = getattr(models_module, 'Restaurant', None)
        Museum = getattr(models_module, 'Museum', None)
        Event = getattr(models_module, 'Event', None)
        Place = getattr(models_module, 'Place', None)
        UserFeedback = getattr(models_module, 'UserFeedback', None)
        ChatSession = getattr(models_module, 'ChatSession', None)
        BlogPost = getattr(models_module, 'BlogPost', None)
        BlogComment = getattr(models_module, 'BlogComment', None)
        BlogLike = getattr(models_module, 'BlogLike', None)
        BlogImage = getattr(models_module, 'BlogImage', None)
        ChatHistory = getattr(models_module, 'ChatHistory', None)
        User = getattr(models_module, 'User', None)
        UserSession = getattr(models_module, 'UserSession', None)
        UserInteraction = getattr(models_module, 'UserInteraction', None)
        EnhancedChatHistory = getattr(models_module, 'EnhancedChatHistory', None)
        
        # Real-time learning models (critical for Phase 1)
        FeedbackEvent = getattr(models_module, 'FeedbackEvent', None)
        UserInteractionAggregate = getattr(models_module, 'UserInteractionAggregate', None)
        ItemFeatureVector = getattr(models_module, 'ItemFeatureVector', None)
        OnlineLearningModel = getattr(models_module, 'OnlineLearningModel', None)
        
        # GPS Navigation models (NEW)
        LocationHistory = getattr(models_module, 'LocationHistory', None)
        NavigationSession = getattr(models_module, 'NavigationSession', None)
        RouteHistory = getattr(models_module, 'RouteHistory', None)
        NavigationEvent = getattr(models_module, 'NavigationEvent', None)
        UserPreferences = getattr(models_module, 'UserPreferences', None)
        ConversationHistory = getattr(models_module, 'ConversationHistory', None)
        
        if FeedbackEvent and UserInteractionAggregate:
            print("✅ Real-time learning models imported successfully")
        else:
            print("⚠️ Warning: Some real-time learning models not found in models.py")
        
        if LocationHistory and NavigationSession:
            print("✅ GPS Navigation models imported successfully")
        else:
            print("⚠️ Warning: Some GPS Navigation models not found in models.py")
            
except Exception as e:
    print(f"⚠️ Warning: Could not import from models.py: {e}")
    # Fallback - define minimal models
    Restaurant = None
    Museum = None
    Event = None
    Place = None
    UserFeedback = None
    ChatSession = None
    FeedbackEvent = None
    UserInteractionAggregate = None
    ItemFeatureVector = None
    OnlineLearningModel = None
    BlogPost = None
    BlogComment = None
    BlogLike = None
    BlogImage = None
    ChatHistory = None
    User = None
    UserSession = None
    UserInteraction = None
    EnhancedChatHistory = None
    LocationHistory = None
    NavigationSession = None
    RouteHistory = None
    NavigationEvent = None
    UserPreferences = None
    ConversationHistory = None

from .intent_feedback import IntentFeedback, FeedbackStatistics, create_tables

__all__ = [
    'Base', 'Restaurant', 'Museum', 'Event', 'Place', 'UserFeedback', 'ChatSession', 
    'BlogPost', 'BlogComment', 'BlogLike', 'BlogImage', 'ChatHistory', 'User',
    'IntentFeedback', 'FeedbackStatistics', 'create_tables',
    'FeedbackEvent', 'UserInteractionAggregate', 'ItemFeatureVector', 'OnlineLearningModel',
    'LocationHistory', 'NavigationSession', 'RouteHistory', 'NavigationEvent',
    'UserPreferences', 'ConversationHistory'
]

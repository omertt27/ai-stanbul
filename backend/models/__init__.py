"""
Backend Models Package - Re-export models from models.py
"""

# Import from central Base location
import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from db.base import Base  # Use central Base - single source of truth

# Import models directly from models.py (NOT using importlib to avoid duplicate registration)
try:
    # Change the import path to use the parent directory
    import sys
    original_path = sys.path.copy()
    
    # Import from sibling models.py file
    import models as models_module
    
    # Re-export all models
    User = models_module.User
    Place = models_module.Place
    Museum = models_module.Museum
    Restaurant = models_module.Restaurant
    Event = models_module.Event
    ChatHistory = models_module.ChatHistory
    BlogPost = models_module.BlogPost
    BlogComment = models_module.BlogComment
    BlogLike = models_module.BlogLike
    ChatSession = models_module.ChatSession
    ConversationHistory = models_module.ConversationHistory
    UserPreferences = models_module.UserPreferences
    
    # Real-time learning models
    FeedbackEvent = models_module.FeedbackEvent
    UserInteractionAggregate = models_module.UserInteractionAggregate
    ItemFeatureVector = models_module.ItemFeatureVector
    OnlineLearningModel = models_module.OnlineLearningModel
    
    # GPS Navigation models
    LocationHistory = models_module.LocationHistory
    NavigationSession = models_module.NavigationSession
    RouteHistory = models_module.RouteHistory
    NavigationEvent = models_module.NavigationEvent
    
    print("✅ Real-time learning models imported successfully")
    print("✅ GPS Navigation models imported successfully")
    
except Exception as e:
    print(f"⚠️ Warning: Could not import from models.py: {e}")
    # Set to None if import fails
    User = Place = Museum = Restaurant = Event = ChatHistory = None
    BlogPost = BlogComment = BlogLike = ChatSession = ConversationHistory = None
    FeedbackEvent = UserInteractionAggregate = ItemFeatureVector = OnlineLearningModel = None
    LocationHistory = NavigationSession = RouteHistory = NavigationEvent = UserPreferences = None

# Temporarily comment out to fix SQLAlchemy MetaData conflict
# from .intent_feedback import IntentFeedback, FeedbackStatistics, create_tables
IntentFeedback = None
FeedbackStatistics = None
create_tables = None

__all__ = [
    'Base', 'Restaurant', 'Museum', 'Event', 'Place', 'ChatSession', 
    'BlogPost', 'BlogComment', 'BlogLike', 'ChatHistory', 'User',
    'UserInteractionAggregate', 'FeedbackEvent',
    'FeedbackEvent', 'UserInteractionAggregate', 'ItemFeatureVector', 'OnlineLearningModel',
    'LocationHistory', 'NavigationSession', 'RouteHistory', 'NavigationEvent',
    'UserPreferences', 'ConversationHistory'
]

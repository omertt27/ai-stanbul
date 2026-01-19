"""
Direct models import - bypasses the models/ package to import from models.py file
Use this when you need to import models without the circular dependency issue
"""
import sys
import os

# Get the backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))

# Import models.py directly (not the models/ package)
import importlib.util
spec = importlib.util.spec_from_file_location("models_direct", os.path.join(backend_dir, "models.py"))
models_direct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models_direct)

# Re-export all models
BlogPost = models_direct.BlogPost
BlogComment = models_direct.BlogComment
BlogLike = models_direct.BlogLike
User = models_direct.User
Place = models_direct.Place
Museum = models_direct.Museum
Restaurant = models_direct.Restaurant
Event = models_direct.Event
ChatHistory = models_direct.ChatHistory
ChatSession = models_direct.ChatSession
ConversationHistory = models_direct.ConversationHistory
UserPreferences = models_direct.UserPreferences
FeedbackEvent = models_direct.FeedbackEvent
UserInteractionAggregate = models_direct.UserInteractionAggregate
ItemFeatureVector = models_direct.ItemFeatureVector
OnlineLearningModel = models_direct.OnlineLearningModel
LocationHistory = models_direct.LocationHistory
NavigationSession = models_direct.NavigationSession
RouteHistory = models_direct.RouteHistory
NavigationEvent = models_direct.NavigationEvent

__all__ = [
    'BlogPost', 'BlogComment', 'BlogLike',
    'User', 'Place', 'Museum', 'Restaurant', 'Event',
    'ChatHistory', 'ChatSession', 'ConversationHistory', 'UserPreferences',
    'FeedbackEvent', 'UserInteractionAggregate', 'ItemFeatureVector', 'OnlineLearningModel',
    'LocationHistory', 'NavigationSession', 'RouteHistory', 'NavigationEvent'
]

"""
Backend Models Package - Re-export models from models.py

NOTE: To avoid circular imports, this module uses lazy imports.
Models are only loaded when actually accessed.
"""

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Base directly (this is safe - no circular dependency)
from db.base import Base

# Lazy loading - models are imported only when needed
_models_loaded = False
_models_cache = {}

def _load_models():
    """Lazy load models from models.py to avoid circular imports"""
    global _models_loaded, _models_cache
    
    if _models_loaded:
        return _models_cache
    
    try:
        # Import the parent models.py file directly by its path
        import importlib.util
        models_path = os.path.join(parent_dir, 'models.py')
        spec = importlib.util.spec_from_file_location("models_file", models_path)
        models_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models_module)
        
        # Cache all model classes
        _models_cache = {
            'User': getattr(models_module, 'User', None),
            'Place': getattr(models_module, 'Place', None),
            'Museum': getattr(models_module, 'Museum', None),
            'Restaurant': getattr(models_module, 'Restaurant', None),
            'Event': getattr(models_module, 'Event', None),
            'ChatHistory': getattr(models_module, 'ChatHistory', None),
            'BlogPost': getattr(models_module, 'BlogPost', None),
            'BlogComment': getattr(models_module, 'BlogComment', None),
            'BlogLike': getattr(models_module, 'BlogLike', None),
            'ChatSession': getattr(models_module, 'ChatSession', None),
            'ConversationHistory': getattr(models_module, 'ConversationHistory', None),
            'UserPreferences': getattr(models_module, 'UserPreferences', None),
            'FeedbackEvent': getattr(models_module, 'FeedbackEvent', None),
            'UserInteractionAggregate': getattr(models_module, 'UserInteractionAggregate', None),
            'ItemFeatureVector': getattr(models_module, 'ItemFeatureVector', None),
            'OnlineLearningModel': getattr(models_module, 'OnlineLearningModel', None),
            'LocationHistory': getattr(models_module, 'LocationHistory', None),
            'NavigationSession': getattr(models_module, 'NavigationSession', None),
            'RouteHistory': getattr(models_module, 'RouteHistory', None),
            'NavigationEvent': getattr(models_module, 'NavigationEvent', None),
        }
        _models_loaded = True
        
    except Exception as e:
        print(f"⚠️ Warning: Could not load models: {e}")
        _models_cache = {}
        _models_loaded = True
    
    return _models_cache

def __getattr__(name):
    """Lazy attribute access for models"""
    if name == 'Base':
        return Base
    
    models = _load_models()
    if name in models:
        return models[name]
    
    raise AttributeError(f"module 'models' has no attribute '{name}'")

# Temporarily disabled
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

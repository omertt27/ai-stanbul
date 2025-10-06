"""
Enhanced Ultra-Specialized Istanbul AI Integration Module
Connects all specialized Istanbul AI systems with full database integration.

This module provides the production-ready enhanced system with direct database connectivity.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import time
import json
import re
from pathlib import Path

# Import the new enhancement services
try:
    from services.integrated_ai_enhancement_service import IntegratedAIEnhancementService, UserContext
    from services.user_feedback_service import UserFeedbackService
    from services.seasonal_calendar_service import SeasonalCalendarService
    from services.daily_life_suggestions_service import DailyLifeSuggestionsService
    from services.enhanced_attractions_service import EnhancedAttractionsService
    from services.scraping_curation_service import ScrapingCurationService
    ENHANCEMENT_SERVICES_AVAILABLE = True
    print("✅ AI Enhancement Services loaded successfully!")
except ImportError as e:
    print(f"⚠️ AI Enhancement Services not available: {e}")
    ENHANCEMENT_SERVICES_AVAILABLE = False
    # Create mock classes for fallback
    class IntegratedAIEnhancementService:
        def __init__(self, *args, **kwargs): pass
        def get_enhanced_recommendations(self, *args, **kwargs): return {'primary_suggestions': []}
        def close(self): pass
    
    class UserContext:
        def __init__(self, user_type="tourist", **kwargs):
            self.user_type = user_type
            self.visit_history = []
            self.preferences = {}
            self.current_location = None
            self.group_size = 1
            self.duration = "1day"
            self.budget_level = "medium"
            self.interests = []

# Database Integration Classes
class IstanbulDatabaseManager:
    """Enhanced database manager for Istanbul AI system"""

    def __init__(self, data_path=None):
        if data_path is None:
            # Use relative path from backend directory
            self.data_path = Path(__file__).parent / "data"
        else:
            self.data_path = Path(data_path)

        self.restaurants = []
        self.attractions = []
        self.museums = {}
        self.cultural_data = {}
        self.cache = {}
        
        # Initialize AI Enhancement Services
        if ENHANCEMENT_SERVICES_AVAILABLE:
            try:
                self.enhancement_service = IntegratedAIEnhancementService()
                print("🚀 AI Enhancement Services initialized in database manager")
            except Exception as e:
                print(f"⚠️ Error initializing enhancement services: {e}")
                self.enhancement_service = None
        else:
            self.enhancement_service = None
        
        self._load_all_databases()

    def _load_all_databases(self):
        """Load all database sources"""
        # Load expanded restaurant database (500+ entries) - fallback to original if needed
        expanded_restaurant_file = self.data_path / "restaurants_database_expanded.json" 
        restaurant_file = self.data_path / "restaurants_database.json"
        
        restaurant_file_to_use = expanded_restaurant_file if expanded_restaurant_file.exists() else restaurant_file
        
        if restaurant_file_to_use.exists():
            try:
                with open(restaurant_file_to_use, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.restaurants = data.get('restaurants', [])
            except Exception as e:
                print(f"Warning: Could not load restaurant database: {e}")

        # Load expanded attractions database (100+ entries) - fallback to original if needed
        expanded_attractions_file = self.data_path / "attractions_database_expanded.json"
        attractions_file = self.data_path / "attractions_database.json"
        
        attractions_file_to_use = expanded_attractions_file if expanded_attractions_file.exists() else attractions_file
        
        if attractions_file_to_use.exists():
            try:
                with open(attractions_file_to_use, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Handle both original format (attractions dict) and expanded format (niche_attractions list)
                    if 'niche_attractions' in data:
                        self.attractions = data['niche_attractions']
                    else:
                        self.attractions = data.get('attractions', [])
                        # Convert dict format to list if needed
                        if isinstance(self.attractions, dict):
                            self.attractions = list(self.attractions.values())
            except Exception as e:
                print(f"Warning: Could not load attractions database: {e}")

        # Load museum database (from Python module)
        try:
            import sys
            sys.path.append(str(self.data_path.parent))
            from accurate_museum_database import istanbul_museums
            self.museums = istanbul_museums.museums
        except ImportError:
            print("Warning: Could not load museum database")

        # Load cultural and seasonal data
        cultural_file = self.data_path / "cultural_seasonal_data.json"
        if cultural_file.exists():
            try:
                with open(cultural_file, 'r', encoding='utf-8') as f:
                    self.cultural_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cultural data: {e}")
                self.cultural_data = {}
        else:
            self.cultural_data = {}

        print(f"📚 Enhanced Database Manager initialized:")
        print(f"   🍽️ Restaurants: {len(self.restaurants)}")
        print(f"   🎭 Attractions: {len(self.attractions)}")
        print(f"   🏛️ Museums: {len(self.museums)}")
        print(f"   🎪 Cultural Data: {'Available' if self.cultural_data else 'Not available'}")
        print(f"   🚀 AI Enhancements: {'Active' if self.enhancement_service else 'Disabled'}")
    
    def get_enhanced_recommendations(self, query: str, user_context: Dict = None) -> Dict[str, Any]:
        """Get recommendations enhanced with new AI services"""
        if not self.enhancement_service:
            return {'primary_suggestions': self.attractions[:5]}  # Fallback
        
        # Create user context
        if user_context is None:
            user_context = {}
        
        context = UserContext(
            user_type=user_context.get('user_type', 'tourist'),
            visit_history=user_context.get('visit_history', []),
            preferences=user_context.get('preferences', {}),
            group_size=user_context.get('group_size', 1),
            duration=user_context.get('duration', '1day'),
            budget_level=user_context.get('budget_level', 'medium'),
            interests=user_context.get('interests', [])
        )
        
        # Get enhanced recommendations
        return self.enhancement_service.get_enhanced_recommendations(context, query)
    
    def add_user_feedback(self, user_id: str, attraction_name: str, ratings: Dict[str, float],
                         comment: str = "", user_type: str = "tourist") -> bool:
        """Add user feedback to improve recommendations"""
        if not self.enhancement_service:
            return False
        
        # Find attraction ID to

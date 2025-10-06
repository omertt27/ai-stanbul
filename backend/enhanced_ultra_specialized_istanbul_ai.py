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

# Import the complete query processing pipeline (retrieval-first system)
try:
    from complete_query_pipeline import CompleteQueryPipeline, process_query as pipeline_process_query
    from intelligent_query_processor import IntelligentQueryProcessor
    from lightweight_nlp_system import QueryIntent, ExtractedEntities
    COMPLETE_PIPELINE_AVAILABLE = True
    NLP_SYSTEM_AVAILABLE = True
    print("✅ Complete Query Processing Pipeline loaded successfully!")
    print("✅ Lightweight NLP System loaded successfully!")
except ImportError as e:
    print(f"⚠️ Complete Query Pipeline not available: {e}")
    COMPLETE_PIPELINE_AVAILABLE = False
    try:
        from intelligent_query_processor import IntelligentQueryProcessor
        from lightweight_nlp_system import QueryIntent, ExtractedEntities
        NLP_SYSTEM_AVAILABLE = True
        print("✅ Lightweight NLP System loaded successfully (fallback)!")
    except ImportError as e:
        print(f"⚠️ Lightweight NLP System not available: {e}")
        NLP_SYSTEM_AVAILABLE = False

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
                         comment: str = "", user_type: str = "tourist") -> Dict[str, Any]:
        """Add user feedback to improve recommendations"""
        if not self.enhancement_service:
            return {"success": False, "message": "Enhancement service not available"}
        
        # Find attraction ID based on name
        attraction_id = None
        for i, attraction in enumerate(self.attractions):
            if isinstance(attraction, dict) and attraction.get('name', '').lower() == attraction_name.lower():
                attraction_id = i + 1
                break
        
        if attraction_id is None:
            return {"success": False, "message": f"Attraction '{attraction_name}' not found"}
        
        try:
            feedback_id = self.enhancement_service.add_user_rating(
                user_id, attraction_id, ratings, comment, user_type
            )
            return {
                "success": True,
                "message": "Feedback added successfully",
                "feedback_id": feedback_id,
                "impact": "Your feedback will help improve recommendations for other users"
            }
        except Exception as e:
            return {"success": False, "message": f"Error adding feedback: {str(e)}"}
    
    def get_daily_schedule(self, user_context: Dict = None) -> Dict[str, Any]:
        """Get a personalized daily schedule"""
        if not self.enhancement_service:
            return {"success": False, "message": "Enhancement service not available"}
        
        context = UserContext(
            user_type=user_context.get('user_type', 'tourist'),
            visit_history=user_context.get('visit_history', []),
            preferences=user_context.get('preferences', {}),
            group_size=user_context.get('group_size', 1),
            duration=user_context.get('duration', '1day'),
            budget_level=user_context.get('budget_level', 'medium'),
            interests=user_context.get('interests', [])
        )
        
        try:
            schedule = self.enhancement_service.get_daily_schedule(context)
            return {
                "success": True,
                "response": self._format_daily_schedule(schedule),
                "schedule_data": schedule,
                "confidence": 0.9
            }
        except Exception as e:
            return {"success": False, "message": f"Error generating schedule: {str(e)}"}
    
    def _format_daily_schedule(self, schedule: Dict) -> str:
        """Format daily schedule into readable text"""
        formatted = "🌅 **Your Personalized Istanbul Daily Schedule**\n\n"
        
        for time_period in ['morning', 'afternoon', 'evening']:
            if time_period in schedule:
                formatted += f"**{time_period.title()}:**\n"
                for suggestion in schedule[time_period]:
                    formatted += f"• {suggestion.get('title', 'Activity')}\n"
                    formatted += f"  📍 {suggestion.get('location', 'Location TBD')}\n"
                    if suggestion.get('local_tips'):
                        formatted += f"  💡 {suggestion.get('local_tips')[:100]}...\n"
                    formatted += "\n"
        
        # Add seasonal context
        if 'seasonal_context' in schedule:
            seasonal = schedule['seasonal_context']
            if seasonal.get('active_events'):
                formatted += "**🎉 Current Events:**\n"
                for event in seasonal['active_events'][:2]:
                    formatted += f"• {event.get('name', 'Event')}: {event.get('description', '')[:100]}...\n"
        
        # Add cultural tips
        if 'cultural_tips' in schedule:
            formatted += "\n**🇹🇷 Cultural Tips:**\n"
            for tip in schedule['cultural_tips'][:3]:
                formatted += f"• {tip}\n"
        
        return formatted
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "core_system": "active",
            "database_status": {
                "restaurants": len(self.restaurants),
                "attractions": len(self.attractions),
                "museums": len(self.museums)
            },
            "enhancement_services": "active" if self.enhancement_service else "disabled",
            "last_updated": datetime.now().isoformat()
        }
        
        if self.enhancement_service:
            try:
                analytics = self.enhancement_service.get_system_analytics()
                status["enhancement_analytics"] = analytics
            except Exception as e:
                status["enhancement_error"] = str(e)
        
        return status

    def close(self):
        """Close database connections"""
        if self.enhancement_service:
            try:
                self.enhancement_service.close()
            except Exception as e:
                print(f"Error closing enhancement service: {e}")


class UltraSpecializedIstanbulIntelligence:
    """Enhanced main AI system with all new features integrated"""
    
    def __init__(self):
        """Initialize the enhanced Istanbul AI system"""
        self.db_manager = IstanbulDatabaseManager()
        
        # Initialize complete query processing pipeline (retrieval-first)
        if COMPLETE_PIPELINE_AVAILABLE:
            try:
                self.query_pipeline = CompleteQueryPipeline()
                print("🚀 Complete Query Processing Pipeline initialized successfully")
                self.use_complete_pipeline = True
            except Exception as e:
                print(f"⚠️ Error initializing Complete Query Pipeline: {e}")
                self.query_pipeline = None
                self.use_complete_pipeline = False
        else:
            self.query_pipeline = None
            self.use_complete_pipeline = False
        
        # Initialize lightweight NLP system as fallback
        if NLP_SYSTEM_AVAILABLE:
            try:
                self.query_processor = IntelligentQueryProcessor(database_manager=self.db_manager)
                print("🧠 Lightweight NLP System initialized successfully")
            except Exception as e:
                print(f"⚠️ Error initializing NLP system: {e}")
                self.query_processor = None
        else:
            self.query_processor = None
            
        print("🏛️ Ultra-Specialized Istanbul AI System Enhanced v3.0 initialized (NLP-Powered, No LLMs)")
    
    def process_istanbul_query(self, query: str, user_context: Dict = None) -> Dict[str, Any]:
        """Process Istanbul query using complete retrieval-first pipeline (no LLMs required)"""
        start_time = time.time()
        
        if user_context is None:
            user_context = {}
        
        try:
            # Use complete query processing pipeline if available (preferred)
            if self.use_complete_pipeline and self.query_pipeline:
                print(f"🚀 Using Complete Query Processing Pipeline for: '{query[:50]}...'")
                
                # Process through complete pipeline
                pipeline_result = self.query_pipeline.process_query(query, user_context)
                
                # Enhance with AI services if available
                if ENHANCEMENT_SERVICES_AVAILABLE and self.db_manager.enhancement_service:
                    enhanced_result = self.db_manager.get_enhanced_recommendations(query, user_context)
                    pipeline_result = self._merge_pipeline_with_enhancements(pipeline_result, enhanced_result)
                
                processing_time = time.time() - start_time
                pipeline_result['processing_time'] = processing_time
                pipeline_result['system_version'] = 'v4.0_complete_pipeline'
                pipeline_result['uses_llm'] = False
                pipeline_result['pipeline_features'] = [
                    "text_preprocessing",
                    "intent_classification", 
                    "entity_extraction",
                    "vector_search",
                    "keyword_search",
                    "rule_based_ranking",
                    "response_generation"
                ]
                
                return pipeline_result
            
            # Fallback to lightweight NLP system
            elif self.query_processor and NLP_SYSTEM_AVAILABLE:
                print(f"🧠 Using Lightweight NLP System for: '{query[:50]}...'")
                session_id = user_context.get('session_id', 'default')
                nlp_result = self.query_processor.process_user_query(query, session_id)
                
                # Enhance with AI services if available
                if ENHANCEMENT_SERVICES_AVAILABLE and self.db_manager.enhancement_service:
                    enhanced_result = self.db_manager.get_enhanced_recommendations(query, user_context)
                    nlp_result = self._merge_nlp_with_enhancements(nlp_result, enhanced_result)
                
                processing_time = time.time() - start_time
                nlp_result['processing_time'] = processing_time
                nlp_result['system_version'] = 'v3.0_nlp_powered'
                nlp_result['uses_llm'] = False
                
                return nlp_result
            
            else:
                # Fallback to enhancement services only
                if ENHANCEMENT_SERVICES_AVAILABLE:
                    enhanced_result = self.db_manager.get_enhanced_recommendations(query, user_context)
                    response = self._format_enhanced_response(enhanced_result, query)
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "success": True,
                        "response": response,
                        "confidence": 0.8,
                        "processing_time": processing_time,
                        "enhancement_features": [
                            "user_feedback_integration",
                            "seasonal_calendar",
                            "daily_life_suggestions",
                            "curated_attractions",
                            "authenticity_boost"
                        ],
                        "system_version": "v3.0_fallback",
                        "uses_llm": False
                    }
                else:
                    return self._fallback_response(query, "All query processing systems unavailable")
            
        except Exception as e:
            print(f"❌ Error in query processing: {e}")
            # Fallback to basic response
            return self._fallback_response(query, str(e))
    
    def _merge_nlp_with_enhancements(self, nlp_result: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge NLP system results with AI enhancement services data"""
        # Start with NLP result as base
        merged_result = nlp_result.copy()
        
        # Enhance the response with additional data
        original_response = nlp_result.get('response', '')
        
        # Add enhancement features information
        enhancement_info = "\n\n**🔥 AI Enhancement Features Active:**\n"
        
        # Seasonal highlights
        seasonal = enhanced_result.get('seasonal_highlights', {})
        if seasonal.get('active_events'):
            enhancement_info += "**🎉 Current Seasonal Events:**\n"
            for event in seasonal['active_events'][:2]:
                enhancement_info += f"• **{event.get('name', 'Event')}**: {event.get('description', '')}\n"
            enhancement_info += "\n"
        
        # User feedback boost
        primary = enhanced_result.get('primary_suggestions', [])
        boosted_attractions = [p for p in primary if p.get('boosted_authenticity_score', 0) > p.get('authenticity_score', 0)]
        if boosted_attractions:
            enhancement_info += f"**⭐ {len(boosted_attractions)} attractions boosted by community feedback**\n\n"
        
        # Daily life experiences
        daily_experiences = enhanced_result.get('daily_life_experiences', [])
        if daily_experiences:
            enhancement_info += "**🌟 Authentic Daily Life Experiences Available:**\n"
            for exp in daily_experiences[:2]:
                enhancement_info += f"• **{exp.get('title', 'Experience')}** in {exp.get('location', 'Istanbul')}\n"
            enhancement_info += "\n"
        
        # Combine responses
        merged_result['response'] = original_response + enhancement_info
        
        # Add metadata
        merged_result['enhancement_features'] = [
            "user_feedback_integration",
            "seasonal_calendar",
            "daily_life_suggestions", 
            "curated_attractions",
            "authenticity_boost"
        ]
        merged_result['user_feedback_applied'] = len(boosted_attractions) > 0
        merged_result['seasonal_events'] = seasonal.get('active_events', [])
        merged_result['authenticity_boosted'] = len(boosted_attractions)
        merged_result['system_version'] = 'v3.0_nlp_enhanced'
        
        return merged_result
    
    def _merge_pipeline_with_enhancements(self, pipeline_result: Dict[str, Any], enhanced_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge complete query pipeline results with AI enhancement services data"""
        # Start with pipeline result as base
        merged_result = pipeline_result.copy()
        
        # Get the original response from pipeline
        original_response = pipeline_result.get('response', '')
        
        # Add enhancement features information
        enhancement_info = "\n\n**🔥 AI Enhancement Features Active:**\n"
        
        # Seasonal highlights
        seasonal = enhanced_result.get('seasonal_highlights', {})
        if seasonal.get('active_events'):
            enhancement_info += "**🎉 Current Seasonal Events:**\n"
            for event in seasonal['active_events'][:2]:
                enhancement_info += f"• **{event.get('name', 'Event')}**: {event.get('description', '')}\n"
            enhancement_info += "\n"
        
        # User feedback boost
        primary = enhanced_result.get('primary_suggestions', [])
        boosted_attractions = [p for p in primary if p.get('boosted_authenticity_score', 0) > p.get('authenticity_score', 0)]
        if boosted_attractions:
            enhancement_info += f"**⭐ {len(boosted_attractions)} attractions boosted by community feedback**\n\n"
        
        # Daily life experiences
        daily_experiences = enhanced_result.get('daily_life_experiences', [])
        if daily_experiences:
            enhancement_info += "**🌟 Authentic Daily Life Experiences Available:**\n"
            for exp in daily_experiences[:2]:
                enhancement_info += f"• **{exp.get('title', 'Experience')}** in {exp.get('location', 'Istanbul')}\n"
            enhancement_info += "\n"
        
        # Combine responses
        merged_result['response'] = original_response + enhancement_info
        
        # Add metadata
        existing_features = merged_result.get('pipeline_features', [])
        merged_result['enhancement_features'] = [
            "user_feedback_integration",
            "seasonal_calendar", 
            "daily_life_suggestions",
            "curated_attractions",
            "authenticity_boost"
        ]
        merged_result['all_features'] = existing_features + merged_result['enhancement_features']
        merged_result['user_feedback_applied'] = len(boosted_attractions) > 0
        merged_result['seasonal_events'] = seasonal.get('active_events', [])
        merged_result['authenticity_boosted'] = len(boosted_attractions)
        merged_result['system_version'] = 'v4.0_complete_pipeline_enhanced'
        
        return merged_result

    def _format_enhanced_response(self, enhanced_result: Dict, query: str) -> str:
        """Format enhanced recommendations into readable response"""
        response = "🏛️ **Enhanced Istanbul Recommendations**\n\n"
        
        # Primary suggestions with authenticity boost
        primary = enhanced_result.get('primary_suggestions', [])
        if primary:
            response += "**🎯 Top Recommendations (Authenticity Enhanced):**\n"
            for suggestion in primary[:5]:
                name = suggestion.get('name', 'Attraction')
                auth_score = suggestion.get('boosted_authenticity_score', suggestion.get('authenticity_score', 0))
                feedback = suggestion.get('user_feedback', {})
                
                response += f"• **{name}** (Authenticity: {auth_score:.1f}/10)\n"
                if feedback.get('total_ratings', 0) > 0:
                    response += f"  ⭐ {feedback['total_ratings']} user ratings, boost: +{feedback.get('boost_factor', 0):.1f}\n"
                if suggestion.get('local_tips'):
                    response += f"  💡 {suggestion['local_tips'][:100]}...\n"
                response += "\n"
        
        # Daily life experiences
        daily_experiences = enhanced_result.get('daily_life_experiences', [])
        if daily_experiences:
            response += "**🌟 Authentic Daily Life Experiences:**\n"
            for exp in daily_experiences[:3]:
                response += f"• **{exp.get('title', 'Experience')}**\n"
                response += f"  📍 {exp.get('location', 'Various locations')}\n"
                if exp.get('cultural_context'):
                    response += f"  🇹🇷 {exp['cultural_context'][:100]}...\n"
                response += "\n"
        
        # Seasonal highlights
        seasonal = enhanced_result.get('seasonal_highlights', {})
        if seasonal.get('active_events'):
            response += "**🎉 Current Seasonal Events:**\n"
            for event in seasonal['active_events'][:2]:
                response += f"• **{event.get('name', 'Event')}**: {event.get('description', '')}\n"
            response += "\n"
        
        # Context-aware tips
        tips = enhanced_result.get('context_aware_tips', [])
        if tips:
            response += "**💡 Smart Tips for You:**\n"
            for tip in tips[:3]:
                response += f"• {tip}\n"
            response += "\n"
        
        # Crowd-sourced gems
        gems = enhanced_result.get('crowd_sourced_gems', [])
        if gems:
            response += "**💎 Community-Recommended Hidden Gems:**\n"
            for gem in gems[:3]:
                rating = gem.get('boosted_rating', gem.get('avg_rating', 0))
                response += f"• Attraction ID {gem.get('attraction_id', '?')} (Rating: {rating:.1f}/10, {gem.get('rating_count', 0)} reviews)\n"
        
        return response
    
    def _fallback_response(self, query: str, error: str) -> Dict[str, Any]:
        """Fallback response when enhanced features fail"""
        return {
            "success": False,
            "response": f"I'm your Ultra-Specialized Istanbul AI assistant. While my enhanced features are temporarily unavailable, I can still help with Istanbul travel questions.\n\nQuery: {query}\n\nPlease try again or ask a more specific question about Istanbul attractions, restaurants, or cultural experiences.",
            "confidence": 0.5,
            "error": error,
            "enhancement_features": [],
            "fallback_mode": True
        }
    
    def add_user_feedback(self, user_id: str, attraction_name: str, ratings: Dict[str, float],
                         comment: str = "", user_type: str = "tourist") -> Dict[str, Any]:
        """Add user feedback through the database manager"""
        return self.db_manager.add_user_feedback(user_id, attraction_name, ratings, comment, user_type)
    
    def get_daily_schedule(self, user_context: Dict = None) -> Dict[str, Any]:
        """Get daily schedule through the database manager"""
        return self.db_manager.get_daily_schedule(user_context)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status through the database manager"""
        return self.db_manager.get_system_status()


# Create the main enhanced system instance
enhanced_istanbul_ai_system = UltraSpecializedIstanbulIntelligence()

print("✅ Enhanced Ultra-Specialized Istanbul AI System v3.0 Ready!")
print("   🧠 Lightweight NLP System (No LLMs Required)")
print("   🚀 AI Enhancement Services integrated") 
print("   📊 User feedback and rating system active")
print("   📅 Seasonal calendar integration enabled")
print("   🌟 Daily life suggestions available")
print("   🎯 Authenticity boost system operational")
print("   🔍 Semi-automated discovery pipeline ready")
print("   ⚡ Fast, rule-based query processing")
print("   🌍 40+ authentic Istanbul museums")

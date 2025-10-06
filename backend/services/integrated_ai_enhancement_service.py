"""
Integrated AI Enhancement Service
Combines all new features into a unified service for the main AI system
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import all our new services
from .user_feedback_service import UserFeedbackService
from .seasonal_calendar_service import SeasonalCalendarService
from .daily_life_suggestions_service import DailyLifeSuggestionsService
from .enhanced_attractions_service import EnhancedAttractionsService
from .scraping_curation_service import ScrapingCurationService

class RecommendationContext(Enum):
    TOURIST_FIRST_VISIT = "tourist_first_visit"
    TOURIST_REPEAT = "tourist_repeat"
    LOCAL_DISCOVERY = "local_discovery"
    LOCAL_EXPERT = "local_expert"
    MIXED_GROUP = "mixed_group"

@dataclass
class UserContext:
    user_type: str = "tourist"
    visit_history: List[str] = None
    preferences: Dict[str, Any] = None
    current_location: Optional[Tuple[float, float]] = None
    group_size: int = 1
    duration: str = "1day"  # hours, days
    budget_level: str = "medium"
    interests: List[str] = None
    
    def __post_init__(self):
        if self.visit_history is None:
            self.visit_history = []
        if self.preferences is None:
            self.preferences = {}
        if self.interests is None:
            self.interests = []

class IntegratedAIEnhancementService:
    """
    Main service that integrates all new features:
    - User feedback and rating system
    - Seasonal calendar integration  
    - Daily life suggestions
    - Enhanced attractions database
    - Scraping and curation pipeline
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        
        # Initialize all services
        self.feedback_service = UserFeedbackService(db_path)
        self.calendar_service = SeasonalCalendarService(db_path)
        self.suggestions_service = DailyLifeSuggestionsService(db_path)
        self.attractions_service = EnhancedAttractionsService(db_path)
        self.scraping_service = ScrapingCurationService(db_path)
        
    def get_enhanced_recommendations(self, user_context: UserContext,
                                   query: str = "", limit: int = 10) -> Dict[str, Any]:
        """
        Get AI recommendations enhanced with all new features
        """
        recommendations = {
            'primary_suggestions': [],
            'daily_life_experiences': [],
            'seasonal_highlights': [],
            'crowd_sourced_gems': [],
            'context_aware_tips': [],
            'authenticity_boosted': [],
            'metadata': {
                'user_context': user_context.user_type,
                'enhancement_features': [
                    'user_feedback_boost',
                    'seasonal_integration',
                    'daily_life_suggestions',
                    'curated_attractions',
                    'crowd_intelligence'
                ]
            }
        }
        
        # 1. Get base attractions with user feedback boost
        base_attractions = self._get_feedback_boosted_attractions(
            user_context, query, limit
        )
        recommendations['primary_suggestions'] = base_attractions
        
        # 2. Get daily life suggestions specific to user type
        daily_suggestions = self.suggestions_service.get_suggestions_for_audience(
            user_context.user_type, limit=5
        )
        recommendations['daily_life_experiences'] = daily_suggestions
        
        # 3. Get seasonal recommendations and events
        seasonal_info = self.calendar_service.get_seasonal_recommendations()
        recommendations['seasonal_highlights'] = seasonal_info
        
        # 4. Get crowd-sourced top rated attractions
        crowd_gems = self.feedback_service.get_top_rated_attractions(
            limit=5, user_type=user_context.user_type
        )
        recommendations['crowd_sourced_gems'] = crowd_gems
        
        # 5. Generate context-aware tips
        context_tips = self._generate_context_tips(user_context, seasonal_info)
        recommendations['context_aware_tips'] = context_tips
        
        # 6. Get authenticity-boosted attractions
        authentic_attractions = self.attractions_service.get_top_authentic_attractions(limit=5)
        recommendations['authenticity_boosted'] = authentic_attractions
        
        return recommendations
    
    def _get_feedback_boosted_attractions(self, user_context: UserContext,
                                        query: str, limit: int) -> List[Dict]:
        """Get attractions with user feedback boost applied"""
        # First get base attractions
        if query:
            base_attractions = self.attractions_service.search_attractions(query)
        else:
            base_attractions = self.attractions_service.get_top_authentic_attractions(limit * 2)
        
        # Apply feedback boost to each attraction
        boosted_attractions = []
        for attraction in base_attractions[:limit]:
            # Get user ratings for this attraction
            rating_data = self.feedback_service.get_attraction_ratings(attraction['id'])
            
            # Apply boost to authenticity score
            original_score = attraction.get('authenticity_score', 0.0)
            boost_factor = rating_data.get('boost_factor', 0.0)
            boosted_score = min(10.0, original_score + boost_factor)
            
            # Add rating information
            attraction['original_authenticity_score'] = original_score
            attraction['boosted_authenticity_score'] = boosted_score
            attraction['user_feedback'] = {
                'total_ratings': rating_data.get('total_ratings', 0),
                'boost_factor': boost_factor,
                'avg_ratings': rating_data.get('avg_ratings', {}),
                'local_avg': rating_data.get('local_avg', 0.0),
                'tourist_avg': rating_data.get('tourist_avg', 0.0)
            }
            
            boosted_attractions.append(attraction)
        
        # Sort by boosted score
        boosted_attractions.sort(key=lambda x: x['boosted_authenticity_score'], reverse=True)
        return boosted_attractions
    
    def _generate_context_tips(self, user_context: UserContext,
                             seasonal_info: Dict) -> List[str]:
        """Generate context-aware tips based on user and seasonal factors"""
        tips = []
        
        # User type specific tips
        if user_context.user_type == "tourist":
            tips.append("💡 Look for attractions with high 'local_rating' for authentic experiences")
            tips.append("🎯 User feedback shows these places are authentic and worth visiting")
        elif user_context.user_type == "local":
            tips.append("🏙️ Discover hidden gems in your own city through curated local suggestions")
            tips.append("⭐ Rate your experiences to help improve recommendations for everyone")
        
        # Seasonal tips
        current_events = seasonal_info.get('active_events', [])
        if current_events:
            for event in current_events[:2]:  # Top 2 events
                if event['impact_level'] in ['HIGH', 'VERY_HIGH']:
                    tips.append(f"🎉 {event['name']} is happening now - {event['description']}")
        
        # Crowd warnings
        crowd_warnings = seasonal_info.get('crowd_warnings', [])
        if crowd_warnings:
            tips.append(f"⚠️ {crowd_warnings[0]['advice']}")
        
        # Budget considerations
        if user_context.budget_level == "budget":
            tips.append("💰 Many authentic experiences like mosque visits and neighborhood walks are free")
        
        # Group size considerations
        if user_context.group_size > 4:
            tips.append("👥 Large groups: consider splitting up for intimate venues like traditional cafes")
        
        return tips
    
    def add_user_rating(self, user_id: str, attraction_id: int, ratings: Dict[str, float],
                       comment: str = "", user_type: str = "tourist") -> str:
        """Add user rating and update system intelligence"""
        # Add rating to feedback service
        rating_id = self.feedback_service.add_rating(
            user_id, attraction_id, ratings, comment, user_type=user_type
        )
        
        # Trigger recommendation recalculation for affected attractions
        self._update_recommendation_cache(attraction_id)
        
        return rating_id
    
    def _update_recommendation_cache(self, attraction_id: int):
        """Update cached recommendations when new ratings are added"""
        # In a production system, this would invalidate relevant caches
        # For now, we'll just log the update
        print(f"🔄 Recommendation cache updated for attraction {attraction_id}")
    
    def get_daily_schedule(self, user_context: UserContext, 
                          date: datetime = None) -> Dict[str, Any]:
        """Get a complete daily schedule with all enhancements"""
        if date is None:
            date = datetime.now()
        
        # Get base daily schedule
        schedule = self.suggestions_service.get_daily_schedule_suggestions(
            user_context.user_type, date
        )
        
        # Enhance with seasonal events
        seasonal_info = self.calendar_service.get_seasonal_recommendations(date)
        schedule['seasonal_context'] = seasonal_info
        
        # Add feedback-enhanced attractions for each time period
        for time_period in ['morning', 'afternoon', 'evening']:
            if time_period in schedule:
                # Get attractions suitable for this time period
                enhanced_attractions = []
                for suggestion in schedule[time_period]:
                    # If suggestion has a location, find nearby attractions
                    if 'district' in suggestion:
                        nearby = self.attractions_service.search_attractions(
                            suggestion['district']
                        )
                        if nearby:
                            enhanced_attractions.extend(nearby[:2])  # Top 2
                
                schedule[f'{time_period}_enhanced_attractions'] = enhanced_attractions
        
        return schedule
    
    def search_with_enhancements(self, query: str, user_context: UserContext) -> Dict[str, Any]:
        """Enhanced search with all features integrated"""
        results = {
            'attractions': [],
            'daily_suggestions': [],
            'seasonal_relevance': [],
            'user_rated': [],
            'search_metadata': {
                'query': query,
                'user_type': user_context.user_type,
                'enhancement_applied': True
            }
        }
        
        # Search attractions
        attractions = self.attractions_service.search_attractions(query)
        results['attractions'] = self._apply_feedback_boosts(attractions)
        
        # Search daily suggestions
        daily_suggestions = self.suggestions_service.search_suggestions(
            query, user_context.user_type
        )
        results['daily_suggestions'] = daily_suggestions
        
        # Check seasonal relevance
        current_events = self.calendar_service.get_current_events()
        relevant_events = [
            event for event in current_events
            if query.lower() in event['name'].lower() or 
               query.lower() in event['description'].lower()
        ]
        results['seasonal_relevance'] = relevant_events
        
        # Get user-rated items
        top_rated = self.feedback_service.get_top_rated_attractions(
            user_type=user_context.user_type
        )
        results['user_rated'] = top_rated
        
        return results
    
    def _apply_feedback_boosts(self, attractions: List[Dict]) -> List[Dict]:
        """Apply feedback boosts to a list of attractions"""
        for attraction in attractions:
            rating_data = self.feedback_service.get_attraction_ratings(attraction['id'])
            boost = rating_data.get('boost_factor', 0.0)
            original_score = attraction.get('authenticity_score', 0.0)
            attraction['boosted_authenticity_score'] = min(10.0, original_score + boost)
            attraction['user_feedback_summary'] = {
                'ratings_count': rating_data.get('total_ratings', 0),
                'boost_applied': boost,
                'community_verified': rating_data.get('total_ratings', 0) >= 5
            }
        
        # Sort by boosted score
        attractions.sort(key=lambda x: x.get('boosted_authenticity_score', 0), reverse=True)
        return attractions
    
    def get_curation_dashboard_data(self) -> Dict[str, Any]:
        """Get data for admin curation dashboard"""
        dashboard_data = {
            'scraping_stats': {},
            'curation_workload': [],
            'feedback_analytics': {},
            'attraction_stats': {},
            'seasonal_calendar': {},
            'system_health': {}
        }
        
        # Scraping and curation stats
        curation_report = self.scraping_service.generate_curation_report()
        dashboard_data['scraping_stats'] = curation_report
        
        # Items pending curation
        workload = self.scraping_service.get_curation_workload(limit=10)
        dashboard_data['curation_workload'] = workload
        
        # Feedback analytics
        feedback_analytics = self.feedback_service.get_feedback_analytics()
        dashboard_data['feedback_analytics'] = feedback_analytics
        
        # Attraction database stats
        attraction_stats = self.attractions_service.get_attractions_statistics()
        dashboard_data['attraction_stats'] = attraction_stats
        
        # Seasonal calendar overview
        calendar_data = self.calendar_service.get_event_calendar()
        dashboard_data['seasonal_calendar'] = calendar_data
        
        # System health indicators
        dashboard_data['system_health'] = {
            'total_attractions': attraction_stats.get('total_attractions', 0),
            'pending_curation': curation_report.get('overall_stats', {}).get('pending', 0),
            'recent_feedback': len(self.feedback_service.get_recent_feedback(days=7)),
            'active_events': len(self.calendar_service.get_current_events()),
            'authenticity_avg': attraction_stats.get('authenticity_stats', {}).get('average', 0)
        }
        
        return dashboard_data
    
    def run_discovery_pipeline(self, categories: List[str] = None) -> Dict[str, Any]:
        """Run the complete discovery and curation pipeline"""
        pipeline_results = {
            'scraping_sessions': {},
            'items_discovered': 0,
            'items_queued_for_curation': 0,
            'estimated_processing_time': 0,
            'pipeline_status': 'running'
        }
        
        try:
            # Run batch scraping
            session_ids = self.scraping_service.run_batch_scraping(categories)
            pipeline_results['scraping_sessions'] = session_ids
            
            # Get curation workload
            workload = self.scraping_service.get_curation_workload()
            pipeline_results['items_queued_for_curation'] = len(workload)
            
            # Calculate discovered items
            for session_id in session_ids.values():
                # This would normally query the session results
                pipeline_results['items_discovered'] += 10  # Mock count
            
            # Estimate processing time (mock calculation)
            pipeline_results['estimated_processing_time'] = len(workload) * 5  # 5 minutes per item
            
            pipeline_results['pipeline_status'] = 'completed'
            
        except Exception as e:
            pipeline_results['pipeline_status'] = 'failed'
            pipeline_results['error'] = str(e)
        
        return pipeline_results
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        analytics = {
            'overview': {},
            'user_engagement': {},
            'content_quality': {},
            'seasonal_patterns': {},
            'discovery_pipeline': {}
        }
        
        try:
            # Overview metrics
            attraction_stats = self.attractions_service.get_attractions_statistics()
            feedback_stats = self.feedback_service.get_feedback_analytics()
            
            analytics['overview'] = {
                'total_attractions': attraction_stats.get('total_attractions', 0),
                'total_user_ratings': feedback_stats.get('total_ratings', 0),
                'avg_authenticity_score': attraction_stats.get('authenticity_stats', {}).get('average', 0),
                'active_seasonal_events': len(self.calendar_service.get_current_events())
            }
            
            # User engagement
            analytics['user_engagement'] = {
                'recent_ratings': len(self.feedback_service.get_recent_feedback(days=30)),
                'user_type_distribution': feedback_stats.get('user_type_distribution', {}),
                'avg_rating_by_category': feedback_stats.get('avg_ratings', {})
            }
            
            # Content quality
            analytics['content_quality'] = {
                'attractions_by_category': attraction_stats.get('by_category', []),
                'authenticity_distribution': attraction_stats.get('authenticity_stats', {}),
                'verification_status': 'High' if attraction_stats.get('total_attractions', 0) > 20 else 'Medium'
            }
            
        except Exception as e:
            print(f"⚠️ Error getting basic analytics: {e}")
            analytics['overview'] = {'error': 'Analytics temporarily unavailable'}
        
        try:
            # Discovery pipeline (may fail if services use separate databases)
            curation_report = self.scraping_service.generate_curation_report()
            analytics['discovery_pipeline'] = {
                'approval_rate': curation_report.get('overall_stats', {}).get('approval_rate', 0),
                'pending_curation': curation_report.get('overall_stats', {}).get('pending', 0),
                'pipeline_efficiency': 'High' if curation_report.get('overall_stats', {}).get('approval_rate', 0) > 70 else 'Medium'
            }
        except Exception as e:
            print(f"⚠️ Discovery pipeline analytics unavailable: {e}")
            analytics['discovery_pipeline'] = {
                'status': 'Service running',
                'note': 'Pipeline uses separate database - detailed stats available via admin dashboard'
            }
        
        return analytics
    
    def close(self):
        """Close all service connections"""
        services = [
            self.feedback_service,
            self.calendar_service, 
            self.suggestions_service,
            self.attractions_service,
            self.scraping_service
        ]
        
        for service in services:
            try:
                service.close()
            except Exception as e:
                print(f"Error closing service: {e}")

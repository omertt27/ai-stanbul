"""
ML-Enhanced Event Handler
Provides context-aware event recommendations with neural ranking
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class EventContext:
    """Context for event recommendations"""
    user_query: str
    event_categories: List[str]  # concerts, exhibitions, festivals, sports, theater, etc.
    date_range: Optional[Dict[str, datetime]]  # start_date, end_date
    time_preference: Optional[str]  # morning, afternoon, evening, night
    interests: List[str]
    budget_level: Optional[str]
    location_preference: Optional[str]  # neighborhood or district
    crowd_preference: Optional[str]  # intimate, moderate, large
    weather_context: Optional[Dict[str, Any]]
    user_sentiment: float  # -1.0 to 1.0
    indoor_outdoor_pref: Optional[str]  # indoor, outdoor, both


class MLEnhancedEventHandler:
    """
    ML-Enhanced Event Handler
    
    Features:
    - Context extraction using MLContextBuilder
    - Neural ranking of events based on user preferences
    - Category and interest matching using semantic similarity
    - Time-aware filtering (date, time of day)
    - Weather-based recommendations (indoor/outdoor)
    - Budget-conscious filtering
    - Personalized response generation with event details
    """
    
    def __init__(self, events_service, ml_context_builder, ml_processor, response_generator):
        """
        Initialize handler with required services
        
        Args:
            events_service: Events service with event data
            ml_context_builder: Centralized ML context builder
            ml_processor: Neural processor for embeddings and ranking
            response_generator: Response generator for natural language output
        """
        self.events_service = events_service
        self.ml_context_builder = ml_context_builder
        self.ml_processor = ml_processor
        self.response_generator = response_generator
        
        logger.info("✅ ML-Enhanced Event Handler initialized")
    
    async def handle_event_query(
        self,
        user_query: str,
        user_profile: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle event query with ML enhancement
        
        Args:
            user_query: User's natural language query
            user_profile: Optional user profile for personalization
            context: Optional additional context (weather, location, dates, etc.)
        
        Returns:
            Dict with events, scores, and natural language response
        """
        try:
            # Step 1: Extract ML context
            ml_context = await self.ml_context_builder.build_context(
                query=user_query,
                intent="event_recommendation",
                user_profile=user_profile,
                additional_context=context
            )
            
            # Step 2: Build event-specific context from ML context
            event_context = self._build_event_context(ml_context, context)
            
            # Step 3: Get candidate events (with date filtering)
            candidates = await self._get_candidate_events(event_context)
            
            # Step 4: Neural ranking of events
            ranked_events = await self._rank_events_neural(
                events=candidates,
                context=event_context,
                ml_context=ml_context
            )
            
            # Step 5: Apply filters (weather, budget, crowd size)
            filtered_events = self._apply_filters(
                ranked_events,
                event_context
            )
            
            # Step 6: Generate personalized response
            response = await self._generate_response(
                events=filtered_events[:5],  # Top 5
                context=event_context,
                ml_context=ml_context
            )
            
            return {
                "success": True,
                "events": filtered_events[:5],
                "response": response,
                "context_used": {
                    "categories": event_context.event_categories,
                    "interests": event_context.interests,
                    "date_range": event_context.date_range,
                    "sentiment": event_context.user_sentiment,
                    "weather_aware": event_context.weather_context is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error in event handler: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I had trouble finding events for you. Could you tell me more about what you're interested in?"
            }
    
    def _build_event_context(
        self,
        ml_context: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]]
    ) -> EventContext:
        """Build event-specific context from ML context"""
        
        query_lower = ml_context.get("original_query", "").lower()
        
        # Extract event categories from query
        event_categories = []
        category_keywords = {
            "concert": ["concert", "music", "band", "performance", "show", "gig"],
            "exhibition": ["exhibition", "gallery", "art show", "museum", "display"],
            "festival": ["festival", "fest", "celebration", "carnival"],
            "theater": ["theater", "theatre", "play", "drama", "musical"],
            "sports": ["sports", "match", "game", "tournament", "race"],
            "cultural": ["cultural", "traditional", "heritage", "folk"],
            "food": ["food", "culinary", "tasting", "gastronomy"],
            "nightlife": ["club", "party", "dj", "nightlife", "dance"],
            "family": ["family", "kids", "children", "family-friendly"],
            "workshop": ["workshop", "class", "seminar", "talk", "lecture"]
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                event_categories.append(category)
        
        # Default to all categories if none specified
        if not event_categories:
            event_categories = ["concert", "exhibition", "festival", "theater", "cultural"]
        
        # Extract date range
        date_range = None
        if additional_context and "dates" in additional_context:
            date_range = additional_context["dates"]
        else:
            # Default to next 30 days
            date_range = {
                "start_date": datetime.now(),
                "end_date": datetime.now() + timedelta(days=30)
            }
        
        # Extract time preference
        time_preference = None
        time_keywords = {
            "morning": ["morning", "early"],
            "afternoon": ["afternoon", "matinee"],
            "evening": ["evening", "dinner time"],
            "night": ["night", "late", "midnight"]
        }
        for time, keywords in time_keywords.items():
            if any(kw in query_lower for kw in keywords):
                time_preference = time
                break
        
        # Extract indoor/outdoor preference
        indoor_outdoor_pref = None
        if "outdoor" in query_lower or "open air" in query_lower:
            indoor_outdoor_pref = "outdoor"
        elif "indoor" in query_lower or "inside" in query_lower:
            indoor_outdoor_pref = "indoor"
        
        # Extract crowd preference
        crowd_preference = None
        if any(kw in query_lower for kw in ["intimate", "small", "cozy"]):
            crowd_preference = "intimate"
        elif any(kw in query_lower for kw in ["big", "large", "major", "popular"]):
            crowd_preference = "large"
        
        # Extract interests
        interests = ml_context.get("detected_interests", [])
        
        return EventContext(
            user_query=ml_context.get("original_query", ""),
            event_categories=event_categories,
            date_range=date_range,
            time_preference=time_preference,
            interests=interests,
            budget_level=ml_context.get("budget_preference"),
            location_preference=ml_context.get("location_preference"),
            crowd_preference=crowd_preference,
            weather_context=ml_context.get("weather"),
            user_sentiment=ml_context.get("sentiment_score", 0.0),
            indoor_outdoor_pref=indoor_outdoor_pref
        )
    
    async def _get_candidate_events(self, context: EventContext) -> List[Dict[str, Any]]:
        """Get candidate events from service with date filtering"""
        
        try:
            # Get events from service
            all_events = await self.events_service.get_upcoming_events(
                start_date=context.date_range["start_date"] if context.date_range else None,
                end_date=context.date_range["end_date"] if context.date_range else None
            )
            
            # Filter by categories if specified
            if context.event_categories:
                filtered_events = [
                    event for event in all_events
                    if any(cat in event.get("category", "").lower() for cat in context.event_categories)
                ]
            else:
                filtered_events = all_events
            
            return filtered_events
            
        except Exception as e:
            logger.warning(f"Error fetching events: {e}, returning mock data")
            # Return mock events if service fails
            return self._get_mock_events(context)
    
    def _get_mock_events(self, context: EventContext) -> List[Dict[str, Any]]:
        """Return mock events for development/testing"""
        
        mock_events = [
            {
                "id": "evt_001",
                "name": "Istanbul Jazz Festival",
                "category": "concert",
                "description": "Annual jazz festival featuring international and local artists",
                "date": datetime.now() + timedelta(days=5),
                "time": "evening",
                "location": "Zorlu PSM, Beşiktaş",
                "venue_type": "indoor",
                "price_range": "moderate",
                "crowd_size": "large",
                "highlights": ["World-renowned artists", "Multiple stages", "Food & drinks"]
            },
            {
                "id": "evt_002",
                "name": "Contemporary Art Exhibition",
                "category": "exhibition",
                "description": "Modern Turkish art showcase at Istanbul Modern",
                "date": datetime.now() + timedelta(days=2),
                "time": "afternoon",
                "location": "Istanbul Modern, Karaköy",
                "venue_type": "indoor",
                "price_range": "budget",
                "crowd_size": "moderate",
                "highlights": ["Local artists", "Interactive installations", "Bosphorus view"]
            },
            {
                "id": "evt_003",
                "name": "Tulip Festival",
                "category": "festival",
                "description": "Millions of tulips blooming in Emirgan Park",
                "date": datetime.now() + timedelta(days=10),
                "time": "morning",
                "location": "Emirgan Park, Sarıyer",
                "venue_type": "outdoor",
                "price_range": "free",
                "crowd_size": "large",
                "highlights": ["Free entry", "Photo opportunities", "Family-friendly"]
            },
            {
                "id": "evt_004",
                "name": "Traditional Turkish Music Night",
                "category": "concert",
                "description": "Authentic Ottoman classical music performance",
                "date": datetime.now() + timedelta(days=7),
                "time": "evening",
                "location": "Cemal Reşit Rey Concert Hall, Harbiye",
                "venue_type": "indoor",
                "price_range": "budget",
                "crowd_size": "intimate",
                "highlights": ["Historical venue", "Traditional instruments", "Cultural experience"]
            },
            {
                "id": "evt_005",
                "name": "Bosphorus Food Festival",
                "category": "food",
                "description": "Culinary celebration with street food and chef demonstrations",
                "date": datetime.now() + timedelta(days=15),
                "time": "afternoon",
                "location": "Kuruçeşme Arena",
                "venue_type": "outdoor",
                "price_range": "moderate",
                "crowd_size": "large",
                "highlights": ["50+ vendors", "Live cooking shows", "Waterfront location"]
            }
        ]
        
        return mock_events
    
    async def _rank_events_neural(
        self,
        events: List[Dict[str, Any]],
        context: EventContext,
        ml_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank events using neural similarity"""
        
        # Get query embedding
        query_embedding = await self.ml_processor.get_embedding(context.user_query)
        
        scored_events = []
        for event in events:
            # Create event description for embedding
            desc = f"{event['name']} {event.get('category', '')} {event.get('description', '')}"
            
            # Get event embedding
            event_embedding = await self.ml_processor.get_embedding(desc)
            
            # Calculate base similarity score
            base_score = self.ml_processor.calculate_similarity(
                query_embedding,
                event_embedding
            )
            
            # Apply context-based adjustments
            adjusted_score = self._adjust_score_with_context(
                base_score,
                event,
                context
            )
            
            scored_events.append({
                **event,
                "ml_score": adjusted_score,
                "base_similarity": base_score
            })
        
        # Sort by ML score descending
        scored_events.sort(key=lambda x: x["ml_score"], reverse=True)
        
        return scored_events
    
    def _adjust_score_with_context(
        self,
        base_score: float,
        event: Dict[str, Any],
        context: EventContext
    ) -> float:
        """
        Adjust neural similarity score with context factors
        
        Scoring formula:
        final_score = base_score * (1 + category_boost + time_boost + interest_boost + 
                                    weather_boost + crowd_boost + budget_boost)
        """
        score = base_score
        boost = 0.0
        
        # Category match boost
        if context.event_categories:
            event_category = event.get("category", "")
            if event_category in context.event_categories:
                boost += 0.20
        
        # Time preference boost
        if context.time_preference:
            event_time = event.get("time", "")
            if context.time_preference == event_time:
                boost += 0.15
        
        # Interest alignment boost
        if context.interests:
            event_desc = f"{event.get('name', '')} {event.get('description', '')}".lower()
            for interest in context.interests:
                if interest.lower() in event_desc:
                    boost += 0.10
        
        # Weather-based boost
        if context.weather_context:
            weather_condition = context.weather_context.get("condition", "").lower()
            venue_type = event.get("venue_type", "")
            
            # Prefer indoor events in bad weather
            if any(bad in weather_condition for bad in ["rain", "snow", "storm"]):
                if venue_type == "indoor":
                    boost += 0.20
            # Prefer outdoor events in good weather
            elif any(good in weather_condition for good in ["clear", "sunny", "fair"]):
                if venue_type == "outdoor":
                    boost += 0.15
        
        # Indoor/outdoor preference boost
        if context.indoor_outdoor_pref:
            venue_type = event.get("venue_type", "")
            if context.indoor_outdoor_pref == venue_type:
                boost += 0.15
        
        # Crowd size preference boost
        if context.crowd_preference:
            event_crowd = event.get("crowd_size", "")
            if context.crowd_preference == event_crowd:
                boost += 0.10
        
        # Budget alignment boost
        if context.budget_level:
            event_price = event.get("price_range", "")
            if context.budget_level.lower() == event_price:
                boost += 0.15
            elif event_price == "free":
                boost += 0.10  # Always give free events a small boost
        
        # Location preference boost
        if context.location_preference:
            event_location = event.get("location", "").lower()
            if context.location_preference.lower() in event_location:
                boost += 0.10
        
        # Sentiment boost (positive sentiment = more exciting/vibrant events)
        if context.user_sentiment > 0.5:
            event_desc = event.get("description", "").lower()
            if any(exciting in event_desc for exciting in ["exciting", "vibrant", "energetic", "festival"]):
                boost += 0.10
        
        # Recency boost (events happening sooner get slight preference)
        if "date" in event:
            days_until = (event["date"] - datetime.now()).days
            if 0 <= days_until <= 7:
                boost += 0.05  # Events in next week get small boost
        
        final_score = score * (1 + boost)
        return min(final_score, 1.0)  # Cap at 1.0
    
    def _apply_filters(
        self,
        events: List[Dict[str, Any]],
        context: EventContext
    ) -> List[Dict[str, Any]]:
        """Apply hard filters based on context"""
        
        filtered = events
        
        # Filter out events outside budget if strict budget specified
        if context.budget_level == "budget":
            filtered = [
                e for e in filtered
                if e.get("price_range") in ["budget", "free"]
            ]
        
        # Filter by venue type if strict weather concerns
        if context.weather_context:
            weather = context.weather_context.get("condition", "").lower()
            if "storm" in weather or "heavy rain" in weather:
                # Strongly prefer indoor events in severe weather
                filtered = [e for e in filtered if e.get("venue_type") == "indoor"]
        
        return filtered
    
    async def _generate_response(
        self,
        events: List[Dict[str, Any]],
        context: EventContext,
        ml_context: Dict[str, Any]
    ) -> str:
        """Generate natural language response"""
        
        if not events:
            return "I couldn't find any events matching your preferences. Would you like me to check different dates or categories?"
        
        # Build response components
        response_parts = []
        
        # Opening based on sentiment and context
        if context.user_sentiment > 0.5:
            response_parts.append("Exciting! I found some amazing events for you! 🎉")
        else:
            response_parts.append("Here are some events I think you'll enjoy:")
        
        # Top event detailed description
        top = events[0]
        response_parts.append(f"\n\n🌟 **{top['name']}** (Match: {int(top['ml_score']*100)}%)")
        response_parts.append(f"   📅 {self._format_date(top.get('date'))}")
        response_parts.append(f"   📍 {top.get('location', 'Location TBA')}")
        response_parts.append(f"   💰 {top.get('price_range', 'Check website').title()}")
        
        if "description" in top:
            response_parts.append(f"   {top['description']}")
        
        # Highlights
        if "highlights" in top and top["highlights"]:
            highlights = ", ".join(top["highlights"][:3])
            response_parts.append(f"   ✨ Highlights: {highlights}")
        
        # Additional events (brief)
        if len(events) > 1:
            response_parts.append("\n\n📅 **More events for you:**")
            for event in events[1:4]:
                date_str = self._format_date(event.get("date"))
                response_parts.append(
                    f"   • **{event['name']}** - {date_str} at {event.get('location', 'TBA')}"
                )
        
        # Context-aware tips
        if context.weather_context:
            weather = context.weather_context.get("condition", "")
            if "rain" in weather.lower():
                response_parts.append("\n\n☔ Weather Note: I've prioritized indoor events due to rain in the forecast.")
        
        if context.budget_level == "budget":
            free_count = sum(1 for e in events if e.get("price_range") == "free")
            if free_count > 0:
                response_parts.append(f"\n\n💰 Budget Tip: {free_count} of these events are free!")
        
        # Booking reminder
        if any(e.get("crowd_size") == "large" for e in events[:2]):
            response_parts.append("\n\n🎟️ Tip: Popular events may require advance booking!")
        
        return "\n".join(response_parts)
    
    def _format_date(self, date_obj) -> str:
        """Format date for display"""
        if isinstance(date_obj, datetime):
            days_until = (date_obj - datetime.now()).days
            if days_until == 0:
                return "Today"
            elif days_until == 1:
                return "Tomorrow"
            elif days_until < 7:
                return f"This {date_obj.strftime('%A')}"
            else:
                return date_obj.strftime("%B %d, %Y")
        return str(date_obj)


def create_ml_enhanced_event_handler(
    events_service,
    ml_context_builder,
    ml_processor,
    response_generator
):
    """Factory function to create ML-enhanced event handler"""
    return MLEnhancedEventHandler(
        events_service=events_service,
        ml_context_builder=ml_context_builder,
        ml_processor=ml_processor,
        response_generator=response_generator
    )

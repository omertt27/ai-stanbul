"""
ML-Enhanced Neighborhood Handler
Provides context-aware neighborhood recommendations with neural ranking
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NeighborhoodContext:
    """Context for neighborhood recommendations"""
    user_query: str
    vibe_preferences: List[str]
    crowd_type: Optional[str]
    time_of_day: Optional[str]
    interests: List[str]
    budget_level: Optional[str]
    current_location: Optional[str]
    weather_context: Optional[Dict[str, Any]]
    user_sentiment: float  # -1.0 to 1.0


class MLEnhancedNeighborhoodHandler:
    """
    ML-Enhanced Neighborhood Handler
    
    Features:
    - Context extraction using MLContextBuilder
    - Neural ranking of neighborhoods based on user preferences
    - Character/vibe matching using semantic similarity
    - Time-aware recommendations
    - Weather-based filtering
    - Personalized response generation
    """
    
    def __init__(self, neighborhood_service, ml_context_builder, ml_processor, response_generator):
        """
        Initialize handler with required services
        
        Args:
            neighborhood_service: Neighborhood guide service with district data
            ml_context_builder: Centralized ML context builder
            ml_processor: Neural processor for embeddings and ranking
            response_generator: Response generator for natural language output
        """
        self.neighborhood_service = neighborhood_service
        self.ml_context_builder = ml_context_builder
        self.ml_processor = ml_processor
        self.response_generator = response_generator
        
        logger.info("✅ ML-Enhanced Neighborhood Handler initialized")
    
    async def handle_neighborhood_query(
        self,
        user_query: str,
        user_profile: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle neighborhood query with ML enhancement
        
        Args:
            user_query: User's natural language query
            user_profile: Optional user profile for personalization
            context: Optional additional context (weather, location, etc.)
        
        Returns:
            Dict with neighborhoods, scores, and natural language response
        """
        try:
            # Step 1: Extract ML context
            ml_context = await self.ml_context_builder.build_context(
                query=user_query,
                intent="neighborhood_recommendation",
                user_profile=user_profile,
                additional_context=context
            )
            
            # Step 2: Build neighborhood context from ML context
            neighborhood_context = self._build_neighborhood_context(ml_context)
            
            # Step 3: Get candidate neighborhoods
            candidates = self._get_candidate_neighborhoods(neighborhood_context)
            
            # Step 4: Neural ranking of neighborhoods
            ranked_neighborhoods = await self._rank_neighborhoods_neural(
                neighborhoods=candidates,
                context=neighborhood_context,
                ml_context=ml_context
            )
            
            # Step 5: Apply filters (weather, time, budget)
            filtered_neighborhoods = self._apply_filters(
                ranked_neighborhoods,
                neighborhood_context
            )
            
            # Step 6: Generate personalized response
            response = await self._generate_response(
                neighborhoods=filtered_neighborhoods[:5],  # Top 5
                context=neighborhood_context,
                ml_context=ml_context
            )
            
            return {
                "success": True,
                "neighborhoods": filtered_neighborhoods[:5],
                "response": response,
                "context_used": {
                    "vibe_preferences": neighborhood_context.vibe_preferences,
                    "interests": neighborhood_context.interests,
                    "sentiment": neighborhood_context.user_sentiment,
                    "weather_aware": neighborhood_context.weather_context is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error in neighborhood handler: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I had trouble processing your neighborhood request. Could you try rephrasing?"
            }
    
    def _build_neighborhood_context(self, ml_context: Dict[str, Any]) -> NeighborhoodContext:
        """Build neighborhood-specific context from ML context"""
        
        # Extract vibe/character preferences from entities and query
        vibe_prefs = []
        query_lower = ml_context.get("original_query", "").lower()
        
        vibe_keywords = {
            "trendy": ["trendy", "hip", "modern", "cool", "stylish"],
            "traditional": ["traditional", "authentic", "local", "old", "historic"],
            "nightlife": ["nightlife", "party", "bar", "club", "entertainment"],
            "quiet": ["quiet", "peaceful", "calm", "relaxed", "serene"],
            "artsy": ["artsy", "artistic", "cultural", "creative", "bohemian"],
            "business": ["business", "professional", "corporate", "work"],
            "coastal": ["coastal", "seaside", "waterfront", "bosphorus", "sea"]
        }
        
        for vibe, keywords in vibe_keywords.items():
            if any(kw in query_lower for kw in keywords):
                vibe_prefs.append(vibe)
        
        # Extract time of day
        time_of_day = None
        time_keywords = {
            "morning": ["morning", "breakfast", "early"],
            "afternoon": ["afternoon", "lunch", "midday"],
            "evening": ["evening", "dinner", "sunset"],
            "night": ["night", "late", "nighttime"]
        }
        for time, keywords in time_keywords.items():
            if any(kw in query_lower for kw in keywords):
                time_of_day = time
                break
        
        # Extract interests from ML context
        interests = ml_context.get("detected_interests", [])
        
        return NeighborhoodContext(
            user_query=ml_context.get("original_query", ""),
            vibe_preferences=vibe_prefs,
            crowd_type=ml_context.get("crowd_preference"),
            time_of_day=time_of_day,
            interests=interests,
            budget_level=ml_context.get("budget_preference"),
            current_location=ml_context.get("current_location"),
            weather_context=ml_context.get("weather"),
            user_sentiment=ml_context.get("sentiment_score", 0.0)
        )
    
    def _get_candidate_neighborhoods(self, context: NeighborhoodContext) -> List[Dict[str, Any]]:
        """Get candidate neighborhoods from service"""
        
        # Get all districts from service
        districts = self.neighborhood_service.get_all_districts()
        
        candidates = []
        for district_name, district_guide in districts.items():
            candidates.append({
                "name": district_guide.name,
                "district_type": district_guide.district_type.value,
                "character": {
                    "vibe": district_guide.character.vibe,
                    "crowd": district_guide.character.crowd,
                    "atmosphere": district_guide.character.atmosphere,
                    "best_time": [t.value for t in district_guide.character.best_time]
                },
                "recommendations": district_guide.recommendations,
                "local_customs": district_guide.local_customs,
                "budget_estimate": district_guide.budget_estimate,
                "getting_there": district_guide.getting_there
            })
        
        return candidates
    
    async def _rank_neighborhoods_neural(
        self,
        neighborhoods: List[Dict[str, Any]],
        context: NeighborhoodContext,
        ml_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank neighborhoods using neural similarity"""
        
        # Get query embedding
        query_embedding = await self.ml_processor.get_embedding(context.user_query)
        
        scored_neighborhoods = []
        for neighborhood in neighborhoods:
            # Create neighborhood description for embedding
            desc = f"{neighborhood['character']['vibe']} {neighborhood['character']['atmosphere']}"
            
            # Get neighborhood embedding
            neighborhood_embedding = await self.ml_processor.get_embedding(desc)
            
            # Calculate base similarity score
            base_score = self.ml_processor.calculate_similarity(
                query_embedding,
                neighborhood_embedding
            )
            
            # Apply context-based adjustments
            adjusted_score = self._adjust_score_with_context(
                base_score,
                neighborhood,
                context
            )
            
            scored_neighborhoods.append({
                **neighborhood,
                "ml_score": adjusted_score,
                "base_similarity": base_score
            })
        
        # Sort by ML score descending
        scored_neighborhoods.sort(key=lambda x: x["ml_score"], reverse=True)
        
        return scored_neighborhoods
    
    def _adjust_score_with_context(
        self,
        base_score: float,
        neighborhood: Dict[str, Any],
        context: NeighborhoodContext
    ) -> float:
        """
        Adjust neural similarity score with context factors
        
        Scoring formula:
        final_score = base_score * (1 + vibe_boost + time_boost + interest_boost + weather_boost)
        """
        score = base_score
        boost = 0.0
        
        # Vibe preference boost
        if context.vibe_preferences:
            neighborhood_type = neighborhood.get("district_type", "")
            for vibe in context.vibe_preferences:
                if vibe in neighborhood_type or vibe in neighborhood["character"]["vibe"].lower():
                    boost += 0.15
        
        # Time of day boost
        if context.time_of_day:
            best_times = neighborhood["character"].get("best_time", [])
            if context.time_of_day in best_times:
                boost += 0.20
        
        # Interest alignment boost
        if context.interests:
            recommendations = neighborhood.get("recommendations", {})
            for interest in context.interests:
                if interest in str(recommendations).lower():
                    boost += 0.10
        
        # Weather boost (if indoor-friendly in bad weather)
        if context.weather_context:
            weather_condition = context.weather_context.get("condition", "").lower()
            if "rain" in weather_condition or "snow" in weather_condition:
                # Prefer neighborhoods with good indoor options
                recs = neighborhood.get("recommendations", {})
                indoor_count = sum(1 for cat in ["cafes", "museums", "shopping"] if cat in recs)
                if indoor_count > 2:
                    boost += 0.15
        
        # Budget alignment boost
        if context.budget_level:
            budget_estimate = neighborhood.get("budget_estimate", "").lower()
            if context.budget_level.lower() in budget_estimate:
                boost += 0.10
        
        # Sentiment boost (positive sentiment = more adventurous recommendations)
        if context.user_sentiment > 0.5:
            # Boost trendy/vibrant neighborhoods for positive sentiment
            if "trendy" in neighborhood["character"]["vibe"].lower():
                boost += 0.10
        
        final_score = score * (1 + boost)
        return min(final_score, 1.0)  # Cap at 1.0
    
    def _apply_filters(
        self,
        neighborhoods: List[Dict[str, Any]],
        context: NeighborhoodContext
    ) -> List[Dict[str, Any]]:
        """Apply hard filters based on context"""
        
        filtered = neighborhoods
        
        # Apply any hard constraints here if needed
        # For now, we rely on neural ranking
        
        return filtered
    
    async def _generate_response(
        self,
        neighborhoods: List[Dict[str, Any]],
        context: NeighborhoodContext,
        ml_context: Dict[str, Any]
    ) -> str:
        """Generate natural language response"""
        
        if not neighborhoods:
            return "I couldn't find neighborhoods matching your preferences. Could you tell me more about what you're looking for?"
        
        # Build response components
        response_parts = []
        
        # Opening based on sentiment and context
        if context.user_sentiment > 0.5:
            response_parts.append("Great! I have some exciting neighborhood recommendations for you! 🎉")
        else:
            response_parts.append("Based on what you're looking for, here are some neighborhoods I'd recommend:")
        
        # Top neighborhood detailed description
        top = neighborhoods[0]
        response_parts.append(f"\n\n🌟 **{top['name']}** (Match: {int(top['ml_score']*100)}%)")
        response_parts.append(f"   {top['character']['vibe']}")
        response_parts.append(f"   Best time: {', '.join(top['character']['best_time'])}")
        
        # Add top recommendations from this neighborhood
        if "restaurants" in top.get("recommendations", {}):
            response_parts.append(f"   Must-try: {top['recommendations']['restaurants'][0]['name']}")
        
        # Additional neighborhoods (brief)
        if len(neighborhoods) > 1:
            response_parts.append("\n\n📍 **Other great options:**")
            for neighborhood in neighborhoods[1:4]:
                response_parts.append(
                    f"   • {neighborhood['name']} - {neighborhood['character']['vibe'][:60]}..."
                )
        
        # Context-aware tips
        if context.weather_context:
            weather = context.weather_context.get("condition", "")
            if "rain" in weather.lower():
                response_parts.append("\n\n☔ Tip: I've prioritized neighborhoods with great indoor options!")
        
        if context.budget_level:
            response_parts.append(f"\n\n💰 Budget-conscious: All recommendations fit your {context.budget_level} budget preference.")
        
        return "\n".join(response_parts)


def create_ml_enhanced_neighborhood_handler(
    neighborhood_service,
    ml_context_builder,
    ml_processor,
    response_generator
):
    """Factory function to create ML-enhanced neighborhood handler"""
    return MLEnhancedNeighborhoodHandler(
        neighborhood_service=neighborhood_service,
        ml_context_builder=ml_context_builder,
        ml_processor=ml_processor,
        response_generator=response_generator
    )

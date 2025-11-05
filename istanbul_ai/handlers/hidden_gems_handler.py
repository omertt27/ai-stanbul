"""
ML-Enhanced Hidden Gems Handler
Provides context-aware hidden gem recommendations with neural ranking
🌐 Full English/Turkish bilingual support

Updated: December 19, 2024 - Added bilingual support
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Import bilingual support
try:
    from ..services.bilingual_manager import BilingualManager, Language
    BILINGUAL_AVAILABLE = True
except ImportError:
    BILINGUAL_AVAILABLE = False
    Language = None

logger = logging.getLogger(__name__)


@dataclass
class HiddenGemContext:
    """Context for hidden gem recommendations"""
    user_query: str
    gem_types: List[str]  # cafes, rooftops, parks, shops, historical, viewpoints
    neighborhood_preference: Optional[str]
    crowd_level: str  # 'very_quiet', 'quiet', 'moderate'
    authenticity_score: float  # 0.0-1.0, higher = more authentic/local
    interests: List[str]
    tourist_comfort: float  # 0.0-1.0, how tourist-friendly should it be
    accessibility_needs: Optional[List[str]]
    budget_level: Optional[str]
    time_of_day: Optional[str]
    weather_context: Optional[Dict[str, Any]]
    user_sentiment: float


class MLEnhancedHiddenGemsHandler:
    """
    ML-Enhanced Hidden Gems Handler
    
    Features:
    - Context extraction for authentic, local experiences
    - Neural ranking based on "hidden" quality and user preferences
    - Authenticity scoring (local vs tourist)
    - Crowd-level filtering
    - Neighborhood and type matching
    - Time-aware and weather-aware filtering
    - Personalized descriptions emphasizing unique qualities
    """
    
    def __init__(self, hidden_gems_service, ml_context_builder, ml_processor, response_generator,
                 bilingual_manager=None, map_integration_service=None):
        """
        Initialize handler with required services
        
        Args:
            hidden_gems_service: Hidden gems service with gem data
            ml_context_builder: Centralized ML context builder
            ml_processor: Neural processor for embeddings and ranking
            response_generator: Response generator for natural language output
            bilingual_manager: BilingualManager for language support
            map_integration_service: MapIntegrationService for map visualization
        """
        self.hidden_gems_service = hidden_gems_service
        self.ml_context_builder = ml_context_builder
        self.ml_processor = ml_processor
        self.response_generator = response_generator
        self.bilingual_manager = bilingual_manager
        self.map_integration_service = map_integration_service
        self.has_bilingual = bilingual_manager is not None and BILINGUAL_AVAILABLE
        self.has_maps = map_integration_service is not None and map_integration_service.is_enabled()
        
        logger.info(f"✅ ML-Enhanced Hidden Gems Handler initialized (Bilingual: {self.has_bilingual}, Maps: {self.has_maps})")
    
    def _extract_language(self, context: Optional[Dict[str, Any]]) -> Language:
        """Extract language from context or detect from query"""
        if not self.has_bilingual:
            return None
        
        # Check context for language
        if context and "language" in context:
            lang_str = context["language"]
            if lang_str == "tr":
                return Language.TURKISH
            elif lang_str == "en":
                return Language.ENGLISH
        
        # Default to English
        return Language.ENGLISH
    
    async def handle_hidden_gems_query(
        self,
        user_query: str,
        user_profile: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle hidden gems query with ML enhancement
        
        Args:
            user_query: User's natural language query
            user_profile: Optional user profile for personalization
            context: Optional additional context (must include 'language' key)
        
        Returns:
            Dict with hidden gems, scores, and natural language response
        """
        # Extract language for bilingual support
        language = self._extract_language(context)
        
        try:
            # Step 1: Extract ML context
            ml_context = await self.ml_context_builder.build_context(
                query=user_query,
                intent="hidden_gems",
                user_profile=user_profile,
                additional_context={
                    **(context or {}),
                    "language": language.value if language else "en"
                }
            )
            
            # Step 2: Build hidden gem context
            gem_context = self._build_gem_context(ml_context, context)
            
            # Step 3: Get candidate hidden gems
            candidates = await self._get_candidate_gems(gem_context)
            
            # Step 4: Neural ranking with authenticity scoring
            ranked_gems = await self._rank_gems_neural(
                gems=candidates,
                context=gem_context,
                ml_context=ml_context
            )
            
            # Step 5: Apply filters (crowd level, authenticity, accessibility)
            filtered_gems = self._apply_filters(
                ranked_gems,
                gem_context
            )
            
            # Step 6: Generate personalized response
            response = await self._generate_response(
                gems=filtered_gems[:5],
                context=gem_context,
                ml_context=ml_context,
                language=language
            )
            
            # Step 7: Generate map visualization for hidden gems
            map_data = None
            if self.has_maps:
                try:
                    # Extract gem locations for mapping
                    gem_locations = []
                    for gem in filtered_gems[:5]:
                        if 'lat' in gem and 'lon' in gem:
                            gem_locations.append(gem)
                    
                    if gem_locations:
                        map_data = self.map_integration_service.create_hidden_gem_map(gem_locations)
                        if map_data:
                            logger.info(f"🗺️ Generated map with {len(gem_locations)} hidden gems")
                except Exception as e:
                    logger.warning(f"Failed to generate map: {e}")
            
            return {
                "success": True,
                "hidden_gems": filtered_gems[:5],
                "response": response,
                "language": language.value if language else "en",
                "map_data": map_data,
                "context_used": {
                    "gem_types": gem_context.gem_types,
                    "authenticity_score": gem_context.authenticity_score,
                    "crowd_level": gem_context.crowd_level,
                    "tourist_comfort": gem_context.tourist_comfort
                }
            }
            
        except Exception as e:
            logger.error(f"Error in hidden gems handler: {e}")
            error_msg = self._get_error_message(language)
            return {
                "success": False,
                "error": str(e),
                "response": error_msg,
                "language": language.value if language else "en",
                "map_data": None
            }
    
    def _get_error_message(self, language: Optional[Language]) -> str:
        """Get error message in appropriate language"""
        if self.has_bilingual and language:
            if language == Language.TURKISH:
                return "Size gizli mekanlar paylaşmak isterim! Ne tür yerler ilginizi çekiyor?"
            else:
                return "I'd love to share some hidden gems with you! Could you tell me what kind of places interest you?"
        return "I'd love to share some hidden gems with you! Could you tell me what kind of places interest you?"
    
    def _build_gem_context(
        self,
        ml_context: Dict[str, Any],
        additional_context: Optional[Dict[str, Any]]
    ) -> HiddenGemContext:
        """Build hidden gem context from ML context"""
        
        query_lower = ml_context.get("original_query", "").lower()
        
        # Extract gem types
        gem_types = []
        type_keywords = {
            "cafe": ["cafe", "coffee", "tea house", "kahve"],
            "rooftop": ["rooftop", "terrace", "view", "panoramic"],
            "park": ["park", "garden", "green space", "nature"],
            "shop": ["shop", "boutique", "market", "store", "bazaar"],
            "historical": ["historical", "ancient", "ottoman", "byzantine", "old"],
            "viewpoint": ["view", "lookout", "panorama", "vista"],
            "restaurant": ["restaurant", "eatery", "dining", "lokanta"],
            "street": ["street", "alley", "walking", "neighborhood"],
            "art": ["art", "gallery", "studio", "creative"],
            "bookshop": ["book", "library", "bookstore"]
        }
        
        for gem_type, keywords in type_keywords.items():
            if any(kw in query_lower for kw in keywords):
                gem_types.append(gem_type)
        
        # Default to general types if none specified
        if not gem_types:
            gem_types = ["cafe", "viewpoint", "historical", "street"]
        
        # Determine authenticity score (how "local" vs "touristy")
        authenticity_keywords = [
            "hidden", "secret", "local", "authentic", "off the beaten",
            "undiscovered", "locals only", "unknown", "tucked away"
        ]
        authenticity_score = 0.5  # Default
        for kw in authenticity_keywords:
            if kw in query_lower:
                authenticity_score = min(authenticity_score + 0.15, 1.0)
        
        # Determine crowd level preference
        crowd_level = "quiet"  # Default for hidden gems
        if any(kw in query_lower for kw in ["very quiet", "empty", "secluded", "peaceful"]):
            crowd_level = "very_quiet"
        elif any(kw in query_lower for kw in ["some people", "moderate", "not too crowded"]):
            crowd_level = "moderate"
        
        # Tourist comfort level (0 = locals only, 1 = tourist-friendly)
        tourist_comfort = 0.5
        if any(kw in query_lower for kw in ["tourist", "english", "easy", "accessible"]):
            tourist_comfort = 0.8
        elif any(kw in query_lower for kw in ["local", "authentic", "turkish only"]):
            tourist_comfort = 0.3
        
        # Time of day
        time_of_day = None
        if any(kw in query_lower for kw in ["morning", "breakfast", "early"]):
            time_of_day = "morning"
        elif any(kw in query_lower for kw in ["afternoon", "lunch"]):
            time_of_day = "afternoon"
        elif any(kw in query_lower for kw in ["evening", "sunset", "dinner"]):
            time_of_day = "evening"
        elif any(kw in query_lower for kw in ["night", "late"]):
            time_of_day = "night"
        
        return HiddenGemContext(
            user_query=ml_context.get("original_query", ""),
            gem_types=gem_types,
            neighborhood_preference=ml_context.get("location_preference"),
            crowd_level=crowd_level,
            authenticity_score=authenticity_score,
            interests=ml_context.get("detected_interests", []),
            tourist_comfort=tourist_comfort,
            accessibility_needs=ml_context.get("accessibility_needs"),
            budget_level=ml_context.get("budget_preference"),
            time_of_day=time_of_day,
            weather_context=ml_context.get("weather"),
            user_sentiment=ml_context.get("sentiment_score", 0.0)
        )
    
    async def _get_candidate_gems(self, context: HiddenGemContext) -> List[Dict[str, Any]]:
        """Get candidate hidden gems from service"""
        
        try:
            # Get gems from service
            all_gems = await self.hidden_gems_service.get_hidden_gems(
                gem_types=context.gem_types,
                neighborhood=context.neighborhood_preference
            )
            
            return all_gems
            
        except Exception as e:
            logger.warning(f"Error fetching gems: {e}, returning mock data")
            return self._get_mock_gems(context)
    
    def _get_mock_gems(self, context: HiddenGemContext) -> List[Dict[str, Any]]:
        """Return mock hidden gems for development with enhanced time-aware data"""
        
        return [
            # Morning gems
            {
                "id": "gem_001",
                "name": "Asmalı Cavit",
                "type": "cafe",
                "description": "Tiny neighborhood cafe in Cihangir, beloved by local writers",
                "neighborhood": "Cihangir, Beyoğlu",
                "authenticity": 0.95,
                "crowd_level": "very_quiet",
                "tourist_ratio": 0.1,
                "best_time": "morning",
                "suitable_times": ["morning", "afternoon"],
                "price_range": "budget",
                "indoor_outdoor": "indoor",
                "highlights": ["Local crowd", "Traditional Turkish coffee", "Books everywhere"],
                "directions": "Hidden alley off Sıraselviler Caddesi",
                "lat": 41.0341,
                "lon": 28.9866
            },
            {
                "id": "gem_006",
                "name": "Balat Morning Market Corner",
                "type": "market",
                "description": "Secret corner of morning market where locals buy fresh simit and tea",
                "neighborhood": "Balat, Fatih",
                "authenticity": 0.97,
                "crowd_level": "moderate",
                "tourist_ratio": 0.05,
                "best_time": "morning",
                "suitable_times": ["morning"],
                "price_range": "budget",
                "indoor_outdoor": "outdoor",
                "highlights": ["Fresh simit", "Morning tea", "Local gossip"],
                "directions": "Behind Balat main square, follow the tea smell",
                "lat": 41.0294,
                "lon": 28.9477
            },
            {
                "id": "gem_005",
                "name": "Yedikule Urban Gardens",
                "type": "park",
                "description": "Community gardens inside ancient Byzantine walls, few tourists",
                "neighborhood": "Yedikule, Fatih",
                "authenticity": 0.90,
                "crowd_level": "moderate",
                "tourist_ratio": 0.1,
                "best_time": "morning",
                "suitable_times": ["morning", "afternoon"],
                "price_range": "free",
                "indoor_outdoor": "outdoor",
                "highlights": ["Byzantine walls", "Local gardeners", "Fresh produce"],
                "directions": "Enter through Yedikule fortress, walk to gardens section",
                "lat": 40.9931,
                "lon": 28.9216
            },
            # Afternoon gems
            {
                "id": "gem_003",
                "name": "Pandeli Rooftop Garden",
                "type": "rooftop",
                "description": "Hidden rooftop garden above Spice Bazaar with tea service",
                "neighborhood": "Eminönü, Fatih",
                "authenticity": 0.85,
                "crowd_level": "quiet",
                "tourist_ratio": 0.2,
                "best_time": "afternoon",
                "suitable_times": ["afternoon", "evening"],
                "price_range": "budget",
                "indoor_outdoor": "outdoor",
                "highlights": ["Bazaar views", "Turkish tea", "Secret entrance"],
                "directions": "Ask shopkeepers in Pandeli Restaurant building",
                "lat": 41.0166,
                "lon": 28.9708
            },
            {
                "id": "gem_004",
                "name": "Kitapevi Çay Bahçesi",
                "type": "bookshop",
                "description": "Bookshop with secret tea garden, Turkish literature focus",
                "neighborhood": "Kadıköy, Asian Side",
                "authenticity": 0.92,
                "crowd_level": "quiet",
                "tourist_ratio": 0.15,
                "best_time": "afternoon",
                "suitable_times": ["afternoon", "evening"],
                "price_range": "budget",
                "indoor_outdoor": "both",
                "highlights": ["Turkish books", "Garden seating", "Local intellectuals"],
                "directions": "Near Kadıköy fish market, ask locals",
                "lat": 40.9903,
                "lon": 29.0257
            },
            {
                "id": "gem_007",
                "name": "Gülhane Secret Rose Garden",
                "type": "garden",
                "description": "Hidden section of Gülhane Park with Ottoman roses, locals only",
                "neighborhood": "Sultanahmet, Fatih",
                "authenticity": 0.88,
                "crowd_level": "quiet",
                "tourist_ratio": 0.1,
                "best_time": "afternoon",
                "suitable_times": ["afternoon"],
                "price_range": "free",
                "indoor_outdoor": "outdoor",
                "highlights": ["Ottoman roses", "Peaceful benches", "No tour groups"],
                "directions": "Enter Gülhane, go left past main path to back corner",
                "lat": 41.0138,
                "lon": 28.9819
            },
            # Evening gems
            {
                "id": "gem_002",
                "name": "Feneraltı Secret Viewpoint",
                "type": "viewpoint",
                "description": "Unmarked spot in Balat with stunning Golden Horn panorama",
                "neighborhood": "Balat, Fatih",
                "authenticity": 0.98,
                "crowd_level": "very_quiet",
                "tourist_ratio": 0.05,
                "best_time": "evening",
                "suitable_times": ["evening", "afternoon"],
                "price_range": "free",
                "indoor_outdoor": "outdoor",
                "highlights": ["Free", "No tourists", "Golden Horn sunset"],
                "directions": "Walk down stairs near Fener Greek Patriarchate",
                "lat": 41.0283,
                "lon": 28.9492
            },
            {
                "id": "gem_008",
                "name": "Şehzadebaşı Sunset Terrace",
                "type": "rooftop",
                "description": "Local family's rooftop terrace serving homemade lemonade at sunset",
                "neighborhood": "Şehzadebaşı, Fatih",
                "authenticity": 0.93,
                "crowd_level": "very_quiet",
                "tourist_ratio": 0.08,
                "best_time": "evening",
                "suitable_times": ["evening"],
                "price_range": "budget",
                "indoor_outdoor": "outdoor",
                "highlights": ["Homemade lemonade", "Family atmosphere", "Mosque silhouettes"],
                "directions": "Ring doorbell at building marked with blue tile, ask for terrace",
                "lat": 41.0158,
                "lon": 28.9578
            },
            {
                "id": "gem_009",
                "name": "Kumkapı Fisherman's Dock",
                "type": "street",
                "description": "Evening gathering spot where fishermen share rakı and stories",
                "neighborhood": "Kumkapı, Fatih",
                "authenticity": 0.94,
                "crowd_level": "moderate",
                "tourist_ratio": 0.12,
                "best_time": "evening",
                "suitable_times": ["evening", "night"],
                "price_range": "budget",
                "indoor_outdoor": "outdoor",
                "highlights": ["Fresh fish", "Local fishermen", "Authentic atmosphere"],
                "directions": "Walk past touristy restaurants to actual fishing dock",
                "lat": 41.0041,
                "lon": 28.9586
            },
            # Night gems
            {
                "id": "gem_010",
                "name": "Galata Tower Secret Viewpoint",
                "type": "viewpoint",
                "description": "Hidden stairs behind Galata Tower with city lights panorama",
                "neighborhood": "Galata, Beyoğlu",
                "authenticity": 0.91,
                "crowd_level": "very_quiet",
                "tourist_ratio": 0.15,
                "best_time": "night",
                "suitable_times": ["night", "evening"],
                "price_range": "free",
                "indoor_outdoor": "outdoor",
                "highlights": ["City lights", "Free", "Romantic spot"],
                "directions": "Behind tower, look for narrow stairs between buildings",
                "lat": 41.0257,
                "lon": 28.9742
            },
            {
                "id": "gem_011",
                "name": "Arnavutköy Midnight Bakery",
                "type": "cafe",
                "description": "24-hour neighborhood bakery with late-night börek and tea",
                "neighborhood": "Arnavutköy, Beşiktaş",
                "authenticity": 0.96,
                "crowd_level": "quiet",
                "tourist_ratio": 0.05,
                "best_time": "night",
                "suitable_times": ["night"],
                "price_range": "budget",
                "indoor_outdoor": "indoor",
                "highlights": ["24-hour", "Fresh börek", "Night owls gathering"],
                "directions": "Bosphorus-side street, look for warm glow and flour smell",
                "lat": 41.0707,
                "lon": 29.0439
            },
            {
                "id": "gem_012",
                "name": "Çengelköy Moonlight Pier",
                "type": "viewpoint",
                "description": "Quiet pier on Asian side, perfect for night reflection with Bosphorus view",
                "neighborhood": "Çengelköy, Asian Side",
                "authenticity": 0.89,
                "crowd_level": "very_quiet",
                "tourist_ratio": 0.1,
                "best_time": "night",
                "suitable_times": ["night", "evening"],
                "price_range": "free",
                "indoor_outdoor": "outdoor",
                "highlights": ["Bosphorus at night", "Very quiet", "Local couples"],
                "directions": "Follow waterfront past main cafes to small fishing pier",
                "lat": 41.0536,
                "lon": 29.0796
            }
        ]
    
    async def _rank_gems_neural(
        self,
        gems: List[Dict[str, Any]],
        context: HiddenGemContext,
        ml_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank gems using neural similarity with heavy authenticity weighting"""
        
        # Get query embedding
        query_embedding = await self.ml_processor.get_embedding(context.user_query)
        
        scored_gems = []
        for gem in gems:
            # Create gem description for embedding
            desc = f"{gem['name']} {gem.get('type', '')} {gem.get('description', '')}"
            
            # Get gem embedding
            gem_embedding = await self.ml_processor.get_embedding(desc)
            
            # Calculate base similarity
            base_score = self.ml_processor.calculate_similarity(
                query_embedding,
                gem_embedding
            )
            
            # Apply context-based adjustments (authenticity heavily weighted)
            adjusted_score = self._adjust_score_with_context(
                base_score,
                gem,
                context
            )
            
            scored_gems.append({
                **gem,
                "ml_score": adjusted_score,
                "base_similarity": base_score
            })
        
        # Sort by ML score descending
        scored_gems.sort(key=lambda x: x["ml_score"], reverse=True)
        
        return scored_gems
    
    def _adjust_score_with_context(
        self,
        base_score: float,
        gem: Dict[str, Any],
        context: HiddenGemContext
    ) -> float:
        """Adjust score with heavy emphasis on authenticity and hidden quality"""
        
        score = base_score
        boost = 0.0
        
        # AUTHENTICITY BOOST (most important for hidden gems!)
        gem_auth = gem.get("authenticity", 0.5)
        if context.authenticity_score > 0.7:
            # User wants very authentic places
            if gem_auth > 0.9:
                boost += 0.40  # Huge boost for very authentic
            elif gem_auth > 0.8:
                boost += 0.25
        
        # TOURIST RATIO PENALTY/BOOST
        tourist_ratio = gem.get("tourist_ratio", 0.5)
        if context.tourist_comfort < 0.5:
            # User wants local-only places
            if tourist_ratio < 0.2:
                boost += 0.30
        else:
            # User okay with some tourists
            if 0.2 <= tourist_ratio <= 0.4:
                boost += 0.10  # Sweet spot
        
        # GEM TYPE MATCH
        gem_type = gem.get("type", "")
        if gem_type in context.gem_types:
            boost += 0.20
        
        # CROWD LEVEL MATCH
        gem_crowd = gem.get("crowd_level", "")
        if gem_crowd == context.crowd_level:
            boost += 0.20
        elif context.crowd_level == "very_quiet" and gem_crowd in ["very_quiet", "quiet"]:
            boost += 0.15
        
        # NEIGHBORHOOD PREFERENCE
        if context.neighborhood_preference:
            gem_neighborhood = gem.get("neighborhood", "").lower()
            if context.neighborhood_preference.lower() in gem_neighborhood:
                boost += 0.15
        
        # TIME OF DAY MATCH
        if context.time_of_day:
            gem_best_time = gem.get("best_time", "")
            if context.time_of_day == gem_best_time:
                boost += 0.15
        
        # BUDGET MATCH
        if context.budget_level:
            gem_price = gem.get("price_range", "")
            if context.budget_level == gem_price or gem_price == "free":
                boost += 0.15
        
        # WEATHER-BASED BOOST
        if context.weather_context:
            weather = context.weather_context.get("condition", "").lower()
            venue = gem.get("indoor_outdoor", "")
            
            if "rain" in weather and venue in ["indoor", "both"]:
                boost += 0.15
            elif "clear" in weather and venue == "outdoor":
                boost += 0.10
        
        # INTEREST ALIGNMENT
        gem_desc = f"{gem.get('name', '')} {gem.get('description', '')}".lower()
        for interest in context.interests:
            if interest.lower() in gem_desc:
                boost += 0.10
        
        # FREE PLACES BONUS (hidden gems are often free!)
        if gem.get("price_range") == "free":
            boost += 0.10
        
        final_score = score * (1 + boost)
        return min(final_score, 1.0)
    
    def _apply_filters(
        self,
        gems: List[Dict[str, Any]],
        context: HiddenGemContext
    ) -> List[Dict[str, Any]]:
        """Apply hard filters including time-aware and weather-aware filtering"""
        
        filtered = gems
        
        # Strict authenticity filter if user wants very authentic
        if context.authenticity_score > 0.8:
            filtered = [g for g in filtered if g.get("authenticity", 0) > 0.7]
        
        # Strict tourist ratio filter for "locals only" requests
        if context.tourist_comfort < 0.3:
            filtered = [g for g in filtered if g.get("tourist_ratio", 1.0) < 0.3]
        
        # WEATHER-AWARE FILTERING (Weather-Aware Gems Feature)
        # Apply weather filtering before time filtering
        if context.weather_context and len(filtered) > 0:
            weather_filtered = self._apply_weather_aware_filter(filtered, context.weather_context)
            # Only use weather-filtered results if we still have gems, otherwise keep all
            if len(weather_filtered) > 0:
                filtered = weather_filtered
        
        # TIME-AWARE FILTERING (Time-Aware Gems Feature)
        # Apply time filtering last and only if we have gems after other filters
        if context.time_of_day and len(filtered) > 0:
            time_filtered = self._apply_time_aware_filter(filtered, context.time_of_day)
            # Only use time-filtered results if we still have gems, otherwise keep all
            if len(time_filtered) > 0:
                filtered = time_filtered
        
        # Accessibility filters
        if context.accessibility_needs:
            # Filter out places that don't meet accessibility needs
            pass  # Implement if accessibility data available
        
        return filtered
    
    def _apply_weather_aware_filter(
        self,
        gems: List[Dict[str, Any]],
        weather_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Apply weather-aware filtering for Weather-Aware Gems feature
        Prioritizes gems that are appropriate for current weather conditions
        
        Args:
            gems: List of gem dictionaries
            weather_context: Weather information dict with 'condition' and 'temperature'
        
        Returns:
            Filtered and re-scored gems based on weather appropriateness
        """
        weather_scored_gems = []
        
        # Extract weather conditions
        condition = weather_context.get("condition", "").lower()
        temperature = weather_context.get("temperature", 20)  # Default 20°C
        
        # Determine weather category
        weather_category = self._categorize_weather(condition, temperature)
        
        for gem in gems:
            venue_type = gem.get("indoor_outdoor", "both")
            gem_type = gem.get("type", "")
            
            # Calculate weather appropriateness score
            weather_score = self._get_weather_appropriateness_score(
                venue_type, gem_type, weather_category, temperature
            )
            
            # Only include gems with reasonable weather score (> 0.4)
            if weather_score > 0.4:
                gem_copy = gem.copy()
                gem_copy["weather_appropriateness_score"] = weather_score
                # Boost ML score based on weather appropriateness
                gem_copy["ml_score"] = gem_copy.get("ml_score", 0.5) * (1 + weather_score * 0.25)
                weather_scored_gems.append(gem_copy)
        
        # Sort by adjusted ML score
        weather_scored_gems.sort(key=lambda x: x["ml_score"], reverse=True)
        
        return weather_scored_gems
    
    def _categorize_weather(self, condition: str, temperature: float) -> str:
        """
        Categorize weather into one of: rainy, sunny, cold, hot
        
        Args:
            condition: Weather condition string (e.g., "rain", "clear", "cloudy")
            temperature: Temperature in Celsius
        
        Returns:
            Weather category string
        """
        # Check for rain/snow
        if any(keyword in condition for keyword in ["rain", "drizzle", "shower", "storm", "snow"]):
            return "rainy"
        
        # Check temperature-based categories
        if temperature < 10:
            return "cold"
        elif temperature > 28:
            return "hot"
        # Check for sunny conditions
        elif any(keyword in condition for keyword in ["clear", "sunny", "fair"]):
            return "sunny"
        else:
            # Default for cloudy/mild conditions
            return "sunny" if temperature > 15 else "cold"
    
    def _get_weather_appropriateness_score(
        self,
        venue_type: str,
        gem_type: str,
        weather_category: str,
        temperature: float
    ) -> float:
        """
        Calculate weather appropriateness score for a gem
        
        Args:
            venue_type: 'indoor', 'outdoor', or 'both'
            gem_type: Type of gem (cafe, park, rooftop, etc.)
            weather_category: One of 'rainy', 'sunny', 'cold', 'hot'
            temperature: Current temperature in Celsius
        
        Returns:
            Score between 0.0 and 1.0
        """
        # Weather-venue compatibility matrix
        weather_venue_scores = {
            "rainy": {
                "indoor": 1.0,
                "both": 0.8,
                "outdoor": 0.3
            },
            "sunny": {
                "outdoor": 1.0,
                "both": 0.9,
                "indoor": 0.6
            },
            "cold": {
                "indoor": 1.0,
                "both": 0.7,
                "outdoor": 0.4
            },
            "hot": {
                "outdoor": 0.9,  # If shaded/waterfront
                "both": 0.8,
                "indoor": 0.9  # Air conditioned indoor spaces
            }
        }
        
        # Weather-type compatibility matrix
        weather_type_scores = {
            "rainy": {
                "cafe": 0.9,
                "restaurant": 0.9,
                "bookshop": 0.95,
                "market": 0.7,  # Covered markets
                "historical": 0.6,  # Museums
                "art": 0.9,
                "rooftop": 0.2,
                "park": 0.2,
                "viewpoint": 0.3,
                "garden": 0.2,
                "street": 0.3
            },
            "sunny": {
                "park": 1.0,
                "garden": 1.0,
                "viewpoint": 0.95,
                "rooftop": 0.95,
                "street": 0.9,
                "cafe": 0.8,
                "market": 0.85,
                "historical": 0.8,
                "restaurant": 0.75
            },
            "cold": {
                "cafe": 1.0,
                "restaurant": 0.95,
                "bookshop": 0.9,
                "historical": 0.8,  # Indoor museums
                "art": 0.85,
                "market": 0.6,  # If covered
                "rooftop": 0.3,
                "park": 0.4,
                "viewpoint": 0.4,
                "street": 0.5
            },
            "hot": {
                "park": 0.8,  # If shaded
                "garden": 0.85,  # Shaded gardens
                "viewpoint": 0.7,  # Waterfront/breezy
                "rooftop": 0.6,  # Evening only
                "cafe": 0.9,  # Air conditioned
                "restaurant": 0.85,
                "bookshop": 0.9,
                "street": 0.5,  # Hot streets
                "market": 0.6,  # If covered
                "historical": 0.85  # Air conditioned museums
            }
        }
        
        # Get base score from venue type
        venue_score = weather_venue_scores.get(weather_category, {}).get(venue_type, 0.7)
        
        # Get type score
        type_score = weather_type_scores.get(weather_category, {}).get(gem_type, 0.7)
        
        # Combine scores (weighted average: 60% type, 40% venue)
        combined_score = (type_score * 0.6) + (venue_score * 0.4)
        
        return combined_score
    
    def _apply_time_aware_filter(
        self,
        gems: List[Dict[str, Any]],
        time_of_day: str
    ) -> List[Dict[str, Any]]:
        """
        Apply time-aware filtering for Time-Aware Gems feature
        Prioritizes gems that are ideal for the specified time of day
        
        Args:
            gems: List of gem dictionaries
            time_of_day: One of 'morning', 'afternoon', 'evening', 'night'
        
        Returns:
            Filtered and re-scored gems based on time appropriateness
        """
        time_scored_gems = []
        
        for gem in gems:
            gem_best_time = gem.get("best_time", "")
            gem_suitable_times = gem.get("suitable_times", [gem_best_time] if gem_best_time else [])
            gem_type = gem.get("type", "")
            
            # Calculate time appropriateness score
            time_score = 0.0
            
            # Perfect match: gem's best time matches query time
            if gem_best_time == time_of_day:
                time_score = 1.0
            # Good match: time is in suitable times
            elif time_of_day in gem_suitable_times:
                time_score = 0.7
            # Type-based matching for generic gems without time data
            else:
                time_score = self._get_type_time_score(gem_type, time_of_day)
            
            # Only include gems with reasonable time score (> 0.2) - more lenient for better results
            if time_score > 0.2:
                gem_copy = gem.copy()
                gem_copy["time_appropriateness_score"] = time_score
                # Boost ML score based on time appropriateness
                gem_copy["ml_score"] = gem_copy.get("ml_score", 0.5) * (1 + time_score * 0.4)
                time_scored_gems.append(gem_copy)
        
        # Sort by adjusted ML score
        time_scored_gems.sort(key=lambda x: x["ml_score"], reverse=True)
        
        return time_scored_gems
    
    def _get_type_time_score(self, gem_type: str, time_of_day: str) -> float:
        """
        Get time appropriateness score based on gem type and time of day
        Used when gem doesn't have specific time data
        
        Returns:
            Float between 0.0 and 1.0
        """
        # Define type-time compatibility matrix
        type_time_matrix = {
            "morning": {
                "cafe": 0.9,
                "park": 0.8,
                "market": 0.9,
                "viewpoint": 0.7,
                "bookshop": 0.6,
                "street": 0.7,
                "garden": 0.8,
                "historical": 0.7,
                "restaurant": 0.5,
                "rooftop": 0.4,
                "bar": 0.1
            },
            "afternoon": {
                "cafe": 0.8,
                "restaurant": 0.9,
                "shop": 0.9,
                "boutique": 0.9,
                "market": 0.8,
                "historical": 0.9,
                "viewpoint": 0.8,
                "park": 0.7,
                "bookshop": 0.8,
                "art": 0.9,
                "rooftop": 0.6,
                "street": 0.8
            },
            "evening": {
                "restaurant": 0.9,
                "rooftop": 1.0,
                "viewpoint": 0.9,
                "cafe": 0.7,
                "bar": 0.8,
                "street": 0.8,
                "art": 0.7,
                "historical": 0.5,
                "park": 0.6
            },
            "night": {
                "rooftop": 1.0,
                "bar": 1.0,
                "restaurant": 0.8,
                "viewpoint": 0.7,
                "street": 0.6,
                "cafe": 0.5,
                "historical": 0.2,
                "park": 0.2,
                "shop": 0.1
            }
        }
        
        return type_time_matrix.get(time_of_day, {}).get(gem_type, 0.5)
    
    async def _generate_response(
        self,
        gems: List[Dict[str, Any]],
        context: HiddenGemContext,
        ml_context: Dict[str, Any],
        language: Optional[Language] = None
    ) -> str:
        """Generate natural language response emphasizing hidden quality with bilingual support"""
        
        if not gems:
            if self.has_bilingual and language:
                if language == Language.TURKISH:
                    return "Bu kriterlere uyan gizli mekanlar bulamadım. Farklı bir semtte aramak ister misiniz?"
                else:
                    return "I couldn't find hidden gems matching those criteria. Would you like me to search in a different neighborhood?"
            return "I couldn't find hidden gems matching those criteria. Would you like me to search in a different neighborhood?"
        
        response_parts = []
        
        # Opening emphasizing "hidden" nature
        opening = self._get_opening_message(context.authenticity_score, language)
        response_parts.append(opening)
        
        # Top gem with detailed description
        top = gems[0]
        
        # Match and Authenticity labels
        if self.has_bilingual and language:
            if language == Language.TURKISH:
                match_label = "Eşleşme"
                auth_label = "Özgünlük"
            else:
                match_label = "Match"
                auth_label = "Authenticity"
        else:
            match_label = "Match"
            auth_label = "Authenticity"
        
        response_parts.append(f"\n\n🌟 **{top['name']}** ({match_label}: {int(top['ml_score']*100)}%, {auth_label}: {int(top.get('authenticity', 0)*100)}%)")
        response_parts.append(f"   📍 {top.get('neighborhood', 'Hidden location')}")
        response_parts.append(f"   {top.get('description', '')}")
        
        # Highlights
        if "highlights" in top and top["highlights"]:
            highlights = ", ".join(top["highlights"][:3])
            if self.has_bilingual and language == Language.TURKISH:
                response_parts.append(f"   ✨ Özel kılan: {highlights}")
            else:
                response_parts.append(f"   ✨ What makes it special: {highlights}")
        
        # TIME-AWARE INFORMATION (Time-Aware Gems Feature)
        if context.time_of_day:
            time_info = self._get_time_aware_info(top, context.time_of_day, language)
            if time_info:
                response_parts.append(time_info)
        
        # WEATHER-AWARE INFORMATION (Weather-Aware Gems Feature)
        if context.weather_context:
            weather_info = self._get_weather_aware_info(top, context.weather_context, language)
            if weather_info:
                response_parts.append(weather_info)
        
        # Crowd info
        crowd_text = self._get_crowd_text(top.get("tourist_ratio", 0), language)
        if crowd_text:
            response_parts.append(crowd_text)
        
        # Directions hint
        if "directions" in top:
            if self.has_bilingual and language == Language.TURKISH:
                response_parts.append(f"   🗺️ Nasıl bulunur: {top['directions']}")
            else:
                response_parts.append(f"   🗺️ How to find: {top['directions']}")
        
        # Additional gems
        if len(gems) > 1:
            if self.has_bilingual and language == Language.TURKISH:
                more_header = "\n\n🗺️ **Daha fazla gizli mekan:**"
            else:
                more_header = "\n\n🗺️ **More hidden spots:**"
            response_parts.append(more_header)
            
            for gem in gems[1:4]:
                auth_emoji = "🔒" if gem.get("authenticity", 0) > 0.9 else "🎯"
                response_parts.append(
                    f"   {auth_emoji} **{gem['name']}** in {gem.get('neighborhood', 'TBA')} - {gem.get('description', '')[:80]}..."
                )
        
        # Context-aware tips
        tips = self._get_context_tips(gems, context, language)
        if tips:
            response_parts.append("\n\n" + "\n".join(tips))
        
        # Final reminder
        final_reminder = self._get_final_reminder(language)
        response_parts.append(final_reminder)
        
        return "\n".join(response_parts)
    
    def _get_opening_message(self, authenticity_score: float, language: Optional[Language]) -> str:
        """Get opening message based on authenticity preference"""
        if authenticity_score > 0.8:
            if self.has_bilingual and language == Language.TURKISH:
                return "🤫 İşte birçok yerli bile bilmeyen gerçekten gizli mekanlar:"
            else:
                return "🤫 Here are some truly hidden spots that even many locals don't know about:"
        else:
            if self.has_bilingual and language == Language.TURKISH:
                return "✨ Sizin için harika gizli mekanlar buldum:"
            else:
                return "✨ I found some wonderful hidden gems for you:"
    
    def _get_crowd_text(self, tourist_ratio: float, language: Optional[Language]) -> Optional[str]:
        """Get crowd description text"""
        if tourist_ratio < 0.2:
            if self.has_bilingual and language == Language.TURKISH:
                return "   🎯 Kalabalık: Neredeyse tamamen yerliler"
            else:
                return "   🎯 Crowd: Almost exclusively locals"
        elif tourist_ratio < 0.4:
            if self.has_bilingual and language == Language.TURKISH:
                return "   🎯 Kalabalık: Çoğunlukla yerliler, az turist"
            else:
                return "   🎯 Crowd: Mostly locals, few tourists"
        return None
    
    def _get_time_aware_info(
        self,
        gem: Dict[str, Any],
        time_of_day: str,
        language: Optional[Language]
    ) -> Optional[str]:
        """
        Get time-aware information for Time-Aware Gems feature
        
        Args:
            gem: The gem dictionary
            time_of_day: Current/requested time of day
            language: Language for the response
        
        Returns:
            Formatted time-aware information string or None
        """
        time_score = gem.get("time_appropriateness_score", 0)
        gem_best_time = gem.get("best_time", "")
        
        # Time emojis
        time_emojis = {
            "morning": "🌅",
            "afternoon": "☀️",
            "evening": "🌆",
            "night": "🌙"
        }
        
        time_emoji = time_emojis.get(time_of_day, "⏰")
        
        # Perfect match
        if gem_best_time == time_of_day and time_score >= 0.9:
            if self.has_bilingual and language == Language.TURKISH:
                time_names = {
                    "morning": "sabah",
                    "afternoon": "öğleden sonra",
                    "evening": "akşam",
                    "night": "gece"
                }
                return f"   {time_emoji} Zaman: {time_names.get(time_of_day, time_of_day)} için mükemmel!"
            else:
                return f"   {time_emoji} Timing: Perfect for {time_of_day}!"
        
        # Good match
        elif time_score >= 0.7:
            if self.has_bilingual and language == Language.TURKISH:
                time_names = {
                    "morning": "sabah",
                    "afternoon": "öğleden sonra",
                    "evening": "akşam",
                    "night": "gece"
                }
                return f"   {time_emoji} Zaman: {time_names.get(time_of_day, time_of_day)} için harika"
            else:
                return f"   {time_emoji} Timing: Great for {time_of_day}"
        
        # Mention best time if different
        elif gem_best_time and gem_best_time != time_of_day:
            if self.has_bilingual and language == Language.TURKISH:
                time_names = {
                    "morning": "sabahları",
                    "afternoon": "öğleden sonra",
                    "evening": "akşamları",
                    "night": "geceleri"
                }
                return f"   {time_emojis.get(gem_best_time, '⏰')} En iyi: {time_names.get(gem_best_time, gem_best_time)}"
            else:
                return f"   {time_emojis.get(gem_best_time, '⏰')} Best time: {gem_best_time}"
        
        return None
    
    def _get_weather_aware_info(
        self,
        gem: Dict[str, Any],
        weather_context: Dict[str, Any],
        language: Optional[Language]
    ) -> Optional[str]:
        """
        Get weather-aware information for Weather-Aware Gems feature
        
        Args:
            gem: The gem dictionary
            weather_context: Weather context with condition and temperature
            language: Language for the response
        
        Returns:
            Formatted weather-aware information string or None
        """
        weather_score = gem.get("weather_appropriateness_score", 0)
        condition = weather_context.get("condition", "").lower()
        temperature = weather_context.get("temperature", 20)
        venue_type = gem.get("indoor_outdoor", "")
        
        # Weather emojis
        weather_emojis = {
            "rainy": "🌧️",
            "sunny": "☀️",
            "cold": "❄️",
            "hot": "🔥"
        }
        
        weather_category = self._categorize_weather(condition, temperature)
        weather_emoji = weather_emojis.get(weather_category, "🌤️")
        
        # Perfect weather match
        if weather_score >= 0.9:
            if self.has_bilingual and language == Language.TURKISH:
                weather_names = {
                    "rainy": "yağmurlu hava",
                    "sunny": "güneşli hava",
                    "cold": "soğuk hava",
                    "hot": "sıcak hava"
                }
                return f"   {weather_emoji} Hava: {weather_names.get(weather_category, 'bu hava')} için mükemmel!"
            else:
                weather_names = {
                    "rainy": "rainy weather",
                    "sunny": "sunny weather",
                    "cold": "cold weather",
                    "hot": "hot weather"
                }
                return f"   {weather_emoji} Weather: Perfect for {weather_names.get(weather_category, 'current conditions')}!"
        
        # Good weather match
        elif weather_score >= 0.75:
            if self.has_bilingual and language == Language.TURKISH:
                venue_names = {
                    "indoor": "kapalı",
                    "outdoor": "açık hava",
                    "both": "her iki"
                }
                return f"   {weather_emoji} {venue_names.get(venue_type, 'mekan')} - bu havada harika"
            else:
                return f"   {weather_emoji} {venue_type.capitalize()} - great for today's weather"
        
        # Mention weather suitability
        elif weather_score >= 0.6:
            if self.has_bilingual and language == Language.TURKISH:
                return f"   {weather_emoji} Bu havada uygun"
            else:
                return f"   {weather_emoji} Suitable for current weather"
        
        return None
    
    def _get_context_tips(
        self,
        gems: List[Dict[str, Any]],
        context: HiddenGemContext,
        language: Optional[Language]
    ) -> List[str]:
        """Get context-aware tips including time-aware and weather-aware suggestions"""
        tips = []
        
        # WEATHER-AWARE TIP (Weather-Aware Gems Feature)
        if context.weather_context:
            weather_tip = self._get_weather_aware_tip(context.weather_context, gems, language)
            if weather_tip:
                tips.append(weather_tip)
        
        # TIME-AWARE TIP (Time-Aware Gems Feature)
        if context.time_of_day:
            time_tip = self._get_time_aware_tip(context.time_of_day, gems, language)
            if time_tip:
                tips.append(time_tip)
        
        # Authenticity tip
        if context.authenticity_score > 0.8:
            if self.has_bilingual and language == Language.TURKISH:
                tips.append("💡 İpucu: Bu yerlerde İngilizce tabelalar olmayabilir - işte bu onları özel kılan!")
            else:
                tips.append("💡 Pro tip: These places might not have English signs - that's what makes them special!")
        
        # Free places tip
        if any(g.get("price_range") == "free" for g in gems[:3]):
            if self.has_bilingual and language == Language.TURKISH:
                tips.append("💰 Bonus: Bu mekanların çoğu ücretsiz!")
            else:
                tips.append("💰 Bonus: Most of these spots are free!")
        
        # Language tip
        if context.tourist_comfort < 0.5:
            if self.has_bilingual and language == Language.TURKISH:
                tips.append("🗣️ Dil ipucu: Birkaç Türkçe kelime öğrenmek bu yerli mekanlarda gerçekten yardımcı olacaktır")
            else:
                tips.append("🗣️ Language tip: Learning a few Turkish phrases will really help at these local spots")
        
        return tips
    
    def _get_weather_aware_tip(
        self,
        weather_context: Dict[str, Any],
        gems: List[Dict[str, Any]],
        language: Optional[Language]
    ) -> Optional[str]:
        """
        Generate weather-aware tips for Weather-Aware Gems feature
        
        Args:
            weather_context: Weather information
            gems: List of recommended gems
            language: Language for the tip
        
        Returns:
            Weather-aware tip string or None
        """
        condition = weather_context.get("condition", "").lower()
        temperature = weather_context.get("temperature", 20)
        weather_category = self._categorize_weather(condition, temperature)
        
        # Weather-specific tips
        weather_tips = {
            "rainy": {
                "en": "🌧️ Weather tip: Perfect day for cozy indoor spots - these gems offer shelter and atmosphere",
                "tr": "🌧️ Hava ipucu: Rahat kapalı mekanlar için mükemmel gün - bu yerler sığınak ve atmosfer sunuyor"
            },
            "sunny": {
                "en": "☀️ Weather tip: Beautiful day to explore outdoor gems - don't forget sunscreen!",
                "tr": "☀️ Hava ipucu: Açık hava mekanlarını keşfetmek için güzel bir gün - güneş kremini unutmayın!"
            },
            "cold": {
                "en": "❄️ Weather tip: Stay warm in cozy indoor spots - perfect for hot tea and local atmosphere",
                "tr": "❄️ Hava ipucu: Rahat kapalı mekanlarda sıcak kalın - sıcak çay ve yerel atmosfer için mükemmel"
            },
            "hot": {
                "en": "🔥 Weather tip: Beat the heat in shaded gardens or air-conditioned cafes - these spots stay cool",
                "tr": "🔥 Hava ipucu: Gölgeli bahçelerde veya klimali kafelerde serinleyin - bu yerler serin kalıyor"
            }
        }
        
        lang_code = "tr" if (self.has_bilingual and language == Language.TURKISH) else "en"
        
        if weather_category in weather_tips:
            return weather_tips[weather_category][lang_code]
        
        return None
    
    def _get_time_aware_tip(
        self,
        time_of_day: str,
        gems: List[Dict[str, Any]],
        language: Optional[Language]
    ) -> Optional[str]:
        """
        Generate time-aware tips for Time-Aware Gems feature
        
        Args:
            time_of_day: Current/requested time of day
            gems: List of recommended gems
            language: Language for the tip
        
        Returns:
            Time-aware tip string or None
        """
        # Time-specific tips
        time_tips = {
            "morning": {
                "en": "☕ Morning tip: These spots are perfect for early risers - arrive before 9 AM for the most authentic local experience",
                "tr": "☕ Sabah ipucu: Bu mekanlar erken kalkanlar için mükemmel - en otantik yerel deneyim için sabah 9'dan önce gelin"
            },
            "afternoon": {
                "en": "🌤️ Afternoon tip: These places are ideal for a leisurely lunch or tea break - locals visit between 2-5 PM",
                "tr": "🌤️ Öğleden sonra ipucu: Bu yerler rahat bir öğle yemeği veya çay molası için ideal - yerliler 14:00-17:00 arası gelir"
            },
            "evening": {
                "en": "🌆 Evening tip: Arrive around sunset (1 hour before dark) for the best views and atmosphere",
                "tr": "🌆 Akşam ipucu: En iyi manzara ve atmosfer için gün batımı civarında (karanlıktan 1 saat önce) gelin"
            },
            "night": {
                "en": "🌙 Night tip: These spots come alive after 9 PM - perfect for experiencing Istanbul's nocturnal side",
                "tr": "🌙 Gece ipucu: Bu mekanlar 21:00'den sonra canlanır - İstanbul'un gece yüzünü deneyimlemek için mükemmel"
            }
        }
        
        lang_code = "tr" if (self.has_bilingual and language == Language.TURKISH) else "en"
        
        if time_of_day in time_tips:
            return time_tips[time_of_day][lang_code]
        
        return None
    
    def _get_final_reminder(self, language: Optional[Language]) -> str:
        """Get final reminder about hidden gems"""
        if self.has_bilingual and language == Language.TURKISH:
            return "\n\n🤫 Unutmayın: Bunlar gizli mekanlar - bulmak zor olabilir, ama işte macera da bu!"
        else:
            return "\n\n🤫 Remember: These are hidden gems - they might be hard to find, but that's part of the adventure!"
    


def create_ml_enhanced_hidden_gems_handler(
    hidden_gems_service,
    ml_context_builder,
    ml_processor,
    response_generator
):
    """Factory function to create ML-enhanced hidden gems handler"""
    return MLEnhancedHiddenGemsHandler(
        hidden_gems_service=hidden_gems_service,
        ml_context_builder=ml_context_builder,
        ml_processor=ml_processor,
        response_generator=response_generator
    )

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
        """Return mock hidden gems for development"""
        
        return [
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
                "price_range": "budget",
                "indoor_outdoor": "indoor",
                "highlights": ["Local crowd", "Traditional Turkish coffee", "Books everywhere"],
                "directions": "Hidden alley off Sıraselviler Caddesi"
            },
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
                "price_range": "free",
                "indoor_outdoor": "outdoor",
                "highlights": ["Free", "No tourists", "Golden Horn sunset"],
                "directions": "Walk down stairs near Fener Greek Patriarchate"
            },
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
                "price_range": "budget",
                "indoor_outdoor": "outdoor",
                "highlights": ["Bazaar views", "Turkish tea", "Secret entrance"],
                "directions": "Ask shopkeepers in Pandeli Restaurant building"
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
                "price_range": "budget",
                "indoor_outdoor": "both",
                "highlights": ["Turkish books", "Garden seating", "Local intellectuals"],
                "directions": "Near Kadıköy fish market, ask locals"
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
                "price_range": "free",
                "indoor_outdoor": "outdoor",
                "highlights": ["Byzantine walls", "Local gardeners", "Fresh produce"],
                "directions": "Enter through Yedikule fortress, walk to gardens section"
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
        """Apply hard filters"""
        
        filtered = gems
        
        # Strict authenticity filter if user wants very authentic
        if context.authenticity_score > 0.8:
            filtered = [g for g in filtered if g.get("authenticity", 0) > 0.7]
        
        # Strict tourist ratio filter for "locals only" requests
        if context.tourist_comfort < 0.3:
            filtered = [g for g in filtered if g.get("tourist_ratio", 1.0) < 0.3]
        
        # Accessibility filters
        if context.accessibility_needs:
            # Filter out places that don't meet accessibility needs
            pass  # Implement if accessibility data available
        
        return filtered
    
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
    
    def _get_context_tips(
        self,
        gems: List[Dict[str, Any]],
        context: HiddenGemContext,
        language: Optional[Language]
    ) -> List[str]:
        """Get context-aware tips"""
        tips = []
        
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
                tips.append("🗣️ Dil ipucu: Birkaç İngilizce kelime öğrenmek bu yerli mekanlarda gerçekten yardımcı olacaktır")
            else:
                tips.append("🗣️ Language tip: Learning a few Turkish phrases will really help at these local spots")
        
        return tips
    
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

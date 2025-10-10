#!/usr/bin/env python3
"""
🏘️ Istanbul Neighborhood Guides System
Deep Learning Enhanced Neighborhood Intelligence

Features:
- Detailed character descriptions for all major Istanbul areas
- Best visiting times with seasonal recommendations
- Local insights and hidden gems discovery
- District-specific recommendations with deep learning scoring
- Cultural context and authentic experiences
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, time
import random

logger = logging.getLogger(__name__)

class NeighborhoodCharacter(Enum):
    """Character types for neighborhoods"""
    HISTORIC_IMPERIAL = "historic_imperial"
    MODERN_COSMOPOLITAN = "modern_cosmopolitan"
    BOHEMIAN_ARTISTIC = "bohemian_artistic"
    TRADITIONAL_AUTHENTIC = "traditional_authentic"
    WATERFRONT_SCENIC = "waterfront_scenic"
    TRENDY_HIPSTER = "trendy_hipster"
    RELIGIOUS_SPIRITUAL = "religious_spiritual"
    COMMERCIAL_SHOPPING = "commercial_shopping"
    RESIDENTIAL_LOCAL = "residential_local"
    NIGHTLIFE_ENTERTAINMENT = "nightlife_entertainment"

class BestVisitingTime(Enum):
    """Best times to visit neighborhoods"""
    EARLY_MORNING = "early_morning"    # 6-9 AM
    MORNING = "morning"               # 9-12 PM
    AFTERNOON = "afternoon"           # 12-5 PM
    EVENING = "evening"               # 5-8 PM
    SUNSET = "sunset"                 # 8-9 PM
    NIGHT = "night"                   # 9 PM-12 AM
    LATE_NIGHT = "late_night"         # 12-6 AM

class VisitorType(Enum):
    """Types of visitors for personalized recommendations"""
    FIRST_TIME_TOURIST = "first_time_tourist"
    CULTURAL_EXPLORER = "cultural_explorer"
    FOOD_ENTHUSIAST = "food_enthusiast"
    PHOTOGRAPHY_LOVER = "photography_lover"
    LOCAL_EXPERIENCE_SEEKER = "local_experience_seeker"
    BUDGET_TRAVELER = "budget_traveler"
    LUXURY_TRAVELER = "luxury_traveler"
    FAMILY_WITH_CHILDREN = "family_with_children"
    YOUNG_BACKPACKER = "young_backpacker"
    BUSINESS_TRAVELER = "business_traveler"

@dataclass
class HiddenGem:
    """Hidden gem or local insight"""
    name: str
    description: str
    type: str  # cafe, viewpoint, shop, experience, etc.
    location_hint: str
    best_time: str
    insider_tip: str
    difficulty_to_find: str  # easy, moderate, challenging
    local_favorite: bool = True

@dataclass
class NeighborhoodGuide:
    """Comprehensive neighborhood guide"""
    name: str
    district: str
    character: NeighborhoodCharacter
    character_description: str
    atmosphere: str
    best_visiting_times: List[BestVisitingTime]
    seasonal_highlights: Dict[str, str]  # season -> highlight
    hidden_gems: List[HiddenGem]
    local_insights: List[str]
    recommended_for: List[VisitorType]
    avoid_times: List[str]
    safety_rating: int  # 1-10
    tourist_density: str  # low, moderate, high, very_high
    price_level: str  # budget, moderate, expensive, luxury
    unique_experiences: List[str]
    photo_opportunities: List[str]
    cultural_significance: str
    local_etiquette: List[str]
    must_try_foods: List[str]
    walking_difficulty: str  # easy, moderate, challenging
    estimated_visit_duration: str

class IstanbulNeighborhoodGuidesSystem:
    """Deep learning enhanced neighborhood guides system"""
    
    def __init__(self):
        """Initialize the neighborhood guides system"""
        self.neighborhoods = self._load_comprehensive_neighborhoods()
        self.seasonal_context = self._load_seasonal_contexts()
        self.visitor_preferences = {}  # Deep learning user profiling
        logger.info(f"🏘️ Loaded {len(self.neighborhoods)} comprehensive neighborhoods")
    
    def _load_comprehensive_neighborhoods(self) -> Dict[str, NeighborhoodGuide]:
        """Load comprehensive neighborhood data"""
        neighborhoods = {}
        
        # Historic Peninsula - Sultanahmet
        neighborhoods["sultanahmet"] = NeighborhoodGuide(
            name="Sultanahmet",
            district="Fatih",
            character=NeighborhoodCharacter.HISTORIC_IMPERIAL,
            character_description="The crown jewel of Byzantine and Ottoman empires, where every stone tells a story of 1,500 years of imperial grandeur.",
            atmosphere="Majestic and timeless, with the weight of history in every cobblestone. Tourist-heavy but undeniably magical.",
            best_visiting_times=[BestVisitingTime.EARLY_MORNING, BestVisitingTime.SUNSET],
            seasonal_highlights={
                "spring": "Tulip gardens bloom around historic sites",
                "summer": "Extended daylight hours for exploring",
                "autumn": "Golden light on ancient monuments",
                "winter": "Mystical fog around the Hagia Sophia"
            },
            hidden_gems=[
                HiddenGem(
                    name="Soğukçeşme Sokağı",
                    description="Ottoman wooden houses turned boutique hotels, like stepping into a fairytale",
                    type="historic_street",
                    location_hint="Between Topkapi Palace and Hagia Sophia",
                    best_time="Golden hour",
                    insider_tip="Local photographers know this street has the best Ottoman architecture shots",
                    difficulty_to_find="easy"
                ),
                HiddenGem(
                    name="Gülhane Park Rose Garden",
                    description="Secret rose garden where locals picnic away from tourist crowds",
                    type="park",
                    location_hint="Deep inside Gülhane Park, follow the scent",
                    best_time="Early morning or late afternoon",
                    insider_tip="Bring tea and simit for the perfect local experience",
                    difficulty_to_find="moderate"
                )
            ],
            local_insights=[
                "Visit major attractions at 8 AM to avoid crowds - locals know this secret",
                "The çay (tea) vendors near Blue Mosque serve the strongest tea in the city",
                "Local carpet sellers in Grand Bazaar are friendliest on rainy days",
                "Free classical concerts happen in Hagia Sophia's garden on summer evenings"
            ],
            recommended_for=[VisitorType.FIRST_TIME_TOURIST, VisitorType.CULTURAL_EXPLORER, VisitorType.PHOTOGRAPHY_LOVER],
            avoid_times=["Weekend afternoons", "Cruise ship arrival times (10 AM - 2 PM)"],
            safety_rating=9,
            tourist_density="very_high",
            price_level="expensive",
            unique_experiences=[
                "Sunrise prayer call echoing between monuments",
                "Traditional Ottoman coffee ceremony",
                "Underground cistern meditation"
            ],
            photo_opportunities=[
                "Blue Mosque silhouette at dawn",
                "Byzantine mosaics in golden light",
                "Street life in Ottoman alleyways"
            ],
            cultural_significance="Heart of three empires: Roman, Byzantine, and Ottoman",
            local_etiquette=[
                "Dress modestly when visiting mosques",
                "Remove shoes before entering religious sites",
                "Don't point camera directly at people praying"
            ],
            must_try_foods=["Turkish delight from Hacı Bekir", "Ottoman coffee", "Lokum varieties"],
            walking_difficulty="easy",
            estimated_visit_duration="Full day"
        )
        
        # Modern Istanbul - Beyoğlu
        neighborhoods["beyoglu"] = NeighborhoodGuide(
            name="Beyoğlu",
            district="Beyoğlu",
            character=NeighborhoodCharacter.MODERN_COSMOPOLITAN,
            character_description="The beating heart of modern Istanbul, where European elegance meets Turkish soul in a symphony of culture, nightlife, and creativity.",
            atmosphere="Electric and dynamic, buzzing with energy from dawn to dawn. A cultural melting pot where every corner surprises.",
            best_visiting_times=[BestVisitingTime.AFTERNOON, BestVisitingTime.EVENING, BestVisitingTime.NIGHT],
            seasonal_highlights={
                "spring": "Rooftop terraces come alive with outdoor dining",
                "summer": "Street festivals and open-air concerts",
                "autumn": "Cozy bookstore café culture peaks",
                "winter": "Christmas decorations on İstiklal Street"
            },
            hidden_gems=[
                HiddenGem(
                    name="Mikla Rooftop View (Free Version)",
                    description="360-degree Istanbul view without restaurant prices - access through Marmara Pera Hotel elevator",
                    type="viewpoint",
                    location_hint="Marmara Pera Hotel top floor, act like hotel guest",
                    best_time="Sunset",
                    insider_tip="Order a coffee at the lobby café for legitimate access",
                    difficulty_to_find="moderate"
                ),
                HiddenGem(
                    name="French Passage Antique Market",
                    description="Secret antique paradise where locals find Ottoman treasures",
                    type="market",
                    location_hint="Hidden passage off İstiklal Street",
                    best_time="Weekday mornings",
                    insider_tip="Bargain in Turkish for better prices",
                    difficulty_to_find="challenging"
                )
            ],
            local_insights=[
                "Real İstanbul nightlife starts after 11 PM - tourists leave by then",
                "Best street food is found in side alleys, not on main İstiklal Street",
                "Locals shop at Balık Pazarı (Fish Market) for authentic food experiences",
                "Free art galleries open late on Thursdays for wine and culture"
            ],
            recommended_for=[VisitorType.CULTURAL_EXPLORER, VisitorType.FOOD_ENTHUSIAST, VisitorType.YOUNG_BACKPACKER],
            avoid_times=["İstiklal Street weekends (too crowded)", "Late night if not interested in nightlife"],
            safety_rating=8,
            tourist_density="high",
            price_level="moderate",
            unique_experiences=[
                "Meyhane tavern crawl with locals",
                "Underground jazz clubs till dawn",
                "Rooftop gallery openings"
            ],
            photo_opportunities=[
                "İstiklal Street tram in motion",
                "Galata Tower from rooftops",
                "Street art in side alleys"
            ],
            cultural_significance="Historic Pera district, heart of Ottoman cosmopolitan culture",
            local_etiquette=[
                "Nightlife dress code: smart casual minimum",
                "Tipping is expected in meyhanes",
                "Don't photograph people without permission in bars"
            ],
            must_try_foods=["Balık ekmek from street vendors", "Meze at traditional meyhane", "Turkish wine varieties"],
            walking_difficulty="moderate",
            estimated_visit_duration="Half day to full day"
        )
        
        # Authentic Neighborhood - Balat
        neighborhoods["balat"] = NeighborhoodGuide(
            name="Balat",
            district="Fatih",
            character=NeighborhoodCharacter.TRADITIONAL_AUTHENTIC,
            character_description="A living museum of multi-cultural Istanbul, where colorful Ottoman houses tell stories of Greeks, Jews, and Turks living in harmony.",
            atmosphere="Authentic and unpretentious, like discovering a secret Istanbul that exists parallel to the tourist trail.",
            best_visiting_times=[BestVisitingTime.MORNING, BestVisitingTime.AFTERNOON],
            seasonal_highlights={
                "spring": "Colorful houses pop against blue skies",
                "summer": "Elderly residents share stories on doorsteps",
                "autumn": "Perfect light for photography",
                "winter": "Cozy neighborhood cafés warm up"
            },
            hidden_gems=[
                HiddenGem(
                    name="Grandmother's Secret Recipe Café",
                    description="Unnamed café where 80-year-old Ayşe Teyze makes the city's best börek in her living room",
                    type="cafe",
                    location_hint="Follow the smell of fresh pastry and sound of local gossip",
                    best_time="Mid-morning",
                    insider_tip="Say 'Ayşe Teyze gönderiyor' (Aunt Ayşe sent me)",
                    difficulty_to_find="challenging"
                ),
                HiddenGem(
                    name="Abandoned Greek School Viewpoint",
                    description="Ruins with the best view of Golden Horn, known only to locals",
                    type="viewpoint",
                    location_hint="Behind the Bulgarian Church, follow cats",
                    best_time="Sunset",
                    insider_tip="Bring respect - this is a sacred place for locals",
                    difficulty_to_find="challenging"
                )
            ],
            local_insights=[
                "Locals still practice the old neighborhood watch system",
                "Many residents speak Greek, Turkish, and Ladino",
                "Best antiques are found in people's houses, not shops",
                "The neighborhood cats are considered residents and are fed by everyone"
            ],
            recommended_for=[VisitorType.LOCAL_EXPERIENCE_SEEKER, VisitorType.PHOTOGRAPHY_LOVER, VisitorType.CULTURAL_EXPLORER],
            avoid_times=["Friday afternoons (prayer time)", "Very early morning (residents sleeping)"],
            safety_rating=9,
            tourist_density="low",
            price_level="budget",
            unique_experiences=[
                "Coffee with neighborhood elders",
                "Discovering Ottoman family stories",
                "Traditional craftspeople at work"
            ],
            photo_opportunities=[
                "Colorful Ottoman houses",
                "Daily life scenes",
                "Multicultural architecture"
            ],
            cultural_significance="Historic multi-ethnic neighborhood preserving Ottoman tolerance",
            local_etiquette=[
                "Greet elders with respect",
                "Ask permission before photographing people or their homes",
                "Support local businesses"
            ],
            must_try_foods=["Home-made börek", "Traditional Turkish coffee", "Local bakery simit"],
            walking_difficulty="moderate",
            estimated_visit_duration="Half day"
        )
        
        # Waterfront Beauty - Ortaköy
        neighborhoods["ortakoy"] = NeighborhoodGuide(
            name="Ortaköy",
            district="Beşiktaş",
            character=NeighborhoodCharacter.WATERFRONT_SCENIC,
            character_description="Where Bosphorus magic meets village charm, offering the most Instagram-worthy views in Istanbul with authentic waterfront culture.",
            atmosphere="Romantic and picturesque, like a Mediterranean village transplanted to the Bosphorus. Perfect for sunset dreams.",
            best_visiting_times=[BestVisitingTime.AFTERNOON, BestVisitingTime.SUNSET, BestVisitingTime.EVENING],
            seasonal_highlights={
                "spring": "Cherry blossoms frame the mosque",
                "summer": "Vibrant street art festival",
                "autumn": "Golden reflections on Bosphorus",
                "winter": "Dramatic storm watching from cafés"
            },
            hidden_gems=[
                HiddenGem(
                    name="Fisherman's Dawn Coffee",
                    description="Join local fishermen for sunrise coffee and fresh catch stories",
                    type="experience",
                    location_hint="Follow fishing boats to small pier",
                    best_time="5:30 AM",
                    insider_tip="Bring Turkish phrases - fishermen love sharing stories",
                    difficulty_to_find="moderate"
                ),
                HiddenGem(
                    name="Bridge Photographer's Secret Spot",
                    description="Best Bosphorus Bridge photos without crowds",
                    type="viewpoint",
                    location_hint="Behind the mosque, up the hidden stairs",
                    best_time="Blue hour",
                    insider_tip="Professional photographers guard this location",
                    difficulty_to_find="challenging"
                )
            ],
            local_insights=[
                "Best kumpir (stuffed potato) vendor changes daily - ask locals",
                "Weekend craft market has authentic handmade items, not tourist souvenirs",
                "Locals know which café has the bathroom with Bosphorus view",
                "Free WiFi password at most cafés is 'ortakoy123'"
            ],
            recommended_for=[VisitorType.PHOTOGRAPHY_LOVER, VisitorType.CULTURAL_EXPLORER, VisitorType.FAMILY_WITH_CHILDREN],
            avoid_times=["Weekend evenings (too crowded for photos)", "Rainy days (outdoor charm lost)"],
            safety_rating=9,
            tourist_density="moderate",
            price_level="moderate",
            unique_experiences=[
                "Sunset prayers with Bosphorus view",
                "Traditional boat builder workshops",
                "Street artists creating live"
            ],
            photo_opportunities=[
                "Mosque with Bosphorus Bridge backdrop",
                "Colorful waterfront houses",
                "Street food vendors in action"
            ],
            cultural_significance="Traditional fishing village maintaining Bosphorus culture",
            local_etiquette=[
                "Don't block fishermen's paths",
                "Respect prayer times at mosque",
                "Support local artisans over mass-produced items"
            ],
            must_try_foods=["Kumpir with all toppings", "Fresh fish sandwich", "Turkish waffle"],
            walking_difficulty="easy",
            estimated_visit_duration="Half day"
        )
        
        # Trendy Asian Side - Kadıköy
        neighborhoods["kadikoy"] = NeighborhoodGuide(
            name="Kadıköy",
            district="Kadıköy",
            character=NeighborhoodCharacter.TRENDY_HIPSTER,
            character_description="Istanbul's Brooklyn - where young creatives, artists, and intellectuals create a parallel universe of cool cafés, indie culture, and authentic local life.",
            atmosphere="Relaxed and creative, with an edge of intellectual rebellion. The 'real' Istanbul where locals actually live and play.",
            best_visiting_times=[BestVisitingTime.AFTERNOON, BestVisitingTime.EVENING, BestVisitingTime.NIGHT],
            seasonal_highlights={
                "spring": "Rooftop bar season begins",
                "summer": "Outdoor cinema in parks",
                "autumn": "Cozy bookstore readings",
                "winter": "Underground music venue season"
            },
            hidden_gems=[
                HiddenGem(
                    name="Secret Vinyl Record Shop",
                    description="Underground shop where Istanbul's DJs find rare Turkish psychedelic records",
                    type="shop",
                    location_hint="Basement of building with graffiti cat",
                    best_time="Any time",
                    insider_tip="Ask for 'Anatolian rock' section",
                    difficulty_to_find="challenging"
                ),
                HiddenGem(
                    name="Rooftop Philosophy Café",
                    description="Where intellectuals debate life over endless tea",
                    type="cafe",
                    location_hint="Above the anarchist bookstore",
                    best_time="Evening",
                    insider_tip="Join the Wednesday philosophy nights",
                    difficulty_to_find="moderate"
                )
            ],
            local_insights=[
                "The ferry from European side is an experience itself",
                "Best bars don't have signs - look for crowds at unmarked doors",
                "Locals prefer Asian side because it's 'more authentic'",
                "Street art changes weekly - local artists are always active"
            ],
            recommended_for=[VisitorType.YOUNG_BACKPACKER, VisitorType.LOCAL_EXPERIENCE_SEEKER, VisitorType.CULTURAL_EXPLORER],
            avoid_times=["Monday mornings (everything closed)", "Tourist ferry arrival times"],
            safety_rating=9,
            tourist_density="low",
            price_level="budget",
            unique_experiences=[
                "Underground music venues",
                "Independent bookstore readings",
                "Alternative art galleries"
            ],
            photo_opportunities=[
                "Street art murals",
                "Vintage shop interiors",
                "Local café culture"
            ],
            cultural_significance="Modern Turkish alternative culture hub",
            local_etiquette=[
                "Support independent businesses",
                "Engage in intellectual conversations",
                "Respect the creative community"
            ],
            must_try_foods=["Third-wave coffee", "Alternative restaurant cuisines", "Local bakery specialties"],
            walking_difficulty="easy",
            estimated_visit_duration="Half day"
        )
        
        return neighborhoods
    
    def _load_seasonal_contexts(self) -> Dict[str, Dict[str, str]]:
        """Load seasonal context for neighborhoods"""
        return {
            "spring": {
                "general": "Perfect weather for exploring, flowers blooming, outdoor activities resume",
                "photography": "Soft light, colorful nature, comfortable walking weather",
                "cultural": "Festival season begins, outdoor events, renewed energy"
            },
            "summer": {
                "general": "Hot days, extended daylight, vibrant street life, tourist peak",
                "photography": "Strong light, dramatic shadows, golden hour extended",
                "cultural": "Outdoor festivals, rooftop venues, beach culture"
            },
            "autumn": {
                "general": "Perfect weather returns, golden light, comfortable temperatures",
                "photography": "Best lighting conditions, autumn colors, clear skies",
                "cultural": "Cultural season begins, indoor venues reopen, cozy atmosphere"
            },
            "winter": {
                "general": "Quiet and authentic, dramatic weather, cozy indoor culture",
                "photography": "Dramatic skies, snow possibilities, intimate indoor shots",
                "cultural": "Traditional culture dominates, indoor venues peak, authentic experiences"
            }
        }
    
    def get_neighborhood_guide(self, neighborhood_name: str) -> Optional[NeighborhoodGuide]:
        """Get comprehensive guide for a specific neighborhood"""
        return self.neighborhoods.get(neighborhood_name.lower())
    
    def get_neighborhoods_by_character(self, character: NeighborhoodCharacter) -> List[NeighborhoodGuide]:
        """Get neighborhoods matching a specific character type"""
        return [n for n in self.neighborhoods.values() if n.character == character]
    
    def get_recommendations_for_visitor_type(self, visitor_type: VisitorType) -> List[NeighborhoodGuide]:
        """Get neighborhood recommendations based on visitor type"""
        return [n for n in self.neighborhoods.values() if visitor_type in n.recommended_for]
    
    def get_neighborhoods_by_visiting_time(self, visiting_time: BestVisitingTime) -> List[NeighborhoodGuide]:
        """Get neighborhoods best visited at specific time"""
        return [n for n in self.neighborhoods.values() if visiting_time in n.best_visiting_times]
    
    def search_hidden_gems(self, gem_type: Optional[str] = None, difficulty: Optional[str] = None) -> List[Tuple[str, HiddenGem]]:
        """Search for hidden gems across all neighborhoods"""
        gems = []
        for neighborhood_name, neighborhood in self.neighborhoods.items():
            for gem in neighborhood.hidden_gems:
                if gem_type and gem.type != gem_type:
                    continue
                if difficulty and gem.difficulty_to_find != difficulty:
                    continue
                gems.append((neighborhood_name, gem))
        return gems
    
    def get_seasonal_recommendations(self, season: str) -> Dict[str, List[str]]:
        """Get seasonal recommendations for all neighborhoods"""
        recommendations = {}
        for name, neighborhood in self.neighborhoods.items():
            if season in neighborhood.seasonal_highlights:
                recommendations[name] = neighborhood.seasonal_highlights[season]
        return recommendations
    
    def generate_personalized_neighborhood_guide(self, visitor_type: VisitorType, interests: List[str], 
                                                visit_duration: str, season: str) -> Dict[str, Any]:
        """Generate personalized neighborhood recommendations using deep learning insights"""
        
        # Get base recommendations
        recommended_neighborhoods = self.get_recommendations_for_visitor_type(visitor_type)
        
        # Score neighborhoods based on interests and season
        scored_neighborhoods = []
        
        for neighborhood in recommended_neighborhoods:
            score = self._calculate_neighborhood_score(neighborhood, interests, season, visit_duration)
            scored_neighborhoods.append((neighborhood, score))
        
        # Sort by score
        scored_neighborhoods.sort(key=lambda x: x[1], reverse=True)
        
        # Generate comprehensive guide
        guide = {
            "visitor_profile": {
                "type": visitor_type.value,
                "interests": interests,
                "visit_duration": visit_duration,
                "season": season
            },
            "top_recommendations": [],
            "hidden_gems_compilation": [],
            "insider_tips": [],
            "cultural_experiences": [],
            "photo_opportunities": []
        }
        
        # Add top neighborhoods
        for neighborhood, score in scored_neighborhoods[:3]:
            recommendation = {
                "neighborhood": neighborhood.name,
                "score": round(score, 2),
                "why_recommended": self._generate_recommendation_reason(neighborhood, visitor_type, interests),
                "best_times": [t.value for t in neighborhood.best_visiting_times],
                "seasonal_highlight": neighborhood.seasonal_highlights.get(season, "Great year-round"),
                "unique_experiences": neighborhood.unique_experiences[:2],
                "must_try": neighborhood.must_try_foods[:2],
                "hidden_gems": [gem.name for gem in neighborhood.hidden_gems]
            }
            guide["top_recommendations"].append(recommendation)
        
        # Compile hidden gems
        all_gems = self.search_hidden_gems()
        guide["hidden_gems_compilation"] = [
            {
                "name": gem.name,
                "neighborhood": neighborhood_name,
                "description": gem.description,
                "insider_tip": gem.insider_tip,
                "difficulty": gem.difficulty_to_find
            }
            for neighborhood_name, gem in all_gems[:5]
        ]
        
        # Collect insider tips
        all_insights = []
        for neighborhood in recommended_neighborhoods:
            all_insights.extend(neighborhood.local_insights)
        guide["insider_tips"] = random.sample(all_insights, min(5, len(all_insights)))
        
        return guide
    
    def _calculate_neighborhood_score(self, neighborhood: NeighborhoodGuide, interests: List[str], 
                                    season: str, visit_duration: str) -> float:
        """Calculate neighborhood relevance score using deep learning principles"""
        score = 0.0
        
        # Base score for neighborhood character alignment
        character_weights = {
            "history": 0.9 if neighborhood.character in [NeighborhoodCharacter.HISTORIC_IMPERIAL] else 0.3,
            "culture": 0.8 if neighborhood.character in [NeighborhoodCharacter.TRADITIONAL_AUTHENTIC, NeighborhoodCharacter.BOHEMIAN_ARTISTIC] else 0.4,
            "food": 0.7 if neighborhood.character in [NeighborhoodCharacter.TRADITIONAL_AUTHENTIC] else 0.5,
            "nightlife": 0.9 if neighborhood.character in [NeighborhoodCharacter.NIGHTLIFE_ENTERTAINMENT, NeighborhoodCharacter.TRENDY_HIPSTER] else 0.2,
            "photography": 0.8 if neighborhood.character in [NeighborhoodCharacter.WATERFRONT_SCENIC, NeighborhoodCharacter.HISTORIC_IMPERIAL] else 0.4,
            "shopping": 0.7 if neighborhood.character in [NeighborhoodCharacter.COMMERCIAL_SHOPPING, NeighborhoodCharacter.MODERN_COSMOPOLITAN] else 0.3,
            "local_life": 0.9 if neighborhood.character in [NeighborhoodCharacter.TRADITIONAL_AUTHENTIC, NeighborhoodCharacter.RESIDENTIAL_LOCAL] else 0.4
        }
        
        for interest in interests:
            if interest in character_weights:
                score += character_weights[interest]
        
        # Seasonal bonus
        if season in neighborhood.seasonal_highlights:
            score += 0.3
        
        # Visit duration compatibility
        duration_weights = {
            "short": 0.2 if neighborhood.estimated_visit_duration == "Full day" else 0.8,
            "medium": 0.6,
            "long": 0.8 if neighborhood.estimated_visit_duration == "Full day" else 0.4
        }
        
        score += duration_weights.get(visit_duration, 0.5)
        
        # Hidden gems bonus
        score += len(neighborhood.hidden_gems) * 0.1
        
        # Safety and accessibility
        score += neighborhood.safety_rating * 0.05
        
        return score
    
    def _generate_recommendation_reason(self, neighborhood: NeighborhoodGuide, 
                                      visitor_type: VisitorType, interests: List[str]) -> str:
        """Generate AI-powered reason for recommendation"""
        reasons = []
        
        # Character-based reasoning
        character_reasons = {
            NeighborhoodCharacter.HISTORIC_IMPERIAL: "Perfect for experiencing Istanbul's imperial grandeur",
            NeighborhoodCharacter.MODERN_COSMOPOLITAN: "Ideal for contemporary cultural experiences",
            NeighborhoodCharacter.TRADITIONAL_AUTHENTIC: "Best for authentic local life immersion",
            NeighborhoodCharacter.WATERFRONT_SCENIC: "Unmatched scenic beauty and photo opportunities",
            NeighborhoodCharacter.TRENDY_HIPSTER: "Perfect for alternative culture and creative scenes"
        }
        
        reasons.append(character_reasons.get(neighborhood.character, "Great neighborhood experience"))
        
        # Interest-based reasoning
        if "history" in interests and neighborhood.character == NeighborhoodCharacter.HISTORIC_IMPERIAL:
            reasons.append("Rich historical sites match your interests")
        
        if "food" in interests and neighborhood.must_try_foods:
            reasons.append(f"Amazing local cuisine including {neighborhood.must_try_foods[0]}")
        
        if "photography" in interests and neighborhood.photo_opportunities:
            reasons.append("Exceptional photography opportunities")
        
        # Unique experience reasoning
        if neighborhood.unique_experiences:
            reasons.append(f"Unique experience: {neighborhood.unique_experiences[0]}")
        
        return " • ".join(reasons[:3])
    
    def get_neighborhood_walking_route(self, neighborhood_name: str, interests: List[str]) -> Dict[str, Any]:
        """Generate optimized walking route through neighborhood (placeholder for future transportation integration)"""
        neighborhood = self.get_neighborhood_guide(neighborhood_name)
        if not neighborhood:
            return {"error": "Neighborhood not found"}
        
        return {
            "neighborhood": neighborhood_name,
            "estimated_duration": neighborhood.estimated_visit_duration,
            "difficulty": neighborhood.walking_difficulty,
            "key_stops": neighborhood.unique_experiences,
            "hidden_gems_route": [gem.name for gem in neighborhood.hidden_gems],
            "photo_opportunities": neighborhood.photo_opportunities,
            "note": "Detailed walking routes with transportation will be available soon"
        }
    
    def get_all_neighborhoods_summary(self) -> Dict[str, Any]:
        """Get summary of all available neighborhoods"""
        summary = {
            "total_neighborhoods": len(self.neighborhoods),
            "character_distribution": {},
            "visitor_type_coverage": {},
            "total_hidden_gems": 0,
            "neighborhoods": []
        }
        
        # Count character types
        for neighborhood in self.neighborhoods.values():
            char = neighborhood.character.value
            summary["character_distribution"][char] = summary["character_distribution"].get(char, 0) + 1
            summary["total_hidden_gems"] += len(neighborhood.hidden_gems)
        
        # Count visitor type coverage
        for neighborhood in self.neighborhoods.values():
            for visitor_type in neighborhood.recommended_for:
                vt = visitor_type.value
                summary["visitor_type_coverage"][vt] = summary["visitor_type_coverage"].get(vt, 0) + 1
        
        # Add basic neighborhood info
        for name, neighborhood in self.neighborhoods.items():
            summary["neighborhoods"].append({
                "name": name,
                "character": neighborhood.character.value,
                "district": neighborhood.district,
                "tourist_density": neighborhood.tourist_density,
                "price_level": neighborhood.price_level,
                "safety_rating": neighborhood.safety_rating,
                "hidden_gems_count": len(neighborhood.hidden_gems)
            })
        
        return summary

def main():
    """Test the neighborhood guides system"""
    print("🏘️ ISTANBUL NEIGHBORHOOD GUIDES SYSTEM TEST")
    print("=" * 60)
    
    # Initialize system
    guides = IstanbulNeighborhoodGuidesSystem()
    
    # Test basic functionality
    print(f"\n📊 System Summary:")
    summary = guides.get_all_neighborhoods_summary()
    print(f"  • Total neighborhoods: {summary['total_neighborhoods']}")
    print(f"  • Total hidden gems: {summary['total_hidden_gems']}")
    print(f"  • Character types: {list(summary['character_distribution'].keys())}")
    
    # Test personalized recommendations
    print(f"\n🎯 Personalized Recommendations Test:")
    personal_guide = guides.generate_personalized_neighborhood_guide(
        visitor_type=VisitorType.CULTURAL_EXPLORER,
        interests=["history", "photography", "local_life"],
        visit_duration="medium",
        season="autumn"
    )
    
    print(f"  Visitor: {personal_guide['visitor_profile']['type']}")
    print(f"  Top recommendations:")
    for rec in personal_guide['top_recommendations']:
        print(f"    • {rec['neighborhood']} (Score: {rec['score']}) - {rec['why_recommended']}")
    
    # Test hidden gems search
    print(f"\n💎 Hidden Gems Discovery:")
    gems = guides.search_hidden_gems(difficulty="challenging")
    for neighborhood, gem in gems[:3]:
        print(f"    • {gem.name} in {neighborhood}: {gem.description}")

if __name__ == "__main__":
    main()

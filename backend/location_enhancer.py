#!/usr/bin/env python3
"""
Location-Specific Response Enhancement System
===========================================

This module provides comprehensive location-specific response enhancement 
to dramatically improve query relevance and location focus for Istanbul tourism.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

@dataclass 
class LocationData:
    """Comprehensive location information for Istanbul neighborhoods"""
    name: str
    description: str
    character: str
    main_attractions: List[str]
    metro_stations: List[str]
    tram_stops: List[str]
    ferry_terminals: List[str]
    walking_distances: Dict[str, str]
    local_landmarks: List[str]
    dining_specialties: List[str]
    shopping_highlights: List[str]
    cultural_notes: List[str]
    practical_tips: List[str]
    best_times: List[str]
    transportation_hub: bool
    tourist_density: str  # "high", "medium", "low"
    local_authenticity: str  # "high", "medium", "low"
    budget_level: str  # "budget", "moderate", "upscale", "mixed"

class LocationEnhancer:
    """Comprehensive location-specific response enhancement system"""
    
    def __init__(self):
        self.location_database = self._build_location_database()
        self.response_enhancers = self._build_response_enhancers()
        
    def _build_location_database(self) -> Dict[str, LocationData]:
        """Build comprehensive database of Istanbul locations"""
        return {
            "sultanahmet": LocationData(
                name="Sultanahmet",
                description="Historic heart of Istanbul and UNESCO World Heritage Site",
                character="Byzantine and Ottoman imperial district with world-famous monuments",
                main_attractions=[
                    "Hagia Sophia", "Blue Mosque", "Topkapi Palace", "Basilica Cistern",
                    "Hippodrome", "Turkish and Islamic Arts Museum", "Sultanahmet Archaeological Museum"
                ],
                metro_stations=["Vezneciler"],
                tram_stops=["Sultanahmet", "Gülhane", "Eminönü"],
                ferry_terminals=["Eminönü"],
                walking_distances={
                    "Hagia Sophia to Blue Mosque": "2-minute walk (150 meters)",
                    "Blue Mosque to Topkapi Palace": "5-minute walk (400 meters)", 
                    "Sultanahmet to Grand Bazaar": "8-minute walk (650 meters)",
                    "Sultanahmet to Galata Bridge": "10-minute walk (800 meters)",
                    "Basilica Cistern to Hagia Sophia": "3-minute walk (200 meters)"
                },
                local_landmarks=[
                    "Sultanahmet Square", "German Fountain", "Obelisk of Theodosius",
                    "Serpentine Column", "Million Stone", "Firuz Ağa Mosque"
                ],
                dining_specialties=[
                    "Traditional Ottoman cuisine", "Tourist-friendly restaurants", 
                    "Rooftop terraces with historical views", "Traditional Turkish breakfast"
                ],
                shopping_highlights=[
                    "Arasta Bazaar", "Carpet and kilim shops", "Traditional ceramics",
                    "Turkish delight and spice shops", "Souvenir boutiques"
                ],
                cultural_notes=[
                    "Dress modestly for mosque visits", "Remove shoes before entering mosques",
                    "Photography restrictions in some historical sites", "Very busy during summer months"
                ],
                practical_tips=[
                    "Start early (8 AM) to avoid crowds", "Buy Museum Pass Istanbul for savings",
                    "Beware of carpet shop touts", "Use official guides for authentic information",
                    "Stay hydrated - limited shade in summer"
                ],
                best_times=["Early morning (8-10 AM)", "Late afternoon (4-6 PM)", "Off-season (Nov-Mar)"],
                transportation_hub=True,
                tourist_density="high", 
                local_authenticity="medium",
                budget_level="mixed"
            ),
            
            "beyoglu": LocationData(
                name="Beyoğlu",
                description="Modern cultural district with vibrant nightlife and European atmosphere",
                character="Cosmopolitan area blending Ottoman heritage with modern Turkish culture",
                main_attractions=[
                    "Istiklal Street", "Galata Tower", "Pera Museum", "Galatasaray High School",
                    "Flower Passage", "Fish Market", "Cicek Pasaji", "Neve Shalom Synagogue"
                ],
                metro_stations=["Şişhane", "Taksim"],
                tram_stops=["Karaköy", "Tophane"],
                ferry_terminals=["Karaköy"],
                walking_distances={
                    "Taksim Square to Istiklal Street": "0 minutes (same location)",
                    "Istiklal Street to Galata Tower": "10-minute walk (750 meters)",
                    "Galata Tower to Karaköy": "8-minute walk downhill (600 meters)",
                    "Taksim to Cihangir": "15-minute walk (1.2 km)",
                    "Karaköy to Galata Bridge": "5-minute walk (400 meters)"
                },
                local_landmarks=[
                    "Taksim Square", "Gezi Park", "French Cultural Center", "British Consulate",
                    "Pera Palace Hotel", "Tunnel funicular station", "Galatasaray Square"
                ],
                dining_specialties=[
                    "International cuisine", "Rooftop restaurants with Bosphorus views",
                    "Traditional meyhanes", "Modern Turkish fusion", "Craft cocktail bars"
                ],
                shopping_highlights=[
                    "Istiklal Street boutiques", "Vintage shops in Cihangir", 
                    "European brand stores", "Local designer fashion", "Antique shops in Galata"
                ],
                cultural_notes=[
                    "Lively nightlife scene", "Art galleries and cultural centers",
                    "Mix of different religious communities", "European architectural influence"
                ],
                practical_tips=[
                    "Evening is best for atmosphere", "Book restaurant reservations in advance",
                    "Use Tünel funicular for Galata area", "Street musicians and performers common",
                    "Dress up for upscale venues"
                ],
                best_times=["Evening (6-11 PM)", "Weekend afternoons", "Cultural event seasons"],
                transportation_hub=True,
                tourist_density="high",
                local_authenticity="medium", 
                budget_level="upscale"
            ),
            
            "kadikoy": LocationData(
                name="Kadıköy",
                description="Authentic Asian-side neighborhood with local markets and seaside charm",
                character="Less touristy, more authentic Turkish neighborhood with great food scene",
                main_attractions=[
                    "Moda neighborhood", "Bahariye Street", "Kadıköy Market", "Fenerbahçe Park",
                    "Sureyya Opera House", "Yeldeğirmeni Art District", "Moda coastline"
                ],
                metro_stations=["Kadıköy"],
                tram_stops=["Kadıköy"],
                ferry_terminals=["Kadıköy"],
                walking_distances={
                    "Kadıköy ferry to Bahariye Street": "3-minute walk (250 meters)",
                    "Bahariye to Moda": "15-minute walk (1.1 km)", 
                    "Moda to seaside promenade": "5-minute walk (350 meters)",
                    "Kadıköy Market to ferry": "8-minute walk (600 meters)",
                    "Yeldeğirmeni to Kadıköy center": "12-minute walk (900 meters)"
                },
                local_landmarks=[
                    "Bull Statue", "Kadıköy Pier", "Moda Pier", "Reşitpaşa Mosque",
                    "Özgürlük Park", "Caferağa Madrassa", "Barış Manço House"
                ],
                dining_specialties=[
                    "Authentic Turkish home cooking", "Fresh fish restaurants",
                    "Traditional kahvaltı (breakfast)", "Local street food", "Family-run lokantası"
                ],
                shopping_highlights=[
                    "Tuesday market (Salı Pazarı)", "Local produce markets",
                    "Independent bookstores", "Vintage clothing shops", "Handmade crafts"
                ],
                cultural_notes=[
                    "More conservative than European side", "Strong local community feel",
                    "Traditional Turkish family culture", "Less English spoken"
                ],
                practical_tips=[
                    "Learn basic Turkish phrases", "Try local markets for authentic experience", 
                    "Ferry ride offers great city views", "More affordable than European side",
                    "Best for experiencing local life"
                ],
                best_times=["Morning for markets", "Sunset at Moda", "Weekday evenings for authentic feel"],
                transportation_hub=True,
                tourist_density="low",
                local_authenticity="high",
                budget_level="budget"
            ),

            "karakoy": LocationData(
                name="Karaköy", 
                description="Historic port district transformed into trendy arts and dining quarter",
                character="Industrial heritage meets contemporary culture with Bosphorus views",
                main_attractions=[
                    "Istanbul Modern Art Museum", "Galata Bridge", "Karaköy Port",
                    "Salt Galata", "Galata Mevlevi Lodge", "Camondo Steps", "Bankalar Caddesi"
                ],
                metro_stations=["Şişhane"],
                tram_stops=["Karaköy", "Tophane"],
                ferry_terminals=["Karaköy"],
                walking_distances={
                    "Karaköy ferry to Galata Tower": "8-minute uphill walk (650 meters)",
                    "Galata Bridge to Karaköy": "3-minute walk (200 meters)",
                    "Istanbul Modern to Tophane": "10-minute walk (750 meters)", 
                    "Karaköy to Galata Bridge": "5-minute walk (400 meters)",
                    "Salt Galata to Galata Tower": "3-minute walk (250 meters)"
                },
                local_landmarks=[
                    "Galata Bridge", "Karaköy Square", "Perşembe Pazarı", 
                    "Yeraltı Mosque", "Maritime Lines Building", "Minerva Han"
                ],
                dining_specialties=[
                    "Seafood with Bosphorus views", "Contemporary Turkish cuisine",
                    "Third-wave coffee culture", "Romantic waterfront dining", "Fusion restaurants"
                ],
                shopping_highlights=[
                    "Contemporary art galleries", "Design studios", "Antique shops on Bankalar Caddesi",
                    "Local designer boutiques", "Specialty coffee roasters"
                ],
                cultural_notes=[
                    "Rapidly gentrifying area", "Mix of old and new architecture",
                    "Growing arts scene", "Popular with young professionals"
                ],
                practical_tips=[
                    "Great for romantic dinners", "Book waterfront restaurants in advance",
                    "Steep hills - wear comfortable shoes", "Perfect for sunset photography",
                    "Combine with Galata Tower visit"
                ],
                best_times=["Sunset for dining", "Afternoon for art galleries", "Evening for nightlife"],
                transportation_hub=True,
                tourist_density="medium",
                local_authenticity="medium",
                budget_level="upscale"
            ),

            "eminonu": LocationData(
                name="Eminönü",
                description="Historic commercial heart with spice markets and ferry terminals",
                character="Bustling trading district where Europe meets Asia via water",
                main_attractions=[
                    "Spice Bazaar (Egyptian Bazaar)", "New Mosque", "Rüstem Pasha Mosque",
                    "Golden Horn", "Ferry terminals", "Pandeli Restaurant", "Hamdi Restaurant"
                ],
                metro_stations=["Eminönü"],
                tram_stops=["Eminönü"],
                ferry_terminals=["Eminönü"],
                walking_distances={
                    "Spice Bazaar to New Mosque": "2-minute walk (150 meters)",
                    "Eminönü to Galata Bridge": "3-minute walk (250 meters)",
                    "Ferry terminal to Spice Bazaar": "1-minute walk (100 meters)",
                    "Eminönü to Sultanahmet": "10-minute walk (800 meters)",
                    "New Mosque to Rüstem Pasha Mosque": "5-minute walk (400 meters)"
                },
                local_landmarks=[
                    "Eminönü Square", "Golden Horn Bridge", "Valide Han",
                    "Tahtakale", "Haseki Hürrem Sultan Hammam", "Büyük Valide Han"
                ],
                dining_specialties=[
                    "Traditional Ottoman cuisine", "Famous fish sandwich (balık ekmek)", 
                    "Turkish delight and baklava", "Spice market tastings", "Historic restaurants"
                ],
                shopping_highlights=[
                    "Spices and Turkish delight", "Traditional Turkish coffee",
                    "Dried fruits and nuts", "Turkish ceramics", "Prayer rugs and textiles"
                ],
                cultural_notes=[
                    "Very busy commercial area", "Traditional Turkish business culture",
                    "Important transportation hub", "Mix of locals and tourists"
                ],
                practical_tips=[
                    "Visit early to avoid crowds", "Try free spice samples", 
                    "Bargaining expected in markets", "Watch for pickpockets", 
                    "Great starting point for ferry tours"
                ],
                best_times=["Early morning (8-10 AM)", "Late afternoon (4-6 PM)", "Avoid rush hours"],
                transportation_hub=True,
                tourist_density="high",
                local_authenticity="medium", 
                budget_level="budget"
            ),

            "taksim": LocationData(
                name="Taksim",
                description="Modern city center and transportation hub with hotels and nightlife",
                character="Bustling modern district serving as the heart of contemporary Istanbul",
                main_attractions=[
                    "Taksim Square", "Gezi Park", "Atatürk Cultural Center",
                    "Republic Monument", "French Cultural Center", "Maksem Art Center"
                ],
                metro_stations=["Taksim"],
                tram_stops=[],
                ferry_terminals=[],
                walking_distances={
                    "Taksim Square to Istiklal Street": "0 minutes (same location)",
                    "Taksim to Gezi Park": "2-minute walk (150 meters)",
                    "Taksim to Nişantaşı": "20-minute walk (1.5 km)",
                    "Taksim to Cihangir": "15-minute walk (1.2 km)",
                    "Taksim to Beşiktaş": "25-minute walk (2 km)"
                },
                local_landmarks=[
                    "Taksim Square", "Republic Monument", "Inönü Stadium",
                    "Hilton Hotel", "Divan Hotel", "Marmara Hotel"
                ],
                dining_specialties=[
                    "International hotel restaurants", "Rooftop bars and lounges",
                    "Late-night dining", "Chain restaurants", "Food courts"
                ],
                shopping_highlights=[
                    "Major international brands", "Shopping malls nearby",
                    "Hotel gift shops", "Street vendors", "Souvenir stands"
                ],
                cultural_notes=[
                    "Political and cultural significance", "Major protest location historically",
                    "Mix of tourists and locals", "24/7 activity"
                ],
                practical_tips=[
                    "Major transportation hub", "Book hotels in advance",
                    "Can be crowded and noisy", "Good base for exploring",
                    "Safe area but watch belongings"
                ],
                best_times=["Any time - always active", "Evening for nightlife", "Morning for transportation"],
                transportation_hub=True,
                tourist_density="high",
                local_authenticity="low",
                budget_level="mixed"
            )
        }

    def _build_response_enhancers(self) -> Dict[str, Dict[str, Any]]:
        """Build response enhancement templates for different query types"""
        return {
            "restaurant": {
                "location_intro": "For dining in {location}, here are the best options in this {character} area:",
                "specialties_intro": "This neighborhood is known for:",
                "walking_context": "Within walking distance you'll find:",
                "practical_dining": "Dining tips for this area:",
                "atmosphere": "The dining atmosphere here is:",
            },
            "transportation": {
                "location_intro": "Getting around {location} is straightforward with these transport options:",
                "stations_context": "Transportation hubs in this area:",
                "walking_context": "Key walking distances:",
                "practical_transport": "Transportation tips:",
                "connections": "This area connects to:",
            },
            "attractions": {
                "location_intro": "{location} offers these {character} attractions:",
                "main_sites": "Must-see attractions:",
                "walking_context": "Walking between major sites:",
                "cultural_context": "Cultural significance:",
                "visiting_tips": "Best visiting strategies:",
            },
            "shopping": {
                "location_intro": "Shopping in {location} offers {character} experiences:",
                "highlights": "Shopping highlights:",
                "walking_context": "Shopping areas within walking distance:",
                "local_specialties": "Local shopping specialties:",
                "practical_shopping": "Shopping tips for this area:",
            },
            "accommodation": {
                "location_intro": "Staying in {location} puts you in a {character} area:",
                "location_benefits": "Benefits of this location:",
                "walking_context": "Within walking distance:",
                "practical_stay": "Tips for staying here:",
                "transportation_access": "Transportation connections:",
            },
            "general": {
                "location_intro": "{location} is a {character} area that offers:",
                "character_description": "The neighborhood character:",
                "main_highlights": "Key highlights:",
                "walking_context": "Getting around on foot:",
                "local_tips": "Local insider tips:",
            }
        }

    def enhance_response_with_location(self, query: str, location: str, 
                                     query_type: str, base_response: str) -> str:
        """Enhance any response with comprehensive location-specific information"""
        
        location_key = location.lower()
        if location_key not in self.location_database:
            return base_response
        
        location_data = self.location_database[location_key]
        
        # Determine query type for enhancement
        enhancement_type = self._map_query_type(query_type)
        enhancer = self.response_enhancers.get(enhancement_type, self.response_enhancers["general"])
        
        # Build enhanced response
        enhanced_parts = []
        
        # Add location-specific introduction
        location_intro = enhancer["location_intro"].format(
            location=location_data.name,
            character=location_data.character
        )
        enhanced_parts.append(location_intro)
        
        # Add base response with location context
        enhanced_base = self._inject_location_context(base_response, location_data)
        enhanced_parts.append(enhanced_base)
        
        # Add location-specific walking distances
        if location_data.walking_distances:
            enhanced_parts.append("\n🚶 **Walking Distances in " + location_data.name + ":**")
            for route, distance in list(location_data.walking_distances.items())[:3]:  # Top 3 most relevant
                enhanced_parts.append(f"• {route}: {distance}")
        
        # Add transportation information
        transport_info = self._build_transport_info(location_data)
        if transport_info:
            enhanced_parts.append(transport_info)
        
        # Add location-specific practical tips
        if location_data.practical_tips:
            enhanced_parts.append(f"\n💡 **Tips for {location_data.name}:**")
            for tip in location_data.practical_tips[:3]:  # Top 3 tips
                enhanced_parts.append(f"• {tip}")
        
        # Add local landmarks for context
        if location_data.local_landmarks:
            enhanced_parts.append(f"\n📍 **Local Landmarks:** {', '.join(location_data.local_landmarks[:4])}")
        
        # Add timing recommendations
        if location_data.best_times:
            enhanced_parts.append(f"\n⏰ **Best Times to Visit:** {', '.join(location_data.best_times)}")
        
        return "\n".join(enhanced_parts)

    def _map_query_type(self, query_type: str) -> str:
        """Map query types to enhancement categories"""
        mapping = {
            "restaurant_specific": "restaurant",
            "restaurant_general": "restaurant", 
            "transportation": "transportation",
            "cultural_sites": "attractions",
            "shopping": "shopping",
            "accommodation": "accommodation",
            "nightlife": "general",
            "practical_info": "general"
        }
        return mapping.get(query_type, "general")

    def _inject_location_context(self, base_response: str, location_data: LocationData) -> str:
        """Inject location-specific context into base response"""
        
        # Add location character description if not present
        if location_data.name.lower() not in base_response.lower():
            base_response = f"In {location_data.name}, {base_response.lower()}"
        
        # Enhance with location-specific details
        enhanced_response = base_response
        
        # Add dining context for food-related queries
        if any(word in base_response.lower() for word in ['restaurant', 'food', 'dining', 'eat']):
            if location_data.dining_specialties:
                enhanced_response += f"\n\nThis area is particularly known for {', '.join(location_data.dining_specialties[:2])}."
        
        # Add shopping context for shopping-related queries  
        if any(word in base_response.lower() for word in ['shop', 'buy', 'market', 'bazaar']):
            if location_data.shopping_highlights:
                enhanced_response += f"\n\nFor shopping, this area offers {', '.join(location_data.shopping_highlights[:2])}."
        
        # Add cultural context for cultural queries
        if any(word in base_response.lower() for word in ['culture', 'history', 'museum', 'mosque', 'site']):
            if location_data.cultural_notes:
                enhanced_response += f"\n\nCultural note: {location_data.cultural_notes[0]}"
                
        return enhanced_response

    def _build_transport_info(self, location_data: LocationData) -> str:
        """Build comprehensive transportation information"""
        transport_parts = []
        
        if location_data.metro_stations or location_data.tram_stops or location_data.ferry_terminals:
            transport_parts.append(f"\n🚇 **Transportation for {location_data.name}:**")
            
            if location_data.metro_stations:
                transport_parts.append(f"• Metro: {', '.join(location_data.metro_stations)}")
            if location_data.tram_stops:
                transport_parts.append(f"• Tram: {', '.join(location_data.tram_stops)}")  
            if location_data.ferry_terminals:
                transport_parts.append(f"• Ferry: {', '.join(location_data.ferry_terminals)}")
        
        return "\n".join(transport_parts) if transport_parts else ""

    def get_location_specific_prompt_enhancement(self, query: str, location: str) -> str:
        """Generate location-specific prompt enhancements for GPT"""
        
        location_key = location.lower()
        if location_key not in self.location_database:
            return ""
        
        location_data = self.location_database[location_key]
        
        enhancement = f"""
SPECIFIC LOCATION FOCUS: {location_data.name}
Area Character: {location_data.character}

KEY LOCAL CONTEXT TO INCLUDE:
- Main attractions: {', '.join(location_data.main_attractions[:4])}
- Walking distances: {'; '.join([f"{k}: {v}" for k, v in list(location_data.walking_distances.items())[:3]])}
- Transportation: Metro {', '.join(location_data.metro_stations)} | Tram {', '.join(location_data.tram_stops)} | Ferry {', '.join(location_data.ferry_terminals)}
- Local specialties: {', '.join(location_data.dining_specialties[:2] if location_data.dining_specialties else [])}
- Character: {location_data.tourist_density} tourist density, {location_data.local_authenticity} authenticity, {location_data.budget_level} budget level

PRACTICAL DETAILS TO MENTION:
{chr(10).join(['- ' + tip for tip in location_data.practical_tips[:3]])}

RESPONSE INSTRUCTIONS:
1. Start with location-specific context about {location_data.name}
2. Include specific walking distances to mentioned places
3. Reference nearby landmarks: {', '.join(location_data.local_landmarks[:3])}
4. Mention transportation options available in this area
5. Include area-specific practical tips
6. Focus entirely on this neighborhood - don't give generic Istanbul advice
"""
        
        return enhancement.strip()

    def detect_location_from_query(self, query: str) -> Optional[str]:
        """Detect location mentioned in query with fuzzy matching"""
        query_lower = query.lower()
        
        # Direct location name matching
        location_patterns = {
            'sultanahmet': [r'sultanahmet', r'sultan ahmet', r'blue mosque', r'hagia sophia', r'topkapi'],
            'beyoglu': [r'beyoğlu', r'beyoglu', r'istiklal', r'galata tower', r'taksim', r'pera'],
            'kadikoy': [r'kadıköy', r'kadikoy', r'moda', r'asian side', r'bahariye'],
            'karakoy': [r'karaköy', r'karakoy', r'galata bridge', r'port'],
            'eminonu': [r'eminönü', r'eminonu', r'spice bazaar', r'egyptian bazaar', r'new mosque'],
            'taksim': [r'taksim', r'taksim square', r'gezi park']
        }
        
        for location, patterns in location_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                return location
                
        return None

    def get_neighborhood_recommendations(self, query_type: str, budget: str = "mixed") -> Dict[str, List[str]]:
        """Get neighborhood recommendations based on query type and budget"""
        recommendations = {}
        
        for location_key, location_data in self.location_database.items():
            score = 0
            reasons = []
            
            # Score based on query type
            if query_type == "restaurant":
                if location_data.dining_specialties:
                    score += 3
                    reasons.append("Great dining scene")
                if location_data.tourist_density == "medium":
                    score += 1
                    reasons.append("Good mix of tourist and local options")
                    
            elif query_type == "cultural":
                if location_data.main_attractions:
                    score += len(location_data.main_attractions)
                    reasons.append("Rich in cultural attractions")
                if location_data.local_authenticity == "high":
                    score += 2
                    reasons.append("Authentic cultural experience")
                    
            elif query_type == "shopping":
                if location_data.shopping_highlights:
                    score += 2
                    reasons.append("Good shopping options")
                if location_data.tourist_density == "high":
                    score += 1
                    reasons.append("Tourist-friendly shopping")
                    
            elif query_type == "nightlife":
                if "nightlife" in location_data.character.lower():
                    score += 3
                    reasons.append("Vibrant nightlife")
                if location_data.budget_level == "upscale":
                    score += 1
                    reasons.append("Upscale venues")
            
            # Score based on budget preference
            if budget == "budget" and location_data.budget_level in ["budget", "mixed"]:
                score += 2
                reasons.append("Budget-friendly")
            elif budget == "upscale" and location_data.budget_level in ["upscale", "mixed"]:
                score += 2
                reasons.append("Upscale options")
                
            if score > 0:
                recommendations[location_key] = {
                    "score": score,
                    "reasons": reasons,
                    "data": location_data
                }
        
        # Sort by score
        sorted_recommendations = dict(sorted(recommendations.items(), key=lambda x: x[1]["score"], reverse=True))
        
        return sorted_recommendations

# Global location enhancer instance
location_enhancer = LocationEnhancer()

def enhance_response_with_location_context(query: str, base_response: str, 
                                         location: Optional[str] = None, 
                                         query_type: str = "general") -> str:
    """Main function to enhance any response with location context"""
    
    # Detect location if not provided
    if not location:
        location = location_enhancer.detect_location_from_query(query)
    
    if not location:
        return base_response
    
    return location_enhancer.enhance_response_with_location(
        query, location, query_type, base_response
    )

def get_location_enhanced_gpt_prompt(query: str, location: Optional[str] = None) -> str:
    """Generate location-enhanced GPT prompt"""
    
    # Detect location if not provided
    if not location:
        location = location_enhancer.detect_location_from_query(query)
    
    if not location:
        return query
        
    enhancement = location_enhancer.get_location_specific_prompt_enhancement(query, location)
    
    return f"{query}\n\n{enhancement}"

# Test the location enhancer
if __name__ == "__main__":
    test_queries = [
        ("I want good restaurants in Sultanahmet", "sultanahmet"),
        ("How do I get around Kadıköy?", "kadikoy"), 
        ("What shopping is available in Beyoğlu?", "beyoglu"),
        ("Tell me about nightlife in Karaköy", "karakoy")
    ]
    
    print("🎯 LOCATION-SPECIFIC RESPONSE ENHANCEMENT TEST")
    print("=" * 60)
    
    for query, location in test_queries:
        print(f"\n📍 Location: {location.title()}")
        print(f"❓ Query: {query}")
        
        # Test location detection
        detected = location_enhancer.detect_location_from_query(query)
        print(f"🔍 Detected Location: {detected}")
        
        # Test prompt enhancement
        enhanced_prompt = get_location_enhanced_gpt_prompt(query, location)
        print(f"🚀 Enhanced Prompt Length: {len(enhanced_prompt)} characters")
        
        # Test response enhancement
        base_response = "Here are some recommendations for your query."
        enhanced_response = enhance_response_with_location_context(
            query, base_response, location, "restaurant"
        )
        print(f"✨ Enhanced Response Preview: {enhanced_response[:200]}...")
        print("-" * 60)

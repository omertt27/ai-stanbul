#!/usr/bin/env python3
"""
Enhanced AI Blog Generator for AI Istanbul
Uses real Google Places API and weather data to generate contextual blog posts
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Import your enhanced API clients
from api_clients.enhanced_google_places import EnhancedGooglePlacesClient
from api_clients.google_weather import GoogleWeatherClient
from api_clients.istanbul_transport import IstanbulTransportClient

logger = logging.getLogger(__name__)

class EnhancedAIBlogGenerator:
    """AI-powered blog generator using real Istanbul data"""
    
    def __init__(self):
        self.places_client = EnhancedGooglePlacesClient()
        self.weather_client = GoogleWeatherClient()
        self.transport_client = IstanbulTransportClient()
        
        # Blog templates for different types of content
        self.templates = {
            "food_guide": self._food_guide_template,
            "neighborhood_guide": self._neighborhood_template,
            "seasonal_guide": self._seasonal_template,
            "hidden_gems": self._hidden_gems_template,
            "weather_based": self._weather_based_template
        }
    
    async def generate_contextual_blog_post(self, 
                                          topic: str, 
                                          post_type: str = "neighborhood_guide",
                                          location: str = "Istanbul") -> Dict[str, Any]:
        """Generate a blog post using real Istanbul data"""
        
        try:
            # Get real data from APIs
            places_data = self._get_places_data(topic, location)
            weather_data = self._get_weather_context()
            transport_data = self._get_transport_info(location)
            
            # Generate content using template
            template_func = self.templates.get(post_type, self._neighborhood_template)
            content = template_func(topic, places_data, weather_data, transport_data)
            
            # Create blog post structure
            blog_post = {
                "id": str(uuid.uuid4()),
                "title": self._generate_title(topic, post_type),
                "content": content,
                "author": "AI Istanbul Guide",
                "category": post_type,
                "tags": self._generate_tags(topic, post_type),
                "featured_image": self._suggest_featured_image(topic),
                "published": True,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "views": 0,
                "likes": 0,
                "meta_data": {
                    "seo_title": self._generate_seo_title(topic),
                    "meta_description": self._generate_meta_description(topic, content),
                    "reading_time": self._estimate_reading_time(content),
                    "difficulty": self._assess_difficulty(places_data),
                    "budget_range": self._estimate_budget(places_data),
                    "best_time_to_visit": self._suggest_best_time(weather_data)
                }
            }
            
            logger.info(f"Generated blog post: {blog_post['title']}")
            return blog_post
            
        except Exception as e:
            logger.error(f"Error generating blog post: {e}")
            return self._create_fallback_post(topic, post_type)
    
    def _get_places_data(self, topic: str, location: str) -> Dict:
        """Get real places data from Google Places API"""
        try:
            # Search for restaurants/places related to the topic
            if "food" in topic.lower() or "restaurant" in topic.lower():
                return self.places_client.search_restaurants(location, topic)
            else:
                # For general places
                return self.places_client.search_restaurants(location, keyword=topic)
        except Exception as e:
            logger.warning(f"Could not fetch places data: {e}")
            return {"results": []}
    
    def _get_weather_context(self) -> Dict:
        """Get current weather context"""
        try:
            return self.weather_client.get_current_weather("Istanbul")
        except Exception as e:
            logger.warning(f"Could not fetch weather data: {e}")
            return {}
    
    def _get_transport_info(self, location: str) -> Dict:
        """Get transport information for the area"""
        try:
            return self.transport_client.get_route_info("Taksim", location)
        except Exception as e:
            logger.warning(f"Could not fetch transport data: {e}")
            return {}
    
    def _food_guide_template(self, topic: str, places_data: Dict, weather_data: Dict, transport_data: Dict) -> str:
        """Generate food guide blog post"""
        
        restaurants = places_data.get('results', [])[:5]  # Top 5 restaurants
        weather_temp = weather_data.get('main', {}).get('temp', 20)
        weather_desc = weather_data.get('weather', [{}])[0].get('description', 'pleasant')
        
        content = f"""# 🍽️ {topic}: A Local's Guide to Istanbul's Best Flavors

*Generated with real-time data from local establishments*

## 🌤️ Today's Weather & Dining Recommendations

With temperatures at {weather_temp}°C and {weather_desc} conditions, it's perfect for exploring Istanbul's diverse food scene!

"""
        
        if restaurants:
            content += "## 🏆 Top Local Favorites\n\n"
            for i, restaurant in enumerate(restaurants, 1):
                name = restaurant.get('name', f'Local Restaurant {i}')
                rating = restaurant.get('rating', 'N/A')
                vicinity = restaurant.get('vicinity', 'Istanbul')
                price_level = '💰' * restaurant.get('price_level', 2)
                
                content += f"""### {i}. {name}
**Rating:** {rating}⭐ | **Price:** {price_level} | **Location:** {vicinity}

{self._generate_restaurant_description(restaurant)}

"""
        
        # Add weather-based recommendations
        if weather_temp > 25:
            content += """## ☀️ Perfect Weather for Outdoor Dining

Take advantage of this beautiful weather:
- **Rooftop terraces** with Bosphorus views
- **Garden restaurants** in Bebek
- **Waterfront dining** along the Golden Horn

"""
        elif weather_temp < 15:
            content += """## 🏠 Cozy Indoor Dining Recommendations

Perfect weather for warm, traditional restaurants:
- **Historic Ottoman lokanta** experiences
- **Traditional Turkish breakfast** spots
- **Warm soup and stew** specialties

"""
        
        content += self._add_practical_info(transport_data)
        
        return content
    
    def _neighborhood_template(self, topic: str, places_data: Dict, weather_data: Dict, transport_data: Dict) -> str:
        """Generate neighborhood guide blog post"""
        
        places = places_data.get('results', [])[:3]
        weather_temp = weather_data.get('main', {}).get('temp', 20)
        
        content = f"""# 🏘️ Exploring {topic}: Your Local Insider Guide

*A comprehensive guide featuring real establishments and current conditions*

## 🎯 Why Visit {topic} Today?

With current temperatures at {weather_temp}°C, {topic} offers the perfect blend of culture, cuisine, and authentic Istanbul atmosphere.

"""
        
        if places:
            content += "## 🍽️ Must-Visit Local Spots\n\n"
            for place in places:
                name = place.get('name', 'Local Establishment')
                rating = place.get('rating', 'N/A')
                types = ', '.join(place.get('types', ['restaurant'])[:2])
                
                content += f"""### {name}
**Type:** {types} | **Rating:** {rating}⭐

{self._generate_place_description(place)}

"""
        
        # Add neighborhood-specific insights
        content += f"""## 🚶‍♂️ Walking Tour Suggestions

### Morning Route (2-3 hours)
Perfect for the current {weather_data.get('weather', [{}])[0].get('description', 'weather')} conditions:

1. **Start at the main square** - Get oriented and grab Turkish coffee
2. **Explore side streets** - Discover hidden architectural gems  
3. **Visit local market** - Experience authentic shopping culture
4. **End at a scenic viewpoint** - Perfect for photos

### What Locals Recommend
- **Best time to visit:** Early morning or late afternoon
- **Dress code:** Comfortable walking shoes recommended
- **Budget:** €20-40 per person for full day experience

"""
        
        content += self._add_practical_info(transport_data)
        
        return content
    
    def _seasonal_template(self, topic: str, places_data: Dict, weather_data: Dict, transport_data: Dict) -> str:
        """Generate seasonal guide blog post"""
        
        current_month = datetime.now().month
        season = self._get_season(current_month)
        weather_temp = weather_data.get('main', {}).get('temp', 20)
        
        content = f"""# 🌅 {topic} in {season}: A Seasonal Istanbul Experience

*Tailored recommendations based on current weather and season*

## 🌡️ Current Conditions: {weather_temp}°C

{self._get_seasonal_intro(season, weather_temp)}

"""
        
        places = places_data.get('results', [])
        if places:
            content += f"""## 🎯 Perfect {season} Activities

Based on real-time data and current weather conditions:

"""
            for place in places[:3]:
                content += f"""### {place.get('name', 'Local Spot')}
{self._generate_seasonal_recommendation(place, season, weather_temp)}

"""
        
        # Add seasonal clothing and preparation advice
        content += f"""## 👕 What to Wear & Bring

For {weather_temp}°C weather in {season}:
{weather_data.get('clothing_advice', 'Dress comfortably for the weather')}

## 📅 Best Times Today
- **Sunrise:** {self._format_time(weather_data.get('sys', {}).get('sunrise'))}
- **Sunset:** {self._format_time(weather_data.get('sys', {}).get('sunset'))}
- **Peak Hours:** Avoid 12:00-14:00 for outdoor activities

"""
        
        return content
    
    def _hidden_gems_template(self, topic: str, places_data: Dict, weather_data: Dict, transport_data: Dict) -> str:
        """Generate hidden gems guide"""
        
        content = f"""# 💎 Hidden Gems: {topic}

*Secret spots that locals love - updated with real-time information*

## 🤫 Why These Places Are Special

These aren't your typical tourist destinations. These are the places where Istanbulites actually go, verified through local data and real visitor experiences.

"""
        
        places = places_data.get('results', [])
        if places:
            content += "## 🗺️ Secret Locations\n\n"
            for i, place in enumerate(places[:4], 1):
                content += f"""### Secret #{i}: {place.get('name', 'Hidden Spot')}
**Why locals love it:** {self._generate_local_insight(place)}
**Best time to visit:** {self._suggest_visit_time(place)}
**Insider tip:** {self._generate_insider_tip(place)}

"""
        
        content += """## 🎯 How to Experience Like a Local

### Do:
- Arrive early morning or late afternoon
- Learn a few Turkish phrases
- Try the daily specials
- Respect photography rules

### Don't:
- Visit during peak tourist hours
- Be loud in quiet neighborhood spots
- Skip tipping (10-15% is standard)
- Forget to remove shoes when required

"""
        
        return content
    
    def _weather_based_template(self, topic: str, places_data: Dict, weather_data: Dict, transport_data: Dict) -> str:
        """Generate weather-specific recommendations"""
        
        weather_temp = weather_data.get('main', {}).get('temp', 20)
        weather_condition = weather_data.get('weather', [{}])[0].get('main', 'Clear')
        
        content = f"""# 🌤️ Perfect Day for {topic} - Weather-Optimized Guide

*Customized recommendations based on today's {weather_temp}°C {weather_condition.lower()} conditions*

## 🎯 Today's Perfect Activities

The weather is {weather_condition.lower()} with {weather_temp}°C - here's how to make the most of it:

"""
        
        # Weather-specific recommendations
        if weather_temp > 25:
            content += """### ☀️ Hot Weather Activities
- **Morning:** Early exploration before it gets too warm
- **Midday:** Indoor attractions with air conditioning
- **Afternoon:** Shaded areas and covered markets
- **Evening:** Outdoor dining and sunset viewpoints

"""
        elif weather_temp < 15:
            content += """### 🧥 Cool Weather Activities
- **Morning:** Warm up with Turkish breakfast
- **Midday:** Indoor museums and cultural sites
- **Afternoon:** Cozy cafes and tea houses
- **Evening:** Traditional bathhouses and warm restaurants

"""
        else:
            content += """### 🌤️ Perfect Weather Activities
- **All day:** Comfortable for any outdoor activity
- **Morning:** Walking tours and sightseeing
- **Afternoon:** Shopping and exploring neighborhoods
- **Evening:** Outdoor dining and entertainment

"""
        
        # Add activity recommendations from weather data
        activity_recommendations = weather_data.get('activity_recommendations', [])
        if activity_recommendations:
            content += "### 🎯 AI-Powered Activity Suggestions\n"
            for activity in activity_recommendations:
                content += f"- {activity}\n"
            content += "\n"
        
        return content
    
    # Helper methods
    def _generate_restaurant_description(self, restaurant: Dict) -> str:
        """Generate description for restaurant"""
        types = restaurant.get('types', [])
        cuisine_type = 'Turkish' if 'restaurant' in types else 'Local'
        
        descriptions = [
            f"Authentic {cuisine_type} cuisine with local flavors",
            f"Popular among locals for its traditional atmosphere",
            f"Known for fresh ingredients and generous portions",
            f"Perfect for experiencing genuine Istanbul dining culture"
        ]
        
        return descriptions[hash(restaurant.get('place_id', '')) % len(descriptions)]
    
    def _generate_place_description(self, place: Dict) -> str:
        """Generate description for general place"""
        return f"A beloved local spot that embodies the authentic spirit of Istanbul, perfect for visitors seeking genuine cultural experiences."
    
    def _generate_title(self, topic: str, post_type: str) -> str:
        """Generate engaging title"""
        title_templates = {
            "food_guide": f"🍽️ {topic}: Where Locals Actually Eat in Istanbul",
            "neighborhood_guide": f"🏘️ Insider's Guide to {topic}: Beyond the Tourist Trail", 
            "seasonal_guide": f"🌅 {topic} This Season: Perfect Weather Activities",
            "hidden_gems": f"💎 {topic}: Secret Spots Locals Don't Want You to Know",
            "weather_based": f"🌤️ Perfect Day for {topic}: Weather-Optimized Istanbul Guide"
        }
        return title_templates.get(post_type, f"{topic}: Your Local Istanbul Guide")
    
    def _generate_tags(self, topic: str, post_type: str) -> List[str]:
        """Generate relevant tags"""
        base_tags = ["istanbul", "travel", "local", "guide"]
        
        if "food" in topic.lower():
            base_tags.extend(["food", "restaurant", "cuisine", "dining"])
        if post_type == "hidden_gems":
            base_tags.extend(["hidden", "secret", "authentic", "locals"])
        if post_type == "seasonal_guide":
            base_tags.extend(["seasonal", "weather", "activities"])
            
        return base_tags[:8]  # Limit to 8 tags
    
    def _generate_seo_title(self, topic: str) -> str:
        """Generate SEO-optimized title"""
        return f"{topic} Istanbul: Local Guide 2024 | AI Istanbul"
    
    def _generate_meta_description(self, topic: str, content: str) -> str:
        """Generate meta description"""
        return f"Discover {topic} in Istanbul with real-time local recommendations. Updated with current weather, authentic spots, and insider tips from AI Istanbul."
    
    def _estimate_reading_time(self, content: str) -> int:
        """Estimate reading time in minutes"""
        words = len(content.split())
        return max(1, words // 200)  # Average 200 words per minute
    
    def _assess_difficulty(self, places_data: Dict) -> str:
        """Assess difficulty level of activities"""
        return "Easy"  # Default for now
    
    def _estimate_budget(self, places_data: Dict) -> str:
        """Estimate budget range"""
        restaurants = places_data.get('results', [])
        if restaurants:
            avg_price_level = sum(r.get('price_level', 2) for r in restaurants) / len(restaurants)
            if avg_price_level <= 1.5:
                return "Budget-friendly (€10-25/person)"
            elif avg_price_level <= 2.5:
                return "Moderate (€25-50/person)"
            else:
                return "Premium (€50+/person)"
        return "Moderate (€25-50/person)"
    
    def _suggest_best_time(self, weather_data: Dict) -> str:
        """Suggest best time to visit based on weather"""
        temp = weather_data.get('main', {}).get('temp', 20)
        if temp > 25:
            return "Early morning (8-10 AM) or evening (6-8 PM)"
        elif temp < 15:
            return "Midday (11 AM - 3 PM) for warmest temperatures"
        else:
            return "Any time - perfect weather!"
    
    def _add_practical_info(self, transport_data: Dict) -> str:
        """Add practical transportation and logistics info"""
        return """## 🚇 Getting There

### Public Transportation
- **Metro:** Multiple lines serve the area
- **Bus:** Regular city bus connections
- **Ferry:** Available for waterfront locations
- **Walking:** Most attractions within walking distance

### 💡 Pro Tips
- Get an Istanbul Card for easy public transport
- Download offline maps before exploring
- Keep some cash - not all places accept cards
- Learn basic Turkish phrases for better experiences

*This guide was generated using real-time data and is updated regularly for accuracy.*
"""
    
    # Additional helper methods
    def _get_season(self, month: int) -> str:
        """Get current season"""
        seasons = {
            (12, 1, 2): "Winter",
            (3, 4, 5): "Spring", 
            (6, 7, 8): "Summer",
            (9, 10, 11): "Autumn"
        }
        for months, season in seasons.items():
            if month in months:
                return season
        return "Spring"
    
    def _get_seasonal_intro(self, season: str, temp: float) -> str:
        """Get seasonal introduction"""
        intros = {
            "Winter": f"Winter in Istanbul brings crisp {temp}°C weather, perfect for cozy indoor experiences and warming Turkish cuisine.",
            "Spring": f"Spring magic at {temp}°C - ideal weather for exploring both indoor and outdoor attractions.",
            "Summer": f"Summer warmth at {temp}°C calls for early morning adventures and evening explorations.",
            "Autumn": f"Autumn's gentle {temp}°C temperatures create perfect conditions for walking and discovery."
        }
        return intros.get(season, f"Beautiful {temp}°C weather makes it perfect for Istanbul exploration!")
    
    def _generate_seasonal_recommendation(self, place: Dict, season: str, temp: float) -> str:
        """Generate seasonal recommendation for a place"""
        name = place.get('name', 'Local Spot')
        if temp > 25:
            return f"Perfect for air-conditioned comfort during hot {season} days. Known for cool interiors and refreshing drinks."
        elif temp < 15:
            return f"Ideal for warming up during cool {season} weather. Famous for hot drinks and cozy atmosphere."
        else:
            return f"Excellent choice for perfect {season} weather. Great for both indoor and outdoor experiences."
    
    def _format_time(self, timestamp: Optional[int]) -> str:
        """Format timestamp to readable time"""
        if not timestamp:
            return "N/A"
        return datetime.fromtimestamp(timestamp).strftime("%H:%M")
    
    def _generate_local_insight(self, place: Dict) -> str:
        """Generate local insight about a place"""
        insights = [
            "Authentic atmosphere that hasn't changed in decades",
            "Where locals bring their families for special occasions", 
            "Known for maintaining traditional preparation methods",
            "A gathering place for the local community"
        ]
        return insights[hash(place.get('place_id', '')) % len(insights)]
    
    def _suggest_visit_time(self, place: Dict) -> str:
        """Suggest best visit time"""
        times = [
            "Early morning for peaceful atmosphere",
            "Late afternoon for golden hour lighting",
            "Evening for local crowd energy",
            "Weekday mornings for authentic experience"
        ]
        return times[hash(place.get('place_id', '')) % len(times)]
    
    def _generate_insider_tip(self, place: Dict) -> str:
        """Generate insider tip"""
        tips = [
            "Ask for the daily special - it's not on the menu",
            "Sit where locals sit for the best experience",
            "Try the traditional preparation method",
            "Visit during local meal times for authenticity"
        ]
        return tips[hash(place.get('place_id', '')) % len(tips)]
    
    def _suggest_featured_image(self, topic: str) -> str:
        """Suggest featured image URL or path"""
        # This would integrate with your image system
        return f"/images/blog/{topic.lower().replace(' ', '_')}_featured.jpg"
    
    def _create_fallback_post(self, topic: str, post_type: str) -> Dict[str, Any]:
        """Create fallback post if API fails"""
        return {
            "id": str(uuid.uuid4()),
            "title": f"Exploring {topic}: Your Istanbul Guide",
            "content": f"# {topic}\n\nDiscover the authentic side of Istanbul with our local recommendations and insider tips.",
            "author": "AI Istanbul Guide",
            "category": post_type,
            "tags": ["istanbul", "travel", "guide"],
            "featured_image": None,
            "published": True,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "views": 0,
            "likes": 0,
            "meta_data": {
                "seo_title": f"{topic} Istanbul Guide",
                "meta_description": f"Explore {topic} in Istanbul with local recommendations",
                "reading_time": 3,
                "difficulty": "Easy",
                "budget_range": "Moderate",
                "best_time_to_visit": "Any time"
            }
        }

# Convenience functions for blog generation
async def generate_food_guide(neighborhood: str) -> Dict[str, Any]:
    """Quick function to generate food guide"""
    generator = EnhancedAIBlogGenerator()
    return await generator.generate_contextual_blog_post(
        topic=f"{neighborhood} Food Scene",
        post_type="food_guide",
        location=neighborhood
    )

async def generate_neighborhood_guide(neighborhood: str) -> Dict[str, Any]:
    """Quick function to generate neighborhood guide"""
    generator = EnhancedAIBlogGenerator()
    return await generator.generate_contextual_blog_post(
        topic=neighborhood,
        post_type="neighborhood_guide",
        location=neighborhood
    )

async def generate_seasonal_guide(activity: str) -> Dict[str, Any]:
    """Quick function to generate seasonal guide"""
    generator = EnhancedAIBlogGenerator()
    return await generator.generate_contextual_blog_post(
        topic=activity,
        post_type="seasonal_guide"
    )

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_blog_generation():
        """Test the blog generator"""
        generator = EnhancedAIBlogGenerator()
        
        # Test different types of blog posts
        test_cases = [
            ("Sultanahmet Restaurants", "food_guide"),
            ("Galata District", "neighborhood_guide"), 
            ("Istanbul Coffee Culture", "hidden_gems"),
            ("Bosphorus Walking Tour", "seasonal_guide")
        ]
        
        for topic, post_type in test_cases:
            print(f"\n🧪 Generating {post_type} for '{topic}'...")
            
            try:
                blog_post = await generator.generate_contextual_blog_post(topic, post_type)
                print(f"✅ Generated: {blog_post['title']}")
                print(f"📊 Reading time: {blog_post['meta_data']['reading_time']} minutes")
                print(f"💰 Budget: {blog_post['meta_data']['budget_range']}")
                print(f"📝 Content preview: {blog_post['content'][:200]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Run the test
    asyncio.run(test_blog_generation())
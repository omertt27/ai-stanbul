from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import sys
import os
import re
import asyncio
import json
import time
from datetime import datetime

# Add the current directory to Python path for Render deployment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import engine, SessionLocal
from models import Base, Restaurant, Museum, Place
from routes import museums, restaurants, places
import json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="AIstanbul API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
        # Production frontend URLs
        "https://aistanbul.vercel.app",
        "https://aistanbul-fdsqdpks5-omers-projects-3eea52d8.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables if needed
Base.metadata.create_all(bind=engine)

def create_fallback_response(user_input, places):
    """Create intelligent fallback responses when OpenAI API is unavailable"""
    user_input_lower = user_input.lower()
    
    # History and culture questions
    if any(word in user_input_lower for word in ['history', 'historical', 'culture', 'byzantine', 'ottoman']):
        return """🏛️ **Istanbul's Rich History**

Istanbul has over 2,500 years of history! Here are key highlights:

**Byzantine Era (330-1453 CE):**
- Originally called Constantinople
- Hagia Sophia built in 537 CE
- Capital of Byzantine Empire

**Ottoman Era (1453-1922):**
- Conquered by Mehmed II in 1453
- Became capital of Ottoman Empire
- Blue Mosque, Topkapi Palace built

**Modern Istanbul:**
- Turkey's largest city with 15+ million people
- Spans Europe and Asia across the Bosphorus
- UNESCO World Heritage sites in historic areas

Would you like to know about specific historical sites or districts?"""

    # Food and cuisine questions (no static restaurant recommendations)
    elif any(word in user_input_lower for word in ['food', 'eat', 'cuisine', 'dish', 'meal', 'breakfast', 'lunch', 'dinner']):
        return """🍽️ **Turkish Cuisine in Istanbul**\n\nI can provide live restaurant recommendations using Google Maps. Please ask for a specific type of restaurant or cuisine, and I'll fetch the latest options for you!\n\nMust-try Turkish dishes include döner kebab, simit, balık ekmek, midye dolma, iskender kebab, manti, lahmacun, börek, baklava, Turkish delight, künefe, Turkish tea, ayran, and raki.\n\nFor restaurant recommendations, please specify your preference (e.g., 'seafood in Kadıköy')."""

    # Transportation questions
    elif any(word in user_input_lower for word in ['transport', 'metro', 'bus', 'ferry', 'taxi', 'getting around']):
        return """🚇 **Getting Around Istanbul**

**Istanbul Card (Istanbulkart):**
- Essential for all public transport
- Buy at metro stations or kiosks
- Works on metro, bus, tram, ferry

**Metro & Tram:**
- Clean, efficient, connects major areas
- M1: Airport to city center
- M2: European side north-south
- Tram: Historic peninsula (Sultanahmet)

**Ferries:**
- Cross between European & Asian sides
- Scenic Bosphorus tours
- Kadıköy ↔ Eminönü popular route

**Taxis & Apps:**
- BiTaksi and Uber available
- Always ask for meter ("taksimetre")
- Airport to city: 45-60 TL

**Tips:**
- Rush hours: 8-10 AM, 5-7 PM
- Download offline maps
- Learn basic Turkish transport terms"""

    # Weather and timing questions
    elif any(word in user_input_lower for word in ['weather', 'climate', 'season', 'when to visit', 'best time']):
        return """🌤️ **Istanbul Weather & Best Times to Visit**

**Seasons:**

**Spring (April-May):** ⭐ BEST
- Perfect weather (15-22°C)
- Blooming tulips in parks
- Fewer crowds

**Summer (June-August):**
- Hot (25-30°C), humid
- Peak tourist season
- Great for Bosphorus activities

**Fall (September-November):** ⭐ EXCELLENT
- Mild weather (18-25°C)
- Beautiful autumn colors
- Ideal for walking tours

**Winter (December-March):**
- Cool, rainy (8-15°C)
- Fewer tourists, lower prices
- Cozy indoor experiences

**What to Pack:**
- Comfortable walking shoes
- Layers for temperature changes
- Light rain jacket
- Modest clothing for mosques"""

    # Shopping questions
    elif any(word in user_input_lower for word in ['shop', 'shopping', 'bazaar', 'market', 'buy']):
        return """🛍️ **Shopping in Istanbul**

**Traditional Markets:**
- **Grand Bazaar** (Kapalıçarşı) - 4,000 shops, carpets, jewelry
- **Spice Bazaar** - Turkish delight, spices, teas
- **Arasta Bazaar** - Near Blue Mosque, smaller crowds

**Modern Shopping:**
- **Istinye Park** - Luxury brands, European side
- **Kanyon** - Unique architecture in Levent
- **Zorlu Center** - High-end shopping in Beşiktaş

**What to Buy:**
- Turkish carpets & kilims
- Ceramic tiles and pottery
- Turkish delight & spices
- Leather goods
- Gold jewelry

**Bargaining Tips:**
- Expected in bazaars, not in modern stores
- Start at 30-50% of asking price
- Be polite and patient
- Compare prices at multiple shops"""

    # General recommendations
    elif any(word in user_input_lower for word in ['recommend', 'suggest', 'what to do', 'attractions', 'sights']):
        return """✨ **Top Istanbul Recommendations**

**Must-See Historic Sites:**
- Hagia Sophia - Byzantine masterpiece
- Blue Mosque - Ottoman architecture
- Topkapi Palace - Ottoman sultans' palace
- Basilica Cistern - Underground marvel

**Neighborhoods to Explore:**
- **Sultanahmet** - Historic peninsula
- **Beyoğlu** - Modern culture, nightlife
- **Galata** - Trendy area, great views
- **Kadıköy** - Asian side, local vibe

**Unique Experiences:**
- Bosphorus ferry cruise at sunset
- Turkish bath (hamam) experience
- Rooftop dining with city views
- Local food tour in Kadıköy

**Day Trip Ideas:**
- Princes' Islands (Büyükada)
- Büyükçekmece Lake
- Belgrade Forest hiking

Ask me about specific areas or activities for more detailed information!"""

    # Default response for unclear queries
    else:
        # Check if the input is very short or unclear
        if len(user_input.strip()) < 3 or not any(char.isalpha() for char in user_input):
            return "Sorry, I couldn't understand. Can you type again?"
        
        return f"""Sorry, I couldn't understand your request about "{user_input}". Can you type again?

I can help you with:

🍽️ **Restaurants** - "restaurants in Kadıköy" or "Turkish cuisine"
🏛️ **Museums & Attractions** - "museums in Istanbul" or "Hagia Sophia"
🏘️ **Districts** - "best neighborhoods" or "Sultanahmet area"
🚇 **Transportation** - "how to get around" or "metro system"
🛍️ **Shopping** - "Grand Bazaar" or "where to shop"
� **Nightlife** - "best bars" or "Beyoğlu nightlife"

Please ask me something more specific about Istanbul!"""

# Routers
app.include_router(museums.router)
app.include_router(restaurants.router)
app.include_router(places.router)

@app.get("/")
def root():
    return {"message": "Welcome to AIstanbul API"}

@app.post("/feedback")
async def receive_feedback(request: Request):
    """Endpoint to receive user feedback on AI responses"""
    try:
        feedback_data = await request.json()
        
        # Log feedback to console for observation
        print(f"\n📊 FEEDBACK RECEIVED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Type: {feedback_data.get('feedbackType', 'unknown')}")
        print(f"Query: {feedback_data.get('userQuery', 'N/A')}")
        print(f"Response: {feedback_data.get('messageText', '')[:100]}...")
        print(f"Session: {feedback_data.get('sessionId', 'N/A')}")
        print("-" * 50)
        
        # You could store this in a database here
        # For now, we just acknowledge receipt
        return {
            "status": "success",
            "message": "Feedback received",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error processing feedback: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/ai")
async def ai_istanbul_router(request: Request):
    data = await request.json()
    user_input = data.get("user_input", "")
    
    try:
        from openai import OpenAI
        import os
        import re
        from api_clients.google_places import search_restaurants
        from sqlalchemy.orm import Session
        
        # Debug logging
        print(f"Received user_input: '{user_input}' (length: {len(user_input)})")
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Create database session
        db = SessionLocal()
        
        try:
            # Check for very specific queries that need database/API data
            restaurant_keywords = [
                'restaurant', 'restaurants', 'restarunt', 'restarunts',  # Add basic words and common misspellings first
                'estrnt', 'resturant', 'restrant', 'restrnt',  # Common misspellings and abbreviations
                'restaurant recommendation', 'restaurant recommendations', 'recommend restaurants',
                'where to eat', 'best restaurants', 'good restaurants', 'top restaurants',
                'food places', 'places to eat', 'good places to eat', 'where can I eat',
                'turkish restaurants', 'local restaurants', 'traditional restaurants',
                'dining in istanbul', 'dinner recommendations', 'lunch places',
                'breakfast places', 'brunch spots', 'fine dining', 'casual dining',
                'cheap eats', 'budget restaurants', 'expensive restaurants', 'high-end restaurants',
                'seafood restaurants', 'kebab places', 'turkish cuisine', 'ottoman cuisine',
                'street food', 'local food', 'authentic food', 'traditional food',
                'rooftop restaurants', 'restaurants with view', 'bosphorus restaurants',
                'sultanahmet restaurants', 'beyoglu restaurants', 'galata restaurants',
                'taksim restaurants', 'kadikoy restaurants', 'besiktas restaurants',
                'asian side restaurants', 'european side restaurants',
                'vegetarian restaurants', 'vegan restaurants', 'halal restaurants',
                'restaurants near me', 'food recommendations', 'eating out',
                'where should I eat', 'suggest restaurants', 'restaurant suggestions'
            ]
            
            # Enhanced location-based restaurant detection
            location_restaurant_patterns = [
                r'restaurants?\s+in\s+\w+',  # "restaurants in taksim"
                r'restaurant\s+in\s+\w+',   # "restaurant in taksim"
                r'restarunts?\s+in\s+\w+',   # "restarunt in taksim" - common misspelling
                r'restarunt\s+in\s+\w+',    # "restarunt in taksim" - common misspelling
                r'resturant\s+in\s+\w+',    # Common misspelling
                r'restrnt\s+in\s+\w+',      # Abbreviated form
                r'estrnt\s+in\s+\w+',       # Abbreviated form
                r'restrant\s+in\s+\w+',     # Common misspelling
                r'restaurants?\s+near\s+\w+',  # "restaurants near galata"
                r'restaurants?\s+around\s+\w+',  # "restaurants around sultanahmet"
                r'eat\s+in\s+\w+',  # "eat in beyoglu"
                r'food\s+in\s+\w+',  # "food in kadikoy"
                r'dining\s+in\s+\w+',  # "dining in taksim"
                r'give\s+me\s+restaurants?\s+in\s+\w+',  # "give me restaurants in taksim"
                r'show\s+me\s+restaurants?\s+in\s+\w+'   # "show me restaurants in galata"
            ]
            museum_keywords = [
                'list museums', 'show museums', 'museum list', 'museums in istanbul',
                'art museum', 'history museum', 'archaeological museum', 'palace museum',
                'topkapi', 'hagia sophia', 'dolmabahce', 'istanbul modern',
                'pera museum', 'sakip sabanci', 'rahmi koc museum', 'museum recommendations',
                'which museums', 'best museums', 'must see museums', 'famous museums'
            ]
            district_keywords = [
                'list districts', 'show districts', 'district list', 'neighborhoods in istanbul',
                'sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar',
                'fatih', 'sisli', 'taksim', 'karakoy', 'ortakoy', 'bebek', 'arnavutkoy',
                'balat', 'fener', 'eminonu', 'bakirkoy', 'maltepe', 'asian side', 'european side',
                'neighborhoods', 'areas in istanbul', 'districts to visit', 'where to stay',
                'best neighborhoods', 'trendy areas', 'historic districts'
            ]
            attraction_keywords = [
                'list attractions', 'show attractions', 'attraction list', 'landmarks in istanbul',
                'tourist attractions', 'sightseeing', 'must see', 'top attractions',
                'blue mosque', 'galata tower', 'bosphorus', 'golden horn', 'maiden tower',
                'basilica cistern', 'grand bazaar', 'spice bazaar', 'princes islands',
                'istiklal street', 'pierre loti', 'camlica hill', 'rumeli fortress',
                'things to do', 'places to visit', 'famous places', 'landmarks'
            ]
            
            # Add new keyword categories for better query routing
            shopping_keywords = [
                'shopping', 'shop', 'buy', 'bazaar', 'market', 'mall', 'stores',
                'grand bazaar', 'spice bazaar', 'istinye park', 'kanyon', 'zorlu center',
                'cevahir', 'outlet', 'souvenir', 'carpet', 'jewelry', 'leather',
                'shopping centers', 'where to shop', 'shopping recommendations', 'best shopping'
            ]
            
            transportation_keywords = [
                'transport', 'transportation', 'metro', 'bus', 'ferry', 'taxi', 'uber',
                'how to get', 'getting around', 'public transport', 'istanbulkart',
                'airport', 'train', 'tram', 'dolmus', 'marmaray', 'metrobus',
                'getting from', 'how to reach', 'travel to', 'transport options'
            ]
            
            nightlife_keywords = [
                'nightlife', 'bars', 'clubs', 'night out', 'drinks', 'pub', 'lounge',
                'rooftop bar', 'live music', 'dancing', 'cocktails', 'beer', 'wine',
                'galata nightlife', 'beyoglu nightlife', 'taksim bars', 'karakoy bars',
                'where to drink', 'best bars', 'night clubs', 'party'
            ]
            
            culture_keywords = [
                'culture', 'cultural', 'tradition', 'festival', 'event', 'show',
                'turkish culture', 'ottoman', 'byzantine', 'hamam', 'turkish bath',
                'folk dance', 'music', 'art', 'theater', 'concert', 'whirling dervish',
                'cultural activities', 'traditional experiences', 'local customs'
            ]
            
            accommodation_keywords = [
                'hotel', 'accommodation', 'where to stay', 'hostel', 'apartment',
                'boutique hotel', 'luxury hotel', 'budget hotel', 'airbnb',
                'sultanahmet hotels', 'galata hotels', 'bosphorus view',
                'hotel recommendations', 'best hotels', 'cheap hotels'
            ]
            
            events_keywords = [
                'events', 'concerts', 'shows', 'exhibitions', 'festivals',
                'what\'s happening', 'events today', 'weekend events', 'cultural events',
                'music events', 'art exhibitions', 'theater shows', 'performances'
            ]
            
            # Add regex patterns for location-based restaurant queries
            location_restaurant_patterns = [
                r'restaurants?\s+in\s+\w+',  # "restaurants in taksim"
                r'restaurants?\s+near\s+\w+',  # "restaurants near galata"
                r'restaurants?\s+at\s+\w+',  # "restaurants at sultanahmet"
                r'restaurants?\s+around\s+\w+',  # "restaurants around beyoglu"
                r'food\s+in\s+\w+',  # "food in kadikoy"
                r'eat\s+in\s+\w+',  # "eat in taksim"
                r'dining\s+in\s+\w+',  # "dining in galata"
                r'\w+\s+restaurants',  # "taksim restaurants", "galata restaurants"
                r'where\s+to\s+eat\s+in\s+\w+',  # "where to eat in beyoglu"
            ]
            
            # Add regex patterns for location-based place queries
            location_place_patterns = [
                r'place\s+in\s+\w+',  # "place in kadikoy"
                r'places\s+in\s+\w+',  # "places in sultanahmet"
                r'attractions?\s+in\s+\w+',  # "attractions in beyoglu"
                r'things?\s+to\s+do\s+in\s+\w+',  # "things to do in galata"
                r'visit\s+in\s+\w+',  # "visit in taksim"
                r'see\s+in\s+\w+',  # "see in kadikoy"
                r'go\s+in\s+\w+',  # "go in fatih"
                r'\w+\s+attractions',  # "kadikoy attractions"
                r'what.*in\s+\w+',  # "what to do in beyoglu"
            ]
            
            # Check if query matches location-based patterns
            is_location_restaurant_query = any(re.search(pattern, user_input.lower()) for pattern in location_restaurant_patterns)
            is_location_place_query = any(re.search(pattern, user_input.lower()) for pattern in location_place_patterns)
            
            # More specific matching for different query types - prioritize restaurant queries
            is_restaurant_query = any(keyword in user_input.lower() for keyword in restaurant_keywords) or is_location_restaurant_query
            
            # Only consider it a district query if it's NOT a restaurant query and NOT a location-based query
            is_district_query = (any(keyword in user_input.lower() for keyword in district_keywords) and 
                               not is_restaurant_query and 
                               not is_location_restaurant_query and 
                               not is_location_place_query)
            
            is_museum_query = any(keyword in user_input.lower() for keyword in museum_keywords)
            is_attraction_query = any(keyword in user_input.lower() for keyword in attraction_keywords) or is_location_place_query
            is_shopping_query = any(keyword in user_input.lower() for keyword in shopping_keywords)
            is_transportation_query = any(keyword in user_input.lower() for keyword in transportation_keywords)
            is_nightlife_query = any(keyword in user_input.lower() for keyword in nightlife_keywords)
            is_culture_query = any(keyword in user_input.lower() for keyword in culture_keywords)
            is_accommodation_query = any(keyword in user_input.lower() for keyword in accommodation_keywords)
            is_events_query = any(keyword in user_input.lower() for keyword in events_keywords)
            
            # Debug query categorization
            print(f"Query categorization:")
            print(f"  is_restaurant_query: {is_restaurant_query}")
            print(f"  is_museum_query: {is_museum_query}")
            print(f"  is_district_query: {is_district_query}")
            print(f"  is_attraction_query: {is_attraction_query}")
            print(f"  is_location_place_query: {is_location_place_query}")
            
            if is_restaurant_query:
                # Get real restaurant data from Google Maps only
                try:
                    # Extract location from query for better results
                    search_location = "Istanbul, Turkey"
                    if is_location_restaurant_query:
                        # Try to extract specific location from the query
                        location_patterns = [
                            r'in\s+([a-zA-Z\s]+)',
                            r'near\s+([a-zA-Z\s]+)',
                            r'at\s+([a-zA-Z\s]+)',
                            r'around\s+([a-zA-Z\s]+)',
                        ]
                        for pattern in location_patterns:
                            match = re.search(pattern, user_input.lower())
                            if match:
                                location = match.group(1).strip()
                                search_location = f"{location}, Istanbul, Turkey"
                                break
                    
                    places_data = search_restaurants(search_location, user_input)
                    
                    if places_data.get('results'):
                        restaurants_info = f"### 🍽️ Here are restaurants I found{' in ' + search_location.split(',')[0] if 'Istanbul' not in search_location else ' in Istanbul'}:\n\n"
                        for i, place in enumerate(places_data['results'][:5]):  # Top 5 results
                            name = place.get('name', 'Unknown')
                            rating = place.get('rating', 'N/A')
                            price_level = place.get('price_level', 'N/A')
                            place_id = place.get('place_id', '')
                            maps_link = f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else "N/A"
                            # Convert price_level to emoji
                            if price_level != 'N/A' and isinstance(price_level, int):
                                price_emoji = '💲' * price_level
                            elif price_level != 'N/A' and str(price_level).isdigit():
                                price_emoji = '💲' * int(price_level)
                            else:
                                price_emoji = ''
                            restaurants_info += f"{i+1}. **{name}**  \n"
                            restaurants_info += f"   ⭐ {rating}/5\t{price_emoji}  \n"
                            restaurants_info += f"   [View on Google Maps]({maps_link})\n\n"
                        return {"message": restaurants_info}
                    else:
                        return {"message": "Sorry, I couldn't find any restaurants matching your request in Istanbul."}
                        
                except Exception as e:
                    print(f"Error fetching Google Places data: {e}")
                    return {"message": "Sorry, I encountered an error while searching for restaurants. Please try again."}
            
            elif is_museum_query or is_attraction_query or is_district_query:
                # Get data from manual database
                places = db.query(Place).all()
                
                # Extract location if this is a location-specific query
                extracted_location = None
                if is_location_place_query:
                    print(f"Location-based place query detected: {user_input}")
                    location_patterns = [
                        r'in\s+([a-zA-Z\s]+)',
                        r'at\s+([a-zA-Z\s]+)',
                        r'around\s+([a-zA-Z\s]+)',
                    ]
                    for pattern in location_patterns:
                        match = re.search(pattern, user_input.lower())
                        if match:
                            extracted_location = match.group(1).strip()
                            print(f"Extracted location: '{extracted_location}'")
                            break
                
                # Filter based on query type
                filtered_places = []
                if is_location_place_query and extracted_location:
                    # For location-specific queries like "place in kadikoy", include all places first
                    print(f"DEBUG: Using location-based filtering for '{extracted_location}'")
                    filtered_places = places
                elif is_museum_query:
                    filtered_places = [p for p in places if p.category and 'museum' in p.category.lower()]
                elif is_district_query:
                    filtered_places = [p for p in places if p.category and 'district' in p.category.lower()]
                    # Also include places by district
                    if not filtered_places:
                        districts = set([p.district for p in places if p.district])
                        district_info = f"Here are the main districts in Istanbul:\n\n"
                        for district in sorted(districts):
                            places_in_district = [p for p in places if p.district == district]
                            district_info += f"**{district}**:\n"
                            for place in places_in_district[:3]:  # Top 3 places per district
                                district_info += f"   - {place.name} ({place.category})\n"
                            district_info += "\n"
                        return {"message": district_info}
                else:  # general attraction query
                    filtered_places = [p for p in places if p.category and p.category.lower() in ['historical place', 'mosque', 'church', 'park']]
                
                # Apply location filter if location was extracted
                if extracted_location and filtered_places:
                    print(f"DEBUG: Before location filter: {len(filtered_places)} places")
                    print(f"DEBUG: Looking for location: '{extracted_location.lower()}'")
                    print(f"DEBUG: Available districts: {[p.district for p in filtered_places[:5]]}")
                    
                    # Normalize location name for matching (case-insensitive)
                    location_lower = extracted_location.lower()
                    original_count = len(filtered_places)
                    filtered_places = [p for p in filtered_places if p.district and location_lower in p.district.lower()]
                    print(f"DEBUG: After location filter: {len(filtered_places)} places (from {original_count})")
                    
                    if filtered_places:
                        print(f"DEBUG: Filtered places found:")
                        for p in filtered_places:
                            print(f"  - {p.name} in {p.district}")
                    else:
                        print(f"DEBUG: No places found matching location '{extracted_location}'")
                
                if filtered_places:
                    location_text = f" in {extracted_location.title()}" if extracted_location else " in Istanbul"
                    places_info = f"Here are the {'museums' if is_museum_query else 'places'}{location_text}:\n\n"
                    for i, place in enumerate(filtered_places[:8]):  # Top 8 results
                        places_info += f"{i+1}. **{place.name}**\n"
                        places_info += f"   - Category: {place.category}\n"
                        places_info += f"   - District: {place.district}\n\n"
                    return {"message": places_info}
                else:
                    if extracted_location:
                        return {"message": f"Sorry, I couldn't find any {'museums' if is_museum_query else 'places'} in {extracted_location.title()} in my database. Try asking about a different district or general attractions in Istanbul."}
                    else:
                        return {"message": f"Sorry, I couldn't find any {'museums' if is_museum_query else 'attractions'} in my database."}
            
            elif is_shopping_query:
                shopping_response = """🛍️ **Shopping in Istanbul**

**Traditional Markets:**
- **Grand Bazaar** (Kapalıçarşı) - 4,000 shops, carpets, jewelry, spices
  - Metro: Beyazıt-Kapalıçarşı (M1) or Vezneciler (M2)
  - Hours: 9 AM - 7 PM (closed Sundays)
  
- **Spice Bazaar** (Mısır Çarşısı) - Turkish delight, spices, teas, nuts
  - Location: Near Eminönü ferry terminal
  - Great for food souvenirs and gifts

**Modern Shopping Centers:**
- **Istinye Park** - Luxury brands, beautiful architecture (European side)
- **Kanyon** - Unique design in Levent business district
- **Zorlu Center** - High-end shopping in Beşiktaş
- **Mall of Istanbul** - Large shopping center on European side
- **Palladium** - Popular mall in Ataşehir (Asian side)

**What to Buy:**
- Turkish carpets & kilims
- Ceramic tiles and pottery
- Evil eye (nazar) jewelry & charms
- Turkish delight & baklava
- Turkish tea & coffee
- Leather goods & shoes
- Traditional textiles
- Handmade soaps

**Shopping Tips:**
- Bargaining is expected in bazaars (start at 30-50% of asking price)
- Fixed prices in modern malls
- Many shops close on Sundays
- Ask for tax-free shopping receipts for purchases over 108 TL"""
                return {"message": shopping_response}
            
            elif is_transportation_query:
                transport_response = """🚇 **Getting Around Istanbul**

**Istanbul Card (Istanbulkart):** 💳
- Essential for ALL public transport
- Buy at metro stations, airports, or kiosks
- Works on metro, bus, tram, ferry, funicular
- Significant discounts vs. single tickets

**Metro Lines:**
- **M1A/M1B**: Airport ↔ Yenikapı ↔ Kirazlı
- **M2**: Vezneciler ↔ Şişli ↔ Hacıosman
- **M3**: Kirazlı ↔ Başakşehir
- **M4**: Kadıköy ↔ Sabiha Gökçen Airport
- **M5**: Üsküdar ↔ Çekmeköy
- **M6**: Levent ↔ Boğaziçi Üniversitesi
- **M7**: Mecidiyeköy ↔ Mahmutbey

**Trams:**
- **T1**: Kabataş ↔ Bağcılar (passes through Sultanahmet)
- **T4**: Topkapı ↔ Mescid-i Selam

**Key Ferry Routes:**
- Eminönü ↔ Kadıköy (20 min, scenic)
- Karaköy ↔ Kadıköy (15 min)
- Beşiktaş ↔ Üsküdar (15 min)
- Bosphorus tours from Eminönü

**Airports:**
- **Istanbul Airport (IST)**: M11 metro to city center
- **Sabiha Gökçen (SAW)**: M4 metro or HAVABUS

**Apps to Download:**
- Moovit - Real-time public transport
- BiTaksi - Local taxi app
- Uber - Available in Istanbul

**Tips:**
- Rush hours: 8-10 AM, 5-7 PM
- Metro announcements in Turkish & English
- Keep your Istanbulkart with you always!"""
                return {"message": transport_response}
            
            elif is_nightlife_query:
                nightlife_response = """🌃 **Istanbul Nightlife**

**Trendy Neighborhoods:**

**Beyoğlu/Galata:**
- Heart of Istanbul's nightlife
- Mix of rooftop bars, clubs, and pubs
- Istiklal Street has many options

**Karaköy:**
- Hip, artistic area with craft cocktail bars
- Great Bosphorus views from rooftop venues
- More upscale crowd

**Beşiktaş:**
- University area with younger crowd
- Good mix of bars and clubs
- More affordable options

**Popular Venues:**
- **360 Istanbul** - Famous rooftop with city views
- **Mikla** - Upscale rooftop restaurant/bar
- **Kloster** - Historic building, great atmosphere
- **Under** - Underground club in Karaköy
- **Sortie** - Upscale club in Maçka
- **Reina** - Famous Bosphorus-side nightclub

**Rooftop Bars:**
- **Leb-i Derya** - Multiple locations, great views
- **Nu Teras** - Sophisticated rooftop in Beyoğlu
- **Banyan** - Asian-inspired rooftop bar
- **The Marmara Pera** - Hotel rooftop with panoramic views

**Tips:**
- Most venues open after 9 PM
- Dress code: Smart casual to upscale
- Credit cards widely accepted
- Many venues have entrance fees on weekends
- Turkish beer (Efes) and rakı are popular local drinks
- Some areas can be crowded on weekends"""
                return {"message": nightlife_response}
            
            elif is_culture_query:
                culture_response = """🎭 **Turkish Culture & Experiences**

**Traditional Experiences:**
- **Turkish Bath (Hamam)** - Historic Cagaloglu or Suleymaniye Hamams
- **Whirling Dervishes** - Sema ceremony at various cultural centers
- **Turkish Coffee** - UNESCO Intangible Cultural Heritage
- **Traditional Music** - Turkish folk or Ottoman classical music

**Cultural Venues:**
- **Hodjapasha Cultural Center** - Traditional shows & performances
- **Galata Mevlevihanesi** - Whirling dervish ceremonies
- **Cemal Reşit Rey Concert Hall** - Classical music & opera
- **Zorlu PSM** - Modern performing arts center

**Festivals & Events:**
- **Istanbul Music Festival** (June) - Classical music in historic venues
- **Istanbul Biennial** (Fall, odd years) - Contemporary art
- **Ramadan** - Special atmosphere, iftar meals at sunset
- **Turkish National Days** - Republic Day (Oct 29), Victory Day (Aug 30)

**Cultural Customs:**
- Remove shoes when entering mosques or homes
- Dress modestly in religious sites
- Greetings: Handshakes common, kisses on both cheeks for friends
- Hospitality is very important in Turkish culture
- Tea (çay) is offered as sign of hospitality

**Traditional Arts:**
- **Calligraphy** - Ottoman Turkish writing art
- **Miniature Painting** - Traditional Ottoman art form
- **Carpet Weaving** - Intricate traditional patterns
- **Ceramic Art** - Iznik tiles and pottery
- **Marbled Paper (Ebru)** - Water-based art technique

**Cultural Districts:**
- **Balat** - Historic Jewish quarter with colorful houses
- **Fener** - Greek Orthodox heritage area
- **Süleymaniye** - Traditional Ottoman neighborhood
- **Eyüp** - Religious significance, local life"""
                return {"message": culture_response}
            
            elif is_accommodation_query:
                accommodation_response = """🏨 **Where to Stay in Istanbul**

**Best Neighborhoods:**

**Sultanahmet (Historic Peninsula):**
- Walk to major attractions (Blue Mosque, Hagia Sophia)
- Traditional Ottoman hotels and boutique properties
- Great for first-time visitors
- Can be touristy and crowded

**Beyoğlu/Galata:**
- Trendy area with modern boutique hotels
- Easy access to nightlife, restaurants, art galleries
- Good transport connections
- More contemporary vibe

**Beşiktaş:**
- Business district with luxury hotels
- Near Dolmabahçe Palace and Bosphorus
- Excellent shopping and dining
- Great transport hub

**Kadıköy (Asian Side):**
- Authentic local experience
- Great food scene and markets
- Less touristy, more affordable
- Easy ferry connection to European side

**Accommodation Types:**

**Luxury Hotels:**
- **Four Seasons Sultanahmet** - Ottoman palace conversion
- **Çırağan Palace Kempinski** - Former Ottoman palace
- **The Ritz-Carlton Istanbul** - Modern luxury in Şişli
- **Shangri-La Bosphorus** - Asian side luxury

**Boutique Hotels:**
- **Museum Hotel** - Antique-filled historic property
- **Vault Karakoy** - Former bank building in Galata
- **Georges Hotel Galata** - Stylish property near Galata Tower
- **Soho House Istanbul** - Trendy members' club with rooms

**Budget Options:**
- **Cheers Hostel** - Well-reviewed hostel in Sultanahmet
- **Marmara Guesthouse** - Budget-friendly in old city
- **Istanbul Hostel** - Central location, clean facilities

**Booking Tips:**
- Book early for peak season (April-October)
- Many hotels offer airport transfer services
- Check if breakfast is included
- Rooftop terraces are common and worth requesting
- Some historic hotels have small rooms - check dimensions"""
                return {"message": accommodation_response}
            
            elif is_events_query:
                events_response = """🎪 **Events & Entertainment in Istanbul**

**Regular Cultural Events:**

**Weekly:**
- **Friday Prayer** - Beautiful call to prayer across the city
- **Weekend Markets** - Kadıköy Saturday Market, various neighborhood pazars

**Monthly/Seasonal:**
- **Whirling Dervish Ceremonies** - Various venues, usually weekends
- **Traditional Music Concerts** - Cultural centers and historic venues
- **Art Gallery Openings** - Especially in Beyoğlu and Karaköy

**Major Annual Festivals:**

**Spring:**
- **Tulip Festival** (April) - Millions of tulips bloom in city parks
- **Istanbul Music Festival** (June) - Classical music in historic venues

**Summer:**
- **Istanbul Jazz Festival** (July) - International artists, multiple venues
- **Rock'n Coke Festival** - Major international music festival

**Fall:**
- **Istanbul Biennial** (Sept-Nov, odd years) - Contemporary art across the city
- **Akbank Jazz Festival** (Oct) - Jazz performances citywide

**Winter:**
- **New Year Celebrations** - Taksim Square and various venues

**Entertainment Venues:**

**Concert Halls:**
- **Zorlu PSM** - Major international shows
- **Cemal Reşit Rey Concert Hall** - Classical music
- **Volkswagen Arena** - Large concerts and sports

**Theaters:**
- **Istanbul State Opera and Ballet**
- **Turkish State Theaters**
- Various private theaters in Beyoğlu

**Current Events Resources:**
- **Biletix.com** - Main ticketing platform
- **Time Out Istanbul** - Event listings
- **Istanbul.com** - Tourism events
- **Eventbrite** - International events

**Tips:**
- Check event calendars at hotel concierge
- Many cultural events are free or low-cost
- Book popular shows in advance
- Some events may be in Turkish only - check beforehand"""
                return {"message": events_response}
            
            else:
                # Use OpenAI for intelligent responses about Istanbul
                # Get context from database
                places = db.query(Place).all()
                restaurants_context = "Available restaurants data from Google Maps API."
                places_context = ""
                
                if places:
                    places_context = "Available places in Istanbul database:\n"
                    for place in places[:20]:  # Limit context size
                        places_context += f"- {place.name} ({place.category}) in {place.district}\n"
                
                # Create prompt for OpenAI
                system_prompt = """You are an expert Istanbul travel guide AI assistant. You have access to:
1. Real-time restaurant data from Google Maps API
2. A database of museums, attractions, and districts in Istanbul

Provide helpful, accurate, and engaging responses about Istanbul tourism, culture, history, food, and attractions. 
Be conversational and enthusiastic about Istanbul. Use emojis appropriately.

When users ask about restaurants, suggest they specify what type of cuisine or area they're interested in.
When users ask about attractions, museums, or districts, use your knowledge of Istanbul combined with the provided database."""

                try:
                    print("Making OpenAI API call...")
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "system", "content": f"Database context:\n{places_context}"},
                            {"role": "user", "content": user_input}
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    ai_response = response.choices[0].message.content
                    print(f"OpenAI response: {ai_response[:100]}...")
                    return {"message": ai_response}
                    
                except Exception as e:
                    print(f"OpenAI API error: {e}")
                    # Smart fallback response based on user input
                    return {"message": create_fallback_response(user_input, places)}
        
        finally:
            db.close()
            
    except Exception as e:
        print(f"Error in AI endpoint: {e}")
        return {"message": "Sorry, I couldn't understand. Can you type again?"}
        
    except Exception as e:
        return {"error": "Internal server error", "details": str(e)}

async def stream_response(message: str):
    """Stream response word by word like ChatGPT"""
    words = message.split(' ')
    
    for i, word in enumerate(words):
        chunk = {
            "delta": {"content": word + (" " if i < len(words) - 1 else "")},
            "finish_reason": None
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.1)  # ChatGPT-like delay between words
    
    # Send final chunk
    final_chunk = {
        "delta": {"content": ""},
        "finish_reason": "stop"
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/ai/stream")
async def ai_istanbul_stream(request: Request):
    """Streaming version of the AI endpoint for ChatGPT-like responses"""
    data = await request.json()
    user_input = data.get("user_input", "")
    speed = data.get("speed", 1.0)  # Speed multiplier: 1.0 = normal, 0.5 = slower, 2.0 = faster
    
    try:
        from openai import OpenAI
        from api_clients.google_places import search_restaurants
        from sqlalchemy.orm import Session
        print(f"Received streaming user_input: '{user_input}' (length: {len(user_input)}) at speed: {speed}x")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        db = SessionLocal()
        
        try:
            # Use fallback response for now
            places = []
            message = create_fallback_response(user_input, places)
            print(f"Generated message length: {len(message)}")
        finally:
            db.close()
            
        return StreamingResponse(stream_response(message), media_type="text/plain")
    except Exception as e:
        print(f"Error in streaming AI endpoint: {e}")
        error_message = "Sorry, I encountered an error. Please try again."
        return StreamingResponse(stream_response(error_message), media_type="text/plain")

# Remove any duplicate or broken code fragments below this point





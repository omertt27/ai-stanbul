# --- Standard Library Imports ---
import sys
import os
import re
import asyncio
import json
import time
from datetime import datetime

# --- Third-Party Imports ---
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process

# --- Project Imports ---
from database import engine, SessionLocal
from models import Base, Restaurant, Museum, Place
from routes import museums, restaurants, places
from api_clients.google_places import GooglePlacesClient
from sqlalchemy.orm import Session

load_dotenv()

# --- OpenAI Import ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    print("[ERROR] openai package not installed. Please install it with 'pip install openai'.")

# Add the current directory to Python path for Render deployment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# (Removed duplicate project imports and load_dotenv)

def clean_text_formatting(text):
    """Remove emojis, hashtags, and markdown formatting from text while preserving line breaks"""
    if not text:
        return text
    
    # Remove emojis (Unicode emoji ranges)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # dingbats
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        u"\U00002700-\U000027BF"  # dingbats
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
    text = re.sub(r'#+ ', '', text)               # Remove hashtags at start of lines
    text = re.sub(r' #\w+', '', text)             # Remove hashtags in text
    
    # Clean up extra whitespace but preserve line breaks
    lines = text.split('\n')
    cleaned_lines = [' '.join(line.split()) for line in lines]
    text = '\n'.join(cleaned_lines)
    
    return text.strip()

def generate_restaurant_info(restaurant_name, location="Istanbul"):
    """Generate a brief, plain-text description for a restaurant based on its name"""
    name_lower = restaurant_name.lower()
    
    # Common Turkish restaurant types and food indicators
    if any(word in name_lower for word in ['kebap', 'kebab', 'çiğ köfte', 'döner', 'dürüm']):
        return "A popular kebab restaurant serving traditional Turkish grilled meats and specialties."
    elif any(word in name_lower for word in ['pizza', 'pizzeria', 'italian']):
        return "An Italian restaurant specializing in authentic pizzas and Mediterranean cuisine."
    elif any(word in name_lower for word in ['sushi', 'japanese', 'asian']):
        return "A Japanese restaurant offering fresh sushi and traditional Asian dishes."
    elif any(word in name_lower for word in ['burger', 'american', 'grill']):
        return "A casual dining spot known for burgers, grilled foods, and American-style cuisine."
    elif any(word in name_lower for word in ['cafe', 'kahve', 'coffee']):
        return "A cozy cafe perfect for coffee, light meals, and a relaxed atmosphere."
    elif any(word in name_lower for word in ['balık', 'fish', 'seafood', 'deniz']):
        return "A seafood restaurant featuring fresh fish and Mediterranean coastal cuisine."
    elif any(word in name_lower for word in ['ev yemeği', 'lokanta', 'traditional']):
        return "A traditional Turkish restaurant serving home-style cooking and local specialties."
    elif any(word in name_lower for word in ['meze', 'rakı', 'taverna']):
        return "A traditional meze restaurant offering small plates and Turkish appetizers."
    elif any(word in name_lower for word in ['pastane', 'tatlı', 'dessert', 'bakery']):
        return "A bakery and dessert shop known for Turkish sweets and fresh pastries."
    elif any(word in name_lower for word in ['steakhouse', 'et', 'meat']):
        return "A steakhouse specializing in premium cuts of meat and grilled dishes."
    else:
        return "A well-regarded restaurant offering quality dining and local cuisine."

app = FastAPI(title="AIstanbul API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
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

def create_fuzzy_keywords():
    """Create a comprehensive list of keywords for fuzzy matching"""
    keywords = {
        # Location names and variations
        'locations': [
            'kadikoy', 'kadıköy', 'sultanahmet', 'beyoglu', 'beyoğlu', 'galata', 
            'taksim', 'besiktas', 'beşiktaş', 'uskudar', 'üsküdar', 'fatih', 
            'sisli', 'şişli', 'karakoy', 'karaköy', 'ortakoy', 'ortaköy', 
            'bebek', 'arnavutkoy', 'arnavutköy', 'balat', 'fener', 'eminonu', 
            'eminönü', 'bakirkoy', 'bakırköy', 'maltepe', 'istanbul', 'instanbul'
        ],
        # Query types and variations
        'places': [
            'places', 'place', 'plases', 'plases', 'plase', 'spots', 'locations', 'areas'
        ],
        'restaurants': [
            'restaurants', 'restaurant', 'restourant', 'resturant', 'food', 
            'eat', 'dining', 'eatery', 'cafe', 'cafes'
        ],
        'attractions': [
            'attractions', 'attraction', 'atraction', 'sights', 'sites', 
            'tourist', 'visit', 'see', 'things to do', 'activities'
        ],
        'museums': [
            'museums', 'museum', 'musem', 'gallery', 'galleries', 'art', 
            'culture', 'cultural', 'history', 'historical'
        ],
        'nightlife': [
            'nightlife', 'night', 'bars', 'bar', 'clubs', 'club', 'party', 
            'drinks', 'entertainment'
        ],
        'shopping': [
            'shopping', 'shop', 'shops', 'market', 'markets', 'bazaar', 
            'bazaars', 'mall', 'malls', 'store', 'stores'
        ],
        'transport': [
            'transport', 'transportation', 'metro', 'bus', 'taxi', 'travel', 
            'getting around', 'how to get'
        ]
    }
    return keywords

def correct_typos(text, threshold=80):
    """Correct typos in user input using fuzzy matching"""
    try:
        keywords = create_fuzzy_keywords()
        words = text.lower().split()
        corrected_words = []
        
        # Common words that should not be corrected
        stop_words = {'in', 'to', 'at', 'on', 'for', 'with', 'by', 'from', 'up', 
                     'about', 'into', 'through', 'during', 'before', 'after', 
                     'above', 'below', 'between', 'among', 'a', 'an', 'the', 
                     'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'what', 'where', 'when', 'how', 'why', 'which', 'who', 'whom'}
        
        for word in words:
            # Skip common stop words
            if word.lower() in stop_words:
                corrected_words.append(word)
                continue
                
            best_match = None
            best_score = 0
            best_category = None
            
            # Check each category of keywords
            for category, keyword_list in keywords.items():
                match = process.extractOne(word, keyword_list)
                if match and match[1] > best_score and match[1] >= threshold:
                    best_match = match[0]
                    best_score = match[1]
                    best_category = category
            
            if best_match and best_score >= threshold:
                corrected_words.append(best_match)
                print(f"Typo correction: '{word}' -> '{best_match}' (score: {best_score}, category: {best_category})")
            else:
                corrected_words.append(word)
        
        corrected_text = ' '.join(corrected_words)
        return corrected_text
    except Exception as e:
        print(f"Error in typo correction: {e}")
        return text

def enhance_query_understanding(user_input):
    """Enhance query understanding by correcting typos and adding context"""
    try:
        # First correct typos
        corrected_input = correct_typos(user_input)
        
        # Add common query pattern recognition
        enhanced_input = corrected_input.lower()
        
        # Handle common patterns and add missing words
        patterns = [
            # "kadikoy restaurant" -> "restaurants in kadikoy"
            (r'^(\w+)\s+(restaurant|food|eat)$', r'restaurants in \1'),
            # "kadikoy place" -> "places in kadikoy"
            (r'^(\w+)\s+(place|spot)$', r'places in \1'),
            # "kadikoy attraction" -> "attractions in kadikoy"
            (r'^(\w+)\s+(attraction|sight)$', r'attractions in \1'),
            # "kadikoy museum" -> "museums in kadikoy"
            (r'^(\w+)\s+(museum|gallery)$', r'museums in \1'),
            # Handle "to visit" patterns - be more specific to avoid wrong enhancements
            (r'^(\w+)\s+to\s+visit$', r'places to visit in \1'),
            # Handle "what in" patterns
            (r'^what\s+(\w+)$', r'what to do in \1'),
        ]
        
        for pattern, replacement in patterns:
            match = re.search(pattern, enhanced_input)
            if match:
                enhanced_input = re.sub(pattern, replacement, enhanced_input)
                print(f"Query enhancement: '{corrected_input}' -> '{enhanced_input}'")
                break
        
        return enhanced_input if enhanced_input != corrected_input.lower() else corrected_input
        
    except Exception as e:
        print(f"Error in query enhancement: {e}")
        return user_input

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
    user_input = data.get("query", data.get("user_input", ""))  # Support both query and user_input

    # Handle greetings and daily talk BEFORE any typo correction or enhancement
    user_input_clean = user_input.lower().strip()
    greeting_patterns = [
        'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon',
        'good evening', 'howdy', 'hiya', 'sup', "what's up", 'whats up',
        'how are you', 'how are you doing', "how's it going", 'hows it going',
        'nice to meet you', 'pleased to meet you', 'good to see you'
    ]
    daily_talk_patterns = [
        'how are things', "what's new", 'whats new', 'how have you been',
        'long time no see', 'good to hear from you', "hope you're well",
        "how's your day", 'hows your day', 'having a good day',
        "what's happening", 'whats happening', "how's life", 'hows life'
    ]
    is_greeting = any(pattern in user_input_clean for pattern in greeting_patterns)
    is_daily_talk = any(pattern in user_input_clean for pattern in daily_talk_patterns)
    if is_greeting or is_daily_talk:
        print(f"[AIstanbul] Detected greeting/daily talk: {user_input}")
        if any(word in user_input_clean for word in ['hi', 'hello', 'hey', 'greetings', 'howdy', 'hiya']):
            return {"message": "Hello there! 👋 I'm your friendly Istanbul travel guide. I'm here to help you discover amazing places, restaurants, attractions, and hidden gems in this beautiful city. What would you like to explore in Istanbul today?"}
        elif 'how are you' in user_input_clean or 'how are you doing' in user_input_clean:
            return {"message": "I'm doing great, thank you for asking! 😊 I'm excited to help you explore Istanbul. There's so much to discover in this incredible city - from historic sites like Hagia Sophia to delicious food in Kadıköy. What interests you most?"}
        elif any(phrase in user_input_clean for phrase in ['good morning', 'good afternoon', 'good evening']):
            return {"message": "Good day to you too! ☀️ What a perfect time to plan your Istanbul adventure. Whether you're looking for restaurants, museums, or unique neighborhoods to explore, I'm here to help. What catches your interest?"}
        elif any(phrase in user_input_clean for phrase in ["what's up", 'whats up', 'sup', "how's it going", 'hows it going']):
            return {"message": "Not much, just here ready to help you discover Istanbul! 🌟 This city has incredible energy - from the bustling Grand Bazaar to peaceful Bosphorus views. What would you like to know about?"}
        else:
            return {"message": "It's so nice to chat with you! 😊 I love helping people discover Istanbul's wonders. From traditional Turkish cuisine to stunning architecture, there's something for everyone here. What aspect of Istanbul interests you most?"}
    
    try:
        # Debug logging
        print(f"Original user_input: '{user_input}' (length: {len(user_input)})")
        
        # Correct typos and enhance query understanding
        enhanced_user_input = enhance_query_understanding(user_input)
        print(f"Enhanced user_input: '{enhanced_user_input}'")
        
        # Use enhanced input for processing
        user_input = enhanced_user_input
        print(f"Received user_input: '{user_input}' (length: {len(user_input)})")

        # --- OpenAI API Key Check ---
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not OpenAI or not openai_api_key:
            print("[ERROR] OpenAI API key not set or openai package missing.")
            raise RuntimeError("OpenAI API key not set or openai package missing.")
        client = OpenAI(api_key=openai_api_key)

        # Create database session
        db = SessionLocal()
        try:
            # Check for very specific queries that need database/API data
            restaurant_keywords = [
                'restaurant', 'restaurants', 'restourant', 'restourants',  # Include corrected typos
                'restarunt', 'restarunts',  # Add basic words and common misspellings first
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
            
            # Add regex patterns for location-based museum queries
            location_museum_patterns = [
                r'museums?\s+in\s+\w+',      # "museums in beyoglu"
                r'museum\s+in\s+\w+',       # "museum in taksim"  
                r'give\s+me\s+museums?\s+in\s+\w+',  # "give me museums in beyoglu"
                r'show\s+me\s+museums?\s+in\s+\w+',  # "show me museums in galata"
                r'museums?\s+near\s+\w+',    # "museums near galata"
                r'museums?\s+around\s+\w+',  # "museums around sultanahmet"
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
            
            # --- Remove duplicate location_restaurant_patterns ---
            # (Already defined above, so do not redefine here)
            
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
                r'\w+\s+places?\s+to\s+visit',  # "kadikoy places to visit"
                r'\w+\s+to\s+places?\s+to\s+visit',  # "kadikoy to places to visit" - double "to" pattern
                r'\w+\s+to\s+visit',  # "kadikoy to visit"
                r'places?\s+to\s+visit\s+in\s+\w+',  # "places to visit in kadikoy"
                r'visit\s+\w+\s+places?',  # "visit kadikoy places"
            ]
            
            # Check if query matches location-based patterns
            is_location_restaurant_query = any(re.search(pattern, user_input.lower()) for pattern in location_restaurant_patterns)
            is_location_place_query = any(re.search(pattern, user_input.lower()) for pattern in location_place_patterns)
            is_location_museum_query = any(re.search(pattern, user_input.lower()) for pattern in location_museum_patterns)
            
            # Check if query is just a single district name (should show places in that district)
            single_district_names = ['sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar',
                                   'fatih', 'sisli', 'taksim', 'karakoy', 'ortakoy', 'bebek', 'arnavutkoy',
                                   'balat', 'fener', 'eminonu', 'bakirkoy', 'maltepe']
            is_single_district_query = (user_input.lower().strip() in single_district_names)
            
            # More specific matching for different query types - prioritize restaurant and museum queries
            is_restaurant_query = any(keyword in user_input.lower() for keyword in restaurant_keywords) or is_location_restaurant_query
            is_museum_query = any(keyword in user_input.lower() for keyword in museum_keywords) or is_location_museum_query
            
            # Only consider it a district query if it's NOT a restaurant, museum, or location-based query and NOT a single district name
            is_district_query = (any(keyword in user_input.lower() for keyword in district_keywords) and 
                               not is_restaurant_query and 
                               not is_museum_query and
                               not is_location_restaurant_query and 
                               not is_location_place_query and
                               not is_location_museum_query and
                               not is_single_district_query)
            
            is_attraction_query = (any(keyword in user_input.lower() for keyword in attraction_keywords) or 
                                 is_location_place_query or is_single_district_query)
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
            print(f"  is_location_museum_query: {is_location_museum_query}")
            print(f"  is_single_district_query: {is_single_district_query}")
            
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
                    
                    client = GooglePlacesClient()
                    places_data = client.search_restaurants(location=search_location, keyword=user_input)
                    
                    if places_data.get('results'):
                        location_text = user_input.lower().split("in ")[-1].strip().title() if "in " in user_input.lower() else "Istanbul"
                        restaurants_info = f"Here are some great restaurants in {location_text}:\n\n"
                        
                        for i, place in enumerate(places_data['results'][:5]):  # Top 5 results
                            name = place.get('name', 'Unknown')
                            rating = place.get('rating', 'N/A')
                            price_level = place.get('price_level', 'N/A')
                            address = place.get('formatted_address', '')
                            
                            # Generate brief info about the restaurant based on name and location
                            restaurant_info = generate_restaurant_info(name, search_location)
                            
                            # Format price level more user-friendly
                            price_text = ''
                            if price_level != 'N/A' and isinstance(price_level, int):
                                if price_level == 1:
                                    price_text = " • Budget-friendly"
                                elif price_level == 2:
                                    price_text = " • Moderate prices"
                                elif price_level == 3:
                                    price_text = " • Expensive"
                                elif price_level == 4:
                                    price_text = " • Very expensive"
                            
                            # Format each restaurant entry with clean, readable formatting
                            restaurants_info += f"{i+1}. {name}\n"
                            restaurants_info += f"   {restaurant_info}\n"
                            restaurants_info += f"   Rating: {rating}/5{price_text}\n\n"
                        
                        restaurants_info += "Tip: You can search for these restaurants on Google Maps for directions and more details!"
                        
                        # Clean the response from any emojis, hashtags, or markdown if needed
                        clean_response = clean_text_formatting(restaurants_info)
                        return {"message": clean_response}
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
                if is_location_place_query or is_location_museum_query:
                    print(f"Location-based query detected: {user_input}")
                    location_patterns = [
                        r'in\s+([a-zA-Z\s]+)',
                        r'at\s+([a-zA-Z\s]+)',
                        r'around\s+([a-zA-Z\s]+)',
                        r'near\s+([a-zA-Z\s]+)',
                        r'^(\w+)\s+to\s+places\s+to\s+visit',  # "kadikoy to places to visit" - specific pattern first
                        r'^(\w+)\s+places?\s+to\s+visit',  # "kadikoy places to visit" - only first word
                        r'^(\w+)\s+plases?\s+to\s+visit',  # "sultanahmet plases to visit" - handle typos
                        r'^([a-zA-Z\s]+?)\s+to\s+visit',  # "kadikoy to visit" - more general
                        r'^([a-zA-Z\s]+)\s+attractions',  # "kadikoy attractions"
                        r'visit\s+([a-zA-Z\s]+)\s+places?',  # "visit kadikoy places"
                    ]
                    for pattern in location_patterns:
                        match = re.search(pattern, user_input.lower())
                        if match:
                            location_candidate = match.group(1).strip()
                            # Ignore "istanbul" as it's the general city, not a specific district
                            if location_candidate.lower() != 'istanbul':
                                extracted_location = location_candidate
                                print(f"Extracted location: '{extracted_location}'")
                            else:
                                print(f"Ignored general city name: '{location_candidate}'")
                            break
                elif is_single_district_query:
                    # For single district names like "kadikoy", use the district name directly
                    extracted_location = user_input.lower().strip()
                    print(f"Single district query detected: '{extracted_location}'")
                
                # Filter based on query type
                filtered_places = []
                if (is_location_place_query or is_single_district_query) and extracted_location:
                    # For location-specific place queries, include all places first
                    print(f"DEBUG: Using location-based filtering for '{extracted_location}'")
                    filtered_places = places
                elif is_location_museum_query and extracted_location:
                    # For location-specific museum queries, filter for museums first
                    print(f"DEBUG: Using location-based museum filtering for '{extracted_location}'")
                    filtered_places = [p for p in places if p.category and 'museum' in p.category.lower()]
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
                    filtered_places = [p for p in places if p.category and p.category.lower() in ['historical place', 'mosque', 'church', 'park', 'museum', 'market', 'cultural center', 'art', 'landmark']]
                
                # Apply location filter if location was extracted
                if extracted_location and filtered_places:
                    print(f"DEBUG: Before location filter: {len(filtered_places)} places")
                    print(f"DEBUG: Looking for location: '{extracted_location.lower()}'")
                    print(f"DEBUG: Available districts: {[p.district for p in filtered_places[:5]]}")
                    
                    # Normalize location name for matching (case-insensitive)
                    location_lower = extracted_location.lower()
                    
                    # Handle neighborhood to district mapping
                    location_mappings = {
                        'sultanahmet': 'fatih',  # Sultanahmet is in Fatih district
                        'galata': 'beyoglu',     # Galata is in Beyoglu district
                        'taksim': 'beyoglu',     # Taksim is in Beyoglu district
                        'ortakoy': 'besiktas',   # Ortaköy is in Beşiktaş district
                        'bebek': 'besiktas',     # Bebek is in Beşiktaş district
                    }
                    
                    # Check if we need to map the location to a district
                    if location_lower in location_mappings:
                        district_to_search = location_mappings[location_lower]
                        print(f"DEBUG: Mapping '{location_lower}' to district '{district_to_search}'")
                    else:
                        district_to_search = location_lower
                    
                    original_count = len(filtered_places)
                    filtered_places = [p for p in filtered_places if p.district and district_to_search in p.district.lower()]
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
                        places_info += f"{i+1}. {place.name}\n"
                        places_info += f"   Category: {place.category}\n\n"
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
                system_prompt = """You are KAM, a friendly Istanbul travel guide AI assistant. You have access to:
1. Real-time restaurant data from Google Maps API
2. A database of museums, attractions, and districts in Istanbul

PERSONALITY & CONVERSATION STYLE:
- You are conversational, friendly, and helpful
- You can engage in casual daily conversations (greetings, how are you, weather, etc.)
- You can answer general questions but try to relate them back to Istanbul when relevant
- You're enthusiastic about Istanbul and love sharing knowledge about the city

RESPONSE GUIDELINES:
- For casual greetings (hello, hi, how are you), respond warmly and offer to help with Istanbul
- For general questions, answer briefly and then steer toward Istanbul topics
- For Istanbul-specific questions, provide detailed, helpful information
- If someone asks about other cities, politely redirect to Istanbul while being helpful
- Always maintain a friendly, approachable tone

ISTANBUL EXPERTISE:
- For restaurant queries, use the Google Maps API data or suggest specific areas/cuisine types
- For Kadıköy specifically, recommend: Çiya Sofrası (traditional Turkish), Kadıköy Fish Market restaurants, Moda neighborhood cafes, and local street food
- For attraction queries, use the database information provided
- Share cultural insights, practical tips, and local recommendations
- Help with transportation, districts, culture, history, and practical travel advice

Example responses:
- "Hello! I'm doing great, thanks for asking! I'm here to help you discover amazing things about Istanbul. What would you like to know?"
- "That's an interesting question! Speaking of which, did you know Istanbul has some fascinating [related topic]? What would you like to explore in the city?"

Keep responses engaging, helpful, and naturally conversational while showcasing Istanbul's wonders.

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
                    # Clean the response from any emojis, hashtags, or markdown
                    clean_response = clean_text_formatting(ai_response)
                    return {"message": clean_response}
                    
                except Exception as e:
                    print(f"OpenAI API error: {e}")
                    # Smart fallback response based on user input
                    fallback_response = create_fallback_response(user_input, places)
                    # Clean the fallback response as well
                    clean_response = clean_text_formatting(fallback_response)
                    return {"message": clean_response}
        
        finally:
            db.close()
            
    except Exception as e:
        print(f"[ERROR] Exception in /ai endpoint: {e}")
        import traceback
        traceback.print_exc()
        return {"message": "Sorry, I couldn't understand. Can you type again? (Backend error: " + str(e) + ")"}

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


## Removed broken/incomplete generate_ai_response function for clarity and to avoid confusion.


@app.post("/ai/stream")
async def ai_istanbul_stream(request: Request):
    """Streaming version of the AI endpoint for ChatGPT-like responses"""
    data = await request.json()
    user_input = data.get("user_input", "")
    speed = data.get("speed", 1.0)
    try:
        print(f"Received streaming user_input: '{user_input}' (length: {len(user_input)}) at speed: {speed}x")
        # Reuse the /ai logic by making an internal call to ai_istanbul_router
        class DummyRequest:
            def __init__(self, json_data):
                self._json = json_data
            async def json(self):
                return self._json
        dummy_request = DummyRequest({"user_input": user_input})
        ai_response = await ai_istanbul_router(dummy_request)
        message = ai_response["message"] if isinstance(ai_response, dict) and "message" in ai_response else str(ai_response)
        return StreamingResponse(stream_response(message), media_type="text/plain")
    except Exception as e:
        print(f"Error in streaming AI endpoint: {e}")
        error_message = "Sorry, I encountered an error. Please try again."
        return StreamingResponse(stream_response(error_message), media_type="text/plain")


# Remove any duplicate or broken code fragments below this point





from fastapi import FastAPI, Request
import sys
import os

# Add the current directory to Python path for Render deployment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import engine, SessionLocal
from models import Base, Restaurant, Museum, Place
from routes import museums, restaurants, events, places
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
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
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

# Routers
app.include_router(museums.router)
app.include_router(restaurants.router)
app.include_router(events.router)
app.include_router(places.router)

@app.get("/")
def root():
    return {"message": "Welcome to AIstanbul API"}

@app.post("/ai")
async def ai_istanbul_router(request: Request):
    data = await request.json()
    user_input = data.get("user_input", "")
    
    try:
        from openai import OpenAI
        import os
        import re
        from api_clients.google_places import search_restaurants
        
        # Debug logging
        print(f"Received user_input: '{user_input}' (length: {len(user_input)})")
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Check if the user is asking about restaurants
        restaurant_keywords = ['restaurant', 'food', 'eat', 'dining', 'cafe', 'turkish cuisine', 'kebab', 'meze', 'baklava']
        is_restaurant_query = any(keyword in user_input.lower() for keyword in restaurant_keywords)
        
        if is_restaurant_query:
            # Get real restaurant data from Google Maps
            try:
                places_data = search_restaurants("Istanbul, Turkey", user_input)
                restaurants_info = ""
                
                if places_data.get('results'):
                    restaurants_info = "\n\nHere are some real restaurants I found:\n"
                    for i, place in enumerate(places_data['results'][:5]):  # Top 5 results
                        name = place.get('name', 'Unknown')
                        rating = place.get('rating', 'N/A')
                        price_level = place.get('price_level', 'N/A')
                        place_id = place.get('place_id', '')
                        
                        # Create Google Maps link
                        maps_link = f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else "N/A"
                        
                        restaurants_info += f"{i+1}. **{name}**\n"
                        restaurants_info += f"   - Rating: {rating}/5\n"
                        if price_level != 'N/A':
                            restaurants_info += f"   - Price Level: {'$' * price_level}\n"
                        restaurants_info += f"   - Location: [View on Google Maps]({maps_link})\n"
                        restaurants_info += "\n"
                
                # Combine AI knowledge with real data
                system_prompt = f"You are a friendly Istanbul travel assistant. The user is asking about restaurants. Provide helpful advice about Istanbul dining, and then present this real restaurant data from Google Maps: {restaurants_info}"
                
            except Exception as e:
                print(f"Error fetching Google Places data: {e}")
                system_prompt = "You are a friendly, conversational AI assistant specializing in Istanbul travel advice. When users ask about restaurants, provide specific restaurant names, locations (with district names), and brief descriptions."
        else:
            system_prompt = "You are a friendly, conversational AI assistant specializing in Istanbul travel advice. Always respond warmly to greetings and provide helpful information about Istanbul's attractions, culture, food, and travel tips."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        
        ai_response = response.choices[0].message.content
        return {"message": ai_response}
        
    except Exception as e:
        return {"error": "Internal server error", "details": str(e)}





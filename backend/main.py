# Smart intent/entity integration endpoint
from fastapi import Request

@app.post("/ai")
async def ai_istanbul_router(request: Request):
    data = await request.json()
    user_input = data.get("user_input", "")
    parsed = parse_user_input(user_input)
    try:
        parsed_json = json.loads(parsed)
    except Exception:
        return {"error": "Failed to parse response", "raw": parsed}
    intent = parsed_json.get("intent", "")
    entities = parsed_json.get("entities", {})

    # Restaurant intent
    if intent in ["find_restaurants", "restaurant_search"]:
        from routes.restaurants import get_restaurants
        db = next(SessionLocal())
        results = db.query(Base.classes.restaurants).all()
        return {"results": [r.name for r in results], "entities": entities}

    # Museum intent
    if intent in ["inquire_about_museums", "museum_info"]:
        from routes.museums import get_museums
        db = next(SessionLocal())
        results = db.query(Base.classes.museums).all()
        return {"results": [r.name for r in results], "entities": entities}

    # Place intent
    if intent in ["inquire_about_places", "place_info"]:
        from routes.places import get_places
        db = next(SessionLocal())
        results = db.query(Base.classes.places).all()
        return {"results": [r.name for r in results], "entities": entities}

    # Fallback
    return {"message": "Sorry, I didn’t quite get that. Can you rephrase?", "entities": entities, "intent": intent}

from fastapi import FastAPI, Request
from database import engine, SessionLocal
from models import Base, Restaurant, Museum, Place
from routes import museums, restaurants, events, places
from intent_utils import parse_user_input
import json

app = FastAPI(title="AIstanbul API")

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
        parsed = parse_user_input(user_input)
        try:
            parsed_json = json.loads(parsed)
        except Exception:
            return {"error": "Failed to parse response from OpenAI", "raw": parsed}
        intent = parsed_json.get("intent", "")
        entities = parsed_json.get("entities", {})

        db = SessionLocal()
        try:
            if intent in ["find_restaurants", "restaurant_search"]:
                results = db.query(Restaurant).all()
                return {"results": [r.name for r in results], "entities": entities}
            if intent in ["inquire_about_museums", "museum_info"]:
                results = db.query(Museum).all()
                return {"results": [r.name for r in results], "entities": entities}
            if intent in ["inquire_about_places", "place_info"]:
                results = db.query(Place).all()
                return {"results": [r.name for r in results], "entities": entities}
        finally:
            db.close()

        return {"message": "Sorry, I didn’t quite get that. Can you rephrase?", "entities": entities, "intent": intent}
    except Exception as e:
        return {"error": "Internal server error", "details": str(e)}



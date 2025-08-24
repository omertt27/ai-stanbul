from intent_utils import parse_user_input
import json
import fastapi
fastapi.__version__
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Hello Istanbul AI Guide!"}
from fastapi import FastAPI
from database import engine, SessionLocal
from models import Base

app = FastAPI(title="Istanbul AI Guide")

# Optional: Create tables if they don’t exist yet
Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Istanbul AI API is running!"}
from fastapi import FastAPI
from database import Base, engine
from routes import museums, restaurants, events, places

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AIstanbul API")


# Routers
# Routers
app.include_router(museums.router)
app.include_router(restaurants.router)
app.include_router(events.router)
app.include_router(places.router)

# Intent/entity extraction endpoint for integration test
@app.post("/parse/")
def parse_query(query: dict):
    user_input = query.get("user_input", "")
    parsed = parse_user_input(user_input)
    try:
        parsed_json = json.loads(parsed)
    except Exception:
        parsed_json = {"error": "Failed to parse response", "raw": parsed}
    return parsed_json

@app.get("/")
def root():
    return {"message": "Welcome to AIstanbul API"}
extend_existing=True



import requests

GOOGLE_API_KEY = "R3g!on9$Plx"
BASE_URL = "https://maps.googleapis.com/maps/api/place/"

def search_restaurants(location, radius=1000, type="restaurant"):
    url = f"{BASE_URL}nearbysearch/json"
    params = {
        "key": GOOGLE_API_KEY,
        "location": location,  
        "radius": radius,
        "type": type
    }
    response = requests.get(url, params=params)
    return response.json()
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..database import SessionLocal
from ..models import Restaurant

router = APIRouter(prefix="/restaurants", tags=["Restaurants"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_restaurants(db: Session = Depends(get_db)):
    return db.query(Restaurant).all()

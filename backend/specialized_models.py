from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, Time
from sqlalchemy.orm import relationship
from db.base import Base  # Use central Base - single source of truth
from datetime import datetime, time

class TransportRoute(Base):
    __tablename__ = "transport_routes"
    id = Column(Integer, primary_key=True, index=True)
    route_name = Column(String, index=True)  # "Kadikoy-Eminonu Ferry"
    transport_type = Column(String)  # "ferry", "metro", "bus", "tram"
    from_location = Column(String)
    to_location = Column(String)
    duration_minutes = Column(Integer)
    frequency_minutes = Column(Integer)  # How often it runs
    first_departure = Column(Time)
    last_departure = Column(Time)
    price_try = Column(Float)  # Price in Turkish Lira
    notes = Column(Text)  # "Scenic Bosphorus views", "Express service"
    is_active = Column(Boolean, default=True)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    dietary_restrictions = Column(String)  # "vegetarian", "halal", "vegan"
    accommodation_district = Column(String)  # "Kadikoy", "Sultanahmet"
    days_remaining = Column(Integer)
    budget_level = Column(String)  # "budget", "mid-range", "luxury"
    interests = Column(Text)  # JSON string of interests
    language_preference = Column(String, default="en")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)

class TurkishPhrases(Base):
    __tablename__ = "turkish_phrases"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String)  # "greetings", "food", "directions", "emergencies"
    english_phrase = Column(String)
    turkish_phrase = Column(String)
    pronunciation = Column(String)  # Phonetic pronunciation
    context = Column(String)  # When to use it
    audio_file = Column(String)  # Path to audio file for offline use

class LocalTips(Base):
    __tablename__ = "local_tips"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String)  # "culture", "etiquette", "money", "safety"
    tip_title = Column(String)
    tip_content = Column(Text)
    importance_level = Column(String)  # "essential", "helpful", "nice-to-know"
    relevant_districts = Column(String)  # JSON array of districts
    is_offline_available = Column(Boolean, default=True)

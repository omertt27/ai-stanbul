from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)

class Place(Base):
    __tablename__ = "places"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    category = Column(String(50))
    district = Column(String(50))
    
class Museum(Base):
    __tablename__ = "museums"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    location = Column(String)
    hours = Column(String)
    ticket_price = Column(Float)
    highlights = Column(String)

class Restaurant(Base):
    __tablename__ = "restaurants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    cuisine = Column(String)
    location = Column(String)
    rating = Column(Float)
    source = Column(String)
    description = Column(String)  # Added description field
    place_id = Column(String, unique=True)  # Google Places ID
    phone = Column(String)
    website = Column(String)
    price_level = Column(Integer)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    venue = Column(String)
    date = Column(DateTime)
    genre = Column(String)
    biletix_id = Column(String, unique=True)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_ip = Column(String(50))

class BlogPost(Base):
    __tablename__ = "blog_posts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(Text, nullable=True)  # JSON string of tags
    district = Column(String(100), nullable=True)  # Single district
    author_name = Column(String(100), nullable=True)  # User name
    author_photo = Column(String(500), nullable=True)  # Profile photo URL
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_published = Column(Boolean, default=True)
    likes_count = Column(Integer, default=0)

    # Relationship with images
    images = relationship("BlogImage", back_populates="blog_post")
    
    # Relationship with likes
    likes = relationship("BlogLike", back_populates="blog_post")
    # Relationship with likes
    likes = relationship("BlogLike", back_populates="blog_post")

class BlogImage(Base):
    __tablename__ = "blog_images"
    id = Column(Integer, primary_key=True, autoincrement=True)
    blog_post_id = Column(Integer, ForeignKey("blog_posts.id"), nullable=False)
    url = Column(String(500), nullable=False)  # URL path to the image
    alt_text = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship with blog post
    blog_post = relationship("BlogPost", back_populates="images")

class BlogLike(Base):
    __tablename__ = "blog_likes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    blog_post_id = Column(Integer, ForeignKey("blog_posts.id"), nullable=False)
    user_identifier = Column(String(255), nullable=False)  # Can be IP address or user ID
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Unique constraint to prevent duplicate likes
    __table_args__ = (
        UniqueConstraint('blog_post_id', 'user_identifier', name='unique_user_like'),
    )
    
    # Relationship with blog post
    blog_post = relationship("BlogPost", back_populates="likes")

# Enhanced AI Models for Intelligent Conversation
class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_ip = Column(String(50))
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    preferences = relationship("UserPreference", back_populates="session", uselist=False)
    conversations = relationship("ConversationContext", back_populates="session")
    interactions = relationship("UserInteraction", back_populates="session")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("user_sessions.session_id"), unique=True, nullable=False)
    
    # Cuisine preferences (JSON array)
    preferred_cuisines = Column(JSON, default=list)  # ["turkish", "italian", "seafood"]
    avoided_cuisines = Column(JSON, default=list)   # ["spicy", "vegetarian"]
    
    # Budget preferences
    budget_level = Column(String(20))  # "budget", "mid-range", "luxury", "any"
    
    # Interest categories (JSON array)
    interests = Column(JSON, default=list)  # ["museums", "nightlife", "shopping", "culture"]
    
    # Travel style
    travel_style = Column(String(30))  # "solo", "couple", "family", "business", "group"
    
    # Time preferences
    preferred_time_of_day = Column(JSON, default=list)  # ["morning", "afternoon", "evening", "night"]
    
    # Districts of interest
    preferred_districts = Column(JSON, default=list)  # ["sultanahmet", "beyoglu", "kadikoy"]
    
    # Transportation preferences
    transportation_preference = Column(String(20))  # "walking", "public", "taxi", "mixed"
    
    # Language preference
    language = Column(String(10), default="en")  # "en", "tr"
    
    # Learning metadata
    confidence_score = Column(Float, default=0.0)  # How confident we are in these preferences
    last_updated = Column(DateTime, default=datetime.utcnow)
    total_interactions = Column(Integer, default=0)
    
    # Relationship
    session = relationship("UserSession", back_populates="preferences")

class ConversationContext(Base):
    __tablename__ = "conversation_context"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("user_sessions.session_id"), nullable=False)
    
    # Conversation state
    current_intent = Column(String(50))  # "restaurant_search", "transportation", "attractions"
    context_data = Column(JSON, default=dict)  # Flexible context storage
    
    # Location context
    current_location = Column(String(100))  # Current area of interest
    previous_locations = Column(JSON, default=list)  # History of searched locations
    
    # Topic tracking
    current_topic = Column(String(50))
    topic_history = Column(JSON, default=list)
    
    # Follow-up handling
    expecting_followup = Column(Boolean, default=False)
    followup_type = Column(String(30))  # "location_clarification", "preference_confirmation"
    followup_data = Column(JSON, default=dict)
    
    # Conversation flow
    conversation_stage = Column(String(30), default="initial")  # "initial", "exploring", "deciding", "completed"
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship
    session = relationship("UserSession", back_populates="conversations")

class UserInteraction(Base):
    __tablename__ = "user_interactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("user_sessions.session_id"), nullable=False)
    
    # Interaction details
    user_message = Column(Text, nullable=False)
    processed_intent = Column(String(50))
    extracted_entities = Column(JSON, default=dict)  # locations, times, preferences mentioned
    
    # AI response details
    bot_response = Column(Text, nullable=False)
    response_type = Column(String(30))  # "restaurant_list", "directions", "general_info"
    confidence_score = Column(Float)
    
    # Recommendations tracking
    recommendations_given = Column(JSON, default=list)  # Track what was recommended
    user_feedback = Column(String(20))  # "helpful", "not_helpful", "partially_helpful"
    
    # Performance metrics
    response_time_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    session = relationship("UserSession", back_populates="interactions")

class IntelligentRecommendation(Base):
    __tablename__ = "intelligent_recommendations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Recommendation metadata
    session_id = Column(String(100), index=True)
    recommendation_type = Column(String(30))  # "restaurant", "attraction", "route"
    
    # Item details
    item_name = Column(String(255), nullable=False)
    item_category = Column(String(50))
    location = Column(String(100))
    
    # Scoring and ranking
    base_score = Column(Float)  # Base popularity/rating score
    personalization_score = Column(Float)  # How well it matches user preferences
    context_score = Column(Float)  # Time, weather, location relevance
    final_score = Column(Float)  # Combined final score
    
    # Reasoning
    recommendation_reason = Column(Text)  # Why this was recommended
    factors_considered = Column(JSON, default=list)  # ["user_preference", "time_of_day", "weather"]
    
    # Outcome tracking
    was_clicked = Column(Boolean, default=False)
    user_rating = Column(Integer)  # 1-5 if user provides feedback
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Enhanced Chat History with AI context
class EnhancedChatHistory(Base):
    __tablename__ = "enhanced_chat_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    
    # Message content
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    
    # AI processing details
    detected_intent = Column(String(50))
    intent_confidence = Column(Float)
    extracted_entities = Column(JSON, default=dict)
    
    # Context at time of message
    user_preferences_snapshot = Column(JSON, default=dict)
    conversation_context_snapshot = Column(JSON, default=dict)
    
    # Performance and learning
    processing_time_ms = Column(Integer)
    model_version = Column(String(20))
    was_helpful = Column(Boolean)  # User feedback
    
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_ip = Column(String(50))


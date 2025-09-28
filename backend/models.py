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
    ai_response = Column(Text, nullable=False)  # Fixed to match actual database column
    timestamp = Column(DateTime, default=datetime.utcnow)

class BlogPost(Base):
    __tablename__ = "blog_posts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    author = Column(String(100), nullable=True)  # Matches existing 'author' column
    district = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    likes_count = Column(Integer, default=0)
    
    # Add missing columns with defaults for compatibility
    @property
    def tags(self):
        return None
    
    @property
    def author_name(self):
        return self.author
    
    @property
    def author_photo(self):
        return None
    
    @property
    def updated_at(self):
        return self.created_at
    
    @property
    def is_published(self):
        return True

    # Relationship with images
    images = relationship("BlogImage", back_populates="blog_post")
    
    # Relationship with likes
    likes = relationship("BlogLike", back_populates="blog_post")
    # Relationship with comments
    comments = relationship("BlogComment", back_populates="blog_post")

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

class BlogComment(Base):
    __tablename__ = "blog_comments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    blog_post_id = Column(Integer, ForeignKey("blog_posts.id"), nullable=False)
    
    # Comment content
    author_name = Column(String(100), nullable=False)
    author_email = Column(String(100), nullable=True)  # Optional
    content = Column(Text, nullable=False)
    
    # Moderation fields
    is_approved = Column(Boolean, default=True)   # Auto-approved by default
    is_flagged = Column(Boolean, default=False)   # Flagged for review
    is_spam = Column(Boolean, default=False)      # Marked as spam
    flagged_reason = Column(String(200), nullable=True)  # Reason for flagging
    
    # Metadata
    user_ip = Column(String(50))
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    approved_at = Column(DateTime, nullable=True)
    approved_by = Column(String(100), nullable=True)  # Admin who approved
    
    # Relationships
    blog_post = relationship("BlogPost", back_populates="comments")

class UserMemory(Base):
    __tablename__ = "user_memory"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("user_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Memory categories
    memory_type = Column(String(50), nullable=False, index=True)  # 'preference', 'visited', 'interest', 'experience'
    memory_key = Column(String(100), nullable=False)  # e.g., 'favorite_district', 'visited_place', 'food_preference'
    memory_value = Column(Text, nullable=False)  # The actual memory content
    memory_context = Column(JSON)  # Additional context like timestamps, ratings, etc.
    
    # Memory metadata
    confidence_score = Column(Float, default=0.8)  # How confident we are about this memory
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_referenced = Column(DateTime, default=datetime.utcnow)
    reference_count = Column(Integer, default=1)
    
    # Memory importance and persistence
    importance_level = Column(Integer, default=1)  # 1-5 scale
    is_persistent = Column(Boolean, default=False)  # Should this memory persist across long periods?
    
    # Unique constraint to prevent duplicate memories
    __table_args__ = (UniqueConstraint('session_id', 'memory_type', 'memory_key', name='unique_user_memory'),)
    
    # Relationships
    session = relationship("UserSession", back_populates="memories")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("user_sessions.session_id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Cuisine preferences
    preferred_cuisines = Column(JSON)  # Array of preferred cuisines
    avoided_cuisines = Column(JSON)   # Array of avoided cuisines
    
    # Budget and travel preferences
    budget_level = Column(String(20))  # 'budget', 'mid-range', 'luxury'
    travel_style = Column(String(30))  # 'family', 'couple', 'business', 'group'
    
    # Location and time preferences
    preferred_districts = Column(JSON)  # Array of preferred districts
    preferred_time_of_day = Column(JSON)  # Array of preferred times
    
    # General preferences
    interests = Column(JSON)  # Array of interest categories
    transportation_preference = Column(String(20))  # 'walking', 'metro', 'taxi', etc.
    language = Column(String(10))  # Language preference
    
    # Metadata
    confidence_score = Column(Float, default=0.8)  # How confident we are about preferences
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_interactions = Column(Integer, default=0)  # Number of interactions that informed preferences
    
    # Relationships
    session = relationship("UserSession", back_populates="user_preferences")

class ConversationContext(Base):
    __tablename__ = "conversation_context"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), ForeignKey("user_sessions.session_id"), nullable=False, index=True)
    
    # Context tracking
    current_topic = Column(String(100))  # Current conversation topic
    topics_discussed = Column(JSON, default=list)  # List of topics covered in session
    places_mentioned = Column(JSON, default=list)  # Places that have been discussed
    
    # User journey tracking
    travel_stage = Column(String(50))  # 'planning', 'visiting', 'exploring', 'departing'
    visit_duration = Column(String(50))  # '1 day', '3 days', '1 week', etc.
    travel_style = Column(String(50))  # 'solo', 'family', 'business', 'romantic', 'adventure'
    
    # Current context
    last_location_discussed = Column(String(200))
    current_need = Column(String(100))  # 'directions', 'recommendations', 'planning', 'cultural_info'
    conversation_mood = Column(String(50))  # 'excited', 'confused', 'planning', 'urgent'
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = relationship("UserSession", back_populates="conversation_contexts")

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
    memories = relationship("UserMemory", back_populates="session")
    user_preferences = relationship("UserPreference", back_populates="session")
    conversation_contexts = relationship("ConversationContext", back_populates="session")
    interactions = relationship("UserInteraction", back_populates="session")

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

# User Feedback for Like/Dislike tracking
class UserFeedback(Base):
    __tablename__ = "user_feedback"
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Feedback metadata
    session_id = Column(String(100), nullable=False, index=True)
    feedback_type = Column(String(20), nullable=False)  # "like", "dislike"
    
    # Original query and response
    user_query = Column(Text)
    response_preview = Column(Text)  # First 200 chars of the response
    message_index = Column(Integer)  # Position in conversation
    
    # Session context
    user_ip = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Additional metadata
    message_content = Column(Text)  # Full AI response content
    conversation_context = Column(JSON, default=dict)  # Any additional context

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String(100), primary_key=True)  # session_id as primary key
    
    # Session metadata
    title = Column(String(200))  # Generated from first query or user-provided
    user_ip = Column(String(50))
    
    # Session statistics
    message_count = Column(Integer, default=0)
    first_message_at = Column(DateTime, default=datetime.utcnow)
    last_activity_at = Column(DateTime, default=datetime.utcnow)
    
    # Save status - only sessions with liked messages are saved
    is_saved = Column(Boolean, default=False)
    saved_at = Column(DateTime)
    
    # Session data
    conversation_history = Column(JSON, default=list)  # Store the full conversation


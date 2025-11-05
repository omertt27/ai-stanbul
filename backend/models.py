from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from database import Base
from datetime import datetime

# User model exists in the database already, using extend_existing to avoid conflicts
class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)

class Place(Base):
    __tablename__ = "places"
    __table_args__ = {'extend_existing': True}
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    category = Column(String(50))
    district = Column(String(50))
    
class Museum(Base):
    __tablename__ = "museums"
    __table_args__ = {'extend_existing': True}
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    location = Column(String)
    hours = Column(String)
    ticket_price = Column(Float)
    highlights = Column(String)

class Restaurant(Base):
    __tablename__ = "restaurants"
    __table_args__ = {'extend_existing': True}
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
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    venue = Column(String)
    date = Column(DateTime)
    genre = Column(String)
    biletix_id = Column(String, unique=True)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)  # Fixed to match actual database column
    timestamp = Column(DateTime, default=datetime.utcnow)

class BlogPost(Base):
    __tablename__ = "blog_posts"
    __table_args__ = {'extend_existing': True}
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


# Real-Time Feedback Models for Online Learning
class FeedbackEvent(Base):
    """
    Stores user feedback events for real-time learning
    High-write table for streaming analytics
    """
    __tablename__ = "feedback_events"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    session_id = Column(String(100), index=True)
    event_type = Column(String(50), nullable=False, index=True)  # view, click, save, rating, etc.
    item_id = Column(String(100), nullable=False, index=True)
    item_type = Column(String(50), nullable=False)  # hidden_gem, restaurant, attraction, etc.
    event_metadata = Column('metadata', JSON)  # Additional event data (rating value, dwell time, etc.)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    processed = Column(Boolean, default=False, index=True)  # For streaming pipeline
    
    def __repr__(self):
        return f"<FeedbackEvent(user={self.user_id}, type={self.event_type}, item={self.item_id})>"


class UserInteractionAggregate(Base):
    """
    Pre-computed aggregates of user interactions for fast feature retrieval
    Updated incrementally by the online learning system
    """
    __tablename__ = "user_interaction_aggregates"
    __table_args__ = (
        UniqueConstraint('user_id', 'item_type', name='uix_user_item_type'),
        {'extend_existing': True}
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    item_type = Column(String(50), nullable=False)  # hidden_gem, restaurant, etc.
    
    # Interaction counts
    view_count = Column(Integer, default=0)
    click_count = Column(Integer, default=0)
    save_count = Column(Integer, default=0)
    rating_count = Column(Integer, default=0)
    conversion_count = Column(Integer, default=0)
    
    # Aggregated metrics
    avg_rating = Column(Float, default=0.0)
    avg_dwell_time = Column(Float, default=0.0)
    click_through_rate = Column(Float, default=0.0)  # clicks / views
    conversion_rate = Column(Float, default=0.0)  # conversions / clicks
    
    # Temporal features
    last_interaction = Column(DateTime, default=datetime.utcnow)
    recency_score = Column(Float, default=0.0)  # Time-decayed interaction score
    
    # Category preferences (JSON for flexibility)
    category_preferences = Column(JSON)  # {"cultural": 0.8, "nature": 0.6, ...}
    
    # Metadata
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserInteractionAggregate(user={self.user_id}, type={self.item_type})>"


class ItemFeatureVector(Base):
    """
    Stores learned feature embeddings for items (places, restaurants, etc.)
    Updated incrementally by the online learning system
    """
    __tablename__ = "item_feature_vectors"
    __table_args__ = (
        UniqueConstraint('item_id', 'item_type', name='uix_item_id_type'),
        {'extend_existing': True}
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(String(100), nullable=False, index=True)
    item_type = Column(String(50), nullable=False)
    
    # Learned embeddings (stored as JSON for flexibility)
    embedding_vector = Column(JSON)  # Dense feature vector [0.1, 0.2, ...]
    embedding_version = Column(String(50), default="v1.0")
    
    # Popularity metrics
    total_views = Column(Integer, default=0)
    total_clicks = Column(Integer, default=0)
    total_saves = Column(Integer, default=0)
    avg_rating = Column(Float, default=0.0)
    conversion_rate = Column(Float, default=0.0)
    
    # Quality score (computed from multiple signals)
    quality_score = Column(Float, default=0.5)
    
    # Metadata
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ItemFeatureVector(item={self.item_id}, type={self.item_type})>"


class OnlineLearningModel(Base):
    """
    Stores metadata and parameters for online learning models
    Enables model versioning and A/B testing
    """
    __tablename__ = "online_learning_models"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, unique=True)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # thompson_sampling, neural_cf, etc.
    
    # Model parameters (stored as JSON)
    parameters = Column(JSON)  # {"learning_rate": 0.01, "embedding_dim": 64, ...}
    hyperparameters = Column(JSON)
    
    # Performance metrics
    metrics = Column(JSON)  # {"precision": 0.85, "recall": 0.72, "ndcg": 0.78, ...}
    
    # Model state
    is_active = Column(Boolean, default=True)
    is_deployed = Column(Boolean, default=False)
    deployment_percentage = Column(Float, default=0.0)  # For A/B testing
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<OnlineLearningModel(name={self.model_name}, version={self.model_version})>"


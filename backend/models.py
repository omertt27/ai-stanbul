from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, UniqueConstraint, JSON
from sqlalchemy.orm import relationship
from db.base import Base  # Use central Base - single source of truth
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
    photo_url = Column(String)  # Local or CDN URL to stored photo
    photo_reference = Column(String)  # Google Places photo reference (for future updates)

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
    slug = Column(String(250), nullable=True, unique=True)
    content = Column(Text, nullable=False)
    excerpt = Column(Text, nullable=True)
    author = Column(String(100), nullable=True)  # Matches existing 'author' column
    status = Column(String(20), default='draft')  # draft, published, archived
    featured_image = Column(String(500), nullable=True)
    category = Column(String(100), nullable=True)
    tags = Column(JSON, default=list)
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    district = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)
    likes_count = Column(Integer, default=0)
    
    # Add missing columns with defaults for compatibility
    @property
    def author_name(self):
        return self.author
    
    @property
    def author_photo(self):
        return None
    
    @property
    def is_published(self):
        return self.status == 'published'


class BlogComment(Base):
    __tablename__ = "blog_comments"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    blog_post_id = Column(Integer, ForeignKey('blog_posts.id', ondelete='CASCADE'), nullable=False)
    author_name = Column(String(100), nullable=False)
    author_email = Column(String(200), nullable=True)
    content = Column(Text, nullable=False)
    status = Column(String(20), default='approved')  # pending, approved, rejected
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    @property
    def is_approved(self):
        return self.status == 'approved'


class BlogLike(Base):
    __tablename__ = "blog_likes"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, autoincrement=True)
    blog_post_id = Column(Integer, ForeignKey('blog_posts.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(100), nullable=True)  # IP or user identifier
    created_at = Column(DateTime, default=datetime.utcnow)


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
    rating = Column(Integer, nullable=True)  # Rating value if applicable
    feedback_text = Column(Text, nullable=True)  # Text feedback if provided
    context = Column(JSON, nullable=True)  # Contextual information
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


# ==============================================
# GPS Navigation & Location Tracking Models
# ==============================================

class LocationHistory(Base):
    """
    Stores user GPS location history for navigation and analytics
    """
    __tablename__ = "location_history"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    session_id = Column(String(100), index=True)
    
    # GPS coordinates
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    accuracy = Column(Float)  # Accuracy in meters
    altitude = Column(Float)  # Altitude in meters
    
    # Motion data
    speed = Column(Float)  # Speed in m/s
    heading = Column(Float)  # Direction in degrees (0-360)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Context
    activity_type = Column(String(50))  # walking, driving, stationary
    is_navigation_active = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<LocationHistory(user={self.user_id}, lat={self.latitude}, lon={self.longitude})>"


class NavigationSession(Base):
    """
    Stores active and completed navigation sessions
    """
    __tablename__ = "navigation_sessions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    chat_session_id = Column(String(100), index=True)  # Link to chat session
    
    # Route information
    origin_lat = Column(Float, nullable=False)
    origin_lon = Column(Float, nullable=False)
    origin_name = Column(String(255))
    
    destination_lat = Column(Float, nullable=False)
    destination_lon = Column(Float, nullable=False)
    destination_name = Column(String(255))
    
    # Waypoints (JSON array of {lat, lon, name})
    waypoints = Column(JSON)
    
    # Route data
    total_distance = Column(Float)  # Total distance in meters
    total_duration = Column(Float)  # Total duration in seconds
    transport_mode = Column(String(50), default='walking')  # walking, driving, transit
    
    # Navigation state
    current_step_index = Column(Integer, default=0)
    steps_completed = Column(Integer, default=0)
    distance_remaining = Column(Float)
    time_remaining = Column(Float)
    
    # Status
    status = Column(String(50), default='active')  # active, completed, cancelled, paused
    is_active = Column(Boolean, default=True)
    
    # Route geometry (encoded polyline or GeoJSON)
    route_geometry = Column(JSON)
    route_steps = Column(JSON)  # Full turn-by-turn instructions
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    last_update = Column(DateTime, default=datetime.utcnow)
    
    # Performance metrics
    actual_duration = Column(Float)  # Actual time taken
    deviations_count = Column(Integer, default=0)  # Times user deviated from route
    reroutes_count = Column(Integer, default=0)  # Times route was recalculated
    
    def __repr__(self):
        return f"<NavigationSession(id={self.session_id}, status={self.status})>"


class RouteHistory(Base):
    """
    Stores completed routes for analytics and recommendations
    """
    __tablename__ = "route_history"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)
    navigation_session_id = Column(String(100), ForeignKey('navigation_sessions.session_id'))
    
    # Route details
    origin = Column(String(255))
    destination = Column(String(255))
    waypoints = Column(JSON)
    
    # Metrics
    distance = Column(Float)  # meters
    duration = Column(Float)  # seconds
    transport_mode = Column(String(50))
    
    # Route data (for replay/analysis)
    route_geometry = Column(JSON)
    steps = Column(JSON)
    
    # User rating
    user_rating = Column(Integer)  # 1-5 stars
    user_feedback = Column(Text)
    
    # Timestamps
    completed_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<RouteHistory(user={self.user_id}, from={self.origin}, to={self.destination})>"


class NavigationEvent(Base):
    """
    Stores navigation events for real-time tracking and analytics
    """
    __tablename__ = "navigation_events"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), nullable=False, index=True)
    
    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # step_started, step_completed, reroute, deviation, arrival, etc.
    event_data = Column(JSON)  # Additional event-specific data
    
    # Location at event time
    latitude = Column(Float)
    longitude = Column(Float)
    
    # Step information
    current_step = Column(Integer)
    step_instruction = Column(Text)
    distance_to_next_step = Column(Float)
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<NavigationEvent(session={self.session_id}, type={self.event_type})>"


class UserPreferences(Base):
    """
    Stores user preferences for navigation and recommendations
    """
    __tablename__ = "user_preferences"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Navigation preferences
    preferred_transport = Column(String(50), default='walking')  # walking, driving, transit
    avoid_highways = Column(Boolean, default=False)
    avoid_tolls = Column(Boolean, default=False)
    avoid_ferries = Column(Boolean, default=False)
    
    # Accessibility
    wheelchair_accessible = Column(Boolean, default=False)
    requires_elevator = Column(Boolean, default=False)
    
    # Language & units
    preferred_language = Column(String(10), default='en')
    distance_units = Column(String(10), default='metric')  # metric, imperial
    
    # Notifications
    voice_guidance = Column(Boolean, default=True)
    notification_sound = Column(Boolean, default=True)
    vibration = Column(Boolean, default=True)
    
    # Privacy
    save_location_history = Column(Boolean, default=True)
    share_location = Column(Boolean, default=False)
    
    # Recommendation preferences (JSON)
    interests = Column(JSON)  # ["cultural", "nature", "food", ...]
    dietary_restrictions = Column(JSON)  # ["vegetarian", "halal", ...]
    budget_level = Column(String(20), default='moderate')  # budget, moderate, luxury
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<UserPreferences(user={self.user_id}, transport={self.preferred_transport})>"


class ChatSession(Base):
    """
    Enhanced chat session with navigation integration
    """
    __tablename__ = "chat_sessions"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    user_id = Column(String(100), index=True)
    
    # Session metadata
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_activity = Column(DateTime, default=datetime.utcnow)
    messages_count = Column(Integer, default=0)
    
    # Navigation context
    active_navigation_session = Column(String(100))  # Current navigation session ID
    has_navigation = Column(Boolean, default=False)
    
    # User context (JSON)
    context = Column(JSON)  # Location, preferences, history, etc.
    
    # Status
    is_active = Column(Boolean, default=True)
    ended_at = Column(DateTime)
    
    def __repr__(self):
        return f"<ChatSession(id={self.session_id}, messages={self.messages_count})>"


class ConversationHistory(Base):
    """
    Stores conversation history with navigation data
    """
    __tablename__ = "conversation_history"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_id = Column(String(100), index=True)
    
    # Message content
    user_message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    
    # Navigation context
    route_data = Column(JSON)  # Route information if message involved navigation
    location_data = Column(JSON)  # User location at time of message
    navigation_active = Column(Boolean, default=False)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    intent = Column(String(100))  # route_request, navigation_help, general_question, etc.
    entities_extracted = Column(JSON)  # Extracted locations, preferences, etc.
    
    def __repr__(self):
        return f"<ConversationHistory(session={self.session_id}, time={self.timestamp})>"


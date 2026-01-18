# Self-Contained AI-stanbul Backend for Production
# All dependencies included in one file to avoid import issues

import sys
import os
import json
import logging
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add parent directory for accessing advanced transportation modules
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import advanced transportation system
try:
    from transportation_integration_helper import TransportationQueryProcessor
    from ml_enhanced_transportation_system import create_ml_enhanced_transportation_system, GPSLocation
    ADVANCED_TRANSPORT_AVAILABLE = True
    logger.info("✅ Advanced transportation system with IBB API loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Advanced transportation system not available: {e}")
    ADVANCED_TRANSPORT_AVAILABLE = False

# Test for python-multipart availability early
try:
    import multipart
    logger.info("✅ python-multipart is available")
except ImportError as e:
    print(f"❌ Warning: python-multipart not found: {e}")
    print("📦 Installing python-multipart...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-multipart==0.0.6"])
        import multipart
        print("✅ python-multipart installed successfully")
    except Exception as install_error:
        print(f"❌ Failed to install python-multipart: {install_error}")
        print("⚠️  Continuing without multipart support...")

# Core imports
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel

# Database imports
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

# Import Base from central location
sys.path.insert(0, os.path.dirname(__file__))
from db.base import Base

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================
# DATABASE CONFIGURATION
# ===============================

# Database URL with fallback
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine
try:
    if "sqlite" in DATABASE_URL:
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Database engine creation failed: {e}")
    # Fallback to in-memory SQLite
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Using fallback in-memory database")

# ===============================
# DATABASE MODELS
# ===============================

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text)
    ai_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String, index=True)

class Restaurant(Base):
    __tablename__ = "restaurants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    cuisine = Column(String)
    location = Column(String)
    rating = Column(Float)
    description = Column(Text)
    phone = Column(String)
    price_level = Column(Integer)

class Museum(Base):
    __tablename__ = "museums"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    location = Column(String)
    hours = Column(String)
    ticket_price = Column(Float)
    highlights = Column(Text)

class BlogPost(Base):
    __tablename__ = "blog_posts"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text)
    author = Column(String)
    district = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    likes_count = Column(Integer, default=0)

# ===============================
# PYDANTIC MODELS
# ===============================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: str
    session_id: Optional[str] = None

# ===============================
# DATABASE DEPENDENCY
# ===============================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===============================
# FASTAPI APP CONFIGURATION
# ===============================

app = FastAPI(
    title="AI-stanbul API",
    description="AI-powered Istanbul travel assistant",
    version="2.0.0"
)

# Initialize advanced transportation system
if ADVANCED_TRANSPORT_AVAILABLE:
    try:
        transport_processor = TransportationQueryProcessor()
        ml_transport_system = create_ml_enhanced_transportation_system()
        logger.info("🚇 Advanced transportation system with IBB API initialized in backend")
    except Exception as e:
        logger.error(f"Failed to initialize advanced transportation: {e}")
        transport_processor = None
        ml_transport_system = None
        ADVANCED_TRANSPORT_AVAILABLE = False
else:
    transport_processor = None
    ml_transport_system = None

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174", 
    "http://localhost:5175",
    "http://localhost:5176",
    "https://ai-stanbul.onrender.com",
    "https://your-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # More permissive for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# ROUTE INTEGRATION - ML-POWERED CHAT
# ===============================

# Import unified chat router with ML systems integration
try:
    from routes.unified_chat import router as unified_chat_router
    app.include_router(unified_chat_router)
    logger.info("✅ Unified ML-powered chat endpoint loaded (/api/chat)")
    UNIFIED_CHAT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Unified chat router not available: {e}")
    logger.warning("⚠️ Falling back to legacy OpenAI-only chat endpoint")
    UNIFIED_CHAT_AVAILABLE = False

# ===============================
# OPENAI INTEGRATION
# ===============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    import openai
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        logger.info("OpenAI API configured")
    else:
        logger.warning("OpenAI API key not found")
        openai = None
except ImportError:
    logger.warning("OpenAI package not installed")
    openai = None

# ===============================
# UTILITY FUNCTIONS
# ===============================

def simple_fuzzy_match(query: str, choices: List[str], threshold: int = 60) -> Optional[str]:
    """Simple fuzzy matching without external dependencies"""
    query = query.lower().strip()
    
    # Exact match
    for choice in choices:
        if query == choice.lower():
            return choice
    
    # Partial match
    for choice in choices:
        if query in choice.lower() or choice.lower() in query:
            return choice
    
    # Word-based matching
    query_words = set(query.split())
    for choice in choices:
        choice_words = set(choice.lower().split())
        if query_words & choice_words:  # If there's any common word
            return choice
    
    return None

def get_advanced_transportation_response(query: str) -> str:
    """Get advanced transportation response using IBB API and ML system"""
    try:
        if ADVANCED_TRANSPORT_AVAILABLE and transport_processor:
            logger.info("🚇 Using advanced transportation system with IBB API")
            
            # Create dummy entities and user profile for processing
            entities = {}
            from datetime import datetime
            from dataclasses import dataclass
            from typing import Optional
            
            @dataclass
            class DummyUserProfile:
                user_id: str = "default"
                language: str = "en"
                gps_location: Optional[tuple] = None
                
            user_profile = DummyUserProfile()
            
            # Process query through advanced system
            enhanced_response = transport_processor.process_transportation_query(
                query, entities, user_profile
            )
            
            if enhanced_response and enhanced_response.strip():
                return enhanced_response
                
    except Exception as e:
        logger.error(f"Advanced transportation system error: {e}")
    
    # Return None to use fallback
    return None

def get_default_response(query: str) -> str:
    """Generate a helpful default response when AI is not available"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['restaurant', 'food', 'eat', 'dining']):
        # Try to extract district/location from query
        district = None
        if 'beyoğlu' in query_lower or 'beyoglu' in query_lower:
            district = 'Beyoğlu'
        elif 'sultanahmet' in query_lower:
            district = 'Sultanahmet'
        elif 'kadıköy' in query_lower or 'kadikoy' in query_lower:
            district = 'Kadıköy'
        elif 'fatih' in query_lower:
            district = 'Fatih'
        elif 'beşiktaş' in query_lower or 'besiktas' in query_lower:
            district = 'Beşiktaş'
        elif 'şişli' in query_lower or 'sisli' in query_lower:
            district = 'Şişli'
        
        try:
            # Import here to avoid circular imports
            from api_clients.google_places import get_istanbul_restaurants_with_descriptions
            
            # Get restaurant data
            restaurants = get_istanbul_restaurants_with_descriptions(
                district=district,
                limit=5
            )
            
            if restaurants:
                # Format the response nicely
                response_lines = []
                if district:
                    response_lines.append(f"🍽️ **Great restaurants in {district}, Istanbul:**\n")
                else:
                    response_lines.append("🍽️ **Great restaurants in Istanbul:**\n")
                
                for i, restaurant in enumerate(restaurants, 1):
                    name = restaurant.get('name', 'Unknown Restaurant')
                    rating = restaurant.get('rating', 'N/A')
                    vicinity = restaurant.get('vicinity', 'Istanbul')
                    description = restaurant.get('description', 'A great restaurant offering quality dining and local cuisine.')
                    price_level = restaurant.get('price_level', 0)
                    
                    # Format price level
                    price_text = ""
                    if price_level == 1:
                        price_text = " • Budget-friendly"
                    elif price_level == 2:
                        price_text = " • Moderate"
                    elif price_level == 3:
                        price_text = " • Upscale"
                    elif price_level == 4:
                        price_text = " • Expensive"
                    
                    # Clean up description and add formatting
                    if len(description) > 100:
                        description = description[:100] + "..."
                    
                    response_lines.append(
                        f"**{i}. {name}**\n"
                        f"📍 {vicinity}\n"
                        f"⭐ Rating: {rating}/5{price_text}\n"
                        f"ℹ️ {description}\n"
                    )
                
                response_lines.append("\n💡 **Tip:** You can search for these restaurants on Google Maps for directions and more details!")
                
                return "\n".join(response_lines)
            
        except Exception as e:
            logger.error(f"Error fetching restaurants: {e}")
        
        # Fallback response if restaurant fetch fails
        fallback_district = district if district else "Istanbul"
        return f"""🍽️ **For {fallback_district} restaurants, I recommend exploring these areas:**

**Sultanahmet**: Traditional Turkish cuisine near historical sites
**Karaköy**: Modern restaurants with Bosphorus views  
**Beyoğlu**: Diverse international and local options
**Kadıköy**: Authentic local eateries and street food

Popular dishes to try: Kebabs, Meze, Baklava, Turkish Breakfast, and fresh seafood by the Bosphorus!

💡 **Tip:** Try asking me about restaurants in specific districts like "restaurants in Beyoğlu" for more targeted recommendations!"""

    elif any(word in query_lower for word in ['museum', 'history', 'culture', 'art', 'historical', 'heritage', 'palace', 'mosque', 'church']):
        return """🏛️ **Top Istanbul Museums & Cultural Sites:**

**1. Hagia Sophia** 🕌
📍 Sultanahmet, Fatih
⭐ Rating: 4.5/5 • World Heritage Site
ℹ️ Iconic Byzantine church turned mosque, showcasing 1,500 years of history with stunning mosaics and architecture.

**2. Topkapi Palace** 👑
📍 Sultanahmet, Fatih  
⭐ Rating: 4.4/5 • Ottoman Imperial Palace
ℹ️ Former residence of Ottoman sultans with magnificent courtyards, treasury, and panoramic Bosphorus views.

**3. Blue Mosque (Sultan Ahmed)** 🔵
📍 Sultanahmet, Fatih
⭐ Rating: 4.5/5 • Active Mosque
ℹ️ Stunning Ottoman architecture with six minarets and beautiful blue Iznik tiles decorating the interior.

**4. Archaeological Museum** 🏺
📍 Sultanahmet, Fatih
⭐ Rating: 4.3/5 • Ancient Artifacts
ℹ️ Houses incredible artifacts from ancient civilizations including the Alexander Sarcophagus and Babylonian treasures.

**5. Basilica Cistern** 💧
📍 Sultanahmet, Fatih
⭐ Rating: 4.2/5 • Underground Wonder
ℹ️ Mysterious 6th-century underground cistern with 336 marble columns and atmospheric lighting.

💡 **Tips:** 
• Most museums are closed on Mondays
• Get a Museum Pass Istanbul for better value
• Visit early morning to avoid crowds
• Sultanahmet area has many sites within walking distance"""

    elif any(word in query_lower for word in ['place', 'visit', 'attraction', 'tourist', 'sightseeing', 'things to do', 'what to see']):
        return """🎯 **Must-Visit Places in Istanbul:**

**1. Galata Tower** 🗼
📍 Galata, Beyoğlu
⭐ Rating: 4.3/5 • Panoramic Views
ℹ️ Medieval stone tower offering 360° views of Istanbul, the Bosphorus, and Golden Horn from its observation deck.

**2. Grand Bazaar** 🛍️
📍 Beyazıt, Fatih
⭐ Rating: 4.1/5 • Historic Shopping
ℹ️ One of the world's oldest covered markets with 4,000 shops selling carpets, jewelry, spices, and Turkish crafts.

**3. Spice Bazaar** 🌶️
📍 Eminönü, Fatih  
⭐ Rating: 4.3/5 • Aromatic Experience
ℹ️ Colorful market filled with exotic spices, Turkish delight, dried fruits, and traditional Ottoman delicacies.

**4. Bosphorus Bridge** 🌉
📍 Ortaköy, Beşiktaş
⭐ Rating: 4.4/5 • Iconic Landmark
ℹ️ Suspension bridge connecting Europe and Asia with stunning views, especially beautiful at sunset and night.

**5. Taksim Square** 🏙️
📍 Taksim, Beyoğlu
⭐ Rating: 4.0/5 • City Center
ℹ️ Bustling heart of modern Istanbul with shops, restaurants, and the famous Istiklal Street pedestrian avenue.

💡 **Tips:**
• Take a Bosphorus cruise for unique city views
• Visit Galata Tower at sunset for best photos
• Bargain respectfully in the bazaars
• Combine multiple nearby attractions in one day"""

    elif any(word in query_lower for word in ['transport', 'metro', 'bus', 'travel', 'get around']):
        # Try advanced transportation system first
        advanced_response = get_advanced_transportation_response(query)
        if advanced_response:
            return advanced_response
            
        # Fallback to static response
        return """🚇 **Istanbul Transportation Guide:**

**1. Istanbulkart** 💳
💰 Cost: ~15₺ card + credit
ℹ️ Essential rechargeable card for all public transport. Buy at metro stations, kiosks, or ferry terminals.

**2. Metro System** 🚇
⭐ Rating: 4.4/5 • Fast & Clean
ℹ️ Modern subway system covering major areas. M11 (IST Airport), M2 (Taksim-Şişli), M1A (Grand Bazaar area).

**3. Metrobus** 🚌
⭐ Rating: 4.2/5 • Rapid Transit
ℹ️ High-capacity bus system with dedicated lanes. Connects European and Asian sides quickly.

**4. Ferry System** ⛴️
⭐ Rating: 4.6/5 • Scenic Route
ℹ️ Beautiful Bosphorus crossings between continents. Try Eminönü-Kadıköy or Beşiktaş-Üsküdar routes.

**5. Dolmuş** 🚐
⭐ Rating: 4.0/5 • Local Experience
ℹ️ Shared minibuses following fixed routes. Authentic local transport, just say "Müsait" to board.

**6. Taxi & Ride-sharing** 🚕
💰 Cost: Moderate • Apps: BiTaksi, Uber
ℹ️ Yellow taxis everywhere. Use apps for better pricing and English support.

💡 **Apps to Download:**
• İETT (Real-time public transport)
• Mobiett (Route planning)
• BiTaksi (Ride-hailing)
• Istanbul Metro Map (Offline maps)"""

    else:
        return f"""👋 **Welcome to AI-stanbul!** I'm here to help you explore Istanbul.

I can assist you with:
🍽️ **Restaurants** - Local cuisine and dining recommendations
🏛️ **Museums** - Historical sites and cultural attractions  
🎯 **Places** - Popular areas and hidden gems
🚇 **Transportation** - Getting around the city
🎪 **Events** - What's happening in Istanbul

You asked: "{query}"

What specific aspect of Istanbul would you like to explore? Try asking about "restaurants in Beyoğlu" or "museums near Sultanahmet" for specific recommendations!"""

# ===============================
# API ENDPOINTS
# ===============================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = True
    except Exception:
        db_status = False
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": db_status,
            "openai": openai is not None and OPENAI_API_KEY is not None
        },
        "version": "2.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI-stanbul API v2.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs",
            "blog": "/api/blog/posts"
        }
    }

# ===============================
# LEGACY CHAT ENDPOINT (DEPRECATED)
# ===============================
# This endpoint is kept as a fallback if unified_chat router fails to load
# The new ML-powered endpoint is at /api/chat (via unified_chat router)
# This legacy endpoint uses OpenAI directly without ML intent classification

@app.post("/chat_legacy")
async def chat_endpoint_legacy(request: ChatRequest, db: Session = Depends(get_db)):
    """
    DEPRECATED: Legacy chat endpoint with OpenAI-only integration
    
    This endpoint is kept as a fallback for compatibility.
    New code should use /api/chat which includes ML intent classification.
    """
    logger.warning("⚠️ Using legacy chat endpoint - consider migrating to /api/chat")
    
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"
        
        # Generate response
        if openai and OPENAI_API_KEY:
            try:
                # Use OpenAI for intelligent responses (OpenAI v1.x+ syntax)
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful Istanbul travel assistant. Provide accurate, friendly advice about Istanbul's restaurants, museums, attractions, transportation, and culture. Keep responses informative but concise."
                        },
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                ai_response = response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                ai_response = get_default_response(user_message)
        else:
            ai_response = get_default_response(user_message)
        
        # Save to database
        try:
            chat_record = ChatHistory(
                user_message=user_message,
                ai_response=ai_response,
                session_id=session_id
            )
            db.add(chat_record)
            db.commit()
        except Exception as e:
            logger.error(f"Database save error: {e}")
            # Continue without saving if DB fails
        
        return ChatResponse(
            response=ai_response or "",
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Legacy chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Fallback /chat endpoint if unified_chat router is not available
if not UNIFIED_CHAT_AVAILABLE:
    @app.post("/chat")
    async def chat_endpoint_fallback(request: ChatRequest, db: Session = Depends(get_db)):
        """Fallback chat endpoint when unified ML system is not available"""
        logger.warning("⚠️ Unified chat not available, using fallback OpenAI endpoint")
        return await chat_endpoint_legacy(request, db)

@app.get("/api/blog/posts")
async def get_blog_posts(db: Session = Depends(get_db)):
    """Get blog posts"""
    try:
        posts = db.query(BlogPost).order_by(BlogPost.created_at.desc()).limit(10).all()
        return {
            "posts": [
                {
                    "id": post.id,
                    "title": post.title,
                    "content": (post.content[:200] + "...") if isinstance(post.content, str) and len(post.content) > 200 else post.content,
                    "author": post.author,
                    "district": post.district,
                    "created_at": post.created_at.isoformat(),
                    "likes_count": post.likes_count
                }
                for post in posts
            ],
            "total": len(posts)
        }
    except Exception as e:
        logger.error(f"Blog posts error: {e}")
        return {"posts": [], "total": 0, "error": "Unable to fetch blog posts"}

@app.get("/api/blog/posts/{post_id}")
async def get_blog_post(post_id: int, db: Session = Depends(get_db)):
    """Get single blog post"""
    try:
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        return {
            "id": post.id,
            "title": post.title,
            "content": post.content,
            "author": post.author,
            "district": post.district,
            "created_at": post.created_at.isoformat(),
            "likes_count": post.likes_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Blog post error: {e}")
        raise HTTPException(status_code=500, detail="Unable to fetch blog post")

@app.get("/api/restaurants")
async def get_restaurants(db: Session = Depends(get_db)):
    """Get restaurants"""
    try:
        restaurants = db.query(Restaurant).limit(20).all()
        return {
            "restaurants": [
                {
                    "id": r.id,
                    "name": r.name,
                    "cuisine": r.cuisine,
                    "location": r.location,
                    "rating": r.rating,
                    "description": r.description
                }
                for r in restaurants
            ]
        }
    except Exception as e:
        logger.error(f"Restaurants error: {e}")
        return {"restaurants": [], "error": "Unable to fetch restaurants"}

@app.get("/api/museums")
async def get_museums(db: Session = Depends(get_db)):
    """Get museums"""
    try:
        museums = db.query(Museum).limit(20).all()
        return {
            "museums": [
                {
                    "id": m.id,
                    "name": m.name,
                    "location": m.location,
                    "hours": m.hours,
                    "ticket_price": m.ticket_price,
                    "highlights": m.highlights
                }
                for m in museums
            ]
        }
    except Exception as e:
        logger.error(f"Museums error: {e}")
        return {"museums": [], "error": "Unable to fetch museums"}

# ===============================
# DATABASE INITIALIZATION
# ===============================

def init_database():
    """Initialize database tables and sample data"""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Add sample data if tables are empty
        db = SessionLocal()
        try:
            # Check if we need sample data
            if db.query(Restaurant).count() == 0:
                sample_restaurants = [
                    Restaurant(
                        name="Pandeli",
                        cuisine="Ottoman",
                        location="Eminönü",
                        rating=4.5,
                        description="Historic Ottoman restaurant above the Spice Bazaar"
                    ),
                    Restaurant(
                        name="Çiya Sofrası",
                        cuisine="Regional Turkish",
                        location="Kadıköy",
                        rating=4.6,
                        description="Authentic regional Turkish dishes"
                    )
                ]
                for restaurant in sample_restaurants:
                    db.add(restaurant)
            
            if db.query(Museum).count() == 0:
                sample_museums = [
                    Museum(
                        name="Hagia Sophia",
                        location="Sultanahmet",
                        hours="9:00-19:00",
                        ticket_price=100.0,
                        highlights="Byzantine architecture, mosaics, Islamic calligraphy"
                    ),
                    Museum(
                        name="Topkapi Palace",
                        location="Sultanahmet",
                        hours="9:00-18:00",
                        ticket_price=200.0,
                        highlights="Ottoman treasures, imperial chambers, Bosphorus views"
                    )
                ]
                for museum in sample_museums:
                    db.add(museum)
            
            db.commit()
            logger.info("Sample data added successfully")
            
        except Exception as e:
            logger.error(f"Sample data error: {e}")
            db.rollback()
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

# Initialize database on startup
init_database()

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting AI-stanbul backend on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

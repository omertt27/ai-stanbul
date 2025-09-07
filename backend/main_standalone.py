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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

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

Base = declarative_base()

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

def get_default_response(query: str) -> str:
    """Generate a helpful default response when AI is not available"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['restaurant', 'food', 'eat', 'dining']):
        return """🍽️ For Istanbul restaurants, I recommend exploring these areas:

**Sultanahmet**: Traditional Turkish cuisine near historical sites
**Karaköy**: Modern restaurants with Bosphorus views  
**Beyoğlu**: Diverse international and local options
**Kadıköy**: Authentic local eateries and street food

Popular dishes to try: Kebabs, Meze, Baklava, Turkish Breakfast, and fresh seafood by the Bosphorus!"""

    elif any(word in query_lower for word in ['museum', 'history', 'culture', 'art']):
        return """🏛️ Top Istanbul Museums & Cultural Sites:

**Hagia Sophia**: Iconic Byzantine architecture
**Topkapi Palace**: Ottoman imperial residence
**Blue Mosque**: Stunning Islamic architecture
**Archaeological Museum**: Ancient artifacts
**Istanbul Modern**: Contemporary Turkish art

Most museums are closed on Mondays. Consider getting a Museum Pass Istanbul for better value!"""

    elif any(word in query_lower for word in ['transport', 'metro', 'bus', 'travel', 'get around']):
        return """🚇 Istanbul Transportation:

**Istanbulkart**: Essential card for all public transport
**Metro**: Fast and efficient, covers major areas
**Dolmuş**: Shared minibuses for local travel
**Ferry**: Scenic Bosphorus crossings
**Taxi**: Widely available, use BiTaksi app

Download the İETT app for real-time public transport info!"""

    else:
        return f"""👋 Welcome to AI-stanbul! I'm here to help you explore Istanbul.

I can assist you with:
🍽️ **Restaurants** - Local cuisine and dining recommendations
🏛️ **Museums** - Historical sites and cultural attractions  
🎯 **Places** - Popular areas and hidden gems
🚇 **Transportation** - Getting around the city
🎪 **Events** - What's happening in Istanbul

You asked: "{query}"

What specific aspect of Istanbul would you like to explore? Just ask me about restaurants, museums, places to visit, or anything else about this amazing city!"""

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

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    """Main chat endpoint with AI integration"""
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
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

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

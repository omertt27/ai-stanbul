# Minimal Production-Ready AI-stanbul Backend
import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-stanbul API",
    description="AI-powered Istanbul travel assistant",
    version="1.0.0"
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "https://your-domain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup with fallback
try:
    import sys
    import os
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Import Base from database module
    sys.path.insert(0, os.path.dirname(__file__))
    from database import Base
    
    DB_URL = "sqlite:///./app.db"
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")
    
except Exception as e:
    logger.warning(f"Database setup failed: {e}")
    SessionLocal = None

# OpenAI setup with fallback
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    import openai
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        logger.info("OpenAI API initialized")
    else:
        logger.warning("OpenAI API key not found")
except ImportError:
    logger.warning("OpenAI package not installed")
    openai = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": SessionLocal is not None,
            "openai": openai is not None and OPENAI_API_KEY is not None
        }
    }

# Simple chat endpoint
@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_message = data.get("message", "")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Simple response for testing
        response = {
            "response": f"Hello! You asked about: {user_message}. I'm currently running in minimal mode. For full functionality, please ensure all dependencies are installed.",
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Blog endpoints (minimal)
@app.get("/api/blog/posts")
async def get_blog_posts():
    return {
        "posts": [],
        "total": 0,
        "message": "Blog functionality available in full mode"
    }

@app.get("/api/blog/posts/{post_id}")
async def get_blog_post(post_id: int):
    return {
        "id": post_id,
        "title": "Sample Post",
        "content": "This is a sample blog post in minimal mode.",
        "created_at": datetime.now().isoformat()
    }

# Restaurant endpoints (minimal)
@app.get("/api/restaurants")
async def get_restaurants():
    return {
        "restaurants": [],
        "message": "Restaurant data available in full mode"
    }

# Museum endpoints (minimal)
@app.get("/api/museums")
async def get_museums():
    return {
        "museums": [],
        "message": "Museum data available in full mode"
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI-stanbul API - Minimal Mode",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)

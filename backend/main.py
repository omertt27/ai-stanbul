from fastapi import FastAPI, Request
from .database import engine, SessionLocal
from .models import Base, Restaurant, Museum, Place
from .routes import museums, restaurants, events, places
from .intent_utils import parse_user_input
import json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="AIstanbul API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        # Production frontend URLs
        "https://aistanbul.vercel.app",
        "https://aistanbul-fdsqdpks5-omers-projects-3eea52d8.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create tables if needed
Base.metadata.create_all(bind=engine)

# Routers
app.include_router(museums.router)
app.include_router(restaurants.router)
app.include_router(events.router)
app.include_router(places.router)

@app.get("/")
def root():
    return {"message": "Welcome to AIstanbul API"}

@app.post("/ai")
async def ai_istanbul_router(request: Request):
    data = await request.json()
    user_input = data.get("user_input", "")
    
    try:
        # Import OpenAI client
        from openai import OpenAI
        import os
        
        # Debug logging
        print(f"Received user_input: '{user_input}' (length: {len(user_input)})")
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly, conversational AI assistant. Answer any question the user asks, whether about Istanbul or any other topic. If the user's message is vague or unclear, ask a clarifying question. Always try to keep the conversation going and be engaging."},
                {"role": "user", "content": user_input}
            ]
        )
        
        ai_response = response.choices[0].message.content
        return {"message": ai_response}
        
    except Exception as e:
        return {"error": "Internal server error", "details": str(e)}





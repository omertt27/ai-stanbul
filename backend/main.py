import fastapi
fastapi.__version__
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Hello Istanbul AI Guide!"}
from fastapi import FastAPI
from database import engine, SessionLocal
from models import Base

app = FastAPI(title="Istanbul AI Guide")

# Optional: Create tables if they don’t exist yet
Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"message": "Istanbul AI API is running!"}


# backend/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from models import Base

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL not found! Check your .env file.")

engine = create_engine(DB_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

# Create tables
Base.metadata.create_all(engine)

# test_db.py
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")  # make sure .env has correct DB URL

# Create engine
engine = create_engine(DB_URL, echo=True)

# Test connection and query
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("Database connected! Test query result:", result.scalar())
except Exception as e:
    print("Error connecting to database:", e)

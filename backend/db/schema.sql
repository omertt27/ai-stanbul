CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE
);

CREATE TABLE places (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    category VARCHAR(50),
    lat DOUBLE PRECISION,
    lng DOUBLE PRECISION
);

CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255),
    venue VARCHAR(255),
    date TIMESTAMP
);
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")  # e.g., postgres://user:pass@localhost:5432/istanbul_ai
engine = create_engine(DB_URL)
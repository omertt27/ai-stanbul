from db.base import Base
from database import engine
from models import User, Place, Museum, Restaurant, Event

Base.metadata.create_all(bind=engine)
print("Database tables created ✅")

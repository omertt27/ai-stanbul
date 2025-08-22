from database import Base, engine
from models import User, Place, Museum, Restaurant, Event

Base.metadata.create_all(bind=engine)
print("Database tables created ✅")

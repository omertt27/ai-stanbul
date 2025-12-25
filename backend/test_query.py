from database import SessionLocal
from sqlalchemy import text

session = SessionLocal()

# Test users
users = session.execute(text("SELECT * FROM users;")).fetchall()
print("Users:", users)

# Test places
places = session.execute(text("SELECT * FROM places;")).fetchall()
print("Places:", places)

# Test events
events = session.execute(text("SELECT * FROM events;")).fetchall()
print("Events:", events)

session.close()

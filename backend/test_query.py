from database import SessionLocal

session = SessionLocal()

# Test users
users = session.execute("SELECT * FROM users;").fetchall()
print("Users:", users)

# Test places
places = session.execute("SELECT * FROM places;").fetchall()
print("Places:", places)

# Test events
events = session.execute("SELECT * FROM events;").fetchall()
print("Events:", events)

session.close()

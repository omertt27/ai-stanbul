-- Sample users
INSERT INTO users (name, email) VALUES
('Alice', 'alice@example.com'),
('Bob', 'bob@example.com');

-- Sample places
INSERT INTO places (name, category, lat, lng) VALUES
('Galata Tower', 'Historic', 41.0256, 28.9744),
('Istiklal Street', 'Shopping', 41.0359, 28.9850);

-- Sample events
INSERT INTO events (title, venue, date) VALUES
('Istanbul Jazz Festival', 'Harbiye', '2025-09-15 19:00'),
('Art Exhibition', 'Istanbul Modern', '2025-08-30 10:00');

from sqlalchemy.orm import Session
from database import SessionLocal
from models import Museum  # you can rename to Place if you want more general

db = SessionLocal()

places = [
    {"name": "Sunny Fenerbahce Park", "location": "Kadikoy"},
    {"name": "Sunny Caddebostan seaside", "location": "Kadikoy"},
    {"name": "Bagdat Avenue", "location": "Kadikoy"},
    {"name": "Moda", "location": "Kadikoy"},
    {"name": "Kadikoy District", "location": "Kadikoy"},
    {"name": "Mihrimah Sultan Mosque", "location": "Uskudar"},
    {"name": "Maiden's Tower", "location": "Uskudar"},
    {"name": "Archeology Museum", "location": "Fatih"},
    {"name": "Nuruosmaniye Mosque", "location": "Fatih"},
    {"name": "Cagaloglu Hammam", "location": "Fatih"},
    {"name": "Egyptian Bazaar", "location": "Fatih"},
    {"name": "Suleymaniye Mosque", "location": "Fatih"},
    {"name": "Grand Bazaar", "location": "Fatih"},
    {"name": "Basilica Cistern", "location": "Fatih"},
    {"name": "Hagia Sophia Grand Mosque", "location": "Fatih"},
    {"name": "Turkish & Islamic Arts Museum", "location": "Fatih"},
    {"name": "The Blue Mosque", "location": "Fatih"},
    {"name": "Sultanahmet Square", "location": "Fatih"},
    {"name": "Topkapi Palace Museum", "location": "Fatih"},
    {"name": "Istanbul Museum of Modern Art", "location": "Beyoglu"},
    {"name": "Pera Museum", "location": "Beyoglu"},
    {"name": "Church of Saint Anthony of Padua", "location": "Beyoglu"},
    {"name": "Çiçek Pasajı", "location": "Beyoglu"},
    {"name": "Hagia Triada Greek Orthodox Church", "location": "Beyoglu"},
    {"name": "Galata Tower", "location": "Beyoglu"},
    {"name": "Taksim Square", "location": "Beyoglu"},
    {"name": "Istiklal Avenue", "location": "Beyoglu"},
    {"name": "Museum Of Illusions Istanbul", "location": "Beyoglu"},
    {"name": "Venerable Patriarchal Church of Saint George", "location": "Fatih"},
    {"name": "Dolmabahçe Palace Museum", "location": "Besiktas"},
    {"name": "National Painting Museum", "location": "Besiktas"},
    {"name": "Naval Museum", "location": "Besiktas"},
    {"name": "National Palaces Depot Museum", "location": "Besiktas"},
    {"name": "Büyük Mecidiye Mosque", "location": "Besiktas"},
    {"name": "Yıldız Park", "location": "Besiktas"},
    {"name": "Yıldız Hamidiye Mosque", "location": "Besiktas"},
    {"name": "Nisantasi", "location": "Sisli"},
    {"name": "Harbiye Military Museum", "location": "Sisli"},
    {"name": "Yapı Kredi Kültür Sanat Museum", "location": "Beyoglu"},
    {"name": "Roman Catholic Church of Santa Maria Draperis", "location": "Beyoglu"},
    {"name": "Taksim Mosque", "location": "Beyoglu"},
    {"name": "Rahmi M. Koç Museum", "location": "Beyoglu"},
    {"name": "İBB Maçka Democracy Park", "location": "Sisli"},
]

for p in places:
    db.merge(Museum(**p))  # or Place(**p) if you generalize
db.commit()
db.close()

print("✅ 43 places inserted successfully")

#!/usr/bin/env python3
"""
Seed script to populate the places table with Istanbul districts and attractions
"""

from sqlalchemy.orm import sessionmaker
from database import engine
from models import Place

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

# Sample places data for Istanbul districts
places_data = [
    # Uskudar
    {"name": "Maiden's Tower (Kız Kulesi)", "category": "Historical Site", "district": "Uskudar"},
    {"name": "Çamlıca Hill", "category": "Viewpoint", "district": "Uskudar"},
    {"name": "Mihrimah Sultan Mosque", "category": "Mosque", "district": "Uskudar"},
    {"name": "Leander's Tower", "category": "Historical Site", "district": "Uskudar"},
    {"name": "Semsi Pasha Mosque", "category": "Mosque", "district": "Uskudar"},
    {"name": "Atik Valide Mosque", "category": "Mosque", "district": "Uskudar"},
    
    # Kadikoy
    {"name": "Moda Park", "category": "Park", "district": "Kadikoy"},
    {"name": "Kadikoy Market", "category": "Market", "district": "Kadikoy"},
    {"name": "Bahariye Street", "category": "Shopping Street", "district": "Kadikoy"},
    {"name": "Fenerbahce Park", "category": "Park", "district": "Kadikoy"},
    {"name": "Moda Pier", "category": "Pier", "district": "Kadikoy"},
    {"name": "Kadikoy Fish Market", "category": "Market", "district": "Kadikoy"},
    
    # Sultanahmet
    {"name": "Hagia Sophia", "category": "Historical Site", "district": "Sultanahmet"},
    {"name": "Blue Mosque", "category": "Mosque", "district": "Sultanahmet"},
    {"name": "Topkapi Palace", "category": "Palace", "district": "Sultanahmet"},
    {"name": "Basilica Cistern", "category": "Historical Site", "district": "Sultanahmet"},
    {"name": "Grand Bazaar", "category": "Market", "district": "Sultanahmet"},
    
    # Beyoglu
    {"name": "Galata Tower", "category": "Tower", "district": "Beyoglu"},
    {"name": "Istiklal Street", "category": "Shopping Street", "district": "Beyoglu"},
    {"name": "Taksim Square", "category": "Square", "district": "Beyoglu"},
    {"name": "Pera Museum", "category": "Museum", "district": "Beyoglu"},
    {"name": "Galata Bridge", "category": "Bridge", "district": "Beyoglu"},
    
    # Besiktas
    {"name": "Dolmabahce Palace", "category": "Palace", "district": "Besiktas"},
    {"name": "Besiktas Park", "category": "Park", "district": "Besiktas"},
    {"name": "Naval Museum", "category": "Museum", "district": "Besiktas"},
    {"name": "Vodafone Park", "category": "Stadium", "district": "Besiktas"},
    
    # Ortakoy
    {"name": "Ortakoy Mosque", "category": "Mosque", "district": "Ortakoy"},
    {"name": "Bosphorus Bridge View", "category": "Viewpoint", "district": "Ortakoy"},
    {"name": "Ortakoy Market", "category": "Market", "district": "Ortakoy"},
    
    # Fatih
    {"name": "Suleymaniye Mosque", "category": "Mosque", "district": "Fatih"},
    {"name": "Chora Church", "category": "Historical Site", "district": "Fatih"},
    {"name": "Fener Greek Patriarch", "category": "Religious Site", "district": "Fatih"},
    {"name": "Golden Horn", "category": "Waterway", "district": "Fatih"},
    
    # Sariyer
    {"name": "Rumeli Fortress", "category": "Historical Site", "district": "Sariyer"},
    {"name": "Belgrade Forest", "category": "Forest", "district": "Sariyer"},
    {"name": "Emirgan Park", "category": "Park", "district": "Sariyer"},
    
    # Sisli
    {"name": "Cevahir Mall", "category": "Shopping Mall", "district": "Sisli"},
    {"name": "Military Museum", "category": "Museum", "district": "Sisli"},
    
    # Bakirkoy
    {"name": "Bakirkoy Beach", "category": "Beach", "district": "Bakirkoy"},
    {"name": "Ataturk Airport Museum", "category": "Museum", "district": "Bakirkoy"},
]

try:
    # Clear existing places first
    session.query(Place).delete()
    
    # Add all places
    for place_data in places_data:
        place = Place(
            name=place_data["name"],
            category=place_data["category"],
            district=place_data["district"]
        )
        session.add(place)
    
    session.commit()
    print(f"✅ Successfully added {len(places_data)} places to the database!")
    
    # Show summary
    districts = session.query(Place.district).distinct().all()
    print(f"\nAdded places for {len(districts)} districts:")
    for district in districts:
        count = session.query(Place).filter(Place.district == district[0]).count()
        print(f"- {district[0]}: {count} places")

except Exception as e:
    print(f"❌ Error: {e}")
    session.rollback()
finally:
    session.close()

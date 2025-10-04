#!/usr/bin/env python3
"""
Enhanced seed script to populate the places table with comprehensive Istanbul attractions
"""

from sqlalchemy.orm import sessionmaker
from database import engine
from models import Place

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

# Comprehensive places data from the user's list
enhanced_places_data = [
    # Kadikoy
    {"name": "Fenerbahce Park", "category": "Park", "district": "Kadikoy"},
    {"name": "Caddebostan Seaside", "category": "Seaside", "district": "Kadikoy"},
    {"name": "Bagdat Avenue", "category": "District", "district": "Kadikoy"},
    {"name": "Moda", "category": "District", "district": "Kadikoy"},
    
    # Uskudar
    {"name": "Mihrimah Sultan Mosque", "category": "Mosque", "district": "Uskudar"},
    {"name": "Maiden's Tower", "category": "Historical Site", "district": "Uskudar"},
    
    # Fatih
    {"name": "Archaeology Museum", "category": "Museum", "district": "Fatih"},
    {"name": "Nuruosmaniye Mosque", "category": "Mosque", "district": "Fatih"},
    {"name": "Cagaloglu Hammam", "category": "Historical Site", "district": "Fatih"},
    {"name": "Egyptian Bazaar", "category": "Historical Site", "district": "Fatih"},
    {"name": "Suleymaniye Mosque", "category": "Mosque", "district": "Fatih"},
    {"name": "Grand Bazaar", "category": "Historical Site", "district": "Fatih"},
    {"name": "Basilica Cistern", "category": "Historical Site", "district": "Fatih"},
    {"name": "Hagia Sophia Grand Mosque", "category": "Mosque", "district": "Fatih"},
    {"name": "Turkish & Islamic Arts Museum", "category": "Museum", "district": "Fatih"},
    {"name": "The Blue Mosque", "category": "Mosque", "district": "Fatih"},
    {"name": "Sultanahmet Square", "category": "Historical Site", "district": "Fatih"},
    {"name": "Topkapi Palace Museum", "category": "Museum", "district": "Fatih"},
    {"name": "Venerable Patriarchal Church of Saint George", "category": "Church", "district": "Fatih"},
    
    # Beyoglu
    {"name": "Istanbul Museum of Modern Art", "category": "Museum", "district": "Beyoglu"},
    {"name": "Pera Museum", "category": "Museum", "district": "Beyoglu"},
    {"name": "Church of Saint Anthony of Padua", "category": "Church", "district": "Beyoglu"},
    {"name": "Çiçek Pasajı", "category": "Historical Site", "district": "Beyoglu"},
    {"name": "Hagia Triada Greek Orthodox Church", "category": "Church", "district": "Beyoglu"},
    {"name": "Galata Tower", "category": "Historical Site", "district": "Beyoglu"},
    {"name": "Taksim Square", "category": "District", "district": "Beyoglu"},
    {"name": "Istıklal Avenue", "category": "District", "district": "Beyoglu"},
    {"name": "Museum Of Illusions Istanbul", "category": "Museum", "district": "Beyoglu"},
    {"name": "Yapı Kredi Kültür Sanat Museum", "category": "Museum", "district": "Beyoglu"},
    {"name": "Roman Catholic Church of Santa Maria Draperis", "category": "Church", "district": "Beyoglu"},
    {"name": "Taksim Mosque", "category": "Mosque", "district": "Beyoglu"},
    {"name": "Rahmi M. Koç Museum", "category": "Museum", "district": "Beyoglu"},
    
    # Besiktas
    {"name": "Dolmabahçe Palace", "category": "Museum", "district": "Besiktas"},
    {"name": "National Painting Museum", "category": "Museum", "district": "Besiktas"},
    {"name": "Naval Museum", "category": "Museum", "district": "Besiktas"},
    {"name": "National Palaces Depot", "category": "Museum", "district": "Besiktas"},
    {"name": "Büyük Mecidiye Mosque", "category": "Mosque", "district": "Besiktas"},
    {"name": "Yıldız Park", "category": "Park", "district": "Besiktas"},
    {"name": "Yıldız Hamidiye Mosque", "category": "Mosque", "district": "Besiktas"},
    
    # Sisli
    {"name": "Nisantasi", "category": "District", "district": "Sisli"},
    {"name": "Harbiye Military Museum", "category": "Museum", "district": "Sisli"},
    {"name": "İBB Maçka Democracy Park", "category": "Park", "district": "Sisli"},
]

try:
    # Clear existing places first to avoid duplicates
    print("🗑️ Clearing existing places...")
    session.query(Place).delete()
    
    # Add all enhanced places
    print(f"📍 Adding {len(enhanced_places_data)} new places...")
    for place_data in enhanced_places_data:
        place = Place(
            name=place_data["name"],
            category=place_data["category"],
            district=place_data["district"]
        )
        session.add(place)
    
    session.commit()
    print(f"✅ Successfully added {len(enhanced_places_data)} places to the database!")
    
    # Show summary by category and district
    print(f"\n📊 Places by Category:")
    from collections import defaultdict
    
    by_category = defaultdict(int)
    by_district = defaultdict(int)
    
    places = session.query(Place).all()
    for place in places:
        by_category[place.category] += 1
        by_district[place.district] += 1
    
    for category, count in sorted(by_category.items()):
        print(f"   {category}: {count}")
    
    print(f"\n📍 Places by District:")
    for district, count in sorted(by_district.items()):
        print(f"   {district}: {count}")

except Exception as e:
    print(f"❌ Error: {e}")
    session.rollback()
finally:
    session.close()

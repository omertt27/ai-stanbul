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

# Comprehensive places data for Istanbul
comprehensive_places_data = [
    # Historical Sites & Museums
    {"name": "Hagia Sophia", "category": "Historical Site", "district": "Sultanahmet"},
    {"name": "Blue Mosque", "category": "Mosque", "district": "Sultanahmet"},
    {"name": "Topkapi Palace", "category": "Palace", "district": "Sultanahmet"},
    {"name": "Basilica Cistern", "category": "Historical Site", "district": "Sultanahmet"},
    {"name": "Grand Bazaar", "category": "Market", "district": "Sultanahmet"},
    {"name": "Spice Bazaar", "category": "Market", "district": "Eminonu"},
    {"name": "Hippodrome", "category": "Historical Site", "district": "Sultanahmet"},
    {"name": "Galata Tower", "category": "Tower", "district": "Beyoglu"},
    {"name": "Dolmabahce Palace", "category": "Palace", "district": "Besiktas"},
    {"name": "Beylerbeyi Palace", "category": "Palace", "district": "Beylerbeyi"},
    {"name": "Rumeli Fortress", "category": "Historical Site", "district": "Sariyer"},
    {"name": "Anadolu Fortress", "category": "Historical Site", "district": "Beykoz"},
    {"name": "Chora Church", "category": "Historical Site", "district": "Fatih"},
    {"name": "Istanbul Archaeology Museum", "category": "Museum", "district": "Sultanahmet"},
    {"name": "Turkish and Islamic Arts Museum", "category": "Museum", "district": "Sultanahmet"},
    {"name": "Pera Museum", "category": "Museum", "district": "Beyoglu"},
    {"name": "Istanbul Modern", "category": "Museum", "district": "Beyoglu"},
    {"name": "Sakıp Sabancı Museum", "category": "Museum", "district": "Emirgan"},
    {"name": "Military Museum", "category": "Museum", "district": "Sisli"},
    {"name": "Naval Museum", "category": "Museum", "district": "Besiktas"},
    {"name": "Rahmi M. Koç Museum", "category": "Museum", "district": "Hasköy"},
    
    # Mosques & Religious Sites
    {"name": "Suleymaniye Mosque", "category": "Mosque", "district": "Fatih"},
    {"name": "Rustem Pasha Mosque", "category": "Mosque", "district": "Eminonu"},
    {"name": "Yeni Mosque", "category": "Mosque", "district": "Eminonu"},
    {"name": "Fatih Mosque", "category": "Mosque", "district": "Fatih"},
    {"name": "Eyup Sultan Mosque", "category": "Mosque", "district": "Eyup"},
    {"name": "Ortakoy Mosque", "category": "Mosque", "district": "Ortakoy"},
    {"name": "Mihrimah Sultan Mosque (Edirnekapi)", "category": "Mosque", "district": "Fatih"},
    {"name": "Mihrimah Sultan Mosque (Uskudar)", "category": "Mosque", "district": "Uskudar"},
    {"name": "Sokollu Mehmet Pasha Mosque", "category": "Mosque", "district": "Sultanahmet"},
    {"name": "Atik Valide Mosque", "category": "Mosque", "district": "Uskudar"},
    {"name": "Yeni Valide Mosque", "category": "Mosque", "district": "Eminonu"},
    {"name": "Semsi Pasha Mosque", "category": "Mosque", "district": "Uskudar"},
    {"name": "Greek Orthodox Patriarchate", "category": "Religious Site", "district": "Fener"},
    {"name": "Armenian Patriarchate", "category": "Religious Site", "district": "Kumkapi"},
    {"name": "St. Antoine Church", "category": "Religious Site", "district": "Beyoglu"},
    
    # Parks & Natural Areas
    {"name": "Gulhane Park", "category": "Park", "district": "Sultanahmet"},
    {"name": "Emirgan Park", "category": "Park", "district": "Emirgan"},
    {"name": "Yildiz Park", "category": "Park", "district": "Besiktas"},
    {"name": "Belgrade Forest", "category": "Forest", "district": "Sariyer"},
    {"name": "Camlica Hill", "category": "Viewpoint", "district": "Uskudar"},
    {"name": "Pierre Loti Hill", "category": "Viewpoint", "district": "Eyup"},
    {"name": "Moda Park", "category": "Park", "district": "Kadikoy"},
    {"name": "Fenerbahce Park", "category": "Park", "district": "Kadikoy"},
    {"name": "Maçka Park", "category": "Park", "district": "Sisli"},
    {"name": "Hidiv Pavilion", "category": "Viewpoint", "district": "Beykoz"},
    
    # Towers & Landmarks
    {"name": "Maiden's Tower", "category": "Tower", "district": "Uskudar"},
    {"name": "Leander's Tower", "category": "Tower", "district": "Uskudar"},
    {"name": "Camlica Tower", "category": "Tower", "district": "Uskudar"},
    
    # Bridges & Transportation Landmarks
    {"name": "Galata Bridge", "category": "Bridge", "district": "Eminonu"},
    {"name": "Golden Horn Bridge", "category": "Bridge", "district": "Balat"},
    {"name": "Bosphorus Bridge", "category": "Bridge", "district": "Ortakoy"},
    {"name": "Fatih Sultan Mehmet Bridge", "category": "Bridge", "district": "Sariyer"},
    {"name": "Yavuz Sultan Selim Bridge", "category": "Bridge", "district": "Sariyer"},
    
    # Shopping Areas
    {"name": "Istiklal Street", "category": "Shopping Street", "district": "Beyoglu"},
    {"name": "Bagdat Avenue", "category": "Shopping Street", "district": "Kadikoy"},
    {"name": "Nisantasi", "category": "Shopping District", "district": "Sisli"},
    {"name": "Kanyon Shopping Mall", "category": "Shopping Mall", "district": "Levent"},
    {"name": "Cevahir Mall", "category": "Shopping Mall", "district": "Sisli"},
    {"name": "Istinye Park", "category": "Shopping Mall", "district": "Sariyer"},
    {"name": "Zorlu Center", "category": "Shopping Mall", "district": "Besiktas"},
    {"name": "Arasta Bazaar", "category": "Market", "district": "Sultanahmet"},
    {"name": "Sahaflar Carsisi", "category": "Market", "district": "Beyazit"},
    {"name": "Kadikoy Fish Market", "category": "Market", "district": "Kadikoy"},
    {"name": "Besiktas Fish Market", "category": "Market", "district": "Besiktas"},
    
    # Squares & Public Areas
    {"name": "Taksim Square", "category": "Square", "district": "Beyoglu"},
    {"name": "Beyazit Square", "category": "Square", "district": "Beyazit"},
    {"name": "Sultanahmet Square", "category": "Square", "district": "Sultanahmet"},
    {"name": "Eminonu Square", "category": "Square", "district": "Eminonu"},
    {"name": "Kadikoy Square", "category": "Square", "district": "Kadikoy"},
    
    # Neighborhoods & Districts (for reference)
    {"name": "Balat", "category": "Historic Neighborhood", "district": "Fatih"},
    {"name": "Fener", "category": "Historic Neighborhood", "district": "Fatih"},
    {"name": "Kumkapi", "category": "Neighborhood", "district": "Fatih"},
    {"name": "Cihangir", "category": "Neighborhood", "district": "Beyoglu"},
    {"name": "Galata", "category": "Historic Neighborhood", "district": "Beyoglu"},
    {"name": "Karakoy", "category": "Neighborhood", "district": "Beyoglu"},
    {"name": "Bebek", "category": "Neighborhood", "district": "Besiktas"},
    {"name": "Arnavutkoy", "category": "Neighborhood", "district": "Besiktas"},
    {"name": "Kuzguncuk", "category": "Historic Neighborhood", "district": "Uskudar"},
    {"name": "Cengelkoy", "category": "Neighborhood", "district": "Uskudar"},
    {"name": "Moda", "category": "Neighborhood", "district": "Kadikoy"},
    
    # Waterfront & Piers
    {"name": "Eminonu Pier", "category": "Pier", "district": "Eminonu"},
    {"name": "Karakoy Pier", "category": "Pier", "district": "Beyoglu"},
    {"name": "Besiktas Pier", "category": "Pier", "district": "Besiktas"},
    {"name": "Uskudar Pier", "category": "Pier", "district": "Uskudar"},
    {"name": "Kadikoy Pier", "category": "Pier", "district": "Kadikoy"},
    {"name": "Ortakoy Pier", "category": "Pier", "district": "Ortakoy"},
    {"name": "Bebek Bay", "category": "Waterfront", "district": "Besiktas"},
    {"name": "Golden Horn", "category": "Waterway", "district": "Fatih"},
    {"name": "Bosphorus Strait", "category": "Waterway", "district": "Istanbul"},
    
    # Universities & Cultural Centers
    {"name": "Bogazici University", "category": "University", "district": "Bebek"},
    {"name": "Istanbul University", "category": "University", "district": "Beyazit"},
    {"name": "Galatasaray High School", "category": "School", "district": "Beyoglu"},
    {"name": "Lutfi Kirdar Convention Center", "category": "Convention Center", "district": "Sisli"},
    {"name": "Cemal Resit Rey Concert Hall", "category": "Concert Hall", "district": "Sisli"},
    {"name": "Atatürk Cultural Center", "category": "Cultural Center", "district": "Beyoglu"},
    
    # Hotels & Luxury Places (major landmarks)
    {"name": "Ciragan Palace Kempinski", "category": "Historic Hotel", "district": "Besiktas"},
    {"name": "Four Seasons Sultanahmet", "category": "Historic Hotel", "district": "Sultanahmet"},
    {"name": "Pera Palace Hotel", "category": "Historic Hotel", "district": "Beyoglu"},
    
    # Sports Venues
    {"name": "Vodafone Park", "category": "Stadium", "district": "Besiktas"},
    {"name": "Türk Telekom Stadium", "category": "Stadium", "district": "Sariyer"},
    {"name": "Fenerbahce Sukru Saracoglu Stadium", "category": "Stadium", "district": "Kadikoy"},
    
    # Baths & Traditional Places
    {"name": "Cagaloglu Hamam", "category": "Turkish Bath", "district": "Sultanahmet"},
    {"name": "Cemberlitas Hamam", "category": "Turkish Bath", "district": "Sultanahmet"},
    {"name": "Galatasaray Hamam", "category": "Turkish Bath", "district": "Beyoglu"},
    {"name": "Kilic Ali Pasha Hamam", "category": "Turkish Bath", "district": "Beyoglu"},
]

try:
    # Clear existing places first
    session.query(Place).delete()
    
    # Add all comprehensive places
    for place_data in comprehensive_places_data:
        place = Place(
            name=place_data["name"],
            category=place_data["category"],
            district=place_data["district"]
        )
        session.add(place)
    
    session.commit()
    print(f"✅ Successfully added {len(comprehensive_places_data)} places to the database!")
    
    # Show summary by category
    from collections import defaultdict
    
    categories = defaultdict(int)
    districts = defaultdict(int)
    
    all_places = session.query(Place).all()
    for place in all_places:
        categories[place.category] += 1
        districts[place.district] += 1
    
    print(f"\n📊 PLACES BY CATEGORY:")
    for category, count in sorted(categories.items()):
        print(f"   {category}: {count}")
    
    print(f"\n📍 PLACES BY DISTRICT:")
    for district, count in sorted(districts.items()):
        print(f"   {district}: {count}")
    
    print(f"\n🎯 TOTAL: {len(all_places)} places across {len(districts)} districts in {len(categories)} categories")

except Exception as e:
    print(f"❌ Error: {e}")
    session.rollback()
finally:
    session.close()

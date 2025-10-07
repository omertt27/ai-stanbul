"""
Seed Enhanced Attractions Database
Populate the enhanced_attractions table with Istanbul's key attractions including coordinates
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session
from backend.database import SessionLocal, engine, Base
from backend.models import EnhancedAttraction
from datetime import datetime

# Create tables
Base.metadata.create_all(bind=engine)

def seed_enhanced_attractions():
    """Seed the enhanced attractions table with curated Istanbul attractions"""
    db = SessionLocal()
    
    try:
        # Clear existing data
        db.query(EnhancedAttraction).delete()
        
        # Istanbul's must-visit attractions with accurate coordinates
        attractions_data = [
            # Historic Sites & Mosques
            {
                "name": "Hagia Sophia",
                "category": "historic_site",
                "subcategory": "museum",
                "address": "Sultan Ahmet, Ayasofya Meydanı No:1, 34122 Fatih/İstanbul",
                "district": "Sultanahmet",
                "coordinates_lat": 41.0086,
                "coordinates_lng": 28.9802,
                "popularity_score": 5.0,
                "estimated_visit_time_minutes": 90,
                "best_time_of_day": "morning",
                "crowd_level": "high",
                "description": "Iconic Byzantine cathedral turned mosque, UNESCO World Heritage Site",
                "opening_hours": "09:00-19:00 (closed during prayer times)",
                "price_range": "mid-range",
                "authenticity_score": 5.0
            },
            {
                "name": "Blue Mosque (Sultan Ahmed Mosque)",
                "category": "mosque",
                "subcategory": "historic",
                "address": "Sultan Ahmet, Atmeydanı Cd. No:7, 34122 Fatih/İstanbul",
                "district": "Sultanahmet",
                "coordinates_lat": 41.0054,
                "coordinates_lng": 28.9768,
                "popularity_score": 4.8,
                "estimated_visit_time_minutes": 60,
                "best_time_of_day": "morning",
                "crowd_level": "high",
                "description": "Famous 17th-century mosque with six minarets and blue tiles",
                "opening_hours": "08:30-18:00 (closed during prayer times)",
                "price_range": "free",
                "authenticity_score": 4.8
            },
            {
                "name": "Topkapi Palace",
                "category": "museum",
                "subcategory": "palace",
                "address": "Cankurtaran, 34122 Fatih/İstanbul",
                "district": "Sultanahmet",
                "coordinates_lat": 41.0115,
                "coordinates_lng": 28.9833,
                "popularity_score": 4.7,
                "estimated_visit_time_minutes": 120,
                "best_time_of_day": "morning",
                "crowd_level": "high",
                "description": "Ottoman imperial palace with treasury and harem",
                "opening_hours": "09:00-18:00 (closed Tuesdays)",
                "price_range": "mid-range",
                "authenticity_score": 4.9
            },
            
            # Basilica Cistern
            {
                "name": "Basilica Cistern",
                "category": "historic_site",
                "subcategory": "underground",
                "address": "Alemdar, Yerebatan Cd. 1/3, 34110 Fatih/İstanbul",
                "district": "Sultanahmet",
                "coordinates_lat": 41.0084,
                "coordinates_lng": 28.9777,
                "popularity_score": 4.5,
                "estimated_visit_time_minutes": 45,
                "best_time_of_day": "any",
                "crowd_level": "medium",
                "description": "Ancient underground cistern with atmospheric lighting",
                "opening_hours": "09:00-17:30",
                "price_range": "budget",
                "authenticity_score": 4.8
            },
            
            # Grand Bazaar & Spice Bazaar
            {
                "name": "Grand Bazaar",
                "category": "bazaar",
                "subcategory": "shopping",
                "address": "Beyazıt, 34126 Fatih/İstanbul",
                "district": "Beyazit",
                "coordinates_lat": 41.0108,
                "coordinates_lng": 28.9678,
                "popularity_score": 4.2,
                "estimated_visit_time_minutes": 90,
                "best_time_of_day": "afternoon",
                "crowd_level": "high",
                "description": "Historic covered market with 4,000 shops",
                "opening_hours": "09:00-19:00 (closed Sundays)",
                "price_range": "varies",
                "authenticity_score": 3.8
            },
            {
                "name": "Spice Bazaar (Egyptian Bazaar)",
                "category": "bazaar",
                "subcategory": "food",
                "address": "Rüstem Paşa, Erzak Ambarı Sok. No:92, 34116 Fatih/İstanbul",
                "district": "Eminonu",
                "coordinates_lat": 41.0166,
                "coordinates_lng": 28.9706,
                "popularity_score": 4.3,
                "estimated_visit_time_minutes": 60,
                "best_time_of_day": "afternoon",
                "crowd_level": "medium",
                "description": "Historic spice market with aromatic Turkish delights",
                "opening_hours": "08:00-19:00",
                "price_range": "budget",
                "authenticity_score": 4.2
            },
            
            # Galata & Beyoglu
            {
                "name": "Galata Tower",
                "category": "viewpoint",
                "subcategory": "historic",
                "address": "Bereketzade, Galata Kulesi Sk., 34421 Beyoğlu/İstanbul",
                "district": "Galata",
                "coordinates_lat": 41.0256,
                "coordinates_lng": 28.9744,
                "popularity_score": 4.4,
                "estimated_visit_time_minutes": 75,
                "best_time_of_day": "evening",
                "crowd_level": "high",
                "description": "Medieval tower with panoramic city views",
                "opening_hours": "08:30-23:00",
                "price_range": "mid-range",
                "authenticity_score": 4.1
            },
            {
                "name": "Istiklal Avenue",
                "category": "shopping",
                "subcategory": "street",
                "address": "İstiklal Cd., Beyoğlu/İstanbul",
                "district": "Beyoglu",
                "coordinates_lat": 41.0362,
                "coordinates_lng": 28.9751,
                "popularity_score": 4.1,
                "estimated_visit_time_minutes": 120,
                "best_time_of_day": "evening",
                "crowd_level": "high",
                "description": "Bustling pedestrian street with shops and cafes",
                "opening_hours": "24/7",
                "price_range": "varies",
                "authenticity_score": 3.5
            },
            
            # Dolmabahce Palace
            {
                "name": "Dolmabahce Palace",
                "category": "museum",
                "subcategory": "palace",
                "address": "Vişnezade, Dolmabahçe Cd., 34357 Beşiktaş/İstanbul",
                "district": "Besiktas",
                "coordinates_lat": 41.0391,
                "coordinates_lng": 29.0002,
                "popularity_score": 4.6,
                "estimated_visit_time_minutes": 100,
                "best_time_of_day": "morning",
                "crowd_level": "medium",
                "description": "Opulent 19th-century Ottoman palace on the Bosphorus",
                "opening_hours": "09:00-16:00 (closed Mondays)",
                "price_range": "mid-range",
                "authenticity_score": 4.7
            },
            
            # Asian Side Attractions
            {
                "name": "Maiden's Tower",
                "category": "historic_site",
                "subcategory": "tower",
                "address": "Üsküdar/İstanbul (Bosphorus)",
                "district": "Uskudar",
                "coordinates_lat": 41.0213,
                "coordinates_lng": 29.0043,
                "popularity_score": 4.2,
                "estimated_visit_time_minutes": 90,
                "best_time_of_day": "evening",
                "crowd_level": "low",
                "description": "Historic tower on small island with restaurant",
                "opening_hours": "09:00-19:00",
                "price_range": "expensive",
                "authenticity_score": 4.5
            },
            {
                "name": "Kadikoy Market",
                "category": "market",
                "subcategory": "local",
                "address": "Kadıköy Çarşısı, Kadıköy/İstanbul",
                "district": "Kadikoy",
                "coordinates_lat": 40.9904,
                "coordinates_lng": 29.0263,
                "popularity_score": 4.0,
                "estimated_visit_time_minutes": 80,
                "best_time_of_day": "morning",
                "crowd_level": "medium",
                "description": "Authentic local market with fresh produce and Turkish specialties",
                "opening_hours": "08:00-20:00",
                "price_range": "budget",
                "authenticity_score": 4.8
            },
            
            # Restaurants & Cafes
            {
                "name": "Pandeli Restaurant",
                "category": "restaurant",
                "subcategory": "ottoman",
                "address": "Rüstem Paşa, Erzak Ambarı Sok. No:1, 34116 Fatih/İstanbul",
                "district": "Eminonu",
                "coordinates_lat": 41.0170,
                "coordinates_lng": 28.9701,
                "popularity_score": 4.3,
                "estimated_visit_time_minutes": 90,
                "best_time_of_day": "afternoon",
                "crowd_level": "medium",
                "description": "Historic Ottoman restaurant above Spice Bazaar",
                "opening_hours": "12:00-17:00 (closed Sundays)",
                "price_range": "expensive",
                "authenticity_score": 4.7
            },
            {
                "name": "Ciya Sofrasi",
                "category": "restaurant",
                "subcategory": "traditional",
                "address": "Caferağa, Güneşli Bahçe Sk. No:43, 34710 Kadıköy/İstanbul",
                "district": "Kadikoy",
                "coordinates_lat": 40.9908,
                "coordinates_lng": 29.0280,
                "popularity_score": 4.5,
                "estimated_visit_time_minutes": 75,
                "best_time_of_day": "afternoon",
                "crowd_level": "medium",
                "description": "Famous for authentic regional Turkish cuisine",
                "opening_hours": "12:00-22:00",
                "price_range": "mid-range",
                "authenticity_score": 4.9
            },
            
            # Coffee & Tea
            {
                "name": "Pierre Loti Cafe",
                "category": "cafe",
                "subcategory": "historic",
                "address": "Karyağdı Sk., 34445 Beyoğlu/İstanbul",
                "district": "Eyup",
                "coordinates_lat": 41.0452,
                "coordinates_lng": 28.9395,
                "popularity_score": 4.1,
                "estimated_visit_time_minutes": 60,
                "best_time_of_day": "afternoon",
                "crowd_level": "low",
                "description": "Historic hilltop cafe with Golden Horn views",
                "opening_hours": "08:00-24:00",
                "price_range": "budget",
                "authenticity_score": 4.3
            },
            
            # Parks & Views
            {
                "name": "Gulhane Park",
                "category": "park",
                "subcategory": "historic",
                "address": "Cankurtaran, Kennedy Cd., 34122 Fatih/İstanbul",
                "district": "Sultanahmet",
                "coordinates_lat": 41.0129,
                "coordinates_lng": 28.9816,
                "popularity_score": 3.8,
                "estimated_visit_time_minutes": 45,
                "best_time_of_day": "morning",
                "crowd_level": "low",
                "description": "Historic park near Topkapi Palace with tulip gardens",
                "opening_hours": "24/7",
                "price_range": "free",
                "authenticity_score": 4.2
            },
            {
                "name": "Emirgan Park",
                "category": "park",
                "subcategory": "nature",
                "address": "Emirgan, 34467 Sarıyer/İstanbul",
                "district": "Sariyer",
                "coordinates_lat": 41.1069,
                "coordinates_lng": 29.0533,
                "popularity_score": 4.0,
                "estimated_visit_time_minutes": 90,
                "best_time_of_day": "morning",
                "crowd_level": "low",
                "description": "Large park famous for tulip festival and Bosphorus views",
                "opening_hours": "24/7",
                "price_range": "free",
                "authenticity_score": 4.4
            },
            
            # Transportation Hubs (useful for route planning)
            {
                "name": "Eminonu Ferry Terminal",
                "category": "transportation",
                "subcategory": "ferry",
                "address": "Eminönü, 34110 Fatih/İstanbul",
                "district": "Eminonu",
                "coordinates_lat": 41.0177,
                "coordinates_lng": 28.9706,
                "popularity_score": 3.5,
                "estimated_visit_time_minutes": 15,
                "best_time_of_day": "any",
                "crowd_level": "medium",
                "description": "Main ferry terminal connecting European and Asian sides",
                "opening_hours": "06:00-23:00",
                "price_range": "budget",
                "authenticity_score": 4.0
            },
            {
                "name": "Galata Bridge",
                "category": "viewpoint",
                "subcategory": "bridge",
                "address": "Kemankeş Karamustafa Paşa, Galata Köprüsü, 34425 Beyoğlu/İstanbul",
                "district": "Karakoy",
                "coordinates_lat": 41.0192,
                "coordinates_lng": 28.9738,
                "popularity_score": 4.2,
                "estimated_visit_time_minutes": 30,
                "best_time_of_day": "evening",
                "crowd_level": "medium",
                "description": "Iconic bridge connecting old and new city with fishing",
                "opening_hours": "24/7",
                "price_range": "free",
                "authenticity_score": 4.5
            },
            
            # Hidden Gems
            {
                "name": "Balat Neighborhood",
                "category": "neighborhood",
                "subcategory": "historic",
                "address": "Balat, Fatih/İstanbul",
                "district": "Balat",
                "coordinates_lat": 41.0290,
                "coordinates_lng": 28.9492,
                "popularity_score": 3.9,
                "estimated_visit_time_minutes": 120,
                "best_time_of_day": "afternoon",
                "crowd_level": "low",
                "description": "Colorful historic neighborhood with Ottoman houses",
                "opening_hours": "24/7",
                "price_range": "free",
                "authenticity_score": 4.8
            },
            {
                "name": "Chora Church",
                "category": "museum",
                "subcategory": "church",
                "address": "Dervişali, Kariye Camii Sk. No:18, 34087 Fatih/İstanbul",
                "district": "Edirnekapi",
                "coordinates_lat": 41.0306,
                "coordinates_lng": 28.9389,
                "popularity_score": 4.4,
                "estimated_visit_time_minutes": 60,
                "best_time_of_day": "morning",
                "crowd_level": "low",
                "description": "Byzantine church with exceptional mosaics and frescoes",
                "opening_hours": "09:00-17:00 (closed Wednesdays)",
                "price_range": "budget",
                "authenticity_score": 4.9
            }
        ]
        
        # Insert attractions
        for attr_data in attractions_data:
            attraction = EnhancedAttraction(**attr_data)
            db.add(attraction)
        
        db.commit()
        print(f"✅ Successfully seeded {len(attractions_data)} enhanced attractions")
        
        # Verify the data
        count = db.query(EnhancedAttraction).count()
        print(f"📊 Total attractions in database: {count}")
        
        # Show category breakdown
        categories = db.query(EnhancedAttraction.category.distinct()).all()
        print(f"🏷️ Categories: {[cat[0] for cat in categories]}")
        
    except Exception as e:
        print(f"❌ Error seeding attractions: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🌱 Seeding Enhanced Attractions Database...")
    seed_enhanced_attractions()
    print("✅ Seeding complete!")

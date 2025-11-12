#!/usr/bin/env python3
"""
Final Database Seed Script - Handles all data type conversions properly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import SessionLocal
from models import Restaurant, Museum, Place
from sqlalchemy import text
import json
import re

def price_to_float(price_str):
    """Convert price string to float"""
    if not price_str or price_str.lower() in ['free', 'n/a', 'none']:
        return 0.0
    
    # Extract first number
    match = re.search(r'\d+', str(price_str))
    if match:
        return float(match.group())
    return 0.0

def price_level_to_int(price_str):
    """Convert price string ($$, $$$) to integer (1-4)"""
    if not price_str:
        return 2
    
    price_map = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
    return price_map.get(str(price_str).strip(), 2)

def seed_museums():
    print("🏛️ Seeding 40+ Museums...")
    
    try:
        from accurate_museum_database import IstanbulMuseumDatabase
        
        db = SessionLocal()
        museum_db = IstanbulMuseumDatabase()
        
        added = 0
        for museum_id, m in museum_db.museums.items():
            if db.query(Museum).filter_by(name=m.name).first():
                continue
            
            museum = Museum(
                name=m.name,
                location=m.location,
                hours=json.dumps(m.opening_hours) if m.opening_hours else None,
                ticket_price=price_to_float(m.entrance_fee),  # Convert to float
                highlights=json.dumps(m.must_see_highlights[:5]) if m.must_see_highlights else None
            )
            db.add(museum)
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} museums")
        return added
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0

def seed_restaurants():
    print("🍽️ Seeding 300+ Restaurants...")
    
    try:
        from services.restaurant_database_service import RestaurantDatabaseService
        
        db = SessionLocal()
        service = RestaurantDatabaseService()
        
        all_rests = service.restaurants[:200]  # Seed first 200
        
        added = 0
        for rest in all_rests:
            if db.query(Restaurant).filter_by(name=rest['name']).first():
                continue
            
            restaurant = Restaurant(
                name=rest['name'],
                cuisine=rest.get('cuisine', 'Turkish'),
                location=rest.get('location', rest.get('neighborhood', 'Istanbul')),
                price_level=price_level_to_int(rest.get('price_range')),  # Convert to int
                rating=float(rest.get('rating', 4.0)),
                description=rest.get('description', '')[:300] if rest.get('description') else None,
                website=rest.get('website'),
                phone=rest.get('phone'),
                place_id=rest.get('place_id')
            )
            db.add(restaurant)
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} restaurants")
        return added
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0

def seed_attractions():
    print("🏞️ Seeding 60+ Attractions...")
    
    try:
        from istanbul_attractions_system import IstanbulAttractionsSystem
        
        db = SessionLocal()
        system = IstanbulAttractionsSystem()
        
        added = 0
        # Handle both list and dict formats
        attractions = system.attractions if isinstance(system.attractions, list) else list(system.attractions.values())
        
        for attr in attractions:
            name = attr['name'] if isinstance(attr, dict) else attr.name
            
            if db.query(Place).filter_by(name=name).first():
                continue
            
            if isinstance(attr, dict):
                place = Place(
                    name=attr['name'],
                    category=attr.get('category', 'attraction'),
                    district=attr.get('district', attr.get('location', 'Istanbul'))
                )
            else:
                place = Place(
                    name=attr.name,
                    category=getattr(attr, 'category', 'attraction'),
                    district=getattr(attr, 'district', getattr(attr, 'location', 'Istanbul'))
                )
            
            db.add(place)
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} attractions")
        return added
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0

def verify_data():
    print("\n" + "=" * 60)
    print("📊 FINAL DATABASE SUMMARY")
    print("=" * 60)
    
    db = SessionLocal()
    
    results = {}
    tables = [
        ("Restaurants", "restaurants"),
        ("Museums", "museums"),
        ("Events", "events"),
        ("Places", "places"),
        ("Enhanced Attractions", "enhanced_attractions"),
    ]
    
    for label, table in tables:
        result = db.execute(text(f"SELECT COUNT(*) FROM {table}")).first()
        count = result[0] if result else 0
        results[label] = count
        
        emoji = "✅" if count >= 20 else "⚠️" if count > 0 else "❌"
        print(f"{emoji} {label:30} {count:>5} records")
    
    db.close()
    print("=" * 60 + "\n")
    
    return results

def main():
    print("\n" + "=" * 60)
    print("🌱 FINAL DATABASE SEEDING - ALL DATASETS")
    print("=" * 60 + "\n")
    
    museums_added = seed_museums()
    restaurants_added = seed_restaurants()
    attractions_added = seed_attractions()
    
    results = verify_data()
    
    total_added = museums_added + restaurants_added + attractions_added
    
    if total_added > 0:
        print(f"✅ Added {total_added} new records!")
    
    # Check if we have enough data
    if results.get("Restaurants", 0) >= 50 and results.get("Museums", 0) >= 20:
        print("🎉 Database has sufficient data for Pure LLM Architecture!")
        print("🚀 Ready to proceed with Phase 3!")
        return 0
    else:
        print("⚠️ Some tables still need more data")
        return 1

if __name__ == "__main__":
    sys.exit(main())

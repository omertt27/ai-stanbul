#!/usr/bin/env python3
"""
Quick Database Seed Script - Load Large Datasets
Seeds museums, restaurants, places, and events into PostgreSQL
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import SessionLocal
from models import Restaurant, Museum, Place
from sqlalchemy import text

def seed_museums():
    """Seed 40+ museums from accurate_museum_database.py"""
    print("🏛️ Seeding Museums from accurate_museum_database.py...")
    
    try:
        from accurate_museum_database import IstanbulMuseumDatabase
        
        db = SessionLocal()
        museum_db = IstanbulMuseumDatabase()
        
        added = 0
        skipped = 0
        
        for museum_id, museum_info in museum_db.museums.items():
            # Check if already exists
            existing = db.query(Museum).filter_by(name=museum_info.name).first()
            if existing:
                skipped += 1
                continue
            
            museum = Museum(
                name=museum_info.name,
                location=museum_info.location,
                description=museum_info.historical_significance[:500] if museum_info.historical_significance else "",
                opening_hours=str(museum_info.opening_hours),
                entrance_fee=museum_info.entrance_fee,
                latitude=41.0082,
                longitude=28.9784
            )
            db.add(museum)
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} museums (skipped {skipped} existing)")
        return added
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def seed_events_directly():
    """Seed events directly using SQL"""
    print("🎭 Seeding Events...")
    
    try:
        from data.events_database import get_all_events
        
        db = SessionLocal()
        events = get_all_events()
        
        added = 0
        for event in events[:50]:  # Add first 50 events
            title = event.get('title', event.get('name', 'Untitled'))
            venue = event.get('location', event.get('venue', 'Istanbul'))
            date = event.get('date', event.get('start_date'))
            
            # Check if exists
            result = db.execute(
                text("SELECT id FROM events WHERE title = :title"),
                {"title": title}
            ).first()
            
            if result:
                continue
            
            # Insert
            db.execute(
                text("INSERT INTO events (title, venue, date) VALUES (:title, :venue, :date)"),
                {"title": title, "venue": venue, "date": date}
            )
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} events")
        return added
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def seed_restaurants_from_service():
    """Seed restaurants from restaurant database service"""
    print("🍽️ Seeding Restaurants...")
    
    try:
        from services.restaurant_database_service import RestaurantDatabaseService
        
        db = SessionLocal()
        service = RestaurantDatabaseService()
        all_restaurants = service.get_all_restaurants()
        
        added = 0
        skipped = 0
        
        for rest in all_restaurants[:200]:  # Add first 200
            existing = db.query(Restaurant).filter_by(name=rest['name']).first()
            if existing:
                skipped += 1
                continue
            
            restaurant = Restaurant(
                name=rest['name'],
                cuisine=rest.get('cuisine', 'Turkish'),
                location=rest.get('location', rest.get('neighborhood', 'Istanbul')),
                price_range=rest.get('price_range', '$$'),
                rating=rest.get('rating', 4.0),
                description=rest.get('description', '')[:500],
                latitude=rest.get('latitude', 41.0082),
                longitude=rest.get('longitude', 28.9784)
            )
            db.add(restaurant)
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} restaurants (skipped {skipped} existing)")
        return added
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def seed_attractions():
    """Seed attractions/places"""
    print("🏞️ Seeding Attractions & Places...")
    
    try:
        # Seed from Istanbul attractions system
        from istanbul_attractions_system import IstanbulAttractionsSystem
        
        db = SessionLocal()
        attractions_system = IstanbulAttractionsSystem()
        
        added = 0
        skipped = 0
        
        for attraction in attractions_system.attractions:
            existing = db.query(Place).filter_by(name=attraction['name']).first()
            if existing:
                skipped += 1
                continue
            
            place = Place(
                name=attraction['name'],
                description=attraction.get('description', '')[:500],
                location=attraction.get('district', 'Istanbul'),
                category=attraction.get('category', 'attraction'),
                latitude=attraction.get('latitude', 41.0082),
                longitude=attraction.get('longitude', 28.9784)
            )
            db.add(place)
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} attractions (skipped {skipped} existing)")
        return added
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def seed_hidden_gems():
    """Seed hidden gems"""
    print("💎 Seeding Hidden Gems...")
    
    try:
        from data.hidden_gems_database import get_all_hidden_gems
        
        db = SessionLocal()
        hidden_gems = get_all_hidden_gems()
        
        added = 0
        skipped = 0
        
        for gem in hidden_gems:
            existing = db.query(Place).filter_by(name=gem.get('name')).first()
            if existing:
                skipped += 1
                continue
            
            place = Place(
                name=gem.get('name'),
                description=gem.get('description', '')[:500],
                location=gem.get('neighborhood', 'Istanbul'),
                category='hidden_gem',
                latitude=gem.get('latitude', 41.0082),
                longitude=gem.get('longitude', 28.9784)
            )
            db.add(place)
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} hidden gems (skipped {skipped} existing)")
        return added
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def verify_data():
    """Show final counts"""
    print("\n" + "=" * 60)
    print("📊 DATABASE SUMMARY")
    print("=" * 60)
    
    db = SessionLocal()
    
    tables = [
        ("Restaurants", "restaurants"),
        ("Museums", "museums"),
        ("Events", "events"),
        ("Places", "places"),
        ("Enhanced Attractions", "enhanced_attractions"),
        ("Chat History", "chat_history")
    ]
    
    for label, table in tables:
        result = db.execute(text(f"SELECT COUNT(*) FROM {table}")).first()
        count = result[0] if result else 0
        emoji = "✅" if count > 10 else "⚠️" if count > 0 else "❌"
        print(f"{emoji} {label:25} {count:>5} records")
    
    db.close()
    print("=" * 60 + "\n")

def main():
    print("\n" + "=" * 60)
    print("🌱 SEEDING LARGE DATASETS INTO POSTGRESQL")
    print("=" * 60 + "\n")
    
    # Seed all data
    museums_added = seed_museums()
    events_added = seed_events_directly()
    restaurants_added = seed_restaurants_from_service()
    attractions_added = seed_attractions()
    gems_added = seed_hidden_gems()
    
    # Show summary
    verify_data()
    
    total = museums_added + events_added + restaurants_added + attractions_added + gems_added
    
    if total > 0:
        print(f"✅ Successfully added {total} total records!")
        print("🚀 Database is ready for LLM integration!")
        return 0
    else:
        print("⚠️ No new data was added. Database may already be seeded.")
        return 0

if __name__ == "__main__":
    sys.exit(main())

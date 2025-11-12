#!/usr/bin/env python3
"""
Comprehensive Data Seeding Script
Seeds all available datasets into the database
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import SessionLocal, engine, Base
from models import Restaurant, Museum, Event, Place
from sqlalchemy import text
import json

def seed_museums():
    """Seed museums from accurate_museum_database.py"""
    print("🏛️ Seeding Museums...")
    
    try:
        from accurate_museum_database import IstanbulMuseumDatabase
        
        db = SessionLocal()
        museum_db = IstanbulMuseumDatabase()
        
        count = 0
        for museum_id, museum_info in museum_db.museums.items():
            # Check if museum already exists
            existing = db.query(Museum).filter_by(name=museum_info.name).first()
            if existing:
                continue
            
            museum = Museum(
                name=museum_info.name,
                location=museum_info.location,
                description=museum_info.historical_significance,
                opening_hours=str(museum_info.opening_hours),
                entrance_fee=museum_info.entrance_fee,
                latitude=41.0082,  # Default Istanbul coordinates
                longitude=28.9784
            )
            db.add(museum)
            count += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {count} new museums")
        
    except Exception as e:
        print(f"   ❌ Error seeding museums: {e}")

def seed_events():
    """Seed events from events_database.py"""
    print("🎭 Seeding Events...")
    
    try:
        from data.events_database import get_all_events
        
        db = SessionLocal()
        events = get_all_events()
        
        count = 0
        for event_data in events:
            # Check if event already exists
            existing = db.query(Event).filter_by(title=event_data.get('title', event_data.get('name'))).first()
            if existing:
                continue
            
            event = Event(
                title=event_data.get('title', event_data.get('name')),
                description=event_data.get('description', ''),
                date=event_data.get('date', event_data.get('start_date', '')),
                location=event_data.get('location', event_data.get('venue', '')),
                category=event_data.get('category', 'cultural')
            )
            db.add(event)
            count += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {count} new events")
        
    except Exception as e:
        print(f"   ❌ Error seeding events: {e}")

def seed_restaurants():
    """Seed restaurants from restaurant_database_service.py"""
    print("🍽️ Seeding Restaurants...")
    
    try:
        from services.restaurant_database_service import RestaurantDatabaseService
        
        db = SessionLocal()
        restaurant_service = RestaurantDatabaseService()
        restaurants_data = restaurant_service.get_all_restaurants()
        
        count = 0
        for restaurant_data in restaurants_data[:100]:  # Seed first 100
            # Check if restaurant already exists
            existing = db.query(Restaurant).filter_by(name=restaurant_data['name']).first()
            if existing:
                continue
            
            restaurant = Restaurant(
                name=restaurant_data['name'],
                cuisine=restaurant_data.get('cuisine', 'Turkish'),
                location=restaurant_data.get('location', 'Istanbul'),
                price_range=restaurant_data.get('price_range', '$$'),
                rating=restaurant_data.get('rating', 4.0),
                description=restaurant_data.get('description', ''),
                latitude=restaurant_data.get('latitude', 41.0082),
                longitude=restaurant_data.get('longitude', 28.9784)
            )
            db.add(restaurant)
            count += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {count} new restaurants")
        
    except Exception as e:
        print(f"   ❌ Error seeding restaurants: {e}")

def seed_places():
    """Seed places/attractions"""
    print("🏞️ Seeding Places & Attractions...")
    
    try:
        # Use seed_enhanced_attractions.py data
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        
        # Try to import from seed scripts
        try:
            exec(open('seed_enhanced_attractions.py').read())
            print("   ✅ Executed seed_enhanced_attractions.py")
        except Exception as e:
            print(f"   ⚠️  Could not execute seed_enhanced_attractions.py: {e}")
        
        # Count total places
        db = SessionLocal()
        count = db.query(Place).count()
        db.close()
        print(f"   ✅ Total places in database: {count}")
        
    except Exception as e:
        print(f"   ❌ Error seeding places: {e}")

def seed_hidden_gems():
    """Seed hidden gems into places or enhanced_attractions"""
    print("💎 Seeding Hidden Gems...")
    
    try:
        from data.hidden_gems_database import get_all_hidden_gems
        
        db = SessionLocal()
        hidden_gems = get_all_hidden_gems()
        
        count = 0
        for gem_data in hidden_gems:
            # Check if gem already exists as a place
            existing = db.query(Place).filter_by(name=gem_data.get('name')).first()
            if existing:
                continue
            
            place = Place(
                name=gem_data.get('name'),
                description=gem_data.get('description', ''),
                location=gem_data.get('neighborhood', ''),
                category=gem_data.get('category', 'hidden_gem'),
                latitude=gem_data.get('latitude', 41.0082),
                longitude=gem_data.get('longitude', 28.9784)
            )
            db.add(place)
            count += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {count} new hidden gems")
        
    except Exception as e:
        print(f"   ❌ Error seeding hidden gems: {e}")

def verify_data():
    """Verify all data was seeded correctly"""
    print("\n📊 Verifying Database Contents...")
    
    db = SessionLocal()
    
    counts = {
        'restaurants': db.query(Restaurant).count(),
        'museums': db.query(Museum).count(),
        'events': db.query(Event).count(),
        'places': db.query(Place).count(),
    }
    
    db.close()
    
    print("\n" + "=" * 50)
    print("DATABASE SUMMARY")
    print("=" * 50)
    for table, count in counts.items():
        emoji = "✅" if count > 0 else "⚠️"
        print(f"{emoji} {table.capitalize()}: {count}")
    print("=" * 50 + "\n")
    
    return all(count > 0 for count in counts.values())

def main():
    """Main seeding function"""
    print("=" * 50)
    print("🌱 COMPREHENSIVE DATA SEEDING")
    print("=" * 50)
    print()
    
    # Create all tables if they don't exist
    Base.metadata.create_all(engine)
    
    # Seed all data
    seed_museums()
    seed_events()
    seed_restaurants()
    seed_places()
    seed_hidden_gems()
    
    # Verify
    success = verify_data()
    
    if success:
        print("✅ All data seeded successfully!")
        print("🚀 Database is ready for production!")
    else:
        print("⚠️ Some tables are still empty. Check errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

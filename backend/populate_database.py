#!/usr/bin/env python3
"""
Database Population Script
Populates Render PostgreSQL with Istanbul data from JSON files
"""

import json
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import SessionLocal, engine
from models import Base, Restaurant, Museum, Event
from sqlalchemy.exc import IntegrityError

def create_tables():
    """Create all tables if they don't exist"""
    print("📦 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created/verified")

def load_restaurants():
    """Load restaurants from JSON file"""
    print("\n🍽️  Loading restaurants...")
    
    data_file = backend_dir / 'data' / 'restaurants_database.json'
    if not data_file.exists():
        print(f"⚠️  File not found: {data_file}")
        return 0
    
    db = SessionLocal()
    count = 0
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            restaurants = data.get('restaurants', [])
            
            for rest_data in restaurants:
                try:
                    # Map JSON fields to model fields
                    restaurant = Restaurant(
                        name=rest_data.get('name'),
                        cuisine=rest_data.get('cuisine_type') or rest_data.get('cuisine'),
                        district=rest_data.get('district'),
                        address=rest_data.get('address'),
                        price_range=rest_data.get('budget') or rest_data.get('price_range'),  # $, $$, $$$
                        rating=float(rest_data.get('rating', 0)) if rest_data.get('rating') else None,
                        reviews_count=rest_data.get('reviews_count'),
                        latitude=rest_data.get('latitude'),
                        longitude=rest_data.get('longitude'),
                        phone=rest_data.get('phone'),
                        website=rest_data.get('website'),
                        description=rest_data.get('description'),
                        language='en'
                    )
                    db.add(restaurant)
                    count += 1
                    
                    # Commit in batches of 50
                    if count % 50 == 0:
                        db.commit()
                        print(f"  ✓ {count} restaurants loaded...")
                        
                except Exception as e:
                    print(f"  ⚠️  Skipped restaurant: {rest_data.get('name')} - {e}")
                    continue
            
            db.commit()
            print(f"✅ Loaded {count} restaurants total")
            
    except Exception as e:
        print(f"❌ Error loading restaurants: {e}")
        db.rollback()
    finally:
        db.close()
    
    return count

def load_attractions():
    """Load museums and attractions from JSON file"""
    print("\n🏛️  Loading attractions...")
    
    data_file = backend_dir / 'data' / 'attractions_database_expanded.json'
    if not data_file.exists():
        print(f"⚠️  File not found: {data_file}")
        return 0
    
    db = SessionLocal()
    count = 0
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            attractions = data.get('attractions', [])
            
            for attr_data in attractions:
                try:
                    museum = Museum(
                        name=attr_data.get('name'),
                        category=attr_data.get('category') or attr_data.get('type'),
                        district=attr_data.get('district'),
                        address=attr_data.get('address'),
                        entry_fee=attr_data.get('entry_fee') or attr_data.get('price'),
                        rating=float(attr_data.get('rating', 0)) if attr_data.get('rating') else None,
                        description=attr_data.get('description'),
                        opening_hours=attr_data.get('opening_hours'),
                        latitude=attr_data.get('latitude'),
                        longitude=attr_data.get('longitude')
                    )
                    db.add(museum)
                    count += 1
                    
                    if count % 25 == 0:
                        db.commit()
                        print(f"  ✓ {count} attractions loaded...")
                        
                except Exception as e:
                    print(f"  ⚠️  Skipped attraction: {attr_data.get('name')} - {e}")
                    continue
            
            db.commit()
            print(f"✅ Loaded {count} attractions total")
            
    except Exception as e:
        print(f"❌ Error loading attractions: {e}")
        db.rollback()
    finally:
        db.close()
    
    return count

def load_events():
    """Load events from Python module"""
    print("\n🎉 Loading events...")
    
    try:
        from data.events_database import ISTANBUL_EVENTS
        
        db = SessionLocal()
        count = 0
        
        for event_data in ISTANBUL_EVENTS:
            try:
                event = Event(
                    title=event_data.get('name') or event_data.get('title'),
                    category=event_data.get('category'),
                    venue=event_data.get('venue') or event_data.get('location'),
                    district=event_data.get('district'),
                    start_date=event_data.get('start_date'),
                    end_date=event_data.get('end_date'),
                    price=event_data.get('price') or event_data.get('entry_fee'),
                    description=event_data.get('description')
                )
                db.add(event)
                count += 1
                
            except Exception as e:
                print(f"  ⚠️  Skipped event: {event_data.get('name')} - {e}")
                continue
        
        db.commit()
        print(f"✅ Loaded {count} events total")
        db.close()
        return count
        
    except ImportError:
        print("⚠️  events_database.py not found or invalid")
        return 0
    except Exception as e:
        print(f"❌ Error loading events: {e}")
        return 0

def get_table_counts():
    """Get current counts from database"""
    db = SessionLocal()
    try:
        restaurant_count = db.query(Restaurant).count()
        museum_count = db.query(Museum).count()
        event_count = db.query(Event).count()
        return restaurant_count, museum_count, event_count
    finally:
        db.close()

def main():
    """Main population script"""
    print("=" * 60)
    print("🚀 AI Istanbul - Database Population Script")
    print("=" * 60)
    
    # Check if DATABASE_URL is set
    if not os.getenv('DATABASE_URL'):
        print("❌ ERROR: DATABASE_URL environment variable not set!")
        print("\nSet it with:")
        print("export DATABASE_URL='postgres://...'")
        print("\nGet the URL from: Render Dashboard → PostgreSQL → Internal Database URL")
        sys.exit(1)
    
    print(f"\n📊 Current database state:")
    before_restaurants, before_museums, before_events = get_table_counts()
    print(f"  Restaurants: {before_restaurants}")
    print(f"  Museums/Attractions: {before_museums}")
    print(f"  Events: {before_events}")
    
    # Create tables
    create_tables()
    
    # Load data
    restaurants_loaded = load_restaurants()
    attractions_loaded = load_attractions()
    events_loaded = load_events()
    
    # Final counts
    print("\n" + "=" * 60)
    print("📊 Final database state:")
    after_restaurants, after_museums, after_events = get_table_counts()
    print(f"  Restaurants: {after_restaurants} (+{after_restaurants - before_restaurants})")
    print(f"  Museums/Attractions: {after_museums} (+{after_museums - before_museums})")
    print(f"  Events: {after_events} (+{after_events - before_events})")
    print("=" * 60)
    
    if restaurants_loaded + attractions_loaded + events_loaded > 0:
        print("\n✅ Database population complete!")
    else:
        print("\n⚠️  No data was loaded. Check error messages above.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

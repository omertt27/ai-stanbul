#!/usr/bin/env python3
"""
Fixed Database Seed Script - Correct Model Fields
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from database import SessionLocal
from models import Restaurant, Museum, Place
from sqlalchemy import text
import json

def seed_museums():
    """Seed museums with correct fields: name, location, hours, ticket_price, highlights"""
    print("🏛️ Seeding Museums...")
    
    try:
        from accurate_museum_database import IstanbulMuseumDatabase
        
        db = SessionLocal()
        museum_db = IstanbulMuseumDatabase()
        
        added = 0
        for museum_id, m in museum_db.museums.items():
            existing = db.query(Museum).filter_by(name=m.name).first()
            if existing:
                continue
            
            museum = Museum(
                name=m.name,
                location=m.location,
                hours=json.dumps(m.opening_hours) if m.opening_hours else None,
                ticket_price=m.entrance_fee,
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
        return 0

def seed_restaurants():
    """Seed restaurants with correct fields"""
    print("🍽️ Seeding Restaurants...")
    
    try:
        from services.restaurant_database_service import RestaurantDatabaseService
        
        db = SessionLocal()
        service = RestaurantDatabaseService()
        
        # Get restaurants from the service's internal data
        all_rests = service.restaurants  # Direct access to the list
        
        added = 0
        for rest in all_rests[:150]:  # Seed first 150
            existing = db.query(Restaurant).filter_by(name=rest['name']).first()
            if existing:
                continue
            
            restaurant = Restaurant(
                name=rest['name'],
                cuisine=rest.get('cuisine', 'Turkish'),
                location=rest.get('location', rest.get('neighborhood', 'Istanbul')),
                price_level=rest.get('price_range', '$$'),
                rating=float(rest.get('rating', 4.0)),
                description=rest.get('description', '')[:300],
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
        return 0

def seed_attractions():
    """Seed attractions with correct Place fields: name, category, district"""
    print("🏞️ Seeding Attractions...")
    
    try:
        from istanbul_attractions_system import IstanbulAttractionsSystem
        
        db = SessionLocal()
        system = IstanbulAttractionsSystem()
        
        added = 0
        for attr in system.attractions:
            existing = db.query(Place).filter_by(name=attr['name']).first()
            if existing:
                continue
            
            place = Place(
                name=attr['name'],
                category=attr.get('category', 'attraction'),
                district=attr.get('district', attr.get('location', 'Istanbul'))
            )
            db.add(place)
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} attractions")
        return added
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def seed_hidden_gems():
    """Seed hidden gems as places"""
    print("💎 Seeding Hidden Gems...")
    
    try:
        from data.hidden_gems_database import get_all_hidden_gems
        
        db = SessionLocal()
        gems = get_all_hidden_gems()
        
        added = 0
        for gem in gems:
            existing = db.query(Place).filter_by(name=gem.get('name')).first()
            if existing:
                continue
            
            place = Place(
                name=gem.get('name'),
                category='hidden_gem',
                district=gem.get('neighborhood', 'Istanbul')
            )
            db.add(place)
            added += 1
        
        db.commit()
        db.close()
        print(f"   ✅ Added {added} hidden gems")
        return added
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def seed_events():
    """Seed events using raw SQL"""
    print("🎭 Seeding Events...")
    
    try:
        db = SessionLocal()
        
        # Add sample events manually since the events_database has complex structure
        sample_events = [
            ("İKSV Film Festival", "İstanbul", "2024-04-05"),
            ("İstanbul Music Festival", "Cemal Reşit Rey Concert Hall", "2024-06-15"),
            ("İstanbul Biennial", "Various Locations", "2024-09-01"),
            ("Chill-Out Festival", "Maçka Democracy Park", "2024-07-20"),
            ("İstanbul Jazz Festival", "Multiple Venues", "2024-07-10"),
            ("İstanbul Theatre Festival", "Various Theatres", "2024-05-15"),
            ("Ramadan Celebrations", "Sultanahmet Square", "2024-03-10"),
            ("Istanbul Marathon", "Bosphorus Bridge", "2024-11-03"),
            ("Tulip Festival", "Emirgan Park", "2024-04-15"),
            ("Istanbul Shopping Fest", "Shopping Malls", "2024-06-01"),
        ]
        
        added = 0
        for title, venue, date in sample_events:
            result = db.execute(
                text("SELECT id FROM events WHERE title = :title"),
                {"title": title}
            ).first()
            
            if not result:
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

def verify_data():
    """Show final database summary"""
    print("\n" + "=" * 60)
    print("📊 DATABASE CONTENTS")
    print("=" * 60)
    
    db = SessionLocal()
    
    tables = [
        ("Restaurants", "restaurants"),
        ("Museums", "museums"),
        ("Events", "events"),
        ("Places (Attractions)", "places"),
        ("Enhanced Attractions", "enhanced_attractions"),
    ]
    
    for label, table in tables:
        result = db.execute(text(f"SELECT COUNT(*) FROM {table}")).first()
        count = result[0] if result else 0
        emoji = "✅" if count >= 10 else "⚠️" if count > 0 else "❌"
        print(f"{emoji} {label:30} {count:>5} records")
    
    db.close()
    print("=" * 60 + "\n")

def main():
    print("\n" + "=" * 60)
    print("🌱 SEEDING DATABASE WITH LARGE DATASETS")
    print("=" * 60 + "\n")
    
    museums_added = seed_museums()
    restaurants_added = seed_restaurants()
    attractions_added = seed_attractions()
    gems_added = seed_hidden_gems()
    events_added = seed_events()
    
    verify_data()
    
    total = museums_added + restaurants_added + attractions_added + gems_added + events_added
    
    if total > 0:
        print(f"✅ Successfully added {total} new records!")
        print("🚀 Database is ready for Pure LLM Architecture!")
    else:
        print("ℹ️  No new data added - database already contains data")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

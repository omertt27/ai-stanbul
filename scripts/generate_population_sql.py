#!/usr/bin/env python3
"""
Generate SQL INSERT statements from JSON data files
This can be used to populate the database via psql if Python script fails
"""

import json
import sys
from pathlib import Path

# Get backend directory
backend_dir = Path(__file__).parent.parent / 'backend'

def escape_sql_string(s):
    """Escape single quotes for SQL"""
    if s is None:
        return 'NULL'
    return f"'{str(s).replace(chr(39), chr(39)+chr(39))}'"

def generate_restaurant_inserts():
    """Generate INSERT statements for restaurants"""
    data_file = backend_dir / 'data' / 'restaurants_database.json'
    if not data_file.exists():
        return []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        restaurants = data.get('restaurants', [])
    
    inserts = []
    for r in restaurants:
        name = escape_sql_string(r.get('name'))
        cuisine = escape_sql_string(r.get('cuisine_type') or r.get('cuisine'))
        district = escape_sql_string(r.get('district'))
        address = escape_sql_string(r.get('address'))
        price_range = escape_sql_string(r.get('budget') or r.get('price_range'))
        rating = r.get('rating') if r.get('rating') else 'NULL'
        reviews_count = r.get('reviews_count') if r.get('reviews_count') else 'NULL'
        latitude = r.get('latitude') if r.get('latitude') else 'NULL'
        longitude = r.get('longitude') if r.get('longitude') else 'NULL'
        phone = escape_sql_string(r.get('phone'))
        website = escape_sql_string(r.get('website'))
        description = escape_sql_string(r.get('description'))
        
        sql = f"""INSERT INTO restaurants (name, cuisine, district, address, price_range, rating, reviews_count, latitude, longitude, phone, website, description, language) 
VALUES ({name}, {cuisine}, {district}, {address}, {price_range}, {rating}, {reviews_count}, {latitude}, {longitude}, {phone}, {website}, {description}, 'en') 
ON CONFLICT DO NOTHING;"""
        inserts.append(sql)
    
    return inserts

def generate_attraction_inserts():
    """Generate INSERT statements for museums/attractions"""
    data_file = backend_dir / 'data' / 'attractions_database_expanded.json'
    if not data_file.exists():
        return []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        attractions = data.get('attractions', [])
    
    inserts = []
    for a in attractions:
        name = escape_sql_string(a.get('name'))
        category = escape_sql_string(a.get('category') or a.get('type'))
        district = escape_sql_string(a.get('district'))
        address = escape_sql_string(a.get('address'))
        entry_fee = escape_sql_string(a.get('entry_fee') or a.get('price'))
        rating = a.get('rating') if a.get('rating') else 'NULL'
        description = escape_sql_string(a.get('description'))
        opening_hours = escape_sql_string(a.get('opening_hours'))
        latitude = a.get('latitude') if a.get('latitude') else 'NULL'
        longitude = a.get('longitude') if a.get('longitude') else 'NULL'
        
        sql = f"""INSERT INTO museums (name, category, district, address, entry_fee, rating, description, opening_hours, latitude, longitude) 
VALUES ({name}, {category}, {district}, {address}, {entry_fee}, {rating}, {description}, {opening_hours}, {latitude}, {longitude}) 
ON CONFLICT DO NOTHING;"""
        inserts.append(sql)
    
    return inserts

def generate_event_inserts():
    """Generate INSERT statements for events"""
    try:
        sys.path.insert(0, str(backend_dir))
        from data.events_database import ISTANBUL_EVENTS
        
        inserts = []
        for e in ISTANBUL_EVENTS:
            title = escape_sql_string(e.get('name') or e.get('title'))
            category = escape_sql_string(e.get('category'))
            venue = escape_sql_string(e.get('venue') or e.get('location'))
            district = escape_sql_string(e.get('district'))
            start_date = escape_sql_string(e.get('start_date'))
            end_date = escape_sql_string(e.get('end_date'))
            price = escape_sql_string(e.get('price') or e.get('entry_fee'))
            description = escape_sql_string(e.get('description'))
            
            sql = f"""INSERT INTO events (title, category, venue, district, start_date, end_date, price, description) 
VALUES ({title}, {category}, {venue}, {district}, {start_date}, {end_date}, {price}, {description}) 
ON CONFLICT DO NOTHING;"""
            inserts.append(sql)
        
        return inserts
    except:
        return []

def main():
    """Generate complete SQL script"""
    print("-- AI Istanbul Database Population SQL Script")
    print("-- Generated automatically from JSON data files")
    print("-- Usage: psql $DATABASE_URL < population_script.sql")
    print()
    
    print("-- Begin transaction")
    print("BEGIN;")
    print()
    
    # Restaurants
    print("-- Inserting restaurants...")
    restaurant_inserts = generate_restaurant_inserts()
    for sql in restaurant_inserts:
        print(sql)
    print(f"-- {len(restaurant_inserts)} restaurant records")
    print()
    
    # Attractions
    print("-- Inserting attractions...")
    attraction_inserts = generate_attraction_inserts()
    for sql in attraction_inserts:
        print(sql)
    print(f"-- {len(attraction_inserts)} attraction records")
    print()
    
    # Events  
    print("-- Inserting events...")
    event_inserts = generate_event_inserts()
    for sql in event_inserts:
        print(sql)
    print(f"-- {len(event_inserts)} event records")
    print()
    
    print("-- Commit transaction")
    print("COMMIT;")
    print()
    
    total = len(restaurant_inserts) + len(attraction_inserts) + len(event_inserts)
    print(f"-- Total: {total} records")
    print(f"-- Restaurants: {len(restaurant_inserts)}")
    print(f"-- Attractions: {len(attraction_inserts)}")
    print(f"-- Events: {len(event_inserts)}")

if __name__ == "__main__":
    main()

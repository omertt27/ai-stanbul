"""Import fetched POIs into SQLite database"""

import json
import sqlite3
import sys
import os

def import_pois(json_file, db_path='places_cache.db'):
    """Import POIs from JSON file into database"""
    
    # Check if JSON file exists
    if not os.path.exists(json_file):
        print(f"❌ Error: File '{json_file}' not found!")
        return False
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Error: Database '{db_path}' not found!")
        print("💡 Run 'python create_db.py' first to create the database.")
        return False
    
    # Load JSON data
    print(f"📂 Loading POIs from {json_file}...")
    with open(json_file, 'r', encoding='utf-8') as f:
        pois = json.load(f)
    
    print(f"   Found {len(pois)} POIs to import")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    imported = 0
    skipped = 0
    errors = 0
    
    print("\n🔄 Importing POIs...")
    
    for i, poi in enumerate(pois, 1):
        try:
            cursor.execute('''
                INSERT INTO cached_pois (
                    name, type, lat, lng, rating, review_count,
                    google_place_id, address, price_level, zone, source,
                    photos, tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                poi.get('name'),
                poi.get('type'),
                poi.get('lat'),
                poi.get('lng'),
                poi.get('rating', 0),
                poi.get('review_count', 0),
                poi.get('google_place_id'),
                poi.get('address'),
                poi.get('price_level', 0),
                poi.get('zone'),
                poi.get('source', 'google'),
                json.dumps(poi.get('photos', [])),
                json.dumps([poi.get('type', '')])
            ))
            imported += 1
            
            # Progress indicator
            if i % 50 == 0:
                print(f"   Processed {i}/{len(pois)}...")
            
        except sqlite3.IntegrityError:
            skipped += 1  # Duplicate place_id
        except Exception as e:
            errors += 1
            print(f"   ✗ Error importing '{poi.get('name', 'Unknown')}': {str(e)}")
    
    conn.commit()
    
    # Get final stats
    cursor.execute('SELECT COUNT(*) FROM cached_pois')
    total_in_db = cursor.fetchone()[0]
    
    conn.close()
    
    print("\n" + "="*60)
    print("✅ Import Complete!")
    print(f"   Imported:  {imported} new POIs")
    print(f"   Skipped:   {skipped} duplicates")
    print(f"   Errors:    {errors}")
    print(f"   Total in DB: {total_in_db} POIs")
    print("="*60)
    print(f"\n📄 Next Step: python verify_db.py")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: No JSON file specified!")
        print("\nUsage: python import_pois.py <json_file>")
        print("Example: python import_pois.py pois_raw_20251018_140523.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    import_pois(json_file)

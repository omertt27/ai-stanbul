"""Create SQLite database for cached POIs"""

import sqlite3
import os

def create_database(db_path='places_cache.db'):
    """Create the POI cache database with proper schema"""
    
    # Check if database already exists
    if os.path.exists(db_path):
        response = input(f"⚠️  Database '{db_path}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted. Database not modified.")
            return False
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("🔨 Creating database schema...")
    
    # Create cached_pois table
    cursor.execute('''
        CREATE TABLE cached_pois (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT,
            lat REAL,
            lng REAL,
            rating REAL,
            review_count INTEGER,
            google_place_id TEXT UNIQUE,
            tripadvisor_id TEXT,
            description TEXT,
            tags TEXT,
            photos TEXT,
            hours TEXT,
            address TEXT,
            price_level INTEGER,
            zone TEXT,
            source TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            manual_verified BOOLEAN DEFAULT 0,
            user_rating_override REAL
        )
    ''')
    
    print("  ✓ Table 'cached_pois' created")
    
    # Create indexes for fast queries
    print("🔨 Creating indexes...")
    
    cursor.execute('CREATE INDEX idx_location ON cached_pois(lat, lng)')
    print("  ✓ Index 'idx_location' created")
    
    cursor.execute('CREATE INDEX idx_type ON cached_pois(type)')
    print("  ✓ Index 'idx_type' created")
    
    cursor.execute('CREATE INDEX idx_rating ON cached_pois(rating DESC, review_count DESC)')
    print("  ✓ Index 'idx_rating' created")
    
    cursor.execute('CREATE INDEX idx_zone ON cached_pois(zone)')
    print("  ✓ Index 'idx_zone' created")
    
    cursor.execute('CREATE INDEX idx_name ON cached_pois(name)')
    print("  ✓ Index 'idx_name' created")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Database created successfully: {db_path}")
    print(f"📊 Schema: 1 table, 5 indexes")
    print(f"💾 Size: {os.path.getsize(db_path)} bytes")
    print(f"\n📄 Next Step: python import_pois.py <your_json_file>")
    
    return True

if __name__ == "__main__":
    create_database()

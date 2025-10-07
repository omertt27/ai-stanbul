"""
Direct SQL Table Creation for Route Maker
Creates the tables using direct SQL commands
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from backend.database import DATABASE_URL, engine

def create_tables_sql():
    """Create tables using direct SQL"""
    print("🔧 Creating Route Maker tables with direct SQL...")
    
    try:
        with engine.connect() as conn:
            # Create enhanced_attractions table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS enhanced_attractions (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    subcategory VARCHAR(50),
                    address VARCHAR(500),
                    district VARCHAR(50),
                    coordinates_lat REAL NOT NULL,
                    coordinates_lng REAL NOT NULL,
                    popularity_score REAL DEFAULT 3.0,
                    estimated_visit_time_minutes INTEGER DEFAULT 60,
                    best_time_of_day VARCHAR(20),
                    crowd_level VARCHAR(20) DEFAULT 'medium',
                    description TEXT,
                    opening_hours VARCHAR(200),
                    price_range VARCHAR(20),
                    authenticity_score REAL DEFAULT 3.0,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Create routes table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS routes (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(200),
                    description TEXT,
                    start_lat REAL NOT NULL,
                    start_lng REAL NOT NULL,
                    end_lat REAL,
                    end_lng REAL,
                    total_distance_km REAL,
                    estimated_duration_hours REAL,
                    difficulty_level VARCHAR(20) DEFAULT 'easy',
                    transportation_mode VARCHAR(20) DEFAULT 'walking',
                    overall_score REAL,
                    diversity_score REAL,
                    efficiency_score REAL,
                    preferences_snapshot JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_saved BOOLEAN DEFAULT FALSE
                );
            """))
            
            # Create route_waypoints table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS route_waypoints (
                    id SERIAL PRIMARY KEY,
                    route_id INTEGER NOT NULL REFERENCES routes(id) ON DELETE CASCADE,
                    attraction_id INTEGER NOT NULL REFERENCES enhanced_attractions(id),
                    waypoint_order INTEGER NOT NULL,
                    estimated_arrival_time VARCHAR(20),
                    suggested_duration_minutes INTEGER,
                    distance_from_previous_km REAL,
                    travel_time_from_previous_minutes REAL,
                    attraction_score REAL,
                    position_score REAL,
                    notes TEXT
                );
            """))
            
            # Create user_route_preferences table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_route_preferences (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(100) NOT NULL,
                    max_walking_distance_km REAL DEFAULT 5.0,
                    preferred_pace VARCHAR(20) DEFAULT 'moderate',
                    available_time_hours REAL DEFAULT 4.0,
                    preferred_categories JSONB,
                    avoided_categories JSONB,
                    min_popularity_score REAL DEFAULT 2.0,
                    max_crowd_level VARCHAR(20) DEFAULT 'high',
                    preferred_start_time VARCHAR(10),
                    preferred_end_time VARCHAR(10),
                    max_total_cost REAL,
                    requires_wheelchair_access BOOLEAN DEFAULT FALSE,
                    route_style VARCHAR(30) DEFAULT 'balanced',
                    include_food_stops BOOLEAN DEFAULT TRUE,
                    include_rest_stops BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            conn.commit()
            print("✅ Tables created successfully")
            
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise

def create_indexes():
    """Create indexes for better performance"""
    print("🔧 Creating database indexes...")
    
    try:
        with engine.connect() as conn:
            # Indexes for enhanced_attractions
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_attractions_coordinates 
                ON enhanced_attractions (coordinates_lat, coordinates_lng);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_attractions_category 
                ON enhanced_attractions (category);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_attractions_district 
                ON enhanced_attractions (district);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_attractions_popularity 
                ON enhanced_attractions (popularity_score DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_attractions_active 
                ON enhanced_attractions (is_active);
            """))
            
            # Indexes for routes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_routes_created_at 
                ON routes (created_at DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_routes_saved 
                ON routes (is_saved);
            """))
            
            # Indexes for route_waypoints
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_route_waypoints_route_id 
                ON route_waypoints (route_id);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_route_waypoints_order 
                ON route_waypoints (route_id, waypoint_order);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_route_waypoints_attraction 
                ON route_waypoints (attraction_id);
            """))
            
            # Indexes for user_route_preferences
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_route_preferences_session 
                ON user_route_preferences (session_id);
            """))
            
            conn.commit()
            print("✅ Database indexes created successfully")
            
    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
        raise

def verify_tables():
    """Verify that all tables were created"""
    print("🔍 Verifying table creation...")
    
    tables_to_check = [
        'enhanced_attractions',
        'routes', 
        'route_waypoints',
        'user_route_preferences'
    ]
    
    try:
        with engine.connect() as conn:
            for table_name in tables_to_check:
                result = conn.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table_name}'
                    );
                """))
                
                if result.scalar():
                    print(f"✅ {table_name} table exists")
                    
                    # Get row count
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = count_result.scalar()
                    print(f"   📊 {count} rows")
                else:
                    print(f"❌ {table_name} table not found")
    
    except Exception as e:
        print(f"❌ Error verifying tables: {e}")

if __name__ == "__main__":
    print("🚀 Starting Direct SQL Route Maker Table Creation...")
    print(f"📊 Database URL: {DATABASE_URL}")
    
    # Create tables
    create_tables_sql()
    
    # Create indexes
    create_indexes()
    
    # Verify tables
    verify_tables()
    
    print("✅ Route Maker database setup completed successfully!")

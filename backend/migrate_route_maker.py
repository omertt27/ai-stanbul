"""
Database Migration: Add Route Maker Tables
Creates the new tables for enhanced attractions, routes, and route maker functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from backend.database import DATABASE_URL, Base, SessionLocal, engine
from backend.models import (
    EnhancedAttraction, Route, RouteWaypoint, UserRoutePreferences
)

def create_route_maker_tables():
    """Create all route maker related tables"""
    print("🔧 Creating Route Maker tables...")
    
    try:
        # Create all tables defined in models
        Base.metadata.create_all(bind=engine)
        print("✅ All tables created successfully")
        
        # Verify tables were created
        with engine.connect() as conn:
            # Check if enhanced_attractions table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'enhanced_attractions'
                );
            """))
            
            if result.scalar():
                print("✅ enhanced_attractions table created")
            else:
                print("❌ enhanced_attractions table not found")
            
            # Check if routes table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'routes'
                );
            """))
            
            if result.scalar():
                print("✅ routes table created")
            else:
                print("❌ routes table not found")
            
            # Check if route_waypoints table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'route_waypoints'
                );
            """))
            
            if result.scalar():
                print("✅ route_waypoints table created")
            else:
                print("❌ route_waypoints table not found")
            
            # Check if user_route_preferences table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'user_route_preferences'
                );
            """))
            
            if result.scalar():
                print("✅ user_route_preferences table created")
            else:
                print("❌ user_route_preferences table not found")
                
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise

def add_indexes():
    """Add database indexes for better performance"""
    print("🔧 Adding database indexes...")
    
    try:
        with engine.connect() as conn:
            # Spatial indexes for coordinates
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_attractions_coordinates 
                ON enhanced_attractions (coordinates_lat, coordinates_lng);
            """))
            
            # Category index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_attractions_category 
                ON enhanced_attractions (category);
            """))
            
            # District index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_attractions_district 
                ON enhanced_attractions (district);
            """))
            
            # Popularity score index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_enhanced_attractions_popularity 
                ON enhanced_attractions (popularity_score DESC);
            """))
            
            # Route indexes
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_routes_created_at 
                ON routes (created_at DESC);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_route_waypoints_route_id 
                ON route_waypoints (route_id);
            """))
            
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_route_waypoints_order 
                ON route_waypoints (route_id, waypoint_order);
            """))
            
            # User preferences index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_user_route_preferences_session 
                ON user_route_preferences (session_id);
            """))
            
            conn.commit()
            print("✅ Database indexes added successfully")
            
    except Exception as e:
        print(f"❌ Error adding indexes: {e}")
        raise

if __name__ == "__main__":
    print("🚀 Starting Route Maker Database Migration...")
    print(f"📊 Database URL: {DATABASE_URL}")
    
    # Create tables
    create_route_maker_tables()
    
    # Add indexes
    add_indexes()
    
    print("✅ Route Maker database migration completed successfully!")

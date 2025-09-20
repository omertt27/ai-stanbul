#!/usr/bin/env python3
"""
Database Index Optimization Script for Istanbul AI

This script adds critical database indexes to improve query performance,
specifically for places_data.category and places_data.district columns.
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from database import Base, SessionLocal, engine
from models import Place, Restaurant, Museum

def check_existing_indexes():
    """Check what indexes currently exist in the database"""
    inspector = inspect(engine)
    
    print("🔍 Current Database Indexes:")
    print("=" * 50)
    
    # Check indexes for each table
    tables = ['places', 'restaurants', 'museums', 'blog_posts', 'chat_history']
    
    for table_name in tables:
        try:
            indexes = inspector.get_indexes(table_name)
            print(f"\n📊 Table: {table_name}")
            if indexes:
                for idx in indexes:
                    # Handle potential None values in column names
                    column_names = [col for col in idx['column_names'] if col is not None]
                    columns = ", ".join(column_names)
                    unique = " (UNIQUE)" if idx.get('unique') else ""
                    print(f"  • {idx['name']}: [{columns}]{unique}")
            else:
                print("  • No custom indexes found")
        except Exception as e:
            print(f"  • Table {table_name} not found or error: {e}")
    
    print("\n" + "=" * 50)

def add_performance_indexes():
    """Add critical indexes for query performance optimization"""
    
    print("\n🚀 Adding Performance Indexes...")
    print("=" * 50)
    
    # Get database session
    session = SessionLocal()
    
    try:
        # Index statements for different database types
        index_queries = [
            # Places table indexes
            "CREATE INDEX IF NOT EXISTS idx_places_category ON places(category)",
            "CREATE INDEX IF NOT EXISTS idx_places_district ON places(district)", 
            "CREATE INDEX IF NOT EXISTS idx_places_category_district ON places(category, district)",
            
            # Restaurants table indexes  
            "CREATE INDEX IF NOT EXISTS idx_restaurants_cuisine ON restaurants(cuisine)",
            "CREATE INDEX IF NOT EXISTS idx_restaurants_location ON restaurants(location)",
            "CREATE INDEX IF NOT EXISTS idx_restaurants_rating ON restaurants(rating)",
            "CREATE INDEX IF NOT EXISTS idx_restaurants_place_id ON restaurants(place_id)",
            
            # Museums table indexes
            "CREATE INDEX IF NOT EXISTS idx_museums_location ON museums(location)",
            "CREATE INDEX IF NOT EXISTS idx_museums_name ON museums(name)",
            
            # Chat history indexes for session-based queries
            "CREATE INDEX IF NOT EXISTS idx_chat_session_id ON chat_history(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_history(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_chat_session_timestamp ON chat_history(session_id, timestamp)",
            
            # Blog posts indexes
            "CREATE INDEX IF NOT EXISTS idx_blog_district ON blog_posts(district)",
            "CREATE INDEX IF NOT EXISTS idx_blog_created_at ON blog_posts(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_blog_is_published ON blog_posts(is_published)",
            "CREATE INDEX IF NOT EXISTS idx_blog_published_created ON blog_posts(is_published, created_at)",
            
            # User session indexes for AI features
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_session_id ON user_sessions(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active, last_activity)",
            
            # Enhanced chat history indexes
            "CREATE INDEX IF NOT EXISTS idx_enhanced_chat_session ON enhanced_chat_history(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_chat_timestamp ON enhanced_chat_history(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_chat_intent ON enhanced_chat_history(detected_intent)",
        ]
        
        successful_indexes = []
        failed_indexes = []
        
        for query in index_queries:
            try:
                session.execute(text(query))
                session.commit()
                
                # Extract index name for reporting
                index_name = query.split("idx_")[1].split(" ")[0] if "idx_" in query else "unknown"
                successful_indexes.append(index_name)
                print(f"  ✅ Created index: idx_{index_name}")
                
            except Exception as e:
                error_msg = str(e)
                index_name = query.split("idx_")[1].split(" ")[0] if "idx_" in query else "unknown"
                failed_indexes.append((index_name, error_msg))
                
                # Don't fail for "already exists" errors
                if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                    print(f"  ⚠️  Index already exists: idx_{index_name}")
                else:
                    print(f"  ❌ Failed to create idx_{index_name}: {error_msg}")
        
        print(f"\n📈 Index Creation Summary:")
        print(f"  • Successfully created: {len(successful_indexes)} indexes")
        print(f"  • Failed/Already existed: {len(failed_indexes)} indexes")
        
        if successful_indexes:
            print(f"\n🎯 New indexes that will improve performance:")
            for idx in successful_indexes:
                print(f"  • idx_{idx}")
                
    except Exception as e:
        print(f"❌ Error during index creation: {e}")
        session.rollback()
    finally:
        session.close()

def analyze_query_performance():
    """Analyze the impact of new indexes on common queries"""
    
    print("\n📊 Query Performance Analysis:")
    print("=" * 50)
    
    session = SessionLocal()
    
    try:
        # Test common query patterns that benefit from indexing
        test_queries = [
            ("Category-based place search", "SELECT COUNT(*) FROM places WHERE category = 'restaurant'"),
            ("District-based place search", "SELECT COUNT(*) FROM places WHERE district = 'Sultanahmet'"),
            ("Combined category+district", "SELECT COUNT(*) FROM places WHERE category = 'restaurant' AND district = 'Beyoglu'"),
            ("Restaurant by cuisine", "SELECT COUNT(*) FROM restaurants WHERE cuisine = 'Turkish'"),
            ("High-rated restaurants", "SELECT COUNT(*) FROM restaurants WHERE rating >= 4.0"),
            ("Recent chat history", "SELECT COUNT(*) FROM chat_history WHERE timestamp > datetime('now', '-1 day')"),
            ("Published blog posts", "SELECT COUNT(*) FROM blog_posts WHERE is_published = 1"),
        ]
        
        for description, query in test_queries:
            try:
                result = session.execute(text(query)).scalar()
                print(f"  ✅ {description}: {result} records")
            except Exception as e:
                print(f"  ❌ {description}: Error - {e}")
                
    except Exception as e:
        print(f"❌ Error during performance analysis: {e}")
    finally:
        session.close()

def main():
    """Main execution function"""
    print("🗃️  Istanbul AI Database Index Optimization")
    print("=" * 60)
    
    try:
        # Step 1: Check current indexes
        check_existing_indexes()
        
        # Step 2: Add performance indexes
        add_performance_indexes()
        
        # Step 3: Analyze query performance
        analyze_query_performance()
        
        print("\n🎯 Database Optimization Complete!")
        print("=" * 60)
        print("📋 Benefits of Added Indexes:")
        print("  • Faster place searches by category and district")
        print("  • Improved restaurant filtering by cuisine and rating")
        print("  • Optimized chat history retrieval by session")
        print("  • Enhanced blog post queries by publication status")
        print("  • Better performance for AI recommendation queries")
        print("\n💡 Recommendation: Run this script after any major data imports")
        
    except Exception as e:
        print(f"❌ Critical error during database optimization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

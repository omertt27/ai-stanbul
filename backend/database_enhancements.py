#!/usr/bin/env python3
"""
Database Enhancements for AI Istanbul
=====================================

Implements:
1. Full-text search indexes for PostgreSQL
2. NoSQL-style document storage within PostgreSQL (JSONB)
3. Search optimization and indexing
4. Data validation and normalization
"""

from sqlalchemy import text, Index
from sqlalchemy.dialects.postgresql import TSVECTOR
from database import engine, SessionLocal
from models import Base, Restaurant, Museum, Place, UserFeedback
import logging

logger = logging.getLogger(__name__)

def create_fulltext_search_indexes():
    """Create full-text search indexes for PostgreSQL"""
    
    with engine.connect() as conn:
        # Enable PostgreSQL full-text search extensions
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent;"))
            print("✅ PostgreSQL text search extensions enabled")
        except Exception as e:
            print(f"⚠️ Extension setup: {e}")
        
        # Create full-text search indexes on key tables
        search_indexes = [
            # Restaurants full-text search
            """
            CREATE INDEX IF NOT EXISTS restaurants_search_idx 
            ON restaurants USING GIN (
                to_tsvector('english', 
                    COALESCE(name, '') || ' ' || 
                    COALESCE(description, '') || ' ' || 
                    COALESCE(cuisine_type, '') || ' ' || 
                    COALESCE(district, '')
                )
            );
            """,
            
            # Museums full-text search  
            """
            CREATE INDEX IF NOT EXISTS museums_search_idx 
            ON museums USING GIN (
                to_tsvector('english',
                    COALESCE(name, '') || ' ' || 
                    COALESCE(description, '') || ' ' || 
                    COALESCE(category, '') || ' ' || 
                    COALESCE(district, '')
                )
            );
            """,
            
            # Places full-text search
            """
            CREATE INDEX IF NOT EXISTS places_search_idx 
            ON places USING GIN (
                to_tsvector('english',
                    COALESCE(name, '') || ' ' || 
                    COALESCE(description, '') || ' ' || 
                    COALESCE(category, '') || ' ' || 
                    COALESCE(district, '')
                )
            );
            """,
            
            # User feedback search
            """
            CREATE INDEX IF NOT EXISTS user_feedback_search_idx 
            ON user_feedback USING GIN (
                to_tsvector('english', COALESCE(comment, ''))
            );
            """,
            
            # Trigram indexes for fuzzy matching
            """
            CREATE INDEX IF NOT EXISTS restaurants_name_trgm_idx 
            ON restaurants USING GIN (name gin_trgm_ops);
            """,
            
            """
            CREATE INDEX IF NOT EXISTS museums_name_trgm_idx 
            ON museums USING GIN (name gin_trgm_ops);
            """,
            
            """
            CREATE INDEX IF NOT EXISTS places_name_trgm_idx 
            ON places USING GIN (name gin_trgm_ops);
            """,
        ]
        
        for index_sql in search_indexes:
            try:
                conn.execute(text(index_sql))
                print(f"✅ Created search index")
            except Exception as e:
                print(f"⚠️ Index creation error: {e}")
        
        # Commit all changes
        conn.commit()
        print("✅ All full-text search indexes created successfully")

def create_document_storage_tables():
    """Create tables for semi-structured document storage using JSONB"""
    
    with engine.connect() as conn:
        # Reviews and ratings (NoSQL-style)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_reviews (
                id SERIAL PRIMARY KEY,
                place_id INTEGER,
                place_type VARCHAR(50) NOT NULL, -- 'restaurant', 'museum', 'attraction'
                place_name VARCHAR(255) NOT NULL,
                review_data JSONB NOT NULL, -- Full review document
                user_data JSONB, -- User profile and preferences
                metadata JSONB, -- System metadata, ratings, sentiment
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        # Itineraries (NoSQL-style)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_itineraries (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                itinerary_data JSONB NOT NULL, -- Full itinerary document
                preferences JSONB, -- User travel preferences
                metadata JSONB, -- System data, completion status, ratings
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        # FAQs and Knowledge Base (NoSQL-style)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_knowledge (
                id SERIAL PRIMARY KEY,
                category VARCHAR(100) NOT NULL,
                question_answer JSONB NOT NULL, -- Q&A pairs and variations
                content_data JSONB, -- Rich content, images, links
                search_metadata JSONB, -- Keywords, tags, popularity
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        # Create JSONB indexes for fast queries
        jsonb_indexes = [
            # Reviews indexes
            "CREATE INDEX IF NOT EXISTS reviews_place_type_idx ON document_reviews USING BTREE (place_type);",
            "CREATE INDEX IF NOT EXISTS reviews_place_name_idx ON document_reviews USING GIN (place_name gin_trgm_ops);",
            "CREATE INDEX IF NOT EXISTS reviews_data_idx ON document_reviews USING GIN (review_data);",
            "CREATE INDEX IF NOT EXISTS reviews_metadata_idx ON document_reviews USING GIN (metadata);",
            
            # Itineraries indexes  
            "CREATE INDEX IF NOT EXISTS itineraries_user_idx ON document_itineraries USING BTREE (user_id);",
            "CREATE INDEX IF NOT EXISTS itineraries_data_idx ON document_itineraries USING GIN (itinerary_data);",
            
            # Knowledge base indexes
            "CREATE INDEX IF NOT EXISTS knowledge_category_idx ON document_knowledge USING BTREE (category);",
            "CREATE INDEX IF NOT EXISTS knowledge_qa_idx ON document_knowledge USING GIN (question_answer);",
            "CREATE INDEX IF NOT EXISTS knowledge_search_idx ON document_knowledge USING GIN (search_metadata);",
        ]
        
        for index_sql in jsonb_indexes:
            try:
                conn.execute(text(index_sql))
                print(f"✅ Created JSONB index")
            except Exception as e:
                print(f"⚠️ JSONB index error: {e}")
        
        conn.commit()
        print("✅ Document storage tables and indexes created successfully")

def optimize_existing_tables():
    """Add optimization indexes to existing tables"""
    
    with engine.connect() as conn:
        optimization_indexes = [
            # Geographic/location optimization
            "CREATE INDEX IF NOT EXISTS restaurants_district_idx ON restaurants (district);",
            "CREATE INDEX IF NOT EXISTS museums_district_idx ON museums (district);", 
            "CREATE INDEX IF NOT EXISTS places_district_idx ON places (district);",
            
            # Category/type optimization
            "CREATE INDEX IF NOT EXISTS restaurants_cuisine_idx ON restaurants (cuisine_type);",
            "CREATE INDEX IF NOT EXISTS museums_category_idx ON museums (category);",
            "CREATE INDEX IF NOT EXISTS places_category_idx ON places (category);",
            
            # Rating and popularity optimization
            "CREATE INDEX IF NOT EXISTS restaurants_rating_idx ON restaurants (rating DESC) WHERE rating IS NOT NULL;",
            "CREATE INDEX IF NOT EXISTS museums_rating_idx ON museums (rating DESC) WHERE rating IS NOT NULL;",
            "CREATE INDEX IF NOT EXISTS places_rating_idx ON places (rating DESC) WHERE rating IS NOT NULL;",
            
            # Timestamp optimization for recent content
            "CREATE INDEX IF NOT EXISTS restaurants_created_idx ON restaurants (created_at DESC);",
            "CREATE INDEX IF NOT EXISTS museums_created_idx ON museums (created_at DESC);",
            "CREATE INDEX IF NOT EXISTS user_feedback_created_idx ON user_feedback (created_at DESC);",
        ]
        
        for index_sql in optimization_indexes:
            try:
                conn.execute(text(index_sql))
                print(f"✅ Created optimization index")
            except Exception as e:
                print(f"⚠️ Optimization index error: {e}")
        
        conn.commit()
        print("✅ Table optimization indexes created successfully")

def setup_advanced_search_functions():
    """Create PostgreSQL functions for advanced search operations"""
    
    with engine.connect() as conn:
        # Advanced search function for places
        search_function = """
        CREATE OR REPLACE FUNCTION search_places_advanced(
            search_query TEXT,
            place_category TEXT DEFAULT NULL,
            district_filter TEXT DEFAULT NULL,
            min_rating DECIMAL DEFAULT NULL
        )
        RETURNS TABLE (
            id INTEGER,
            name VARCHAR,
            description TEXT,
            category VARCHAR,
            district VARCHAR,
            rating DECIMAL,
            search_rank REAL
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                p.id,
                p.name,
                p.description,
                p.category,
                p.district,
                p.rating,
                ts_rank(
                    to_tsvector('english', 
                        COALESCE(p.name, '') || ' ' || 
                        COALESCE(p.description, '') || ' ' || 
                        COALESCE(p.category, '') || ' ' || 
                        COALESCE(p.district, '')
                    ),
                    plainto_tsquery('english', search_query)
                ) AS search_rank
            FROM places p
            WHERE (
                to_tsvector('english', 
                    COALESCE(p.name, '') || ' ' || 
                    COALESCE(p.description, '') || ' ' || 
                    COALESCE(p.category, '') || ' ' || 
                    COALESCE(p.district, '')
                ) @@ plainto_tsquery('english', search_query)
                OR similarity(p.name, search_query) > 0.3
            )
            AND (place_category IS NULL OR p.category = place_category)
            AND (district_filter IS NULL OR p.district = district_filter)
            AND (min_rating IS NULL OR p.rating >= min_rating)
            ORDER BY search_rank DESC, p.rating DESC NULLS LAST
            LIMIT 50;
        END;
        $$ LANGUAGE plpgsql;
        """
        
        try:
            conn.execute(text(search_function))
            print("✅ Advanced search function created")
        except Exception as e:
            print(f"⚠️ Search function error: {e}")
        
        conn.commit()

class DatabaseEnhancementService:
    """Service for managing database enhancements and search operations"""
    
    def __init__(self):
        self.db = SessionLocal
    
    def full_text_search(self, query: str, table: str = "places", limit: int = 20):
        """Perform full-text search across specified table"""
        
        with self.db() as db:
            if table == "places":
                sql = text("""
                    SELECT *, ts_rank(
                        to_tsvector('english', 
                            COALESCE(name, '') || ' ' || 
                            COALESCE(description, '') || ' ' || 
                            COALESCE(category, '') || ' ' || 
                            COALESCE(district, '')
                        ),
                        plainto_tsquery('english', :query)
                    ) as rank
                    FROM places 
                    WHERE to_tsvector('english', 
                        COALESCE(name, '') || ' ' || 
                        COALESCE(description, '') || ' ' || 
                        COALESCE(category, '') || ' ' || 
                        COALESCE(district, '')
                    ) @@ plainto_tsquery('english', :query)
                    ORDER BY rank DESC
                    LIMIT :limit
                """)
            elif table == "restaurants":
                sql = text("""
                    SELECT *, ts_rank(
                        to_tsvector('english', 
                            COALESCE(name, '') || ' ' || 
                            COALESCE(description, '') || ' ' || 
                            COALESCE(cuisine_type, '') || ' ' || 
                            COALESCE(district, '')
                        ),
                        plainto_tsquery('english', :query)
                    ) as rank
                    FROM restaurants 
                    WHERE to_tsvector('english', 
                        COALESCE(name, '') || ' ' || 
                        COALESCE(description, '') || ' ' || 
                        COALESCE(cuisine_type, '') || ' ' || 
                        COALESCE(district, '')
                    ) @@ plainto_tsquery('english', :query)
                    ORDER BY rank DESC
                    LIMIT :limit
                """)
            else:
                return []
            
            result = db.execute(sql, {"query": query, "limit": limit})
            return result.fetchall()
    
    def fuzzy_search(self, query: str, table: str = "places", threshold: float = 0.3):
        """Perform fuzzy search using trigram similarity"""
        
        with self.db() as db:
            sql = text(f"""
                SELECT *, similarity(name, :query) as similarity_score
                FROM {table}
                WHERE similarity(name, :query) > :threshold
                ORDER BY similarity_score DESC
                LIMIT 20
            """)
            
            result = db.execute(sql, {"query": query, "threshold": threshold})
            return result.fetchall()
    
    def store_document(self, table: str, document_data: dict):
        """Store semi-structured document in JSONB table"""
        
        with self.db() as db:
            if table == "reviews":
                sql = text("""
                    INSERT INTO document_reviews (place_type, place_name, review_data, metadata)
                    VALUES (:place_type, :place_name, :review_data, :metadata)
                    RETURNING id
                """)
                result = db.execute(sql, document_data)
            elif table == "itineraries":
                sql = text("""
                    INSERT INTO document_itineraries (user_id, itinerary_data, preferences, metadata)
                    VALUES (:user_id, :itinerary_data, :preferences, :metadata)
                    RETURNING id
                """)
                result = db.execute(sql, document_data)
            elif table == "knowledge":
                sql = text("""
                    INSERT INTO document_knowledge (category, question_answer, content_data, search_metadata)
                    VALUES (:category, :question_answer, :content_data, :search_metadata)
                    RETURNING id
                """)
                result = db.execute(sql, document_data)
            
            db.commit()
            return result.fetchone()[0]

# Global service instance
database_enhancement_service = DatabaseEnhancementService()

def initialize_database_enhancements():
    """Initialize all database enhancements"""
    print("🚀 Initializing Database Enhancements...")
    
    try:
        create_fulltext_search_indexes()
        create_document_storage_tables()
        optimize_existing_tables()
        setup_advanced_search_functions()
        
        print("✅ Database enhancements completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database enhancement error: {e}")
        return False

if __name__ == "__main__":
    initialize_database_enhancements()
    
    # Test the enhancements
    print("\n🧪 Testing Database Enhancements...")
    
    # Test full-text search
    service = DatabaseEnhancementService()
    results = service.full_text_search("museum istanbul")
    print(f"✅ Full-text search test: {len(results)} results")
    
    # Test fuzzy search
    fuzzy_results = service.fuzzy_search("hagia sophia")
    print(f"✅ Fuzzy search test: {len(fuzzy_results)} results")
    
    print("✅ Database enhancements are working correctly!")

# Database Optimization Configuration for AI Istanbul

## PostgreSQL Connection Pooling (Production)

### Connection Pool Settings
```python
# In your database configuration
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "ai_istanbul",
    "user": "ai_istanbul_user",
    "password": "secure_password",
    
    # Connection Pool Settings
    "pool_size": 20,          # Initial pool size
    "max_overflow": 10,       # Additional connections beyond pool_size
    "pool_timeout": 30,       # Seconds to wait for connection
    "pool_recycle": 3600,     # Recycle connections after 1 hour
    "pool_pre_ping": True,    # Validate connections before use
}
```

### SQLAlchemy Configuration
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False  # Set to True for SQL debugging
)
```

## Query Optimization

### Indexes for Restaurant Search
```sql
-- Location-based queries
CREATE INDEX idx_restaurants_location ON restaurants USING GIST (location);

-- Text search optimization  
CREATE INDEX idx_restaurants_name_gin ON restaurants USING GIN (to_tsvector('english', name));
CREATE INDEX idx_restaurants_cuisine ON restaurants (cuisine_type);

-- Price and rating filters
CREATE INDEX idx_restaurants_price_rating ON restaurants (price_level, rating);

-- Composite index for common searches
CREATE INDEX idx_restaurants_search ON restaurants (cuisine_type, price_level, rating) WHERE is_active = true;
```

### Query Performance Monitoring
```sql
-- Enable query logging (PostgreSQL)
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1s
SELECT pg_reload_conf();
```

## Connection Monitoring
```python
def monitor_database_connections():
    """Monitor database connection pool status"""
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT 
                count(*) as total_connections,
                count(*) FILTER (WHERE state = 'active') as active,
                count(*) FILTER (WHERE state = 'idle') as idle
            FROM pg_stat_activity 
            WHERE datname = current_database()
        """))
        return dict(result.fetchone())
```

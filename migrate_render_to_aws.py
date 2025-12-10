"""
Database Migration: Render PostgreSQL → AWS RDS PostgreSQL
Migrates all tables, data, sequences, and indexes from Render to AWS RDS
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# Add backend to path
backend_path = Path(__file__).parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handles database migration from Render to AWS RDS"""
    
    def __init__(self):
        """Initialize migration connections"""
        # Source: Render PostgreSQL
        self.source_url = os.getenv('RENDER_DATABASE_URL') or os.getenv('OLD_DATABASE_URL')
        
        # Target: AWS RDS PostgreSQL
        self.target_url = os.getenv('DATABASE_URL') or os.getenv('AWS_DATABASE_URL')
        
        if not self.source_url:
            raise ValueError("❌ RENDER_DATABASE_URL or OLD_DATABASE_URL not found in environment")
        
        if not self.target_url:
            raise ValueError("❌ DATABASE_URL or AWS_DATABASE_URL not found in environment")
        
        # Ensure postgresql:// format
        if self.source_url.startswith('postgres://'):
            self.source_url = self.source_url.replace('postgres://', 'postgresql://', 1)
        if self.target_url.startswith('postgres://'):
            self.target_url = self.target_url.replace('postgres://', 'postgresql://', 1)
        
        self.source_conn = None
        self.target_conn = None
        
        # Migration stats
        self.stats = {
            'tables_migrated': 0,
            'rows_migrated': 0,
            'sequences_migrated': 0,
            'indexes_migrated': 0,
            'errors': []
        }
    
    def connect_source(self):
        """Connect to source database (Render)"""
        try:
            logger.info("📡 Connecting to source database (Render)...")
            self.source_conn = psycopg2.connect(self.source_url)
            logger.info("✅ Connected to source database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to source: {e}")
            return False
    
    def connect_target(self):
        """Connect to target database (AWS RDS)"""
        try:
            logger.info("📡 Connecting to target database (AWS RDS)...")
            self.target_conn = psycopg2.connect(self.target_url)
            logger.info("✅ Connected to target database")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to target: {e}")
            return False
    
    def get_table_list(self):
        """Get list of all user tables from source database"""
        try:
            cursor = self.source_conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            logger.info(f"📋 Found {len(tables)} tables to migrate")
            return tables
        except Exception as e:
            logger.error(f"❌ Failed to get table list: {e}")
            return []
    
    def get_table_schema(self, table_name):
        """Get CREATE TABLE statement for a table"""
        try:
            cursor = self.source_conn.cursor()
            
            # Get columns with proper type formatting
            cursor.execute(f"""
                SELECT 
                    column_name,
                    CASE 
                        -- Handle character types with length
                        WHEN data_type IN ('character varying', 'varchar') AND character_maximum_length IS NOT NULL 
                        THEN 'VARCHAR(' || character_maximum_length || ')'
                        WHEN data_type IN ('character', 'char') AND character_maximum_length IS NOT NULL 
                        THEN 'CHAR(' || character_maximum_length || ')'
                        -- Handle numeric types WITHOUT precision/scale (PostgreSQL doesn't use them like this)
                        WHEN data_type = 'numeric' AND numeric_precision IS NOT NULL AND numeric_scale IS NOT NULL AND numeric_scale > 0
                        THEN 'NUMERIC(' || numeric_precision || ',' || numeric_scale || ')'
                        WHEN data_type = 'numeric' AND numeric_precision IS NOT NULL
                        THEN 'NUMERIC(' || numeric_precision || ')'
                        -- Integer types don't take precision in PostgreSQL
                        WHEN data_type IN ('integer', 'bigint', 'smallint') 
                        THEN UPPER(data_type)
                        -- Timestamp types
                        WHEN data_type = 'timestamp without time zone'
                        THEN 'TIMESTAMP'
                        WHEN data_type = 'timestamp with time zone'
                        THEN 'TIMESTAMPTZ'
                        -- Text types
                        WHEN data_type = 'text'
                        THEN 'TEXT'
                        -- Boolean
                        WHEN data_type = 'boolean'
                        THEN 'BOOLEAN'
                        -- JSON types
                        WHEN data_type = 'json'
                        THEN 'JSON'
                        WHEN data_type = 'jsonb'
                        THEN 'JSONB'
                        -- UUID
                        WHEN data_type = 'uuid'
                        THEN 'UUID'
                        -- Array types
                        WHEN data_type = 'ARRAY'
                        THEN udt_name
                        -- Default: use data_type as-is
                        ELSE UPPER(data_type)
                    END as column_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' 
                AND table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            
            columns = []
            for row in cursor.fetchall():
                col_name, col_type, nullable, default = row
                
                col_def = f'"{col_name}" {col_type}'
                
                if nullable == 'NO':
                    col_def += ' NOT NULL'
                
                if default:
                    # Clean up default value
                    default_clean = default.strip()
                    # Handle sequence defaults
                    if 'nextval' in default_clean:
                        col_def += f' DEFAULT {default_clean}'
                    # Handle other defaults
                    elif default_clean:
                        col_def += f' DEFAULT {default_clean}'
                
                columns.append(col_def)
            
            cursor.close()
            
            if not columns:
                raise ValueError(f"No columns found for table {table_name}")
            
            return f'CREATE TABLE "{table_name}" ({", ".join(columns)});'
                
        except Exception as e:
            logger.error(f"❌ Failed to get schema for {table_name}: {e}")
            raise
    
    def migrate_sequences(self):
        """Migrate all sequences from source to target"""
        try:
            cursor = self.source_conn.cursor()
            cursor.execute("""
                SELECT sequence_name 
                FROM information_schema.sequences
                WHERE sequence_schema = 'public';
            """)
            sequences = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            if not sequences:
                logger.info("📋 No sequences to migrate")
                return True
            
            logger.info(f"📋 Migrating {len(sequences)} sequences...")
            
            target_cursor = self.target_conn.cursor()
            for seq_name in sequences:
                try:
                    # Get current value from source
                    cursor = self.source_conn.cursor()
                    cursor.execute(f"SELECT last_value FROM \"{seq_name}\";")
                    last_value = cursor.fetchone()[0]
                    cursor.close()
                    
                    # Create sequence in target
                    target_cursor.execute(f'DROP SEQUENCE IF EXISTS "{seq_name}" CASCADE;')
                    target_cursor.execute(f'CREATE SEQUENCE "{seq_name}";')
                    target_cursor.execute(f"SELECT setval('\"{seq_name}\"', {last_value});")
                    self.target_conn.commit()
                    
                    logger.info(f"  ✅ Migrated sequence: {seq_name} (value: {last_value})")
                    self.stats['sequences_migrated'] += 1
                    
                except Exception as e:
                    logger.warning(f"  ⚠️  Failed to migrate sequence {seq_name}: {e}")
            
            target_cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate sequences: {e}")
            return False
    
    def create_table_in_target(self, table_name):
        """Create table in target database"""
        try:
            # Drop table if exists
            cursor = self.target_conn.cursor()
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')
            self.target_conn.commit()
            
            # Get schema from source
            schema = self.get_table_schema(table_name)
            
            # Create table in target
            cursor.execute(schema)
            self.target_conn.commit()
            cursor.close()
            
            logger.info(f"✅ Created table: {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create table {table_name}: {e}")
            self.stats['errors'].append(f"Table creation failed for {table_name}: {str(e)}")
            return False
    
    def migrate_table_data(self, table_name):
        """Migrate all data from source table to target table"""
        try:
            # Get column names
            cursor = self.source_conn.cursor()
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns
                WHERE table_name = '{table_name}' AND table_schema = 'public'
                ORDER BY ordinal_position;
            """)
            columns = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
            # Count rows
            cursor = self.source_conn.cursor()
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}";')
            total_rows = cursor.fetchone()[0]
            cursor.close()
            
            if total_rows == 0:
                logger.info(f"⏭️  Skipping {table_name} (empty)")
                return True
            
            logger.info(f"📦 Migrating {total_rows} rows from {table_name}...")
            
            # Fetch all data
            cursor = self.source_conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute(f'SELECT * FROM "{table_name}";')
            
            # Insert in batches
            batch_size = 1000
            rows_migrated = 0
            
            target_cursor = self.target_conn.cursor()
            
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                
                # Build INSERT query
                col_names = ', '.join([f'"{col}"' for col in columns])
                placeholders = ', '.join(['%s'] * len(columns))
                insert_query = f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})'
                
                # Insert batch
                for row in rows:
                    values = [row[col] for col in columns]
                    target_cursor.execute(insert_query, values)
                    rows_migrated += 1
                
                self.target_conn.commit()
                logger.info(f"   📊 {rows_migrated}/{total_rows} rows migrated...")
            
            cursor.close()
            target_cursor.close()
            
            self.stats['rows_migrated'] += rows_migrated
            logger.info(f"✅ Migrated {rows_migrated} rows from {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to migrate data for {table_name}: {e}")
            self.stats['errors'].append(f"Data migration failed for {table_name}: {str(e)}")
            return False
    
    def migrate_indexes(self, table_name):
        """Migrate indexes for table"""
        try:
            cursor = self.source_conn.cursor()
            
            # Get indexes
            cursor.execute(f"""
                SELECT indexdef 
                FROM pg_indexes 
                WHERE tablename = '{table_name}' 
                AND schemaname = 'public'
                AND indexname NOT LIKE '%_pkey';
            """)
            
            indexes = cursor.fetchall()
            cursor.close()
            
            if not indexes:
                return True
            
            target_cursor = self.target_conn.cursor()
            
            for (index_def,) in indexes:
                try:
                    target_cursor.execute(index_def)
                    self.target_conn.commit()
                    logger.info(f"✅ Created index: {index_def[:80]}...")
                    self.stats['indexes_migrated'] += 1
                except Exception as e:
                    logger.warning(f"⚠️ Could not create index: {e}")
            
            target_cursor.close()
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to migrate indexes for {table_name}: {e}")
            return True  # Non-critical, continue
    
    def migrate(self):
        """Run full migration"""
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("🚀 Database Migration: Render → AWS RDS")
        print("="*70 + "\n")
        
        # Connect to databases
        if not self.connect_source():
            return False
        
        if not self.connect_target():
            return False
        
        # Get tables
        tables = self.get_table_list()
        
        if not tables:
            logger.error("❌ No tables found to migrate")
            return False
        
        print(f"\n📋 Tables to migrate: {', '.join(tables)}\n")
        
        # Migrate sequences first (needed for auto-increment columns)
        print("\n🔢 Migrating sequences...")
        print("-" * 70)
        self.migrate_sequences()
        
        # Migrate each table
        for i, table_name in enumerate(tables, 1):
            print(f"\n[{i}/{len(tables)}] Migrating table: {table_name}")
            print("-" * 70)
            
            # Create table
            if not self.create_table_in_target(table_name):
                continue
            
            # Migrate data
            if not self.migrate_table_data(table_name):
                continue
            
            # Migrate indexes
            self.migrate_indexes(table_name)
            
            self.stats['tables_migrated'] += 1
            print(f"✅ Completed: {table_name}\n")
        
        # Close connections
        self.source_conn.close()
        self.target_conn.close()
        
        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "="*70)
        print("📊 Migration Summary")
        print("="*70)
        print(f"✅ Tables migrated: {self.stats['tables_migrated']}/{len(tables)}")
        print(f"✅ Rows migrated: {self.stats['rows_migrated']:,}")
        print(f"✅ Sequences migrated: {self.stats['sequences_migrated']}")
        print(f"✅ Indexes migrated: {self.stats['indexes_migrated']}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        if self.stats['errors']:
            print(f"\n⚠️  Errors ({len(self.stats['errors'])}):")
            for error in self.stats['errors']:
                print(f"   - {error}")
        else:
            print("\n✅ Migration completed successfully with no errors!")
        
        print("\n" + "="*70 + "\n")
        
        # Save migration report
        report = {
            'source': 'Render PostgreSQL',
            'target': 'AWS RDS PostgreSQL',
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'stats': self.stats,
            'tables': tables
        }
        
        report_file = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Migration report saved: {report_file}\n")
        
        return True


def main():
    """Main migration function"""
    print("\n⚠️  IMPORTANT: This will migrate your database from Render to AWS RDS")
    print("⚠️  Make sure you have:")
    print("   1. Set RENDER_DATABASE_URL (or OLD_DATABASE_URL) in .env")
    print("   2. Set DATABASE_URL (AWS RDS) in .env")
    print("   3. Backed up your Render database")
    print("   4. Tested AWS RDS connection\n")
    
    response = input("Continue with migration? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("❌ Migration cancelled")
        return
    
    try:
        migrator = DatabaseMigrator()
        success = migrator.migrate()
        
        if success:
            print("\n✅ Migration completed successfully!")
            print("\n📝 Next steps:")
            print("   1. Verify data in AWS RDS")
            print("   2. Update DATABASE_URL in production")
            print("   3. Test application with new database")
            print("   4. Keep Render backup for 7-30 days\n")
        else:
            print("\n❌ Migration failed. Check logs above.")
    
    except Exception as e:
        logger.error(f"❌ Migration error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

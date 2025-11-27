#!/usr/bin/env python3
"""
Database Migration: Add Photo Columns to Restaurants Table
============================================================

This script adds photo_url and photo_reference columns to the restaurants table.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from database import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_photo_columns():
    """Add photo columns to restaurants table if they don't exist"""
    
    logger.info("=" * 80)
    logger.info("🔄 Database Migration: Adding Photo Columns")
    logger.info("=" * 80)
    
    try:
        with engine.connect() as conn:
            # Check if columns already exist
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'restaurants' 
                AND column_name IN ('photo_url', 'photo_reference')
            """))
            
            existing_columns = [row[0] for row in result]
            
            # Add photo_url if it doesn't exist
            if 'photo_url' not in existing_columns:
                logger.info("Adding photo_url column...")
                conn.execute(text("""
                    ALTER TABLE restaurants 
                    ADD COLUMN photo_url VARCHAR(500)
                """))
                conn.commit()
                logger.info("✅ Added photo_url column")
            else:
                logger.info("✓ photo_url column already exists")
            
            # Add photo_reference if it doesn't exist
            if 'photo_reference' not in existing_columns:
                logger.info("Adding photo_reference column...")
                conn.execute(text("""
                    ALTER TABLE restaurants 
                    ADD COLUMN photo_reference VARCHAR(500)
                """))
                conn.commit()
                logger.info("✅ Added photo_reference column")
            else:
                logger.info("✓ photo_reference column already exists")
            
            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ Migration Complete!")
            logger.info("=" * 80)
            
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


if __name__ == '__main__':
    add_photo_columns()

#!/usr/bin/env python3
"""
Fetch and Store Restaurant Photos
==================================

This script fetches restaurant photos from Google Places API once
and stores them in the database, so we don't need to make API calls every time.

Features:
- Fetches photo from Google Places API
- Downloads and stores in local storage (or S3)
- Updates database with photo URL
- Rate limiting to avoid quota issues
"""

import os
import sys
import time
import requests
import logging
from pathlib import Path
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_db
from models import Restaurant

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
GOOGLE_PLACES_API_KEY = os.getenv('GOOGLE_PLACES_API_KEY')
PHOTO_STORAGE_PATH = Path(__file__).parent.parent / 'static' / 'restaurant_photos'
MAX_PHOTO_SIZE = 800  # Max width/height in pixels
RATE_LIMIT_DELAY = 0.5  # Seconds between API calls

# Create photo storage directory
PHOTO_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


def fetch_google_place_photo(place_id: str, api_key: str) -> Optional[str]:
    """
    Fetch photo reference from Google Places API
    
    Args:
        place_id: Google Place ID
        api_key: Google API key
        
    Returns:
        Photo reference string or None
    """
    try:
        # Get place details including photos
        url = 'https://maps.googleapis.com/maps/api/place/details/json'
        params = {
            'place_id': place_id,
            'fields': 'photos',
            'key': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'OK':
            logger.warning(f"Places API error for {place_id}: {data.get('status')}")
            return None
            
        photos = data.get('result', {}).get('photos', [])
        if not photos:
            logger.info(f"No photos found for place_id: {place_id}")
            return None
            
        # Return first photo reference
        return photos[0].get('photo_reference')
        
    except Exception as e:
        logger.error(f"Error fetching photo reference for {place_id}: {e}")
        return None


def download_photo(photo_reference: str, api_key: str, output_path: Path) -> bool:
    """
    Download photo from Google Places API
    
    Args:
        photo_reference: Google Places photo reference
        api_key: Google API key
        output_path: Where to save the photo
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Construct photo URL
        url = 'https://maps.googleapis.com/maps/api/place/photo'
        params = {
            'maxwidth': MAX_PHOTO_SIZE,
            'photo_reference': photo_reference,
            'key': api_key
        }
        
        # Download photo
        response = requests.get(url, params=params, timeout=30, stream=True)
        response.raise_for_status()
        
        # Save to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Downloaded photo to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading photo: {e}")
        return False


def process_restaurant(restaurant: Restaurant, db: Session, api_key: str) -> bool:
    """
    Process a single restaurant - fetch and store photo
    
    Args:
        restaurant: Restaurant model instance
        db: Database session
        api_key: Google API key
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Skip if already has photo
        if restaurant.photo_url:
            logger.info(f"Restaurant {restaurant.id} already has photo: {restaurant.photo_url}")
            return True
            
        # Skip if no place_id
        if not restaurant.place_id:
            logger.warning(f"Restaurant {restaurant.id} has no place_id")
            return False
            
        logger.info(f"Processing restaurant: {restaurant.name} (ID: {restaurant.id})")
        
        # Fetch photo reference
        photo_ref = fetch_google_place_photo(restaurant.place_id, api_key)
        if not photo_ref:
            logger.warning(f"No photo reference for {restaurant.name}")
            return False
            
        # Download photo
        photo_filename = f"restaurant_{restaurant.id}.jpg"
        photo_path = PHOTO_STORAGE_PATH / photo_filename
        
        if download_photo(photo_ref, api_key, photo_path):
            # Update database with photo URL
            restaurant.photo_url = f"/static/restaurant_photos/{photo_filename}"
            restaurant.photo_reference = photo_ref  # Store reference for future updates
            db.commit()
            logger.info(f"✅ Updated restaurant {restaurant.id} with photo")
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error processing restaurant {restaurant.id}: {e}")
        db.rollback()
        return False


def fetch_all_restaurant_photos(limit: Optional[int] = None, skip_existing: bool = True):
    """
    Fetch photos for all restaurants in database
    
    Args:
        limit: Maximum number of restaurants to process (None = all)
        skip_existing: Skip restaurants that already have photos
    """
    if not GOOGLE_PLACES_API_KEY:
        logger.error("GOOGLE_PLACES_API_KEY not set in environment")
        return
        
    logger.info("=" * 80)
    logger.info("🖼️  Restaurant Photo Fetcher")
    logger.info("=" * 80)
    logger.info(f"Photo storage: {PHOTO_STORAGE_PATH}")
    logger.info(f"Max photo size: {MAX_PHOTO_SIZE}px")
    logger.info(f"Rate limit delay: {RATE_LIMIT_DELAY}s")
    logger.info("")
    
    # Get database session
    db = next(get_db())
    
    try:
        # Query restaurants
        query = db.query(Restaurant)
        
        if skip_existing:
            query = query.filter(Restaurant.photo_url == None)
            
        if limit:
            query = query.limit(limit)
            
        restaurants = query.all()
        total = len(restaurants)
        
        logger.info(f"Found {total} restaurants to process")
        logger.info("")
        
        # Process each restaurant
        success_count = 0
        failed_count = 0
        
        for i, restaurant in enumerate(restaurants, 1):
            logger.info(f"[{i}/{total}] Processing: {restaurant.name}")
            
            if process_restaurant(restaurant, db, GOOGLE_PLACES_API_KEY):
                success_count += 1
            else:
                failed_count += 1
                
            # Rate limiting
            if i < total:
                time.sleep(RATE_LIMIT_DELAY)
                
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"✅ Complete!")
        logger.info(f"   Success: {success_count}")
        logger.info(f"   Failed: {failed_count}")
        logger.info(f"   Total: {total}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        db.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch and store restaurant photos')
    parser.add_argument('--limit', type=int, help='Limit number of restaurants to process')
    parser.add_argument('--all', action='store_true', help='Process all restaurants (including those with photos)')
    parser.add_argument('--test', action='store_true', help='Test mode - only process 5 restaurants')
    
    args = parser.parse_args()
    
    limit = args.limit
    if args.test:
        limit = 5
        
    skip_existing = not args.all
    
    fetch_all_restaurant_photos(limit=limit, skip_existing=skip_existing)

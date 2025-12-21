#!/usr/bin/env python3
"""
Update blog posts with S3 featured image URLs
Maps each blog post to an appropriate image from AWS S3
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from sqlalchemy import create_engine, text

# S3 bucket base URL
S3_BASE_URL = "https://aistanbul-info.s3.eu-central-1.amazonaws.com/blog"

# Map blog post topics to appropriate images
IMAGE_MAPPING = {
    # Istanbul landmarks and views
    "hidden gems": f"{S3_BASE_URL}/hidden-gems-istanbul.jpg",
    "rooftop": f"{S3_BASE_URL}/hidden-gems-istanbul.jpg",
    "sultanahmet": f"{S3_BASE_URL}/sultanahmet-guide.jpg",
    "hagia sophia": f"{S3_BASE_URL}/sultanahmet-guide.jpg",
    "blue mosque": f"{S3_BASE_URL}/sultanahmet-guide.jpg",
    "topkapi": f"{S3_BASE_URL}/sultanahmet-guide.jpg",
    
    # Food and dining
    "coffee": f"{S3_BASE_URL}/turkish-street-food.jpg",
    "food": f"{S3_BASE_URL}/turkish-street-food.jpg",
    "street food": f"{S3_BASE_URL}/turkish-street-food.jpg",
    "breakfast": f"{S3_BASE_URL}/turkish-street-food.jpg",
    "restaurant": f"{S3_BASE_URL}/turkish-street-food.jpg",
    "dining": f"{S3_BASE_URL}/turkish-street-food.jpg",
    "cuisine": f"{S3_BASE_URL}/turkish-street-food.jpg",
    
    # Nightlife and entertainment
    "nightlife": f"{S3_BASE_URL}/beyoglu-nightlife.jpg",
    "beyoğlu": f"{S3_BASE_URL}/beyoglu-nightlife.jpg",
    "beyoglu": f"{S3_BASE_URL}/beyoglu-nightlife.jpg",
    "galata": f"{S3_BASE_URL}/beyoglu-nightlife.jpg",
    "taksim": f"{S3_BASE_URL}/beyoglu-nightlife.jpg",
    "bar": f"{S3_BASE_URL}/beyoglu-nightlife.jpg",
    "music": f"{S3_BASE_URL}/beyoglu-nightlife.jpg",
    
    # Transportation
    "transport": f"{S3_BASE_URL}/istanbul-transport.jpg",
    "metro": f"{S3_BASE_URL}/istanbul-transport.jpg",
    "ferry": f"{S3_BASE_URL}/istanbul-transport.jpg",
    "bosphorus": f"{S3_BASE_URL}/istanbul-transport.jpg",
    "getting around": f"{S3_BASE_URL}/istanbul-transport.jpg",
    
    # Markets and shopping
    "market": f"{S3_BASE_URL}/hidden-gems-istanbul.jpg",
    "shopping": f"{S3_BASE_URL}/hidden-gems-istanbul.jpg",
    "bazaar": f"{S3_BASE_URL}/hidden-gems-istanbul.jpg",
    
    # Default fallback image
    "default": f"{S3_BASE_URL}/hidden-gems-istanbul.jpg"
}

def get_image_for_post(title, content):
    """Determine the best image for a blog post based on title and content"""
    search_text = (title + " " + content[:500]).lower()
    
    for keyword, image_url in IMAGE_MAPPING.items():
        if keyword in search_text:
            return image_url
    
    return IMAGE_MAPPING["default"]

def update_featured_images():
    """Update all blog posts with featured images"""
    
    print("\n" + "="*80)
    print("🖼️  Updating Blog Posts with S3 Featured Images")
    print("="*80)
    
    database_url = "postgresql://postgres:NewSecurePassword123!@127.0.0.1:5433/postgres"
    engine = create_engine(database_url)
    
    with engine.connect() as conn:
        # Get all blog posts
        result = conn.execute(text("""
            SELECT id, title, content 
            FROM blog_posts 
            ORDER BY id
        """))
        
        posts = result.fetchall()
        print(f"\n📝 Found {len(posts)} blog posts to update\n")
        
        updated_count = 0
        for post in posts:
            post_id, title, content = post
            
            # Determine best image
            image_url = get_image_for_post(title, content or "")
            
            # Update the post
            conn.execute(text("""
                UPDATE blog_posts 
                SET featured_image = :image_url,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = :post_id
            """), {"image_url": image_url, "post_id": post_id})
            conn.commit()
            
            # Extract image filename for display
            image_name = image_url.split('/')[-1]
            print(f"  ✅ [{post_id:2d}] {title[:50]:<50} → {image_name}")
            updated_count += 1
        
        print(f"\n" + "="*80)
        print(f"✅ Updated {updated_count} blog posts with featured images!")
        print("="*80 + "\n")
        
        # Show summary
        print("📊 Image Distribution:")
        print("-"*80)
        result = conn.execute(text("""
            SELECT 
                substring(featured_image from '.*/(.+)$') as image_name,
                COUNT(*) as count
            FROM blog_posts
            WHERE featured_image IS NOT NULL
            GROUP BY featured_image
            ORDER BY count DESC
        """))
        
        for row in result:
            print(f"  {row[0]:<40} {row[1]:>3} posts")
        
        print("-"*80 + "\n")

if __name__ == "__main__":
    update_featured_images()

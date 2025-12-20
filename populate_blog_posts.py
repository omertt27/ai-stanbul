#!/usr/bin/env python3
"""
Populate Blog Posts in AWS Database
Adds sample blog posts with S3-hosted images to the PostgreSQL database

Usage:
    cd backend
    python ../populate_blog_posts.py
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / 'backend'
sys.path.insert(0, str(backend_path))

from database import SessionLocal
from models import BlogPost
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 bucket URL
S3_BUCKET = "aistanbul-info"
S3_REGION = "eu-central-1"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"

# Sample blog posts with S3 image URLs
SAMPLE_POSTS = [
    {
        "title": "Ultimate Guide to Istanbul's Hidden Gems",
        "slug": "istanbul-hidden-gems-guide",
        "content": """# Discover Istanbul's Secret Treasures

Istanbul is a city of endless discoveries, where every corner holds a story waiting to be unveiled. Beyond the famous landmarks lies a world of hidden gems that only locals know about.

## 🔮 Secret Neighborhoods

### Fener & Balat
These colorful Byzantine neighborhoods offer an authentic glimpse into Istanbul's multicultural past:
- **Historic Greek Houses**: Beautifully painted in vibrant colors
- **Antique Shops**: Treasure hunting opportunities around every corner
- **Local Cafes**: Experience authentic Turkish coffee culture
- **Photography Paradise**: Instagram-worthy streets at every turn

### Kuzguncuk
A peaceful village on the Asian side that time forgot:
- **Multi-cultural Heritage**: Churches, synagogues, and mosques standing side by side
- **Wooden Houses**: Perfectly preserved Ottoman-era architecture
- **Artisan Workshops**: Watch local craftspeople at work
- **Quiet Cafes**: Perfect spots for reading and relaxation

## 🍽️ Hidden Food Spots

### Local Lokantas
- **Develi Restaurant** (Samatya): Famous for their lamb dishes
- **Pandeli** (Eminönü): Historic Ottoman cuisine in a stunning setting
- **Kiva Han** (Beyoğlu): Traditional meze selection that's second to none

### Street Food Secrets
- **Balık Ekmek** at Galata Bridge: Fresh fish sandwiches right by the water
- **Dönerci Engin** (Aksaray): Locals swear it's the best döner in the city
- **Kokoreç at Ortaköy**: A late-night specialty you must try

## 🎨 Cultural Discoveries

Explore lesser-known museums and galleries that showcase Istanbul's rich artistic heritage. From underground cisterns to contemporary art spaces, there's always something new to discover.

Ready to explore Istanbul like a local? These hidden gems will give you authentic experiences that most tourists never discover!""",
        "excerpt": "Discover the secret neighborhoods, hidden food spots, and cultural treasures that only locals know about in Istanbul.",
        "author": "Istanbul Explorer",
        "status": "published",
        "featured_image": f"{S3_BASE_URL}/blog/istanbul-balat-colorful-houses.jpg",
        "category": "Travel Guide",
        "tags": ["hidden gems", "local culture", "neighborhoods", "insider tips"],
        "district": "Istanbul (general)",
        "likes_count": 23,
        "views": 156
    },
    {
        "title": "Best Turkish Street Food You Must Try",
        "slug": "best-turkish-street-food",
        "content": """# Turkish Street Food Adventure

Turkish street food is a culinary journey that will tantalize your taste buds and introduce you to the heart of Turkish culture. Here's your complete guide to the must-try street foods in Istanbul.

## 🥙 Essential Street Foods

### Döner Kebab
The undisputed king of Turkish street food:
- **Best Places**: Öz Konyalı, Karadeniz Döner
- **What to Order**: Döner in bread (ekmek arası) or wrapped in lavash
- **Pro Tip**: Always ask for extra salad and sauce

### Balık Ekmek (Fish Sandwich)
Istanbul's most iconic waterfront snack:
- **Where**: Eminönü waterfront, underneath Galata Bridge
- **Fish**: Usually fresh mackerel, grilled to perfection
- **Sides**: Pickled vegetables and onions for the perfect balance

### Simit
The Turkish sesame bagel, perfect for any time:
- **Toppings**: Traditional plain, with sesame, or filled with cheese
- **Best With**: A glass of Turkish tea
- **Where**: Street vendors on every corner

## 🍢 Grilled Specialties

### Kokoreç
For the adventurous - grilled lamb intestines:
- **Flavor**: Rich, spiced, and absolutely delicious
- **Best At**: Ortaköy square, especially late at night
- **Served**: In fresh bread with spices and lemon

### Midye Dolma (Stuffed Mussels)
A popular seaside snack:
- **Filling**: Aromatic rice with pine nuts and spices
- **How to Eat**: Squeeze fresh lemon juice over them
- **Important**: Only buy from busy vendors for freshness

## 🥘 Traditional Favorites

### Kumpir (Loaded Baked Potato)
Ortaköy's famous street food:
- **Base**: Huge baked potato mashed with butter and cheese
- **Toppings**: Choose from dozens of options
- **Best**: Perfect for lunch with a Bosphorus view

### Lahmacun (Turkish Pizza)
Thin, crispy, and absolutely delicious:
- **Topping**: Minced meat, vegetables, and herbs
- **Served**: Rolled up with fresh parsley and lemon
- **Where**: Street stalls throughout the city

Experience the authentic flavors of Istanbul's street food scene - it's affordable, delicious, and unforgettable!""",
        "excerpt": "From döner kebab to balık ekmek, explore the delicious world of Turkish street food and where to find the best bites in Istanbul.",
        "author": "Food Explorer",
        "status": "published",
        "featured_image": f"{S3_BASE_URL}/blog/turkish-street-food-doner.jpg",
        "category": "Food & Cuisine",
        "tags": ["street food", "turkish cuisine", "local food", "budget eating"],
        "district": "Istanbul (general)",
        "likes_count": 15,
        "views": 89
    },
    {
        "title": "Navigating Istanbul's Transport System Like a Pro",
        "slug": "istanbul-transport-guide",
        "content": """# Master Istanbul's Transportation

Istanbul's transport system can seem overwhelming at first, but with the right knowledge, you'll be navigating the city like a true Istanbullu. Here's your complete guide.

## 🎫 Istanbul Card (Istanbulkart)

### The Essential Card
Your key to all public transport in the city:
- **Where to Buy**: Metro stations, kiosks, post offices
- **Cost**: 13 TL for the card + credit you add
- **Works On**: Metro, bus, tram, ferry, funicular, and Marmaray
- **Discounts**: Transfers within 2 hours get discounted rates

### Loading Your Card
- **Machines**: Available at all stations with English language option
- **Payment**: Both cash and credit cards accepted
- **Pro Tip**: Load enough for several days to avoid queues

## 🚇 Metro System

### Main Metro Lines

**M1 Red Line**: Your airport connection
- **Route**: Istanbul Airport → Zeytinburnu → Aksaray
- **Useful For**: Airport transfers to city center
- **Travel Time**: About 45 minutes to the center

**M2 Green Line**: The tourist favorite
- **Key Stops**: Şişhane (Galata), Vezneciler (Grand Bazaar), Taksim
- **Most Used**: By both tourists and locals
- **Frequency**: Every 2-4 minutes during peak hours

**M4 Pink Line**: Explore the Asian side
- **Route**: Kadıköy to Tavşantepe
- **Connects**: Major Asian side neighborhoods
- **Perfect For**: Discovering authentic local life

## 🚋 Tram Lines

### T1 Historic Tram - The Golden Route
**Route**: Kabataş → Eminönü → Sultanahmet → Zeytinburnu

Major tourist stops include:
- **Karaköy**: Galata Tower and waterfront
- **Eminönü**: Spice Bazaar and ferries
- **Sultanahmet**: Blue Mosque and Hagia Sophia
- **Beyazıt**: Grand Bazaar entrance

## 🚢 Ferry System

### The Most Scenic Way to Travel

**European to Asian Side**:
- **Karaköy ↔ Kadıköy**: The most popular route
- **Eminönü ↔ Üsküdar**: Historic connection
- **Kabataş ↔ Üsküdar**: Quick and convenient

**Bosphorus Tours**:
- **Short Tour**: Eminönü to Anadolu Kavağı (1.5 hours)
- **Long Tour**: Full Bosphorus exploration
- **Pro Tip**: Public ferries are much cheaper than tourist boats

## 💡 Insider Tips

### Best Practices
- **Rush Hours**: Avoid 7:30-9:30 AM and 5:30-7:30 PM if possible
- **Direction**: Always check the end station name for direction
- **Apps**: Use Moovit or Google Maps for real-time updates
- **Accessibility**: Most stations are wheelchair accessible

### Money-Saving Tips
- **Use Istanbulkart**: Much cheaper than single tickets
- **Transfer Discounts**: Switch lines within 2 hours for discounted fares
- **Weekly Passes**: Not available - just load your card as needed

Master these transportation basics and you'll navigate Istanbul with confidence!""",
        "excerpt": "A comprehensive guide to Istanbul's metro, tram, ferry, and bus systems - navigate the city like a local.",
        "author": "Transit Guide",
        "status": "published",
        "featured_image": f"{S3_BASE_URL}/blog/istanbul-metro-tram.jpg",
        "category": "Travel Tips",
        "tags": ["transportation", "metro", "tram", "ferry", "travel tips"],
        "district": "Istanbul (general)",
        "likes_count": 31,
        "views": 142
    },
    {
        "title": "Beyoğlu District: The Heart of Modern Istanbul",
        "slug": "beyoglu-district-guide",
        "content": """# Beyoğlu: Where History Meets Modernity

Beyoğlu is Istanbul's most vibrant district, where historic architecture meets contemporary culture, creating an atmosphere unlike anywhere else in the city.

## 🏛️ Historic Landmarks

### Galata Tower
The iconic medieval stone tower:
- **Height**: 67 meters with panoramic views
- **History**: Built in 1348 by the Genoese
- **Tip**: Visit at sunset for magical views
- **Avoid**: Weekends when it's most crowded

### İstiklal Avenue
The beating heart of Beyoğlu:
- **Length**: 1.4 km pedestrian street
- **Historic Tram**: Red nostalgic tram runs the length
- **Shopping**: Mix of international brands and local shops
- **Nightlife**: Countless bars, clubs, and live music venues

## 🎨 Art & Culture

### Museums and Galleries
- **Pera Museum**: Outstanding Orientalist paintings collection
- **SALT Galata**: Contemporary art and cultural programs
- **Istanbul Modern**: Turkey's leading contemporary art museum

### Performance Venues
- **Babylon**: Legendary live music venue
- **Akbank Sanat**: Cultural center with exhibitions and concerts
- **Nardis Jazz Club**: Intimate jazz performances

## 🍴 Culinary Scene

### Traditional Meyhanes
Turkish taverns serving meze and rakı:
- **Nevizade Street**: Famous for traditional meyhanes
- **Asmalımescit**: Trendy bars and restaurants
- **Çiçek Pasajı**: Historic passage filled with taverns

### International Cuisine
- **French**: Fransız Sokağı (French Street)
- **Asian**: Numerous sushi and Asian fusion restaurants
- **Middle Eastern**: Authentic Lebanese and Syrian cuisine

## 🌙 Nightlife

### Bar Districts
- **Asmalımescit**: Sophisticated cocktail bars
- **Nevizade**: Traditional rakı taverns
- **Karaköy**: Waterfront bars with Bosphorus views

### Live Music
- **Babylon**: International and local acts
- **Nardis Jazz Club**: Jazz every night
- **Salon İKSV**: Concert hall with diverse programming

## 🛍️ Shopping

### Unique Finds
- **Çukurcuma**: Antique shops and vintage treasures
- **Galata**: Artisan workshops and independent boutiques
- **İstiklal Avenue**: Everything from haute couture to street fashion

Beyoğlu never sleeps - there's always something happening in this dynamic district!""",
        "excerpt": "Explore Beyoğlu, Istanbul's most dynamic district, from historic Galata Tower to trendy İstiklal Avenue and vibrant nightlife.",
        "author": "District Explorer",
        "status": "published",
        "featured_image": f"{S3_BASE_URL}/blog/beyoglu-istiklal-avenue.jpg",
        "category": "Districts",
        "tags": ["beyoglu", "galata", "istiklal", "nightlife", "culture"],
        "district": "Beyoğlu",
        "likes_count": 28,
        "views": 134
    },
    {
        "title": "Sultanahmet: Istanbul's Historic Peninsula",
        "slug": "sultanahmet-historic-guide",
        "content": """# Sultanahmet: The Heart of Historic Istanbul

Sultanahmet is where empires rose and fell, where Byzantine and Ottoman legacies intertwine, and where some of the world's most magnificent monuments stand as testaments to Istanbul's rich history.

## 🕌 Must-See Monuments

### Hagia Sophia (Ayasofya)
One of the world's greatest architectural achievements:
- **History**: Built in 537 AD, converted to mosque, then museum, now mosque
- **Architecture**: Massive dome that inspired countless buildings
- **Visiting**: Free entry as a mosque, dress modestly
- **Best Time**: Early morning to avoid crowds

### Blue Mosque (Sultan Ahmed Mosque)
Istanbul's most iconic Ottoman mosque:
- **Built**: 1616 by Sultan Ahmed I
- **Unique**: Six minarets (unusual for its time)
- **Interior**: Stunning blue İznik tiles
- **Prayer Times**: Closed to tourists during prayers

### Topkapı Palace
The grand palace of Ottoman sultans:
- **Size**: Massive complex with multiple courtyards
- **Highlights**: Harem, Treasury, Sacred Relics
- **Ticket**: Separate tickets for palace and Harem
- **Time Needed**: Minimum 3-4 hours

## 🏛️ Museums

### Istanbul Archaeology Museums
Three museums in one complex:
- **Archaeological Museum**: Ancient civilizations
- **Museum of the Ancient Orient**: Mesopotamian artifacts
- **Tiled Kiosk Museum**: Ottoman tiles and ceramics

### Turkish and Islamic Arts Museum
- **Location**: İbrahim Paşa Palace
- **Collection**: Carpets, calligraphy, ceramics
- **Highlight**: World's finest carpet collection

## 🌳 Parks and Gardens

### Gülhane Park
Beautiful park next to Topkapı Palace:
- **History**: Former palace gardens
- **Perfect For**: Picnics and relaxation
- **Spring**: Tulip festival in April

### Sultanahmet Square
The historic heart between Hagia Sophia and Blue Mosque:
- **Atmosphere**: Always buzzing with tourists and locals
- **Events**: Cultural events and festivals
- **Cafes**: Surrounding cafes with great views

## 🍽️ Where to Eat

### Traditional Turkish Cuisine
- **Tarihi Sultanahmet Köftecisi**: Famous meatballs since 1920
- **Pandeli**: Historic restaurant in Spice Bazaar
- **Balıkçı Sabahattin**: Traditional fish restaurant

### Rooftop Restaurants
Stunning views of Hagia Sophia and Blue Mosque:
- **Seven Hills Restaurant**: Panoramic terrace
- **Albura Kathisma**: Ottoman mansion setting
- **Roof Mezze**: Modern Turkish cuisine

## 💡 Visiting Tips

### Planning Your Visit
- **Time**: Allocate minimum 2 full days
- **Start Early**: Major sites open at 9 AM
- **Museum Pass**: Consider Istanbul Museum Pass
- **Dress Code**: Modest clothing for mosques

### Avoiding Crowds
- **Best Months**: April-May, September-October
- **Worst Times**: Summer months and weekends
- **Early Morning**: Sites are quieter before 10 AM

### Getting There
- **Tram**: T1 to Sultanahmet station
- **Walking**: Easy to explore on foot
- **Tours**: Many walking tours available

Sultanahmet is living history - every stone tells a story of empires, faith, and human achievement!""",
        "excerpt": "Discover Sultanahmet's magnificent monuments including Hagia Sophia, Blue Mosque, and Topkapı Palace in Istanbul's historic heart.",
        "author": "History Explorer",
        "status": "published",
        "featured_image": f"{S3_BASE_URL}/blog/sultanahmet-blue-mosque.jpg",
        "category": "Districts",
        "tags": ["sultanahmet", "hagia sophia", "blue mosque", "topkapi palace", "history"],
        "district": "Sultanahmet",
        "likes_count": 45,
        "views": 287
    }
]


def populate_blog_posts():
    """Populate the database with sample blog posts"""
    db = SessionLocal()
    
    try:
        # Check existing posts
        existing_count = db.query(BlogPost).count()
        logger.info(f"📊 Current blog posts in database: {existing_count}")
        
        # Add sample posts
        added = 0
        skipped = 0
        
        for i, post_data in enumerate(SAMPLE_POSTS, 1):
            # Check if post already exists by slug
            existing_post = db.query(BlogPost).filter(BlogPost.slug == post_data['slug']).first()
            
            if existing_post:
                logger.info(f"⏭️  Skipping '{post_data['title']}' - already exists")
                skipped += 1
                continue
            
            # Create new blog post
            post = BlogPost(
                title=post_data['title'],
                slug=post_data['slug'],
                content=post_data['content'],
                excerpt=post_data['excerpt'],
                author=post_data['author'],
                status=post_data['status'],
                featured_image=post_data['featured_image'],
                category=post_data['category'],
                tags=post_data['tags'],
                district=post_data['district'],
                likes_count=post_data['likes_count'],
                views=post_data['views'],
                created_at=datetime.utcnow() - timedelta(days=(len(SAMPLE_POSTS) - i) * 3),
                published_at=datetime.utcnow() - timedelta(days=(len(SAMPLE_POSTS) - i) * 3)
            )
            
            db.add(post)
            added += 1
            logger.info(f"✅ Added: '{post_data['title']}'")
        
        # Commit all changes
        db.commit()
        
        # Final count
        final_count = db.query(BlogPost).count()
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 Blog Post Population Complete!")
        logger.info("=" * 80)
        logger.info(f"✅ Posts added: {added}")
        logger.info(f"⏭️  Posts skipped: {skipped}")
        logger.info(f"📊 Total posts in database: {final_count}")
        logger.info("=" * 80)
        
        # Show sample posts
        logger.info("\n📋 Sample posts:")
        sample_posts = db.query(BlogPost).order_by(BlogPost.created_at.desc()).limit(5).all()
        for post in sample_posts:
            logger.info(f"  - {post.id}: {post.title}")
            logger.info(f"    Category: {post.category}, District: {post.district}")
            logger.info(f"    Image: {post.featured_image}")
            logger.info(f"    Likes: {post.likes_count}, Views: {post.views}")
            logger.info("")
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error populating blog posts: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("🚀 Starting blog post population...")
    logger.info(f"📦 S3 Bucket: {S3_BUCKET}")
    logger.info(f"🌐 S3 Base URL: {S3_BASE_URL}")
    logger.info(f"📝 Posts to add: {len(SAMPLE_POSTS)}")
    logger.info("")
    
    populate_blog_posts()

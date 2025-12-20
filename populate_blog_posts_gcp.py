#!/usr/bin/env python3
"""
Populate Blog Posts in GCP Cloud SQL Database

This script creates sample blog posts in your Google Cloud SQL (PostgreSQL) database
with references to images that should be stored in AWS S3.

Usage:
    python populate_blog_posts_gcp.py

Author: AI Istanbul Team
Date: December 2024
"""

import sys
import os
from datetime import datetime, timedelta

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Sample blog posts with Istanbul content
SAMPLE_BLOG_POSTS = [
    {
        "title": "Top 10 Hidden Gems in Istanbul",
        "slug": "top-10-hidden-gems-istanbul",
        "content": """
# Discover Istanbul's Best Kept Secrets

Istanbul is a city of endless wonders, but some of its most magical spots remain hidden from the typical tourist trail. Here are 10 hidden gems that will make your Istanbul experience truly unforgettable.

## 1. Balat Neighborhood
This colorful Byzantine neighborhood is a photographer's paradise with its rainbow-painted houses, antique shops, and authentic cafes.

## 2. Pierre Loti Hill
Named after the French novelist, this hilltop cafe offers stunning views of the Golden Horn. Take the cable car up for a romantic sunset.

## 3. Fener Greek Patriarchate
The seat of the Ecumenical Patriarch, this historic complex showcases Byzantine architecture at its finest.

## 4. Miniaturk Park
See all of Turkey in one place! This park features miniature replicas of Turkey's most famous landmarks.

## 5. Rahmi M. Koç Museum
A fascinating industrial museum housed in a restored anchor factory, perfect for families and history buffs.

## 6. Kuzguncuk
A peaceful village on the Asian side where mosques, churches, and synagogues sit side by side.

## 7. Çukurcuma Antique District
Hunt for treasures in this neighborhood filled with antique shops and vintage boutiques.

## 8. Yedikule Fortress
The "Fortress of Seven Towers" offers a glimpse into Byzantine and Ottoman military architecture.

## 9. Suleymaniye Hamam
Experience an authentic Turkish bath in this 16th-century hamam near the Suleymaniye Mosque.

## 10. Büyükada Island
Escape the city chaos on this car-free Princes' Island. Rent a bicycle and explore Victorian mansions.

Visit these hidden gems and experience Istanbul like a local!
        """,
        "excerpt": "Discover 10 hidden gems in Istanbul that most tourists never see. From colorful Balat to peaceful Büyükada Island.",
        "author": "Istanbul Explorer",
        "status": "published",
        "featured_image": "https://aistanbul-info.s3.eu-central-1.amazonaws.com/blog/hidden-gems-istanbul.jpg",
        "category": "Travel Guide",
        "district": "Istanbul (general)",
        "tags": ["hidden gems", "local culture", "neighborhoods", "insider tips"],
        "views": 245,
        "likes_count": 34,
        "published_at": datetime.utcnow() - timedelta(days=7)
    },
    {
        "title": "Best Turkish Street Food You Must Try",
        "slug": "best-turkish-street-food",
        "content": """
# Turkish Street Food Adventure

Turkish street food is a culinary journey that will tantalize your taste buds and introduce you to the heart of Turkish culture.

## Essential Street Foods

### Simit - The Turkish Bagel
Start your day like a local with a fresh simit from street vendors. This sesame-crusted bread ring is perfect with Turkish tea.

### Döner Kebab
The king of Turkish street food! Fresh meat carved from a rotating spit, served in bread or lavash with salad and sauce.

### Balık Ekmek (Fish Sandwich)
Istanbul's iconic sandwich! Fresh grilled mackerel served in bread with onions and lettuce, best enjoyed by the Bosphorus.

### Midye Dolma (Stuffed Mussels)
Rice-stuffed mussels seasoned with spices, pine nuts, and herbs. Squeeze lemon on top for the perfect bite.

### Kokoreç
For the adventurous eater! Grilled lamb intestines seasoned with oregano and red pepper flakes.

## Where to Find the Best

- **Eminönü**: Best for balık ekmek and döner
- **Ortaköy**: Famous for kumpir (stuffed baked potatoes)
- **Kadıköy**: Less touristy, authentic local food
- **Beyoğlu**: Great variety and late-night options

## Pro Tips

1. Look for busy vendors - high turnover means fresher food
2. Eat where the locals eat
3. Don't be shy to ask for recommendations
4. Carry cash - many vendors don't accept cards
5. Learn basic Turkish phrases - it enhances the experience!

Turkish street food isn't just about eating—it's about experiencing Turkish culture and hospitality!
        """,
        "excerpt": "A comprehensive guide to Turkish street food, from simit to döner kebab. Discover where to find the best flavors in Istanbul.",
        "author": "Food Explorer",
        "status": "published",
        "featured_image": "https://aistanbul-info.s3.eu-central-1.amazonaws.com/blog/turkish-street-food.jpg",
        "category": "Food & Cuisine",
        "district": "Istanbul (general)",
        "tags": ["street food", "turkish cuisine", "local food", "budget eating"],
        "views": 189,
        "likes_count": 27,
        "published_at": datetime.utcnow() - timedelta(days=5)
    },
    {
        "title": "Navigating Istanbul's Public Transportation",
        "slug": "istanbul-public-transportation-guide",
        "content": """
# Master Istanbul's Transportation System

Istanbul's public transport is efficient, affordable, and easy to use once you know the basics.

## Istanbulkart - Your Essential Card

The Istanbulkart is your key to all public transport in Istanbul.

**Where to buy**: Metro stations, kiosks, and some convenience stores
**Cost**: 70 TL for the card + credit you add
**Coverage**: Metro, tram, bus, ferry, funicular, and Marmaray

### Loading Your Card
- Use machines at metro stations (English available)
- Add at least 100 TL for several days of travel
- Transfers within 2 hours get discounted rates

## Metro System

### Key Lines for Tourists

**M1 (Red Line)**: Airport → Yenikapı
- Connects IST Airport to city center
- Journey time: ~45 minutes

**M2 (Green Line)**: Yenikapı → Hacıosman
- Stops at: Taksim, Şişhane (Galata), Vezneciler (Grand Bazaar)
- Most useful for tourists

**M4 (Pink Line)**: Kadıköy → Tavşantepe
- Explores the Asian side

## Tram Lines

### T1 (Historic Tram)
The most scenic route for tourists!
- Kabataş → Bağcılar
- Key stops: Karaköy, Eminönü, Sultanahmet, Beyazıt, Laleli

Runs every 5-10 minutes, 6 AM - midnight

## Ferry System

The most romantic way to travel!

**Popular Routes**:
- Eminönü ↔ Kadıköy (20 min)
- Karaköy ↔ Kadıköy (25 min)
- Kabataş ↔ Üsküdar (15 min)

**Bosphorus Tour**: Full tour from Eminönü to Anadolu Kavağı (1.5 hours each way)

## Buses

- Extensive network covering all areas
- Pay only with Istanbulkart
- Use Moovit app for real-time tracking
- MetroBus: Rapid transit on dedicated lanes

## Pro Tips

1. **Download Apps**: Moovit, Citymapper, or IBB Mobile
2. **Avoid Rush Hours**: 7:30-9:30 AM and 5:30-7:30 PM
3. **Keep Card Handy**: You'll use it frequently
4. **Try Ferries**: Best views and most relaxing option
5. **Ask Locals**: They're usually happy to help with directions

With these tips, you'll navigate Istanbul like a pro!
        """,
        "excerpt": "Complete guide to Istanbul's public transportation: metro, tram, ferry, and bus. Get your Istanbulkart and explore the city!",
        "author": "Istanbul Navigator",
        "status": "published",
        "featured_image": "https://aistanbul-info.s3.eu-central-1.amazonaws.com/blog/istanbul-transport.jpg",
        "category": "Travel Tips",
        "district": "Istanbul (general)",
        "tags": ["transportation", "metro", "ferry", "travel tips", "istanbulkart"],
        "views": 312,
        "likes_count": 45,
        "published_at": datetime.utcnow() - timedelta(days=3)
    },
    {
        "title": "A Day in Sultanahmet: Complete Guide",
        "slug": "day-in-sultanahmet-guide",
        "content": """
# Sultanahmet: Istanbul's Historic Heart

Sultanahmet is where Byzantine and Ottoman history come alive. Here's how to spend a perfect day in this UNESCO World Heritage site.

## Morning (9:00 AM - 12:00 PM)

### Hagia Sophia (9:00 AM)
Start early to beat the crowds at this architectural marvel.
- **History**: Built in 537 AD, served as a church, mosque, and now a mosque again
- **Don't Miss**: Byzantine mosaics, enormous dome, marble columns
- **Tip**: Dress modestly (scarves provided for women)

### Blue Mosque (10:30 AM)
Just across the square, visit the Sultan Ahmed Mosque.
- **Famous For**: Six minarets and 20,000 blue Iznik tiles
- **Entrance**: Free (donations welcome)
- **Prayer Times**: Closed during prayer times (5 times daily)

## Midday (12:00 PM - 2:00 PM)

### Lunch at Tarihi Sultanahmet Köftecisi
Try authentic Turkish meatballs (köfte) at this historic restaurant.
- **Must Order**: Köfte with rice pilaf and salad
- **Price**: Very reasonable, ~150 TL per person

### Explore the Hippodrome
Walk through the ancient Byzantine chariot racing arena.
- **Monuments**: Egyptian Obelisk, Serpent Column, German Fountain
- **Great For**: Photos and people watching

## Afternoon (2:00 PM - 6:00 PM)

### Topkapi Palace (2:00 PM)
Explore the opulent residence of Ottoman sultans.
- **Highlights**: Harem, Treasury, Sacred Relics
- **Time Needed**: 2-3 hours minimum
- **Tickets**: Buy online to skip queues

### Basilica Cistern (5:00 PM)
Cool off in this underground water reservoir.
- **Famous For**: Medusa head columns
- **Atmosphere**: Mystical lighting and dripping water
- **Duration**: 30-45 minutes

## Evening (6:00 PM onwards)

### Sunset at Sultanahmet Park
Relax in the park between Hagia Sophia and Blue Mosque as the sun sets.

### Dinner with a View
Head to a rooftop restaurant for views of the illuminated monuments.
- **Recommendations**: Seven Hills Restaurant, Balkon Restaurant
- **Perfect For**: Romantic dinner with incredible views

## Budget Breakdown

- Hagia Sophia: Free (mosque)
- Blue Mosque: Free
- Topkapi Palace: 700 TL
- Basilica Cistern: 450 TL
- Lunch: 150 TL
- Dinner: 500-800 TL
- **Total**: ~1,800-2,100 TL per person

## Pro Tips

1. Buy Museum Pass Istanbul (850 TL for 5 days) if visiting multiple museums
2. Wear comfortable shoes - lots of walking!
3. Bring a scarf and modest clothing
4. Book skip-the-line tickets online
5. Visit on weekdays if possible (less crowded)
6. Download offline maps
7. Stay hydrated - water fountains available

Sultanahmet is Istanbul's crown jewel. Take your time and soak in centuries of history!
        """,
        "excerpt": "Complete day itinerary for Sultanahmet: Hagia Sophia, Blue Mosque, Topkapi Palace, and more. Make the most of Istanbul's historic center.",
        "author": "History Enthusiast",
        "status": "published",
        "featured_image": "https://aistanbul-info.s3.eu-central-1.amazonaws.com/blog/sultanahmet-guide.jpg",
        "category": "Itineraries",
        "district": "Sultanahmet",
        "tags": ["sultanahmet", "hagia sophia", "blue mosque", "topkapi palace", "itinerary"],
        "views": 428,
        "likes_count": 62,
        "published_at": datetime.utcnow() - timedelta(days=2)
    },
    {
        "title": "Beyoğlu Nightlife: Where to Go After Dark",
        "slug": "beyoglu-nightlife-guide",
        "content": """
# Beyoğlu After Dark: Istanbul's Nightlife Hub

When the sun sets, Beyoğlu comes alive with music, lights, and energy. Here's your guide to the best nightlife in Istanbul.

## Istiklal Avenue: The Main Artery

### Early Evening (7:00 PM - 10:00 PM)

**Dinner Options**:
- **Mikla**: Rooftop fine dining with Bosphorus views
- **360 Istanbul**: Panoramic views and international cuisine
- **Nevizade Street**: Traditional meyhanes (taverns) with live music

## Late Night Venues (10:00 PM - 4:00 AM)

### For Live Music

**Jazz**:
- **Nardis Jazz Club**: Intimate venue, world-class performers
- **Babylon**: Multi-genre, international and local acts

**Turkish Music**:
- **Munzur Cafe**: Traditional Kurdish and Turkish folk music
- **Eski Beyoğlu**: Classic Turkish songs in vintage setting

### For Dancing

**Electronic/House**:
- **Klein**: Underground techno club
- **Suma Beach**: Beach club atmosphere in the city

**Mixed Genre**:
- **Indigo**: Popular with young crowds, hip-hop and pop
- **Reina**: Upscale Bosphorus-side club (dress code enforced)

## Karaköy & Galata

### Trendy Bars

- **Under**: Speakeasy-style cocktail bar
- **Geyik**: Laid-back with creative drinks
- **Müzedechanga**: Sophisticated cocktails in museum setting

### Rooftop Bars

- **360 Istanbul**: 360-degree city views
- **Leb-i Derya**: Bosphorus sunset views
- **Mikla**: High-end cocktails and cuisine

## Asmalımescit & Nevizade

Traditional meyhane experience:
- **Şark Sofrası**: Traditional food and rakı
- **Cumhuriyet Meyhane**: Live fasıl music
- **Sofyalı 9**: Modern twist on meyhane culture

## Safety & Tips

1. **Getting Around**: Use taxis or Uber late at night (metro closes at midnight)
2. **Dress Code**: Smart casual minimum for upscale venues
3. **Budget**: Expect 500-1,500 TL per person per venue
4. **Reservations**: Book ahead for popular spots
5. **Cover Charges**: Some clubs charge 200-500 TL entrance
6. **Drinks**: Cocktails 150-400 TL, beer 80-150 TL
7. **Stay Safe**: Keep valuables secure, don't accept drinks from strangers

## Night Buses

If you miss the metro:
- Night buses run from Taksim
- Download IBB Mobile app for routes
- Have Istanbulkart loaded

## Weekly Schedule

- **Monday-Tuesday**: Quieter, best for intimate bars
- **Wednesday-Thursday**: Mid-week buzz, better prices
- **Friday-Saturday**: Full energy, expect crowds and queues
- **Sunday**: Chilled rooftop bars and jazz clubs

Beyoğlu offers something for every night owl. Start with dinner, follow with cocktails, and end with dancing until dawn!
        """,
        "excerpt": "Complete guide to Beyoğlu nightlife: best bars, clubs, live music venues, and rooftop spots. Experience Istanbul after dark.",
        "author": "Night Owl",
        "status": "published",
        "featured_image": "https://aistanbul-info.s3.eu-central-1.amazonaws.com/blog/beyoglu-nightlife.jpg",
        "category": "Nightlife",
        "district": "Beyoğlu",
        "tags": ["nightlife", "bars", "clubs", "beyoglu", "live music", "rooftop bars"],
        "views": 356,
        "likes_count": 48,
        "published_at": datetime.utcnow() - timedelta(days=1)
    }
]


def get_gcp_database_url():
    """Get GCP Cloud SQL database URL"""
    # Try to read from .env file
    env_path = os.path.join(os.path.dirname(__file__), 'backend', '.env')
    
    print("\n" + "="*80)
    print("📊 GCP Cloud SQL Database Connection Setup")
    print("="*80)
    
    # GCP Cloud SQL connection details from .env
    gcp_host = "34.38.193.1"
    gcp_password = "*iwP#MDmX5dn8V:1LExE|70:O>|i"
    
    print("\n⚠️  IMPORTANT: GCP Cloud SQL Firewall Configuration")
    print("\nYour GCP Cloud SQL instance needs to allow connections from your IP.")
    print("\nTo configure:")
    print("1. Go to: https://console.cloud.google.com/sql/instances")
    print("2. Select instance: dfsadasdsadsa23123")
    print("3. Click 'Connections' → 'Networking'")
    print("4. Add your current IP to 'Authorized networks'")
    print("5. Save changes\n")
    
    choice = input("Have you whitelisted your IP in GCP? (y/n) [default: n]: ").strip().lower()
    
    if choice != 'y':
        print("\n❌ Please whitelist your IP first, then run this script again.")
        print("\nAlternatively, you can:")
        print("1. Run this script from GCP Cloud Shell")
        print("2. Run this script from a GCP Compute Engine instance")
        print("3. Use the Cloud SQL Proxy\n")
        return None
    
    # URL encode the password
    from urllib.parse import quote_plus
    encoded_password = quote_plus(gcp_password)
    
    database_url = f"postgresql://postgres:{encoded_password}@{gcp_host}:5432/postgres"
    
    print(f"\n✅ Using GCP Cloud SQL: {gcp_host}")
    return database_url


def populate_blog_posts(database_url: str):
    """Populate blog posts in the database"""
    try:
        # Create engine
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        print("\n🔗 Connecting to database...")
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ Connected to PostgreSQL: {version[:50]}...")
        
        # Check if blog_posts table exists
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'blog_posts'
                );
            """))
            table_exists = result.fetchone()[0]
        
        if not table_exists:
            print("\n❌ Table 'blog_posts' does not exist!")
            print("Please run database migrations first:")
            print("  cd backend && alembic upgrade head")
            return False
        
        # Check current posts
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM blog_posts;"))
            current_count = result.fetchone()[0]
            print(f"\n📊 Current blog posts in database: {current_count}")
        
        # Ask for confirmation
        if current_count > 0:
            print(f"\n⚠️  There are already {current_count} blog posts in the database.")
            choice = input("Do you want to add more posts? (y/n) [default: y]: ").strip().lower() or 'y'
            if choice != 'y':
                print("❌ Operation cancelled.")
                return False
        
        # Insert blog posts
        print(f"\n📝 Inserting {len(SAMPLE_BLOG_POSTS)} blog posts...")
        
        for i, post in enumerate(SAMPLE_BLOG_POSTS, 1):
            try:
                # Convert tags list to JSON string
                import json
                tags_json = json.dumps(post['tags'])
                
                with engine.connect() as conn:
                    # Check if post with same slug already exists
                    result = conn.execute(
                        text("SELECT id FROM blog_posts WHERE slug = :slug"),
                        {"slug": post['slug']}
                    )
                    existing = result.fetchone()
                    
                    if existing:
                        print(f"  ⏭️  Skipping '{post['title']}' (already exists)")
                        continue
                    
                    # Insert the post
                    conn.execute(text("""
                        INSERT INTO blog_posts (
                            title, slug, content, excerpt, author, status,
                            featured_image, category, district, tags,
                            views, likes_count, created_at, updated_at, published_at
                        ) VALUES (
                            :title, :slug, :content, :excerpt, :author, :status,
                            :featured_image, :category, :district, :tags,
                            :views, :likes_count, :created_at, :updated_at, :published_at
                        )
                    """), {
                        "title": post['title'],
                        "slug": post['slug'],
                        "content": post['content'],
                        "excerpt": post['excerpt'],
                        "author": post['author'],
                        "status": post['status'],
                        "featured_image": post['featured_image'],
                        "category": post['category'],
                        "district": post['district'],
                        "tags": tags_json,
                        "views": post['views'],
                        "likes_count": post['likes_count'],
                        "created_at": post['published_at'],
                        "updated_at": post['published_at'],
                        "published_at": post['published_at']
                    })
                    conn.commit()
                    
                    print(f"  ✅ {i}. {post['title']}")
            
            except Exception as e:
                print(f"  ❌ Failed to insert '{post['title']}': {e}")
                continue
        
        # Show final count
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM blog_posts;"))
            final_count = result.fetchone()[0]
            print(f"\n✅ Total blog posts in database: {final_count}")
        
        print("\n" + "="*80)
        print("🎉 Blog posts populated successfully!")
        print("="*80)
        print("\n💡 Next steps:")
        print("  1. Upload actual images to S3 bucket: aistanbul-info")
        print("  2. Update featured_image URLs if needed")
        print("  3. Test the blog page: http://localhost:3000/blog")
        print("  4. Verify posts appear correctly\n")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    print("\n" + "="*80)
    print("🗃️  Blog Posts Database Populator for GCP Cloud SQL")
    print("="*80)
    
    # Get database URL
    database_url = get_gcp_database_url()
    
    if not database_url:
        sys.exit(1)
    
    # Populate blog posts
    success = populate_blog_posts(database_url)
    
    if success:
        print("\n✅ Success! Your blog should now display posts.")
    else:
        print("\n❌ Failed to populate blog posts.")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Seed Blog Posts into Database
Creates sample blog posts for the AI Istanbul blog
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from datetime import datetime
from database import SessionLocal
from models import BlogPost

def create_sample_posts():
    """Create sample blog posts"""
    db = SessionLocal()
    
    try:
        # Check if posts already exist
        existing_count = db.query(BlogPost).count()
        if existing_count > 0:
            print(f"✅ Database already has {existing_count} blog posts")
            response = input("Do you want to add more sample posts? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        sample_posts = [
            {
                "title": "Ultimate Guide to Istanbul's Hidden Gems",
                "content": """# Discover Istanbul's Secret Treasures

Istanbul is a city of endless discoveries, where every corner holds a story waiting to be unveiled. Beyond the famous landmarks lies a world of hidden gems that only locals know about.

## 🔮 Secret Neighborhoods

### Fener & Balat
These colorful Byzantine neighborhoods offer:
- Historic Greek Houses painted in vibrant colors
- Antique Shops for treasure hunting
- Local Cafes with authentic Turkish coffee culture
- Photography Paradise with Instagram-worthy streets

### Kuzguncuk
A peaceful village on the Asian side:
- Multi-cultural Heritage with churches, synagogues, and mosques
- Wooden Houses with Ottoman-era architecture
- Artisan Workshops with local craftspeople
- Quiet Cafes perfect for reading and relaxation

## 🍽️ Hidden Food Spots

Discover the best local eateries that tourists rarely find!

## 🎨 Cultural Discoveries

Explore underground cisterns, art galleries, and more.""",
                "author": "Istanbul Local Guide"
            },
            {
                "title": "Best Restaurants in Beyoğlu",
                "content": """# Beyoğlu Culinary Adventure

Beyoğlu is the heart of Istanbul's modern food scene, where traditional Turkish cuisine meets contemporary innovation.

## Top Picks

### Mikla
- Rooftop dining with Bosphorus views
- Modern Anatolian cuisine
- Celebrity chef Mehmet Gürs
- Reservations essential

### Zübeyir Ocakbaşı
- Traditional grill house
- Authentic kebabs
- Local atmosphere
- Affordable prices

### Karaköy Lokantası
- Ottoman-era recipes
- Historic building
- Daily changing menu
- Perfect for lunch

## Street Food

Don't miss the street vendors selling:
- Midye Dolma (stuffed mussels)
- Simit (sesame bread rings)
- Döner kebab
- Fresh corn on the cob

Visit Beyoğlu hungry and leave happy!""",
                "author": "Food Critic Istanbul"
            },
            {
                "title": "A Perfect Day in Sultanahmet",
                "content": """# Sultanahmet: The Historic Heart of Istanbul

Sultanahmet is where East meets West, where Byzantine meets Ottoman, and where ancient history comes alive.

## Morning: Museums & Marvels

### 9:00 AM - Hagia Sophia
Start your day at this architectural masterpiece. Arrive early to beat the crowds.

**Tips:**
- Buy tickets online
- Spend 60-90 minutes
- Don't miss the upper gallery

### 11:00 AM - Blue Mosque
Just a 5-minute walk away, the Blue Mosque is equally stunning.

**Remember:**
- Remove shoes
- Dress modestly
- Free entry
- Closed during prayer times

## Afternoon: Underground Wonders

### 1:00 PM - Lunch Break
Try traditional Turkish lunch at:
- Sultanahmet Köftecisi
- Erol Lokantası
- Hamdi Restaurant

### 2:30 PM - Basilica Cistern
Cool off in this underground Byzantine water reservoir.

## Evening: Sunset Views

### 5:00 PM - Topkapı Palace
Explore the Ottoman sultans' residence.

### 7:00 PM - Sunset at Sultanahmet Square
Watch the sunset paint the historic buildings in golden light.

## Dinner

End your day with dinner overlooking the Bosphorus.""",
                "author": "Istanbul Travel Expert"
            },
            {
                "title": "Navigating Istanbul's Public Transport",
                "content": """# Master Istanbul's Transport System

Istanbul's transport system is extensive, efficient, and surprisingly easy to use once you know the basics.

## Istanbulkart: Your Magic Card

The Istanbulkart is essential for all public transport:

**Where to Buy:**
- Metro stations
- Ferry terminals
- Kiosks near transport hubs

**Cost:**
- Card: 50 TL (refundable deposit)
- Top-up: Any amount
- Discounts: Transfers are cheaper

## Metro Lines

### M1: Airport to City
- Connects Atatürk Airport
- Runs to Yenikapı
- Transfers to other lines

### M2: Golden Line
- Yenikapı to Hacıosman
- Stops at Taksim
- Most useful for tourists

### Marmaray: Under the Bosphorus
- Connects Europe and Asia
- Underground railway
- Fast and modern

## Ferries

The most scenic way to travel:

**Routes:**
- Kadıköy ↔ Eminönü
- Üsküdar ↔ Beşiktaş
- Bosphorus cruises

**Tips:**
- Sit outside for views
- Buy simit from vendors
- Watch for dolphins

## Trams

### T1: The Tourist Tram
- Kabataş to Bağcılar
- Stops at major sites
- Can get crowded

## Buses

- Extensive network
- Same Istanbulkart
- Use Google Maps for routes

## Taxis

**Important:**
- Use BiTaksi or Uber
- Insist on meter
- Have address in Turkish
- Typical fare: 50-150 TL

## Pro Tips

1. Download Moovit app
2. Buy Istanbulkart immediately
3. Avoid rush hours (8-10 AM, 5-7 PM)
4. Keep small change for tips
5. Learn basic Turkish phrases

Happy traveling!""",
                "author": "Istanbul Commuter"
            },
            {
                "title": "Best Time to Visit Istanbul",
                "content": """# When to Visit Istanbul: A Seasonal Guide

Istanbul is a year-round destination, but each season offers something unique.

## Spring (March-May) ⭐ BEST

### Why Visit:
- Perfect weather (15-25°C)
- Tulip Festival in April
- Fewer crowds than summer
- Blooming gardens

### Events:
- Istanbul Tulip Festival
- Music festivals
- Film festivals

## Summer (June-August) ☀️

### Pros:
- Long daylight hours
- Outdoor dining culture
- Beach escapes nearby
- Vibrant nightlife

### Cons:
- Very hot (30-40°C)
- Crowded tourist sites
- Higher prices
- Humidity

**Tips for Summer:**
- Book hotels with AC
- Visit museums midday
- Explore early morning
- Dress light

## Autumn (September-November) ⭐ BEST

### Why Visit:
- Pleasant temperatures
- Fall colors
- Art biennale (odd years)
- Best food season

### Perfect For:
- Walking tours
- Photography
- Food festivals
- Cultural events

## Winter (December-February) ❄️

### Pros:
- Lowest prices
- No crowds
- Cozy cafes
- Snow-capped mosques

### Cons:
- Cold and rainy
- Short days
- Some sites close early

**Winter Magic:**
- Hot Turkish tea
- Roasted chestnuts
- Indoor museums
- Hamam experience

## Special Considerations

### Ramadan
- Varies by year
- Evening festivities
- Some restaurants closed during day
- Beautiful night markets

### National Holidays
- Book accommodation early
- Expect closures
- Plan around major holidays

## Budget Considerations

**High Season:** May-September
- Prices up 50-100%
- Book 3 months ahead

**Shoulder Season:** March-April, October-November
- Best value
- Good weather
- Fewer crowds

**Low Season:** December-February
- Best deals
- Up to 50% off
- Flexible booking

## My Recommendation

**Best Overall:** April or October
- Perfect weather
- Manageable crowds
- Good prices
- All attractions open

**Best Budget:** January or February
- Rock-bottom prices
- Authentic experience
- Cozy atmosphere

Plan your visit according to your priorities!""",
                "author": "Istanbul Weather Expert"
            },
            {
                "title": "Kadıköy: Asian Side's Gem",
                "content": """# Discover Kadıköy: Istanbul's Cool Neighborhood

Kadıköy represents the authentic, artsy, and alternative side of Istanbul. It's where locals live, work, and play.

## Why Visit Kadıköy?

### Authentic Istanbul
- Real neighborhood feel
- Fewer tourists
- Local prices
- Genuine experiences

### Food Paradise
- Çiya Sofrası: Regional Anatolian cuisine
- Baylan: Historic patisserie since 1923
- Fish market: Fresh seafood
- Street food: Everything imaginable

## What to See & Do

### Moda
Waterfront neighborhood perfect for:
- Walking promenade
- Sunset watching
- Cute cafes
- Parks and green spaces

### Barlar Sokağı (Bar Street)
Istanbul's nightlife hub:
- Live music venues
- Craft beer bars
- Meyhanes (taverns)
- Rooftop bars

### Kadıköy Market
Traditional bazaar with:
- Fresh produce
- Spices and herbs
- Local cheese
- Turkish delights
- Antiques

## Street Art

Kadıköy is famous for murals:
- Yeldegirmeni neighborhood
- International artists
- Urban art festivals
- Instagram paradise

## Shopping

### Moda Caddesi
Boutique shopping:
- Independent designers
- Vintage stores
- Bookshops
- Artisan crafts

## Where to Eat

### Must-Try Restaurants
1. **Çiya Sofrası**: Don't miss this! Historic recipes from all over Turkey
2. **Ali Usta**: Best ice cream in Istanbul
3. **Kadıköy Balık Pazar**: Fresh seafood experience
4. **Kızılkayalar**: Traditional gözleme
5. **Baylan**: Historic pastry shop

## Coffee Culture

Kadıköy has Istanbul's best coffee:
- Petra Roasting Co.
- Coffee Manifesto
- Kronotrop
- Fazıl Bey'in Türk Kahvesi

## How to Get There

### From European Side:
- Ferry from Eminönü (25 min)
- Ferry from Karaköy (25 min)
- Marmaray train

### Getting Around Kadıköy:
- Walk everywhere
- Use trams for longer distances
- Bike rentals available

## Pro Tips

1. **Visit on weekends**: Best atmosphere
2. **Start early**: Markets open early
3. **Take the ferry**: Best way to arrive
4. **Explore side streets**: Hidden gems everywhere
5. **Try street food**: It's amazing and safe

## Sample Itinerary

### Morning (10 AM - 1 PM):
- Arrive by ferry
- Breakfast at a local cafe
- Explore the market
- Visit Moda

### Afternoon (1 PM - 6 PM):
- Lunch at Çiya
- Coffee at Kronotrop
- Shopping on Moda Caddesi
- Walk along the waterfront

### Evening (6 PM - late):
- Sunset at Moda coast
- Dinner at a meyhane
- Drinks on Barlar Sokağı
- Live music

Kadıköy is where Istanbul's soul lives. Don't miss it!""",
                "author": "Kadıköy Resident"
            }
        ]
        
        created_count = 0
        for post_data in sample_posts:
            post = BlogPost(
                title=post_data["title"],
                content=post_data["content"],
                author=post_data["author"],
                likes_count=0,
                created_at=datetime.utcnow()
            )
            db.add(post)
            created_count += 1
            print(f"✅ Created: {post_data['title']}")
        
        db.commit()
        print(f"\n🎉 Successfully created {created_count} blog posts!")
        print(f"📊 Total posts in database: {db.query(BlogPost).count()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("🌱 Seeding blog posts...")
    create_sample_posts()

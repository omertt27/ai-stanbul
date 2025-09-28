#!/usr/bin/env python3
"""
Script to load the 12 mock blog posts into the database
"""
import json
from datetime import datetime
from database import get_db
from models import BlogPost

# Mock blog posts from the frontend
mock_posts = [
  {
    "id": 1,
    "title": "Hidden Gems in Sultanahmet: Beyond the Tourist Trail",
    "content": "Discover the secret courtyards, ancient cisterns, and local eateries that most visitors miss in Istanbul's historic heart. From the peaceful Soğukçeşme Sokağı to the underground wonders of Şerefiye Cistern, explore Istanbul's hidden treasures that showcase the authentic local culture away from crowded tourist spots.",
    "author": "Mehmet Yılmaz",
    "district": "Sultanahmet",
    "created_at": "2024-12-01T10:00:00Z",
    "likes_count": 47
  },
  {
    "id": 2,
    "title": "Best Rooftop Views for Sunset in Galata",
    "content": "Experience Istanbul's magic hour from the best rooftop terraces in Galata. From trendy bars to quiet cafes, here are the spots where locals go to watch the sun set over the Golden Horn. Discover hidden rooftops with panoramic views of the city's skyline and historic landmarks.",
    "author": "Ayşe Demir",
    "district": "Galata",
    "created_at": "2024-11-28T15:30:00Z",
    "likes_count": 73
  },
  {
    "id": 3,
    "title": "Street Food Paradise: Kadıköy's Culinary Adventures",
    "content": "Dive into the vibrant food scene of Kadıköy, where traditional Turkish flavors meet modern creativity. From the famous fish sandwich vendors to hidden meyhanes serving authentic mezze, explore the Asian side's culinary gems that locals have cherished for generations.",
    "author": "Can Özkan",
    "district": "Kadıköy",
    "created_at": "2024-11-25T12:15:00Z",
    "likes_count": 92
  },
  {
    "id": 4,
    "title": "Shopping Like a Local: Beyoğlu's Alternative Markets",
    "content": "Skip the tourist shops and discover where Istanbulites really shop. From vintage treasures in Çukurcuma to artisan crafts in the backstreets of Galata, here's your insider guide to authentic shopping experiences that support local businesses and craftspeople.",
    "author": "Zeynep Kaya",
    "district": "Beyoğlu",
    "created_at": "2024-11-20T09:45:00Z",
    "likes_count": 64
  },
  {
    "id": 5,
    "title": "Early Morning Magic: Bosphorus at Dawn",
    "content": "Join the fishermen and early risers for a completely different perspective of Istanbul. The city awakens slowly along the Bosphorus shores, offering peaceful moments and stunning photography opportunities. Experience the tranquil side of this bustling metropolis.",
    "author": "Emre Şahin",
    "district": "Beşiktaş",
    "created_at": "2024-11-18T06:00:00Z",
    "likes_count": 38
  },
  {
    "id": 6,
    "title": "Traditional Hammam Experience: A First-Timer's Guide",
    "content": "Nervous about trying a Turkish bath? This comprehensive guide covers everything from what to expect to proper etiquette, helping you enjoy this centuries-old Istanbul tradition with confidence. Learn about the rituals, benefits, and best hammams to visit.",
    "author": "Fatma Arslan",
    "district": "Fatih",
    "created_at": "2024-11-15T14:20:00Z",
    "likes_count": 156
  },
  {
    "id": 7,
    "title": "Art & Culture in Karaköy: Where Creativity Meets History",
    "content": "Explore the vibrant art scene in Karaköy district, where contemporary galleries blend with historic Ottoman architecture. From the Istanbul Modern to hidden artist studios, discover how this former merchant quarter has transformed into the city's creative hub.",
    "author": "Selin Yücel",
    "district": "Karaköy",
    "created_at": "2024-11-12T16:45:00Z",
    "likes_count": 82
  },
  {
    "id": 8,
    "title": "Ferry Adventures: Island Hopping from Eminönü",
    "content": "Take the scenic route to Büyükada and Heybeliada. This guide covers the best ferry schedules, what to see on each island, and local seafood restaurants you shouldn't miss. Experience the peaceful escape from city life just a ferry ride away.",
    "author": "Burak Şen",
    "district": "Eminönü",
    "created_at": "2024-11-10T11:30:00Z",
    "likes_count": 95
  },
  {
    "id": 9,
    "title": "Night Markets and Street Life in Şişli",
    "content": "When the sun goes down, Şişli comes alive with bustling night markets, late-night eateries, and vibrant street culture. Here's your guide to experiencing Istanbul after dark, from midnight snacks to 24-hour entertainment venues.",
    "author": "Deniz Aktaş",
    "district": "Şişli",
    "created_at": "2024-11-08T20:15:00Z",
    "likes_count": 67
  },
  {
    "id": 10,
    "title": "Historic Churches and Mosques: A Spiritual Journey",
    "content": "Discover the religious heritage of Istanbul through its magnificent churches and mosques. From Hagia Sophia to Chora Church, explore the spiritual heart of the city and learn about the diverse faiths that have shaped Istanbul's cultural landscape.",
    "author": "Prof. Ahmet Güler",
    "district": "Fatih",
    "created_at": "2024-11-05T13:00:00Z",
    "likes_count": 134
  },
  {
    "id": 11,
    "title": "Coffee Culture: From Traditional to Third Wave",
    "content": "Journey through Istanbul's evolving coffee scene, from traditional Turkish coffee ceremonies to modern specialty coffee shops. Discover the best cafes in every district and learn how coffee culture bridges the gap between tradition and innovation.",
    "author": "Elif Özdemir",
    "district": "Beyoğlu",
    "created_at": "2024-11-02T08:30:00Z",
    "likes_count": 113
  },
  {
    "id": 12,
    "title": "Weekend Escape: Üsküdar's Asian Side Charm",
    "content": "Cross the Bosphorus to discover Üsküdar's peaceful atmosphere, historic sites, and stunning views of the European side. Perfect for a relaxing weekend exploration away from the tourist crowds, featuring traditional neighborhoods and waterfront cafes.",
    "author": "Murat Kaya",
    "district": "Üsküdar",
    "created_at": "2024-10-30T15:45:00Z",
    "likes_count": 78
  }
]

def load_mock_posts():
    """Load mock blog posts into the database"""
    print("🔧 Loading mock blog posts into database...")
    
    db = next(get_db())
    
    # First, clear existing posts to avoid conflicts
    existing_posts = db.query(BlogPost).all()
    for post in existing_posts:
        db.delete(post)
    db.commit()
    print(f"✅ Cleared {len(existing_posts)} existing posts")
    
    # Load mock posts
    posts_added = 0
    for post_data in mock_posts:
        # Convert ISO date string to datetime object
        created_at = datetime.fromisoformat(post_data["created_at"].replace('Z', '+00:00'))
        
        blog_post = BlogPost(
            title=post_data["title"],
            content=post_data["content"],
            author=post_data["author"],
            district=post_data["district"],
            created_at=created_at,
            likes_count=post_data["likes_count"]
        )
        
        db.add(blog_post)
        posts_added += 1
        print(f"➕ Added: {post_data['title']}")
    
    db.commit()
    print(f"✅ Successfully loaded {posts_added} blog posts into database")
    
    # Verify the posts were added
    total_posts = db.query(BlogPost).count()
    print(f"📊 Total posts in database: {total_posts}")

if __name__ == "__main__":
    load_mock_posts()

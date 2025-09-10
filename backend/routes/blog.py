#!/usr/bin/env python3
"""
Blog Routes for AIstanbul
Handles blog posts, travel guides, and content management
"""

import os
import json
import uuid
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import logging

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/blog", tags=["blog"])

# Blog post data model
class BlogPost(BaseModel):
    id: str
    title: str
    content: str
    author: str
    category: str
    tags: List[str]
    featured_image: Optional[str] = None
    published: bool = True
    created_at: datetime
    updated_at: datetime
    views: int = 0
    likes: int = 0
    
    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()
    
    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()

class BlogPostCreate(BaseModel):
    title: str
    content: str
    author: str
    category: str
    tags: List[str] = []
    featured_image: Optional[str] = None
    published: bool = True

class BlogPostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    featured_image: Optional[str] = None
    published: Optional[bool] = None

# Blog storage (in production, this would be a database)
BLOG_DATA_FILE = "blog_posts.json"
UPLOAD_DIRECTORY = "uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

def load_blog_posts() -> List[Dict[str, Any]]:
    """Load blog posts from JSON file"""
    try:
        if os.path.exists(BLOG_DATA_FILE):
            with open(BLOG_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading blog posts: {e}")
        return []

def save_blog_posts(posts: List[Dict[str, Any]]) -> bool:
    """Save blog posts to JSON file"""
    try:
        with open(BLOG_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        logger.error(f"Error saving blog posts: {e}")
        return False

def get_default_blog_posts() -> List[Dict[str, Any]]:
    """Create default blog posts if none exist"""
    return [
        {
            "id": "1",
            "title": "Ultimate Guide to Istanbul's Hidden Gems",
            "content": """# Discover Istanbul's Secret Treasures

Istanbul is a city of endless discoveries, where every corner holds a story waiting to be unveiled. Beyond the famous landmarks lies a world of hidden gems that only locals know about.

## 🔮 Secret Neighborhoods

### Fener & Balat
These colorful Byzantine neighborhoods offer:
- **Historic Greek Houses**: Painted in vibrant colors
- **Antique Shops**: Treasure hunting opportunities
- **Local Cafes**: Authentic Turkish coffee culture
- **Photography Paradise**: Instagram-worthy streets

### Kuzguncuk
A peaceful village on the Asian side:
- **Multi-cultural Heritage**: Churches, synagogues, and mosques side by side
- **Wooden Houses**: Ottoman-era architecture
- **Artisan Workshops**: Local craftspeople at work
- **Quiet Cafes**: Perfect for reading and relaxation

## 🍽️ Hidden Food Spots

### Local Lokantas
- **Develi Restaurant** (Samatya): Famous lamb dishes
- **Pandeli** (Eminönü): Historic Ottoman cuisine
- **Kiva Han** (Beyoğlu): Traditional meze selection

### Street Food Secrets
- **Balık Ekmek** at Galata Bridge: Fresh fish sandwiches
- **Dönerci Engin** (Aksaray): Best döner in the city
- **Kokorec at Ortaköy**: Late-night specialty

## 🎨 Cultural Discoveries

### Underground Cisterns
- **Basilica Cistern**: Famous but magical
- **Binbirdirek Cistern**: Less crowded alternative
- **Theodosius Cistern**: Hidden underground palace

### Art Galleries
- **Istanbul Modern**: Contemporary Turkish art
- **Pera Museum**: Orientalist paintings
- **Salt Galata**: Innovative cultural center

## 🚶‍♂️ Walking Routes

### Golden Horn Walk
1. Start at Galata Bridge
2. Walk along the waterfront
3. Discover hidden stairs and passages
4. End at Pierre Loti Hill

### Bosphorus Village Tour
1. Ortaköy → Arnavutköy → Bebek
2. Explore waterfront mansions
3. Stop at local fish restaurants
4. Take ferry back

## 💡 Insider Tips

### Best Times to Visit
- **Early Morning**: Peaceful neighborhood exploration
- **Late Afternoon**: Perfect light for photography
- **Evening**: Local life comes alive

### Transportation Secrets
- **Vapur (Ferry)**: Most scenic way to travel
- **Dolmuş**: Shared taxis for local experience
- **Walking**: Best way to discover hidden spots

### Cultural Etiquette
- **Greet Shopkeepers**: A simple "Merhaba" goes far
- **Accept Tea**: Turkish hospitality tradition
- **Dress Modestly**: Especially in religious areas
- **Learn Basic Turkish**: Locals appreciate effort

Ready to explore Istanbul like a local? These hidden gems will give you authentic experiences that most tourists never discover!""",
            "author": "Istanbul Explorer",
            "category": "Travel Guide",
            "tags": ["hidden gems", "local culture", "neighborhoods", "insider tips"],
            "featured_image": None,
            "published": True,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "views": 156,
            "likes": 23
        },
        {
            "id": "2", 
            "title": "Best Turkish Street Food You Must Try",
            "content": """# Turkish Street Food Adventure

Turkish street food is a culinary journey that will tantalize your taste buds and introduce you to the heart of Turkish culture. Here's your guide to the must-try street foods in Istanbul.

## 🥙 Essential Street Foods

### Döner Kebab
The king of Turkish street food:
- **Best Places**: Öz Konyalı, Karadeniz Döner
- **What to Order**: Döner in bread or lavash
- **Pro Tip**: Ask for extra salad and sauce

### Balık Ekmek (Fish Sandwich)
Istanbul's iconic sandwich:
- **Where**: Eminönü waterfront, Galata Bridge
- **Fish**: Usually mackerel, grilled fresh
- **Sides**: Pickled vegetables, onions

### Simit
Turkish bagel, perfect for breakfast:
- **Toppings**: Plain, sesame, or with cheese
- **Best With**: Turkish tea
- **Where**: Street vendors everywhere

## 🍢 Grilled Specialties

### Kokoreç
Adventurous choice - grilled lamb intestines:
- **Flavor**: Rich and spiced
- **Best At**: Ortaköy, late-night spots
- **Served**: In bread with spices

### Midye Dolma (Stuffed Mussels)
Popular seaside snack:
- **Filling**: Rice, pine nuts, spices
- **Eaten**: With lemon juice
- **Caution**: Only from busy vendors

### İskender Kebab
Not exactly street food, but a must:
- **Components**: Döner, yogurt, tomato sauce, bread
- **Origin**: Bursa (invented here!)
- **Best**: Kebabçı İskender (original)

## 🧆 Vegetarian Options

### Stuffed Grape Leaves (Dolma)
- **Filling**: Rice, herbs, pine nuts
- **Served**: Cold with lemon
- **Perfect**: Light lunch option

### Gözleme
Thin flatbread with various fillings:
- **Popular**: Cheese, spinach, potato
- **Made**: Fresh on griddle
- **Best**: At Çanakkale restaurants

### Börek
Flaky pastry layers:
- **Types**: Su böreği, sigara böreği
- **Fillings**: Cheese, spinach, meat
- **Where**: Bakeries and street stalls

## 🍰 Sweet Street Treats

### Turkish Delight (Lokum)
- **Flavors**: Rose, lemon, pomegranate
- **Quality**: Avoid tourist traps
- **Best**: Hacı Bekir, Koska

### Baklava
Layered phyllo with nuts:
- **Types**: Pistachios, walnuts
- **Best**: Güllüoğlu, Karaköy Güllüoğlu
- **Served**: With Turkish tea

### Tavuk Göğsü
Sweet pudding made with chicken:
- **Texture**: Creamy and unique
- **Flavor**: Subtle and sweet
- **Try**: At traditional milk shops

## 🥤 Street Drinks

### Turkish Tea (Çay)
- **Served**: In small glasses
- **Sugar**: On the side
- **Culture**: Social bonding drink

### Turkish Coffee
- **Preparation**: Slow-cooked in sand
- **Served**: With Turkish delight
- **UNESCO**: Intangible cultural heritage

### Şalgam Suyu
Fermented turnip juice:
- **Taste**: Salty and tangy
- **Popular**: In Adana region
- **Acquired**: Taste for most foreigners

## 📍 Best Street Food Areas

### Eminönü
- **Specialty**: Balık ekmek, döner
- **Atmosphere**: Busy, authentic
- **Best Time**: Lunch hours

### Ortaköy
- **Famous**: Kumpir (stuffed potatoes)
- **View**: Bosphorus waterfront
- **Evening**: Perfect for sunset

### Beyoğlu/Galata
- **Variety**: International and Turkish
- **Scene**: Young and vibrant
- **Late Night**: Many options open

### Kadıköy
- **Local**: Less touristy
- **Quality**: High, local standards
- **Exploration**: Great for wandering

## 💰 Budget Tips

### Prices
- **Simit**: 3-5 TL
- **Döner**: 15-25 TL
- **Balık Ekmek**: 10-15 TL
- **Turkish Tea**: 2-3 TL

### Money-Saving Tips
- **Lunch Hours**: Better portions
- **Local Areas**: Better prices
- **Cash**: Often preferred
- **Share**: Many portions are large

Turkish street food is not just about eating—it's about experiencing Turkish culture, hospitality, and the joy of simple, delicious food shared with friends!""",
            "author": "Food Explorer",
            "category": "Food & Cuisine",
            "tags": ["street food", "turkish cuisine", "local food", "budget eating"],
            "featured_image": None,
            "published": True,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "views": 89,
            "likes": 15
        },
        {
            "id": "3",
            "title": "Navigating Istanbul's Transport System Like a Pro",
            "content": """# Master Istanbul's Transportation

Istanbul's transport system can seem overwhelming at first, but with the right knowledge, you'll be navigating the city like a true Istanbullu. Here's your complete guide.

## 🎫 Istanbul Card (Istanbulkart)

### The Essential Card
Your key to all public transport:
- **Where to Buy**: Metro stations, kiosks, post offices
- **Cost**: 13 TL card + credit you add
- **Works On**: Metro, bus, tram, ferry, Marmaray
- **Discounts**: Transfers within 2 hours

### Loading Your Card
- **Machines**: Available at all stations
- **Languages**: Turkish and English
- **Payment**: Cash or card
- **Tip**: Load enough for several days

## 🚇 Metro System

### Main Lines
**M1 Red Line**: Airport to city center
- **Route**: Airport → Zeytinburnu → Aksaray
- **Useful**: Airport connection
- **Travel Time**: 45 minutes to center

**M2 Green Line**: European side north-south
- **Key Stops**: Şişhane (Galata), Vezneciler (Grand Bazaar), Taksim
- **Most Used**: By tourists and locals
- **Frequency**: 2-4 minutes peak hours

**M4 Pink Line**: Asian side
- **Route**: Kadıköy to Tavşantepe
- **Connects**: Asian side neighborhoods
- **Useful**: Exploring Asian Istanbul

### Metro Tips
- **Rush Hours**: 7:30-9:30 AM, 5:30-7:30 PM
- **Direction**: Check end station names
- **Platforms**: Separate for each direction
- **Accessibility**: Most stations wheelchair accessible

## 🚋 Tram Lines

### T1 Historic Tram
**Golden Route**: Kabataş → Zeytinburnu
- **Tourist Stops**: Karaköy, Eminönü, Sultanahmet, Beyazıt
- **Frequency**: 5-10 minutes
- **Scenic**: Travels through historic peninsula

### T4 Topkapı Line
- **Route**: Topkapı → Mescid-i Selam
- **Connects**: Historic areas
- **Less Crowded**: Good alternative

### Nostalgic Tram
- **Route**: Short Taksim-Tünel line
- **Purpose**: Historic preservation
- **Tourist**: Fun experience, limited practical use

## 🚢 Ferry System

### Essential Routes
**European to Asian Side**:
- **Karaköy ↔ Kadıköy**: Most popular
- **Eminönü ↔ Üsküdar**: Historic route
- **Kabataş ↔ Üsküdar**: Quick connection

**Bosphorus Tours**:
- **Short Tour**: Eminönü to Anadolu Kavağı
- **Long Tour**: Full Bosphorus (1.5 hours)
- **Price**: Tourist boats vs public ferries

### Ferry Tips
- **Weather**: Can be cancelled in storms
- **Views**: Best from upper deck
- **Timing**: Check schedules (seasonal)
- **Tea**: Available on board

## 🚌 Bus System

### Types of Buses
**Public Buses (İETT)**:
- **Coverage**: Extensive city network
- **Payment**: Istanbulkart only
- **Route Numbers**: Displayed clearly

**Metrobüs**:
- **Route**: Dedicated rapid transit
- **Useful**: Long-distance travel
- **Crowded**: Peak hours very busy

**Private Buses**:
- **Payment**: Cash to driver
- **Routes**: Supplementary to public
- **Comfort**: Varies widely

### Bus Navigation
- **Apps**: Moovit, Citymapper, IBB
- **Real-time**: Bus tracking available
- **Route Maps**: At bus stops
- **Night Buses**: Limited service

## 🚊 Marmaray

### Underground Rail
**Cross-Continental**:
- **Route**: Connects Europe and Asia underground
- **Stations**: Sirkeci, Üsküdar, Ayrılık Çeşmesi
- **Historic**: Tunnel under Bosphorus
- **Fast**: Direct connection between sides

### Marmaray Tips
- **Frequency**: 10-15 minutes
- **Integration**: Connects with other transport
- **Crowded**: Rush hours
- **Archaeological**: Station displays historical artifacts

## 🚖 Taxis and Ride-sharing

### Traditional Taxis
**Yellow Taxis**:
- **Meter**: Always insist on meter
- **Tips**: Round up fare
- **Rush Hour**: Traffic can be expensive
- **Night**: 50% surcharge after midnight

### App-based Services
**BiTaksi**:
- **Popular**: Local Turkish app
- **Payment**: Cash or card
- **Features**: GPS tracking

**Uber**:
- **Available**: Limited areas
- **Payment**: Card only
- **Convenience**: English interface

## 🚶‍♂️ Walking and Biking

### Pedestrian Areas
**Istiklal Avenue**:
- **Length**: 1.4 km pedestrian street
- **Transport**: Nostalgic tram
- **Shopping**: Main commercial area

**Historic Peninsula**:
- **Walkable**: Between major sites
- **Distances**: 15-20 minutes between attractions
- **Surface**: Some cobblestones

### Bike Sharing
**İSBİKE**:
- **Stations**: Throughout the city
- **Cost**: Hourly rates
- **Areas**: Best in Kadıköy, Beşiktaş

## 📱 Essential Apps

### Navigation
- **Citymapper**: Best overall transport app
- **Google Maps**: Works well for Istanbul
- **Moovit**: Real-time public transport
- **IBB**: Official city transport app

### Helpful Features
- **Real-time**: Live departure times
- **Route Planning**: Multiple transport modes
- **Offline**: Download maps
- **Language**: English support

## 💰 Cost Guide

### Daily Transport Budget
- **Tourist**: 30-50 TL per day
- **Local Style**: 20-30 TL per day
- **Heavy User**: 50-70 TL per day

### Individual Costs
- **Metro/Bus/Tram**: 4 TL per ride
- **Ferry**: 4-7 TL depending on distance
- **Taxi**: 15 TL starting fare
- **Transfer Discount**: Cheaper consecutive rides

## ⚡ Pro Tips

### Rush Hour Strategy
- **Avoid**: 7:30-9:30 AM, 5:30-7:30 PM
- **Alternative**: Walk or work from cafes
- **Ferry**: Less affected by rush hour

### Tourist Traps
- **Airport Taxi**: Use Havabus or metro instead
- **Tourist Boats**: More expensive than public ferries
- **Unlicensed Taxis**: Stick to official yellow taxis

### Rainy Day Options
- **Underground**: Metro and Marmaray
- **Covered**: Tram and bus
- **Ferry**: Can be cancelled in storms

### Cultural Notes
- **Priority Seats**: For elderly, pregnant, disabled
- **Quiet**: Talking softly is appreciated
- **Helping**: Locals often help tourists
- **Payment**: Always have Istanbulkart ready

With this guide, you'll master Istanbul's transport system and travel efficiently around this amazing city!""",
            "author": "Transport Expert",
            "category": "Transportation",
            "tags": ["transport", "metro", "ferry", "istanbulkart", "travel tips"],
            "featured_image": None,
            "published": True,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "views": 134,
            "likes": 19
        }
    ]

# Initialize blog posts if file doesn't exist
def initialize_blog_posts():
    """Initialize blog posts with default content if none exist"""
    posts = load_blog_posts()
    if not posts:
        default_posts = get_default_blog_posts()
        save_blog_posts(default_posts)
        logger.info("Initialized blog with default posts")
        return default_posts
    return posts

# API Endpoints

@router.get("/")
async def get_all_posts(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    published: bool = True,
    limit: int = 10,
    offset: int = 0
):
    """Get all blog posts with optional filtering"""
    try:
        posts = load_blog_posts()
        
        # Filter posts
        filtered_posts = []
        for post in posts:
            if published and not post.get('published', True):
                continue
            if category and post.get('category', '').lower() != category.lower():
                continue
            if tag and tag.lower() not in [t.lower() for t in post.get('tags', [])]:
                continue
            filtered_posts.append(post)
        
        # Sort by creation date (newest first)
        filtered_posts.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Apply pagination
        total = len(filtered_posts)
        paginated_posts = filtered_posts[offset:offset + limit]
        
        return {
            "posts": paginated_posts,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        }
    
    except Exception as e:
        logger.error(f"Error getting blog posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve blog posts")

@router.get("/featured")
async def get_featured_posts(limit: int = 3):
    """Get featured blog posts (most viewed/liked)"""
    try:
        posts = load_blog_posts()
        published_posts = [p for p in posts if p.get('published', True)]
        
        # Sort by views + likes for featured
        featured_posts = sorted(
            published_posts,
            key=lambda x: x.get('views', 0) + x.get('likes', 0) * 5,
            reverse=True
        )[:limit]
        
        return {"featured_posts": featured_posts}
    
    except Exception as e:
        logger.error(f"Error getting featured posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve featured posts")

@router.get("/categories")
async def get_categories():
    """Get all available categories"""
    try:
        posts = load_blog_posts()
        categories = list(set(post.get('category', '') for post in posts if post.get('published', True)))
        categories = [cat for cat in categories if cat]  # Remove empty categories
        return {"categories": sorted(categories)}
    
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve categories")

@router.get("/tags")
async def get_tags():
    """Get all available tags"""
    try:
        posts = load_blog_posts()
        all_tags = []
        for post in posts:
            if post.get('published', True):
                all_tags.extend(post.get('tags', []))
        
        unique_tags = list(set(all_tags))
        return {"tags": sorted(unique_tags)}
    
    except Exception as e:
        logger.error(f"Error getting tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tags")

@router.get("/{post_id}")
async def get_post(post_id: str):
    """Get a specific blog post by ID"""
    try:
        posts = load_blog_posts()
        
        for i, post in enumerate(posts):
            if post.get('id') == post_id:
                # Increment view count
                posts[i]['views'] = post.get('views', 0) + 1
                save_blog_posts(posts)
                return {"post": posts[i]}
        
        raise HTTPException(status_code=404, detail="Blog post not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting blog post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve blog post")

@router.post("/")
async def create_post(post_data: BlogPostCreate):
    """Create a new blog post"""
    try:
        posts = load_blog_posts()
        
        # Create new post
        new_post = {
            "id": str(uuid.uuid4()),
            "title": post_data.title,
            "content": post_data.content,
            "author": post_data.author,
            "category": post_data.category,
            "tags": post_data.tags,
            "featured_image": post_data.featured_image,
            "published": post_data.published,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "views": 0,
            "likes": 0
        }
        
        posts.append(new_post)
        
        if save_blog_posts(posts):
            return {"message": "Blog post created successfully", "post": new_post}
        else:
            raise HTTPException(status_code=500, detail="Failed to save blog post")
    
    except Exception as e:
        logger.error(f"Error creating blog post: {e}")
        raise HTTPException(status_code=500, detail="Failed to create blog post")

@router.put("/{post_id}")
async def update_post(post_id: str, post_data: BlogPostUpdate):
    """Update an existing blog post"""
    try:
        posts = load_blog_posts()
        
        for i, post in enumerate(posts):
            if post.get('id') == post_id:
                # Update only provided fields
                if post_data.title is not None:
                    posts[i]['title'] = post_data.title
                if post_data.content is not None:
                    posts[i]['content'] = post_data.content
                if post_data.author is not None:
                    posts[i]['author'] = post_data.author
                if post_data.category is not None:
                    posts[i]['category'] = post_data.category
                if post_data.tags is not None:
                    posts[i]['tags'] = post_data.tags
                if post_data.featured_image is not None:
                    posts[i]['featured_image'] = post_data.featured_image
                if post_data.published is not None:
                    posts[i]['published'] = post_data.published
                
                posts[i]['updated_at'] = datetime.now().isoformat()
                
                if save_blog_posts(posts):
                    return {"message": "Blog post updated successfully", "post": posts[i]}
                else:
                    raise HTTPException(status_code=500, detail="Failed to save blog post")
        
        raise HTTPException(status_code=404, detail="Blog post not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating blog post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update blog post")

@router.delete("/{post_id}")
async def delete_post(post_id: str):
    """Delete a blog post"""
    try:
        posts = load_blog_posts()
        
        for i, post in enumerate(posts):
            if post.get('id') == post_id:
                deleted_post = posts.pop(i)
                
                if save_blog_posts(posts):
                    return {"message": "Blog post deleted successfully", "deleted_post": deleted_post}
                else:
                    raise HTTPException(status_code=500, detail="Failed to save changes")
        
        raise HTTPException(status_code=404, detail="Blog post not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting blog post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete blog post")

@router.post("/{post_id}/like")
async def like_post(post_id: str):
    """Like a blog post"""
    try:
        posts = load_blog_posts()
        
        for i, post in enumerate(posts):
            if post.get('id') == post_id:
                posts[i]['likes'] = post.get('likes', 0) + 1
                
                if save_blog_posts(posts):
                    return {
                        "message": "Post liked successfully",
                        "likes": posts[i]['likes']
                    }
                else:
                    raise HTTPException(status_code=500, detail="Failed to save like")
        
        raise HTTPException(status_code=404, detail="Blog post not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error liking blog post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to like blog post")

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for blog posts"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
            
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Return file URL
        file_url = f"/uploads/{unique_filename}"
        return {
            "message": "Image uploaded successfully",
            "file_url": file_url,
            "filename": unique_filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image")

@router.get("/search/{query}")
async def search_posts(query: str, limit: int = 10):
    """Search blog posts by title and content"""
    try:
        posts = load_blog_posts()
        published_posts = [p for p in posts if p.get('published', True)]
        
        query_lower = query.lower()
        matching_posts = []
        
        for post in published_posts:
            title_match = query_lower in post.get('title', '').lower()
            content_match = query_lower in post.get('content', '').lower()
            tag_match = any(query_lower in tag.lower() for tag in post.get('tags', []))
            category_match = query_lower in post.get('category', '').lower()
            
            if title_match or content_match or tag_match or category_match:
                # Calculate relevance score
                score = 0
                if title_match:
                    score += 3
                if tag_match:
                    score += 2
                if category_match:
                    score += 2
                if content_match:
                    score += 1
                
                post['relevance_score'] = score
                matching_posts.append(post)
        
        # Sort by relevance score
        matching_posts.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return {
            "query": query,
            "results": matching_posts[:limit],
            "total_found": len(matching_posts)
        }
    
    except Exception as e:
        logger.error(f"Error searching blog posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to search blog posts")

# Initialize blog posts when module is imported
initialize_blog_posts()

logger.info("Blog router initialized successfully")

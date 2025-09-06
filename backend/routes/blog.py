from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_, func
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import os
import uuid
import logging
import json
import math
import random

from database import get_db
from models import BlogPost, BlogImage, BlogLike

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["blog"])

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic models for requests/responses
class ImageCreate(BaseModel):
    url: str
    alt_text: Optional[str] = None

class BlogPostCreate(BaseModel):
    title: str
    content: str
    tags: Optional[List[str]] = None
    district: Optional[str] = None
    author_name: Optional[str] = None
    author_photo: Optional[str] = None
    images: Optional[List[ImageCreate]] = None

class BlogPostResponse(BaseModel):
    id: int
    title: str
    content: str
    tags: Optional[List[str]] = None
    district: Optional[str] = None
    author_name: Optional[str] = None
    author_photo: Optional[str] = None
    likes_count: int = 0
    created_at: datetime
    images: Optional[List[dict]] = None
    
    class Config:
        from_attributes = True

@router.get("/posts")
async def get_blog_posts(
    page: int = 1,
    limit: int = 10,
    search: Optional[str] = None,
    tag: Optional[str] = None,
    district: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get paginated list of blog posts with optional filtering"""
    try:
        # Calculate offset for pagination
        offset = (page - 1) * limit
        
        # Base query
        query = db.query(BlogPost).filter(BlogPost.is_published == True)
        total_query = db.query(BlogPost).filter(BlogPost.is_published == True)
        
        # Apply search filter
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                or_(
                    BlogPost.title.ilike(search_filter),
                    BlogPost.content.ilike(search_filter)
                )
            )
            total_query = total_query.filter(
                or_(
                    BlogPost.title.ilike(search_filter),
                    BlogPost.content.ilike(search_filter)
                )
            )
        
        # Apply tag filter
        if tag:
            # Convert stored tags JSON to searchable format
            query = query.filter(BlogPost.tags != None)
            query = query.filter(BlogPost.tags.contains(f'"{tag}"'))
            total_query = total_query.filter(BlogPost.tags != None)
            total_query = total_query.filter(BlogPost.tags.contains(f'"{tag}"'))
            
        # Apply district filter
        if district:
            query = query.filter(BlogPost.district == district)
            total_query = total_query.filter(BlogPost.district == district)
        
        # Get total count for pagination
        total = total_query.count()
        
        # Get posts for current page
        posts = query.order_by(desc(BlogPost.created_at)).offset(offset).limit(limit).all()
        
        # Convert to response format
        response_posts = []
        for post in posts:
            # Parse tags from JSON
            tags = []
            if post.tags:
                try:
                    tags = json.loads(post.tags) if isinstance(post.tags, str) else post.tags
                except:
                    pass
            
            # Get associated images
            images = []
            for img in post.images:
                images.append({
                    'id': img.id,
                    'url': img.url,
                    'alt_text': img.alt_text
                })
            
            response_posts.append({
                'id': post.id,
                'title': post.title,
                'content': post.content,
                'tags': tags,
                'district': post.district,
                'author_name': post.author_name,
                'author_photo': post.author_photo,
                'likes_count': post.likes_count,
                'created_at': post.created_at,
                'images': images
            })
        
        return {
            "posts": response_posts,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": math.ceil(total / limit) if limit > 0 else 1
        }
        
    except Exception as e:
        logger.error(f"Error fetching blog posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch blog posts")

@router.get("/posts/{post_id}", response_model=BlogPostResponse)
async def get_blog_post(post_id: int, db: Session = Depends(get_db)):
    """Get a specific blog post by ID"""
    try:
        post = db.query(BlogPost).filter(BlogPost.id == post_id, BlogPost.is_published == True).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Parse tags from JSON
        tags = []
        if post.tags:
            try:
                tags = json.loads(post.tags) if isinstance(post.tags, str) else post.tags
            except:
                pass
        
        # Get associated images
        images = []
        for img in post.images:
            images.append({
                'id': img.id,
                'url': img.url,
                'alt_text': img.alt_text
            })
        
        return {
            'id': post.id,
            'title': post.title,
            'content': post.content,
            'tags': tags,
            'district': post.district,
            'author_name': post.author_name,
            'author_photo': post.author_photo,
            'likes_count': post.likes_count,
            'created_at': post.created_at,
            'images': images
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching blog post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch blog post")

@router.post("/posts", response_model=BlogPostResponse)
async def create_blog_post(post_data: BlogPostCreate, db: Session = Depends(get_db)):
    """Create a new blog post"""
    try:
        # Convert tags to JSON string
        tags_json = None
        if post_data.tags:
            tags_json = json.dumps(post_data.tags)
        
        # Create new blog post
        new_post = BlogPost(
            title=post_data.title,
            content=post_data.content,
            tags=tags_json,
            district=post_data.district,
            author_name=post_data.author_name,
            author_photo=post_data.author_photo,
            is_published=True,
            likes_count=0
        )
        
        db.add(new_post)
        db.flush()  # To get the post ID
        
        # Add images if provided
        if post_data.images:
            for img_data in post_data.images:
                image = BlogImage(
                    blog_post_id=new_post.id,
                    url=img_data.url,
                    alt_text=img_data.alt_text
                )
                db.add(image)
        
        db.commit()
        db.refresh(new_post)
        
        # Return created post
        return await get_blog_post(new_post.id, db)
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating blog post: {e}")
        raise HTTPException(status_code=500, detail="Failed to create blog post")

@router.post("/posts/{post_id}/like")
async def like_blog_post(post_id: int, request: Request, db: Session = Depends(get_db)):
    """Like a blog post (increment like count) - one like per user"""
    try:
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Use client IP address as user identifier
        # In a real app, you'd use authenticated user ID
        user_identifier = request.client.host
        
        # Check if user already liked this post
        existing_like = db.query(BlogLike).filter(
            BlogLike.blog_post_id == post_id,
            BlogLike.user_identifier == user_identifier
        ).first()
        
        if existing_like:
            raise HTTPException(status_code=400, detail="You have already liked this post")
        
        # Create new like record
        new_like = BlogLike(
            blog_post_id=post_id,
            user_identifier=user_identifier
        )
        db.add(new_like)
        
        # Increment like count
        post.likes_count += 1
        db.commit()
        
        return {
            "message": "Post liked successfully", 
            "likes_count": post.likes_count,
            "already_liked": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error liking post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to like post")

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for blog posts"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Return the URL path
        return {
            "url": f"/uploads/{unique_filename}",
            "filename": unique_filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading image: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image")

@router.get("/tags")
async def get_popular_tags(limit: int = 20, db: Session = Depends(get_db)):
    """Get popular tags from blog posts"""
    try:
        posts = db.query(BlogPost).filter(
            BlogPost.is_published == True,
            BlogPost.tags != None
        ).all()
        
        tag_count = {}
        for post in posts:
            if post.tags:
                try:
                    tags = json.loads(post.tags) if isinstance(post.tags, str) else post.tags
                    if isinstance(tags, list):
                        for tag in tags:
                            tag_count[tag] = tag_count.get(tag, 0) + 1
                except:
                    pass
        
        # Sort by count and return top tags
        popular_tags = sorted(tag_count.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, count in popular_tags[:limit]]
        
    except Exception as e:
        logger.error(f"Error fetching tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tags")

@router.get("/districts")
async def get_popular_districts(limit: int = 20, db: Session = Depends(get_db)):
    """Get popular districts from blog posts"""
    try:
        result = db.query(BlogPost.district, func.count(BlogPost.district).label('count')).filter(
            BlogPost.is_published == True,
            BlogPost.district != None
        ).group_by(BlogPost.district).order_by(desc('count')).limit(limit).all()
        
        return [district for district, count in result]
        
    except Exception as e:
        logger.error(f"Error fetching districts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch districts")

@router.get("/posts/{post_id}/like-status")
async def get_like_status(post_id: int, request: Request, db: Session = Depends(get_db)):
    """Check if current user has liked this post"""
    try:
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Use client IP address as user identifier
        user_identifier = request.client.host
        
        # Check if user has liked this post
        existing_like = db.query(BlogLike).filter(
            BlogLike.blog_post_id == post_id,
            BlogLike.user_identifier == user_identifier
        ).first()
        
        return {
            "already_liked": existing_like is not None,
            "likes_count": post.likes_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking like status for post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to check like status")

@router.get("/featured")
async def get_featured_posts(limit: int = 3, db: Session = Depends(get_db)):
    """Get featured blog posts (most liked posts from last 30 days)"""
    try:
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        featured_posts = db.query(BlogPost).filter(
            BlogPost.is_published == True,
            BlogPost.created_at >= thirty_days_ago
        ).order_by(desc(BlogPost.likes_count), desc(BlogPost.created_at)).limit(limit).all()
        
        # If not enough recent popular posts, fill with most recent
        if len(featured_posts) < limit:
            recent_posts = db.query(BlogPost).filter(
                BlogPost.is_published == True
            ).order_by(desc(BlogPost.created_at)).limit(limit - len(featured_posts)).all()
            
            # Avoid duplicates
            featured_ids = {p.id for p in featured_posts}
            for post in recent_posts:
                if post.id not in featured_ids:
                    featured_posts.append(post)
                    if len(featured_posts) >= limit:
                        break
        
        # Convert to response format
        response_posts = []
        for post in featured_posts:
            tags = []
            if post.tags:
                try:
                    tags = json.loads(post.tags) if isinstance(post.tags, str) else post.tags
                except:
                    pass
            
            images = []
            for img in post.images:
                images.append({
                    'id': img.id,
                    'url': img.url,
                    'alt_text': img.alt_text
                })
            
            response_posts.append({
                'id': post.id,
                'title': post.title,
                'content': post.content[:200] + '...' if len(post.content) > 200 else post.content,
                'tags': tags,
                'district': post.district,
                'author_name': post.author_name,
                'author_photo': post.author_photo,
                'likes_count': post.likes_count,
                'created_at': post.created_at,
                'images': images[:1]  # Only include first image for featured
            })
        
        return {"featured_posts": response_posts}
        
    except Exception as e:
        logger.error(f"Error fetching featured posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch featured posts")

@router.get("/trending")
async def get_trending_posts(limit: int = 5, db: Session = Depends(get_db)):
    """Get trending blog posts (posts with most likes in last 7 days)"""
    try:
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        
        # Get posts with likes in the last 7 days
        trending_query = db.query(BlogPost).join(BlogLike).filter(
            BlogPost.is_published == True,
            BlogLike.created_at >= seven_days_ago
        ).group_by(BlogPost.id).order_by(
            desc(func.count(BlogLike.id)),
            desc(BlogPost.created_at)
        ).limit(limit)
        
        trending_posts = trending_query.all()
        
        # If not enough trending posts, fill with recent posts
        if len(trending_posts) < limit:
            recent_posts = db.query(BlogPost).filter(
                BlogPost.is_published == True
            ).order_by(desc(BlogPost.created_at)).limit(limit).all()
            
            # Combine and remove duplicates
            trending_ids = {p.id for p in trending_posts}
            for post in recent_posts:
                if post.id not in trending_ids:
                    trending_posts.append(post)
                    if len(trending_posts) >= limit:
                        break
        
        response_posts = []
        for post in trending_posts:
            tags = []
            if post.tags:
                try:
                    tags = json.loads(post.tags) if isinstance(post.tags, str) else post.tags
                except:
                    pass
            
            response_posts.append({
                'id': post.id,
                'title': post.title,
                'content': post.content[:150] + '...' if len(post.content) > 150 else post.content,
                'tags': tags,
                'district': post.district,
                'author_name': post.author_name,
                'likes_count': post.likes_count,
                'created_at': post.created_at
            })
        
        return {"trending_posts": response_posts}
        
    except Exception as e:
        logger.error(f"Error fetching trending posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch trending posts")

@router.get("/stats")
async def get_blog_stats(db: Session = Depends(get_db)):
    """Get blog statistics for dashboard"""
    try:
        total_posts = db.query(BlogPost).filter(BlogPost.is_published == True).count()
        total_likes = db.query(func.sum(BlogPost.likes_count)).scalar() or 0
        total_authors = db.query(BlogPost.author_name).filter(
            BlogPost.is_published == True,
            BlogPost.author_name != None
        ).distinct().count()
        
        # Recent activity (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_posts = db.query(BlogPost).filter(
            BlogPost.is_published == True,
            BlogPost.created_at >= seven_days_ago
        ).count()
        
        recent_likes = db.query(BlogLike).filter(
            BlogLike.created_at >= seven_days_ago
        ).count()
        
        return {
            "total_posts": total_posts,
            "total_likes": total_likes,
            "total_authors": total_authors,
            "recent_posts": recent_posts,
            "recent_likes": recent_likes
        }
        
    except Exception as e:
        logger.error(f"Error fetching blog stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch blog stats")

@router.get("/posts/{post_id}/related")
async def get_related_posts(post_id: int, limit: int = 4, db: Session = Depends(get_db)):
    """Get related posts based on district and tags"""
    try:
        # Get the original post
        original_post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not original_post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Find related posts by district first
        related_posts = []
        if original_post.district:
            district_posts = db.query(BlogPost).filter(
                BlogPost.is_published == True,
                BlogPost.district == original_post.district,
                BlogPost.id != post_id
            ).order_by(desc(BlogPost.likes_count), desc(BlogPost.created_at)).limit(limit).all()
            related_posts.extend(district_posts)
        
        # If we need more posts, find by similar tags
        if len(related_posts) < limit and original_post.tags:
            try:
                original_tags = json.loads(original_post.tags) if isinstance(original_post.tags, str) else original_post.tags
                if original_tags:
                    existing_ids = {p.id for p in related_posts} | {post_id}
                    
                    for tag in original_tags:
                        tag_posts = db.query(BlogPost).filter(
                            BlogPost.is_published == True,
                            BlogPost.tags.contains(f'"{tag}"'),
                            ~BlogPost.id.in_(existing_ids)
                        ).order_by(desc(BlogPost.likes_count)).limit(limit - len(related_posts)).all()
                        
                        for post in tag_posts:
                            if len(related_posts) < limit:
                                related_posts.append(post)
                                existing_ids.add(post.id)
            except:
                pass
        
        # If still need more, get recent posts
        if len(related_posts) < limit:
            existing_ids = {p.id for p in related_posts} | {post_id}
            recent_posts = db.query(BlogPost).filter(
                BlogPost.is_published == True,
                ~BlogPost.id.in_(existing_ids)
            ).order_by(desc(BlogPost.created_at)).limit(limit - len(related_posts)).all()
            related_posts.extend(recent_posts)
        
        # Convert to response format
        response_posts = []
        for post in related_posts[:limit]:
            tags = []
            if post.tags:
                try:
                    tags = json.loads(post.tags) if isinstance(post.tags, str) else post.tags
                except:
                    pass
            
            images = []
            for img in post.images[:1]:  # Only first image for related posts
                images.append({
                    'id': img.id,
                    'url': img.url,
                    'alt_text': img.alt_text
                })
            
            response_posts.append({
                'id': post.id,
                'title': post.title,
                'content': post.content[:120] + '...' if len(post.content) > 120 else post.content,
                'tags': tags,
                'district': post.district,
                'author_name': post.author_name,
                'likes_count': post.likes_count,
                'created_at': post.created_at,
                'images': images
            })
        
        return {"related_posts": response_posts}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching related posts for {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch related posts")

@router.post("/seed-sample-posts")
async def seed_sample_posts(db: Session = Depends(get_db)):
    """Seed the blog with sample Istanbul travel posts"""
    try:
        # Check if posts already exist
        existing_posts = db.query(BlogPost).filter(BlogPost.author_name == "AI Istanbul Team").count()
        if existing_posts > 0:
            return {"message": "Sample posts already exist", "posts_created": 0}
        
        sample_posts = [
            {
                "title": "Hidden Gems of Beyoğlu: Beyond Istiklal Street",
                "content": """Beyoğlu district offers so much more than the famous Istiklal Street! During my recent exploration, I discovered incredible spots that most tourists miss.

**Galata Mevlevihanesi (Whirling Dervish Museum)**
This 15th-century monastery offers an authentic glimpse into Sufi culture. The weekend ceremonies are mesmerizing, and the peaceful courtyard provides a perfect escape from the bustling streets.

**Kamondo Stairs**
These Art Nouveau stairs from 1870 connect Galata to Karaköy. Perfect for photography and a unique perspective of the Golden Horn. Best visited during golden hour for amazing light.

**Balat Neighborhood**
Wander through this colorful historic district with its rainbow houses, antique shops, and cozy cafes. Don't miss Fener Orthodox Patriarchate and the stunning Church of St. Stephen.

**Local Food Spots:**
- Pandeli Restaurant (1901) - Ottoman cuisine in a historic setting
- Çiya Sofrası - Amazing traditional Anatolian dishes
- Hamdi Restaurant - Best lahmacun in the city

**Pro Tips:**
- Visit Galata Tower early morning (9 AM) to avoid crowds
- Take the nostalgic tram for a fun way to navigate Istiklal
- Explore the side streets - that's where the real character lies!

The authentic charm of old Istanbul lives on in these neighborhoods. Each corner tells a story spanning centuries of history.""",
                "tags": ["beyoğlu", "hidden gems", "local spots", "galata", "balat", "food", "culture"],
                "district": "Beyoğlu",
                "author_name": "AI Istanbul Team",
                "author_photo": None
            },
            {
                "title": "Kadıköy Food Tour: Asian Side's Culinary Treasures",
                "content": """Kadıköy on Istanbul's Asian side is a food lover's paradise! This vibrant district offers some of the best authentic Turkish cuisine away from tourist crowds.

**Must-Visit Food Spots:**

**Kadıköy Fish Market (Balık Pazarı)**
Fresh seafood heaven! Watch fishmongers prepare your catch while you sip tea. The surrounding meyhanes (taverns) serve the freshest fish with stunning Bosphorus views.

**Çiya Sofrası**
This legendary restaurant showcases regional Turkish cuisine you won't find elsewhere. Try their weekly specials featuring dishes from different Anatolian regions.

**Fazıl Bey Turkish Coffee**
Since 1923, this tiny shop has been roasting the perfect Turkish coffee. The owner still operates the original copper roaster!

**Moda Neighborhood**
Stroll along the seaside promenade, then explore the trendy cafes and bistros. Perfect for a sunset dinner with Asian side vibes.

**Local Delicacies to Try:**
- Kumru (special İzmir sandwich)
- Turkish breakfast at Dem Karakoy
- Artisanal ice cream at Mini Dondurma
- Fresh döner at Pandeli

**Getting There:**
Take the ferry from Eminönü to Kadıköy (20 minutes) - the journey itself is part of the experience with stunning city views!

**Best Times:**
Weekday mornings for authentic local atmosphere, or weekend evenings for the vibrant nightlife scene.

Kadıköy proves that Istanbul's best food experiences often lie off the beaten path.""",
                "tags": ["kadıköy", "food tour", "asian side", "fish market", "local cuisine", "ferry", "authentic"],
                "district": "Kadıköy",
                "author_name": "Food Explorer",
                "author_photo": None
            },
            {
                "title": "Sultanahmet After Dark: When the Crowds Go Home",
                "content": """Most visitors see Sultanahmet during peak hours, but the real magic happens after sunset when the ancient district transforms into something truly special.

**Evening Experiences:**

**Blue Mosque at Sunset**
The evening call to prayer echoing through the courtyard is deeply moving. The warm lighting creates an ethereal atmosphere that photos can't capture.

**Hagia Sophia Night Illumination**
After dark, this architectural marvel is beautifully lit, revealing details invisible during crowded daytime visits. The surrounding park becomes peaceful and perfect for reflection.

**Traditional Turkish Baths (Hamam)**
End your day at Cagaloglu Hamami (1584) - one of the oldest operating baths in the world. The evening sessions are less crowded and more relaxing.

**Rooftop Dining**
Several hotels offer rooftop restaurants with incredible views of illuminated monuments. Try:
- Seven Hills Restaurant - Blue Mosque views
- Pandeli - Historic restaurant since 1901
- Matbah Ottoman Palace Cuisine

**Night Photography Tips:**
- Blue hour (30 minutes after sunset) provides the best lighting
- Bring a tripod for long exposures
- The fountain in Sultanahmet Square creates beautiful reflections

**Safety & Logistics:**
The area is very safe at night with good police presence. Most attractions are walkable, and taxis are readily available.

**Local Secret:**
Visit the small tea gardens behind the Blue Mosque where locals gather for evening tea and backgammon games.""",
                "tags": ["sultanahmet", "evening", "night photography", "blue mosque", "hagia sophia", "hamam", "rooftop dining"],
                "district": "Sultanahmet",
                "author_name": "Night Explorer",
                "author_photo": None
            },
            {
                "title": "Bosphorus Ferry Guide: Best Routes and Hidden Stops",
                "content": """The Bosphorus ferry isn't just transportation - it's one of Istanbul's greatest experiences! Here's your complete guide to making the most of these scenic journeys.

**Best Ferry Routes:**

**Long Bosphorus Tour (6 hours round trip)**
- Start: Eminönü 
- Stops: Beşiktaş, Ortaköy, Bebek, Rumeli Kavağı, Anadolu Kavağı
- Perfect for a full day exploring both European and Asian coastlines

**Short Bosphorus Cruise (2 hours)**
- Circular route from Eminönü
- Great for photography and first-time visitors
- Operates multiple times daily

**Hidden Gem Stops:**

**Anadolu Kavağı**
This charming fishing village at the Black Sea entrance offers:
- Fresh fish restaurants with sea views
- Yoros Castle ruins for hiking enthusiasts
- Peaceful beaches away from city crowds

**Bebek**
Upscale neighborhood perfect for:
- Waterfront cafes and restaurants
- Beautiful art nouveau architecture
- Weekend markets with local crafts

**Çengelköy**
Asian side gem featuring:
- Historic wooden mansions (yalıs)
- Traditional Turkish breakfast spots
- Peaceful walking paths along the water

**Pro Ferry Tips:**
- Buy Istanbulkart for significant savings
- Sit on the right side when going north for best European side views
- Bring warm clothes - it gets windy on deck!
- Pack snacks or buy simit (Turkish bagel) and tea on board

**Best Times:**
- Morning (8-10 AM): Clear views, fewer crowds
- Sunset (6-8 PM): Magical lighting, romantic atmosphere
- Winter: Dramatic stormy seas, cozy cabin atmosphere

**Photography Spots:**
Upper deck provides unobstructed 360° views of palaces, mosques, and modern Istanbul skyline.""",
                "tags": ["bosphorus", "ferry", "cruise", "bebek", "ortaköy", "transportation", "scenic routes"],
                "district": "General",
                "author_name": "Ferry Captain",
                "author_photo": None
            },
            {
                "title": "Grand Bazaar Insider Tips: Navigate Like a Local",
                "content": """The Grand Bazaar can be overwhelming, but with these insider tips, you'll shop like a pro and discover treasures beyond the tourist traps.

**Navigation Strategy:**

**Main Entrances:**
- Beyazıt Gate: Less crowded, leads to authentic carpet section
- Nuruosmaniye Gate: Tourist entrance, but good for initial orientation
- Mahmutpaşa Gate: Local entrance, leads to gold jewelry section

**What to Buy & Where:**

**Authentic Turkish Carpets**
- Look for hand-knotted pieces (feel the back for knots)
- Best shops: Around Beyazıt Gate area
- Negotiate: Start at 40% of asking price

**Jewelry & Gold**
- Turkey has strict gold purity standards
- Check for government hallmarks
- Kuyumcular Çarşısı section has best selection

**Leather Goods**
- High-quality leather jackets and bags
- Ask to see the leather's flexibility and stitching quality
- Bargaining is expected and part of the experience

**Turkish Delight & Spices**
- Sample before buying - quality varies greatly
- Avoid pre-packaged tourist versions
- Ask for recommendations from local shoppers

**Bargaining Rules:**
1. Always negotiate - starting prices are inflated
2. Be prepared to walk away (often brings better offers)
3. Cash payments get better discounts
4. Bundle multiple items for better deals

**Hidden Gems:**
- Şark Kahvesi: Historic coffee house for authentic atmosphere
- Zincirli Han: Antique books and vintage items
- Cevahir Bedesteni: High-end antiques and collectibles

**Best Times to Visit:**
- Early morning (9-10 AM): Fewer crowds, better attention from shopkeepers
- Late afternoon: Good lighting for photos, locals doing shopping

**Cultural Tips:**
- Accept offered tea - it's hospitality, not pressure to buy
- Learn basic Turkish greetings - shopkeepers appreciate the effort
- Dress modestly and comfortably for walking on uneven surfaces""",
                "tags": ["grand bazaar", "shopping", "carpets", "jewelry", "bargaining", "local tips", "authentic"],
                "district": "Fatih",
                "author_name": "Bazaar Expert",
                "author_photo": None
            }
        ]
        
        created_posts = []
        for post_data in sample_posts:
            # Create new blog post
            new_post = BlogPost(
                title=post_data["title"],
                content=post_data["content"],
                tags=json.dumps(post_data["tags"]),
                district=post_data["district"],
                author_name=post_data["author_name"],
                author_photo=post_data["author_photo"],
                is_published=True,
                likes_count=random.randint(5, 25)  # Random initial likes for demo
            )
            
            db.add(new_post)
            created_posts.append(new_post.title)
        
        db.commit()
        
        return {
            "message": "Sample posts created successfully", 
            "posts_created": len(created_posts),
            "titles": created_posts
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error seeding sample posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to create sample posts")

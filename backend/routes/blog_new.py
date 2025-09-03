from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_, func
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
import os
import uuid
import logging
import json
import math

from database import get_db
from models import BlogPost, BlogImage

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
    images: Optional[List[ImageCreate]] = None

class BlogPostResponse(BaseModel):
    id: int
    title: str
    content: str
    tags: Optional[List[str]] = None
    district: Optional[str] = None
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
async def like_blog_post(post_id: int, db: Session = Depends(get_db)):
    """Like a blog post (increment like count)"""
    try:
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        post.likes_count += 1
        db.commit()
        
        return {"message": "Post liked successfully", "likes_count": post.likes_count}
        
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

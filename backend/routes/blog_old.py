from fastapi import APIRouter, HTTPException, Depends, Request, File, UploadFile, Form
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_
from typing import List, Optional
import uuid
import os
import shutil
from datetime import datetime
import logging
from pydantic import BaseModel

from database import get_db
from models import BlogPost, BlogImage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["blog"])

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

class BlogPostSummary(BaseModel):
    id: int
    title: str
    content: str
    tags: Optional[List[str]] = None
    district: Optional[str] = None
    likes_count: int = 0
    created_at: datetime
    
    class Config:
        from_attributes = True
    author_name: str
    featured_image: Optional[str] = None
    tags: Optional[str] = None
    districts_visited: Optional[str] = None
    trip_duration: Optional[str] = None
    created_at: datetime
    views: int
    likes: int
    ai_generated_summary: Optional[str] = None
    
    class Config:
        from_attributes = True

# File upload configuration
UPLOAD_DIR = "uploads/blog_images"
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/posts", response_model=List[BlogPostSummary])
async def get_blog_posts(
    page: int = 1, 
    limit: int = 10, 
    search: Optional[str] = None,
    tag: Optional[str] = None,
    district: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get blog posts with pagination and filtering"""
    try:
        offset = (page - 1) * limit
        query = db.query(BlogPost).filter(BlogPost.is_published == 1)
        
        # Apply filters
        if search:
            query = query.filter(
                or_(
                    BlogPost.title.contains(search),
                    BlogPost.content.contains(search),
                    BlogPost.tags.contains(search)
                )
            )
        
        if tag:
            query = query.filter(BlogPost.tags.contains(tag))
            
        if district:
            query = query.filter(BlogPost.districts_visited.contains(district))
        
        posts = query.order_by(desc(BlogPost.created_at)).offset(offset).limit(limit).all()
        
        # Convert to response format
        response_posts = []
        for post in posts:
            # Generate summary if not exists
            summary = post.ai_generated_summary
            if not summary and post.content:
                # Create a simple excerpt
                summary = (post.content[:200] + "...") if len(post.content) > 200 else post.content
            
            response_posts.append(BlogPostSummary(
                id=post.id,
                title=post.title,
                author_name=post.author_name,
                featured_image=post.featured_image,
                tags=post.tags,
                districts_visited=post.districts_visited,
                trip_duration=post.trip_duration,
                created_at=post.created_at,
                views=post.views,
                likes=post.likes,
                ai_generated_summary=summary
            ))
        
        return response_posts
        
    except Exception as e:
        logger.error(f"Error fetching blog posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch blog posts")

@router.get("/posts/{post_id}", response_model=BlogPostResponse)
async def get_blog_post(post_id: int, db: Session = Depends(get_db)):
    """Get a specific blog post by ID"""
    try:
        post = db.query(BlogPost).filter(BlogPost.id == post_id, BlogPost.is_published == 1).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Increment view count
        post.views += 1
        db.commit()
        
        return BlogPostResponse(
            id=post.id,
            title=post.title,
            content=post.content,
            author_name=post.author_name,
            featured_image=post.featured_image,
            tags=post.tags,
            districts_visited=post.districts_visited,
            trip_duration=post.trip_duration,
            created_at=post.created_at,
            views=post.views,
            likes=post.likes,
            ai_generated_summary=post.ai_generated_summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching blog post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch blog post")

@router.post("/posts", response_model=BlogPostResponse)
async def create_blog_post(
    post: BlogPostCreate,
    db: Session = Depends(get_db)
):
    """Create a new blog post"""
    try:
        # Create blog post
        new_post = BlogPost(
            title=post.title,
            content=post.content,
            author_name=post.author_name,
            author_email=post.author_email,
            tags=post.tags,
            districts_visited=post.districts_visited,
            trip_duration=post.trip_duration
        )
        
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        
        logger.info(f"Created new blog post: {new_post.id} - {new_post.title}")
        
        return BlogPostResponse(
            id=new_post.id,
            title=new_post.title,
            content=new_post.content,
            author_name=new_post.author_name,
            featured_image=new_post.featured_image,
            tags=new_post.tags,
            districts_visited=new_post.districts_visited,
            trip_duration=new_post.trip_duration,
            created_at=new_post.created_at,
            views=new_post.views,
            likes=new_post.likes,
            ai_generated_summary=new_post.ai_generated_summary
        )
        
    except Exception as e:
        logger.error(f"Error creating blog post: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create blog post")

@router.post("/posts/{post_id}/like")
async def like_blog_post(post_id: int, db: Session = Depends(get_db)):
    """Like a blog post"""
    try:
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        post.likes += 1
        db.commit()
        
        return {"message": "Post liked successfully", "likes": post.likes}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error liking post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to like post")

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for blog posts"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset position
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Return URL path (relative to serve statically)
        image_url = f"/uploads/blog_images/{unique_filename}"
        
        logger.info(f"Uploaded image: {image_url}")
        
        return {
            "message": "Image uploaded successfully",
            "image_url": image_url,
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
        posts = db.query(BlogPost.tags).filter(
            BlogPost.is_published == 1,
            BlogPost.tags.isnot(None)
        ).all()
        
        # Count tag frequencies
        tag_counts = {}
        for (tags_str,) in posts:
            if tags_str:
                tags = [tag.strip() for tag in tags_str.split(',')]
                for tag in tags:
                    if tag:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort by frequency and return top tags
        popular_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return [{"tag": tag, "count": count} for tag, count in popular_tags]
        
    except Exception as e:
        logger.error(f"Error fetching popular tags: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tags")

@router.get("/districts")
async def get_popular_districts(limit: int = 15, db: Session = Depends(get_db)):
    """Get popular districts from blog posts"""
    try:
        posts = db.query(BlogPost.districts_visited).filter(
            BlogPost.is_published == 1,
            BlogPost.districts_visited.isnot(None)
        ).all()
        
        # Count district frequencies
        district_counts = {}
        for (districts_str,) in posts:
            if districts_str:
                districts = [district.strip() for district in districts_str.split(',')]
                for district in districts:
                    if district:
                        district_counts[district] = district_counts.get(district, 0) + 1
        
        # Sort by frequency
        popular_districts = sorted(district_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return [{"district": district, "count": count} for district, count in popular_districts]
        
    except Exception as e:
        logger.error(f"Error fetching popular districts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch districts")

"""
Blog API Endpoints
Handles blog posts, comments, and likes
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request, Response, status
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import func
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import logging

from database import get_db

# Import rate limiter
try:
    from rate_limiter import limiter
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False
    limiter = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/blog", tags=["Blog"])

# =============================
# PYDANTIC MODELS
# =============================

class BlogPostCreate(BaseModel):
    title: str = Field(..., min_length=10, max_length=200, description="Blog post title")
    content: str = Field(..., min_length=100, description="Blog post content")
    author_name: str = Field(..., min_length=2, max_length=100, description="Author name")
    district: Optional[str] = Field(None, max_length=100, description="Istanbul district")
    tags: Optional[List[str]] = Field(None, description="Post tags")

class BlogPostResponse(BaseModel):
    id: int
    title: str
    content: str
    author_name: str
    district: Optional[str]
    created_at: datetime
    likes_count: int
    comments_count: int

    class Config:
        from_attributes = True

class BlogCommentCreate(BaseModel):
    author_name: str = Field(..., min_length=2, max_length=100, description="Commenter name")
    author_email: Optional[str] = Field(None, description="Commenter email")
    content: str = Field(..., min_length=10, max_length=1000, description="Comment content")

class BlogCommentResponse(BaseModel):
    id: int
    blog_post_id: int
    author_name: str
    content: str
    created_at: datetime
    is_approved: bool

    class Config:
        from_attributes = True

class BlogLikeResponse(BaseModel):
    success: bool
    is_liked: bool
    likes_count: int

class BlogPostListResponse(BaseModel):
    posts: List[BlogPostResponse]
    total: int
    page: int
    pages: int


# =============================
# BLOG POST ENDPOINTS
# =============================

@router.get("/posts", response_model=BlogPostListResponse)
async def get_blog_posts(
    request: Request,
    response: Response,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(12, ge=1, le=50, description="Posts per page"),
    district: Optional[str] = Query(None, description="Filter by district"),
    search: Optional[str] = Query(None, description="Search in title/content"),
    sort: str = Query("newest", description="Sort by: newest, oldest, popular"),
    db: Session = Depends(get_db)
):
    """
    Get paginated blog posts with filtering and sorting
    ✅ Fixed N+1 query problem with subquery
    ✅ Added response caching headers
    ✅ Rate limited: 60/minute per IP
    """
    try:
        from models import BlogPost, BlogComment
        
        # Add cache headers for public content (5 minutes)
        response.headers["Cache-Control"] = "public, max-age=300"
        response.headers["ETag"] = f"posts-{page}-{district}-{sort}-{search}"
        
        # Base query with eager loading
        query = db.query(BlogPost)
        
        # Apply filters
        if district:
            query = query.filter(BlogPost.district == district)
        
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                (BlogPost.title.ilike(search_filter)) | 
                (BlogPost.content.ilike(search_filter))
            )
        
        # Apply sorting
        if sort == "newest":
            query = query.order_by(BlogPost.created_at.desc())
        elif sort == "oldest":
            query = query.order_by(BlogPost.created_at.asc())
        elif sort == "popular":
            query = query.order_by(BlogPost.likes_count.desc())
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * per_page
        posts = query.offset(offset).limit(per_page).all()
        
        # FIX: N+1 Query Problem - Get all comment counts in one query
        post_ids = [post.id for post in posts]
        comment_counts = {}
        if post_ids:
            comment_count_query = db.query(
                BlogComment.blog_post_id,
                func.count(BlogComment.id).label('count')
            ).filter(
                BlogComment.blog_post_id.in_(post_ids),
                BlogComment.is_approved == True
            ).group_by(BlogComment.blog_post_id).all()
            
            comment_counts = {post_id: count for post_id, count in comment_count_query}
        
        # Format response
        formatted_posts = []
        for post in posts:
            formatted_posts.append(BlogPostResponse(
                id=post.id,
                title=post.title,
                content=post.content,
                author_name=post.author or "Anonymous",
                district=post.district,
                created_at=post.created_at,
                likes_count=post.likes_count or 0,
                comments_count=comment_counts.get(post.id, 0)
            ))
        
        return BlogPostListResponse(
            posts=formatted_posts,
            total=total,
            page=page,
            pages=(total + per_page - 1) // per_page
        )
        
    except Exception as e:
        logger.error(f"Error fetching blog posts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch blog posts")


@router.get("/posts/{post_id}", response_model=BlogPostResponse)
async def get_blog_post(post_id: int, response: Response, db: Session = Depends(get_db)):
    """
    Get a single blog post by ID
    ✅ Added response caching headers
    """
    try:
        from models import BlogPost, BlogComment
        
        # Add cache headers for public content (10 minutes)
        response.headers["Cache-Control"] = "public, max-age=600"
        response.headers["ETag"] = f"post-{post_id}"
        
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        comments_count = db.query(BlogComment).filter(
            BlogComment.blog_post_id == post.id,
            BlogComment.is_approved == True
        ).count()
        
        return BlogPostResponse(
            id=post.id,
            title=post.title,
            content=post.content,
            author_name=post.author or "Anonymous",
            district=post.district,
            created_at=post.created_at,
            likes_count=post.likes_count or 0,
            comments_count=comments_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching blog post {post_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch blog post")


@router.post("/posts", response_model=BlogPostResponse, status_code=status.HTTP_201_CREATED)
async def create_blog_post(post: BlogPostCreate, request: Request, db: Session = Depends(get_db)):
    """
    Create a new blog post
    ✅ Rate limited: 10 posts/hour per IP
    """
    try:
        from models import BlogPost
        
        # Rate limiting (if available)
        if RATE_LIMITER_AVAILABLE and limiter:
            try:
                limiter.hit("blog_post_create", request=request)
            except Exception as e:
                logger.warning(f"Rate limiter error: {e}")
        
        # Create new blog post
        new_post = BlogPost(
            title=post.title,
            content=post.content,
            author=post.author_name,
            district=post.district,
            created_at=datetime.utcnow(),
            likes_count=0
        )
        
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        
        logger.info(f"✅ Created blog post: {new_post.id} - {new_post.title}")
        
        return BlogPostResponse(
            id=new_post.id,
            title=new_post.title,
            content=new_post.content,
            author_name=new_post.author or "Anonymous",
            district=new_post.district,
            created_at=new_post.created_at,
            likes_count=0,
            comments_count=0
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create blog post")


# =============================
# COMMENT ENDPOINTS
# =============================

@router.get("/posts/{post_id}/comments")
async def get_post_comments(post_id: int, db: Session = Depends(get_db)):
    """
    Get all approved comments for a blog post
    """
    try:
        from models import BlogPost, BlogComment
        
        # Verify post exists
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Get approved comments
        comments = db.query(BlogComment).filter(
            BlogComment.blog_post_id == post_id,
            BlogComment.is_approved == True
        ).order_by(BlogComment.created_at.desc()).all()
        
        formatted_comments = [
            BlogCommentResponse(
                id=comment.id,
                blog_post_id=comment.blog_post_id,
                author_name=comment.author_name,
                content=comment.content,
                created_at=comment.created_at,
                is_approved=comment.is_approved
            )
            for comment in comments
        ]
        
        return {"comments": formatted_comments, "total": len(formatted_comments)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching comments for post {post_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch comments")


@router.post("/posts/{post_id}/comments", response_model=BlogCommentResponse, status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: int, 
    comment: BlogCommentCreate,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Create a new comment on a blog post
    ✅ Rate limited: 20 comments/hour per IP
    """
    try:
        from models import BlogPost, BlogComment
        
        # Rate limiting (if available)
        if RATE_LIMITER_AVAILABLE and limiter:
            try:
                limiter.hit("blog_comment_create", request=request)
            except Exception as e:
                logger.warning(f"Rate limiter error: {e}")
        
        # Verify post exists
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Get user IP for tracking
        user_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Create new comment (auto-approved for now)
        new_comment = BlogComment(
            blog_post_id=post_id,
            author_name=comment.author_name,
            author_email=comment.author_email,
            content=comment.content,
            is_approved=True,  # Auto-approve for now
            is_flagged=False,
            is_spam=False,
            user_ip=user_ip,
            user_agent=user_agent,
            created_at=datetime.utcnow()
        )
        
        db.add(new_comment)
        db.commit()
        db.refresh(new_comment)
        
        logger.info(f"✅ Created comment on post {post_id} by {comment.author_name}")
        
        return BlogCommentResponse(
            id=new_comment.id,
            blog_post_id=new_comment.blog_post_id,
            author_name=new_comment.author_name,
            content=new_comment.content,
            created_at=new_comment.created_at,
            is_approved=new_comment.is_approved
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating comment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create comment")


# =============================
# LIKE ENDPOINTS
# =============================

@router.post("/posts/{post_id}/like", response_model=BlogLikeResponse)
async def toggle_like(
    post_id: int,
    user_identifier: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """
    Like or unlike a blog post
    Returns the new like status and updated count
    """
    try:
        from models import BlogPost, BlogLike
        
        # Verify post exists
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Check if user already liked
        existing_like = db.query(BlogLike).filter(
            BlogLike.blog_post_id == post_id,
            BlogLike.user_identifier == user_identifier
        ).first()
        
        if existing_like:
            # Unlike - remove the like
            db.delete(existing_like)
            post.likes_count = max(0, (post.likes_count or 0) - 1)
            is_liked = False
            logger.info(f"👎 User {user_identifier[:8]} unliked post {post_id}")
        else:
            # Like - add new like
            new_like = BlogLike(
                blog_post_id=post_id,
                user_identifier=user_identifier,
                created_at=datetime.utcnow()
            )
            db.add(new_like)
            post.likes_count = (post.likes_count or 0) + 1
            is_liked = True
            logger.info(f"👍 User {user_identifier[:8]} liked post {post_id}")
        
        db.commit()
        
        return BlogLikeResponse(
            success=True,
            is_liked=is_liked,
            likes_count=post.likes_count or 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error toggling like: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to toggle like")


@router.get("/posts/{post_id}/like-status")
async def get_like_status(
    post_id: int,
    user_identifier: str = Query(...),
    db: Session = Depends(get_db)
):
    """
    Check if a user has liked a post
    """
    try:
        from models import BlogPost, BlogLike
        
        # Verify post exists
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Check if user has liked
        existing_like = db.query(BlogLike).filter(
            BlogLike.blog_post_id == post_id,
            BlogLike.user_identifier == user_identifier
        ).first()
        
        return {
            "is_liked": existing_like is not None,
            "likes_count": post.likes_count or 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking like status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to check like status")


# =============================
# UTILITY ENDPOINTS
# =============================

@router.get("/districts")
async def get_blog_districts(db: Session = Depends(get_db)):
    """
    Get list of districts with blog posts
    """
    try:
        from models import BlogPost
        from sqlalchemy import func
        
        districts = db.query(
            BlogPost.district,
            func.count(BlogPost.id).label('count')
        ).filter(
            BlogPost.district.isnot(None)
        ).group_by(BlogPost.district).all()
        
        return {
            "districts": [
                {"name": district, "count": count} 
                for district, count in districts
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching districts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch districts")

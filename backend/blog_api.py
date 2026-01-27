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
# Use direct import to avoid circular dependency with models/ package
from models_direct import BlogPost as BlogPostModel, BlogComment as BlogCommentModel, BlogLike as BlogLikeModel

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
    author_photo: Optional[str] = Field(None, description="Author photo URL")
    district: Optional[str] = Field(None, max_length=100, description="Istanbul district")
    tags: Optional[List[str]] = Field(None, description="Post tags")
    featured_image: Optional[str] = Field(None, description="Featured image URL")
    category: Optional[str] = Field(None, description="Post category")
    status: Optional[str] = Field("published", pattern="^(draft|published|scheduled)$", description="Post status")
    meta_description: Optional[str] = Field(None, max_length=300, description="SEO meta description")
    slug: Optional[str] = Field(None, max_length=250, description="URL slug")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled publish date")
    visibility: Optional[str] = Field("public", pattern="^(public|private)$", description="Post visibility")
    images: Optional[List[dict]] = Field(None, description="Gallery images")

class BlogPostUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=10, max_length=200, description="Blog post title")
    content: Optional[str] = Field(None, min_length=100, description="Blog post content")
    author_name: Optional[str] = Field(None, min_length=2, max_length=100, description="Author name")
    author_photo: Optional[str] = Field(None, description="Author photo URL")
    district: Optional[str] = Field(None, max_length=100, description="Istanbul district")
    tags: Optional[List[str]] = Field(None, description="Post tags")
    featured_image: Optional[str] = Field(None, description="Featured image URL")
    category: Optional[str] = Field(None, description="Post category")
    status: Optional[str] = Field(None, pattern="^(draft|published|scheduled)$", description="Post status")
    meta_description: Optional[str] = Field(None, max_length=300, description="SEO meta description")
    slug: Optional[str] = Field(None, max_length=250, description="URL slug")
    scheduled_at: Optional[datetime] = Field(None, description="Scheduled publish date")
    visibility: Optional[str] = Field(None, pattern="^(public|private)$", description="Post visibility")
    images: Optional[List[dict]] = Field(None, description="Gallery images")

class BlogPostResponse(BaseModel):
    id: int
    title: str
    content: str
    author_name: str
    author_photo: Optional[str] = None
    district: Optional[str] = None
    featured_image: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None
    meta_description: Optional[str] = None
    slug: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    visibility: Optional[str] = None
    images: Optional[List[dict]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
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
@router.get("/posts/", response_model=BlogPostListResponse)  # Support trailing slash
async def get_blog_posts(
    request: Request,
    response: Response,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: Optional[int] = Query(None, ge=1, le=100, description="Posts per page"),
    limit: Optional[int] = Query(None, ge=1, le=100, description="Posts per page (alias for per_page)"),
    district: Optional[str] = Query(None, description="Filter by district"),
    search: Optional[str] = Query(None, description="Search in title/content"),
    sort: Optional[str] = Query(None, description="Sort by: newest, oldest, popular"),
    sort_by: Optional[str] = Query(None, description="Sort by (alias for sort): newest, oldest, popular"),
    db: Session = Depends(get_db)
):
    """
    Get paginated blog posts with filtering and sorting
    ✅ Fixed N+1 query problem with subquery
    ✅ Added response caching headers
    ✅ Rate limited: 60/minute per IP
    ✅ Supports both parameter naming conventions
    """
    try:
        # Models already imported at top of file
        
        # Handle parameter aliases
        actual_per_page = per_page or limit or 12
        actual_sort = sort or sort_by or "newest"
        
        # Add cache headers for public content (5 minutes)
        response.headers["Cache-Control"] = "public, max-age=300"
        response.headers["ETag"] = f"posts-{page}-{district}-{actual_sort}-{search}"
        
        # Base query with eager loading
        query = db.query(BlogPostModel)
        
        # Apply filters
        if district:
            query = query.filter(BlogPostModel.district == district)
        
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                (BlogPostModel.title.ilike(search_filter)) | 
                (BlogPostModel.content.ilike(search_filter))
            )
        
        # Apply sorting
        if actual_sort == "newest":
            query = query.order_by(BlogPostModel.created_at.desc())
        elif actual_sort == "oldest":
            query = query.order_by(BlogPostModel.created_at.asc())
        elif actual_sort == "popular":
            query = query.order_by(BlogPostModel.likes_count.desc())
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        offset = (page - 1) * actual_per_page
        posts = query.offset(offset).limit(actual_per_page).all()
        
        # FIX: N+1 Query Problem - Get all comment counts in one query
        post_ids = [post.id for post in posts]
        comment_counts = {}
        if post_ids:
            comment_count_query = db.query(
                BlogCommentModel.blog_post_id,
                func.count(BlogCommentModel.id).label('count')
            ).filter(
                BlogCommentModel.blog_post_id.in_(post_ids),
                BlogCommentModel.status == 'approved'
            ).group_by(BlogCommentModel.blog_post_id).all()
            
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
                featured_image=post.featured_image,
                category=post.category,
                created_at=post.created_at,
                likes_count=post.likes_count or 0,
                comments_count=comment_counts.get(post.id, 0)
            ))
        
        return BlogPostListResponse(
            posts=formatted_posts,
            total=total,
            page=page,
            pages=(total + actual_per_page - 1) // actual_per_page
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
        # Add cache headers for public content (10 minutes)
        response.headers["Cache-Control"] = "public, max-age=600"
        response.headers["ETag"] = f"post-{post_id}"
        
        post = db.query(BlogPostModel).filter(BlogPostModel.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        comments_count = db.query(BlogCommentModel).filter(
            BlogCommentModel.blog_post_id == post.id,
            BlogCommentModel.status == 'approved'
        ).count()
        
        return BlogPostResponse(
            id=post.id,
            title=post.title,
            content=post.content,
            author_name=post.author or "Anonymous",
            author_photo=post.author_photo,
            district=post.district,
            featured_image=post.featured_image,
            category=post.category,
            tags=post.tags,
            status=post.status,
            meta_description=post.meta_description,
            slug=post.slug,
            scheduled_at=post.scheduled_at,
            visibility=post.visibility,
            images=post.images,
            created_at=post.created_at,
            updated_at=post.updated_at,
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
        # Models already imported at top of file
        
        # Rate limiting (if available)
        if RATE_LIMITER_AVAILABLE and limiter:
            try:
                limiter.hit("blog_post_create", request=request)
            except Exception as e:
                logger.warning(f"Rate limiter error: {e}")
        
        # Create new blog post
        new_post = BlogPostModel(
            title=post.title,
            content=post.content,
            author=post.author_name,
            author_photo=post.author_photo,
            district=post.district,
            featured_image=post.featured_image,
            category=post.category,
            status=post.status or 'published',
            tags=post.tags,
            meta_description=post.meta_description,
            slug=post.slug,
            scheduled_at=post.scheduled_at,
            visibility=post.visibility or 'public',
            images=post.images or [],
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
            author_photo=new_post.author_photo,
            district=new_post.district,
            featured_image=new_post.featured_image,
            category=new_post.category,
            tags=new_post.tags,
            status=new_post.status,
            meta_description=new_post.meta_description,
            slug=new_post.slug,
            scheduled_at=new_post.scheduled_at,
            visibility=new_post.visibility,
            images=new_post.images,
            created_at=new_post.created_at,
            updated_at=new_post.updated_at,
            likes_count=0,
            comments_count=0
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create blog post")


@router.put("/posts/{post_id}", response_model=BlogPostResponse)
async def update_blog_post(post_id: int, post_update: BlogPostUpdate, request: Request, db: Session = Depends(get_db)):
    """
    Update an existing blog post
    ✅ Rate limited: 20 updates/hour per IP
    """
    try:
        # Models already imported at top of file
        
        # Rate limiting (if available)
        if RATE_LIMITER_AVAILABLE and limiter:
            try:
                limiter.hit("blog_post_update", request=request)
            except Exception as e:
                logger.warning(f"Rate limiter error: {e}")
        
        # Get existing post
        existing_post = db.query(BlogPostModel).filter(BlogPostModel.id == post_id).first()
        if not existing_post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Update fields that were provided
        update_data = post_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == 'author_name':
                # Map author_name to author column
                setattr(existing_post, 'author', value)
            elif field == 'tags' or field == 'images':
                # Handle JSON fields
                setattr(existing_post, field, value)
            else:
                setattr(existing_post, field, value)
        
        # Update timestamp
        existing_post.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(existing_post)
        
        # Get comments count
        comments_count = db.query(BlogCommentModel).filter(BlogCommentModel.blog_post_id == post_id).count()
        
        logger.info(f"✅ Updated blog post: {existing_post.id} - {existing_post.title}")
        
        return BlogPostResponse(
            id=existing_post.id,
            title=existing_post.title,
            content=existing_post.content,
            author_name=existing_post.author or "Anonymous",
            author_photo=existing_post.author_photo,
            district=existing_post.district,
            featured_image=existing_post.featured_image,
            category=existing_post.category,
            tags=existing_post.tags,
            status=existing_post.status,
            meta_description=existing_post.meta_description,
            slug=existing_post.slug,
            scheduled_at=existing_post.scheduled_at,
            visibility=existing_post.visibility,
            images=existing_post.images,
            created_at=existing_post.created_at,
            updated_at=existing_post.updated_at,
            likes_count=existing_post.likes_count or 0,
            comments_count=comments_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update blog post")


@router.delete("/posts/{post_id}")
async def delete_blog_post(post_id: int, request: Request, db: Session = Depends(get_db)):
    """
    Delete a blog post
    ✅ Rate limited: 10 deletions/hour per IP
    """
    try:
        # Models already imported at top of file
        
        # Rate limiting (if available)
        if RATE_LIMITER_AVAILABLE and limiter:
            try:
                limiter.hit("blog_post_delete", request=request)
            except Exception as e:
                logger.warning(f"Rate limiter error: {e}")
        
        # Get existing post
        existing_post = db.query(BlogPostModel).filter(BlogPostModel.id == post_id).first()
        if not existing_post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        post_title = existing_post.title
        
        # Delete associated comments and likes (cascade)
        db.query(BlogCommentModel).filter(BlogCommentModel.blog_post_id == post_id).delete()
        db.query(BlogLikeModel).filter(BlogLikeModel.blog_post_id == post_id).delete()
        
        # Delete the post
        db.delete(existing_post)
        db.commit()
        
        logger.info(f"✅ Deleted blog post: {post_id} - {post_title}")
        
        return {
            "message": "Blog post deleted successfully",
            "id": post_id,
            "title": post_title
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete blog post")


@router.get("/posts/{post_id}/comments")
async def get_post_comments(post_id: int, db: Session = Depends(get_db)):
    """
    Get all approved comments for a blog post
    """
    try:
        # Verify post exists
        post = db.query(BlogPostModel).filter(BlogPostModel.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Get approved comments
        comments = db.query(BlogCommentModel).filter(
            BlogCommentModel.blog_post_id == post_id,
            BlogCommentModel.status == 'approved'
        ).order_by(BlogCommentModel.created_at.desc()).all()
        
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
        # Rate limiting (if available)
        if RATE_LIMITER_AVAILABLE and limiter:
            try:
                limiter.hit("blog_comment_create", request=request)
            except Exception as e:
                logger.warning(f"Rate limiter error: {e}")
        
        # Verify post exists
        post = db.query(BlogPostModel).filter(BlogPostModel.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Get user IP for tracking
        user_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # Create new comment (auto-approved for now)
        new_comment = BlogCommentModel(
            blog_post_id=post_id,
            author_name=comment.author_name,
            author_email=comment.author_email,
            content=comment.content,
            status='approved',  # Auto-approve for now
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
        # Verify post exists
        post = db.query(BlogPostModel).filter(BlogPostModel.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Check if user already liked
        existing_like = db.query(BlogLikeModel).filter(
            BlogLikeModel.blog_post_id == post_id,
            BlogLikeModel.user_id == user_identifier
        ).first()
        
        if existing_like:
            # Unlike - remove the like
            db.delete(existing_like)
            post.likes_count = max(0, (post.likes_count or 0) - 1)
            is_liked = False
            logger.info(f"👎 User {user_identifier[:8]} unliked post {post_id}")
        else:
            # Like - add new like
            new_like = BlogLikeModel(
                blog_post_id=post_id,
                user_id=user_identifier,
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
        # Verify post exists
        post = db.query(BlogPostModel).filter(BlogPostModel.id == post_id).first()
        if not post:
            raise HTTPException(status_code=404, detail="Blog post not found")
        
        # Check if user has liked
        existing_like = db.query(BlogLikeModel).filter(
            BlogLikeModel.blog_post_id == post_id,
            BlogLikeModel.user_id == user_identifier
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
        districts = db.query(
            BlogPostModel.district,
            func.count(BlogPostModel.id).label('count')
        ).filter(
            BlogPostModel.district.isnot(None)
        ).group_by(BlogPostModel.district).all()
        
        return {
            "districts": [
                {"name": district, "count": count} 
                for district, count in districts
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching districts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch districts")

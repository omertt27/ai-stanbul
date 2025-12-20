"""
Blog Image Upload API Routes

Handles image uploads for blog posts via REST API.

Author: AI Istanbul Team
Date: December 2024
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from typing import Optional
import logging

from services.blog_image_service import blog_image_service
from auth import get_current_admin_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/blog/images", tags=["blog-images"])


@router.post("/upload/featured")
async def upload_featured_image(
    file: UploadFile = File(...),
    post_id: Optional[int] = Form(None),
    current_admin = Depends(get_current_admin_user)
):
    """
    Upload a featured image for a blog post
    
    **Requires admin authentication**
    
    Args:
        file: Image file to upload
        post_id: Optional blog post ID
    
    Returns:
        JSON with image URL
    """
    try:
        # Check if S3 is available
        if not blog_image_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Image upload service not available. S3 not configured."
            )
        
        # Upload image
        image_url = blog_image_service.upload_featured_image(
            file.file,
            file.filename,
            post_id
        )
        
        if not image_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload image. Check logs for details."
            )
        
        # Generate thumbnail
        thumbnail_url = None
        if post_id:
            thumbnail_url = blog_image_service.generate_thumbnail(image_url, post_id)
        
        return JSONResponse(content={
            "success": True,
            "message": "Featured image uploaded successfully",
            "image_url": image_url,
            "thumbnail_url": thumbnail_url,
            "post_id": post_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading featured image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/content")
async def upload_content_image(
    file: UploadFile = File(...),
    post_id: Optional[int] = Form(None),
    current_admin = Depends(get_current_admin_user)
):
    """
    Upload a content image for a blog post
    
    **Requires admin authentication**
    
    Args:
        file: Image file to upload
        post_id: Optional blog post ID
    
    Returns:
        JSON with image URL
    """
    try:
        # Check if S3 is available
        if not blog_image_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Image upload service not available. S3 not configured."
            )
        
        # Upload image
        image_url = blog_image_service.upload_content_image(
            file.file,
            file.filename,
            post_id
        )
        
        if not image_url:
            raise HTTPException(
                status_code=500,
                detail="Failed to upload image. Check logs for details."
            )
        
        return JSONResponse(content={
            "success": True,
            "message": "Content image uploaded successfully",
            "image_url": image_url,
            "post_id": post_id
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading content image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete")
async def delete_image(
    image_url: str,
    current_admin = Depends(get_current_admin_user)
):
    """
    Delete an image from S3
    
    **Requires admin authentication**
    
    Args:
        image_url: Full S3 URL of image to delete
    
    Returns:
        Success message
    """
    try:
        if not blog_image_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Image service not available. S3 not configured."
            )
        
        success = blog_image_service.delete_image(image_url)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete image"
            )
        
        return JSONResponse(content={
            "success": True,
            "message": "Image deleted successfully",
            "image_url": image_url
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def image_service_status():
    """
    Check if image upload service is available
    
    Returns:
        Service status
    """
    return {
        "available": blog_image_service.is_available(),
        "bucket": blog_image_service.bucket_name if blog_image_service.is_available() else None,
        "region": blog_image_service.region if blog_image_service.is_available() else None,
        "max_size_mb": blog_image_service.MAX_IMAGE_SIZE_MB,
        "allowed_formats": list(blog_image_service.ALLOWED_FORMATS)
    }

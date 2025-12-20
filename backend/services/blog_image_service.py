"""
Blog Image Upload Service

Handles image uploads for blog posts to AWS S3.
Supports featured images, content images, and thumbnails.

Author: AI Istanbul Team
Date: December 2024
"""

import os
import uuid
import logging
from typing import Optional, BinaryIO, Tuple
from datetime import datetime
from PIL import Image
from io import BytesIO
import boto3
from botocore.exceptions import ClientError

from config.settings import settings

logger = logging.getLogger(__name__)


class BlogImageService:
    """Service for handling blog image uploads to S3"""
    
    # Image size constraints
    MAX_IMAGE_SIZE_MB = 10
    MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
    
    # Thumbnail sizes
    THUMBNAIL_SIZE = (300, 200)
    FEATURED_IMAGE_SIZE = (1200, 630)  # Optimal for social media
    
    # Allowed formats
    ALLOWED_FORMATS = {'jpg', 'jpeg', 'png', 'webp', 'gif'}
    
    def __init__(self):
        """Initialize S3 client for blog images"""
        self.s3_client = None
        self.bucket_name = settings.AWS_S3_BUCKET
        self.region = settings.AWS_REGION
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize boto3 S3 client"""
        try:
            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                self.s3_client = boto3.client(
                    's3',
                    region_name=self.region,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
                )
                logger.info(f"✅ Blog Image S3 client initialized (bucket: {self.bucket_name})")
            else:
                logger.warning("⚠️ AWS S3 credentials not configured. Image uploads disabled.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {e}")
    
    def is_available(self) -> bool:
        """Check if S3 is available for uploads"""
        return self.s3_client is not None
    
    def validate_image(self, file_obj: BinaryIO, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate image file
        
        Args:
            file_obj: File object
            filename: Original filename
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file extension
        file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if file_ext not in self.ALLOWED_FORMATS:
            return False, f"Invalid format. Allowed: {', '.join(self.ALLOWED_FORMATS)}"
        
        # Check file size
        file_obj.seek(0, 2)  # Seek to end
        file_size = file_obj.tell()
        file_obj.seek(0)  # Reset to beginning
        
        if file_size > self.MAX_IMAGE_SIZE_BYTES:
            return False, f"File too large. Max size: {self.MAX_IMAGE_SIZE_MB}MB"
        
        # Try to open as image
        try:
            img = Image.open(file_obj)
            img.verify()
            file_obj.seek(0)  # Reset after verify
            return True, None
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def generate_s3_key(self, image_type: str, post_id: Optional[int] = None) -> str:
        """
        Generate unique S3 key for image
        
        Args:
            image_type: 'featured', 'content', or 'thumbnail'
            post_id: Optional blog post ID
        
        Returns:
            S3 key path
        """
        timestamp = datetime.utcnow().strftime('%Y/%m')
        unique_id = uuid.uuid4().hex[:8]
        
        if image_type == 'featured':
            base_path = settings.AWS_S3_FEATURED_PATH
        elif image_type == 'thumbnail':
            base_path = settings.AWS_S3_THUMBNAILS_PATH
        else:
            base_path = settings.AWS_S3_CONTENT_PATH
        
        if post_id:
            return f"{base_path}{timestamp}/post-{post_id}-{unique_id}.jpg"
        else:
            return f"{base_path}{timestamp}/{unique_id}.jpg"
    
    def upload_featured_image(
        self,
        file_obj: BinaryIO,
        filename: str,
        post_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Upload featured image for blog post
        
        Args:
            file_obj: File object
            filename: Original filename
            post_id: Blog post ID
        
        Returns:
            S3 URL or None if failed
        """
        if not self.is_available():
            logger.error("S3 not available for image upload")
            return None
        
        # Validate image
        is_valid, error_msg = self.validate_image(file_obj, filename)
        if not is_valid:
            logger.error(f"Image validation failed: {error_msg}")
            return None
        
        try:
            # Open and resize image
            img = Image.open(file_obj)
            
            # Convert to RGB if necessary (for PNG with transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Resize to featured image size while maintaining aspect ratio
            img.thumbnail(self.FEATURED_IMAGE_SIZE, Image.Resampling.LANCZOS)
            
            # Save to BytesIO
            output = BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            output.seek(0)
            
            # Generate S3 key
            s3_key = self.generate_s3_key('featured', post_id)
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                output,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/jpeg',
                    'CacheControl': 'max-age=31536000',  # 1 year
                    'Metadata': {
                        'original-filename': filename,
                        'image-type': 'featured',
                        'post-id': str(post_id) if post_id else 'unknown'
                    }
                }
            )
            
            # Generate URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Featured image uploaded: {url}")
            return url
            
        except Exception as e:
            logger.error(f"❌ Error uploading featured image: {e}")
            return None
    
    def upload_content_image(
        self,
        file_obj: BinaryIO,
        filename: str,
        post_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Upload content image for blog post
        
        Args:
            file_obj: File object
            filename: Original filename
            post_id: Blog post ID
        
        Returns:
            S3 URL or None if failed
        """
        if not self.is_available():
            return None
        
        # Validate image
        is_valid, error_msg = self.validate_image(file_obj, filename)
        if not is_valid:
            logger.error(f"Image validation failed: {error_msg}")
            return None
        
        try:
            # Open image
            img = Image.open(file_obj)
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Save to BytesIO (no resizing for content images, preserve quality)
            output = BytesIO()
            img.save(output, format='JPEG', quality=90, optimize=True)
            output.seek(0)
            
            # Generate S3 key
            s3_key = self.generate_s3_key('content', post_id)
            
            # Upload to S3
            self.s3_client.upload_fileobj(
                output,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'image/jpeg',
                    'CacheControl': 'max-age=31536000',
                    'Metadata': {
                        'original-filename': filename,
                        'image-type': 'content',
                        'post-id': str(post_id) if post_id else 'unknown'
                    }
                }
            )
            
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            logger.info(f"✅ Content image uploaded: {url}")
            return url
            
        except Exception as e:
            logger.error(f"❌ Error uploading content image: {e}")
            return None
    
    def generate_thumbnail(
        self,
        source_url: str,
        post_id: int
    ) -> Optional[str]:
        """
        Generate thumbnail from featured image
        
        Args:
            source_url: URL of source image
            post_id: Blog post ID
        
        Returns:
            Thumbnail S3 URL or None
        """
        if not self.is_available():
            return None
        
        try:
            # Download source image from S3
            # Extract S3 key from URL
            s3_key = source_url.split(f"{self.bucket_name}.s3.{self.region}.amazonaws.com/")[-1]
            
            # Download image
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            img_data = response['Body'].read()
            
            # Open and create thumbnail
            img = Image.open(BytesIO(img_data))
            img.thumbnail(self.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            
            # Save to BytesIO
            output = BytesIO()
            img.save(output, format='JPEG', quality=80, optimize=True)
            output.seek(0)
            
            # Generate thumbnail S3 key
            thumb_s3_key = self.generate_s3_key('thumbnail', post_id)
            
            # Upload thumbnail
            self.s3_client.upload_fileobj(
                output,
                self.bucket_name,
                thumb_s3_key,
                ExtraArgs={
                    'ContentType': 'image/jpeg',
                    'CacheControl': 'max-age=31536000',
                    'Metadata': {
                        'image-type': 'thumbnail',
                        'post-id': str(post_id),
                        'source-image': s3_key
                    }
                }
            )
            
            thumb_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{thumb_s3_key}"
            logger.info(f"✅ Thumbnail generated: {thumb_url}")
            return thumb_url
            
        except Exception as e:
            logger.error(f"❌ Error generating thumbnail: {e}")
            return None
    
    def delete_image(self, image_url: str) -> bool:
        """
        Delete image from S3
        
        Args:
            image_url: Full S3 URL of image
        
        Returns:
            True if deleted, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            # Extract S3 key from URL
            s3_key = image_url.split(f"{self.bucket_name}.s3.{self.region}.amazonaws.com/")[-1]
            
            # Delete from S3
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"✅ Image deleted: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting image: {e}")
            return False


# Singleton instance
blog_image_service = BlogImageService()

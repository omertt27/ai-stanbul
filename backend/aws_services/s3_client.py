"""
AWS S3 Client
Handles file uploads, storage, and retrieval from S3
"""

import os
import logging
from typing import Optional, BinaryIO, Dict, Any
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class S3Client:
    """Client for interacting with AWS S3"""
    
    def __init__(self):
        """Initialize S3 client"""
        self.aws_region = os.getenv('AWS_REGION', 'eu-central-1')
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Bucket names
        self.uploads_bucket = os.getenv('AWS_S3_UPLOADS_BUCKET', 'ai-istanbul-uploads-prod')
        self.backups_bucket = os.getenv('AWS_S3_BACKUPS_BUCKET', 'ai-istanbul-backups-prod')
        self.assets_bucket = os.getenv('AWS_S3_ASSETS_BUCKET', 'ai-istanbul-assets-prod')
        
        # Initialize client
        self.s3_client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize boto3 S3 client"""
        try:
            if self.aws_access_key and self.aws_secret_key:
                self.s3_client = boto3.client(
                    's3',
                    region_name=self.aws_region,
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key
                )
                logger.info(f"✅ S3 client initialized (region: {self.aws_region})")
            else:
                logger.warning("⚠️ AWS credentials not configured for S3")
        except Exception as e:
            logger.error(f"❌ Failed to initialize S3 client: {e}")
    
    def is_available(self) -> bool:
        """Check if S3 is available"""
        return self.s3_client is not None
    
    # ===== File Upload =====
    
    async def upload_file(
        self,
        file_obj: BinaryIO,
        file_name: str,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Upload file to S3
        
        Args:
            file_obj: File object to upload
            file_name: Name of the file
            bucket: Bucket name (defaults to uploads_bucket)
            content_type: MIME type of the file
            metadata: Additional metadata to store with file
        
        Returns:
            S3 URL of uploaded file, or None if failed
        """
        if not self.is_available():
            return None
        
        bucket = bucket or self.uploads_bucket
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Generate unique key with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            s3_key = f"uploads/{timestamp}_{file_name}"
            
            # Upload file
            self.s3_client.upload_fileobj(
                file_obj,
                bucket,
                s3_key,
                ExtraArgs=extra_args
            )
            
            # Generate URL
            url = f"https://{bucket}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            logger.info(f"File uploaded to S3: {url}")
            return url
            
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            return None
    
    async def upload_from_path(
        self,
        file_path: str,
        s3_key: Optional[str] = None,
        bucket: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload file from local path to S3
        
        Args:
            file_path: Local file path
            s3_key: S3 key (defaults to filename)
            bucket: Bucket name (defaults to uploads_bucket)
            content_type: MIME type
        
        Returns:
            S3 URL of uploaded file
        """
        if not self.is_available():
            return None
        
        bucket = bucket or self.uploads_bucket
        s3_key = s3_key or os.path.basename(file_path)
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self.s3_client.upload_file(
                file_path,
                bucket,
                s3_key,
                ExtraArgs=extra_args
            )
            
            url = f"https://{bucket}.s3.{self.aws_region}.amazonaws.com/{s3_key}"
            logger.info(f"File uploaded to S3: {url}")
            return url
            
        except ClientError as e:
            logger.error(f"Error uploading file to S3: {e}")
            return None
    
    # ===== File Download =====
    
    async def download_file(
        self,
        s3_key: str,
        local_path: str,
        bucket: Optional[str] = None
    ) -> bool:
        """
        Download file from S3 to local path
        
        Args:
            s3_key: S3 object key
            local_path: Local file path to save to
            bucket: Bucket name (defaults to uploads_bucket)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        bucket = bucket or self.uploads_bucket
        
        try:
            self.s3_client.download_file(bucket, s3_key, local_path)
            logger.info(f"File downloaded from S3: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error downloading file from S3: {e}")
            return False
    
    async def get_file_content(
        self,
        s3_key: str,
        bucket: Optional[str] = None
    ) -> Optional[bytes]:
        """
        Get file content as bytes
        
        Args:
            s3_key: S3 object key
            bucket: Bucket name
        
        Returns:
            File content as bytes, or None if failed
        """
        if not self.is_available():
            return None
        
        bucket = bucket or self.uploads_bucket
        
        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=s3_key)
            content = response['Body'].read()
            return content
            
        except ClientError as e:
            logger.error(f"Error getting file content from S3: {e}")
            return None
    
    # ===== File Management =====
    
    async def delete_file(self, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Delete file from S3"""
        if not self.is_available():
            return False
        
        bucket = bucket or self.uploads_bucket
        
        try:
            self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
            logger.info(f"File deleted from S3: {s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting file from S3: {e}")
            return False
    
    async def list_files(
        self,
        prefix: str = "",
        bucket: Optional[str] = None,
        max_keys: int = 1000
    ) -> list:
        """
        List files in S3 bucket
        
        Args:
            prefix: Key prefix to filter by
            bucket: Bucket name
            max_keys: Maximum number of keys to return
        
        Returns:
            List of file keys
        """
        if not self.is_available():
            return []
        
        bucket = bucket or self.uploads_bucket
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            files = []
            for obj in response.get('Contents', []):
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
            
            return files
            
        except ClientError as e:
            logger.error(f"Error listing files in S3: {e}")
            return []
    
    async def file_exists(self, s3_key: str, bucket: Optional[str] = None) -> bool:
        """Check if file exists in S3"""
        if not self.is_available():
            return False
        
        bucket = bucket or self.uploads_bucket
        
        try:
            self.s3_client.head_object(Bucket=bucket, Key=s3_key)
            return True
        except ClientError:
            return False
    
    # ===== Presigned URLs =====
    
    async def generate_presigned_url(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate presigned URL for temporary access
        
        Args:
            s3_key: S3 object key
            bucket: Bucket name
            expiration: URL expiration in seconds (default: 1 hour)
        
        Returns:
            Presigned URL, or None if failed
        """
        if not self.is_available():
            return None
        
        bucket = bucket or self.uploads_bucket
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
            
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None
    
    async def generate_upload_presigned_url(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
        expiration: int = 3600,
        content_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate presigned URL for direct upload
        
        Args:
            s3_key: S3 object key
            bucket: Bucket name
            expiration: URL expiration in seconds
            content_type: MIME type
        
        Returns:
            Dict with 'url' and 'fields' for POST upload
        """
        if not self.is_available():
            return None
        
        bucket = bucket or self.uploads_bucket
        
        try:
            conditions = []
            if content_type:
                conditions.append({'Content-Type': content_type})
            
            response = self.s3_client.generate_presigned_post(
                Bucket=bucket,
                Key=s3_key,
                ExpiresIn=expiration,
                Conditions=conditions if conditions else None
            )
            
            return response
            
        except ClientError as e:
            logger.error(f"Error generating presigned POST: {e}")
            return None
    
    # ===== Backup Operations =====
    
    async def backup_file(
        self,
        source_key: str,
        source_bucket: Optional[str] = None
    ) -> bool:
        """
        Copy file to backup bucket
        
        Args:
            source_key: Source S3 key
            source_bucket: Source bucket (defaults to uploads_bucket)
        
        Returns:
            True if successful
        """
        if not self.is_available():
            return False
        
        source_bucket = source_bucket or self.uploads_bucket
        
        try:
            # Generate backup key with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            backup_key = f"backups/{timestamp}_{source_key}"
            
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.backups_bucket,
                Key=backup_key
            )
            
            logger.info(f"File backed up to S3: {backup_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error backing up file to S3: {e}")
            return False


# Singleton instance
_s3_client: Optional[S3Client] = None


def get_s3_client() -> S3Client:
    """Get or create S3 client singleton"""
    global _s3_client
    if _s3_client is None:
        _s3_client = S3Client()
    return _s3_client

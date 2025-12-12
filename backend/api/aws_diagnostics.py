"""
AWS Integration Test Endpoints
Test S3 and Redis connectivity from Cloud Run
"""

import os
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/test", tags=["test"])


@router.get("/s3")
async def test_s3_connection() -> Dict[str, Any]:
    """
    Test AWS S3 connection and configuration
    
    Returns:
        Connection status, region, and bucket configuration
    """
    try:
        from aws_services import get_s3_client
        
        s3_client = get_s3_client()
        
        if not s3_client.is_available():
            return {
                "status": "not_configured",
                "message": "AWS credentials not set",
                "required_env_vars": [
                    "AWS_REGION",
                    "AWS_ACCESS_KEY_ID",
                    "AWS_SECRET_ACCESS_KEY"
                ]
            }
        
        # Try to list buckets (test permissions)
        try:
            import boto3
            s3 = boto3.client(
                's3',
                region_name=s3_client.aws_region,
                aws_access_key_id=s3_client.aws_access_key,
                aws_secret_access_key=s3_client.aws_secret_key
            )
            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            
            return {
                "status": "connected",
                "region": s3_client.aws_region,
                "configured_buckets": {
                    "uploads": s3_client.uploads_bucket,
                    "backups": s3_client.backups_bucket,
                    "assets": s3_client.assets_bucket
                },
                "available_buckets": bucket_names,
                "message": "✅ S3 connection successful"
            }
        except Exception as e:
            logger.error(f"S3 permission error: {e}")
            return {
                "status": "permission_error",
                "region": s3_client.aws_region,
                "configured_buckets": {
                    "uploads": s3_client.uploads_bucket,
                    "backups": s3_client.backups_bucket,
                    "assets": s3_client.assets_bucket
                },
                "error": str(e),
                "message": "⚠️ S3 configured but cannot list buckets (check IAM permissions)"
            }
            
    except ImportError:
        return {
            "status": "module_not_found",
            "message": "AWS services module not available",
            "hint": "Ensure boto3 is installed: pip install boto3"
        }
    except Exception as e:
        logger.error(f"S3 test error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "❌ S3 test failed"
        }


@router.get("/redis")
async def test_redis_connection() -> Dict[str, Any]:
    """
    Test Redis connection
    
    Returns:
        Connection status and basic operations test
    """
    try:
        import redis
        from datetime import datetime
        
        redis_url = os.getenv('REDIS_URL')
        
        if not redis_url:
            return {
                "status": "not_configured",
                "message": "REDIS_URL environment variable not set",
                "example": "redis://default:password@host:port"
            }
        
        # Mask password in URL for display
        display_url = redis_url
        if '@' in redis_url:
            parts = redis_url.split('@')
            if len(parts) == 2:
                display_url = f"redis://***@{parts[1]}"
        
        try:
            # Connect to Redis
            r = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5)
            
            # Test ping
            r.ping()
            
            # Test set/get
            test_key = f'test_connection_{datetime.utcnow().isoformat()}'
            test_value = f'test_value_{datetime.utcnow().timestamp()}'
            
            r.set(test_key, test_value, ex=60)  # Expire in 60 seconds
            retrieved_value = r.get(test_key)
            
            # Get Redis info
            info = r.info('server')
            
            return {
                "status": "connected",
                "redis_url": display_url,
                "redis_version": info.get('redis_version', 'unknown'),
                "test_operations": {
                    "ping": "success",
                    "set_key": test_key,
                    "set_value": test_value,
                    "retrieved_value": retrieved_value,
                    "match": retrieved_value == test_value
                },
                "message": "✅ Redis connection successful"
            }
            
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            return {
                "status": "connection_error",
                "redis_url": display_url,
                "error": str(e),
                "message": "❌ Cannot connect to Redis",
                "troubleshooting": [
                    "Check if Redis URL is correct",
                    "Verify Redis server is running",
                    "Check firewall rules (Redis Cloud should allow all IPs)",
                    "Verify password is correct"
                ]
            }
        except redis.AuthenticationError as e:
            logger.error(f"Redis authentication error: {e}")
            return {
                "status": "authentication_error",
                "redis_url": display_url,
                "error": str(e),
                "message": "❌ Redis authentication failed",
                "troubleshooting": [
                    "Check Redis password in REDIS_URL",
                    "Verify ACL rules (if using Redis 6+)"
                ]
            }
        except Exception as e:
            logger.error(f"Redis error: {e}")
            return {
                "status": "error",
                "redis_url": display_url,
                "error": str(e),
                "message": "❌ Redis test failed"
            }
            
    except ImportError:
        return {
            "status": "module_not_found",
            "message": "Redis module not available",
            "hint": "Ensure redis is installed: pip install redis"
        }
    except Exception as e:
        logger.error(f"Redis test error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "❌ Redis test failed"
        }


@router.get("/aws-integration")
async def test_aws_integration() -> Dict[str, Any]:
    """
    Test complete AWS integration (S3 + Redis)
    
    Returns:
        Combined status of all AWS services
    """
    try:
        # Test S3
        s3_result = await test_s3_connection()
        
        # Test Redis
        redis_result = await test_redis_connection()
        
        # Overall status
        all_good = (
            s3_result.get('status') == 'connected' and 
            redis_result.get('status') == 'connected'
        )
        
        return {
            "overall_status": "healthy" if all_good else "degraded",
            "s3": s3_result,
            "redis": redis_result,
            "cross_cloud_setup": {
                "gcp_backend": "europe-west1 (Belgium)",
                "aws_s3": s3_result.get('region', 'not_configured'),
                "redis": "configured" if redis_result.get('status') != 'not_configured' else "not_configured"
            },
            "message": "✅ All services healthy" if all_good else "⚠️ Some services need attention"
        }
        
    except Exception as e:
        logger.error(f"Integration test error: {e}")
        return {
            "overall_status": "error",
            "error": str(e),
            "message": "❌ Integration test failed"
        }


@router.get("/env-vars")
async def test_env_vars() -> Dict[str, Any]:
    """
    Test environment variables configuration (sensitive values masked)
    
    Returns:
        Environment variable status
    """
    try:
        env_vars = {
            "AWS_REGION": os.getenv('AWS_REGION'),
            "AWS_ACCESS_KEY_ID": "***" + os.getenv('AWS_ACCESS_KEY_ID', '')[-4:] if os.getenv('AWS_ACCESS_KEY_ID') else None,
            "AWS_SECRET_ACCESS_KEY": "***" if os.getenv('AWS_SECRET_ACCESS_KEY') else None,
            "AWS_S3_UPLOADS_BUCKET": os.getenv('AWS_S3_UPLOADS_BUCKET'),
            "AWS_S3_BACKUPS_BUCKET": os.getenv('AWS_S3_BACKUPS_BUCKET'),
            "AWS_S3_ASSETS_BUCKET": os.getenv('AWS_S3_ASSETS_BUCKET'),
            "REDIS_URL": "configured" if os.getenv('REDIS_URL') else None,
            "DATABASE_URL": "configured" if os.getenv('DATABASE_URL') else None,
            "LLM_API_URL": os.getenv('LLM_API_URL'),
            "AI_ISTANBUL_LLM_MODE": os.getenv('AI_ISTANBUL_LLM_MODE'),
        }
        
        # Count configured vs not configured
        configured = sum(1 for v in env_vars.values() if v is not None)
        total = len(env_vars)
        
        return {
            "status": "ok",
            "configured": f"{configured}/{total}",
            "environment_variables": env_vars,
            "message": f"✅ {configured} of {total} environment variables configured"
        }
        
    except Exception as e:
        logger.error(f"Env vars test error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "❌ Environment variables test failed"
        }

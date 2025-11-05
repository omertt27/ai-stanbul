"""
A/B Testing API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging

from backend.services.recommendation_ab_testing import (
    get_recommendation_ab_testing,
    ABTestConfig,
    ABVariant
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ab-testing", tags=["ab-testing"])


class CreateTestRequest(BaseModel):
    """Request to create a new A/B test"""
    test_id: str
    control_percentage: float  # 0.0 to 1.0
    variant_a_percentage: Optional[float] = None
    variant_b_percentage: Optional[float] = None
    enabled: bool = True


@router.get("/tests")
async def get_all_tests():
    """Get all registered A/B tests"""
    try:
        framework = get_recommendation_ab_testing()
        tests = framework.get_all_tests()
        
        return {
            'tests': tests,
            'count': len(tests)
        }
    except Exception as e:
        logger.error(f"❌ Failed to get tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tests")
async def create_test(request: CreateTestRequest):
    """Create a new A/B test"""
    try:
        # Build variants dict
        variants = {ABVariant.CONTROL: request.control_percentage}
        
        if request.variant_a_percentage:
            variants[ABVariant.VARIANT_A] = request.variant_a_percentage
        
        if request.variant_b_percentage:
            variants[ABVariant.VARIANT_B] = request.variant_b_percentage
        
        # Validate percentages sum to 1.0
        total = sum(variants.values())
        if abs(total - 1.0) > 0.001:
            raise HTTPException(
                status_code=400,
                detail=f"Variant percentages must sum to 1.0, got {total}"
            )
        
        # Create test config
        config = ABTestConfig(
            test_id=request.test_id,
            variants=variants,
            enabled=request.enabled
        )
        
        # Register test
        framework = get_recommendation_ab_testing()
        framework.register_test(config)
        
        return {
            'status': 'success',
            'message': f'Created A/B test: {request.test_id}',
            'config': config.to_dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Failed to create test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tests/{test_id}/variant")
async def get_user_variant(test_id: str, user_id: str):
    """Get the assigned variant for a user"""
    try:
        framework = get_recommendation_ab_testing()
        variant = framework.get_variant(test_id, user_id)
        
        return {
            'test_id': test_id,
            'user_id': user_id,
            'variant': variant.value
        }
    except Exception as e:
        logger.error(f"❌ Failed to get variant: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tests/{test_id}/results")
async def get_test_results(test_id: str, hours: int = 24):
    """Get results for an A/B test"""
    try:
        framework = get_recommendation_ab_testing()
        results = framework.get_test_results(test_id, hours)
        
        if 'error' in results:
            raise HTTPException(status_code=404, detail=results['error'])
        
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tests/{test_id}/enable")
async def enable_test(test_id: str):
    """Enable an A/B test"""
    try:
        framework = get_recommendation_ab_testing()
        framework.enable_test(test_id)
        
        return {
            'status': 'success',
            'message': f'Enabled test: {test_id}'
        }
    except Exception as e:
        logger.error(f"❌ Failed to enable test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tests/{test_id}/disable")
async def disable_test(test_id: str):
    """Disable an A/B test"""
    try:
        framework = get_recommendation_ab_testing()
        framework.disable_test(test_id)
        
        return {
            'status': 'success',
            'message': f'Disabled test: {test_id}'
        }
    except Exception as e:
        logger.error(f"❌ Failed to disable test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Example: Track exposure (called when showing recommendations)
@router.post("/tests/{test_id}/exposure")
async def track_exposure(test_id: str, user_id: str, session_id: Optional[str] = None):
    """Track that a user was exposed to a test variant"""
    try:
        framework = get_recommendation_ab_testing()
        variant = framework.get_variant(test_id, user_id)
        framework.track_exposure(test_id, user_id, variant, session_id)
        
        return {
            'test_id': test_id,
            'user_id': user_id,
            'variant': variant.value,
            'tracked': True
        }
    except Exception as e:
        logger.error(f"❌ Failed to track exposure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

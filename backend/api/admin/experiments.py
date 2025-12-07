"""
experiments.py - A/B Testing and Feature Flags API Endpoints

Admin API for managing experiments, feature flags, and analytics.

Author: AI Istanbul Team
Date: December 7, 2025
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Import A/B testing and feature flag managers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.llm.ab_testing import ExperimentManager, ExperimentConfig, ExperimentStatus
from services.llm.feature_flags import FeatureFlagManager, FeatureFlag

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/experiments", tags=["experiments"])

# Global managers (initialized at startup)
experiment_manager: Optional[ExperimentManager] = None
feature_flag_manager: Optional[FeatureFlagManager] = None


# ===== MODELS =====

class CreateExperimentRequest(BaseModel):
    """Request to create a new experiment."""
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    start_date: str = Field(..., description="Start date (ISO format)")
    end_date: str = Field(..., description="End date (ISO format)")
    variants: Dict[str, Dict[str, Any]] = Field(..., description="Variants configuration")
    metrics: List[str] = Field(..., description="Metrics to track")
    minimum_sample_size: int = Field(100, description="Minimum sample size")


class RecordMetricRequest(BaseModel):
    """Request to record a metric."""
    experiment_id: str = Field(..., description="Experiment ID")
    user_id: str = Field(..., description="User ID")
    metric_name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")


class CreateFeatureFlagRequest(BaseModel):
    """Request to create a feature flag."""
    name: str = Field(..., description="Feature flag name")
    enabled: bool = Field(True, description="Whether flag is enabled")
    rollout_percentage: int = Field(100, description="Rollout percentage (0-100)")
    description: str = Field("", description="Flag description")
    whitelist: Optional[List[str]] = Field(None, description="User whitelist")
    blacklist: Optional[List[str]] = Field(None, description="User blacklist")
    rules: Optional[List[Dict[str, Any]]] = Field(None, description="Context-based rules")


class UpdateFeatureFlagRequest(BaseModel):
    """Request to update a feature flag."""
    enabled: Optional[bool] = None
    rollout_percentage: Optional[int] = None
    description: Optional[str] = None
    whitelist: Optional[List[str]] = None
    blacklist: Optional[List[str]] = None
    rules: Optional[List[Dict[str, Any]]] = None


class EvaluateFlagRequest(BaseModel):
    """Request to evaluate a feature flag."""
    flag_name: str = Field(..., description="Feature flag name")
    user_id: str = Field(..., description="User ID")
    context: Optional[Dict[str, Any]] = Field(None, description="Evaluation context")


# ===== INITIALIZATION =====

def initialize_managers():
    """Initialize experiment and feature flag managers."""
    global experiment_manager, feature_flag_manager
    
    if experiment_manager is None:
        experiment_manager = ExperimentManager()
        logger.info("✅ Experiment Manager initialized")
    
    if feature_flag_manager is None:
        feature_flag_manager = FeatureFlagManager()
        logger.info("✅ Feature Flag Manager initialized")


# ===== EXPERIMENT ENDPOINTS =====

@router.post("/experiments", response_model=Dict[str, Any])
async def create_experiment(request: CreateExperimentRequest):
    """Create a new A/B test experiment."""
    initialize_managers()
    
    try:
        config = ExperimentConfig(
            id=f"exp_{int(datetime.now().timestamp())}",
            name=request.name,
            description=request.description,
            start_date=request.start_date,
            end_date=request.end_date,
            variants=request.variants,
            metrics=request.metrics,
            minimum_sample_size=request.minimum_sample_size,
            status=ExperimentStatus.DRAFT.value
        )
        
        result = experiment_manager.create_experiment(config)
        return {
            "success": True,
            "experiment": result
        }
    except Exception as e:
        logger.error(f"Error creating experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments", response_model=Dict[str, Any])
async def list_experiments(
    status: Optional[str] = Query(None, description="Filter by status")
):
    """List all experiments."""
    initialize_managers()
    
    try:
        experiments = experiment_manager.list_experiments(status_filter=status)
        return {
            "success": True,
            "experiments": experiments,
            "total": len(experiments)
        }
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}", response_model=Dict[str, Any])
async def get_experiment(experiment_id: str):
    """Get experiment details and results."""
    initialize_managers()
    
    try:
        experiment = experiment_manager.get_experiment(experiment_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        results = experiment_manager.get_experiment_results(experiment_id)
        
        return {
            "success": True,
            "experiment": experiment,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/start", response_model=Dict[str, Any])
async def start_experiment(experiment_id: str):
    """Start an experiment."""
    initialize_managers()
    
    try:
        result = experiment_manager.start_experiment(experiment_id)
        return {
            "success": True,
            "experiment": result
        }
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/stop", response_model=Dict[str, Any])
async def stop_experiment(experiment_id: str):
    """Stop an experiment."""
    initialize_managers()
    
    try:
        result = experiment_manager.stop_experiment(experiment_id)
        return {
            "success": True,
            "experiment": result
        }
    except Exception as e:
        logger.error(f"Error stopping experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/experiments/{experiment_id}", response_model=Dict[str, Any])
async def delete_experiment(experiment_id: str):
    """Delete an experiment."""
    initialize_managers()
    
    try:
        experiment_manager.delete_experiment(experiment_id)
        return {
            "success": True,
            "message": f"Experiment {experiment_id} deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/{experiment_id}/metrics", response_model=Dict[str, Any])
async def record_metric(experiment_id: str, request: RecordMetricRequest):
    """Record a metric for an experiment."""
    initialize_managers()
    
    try:
        experiment_manager.record_metric(
            experiment_id=experiment_id,
            user_id=request.user_id,
            metric_name=request.metric_name,
            value=request.value
        )
        return {
            "success": True,
            "message": "Metric recorded"
        }
    except Exception as e:
        logger.error(f"Error recording metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/variant/{user_id}", response_model=Dict[str, Any])
async def get_user_variant(experiment_id: str, user_id: str):
    """Get assigned variant for a user."""
    initialize_managers()
    
    try:
        variant = experiment_manager.get_variant(experiment_id, user_id)
        return {
            "success": True,
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant": variant
        }
    except Exception as e:
        logger.error(f"Error getting variant: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== FEATURE FLAG ENDPOINTS =====

@router.post("/flags", response_model=Dict[str, Any])
async def create_feature_flag(request: CreateFeatureFlagRequest):
    """Create a new feature flag."""
    initialize_managers()
    
    try:
        flag = FeatureFlag(
            name=request.name,
            enabled=request.enabled,
            rollout_percentage=request.rollout_percentage,
            description=request.description,
            whitelist=request.whitelist,
            blacklist=request.blacklist,
            rules=request.rules
        )
        
        result = feature_flag_manager.create_flag(flag)
        return {
            "success": True,
            "flag": result
        }
    except Exception as e:
        logger.error(f"Error creating flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flags", response_model=Dict[str, Any])
async def list_feature_flags():
    """List all feature flags."""
    initialize_managers()
    
    try:
        flags = feature_flag_manager.list_flags()
        return {
            "success": True,
            "flags": flags,
            "total": len(flags)
        }
    except Exception as e:
        logger.error(f"Error listing flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flags/{flag_name}", response_model=Dict[str, Any])
async def get_feature_flag(flag_name: str):
    """Get feature flag details."""
    initialize_managers()
    
    try:
        flag = feature_flag_manager.get_flag(flag_name)
        if not flag:
            raise HTTPException(status_code=404, detail="Flag not found")
        
        return {
            "success": True,
            "flag": flag
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/flags/{flag_name}", response_model=Dict[str, Any])
async def update_feature_flag(flag_name: str, request: UpdateFeatureFlagRequest):
    """Update a feature flag."""
    initialize_managers()
    
    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}
        result = feature_flag_manager.update_flag(flag_name, updates)
        
        return {
            "success": True,
            "flag": result
        }
    except Exception as e:
        logger.error(f"Error updating flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/flags/{flag_name}", response_model=Dict[str, Any])
async def delete_feature_flag(flag_name: str):
    """Delete a feature flag."""
    initialize_managers()
    
    try:
        feature_flag_manager.delete_flag(flag_name)
        return {
            "success": True,
            "message": f"Flag '{flag_name}' deleted"
        }
    except Exception as e:
        logger.error(f"Error deleting flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flags/evaluate", response_model=Dict[str, Any])
async def evaluate_feature_flag(request: EvaluateFlagRequest):
    """Evaluate a feature flag for a user."""
    initialize_managers()
    
    try:
        enabled = feature_flag_manager.is_enabled(
            flag_name=request.flag_name,
            user_id=request.user_id,
            context=request.context
        )
        
        return {
            "success": True,
            "flag_name": request.flag_name,
            "user_id": request.user_id,
            "enabled": enabled
        }
    except Exception as e:
        logger.error(f"Error evaluating flag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== ANALYTICS ENDPOINTS =====

@router.get("/analytics/overview", response_model=Dict[str, Any])
async def get_analytics_overview():
    """Get overall analytics overview."""
    initialize_managers()
    
    try:
        experiments = experiment_manager.list_experiments()
        flags = feature_flag_manager.list_flags()
        
        # Calculate stats
        running_experiments = [e for e in experiments if e.get("status") == "running"]
        completed_experiments = [e for e in experiments if e.get("status") == "completed"]
        active_flags = [f for f in flags if f.get("enabled")]
        
        return {
            "success": True,
            "overview": {
                "experiments": {
                    "total": len(experiments),
                    "running": len(running_experiments),
                    "completed": len(completed_experiments)
                },
                "feature_flags": {
                    "total": len(flags),
                    "active": len(active_flags),
                    "inactive": len(flags) - len(active_flags)
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== CONTINUOUS LEARNING ENDPOINTS =====

@router.get("/learning/statistics", response_model=Dict[str, Any])
async def get_learning_statistics():
    """Get continuous learning pipeline statistics."""
    try:
        from services.llm.continuous_learning import get_pipeline
        
        pipeline = get_pipeline()
        stats = pipeline.get_statistics()
        
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error getting learning statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/feedback", response_model=Dict[str, Any])
async def collect_learning_feedback(
    user_id: str = Query(..., description="User ID"),
    query: str = Query(..., description="User query"),
    response: str = Query(..., description="System response"),
    feedback_type: str = Query(..., description="Feedback type"),
    rating: Optional[float] = Query(None, description="Rating (0-5)"),
    correction: Optional[str] = Query(None, description="User correction")
):
    """Collect user feedback for continuous learning."""
    try:
        from services.llm.continuous_learning import get_pipeline
        
        pipeline = get_pipeline()
        event = pipeline.feedback_collector.collect_feedback(
            user_id=user_id,
            query=query,
            response=response,
            feedback_type=feedback_type,
            rating=rating,
            correction=correction
        )
        
        return {
            "success": True,
            "feedback_id": event.id
        }
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/run-cycle", response_model=Dict[str, Any])
async def run_learning_cycle():
    """Manually trigger a learning cycle."""
    try:
        from services.llm.continuous_learning import get_pipeline
        
        pipeline = get_pipeline()
        result = await pipeline.run_learning_cycle()
        
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Error running learning cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/patterns", response_model=Dict[str, Any])
async def get_learned_patterns(
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    min_confidence: Optional[float] = Query(None, description="Minimum confidence")
):
    """Get learned patterns."""
    try:
        from services.llm.continuous_learning import get_pipeline
        
        pipeline = get_pipeline()
        patterns = pipeline.pattern_learner.get_patterns(
            pattern_type=pattern_type,
            min_confidence=min_confidence
        )
        
        return {
            "success": True,
            "patterns": [p.to_dict() for p in patterns],
            "total": len(patterns)
        }
    except Exception as e:
        logger.error(f"Error getting patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/canary/deployments", response_model=Dict[str, Any])
async def list_canary_deployments(
    status: Optional[str] = Query(None, description="Filter by status")
):
    """List canary deployments."""
    try:
        from services.llm.continuous_learning import get_pipeline
        
        pipeline = get_pipeline()
        deployments = pipeline.canary_deployment.list_deployments(status=status)
        
        return {
            "success": True,
            "deployments": deployments,
            "total": len(deployments)
        }
    except Exception as e:
        logger.error(f"Error listing deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/canary/{deployment_id}/promote", response_model=Dict[str, Any])
async def promote_canary(deployment_id: str):
    """Promote canary to production."""
    try:
        from services.llm.continuous_learning import get_pipeline
        
        pipeline = get_pipeline()
        result = pipeline.canary_deployment.promote_to_production(deployment_id)
        
        return {
            "success": True,
            "deployment": result
        }
    except Exception as e:
        logger.error(f"Error promoting canary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/canary/{deployment_id}/rollback", response_model=Dict[str, Any])
async def rollback_canary(deployment_id: str):
    """Rollback a canary deployment."""
    try:
        from services.llm.continuous_learning import get_pipeline
        
        pipeline = get_pipeline()
        result = pipeline.canary_deployment.rollback(deployment_id)
        
        return {
            "success": True,
            "deployment": result
        }
    except Exception as e:
        logger.error(f"Error rolling back canary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/canary/monitor", response_model=Dict[str, Any])
async def monitor_canaries():
    """Monitor all canary deployments and auto-promote/rollback."""
    try:
        from services.llm.continuous_learning import get_pipeline
        
        pipeline = get_pipeline()
        results = await pipeline.monitor_canaries()
        
        return {
            "success": True,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error monitoring canaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== HEALTH CHECK =====

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Check health of experiments and feature flags systems."""
    initialize_managers()
    
    return {
        "success": True,
        "status": "healthy",
        "components": {
            "experiment_manager": experiment_manager is not None,
            "feature_flag_manager": feature_flag_manager is not None
        }
    }

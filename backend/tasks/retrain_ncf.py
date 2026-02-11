"""
NCF Model Retraining Celery Tasks

Background tasks for automated NCF model retraining, deployment, and maintenance.

Tasks:
- retrain_ncf_model: Complete retraining pipeline
- aggregate_training_data: Periodic data aggregation
- cleanup_old_models: Remove old model versions

Author: AI Istanbul Team
Date: February 10, 2026
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

from celery import Task
from backend.celery_app import app
from backend.services.training_data_collector import TrainingDataCollector
from backend.ml.deep_learning.model_validator import ModelValidator
from backend.services.model_deployment_service import get_deployment_service
from backend.services.monitoring.ncf_monitoring_helper import get_ncf_monitoring_helper

logger = logging.getLogger(__name__)


class RetrainingTask(Task):
    """Base task with logging."""
    
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"✅ Task {self.name} [{task_id}] succeeded")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"❌ Task {self.name} [{task_id}] failed: {exc}")
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(f"🔄 Task {self.name} [{task_id}] retrying: {exc}")


@app.task(base=RetrainingTask, bind=True, max_retries=3)
def retrain_ncf_model(
    self,
    min_interactions: int = 1000,
    days_of_data: int = 30
) -> Dict[str, Any]:
    """
    Complete NCF model retraining pipeline.
    
    Steps:
    1. Collect training data from user interactions
    2. Train new NCF model
    3. Export to ONNX format
    4. Validate new model
    5. Deploy using blue-green strategy
    
    Args:
        min_interactions: Minimum interactions required for retraining
        days_of_data: Number of days of data to collect
        
    Returns:
        Dictionary with retraining results
    """
    logger.info("🚀 Starting NCF model retraining task...")
    logger.info(f"⏱️ Task ID: {self.request.id}")
    
    try:
        # Step 1: Collect training data
        logger.info("📊 Step 1/5: Collecting training data...")
        
        from backend.database import get_db
        db = next(get_db())
        
        collector = TrainingDataCollector(db)
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_of_data)
        
        training_data = collector.prepare_training_data(
            start_date=start_date,
            end_date=end_date
        )
        
        n_interactions = training_data['metadata']['train_size'] + training_data['metadata']['test_size']
        
        if n_interactions < min_interactions:
            logger.warning(f"⚠️ Insufficient data: {n_interactions} < {min_interactions}")
            return {
                'status': 'skipped',
                'reason': 'insufficient_data',
                'interactions': n_interactions,
                'min_required': min_interactions
            }
        
        logger.info(f"✅ Collected {n_interactions} interactions")
        
        # Step 2: Train new model
        logger.info("🧠 Step 2/5: Training NCF model...")
        
        from backend.ml.deep_learning.train_ncf import NCFTrainer
        
        trainer = NCFTrainer(
            num_users=training_data['num_users'],
            num_items=training_data['num_items'],
            embedding_dim=64,
            hidden_layers=[128, 64, 32]
        )
        
        training_history = trainer.train(
            train_data=training_data['train'],
            val_data=training_data['test'],
            epochs=20,
            batch_size=256,
            learning_rate=0.001
        )
        
        logger.info(f"✅ Training complete: {training_history['best_loss']:.4f} best loss")
        
        # Step 3: Export to ONNX
        logger.info("📦 Step 3/5: Exporting to ONNX...")
        
        from backend.ml.deep_learning.onnx_export import ONNXExporter
        
        exporter = ONNXExporter(trainer.model)
        
        # Create temporary path for new model
        models_dir = Path("backend/ml/deep_learning/models")
        temp_model_path = models_dir / f"ncf_model_temp_{int(datetime.utcnow().timestamp())}.onnx"
        
        onnx_info = exporter.export(
            output_path=str(temp_model_path),
            num_users=training_data['num_users'],
            num_items=training_data['num_items']
        )
        
        logger.info(f"✅ ONNX export complete: {onnx_info['model_size_bytes'] / 1024 / 1024:.2f} MB")
        
        # Step 4: Validate new model
        logger.info("🔍 Step 4/5: Validating new model...")
        
        from backend.ml.deep_learning.onnx_inference import ONNXPredictor
        
        new_predictor = ONNXPredictor(str(temp_model_path))
        validator = ModelValidator()
        
        # Get current model metrics for comparison
        from backend.services.production_ncf_service import get_production_ncf_service
        current_service = get_production_ncf_service()
        current_metrics = None
        
        if current_service and hasattr(current_service, 'onnx_predictor'):
            current_metrics = {
                'accuracy': 0.65,  # Would be loaded from metrics store
                'latency_p95_ms': 45.0
            }
        
        validation_report = validator.validate_model(
            new_model=new_predictor,
            test_data=training_data['test'],
            current_metrics=current_metrics
        )
        
        if not validation_report['passed']:
            logger.warning("⚠️ Model validation failed, deployment aborted")
            
            # Clean up temp model
            temp_model_path.unlink()
            
            return {
                'status': 'validation_failed',
                'validation_report': validation_report,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        logger.info("✅ Validation passed!")
        
        # Step 5: Deploy new model
        logger.info("🚀 Step 5/5: Deploying new model...")
        
        deployment_service = get_deployment_service()
        
        # Register new version
        metadata = {
            'created_at': datetime.utcnow().isoformat(),
            'training_data': training_data['metadata'],
            'validation_report': validation_report,
            'onnx_info': onnx_info,
            'task_id': self.request.id
        }
        
        version = deployment_service.register_version(
            model_path=temp_model_path,
            metadata=metadata
        )
        
        # Deploy using blue-green
        deployment_success = deployment_service.deploy_blue_green(
            new_version=version,
            validation_minutes=int(os.getenv('NCF_DEPLOYMENT_VALIDATION_MINUTES', '10'))
        )
        
        # Clean up temp file
        if temp_model_path.exists():
            temp_model_path.unlink()
        
        if deployment_success:
            logger.info(f"🎉 Model retraining complete! Version {version} deployed")
            
            # Update monitoring metrics
            monitoring = get_ncf_monitoring_helper()
            monitoring.update_model_quality(
                accuracy=validation_report['new_model']['accuracy'],
                diversity=validation_report['new_model']['diversity']
            )
            
            return {
                'status': 'success',
                'version': version,
                'validation_report': validation_report,
                'deployment': 'blue_green',
                'timestamp': datetime.utcnow().isoformat()
            }
        else:
            logger.error("❌ Deployment failed")
            return {
                'status': 'deployment_failed',
                'version': version,
                'timestamp': datetime.utcnow().isoformat()
            }
        
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}", exc_info=True)
        
        # Retry task
        raise self.retry(exc=e, countdown=60 * 15)  # Retry after 15 minutes


@app.task(base=RetrainingTask)
def aggregate_training_data() -> Dict[str, Any]:
    """
    Aggregate training data for faster access during retraining.
    
    Runs hourly to pre-process interaction data.
    
    Returns:
        Statistics about aggregated data
    """
    logger.info("📊 Aggregating training data...")
    
    try:
        from backend.database import get_db
        db = next(get_db())
        
        collector = TrainingDataCollector(db)
        
        # Collect last 7 days of data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        df = collector.collect_interactions(start_date, end_date)
        
        stats = {
            'total_interactions': len(df),
            'unique_users': df['user_id'].nunique() if not df.empty else 0,
            'unique_items': df['item_id'].nunique() if not df.empty else 0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"✅ Aggregated {stats['total_interactions']} interactions")
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Data aggregation failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


@app.task(base=RetrainingTask)
def cleanup_old_models(keep_latest: int = 5) -> Dict[str, Any]:
    """
    Clean up old model versions to save disk space.
    
    Runs weekly to remove retired model versions.
    
    Args:
        keep_latest: Number of recent versions to keep
        
    Returns:
        Cleanup statistics
    """
    logger.info("🧹 Cleaning up old model versions...")
    
    try:
        deployment_service = get_deployment_service()
        
        versions_before = len(deployment_service.versions)
        
        deployment_service.cleanup_old_versions(keep_latest=keep_latest)
        
        versions_after = len(deployment_service.versions)
        removed = versions_before - versions_after
        
        logger.info(f"✅ Removed {removed} old versions")
        
        return {
            'status': 'success',
            'versions_removed': removed,
            'versions_remaining': versions_after,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }

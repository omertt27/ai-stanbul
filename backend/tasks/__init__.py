"""
Celery Tasks Module

Background tasks for AI Istanbul system.
"""

from backend.tasks.retrain_ncf import (
    retrain_ncf_model,
    aggregate_training_data,
    cleanup_old_models
)

__all__ = [
    'retrain_ncf_model',
    'aggregate_training_data',
    'cleanup_old_models'
]

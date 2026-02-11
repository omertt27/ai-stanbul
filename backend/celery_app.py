"""
Celery Application Configuration

Handles background tasks for AI Istanbul, including:
- NCF model retraining
- Data aggregation
- Cache warming
- Scheduled maintenance

Author: AI Istanbul Team
Date: February 10, 2026
"""

import os
import logging
from celery import Celery
from celery.schedules import crontab

logger = logging.getLogger(__name__)

# Initialize Celery app
app = Celery(
    'ai_istanbul',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    include=[
        'backend.tasks.retrain_ncf',
        # Add more task modules here
    ]
)

# Celery Configuration
app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task settings
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # Soft limit at 55 minutes
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for long-running tasks
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks (prevent memory leaks)
    
    # Result backend
    result_expires=86400,  # Results expire after 24 hours
    result_backend_transport_options={'master_name': 'mymaster'},
    
    # Beat schedule (periodic tasks)
    beat_schedule={
        'retrain-ncf-model-daily': {
            'task': 'backend.tasks.retrain_ncf.retrain_ncf_model',
            'schedule': crontab(
                hour=os.getenv('NCF_RETRAIN_HOUR', '2'),  # Default: 2 AM
                minute=os.getenv('NCF_RETRAIN_MINUTE', '0')
            ),
            'options': {
                'expires': 3600,  # Task expires if not picked up within 1 hour
            }
        },
        'aggregate-training-data-hourly': {
            'task': 'backend.tasks.retrain_ncf.aggregate_training_data',
            'schedule': crontab(minute='0'),  # Every hour
            'options': {
                'expires': 600,
            }
        },
        'cleanup-old-models-weekly': {
            'task': 'backend.tasks.retrain_ncf.cleanup_old_models',
            'schedule': crontab(
                hour='3',
                minute='0',
                day_of_week='0'  # Sunday
            ),
            'options': {
                'expires': 7200,
            }
        },
    },
)

# Task routes (queue management)
app.conf.task_routes = {
    'backend.tasks.retrain_ncf.*': {'queue': 'ml_training'},
    # Other tasks use 'default' queue
}

logger.info("✅ Celery app configured")
logger.info(f"📍 Broker: {app.conf.broker_url}")
logger.info(f"📍 Backend: {app.conf.result_backend}")

if __name__ == '__main__':
    app.start()

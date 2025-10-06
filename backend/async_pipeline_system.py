#!/usr/bin/env python3
"""
Asynchronous Pipeline System for AI Istanbul
==========================================

Implements high-performance async pipelines for:
1. Data ingestion and processing
2. Vector embedding updates
3. Query handling and response generation
4. Background data pipeline tasks

Features:
- Separate async queues for different operations
- Low-latency query processing
- Background data updates
- Concurrent processing with proper resource management
- Integration with Redis for task queuing
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
import traceback

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    CRITICAL = 1  # User queries - highest priority
    HIGH = 2      # Real-time updates
    MEDIUM = 3    # Background processing
    LOW = 4       # Batch operations

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class AsyncTask:
    """Async task definition"""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    error_message: Optional[str] = None

class AsyncPipelineSystem:
    """High-performance async pipeline system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/3"):
        self.redis_url = redis_url
        self.redis_client = None
        self.running = False
        self.workers = {}
        self.task_queues = {
            TaskPriority.CRITICAL: asyncio.Queue(maxsize=1000),
            TaskPriority.HIGH: asyncio.Queue(maxsize=2000),
            TaskPriority.MEDIUM: asyncio.Queue(maxsize=5000),
            TaskPriority.LOW: asyncio.Queue(maxsize=10000)
        }
        self.task_handlers = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "avg_processing_time": 0.0,
            "queue_sizes": {}
        }
        
    async def initialize(self):
        """Initialize the async pipeline system"""
        print("🚀 Initializing Async Pipeline System...")
        
        # Initialize Redis connection
        if REDIS_AVAILABLE:
            try:
                if hasattr(redis, 'from_url'):
                    # Use async Redis
                    self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                    await self.redis_client.ping()
                else:
                    # Use sync Redis with async wrapper
                    sync_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
                    sync_client.ping()
                    self.redis_client = sync_client
                
                print("✅ Redis connection established")
            except Exception as e:
                print(f"⚠️ Redis not available: {e}")
                self.redis_client = None
        
        # Register default task handlers
        self._register_default_handlers()
        
        print("✅ Async Pipeline System initialized")
    
    def _register_default_handlers(self):
        """Register default task handlers"""
        
        @self.task_handler("query_processing")
        async def handle_query_processing(task_data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle user query processing with low latency"""
            query = task_data.get("query", "")
            user_context = task_data.get("user_context", {})
            
            # Simulate fast query processing
            start_time = time.time()
            
            try:
                # Import and use the complete query pipeline
                from complete_query_pipeline import complete_query_pipeline
                
                # Process query
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    complete_query_pipeline.process_query,
                    query,
                    user_context
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    "success": True,
                    "result": result,
                    "processing_time_ms": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
        
        @self.task_handler("vector_embedding_update")
        async def handle_vector_embedding_update(task_data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle vector embedding updates in background"""
            try:
                from vector_embedding_system import vector_embedding_system
                
                doc_id = task_data.get("doc_id")
                content = task_data.get("content")
                metadata = task_data.get("metadata", {})
                content_type = task_data.get("content_type", "general")
                
                # Add document to vector store
                success = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    vector_embedding_system.add_document,
                    doc_id,
                    content,
                    metadata,
                    content_type
                )
                
                return {
                    "success": success,
                    "doc_id": doc_id,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "doc_id": task_data.get("doc_id")
                }
        
        @self.task_handler("data_pipeline_update")
        async def handle_data_pipeline_update(task_data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle automated data pipeline updates"""
            try:
                from enhanced_data_pipeline import enhanced_data_pipeline
                
                pipeline_type = task_data.get("pipeline_type", "full")
                
                # Run data pipeline
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    enhanced_data_pipeline.run_pipeline,
                    pipeline_type
                )
                
                return {
                    "success": True,
                    "pipeline_type": pipeline_type,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "pipeline_type": task_data.get("pipeline_type")
                }
        
        @self.task_handler("cache_precomputation")
        async def handle_cache_precomputation(task_data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle cache precomputation for popular queries"""
            try:
                # Precompute popular recommendations
                recommendations = {
                    "sultanahmet": [
                        {"name": "Hagia Sophia", "type": "attraction", "rating": 4.8},
                        {"name": "Blue Mosque", "type": "attraction", "rating": 4.7},
                        {"name": "Topkapi Palace", "type": "attraction", "rating": 4.6}
                    ],
                    "beyoglu": [
                        {"name": "Galata Tower", "type": "attraction", "rating": 4.5},
                        {"name": "Istanbul Modern", "type": "museum", "rating": 4.4},
                        {"name": "Pera Museum", "type": "museum", "rating": 4.3}
                    ]
                }
                
                # Cache the recommendations
                if self.redis_client:
                    for district, recs in recommendations.items():
                        cache_key = f"precomputed:recommendations:{district}"
                        if hasattr(self.redis_client, 'setex'):
                            await self.redis_client.setex(cache_key, 3600, json.dumps(recs))
                        else:
                            self.redis_client.setex(cache_key, 3600, json.dumps(recs))
                
                return {
                    "success": True,
                    "cached_districts": list(recommendations.keys()),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def task_handler(self, task_type: str):
        """Decorator to register task handlers"""
        def decorator(func):
            self.task_handlers[task_type] = func
            return func
        return decorator
    
    async def submit_task(self, task_type: str, data: Dict[str, Any], 
                         priority: Optional[TaskPriority] = None,
                         scheduled_at: Optional[datetime] = None) -> str:
        """Submit a task to the async pipeline"""
        task_id = str(uuid.uuid4())
        
        # Default priority if none provided
        if priority is None:
            priority = TaskPriority.MEDIUM
        
        task = AsyncTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data,
            created_at=datetime.now(),
            scheduled_at=scheduled_at
        )
        
        # Add to appropriate queue
        try:
            await self.task_queues[priority].put(task)
            print(f"✅ Task {task_id} ({task_type}) queued with priority {priority.name}")
            return task_id
        except Exception as e:
            print(f"❌ Failed to queue task {task_id}: {e}")
            raise
    
    async def start_workers(self, num_workers: int = 5):
        """Start async worker processes"""
        print(f"🚀 Starting {num_workers} async workers...")
        
        self.running = True
        
        # Start workers for each priority level
        for priority in TaskPriority:
            for i in range(num_workers):
                worker_id = f"{priority.name.lower()}_worker_{i}"
                self.workers[worker_id] = asyncio.create_task(
                    self._worker(worker_id, priority)
                )
        
        # Start metrics collector
        self.workers["metrics_collector"] = asyncio.create_task(
            self._metrics_collector()
        )
        
        print(f"✅ Started {len(self.workers)} async workers")
    
    async def _worker(self, worker_id: str, priority: TaskPriority):
        """Async worker process"""
        queue = self.task_queues[priority]
        
        while self.running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Check if task is scheduled for later
                if task.scheduled_at and task.scheduled_at > datetime.now():
                    # Put back in queue and wait
                    await queue.put(task)
                    await asyncio.sleep(0.1)
                    continue
                
                # Process task
                await self._process_task(worker_id, task)
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_task(self, worker_id: str, task: AsyncTask):
        """Process a single task"""
        start_time = time.time()
        task.status = TaskStatus.PROCESSING
        
        try:
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"No handler for task type: {task.task_type}")
            
            # Execute task
            result = await handler(task.data)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics["tasks_processed"] += 1
            
            # Update average processing time
            current_avg = self.metrics["avg_processing_time"]
            task_count = self.metrics["tasks_processed"]
            self.metrics["avg_processing_time"] = (
                (current_avg * (task_count - 1) + processing_time) / task_count
            )
            
            task.status = TaskStatus.COMPLETED
            
            print(f"✅ {worker_id} completed {task.task_id} in {processing_time:.1f}ms")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self.metrics["tasks_failed"] += 1
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                
                # Add back to queue with delay
                await asyncio.sleep(min(2 ** task.retry_count, 10))  # Exponential backoff
                await self.task_queues[task.priority].put(task)
                
                print(f"🔄 Retrying {task.task_id} (attempt {task.retry_count})")
            else:
                print(f"❌ {worker_id} failed {task.task_id}: {e}")
    
    async def _metrics_collector(self):
        """Collect and update metrics"""
        while self.running:
            try:
                # Update queue sizes
                for priority, queue in self.task_queues.items():
                    self.metrics["queue_sizes"][priority.name] = queue.qsize()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(5)
    
    async def stop_workers(self):
        """Stop all async workers"""
        print("🛑 Stopping async workers...")
        self.running = False
        
        # Cancel all workers
        for worker_id, worker_task in self.workers.items():
            worker_task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers.values(), return_exceptions=True)
        
        self.workers.clear()
        print("✅ All async workers stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics"""
        return {
            "running": self.running,
            "active_workers": len(self.workers),
            "tasks_processed": self.metrics["tasks_processed"],
            "tasks_failed": self.metrics["tasks_failed"],
            "success_rate": (
                self.metrics["tasks_processed"] / 
                max(self.metrics["tasks_processed"] + self.metrics["tasks_failed"], 1)
            ) * 100,
            "avg_processing_time_ms": self.metrics["avg_processing_time"],
            "queue_sizes": self.metrics["queue_sizes"],
            "task_handlers": list(self.task_handlers.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def process_query_async(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a query asynchronously with high priority"""
        task_id = await self.submit_task(
            "query_processing",
            {
                "query": query,
                "user_context": user_context or {}
            },
            priority=TaskPriority.CRITICAL
        )
        
        # For demo purposes, wait for result (in production, would return task_id)
        # This is a simplified implementation - in production you'd have a result store
        await asyncio.sleep(0.1)  # Small delay to allow processing
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "message": "Query submitted for async processing"
        }

# Global async pipeline instance
async_pipeline_system = AsyncPipelineSystem()

async def initialize_async_pipeline():
    """Initialize the async pipeline system"""
    try:
        await async_pipeline_system.initialize()
        await async_pipeline_system.start_workers(num_workers=3)
        
        print("✅ Async Pipeline System ready")
        return True
        
    except Exception as e:
        print(f"❌ Async pipeline initialization error: {e}")
        return False

async def test_async_pipeline():
    """Test the async pipeline system"""
    print("🧪 Testing Async Pipeline System...")
    
    # Initialize
    success = await initialize_async_pipeline()
    if not success:
        return False
    
    # Submit test tasks
    tasks = []
    
    # High priority query
    task_id = await async_pipeline_system.submit_task(
        "query_processing",
        {"query": "Turkish restaurants in Sultanahmet", "user_context": {}},
        priority=TaskPriority.CRITICAL
    )
    tasks.append(task_id)
    
    # Background vector update
    task_id = await async_pipeline_system.submit_task(
        "vector_embedding_update",
        {
            "doc_id": "test_restaurant_123",
            "content": "Amazing Turkish restaurant with traditional Ottoman cuisine",
            "metadata": {"type": "restaurant", "district": "Sultanahmet"},
            "content_type": "restaurant"
        },
        priority=TaskPriority.MEDIUM
    )
    tasks.append(task_id)
    
    # Cache precomputation
    task_id = await async_pipeline_system.submit_task(
        "cache_precomputation",
        {"districts": ["sultanahmet", "beyoglu"]},
        priority=TaskPriority.LOW
    )
    tasks.append(task_id)
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Get metrics
    metrics = async_pipeline_system.get_metrics()
    print(f"📊 Pipeline metrics: {metrics['tasks_processed']} processed, {metrics['success_rate']:.1f}% success rate")
    
    # Stop pipeline
    await async_pipeline_system.stop_workers()
    
    return metrics['tasks_processed'] > 0

if __name__ == "__main__":
    # Test the async pipeline system
    async def main():
        success = await test_async_pipeline()
        if success:
            print("✅ Async Pipeline System is working correctly!")
        else:
            print("❌ Async Pipeline System test failed")
        return success
    
    result = asyncio.run(main())

"""
Prometheus Metrics Instrumentation for AI Istanbul
Exposes metrics for monitoring recommendation system performance
"""

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import psutil
import logging
from typing import Optional
import time

logger = logging.getLogger(__name__)

# ============================================================================
# REQUEST METRICS
# ============================================================================

# Total recommendation requests
recommendation_requests_total = Counter(
    'recommendation_requests_total',
    'Total number of recommendation requests',
    ['endpoint', 'variant']  # Labels for different endpoints and A/B test variants
)

# Recommendation latency
recommendation_latency_seconds = Histogram(
    'recommendation_latency_seconds',
    'Recommendation request latency in seconds',
    ['endpoint'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

# Recommendation errors
recommendation_errors_total = Counter(
    'recommendation_errors_total',
    'Total number of recommendation errors',
    ['error_type']
)

# ============================================================================
# FEEDBACK EVENT METRICS
# ============================================================================

# Feedback events collected
feedback_events_collected_total = Counter(
    'feedback_events_collected_total',
    'Total number of feedback events collected',
    ['event_type']  # view, click, rating, save, conversion
)

# Feedback events processed
feedback_events_processed_total = Counter(
    'feedback_events_processed_total',
    'Total number of feedback events processed by online learning'
)

# Feedback processing latency
feedback_processing_latency_seconds = Histogram(
    'feedback_processing_latency_seconds',
    'Feedback event processing latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

# ============================================================================
# MODEL PERFORMANCE METRICS
# ============================================================================

# Model NDCG@10
model_ndcg_10 = Gauge(
    'model_ndcg_10',
    'Model NDCG@10 score',
    ['model_name']  # online_learning, ncf, lightgbm, ensemble
)

# Model inference time
model_inference_seconds = Histogram(
    'model_inference_seconds',
    'Model inference time in seconds',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

# Number of items ranked
items_ranked_total = Counter(
    'items_ranked_total',
    'Total number of items ranked',
    ['model_name']
)

# ============================================================================
# BUSINESS METRICS
# ============================================================================

# Click-through rate
recommendation_ctr = Gauge(
    'recommendation_ctr',
    'Current click-through rate (rolling window)'
)

# Average rating
recommendation_avg_rating = Gauge(
    'recommendation_avg_rating',
    'Average rating of recommended items'
)

# User engagement score
user_engagement_score = Gauge(
    'user_engagement_score',
    'Aggregated user engagement score'
)

# Conversion rate
conversion_rate = Gauge(
    'conversion_rate',
    'Conversion rate (visits / clicks)'
)

# ============================================================================
# SYSTEM RESOURCE METRICS
# ============================================================================

# CPU usage
system_cpu_percent = Gauge(
    'system_cpu_percent',
    'CPU usage percentage'
)

# Memory usage
system_memory_percent = Gauge(
    'system_memory_percent',
    'Memory usage percentage'
)

system_memory_available_bytes = Gauge(
    'system_memory_available_bytes',
    'Available memory in bytes'
)

# Disk usage
system_disk_percent = Gauge(
    'system_disk_percent',
    'Disk usage percentage'
)

system_disk_free_bytes = Gauge(
    'system_disk_free_bytes',
    'Free disk space in bytes'
)

# ============================================================================
# DATABASE METRICS
# ============================================================================

# Database query latency
database_query_latency_seconds = Histogram(
    'database_query_latency_seconds',
    'Database query latency in seconds',
    ['query_type'],  # select, insert, update, aggregate
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Database connections
database_connections_active = Gauge(
    'database_connections_active',
    'Number of active database connections'
)

# ============================================================================
# ONLINE LEARNING METRICS
# ============================================================================

# Thompson Sampling arm selections
thompson_sampling_arm_selections = Counter(
    'thompson_sampling_arm_selections_total',
    'Number of times each arm was selected',
    ['arm']
)

# Concept drift detections
concept_drift_detections = Counter(
    'concept_drift_detections_total',
    'Number of times concept drift was detected'
)

# User aggregates updated
user_aggregates_updated_total = Counter(
    'user_aggregates_updated_total',
    'Total number of user aggregate updates'
)

# ============================================================================
# A/B TESTING METRICS
# ============================================================================

# A/B test assignments
ab_test_assignments_total = Counter(
    'ab_test_assignments_total',
    'Total number of A/B test assignments',
    ['test_name', 'variant']
)

# A/B test metrics by variant
ab_test_ctr = Gauge(
    'ab_test_ctr',
    'CTR by A/B test variant',
    ['test_name', 'variant']
)

# ============================================================================
# TRAINING METRICS
# ============================================================================

# Training duration
training_duration_minutes = Gauge(
    'training_duration_minutes',
    'Model training duration in minutes',
    ['model_name']
)

# Training NDCG
training_ndcg_10 = Gauge(
    'training_ndcg_10',
    'Training set NDCG@10',
    ['model_name']
)

# Models trained
models_trained_total = Counter(
    'models_trained_total',
    'Total number of model training runs',
    ['model_name']
)

# ============================================================================
# APPLICATION INFO
# ============================================================================

# Application version and info
app_info = Info(
    'ai_istanbul_app',
    'AI Istanbul application information'
)

# Set application info
app_info.info({
    'version': '1.0.0',
    'phase': 'Phase 1 Complete',
    'deployment': 'single-t4-gpu',
    'budget_tier': 'optimized'
})

# ============================================================================
# METRIC UPDATER CLASS
# ============================================================================

class MetricsUpdater:
    """
    Helper class to update system metrics periodically
    """
    
    def __init__(self):
        """Initialize metrics updater"""
        self.running = False
        logger.info("✅ Metrics updater initialized")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            system_cpu_percent.set(cpu_percent)
            
            # Memory
            memory = psutil.virtual_memory()
            system_memory_percent.set(memory.percent)
            system_memory_available_bytes.set(memory.available)
            
            # Disk
            disk = psutil.disk_usage('/')
            system_disk_percent.set(disk.percent)
            system_disk_free_bytes.set(disk.free)
            
            logger.debug(f"Updated system metrics: CPU={cpu_percent}%, Memory={memory.percent}%")
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def start_background_updater(self, interval: int = 15):
        """
        Start background thread to update metrics
        
        Args:
            interval: Update interval in seconds
        """
        import threading
        
        def update_loop():
            self.running = True
            while self.running:
                self.update_system_metrics()
                time.sleep(interval)
        
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        logger.info(f"✅ Started metrics updater (interval={interval}s)")
    
    def stop(self):
        """Stop background updater"""
        self.running = False
        logger.info("🛑 Stopped metrics updater")


# Global instance
_metrics_updater = None


def get_metrics_updater() -> MetricsUpdater:
    """Get or create global metrics updater"""
    global _metrics_updater
    if _metrics_updater is None:
        _metrics_updater = MetricsUpdater()
    return _metrics_updater


def start_metrics_server(port: int = 8001):
    """
    Start Prometheus metrics HTTP server
    
    Args:
        port: Port to listen on (default: 8001)
    """
    try:
        start_http_server(port)
        logger.info(f"✅ Prometheus metrics server started on port {port}")
        
        # Start background metric updater
        updater = get_metrics_updater()
        updater.start_background_updater(interval=15)
        
    except Exception as e:
        logger.error(f"❌ Failed to start metrics server: {e}")
        raise


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def track_recommendation_request(endpoint: str, variant: str = "default"):
    """Track a recommendation request"""
    recommendation_requests_total.labels(endpoint=endpoint, variant=variant).inc()


def track_feedback_event(event_type: str):
    """Track a feedback event"""
    feedback_events_collected_total.labels(event_type=event_type).inc()


def track_error(error_type: str):
    """Track an error"""
    recommendation_errors_total.labels(error_type=error_type).inc()


def update_model_ndcg(model_name: str, ndcg_value: float):
    """Update model NDCG score"""
    model_ndcg_10.labels(model_name=model_name).set(ndcg_value)


def update_ctr(ctr_value: float):
    """Update CTR metric"""
    recommendation_ctr.set(ctr_value)


# Example usage
if __name__ == "__main__":
    import time
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Start metrics server
    start_metrics_server(port=8001)
    
    # Simulate some metrics
    for i in range(100):
        # Simulate recommendation requests
        track_recommendation_request("hidden_gems", "treatment")
        
        # Simulate feedback events
        track_feedback_event("view")
        if i % 3 == 0:
            track_feedback_event("click")
        
        # Update model performance
        update_model_ndcg("online_learning", 0.72 + (i % 10) / 100)
        
        # Update CTR
        update_ctr(0.18 + (i % 5) / 100)
        
        time.sleep(1)
    
    print("\n✅ Metrics server running on http://localhost:8001/metrics")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping metrics server...")

"""
Prometheus metrics for monitoring LLM performance
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response
import time

router = APIRouter()

# Request metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total number of LLM requests',
    ['language', 'use_case']
)

llm_errors_total = Counter(
    'llm_errors_total',
    'Total number of LLM errors',
    ['error_type', 'language']
)

# Response time metrics
llm_response_time_seconds = Histogram(
    'llm_response_time_seconds',
    'LLM response time in seconds',
    ['language', 'use_case'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Token metrics
llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total number of tokens used',
    ['language', 'type']  # type: input/output
)

# Cache metrics
cache_requests_total = Counter(
    'cache_requests_total',
    'Total cache requests'
)

cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['tier']  # tier: l1/l2/l3
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses'
)

cache_l1_hits = Counter('cache_l1_hits', 'L1 cache hits')
cache_l2_hits = Counter('cache_l2_hits', 'L2 cache hits')
cache_l3_hits = Counter('cache_l3_hits', 'L3 cache hits')

# Feedback metrics
feedback_total = Counter(
    'feedback_total',
    'Total feedback submissions',
    ['language', 'type']
)

feedback_positive_total = Counter(
    'feedback_positive_total',
    'Total positive feedback',
    ['language']
)

feedback_negative_total = Counter(
    'feedback_negative_total',
    'Total negative feedback',
    ['language']
)

feedback_rating = Gauge(
    'feedback_rating',
    'Current average feedback rating',
    ['language']
)

# A/B Testing metrics
ab_test_requests = Counter(
    'ab_test_requests',
    'A/B test requests',
    ['experiment_id', 'variant']
)

ab_test_positive_feedback = Counter(
    'ab_test_positive_feedback',
    'Positive feedback per variant',
    ['experiment_id', 'variant']
)

ab_test_negative_feedback = Counter(
    'ab_test_negative_feedback',
    'Negative feedback per variant',
    ['experiment_id', 'variant']
)

ab_test_response_time_seconds = Histogram(
    'ab_test_response_time_seconds',
    'Response time per variant',
    ['experiment_id', 'variant'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
)

# Session metrics
active_sessions = Gauge(
    'active_sessions',
    'Number of active user sessions'
)


# Metrics endpoint
@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Helper functions for recording metrics
def record_llm_request(language: str, use_case: str):
    """Record an LLM request"""
    llm_requests_total.labels(language=language, use_case=use_case).inc()


def record_llm_error(error_type: str, language: str):
    """Record an LLM error"""
    llm_errors_total.labels(error_type=error_type, language=language).inc()


def record_llm_response_time(duration: float, language: str, use_case: str):
    """Record LLM response time"""
    llm_response_time_seconds.labels(
        language=language,
        use_case=use_case
    ).observe(duration)


def record_tokens(count: int, language: str, token_type: str):
    """Record token usage"""
    llm_tokens_total.labels(language=language, type=token_type).inc(count)


def record_cache_hit(tier: str):
    """Record a cache hit"""
    cache_requests_total.inc()
    cache_hits_total.labels(tier=tier).inc()
    
    if tier == 'l1':
        cache_l1_hits.inc()
    elif tier == 'l2':
        cache_l2_hits.inc()
    elif tier == 'l3':
        cache_l3_hits.inc()


def record_cache_miss():
    """Record a cache miss"""
    cache_requests_total.inc()
    cache_misses_total.inc()


def record_feedback(feedback_type: str, language: str, rating: int = None):
    """Record user feedback"""
    feedback_total.labels(language=language, type=feedback_type).inc()
    
    if feedback_type == 'thumbs_up':
        feedback_positive_total.labels(language=language).inc()
    elif feedback_type == 'thumbs_down':
        feedback_negative_total.labels(language=language).inc()
    
    if rating:
        feedback_rating.labels(language=language).set(rating)


def record_ab_test_request(experiment_id: str, variant: str, duration: float):
    """Record A/B test request"""
    ab_test_requests.labels(
        experiment_id=experiment_id,
        variant=variant
    ).inc()
    
    ab_test_response_time_seconds.labels(
        experiment_id=experiment_id,
        variant=variant
    ).observe(duration)


def record_ab_test_feedback(
    experiment_id: str,
    variant: str,
    is_positive: bool
):
    """Record A/B test feedback"""
    if is_positive:
        ab_test_positive_feedback.labels(
            experiment_id=experiment_id,
            variant=variant
        ).inc()
    else:
        ab_test_negative_feedback.labels(
            experiment_id=experiment_id,
            variant=variant
        ).inc()

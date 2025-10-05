# Sample Data Collection Configuration
# Copy this to config_local.py and customize

COLLECTION_CONFIG = {
    'max_concurrent_requests': 5,
    'request_delay': 1.0,
    'retry_attempts': 3,
    'timeout_seconds': 30,
    
    'data_sources': {
        'istanbul_tourism_official': {
            'enabled': True,
            'priority': 1,
            'rate_limit': 30
        },
        'google_places': {
            'enabled': False,  # Requires API key
            'priority': 2,
            'rate_limit': 100
        }
    },
    
    'quality_thresholds': {
        'min_text_length': 50,
        'max_text_length': 5000,
        'min_relevance_score': 0.3
    }
}

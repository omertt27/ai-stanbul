"""
Configuration for AI Istanbul Load Testing Suite
"""

import os
from typing import Dict, List, Any

# Base URLs
BACKEND_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FRONTEND_BASE_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# Production URLs
PRODUCTION_BACKEND_URL = "https://ai-istanbul-backend.render.com"
PRODUCTION_FRONTEND_URL = "https://ai-stanbul.onrender.com"

# Test Configuration
DEFAULT_TEST_CONFIG = {
    "users": 10,  # Concurrent users
    "spawn_rate": 2,  # Users spawned per second
    "duration": 300,  # Test duration in seconds (5 minutes)
    "host": BACKEND_BASE_URL
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time_p95": {
        "chat": 2000,  # ms
        "search": 1000,  # ms
        "crud": 500,   # ms
        "static": 100  # ms
    },
    "throughput": {
        "min_rps": 10,  # Minimum requests per second
        "target_rps": 50  # Target requests per second
    },
    "error_rate": {
        "max_4xx": 5,   # Max 5% 4xx errors
        "max_5xx": 1    # Max 1% 5xx errors
    },
    "resource_usage": {
        "max_cpu": 80,    # Max 80% CPU usage
        "max_memory": 512  # Max 512MB memory usage
    }
}

# API Endpoints to Test (Updated for minimal backend)
API_ENDPOINTS = {
    "health": "/health",
    "root": "/",
    "chat": "/chat",
    "blog_posts": "/api/blog/posts",
    "blog_post": "/api/blog/posts/1",
    "restaurants": "/api/restaurants", 
    "museums": "/api/museums",
    "docs": "/docs",
    "openapi": "/openapi.json",
    # Location-based endpoints (for mobile testing)
    "location_validate": "/api/location/validate",
    "location_nearby": "/api/location/nearby",
    "location_health": "/api/location/health"
}

# Test Data
TEST_DATA = {
    "chat_queries": [
        "What are the best restaurants in Sultanahmet?",
        "Tell me about Hagia Sophia",
        "How do I get from Taksim to Galata Tower?",
        "What's the weather like in Istanbul?",
        "Recommend a 3-day itinerary for Istanbul",
        "Best Turkish breakfast places in Beyoğlu",
        "Istanbul nightlife recommendations",
        "Museums near Sultanahmet Square",
        "Shopping areas in Istanbul",
        "Traditional Turkish baths"
    ],
    "restaurant_queries": [
        {"district": "Sultanahmet", "limit": 5},
        {"district": "Beyoğlu", "cuisine": "Turkish", "limit": 10},
        {"district": "Galata", "budget": "moderate", "limit": 8},
        {"keyword": "kebab", "limit": 15},
        {"cuisine": "seafood", "district": "Beşiktaş", "limit": 12}
    ],
    "route_requests": [
        {
            "start_lat": 41.0082,
            "start_lng": 28.9784,
            "max_distance_km": 5,
            "available_time_hours": 4,
            "preferred_categories": ["museum", "restaurant"],
            "transport_mode": "walking"
        },
        {
            "start_lat": 41.0258,
            "start_lng": 28.9744,
            "max_distance_km": 3,
            "available_time_hours": 2,
            "preferred_categories": ["shopping", "food"],
            "transport_mode": "walking"
        }
    ],
    "blog_data": {
        "title": "Test Blog Post",
        "content": "This is a test blog post for load testing.",
        "author": "Load Tester",
        "district": "Test District",
        "tags": ["test", "load-testing"]
    }
}

# Database Configuration
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600
}

# Redis Configuration  
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "decode_responses": True
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "metrics_interval": 5,  # seconds
    "resource_monitoring": True,
    "database_monitoring": True,
    "cache_monitoring": True,
    "log_level": "INFO"
}

# Report Configuration
REPORT_CONFIG = {
    "output_dir": "reports",
    "generate_graphs": True,
    "include_raw_data": True,
    "format": "html"  # html, json, csv
}

# Load Test Scenarios
LOAD_SCENARIOS = {
    "light_load": {
        "users": 5,
        "spawn_rate": 1,
        "duration": 60,
        "description": "Light load test - 5 users for 1 minute"
    },
    "normal_load": {
        "users": 25,
        "spawn_rate": 2,
        "duration": 300,
        "description": "Normal load test - 25 users for 5 minutes"
    },
    "heavy_load": {
        "users": 50,
        "spawn_rate": 5,
        "duration": 600,
        "description": "Heavy load test - 50 users for 10 minutes"
    },
    "stress_test": {
        "users": 100,
        "spawn_rate": 10,
        "duration": 900,
        "description": "Stress test - 100 users for 15 minutes"
    },
    "endurance_test": {
        "users": 20,
        "spawn_rate": 2,
        "duration": 3600,
        "description": "Endurance test - 20 users for 1 hour"
    }
}

# Frontend Testing Configuration
FRONTEND_CONFIG = {
    "browser": "chromium",
    "headless": True,
    "viewport": {"width": 1920, "height": 1080},
    "timeout": 30000,
    "pages_to_test": [
        "/",
        "/chat",
        "/blog",
        "/about",
        "/routes"
    ]
}

# Mobile Testing Configuration
MOBILE_CONFIG = {
    "viewport_sizes": [
        {"width": 375, "height": 667, "name": "iPhone SE"},
        {"width": 414, "height": 896, "name": "iPhone 11 Pro"},
        {"width": 360, "height": 640, "name": "Samsung Galaxy S8"},
        {"width": 768, "height": 1024, "name": "iPad"},
        {"width": 1024, "height": 768, "name": "iPad Landscape"}
    ],
    "test_locations": [
        {"lat": 41.0082, "lng": 28.9784, "name": "Sultanahmet Square"},
        {"lat": 41.0096, "lng": 28.9651, "name": "Galata Tower"},
        {"lat": 41.0058, "lng": 28.9768, "name": "Blue Mosque"},
        {"lat": 41.0255, "lng": 28.9742, "name": "Taksim Square"},
        {"lat": 41.0136, "lng": 28.9550, "name": "Grand Bazaar"}
    ],
    "location_features_to_test": [
        "nearby_restaurants",
        "route_planning", 
        "location_chat_context",
        "distance_calculations",
        "gps_accuracy"
    ]
}

# Enhanced Mobile Location Testing Configuration
ENHANCED_MOBILE_CONFIG = {
    'real_device_testing': {
        'enabled': True,
        'device_farm_url': None,  # Set if using cloud device testing
        'local_devices': ['iPhone', 'Android'],
        'test_duration_minutes': 30
    },
    
    'gps_accuracy': {
        'precision_threshold_meters': 10,
        'timeout_seconds': 30,
        'retry_attempts': 3,
        'test_real_locations': True
    },
    
    'network_simulation': {
        'test_conditions': ['4G', '3G', 'Slow 3G', 'Offline'],
        'timeout_multiplier': 2.0,
        'retry_on_failure': True
    },
    
    'performance_monitoring': {
        'memory_threshold_mb': 50,
        'battery_drain_monitoring': True,
        'cpu_usage_tracking': True,
        'network_usage_tracking': True
    },
    
    'accessibility_testing': {
        'screen_reader_simulation': True,
        'color_contrast_check': True,
        'touch_target_size_check': True,
        'keyboard_navigation_test': True
    },
    
    'location_contexts': {
        'istanbul_districts': [
            'Sultanahmet', 'Beyoğlu', 'Beşiktaş', 'Kadıköy', 
            'Üsküdar', 'Fatih', 'Galata', 'Taksim'
        ],
        'poi_categories': [
            'historic', 'religious', 'museum', 'shopping', 
            'restaurant', 'viewpoint', 'transportation'
        ]
    }
}

# Location Permission Testing
LOCATION_PERMISSION_CONFIG = {
    "popup_messages": [
        "Enable location to find nearby attractions",
        "Allow location access for personalized recommendations", 
        "Share your location for better Istanbul experience"
    ],
    "manual_entry_fallbacks": [
        "Enter your location manually",
        "Choose your area in Istanbul",
        "Select your district"
    ],
    "common_istanbul_locations": [
        "Sultanahmet",
        "Beyoğlu",
        "Galata", 
        "Taksim",
        "Beşiktaş",
        "Kadıköy",
        "Üsküdar",
        "Fatih"
    ]
}

def get_config(environment: str = "local") -> Dict[str, Any]:
    """Get configuration for specific environment"""
    config = {
        "backend_url": BACKEND_BASE_URL,
        "frontend_url": FRONTEND_BASE_URL,
        **DEFAULT_TEST_CONFIG
    }
    
    if environment == "production":
        config.update({
            "backend_url": PRODUCTION_BACKEND_URL,
            "frontend_url": PRODUCTION_FRONTEND_URL,
            "host": PRODUCTION_BACKEND_URL
        })
    
    return config

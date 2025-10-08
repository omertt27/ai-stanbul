"""
AI Istanbul Backend Load Testing with Locust
Comprehensive load testing for chat endpoints, streaming, and location services
"""

import json
import time
import random
from typing import Dict, List
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask

# Test data for realistic load testing
SAMPLE_QUERIES = [
    "What are the best restaurants in Sultanahmet?",
    "How do I get from Taksim to Galata Tower?", 
    "Tell me about museums in Beyoğlu",
    "Where can I find good Turkish breakfast?",
    "What's the best way to travel around Istanbul?",
    "Recommend some cafes in Kadıköy",
    "What are the opening hours for Hagia Sophia?",
    "Where should I go shopping in Istanbul?",
    "Tell me about Istanbul's nightlife",
    "What are some hidden gems in Istanbul?",
    "Best places for Turkish coffee?",
    "How to use public transport in Istanbul?",
    "Romantic restaurants for dinner?",
    "Family-friendly activities in Istanbul?",
    "Traditional Turkish food recommendations"
]

DISTRICTS = [
    "Sultanahmet", "Beyoğlu", "Taksim", "Kadıköy", "Beşiktaş",
    "Üsküdar", "Fatih", "Şişli", "Bakırköy", "Maltepe"
]

LOCATION_COORDINATES = [
    {"lat": 41.0082, "lng": 28.9784, "district": "Sultanahmet"},
    {"lat": 41.0361, "lng": 28.9855, "district": "Beyoğlu"},
    {"lat": 41.0370, "lng": 28.9859, "district": "Taksim"},
    {"lat": 40.9904, "lng": 29.0242, "district": "Kadıköy"},
    {"lat": 41.0426, "lng": 29.0007, "district": "Beşiktaş"},
]

class AIIstanbulUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.session_id = f"load_test_{int(time.time())}_{random.randint(1000, 9999)}"
        self.user_preferences = {
            "preferred_districts": random.sample(DISTRICTS, random.randint(1, 3)),
            "interests": random.choice(["food", "culture", "shopping", "nightlife", "history"])
        }
        
    @task(10)  # High frequency - main chat functionality
    def chat_with_ai(self):
        """Test main chat endpoint"""
        query = random.choice(SAMPLE_QUERIES)
        
        # Sometimes include location context
        location_context = None
        if random.random() < 0.3:  # 30% chance of location context
            location = random.choice(LOCATION_COORDINATES)
            location_context = {
                "has_location": True,
                "latitude": location["lat"],
                "longitude": location["lng"], 
                "district": location["district"],
                "nearby_pois": ["Test POI 1", "Test POI 2"],
                "session_id": self.session_id,
                "accuracy": random.uniform(10, 100)
            }
        
        payload = {
            "message": query,
            "session_id": self.session_id,
            "location_context": location_context,
            "context_type": "location_aware" if location_context else "general"
        }
        
        with self.client.post("/api/chat", json=payload, name="chat_with_ai") as response:
            if response.status_code != 200:
                print(f"Chat error: {response.status_code} - {response.text}")
            
    @task(5)  # Medium frequency - streaming chat
    def streaming_chat(self):
        """Test streaming chat endpoint"""
        query = random.choice(SAMPLE_QUERIES)
        
        location_context = None
        if random.random() < 0.4:  # 40% chance for streaming
            location = random.choice(LOCATION_COORDINATES)
            location_context = {
                "has_location": True,
                "latitude": location["lat"],
                "longitude": location["lng"],
                "district": location["district"],
                "nearby_pois": ["Stream POI 1", "Stream POI 2"],
                "session_id": self.session_id
            }
        
        payload = {
            "message": query,
            "session_id": self.session_id,
            "location_context": location_context
        }
        
        try:
            with self.client.post("/ai/stream", json=payload, 
                                stream=True, name="streaming_chat") as response:
                if response.status_code == 200:
                    # Read streaming response
                    chunk_count = 0
                    for chunk in response.iter_lines():
                        if chunk:
                            chunk_count += 1
                            if chunk_count > 50:  # Prevent infinite loops
                                break
                else:
                    print(f"Streaming error: {response.status_code}")
        except Exception as e:
            print(f"Streaming exception: {e}")
            
    @task(3)  # Lower frequency - restaurant search
    def search_restaurants(self):
        """Test restaurant search endpoint"""
        district = random.choice(DISTRICTS)
        cuisines = ["Turkish", "Mediterranean", "International", "Seafood", "Kebab"]
        
        params = {
            "district": district,
            "cuisine": random.choice(cuisines),
            "limit": random.randint(3, 10)
        }
        
        with self.client.get("/api/restaurants/search", params=params, 
                           name="restaurant_search") as response:
            if response.status_code != 200:
                print(f"Restaurant search error: {response.status_code}")
                
    @task(2)  # Lower frequency - location services
    def location_services(self):
        """Test location-based services"""
        location = random.choice(LOCATION_COORDINATES)
        
        # Test location session start
        payload = {
            "latitude": location["lat"],
            "longitude": location["lng"],
            "accuracy": random.uniform(10, 50),
            "user_id": f"load_test_user_{random.randint(1, 1000)}"
        }
        
        with self.client.post("/api/location/start-session", json=payload,
                            name="location_session") as response:
            if response.status_code == 200:
                session_data = response.json()
                session_id = session_data.get("session_id")
                
                if session_id:
                    # Test recommendations
                    rec_payload = {
                        "session_id": session_id,
                        "categories": ["restaurant", "attraction"],
                        "limit": 5
                    }
                    
                    self.client.post("/api/location/recommendations", 
                                   json=rec_payload, name="location_recommendations")
    
    @task(1)  # Low frequency - health and stats
    def health_check(self):
        """Test health and system endpoints"""
        endpoints = [
            "/health",
            "/api/restaurants/stats", 
            "/api/redis/sessions/active"
        ]
        
        endpoint = random.choice(endpoints)
        with self.client.get(endpoint, name="health_stats") as response:
            if response.status_code != 200:
                print(f"Health check error for {endpoint}: {response.status_code}")

class HighLoadUser(AIIstanbulUser):
    """Simulate high-load users with more aggressive patterns"""
    wait_time = between(0.5, 1.5)  # Faster requests
    
    @task(15)
    def rapid_chat(self):
        """Rapid-fire chat requests"""
        queries = random.sample(SAMPLE_QUERIES, 3)  # Send 3 queries in sequence
        
        for query in queries:
            payload = {
                "message": query,
                "session_id": self.session_id,
                "context_type": "rapid_test"
            }
            
            self.client.post("/api/chat", json=payload, name="rapid_chat")
            time.sleep(0.1)  # Brief pause between rapid requests

class MobileUser(AIIstanbulUser):
    """Simulate mobile users with location-heavy usage"""
    wait_time = between(2, 5)  # Mobile users wait longer
    
    @task(8)
    def location_heavy_chat(self):
        """Location-heavy chat patterns"""
        location = random.choice(LOCATION_COORDINATES)
        
        location_context = {
            "has_location": True,
            "latitude": location["lat"] + random.uniform(-0.01, 0.01),  # Slight variation
            "longitude": location["lng"] + random.uniform(-0.01, 0.01),
            "district": location["district"],
            "nearby_pois": [f"Mobile POI {i}" for i in range(random.randint(1, 5))],
            "session_id": self.session_id,
            "accuracy": random.uniform(5, 30)  # Better mobile accuracy
        }
        
        queries = [
            f"What's near me in {location['district']}?",
            "Find restaurants within walking distance",
            "How do I get to the nearest attraction?",
            "What's the closest metro station?"
        ]
        
        payload = {
            "message": random.choice(queries),
            "session_id": self.session_id,
            "location_context": location_context,
            "context_type": "mobile"
        }
        
        self.client.post("/api/chat", json=payload, name="mobile_location_chat")

# Performance monitoring events
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Log slow requests"""
    if response_time > 2000:  # Log requests slower than 2 seconds
        print(f"SLOW REQUEST: {name} took {response_time}ms")

@events.test_start.add_listener  
def on_test_start(environment, **kwargs):
    """Initialize load test"""
    print("🚀 Starting AI Istanbul Load Test")
    print(f"Target host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Clean up after load test"""
    print("🏁 Load test completed")
    
    # Print summary statistics
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failed requests: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")

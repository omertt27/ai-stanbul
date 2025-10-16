#!/usr/bin/env python3
"""
Cache Warming Script for AI Istanbul System
Improves cache hit rates by pre-loading frequently accessed data
"""

import asyncio
import json
import redis
import requests
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheWarmer:
    """Warms up caches with frequently accessed Istanbul data"""
    
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                db=0, 
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    async def warm_attractions_cache(self):
        """Pre-cache popular attractions data"""
        if not self.redis_client:
            return
        
        popular_attractions = [
            "hagia sophia", "blue mosque", "topkapi palace", "grand bazaar",
            "galata tower", "taksim square", "dolmabahce palace", "basilica cistern",
            "spice bazaar", "bosphorus", "ortakoy", "sultanahmet"
        ]
        
        try:
            # Simulate API calls to cache popular attractions
            base_url = "http://localhost:8000"
            
            for attraction in popular_attractions:
                try:
                    # Cache attraction information
                    cache_key = f"attraction:{attraction.replace(' ', '_')}"
                    response = requests.get(f"{base_url}/api/chat", 
                                          json={"message": f"Tell me about {attraction}"},
                                          timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Store in cache with 6 hour expiry
                        self.redis_client.setex(cache_key, 21600, json.dumps(data))
                        logger.info(f"📦 Cached: {attraction}")
                        
                        # Add small delay to avoid overwhelming the system
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cache {attraction}: {e}")
                    continue
                    
            logger.info(f"🔥 Warmed attractions cache with {len(popular_attractions)} items")
            
        except Exception as e:
            logger.error(f"❌ Attractions cache warming failed: {e}")
    
    async def warm_location_cache(self):
        """Pre-cache popular location queries"""
        if not self.redis_client:
            return
            
        popular_locations = [
            "sultanahmet", "beyoglu", "taksim", "galata", "ortakoy",
            "kadikoy", "besiktas", "uskudar", "fatih", "sisli"
        ]
        
        try:
            for location in popular_locations:
                try:
                    cache_key = f"location_events:{location}"
                    # Pre-cache location-based events
                    events_data = {
                        "location": location,
                        "events": [],
                        "cached_at": datetime.now().isoformat(),
                        "expiry": (datetime.now() + timedelta(hours=2)).isoformat()
                    }
                    
                    # Store with 2 hour expiry
                    self.redis_client.setex(cache_key, 7200, json.dumps(events_data))
                    logger.info(f"📍 Cached location: {location}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cache location {location}: {e}")
                    continue
                    
            logger.info(f"🗺️ Warmed location cache with {len(popular_locations)} items")
            
        except Exception as e:
            logger.error(f"❌ Location cache warming failed: {e}")
    
    async def warm_restaurants_cache(self):
        """Pre-cache popular restaurant queries"""
        if not self.redis_client:
            return
            
        restaurant_queries = [
            "best restaurants in sultanahmet",
            "turkish breakfast istanbul",
            "seafood restaurants bosphorus",
            "vegetarian restaurants beyoglu",
            "traditional turkish cuisine",
            "rooftop restaurants istanbul"
        ]
        
        try:
            base_url = "http://localhost:8000"
            
            for query in restaurant_queries:
                try:
                    cache_key = f"restaurants:{query.replace(' ', '_')}"
                    response = requests.get(f"{base_url}/api/chat",
                                          json={"message": query},
                                          timeout=5)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Store with 4 hour expiry
                        self.redis_client.setex(cache_key, 14400, json.dumps(data))
                        logger.info(f"🍽️ Cached restaurant query: {query}")
                        
                        await asyncio.sleep(0.5)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cache restaurant query {query}: {e}")
                    continue
                    
            logger.info(f"🍽️ Warmed restaurant cache with {len(restaurant_queries)} items")
            
        except Exception as e:
            logger.error(f"❌ Restaurant cache warming failed: {e}")
    
    async def warm_transportation_cache(self):
        """Pre-cache transportation route data"""
        if not self.redis_client:
            return
            
        popular_routes = [
            {"from": "taksim", "to": "sultanahmet"},
            {"from": "galata", "to": "karakoy"},
            {"from": "kadikoy", "to": "eminonu"},
            {"from": "airport", "to": "sultanahmet"},
            {"from": "besiktas", "to": "ortakoy"}
        ]
        
        try:
            for route in popular_routes:
                try:
                    cache_key = f"route:{route['from']}_to_{route['to']}"
                    route_data = {
                        "from": route["from"],
                        "to": route["to"],
                        "cached_at": datetime.now().isoformat(),
                        "methods": ["metro", "bus", "ferry", "walking"],
                        "estimated_time": "15-30 minutes"
                    }
                    
                    # Store with 6 hour expiry
                    self.redis_client.setex(cache_key, 21600, json.dumps(route_data))
                    logger.info(f"🚇 Cached route: {route['from']} → {route['to']}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cache route {route}: {e}")
                    continue
                    
            logger.info(f"🚇 Warmed transportation cache with {len(popular_routes)} routes")
            
        except Exception as e:
            logger.error(f"❌ Transportation cache warming failed: {e}")
    
    async def check_cache_status(self):
        """Check current cache status and statistics"""
        if not self.redis_client:
            logger.error("❌ Redis not available for cache status check")
            return
            
        try:
            # Get basic Redis info
            info = self.redis_client.info()
            
            # Count keys by pattern
            attraction_keys = len(self.redis_client.keys("attraction:*"))
            location_keys = len(self.redis_client.keys("location_events:*"))
            restaurant_keys = len(self.redis_client.keys("restaurants:*"))
            route_keys = len(self.redis_client.keys("route:*"))
            
            total_keys = self.redis_client.dbsize()
            
            logger.info("📊 Cache Status Report:")
            logger.info(f"   🔑 Total Keys: {total_keys}")
            logger.info(f"   🏛️ Attraction Keys: {attraction_keys}")
            logger.info(f"   📍 Location Keys: {location_keys}")
            logger.info(f"   🍽️ Restaurant Keys: {restaurant_keys}")
            logger.info(f"   🚇 Route Keys: {route_keys}")
            logger.info(f"   💾 Memory Usage: {info.get('used_memory_human', 'N/A')}")
            logger.info(f"   🔥 Cache Hit Rate: {info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)) * 100:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Cache status check failed: {e}")
    
    async def run_full_warming(self):
        """Run complete cache warming process"""
        logger.info("🔥 Starting comprehensive cache warming...")
        
        await self.check_cache_status()
        
        # Run all warming tasks
        tasks = [
            self.warm_attractions_cache(),
            self.warm_location_cache(),
            self.warm_restaurants_cache(),
            self.warm_transportation_cache()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        await self.check_cache_status()
        logger.info("🎉 Cache warming completed!")

async def main():
    """Main cache warming entry point"""
    warmer = CacheWarmer()
    await warmer.run_full_warming()

if __name__ == "__main__":
    asyncio.run(main())

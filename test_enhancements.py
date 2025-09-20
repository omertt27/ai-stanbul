#!/usr/bin/env python3
"""
Test script for Istanbul AI chatbot enhanced features.
Tests query caching, rate limiting, input sanitization, and structured logging.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8001"

async def test_enhanced_features():
    """Test all enhanced features of the Istanbul AI chatbot."""
    
    print("🧪 Testing Istanbul AI Enhanced Features")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: Basic health check
        print("1. Testing health check...")
        try:
            async with session.get(f"{BASE_URL}/health") as resp:
                if resp.status == 200:
                    print("   ✅ Health check passed")
                else:
                    print(f"   ❌ Health check failed: {resp.status}")
        except Exception as e:
            print(f"   ❌ Health check error: {e}")
        
        # Test 2: System status with enhancements
        print("\n2. Testing enhanced system status...")
        try:
            async with session.get(f"{BASE_URL}/admin/system-status") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("   ✅ Enhanced system status retrieved")
                    if data.get("success"):
                        features = data["system_status"]["features"]
                        print(f"   - Query Caching: {'✅' if features['query_caching']['active'] else '❌'}")
                        print(f"   - Rate Limiting: {'✅' if features['rate_limiting']['active'] else '❌'}")
                        print(f"   - Structured Logging: {'✅' if features['structured_logging']['active'] else '❌'}")
                        print(f"   - Input Sanitization: {'✅' if features['input_sanitization']['active'] else '❌'}")
                else:
                    print(f"   ❌ System status failed: {resp.status}")
        except Exception as e:
            print(f"   ❌ System status error: {e}")
        
        # Test 3: Cache functionality
        print("\n3. Testing query caching...")
        test_query = "What are the best restaurants in Beyoğlu?"
        
        # First request (should miss cache)
        print("   Making first request (cache miss expected)...")
        start_time = time.time()
        try:
            async with session.post(f"{BASE_URL}/ai", 
                                  json={"query": test_query, "session_id": "test_session"}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    first_response_time = time.time() - start_time
                    print(f"   ✅ First request completed in {first_response_time:.2f}s")
                    
                    # Second request (should hit cache)
                    print("   Making second request (cache hit expected)...")
                    start_time = time.time()
                    async with session.post(f"{BASE_URL}/ai", 
                                          json={"query": test_query, "session_id": "test_session"}) as resp2:
                        if resp2.status == 200:
                            data2 = await resp2.json()
                            second_response_time = time.time() - start_time
                            print(f"   ✅ Second request completed in {second_response_time:.2f}s")
                            
                            # Check if second request was faster (cache hit)
                            if second_response_time < first_response_time * 0.5:
                                print("   ✅ Cache appears to be working (faster response)")
                            else:
                                print("   ⚠️ Cache may not be working (similar response times)")
                        else:
                            print(f"   ❌ Second request failed: {resp2.status}")
                else:
                    print(f"   ❌ First request failed: {resp.status}")
        except Exception as e:
            print(f"   ❌ Cache test error: {e}")
        
        # Test 4: Input sanitization
        print("\n4. Testing input sanitization...")
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "javascript:alert(1)",
            "../../../etc/passwd"
        ]
        
        for malicious_input in malicious_inputs:
            try:
                async with session.post(f"{BASE_URL}/ai", 
                                      json={"query": malicious_input}) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Check if response indicates input was sanitized
                        response_text = data.get("message", "").lower()
                        if "can't process" in response_text or "rephrase" in response_text:
                            print(f"   ✅ Malicious input blocked: {malicious_input[:20]}...")
                        else:
                            print(f"   ⚠️ Input may not be properly sanitized: {malicious_input[:20]}...")
                    else:
                        print(f"   ❌ Sanitization test failed: {resp.status}")
            except Exception as e:
                print(f"   ❌ Sanitization test error: {e}")
        
        # Test 5: Rate limiting
        print("\n5. Testing rate limiting...")
        print("   Making rapid requests to test rate limiting...")
        rate_limit_triggered = False
        
        for i in range(35):  # Try to exceed 30/minute limit
            try:
                async with session.post(f"{BASE_URL}/ai", 
                                      json={"query": f"Test query {i}"}) as resp:
                    if resp.status == 429:
                        rate_limit_triggered = True
                        print(f"   ✅ Rate limit triggered after {i+1} requests")
                        break
                    elif resp.status != 200:
                        print(f"   ❌ Unexpected status: {resp.status}")
                        break
            except Exception as e:
                print(f"   ❌ Rate limit test error: {e}")
                break
        
        if not rate_limit_triggered:
            print("   ⚠️ Rate limiting may not be working (no 429 status received)")
        
        # Test 6: Cache statistics
        print("\n6. Testing cache statistics...")
        try:
            async with session.get(f"{BASE_URL}/admin/cache-stats") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success"):
                        print("   ✅ Cache statistics retrieved")
                        cache_stats = data["cache_stats"]
                        print(f"   - Cache type: {cache_stats.get('cache_type', 'unknown')}")
                        print(f"   - Total entries: {cache_stats.get('total_entries', 0)}")
                    else:
                        print(f"   ❌ Cache stats failed: {data.get('error')}")
                else:
                    print(f"   ❌ Cache stats request failed: {resp.status}")
        except Exception as e:
            print(f"   ❌ Cache stats error: {e}")
        
        # Test 7: Rate limit statistics
        print("\n7. Testing rate limit statistics...")
        try:
            async with session.get(f"{BASE_URL}/admin/rate-limit-stats") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("success"):
                        print("   ✅ Rate limit statistics retrieved")
                        rate_stats = data["rate_limit_stats"]
                        print(f"   - Storage backend: {rate_stats.get('storage_backend', 'unknown')}")
                        print(f"   - Memory entries: {rate_stats.get('memory_entries', 0)}")
                    else:
                        print(f"   ❌ Rate limit stats failed: {data.get('error')}")
                else:
                    print(f"   ❌ Rate limit stats request failed: {resp.status}")
        except Exception as e:
            print(f"   ❌ Rate limit stats error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Enhanced features testing completed!")
    print("\nTo start the server for testing:")
    print("cd /Users/omer/Desktop/ai-stanbul/backend")
    print("/Users/omer/Desktop/ai-stanbul/.venv/bin/python main.py")

if __name__ == "__main__":
    asyncio.run(test_enhanced_features())

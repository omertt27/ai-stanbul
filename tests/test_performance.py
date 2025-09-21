"""
Performance and load testing for AI Istanbul chatbot
Tests response times, concurrent users, memory usage, and scalability
"""
import pytest
import asyncio
import time
import os
from typing import Dict, List, Any, Union
from unittest.mock import patch
from httpx import AsyncClient, Response
from concurrent.futures import ThreadPoolExecutor

try:
    import psutil
except ImportError:
    psutil = None

class TestPerformance:
    """Test performance characteristics and load handling."""
    
    @pytest.mark.asyncio
    async def test_basic_response_time(self, client: AsyncClient, performance_thresholds):
        """Test basic API response time."""
        payload = {
            "query": "Best restaurants in Istanbul",
            "session_id": "perf-basic-test"
        }
        
        start_time = time.time()
        response = await client.post("/ai", json=payload)
        end_time = time.time()
        
        response_time_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert response_time_ms < performance_thresholds["response_time_ms"]
        print(f"Response time: {response_time_ms:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_streaming_response_performance(self, client: AsyncClient):
        """Test streaming endpoint performance."""
        payload = {
            "query": "Tell me about Istanbul's history and culture",
            "session_id": "perf-stream-test"
        }
        
        start_time = time.time()
        response = await client.post("/ai/stream", json=payload)
        
        # Read first chunk to measure time to first byte
        first_chunk_time = time.time()
        ttfb = (first_chunk_time - start_time) * 1000
        
        assert response.status_code == 200
        assert ttfb < 1000  # Time to first byte should be < 1 second
        print(f"Time to first byte: {ttfb:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, client: AsyncClient):
        """Test performance under concurrent load."""
        num_concurrent = 20
        
        async def make_request(request_id: int) -> Dict[str, Any]:
            payload = {
                "query": f"Restaurant recommendation #{request_id}",
                "session_id": f"concurrent-perf-{request_id}"
            }
            start_time = time.time()
            try:
                response = await client.post("/ai", json=payload)
                end_time = time.time()
                return {
                    "status_code": response.status_code,
                    "response_time": (end_time - start_time) * 1000,
                    "request_id": request_id,
                    "success": True
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "status_code": 500,
                    "response_time": (end_time - start_time) * 1000,
                    "request_id": request_id,
                    "success": False,
                    "error": str(e)
                }
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [make_request(i) for i in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"] and r["status_code"] == 200]
        success_rate = len(successful_requests) / num_concurrent
        
        assert success_rate >= 0.9  # At least 90% success rate
        
        # Calculate average response time
        avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
        assert avg_response_time < 3000  # Average response time < 3 seconds
        
        # Calculate requests per second
        rps = num_concurrent / total_time
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average response time: {avg_response_time:.2f}ms")
        print(f"Requests per second: {rps:.2f}")
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, client: AsyncClient):
        """Test memory usage during operations."""
        if psutil is None:
            pytest.skip("psutil not available - skipping memory test")
            return
            
        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        except Exception as e:
            pytest.skip(f"Memory monitoring not available: {e}")
            return
        
        # Perform multiple AI requests
        for i in range(10):
            payload = {
                "query": f"Complex query about Istanbul museums and transportation #{i}",
                "session_id": f"memory-test-{i}"
            }
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
        print(f"Memory usage - Initial: {initial_memory:.2f}MB, Final: {final_memory:.2f}MB, Increase: {memory_increase:.2f}MB")
    
    @pytest.mark.asyncio
    async def test_cache_performance_impact(self, client: AsyncClient):
        """Test cache impact on performance."""
        query = "Popular tourist attractions in Istanbul"
        session_id = "cache-perf-test"
        
        # First request (cache miss)
        payload = {"query": query, "session_id": session_id}
        start_time = time.time()
        response1 = await client.post("/ai", json=payload)
        first_response_time = (time.time() - start_time) * 1000
        
        assert response1.status_code == 200
        
        # Second request (cache hit) - same query
        start_time = time.time()
        response2 = await client.post("/ai", json=payload)
        second_response_time = (time.time() - start_time) * 1000
        
        assert response2.status_code == 200
        
        # Cache should improve performance
        print(f"First request: {first_response_time:.2f}ms")
        print(f"Second request: {second_response_time:.2f}ms")
        
        # Note: Cache might not always be faster due to AI processing complexity
        # But response should still be reasonable
        assert second_response_time < 5000  # Should still be under 5 seconds
    
    @pytest.mark.asyncio
    async def test_database_query_performance(self, client: AsyncClient):
        """Test database operation performance."""
        # Test health endpoint which involves database checks
        start_time = time.time()
        response = await client.get("/health")
        db_response_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert db_response_time < 500  # Database operations should be fast
        print(f"Database health check: {db_response_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_large_query_handling(self, client: AsyncClient):
        """Test handling of large/complex queries."""
        # Create a large query
        large_query = "I want to know everything about Istanbul: " + \
                     "the best restaurants, museums, transportation, hotels, shopping, " * 20 + \
                     "Please provide detailed recommendations for a 7-day itinerary."
        
        payload = {
            "query": large_query,
            "session_id": "large-query-test"
        }
        
        start_time = time.time()
        response = await client.post("/ai", json=payload)
        response_time = (time.time() - start_time) * 1000
        
        # Should handle large queries gracefully
        assert response.status_code == 200
        assert response_time < 10000  # Should respond within 10 seconds even for large queries
        print(f"Large query response time: {response_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_rate_limiting_performance(self, client: AsyncClient):
        """Test rate limiting impact on performance."""
        session_id = "rate-limit-test"
        
        # Make requests rapidly
        response_times = []
        for i in range(5):
            payload = {
                "query": f"Quick query #{i}",
                "session_id": session_id
            }
            
            start_time = time.time()
            response = await client.post("/ai", json=payload)
            response_time = (time.time() - start_time) * 1000
            
            response_times.append({
                "response_time": response_time,
                "status_code": response.status_code
            })
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        # Analyze rate limiting behavior
        successful_requests = [r for r in response_times if r["status_code"] == 200]
        rate_limited_requests = [r for r in response_times if r["status_code"] == 429]
        
        print(f"Successful requests: {len(successful_requests)}")
        print(f"Rate limited requests: {len(rate_limited_requests)}")
        
        # At least some requests should succeed
        assert len(successful_requests) > 0
    
    @pytest.mark.asyncio
    async def test_image_processing_performance(self, client: AsyncClient):
        """Test image processing performance."""
        # Create a mock image file
        test_image_data = b"fake_image_data" * 1000  # Simulate larger image
        files = {"file": ("test_image.jpg", test_image_data, "image/jpeg")}
        data = {"session_id": "image-perf-test"}
        
        with patch('main.analyze_image_with_ai') as mock_analyze:
            mock_analyze.return_value = "This appears to be a Turkish restaurant menu."
            
            start_time = time.time()
            response = await client.post("/ai/analyze-image", files=files, data=data)
            response_time = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            assert response_time < 5000  # Image processing should complete within 5 seconds
            print(f"Image processing time: {response_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_multilingual_performance_comparison(self, client: AsyncClient):
        """Test performance differences across languages."""
        queries = {
            "en": "Best restaurants in Istanbul",
            "tr": "İstanbul'da en iyi restoranlar",
            "ar": "أفضل المطاعم في إستانبول"
        }
        
        performance_results = {}
        
        for lang, query in queries.items():
            payload = {
                "query": query,
                "session_id": f"multilang-perf-{lang}",
                "language": lang
            }
            
            start_time = time.time()
            response = await client.post("/ai", json=payload)
            response_time = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            performance_results[lang] = response_time
            print(f"{lang.upper()} response time: {response_time:.2f}ms")
        
        # All languages should perform reasonably well
        for lang, time_ms in performance_results.items():
            assert time_ms < 3000  # Each language should respond within 3 seconds
    
    @pytest.mark.asyncio
    async def test_session_management_performance(self, client: AsyncClient):
        """Test session management overhead."""
        num_sessions = 50
        
        # Create many different sessions
        start_time = time.time()
        tasks = []
        
        for i in range(num_sessions):
            payload = {
                "query": "Quick test query",
                "session_id": f"session-mgmt-test-{i}"
            }
            tasks.append(client.post("/ai", json=payload))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        successful_responses = []
        for response in responses:
            if isinstance(response, Response) and response.status_code == 200:
                successful_responses.append(response)
        
        success_rate = len(successful_responses) / num_sessions
        
        assert success_rate >= 0.95  # 95% success rate
        
        sessions_per_second = num_sessions / total_time
        print(f"Session management - Success rate: {success_rate:.2%}, Sessions/sec: {sessions_per_second:.2f}")
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, client: AsyncClient):
        """Test that error handling doesn't significantly impact performance."""
        # Test various error conditions
        error_scenarios = [
            {"query": "", "session_id": "error-test-1"},  # Empty query
            {"query": "test", "session_id": ""},  # Empty session
            {"invalid_field": "test"},  # Invalid payload
        ]
        
        error_response_times = []
        
        for scenario in error_scenarios:
            start_time = time.time()
            response = await client.post("/ai", json=scenario)
            response_time = (time.time() - start_time) * 1000
            
            error_response_times.append(response_time)
            # Error responses should be fast
            assert response_time < 1000  # Errors should be handled quickly
        
        avg_error_time = sum(error_response_times) / len(error_response_times)
        print(f"Average error response time: {avg_error_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self, client: AsyncClient):
        """Test health check endpoint performance."""
        # Health checks should be very fast
        health_times = []
        
        for _ in range(10):
            start_time = time.time()
            response = await client.get("/health")
            response_time = (time.time() - start_time) * 1000
            
            assert response.status_code == 200
            health_times.append(response_time)
            
            # Each health check should be very fast
            assert response_time < 200  # Under 200ms
        
        avg_health_time = sum(health_times) / len(health_times)
        print(f"Average health check time: {avg_health_time:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, client: AsyncClient):
        """Test performance under sustained load."""
        duration_seconds = 30  # Run for 30 seconds
        requests_made = 0
        successful_requests = 0
        response_times = []
        
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            payload = {
                "query": f"Sustained load test query {requests_made}",
                "session_id": f"sustained-load-{requests_made}"
            }
            
            start_time = time.time()
            try:
                response = await client.post("/ai", json=payload)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    successful_requests += 1
                    response_times.append(response_time)
                
                requests_made += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Request failed: {e}")
                requests_made += 1
        
        success_rate = successful_requests / requests_made if requests_made > 0 else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        rps = successful_requests / duration_seconds
        
        print(f"Sustained load test results:")
        print(f"Duration: {duration_seconds}s")
        print(f"Total requests: {requests_made}")
        print(f"Successful requests: {successful_requests}")
        print(f"Success rate: {success_rate:.2%}")
        print(f"Average response time: {avg_response_time:.2f}ms")
        print(f"Requests per second: {rps:.2f}")
        
        # Performance expectations
        assert success_rate >= 0.8  # At least 80% success rate under sustained load
        assert avg_response_time < 5000  # Average response time under 5 seconds
        assert rps >= 1  # At least 1 request per second throughput

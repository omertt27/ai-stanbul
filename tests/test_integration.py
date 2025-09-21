"""
End-to-end integration tests for AI Istanbul chatbot
Tests complete user journeys and system integration
"""
import pytest
import asyncio
import json
from typing import List, Dict, Any, Union
from unittest.mock import patch, MagicMock
from httpx import AsyncClient

class TestIntegration:
    """Test complete user journeys and system integration."""
    
    @pytest.mark.asyncio
    async def test_complete_tourist_journey(self, client: AsyncClient):
        """Test a complete tourist interaction flow."""
        session_id = "tourist-journey-test"
        
        # Step 1: Tourist greets the system
        greeting_payload = {
            "query": "Hello, I'm visiting Istanbul for the first time",
            "session_id": session_id
        }
        
        response1 = await client.post("/ai", json=greeting_payload)
        assert response1.status_code == 200
        data1 = response1.json()
        assert "welcome" in data1["response"].lower() or "hello" in data1["response"].lower()
        
        # Step 2: Ask for restaurant recommendations
        restaurant_payload = {
            "query": "Can you recommend some traditional Turkish restaurants?",
            "session_id": session_id
        }
        
        response2 = await client.post("/ai", json=restaurant_payload)
        assert response2.status_code == 200
        data2 = response2.json()
        assert any(keyword in data2["response"].lower() for keyword in ["restaurant", "turkish", "traditional"])
        
        # Step 3: Ask about transportation
        transport_payload = {
            "query": "How do I get around the city?",
            "session_id": session_id
        }
        
        response3 = await client.post("/ai", json=transport_payload)
        assert response3.status_code == 200
        data3 = response3.json()
        assert any(keyword in data3["response"].lower() for keyword in ["metro", "bus", "tram", "taxi"])
        
        # Step 4: Provide feedback
        feedback_payload = {
            "session_id": session_id,
            "rating": 5,
            "feedback": "Very helpful recommendations!",
            "query": "overall experience"
        }
        
        response4 = await client.post("/feedback", json=feedback_payload)
        assert response4.status_code == 200
        data4 = response4.json()
        assert data4["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_multilingual_conversation_flow(self, client: AsyncClient):
        """Test conversation flow switching between languages."""
        session_id = "multilingual-flow-test"
        
        # Start in English
        payload1 = {
            "query": "I want to visit museums in Istanbul",
            "session_id": session_id,
            "language": "en"
        }
        
        response1 = await client.post("/ai", json=payload1)
        assert response1.status_code == 200
        
        # Continue in Turkish
        payload2 = {
            "query": "Hangi müze en popüler?",  # Which museum is most popular?
            "session_id": session_id
        }
        
        response2 = await client.post("/ai", json=payload2)
        assert response2.status_code == 200
        data2 = response2.json()
        assert "müze" in data2["response"].lower() or "museum" in data2["response"].lower()
        
        # Switch to Arabic
        payload3 = {
            "query": "كم تكلفة دخول المتحف؟",  # How much does museum entry cost?
            "session_id": session_id
        }
        
        response3 = await client.post("/ai", json=payload3)
        assert response3.status_code == 200
        # Should maintain context across languages
    
    @pytest.mark.asyncio
    async def test_image_to_recommendation_flow(self, client: AsyncClient):
        """Test flow from image analysis to recommendations."""
        session_id = "image-recommendation-test"
        
        # Step 1: Upload a menu image
        test_image_data = b"fake_menu_image_data"
        files = {"file": ("menu.jpg", test_image_data, "image/jpeg")}
        data = {"session_id": session_id}
        
        with patch('main.analyze_menu_with_ai') as mock_analyze:
            mock_analyze.return_value = {
                "dishes": ["Kebab", "Baklava", "Turkish Coffee"],
                "recommendations": "Try the lamb kebab",
                "price_range": "moderate"
            }
            
            response1 = await client.post("/ai/analyze-menu", files=files, data=data)
            assert response1.status_code == 200
            menu_data = response1.json()
            assert "dishes" in menu_data
        
        # Step 2: Ask for similar restaurants
        payload2 = {
            "query": "Can you recommend restaurants with similar food?",
            "session_id": session_id
        }
        
        response2 = await client.post("/ai", json=payload2)
        assert response2.status_code == 200
        data2 = response2.json()
        # Should consider the previously analyzed menu
        assert any(keyword in data2["response"].lower() for keyword in ["kebab", "similar", "restaurant"])
    
    @pytest.mark.asyncio
    async def test_real_time_data_integration_flow(self, client: AsyncClient):
        """Test integration of real-time data into recommendations."""
        session_id = "realtime-integration-test"
        
        # Mock real-time data
        with patch('main.get_real_time_istanbul_data') as mock_data:
            mock_data.return_value = {
                "weather": {
                    "temperature": 15,
                    "condition": "rainy",
                    "humidity": 80
                },
                "traffic": {"status": "heavy"},
                "events": ["Art Exhibition at Modern Art Museum"]
            }
            
            # Ask for activity recommendations
            payload = {
                "query": "What should I do today in Istanbul?",
                "session_id": session_id
            }
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            # Should consider weather in recommendations
            response_text = data["response"].lower()
            # With rainy weather, should suggest indoor activities
            assert any(keyword in response_text for keyword in ["indoor", "museum", "mall", "covered"])
    
    @pytest.mark.asyncio
    async def test_personalization_learning_flow(self, client: AsyncClient):
        """Test that system learns user preferences over time."""
        session_id = "personalization-test"
        
        # Step 1: User expresses preferences
        payload1 = {
            "query": "I'm vegetarian and love art",
            "session_id": session_id
        }
        
        response1 = await client.post("/ai", json=payload1)
        assert response1.status_code == 200
        
        # Step 2: Ask for restaurant recommendations
        payload2 = {
            "query": "Where should I eat?",
            "session_id": session_id
        }
        
        response2 = await client.post("/ai", json=payload2)
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Should consider vegetarian preference
        response_text = data2["response"].lower()
        assert any(keyword in response_text for keyword in ["vegetarian", "vegan", "plant"])
        
        # Step 3: Ask for activity recommendations  
        payload3 = {
            "query": "What should I visit?",
            "session_id": session_id
        }
        
        response3 = await client.post("/ai", json=payload3)
        assert response3.status_code == 200
        data3 = response3.json()
        
        # Should consider art interest
        response_text = data3["response"].lower()
        assert any(keyword in response_text for keyword in ["art", "gallery", "museum"])
    
    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, client: AsyncClient):
        """Test system recovery from errors in conversation flow."""
        session_id = "error-recovery-test"
        
        # Step 1: Normal query
        payload1 = {
            "query": "Best restaurants in Istanbul",
            "session_id": session_id
        }
        
        response1 = await client.post("/ai", json=payload1)
        assert response1.status_code == 200
        
        # Step 2: Problematic query
        payload2 = {
            "query": "",  # Empty query
            "session_id": session_id
        }
        
        response2 = await client.post("/ai", json=payload2)
        # Should handle error gracefully
        assert response2.status_code in [400, 422]  # Validation error
        
        # Step 3: Recovery with valid query
        payload3 = {
            "query": "Can you tell me about Turkish cuisine?",
            "session_id": session_id
        }
        
        response3 = await client.post("/ai", json=payload3)
        assert response3.status_code == 200
        data3 = response3.json()
        
        # Should recover and provide normal response
        assert "turkish" in data3["response"].lower()
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance_integration(self, client: AsyncClient):
        """Test GDPR compliance throughout user journey."""
        session_id = "gdpr-integration-test"
        email = "user@example.com"
        
        # Step 1: User provides consent
        consent_payload = {
            "session_id": session_id,
            "consent_given": True,
            "consent_types": ["essential", "analytics"]
        }
        
        response1 = await client.post("/gdpr/consent", json=consent_payload)
        assert response1.status_code == 200
        
        # Step 2: Normal interaction
        ai_payload = {
            "query": "Best places to visit in Istanbul",
            "session_id": session_id
        }
        
        response2 = await client.post("/ai", json=ai_payload)
        assert response2.status_code == 200
        
        # Step 3: Data export request
        export_payload = {
            "session_id": session_id,
            "email": email,
            "request_type": "data_export"
        }
        
        response3 = await client.post("/gdpr/data-request", json=export_payload)
        assert response3.status_code == 200
        data3 = response3.json()
        assert "request_id" in data3
        
        # Step 4: Data deletion
        deletion_payload = {
            "session_id": session_id,
            "email": email,
            "confirmation_token": "test-token"
        }
        
        response4 = await client.post("/gdpr/data-deletion", json=deletion_payload)
        assert response4.status_code == 200
    
    @pytest.mark.asyncio
    async def test_caching_and_performance_integration(self, client: AsyncClient):
        """Test caching behavior in realistic usage patterns."""
        session_id = "cache-integration-test"
        
        # Repeat common queries to test caching
        common_queries = [
            "Best restaurants in Sultanahmet",
            "How to get to Hagia Sophia",
            "Istanbul museum opening hours"
        ]
        
        response_times = {}
        
        for query in common_queries:
            # First request (cache miss)
            payload = {"query": query, "session_id": session_id}
            
            import time
            start_time = time.time()
            response1 = await client.post("/ai", json=payload)
            first_time = time.time() - start_time
            
            assert response1.status_code == 200
            
            # Second request (potential cache hit)
            start_time = time.time()
            response2 = await client.post("/ai", json=payload)
            second_time = time.time() - start_time
            
            assert response2.status_code == 200
            
            response_times[query] = {
                "first": first_time * 1000,
                "second": second_time * 1000
            }
        
        # Check cache statistics
        cache_response = await client.get("/ai/cache-stats")
        assert cache_response.status_code == 200
        cache_data = cache_response.json()
        assert "hit_rate" in cache_data
    
    @pytest.mark.asyncio
    async def test_concurrent_users_integration(self, client: AsyncClient):
        """Test realistic concurrent user scenarios."""
        num_users = 10
        
        async def simulate_user_session(user_id):
            session_id = f"concurrent-user-{user_id}"
            
            # Each user asks different types of questions
            queries = [
                f"Hello, I'm user {user_id}",
                "Best restaurants for dinner",
                "How to get around Istanbul",
                "Museums to visit"
            ]
            
            results = []
            for query in queries:
                payload = {
                    "query": query,
                    "session_id": session_id
                }
                
                response = await client.post("/ai", json=payload)
                results.append({
                    "user_id": user_id,
                    "query": query,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                })
                
                # Small delay between user queries
                await asyncio.sleep(0.5)
            
            return results
        
        # Run concurrent user sessions
        tasks = [simulate_user_session(i) for i in range(num_users)]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        total_queries = 0
        successful_queries = 0
        
        for user_results in all_results:
            if isinstance(user_results, list):  # Ensure it's a list, not an Exception
                for result in user_results:
                    total_queries += 1
                    if isinstance(result, dict) and result.get("success", False):
                        successful_queries += 1
        
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        assert success_rate >= 0.9  # 90% success rate with concurrent users
    
    @pytest.mark.asyncio
    async def test_mobile_vs_web_integration(self, client: AsyncClient):
        """Test different client types (mobile vs web)."""
        base_session = "client-type-test"
        
        # Simulate web client
        web_payload = {
            "query": "Best restaurants near me",
            "session_id": f"{base_session}-web",
            "client_type": "web",
            "user_agent": "Mozilla/5.0 (compatible web browser)"
        }
        
        web_response = await client.post("/ai", json=web_payload)
        assert web_response.status_code == 200
        
        # Simulate mobile client
        mobile_payload = {
            "query": "Best restaurants near me",
            "session_id": f"{base_session}-mobile",
            "client_type": "mobile",
            "user_agent": "Istanbul AI Mobile App v1.0"
        }
        
        mobile_response = await client.post("/ai", json=mobile_payload)
        assert mobile_response.status_code == 200
        
        # Both should work, potentially with different optimizations
        web_data = web_response.json()
        mobile_data = mobile_response.json()
        
        assert "response" in web_data
        assert "response" in mobile_data
    
    @pytest.mark.asyncio
    async def test_analytics_integration(self, client: AsyncClient):
        """Test analytics data collection during user interactions."""
        session_id = "analytics-integration-test"
        
        # Perform various interactions
        interactions = [
            {"type": "greeting", "query": "Hello"},
            {"type": "restaurant", "query": "Best Turkish restaurants"},
            {"type": "museum", "query": "Museum recommendations"},
            {"type": "feedback", "rating": 5}
        ]
        
        for interaction in interactions:
            if interaction["type"] == "feedback":
                payload = {
                    "session_id": session_id,
                    "rating": interaction["rating"],
                    "feedback": "Great service"
                }
                response = await client.post("/feedback", json=payload)
            else:
                payload = {
                    "query": interaction["query"],
                    "session_id": session_id
                }
                response = await client.post("/ai", json=payload)
            
            assert response.status_code == 200
        
        # Analytics should be collected (this would typically be verified 
        # through analytics dashboards or database queries in real implementation)
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring_integration(self, client: AsyncClient):
        """Test health monitoring throughout system usage."""
        # Check initial health
        health_response = await client.get("/health")
        assert health_response.status_code == 200
        initial_health = health_response.json()
        
        # Perform some system operations
        for i in range(5):
            payload = {
                "query": f"System load test query {i}",
                "session_id": f"health-monitoring-{i}"
            }
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
        
        # Check health after operations
        health_response2 = await client.get("/health")
        assert health_response2.status_code == 200
        final_health = health_response2.json()
        
        # System should remain healthy
        assert final_health["status"] == "healthy"
        
        # Check cache stats
        cache_response = await client.get("/ai/cache-stats")
        assert cache_response.status_code == 200

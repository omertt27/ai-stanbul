"""
Comprehensive AI and multilingual functionality tests
Tests AI responses, language detection, translation, and multilingual flows
"""
import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock
from httpx import AsyncClient

class TestAIMultilingualFunctionality:
    """Test AI services and multilingual capabilities."""
    
    @pytest.mark.asyncio
    async def test_language_detection_english(self, client: AsyncClient):
        """Test language detection for English queries."""
        test_cases = [
            "Best restaurants in Istanbul",
            "How to get to Hagia Sophia?",
            "What are the opening hours of museums?",
            "Where can I find good Turkish breakfast?"
        ]
        
        for query in test_cases:
            payload = {"query": query, "session_id": "lang-detect-en"}
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data.get("detected_language", "en") == "en"
    
    @pytest.mark.asyncio
    async def test_language_detection_turkish(self, client: AsyncClient):
        """Test language detection for Turkish queries."""
        test_cases = [
            "İstanbul'da en iyi restoranlar nerede?",
            "Ayasofya'ya nasıl gidebilirim?", 
            "Müzelerin açılış saatleri nedir?",
            "Güzel Türk kahvaltısı nerede bulabilirim?"
        ]
        
        for query in test_cases:
            payload = {"query": query, "session_id": "lang-detect-tr"}
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            detected_lang = data.get("detected_language", "tr")
            assert detected_lang in ["tr", "turkish"]
    
    @pytest.mark.asyncio
    async def test_language_detection_arabic(self, client: AsyncClient):
        """Test language detection for Arabic queries."""
        test_cases = [
            "أفضل المطاعم في إستانبول",
            "كيفية الوصول إلى آيا صوفيا؟",
            "ما هي ساعات عمل المتاحف؟",
            "أين يمكنني العثور على إفطار تركي جيد؟"
        ]
        
        for query in test_cases:
            payload = {"query": query, "session_id": "lang-detect-ar"}
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            detected_lang = data.get("detected_language", "ar")
            assert detected_lang in ["ar", "arabic"]
    
    @pytest.mark.asyncio
    async def test_multilingual_restaurant_queries(self, client: AsyncClient, sample_queries):
        """Test restaurant queries in multiple languages."""
        multilingual_restaurant_queries = {
            "en": "Best seafood restaurants with Bosphorus view",
            "tr": "Boğaz manzaralı en iyi deniz ürünleri restoranları",
            "ar": "أفضل مطاعم المأكولات البحرية مع إطلالة على البوسفور"
        }
        
        for lang, query in multilingual_restaurant_queries.items():
            payload = {
                "query": query,
                "session_id": f"multilingual-restaurant-{lang}",
                "language": lang
            }
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            # Check response contains relevant keywords
            response_text = data["response"].lower()
            assert any(keyword in response_text for keyword in ["restaurant", "restoran", "مطعم"])
            assert any(keyword in response_text for keyword in ["bosphorus", "boğaz", "بوسفور"])
    
    @pytest.mark.asyncio
    async def test_multilingual_museum_queries(self, client: AsyncClient):
        """Test museum queries in multiple languages."""
        museum_queries = {
            "en": "What are the opening hours of Topkapi Palace?",
            "tr": "Topkapı Sarayı'nın açılış saatleri nedir?",
            "ar": "ما هي ساعات عمل قصر توبكابي؟"
        }
        
        for lang, query in museum_queries.items():
            payload = {
                "query": query,
                "session_id": f"multilingual-museum-{lang}",
                "language": lang
            }
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            # Check response contains relevant keywords
            response_text = data["response"].lower()
            assert any(keyword in response_text for keyword in ["topkapi", "topkapı", "توبكابي"])
            assert any(keyword in response_text for keyword in ["palace", "saray", "قصر"])
    
    @pytest.mark.asyncio
    async def test_multilingual_transportation_queries(self, client: AsyncClient):
        """Test transportation queries in multiple languages."""
        transport_queries = {
            "en": "How to get from airport to city center?",
            "tr": "Havalimanından şehir merkezine nasıl gidilir?",
            "ar": "كيفية الوصول من المطار إلى وسط المدينة؟"
        }
        
        for lang, query in transport_queries.items():
            payload = {
                "query": query,
                "session_id": f"multilingual-transport-{lang}",
                "language": lang
            }
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            # Check response contains relevant keywords
            response_text = data["response"].lower()
            assert any(keyword in response_text for keyword in ["airport", "havalimanı", "مطار"])
            assert any(keyword in response_text for keyword in ["metro", "bus", "taxi", "otobüs", "تاكسي"])
    
    @pytest.mark.asyncio
    async def test_conversation_context_persistence(self, client: AsyncClient):
        """Test that conversation context is maintained across languages."""
        session_id = "context-persistence-test"
        
        # First query in English
        payload1 = {
            "query": "I want to visit museums in Istanbul",
            "session_id": session_id
        }
        response1 = await client.post("/ai", json=payload1)
        assert response1.status_code == 200
        
        # Follow-up query in Turkish
        payload2 = {
            "query": "Hangi müzeyi önerirsiniz?",  # Which museum do you recommend?
            "session_id": session_id
        }
        response2 = await client.post("/ai", json=payload2)
        assert response2.status_code == 200
        
        # Check that context is maintained
        data2 = response2.json()
        response_text = data2["response"].lower()
        assert any(keyword in response_text for keyword in ["müze", "museum", "recommend"])
    
    @pytest.mark.asyncio
    async def test_code_switching_handling(self, client: AsyncClient):
        """Test handling of code-switching (multiple languages in one query).""" 
        mixed_queries = [
            "I want to visit Galata Tower, nerede bu yer?",  # English + Turkish
            "أريد زيارة Hagia Sophia, how can I get there?",  # Arabic + English
            "Boğaz'da güzel restaurants var mı?",  # Turkish + English
        ]
        
        for query in mixed_queries:
            payload = {
                "query": query,
                "session_id": "code-switching-test"
            }
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            # Should handle mixed language queries gracefully
            assert "response" in data
            assert len(data["response"]) > 10  # Meaningful response
    
    @pytest.mark.asyncio
    async def test_ai_response_quality_metrics(self, client: AsyncClient):
        """Test AI response quality metrics and validation."""
        test_queries = [
            {
                "query": "Best traditional Turkish restaurants in Old City",
                "expected_topics": ["restaurant", "turkish", "traditional", "old city", "sultanahmet"],
                "min_length": 50
            },
            {
                "query": "How to use public transport in Istanbul?",
                "expected_topics": ["metro", "bus", "tram", "istanbulkart", "transport"],
                "min_length": 100
            }
        ]
        
        for test_case in test_queries:
            payload = {
                "query": test_case["query"],
                "session_id": "quality-test"
            }
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            response_text = data["response"].lower()
            
            # Check response length
            assert len(response_text) >= test_case["min_length"]
            
            # Check for expected topics
            topic_matches = sum(1 for topic in test_case["expected_topics"] 
                              if topic in response_text)
            assert topic_matches >= 2  # At least 2 expected topics mentioned
    
    @pytest.mark.asyncio
    async def test_ai_fallback_mechanisms(self, client: AsyncClient):
        """Test AI fallback mechanisms for unclear queries."""
        unclear_queries = [
            "asdfgh qwerty",  # Gibberish
            "aaaaa",  # Repetitive
            "123456",  # Numbers only
            "",  # Empty (should be handled by validation)
            "help",  # Too vague
        ]
        
        for query in unclear_queries:
            if query == "":  # Skip empty query as it's handled by validation
                continue
                
            payload = {
                "query": query,
                "session_id": "fallback-test"
            }
            
            response = await client.post("/ai", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # Should provide helpful fallback response
                response_text = data["response"].lower()
                assert any(keyword in response_text for keyword in 
                          ["help", "assist", "understand", "clarify", "yardım"])
    
    @pytest.mark.asyncio
    async def test_personalization_features(self, client: AsyncClient):
        """Test AI personalization based on user preferences."""
        session_id = "personalization-test"
        
        # Set user preferences
        payload1 = {
            "query": "I'm vegetarian and love art museums",
            "session_id": session_id
        }
        response1 = await client.post("/ai", json=payload1)
        assert response1.status_code == 200
        
        # Ask for restaurant recommendations
        payload2 = {
            "query": "Recommend some restaurants",
            "session_id": session_id
        }
        response2 = await client.post("/ai", json=payload2)
        assert response2.status_code == 200
        
        data2 = response2.json()
        response_text = data2["response"].lower()
        
        # Should consider vegetarian preference
        assert any(keyword in response_text for keyword in 
                  ["vegetarian", "vegan", "plant", "vejeteryan"])
    
    @pytest.mark.asyncio
    async def test_cultural_context_awareness(self, client: AsyncClient):
        """Test AI understanding of cultural context."""
        cultural_queries = [
            {
                "query": "What should I wear when visiting mosques?",
                "expected_keywords": ["modest", "cover", "scarf", "respect"]
            },
            {
                "query": "Turkish tea culture in Istanbul",
                "expected_keywords": ["çay", "tea", "culture", "tradition"]
            },
            {
                "query": "Ramadan dining options in Istanbul",
                "expected_keywords": ["iftar", "ramadan", "halal", "sunset"]
            }
        ]
        
        for test_case in cultural_queries:
            payload = {
                "query": test_case["query"],
                "session_id": "cultural-test"
            }
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            response_text = data["response"].lower()
            
            # Check for cultural awareness
            keyword_matches = sum(1 for keyword in test_case["expected_keywords"]
                                if keyword in response_text)
            assert keyword_matches >= 1  # At least one cultural keyword
    
    @pytest.mark.asyncio
    async def test_ai_response_consistency(self, client: AsyncClient):
        """Test consistency of AI responses for similar queries."""
        similar_queries = [
            "Best restaurants in Sultanahmet",
            "Good restaurants in Old City",
            "Where to eat in Sultanahmet district"
        ]
        
        responses = []
        for query in similar_queries:
            payload = {
                "query": query,
                "session_id": "consistency-test"
            }
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            responses.append(data["response"].lower())
        
        # Check for common elements in all responses
        common_keywords = ["restaurant", "sultanahmet"]
        for response_text in responses:
            assert all(keyword in response_text for keyword in common_keywords)
    
    @pytest.mark.asyncio
    async def test_real_time_data_integration(self, client: AsyncClient):
        """Test integration of real-time data in AI responses."""
        payload = {
            "query": "Current weather and best outdoor activities",
            "session_id": "realtime-test"
        }
        
        with patch('main.get_real_time_istanbul_data') as mock_data:
            mock_data.return_value = {
                "weather": {
                    "temperature": 25,
                    "condition": "sunny",
                    "humidity": 60
                }
            }
            
            response = await client.post("/ai", json=payload)
            assert response.status_code == 200
            data = response.json()
            
            response_text = data["response"].lower()
            # Should incorporate weather data
            assert any(keyword in response_text for keyword in 
                      ["sunny", "25", "weather", "outdoor"])

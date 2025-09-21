"""
Comprehensive tests for Multimodal AI Service
Tests image analysis, menu analysis, and multimodal AI capabilities
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List
import io

class TestMultimodalAIService:
    """Test Multimodal AI Service functionality."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for testing."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def multimodal_service(self, mock_openai_client):
        """Create multimodal AI service instance for testing."""
        with patch('backend.api_clients.multimodal_ai.AsyncOpenAI', return_value=mock_openai_client):
            from backend.api_clients.multimodal_ai import MultimodalAIService
            service = MultimodalAIService()
            return service
    
    @pytest.fixture
    def sample_image_data(self):
        """Sample image data for testing."""
        return b"fake_image_data_for_testing"
    
    @pytest.fixture
    def sample_menu_image(self):
        """Sample menu image data for testing."""
        return b"fake_menu_image_data"
    
    def test_service_initialization(self, multimodal_service):
        """Test multimodal AI service initializes correctly."""
        assert multimodal_service is not None
        assert hasattr(multimodal_service, 'analyze_image_comprehensive')
        assert hasattr(multimodal_service, 'analyze_menu_image')
    
    @pytest.mark.asyncio
    async def test_analyze_image_comprehensive_success(self, multimodal_service, mock_openai_client, sample_image_data):
        """Test successful comprehensive image analysis."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "detected_objects": ["building", "architecture", "landmark"],
            "location_suggestions": ["Hagia Sophia, Istanbul"],
            "landmarks_identified": ["Hagia Sophia"],
            "scene_description": "Historic Byzantine cathedral in Istanbul",
            "confidence_score": 0.95,
            "recommendations": ["Visit during golden hour for best photos"],
            "is_food_image": false,
            "is_location_image": true,
            "extracted_text": "Hagia Sophia Museum"
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = await multimodal_service.analyze_image_comprehensive(
            sample_image_data, 
            "What landmark is this in Istanbul?"
        )
        
        assert result is not None
        assert result.detected_objects == ["building", "architecture", "landmark"]
        assert result.location_suggestions == ["Hagia Sophia, Istanbul"]
        assert result.landmarks_identified == ["Hagia Sophia"]
        assert result.confidence_score == 0.95
        assert result.is_location_image is True
        assert result.is_food_image is False
    
    @pytest.mark.asyncio
    async def test_analyze_menu_image_success(self, multimodal_service, mock_openai_client, sample_menu_image):
        """Test successful menu image analysis."""
        # Mock OpenAI response for menu analysis
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "detected_items": [
                {"name": "Döner Kebab", "price": 25.0, "description": "Traditional Turkish kebab"},
                {"name": "Baklava", "price": 15.0, "description": "Sweet pastry with nuts"},
                {"name": "Turkish Tea", "price": 8.0, "description": "Traditional black tea"}
            ],
            "cuisine_type": "Turkish",
            "price_range": [8.0, 25.0],
            "recommendations": ["Try the döner kebab with ayran", "Baklava is perfect for dessert"],
            "dietary_info": {
                "vegetarian": true,
                "vegan": false,
                "halal": true,
                "gluten_free": false
            },
            "confidence_score": 0.88
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = await multimodal_service.analyze_menu_image(
            sample_menu_image,
            dietary_restrictions="vegetarian"
        )
        
        assert result is not None
        assert len(result.detected_items) == 3
        assert result.cuisine_type == "Turkish"
        assert result.price_range == (8.0, 25.0)
        assert result.confidence_score == 0.88
        assert result.dietary_info["vegetarian"] is True
        assert result.dietary_info["halal"] is True
    
    @pytest.mark.asyncio
    async def test_analyze_image_food_detection(self, multimodal_service, mock_openai_client, sample_image_data):
        """Test food image detection and analysis."""
        # Mock response for food image
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "detected_objects": ["food", "plate", "restaurant"],
            "location_suggestions": ["Turkish restaurant in Sultanahmet"],
            "landmarks_identified": [],
            "scene_description": "Delicious Turkish breakfast spread with various dishes",
            "confidence_score": 0.92,
            "recommendations": ["Try the menemen and Turkish coffee"],
            "is_food_image": true,
            "is_location_image": false,
            "extracted_text": "Turkish Breakfast"
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = await multimodal_service.analyze_image_comprehensive(
            sample_image_data,
            "What food is in this image?"
        )
        
        assert result.is_food_image is True
        assert result.is_location_image is False
        assert "food" in result.detected_objects
        assert result.confidence_score == 0.92
    
    @pytest.mark.asyncio
    async def test_analyze_image_landmark_detection(self, multimodal_service, mock_openai_client, sample_image_data):
        """Test landmark detection and identification."""
        # Mock response for landmark image
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "detected_objects": ["tower", "architecture", "landmark"],
            "location_suggestions": ["Galata Tower area, Beyoğlu"],
            "landmarks_identified": ["Galata Tower"],
            "scene_description": "Historic Galata Tower with panoramic city views",
            "confidence_score": 0.97,
            "recommendations": ["Visit at sunset for amazing views", "Check opening hours"],
            "is_food_image": false,
            "is_location_image": true,
            "extracted_text": "Galata Kulesi"
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = await multimodal_service.analyze_image_comprehensive(
            sample_image_data,
            "What landmark is this?"
        )
        
        assert "Galata Tower" in result.landmarks_identified
        assert result.is_location_image is True
        assert result.confidence_score == 0.97
        assert "tower" in result.detected_objects
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_json(self, multimodal_service, mock_openai_client, sample_image_data):
        """Test error handling when AI returns invalid JSON."""
        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid JSON response from AI"
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Should handle gracefully and return default result
        result = await multimodal_service.analyze_image_comprehensive(
            sample_image_data,
            "Analyze this image"
        )
        
        # Should return a valid result object with defaults
        assert result is not None
        assert hasattr(result, 'detected_objects')
        assert hasattr(result, 'confidence_score')
    
    @pytest.mark.asyncio
    async def test_error_handling_api_failure(self, multimodal_service, mock_openai_client, sample_image_data):
        """Test error handling when OpenAI API fails."""
        # Mock API failure
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        
        # Should handle gracefully
        result = await multimodal_service.analyze_image_comprehensive(
            sample_image_data,
            "Analyze this image"
        )
        
        # Should return error result or handle gracefully
        assert result is not None
    
    def test_image_preprocessing(self, multimodal_service, sample_image_data):
        """Test image preprocessing and validation."""
        # Test image data validation
        processed = multimodal_service._preprocess_image(sample_image_data)
        assert processed is not None
        
        # Test with invalid data
        with pytest.raises((ValueError, TypeError)):
            multimodal_service._preprocess_image(None)
    
    @pytest.mark.asyncio
    async def test_menu_analysis_dietary_restrictions(self, multimodal_service, mock_openai_client, sample_menu_image):
        """Test menu analysis with specific dietary restrictions."""
        # Mock response with dietary information
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "detected_items": [
                {"name": "Vegetable Dolma", "price": 18.0, "description": "Stuffed vegetables"},
                {"name": "Hummus", "price": 12.0, "description": "Chickpea dip"},
                {"name": "Falafel", "price": 20.0, "description": "Fried chickpea balls"}
            ],
            "cuisine_type": "Mediterranean",
            "price_range": [12.0, 20.0],
            "recommendations": ["All items are vegan-friendly"],
            "dietary_info": {
                "vegetarian": true,
                "vegan": true,
                "halal": true,
                "gluten_free": false
            },
            "confidence_score": 0.85
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        result = await multimodal_service.analyze_menu_image(
            sample_menu_image,
            dietary_restrictions="vegan"
        )
        
        assert result.dietary_info["vegan"] is True
        assert result.dietary_info["vegetarian"] is True
        assert all("vegan" in rec.lower() or "vegetarian" in rec.lower() 
                  for rec in result.recommendations)
    
    @pytest.mark.asyncio
    async def test_concurrent_image_analysis(self, multimodal_service, mock_openai_client, sample_image_data):
        """Test handling multiple concurrent image analysis requests."""
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "detected_objects": ["test"],
            "location_suggestions": [],
            "landmarks_identified": [],
            "scene_description": "Test image",
            "confidence_score": 0.8,
            "recommendations": [],
            "is_food_image": false,
            "is_location_image": false,
            "extracted_text": ""
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Create multiple concurrent requests
        tasks = [
            multimodal_service.analyze_image_comprehensive(sample_image_data, f"Query {i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should complete successfully
        assert len(results) == 5
        for result in results:
            assert not isinstance(result, Exception)
            assert result is not None
    
    def test_confidence_score_validation(self, multimodal_service):
        """Test confidence score validation and normalization."""
        # Test various confidence score formats
        test_scores = [0.95, "0.88", 1.2, -0.1, "invalid"]
        
        for score in test_scores:
            normalized = multimodal_service._normalize_confidence_score(score)
            assert 0.0 <= normalized <= 1.0
    
    @pytest.mark.asyncio
    async def test_large_image_handling(self, multimodal_service, mock_openai_client):
        """Test handling of large image files."""
        # Create large mock image data
        large_image_data = b"x" * 1024 * 1024  # 1MB image
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "detected_objects": ["large_image"],
            "location_suggestions": [],
            "landmarks_identified": [],
            "scene_description": "Large image processed",
            "confidence_score": 0.8,
            "recommendations": [],
            "is_food_image": false,
            "is_location_image": false,
            "extracted_text": ""
        }
        '''
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        # Should handle large images appropriately
        result = await multimodal_service.analyze_image_comprehensive(
            large_image_data,
            "Analyze this large image"
        )
        
        assert result is not None
    
    def test_text_extraction_accuracy(self, multimodal_service):
        """Test text extraction accuracy from images."""
        # This would typically test OCR functionality
        sample_text = "RESTAURANT MENU\nKebab - 25 TL\nBaklava - 15 TL"
        
        # Test text processing
        processed_text = multimodal_service._process_extracted_text(sample_text)
        assert "RESTAURANT MENU" in processed_text
        assert "Kebab" in processed_text
        assert "25 TL" in processed_text

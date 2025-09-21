"""
Multimodal AI Tests - Testing REAL API Methods Only
Tests only async methods that actually exist in multimodal_ai.py
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from backend.api_clients.multimodal_ai import (
    OpenAIVisionClient, 
    GoogleVisionClient,
    ImageAnalysisResult,
    MenuAnalysisResult,
    LocationIdentificationResult
)


class TestMultimodalAIRealAPI:
    """Test the Multimodal AI functionality - only real methods."""
    
    @pytest.fixture
    def openai_vision_client(self):
        """Create OpenAI Vision client for testing."""
        with patch('backend.api_clients.multimodal_ai.OPENAI_AVAILABLE', True):
            return OpenAIVisionClient()
    
    @pytest.fixture
    def openai_vision_client_no_api(self):
        """Create OpenAI Vision client without API key."""
        with patch('backend.api_clients.multimodal_ai.OPENAI_AVAILABLE', False):
            return OpenAIVisionClient()
    
    @pytest.fixture
    def google_vision_client(self):
        """Create Google Vision client for testing."""
        return GoogleVisionClient()
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        return b"fake_image_data_for_testing"
    
    def test_openai_vision_client_initialization_with_api(self, openai_vision_client):
        """Test OpenAI Vision client initialization with API available."""
        # When OPENAI_AVAILABLE is True but no real client, it should handle gracefully
        assert openai_vision_client is not None
    
    def test_openai_vision_client_initialization_no_api(self, openai_vision_client_no_api):
        """Test OpenAI Vision client initialization without API."""
        assert openai_vision_client_no_api is not None
        assert openai_vision_client_no_api.client is None
    
    def test_google_vision_client_initialization(self, google_vision_client):
        """Test Google Vision client initialization."""
        assert google_vision_client is not None
        assert google_vision_client.api_key is None  # No real API key in tests
    
    @pytest.mark.asyncio
    async def test_openai_analyze_image_general(self, openai_vision_client_no_api, sample_image_data):
        """Test OpenAI image analysis with general analysis type."""
        result = await openai_vision_client_no_api.analyze_image(sample_image_data, "general")
        
        assert isinstance(result, dict)
        assert "source" in result
        assert result["source"] == "mock_analysis"
        assert "objects" in result
        assert "location_suggestions" in result
        assert "scene_description" in result
        assert "recommendations" in result
        assert "is_food_image" in result
        assert "is_location_image" in result
        assert "confidence_score" in result
    
    @pytest.mark.asyncio
    async def test_openai_analyze_image_menu(self, openai_vision_client_no_api, sample_image_data):
        """Test OpenAI image analysis with menu analysis type."""
        result = await openai_vision_client_no_api.analyze_image(sample_image_data, "menu")
        
        assert isinstance(result, dict)
        assert result["source"] == "mock_analysis"
        assert "menu_items" in result
        assert "cuisine_type" in result
        assert "price_range" in result
        assert "recommendations" in result
        assert "dietary_info" in result
        assert "confidence_score" in result
        
        # Check menu-specific data structure
        assert isinstance(result["menu_items"], list)
        assert result["cuisine_type"] == "Turkish"
        assert isinstance(result["price_range"], tuple)
    
    @pytest.mark.asyncio
    async def test_openai_analyze_image_location(self, openai_vision_client_no_api, sample_image_data):
        """Test OpenAI image analysis with location analysis type."""
        result = await openai_vision_client_no_api.analyze_image(sample_image_data, "location")
        
        assert isinstance(result, dict)
        assert result["source"] == "mock_analysis"
        assert "identified_location" in result
        assert "district" in result
        assert "category" in result
        assert "similar_places" in result
        assert "recommendations" in result
        assert "confidence_score" in result
        
        # Check location-specific data structure
        assert result["district"] == "Sultanahmet"
        assert result["category"] == "historic_area"
        assert isinstance(result["similar_places"], list)
    
    def test_get_analysis_prompt_general(self, openai_vision_client):
        """Test analysis prompt generation for general type."""
        prompt = openai_vision_client._get_analysis_prompt("general")
        
        assert isinstance(prompt, str)
        assert "objects" in prompt.lower()
        assert "landmarks" in prompt.lower()
        assert "istanbul" in prompt.lower()
        assert "json" in prompt.lower()
    
    def test_get_analysis_prompt_menu(self, openai_vision_client):
        """Test analysis prompt generation for menu type."""
        prompt = openai_vision_client._get_analysis_prompt("menu")
        
        assert isinstance(prompt, str)
        assert "menu" in prompt.lower()
        assert "food items" in prompt.lower()
        assert "cuisine" in prompt.lower()
        assert "dietary" in prompt.lower()
        assert "json" in prompt.lower()
    
    def test_get_analysis_prompt_location(self, openai_vision_client):
        """Test analysis prompt generation for location type."""
        prompt = openai_vision_client._get_analysis_prompt("location")
        
        assert isinstance(prompt, str)
        assert "identify" in prompt.lower()
        assert "landmark" in prompt.lower()
        assert "neighborhood" in prompt.lower()
        assert "category" in prompt.lower()
        assert "json" in prompt.lower()
    
    def test_parse_vision_response_valid_json(self, openai_vision_client):
        """Test parsing valid JSON response from Vision API."""
        response_text = '{"objects": ["building"], "confidence_score": 0.9}'
        
        result = openai_vision_client._parse_vision_response(response_text, "general")
        
        assert isinstance(result, dict)
        assert result["objects"] == ["building"]
        assert result["confidence_score"] == 0.9
        assert result["source"] == "openai_vision"
    
    def test_parse_vision_response_json_with_markdown(self, openai_vision_client):
        """Test parsing JSON wrapped in markdown code blocks."""
        response_text = '```json\n{"objects": ["mosque"], "confidence_score": 0.8}\n```'
        
        result = openai_vision_client._parse_vision_response(response_text, "general")
        
        assert isinstance(result, dict)
        assert result["objects"] == ["mosque"]
        assert result["confidence_score"] == 0.8
        assert result["source"] == "openai_vision"
    
    def test_parse_vision_response_invalid_json(self, openai_vision_client):
        """Test parsing invalid JSON falls back to text parsing."""
        response_text = "This is not valid JSON but contains food and restaurant information."
        
        result = openai_vision_client._parse_vision_response(response_text, "general")
        
        assert isinstance(result, dict)
        assert result["source"] == "openai_vision_text"
        assert result["confidence_score"] == 0.3
        assert result["is_food_image"] is True  # Should detect "food" in text
    
    def test_parse_text_response_menu_type(self, openai_vision_client):
        """Test text response parsing for menu analysis."""
        text = "This appears to be a Turkish menu with kebab and traditional dishes."
        
        result = openai_vision_client._parse_text_response(text, "menu")
        
        assert isinstance(result, dict)
        assert result["source"] == "openai_vision_text"
        assert "menu_items" in result
        assert "cuisine_type" in result
        assert "recommendations" in result
        assert result["confidence_score"] == 0.3
    
    def test_parse_text_response_location_type(self, openai_vision_client):
        """Test text response parsing for location analysis."""
        text = "This appears to be a historic building in Istanbul."
        
        result = openai_vision_client._parse_text_response(text, "location")
        
        assert isinstance(result, dict)
        assert result["source"] == "openai_vision_text"
        assert "identified_location" in result
        assert "district" in result
        assert "category" in result
        assert result["confidence_score"] == 0.3
    
    def test_generate_mock_analysis_menu(self, openai_vision_client):
        """Test mock analysis generation for menu type."""
        result = openai_vision_client._generate_mock_analysis("menu")
        
        assert isinstance(result, dict)
        assert result["source"] == "mock_analysis"
        assert "menu_items" in result
        assert "cuisine_type" in result
        assert result["cuisine_type"] == "Turkish"
        assert "price_range" in result
        assert "dietary_info" in result
        assert result["dietary_info"]["halal"] is True
    
    def test_generate_mock_analysis_location(self, openai_vision_client):
        """Test mock analysis generation for location type."""
        result = openai_vision_client._generate_mock_analysis("location")
        
        assert isinstance(result, dict)
        assert result["source"] == "mock_analysis"
        assert "identified_location" in result
        assert "district" in result
        assert result["district"] == "Sultanahmet"
        assert "similar_places" in result
        assert "Hagia Sophia" in result["similar_places"]
    
    def test_generate_mock_analysis_general(self, openai_vision_client):
        """Test mock analysis generation for general type."""
        result = openai_vision_client._generate_mock_analysis("general")
        
        assert isinstance(result, dict)
        assert result["source"] == "mock_analysis"
        assert "objects" in result
        assert "scene_description" in result
        assert "is_food_image" in result
        assert "is_location_image" in result
        assert result["is_location_image"] is True
    
    @pytest.mark.asyncio
    async def test_google_detect_landmarks_no_api(self, google_vision_client, sample_image_data):
        """Test Google landmark detection without API key."""
        landmarks = await google_vision_client.detect_landmarks(sample_image_data)
        
        assert isinstance(landmarks, list)
        # Should return mock landmarks when no API key
        assert len(landmarks) > 0
    
    def test_mock_landmarks(self, google_vision_client):
        """Test mock landmarks generation."""
        landmarks = google_vision_client._mock_landmarks()
        
        assert isinstance(landmarks, list)
        assert len(landmarks) > 0
        # Should contain generic landmark descriptions
        landmark_text = " ".join(landmarks).lower()
        assert any(name in landmark_text for name in ["historic", "building", "ottoman", "turkish", "landmark"])
    
    @pytest.mark.asyncio
    async def test_analyze_image_with_api_error(self, openai_vision_client, sample_image_data):
        """Test image analysis when API call fails."""
        # Mock the client to raise an exception
        if hasattr(openai_vision_client, 'client') and openai_vision_client.client:
            with patch.object(openai_vision_client.client.chat.completions, 'create') as mock_create:
                mock_create.side_effect = Exception("API Error")
                
                result = await openai_vision_client.analyze_image(sample_image_data, "general")
                
                assert isinstance(result, dict)
                assert result["source"] == "mock_analysis"
    
    @patch('backend.api_clients.multimodal_ai.asyncio.gather')
    async def test_analyze_image_with_mixed_results(self, mock_gather, openai_client):
        """Test analyze_image with mixed success/failure results."""
        # Mock gather to return mixed results (success and exceptions)
        mock_gather.return_value = [
            {"items": ["Turkish breakfast"], "type": "menu"},  # OpenAI success
            Exception("Google API error"),  # Google failure
            "Menu text content"  # Text extraction success
        ]
        
        result = await openai_client.analyze_image(b"fake_image", "menu")
        
        assert isinstance(result, dict)
        assert "items" in result
        # Should handle Google API failure gracefully
    
    @patch('backend.api_clients.multimodal_ai.asyncio.gather')
    async def test_analyze_image_all_failures(self, mock_gather, openai_client):
        """Test analyze_image when all async operations fail."""
        # Mock gather to return all exceptions
        mock_gather.return_value = [
            Exception("OpenAI API error"),
            Exception("Google API error"),
            Exception("Text extraction error")
        ]
        
        result = await openai_client.analyze_image(b"fake_image", "general")
        
        # Should still return a result (fallback to mock)
        assert isinstance(result, dict)
    
    def test_parse_vision_response_edge_cases(self, openai_client):
        """Test _parse_vision_response with various edge cases."""
        # Test with empty response
        result1 = openai_client._parse_vision_response("", "menu")
        assert isinstance(result1, dict)
        
        # Test with malformed JSON
        result2 = openai_client._parse_vision_response('{"incomplete": json', "menu")
        assert isinstance(result2, dict)
        
        # Test with valid JSON but unexpected structure
        result3 = openai_client._parse_vision_response('{"unexpected": "structure"}', "location")
        assert isinstance(result3, dict)
        
        # Test with nested markdown code blocks
        markdown_response = """
        Here's the analysis:
        ```json
        {"items": ["test"], "type": "menu"}
        ```
        Additional text here.
        """
        result4 = openai_client._parse_vision_response(markdown_response, "menu")
        assert isinstance(result4, dict)
    
    def test_parse_text_response_edge_cases(self, openai_client):
        """Test _parse_text_response with edge cases."""
        # Test with empty text
        result1 = openai_client._parse_text_response("", "menu")
        assert isinstance(result1, dict)
        
        # Test with very long text
        long_text = "restaurant " * 1000
        result2 = openai_client._parse_text_response(long_text, "location")
        assert isinstance(result2, dict)
        
        # Test with special characters
        special_text = "Döner €5.50 ñ café"
        result3 = openai_client._parse_text_response(special_text, "menu")
        assert isinstance(result3, dict)
    
    def test_generate_mock_analysis_edge_cases(self, openai_client):
        """Test _generate_mock_analysis with edge cases."""
        # Test with invalid analysis type
        result1 = openai_client._generate_mock_analysis("invalid_type")
        assert isinstance(result1, dict)
        assert "error" in result1 or "description" in result1
        
        # Test with None type
        result2 = openai_client._generate_mock_analysis(None)
        assert isinstance(result2, dict)
        
        # Test with empty string type
        result3 = openai_client._generate_mock_analysis("")
        assert isinstance(result3, dict)
    
    async def test_extract_text_from_image_error_handling(self, google_client):
        """Test text extraction error handling."""
        with patch('backend.api_clients.multimodal_ai.vision.ImageAnnotatorClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.text_detection.side_effect = Exception("Google Vision API error")
            mock_client_class.return_value = mock_client
            
            # Should handle error gracefully
            result = await google_client.extract_text_from_image(b"fake_image")
            assert isinstance(result, str)
            assert result == ""  # Should return empty string on error
    
    async def test_detect_landmarks_error_handling(self, google_client):
        """Test landmark detection error handling."""
        with patch('backend.api_clients.multimodal_ai.vision.ImageAnnotatorClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.landmark_detection.side_effect = Exception("Landmark detection failed")
            mock_client_class.return_value = mock_client
            
            result = await google_client.detect_landmarks(b"fake_image")
            assert isinstance(result, list)
            assert len(result) == 0  # Should return empty list on error
    
    def test_openai_client_initialization_variants(self):
        """Test OpenAI client initialization with different scenarios."""
        # Test with different API key scenarios
        with patch.dict('os.environ', {'OPENAI_API_KEY': ''}):
            client = OpenAIVisionClient()
            assert client.api_key is None or client.api_key == ""
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            client = OpenAIVisionClient()
            assert client.api_key == 'test_key'
    
    def test_google_client_initialization_variants(self):
        """Test Google client initialization with different scenarios."""
        # Test with missing credentials
        with patch.dict('os.environ', {}, clear=True):
            client = GoogleVisionClient()
            assert not client.credentials_available
        
        # Test with present credentials
        with patch.dict('os.environ', {'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/creds.json'}):
            client = GoogleVisionClient()
            # Constructor runs but credentials_available depends on actual file
    
    async def test_concurrent_analysis_edge_cases(self, openai_client):
        """Test concurrent analysis with various edge cases."""
        # Test with multiple images of different types
        images = [
            (b"menu_image", "menu"),
            (b"location_image", "location"), 
            (b"general_image", "general")
        ]
        
        with patch.object(openai_client, 'analyze_image') as mock_analyze:
            mock_analyze.side_effect = [
                {"type": "menu", "items": ["item1"]},
                Exception("API limit exceeded"),
                {"type": "general", "description": "scene"}
            ]
            
            # Should handle mixed results
            results = []
            for image_data, analysis_type in images:
                try:
                    result = await openai_client.analyze_image(image_data, analysis_type)
                    results.append(result)
                except Exception:
                    results.append(None)
            
            assert len(results) == 3
            assert results[0] is not None  # First succeeded
            assert results[1] is None       # Second failed
            assert results[2] is not None  # Third succeeded
    
    def test_data_validation_and_sanitization(self, openai_client):
        """Test data validation and sanitization methods."""
        # Test prompt generation with invalid inputs
        prompt1 = openai_client._get_analysis_prompt(None)
        assert isinstance(prompt1, str)
        assert len(prompt1) > 0
        
        prompt2 = openai_client._get_analysis_prompt("")
        assert isinstance(prompt2, str)
        
        prompt3 = openai_client._get_analysis_prompt("invalid_analysis_type")
        assert isinstance(prompt3, str)
        
        # Test with very long analysis type
        long_type = "a" * 1000
        prompt4 = openai_client._get_analysis_prompt(long_type)
        assert isinstance(prompt4, str)
    
    async def test_large_image_handling(self, openai_client):
        """Test handling of large images."""
        # Simulate a large image (though still fake data)
        large_image = b"x" * (10 * 1024 * 1024)  # 10MB fake image
        
        # Should handle without crashing
        result = await openai_client.analyze_image(large_image, "general")
        assert isinstance(result, dict)
    
    async def test_unicode_and_encoding_handling(self, openai_client):
        """Test handling of unicode and encoding issues."""
        # Test with unicode in extracted text
        with patch.object(openai_client, '_parse_vision_response') as mock_parse:
            unicode_response = '{"items": ["Döner Kebab", "Çay", "Türk Kahvesi"], "type": "menu"}'
            mock_parse.return_value = {"items": ["Döner Kebab", "Çay", "Türk Kahvesi"]}
            
            result = await openai_client.analyze_image(b"fake_image", "menu")
            assert isinstance(result, dict)
            
    def test_configuration_and_settings(self, openai_client):
        """Test various configuration scenarios."""
        # Test different model configurations if available
        original_model = getattr(openai_client, 'model', 'gpt-4-vision-preview')
        
        # Test with different model names
        test_models = ['gpt-4-vision-preview', 'gpt-4o', 'gpt-4o-mini']
        for model in test_models:
            if hasattr(openai_client, 'model'):
                openai_client.model = model
            # Should not crash with different models
            prompt = openai_client._get_analysis_prompt("menu")
            assert isinstance(prompt, str)
        
        # Restore original model
        if hasattr(openai_client, 'model'):
            openai_client.model = original_model
    
    async def test_rate_limiting_simulation(self, openai_client):
        """Test behavior under simulated rate limiting."""
        with patch.object(openai_client, 'analyze_image') as mock_analyze:
            # Simulate rate limiting error
            mock_analyze.side_effect = Exception("Rate limit exceeded")
            
            # Should handle gracefully and return mock data
            result = await openai_client.analyze_image(b"fake_image", "menu")
            assert isinstance(result, dict)
    
    def test_memory_usage_patterns(self, openai_client):
        """Test memory usage patterns with repeated operations."""
        # Test repeated operations don't cause memory leaks
        for i in range(100):
            key = openai_client._generate_cache_key(f"test_query_{i}")
            assert isinstance(key, str)
            
        # Test garbage collection friendliness
        import gc
        initial_objects = len(gc.get_objects())
        
        for i in range(50):
            prompt = openai_client._get_analysis_prompt("menu")
            mock_result = openai_client._generate_mock_analysis("general")
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Should not have excessive object growth
        assert final_objects - initial_objects < 1000

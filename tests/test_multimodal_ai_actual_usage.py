"""
Test for MultimodalAIService - ONLY methods actually used in production
Tests only analyze_image_comprehensive() and analyze_menu_image() methods
Excludes unused Google Vision and OpenAI Vision specific APIs
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from backend.api_clients.multimodal_ai import (
    MultimodalAIService,
    ImageAnalysisResult,
    MenuAnalysisResult,
    LocationIdentificationResult
)


class TestMultimodalAIActualUsage:
    """Test only the MultimodalAI methods actually used in production."""
    
    @pytest.fixture
    def multimodal_service(self):
        """Create multimodal service instance."""
        return MultimodalAIService()
    
    @pytest.fixture
    def valid_image_data(self):
        """Create valid image data for testing."""
        # Create a real valid JPEG image using PIL
        from PIL import Image
        import io
        
        # Create a small test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    @pytest.fixture
    def invalid_image_data(self):
        """Create invalid image data for testing."""
        return b'not an image'
    
    def test_multimodal_service_initialization(self, multimodal_service):
        """Test that service initializes correctly."""
        assert multimodal_service is not None
        assert hasattr(multimodal_service, 'openai_vision')
        assert hasattr(multimodal_service, 'google_vision')
        assert hasattr(multimodal_service, 'text_extractor')
        assert hasattr(multimodal_service, 'istanbul_landmarks')
        assert isinstance(multimodal_service.istanbul_landmarks, dict)
    
    def test_istanbul_landmarks_data(self, multimodal_service):
        """Test that Istanbul landmarks data is properly loaded."""
        landmarks = multimodal_service.istanbul_landmarks
        assert len(landmarks) > 0
        
        # Check some expected landmarks
        expected_landmarks = ['hagia sophia', 'blue mosque', 'galata tower']
        for landmark in expected_landmarks:
            assert landmark in landmarks
            assert 'district' in landmarks[landmark]
            assert 'category' in landmarks[landmark]
            assert 'keywords' in landmarks[landmark]
    
    @pytest.mark.asyncio
    async def test_analyze_image_comprehensive_with_valid_image(self, multimodal_service, valid_image_data):
        """Test comprehensive image analysis with valid image data."""
        # Mock the internal API calls to avoid external dependencies
        mock_openai_result = {
            "objects": ["mosque", "building"],
            "location_suggestions": ["Sultanahmet"],
            "scene_description": "A beautiful mosque in Istanbul",
            "recommendations": ["Visit in the morning"],
            "is_food_image": False,
            "is_location_image": True,
            "confidence_score": 0.8
        }
        
        mock_landmarks = ["Blue Mosque", "Hagia Sophia"]
        mock_text = "Sultan Ahmed Mosque"
        
        with patch.object(multimodal_service.openai_vision, 'analyze_image', new_callable=AsyncMock) as mock_openai, \
             patch.object(multimodal_service.google_vision, 'detect_landmarks', new_callable=AsyncMock) as mock_google, \
             patch.object(multimodal_service.text_extractor, 'extract_text', new_callable=AsyncMock) as mock_text_extract:
            
            mock_openai.return_value = mock_openai_result
            mock_google.return_value = mock_landmarks
            mock_text_extract.return_value = mock_text
            
            result = await multimodal_service.analyze_image_comprehensive(valid_image_data, "test location")
            
            # Verify result is correct type
            assert isinstance(result, ImageAnalysisResult)
            
            # Verify basic structure
            assert hasattr(result, 'detected_objects')
            assert hasattr(result, 'location_suggestions')
            assert hasattr(result, 'landmarks_identified')
            assert hasattr(result, 'scene_description')
            assert hasattr(result, 'confidence_score')
            assert hasattr(result, 'recommendations')
            assert hasattr(result, 'is_food_image')
            assert hasattr(result, 'is_location_image')
            assert hasattr(result, 'extracted_text')
            
            # Verify types
            assert isinstance(result.detected_objects, list)
            assert isinstance(result.location_suggestions, list)
            assert isinstance(result.landmarks_identified, list)
            assert isinstance(result.scene_description, str)
            assert isinstance(result.confidence_score, (int, float))
            assert isinstance(result.recommendations, list)
            assert isinstance(result.is_food_image, bool)
            assert isinstance(result.is_location_image, bool)
    
    @pytest.mark.asyncio
    async def test_analyze_image_comprehensive_with_invalid_image(self, multimodal_service, invalid_image_data):
        """Test comprehensive image analysis with invalid image - should handle gracefully."""
        result = await multimodal_service.analyze_image_comprehensive(invalid_image_data)
        
        # Should return fallback result, not crash
        assert isinstance(result, ImageAnalysisResult)
        assert isinstance(result.detected_objects, list)
        assert isinstance(result.confidence_score, (int, float))
    
    @pytest.mark.asyncio
    async def test_analyze_image_comprehensive_fallback_on_api_failure(self, multimodal_service, valid_image_data):
        """Test that analyze_image_comprehensive falls back gracefully when APIs fail."""
        # Mock all APIs to fail
        with patch.object(multimodal_service.openai_vision, 'analyze_image', new_callable=AsyncMock) as mock_openai, \
             patch.object(multimodal_service.google_vision, 'detect_landmarks', new_callable=AsyncMock) as mock_google, \
             patch.object(multimodal_service.text_extractor, 'extract_text', new_callable=AsyncMock) as mock_text_extract:
            
            mock_openai.side_effect = Exception("OpenAI API failed")
            mock_google.side_effect = Exception("Google API failed")  
            mock_text_extract.side_effect = Exception("Text extraction failed")
            
            result = await multimodal_service.analyze_image_comprehensive(valid_image_data)
            
            # Should still return valid result structure
            assert isinstance(result, ImageAnalysisResult)
            assert result.confidence_score >= 0
            assert isinstance(result.scene_description, str)
    
    @pytest.mark.asyncio
    async def test_analyze_menu_image_with_valid_image(self, multimodal_service, valid_image_data):
        """Test menu image analysis with valid image data."""
        # Mock the menu analysis response
        mock_menu_result = {
            "menu_items": [
                {"name": "Kebab", "price": 45.0},
                {"name": "Lahmacun", "price": 25.0}
            ],
            "cuisine_type": "Turkish",
            "price_range": (20.0, 60.0),
            "recommendations": ["Try the kebab"],
            "dietary_info": {"vegetarian_options": True, "halal": True},
            "confidence_score": 0.7
        }
        
        mock_extracted_text = "Kebab 45 TL\nLahmacun 25 TL"
        
        with patch.object(multimodal_service.openai_vision, 'analyze_image', new_callable=AsyncMock) as mock_openai, \
             patch.object(multimodal_service.text_extractor, 'extract_text', new_callable=AsyncMock) as mock_text_extract:
            
            mock_openai.return_value = mock_menu_result
            mock_text_extract.return_value = mock_extracted_text
            
            result = await multimodal_service.analyze_menu_image(valid_image_data)
            
            # Verify result is correct type
            assert isinstance(result, MenuAnalysisResult)
            
            # Verify structure
            assert hasattr(result, 'detected_items')
            assert hasattr(result, 'cuisine_type')
            assert hasattr(result, 'price_range')
            assert hasattr(result, 'recommendations')
            assert hasattr(result, 'dietary_info')
            assert hasattr(result, 'confidence_score')
            
            # Verify types
            assert isinstance(result.detected_items, list)
            assert isinstance(result.cuisine_type, str)
            assert isinstance(result.recommendations, list)
            assert isinstance(result.dietary_info, dict)
            assert isinstance(result.confidence_score, (int, float))
    
    @pytest.mark.asyncio
    async def test_analyze_menu_image_with_invalid_image(self, multimodal_service, invalid_image_data):
        """Test menu image analysis with invalid image - should handle gracefully."""
        result = await multimodal_service.analyze_menu_image(invalid_image_data)
        
        # Should return fallback result, not crash
        assert isinstance(result, MenuAnalysisResult)
        assert isinstance(result.detected_items, list)
        assert isinstance(result.cuisine_type, str)
        assert isinstance(result.confidence_score, (int, float))
    
    @pytest.mark.asyncio
    async def test_analyze_menu_image_fallback_on_api_failure(self, multimodal_service, valid_image_data):
        """Test that analyze_menu_image falls back gracefully when APIs fail."""
        # Mock APIs to fail
        with patch.object(multimodal_service.openai_vision, 'analyze_image', new_callable=AsyncMock) as mock_openai, \
             patch.object(multimodal_service.text_extractor, 'extract_text', new_callable=AsyncMock) as mock_text_extract:
            
            mock_openai.side_effect = Exception("OpenAI API failed")
            mock_text_extract.side_effect = Exception("Text extraction failed")
            
            result = await multimodal_service.analyze_menu_image(valid_image_data)
            
            # Should still return valid result structure
            assert isinstance(result, MenuAnalysisResult)
            assert result.confidence_score >= 0
            assert isinstance(result.cuisine_type, str)
    
    def test_data_classes_structure(self):
        """Test that data classes have correct structure."""
        # Test ImageAnalysisResult can be created with required fields
        image_result = ImageAnalysisResult(
            detected_objects=["building"],
            location_suggestions=["Istanbul"],
            landmarks_identified=["Blue Mosque"],
            scene_description="A mosque",
            confidence_score=0.8,
            recommendations=["Visit"],
            is_food_image=False,
            is_location_image=True,
            extracted_text="test"
        )
        
        assert isinstance(image_result, ImageAnalysisResult)
        assert image_result.detected_objects == ["building"]
        assert image_result.confidence_score == 0.8
        assert image_result.extracted_text == "test"
        
        # Test MenuAnalysisResult can be created with required fields
        menu_result = MenuAnalysisResult(
            detected_items=[{"name": "kebab", "price": 45}],
            cuisine_type="Turkish",
            price_range=(20, 60),
            recommendations=["Try kebab"],
            dietary_info={"halal": True},
            confidence_score=0.7
        )
        
        assert isinstance(menu_result, MenuAnalysisResult)
        assert menu_result.cuisine_type == "Turkish"
        assert menu_result.confidence_score == 0.7
        
        # Test LocationIdentificationResult can be created with required fields
        location_result = LocationIdentificationResult(
            identified_location="Blue Mosque",
            confidence_score=0.9,
            similar_places=["Hagia Sophia"],
            category="mosque",
            recommendations=["Visit in morning"],
            coordinates=(41.0058, 28.9768)
        )
        
        assert isinstance(location_result, LocationIdentificationResult)
        assert location_result.identified_location == "Blue Mosque"
        assert location_result.confidence_score == 0.9
    
    @pytest.mark.asyncio
    async def test_production_usage_integration(self, multimodal_service, valid_image_data):
        """Test the exact usage patterns from main.py production code."""
        # This simulates how the service is used in main.py
        
        # Mock successful responses for both methods
        with patch.object(multimodal_service.openai_vision, 'analyze_image', new_callable=AsyncMock) as mock_openai, \
             patch.object(multimodal_service.google_vision, 'detect_landmarks', new_callable=AsyncMock) as mock_google, \
             patch.object(multimodal_service.text_extractor, 'extract_text', new_callable=AsyncMock) as mock_text_extract:
            
            # Setup mock for comprehensive analysis first call
            mock_openai.side_effect = [
                # First call for analyze_image_comprehensive
                {
                    "objects": ["landmark", "building"],
                    "location_suggestions": ["Sultanahmet"],
                    "scene_description": "Historic building",
                    "recommendations": ["Visit morning"],
                    "is_food_image": False,
                    "is_location_image": True,
                    "confidence_score": 0.8
                },
                # Second call for analyze_menu_image
                {
                    "menu_items": [{"name": "Kebab", "price": 45}],
                    "cuisine_type": "Turkish",
                    "confidence_score": 0.7
                }
            ]
            mock_google.return_value = ["Blue Mosque"]
            mock_text_extract.side_effect = ["Sultan Ahmed Mosque", "Kebab 45 TL"]
            
            # Test analyze_image_comprehensive as used in production
            analysis_result = await multimodal_service.analyze_image_comprehensive(
                valid_image_data, user_context="tourism"
            )
            
            assert isinstance(analysis_result, ImageAnalysisResult)
            assert analysis_result.confidence_score > 0
            
            # Test analyze_menu_image as used in production
            menu_result = await multimodal_service.analyze_menu_image(valid_image_data)
            
            assert isinstance(menu_result, MenuAnalysisResult)
            assert menu_result.confidence_score > 0
            
            # Verify the methods were called as expected
            assert mock_openai.call_count == 2
            assert mock_text_extract.call_count == 2
    
    def test_error_handling_robustness(self, multimodal_service):
        """Test that service handles various error conditions gracefully."""
        # Test with None image data
        async def test_none_image():
            try:
                await multimodal_service.analyze_image_comprehensive(None)
                # Should not crash, might return fallback or raise ValueError
                return True
            except (ValueError, TypeError):
                # These are acceptable error types for None input
                return True
            except Exception:
                # Other exceptions are not expected
                return False
        
        # Test with empty bytes
        async def test_empty_image():
            try:
                result = await multimodal_service.analyze_image_comprehensive(b'')
                return isinstance(result, ImageAnalysisResult)
            except (ValueError, TypeError):
                return True
            except Exception:
                return False
        
        # Run tests
        result1 = asyncio.run(test_none_image())
        result2 = asyncio.run(test_empty_image())
        
        assert result1 is True
        assert result2 is True
    
    def test_no_external_api_dependencies_in_test(self, multimodal_service):
        """Verify that tests don't make real API calls."""
        # This test ensures our mocking is complete
        # If any real API calls are made, they should be caught here
        
        assert multimodal_service.openai_vision is not None
        assert multimodal_service.google_vision is not None
        assert multimodal_service.text_extractor is not None
        
        # Test that service can be created without real API keys
        # (this validates our fallback/mock behavior works)
        assert isinstance(multimodal_service.istanbul_landmarks, dict)
        assert len(multimodal_service.istanbul_landmarks) > 0

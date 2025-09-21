"""
Multimodal AI Service Tests - Focused on Core Service Only
Tests only the MultimodalAIService that's actually used in the application
Excludes Google Vision and OpenAI Vision specific functionality
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


class TestMultimodalAIServiceCore:
    """Test the core MultimodalAIService functionality."""
    
    @pytest.fixture
    def multimodal_service(self):
        """Create multimodal AI service for testing."""
        return MultimodalAIService()
    
    def test_multimodal_service_initialization(self, multimodal_service):
        """Test multimodal AI service initialization."""
        assert multimodal_service is not None
        assert hasattr(multimodal_service, 'openai_vision')
        assert hasattr(multimodal_service, 'google_vision')
        assert hasattr(multimodal_service, 'ocr_service')
    
    def test_validate_image_valid(self, multimodal_service):
        """Test image validation with valid image data."""
        # Valid JPEG header
        valid_jpeg = b'\xff\xd8\xff\xe0' + b'\x00' * 100
        assert multimodal_service._validate_image(valid_jpeg) is True
        
        # Valid PNG header
        valid_png = b'\x89PNG\r\n\x1a\n' + b'\x00' * 100
        assert multimodal_service._validate_image(valid_png) is True
    
    def test_validate_image_invalid(self, multimodal_service):
        """Test image validation with invalid image data."""
        # Too small
        assert multimodal_service._validate_image(b'small') is False
        
        # Invalid header
        assert multimodal_service._validate_image(b'invalid_header' + b'\x00' * 100) is False
        
        # Empty data
        assert multimodal_service._validate_image(b'') is False
    
    @pytest.mark.asyncio
    async def test_analyze_image_comprehensive_with_invalid_image(self, multimodal_service):
        """Test comprehensive image analysis with invalid image."""
        invalid_image = b'invalid'
        
        result = await multimodal_service.analyze_image_comprehensive(invalid_image)
        
        assert isinstance(result, ImageAnalysisResult)
        # Should return empty/default results for invalid image
        assert len(result.detected_objects) == 0 or result.confidence_score == 0
    
    @pytest.mark.asyncio
    async def test_analyze_image_comprehensive_mock_fallback(self, multimodal_service):
        """Test comprehensive image analysis falls back to mock data."""
        # Valid image format
        valid_image = b'\xff\xd8\xff\xe0' + b'\x00' * 1000
        
        # Mock all API calls to fail, should fallback to mock data
        with patch.object(multimodal_service.openai_vision, 'analyze_image', side_effect=Exception("API Error")), \
             patch.object(multimodal_service.google_vision, 'detect_landmarks', side_effect=Exception("API Error")), \
             patch.object(multimodal_service.ocr_service, 'extract_text', side_effect=Exception("API Error")):
            
            result = await multimodal_service.analyze_image_comprehensive(valid_image, "Istanbul landmark")
            
            assert isinstance(result, ImageAnalysisResult)
            assert result.confidence_score > 0
            assert len(result.scene_description) > 0
    
    @pytest.mark.asyncio 
    async def test_analyze_menu_image_with_invalid_image(self, multimodal_service):
        """Test menu image analysis with invalid image."""
        invalid_image = b'invalid'
        
        result = await multimodal_service.analyze_menu_image(invalid_image)
        
        assert isinstance(result, MenuAnalysisResult)
        assert result.success is False
        assert "Invalid image format" in result.error
    
    @pytest.mark.asyncio
    async def test_analyze_menu_image_mock_fallback(self, multimodal_service):
        """Test menu image analysis falls back to mock data."""
        # Valid image format
        valid_image = b'\xff\xd8\xff\xe0' + b'\x00' * 1000
        
        # Mock API calls to fail
        with patch.object(multimodal_service.openai_vision, 'analyze_image', side_effect=Exception("API Error")), \
             patch.object(multimodal_service.ocr_service, 'extract_text', side_effect=Exception("API Error")):
            
            result = await multimodal_service.analyze_menu_image(valid_image)
            
            assert isinstance(result, MenuAnalysisResult)
            assert result.success is True
            assert len(result.menu_items) > 0
            assert isinstance(result.menu_items[0], dict)
            assert "name" in result.menu_items[0]
    
    @pytest.mark.asyncio
    async def test_identify_location_with_invalid_image(self, multimodal_service):
        """Test location identification with invalid image."""
        invalid_image = b'invalid'
        
        result = await multimodal_service.identify_location(invalid_image)
        
        assert isinstance(result, LocationIdentificationResult)
        assert result.success is False
        assert "Invalid image format" in result.error
    
    @pytest.mark.asyncio
    async def test_identify_location_mock_fallback(self, multimodal_service):
        """Test location identification falls back to mock data."""
        # Valid image format
        valid_image = b'\xff\xd8\xff\xe0' + b'\x00' * 1000
        
        # Mock API calls to fail
        with patch.object(multimodal_service.openai_vision, 'analyze_image', side_effect=Exception("API Error")), \
             patch.object(multimodal_service.google_vision, 'detect_landmarks', side_effect=Exception("API Error")):
            
            result = await multimodal_service.identify_location(valid_image, "Istanbul")
            
            assert isinstance(result, LocationIdentificationResult)
            assert result.success is True
            assert result.location_name is not None
            assert len(result.location_name) > 0
            assert result.confidence > 0
    
    def test_combine_analysis_results(self, multimodal_service):
        """Test combining analysis results from different sources."""
        # Mock analysis results
        openai_analysis = {
            "description": "A beautiful mosque",
            "confidence": 0.8,
            "type": "religious_building"
        }
        landmarks = ["Blue Mosque", "Hagia Sophia"]
        extracted_text = "Sultan Ahmed Mosque"
        
        result = multimodal_service._combine_analysis_results(
            openai_analysis, landmarks, extracted_text, "location"
        )
        
        assert isinstance(result, dict)
        assert "description" in result
        assert "landmarks" in result
        assert "extracted_text" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.8  # Should maintain or improve confidence
    
    def test_enhance_menu_analysis(self, multimodal_service):
        """Test menu analysis enhancement with extracted text."""
        menu_analysis = {
            "menu_items": [
                {"name": "Kebab", "price": 50, "description": "Grilled meat"}
            ],
            "cuisine_type": "Turkish",
            "confidence": 0.7
        }
        extracted_text = "Döner Kebab - 50 TL\nAdana Kebab - 60 TL\nLahmacun - 25 TL"
        
        enhanced = multimodal_service._enhance_menu_analysis(menu_analysis, extracted_text)
        
        assert isinstance(enhanced, dict)
        assert "menu_items" in enhanced
        assert "extracted_text" in enhanced
        assert "confidence" in enhanced
        # Should have more items after enhancement
        assert len(enhanced["menu_items"]) >= len(menu_analysis["menu_items"])
    
    def test_data_classes_creation(self):
        """Test creation of data class instances."""
        # Test ImageAnalysisResult
        image_result = ImageAnalysisResult(
            success=True,
            description="A beautiful landmark",
            landmarks=["Blue Mosque"],
            extracted_text="Sultan Ahmed",
            confidence=0.85,
            analysis_type="location",
            error=None
        )
        
        assert image_result.success is True
        assert image_result.description == "A beautiful landmark"
        assert image_result.landmarks == ["Blue Mosque"]
        assert image_result.confidence == 0.85
        
        # Test MenuAnalysisResult
        menu_result = MenuAnalysisResult(
            success=True,
            menu_items=[{"name": "Kebab", "price": 50}],
            cuisine_type="Turkish",
            confidence=0.9,
            extracted_text="Menu text",
            error=None
        )
        
        assert menu_result.success is True
        assert len(menu_result.menu_items) == 1
        assert menu_result.cuisine_type == "Turkish"
        assert menu_result.confidence == 0.9
        
        # Test LocationIdentificationResult
        location_result = LocationIdentificationResult(
            success=True,
            location_name="Blue Mosque",
            coordinates=(41.0058, 28.9766),
            confidence=0.95,
            landmarks=["Sultan Ahmed Mosque"],
            additional_info={"district": "Sultanahmet"},
            error=None
        )
        
        assert location_result.success is True
        assert location_result.location_name == "Blue Mosque"
        assert location_result.coordinates == (41.0058, 28.9766)
        assert location_result.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_integration_full_multimodal_workflow(self, multimodal_service):
        """Test complete multimodal AI workflow without external APIs."""
        # Valid image format
        valid_image = b'\xff\xd8\xff\xe0' + b'\x00' * 1000
        
        # Mock all external API calls to ensure we test fallback behavior
        with patch.object(multimodal_service.openai_vision, 'analyze_image', side_effect=Exception("No API")), \
             patch.object(multimodal_service.google_vision, 'detect_landmarks', side_effect=Exception("No API")), \
             patch.object(multimodal_service.ocr_service, 'extract_text', side_effect=Exception("No API")):
            
            # Test comprehensive analysis
            comprehensive_result = await multimodal_service.analyze_image_comprehensive(
                valid_image, "Istanbul landmark"
            )
            assert isinstance(comprehensive_result, ImageAnalysisResult)
            assert comprehensive_result.success is True
            
            # Test menu analysis
            menu_result = await multimodal_service.analyze_menu_image(valid_image)
            assert isinstance(menu_result, MenuAnalysisResult)
            assert menu_result.success is True
            
            # Test location identification
            location_result = await multimodal_service.identify_location(valid_image, "Sultanahmet")
            assert isinstance(location_result, LocationIdentificationResult)
            assert location_result.success is True
    
    def test_error_handling_comprehensive(self, multimodal_service):
        """Test comprehensive error handling across the service."""
        # Test with None image data
        with pytest.raises((TypeError, AttributeError)):
            multimodal_service._validate_image(None)
        
        # Test combine_analysis_results with invalid data
        result = multimodal_service._combine_analysis_results(
            {}, [], "", "invalid_type"
        )
        assert isinstance(result, dict)
        assert result.get("confidence", 0) >= 0
        
        # Test enhance_menu_analysis with empty data
        enhanced = multimodal_service._enhance_menu_analysis({}, "")
        assert isinstance(enhanced, dict)

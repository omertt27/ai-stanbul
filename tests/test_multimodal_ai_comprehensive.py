"""
Comprehensive tests for Multimodal AI Service - ACTIVELY USED MODULE  
Tests image analysis, menu analysis, and multimodal AI functionality
"""
import pytest
from unittest.mock import patch, MagicMock
from backend.api_clients.multimodal_ai import MultimodalAIService, ImageAnalysisResult, MenuAnalysisResult


class TestMultimodalAIService:
    """Test the multimodal AI service functionality."""
    
    @pytest.fixture
    def ai_service(self):
        """Create multimodal AI service instance for testing."""
        return MultimodalAIService()
    
    def test_initialization(self, ai_service):
        """Test multimodal AI service initialization."""
        assert ai_service is not None
        assert hasattr(ai_service, 'analyze_image_comprehensive')
        assert hasattr(ai_service, 'analyze_menu_image')
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_analyze_image_comprehensive_success(self, mock_openai, ai_service):
        """Test successful comprehensive image analysis."""
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
            "detected_objects": ["building", "minaret", "dome"],
            "location_suggestions": ["Blue Mosque", "Sultanahmet"],
            "landmarks_identified": ["Blue Mosque"],
            "scene_description": "Historic mosque with minarets and dome",
            "confidence_score": 0.92,
            "recommendations": ["Visit during sunrise", "Explore nearby attractions"],
            "is_food_image": false,
            "is_location_image": true,
            "extracted_text": "Blue Mosque"
        }'''
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test image analysis
        test_image_data = b"fake_image_data"
        context = "Tourist attraction in Istanbul"
        
        result = ai_service.analyze_image_comprehensive(test_image_data, context)
        
        assert isinstance(result, ImageAnalysisResult)
        assert "building" in result.detected_objects
        assert "Blue Mosque" in result.landmarks_identified
        assert result.is_location_image is True
        assert result.is_food_image is False
        assert result.confidence_score == 0.92
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_analyze_image_comprehensive_food(self, mock_openai, ai_service):
        """Test image analysis for food images."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
            "detected_objects": ["kebab", "rice", "vegetables", "plate"],
            "location_suggestions": ["Turkish restaurant", "Kebab house"],
            "landmarks_identified": [],
            "scene_description": "Traditional Turkish kebab with rice and vegetables",
            "confidence_score": 0.88,
            "recommendations": ["Try with ayran", "Order baklava for dessert"],
            "is_food_image": true,
            "is_location_image": false,
            "extracted_text": "Adana Kebab"
        }'''
        mock_client.chat.completions.create.return_value = mock_response
        
        test_image_data = b"fake_food_image"
        context = "Restaurant meal"
        
        result = ai_service.analyze_image_comprehensive(test_image_data, context)
        
        assert isinstance(result, ImageAnalysisResult)
        assert result.is_food_image is True
        assert result.is_location_image is False
        assert "kebab" in result.detected_objects
        assert "Try with ayran" in result.recommendations
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_analyze_menu_image_success(self, mock_openai, ai_service):
        """Test successful menu image analysis."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
            "detected_items": [
                {"name": "Adana Kebab", "price": 45.00, "description": "Spicy minced meat kebab"},
                {"name": "Baklava", "price": 25.00, "description": "Sweet pastry with nuts"},
                {"name": "Turkish Tea", "price": 8.00, "description": "Traditional black tea"}
            ],
            "cuisine_type": "Turkish",
            "price_range": [8.00, 45.00],
            "recommendations": ["Adana Kebab is the specialty", "Try baklava for dessert"],
            "dietary_info": {
                "vegetarian": true,
                "vegan": false,
                "halal": true,
                "gluten_free": false
            },
            "confidence_score": 0.85
        }'''
        mock_client.chat.completions.create.return_value = mock_response
        
        test_menu_image = b"fake_menu_image"
        dietary_restrictions = "halal"
        
        result = ai_service.analyze_menu_image(test_menu_image, dietary_restrictions)
        
        assert isinstance(result, MenuAnalysisResult)
        assert len(result.detected_items) == 3
        assert result.cuisine_type == "Turkish"
        assert result.price_range == (8.00, 45.00)
        assert result.dietary_info["halal"] is True
        assert result.confidence_score == 0.85
        
        # Check individual menu items
        kebab_item = next(item for item in result.detected_items if item["name"] == "Adana Kebab")
        assert kebab_item["price"] == 45.00
        assert "Spicy" in kebab_item["description"]
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_analyze_image_api_error(self, mock_openai, ai_service):
        """Test image analysis with API error."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        test_image_data = b"fake_image"
        
        result = ai_service.analyze_image_comprehensive(test_image_data, "test")
        
        # Should return default result with error info
        assert isinstance(result, ImageAnalysisResult)
        assert result.confidence_score == 0.0
        assert len(result.detected_objects) == 0
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_analyze_image_invalid_json(self, mock_openai, ai_service):
        """Test image analysis with invalid JSON response."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        mock_client.chat.completions.create.return_value = mock_response
        
        test_image_data = b"fake_image"
        
        result = ai_service.analyze_image_comprehensive(test_image_data, "test")
        
        # Should handle JSON parse error gracefully
        assert isinstance(result, ImageAnalysisResult)
        assert result.confidence_score == 0.0
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_analyze_menu_dietary_filtering(self, mock_openai, ai_service):
        """Test menu analysis with dietary restriction filtering."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
            "detected_items": [
                {"name": "Vegetable Kebab", "price": 35.00, "description": "Grilled vegetables"},
                {"name": "Meat Kebab", "price": 45.00, "description": "Grilled meat"},
                {"name": "Hummus", "price": 15.00, "description": "Chickpea dip"}
            ],
            "cuisine_type": "Turkish",
            "price_range": [15.00, 45.00],
            "recommendations": ["Vegetable Kebab for vegetarians", "Hummus is vegan-friendly"],
            "dietary_info": {
                "vegetarian": true,
                "vegan": true,
                "halal": true,
                "gluten_free": false
            },
            "confidence_score": 0.80
        }'''
        mock_client.chat.completions.create.return_value = mock_response
        
        result = ai_service.analyze_menu_image(b"menu_image", "vegetarian")
        
        assert result.dietary_info["vegetarian"] is True
        assert "Vegetable Kebab for vegetarians" in result.recommendations
    
    def test_image_analysis_result_properties(self):
        """Test ImageAnalysisResult data structure."""
        result = ImageAnalysisResult(
            detected_objects=["mosque", "minaret"],
            location_suggestions=["Blue Mosque"],
            landmarks_identified=["Sultanahmet Mosque"],
            scene_description="Historic mosque",
            confidence_score=0.9,
            recommendations=["Visit early morning"],
            is_food_image=False,
            is_location_image=True,
            extracted_text="Blue Mosque"
        )
        
        assert result.detected_objects == ["mosque", "minaret"]
        assert result.confidence_score == 0.9
        assert result.is_location_image is True
        assert result.is_food_image is False
    
    def test_menu_analysis_result_properties(self):
        """Test MenuAnalysisResult data structure."""
        items = [
            {"name": "Kebab", "price": 30.0, "description": "Grilled meat"},
            {"name": "Salad", "price": 20.0, "description": "Fresh vegetables"}
        ]
        
        result = MenuAnalysisResult(
            detected_items=items,
            cuisine_type="Turkish",
            price_range=(20.0, 30.0),
            recommendations=["Try the kebab"],
            dietary_info={"vegetarian": True, "halal": True},
            confidence_score=0.85
        )
        
        assert len(result.detected_items) == 2
        assert result.cuisine_type == "Turkish"
        assert result.price_range == (20.0, 30.0)
        assert result.dietary_info["vegetarian"] is True
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_analyze_landmark_image(self, mock_openai, ai_service):
        """Test analysis of Istanbul landmark images."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
            "detected_objects": ["dome", "minaret", "courtyard", "people"],
            "location_suggestions": ["Hagia Sophia", "Sultanahmet Square"],
            "landmarks_identified": ["Hagia Sophia"],
            "scene_description": "The iconic Hagia Sophia with its massive dome and historical architecture",
            "confidence_score": 0.95,
            "recommendations": ["Visit the upper gallery", "Learn about Byzantine history", "Check opening hours"],
            "is_food_image": false,
            "is_location_image": true,
            "extracted_text": "Hagia Sophia Museum"
        }'''
        mock_client.chat.completions.create.return_value = mock_response
        
        result = ai_service.analyze_image_comprehensive(b"hagia_sophia_image", "Historic building")
        
        assert "Hagia Sophia" in result.landmarks_identified
        assert result.confidence_score == 0.95
        assert "Byzantine history" in " ".join(result.recommendations)
        assert result.is_location_image is True
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_analyze_restaurant_interior(self, mock_openai, ai_service):
        """Test analysis of restaurant interior images."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
            "detected_objects": ["tables", "chairs", "decorations", "traditional_carpet"],
            "location_suggestions": ["Traditional Turkish restaurant", "Ottoman-style dining"],
            "landmarks_identified": [],
            "scene_description": "Traditional Turkish restaurant interior with Ottoman decorations",
            "confidence_score": 0.82,
            "recommendations": ["Try traditional dishes", "Ask about live music", "Make reservations"],
            "is_food_image": false,
            "is_location_image": true,
            "extracted_text": ""
        }'''
        mock_client.chat.completions.create.return_value = mock_response
        
        result = ai_service.analyze_image_comprehensive(b"restaurant_interior", "Restaurant atmosphere")
        
        assert "Traditional Turkish restaurant" in result.location_suggestions
        assert "Ottoman decorations" in result.scene_description
        assert "traditional dishes" in " ".join(result.recommendations).lower()
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_analyze_street_food_menu(self, mock_openai, ai_service):
        """Test analysis of street food menu."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
            "detected_items": [
                {"name": "Döner", "price": 15.00, "description": "Sliced meat in bread"},
                {"name": "Lahmacun", "price": 12.00, "description": "Turkish pizza"},
                {"name": "Simit", "price": 3.00, "description": "Turkish bagel"},
                {"name": "Ayran", "price": 5.00, "description": "Yogurt drink"}
            ],
            "cuisine_type": "Turkish Street Food",
            "price_range": [3.00, 15.00],
            "recommendations": ["Döner is very popular", "Ayran pairs well with döner"],
            "dietary_info": {
                "vegetarian": false,
                "vegan": false,
                "halal": true,
                "gluten_free": false
            },
            "confidence_score": 0.90
        }'''
        mock_client.chat.completions.create.return_value = mock_response
        
        result = ai_service.analyze_menu_image(b"street_food_menu", "quick meal")
        
        assert result.cuisine_type == "Turkish Street Food"
        assert len(result.detected_items) == 4
        assert any(item["name"] == "Döner" for item in result.detected_items)
        assert result.price_range == (3.00, 15.00)
        assert "Ayran pairs well" in " ".join(result.recommendations)
    
    def test_error_handling_empty_image(self, ai_service):
        """Test handling of empty image data."""
        result = ai_service.analyze_image_comprehensive(b"", "test")
        
        # Should return default result for empty image
        assert isinstance(result, ImageAnalysisResult)
        assert result.confidence_score == 0.0
        assert len(result.detected_objects) == 0
    
    @patch('backend.api_clients.multimodal_ai.openai.OpenAI')
    def test_integration_full_multimodal_workflow(self, mock_openai, ai_service):
        """Test complete multimodal AI workflow."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Test image analysis first
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = '''{
            "detected_objects": ["restaurant", "sign", "people"],
            "location_suggestions": ["Turkish restaurant"],
            "landmarks_identified": [],
            "scene_description": "Restaurant exterior with menu board",
            "confidence_score": 0.85,
            "recommendations": ["Check the menu", "Read reviews"],
            "is_food_image": false,
            "is_location_image": true,
            "extracted_text": "Restaurant Menu"
        }'''
        
        # Test menu analysis second  
        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = '''{
            "detected_items": [
                {"name": "Mixed Grill", "price": 55.00, "description": "Variety of grilled meats"}
            ],
            "cuisine_type": "Turkish",
            "price_range": [25.00, 55.00],
            "recommendations": ["Mixed Grill for sharing"],
            "dietary_info": {"vegetarian": false, "halal": true},
            "confidence_score": 0.88
        }'''
        
        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
        
        # 1. Analyze restaurant exterior
        exterior_result = ai_service.analyze_image_comprehensive(b"restaurant_exterior", "restaurant")
        assert exterior_result.is_location_image is True
        assert "Turkish restaurant" in exterior_result.location_suggestions
        
        # 2. Analyze menu
        menu_result = ai_service.analyze_menu_image(b"menu_image", "halal")
        assert menu_result.cuisine_type == "Turkish"
        assert menu_result.dietary_info["halal"] is True
        
        # Both calls should have been made
        assert mock_client.chat.completions.create.call_count == 2

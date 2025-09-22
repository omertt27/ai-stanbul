"""
Multimodal AI Module for AI Istanbul Travel Guide
Provides image understanding, visual search, and multimodal capabilities
"""

import os
import io
import base64
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging
from dataclasses import dataclass

# Handle PIL import gracefully
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
    print("✅ Advanced AI features loaded successfully")
except ImportError as e:
    PIL_AVAILABLE = False
    print(f"⚠️ Advanced AI features not available: {e}")
    # Create dummy Image module for graceful degradation
    class DummyImage:
        @staticmethod
        def open(*args, **kwargs):
            raise NotImplementedError("PIL not available")
        @staticmethod
        def new(*args, **kwargs):
            raise NotImplementedError("PIL not available")
    
    PILImage = DummyImage

import requests
import asyncio

# Optional aiohttp import
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    # Create dummy aiohttp for graceful degradation
    class DummyResponse:
        def __init__(self):
            self.status = 503
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        async def json(self):
            return {}
    
    class DummyClientSession:
        def __init__(self):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def post(self, *args, **kwargs):
            return DummyResponse()
    
    class DummyAiohttp:
        ClientSession = DummyClientSession
    
    aiohttp = DummyAiohttp()
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

# Try to import OpenAI for vision capabilities
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available for vision capabilities")

@dataclass
class ImageAnalysisResult:
    """Result from image analysis"""
    detected_objects: List[str]
    location_suggestions: List[str]
    landmarks_identified: List[str]
    scene_description: str
    confidence_score: float
    recommendations: List[str]
    is_food_image: bool
    is_location_image: bool
    extracted_text: Optional[str] = None

@dataclass
class MenuAnalysisResult:
    """Result from menu image analysis"""
    detected_items: List[Dict[str, Any]]
    cuisine_type: str
    price_range: Optional[Tuple[float, float]]
    recommendations: List[str]
    dietary_info: Dict[str, bool]  # vegetarian, vegan, halal, etc.
    confidence_score: float

@dataclass
class LocationIdentificationResult:
    """Result from location identification"""
    identified_location: Optional[str]
    confidence_score: float
    similar_places: List[str]
    category: str  # landmark, restaurant, street, etc.
    recommendations: List[str]
    coordinates: Optional[Tuple[float, float]]

class OpenAIVisionClient:
    """Client for OpenAI Vision API"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and OPENAI_AVAILABLE:
            try:
                from openai import OpenAI as OpenAIClient
                self.client = OpenAIClient(
                    api_key=self.api_key,
                    timeout=30.0,
                    max_retries=2
                )
            except Exception as e:
                logger.error(f"OpenAI client initialization failed: {e}")
                self.client = None
        else:
            self.client = None
        
        if not self.client:
            logger.warning("OpenAI Vision client not available - using mock responses")
    
    async def analyze_image(self, image_data: bytes, analysis_type: str = "general") -> Dict[str, Any]:
        """Analyze image using OpenAI Vision API"""
        if not self.client:
            return self._generate_mock_analysis(analysis_type)
        
        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create appropriate prompt based on analysis type
            prompt = self._get_analysis_prompt(analysis_type)
            
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            content = response.choices[0].message.content or ""
            return self._parse_vision_response(content, analysis_type)
            
        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            return self._generate_mock_analysis(analysis_type)
    
    def _get_analysis_prompt(self, analysis_type: str) -> str:
        """Get appropriate prompt for different analysis types"""
        prompts = {
            "general": """
            Analyze this image and identify:
            1. What objects, landmarks, or places can you see?
            2. If this is a location in Istanbul, which area or landmark might it be?
            3. Describe the scene briefly
            4. What would you recommend to someone visiting this place?
            5. Is this a food/restaurant image or a location/landmark image?
            
            Please format your response as JSON with keys: objects, location_suggestions, scene_description, recommendations, is_food_image, is_location_image, confidence_score
            """,
            
            "menu": """
            Analyze this menu image and extract:
            1. List of food items with estimated prices (in Turkish Lira if visible)
            2. Type of cuisine (Turkish, Italian, etc.)
            3. Any dietary information (vegetarian options, etc.)
            4. Recommendations for what to order
            5. Price range estimation
            
            Please format your response as JSON with keys: menu_items, cuisine_type, price_range, recommendations, dietary_info, confidence_score
            """,
            
            "location": """
            Identify this Istanbul location:
            1. What specific landmark, building, or area is this?
            2. What neighborhood/district is this likely in?
            3. What category does this place belong to? (landmark, restaurant, mosque, museum, bridge, etc.)
            4. What are some similar places nearby?
            5. What would you recommend doing here?
            
            Please format your response as JSON with keys: identified_location, district, category, similar_places, recommendations, confidence_score
            """
        }
        
        return prompts.get(analysis_type, prompts["general"])
    
    def _parse_vision_response(self, response_text: str, analysis_type: str) -> Dict[str, Any]:
        """Parse OpenAI Vision API response"""
        try:
            # Try to extract JSON from the response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            result = json.loads(response_text)
            result["source"] = "openai_vision"
            return result
            
        except json.JSONDecodeError:
            # If JSON parsing fails, create structured response from text
            logger.warning("Failed to parse JSON from Vision API, using text parsing")
            return self._parse_text_response(response_text, analysis_type)
    
    def _parse_text_response(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Parse text response when JSON parsing fails"""
        # Basic text parsing fallback
        if analysis_type == "menu":
            return {
                "menu_items": [],
                "cuisine_type": "Unknown",
                "price_range": None,
                "recommendations": [text[:200]],
                "dietary_info": {},
                "confidence_score": 0.3,
                "source": "openai_vision_text"
            }
        elif analysis_type == "location":
            return {
                "identified_location": "Unknown Istanbul Location",
                "district": "Unknown",
                "category": "unknown",
                "similar_places": [],
                "recommendations": [text[:200]],
                "confidence_score": 0.3,
                "source": "openai_vision_text"
            }
        else:
            return {
                "objects": [],
                "location_suggestions": [],
                "scene_description": text[:200],
                "recommendations": [],
                "is_food_image": "food" in text.lower() or "menu" in text.lower(),
                "is_location_image": "building" in text.lower() or "landmark" in text.lower(),
                "confidence_score": 0.3,
                "source": "openai_vision_text"
            }
    
    def _generate_mock_analysis(self, analysis_type: str) -> Dict[str, Any]:
        """Generate mock analysis when Vision API is not available"""
        if analysis_type == "menu":
            return {
                "menu_items": [
                    {"name": "Turkish Kebab", "price": 45.0},
                    {"name": "Lahmacun", "price": 25.0},
                    {"name": "Turkish Tea", "price": 8.0}
                ],
                "cuisine_type": "Turkish",
                "price_range": (20.0, 60.0),
                "recommendations": ["Try the Turkish kebab - it's a local specialty"],
                "dietary_info": {"vegetarian_options": True, "halal": True},
                "confidence_score": 0.4,
                "source": "mock_analysis"
            }
        elif analysis_type == "location":
            return {
                "identified_location": "Historic Peninsula",
                "district": "Sultanahmet",
                "category": "historic_area",
                "similar_places": ["Hagia Sophia", "Blue Mosque", "Topkapi Palace"],
                "recommendations": ["Visit nearby historic landmarks", "Take a guided tour"],
                "confidence_score": 0.4,
                "source": "mock_analysis"
            }
        else:
            return {
                "objects": ["building", "people", "street"],
                "location_suggestions": ["Historic area of Istanbul"],
                "scene_description": "Urban scene in Istanbul with traditional architecture",
                "recommendations": ["Explore the historic neighborhood", "Try local restaurants"],
                "is_food_image": False,
                "is_location_image": True,
                "confidence_score": 0.4,
                "source": "mock_analysis"
            }

class GoogleVisionClient:
    """Client for Google Cloud Vision API"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        
        if not self.api_key:
            logger.warning("Google Vision API key not available")
    
    async def detect_landmarks(self, image_data: bytes) -> List[str]:
        """Detect landmarks in image using Google Vision API"""
        if not self.api_key:
            return self._mock_landmarks()
        
        try:
            url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
            
            # Prepare the request
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            request_data = {
                "requests": [
                    {
                        "image": {"content": image_base64},
                        "features": [
                            {"type": "LANDMARK_DETECTION", "maxResults": 10},
                            {"type": "OBJECT_LOCALIZATION", "maxResults": 10},
                            {"type": "TEXT_DETECTION", "maxResults": 5}
                        ]
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=request_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_google_vision_response(data)
                    else:
                        logger.error(f"Google Vision API error: {response.status}")
                        return self._mock_landmarks()
        
        except Exception as e:
            logger.error(f"Google Vision API error: {e}")
            return self._mock_landmarks()
    
    def _parse_google_vision_response(self, data: Dict) -> List[str]:
        """Parse Google Vision API response"""
        landmarks = []
        
        if "responses" in data and data["responses"]:
            response = data["responses"][0]
            
            # Extract landmark annotations
            if "landmarkAnnotations" in response:
                for landmark in response["landmarkAnnotations"]:
                    landmarks.append(landmark["description"])
            
            # Extract object annotations as fallback
            if not landmarks and "localizedObjectAnnotations" in response:
                for obj in response["localizedObjectAnnotations"]:
                    landmarks.append(obj["name"])
        
        return landmarks[:5]  # Return top 5
    
    def _mock_landmarks(self) -> List[str]:
        """Mock landmark detection"""
        return ["Historic Building", "Ottoman Architecture", "Turkish Landmark"]

class TextExtractionClient:
    """Client for extracting text from images (OCR)"""
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
    
    async def extract_text(self, image_data: bytes) -> str:
        """Extract text from image using OCR"""
        if self.google_api_key:
            try:
                return await self._google_ocr(image_data)
            except Exception as e:
                logger.warning(f"Google OCR failed: {e}")
        
        # Fallback to mock text extraction
        return self._mock_text_extraction()
    
    async def _google_ocr(self, image_data: bytes) -> str:
        """Use Google Vision API for OCR"""
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.google_api_key}"
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        request_data = {
            "requests": [
                {
                    "image": {"content": image_base64},
                    "features": [{"type": "TEXT_DETECTION", "maxResults": 1}]
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=request_data) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if "responses" in data and data["responses"]:
                        response_data = data["responses"][0]
                        if "textAnnotations" in response_data:
                            return response_data["textAnnotations"][0]["description"]
        
        return ""
    
    def _mock_text_extraction(self) -> str:
        """Mock text extraction"""
        return "Sample text from image - OCR not available"

class MultimodalAIService:
    """Main service for multimodal AI capabilities"""
    
    def __init__(self):
        self.openai_vision = OpenAIVisionClient()
        self.google_vision = GoogleVisionClient()
        self.text_extractor = TextExtractionClient()
        
        # Istanbul-specific knowledge for better location identification
        self.istanbul_landmarks = {
            "hagia sophia": {"district": "Sultanahmet", "category": "museum", "keywords": ["dome", "byzantine", "ancient"]},
            "blue mosque": {"district": "Sultanahmet", "category": "mosque", "keywords": ["minarets", "blue", "tiles"]},
            "galata tower": {"district": "Galata", "category": "tower", "keywords": ["tower", "cylindrical", "stone"]},
            "bosphorus bridge": {"district": "Various", "category": "bridge", "keywords": ["bridge", "suspension", "water"]},
            "grand bazaar": {"district": "Beyazıt", "category": "market", "keywords": ["covered", "shops", "bazaar"]},
            "topkapi palace": {"district": "Sultanahmet", "category": "palace", "keywords": ["palace", "ottoman", "gardens"]},
            "dolmabahce palace": {"district": "Beşiktaş", "category": "palace", "keywords": ["palace", "baroque", "bosphorus"]},
            "maiden's tower": {"district": "Üsküdar", "category": "tower", "keywords": ["tower", "water", "small island"]},
            "taksim square": {"district": "Beyoğlu", "category": "square", "keywords": ["square", "monument", "modern"]},
            "istiklal street": {"district": "Beyoğlu", "category": "street", "keywords": ["pedestrian", "street", "tram"]}
        }
    
    async def analyze_image_comprehensive(self, image_data: bytes, user_context: Optional[str] = None) -> ImageAnalysisResult:
        """Comprehensive image analysis combining multiple AI services"""
        try:
            # Validate image
            if not self._validate_image(image_data):
                raise ValueError("Invalid image format")
            
            # Run multiple analyses in parallel
            tasks = [
                self.openai_vision.analyze_image(image_data, "general"),
                self.google_vision.detect_landmarks(image_data),
                self.text_extractor.extract_text(image_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            openai_result = results[0] if not isinstance(results[0], Exception) else {}
            google_landmarks = results[1] if not isinstance(results[1], Exception) else []
            extracted_text = results[2] if not isinstance(results[2], Exception) else ""
            
            # Ensure proper types
            if not isinstance(openai_result, dict):
                openai_result = {}
            if not isinstance(google_landmarks, list):
                google_landmarks = []
            if not isinstance(extracted_text, str):
                extracted_text = ""
            
            # Combine results
            combined_result = self._combine_analysis_results(
                openai_result, google_landmarks, extracted_text, user_context
            )
            
            return ImageAnalysisResult(**combined_result)
            
        except Exception as e:
            logger.error(f"Comprehensive image analysis failed: {e}")
            return self._fallback_analysis_result()
    
    async def analyze_menu_image(self, image_data: bytes) -> MenuAnalysisResult:
        """Specialized analysis for menu images"""
        try:
            if not self._validate_image(image_data):
                raise ValueError("Invalid image format")
            
            # Use OpenAI Vision for menu analysis
            menu_analysis = await self.openai_vision.analyze_image(image_data, "menu")
            
            # Extract text for additional menu parsing
            extracted_text = await self.text_extractor.extract_text(image_data)
            
            # Enhance analysis with Istanbul-specific menu knowledge
            enhanced_result = self._enhance_menu_analysis(menu_analysis, extracted_text)
            
            return MenuAnalysisResult(**enhanced_result)
            
        except Exception as e:
            logger.error(f"Menu analysis failed: {e}")
            return self._fallback_menu_result()
    
    async def identify_location(self, image_data: bytes, user_hint: Optional[str] = None) -> LocationIdentificationResult:
        """Specialized location identification"""
        try:
            if not self._validate_image(image_data):
                raise ValueError("Invalid image format")
            
            # Run location analysis and landmark detection
            tasks = [
                self.openai_vision.analyze_image(image_data, "location"),
                self.google_vision.detect_landmarks(image_data)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            openai_result = results[0] if not isinstance(results[0], Exception) else {}
            google_landmarks = results[1] if not isinstance(results[1], Exception) else []
            
            # Ensure proper types
            if not isinstance(openai_result, dict):
                openai_result = {}
            if not isinstance(google_landmarks, list):
                google_landmarks = []
            
            # Match against Istanbul landmarks
            location_result = self._match_istanbul_location(openai_result, google_landmarks, user_hint)
            
            return LocationIdentificationResult(**location_result)
            
        except Exception as e:
            logger.error(f"Location identification failed: {e}")
            return self._fallback_location_result()
    
    def _validate_image(self, image_data: bytes) -> bool:
        """Validate image format and size"""
        try:
            image = PILImage.open(io.BytesIO(image_data))
            
            # Check format
            if image.format not in ['JPEG', 'PNG', 'WebP']:
                return False
            
            # Check size (max 10MB)
            if len(image_data) > 10 * 1024 * 1024:
                return False
            
            # Check dimensions
            if image.size[0] > 4096 or image.size[1] > 4096:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _combine_analysis_results(
        self, 
        openai_result: Dict, 
        google_landmarks: List[str], 
        extracted_text: str,
        user_context: Optional[str]
    ) -> Dict[str, Any]:
        """Combine results from multiple AI services"""
        
        # Start with OpenAI results as base
        detected_objects = openai_result.get("objects", [])
        location_suggestions = openai_result.get("location_suggestions", [])
        scene_description = openai_result.get("scene_description", "")
        recommendations = openai_result.get("recommendations", [])
        is_food_image = openai_result.get("is_food_image", False)
        is_location_image = openai_result.get("is_location_image", True)
        confidence_score = openai_result.get("confidence_score", 0.5)
        
        # Enhance with Google landmarks
        landmarks_identified = google_landmarks
        
        # Enhance location suggestions with Istanbul-specific knowledge
        for landmark in google_landmarks:
            landmark_lower = landmark.lower()
            for istanbul_landmark, info in self.istanbul_landmarks.items():
                if any(keyword in landmark_lower for keyword in info["keywords"]):
                    if istanbul_landmark not in [loc.lower() for loc in location_suggestions]:
                        location_suggestions.append(istanbul_landmark.title())
                    
                    # Add district-specific recommendations
                    district_rec = f"Explore {info['district']} district for more {info['category']}s"
                    if district_rec not in recommendations:
                        recommendations.append(district_rec)
        
        # Enhance with user context
        if user_context:
            recommendations.append(f"Based on your interest in {user_context}, consider visiting similar places")
        
        # Enhance with extracted text
        if extracted_text and len(extracted_text.strip()) > 5:
            # Check if text contains Turkish words or place names
            turkish_indicators = ['türk', 'istanbul', 'lira', 'tl', 'beyoğlu', 'sultanahmet']
            if any(indicator in extracted_text.lower() for indicator in turkish_indicators):
                confidence_score = min(1.0, confidence_score + 0.2)
        
        return {
            "detected_objects": detected_objects,
            "location_suggestions": location_suggestions[:5],  # Top 5
            "landmarks_identified": landmarks_identified[:3],  # Top 3
            "scene_description": scene_description,
            "confidence_score": confidence_score,
            "recommendations": recommendations[:5],  # Top 5
            "is_food_image": is_food_image,
            "is_location_image": is_location_image,
            "extracted_text": extracted_text[:200] if extracted_text else None
        }
    
    def _enhance_menu_analysis(self, menu_analysis: Dict, extracted_text: str) -> Dict[str, Any]:
        """Enhance menu analysis with Istanbul-specific knowledge"""
        
        detected_items = menu_analysis.get("menu_items", [])
        cuisine_type = menu_analysis.get("cuisine_type", "Unknown")
        price_range = menu_analysis.get("price_range")
        recommendations = menu_analysis.get("recommendations", [])
        dietary_info = menu_analysis.get("dietary_info", {})
        confidence_score = menu_analysis.get("confidence_score", 0.5)
        
        # Enhance with Turkish food knowledge
        turkish_foods = {
            'kebab': {'category': 'main', 'typical_price': 45},
            'döner': {'category': 'main', 'typical_price': 25},
            'lahmacun': {'category': 'appetizer', 'typical_price': 20},
            'baklava': {'category': 'dessert', 'typical_price': 15},
            'turkish tea': {'category': 'beverage', 'typical_price': 8},
            'turkish coffee': {'category': 'beverage', 'typical_price': 12},
            'meze': {'category': 'appetizer', 'typical_price': 35},
            'pide': {'category': 'main', 'typical_price': 30},
            'köfte': {'category': 'main', 'typical_price': 40},
            'börek': {'category': 'appetizer', 'typical_price': 25}
        }
        
        # Check extracted text for Turkish food items
        if extracted_text:
            text_lower = extracted_text.lower()
            for food, info in turkish_foods.items():
                if food in text_lower:
                    # Add to detected items if not already present
                    existing_names = [item.get('name', '').lower() for item in detected_items]
                    if food not in existing_names:
                        detected_items.append({
                            'name': food.title(),
                            'price': info['typical_price'],
                            'category': info['category']
                        })
                    
                    # Update cuisine type
                    if cuisine_type == "Unknown":
                        cuisine_type = "Turkish"
        
        # Add Turkish cuisine recommendations
        if cuisine_type.lower() == "turkish":
            turkish_recommendations = [
                "Try the traditional Turkish tea with your meal",
                "Ask for recommendations on regional specialties",
                "Don't miss the Turkish desserts if available"
            ]
            recommendations.extend([rec for rec in turkish_recommendations if rec not in recommendations])
            
            # Set dietary info for Turkish cuisine
            dietary_info.update({
                "halal_options": True,
                "vegetarian_options": True,
                "dairy_free_options": True
            })
        
        return {
            "detected_items": detected_items[:10],  # Top 10 items
            "cuisine_type": cuisine_type,
            "price_range": price_range,
            "recommendations": recommendations[:5],
            "dietary_info": dietary_info,
            "confidence_score": min(1.0, confidence_score + 0.1)  # Slight boost for Turkish cuisine
        }
    
    def _match_istanbul_location(
        self, 
        openai_result: Dict, 
        google_landmarks: List[str], 
        user_hint: Optional[str]
    ) -> Dict[str, Any]:
        """Match detected features against known Istanbul locations"""
        
        identified_location = openai_result.get("identified_location", "Unknown Location")
        district = openai_result.get("district", "Unknown")
        category = openai_result.get("category", "unknown")
        similar_places = openai_result.get("similar_places", [])
        recommendations = openai_result.get("recommendations", [])
        confidence_score = openai_result.get("confidence_score", 0.5)
        
        # Try to match against known landmarks
        all_features = google_landmarks + [identified_location] + similar_places
        if user_hint:
            all_features.append(user_hint)
        
        best_match = None
        best_confidence = 0
        
        for feature in all_features:
            feature_lower = feature.lower()
            for landmark, info in self.istanbul_landmarks.items():
                # Check direct name match
                if landmark in feature_lower or feature_lower in landmark:
                    if len(feature_lower) > best_confidence:
                        best_match = landmark
                        best_confidence = len(feature_lower)
                        continue
                
                # Check keyword matches
                keyword_matches = sum(1 for keyword in info["keywords"] if keyword in feature_lower)
                if keyword_matches > 0 and keyword_matches > best_confidence:
                    best_match = landmark
                    best_confidence = keyword_matches
        
        if best_match:
            landmark_info = self.istanbul_landmarks[best_match]
            identified_location = best_match.title()
            district = landmark_info["district"]
            category = landmark_info["category"]
            confidence_score = min(1.0, confidence_score + 0.3)
            
            # Add relevant similar places
            similar_places = [
                landmark.title() for landmark, info in self.istanbul_landmarks.items()
                if info["district"] == landmark_info["district"] and landmark != best_match
            ][:3]
            
            # Add location-specific recommendations
            location_recommendations = [
                f"Perfect spot to explore {district} district",
                f"Don't miss other {category}s in the area",
                f"Great for photography and sightseeing"
            ]
            recommendations.extend([rec for rec in location_recommendations if rec not in recommendations])
        
        return {
            "identified_location": identified_location,
            "confidence_score": confidence_score,
            "similar_places": similar_places[:5],
            "category": category,
            "recommendations": recommendations[:5],
            "coordinates": None  # Would need geocoding service to add coordinates
        }
    
    def _fallback_analysis_result(self) -> ImageAnalysisResult:
        """Fallback result when analysis fails"""
        return ImageAnalysisResult(
            detected_objects=["Unknown objects"],
            location_suggestions=["Istanbul area"],
            landmarks_identified=[],
            scene_description="Unable to analyze image - please try again",
            confidence_score=0.1,
            recommendations=["Try uploading a clearer image"],
            is_food_image=False,
            is_location_image=True,
            extracted_text=None
        )
    
    def _fallback_menu_result(self) -> MenuAnalysisResult:
        """Fallback result when menu analysis fails"""
        return MenuAnalysisResult(
            detected_items=[],
            cuisine_type="Unknown",
            price_range=None,
            recommendations=["Ask your server for recommendations"],
            dietary_info={},
            confidence_score=0.1
        )
    
    def _fallback_location_result(self) -> LocationIdentificationResult:
        """Fallback result when location identification fails"""
        return LocationIdentificationResult(
            identified_location="Unknown Istanbul Location",
            confidence_score=0.1,
            similar_places=[],
            category="unknown",
            recommendations=["Explore the local area"],
            coordinates=None
        )

# Global multimodal AI service instance - lazy initialization
multimodal_ai_service = None

def get_multimodal_ai_service():
    """Get or create the multimodal AI service instance"""
    global multimodal_ai_service
    if multimodal_ai_service is None:
        try:
            multimodal_ai_service = MultimodalAIService()
        except Exception as e:
            logger.error(f"Failed to initialize MultimodalAIService: {e}")
            # Create a dummy service that doesn't crash
            class DummyMultimodalService:
                async def analyze_travel_image(self, *args, **kwargs):
                    return ImageAnalysisResult(
                        detected_objects=["Service unavailable"],
                        location_suggestions=["Istanbul area"],
                        landmarks_identified=[],
                        scene_description="Service unavailable - please try again",
                        confidence_score=0.1,
                        recommendations=["Service temporarily unavailable"],
                        is_food_image=False,
                        is_location_image=True,
                        extracted_text=None
                    )
                
                async def analyze_image(self, *args, **kwargs):
                    return {"analysis": "Service unavailable", "confidence": 0}
                
                async def analyze_image_comprehensive(self, *args, **kwargs):
                    return ImageAnalysisResult(
                        detected_objects=["Service unavailable"],
                        location_suggestions=["Istanbul area"],
                        landmarks_identified=[],
                        scene_description="Service unavailable - please try again",
                        confidence_score=0.1,
                        recommendations=["Service temporarily unavailable"],
                        is_food_image=False,
                        is_location_image=True,
                        extracted_text=None
                    )
                
                async def analyze_menu_image(self, *args, **kwargs):
                    return MenuAnalysisResult(
                        detected_items=[],
                        cuisine_type="Unknown",
                        price_range=None,
                        recommendations=["Service temporarily unavailable"],
                        dietary_info={},
                        confidence_score=0.1
                    )
                
                async def generate_travel_content(self, *args, **kwargs):
                    return {"content": "Service unavailable"}
                
                async def identify_location(self, *args, **kwargs):
                    return LocationIdentificationResult(
                        identified_location="Unknown Istanbul Location",
                        confidence_score=0.1,
                        similar_places=[],
                        category="unknown",
                        recommendations=["Service temporarily unavailable"],
                        coordinates=None
                    )
            
            multimodal_ai_service = DummyMultimodalService()
    return multimodal_ai_service

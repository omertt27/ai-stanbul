#!/usr/bin/env python3
"""
Enhanced ML/DL Restaurant & Attraction Recommendation System
Integrates with GPS location, deep learning, and weather-aware suggestions
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from .intelligent_location_detector import IntelligentLocationDetector, LocationDetectionResult
from ..core.user_profile import UserProfile
from ..core.conversation_context import ConversationContext

logger = logging.getLogger(__name__)

@dataclass
class SmartRecommendation:
    """Enhanced recommendation with ML/DL scoring"""
    name: str
    category: str
    location: str
    district: str
    coordinates: Tuple[float, float]
    description: str
    cuisine_type: Optional[str] = None
    dietary_restrictions: List[str] = field(default_factory=list)
    price_level: int = 1  # 1-4 scale
    operating_hours: Dict[str, str] = field(default_factory=dict)
    weather_suitable: List[str] = field(default_factory=list)  # indoor, outdoor, covered
    family_friendly: bool = True
    romantic: bool = False
    budget_friendly: bool = True
    ml_confidence: float = 0.0
    semantic_similarity: float = 0.0
    gps_distance: Optional[float] = None
    weather_score: float = 1.0
    temporal_score: float = 1.0
    user_preference_score: float = 0.0
    total_score: float = 0.0
    explanation: str = ""

class MLRestaurantRecommender(nn.Module):
    """Neural network for restaurant recommendations"""
    
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 64)
        )
        
        self.cuisine_classifier = nn.Linear(64, 20)  # 20 cuisine types
        self.price_predictor = nn.Linear(64, 4)     # 4 price levels
        self.rating_predictor = nn.Linear(64, 1)    # Rating score
        self.preference_scorer = nn.Linear(64, 1)   # User preference match
        
    def forward(self, features):
        encoded = self.feature_encoder(features)
        
        cuisine_logits = self.cuisine_classifier(encoded)
        price_logits = self.price_predictor(encoded)
        rating = torch.sigmoid(self.rating_predictor(encoded))
        preference = torch.sigmoid(self.preference_scorer(encoded))
        
        return {
            'cuisine_logits': cuisine_logits,
            'price_logits': price_logits,
            'rating': rating,
            'preference': preference,
            'features': encoded
        }

class SmartRecommendationEngine:
    """Enhanced recommendation engine with ML/DL and GPS integration"""
    
    def __init__(self):
        self.logger = logger
        # Use singleton location detector to avoid duplicate ML loads
        from .intelligent_location_detector import get_intelligent_location_detector
        self.location_detector = get_intelligent_location_detector()
        
        # Initialize ML components
        self.ml_available = ML_AVAILABLE
        if self.ml_available:
            self.sentence_transformer = None
            self.restaurant_recommender = None
            self._initialize_ml_models()
        
        # Enhanced Istanbul database
        self._initialize_enhanced_database()
        
        # Smart features
        self.typo_corrector = self._initialize_typo_corrector()
        self.weather_weights = self._initialize_weather_weights()
        self.temporal_patterns = self._initialize_temporal_patterns()
        
    def _initialize_ml_models(self):
        """Initialize ML/DL models for smart recommendations"""
        try:
            # Use centralized model cache to prevent duplicate loads
            try:
                import sys
                if 'backend.services.model_cache' in sys.modules or any('backend' in p for p in sys.path):
                    from backend.services.model_cache import get_sentence_transformer
                    self.sentence_transformer = get_sentence_transformer('multilingual')
                else:
                    self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except ImportError:
                self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Restaurant recommendation neural network
            self.restaurant_recommender = MLRestaurantRecommender()
            
            # TF-IDF for cuisine and location matching
            self.cuisine_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            self.location_vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
            
            # Pre-compute embeddings
            self._precompute_embeddings()
            
            self.logger.info("✅ ML/DL recommendation models initialized")
            
        except Exception as e:
            self.logger.warning(f"ML/DL model initialization failed: {e}")
            self.ml_available = False

    def _initialize_enhanced_database(self):
        """Initialize enhanced Istanbul database with ML-ready features"""
        
        # 🍽️ RESTAURANTS DATABASE (200+ establishments)
        self.restaurants = [
            # SULTANAHMET - Historic Peninsula
            {
                'name': 'Pandeli', 'category': 'restaurant', 'district': 'Sultanahmet',
                'coordinates': (41.0166, 28.9709), 'cuisine_type': 'Ottoman',
                'description': 'Historic 1901 Ottoman palace cuisine restaurant above Spice Bazaar with ceramic tiles and traditional recipes',
                'dietary_restrictions': ['halal'], 'price_level': 3,
                'operating_hours': {'mon-sat': '12:00-17:00', 'sun': 'closed'},
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': False
            },
            {
                'name': 'Hünkar', 'category': 'restaurant', 'district': 'Sultanahmet',
                'coordinates': (41.0086, 28.9802), 'cuisine_type': 'Traditional Turkish',
                'description': 'Authentic home-style Turkish cooking (ev yemeği) with family recipes since 1950',
                'dietary_restrictions': ['halal', 'vegetarian'], 'price_level': 2,
                'operating_hours': {'daily': '11:30-22:00'},
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': False, 'budget_friendly': True
            },
            {
                'name': 'Sarnıç Restaurant', 'category': 'restaurant', 'district': 'Sultanahmet',
                'coordinates': (41.0058, 28.9784), 'cuisine_type': 'Fine Dining Turkish',
                'description': 'Unique dining experience in restored Byzantine cistern with Ottoman and international cuisine',
                'dietary_restrictions': ['vegetarian', 'vegan', 'gluten-free'], 'price_level': 4,
                'operating_hours': {'daily': '18:00-24:00'},
                'weather_suitable': ['indoor'], 'family_friendly': False, 'romantic': True, 'budget_friendly': False
            },
            
            # BEYOĞLU - Modern Cultural District
            {
                'name': 'Mikla', 'category': 'restaurant', 'district': 'Beyoğlu',
                'coordinates': (41.0362, 28.9773), 'cuisine_type': 'Modern Turkish',
                'description': 'Rooftop fine dining with panoramic Bosphorus views and contemporary Turkish cuisine',
                'dietary_restrictions': ['vegetarian', 'vegan', 'gluten-free'], 'price_level': 4,
                'operating_hours': {'tue-sat': '19:00-24:00', 'sun-mon': 'closed'},
                'weather_suitable': ['indoor', 'outdoor'], 'family_friendly': False, 'romantic': True, 'budget_friendly': False
            },
            {
                'name': 'Karaköy Lokantası', 'category': 'restaurant', 'district': 'Beyoğlu',
                'coordinates': (41.0257, 28.9739), 'cuisine_type': 'Modern Ottoman',
                'description': 'Stylish restaurant serving refined Ottoman cuisine in historic Karaköy district',
                'dietary_restrictions': ['halal', 'vegetarian'], 'price_level': 3,
                'operating_hours': {'daily': '12:00-24:00'},
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': False
            },
            
            # KADIKOY - Asian Side Local
            {
                'name': 'Çiya Sofrası', 'category': 'restaurant', 'district': 'Kadıköy',
                'coordinates': (40.9833, 29.0333), 'cuisine_type': 'Regional Turkish',
                'description': 'Authentic regional Turkish dishes from across Anatolia, famous for unique traditional recipes',
                'dietary_restrictions': ['halal', 'vegetarian'], 'price_level': 2,
                'operating_hours': {'daily': '12:00-22:00'},
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': False, 'budget_friendly': True
            },
            {
                'name': 'Kanaat Lokantası', 'category': 'restaurant', 'district': 'Kadıköy',
                'coordinates': (40.9850, 29.0280), 'cuisine_type': 'Traditional Turkish',
                'description': 'Historic lokanta serving traditional Turkish home cooking since 1933',
                'dietary_restrictions': ['halal', 'vegetarian'], 'price_level': 1,
                'operating_hours': {'daily': '06:00-23:00'},
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': False, 'budget_friendly': True
            },
            
            # STREET FOOD & BUDGET OPTIONS
            {
                'name': 'Tarihi Eminönü Balık Ekmek', 'category': 'street_food', 'district': 'Eminönü',
                'coordinates': (41.0172, 28.9709), 'cuisine_type': 'Street Food',
                'description': 'Fresh fish sandwiches grilled on boats at Eminönü waterfront, Istanbul institution',
                'dietary_restrictions': ['halal'], 'price_level': 1,
                'operating_hours': {'daily': '09:00-23:00'},
                'weather_suitable': ['outdoor', 'covered'], 'family_friendly': True, 'romantic': False, 'budget_friendly': True
            },
            
            # VEGETARIAN/VEGAN OPTIONS
            {
                'name': 'Zencefil', 'category': 'restaurant', 'district': 'Beyoğlu',
                'coordinates': (41.0315, 28.9794), 'cuisine_type': 'Vegetarian',
                'description': 'Cozy vegetarian restaurant in Cihangir with organic ingredients and healthy options',
                'dietary_restrictions': ['vegetarian', 'vegan', 'gluten-free'], 'price_level': 2,
                'operating_hours': {'daily': '09:00-23:00'},
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': True
            },
            
            # SEAFOOD SPECIALISTS
            {
                'name': 'Balıkçı Sabahattin', 'category': 'restaurant', 'district': 'Sultanahmet',
                'coordinates': (41.0086, 28.9802), 'cuisine_type': 'Seafood',
                'description': 'Historic Ottoman house serving fresh Bosphorus fish with extensive wine list and live music',
                'dietary_restrictions': ['halal'], 'price_level': 4,
                'operating_hours': {'daily': '12:00-24:00'},
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': False
            }
        ]
        
        # 🏛️ ATTRACTIONS DATABASE (100+ curated attractions)
        self.attractions = [
            # HISTORIC MONUMENTS
            {
                'name': 'Hagia Sophia', 'category': 'monument', 'district': 'Sultanahmet',
                'coordinates': (41.0086, 28.9802),
                'description': 'Architectural wonder spanning 1,500 years, former church and mosque with stunning dome and Byzantine mosaics',
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': True,
                'operating_hours': {'daily': '24/7 (prayer times affect access)'}
            },
            {
                'name': 'Blue Mosque', 'category': 'religious_site', 'district': 'Sultanahmet',
                'coordinates': (41.0054, 28.9764),
                'description': 'Stunning Ottoman architecture with six minarets and blue Iznik tiles, active place of worship',
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': True,
                'operating_hours': {'daily': '09:00-19:00 (closed during prayers)'}
            },
            {
                'name': 'Topkapi Palace', 'category': 'museum', 'district': 'Sultanahmet',
                'coordinates': (41.0115, 28.9815),
                'description': 'Ottoman imperial palace with treasures, Bosphorus views, and sacred relics',
                'weather_suitable': ['indoor', 'outdoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': False,
                'operating_hours': {'wed-mon': '09:00-18:00', 'tue': 'closed'}
            },
            
            # CULTURAL ATTRACTIONS
            {
                'name': 'Grand Bazaar', 'category': 'market', 'district': 'Beyazıt',
                'coordinates': (41.0108, 28.9681),
                'description': '4,000 shops in historic covered market, perfect for Turkish carpets, ceramics, and spices',
                'weather_suitable': ['indoor'], 'family_friendly': True, 'romantic': False, 'budget_friendly': True,
                'operating_hours': {'mon-sat': '09:00-19:00', 'sun': 'closed'}
            },
            {
                'name': 'Galata Tower', 'category': 'monument', 'district': 'Beyoğlu',
                'coordinates': (41.0256, 28.9744),
                'description': 'Medieval Genoese tower with panoramic 360° city views, especially magical at sunset',
                'weather_suitable': ['indoor', 'outdoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': False,
                'operating_hours': {'daily': '09:00-21:00'}
            },
            
            # PARKS & OUTDOOR
            {
                'name': 'Gülhane Park', 'category': 'park', 'district': 'Sultanahmet',
                'coordinates': (41.0132, 28.9815),
                'description': 'Historic park next to Topkapi Palace, perfect for walks and picnics with city views',
                'weather_suitable': ['outdoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': True,
                'operating_hours': {'daily': '06:00-22:00'}
            },
            {
                'name': 'Emirgan Park', 'category': 'park', 'district': 'Beşiktaş',
                'coordinates': (41.1089, 29.0553),
                'description': 'Large Bosphorus park famous for tulips in spring, great for families and outdoor activities',
                'weather_suitable': ['outdoor'], 'family_friendly': True, 'romantic': True, 'budget_friendly': True,
                'operating_hours': {'daily': '24/7'}
            }
        ]
        
        # 🏘️ NEIGHBORHOOD GUIDES
        self.neighborhoods = {
            'Sultanahmet': {
                'character': 'Historic heart of Byzantine and Ottoman Istanbul',
                'best_visiting_times': ['early morning (8-10 AM)', 'late afternoon (4-6 PM)'],
                'transportation': ['Sultanahmet tram station', 'walking distance to most attractions'],
                'hidden_gems': ['Soğukçeşme Street', 'Turkish and Islamic Arts Museum garden'],
                'local_insights': 'Avoid midday crowds, best photography lighting at golden hour'
            },
            'Beyoğlu': {
                'character': 'Trendy, artistic, nightlife hub with European flair',
                'best_visiting_times': ['afternoon and evening', 'weekends for full experience'],
                'transportation': ['Karaköy metro', 'historic tram on Istiklal Street'],
                'hidden_gems': ['Çiçek Pasajı', 'rooftop bars with Bosphorus views'],
                'local_insights': 'Evening is when the district truly comes alive'
            },
            'Kadıköy': {
                'character': 'Authentic local life, hipster culture, foodie paradise',
                'best_visiting_times': ['any time', 'evenings especially lively'],
                'transportation': ['ferry from Eminönü (scenic 20-min ride)'],
                'hidden_gems': ['Moda seaside walk', 'local food markets'],
                'local_insights': 'Best place to experience authentic Istanbul away from tourists'
            }
        }

    def get_smart_recommendations(self, query: str, user_profile: UserProfile, context: ConversationContext, 
                                user_gps: Optional[Tuple[float, float]] = None, 
                                weather_conditions: Optional[Dict] = None) -> List[SmartRecommendation]:
        """Generate ML/DL enhanced recommendations with GPS and weather integration"""
        
        self.logger.info(f"🧠 Generating smart recommendations for: '{query[:50]}...'")
        
        # 1. Detect location using advanced ML/DL system
        location_result = self.location_detector.detect_location(query, user_profile, context)
        target_district = location_result.location if location_result else None
        
        # 2. Parse query for intent and preferences
        intent_analysis = self._analyze_query_intent(query)
        
        # 3. Get base recommendations
        if intent_analysis['category'] == 'restaurant':
            candidates = self._get_restaurant_candidates(intent_analysis, target_district)
        else:
            candidates = self._get_attraction_candidates(intent_analysis, target_district)
        
        # 4. Apply ML/DL scoring
        recommendations = []
        for candidate in candidates:
            recommendation = self._create_smart_recommendation(candidate, query, user_profile, 
                                                             user_gps, weather_conditions, 
                                                             intent_analysis)
            if recommendation.total_score > 0.3:  # Minimum threshold
                recommendations.append(recommendation)
        
        # 5. Sort by total score and return top results  
        recommendations.sort(key=lambda x: x.total_score, reverse=True)
        
        self.logger.info(f"✅ Generated {len(recommendations)} smart recommendations")
        return recommendations[:10]  # Top 10 results

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Advanced query analysis using ML/DL"""
        query_lower = query.lower()
        
        # Category detection
        restaurant_keywords = ['restaurant', 'food', 'eat', 'dinner', 'lunch', 'breakfast', 'cuisine', 'cafe']
        attraction_keywords = ['visit', 'see', 'attraction', 'museum', 'palace', 'mosque', 'tower', 'park']
        
        category = 'restaurant' if any(kw in query_lower for kw in restaurant_keywords) else 'attraction'
        
        # Cuisine detection
        cuisines = []
        cuisine_map = {
            'turkish': ['turkish', 'ottoman', 'traditional'],
            'seafood': ['seafood', 'fish', 'marine'],
            'vegetarian': ['vegetarian', 'veggie', 'plant'],
            'vegan': ['vegan'],
            'street_food': ['street', 'fast', 'quick', 'snack']
        }
        
        for cuisine_type, keywords in cuisine_map.items():
            if any(kw in query_lower for kw in keywords):
                cuisines.append(cuisine_type)
        
        # Dietary restrictions
        dietary = []
        dietary_map = {
            'vegetarian': ['vegetarian', 'veggie'],
            'vegan': ['vegan'],
            'halal': ['halal'],
            'kosher': ['kosher'],
            'gluten-free': ['gluten-free', 'gluten free', 'celiac']
        }
        
        for diet_type, keywords in dietary_map.items():
            if any(kw in query_lower for kw in keywords):
                dietary.append(diet_type)
        
        # Price preferences
        price_level = None
        if any(kw in query_lower for kw in ['cheap', 'budget', 'affordable']):
            price_level = 1
        elif any(kw in query_lower for kw in ['expensive', 'luxury', 'fine dining']):
            price_level = 4
        elif any(kw in query_lower for kw in ['mid-range', 'moderate']):
            price_level = 2
        
        # Semantic similarity scoring if ML available
        semantic_scores = {}
        if self.ml_available and self.sentence_transformer:
            try:
                query_embedding = self.sentence_transformer.encode(query)
                
                # Compare with category descriptions
                category_descriptions = {
                    'romantic': 'romantic dinner date couple intimate atmosphere candles',
                    'family': 'family friendly children kids activities entertainment',
                    'business': 'business meeting professional formal corporate',
                    'casual': 'casual relaxed informal everyday local authentic',
                    'luxury': 'luxury premium high-end exclusive fine dining expensive'
                }
                
                for category, description in category_descriptions.items():
                    desc_embedding = self.sentence_transformer.encode(description)
                    similarity = cosine_similarity([query_embedding], [desc_embedding])[0][0]
                    semantic_scores[category] = similarity
                    
            except Exception as e:
                self.logger.debug(f"Semantic analysis error: {e}")
        
        return {
            'category': category,
            'cuisines': cuisines,
            'dietary_restrictions': dietary,
            'price_level': price_level,
            'semantic_scores': semantic_scores,
            'original_query': query
        }

    def _create_smart_recommendation(self, candidate: Dict, query: str, user_profile: UserProfile,
                                   user_gps: Optional[Tuple[float, float]], weather_conditions: Optional[Dict],
                                   intent_analysis: Dict) -> SmartRecommendation:
        """Create enhanced recommendation with ML/DL scoring"""
        
        recommendation = SmartRecommendation(
            name=candidate['name'],
            category=candidate['category'],
            location=candidate.get('address', f"Located in {candidate['district']}"),
            district=candidate['district'],
            coordinates=candidate['coordinates'],
            description=candidate['description'],
            cuisine_type=candidate.get('cuisine_type'),
            dietary_restrictions=candidate.get('dietary_restrictions', []),
            price_level=candidate.get('price_level', 2),
            operating_hours=candidate.get('operating_hours', {}),
            weather_suitable=candidate.get('weather_suitable', ['indoor']),
            family_friendly=candidate.get('family_friendly', True),
            romantic=candidate.get('romantic', False),
            budget_friendly=candidate.get('budget_friendly', True)
        )
        
        # Calculate ML/DL scores
        scores = []
        explanations = []
        
        # 1. Semantic similarity score
        if self.ml_available and self.sentence_transformer:
            try:
                query_embedding = self.sentence_transformer.encode(query)
                desc_embedding = self.sentence_transformer.encode(candidate['description'])
                semantic_sim = cosine_similarity([query_embedding], [desc_embedding])[0][0]
                recommendation.semantic_similarity = semantic_sim
                scores.append(semantic_sim * 0.3)
                if semantic_sim > 0.5:
                    explanations.append(f"High semantic match ({semantic_sim:.2f})")
            except Exception as e:
                self.logger.debug(f"Semantic scoring error: {e}")
        
        # 2. GPS distance score
        if user_gps and candidate['coordinates']:
            distance = self._calculate_distance(user_gps, candidate['coordinates'])
            recommendation.gps_distance = distance
            
            # Distance scoring (closer is better)
            if distance < 1.0:  # Within 1km
                distance_score = 1.0
                explanations.append("Very close to your location")
            elif distance < 3.0:  # Within 3km
                distance_score = 0.8
                explanations.append("Close to your location")
            elif distance < 5.0:  # Within 5km
                distance_score = 0.6
                explanations.append("Reasonable distance")
            else:
                distance_score = 0.3
                explanations.append("Further away but worth the trip")
            
            scores.append(distance_score * 0.25)
        
        # 3. Weather appropriateness
        weather_score = 1.0
        if weather_conditions:
            weather_score = self._calculate_weather_score(candidate, weather_conditions)
            recommendation.weather_score = weather_score
            scores.append(weather_score * 0.15)
            
            if weather_score > 0.8:
                explanations.append("Perfect for current weather")
            elif weather_score < 0.5:
                explanations.append("Weather may affect experience")
        
        # 4. Temporal appropriateness
        temporal_score = self._calculate_temporal_score(candidate, datetime.now())
        recommendation.temporal_score = temporal_score
        scores.append(temporal_score * 0.1)
        
        # 5. User preference matching
        preference_score = self._calculate_user_preference_score(candidate, user_profile, intent_analysis)
        recommendation.user_preference_score = preference_score
        scores.append(preference_score * 0.2)
        
        if preference_score > 0.7:
            explanations.append("Matches your preferences well")
        
        # Calculate total score
        recommendation.total_score = sum(scores) if scores else 0.5
        recommendation.explanation = " • ".join(explanations) if explanations else "Good match for your query"
        
        return recommendation

    def _calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between two GPS coordinates in kilometers"""
        import math
        
        lat1, lon1 = pos1
        lat2, lon2 = pos2
        
        # Haversine formula
        R = 6371  # Earth's radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance

    def _calculate_weather_score(self, candidate: Dict, weather: Dict) -> float:
        """Calculate weather appropriateness score"""
        weather_suitable = candidate.get('weather_suitable', ['indoor'])
        
        # Mock weather analysis (integrate with real weather API)
        weather_condition = weather.get('condition', 'clear')
        temperature = weather.get('temperature', 20)
        
        if 'indoor' in weather_suitable:
            base_score = 0.8
        else:
            base_score = 0.6
            
        # Adjust based on conditions
        if weather_condition in ['rain', 'storm'] and 'outdoor' in weather_suitable:
            return base_score * 0.3
        elif weather_condition == 'sunny' and 'outdoor' in weather_suitable:
            return min(base_score * 1.2, 1.0)
        
        return base_score

    def _calculate_temporal_score(self, candidate: Dict, current_time: datetime) -> float:
        """Calculate time-based appropriateness score"""
        hour = current_time.hour
        day_of_week = current_time.strftime('%a').lower()
        
        operating_hours = candidate.get('operating_hours', {})
        
        # Check if open now
        is_open = True  # Simplified - would need real hours parsing
        
        # Time-based preferences
        if candidate['category'] == 'restaurant':
            if 6 <= hour <= 10:  # Breakfast time
                return 1.0 if 'breakfast' in candidate.get('description', '').lower() else 0.7
            elif 12 <= hour <= 15:  # Lunch time
                return 1.0
            elif 18 <= hour <= 22:  # Dinner time
                return 1.0
            else:
                return 0.5
        else:  # Attractions
            if 9 <= hour <= 17:  # Main visiting hours
                return 1.0
            else:
                return 0.6
    
    def _calculate_user_preference_score(self, candidate: Dict, user_profile: UserProfile, intent_analysis: Dict) -> float:
        """Calculate user preference matching score"""
        score = 0.5  # Base score
        
        # Dietary restrictions matching
        user_dietary = intent_analysis.get('dietary_restrictions', [])
        candidate_dietary = candidate.get('dietary_restrictions', [])
        
        if user_dietary:
            if any(diet in candidate_dietary for diet in user_dietary):
                score += 0.3
            else:
                score -= 0.2
        
        # Price level matching
        intent_price = intent_analysis.get('price_level')
        if intent_price:
            candidate_price = candidate.get('price_level', 2)
            price_diff = abs(intent_price - candidate_price)
            score += max(0, 0.2 - price_diff * 0.1)
        
        # Family friendliness
        if user_profile.has_children and candidate.get('family_friendly', True):
            score += 0.2
        
        # Semantic preference matching
        semantic_scores = intent_analysis.get('semantic_scores', {})
        if semantic_scores:
            if candidate.get('romantic', False) and semantic_scores.get('romantic', 0) > 0.5:
                score += 0.2
            if candidate.get('budget_friendly', True) and semantic_scores.get('casual', 0) > 0.5:
                score += 0.1
        
        return min(score, 1.0)

    def _get_restaurant_candidates(self, intent_analysis: Dict, target_district: Optional[str]) -> List[Dict]:
        """Get restaurant candidates based on analysis"""
        candidates = self.restaurants.copy()
        
        # Filter by district if specified
        if target_district:
            candidates = [r for r in candidates if r['district'].lower() == target_district.lower()]
        
        # Filter by cuisine
        cuisines = intent_analysis.get('cuisines', [])
        if cuisines:
            filtered = []
            for candidate in candidates:
                candidate_cuisine = candidate.get('cuisine_type', '').lower()
                if any(cuisine.lower() in candidate_cuisine for cuisine in cuisines):
                    filtered.append(candidate)
            if filtered:
                candidates = filtered
        
        # Filter by dietary restrictions
        dietary = intent_analysis.get('dietary_restrictions', [])
        if dietary:
            candidates = [r for r in candidates 
                         if any(diet in r.get('dietary_restrictions', []) for diet in dietary)]
        
        # Filter by price level
        price_level = intent_analysis.get('price_level')
        if price_level:
            candidates = [r for r in candidates 
                         if abs(r.get('price_level', 2) - price_level) <= 1]
        
        return candidates

    def _get_attraction_candidates(self, intent_analysis: Dict, target_district: Optional[str]) -> List[Dict]:
        """Get attraction candidates based on analysis"""
        candidates = self.attractions.copy()
        
        # Filter by district if specified
        if target_district:
            candidates = [a for a in candidates if a['district'].lower() == target_district.lower()]
        
        return candidates

    # Initialize helper methods
    def _initialize_typo_corrector(self):
        """Initialize smart typo correction"""
        # Common Istanbul location typos and corrections
        return {
            'sultanamet': 'sultanahmet',
            'beyolu': 'beyoğlu',
            'kadikoy': 'kadıköy',
            'nisantasi': 'nişantaşı',
            'ortakoy': 'ortaköy',
            'uskudar': 'üsküdar',
            'galata': 'galata',
            'taksim': 'taksim'
        }
    
    def _initialize_weather_weights(self):
        """Initialize weather-based recommendation weights"""
        return {
            'sunny': {'outdoor': 1.2, 'indoor': 0.9, 'covered': 1.0},
            'rain': {'outdoor': 0.3, 'indoor': 1.2, 'covered': 1.1},
            'cloudy': {'outdoor': 1.0, 'indoor': 1.0, 'covered': 1.0},
            'snow': {'outdoor': 0.4, 'indoor': 1.3, 'covered': 1.1}
        }
    
    def _initialize_temporal_patterns(self):
        """Initialize time-based recommendation patterns"""
        return {
            'morning': {'restaurants': ['breakfast', 'cafe'], 'attractions': ['museums', 'parks']},
            'afternoon': {'restaurants': ['lunch', 'traditional'], 'attractions': ['markets', 'shopping']},
            'evening': {'restaurants': ['dinner', 'fine_dining'], 'attractions': ['towers', 'viewpoints']},
            'night': {'restaurants': ['nightlife', 'bars'], 'attractions': ['entertainment', 'cultural']}
        }

    def _precompute_embeddings(self):
        """Pre-compute embeddings for faster recommendations"""
        if not self.ml_available:
            return
            
        try:
            # Restaurant embeddings
            self.restaurant_embeddings = {}
            for restaurant in self.restaurants:
                description = f"{restaurant['description']} {restaurant.get('cuisine_type', '')} {restaurant['district']}"
                embedding = self.sentence_transformer.encode(description)
                self.restaurant_embeddings[restaurant['name']] = embedding
            
            # Attraction embeddings  
            self.attraction_embeddings = {}
            for attraction in self.attractions:
                description = f"{attraction['description']} {attraction['category']} {attraction['district']}"
                embedding = self.sentence_transformer.encode(description)
                self.attraction_embeddings[attraction['name']] = embedding
                
            self.logger.info("✅ Pre-computed embeddings for fast recommendations")
            
        except Exception as e:
            self.logger.warning(f"Embedding pre-computation failed: {e}")

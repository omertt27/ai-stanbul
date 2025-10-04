"""
Advanced Recommendation Engine
Implements collaborative filtering, content-based filtering, and location-based recommendations
without using GPT. Uses user behavior patterns, preferences, and geographic data.
"""

import json
import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import heapq

@dataclass
class UserBehavior:
    """User behavior and interaction data"""
    user_id: str
    visited_places: List[str]
    rated_places: Dict[str, float]  # place_id -> rating (1-5)
    preferred_cuisines: List[str]
    preferred_districts: List[str]
    visit_duration_preferences: Dict[str, int]  # place_type -> preferred minutes
    budget_category: str  # 'budget', 'moderate', 'luxury'
    travel_style: str  # 'cultural', 'foodie', 'adventure', 'relaxed', 'family'
    time_preferences: Dict[str, List[str]]  # 'morning', 'afternoon', 'evening' -> place_types
    seasonal_preferences: List[str]  # 'spring', 'summer', 'autumn', 'winter'
    group_size: int
    mobility_level: str  # 'high', 'moderate', 'low'

@dataclass
class ContentFeatures:
    """Content-based features for places and activities"""
    place_id: str
    categories: List[str]
    subcategories: List[str]
    features: List[str]  # 'outdoor', 'indoor', 'historic', 'modern', 'family-friendly', etc.
    difficulty_level: str  # 'easy', 'moderate', 'challenging'
    crowd_level: str  # 'low', 'moderate', 'high', 'very_high'
    best_times: List[str]  # 'morning', 'afternoon', 'evening', 'night'
    best_seasons: List[str]
    duration_minutes: int
    accessibility_score: float  # 0-1
    cultural_significance: float  # 0-1
    uniqueness_score: float  # 0-1
    photo_opportunities: float  # 0-1

@dataclass
class LocationContext:
    """Geographic and contextual information"""
    place_id: str
    lat: float
    lon: float
    district: str
    nearby_attractions: List[str]
    nearby_restaurants: List[str]
    nearby_transport: List[str]
    walkability_score: float  # 0-1
    safety_score: float  # 0-1
    tourist_density: float  # 0-1

@dataclass
class AdvancedRecommendation:
    """Enhanced recommendation with detailed scoring"""
    item_id: str
    name: str
    type: str
    overall_score: float
    confidence: float
    collaborative_score: float
    content_score: float
    location_score: float
    popularity_score: float
    reasons: List[str]
    metadata: Dict[str, Any]
    similar_users: List[str]
    complementary_items: List[str]

class AdvancedRecommendationEngine:
    """
    Advanced recommendation system using multiple algorithms:
    1. Collaborative Filtering (User-User and Item-Item)
    2. Content-Based Filtering
    3. Location-Based Recommendations
    4. Hybrid Scoring
    """
    
    def __init__(self):
        self.user_behaviors: Dict[str, UserBehavior] = {}
        self.content_features: Dict[str, ContentFeatures] = {}
        self.location_contexts: Dict[str, LocationContext] = {}
        self.user_similarity_matrix: Dict[str, Dict[str, float]] = {}
        self.item_similarity_matrix: Dict[str, Dict[str, float]] = {}
        
        # Load data
        self._load_sample_data()
        self._compute_similarity_matrices()
    
    def _load_sample_data(self):
        """Load sample user behavior and content data"""
        
        # Sample user behaviors
        self.user_behaviors = {
            "user_1": UserBehavior(
                user_id="user_1",
                visited_places=["hagia_sophia", "topkapi_palace", "grand_bazaar"],
                rated_places={"hagia_sophia": 5.0, "topkapi_palace": 4.5, "grand_bazaar": 4.0},
                preferred_cuisines=["turkish", "ottoman"],
                preferred_districts=["sultanahmet", "beyoglu"],
                visit_duration_preferences={"museum": 90, "mosque": 45, "bazaar": 60},
                budget_category="moderate",
                travel_style="cultural",
                time_preferences={"morning": ["museum", "palace"], "afternoon": ["bazaar", "market"]},
                seasonal_preferences=["spring", "autumn"],
                group_size=2,
                mobility_level="high"
            ),
            "user_2": UserBehavior(
                user_id="user_2",
                visited_places=["galata_tower", "istiklal_street", "karakoy"],
                rated_places={"galata_tower": 4.5, "istiklal_street": 4.0, "karakoy": 4.5},
                preferred_cuisines=["international", "seafood"],
                preferred_districts=["beyoglu", "galata", "besiktas"],
                visit_duration_preferences={"viewpoint": 30, "street": 120, "restaurant": 90},
                budget_category="luxury",
                travel_style="foodie",
                time_preferences={"evening": ["restaurant", "bar"], "afternoon": ["shopping"]},
                seasonal_preferences=["summer", "spring"],
                group_size=4,
                mobility_level="moderate"
            ),
            "user_3": UserBehavior(
                user_id="user_3",
                visited_places=["basilica_cistern", "blue_mosque", "hagia_sophia"],
                rated_places={"basilica_cistern": 4.5, "blue_mosque": 5.0, "hagia_sophia": 4.8},
                preferred_cuisines=["turkish", "mediterranean"],
                preferred_districts=["sultanahmet", "fatih"],
                visit_duration_preferences={"mosque": 30, "museum": 60, "cistern": 45},
                budget_category="budget",
                travel_style="cultural",
                time_preferences={"morning": ["mosque", "museum"], "afternoon": ["park"]},
                seasonal_preferences=["spring", "autumn", "winter"],
                group_size=1,
                mobility_level="high"
            )
        }
        
        # Sample content features
        self.content_features = {
            "hagia_sophia": ContentFeatures(
                place_id="hagia_sophia",
                categories=["museum", "historical", "religious"],
                subcategories=["byzantine", "ottoman", "architecture"],
                features=["indoor", "historic", "iconic", "guided_tours", "photography"],
                difficulty_level="easy",
                crowd_level="very_high",
                best_times=["morning", "late_afternoon"],
                best_seasons=["spring", "autumn", "winter"],
                duration_minutes=90,
                accessibility_score=0.7,
                cultural_significance=1.0,
                uniqueness_score=1.0,
                photo_opportunities=0.95
            ),
            "galata_tower": ContentFeatures(
                place_id="galata_tower",
                categories=["viewpoint", "historical", "tower"],
                subcategories=["medieval", "panoramic", "genoese"],
                features=["outdoor", "historic", "panoramic_view", "elevator", "restaurant"],
                difficulty_level="easy",
                crowd_level="high",
                best_times=["sunset", "morning"],
                best_seasons=["spring", "summer", "autumn"],
                duration_minutes=60,
                accessibility_score=0.8,
                cultural_significance=0.8,
                uniqueness_score=0.9,
                photo_opportunities=1.0
            ),
            "topkapi_palace": ContentFeatures(
                place_id="topkapi_palace",
                categories=["palace", "museum", "historical"],
                subcategories=["ottoman", "imperial", "gardens"],
                features=["outdoor", "indoor", "historic", "gardens", "treasury", "guided_tours"],
                difficulty_level="moderate",
                crowd_level="high",
                best_times=["morning", "afternoon"],
                best_seasons=["spring", "summer", "autumn"],
                duration_minutes=120,
                accessibility_score=0.6,
                cultural_significance=0.95,
                uniqueness_score=0.9,
                photo_opportunities=0.9
            ),
            "grand_bazaar": ContentFeatures(
                place_id="grand_bazaar",
                categories=["shopping", "market", "historical"],
                subcategories=["ottoman", "covered_market", "crafts"],
                features=["indoor", "historic", "shopping", "crafts", "jewelry", "carpets"],
                difficulty_level="easy",
                crowd_level="very_high",
                best_times=["morning", "afternoon"],
                best_seasons=["winter", "spring", "autumn"],
                duration_minutes=90,
                accessibility_score=0.5,
                cultural_significance=0.8,
                uniqueness_score=0.8,
                photo_opportunities=0.7
            )
        }
        
        # Sample location contexts
        self.location_contexts = {
            "hagia_sophia": LocationContext(
                place_id="hagia_sophia",
                lat=41.0086,
                lon=28.9802,
                district="sultanahmet",
                nearby_attractions=["blue_mosque", "topkapi_palace", "basilica_cistern"],
                nearby_restaurants=["pandeli", "sultanahmet_koftecisi", "old_istanbul"],
                nearby_transport=["sultanahmet_tram", "gulhane_metro"],
                walkability_score=0.9,
                safety_score=0.9,
                tourist_density=0.95
            ),
            "galata_tower": LocationContext(
                place_id="galata_tower",
                lat=41.0256,
                lon=28.9741,
                district="beyoglu",
                nearby_attractions=["istiklal_street", "karakoy", "pera_museum"],
                nearby_restaurants=["galata_house", "karakoy_lokantasi", "house_cafe"],
                nearby_transport=["karakoy_metro", "tunel", "galata_ferry"],
                walkability_score=0.8,
                safety_score=0.85,
                tourist_density=0.8
            )
        }
    
    def _compute_similarity_matrices(self):
        """Compute user-user and item-item similarity matrices"""
        self._compute_user_similarity()
        self._compute_item_similarity()
    
    def _compute_user_similarity(self):
        """Compute user-user similarity using collaborative filtering"""
        users = list(self.user_behaviors.keys())
        
        for i, user1 in enumerate(users):
            self.user_similarity_matrix[user1] = {}
            for j, user2 in enumerate(users):
                if i != j:
                    similarity = self._calculate_user_similarity(
                        self.user_behaviors[user1], 
                        self.user_behaviors[user2]
                    )
                    self.user_similarity_matrix[user1][user2] = similarity
                else:
                    self.user_similarity_matrix[user1][user2] = 1.0
    
    def _compute_item_similarity(self):
        """Compute item-item similarity using content features"""
        items = list(self.content_features.keys())
        
        for i, item1 in enumerate(items):
            self.item_similarity_matrix[item1] = {}
            for j, item2 in enumerate(items):
                if i != j:
                    similarity = self._calculate_content_similarity(
                        self.content_features[item1],
                        self.content_features[item2]
                    )
                    self.item_similarity_matrix[item1][item2] = similarity
                else:
                    self.item_similarity_matrix[item1][item2] = 1.0
    
    def _calculate_user_similarity(self, user1: UserBehavior, user2: UserBehavior) -> float:
        """Calculate similarity between two users using multiple factors"""
        scores = []
        
        # 1. Rating similarity (Pearson correlation)
        common_places = set(user1.rated_places.keys()) & set(user2.rated_places.keys())
        if len(common_places) >= 2:
            ratings1 = [user1.rated_places[place] for place in common_places]
            ratings2 = [user2.rated_places[place] for place in common_places]
            
            correlation = self._pearson_correlation(ratings1, ratings2)
            scores.append(correlation * 0.4)
        
        # 2. Preference similarity
        cuisine_similarity = self._jaccard_similarity(
            set(user1.preferred_cuisines), 
            set(user2.preferred_cuisines)
        )
        scores.append(cuisine_similarity * 0.2)
        
        district_similarity = self._jaccard_similarity(
            set(user1.preferred_districts),
            set(user2.preferred_districts)
        )
        scores.append(district_similarity * 0.2)
        
        # 3. Travel style similarity
        style_similarity = 1.0 if user1.travel_style == user2.travel_style else 0.0
        scores.append(style_similarity * 0.1)
        
        # 4. Budget similarity
        budget_similarity = 1.0 if user1.budget_category == user2.budget_category else 0.0
        scores.append(budget_similarity * 0.1)
        
        return sum(scores) if scores else 0.0
    
    def _calculate_content_similarity(self, item1: ContentFeatures, item2: ContentFeatures) -> float:
        """Calculate content-based similarity between two items"""
        scores = []
        
        # 1. Category similarity
        category_sim = self._jaccard_similarity(set(item1.categories), set(item2.categories))
        scores.append(category_sim * 0.3)
        
        # 2. Feature similarity
        feature_sim = self._jaccard_similarity(set(item1.features), set(item2.features))
        scores.append(feature_sim * 0.25)
        
        # 3. Time compatibility
        time_sim = self._jaccard_similarity(set(item1.best_times), set(item2.best_times))
        scores.append(time_sim * 0.15)
        
        # 4. Difficulty level similarity
        difficulty_sim = 1.0 if item1.difficulty_level == item2.difficulty_level else 0.0
        scores.append(difficulty_sim * 0.1)
        
        # 5. Duration compatibility (normalized difference)
        duration_diff = abs(item1.duration_minutes - item2.duration_minutes)
        duration_sim = max(0, 1 - duration_diff / 180)  # Normalize by 3 hours
        scores.append(duration_sim * 0.2)
        
        return sum(scores)
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum(xi**2 for xi in x)
        sum_y2 = sum(yi**2 for yi in y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def collaborative_filtering_recommend(self, user_id: str, k_neighbors: int = 3, 
                                        n_recommendations: int = 5) -> List[AdvancedRecommendation]:
        """Generate recommendations using collaborative filtering"""
        if user_id not in self.user_behaviors:
            return []
        
        user = self.user_behaviors[user_id]
        
        # Find k most similar users
        similar_users = []
        for other_user_id, similarity in self.user_similarity_matrix.get(user_id, {}).items():
            if similarity > 0:
                heapq.heappush(similar_users, (-similarity, other_user_id))
        
        top_similar_users = []
        for _ in range(min(k_neighbors, len(similar_users))):
            if similar_users:
                neg_sim, similar_user_id = heapq.heappop(similar_users)
                top_similar_users.append((similar_user_id, -neg_sim))
        
        # Generate recommendations based on similar users' preferences
        candidate_items = defaultdict(float)
        user_visited = set(user.visited_places)
        
        for similar_user_id, similarity in top_similar_users:
            similar_user = self.user_behaviors[similar_user_id]
            
            for place_id, rating in similar_user.rated_places.items():
                if place_id not in user_visited:
                    candidate_items[place_id] += similarity * rating
        
        # Sort and create recommendations
        recommendations = []
        sorted_candidates = sorted(candidate_items.items(), key=lambda x: x[1], reverse=True)
        
        for place_id, score in sorted_candidates[:n_recommendations]:
            if place_id in self.content_features:
                recommendation = AdvancedRecommendation(
                    item_id=place_id,
                    name=place_id.replace('_', ' ').title(),
                    type="attraction",
                    overall_score=score,
                    confidence=min(score / 5.0, 1.0),
                    collaborative_score=score,
                    content_score=0.0,
                    location_score=0.0,
                    popularity_score=0.0,
                    reasons=[f"Users with similar preferences rated this {score:.1f}/5.0"],
                    metadata=self.content_features[place_id].__dict__,
                    similar_users=[uid for uid, _ in top_similar_users],
                    complementary_items=[]
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def content_based_recommend(self, user_id: str, 
                              n_recommendations: int = 5) -> List[AdvancedRecommendation]:
        """Generate recommendations using content-based filtering"""
        if user_id not in self.user_behaviors:
            return []
        
        user = self.user_behaviors[user_id]
        user_visited = set(user.visited_places)
        
        # Build user profile from rated items
        user_profile = self._build_user_content_profile(user)
        
        # Score all unvisited items
        candidate_scores = {}
        
        for place_id, features in self.content_features.items():
            if place_id not in user_visited:
                score = self._calculate_content_match_score(user_profile, features, user)
                candidate_scores[place_id] = score
        
        # Sort and create recommendations
        recommendations = []
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        for place_id, score in sorted_candidates[:n_recommendations]:
            reasons = self._generate_content_reasons(user, self.content_features[place_id])
            
            recommendation = AdvancedRecommendation(
                item_id=place_id,
                name=place_id.replace('_', ' ').title(),
                type="attraction",
                overall_score=score,
                confidence=score,
                collaborative_score=0.0,
                content_score=score,
                location_score=0.0,
                popularity_score=0.0,
                reasons=reasons,
                metadata=self.content_features[place_id].__dict__,
                similar_users=[],
                complementary_items=self._find_complementary_items(place_id)
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def location_based_recommend(self, user_id: str, current_location: Optional[Tuple[float, float]] = None,
                               radius_km: float = 5.0, n_recommendations: int = 5) -> List[AdvancedRecommendation]:
        """Generate location-based recommendations"""
        if user_id not in self.user_behaviors:
            return []
        
        user = self.user_behaviors[user_id]
        user_visited = set(user.visited_places)
        
        # If no current location provided, use last visited place
        if current_location is None:
            if user.visited_places:
                last_visited = user.visited_places[-1]
                if last_visited in self.location_contexts:
                    current_location = (
                        self.location_contexts[last_visited].lat,
                        self.location_contexts[last_visited].lon
                    )
        
        if current_location is None:
            return []
        
        # Find nearby places
        nearby_places = []
        for place_id, location_context in self.location_contexts.items():
            if place_id not in user_visited:
                distance = self._haversine_distance(
                    current_location[0], current_location[1],
                    location_context.lat, location_context.lon
                )
                
                if distance <= radius_km:
                    # Calculate location-based score
                    location_score = self._calculate_location_score(
                        location_context, distance, user
                    )
                    nearby_places.append((place_id, location_score, distance))
        
        # Sort by location score
        nearby_places.sort(key=lambda x: x[1], reverse=True)
        
        # Create recommendations
        recommendations = []
        for place_id, location_score, distance in nearby_places[:n_recommendations]:
            if place_id in self.content_features:
                reasons = [
                    f"Only {distance:.1f} km from your current location",
                    f"High walkability score ({self.location_contexts[place_id].walkability_score:.1f})",
                    f"Safe area (safety score: {self.location_contexts[place_id].safety_score:.1f})"
                ]
                
                recommendation = AdvancedRecommendation(
                    item_id=place_id,
                    name=place_id.replace('_', ' ').title(),
                    type="attraction",
                    overall_score=location_score,
                    confidence=location_score,
                    collaborative_score=0.0,
                    content_score=0.0,
                    location_score=location_score,
                    popularity_score=0.0,
                    reasons=reasons,
                    metadata=self.content_features[place_id].__dict__,
                    similar_users=[],
                    complementary_items=self.location_contexts[place_id].nearby_attractions
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def hybrid_recommend(self, user_id: str, current_location: Optional[Tuple[float, float]] = None,
                        n_recommendations: int = 10) -> List[AdvancedRecommendation]:
        """Generate hybrid recommendations combining all methods"""
        
        # Get recommendations from all methods
        collab_recs = self.collaborative_filtering_recommend(user_id, n_recommendations=15)
        content_recs = self.content_based_recommend(user_id, n_recommendations=15)
        location_recs = self.location_based_recommend(user_id, current_location, n_recommendations=15)
        
        # Combine and weight recommendations
        combined_scores = defaultdict(lambda: {
            'collaborative': 0.0, 'content': 0.0, 'location': 0.0, 'count': 0
        })
        
        all_recommendations = {}
        
        # Process collaborative recommendations
        for rec in collab_recs:
            combined_scores[rec.item_id]['collaborative'] = rec.collaborative_score
            combined_scores[rec.item_id]['count'] += 1
            all_recommendations[rec.item_id] = rec
        
        # Process content-based recommendations
        for rec in content_recs:
            combined_scores[rec.item_id]['content'] = rec.content_score
            combined_scores[rec.item_id]['count'] += 1
            if rec.item_id not in all_recommendations:
                all_recommendations[rec.item_id] = rec
        
        # Process location-based recommendations
        for rec in location_recs:
            combined_scores[rec.item_id]['location'] = rec.location_score
            combined_scores[rec.item_id]['count'] += 1
            if rec.item_id not in all_recommendations:
                all_recommendations[rec.item_id] = rec
        
        # Calculate hybrid scores
        hybrid_recommendations = []
        
        for item_id, scores in combined_scores.items():
            if item_id in all_recommendations:
                # Weighted combination
                weights = {'collaborative': 0.4, 'content': 0.35, 'location': 0.25}
                
                hybrid_score = (
                    scores['collaborative'] * weights['collaborative'] +
                    scores['content'] * weights['content'] +
                    scores['location'] * weights['location']
                )
                
                # Boost score if item appears in multiple recommendation types
                diversity_bonus = (scores['count'] - 1) * 0.1
                hybrid_score += diversity_bonus
                
                # Update recommendation object
                rec = all_recommendations[item_id]
                rec.overall_score = hybrid_score
                rec.collaborative_score = scores['collaborative']
                rec.content_score = scores['content']
                rec.location_score = scores['location']
                rec.confidence = min(hybrid_score, 1.0)
                
                # Enhanced reasons
                method_reasons = []
                if scores['collaborative'] > 0:
                    method_reasons.append("Similar users loved this place")
                if scores['content'] > 0:
                    method_reasons.append("Matches your interests and preferences")
                if scores['location'] > 0:
                    method_reasons.append("Conveniently located near you")
                
                rec.reasons = method_reasons + rec.reasons[:2]  # Keep top 2 original reasons
                
                hybrid_recommendations.append(rec)
        
        # Sort by hybrid score and return top recommendations
        hybrid_recommendations.sort(key=lambda x: x.overall_score, reverse=True)
        return hybrid_recommendations[:n_recommendations]
    
    def _build_user_content_profile(self, user: UserBehavior) -> Dict[str, float]:
        """Build user content profile from rated items"""
        profile = defaultdict(float)
        total_ratings = len(user.rated_places)
        
        if total_ratings == 0:
            return profile
        
        for place_id, rating in user.rated_places.items():
            if place_id in self.content_features:
                features = self.content_features[place_id]
                weight = rating / 5.0  # Normalize rating
                
                for category in features.categories:
                    profile[f"category_{category}"] += weight
                
                for feature in features.features:
                    profile[f"feature_{feature}"] += weight
                
                for time in features.best_times:
                    profile[f"time_{time}"] += weight
        
        # Normalize by number of ratings
        for key in profile:
            profile[key] /= total_ratings
        
        return profile
    
    def _calculate_content_match_score(self, user_profile: Dict[str, float], 
                                     features: ContentFeatures, user: UserBehavior) -> float:
        """Calculate how well content matches user profile"""
        score = 0.0
        
        # Category matching
        for category in features.categories:
            score += user_profile.get(f"category_{category}", 0) * 0.3
        
        # Feature matching
        for feature in features.features:
            score += user_profile.get(f"feature_{feature}", 0) * 0.25
        
        # Time preference matching
        for time in features.best_times:
            score += user_profile.get(f"time_{time}", 0) * 0.15
        
        # Difficulty level matching
        if user.mobility_level == "high" or features.difficulty_level == "easy":
            score += 0.1
        elif user.mobility_level == "moderate" and features.difficulty_level in ["easy", "moderate"]:
            score += 0.05
        
        # Duration preference matching
        if features.place_id.split('_')[0] in user.visit_duration_preferences:
            preferred_duration = user.visit_duration_preferences[features.place_id.split('_')[0]]
            duration_diff = abs(features.duration_minutes - preferred_duration)
            duration_score = max(0, 1 - duration_diff / 120)  # Normalize by 2 hours
            score += duration_score * 0.15
        
        return min(score, 1.0)
    
    def _calculate_location_score(self, location_context: LocationContext, 
                                distance_km: float, user: UserBehavior) -> float:
        """Calculate location-based score"""
        score = 0.0
        
        # Distance penalty (closer is better)
        distance_score = max(0, 1 - distance_km / 10)  # Normalize by 10km
        score += distance_score * 0.4
        
        # Walkability score
        score += location_context.walkability_score * 0.25
        
        # Safety score
        score += location_context.safety_score * 0.2
        
        # District preference
        if location_context.district in user.preferred_districts:
            score += 0.15
        
        return min(score, 1.0)
    
    def _generate_content_reasons(self, user: UserBehavior, features: ContentFeatures) -> List[str]:
        """Generate reasons for content-based recommendations"""
        reasons = []
        
        # Category matches
        user_categories = set()
        for place_id in user.visited_places:
            if place_id in self.content_features:
                user_categories.update(self.content_features[place_id].categories)
        
        matching_categories = set(features.categories) & user_categories
        if matching_categories:
            reasons.append(f"You enjoyed similar {', '.join(matching_categories)} attractions")
        
        # Travel style match
        if user.travel_style == "cultural" and "historic" in features.features:
            reasons.append("Perfect for cultural exploration")
        elif user.travel_style == "foodie" and features.place_id.endswith("restaurant"):
            reasons.append("Great for food enthusiasts")
        
        # Duration match
        if features.duration_minutes <= 60:
            reasons.append("Quick visit option")
        elif features.duration_minutes >= 120:
            reasons.append("Perfect for a deep exploration")
        
        return reasons[:3]
    
    def _find_complementary_items(self, place_id: str) -> List[str]:
        """Find items that complement the given place"""
        if place_id not in self.location_contexts:
            return []
        
        location = self.location_contexts[place_id]
        return location.nearby_attractions[:3]
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c

# Example usage and testing
if __name__ == "__main__":
    engine = AdvancedRecommendationEngine()
    
    # Test collaborative filtering
    print("=== Collaborative Filtering Recommendations ===")
    collab_recs = engine.collaborative_filtering_recommend("user_1", n_recommendations=3)
    for rec in collab_recs:
        print(f"{rec.name}: Score {rec.overall_score:.2f} - {rec.reasons[0] if rec.reasons else 'No reason'}")
    
    # Test content-based filtering
    print("\\n=== Content-Based Recommendations ===")
    content_recs = engine.content_based_recommend("user_1", n_recommendations=3)
    for rec in content_recs:
        print(f"{rec.name}: Score {rec.overall_score:.2f} - {rec.reasons[0] if rec.reasons else 'No reason'}")
    
    # Test hybrid recommendations
    print("\\n=== Hybrid Recommendations ===")
    hybrid_recs = engine.hybrid_recommend("user_1", n_recommendations=5)
    for rec in hybrid_recs:
        print(f"{rec.name}: Score {rec.overall_score:.2f}")
        print(f"  Collaborative: {rec.collaborative_score:.2f}, Content: {rec.content_score:.2f}, Location: {rec.location_score:.2f}")
        print(f"  Reasons: {', '.join(rec.reasons[:2])}")
        print()

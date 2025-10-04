"""
Rule-Based User Preferences System
Implements intelligent user preference management and personalization rules
without using GPT. Uses behavioral analysis, preference inference, and rule engines.
"""

import json
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from enum import Enum

class PreferenceStrength(Enum):
    """Strength levels for user preferences"""
    VERY_WEAK = 0.1
    WEAK = 0.3
    MODERATE = 0.5
    STRONG = 0.7
    VERY_STRONG = 0.9

class PreferenceSource(Enum):
    """Sources of preference information"""
    EXPLICIT = "explicit"  # User explicitly stated
    BEHAVIORAL = "behavioral"  # Inferred from behavior
    DEMOGRAPHIC = "demographic"  # Based on user demographics
    CONTEXTUAL = "contextual"  # Based on context/situation
    SOCIAL = "social"  # From similar users

@dataclass
class UserPreference:
    """Individual user preference with metadata"""
    preference_id: str
    category: str  # 'cuisine', 'activity_type', 'time_preference', etc.
    value: str  # The actual preference value
    strength: float  # 0.0 to 1.0
    confidence: float  # How confident we are about this preference
    source: PreferenceSource
    created_at: datetime
    updated_at: datetime
    evidence_count: int  # Number of supporting evidences
    context_tags: List[str]  # Situational contexts where this applies

@dataclass
class PreferenceRule:
    """Rule for inferring or modifying preferences"""
    rule_id: str
    name: str
    conditions: Dict[str, Any]  # Conditions that must be met
    actions: Dict[str, Any]  # Actions to take when conditions are met
    priority: int  # Higher priority rules are applied first
    enabled: bool
    success_rate: float  # Historical success rate of this rule

@dataclass
class UserContext:
    """Current user context for personalization"""
    time_of_day: str  # 'morning', 'afternoon', 'evening', 'night'
    day_of_week: str  # 'monday', 'tuesday', etc.
    season: str  # 'spring', 'summer', 'autumn', 'winter'
    weather: str  # 'sunny', 'rainy', 'cloudy', 'snowy'
    temperature_c: Optional[int]
    location: Optional[Tuple[float, float]]  # lat, lon
    group_size: int
    budget_range: str  # 'budget', 'moderate', 'luxury'
    available_time_hours: float
    mobility_level: str  # 'low', 'moderate', 'high'
    special_occasions: List[str]  # 'birthday', 'anniversary', etc.
    energy_level: str  # 'low', 'moderate', 'high'

@dataclass
class PersonalizationResult:
    """Result of personalization process"""
    recommendations: List[Dict[str, Any]]
    applied_preferences: List[str]
    applied_rules: List[str]
    confidence_score: float
    explanation: List[str]
    alternative_options: List[Dict[str, Any]]

class RuleBasedPersonalizationEngine:
    """
    Advanced rule-based personalization system that learns and adapts
    user preferences without using machine learning or GPT
    """
    
    def __init__(self):
        self.user_preferences: Dict[str, List[UserPreference]] = defaultdict(list)
        self.preference_rules: List[PreferenceRule] = []
        self.user_behavior_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.global_preference_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Initialize rule engine
        self._initialize_preference_rules()
        self._load_global_statistics()
    
    def _initialize_preference_rules(self):
        """Initialize the rule engine with predefined rules"""
        
        # Time-based preference rules
        self.preference_rules.extend([
            PreferenceRule(
                rule_id="morning_museum_preference",
                name="Morning Museum Preference",
                conditions={
                    "time_of_day": "morning",
                    "visited_museums": {"min_count": 2, "avg_rating": 4.0}
                },
                actions={
                    "boost_category": {"category": "museum", "boost": 0.3},
                    "set_preference": {"category": "best_time", "value": "morning", "strength": 0.7}
                },
                priority=7,
                enabled=True,
                success_rate=0.85
            ),
            
            PreferenceRule(
                rule_id="evening_dining_preference",
                name="Evening Dining Preference",
                conditions={
                    "time_of_day": "evening",
                    "has_dining_history": True
                },
                actions={
                    "boost_category": {"category": "restaurant", "boost": 0.4},
                    "filter_by": {"opening_hours": "evening"}
                },
                priority=8,
                enabled=True,
                success_rate=0.92
            ),
            
            # Weather-based rules
            PreferenceRule(
                rule_id="rainy_day_indoor",
                name="Rainy Day Indoor Preference",
                conditions={
                    "weather": "rainy",
                    "temperature_c": {"max": 15}
                },
                actions={
                    "boost_features": {"indoor": 0.5, "covered": 0.3},
                    "penalize_features": {"outdoor": -0.4}
                },
                priority=9,
                enabled=True,
                success_rate=0.88
            ),
            
            # Social context rules
            PreferenceRule(
                rule_id="group_family_friendly",
                name="Group Family Friendly",
                conditions={
                    "group_size": {"min": 3},
                    "has_children_indicators": True
                },
                actions={
                    "boost_features": {"family_friendly": 0.4, "accessible": 0.3},
                    "filter_duration": {"max_minutes": 90}
                },
                priority=6,
                enabled=True,
                success_rate=0.79
            ),
            
            # Budget-based rules
            PreferenceRule(
                rule_id="budget_conscious",
                name="Budget Conscious Filtering",
                conditions={
                    "budget_range": "budget",
                    "price_sensitive_behavior": True
                },
                actions={
                    "filter_by_cost": {"max_cost": 20},
                    "boost_features": {"free_entry": 0.5},
                    "prioritize_category": {"free_attractions": 0.3}
                },
                priority=8,
                enabled=True,
                success_rate=0.83
            ),
            
            # Behavioral pattern rules
            PreferenceRule(
                rule_id="cultural_enthusiast",
                name="Cultural Enthusiast Pattern",
                conditions={
                    "cultural_visits": {"min_count": 3, "avg_rating": 4.2},
                    "avg_duration_cultural": {"min_minutes": 90}
                },
                actions={
                    "boost_category": {"historical": 0.4, "museum": 0.3, "cultural": 0.4},
                    "set_travel_style": {"value": "cultural", "strength": 0.8}
                },
                priority=5,
                enabled=True,
                success_rate=0.91
            ),
            
            # Seasonal rules
            PreferenceRule(
                rule_id="summer_outdoor_preference",
                name="Summer Outdoor Activities",
                conditions={
                    "season": "summer",
                    "temperature_c": {"min": 20},
                    "weather": {"not": "rainy"}
                },
                actions={
                    "boost_features": {"outdoor": 0.4, "garden": 0.3, "viewpoint": 0.3},
                    "prefer_time": {"afternoon": 0.2, "evening": 0.3}
                },
                priority=6,
                enabled=True,
                success_rate=0.87
            )
        ])
    
    def _load_global_statistics(self):
        """Load global preference statistics for reference"""
        # Simulated global statistics (in real system, this would come from database)
        self.global_preference_stats = {
            "cuisine_preferences": {
                "turkish": 0.75, "mediterranean": 0.45, "international": 0.35,
                "seafood": 0.25, "vegetarian": 0.15, "fast_food": 0.12
            },
            "activity_preferences": {
                "historical": 0.68, "cultural": 0.62, "shopping": 0.45,
                "nature": 0.38, "nightlife": 0.22, "adventure": 0.18
            },
            "time_preferences": {
                "morning": 0.35, "afternoon": 0.55, "evening": 0.45, "night": 0.15
            },
            "duration_preferences": {
                "short": 0.42, "medium": 0.48, "long": 0.25, "full_day": 0.12
            }
        }
    
    def add_user_behavior(self, user_id: str, interaction: Dict[str, Any]):
        """Add user behavior data for preference inference"""
        interaction['timestamp'] = datetime.now()
        self.user_behavior_history[user_id].append(interaction)
        
        # Trigger preference inference
        self._infer_preferences_from_behavior(user_id, interaction)
    
    def _infer_preferences_from_behavior(self, user_id: str, interaction: Dict[str, Any]):
        """Infer user preferences from behavioral data"""
        
        # Analyze the interaction
        place_type = interaction.get('place_type')
        rating = interaction.get('rating', 0)
        duration_minutes = interaction.get('duration_minutes', 0)
        time_of_day = interaction.get('time_of_day')
        context = interaction.get('context', {})
        
        # Infer category preferences
        if place_type and rating >= 4.0:
            self._update_or_create_preference(
                user_id=user_id,
                category="place_type",
                value=place_type,
                strength=min(rating / 5.0, 1.0),
                source=PreferenceSource.BEHAVIORAL,
                context_tags=[time_of_day] if time_of_day else []
            )
        
        # Infer time preferences
        if time_of_day and rating >= 4.0:
            self._update_or_create_preference(
                user_id=user_id,
                category="time_preference",
                value=time_of_day,
                strength=0.6,
                source=PreferenceSource.BEHAVIORAL,
                context_tags=[f"enjoyed_{place_type}"] if place_type else []
            )
        
        # Infer duration preferences
        if duration_minutes > 0 and rating >= 4.0:
            duration_category = self._categorize_duration(duration_minutes)
            self._update_or_create_preference(
                user_id=user_id,
                category="duration_preference",
                value=duration_category,
                strength=0.5,
                source=PreferenceSource.BEHAVIORAL,
                context_tags=[place_type] if place_type else []
            )
    
    def _update_or_create_preference(self, user_id: str, category: str, value: str,
                                   strength: float, source: PreferenceSource,
                                   context_tags: List[str] = None):
        """Update existing preference or create new one"""
        context_tags = context_tags or []
        
        # Look for existing preference
        existing_pref = None
        for pref in self.user_preferences[user_id]:
            if pref.category == category and pref.value == value:
                existing_pref = pref
                break
        
        if existing_pref:
            # Update existing preference
            # Use weighted average for strength
            total_evidence = existing_pref.evidence_count + 1
            existing_pref.strength = (
                (existing_pref.strength * existing_pref.evidence_count + strength) / total_evidence
            )
            existing_pref.evidence_count = total_evidence
            existing_pref.updated_at = datetime.now()
            existing_pref.confidence = min(existing_pref.confidence + 0.1, 1.0)
            
            # Merge context tags
            existing_pref.context_tags = list(set(existing_pref.context_tags + context_tags))
        else:
            # Create new preference
            preference = UserPreference(
                preference_id=f"{user_id}_{category}_{value}_{datetime.now().timestamp()}",
                category=category,
                value=value,
                strength=strength,
                confidence=0.6,
                source=source,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                evidence_count=1,
                context_tags=context_tags
            )
            self.user_preferences[user_id].append(preference)
    
    def _categorize_duration(self, minutes: int) -> str:
        """Categorize visit duration"""
        if minutes <= 30:
            return "very_short"
        elif minutes <= 60:
            return "short"
        elif minutes <= 120:
            return "medium"
        elif minutes <= 240:
            return "long"
        else:
            return "very_long"
    
    def get_user_preferences(self, user_id: str, category: Optional[str] = None) -> List[UserPreference]:
        """Get user preferences, optionally filtered by category"""
        preferences = self.user_preferences.get(user_id, [])
        
        if category:
            preferences = [p for p in preferences if p.category == category]
        
        # Sort by strength and confidence
        preferences.sort(key=lambda p: (p.strength * p.confidence), reverse=True)
        return preferences
    
    def set_explicit_preference(self, user_id: str, category: str, value: str, 
                              strength: float = 0.9):
        """Set explicit user preference"""
        self._update_or_create_preference(
            user_id=user_id,
            category=category,
            value=value,
            strength=strength,
            source=PreferenceSource.EXPLICIT,
            context_tags=["user_stated"]
        )
    
    def personalize_recommendations(self, user_id: str, candidates: List[Dict[str, Any]],
                                  context: UserContext) -> PersonalizationResult:
        """Apply personalization rules to recommendation candidates"""
        
        # Get user preferences
        user_preferences = self.get_user_preferences(user_id)
        
        # Apply preference rules
        applied_rules = []
        rule_modifications = {}
        
        for rule in sorted(self.preference_rules, key=lambda r: r.priority, reverse=True):
            if rule.enabled and self._evaluate_rule_conditions(rule, user_id, context):
                applied_rules.append(rule.rule_id)
                self._apply_rule_actions(rule, rule_modifications, context)
        
        # Score and rank candidates
        scored_candidates = []
        applied_preferences = []
        
        for candidate in candidates:
            score, pref_reasons = self._calculate_personalization_score(
                candidate, user_preferences, rule_modifications, context
            )
            
            candidate_with_score = candidate.copy()
            candidate_with_score['personalization_score'] = score
            candidate_with_score['preference_reasons'] = pref_reasons
            
            scored_candidates.append(candidate_with_score)
            applied_preferences.extend(pref_reasons)
        
        # Sort by personalization score
        scored_candidates.sort(key=lambda c: c['personalization_score'], reverse=True)
        
        # Generate explanation
        explanation = self._generate_personalization_explanation(
            user_preferences, applied_rules, context
        )
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence_score(user_preferences, applied_rules)
        
        # Create alternative options (lower scored but still relevant)
        top_recommendations = scored_candidates[:5]
        alternative_options = scored_candidates[5:10]
        
        return PersonalizationResult(
            recommendations=top_recommendations,
            applied_preferences=list(set(applied_preferences)),
            applied_rules=applied_rules,
            confidence_score=confidence_score,
            explanation=explanation,
            alternative_options=alternative_options
        )
    
    def _evaluate_rule_conditions(self, rule: PreferenceRule, user_id: str, 
                                context: UserContext) -> bool:
        """Evaluate if rule conditions are met"""
        conditions = rule.conditions
        user_history = self.user_behavior_history.get(user_id, [])
        
        # Check time-based conditions
        if "time_of_day" in conditions:
            if context.time_of_day != conditions["time_of_day"]:
                return False
        
        if "day_of_week" in conditions:
            if context.day_of_week != conditions["day_of_week"]:
                return False
        
        if "season" in conditions:
            if context.season != conditions["season"]:
                return False
        
        # Check weather conditions
        if "weather" in conditions:
            if isinstance(conditions["weather"], dict):
                if "not" in conditions["weather"]:
                    if context.weather == conditions["weather"]["not"]:
                        return False
            elif context.weather != conditions["weather"]:
                return False
        
        if "temperature_c" in conditions:
            temp_condition = conditions["temperature_c"]
            if isinstance(temp_condition, dict):
                if "min" in temp_condition and (context.temperature_c or 0) < temp_condition["min"]:
                    return False
                if "max" in temp_condition and (context.temperature_c or 30) > temp_condition["max"]:
                    return False
        
        # Check group size conditions
        if "group_size" in conditions:
            group_condition = conditions["group_size"]
            if isinstance(group_condition, dict):
                if "min" in group_condition and context.group_size < group_condition["min"]:
                    return False
                if "max" in group_condition and context.group_size > group_condition["max"]:
                    return False
        
        # Check behavioral history conditions
        if "visited_museums" in conditions:
            museum_condition = conditions["visited_museums"]
            museum_visits = [h for h in user_history if h.get('place_type') == 'museum']
            
            if "min_count" in museum_condition:
                if len(museum_visits) < museum_condition["min_count"]:
                    return False
            
            if "avg_rating" in museum_condition and museum_visits:
                avg_rating = sum(v.get('rating', 0) for v in museum_visits) / len(museum_visits)
                if avg_rating < museum_condition["avg_rating"]:
                    return False
        
        # Check cultural visits condition
        if "cultural_visits" in conditions:
            cultural_condition = conditions["cultural_visits"]
            cultural_visits = [h for h in user_history 
                             if h.get('place_type') in ['museum', 'historical', 'cultural']]
            
            if "min_count" in cultural_condition:
                if len(cultural_visits) < cultural_condition["min_count"]:
                    return False
            
            if "avg_rating" in cultural_condition and cultural_visits:
                avg_rating = sum(v.get('rating', 0) for v in cultural_visits) / len(cultural_visits)
                if avg_rating < cultural_condition["avg_rating"]:
                    return False
        
        # Check budget conditions
        if "budget_range" in conditions:
            if context.budget_range != conditions["budget_range"]:
                return False
        
        # Check dining history
        if "has_dining_history" in conditions:
            dining_visits = [h for h in user_history if h.get('place_type') == 'restaurant']
            if conditions["has_dining_history"] and not dining_visits:
                return False
        
        return True
    
    def _apply_rule_actions(self, rule: PreferenceRule, modifications: Dict[str, Any], 
                          context: UserContext):
        """Apply rule actions to modify recommendations"""
        actions = rule.actions
        
        # Initialize modification categories if not exist
        if 'category_boosts' not in modifications:
            modifications['category_boosts'] = {}
        if 'feature_boosts' not in modifications:
            modifications['feature_boosts'] = {}
        if 'filters' not in modifications:
            modifications['filters'] = {}
        
        # Apply category boosts
        if "boost_category" in actions:
            boost_action = actions["boost_category"]
            category = boost_action["category"]
            boost = boost_action["boost"]
            modifications['category_boosts'][category] = modifications['category_boosts'].get(category, 0) + boost
        
        # Apply feature boosts
        if "boost_features" in actions:
            for feature, boost in actions["boost_features"].items():
                modifications['feature_boosts'][feature] = modifications['feature_boosts'].get(feature, 0) + boost
        
        if "penalize_features" in actions:
            for feature, penalty in actions["penalize_features"].items():
                modifications['feature_boosts'][feature] = modifications['feature_boosts'].get(feature, 0) + penalty
        
        # Apply filters
        if "filter_by_cost" in actions:
            modifications['filters']['max_cost'] = actions["filter_by_cost"]["max_cost"]
        
        if "filter_duration" in actions:
            modifications['filters']['max_duration'] = actions["filter_duration"]["max_minutes"]
        
        if "filter_by" in actions:
            modifications['filters'].update(actions["filter_by"])
    
    def _calculate_personalization_score(self, candidate: Dict[str, Any], 
                                       preferences: List[UserPreference],
                                       rule_modifications: Dict[str, Any],
                                       context: UserContext) -> Tuple[float, List[str]]:
        """Calculate personalization score for a candidate"""
        base_score = candidate.get('base_score', 0.5)
        personalization_boost = 0.0
        preference_reasons = []
        
        # Apply user preferences
        for pref in preferences:
            boost = 0.0
            
            if pref.category == "place_type" and candidate.get('type') == pref.value:
                boost = pref.strength * pref.confidence * 0.3
                preference_reasons.append(f"Matches your {pref.value} interest")
            
            elif pref.category == "cuisine" and candidate.get('cuisine') == pref.value:
                boost = pref.strength * pref.confidence * 0.25
                preference_reasons.append(f"Your preferred {pref.value} cuisine")
            
            elif pref.category == "time_preference" and context.time_of_day == pref.value:
                boost = pref.strength * pref.confidence * 0.2
                preference_reasons.append(f"Perfect for {pref.value} visits")
            
            elif pref.category == "duration_preference":
                candidate_duration = candidate.get('duration_minutes', 60)
                candidate_duration_cat = self._categorize_duration(candidate_duration)
                if candidate_duration_cat == pref.value:
                    boost = pref.strength * pref.confidence * 0.15
                    preference_reasons.append(f"Fits your preferred visit length")
            
            personalization_boost += boost
        
        # Apply rule modifications
        # Category boosts
        category_boosts = rule_modifications.get('category_boosts', {})
        for category, boost in category_boosts.items():
            if candidate.get('type') == category or candidate.get('category') == category:
                personalization_boost += boost
                preference_reasons.append(f"Recommended for current conditions")
        
        # Feature boosts
        feature_boosts = rule_modifications.get('feature_boosts', {})
        candidate_features = candidate.get('features', [])
        for feature, boost in feature_boosts.items():
            if feature in candidate_features:
                personalization_boost += boost
                if boost > 0:
                    preference_reasons.append(f"Has {feature} feature you'll appreciate")
        
        # Apply filters (negative score if doesn't meet criteria)
        filters = rule_modifications.get('filters', {})
        
        if 'max_cost' in filters:
            candidate_cost = candidate.get('cost', 0)
            if candidate_cost > filters['max_cost']:
                personalization_boost -= 0.5  # Heavy penalty for exceeding budget
        
        if 'max_duration' in filters:
            candidate_duration = candidate.get('duration_minutes', 60)
            if candidate_duration > filters['max_duration']:
                personalization_boost -= 0.3  # Penalty for being too long
        
        # Final score calculation
        final_score = base_score + personalization_boost
        final_score = max(0.0, min(1.0, final_score))  # Clamp between 0 and 1
        
        return final_score, preference_reasons[:3]  # Return top 3 reasons
    
    def _generate_personalization_explanation(self, preferences: List[UserPreference],
                                            applied_rules: List[str],
                                            context: UserContext) -> List[str]:
        """Generate human-readable explanation of personalization"""
        explanations = []
        
        # Explain top preferences
        top_preferences = preferences[:3]
        for pref in top_preferences:
            if pref.strength > 0.6:
                explanations.append(
                    f"Based on your {pref.source.value} preference for {pref.value}"
                )
        
        # Explain applied rules
        rule_explanations = {
            "morning_museum_preference": "Prioritized museums for morning visits",
            "evening_dining_preference": "Highlighted dining options for evening",
            "rainy_day_indoor": "Focused on indoor activities due to weather",
            "group_family_friendly": "Selected family-friendly options for your group",
            "budget_conscious": "Filtered by budget-friendly options",
            "cultural_enthusiast": "Emphasized cultural attractions based on your interests",
            "summer_outdoor_preference": "Prioritized outdoor activities for the season"
        }
        
        for rule_id in applied_rules:
            if rule_id in rule_explanations:
                explanations.append(rule_explanations[rule_id])
        
        # Context-based explanations
        if context.weather == "rainy":
            explanations.append("Adjusted recommendations for rainy weather")
        
        if context.group_size > 2:
            explanations.append(f"Considered options suitable for groups of {context.group_size}")
        
        return explanations[:5]  # Return top 5 explanations
    
    def _calculate_confidence_score(self, preferences: List[UserPreference],
                                  applied_rules: List[str]) -> float:
        """Calculate overall confidence in personalization"""
        if not preferences and not applied_rules:
            return 0.3  # Low confidence with no data
        
        # Preference confidence
        pref_confidence = 0.0
        if preferences:
            total_evidence = sum(p.evidence_count for p in preferences)
            weighted_confidence = sum(p.confidence * p.evidence_count for p in preferences)
            pref_confidence = weighted_confidence / total_evidence if total_evidence > 0 else 0.0
        
        # Rule confidence
        rule_confidence = 0.0
        if applied_rules:
            rule_success_rates = []
            for rule in self.preference_rules:
                if rule.rule_id in applied_rules:
                    rule_success_rates.append(rule.success_rate)
            
            rule_confidence = sum(rule_success_rates) / len(rule_success_rates) if rule_success_rates else 0.0
        
        # Combined confidence
        if preferences and applied_rules:
            combined_confidence = (pref_confidence * 0.6 + rule_confidence * 0.4)
        elif preferences:
            combined_confidence = pref_confidence * 0.8
        elif applied_rules:
            combined_confidence = rule_confidence * 0.7
        else:
            combined_confidence = 0.3
        
        return min(combined_confidence, 1.0)
    
    def get_user_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of user's preference profile"""
        preferences = self.get_user_preferences(user_id)
        behavior_history = self.user_behavior_history.get(user_id, [])
        
        # Group preferences by category
        pref_by_category = defaultdict(list)
        for pref in preferences:
            pref_by_category[pref.category].append(pref)
        
        # Calculate statistics
        total_interactions = len(behavior_history)
        avg_rating = 0.0
        if behavior_history:
            ratings = [h.get('rating', 0) for h in behavior_history if h.get('rating', 0) > 0]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        
        # Most visited types
        place_types = [h.get('place_type') for h in behavior_history if h.get('place_type')]
        most_visited_types = Counter(place_types).most_common(3)
        
        return {
            'user_id': user_id,
            'total_preferences': len(preferences),
            'preferences_by_category': {
                cat: [(p.value, p.strength, p.confidence) for p in prefs[:3]]
                for cat, prefs in pref_by_category.items()
            },
            'total_interactions': total_interactions,
            'average_rating': round(avg_rating, 2),
            'most_visited_types': most_visited_types,
            'profile_maturity': min(total_interactions / 10.0, 1.0),  # 0-1 scale
            'preference_strength': sum(p.strength * p.confidence for p in preferences) / len(preferences) if preferences else 0.0
        }

# Example usage and testing
if __name__ == "__main__":
    engine = RuleBasedPersonalizationEngine()
    
    # Simulate user behavior
    user_id = "test_user_1"
    
    # Add some behavioral data
    behaviors = [
        {"place_type": "museum", "rating": 4.5, "duration_minutes": 90, "time_of_day": "morning"},
        {"place_type": "historical", "rating": 4.8, "duration_minutes": 120, "time_of_day": "afternoon"},
        {"place_type": "restaurant", "rating": 4.2, "duration_minutes": 75, "time_of_day": "evening"},
        {"place_type": "museum", "rating": 4.0, "duration_minutes": 85, "time_of_day": "morning"}
    ]
    
    for behavior in behaviors:
        engine.add_user_behavior(user_id, behavior)
    
    # Set explicit preference
    engine.set_explicit_preference(user_id, "cuisine", "turkish", 0.9)
    
    # Test personalization
    context = UserContext(
        time_of_day="morning",
        day_of_week="saturday",
        season="spring",
        weather="sunny",
        temperature_c=18,
        location=None,
        group_size=2,
        budget_range="moderate",
        available_time_hours=4.0,
        mobility_level="high",
        special_occasions=[],
        energy_level="high"
    )
    
    # Sample candidates
    candidates = [
        {"id": "hagia_sophia", "type": "museum", "base_score": 0.8, "features": ["indoor", "historic"], "duration_minutes": 90, "cost": 15},
        {"id": "galata_tower", "type": "viewpoint", "base_score": 0.7, "features": ["outdoor", "panoramic"], "duration_minutes": 45, "cost": 10},
        {"id": "turkish_restaurant", "type": "restaurant", "cuisine": "turkish", "base_score": 0.6, "features": ["indoor"], "duration_minutes": 60, "cost": 25}
    ]
    
    result = engine.personalize_recommendations(user_id, candidates, context)
    
    print("=== Personalization Results ===")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"Applied Rules: {result.applied_rules}")
    print(f"Applied Preferences: {result.applied_preferences}")
    print("\\nTop Recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i}. {rec['id']}: Score {rec['personalization_score']:.2f}")
        print(f"   Reasons: {', '.join(rec['preference_reasons'])}")
    
    print("\\nExplanation:")
    for explanation in result.explanation:
        print(f"- {explanation}")
    
    # Get user profile summary
    profile = engine.get_user_profile_summary(user_id)
    print("\\n=== User Profile Summary ===")
    print(f"Total Preferences: {profile['total_preferences']}")
    print(f"Profile Maturity: {profile['profile_maturity']:.2f}")
    print(f"Average Rating: {profile['average_rating']}")
    print(f"Most Visited: {profile['most_visited_types']}")

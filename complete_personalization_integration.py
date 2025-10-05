# complete_personalization_integration.py - Full Personalization Pipeline Integration

import sys
import os
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any

# Import all personalization components
from user_profiling_system import UserProfilingSystem
from preference_learning_engine import PreferenceLearningEngine
from recommendation_enhancement_system import RecommendationEnhancementEngine
from personalization_ab_testing import PersonalizationABTesting, ABTestConfig
from hybrid_integration_system import HybridIntegrationSystem

class CompletePersonalizationSystem:
    """Integrated personalization system combining all components"""
    
    def __init__(self, db_path: str = 'ai_istanbul_users.db'):
        self.db_path = db_path
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize all components
        self.user_profiling = UserProfilingSystem(db_path=db_path)
        self.preference_learning = PreferenceLearningEngine(db_path=db_path)
        self.recommendation_engine = RecommendationEnhancementEngine(db_path=db_path)
        self.ab_testing = PersonalizationABTesting(db_path=db_path)
        self.hybrid_system = HybridIntegrationSystem(db_path=db_path)
        
        self.logger.info("Complete personalization system initialized")
        
    def initialize_system(self):
        """Initialize all system components"""
        try:
            # Initialize attraction embeddings
            self.recommendation_engine.initialize_attraction_embeddings()
            
            # Build embedding models
            self.recommendation_engine.embedding_system.build_interaction_model()
            
            # Initialize hybrid system
            hybrid_config = {
                "control_weight": 0.3,
                "treatment_weight": 0.4,
                "hybrid_weight": 0.3,
                "fallback_threshold": 0.5,
                "performance_threshold": 0.7
            }
            self.hybrid_system.initialize_system(hybrid_config)
            
            self.logger.info("System initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            return False
            
    def create_user_profile(self, user_id: str, initial_data: Dict = None) -> bool:
        """Create and initialize user profile"""
        try:
            # Create basic profile
            profile_created = self.user_profiling.create_user_profile(
                user_id=user_id,
                initial_preferences=initial_data.get("preferences", {}) if initial_data else {},
                demographic_info=initial_data.get("demographics", {}) if initial_data else {}
            )
            
            if not profile_created:
                return False
                
            # Initialize preference learning for user
            self.preference_learning.initialize_user_preferences(user_id)
            
            self.logger.info(f"User profile created for {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating user profile: {str(e)}")
            return False
            
    def get_personalized_recommendations(self, user_id: str, context: Dict = None,
                                       use_ab_testing: bool = True) -> Dict:
        """Get personalized recommendations with optional A/B testing"""
        try:
            # Check if user profile exists
            profile = self.user_profiling.get_user_profile(user_id)
            if not profile:
                # Create basic profile if doesn't exist
                self.create_user_profile(user_id)
                
            # Update user preferences based on recent interactions
            self.preference_learning.update_user_preferences(user_id)
            
            if use_ab_testing:
                # Get recommendations through A/B testing
                test_config = ABTestConfig(
                    test_name="Personalized Recommendations Test",
                    start_date=datetime.now(),
                    end_date=datetime.now() + timedelta(days=30),
                    traffic_allocation={"control": 0.3, "treatment": 0.7},
                    success_metrics=["click_through_rate", "engagement_score"],
                    minimum_sample_size=50,
                    confidence_level=0.95
                )
                
                # Create or get existing test
                test_id = self.ab_testing.create_ab_test(test_config)
                
                # Get recommendations based on variant
                recommendations = self.ab_testing.get_recommendations_by_variant(
                    user_id, test_id, self.recommendation_engine
                )
                
            else:
                # Get direct personalized recommendations
                recommendations = self.recommendation_engine.generate_enhanced_recommendations(
                    user_id, context
                )
                
            # Add personalization metadata
            recommendations["personalization_metadata"] = {
                "user_profile_exists": profile is not None,
                "preference_learning_active": True,
                "context_applied": context is not None,
                "ab_testing": use_ab_testing,
                "system_version": "2.0"
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting personalized recommendations: {str(e)}")
            return {"error": str(e)}
            
    def track_user_interaction(self, user_id: str, interaction_data: Dict) -> bool:
        """Track user interaction across all systems"""
        try:
            # Track in user profiling system
            self.user_profiling.track_user_interaction(
                user_id=user_id,
                interaction_type=interaction_data.get("type", "view"),
                attraction_id=interaction_data.get("attraction_id"),
                rating=interaction_data.get("rating"),
                metadata=interaction_data.get("metadata", {})
            )
            
            # Update preference learning
            if interaction_data.get("attraction_id"):
                self.preference_learning.learn_from_interaction(
                    user_id=user_id,
                    attraction_id=interaction_data["attraction_id"],
                    interaction_type=interaction_data.get("type", "view"),
                    rating=interaction_data.get("rating", 3.0)
                )
                
            # Track in A/B testing if test_id provided
            if interaction_data.get("test_id"):
                self.ab_testing.track_user_interaction(
                    user_id=user_id,
                    test_id=interaction_data["test_id"],
                    interaction_type=interaction_data.get("type", "view"),
                    attraction_id=interaction_data.get("attraction_id"),
                    satisfaction_rating=interaction_data.get("rating")
                )
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error tracking interaction: {str(e)}")
            return False
            
    def generate_system_report(self) -> Dict:
        """Generate comprehensive system performance report"""
        try:
            report = {
                "report_date": datetime.now().isoformat(),
                "system_status": "operational",
                "components": {},
                "performance_metrics": {},
                "recommendations": []
            }
            
            # User profiling statistics
            profiling_stats = self.user_profiling.get_system_statistics()
            report["components"]["user_profiling"] = profiling_stats
            
            # Preference learning metrics  
            learning_metrics = self.preference_learning.get_learning_metrics()
            report["components"]["preference_learning"] = learning_metrics
            
            # A/B testing results (if any active tests)
            # Note: Would need test_id in real implementation
            report["components"]["ab_testing"] = {
                "status": "monitoring",
                "active_tests": 0  # Placeholder
            }
            
            # Overall performance metrics
            report["performance_metrics"] = {
                "total_users": profiling_stats.get("total_users", 0),
                "active_profiles": profiling_stats.get("active_profiles", 0),
                "recommendation_requests_today": 0,  # Would track in real system
                "average_satisfaction": learning_metrics.get("average_rating", 0)
            }
            
            # System recommendations
            if profiling_stats.get("total_users", 0) > 100:
                report["recommendations"].append("Consider scaling recommendation engine")
                
            if learning_metrics.get("average_rating", 0) < 3.5:
                report["recommendations"].append("Review recommendation algorithms")
                
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating system report: {str(e)}")
            return {"error": str(e)}
            
    def optimize_system_performance(self) -> Dict:
        """Optimize system performance based on usage patterns"""
        try:
            optimization_results = {
                "timestamp": datetime.now().isoformat(),
                "optimizations_applied": [],
                "performance_improvements": {}
            }
            
            # Retrain embedding models if enough new data
            stats = self.user_profiling.get_system_statistics()
            if stats.get("interactions_since_last_training", 0) > 1000:
                train_result = self.recommendation_engine.embedding_system.train_embeddings()
                if train_result:
                    optimization_results["optimizations_applied"].append("embedding_retraining")
                    
            # Update preference learning models
            learning_update = self.preference_learning.retrain_models()
            if learning_update:
                optimization_results["optimizations_applied"].append("preference_model_update")
                
            # Optimize user profiles (remove inactive users, compress data)
            profile_optimization = self.user_profiling.optimize_profiles()
            if profile_optimization:
                optimization_results["optimizations_applied"].append("profile_optimization")
                
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing system: {str(e)}")
            return {"error": str(e)}

# Example usage and comprehensive testing
def run_personalization_demo():
    """Run comprehensive personalization system demo"""
    print("=== Istanbul AI Tourism - Complete Personalization System Demo ===")
    
    # Initialize system
    personalization_system = CompletePersonalizationSystem()
    
    print("\n1. Initializing system components...")
    init_success = personalization_system.initialize_system()
    print(f"System initialization: {'SUCCESS' if init_success else 'FAILED'}")
    
    # Create test users
    test_users = [
        {
            "user_id": "tourist_001",
            "preferences": {"categories": ["historical", "cultural"], "interests": ["byzantine", "ottoman"]},
            "demographics": {"age": 35, "country": "USA", "travel_style": "cultural"}
        },
        {
            "user_id": "tourist_002", 
            "preferences": {"categories": ["shopping", "food"], "interests": ["traditional", "local"]},
            "demographics": {"age": 28, "country": "Germany", "travel_style": "experiential"}
        },
        {
            "user_id": "tourist_003",
            "preferences": {"categories": ["landmark", "views"], "interests": ["photography", "panoramic"]},
            "demographics": {"age": 42, "country": "Japan", "travel_style": "sightseeing"}
        }
    ]
    
    print("\n2. Creating user profiles...")
    for user_data in test_users:
        success = personalization_system.create_user_profile(
            user_data["user_id"], 
            {"preferences": user_data["preferences"], "demographics": user_data["demographics"]}
        )
        print(f"Profile for {user_data['user_id']}: {'SUCCESS' if success else 'FAILED'}")
        
    print("\n3. Getting personalized recommendations...")
    for user_data in test_users:
        user_id = user_data["user_id"]
        
        # Test context-aware recommendations
        context = {
            "time_of_day": "morning",
            "weather": "clear", 
            "group_size": 2,
            "budget": "medium"
        }
        
        recommendations = personalization_system.get_personalized_recommendations(
            user_id, context, use_ab_testing=True
        )
        
        print(f"\nRecommendations for {user_id}:")
        if "error" not in recommendations:
            if "recommendations" in recommendations:
                for i, rec in enumerate(recommendations["recommendations"][:3], 1):
                    print(f"  {i}. {rec.get('attraction_id', 'N/A')} (Score: {rec.get('score', 0):.3f})")
            print(f"  Variant: {recommendations.get('variant', 'N/A')}")
            print(f"  Personalization: {recommendations.get('personalization_metadata', {}).get('system_version', 'N/A')}")
        else:
            print(f"  Error: {recommendations['error']}")
            
    print("\n4. Simulating user interactions...")
    interactions = [
        {"user_id": "tourist_001", "type": "click", "attraction_id": "hagia_sophia", "rating": 4.5},
        {"user_id": "tourist_001", "type": "booking", "attraction_id": "topkapi_palace", "rating": 4.8},
        {"user_id": "tourist_002", "type": "click", "attraction_id": "grand_bazaar", "rating": 4.2},
        {"user_id": "tourist_003", "type": "click", "attraction_id": "galata_tower", "rating": 4.7},
        {"user_id": "tourist_003", "type": "rating", "attraction_id": "galata_tower", "rating": 5.0}
    ]
    
    for interaction in interactions:
        success = personalization_system.track_user_interaction(
            interaction["user_id"], interaction
        )
        print(f"Tracked interaction for {interaction['user_id']}: {'SUCCESS' if success else 'FAILED'}")
        
    print("\n5. Generating system performance report...")
    report = personalization_system.generate_system_report()
    if "error" not in report:
        print(f"Total Users: {report['performance_metrics']['total_users']}")
        print(f"Active Profiles: {report['performance_metrics']['active_profiles']}")
        print(f"Average Satisfaction: {report['performance_metrics']['average_satisfaction']:.2f}")
        print(f"System Recommendations: {len(report['recommendations'])}")
    else:
        print(f"Report Error: {report['error']}")
        
    print("\n6. System optimization...")
    optimization = personalization_system.optimize_system_performance()
    if "error" not in optimization:
        print(f"Optimizations Applied: {len(optimization['optimizations_applied'])}")
        for opt in optimization['optimizations_applied']:
            print(f"  - {opt}")
    else:
        print(f"Optimization Error: {optimization['error']}")
        
    print("\n=== Personalization System Demo Complete ===")

if __name__ == "__main__":
    run_personalization_demo()

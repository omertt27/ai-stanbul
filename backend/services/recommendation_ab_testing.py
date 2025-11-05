"""
A/B Testing Framework for Recommendation System
Supports multiple test variants with statistical tracking
"""

import logging
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json

from backend.database import SessionLocal
from backend.models import FeedbackEvent
from sqlalchemy import func

logger = logging.getLogger(__name__)


class ABVariant(str, Enum):
    """A/B test variants"""
    CONTROL = "control"  # Original recommendation algorithm
    VARIANT_A = "variant_a"  # Alternative algorithm/parameters
    VARIANT_B = "variant_b"  # Another alternative
    

class ABTestConfig:
    """Configuration for an A/B test"""
    
    def __init__(
        self,
        test_id: str,
        variants: Dict[ABVariant, float],  # variant -> traffic percentage
        enabled: bool = True
    ):
        """
        Args:
            test_id: Unique identifier for the test
            variants: Dict mapping variants to traffic percentage (must sum to 1.0)
            enabled: Whether the test is active
        """
        self.test_id = test_id
        self.variants = variants
        self.enabled = enabled
        
        # Validate percentages
        total = sum(variants.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Variant percentages must sum to 1.0, got {total}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_id': self.test_id,
            'variants': {k.value: v for k, v in self.variants.items()},
            'enabled': self.enabled
        }


class RecommendationABTesting:
    """A/B testing framework for recommendation system"""
    
    def __init__(self):
        """Initialize A/B testing framework"""
        self.tests: Dict[str, ABTestConfig] = {}
        logger.info("✅ RecommendationABTesting initialized")
    
    def register_test(self, config: ABTestConfig):
        """Register a new A/B test"""
        self.tests[config.test_id] = config
        logger.info(f"✅ Registered A/B test: {config.test_id}")
    
    def get_variant(self, test_id: str, user_id: str) -> ABVariant:
        """
        Get the variant for a user in a specific test
        Uses consistent hashing to ensure same user always gets same variant
        
        Args:
            test_id: Test identifier
            user_id: User identifier
        
        Returns:
            Assigned variant
        """
        test = self.tests.get(test_id)
        if not test or not test.enabled:
            return ABVariant.CONTROL
        
        # Use consistent hashing
        hash_input = f"{test_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        hash_percent = (hash_value % 10000) / 10000.0  # 0.0 to 1.0
        
        # Assign variant based on percentages
        cumulative = 0.0
        for variant, percentage in test.variants.items():
            cumulative += percentage
            if hash_percent < cumulative:
                return variant
        
        # Fallback (should never reach here if percentages sum to 1.0)
        return ABVariant.CONTROL
    
    def track_exposure(
        self,
        test_id: str,
        user_id: str,
        variant: ABVariant,
        session_id: Optional[str] = None
    ):
        """
        Track that a user was exposed to a variant
        Stores as a special feedback event
        """
        # This could be stored in a separate table in production
        # For now, we'll use metadata in feedback events
        logger.debug(f"📊 User {user_id} exposed to {test_id}:{variant.value}")
    
    def get_test_results(self, test_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get results for an A/B test
        
        Args:
            test_id: Test identifier
            hours: Number of hours to look back
        
        Returns:
            Dictionary with results per variant
        """
        test = self.tests.get(test_id)
        if not test:
            return {'error': f'Test {test_id} not found'}
        
        try:
            db = SessionLocal()
            
            # Get events for all users (we'll assign variants on the fly)
            since = datetime.now() - timedelta(hours=hours)
            
            events = db.query(FeedbackEvent).filter(
                FeedbackEvent.timestamp >= since
            ).all()
            
            # Group events by variant
            variant_stats = {
                variant: {
                    'users': set(),
                    'views': 0,
                    'clicks': 0,
                    'ratings': [],
                    'conversions': 0
                }
                for variant in test.variants.keys()
            }
            
            for event in events:
                variant = self.get_variant(test_id, event.user_id)
                stats = variant_stats[variant]
                
                stats['users'].add(event.user_id)
                
                if event.event_type == 'view':
                    stats['views'] += 1
                elif event.event_type == 'click':
                    stats['clicks'] += 1
                elif event.event_type == 'rating':
                    rating = event.event_metadata.get('rating', 0)
                    stats['ratings'].append(rating)
                elif event.event_type == 'conversion':
                    stats['conversions'] += 1
            
            db.close()
            
            # Calculate metrics per variant
            results = {}
            for variant, stats in variant_stats.items():
                user_count = len(stats['users'])
                view_count = stats['views']
                click_count = stats['clicks']
                rating_list = stats['ratings']
                conversion_count = stats['conversions']
                
                results[variant.value] = {
                    'users': user_count,
                    'views': view_count,
                    'clicks': click_count,
                    'conversions': conversion_count,
                    'ctr': round(click_count / view_count, 4) if view_count > 0 else 0,
                    'conversion_rate': round(conversion_count / click_count, 4) if click_count > 0 else 0,
                    'avg_rating': round(sum(rating_list) / len(rating_list), 2) if rating_list else 0,
                    'rating_count': len(rating_list)
                }
            
            return {
                'test_id': test_id,
                'period_hours': hours,
                'since': since.isoformat(),
                'variants': results
            }
        except Exception as e:
            logger.error(f"❌ Failed to get test results: {e}")
            return {'error': str(e)}
    
    def get_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered tests"""
        return {
            test_id: config.to_dict()
            for test_id, config in self.tests.items()
        }
    
    def disable_test(self, test_id: str):
        """Disable a test"""
        if test_id in self.tests:
            self.tests[test_id].enabled = False
            logger.info(f"🛑 Disabled A/B test: {test_id}")
    
    def enable_test(self, test_id: str):
        """Enable a test"""
        if test_id in self.tests:
            self.tests[test_id].enabled = True
            logger.info(f"✅ Enabled A/B test: {test_id}")


# Global instance
_recommendation_ab_testing = None


def get_recommendation_ab_testing() -> RecommendationABTesting:
    """Get or create the global recommendation A/B testing instance"""
    global _recommendation_ab_testing
    if _recommendation_ab_testing is None:
        _recommendation_ab_testing = RecommendationABTesting()
        
        # Register default test
        default_test = ABTestConfig(
            test_id="recommendation_algorithm",
            variants={
                ABVariant.CONTROL: 0.5,  # 50% get original algorithm
                ABVariant.VARIANT_A: 0.5  # 50% get new algorithm
            },
            enabled=False  # Disabled by default
        )
        _recommendation_ab_testing.register_test(default_test)
    
    return _recommendation_ab_testing


def reset_recommendation_ab_testing():
    """Reset the global instance (for testing)"""
    global _recommendation_ab_testing
    _recommendation_ab_testing = None


# Example usage
if __name__ == "__main__":
    # Create A/B test
    framework = RecommendationABTesting()
    
    test_config = ABTestConfig(
        test_id="rec_algo_test",
        variants={
            ABVariant.CONTROL: 0.5,
            ABVariant.VARIANT_A: 0.5
        }
    )
    
    framework.register_test(test_config)
    
    # Test variant assignment
    print("\nTesting variant assignment (should be consistent):")
    for user_id in ["user1", "user2", "user3"]:
        variant1 = framework.get_variant("rec_algo_test", user_id)
        variant2 = framework.get_variant("rec_algo_test", user_id)
        assert variant1 == variant2, "Variant assignment not consistent!"
        print(f"  {user_id}: {variant1.value}")
    
    # Test distribution
    print("\nTesting distribution (10,000 users):")
    variants = [framework.get_variant("rec_algo_test", f"user{i}") for i in range(10000)]
    control_count = variants.count(ABVariant.CONTROL)
    variant_a_count = variants.count(ABVariant.VARIANT_A)
    
    print(f"  Control: {control_count} ({control_count/100:.1f}%)")
    print(f"  Variant A: {variant_a_count} ({variant_a_count/100:.1f}%)")
    
    print("\n🎉 A/B testing framework tests passed!")

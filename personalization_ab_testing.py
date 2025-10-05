# personalization_ab_testing.py - A/B Testing for Personalized Recommendations

import numpy as np
import pandas as pd
import sqlite3
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import uuid
import random

@dataclass
class ABTestConfig:
    """Configuration for A/B testing"""
    test_name: str
    start_date: datetime
    end_date: datetime
    traffic_allocation: Dict[str, float]  # {"control": 0.5, "treatment": 0.5}
    success_metrics: List[str]
    minimum_sample_size: int
    confidence_level: float

class PersonalizationABTesting:
    """A/B Testing system for personalized vs generic recommendations"""
    
    def __init__(self, db_path: str = 'ai_istanbul_users.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize database for A/B test tracking
        self._initialize_ab_database()
        
        # Default Istanbul attractions for generic recommendations
        self.generic_recommendations = [
            "hagia_sophia", "blue_mosque", "grand_bazaar", 
            "topkapi_palace", "galata_tower"
        ]
        
    def _initialize_ab_database(self):
        """Initialize A/B testing database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # A/B test configurations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_configs (
                    test_id TEXT PRIMARY KEY,
                    test_name TEXT,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    traffic_allocation TEXT,
                    success_metrics TEXT,
                    minimum_sample_size INTEGER,
                    confidence_level REAL,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # User test assignments
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_test_assignments (
                    user_id TEXT,
                    test_id TEXT,
                    variant TEXT,
                    assignment_date TIMESTAMP,
                    PRIMARY KEY (user_id, test_id)
                )
            ''')
            
            # Interaction events for A/B testing
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ab_test_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    test_id TEXT,
                    variant TEXT,
                    event_type TEXT,
                    event_data TEXT,
                    timestamp TIMESTAMP
                )
            ''')
            
            # Recommendation performance metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recommendation_metrics (
                    metric_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    test_id TEXT,
                    variant TEXT,
                    recommendations TEXT,
                    click_through_rate REAL,
                    engagement_score REAL,
                    satisfaction_rating REAL,
                    conversion_rate REAL,
                    timestamp TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("A/B testing database initialized")
            
        except Exception as e:
            self.logger.error(f"A/B database initialization error: {str(e)}")
            
    def create_ab_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test configuration"""
        try:
            test_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ab_test_configs 
                (test_id, test_name, start_date, end_date, traffic_allocation, 
                 success_metrics, minimum_sample_size, confidence_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_id,
                config.test_name,
                config.start_date.isoformat(),
                config.end_date.isoformat(),
                json.dumps(config.traffic_allocation),
                json.dumps(config.success_metrics),
                config.minimum_sample_size,
                config.confidence_level
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"A/B test created: {config.test_name} (ID: {test_id})")
            return test_id
            
        except Exception as e:
            self.logger.error(f"Error creating A/B test: {str(e)}")
            return ""
            
    def assign_user_to_variant(self, user_id: str, test_id: str) -> str:
        """Assign user to A/B test variant using consistent hashing"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already assigned
            cursor.execute('''
                SELECT variant FROM user_test_assignments 
                WHERE user_id = ? AND test_id = ?
            ''', (user_id, test_id))
            
            existing_assignment = cursor.fetchone()
            if existing_assignment:
                return existing_assignment[0]
                
            # Get test configuration
            cursor.execute('''
                SELECT traffic_allocation FROM ab_test_configs 
                WHERE test_id = ?
            ''', (test_id,))
            
            config_row = cursor.fetchone()
            if not config_row:
                return "control"  # Default fallback
                
            traffic_allocation = json.loads(config_row[0])
            
            # Use consistent hashing for assignment
            hash_input = f"{user_id}_{test_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            probability = (hash_value % 10000) / 10000.0
            
            # Assign based on traffic allocation
            cumulative_probability = 0
            for variant, allocation in traffic_allocation.items():
                cumulative_probability += allocation
                if probability <= cumulative_probability:
                    assigned_variant = variant
                    break
            else:
                assigned_variant = "control"
                
            # Store assignment
            cursor.execute('''
                INSERT INTO user_test_assignments 
                (user_id, test_id, variant, assignment_date)
                VALUES (?, ?, ?, ?)
            ''', (user_id, test_id, assigned_variant, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            return assigned_variant
            
        except Exception as e:
            self.logger.error(f"Error assigning user to variant: {str(e)}")
            return "control"
            
    def get_recommendations_by_variant(self, user_id: str, test_id: str, 
                                     personalized_engine=None) -> Dict:
        """Get recommendations based on A/B test variant"""
        try:
            variant = self.assign_user_to_variant(user_id, test_id)
            
            recommendations = {
                "user_id": user_id,
                "test_id": test_id,
                "variant": variant,
                "timestamp": datetime.now().isoformat(),
                "recommendations": []
            }
            
            if variant == "control":
                # Generic recommendations
                recommendations["recommendations"] = [
                    {
                        "attraction_id": attr_id,
                        "score": random.uniform(0.5, 1.0),  # Simulated generic score
                        "reason": "popular_attraction"
                    }
                    for attr_id in self.generic_recommendations
                ]
                
            elif variant == "treatment" and personalized_engine:
                # Personalized recommendations
                personalized_recs = personalized_engine.generate_enhanced_recommendations(user_id)
                
                if "hybrid_score" in personalized_recs:
                    recommendations["recommendations"] = [
                        {
                            "attraction_id": rec["attraction_id"],
                            "score": rec["hybrid_score"],
                            "reason": "personalized_match"
                        }
                        for rec in personalized_recs["hybrid_score"][:5]
                    ]
                else:
                    # Fallback to generic if personalization fails
                    recommendations["recommendations"] = [
                        {
                            "attraction_id": attr_id,
                            "score": random.uniform(0.5, 1.0),
                            "reason": "fallback_generic"
                        }
                        for attr_id in self.generic_recommendations
                    ]
                    
            # Log recommendation event
            self._log_ab_event(user_id, test_id, variant, "recommendation_served", recommendations)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting variant recommendations: {str(e)}")
            return {"error": str(e)}
            
    def _log_ab_event(self, user_id: str, test_id: str, variant: str, 
                     event_type: str, event_data: Dict):
        """Log A/B test event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            event_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO ab_test_events 
                (event_id, user_id, test_id, variant, event_type, event_data, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id, user_id, test_id, variant, event_type,
                json.dumps(event_data), datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging A/B event: {str(e)}")
            
    def track_user_interaction(self, user_id: str, test_id: str, 
                              interaction_type: str, attraction_id: str = None,
                              satisfaction_rating: float = None):
        """Track user interaction for A/B test analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user's variant
            cursor.execute('''
                SELECT variant FROM user_test_assignments 
                WHERE user_id = ? AND test_id = ?
            ''', (user_id, test_id))
            
            variant_row = cursor.fetchone()
            if not variant_row:
                return False
                
            variant = variant_row[0]
            
            # Log interaction event
            interaction_data = {
                "interaction_type": interaction_type,
                "attraction_id": attraction_id,
                "satisfaction_rating": satisfaction_rating
            }
            
            self._log_ab_event(user_id, test_id, variant, "user_interaction", interaction_data)
            
            # Update metrics if this is a measurable interaction
            if interaction_type in ["click", "booking", "rating"]:
                self._update_recommendation_metrics(user_id, test_id, variant, interaction_data)
                
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error tracking interaction: {str(e)}")
            return False
            
    def _update_recommendation_metrics(self, user_id: str, test_id: str, 
                                     variant: str, interaction_data: Dict):
        """Update recommendation performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get or create metrics record
            cursor.execute('''
                SELECT metric_id, click_through_rate, engagement_score, 
                       satisfaction_rating, conversion_rate 
                FROM recommendation_metrics 
                WHERE user_id = ? AND test_id = ? AND variant = ?
            ''', (user_id, test_id, variant))
            
            existing_metrics = cursor.fetchone()
            
            if existing_metrics:
                metric_id = existing_metrics[0]
                current_ctr = existing_metrics[1] or 0
                current_engagement = existing_metrics[2] or 0
                current_satisfaction = existing_metrics[3] or 0
                current_conversion = existing_metrics[4] or 0
            else:
                metric_id = str(uuid.uuid4())
                current_ctr = current_engagement = current_satisfaction = current_conversion = 0
                
            # Update metrics based on interaction type
            if interaction_data["interaction_type"] == "click":
                current_ctr += 0.2  # Increment CTR
                current_engagement += 0.1
                
            elif interaction_data["interaction_type"] == "booking":
                current_conversion += 0.5
                current_engagement += 0.3
                
            elif interaction_data["interaction_type"] == "rating":
                if interaction_data.get("satisfaction_rating"):
                    current_satisfaction = interaction_data["satisfaction_rating"]
                    
            # Store updated metrics
            cursor.execute('''
                INSERT OR REPLACE INTO recommendation_metrics 
                (metric_id, user_id, test_id, variant, click_through_rate, 
                 engagement_score, satisfaction_rating, conversion_rate, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric_id, user_id, test_id, variant, current_ctr,
                current_engagement, current_satisfaction, current_conversion,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
            
    def analyze_ab_test_results(self, test_id: str) -> Dict:
        """Analyze A/B test results with statistical significance testing"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get test configuration
            test_config_df = pd.read_sql_query('''
                SELECT * FROM ab_test_configs WHERE test_id = ?
            ''', conn, params=(test_id,))
            
            if test_config_df.empty:
                return {"error": "Test not found"}
                
            # Get metrics for all variants
            metrics_df = pd.read_sql_query('''
                SELECT variant, click_through_rate, engagement_score, 
                       satisfaction_rating, conversion_rate
                FROM recommendation_metrics 
                WHERE test_id = ?
            ''', conn, params=(test_id,))
            
            if metrics_df.empty:
                return {"error": "No metrics data found"}
                
            # Calculate summary statistics by variant
            results = {
                "test_id": test_id,
                "test_name": test_config_df.iloc[0]["test_name"],
                "analysis_date": datetime.now().isoformat(),
                "variants": {},
                "statistical_tests": {},
                "recommendations": []
            }
            
            # Analyze each variant
            for variant in metrics_df["variant"].unique():
                variant_data = metrics_df[metrics_df["variant"] == variant]
                
                results["variants"][variant] = {
                    "sample_size": len(variant_data),
                    "metrics": {
                        "click_through_rate": {
                            "mean": float(variant_data["click_through_rate"].mean()),
                            "std": float(variant_data["click_through_rate"].std()),
                            "confidence_interval": self._calculate_confidence_interval(
                                variant_data["click_through_rate"]
                            )
                        },
                        "engagement_score": {
                            "mean": float(variant_data["engagement_score"].mean()),
                            "std": float(variant_data["engagement_score"].std()),
                            "confidence_interval": self._calculate_confidence_interval(
                                variant_data["engagement_score"]
                            )
                        },
                        "satisfaction_rating": {
                            "mean": float(variant_data["satisfaction_rating"].mean()),
                            "std": float(variant_data["satisfaction_rating"].std()),
                            "confidence_interval": self._calculate_confidence_interval(
                                variant_data["satisfaction_rating"]
                            )
                        },
                        "conversion_rate": {
                            "mean": float(variant_data["conversion_rate"].mean()),
                            "std": float(variant_data["conversion_rate"].std()),
                            "confidence_interval": self._calculate_confidence_interval(
                                variant_data["conversion_rate"]
                            )
                        }
                    }
                }
                
            # Statistical significance testing
            if len(results["variants"]) == 2:
                variants = list(results["variants"].keys())
                control_data = metrics_df[metrics_df["variant"] == variants[0]]
                treatment_data = metrics_df[metrics_df["variant"] == variants[1]]
                
                # T-tests for each metric
                for metric in ["click_through_rate", "engagement_score", "satisfaction_rating", "conversion_rate"]:
                    control_values = control_data[metric].dropna()
                    treatment_values = treatment_data[metric].dropna()
                    
                    if len(control_values) > 1 and len(treatment_values) > 1:
                        t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
                        
                        results["statistical_tests"][metric] = {
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "significant": p_value < (1 - float(test_config_df.iloc[0]["confidence_level"])),
                            "effect_size": float(treatment_values.mean() - control_values.mean())
                        }
                        
            # Generate recommendations
            results["recommendations"] = self._generate_test_recommendations(results)
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing A/B test: {str(e)}")
            return {"error": str(e)}
            
    def _calculate_confidence_interval(self, data: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a data series"""
        try:
            data_clean = data.dropna()
            if len(data_clean) < 2:
                return (0.0, 0.0)
                
            mean = data_clean.mean()
            sem = stats.sem(data_clean)
            interval = stats.t.interval(confidence, len(data_clean)-1, loc=mean, scale=sem)
            
            return (float(interval[0]), float(interval[1]))
            
        except Exception:
            return (0.0, 0.0)
            
    def _generate_test_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on A/B test results"""
        recommendations = []
        
        try:
            if "statistical_tests" in results:
                for metric, test_result in results["statistical_tests"].items():
                    if test_result["significant"]:
                        if test_result["effect_size"] > 0:
                            recommendations.append(
                                f"Treatment variant shows significant improvement in {metric} "
                                f"(effect size: {test_result['effect_size']:.3f})"
                            )
                        else:
                            recommendations.append(
                                f"Control variant performs significantly better in {metric} "
                                f"(effect size: {abs(test_result['effect_size']):.3f})"
                            )
                            
            if not recommendations:
                recommendations.append("No statistically significant differences found between variants")
                
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
            
        return recommendations
        
    def generate_ab_test_report(self, test_id: str, save_plots: bool = True) -> Dict:
        """Generate comprehensive A/B test report with visualizations"""
        try:
            results = self.analyze_ab_test_results(test_id)
            
            if "error" in results:
                return results
                
            # Create visualizations if requested
            if save_plots:
                self._create_ab_test_plots(test_id, results)
                
            # Add executive summary
            results["executive_summary"] = self._create_executive_summary(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error generating A/B test report: {str(e)}")
            return {"error": str(e)}
            
    def _create_ab_test_plots(self, test_id: str, results: Dict):
        """Create visualization plots for A/B test results"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'A/B Test Results: {results["test_name"]}', fontsize=16)
            
            metrics = ["click_through_rate", "engagement_score", "satisfaction_rating", "conversion_rate"]
            metric_titles = ["Click Through Rate", "Engagement Score", "Satisfaction Rating", "Conversion Rate"]
            
            for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
                ax = axes[i//2, i%2]
                
                variants = list(results["variants"].keys())
                values = [results["variants"][variant]["metrics"][metric]["mean"] for variant in variants]
                errors = [results["variants"][variant]["metrics"][metric]["std"] for variant in variants]
                
                bars = ax.bar(variants, values, yerr=errors, capsize=5, alpha=0.7)
                ax.set_title(title)
                ax.set_ylabel('Score')
                
                # Add significance annotation if available
                if metric in results.get("statistical_tests", {}):
                    if results["statistical_tests"][metric]["significant"]:
                        ax.text(0.5, max(values) * 0.9, "Significant*", 
                               ha='center', va='bottom', transform=ax.transData)
                        
            plt.tight_layout()
            plt.savefig(f'ab_test_results_{test_id}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"A/B test plots saved: ab_test_results_{test_id}.png")
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {str(e)}")
            
    def _create_executive_summary(self, results: Dict) -> Dict:
        """Create executive summary of A/B test results"""
        try:
            summary = {
                "test_duration": "N/A",
                "total_participants": sum(v["sample_size"] for v in results["variants"].values()),
                "key_findings": [],
                "recommendation": "Continue monitoring"
            }
            
            # Identify key findings
            if "statistical_tests" in results:
                significant_improvements = []
                for metric, test_result in results["statistical_tests"].items():
                    if test_result["significant"] and test_result["effect_size"] > 0:
                        significant_improvements.append(metric)
                        
                if significant_improvements:
                    summary["key_findings"].append(
                        f"Treatment variant shows significant improvements in: {', '.join(significant_improvements)}"
                    )
                    summary["recommendation"] = "Implement treatment variant"
                else:
                    summary["key_findings"].append("No significant improvements found in treatment variant")
                    
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating executive summary: {str(e)}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Initialize A/B testing system
    ab_testing = PersonalizationABTesting()
    
    # Create A/B test configuration
    config = ABTestConfig(
        test_name="Personalized vs Generic Recommendations",
        start_date=datetime.now(),
        end_date=datetime.now() + timedelta(days=14),
        traffic_allocation={"control": 0.5, "treatment": 0.5},
        success_metrics=["click_through_rate", "engagement_score", "satisfaction_rating"],
        minimum_sample_size=100,
        confidence_level=0.95
    )
    
    test_id = ab_testing.create_ab_test(config)
    print(f"Created A/B Test: {test_id}")
    
    # Simulate user interactions
    for i in range(20):
        user_id = f"user_{i:03d}"
        
        # Get recommendations based on variant
        recommendations = ab_testing.get_recommendations_by_variant(user_id, test_id)
        print(f"User {user_id} assigned to variant: {recommendations['variant']}")
        
        # Simulate user interactions
        if random.random() < 0.3:  # 30% click rate
            ab_testing.track_user_interaction(user_id, test_id, "click", "hagia_sophia")
            
        if random.random() < 0.1:  # 10% conversion rate
            ab_testing.track_user_interaction(user_id, test_id, "booking", "blue_mosque")
            
        if random.random() < 0.2:  # 20% rating rate
            rating = random.uniform(3.5, 5.0)
            ab_testing.track_user_interaction(user_id, test_id, "rating", satisfaction_rating=rating)
            
    # Analyze results
    results = ab_testing.analyze_ab_test_results(test_id)
    print("\nA/B Test Analysis Results:")
    print(json.dumps(results, indent=2, default=str))

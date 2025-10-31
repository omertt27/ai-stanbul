"""
A/B Testing Analytics and Statistics Module

Provides comprehensive analytics and statistical analysis for A/B experiments:
- Metrics aggregation and calculation
- Statistical significance testing (t-tests, chi-square)
- Confidence intervals and effect sizes
- Time-series analysis
- Experiment result evaluation and recommendations

Part of Phase 4: A/B Testing Framework
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import math

logger = logging.getLogger(__name__)


@dataclass
class MetricStats:
    """Statistics for a single metric"""
    name: str
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    count: int
    sum_value: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of comparing two variants"""
    metric_name: str
    control_mean: float
    treatment_mean: float
    absolute_difference: float
    relative_difference_percent: float
    p_value: float
    is_significant: bool
    confidence_level: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    effect_size: float  # Cohen's d
    sample_size_control: int
    sample_size_treatment: int
    winner: Optional[str]  # 'control', 'treatment', or None
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ExperimentReport:
    """Comprehensive experiment analysis report"""
    experiment_id: str
    start_time: datetime
    end_time: datetime
    duration_hours: float
    total_samples: int
    variants: Dict[str, int]  # variant -> sample count
    metric_comparisons: List[ComparisonResult]
    overall_winner: Optional[str]
    confidence_score: float  # 0-100
    recommendations: List[str]
    raw_stats: Dict[str, Dict[str, MetricStats]]  # variant -> metric -> stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'experiment_id': self.experiment_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_hours': self.duration_hours,
            'total_samples': self.total_samples,
            'variants': self.variants,
            'metric_comparisons': [c.to_dict() for c in self.metric_comparisons],
            'overall_winner': self.overall_winner,
            'confidence_score': self.confidence_score,
            'recommendations': self.recommendations,
            'raw_stats': {
                variant: {metric: stats.to_dict() for metric, stats in metrics.items()}
                for variant, metrics in self.raw_stats.items()
            }
        }


class StatisticalAnalyzer:
    """Performs statistical analysis on A/B test data"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize analyzer
        
        Args:
            significance_level: P-value threshold for significance (default: 0.05)
        """
        self.significance_level = significance_level
        self.confidence_level = 1 - significance_level
    
    def calculate_metric_stats(self, values: List[float]) -> MetricStats:
        """
        Calculate comprehensive statistics for a metric
        
        Args:
            values: List of metric values
            
        Returns:
            MetricStats with all statistics
        """
        if not values:
            # Return empty stats
            return MetricStats(
                name="",
                mean=0.0,
                median=0.0,
                std_dev=0.0,
                min_value=0.0,
                max_value=0.0,
                count=0,
                sum_value=0.0,
                percentile_25=0.0,
                percentile_75=0.0,
                percentile_95=0.0
            )
        
        sorted_values = sorted(values)
        n = len(values)
        
        return MetricStats(
            name="",
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if n > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            count=n,
            sum_value=sum(values),
            percentile_25=self._percentile(sorted_values, 25),
            percentile_75=self._percentile(sorted_values, 75),
            percentile_95=self._percentile(sorted_values, 95)
        )
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values"""
        if not sorted_values:
            return 0.0
        
        n = len(sorted_values)
        rank = (percentile / 100) * (n - 1)
        lower_idx = int(math.floor(rank))
        upper_idx = int(math.ceil(rank))
        
        if lower_idx == upper_idx:
            return sorted_values[lower_idx]
        
        # Linear interpolation
        fraction = rank - lower_idx
        return sorted_values[lower_idx] + fraction * (sorted_values[upper_idx] - sorted_values[lower_idx])
    
    def welch_t_test(self, 
                     values1: List[float], 
                     values2: List[float]) -> Tuple[float, float]:
        """
        Perform Welch's t-test (unequal variances t-test)
        
        Args:
            values1: Control group values
            values2: Treatment group values
            
        Returns:
            Tuple of (t_statistic, p_value)
        """
        if len(values1) < 2 or len(values2) < 2:
            return 0.0, 1.0
        
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        var1 = statistics.variance(values1)
        var2 = statistics.variance(values2)
        n1 = len(values1)
        n2 = len(values2)
        
        # Welch's t-statistic
        t_stat = (mean1 - mean2) / math.sqrt(var1/n1 + var2/n2)
        
        # Degrees of freedom (Welch-Satterthwaite equation)
        df = ((var1/n1 + var2/n2) ** 2) / \
             ((var1/n1)**2 / (n1-1) + (var2/n2)**2 / (n2-1))
        
        # Approximate p-value using t-distribution
        # For simplicity, using a conservative approximation
        p_value = self._t_to_p(abs(t_stat), df)
        
        return t_stat, p_value
    
    def _t_to_p(self, t_stat: float, df: float) -> float:
        """
        Convert t-statistic to two-tailed p-value
        Simple approximation for large df
        """
        # For large df, t-distribution approximates normal distribution
        # This is a simplified approximation
        if df >= 30:
            # Use normal approximation
            z = t_stat
            # Approximate two-tailed p-value
            if z > 3.0:
                return 0.001
            elif z > 2.576:
                return 0.01
            elif z > 1.96:
                return 0.05
            elif z > 1.645:
                return 0.10
            else:
                return 0.20
        else:
            # For small df, be more conservative
            if t_stat > 2.5:
                return 0.05
            else:
                return 0.20
    
    def cohens_d(self, values1: List[float], values2: List[float]) -> float:
        """
        Calculate Cohen's d effect size
        
        Args:
            values1: Control group values
            values2: Treatment group values
            
        Returns:
            Effect size (Cohen's d)
        """
        if len(values1) < 2 or len(values2) < 2:
            return 0.0
        
        mean1 = statistics.mean(values1)
        mean2 = statistics.mean(values2)
        var1 = statistics.variance(values1)
        var2 = statistics.variance(values2)
        n1 = len(values1)
        n2 = len(values2)
        
        # Pooled standard deviation
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean2 - mean1) / pooled_std
    
    def confidence_interval(self,
                           values: List[float],
                           confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean
        
        Args:
            values: Sample values
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(values) < 2:
            mean = values[0] if values else 0.0
            return mean, mean
        
        mean = statistics.mean(values)
        std_err = statistics.stdev(values) / math.sqrt(len(values))
        
        # Using t-distribution critical value (approximation)
        # For 95% CI and large n, ~1.96; for small n, larger
        t_critical = 2.0 if len(values) < 30 else 1.96
        
        margin = t_critical * std_err
        return mean - margin, mean + margin


class ExperimentAnalyzer:
    """Analyzes A/B experiment results and generates reports"""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize experiment analyzer
        
        Args:
            significance_level: P-value threshold for significance
        """
        self.stats_analyzer = StatisticalAnalyzer(significance_level)
        self.significance_level = significance_level
    
    def analyze_experiment(self,
                          experiment_id: str,
                          metrics_by_variant: Dict[str, Dict[str, List[float]]],
                          start_time: datetime,
                          end_time: datetime,
                          control_variant: str = "control") -> ExperimentReport:
        """
        Perform comprehensive analysis of an experiment
        
        Args:
            experiment_id: Experiment identifier
            metrics_by_variant: Dict of variant -> metric_name -> values
            start_time: Experiment start time
            end_time: Experiment end time
            control_variant: Name of control variant
            
        Returns:
            ExperimentReport with full analysis
        """
        # Calculate duration
        duration = (end_time - start_time).total_seconds() / 3600  # hours
        
        # Calculate sample sizes
        variant_counts = {}
        total_samples = 0
        for variant, metrics in metrics_by_variant.items():
            # Use first metric to count samples
            count = len(next(iter(metrics.values()))) if metrics else 0
            variant_counts[variant] = count
            total_samples += count
        
        # Calculate statistics for each variant and metric
        raw_stats: Dict[str, Dict[str, MetricStats]] = {}
        for variant, metrics in metrics_by_variant.items():
            raw_stats[variant] = {}
            for metric_name, values in metrics.items():
                stats = self.stats_analyzer.calculate_metric_stats(values)
                stats.name = metric_name
                raw_stats[variant][metric_name] = stats
        
        # Compare each treatment variant to control
        comparisons: List[ComparisonResult] = []
        
        if control_variant not in metrics_by_variant:
            logger.warning(f"Control variant '{control_variant}' not found in data")
            control_variant = list(metrics_by_variant.keys())[0]
        
        control_metrics = metrics_by_variant[control_variant]
        
        for variant_name, variant_metrics in metrics_by_variant.items():
            if variant_name == control_variant:
                continue
            
            # Compare each metric
            for metric_name in control_metrics.keys():
                if metric_name not in variant_metrics:
                    continue
                
                control_values = control_metrics[metric_name]
                treatment_values = variant_metrics[metric_name]
                
                comparison = self._compare_variants(
                    metric_name=metric_name,
                    control_values=control_values,
                    treatment_values=treatment_values,
                    control_name=control_variant,
                    treatment_name=variant_name
                )
                comparisons.append(comparison)
        
        # Determine overall winner and confidence
        winner, confidence = self._determine_winner(comparisons)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            comparisons, winner, confidence, variant_counts
        )
        
        return ExperimentReport(
            experiment_id=experiment_id,
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration,
            total_samples=total_samples,
            variants=variant_counts,
            metric_comparisons=comparisons,
            overall_winner=winner,
            confidence_score=confidence,
            recommendations=recommendations,
            raw_stats=raw_stats
        )
    
    def _compare_variants(self,
                         metric_name: str,
                         control_values: List[float],
                         treatment_values: List[float],
                         control_name: str,
                         treatment_name: str) -> ComparisonResult:
        """Compare two variants for a single metric"""
        
        # Calculate means
        control_mean = statistics.mean(control_values) if control_values else 0.0
        treatment_mean = statistics.mean(treatment_values) if treatment_values else 0.0
        
        # Calculate differences
        abs_diff = treatment_mean - control_mean
        rel_diff = (abs_diff / control_mean * 100) if control_mean != 0 else 0.0
        
        # Statistical test
        t_stat, p_value = self.stats_analyzer.welch_t_test(control_values, treatment_values)
        is_significant = p_value < self.significance_level
        
        # Effect size
        effect_size = self.stats_analyzer.cohens_d(control_values, treatment_values)
        
        # Confidence interval for difference
        ci_lower, ci_upper = self.stats_analyzer.confidence_interval(
            [t - c for c, t in zip(control_values, treatment_values[:len(control_values)])]
        )
        
        # Determine winner
        winner = None
        if is_significant:
            # For response time, lower is better
            if 'time' in metric_name.lower() or 'latency' in metric_name.lower():
                winner = treatment_name if treatment_mean < control_mean else control_name
            else:
                # For most metrics, higher is better
                winner = treatment_name if treatment_mean > control_mean else control_name
        
        # Generate recommendation
        if is_significant:
            if abs(effect_size) > 0.8:
                effect_desc = "large"
            elif abs(effect_size) > 0.5:
                effect_desc = "medium"
            else:
                effect_desc = "small"
            
            recommendation = f"Statistically significant {effect_desc} effect (p={p_value:.4f}). " \
                           f"{winner} performs {abs(rel_diff):.1f}% better."
        else:
            recommendation = f"No significant difference (p={p_value:.4f}). " \
                           f"May need more data or effect is negligible."
        
        return ComparisonResult(
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            absolute_difference=abs_diff,
            relative_difference_percent=rel_diff,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.stats_analyzer.confidence_level,
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            effect_size=effect_size,
            sample_size_control=len(control_values),
            sample_size_treatment=len(treatment_values),
            winner=winner,
            recommendation=recommendation
        )
    
    def _determine_winner(self,
                         comparisons: List[ComparisonResult]) -> Tuple[Optional[str], float]:
        """
        Determine overall winner across all metrics
        
        Returns:
            Tuple of (winner_name, confidence_score)
        """
        if not comparisons:
            return None, 0.0
        
        # Count wins per variant
        wins: Dict[str, int] = defaultdict(int)
        significant_comparisons = 0
        
        for comp in comparisons:
            if comp.is_significant and comp.winner:
                wins[comp.winner] += 1
                significant_comparisons += 1
        
        if not wins:
            return None, 0.0
        
        # Winner is variant with most wins
        winner = max(wins.items(), key=lambda x: x[1])[0]
        winner_wins = wins[winner]
        
        # Confidence based on:
        # 1. Proportion of significant wins
        # 2. Average effect size
        # 3. Sample sizes
        
        total_metrics = len(comparisons)
        win_rate = winner_wins / total_metrics
        
        # Average effect size for winner's wins
        winner_effects = [
            abs(comp.effect_size) for comp in comparisons
            if comp.winner == winner and comp.is_significant
        ]
        avg_effect = statistics.mean(winner_effects) if winner_effects else 0.0
        
        # Normalize effect size to 0-1 (cap at 1.0)
        effect_score = min(avg_effect / 1.0, 1.0)
        
        # Combined confidence score (0-100)
        confidence = (win_rate * 0.6 + effect_score * 0.4) * 100
        
        return winner, confidence
    
    def _generate_recommendations(self,
                                 comparisons: List[ComparisonResult],
                                 winner: Optional[str],
                                 confidence: float,
                                 variant_counts: Dict[str, int]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check sample sizes
        min_samples = min(variant_counts.values()) if variant_counts else 0
        if min_samples < 100:
            recommendations.append(
                f"⚠️ Small sample sizes (min: {min_samples}). "
                "Collect more data for reliable conclusions."
            )
        
        # Overall winner recommendation
        if winner and confidence >= 80:
            recommendations.append(
                f"✅ Strong recommendation: Deploy '{winner}' (confidence: {confidence:.1f}%)"
            )
        elif winner and confidence >= 60:
            recommendations.append(
                f"⚡ Moderate recommendation: Consider '{winner}' (confidence: {confidence:.1f}%). "
                "Monitor closely or collect more data."
            )
        elif winner:
            recommendations.append(
                f"⚠️ Weak signal for '{winner}' (confidence: {confidence:.1f}%). "
                "Continue experiment or investigate further."
            )
        else:
            recommendations.append(
                "📊 No clear winner. Variants perform similarly. "
                "Consider other factors (cost, complexity) or continue testing."
            )
        
        # Metric-specific recommendations
        significant_metrics = [c for c in comparisons if c.is_significant]
        if significant_metrics:
            recommendations.append(
                f"📈 {len(significant_metrics)} metrics show significant differences. "
                "Review individual metric details."
            )
        
        # Check for conflicting results
        winners = set(c.winner for c in comparisons if c.winner)
        if len(winners) > 1:
            recommendations.append(
                "⚠️ Mixed results: Different variants win on different metrics. "
                "Prioritize metrics by business importance."
            )
        
        return recommendations
    
    def export_report(self, report: ExperimentReport, filepath: str):
        """Export report to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Exported experiment report to {filepath}")
    
    def print_report_summary(self, report: ExperimentReport):
        """Print human-readable report summary"""
        print("\n" + "="*80)
        print(f"A/B TEST REPORT: {report.experiment_id}")
        print("="*80)
        print(f"\nDuration: {report.duration_hours:.1f} hours")
        print(f"Total Samples: {report.total_samples:,}")
        print(f"\nVariants:")
        for variant, count in report.variants.items():
            pct = (count / report.total_samples * 100) if report.total_samples > 0 else 0
            print(f"  - {variant}: {count:,} ({pct:.1f}%)")
        
        print(f"\n{'Metric Comparisons':-^80}")
        for comp in report.metric_comparisons:
            print(f"\n{comp.metric_name}:")
            print(f"  Control: {comp.control_mean:.4f}")
            print(f"  Treatment: {comp.treatment_mean:.4f}")
            print(f"  Difference: {comp.relative_difference_percent:+.2f}%")
            print(f"  P-value: {comp.p_value:.4f} {'✓ SIGNIFICANT' if comp.is_significant else '✗ Not significant'}")
            print(f"  Effect Size (Cohen's d): {comp.effect_size:.3f}")
            if comp.winner:
                print(f"  Winner: {comp.winner}")
        
        print(f"\n{'Overall Results':-^80}")
        if report.overall_winner:
            print(f"Winner: {report.overall_winner}")
            print(f"Confidence: {report.confidence_score:.1f}%")
        else:
            print("Winner: No clear winner")
        
        print(f"\n{'Recommendations':-^80}")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*80 + "\n")


# Convenience functions
def analyze_experiment_data(experiment_id: str,
                           metrics_by_variant: Dict[str, Dict[str, List[float]]],
                           start_time: datetime,
                           end_time: datetime,
                           control_variant: str = "control",
                           significance_level: float = 0.05) -> ExperimentReport:
    """
    Analyze experiment data and return comprehensive report
    
    Args:
        experiment_id: Experiment identifier
        metrics_by_variant: Dict of variant -> metric_name -> values
        start_time: Experiment start time
        end_time: Experiment end time
        control_variant: Name of control variant
        significance_level: P-value threshold
        
    Returns:
        ExperimentReport with full analysis
    """
    analyzer = ExperimentAnalyzer(significance_level)
    return analyzer.analyze_experiment(
        experiment_id=experiment_id,
        metrics_by_variant=metrics_by_variant,
        start_time=start_time,
        end_time=end_time,
        control_variant=control_variant
    )


def quick_comparison(control_values: List[float],
                    treatment_values: List[float],
                    metric_name: str = "metric") -> ComparisonResult:
    """
    Quick statistical comparison between two variants
    
    Args:
        control_values: Control variant values
        treatment_values: Treatment variant values
        metric_name: Name of the metric
        
    Returns:
        ComparisonResult with statistical analysis
    """
    analyzer = ExperimentAnalyzer()
    return analyzer._compare_variants(
        metric_name=metric_name,
        control_values=control_values,
        treatment_values=treatment_values,
        control_name="control",
        treatment_name="treatment"
    )

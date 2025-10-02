#!/usr/bin/env python3
"""
AI Istanbul - Google API Cost Analysis for 50k Monthly Users
Comprehensive cost calculation with optimized caching and field selection
"""

import json
from datetime import datetime
from typing import Dict, List, Any

class GoogleApiCostCalculator:
    """Calculate Google API costs with optimization strategies"""
    
    def __init__(self):
        # Google Places API pricing (as of October 2024)
        self.places_api_pricing = {
            'text_search': 0.032,      # $32 per 1000 requests
            'place_details': 0.017,    # $17 per 1000 requests  
            'find_place': 0.017,       # $17 per 1000 requests
            'nearby_search': 0.032,    # $32 per 1000 requests
            'photos': 0.007,           # $7 per 1000 requests
            'autocomplete': 0.00283,   # $2.83 per 1000 requests
        }
        
        # Field-based pricing for Places API
        self.field_pricing = {
            # Basic fields (included in base price)
            'basic': ['place_id', 'name', 'rating', 'geometry', 'types', 'vicinity'],
            
            # Contact fields (+$3 per 1000 requests)
            'contact': ['formatted_address', 'formatted_phone_number', 'website', 'url'],
            'contact_cost': 0.003,
            
            # Atmosphere fields (+$5 per 1000 requests)  
            'atmosphere': ['price_level', 'opening_hours', 'wheelchair_accessible_entrance'],
            'atmosphere_cost': 0.005,
            
            # Premium fields (+$15 per 1000 requests)
            'premium': ['photos', 'reviews', 'editorial_summary', 'current_opening_hours'],
            'premium_cost': 0.015
        }
        
        # User behavior patterns for 50k monthly users
        self.user_patterns = {
            'total_monthly_users': 50000,
            'active_users_percentage': 0.35,        # 35% of users are active monthly
            'queries_per_active_user': 8.5,         # Average queries per active user
            'restaurant_query_percentage': 0.45,     # 45% of queries are restaurant-related
            'repeat_query_percentage': 0.25,        # 25% are similar/repeated queries
            'peak_hour_multiplier': 1.8,            # 80% more queries during peak hours
            'seasonal_variance': 1.2,               # 20% increase during tourist season
        }
        
        # Optimization factors
        self.optimizations = {
            'cache_hit_rate': 0.72,                 # 72% cache hit rate (aggressive caching)
            'field_selection_savings': 0.20,        # 20% savings from field optimization
            'query_deduplication': 0.15,            # 15% savings from smart deduplication
            'rate_limiting_efficiency': 0.08,       # 8% savings from intelligent rate limiting
            'request_batching': 0.12,               # 12% savings from request batching
        }
    
    def calculate_base_usage(self) -> Dict[str, Any]:
        """Calculate base API usage without optimizations"""
        
        active_users = int(self.user_patterns['total_monthly_users'] * 
                          self.user_patterns['active_users_percentage'])
        
        total_queries = int(active_users * self.user_patterns['queries_per_active_user'])
        
        restaurant_queries = int(total_queries * 
                               self.user_patterns['restaurant_query_percentage'])
        
        # Apply seasonal and peak hour variations
        adjusted_queries = int(restaurant_queries * 
                             self.user_patterns['seasonal_variance'] *
                             self.user_patterns['peak_hour_multiplier'])
        
        return {
            'total_monthly_users': self.user_patterns['total_monthly_users'],
            'active_users': active_users,
            'total_queries': total_queries,
            'restaurant_queries': restaurant_queries,
            'adjusted_restaurant_queries': adjusted_queries,
            'api_requests_needed': adjusted_queries  # 1:1 ratio for now
        }
    
    def calculate_optimized_usage(self, base_usage: Dict) -> Dict[str, Any]:
        """Calculate API usage with all optimizations applied"""
        
        base_requests = base_usage['api_requests_needed']
        
        # Apply cache optimization (most significant)
        cache_savings = int(base_requests * self.optimizations['cache_hit_rate'])
        requests_after_cache = base_requests - cache_savings
        
        # Apply field selection optimization
        field_savings = int(requests_after_cache * self.optimizations['field_selection_savings'])
        requests_after_fields = requests_after_cache - field_savings
        
        # Apply deduplication
        dedup_savings = int(requests_after_fields * self.optimizations['query_deduplication'])
        requests_after_dedup = requests_after_fields - dedup_savings
        
        # Apply rate limiting efficiency
        rate_limit_savings = int(requests_after_dedup * self.optimizations['rate_limiting_efficiency'])
        requests_after_rate_limit = requests_after_dedup - rate_limit_savings
        
        # Apply request batching
        batching_savings = int(requests_after_rate_limit * self.optimizations['request_batching'])
        final_requests = requests_after_rate_limit - batching_savings
        
        return {
            'base_requests': base_requests,
            'cache_savings': cache_savings,
            'requests_after_cache': requests_after_cache,
            'field_selection_savings': field_savings,
            'deduplication_savings': dedup_savings,
            'rate_limiting_savings': rate_limit_savings,
            'batching_savings': batching_savings,
            'final_api_requests': final_requests,
            'total_requests_saved': base_requests - final_requests,
            'optimization_percentage': round(((base_requests - final_requests) / base_requests) * 100, 1)
        }
    
    def calculate_field_costs(self, requests: int, optimization_level: str = "aggressive") -> Dict[str, Any]:
        """Calculate costs based on field selection optimization"""
        
        if optimization_level == "none":
            # Request all fields (worst case)
            contact_requests = requests
            atmosphere_requests = requests  
            premium_requests = requests
        elif optimization_level == "moderate":
            # Some field optimization
            contact_requests = int(requests * 0.6)     # 60% of requests need contact info
            atmosphere_requests = int(requests * 0.4)   # 40% need atmosphere data
            premium_requests = int(requests * 0.2)      # 20% need premium fields
        else:  # aggressive
            # Dynamic field selection (our implementation)
            contact_requests = int(requests * 0.35)     # 35% need contact info
            atmosphere_requests = int(requests * 0.25)  # 25% need atmosphere data  
            premium_requests = int(requests * 0.10)     # 10% need premium fields
        
        base_cost = requests * self.places_api_pricing['text_search']
        contact_cost = contact_requests * self.field_pricing['contact_cost']
        atmosphere_cost = atmosphere_requests * self.field_pricing['atmosphere_cost']
        premium_cost = premium_requests * self.field_pricing['premium_cost']
        
        total_cost = base_cost + contact_cost + atmosphere_cost + premium_cost
        
        return {
            'base_requests': requests,
            'base_cost': base_cost,
            'contact_requests': contact_requests,
            'contact_cost': contact_cost,
            'atmosphere_requests': atmosphere_requests, 
            'atmosphere_cost': atmosphere_cost,
            'premium_requests': premium_requests,
            'premium_cost': premium_cost,
            'total_monthly_cost': total_cost,
            'cost_per_1k_requests': (total_cost / requests) * 1000 if requests > 0 else 0
        }
    
    def generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate complete cost analysis for 50k users"""
        
        # Calculate base usage
        base_usage = self.calculate_base_usage()
        
        # Calculate optimized usage
        optimized_usage = self.calculate_optimized_usage(base_usage)
        
        # Calculate costs for different scenarios
        unoptimized_costs = self.calculate_field_costs(
            base_usage['api_requests_needed'], 
            "none"
        )
        
        moderate_costs = self.calculate_field_costs(
            optimized_usage['final_api_requests'], 
            "moderate"
        )
        
        optimized_costs = self.calculate_field_costs(
            optimized_usage['final_api_requests'], 
            "aggressive"
        )
        
        # Calculate savings
        monthly_savings = unoptimized_costs['total_monthly_cost'] - optimized_costs['total_monthly_cost']
        annual_savings = monthly_savings * 12
        savings_percentage = ((monthly_savings / unoptimized_costs['total_monthly_cost']) * 100 
                            if unoptimized_costs['total_monthly_cost'] > 0 else 0)
        
        # Additional calculations
        cost_per_user = optimized_costs['total_monthly_cost'] / self.user_patterns['total_monthly_users']
        cost_per_active_user = optimized_costs['total_monthly_cost'] / base_usage['active_users']
        
        return {
            'analysis_date': datetime.now().isoformat(),
            'user_metrics': base_usage,
            'optimization_impact': optimized_usage,
            'cost_breakdown': {
                'unoptimized': unoptimized_costs,
                'moderate_optimization': moderate_costs,
                'full_optimization': optimized_costs
            },
            'savings_analysis': {
                'monthly_savings_usd': round(monthly_savings, 2),
                'annual_savings_usd': round(annual_savings, 2),
                'savings_percentage': round(savings_percentage, 1),
                'cost_per_user_monthly': round(cost_per_user, 4),
                'cost_per_active_user_monthly': round(cost_per_active_user, 2)
            },
            'optimization_breakdown': {
                'cache_savings_percent': round(self.optimizations['cache_hit_rate'] * 100, 1),
                'field_selection_savings_percent': round(self.optimizations['field_selection_savings'] * 100, 1),
                'deduplication_savings_percent': round(self.optimizations['query_deduplication'] * 100, 1),
                'rate_limiting_savings_percent': round(self.optimizations['rate_limiting_efficiency'] * 100, 1),
                'batching_savings_percent': round(self.optimizations['request_batching'] * 100, 1)
            },
            'recommendations': self._generate_recommendations(optimized_costs, monthly_savings)
        }
    
    def _generate_recommendations(self, costs: Dict, savings: float) -> List[str]:
        """Generate cost optimization recommendations"""
        
        recommendations = []
        
        if costs['total_monthly_cost'] > 1000:
            recommendations.append("💰 Consider implementing request quotas per user to control costs")
        
        if savings > 500:
            recommendations.append("🎯 Current optimization strategy is highly effective - maintain these practices")
        
        recommendations.extend([
            "📊 Monitor field usage patterns monthly to optimize field selection further",
            "🔄 Implement progressive cache warming during off-peak hours",
            "⚡ Consider implementing query prediction to pre-cache popular searches",
            "🎛️ Implement dynamic pricing alerts when approaching budget limits",
            "📈 Track user satisfaction metrics to ensure optimizations don't impact UX"
        ])
        
        return recommendations

def generate_cost_report():
    """Generate and display comprehensive cost analysis"""
    
    calculator = GoogleApiCostCalculator()
    analysis = calculator.generate_comprehensive_analysis()
    
    print("🏙️  AI ISTANBUL - GOOGLE API COST ANALYSIS")
    print("=" * 80)
    print(f"📅 Analysis Date: {analysis['analysis_date'][:10]}")
    print(f"👥 Total Users: {analysis['user_metrics']['total_monthly_users']:,}")
    print(f"🔥 Active Users: {analysis['user_metrics']['active_users']:,}")
    print()
    
    print("📊 USAGE METRICS")
    print("-" * 40)
    print(f"Total Queries: {analysis['user_metrics']['total_queries']:,}")
    print(f"Restaurant Queries: {analysis['user_metrics']['restaurant_queries']:,}")
    print(f"API Requests (before optimization): {analysis['user_metrics']['api_requests_needed']:,}")
    print(f"API Requests (after optimization): {analysis['optimization_impact']['final_api_requests']:,}")
    print(f"Optimization Efficiency: {analysis['optimization_impact']['optimization_percentage']}%")
    print()
    
    print("💰 COST BREAKDOWN")
    print("-" * 40)
    unopt = analysis['cost_breakdown']['unoptimized']
    opt = analysis['cost_breakdown']['full_optimization']
    
    print(f"Without Optimization: ${unopt['total_monthly_cost']:.2f}/month")
    print(f"With Full Optimization: ${opt['total_monthly_cost']:.2f}/month")
    print(f"Monthly Savings: ${analysis['savings_analysis']['monthly_savings_usd']:.2f}")
    print(f"Annual Savings: ${analysis['savings_analysis']['annual_savings_usd']:.2f}")
    print(f"Cost Reduction: {analysis['savings_analysis']['savings_percentage']}%")
    print()
    
    print("🎯 OPTIMIZATION IMPACT")
    print("-" * 40)
    opt_breakdown = analysis['optimization_breakdown']
    print(f"Cache Hit Rate: {opt_breakdown['cache_savings_percent']}%")
    print(f"Field Selection Savings: {opt_breakdown['field_selection_savings_percent']}%")
    print(f"Query Deduplication: {opt_breakdown['deduplication_savings_percent']}%")
    print(f"Rate Limiting Efficiency: {opt_breakdown['rate_limiting_savings_percent']}%")
    print(f"Request Batching: {opt_breakdown['batching_savings_percent']}%")
    print()
    
    print("📈 PER-USER COSTS")
    print("-" * 40)
    savings = analysis['savings_analysis']
    print(f"Cost per User (monthly): ${savings['cost_per_user_monthly']:.4f}")
    print(f"Cost per Active User (monthly): ${savings['cost_per_active_user_monthly']:.2f}")
    print()
    
    print("🎯 RECOMMENDATIONS")
    print("-" * 40)
    for i, rec in enumerate(analysis['recommendations'], 1):
        print(f"{i}. {rec}")
    print()
    
    print("🚀 IMPLEMENTATION STATUS")
    print("-" * 40)
    print("✅ Dynamic Field Selection - IMPLEMENTED")
    print("✅ Aggressive Caching (72% hit rate) - IMPLEMENTED") 
    print("✅ Query Deduplication - IMPLEMENTED")
    print("✅ Rate Limiting - IMPLEMENTED")
    print("⚠️  Request Batching - PARTIALLY IMPLEMENTED")
    print("⚠️  Predictive Caching - PLANNED")
    
    return analysis

if __name__ == "__main__":
    # Generate comprehensive cost analysis
    analysis = generate_cost_report()
    
    # Save detailed analysis to JSON file
    with open('google_api_cost_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to: google_api_cost_analysis.json")

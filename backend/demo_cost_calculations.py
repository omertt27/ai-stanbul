#!/usr/bin/env python3
"""
Cost Calculation Demo Script
===========================

Demonstrates the cost savings achieved through the integrated cache system
and field optimization features.
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def demo_cost_calculations():
    """Demo comprehensive cost calculation features"""
    
    print("🧮 AI Istanbul - Cost Optimization Demo")
    print("=" * 60)
    print(f"Demo started at: {datetime.now().isoformat()}")
    print()
    
    # Test queries with different intents
    test_scenarios = [
        {
            "query": "best Turkish restaurants in Sultanahmet",
            "intent": "basic_search",
            "description": "Basic restaurant search - high optimization potential"
        },
        {
            "query": "tell me everything about Pandeli restaurant hours menu prices",
            "intent": "detailed_info", 
            "description": "Detailed info request - moderate optimization"
        },
        {
            "query": "quick lunch nearby",
            "intent": "quick_recommendation",
            "description": "Quick recommendation - maximum optimization"
        },
        {
            "query": "wheelchair accessible restaurants in Beyoğlu",
            "intent": "accessibility",
            "description": "Accessibility focused - specialized optimization"
        },
        {
            "query": "romantic restaurants open now with live music",
            "intent": "dining_decision",
            "description": "Complex dining decision - multiple factors"
        }
    ]
    
    print("📊 Testing Cost Optimization Scenarios:")
    print("-" * 60)
    
    total_optimized_cost = 0
    total_unoptimized_cost = 0
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. {scenario['description']}")
        print(f"   Query: '{scenario['query']}'")
        print(f"   Intent: {scenario['intent']}")
        
        # Simulate different optimization outcomes
        cache_hit = i % 2 == 0  # Alternate cache hits/misses
        fields_requested = {
            "basic_search": 8,
            "detailed_info": 15,
            "quick_recommendation": 5,
            "accessibility": 10,
            "dining_decision": 12
        }.get(scenario['intent'], 10)
        
        total_fields = 25  # Total available fields
        response_time = 50 if cache_hit else 400
        
        # Import cost calculation function
        try:
            from routes.restaurants import _calculate_enhanced_cost_analysis
            
            cost_analysis = _calculate_enhanced_cost_analysis(
                cache_hit=cache_hit,
                response_time_ms=response_time,
                cache_type="restaurant_basic_info" if cache_hit else "unknown",
                fields_requested=fields_requested,
                total_available_fields=total_fields,
                original_cost_score=100,
                optimized_cost_score=30,
                query=scenario['query'],
                intent=scenario['intent']
            )
            
            current_cost = cost_analysis['current_request']['total_cost_usd']
            unoptimized_cost = cost_analysis['cost_comparison']['unoptimized_cost_usd']
            savings_percent = cost_analysis['cost_comparison']['savings_percent']
            
            total_optimized_cost += current_cost
            total_unoptimized_cost += unoptimized_cost
            
            print(f"   💰 Cost: ${current_cost:.6f} (vs ${unoptimized_cost:.6f} unoptimized)")
            print(f"   💾 Cache: {'HIT' if cache_hit else 'MISS'} | ⏱️ Time: {response_time}ms")
            print(f"   📊 Fields: {fields_requested}/{total_fields} | 💸 Savings: {savings_percent:.1f}%")
            print(f"   🎯 Efficiency: {cost_analysis['performance_metrics']['efficiency_rating']}")
            
        except ImportError:
            print("   ⚠️  Cost calculation functions not available")
            # Use estimated values for demo
            base_cost = 0.017 + (fields_requested * 0.003)
            optimized_cost = 0 if cache_hit else base_cost * 0.6
            savings = ((base_cost - optimized_cost) / base_cost) * 100
            
            total_optimized_cost += optimized_cost
            total_unoptimized_cost += base_cost
            
            print(f"   💰 Estimated Cost: ${optimized_cost:.6f} (vs ${base_cost:.6f} unoptimized)")
            print(f"   💾 Cache: {'HIT' if cache_hit else 'MISS'} | ⏱️ Time: {response_time}ms")
            print(f"   📊 Fields: {fields_requested}/{total_fields} | 💸 Estimated Savings: {savings:.1f}%")
    
    # Overall summary
    overall_savings = total_unoptimized_cost - total_optimized_cost
    overall_savings_percent = (overall_savings / total_unoptimized_cost) * 100 if total_unoptimized_cost > 0 else 0
    
    print("\n" + "=" * 60)
    print("📈 OVERALL COST ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Optimized Cost:   ${total_optimized_cost:.6f}")
    print(f"Total Unoptimized Cost: ${total_unoptimized_cost:.6f}")
    print(f"Total Savings:          ${overall_savings:.6f}")
    print(f"Overall Savings:        {overall_savings_percent:.1f}%")
    
    # Monthly projections for different business scales
    print(f"\n💼 MONTHLY COST PROJECTIONS:")
    print("-" * 60)
    
    business_scales = [
        ("Small Business", 500, "Local restaurant with basic search needs"),
        ("Medium Business", 5000, "Restaurant chain with moderate usage"),
        ("Large Enterprise", 50000, "Food delivery platform with high volume"),
        ("Startup MVP", 1000, "New restaurant discovery app")
    ]
    
    avg_cost_per_query = total_optimized_cost / len(test_scenarios)
    avg_unoptimized_cost = total_unoptimized_cost / len(test_scenarios)
    
    for scale_name, monthly_requests, description in business_scales:
        monthly_optimized = avg_cost_per_query * monthly_requests
        monthly_unoptimized = avg_unoptimized_cost * monthly_requests
        monthly_savings = monthly_unoptimized - monthly_optimized
        annual_savings = monthly_savings * 12
        
        print(f"\n{scale_name} ({monthly_requests:,} monthly requests)")
        print(f"  📝 {description}")
        print(f"  💰 Monthly Cost: ${monthly_optimized:.2f} (vs ${monthly_unoptimized:.2f})")
        print(f"  💸 Monthly Savings: ${monthly_savings:.2f}")
        print(f"  📅 Annual Savings: ${annual_savings:.2f}")
        
        if annual_savings > 1000:
            print(f"  🎯 Impact: HIGH - Significant cost reduction")
        elif annual_savings > 100:
            print(f"  🎯 Impact: MEDIUM - Good operational savings")
        else:
            print(f"  🎯 Impact: LOW - Modest efficiency gains")
    
    # ROI Analysis
    print(f"\n📊 ROI ANALYSIS:")
    print("-" * 60)
    
    # Assume infrastructure costs for Redis + monitoring
    monthly_infrastructure_cost = 50  # $50/month for Redis + monitoring
    
    for scale_name, monthly_requests, _ in business_scales:
        monthly_savings = (avg_unoptimized_cost - avg_cost_per_query) * monthly_requests
        net_monthly_savings = monthly_savings - monthly_infrastructure_cost
        
        if net_monthly_savings > 0:
            roi_months = monthly_infrastructure_cost / monthly_savings if monthly_savings > 0 else float('inf')
            annual_roi = (net_monthly_savings * 12) / (monthly_infrastructure_cost * 12) * 100
            
            print(f"{scale_name}:")
            print(f"  💵 Net Monthly Savings: ${net_monthly_savings:.2f}")
            print(f"  ⏰ Payback Period: {roi_months:.1f} months")
            print(f"  📈 Annual ROI: {annual_roi:.0f}%")
        else:
            print(f"{scale_name}:")
            print(f"  ⚠️  Infrastructure cost exceeds savings for this scale")
    
    # Best practices and recommendations
    print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 60)
    print("1. 🔥 Enable cache warming for popular queries")
    print("2. 🎯 Use intent-based field optimization") 
    print("3. ⏰ Implement dynamic TTL adjustment")
    print("4. 📊 Monitor cache hit rates (target >75%)")
    print("5. 🔄 Regular review of cost analytics")
    print("6. 💡 Consider budget mode for cost-sensitive operations")
    
    print(f"\n✅ Cost optimization demo completed!")
    print("   Check the monitoring dashboard for real-time analytics")
    print("   Use /api/cache/analytics for detailed system metrics")

if __name__ == "__main__":
    asyncio.run(demo_cost_calculations())

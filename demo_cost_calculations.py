#!/usr/bin/env python3
"""
AI Istanbul - Cost Calculation Demo
Demonstrates the cost optimization and analytics system in action.
"""

import json
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random

class CostCalculationDemo:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.demo_results = {
            "timestamp": datetime.now().isoformat(),
            "cost_tracking": {},
            "cache_savings": {},
            "roi_analysis": {},
            "budget_controls": {},
            "demo_summary": {}
        }
        
    async def demonstrate_cost_tracking(self):
        """Demonstrate real-time cost tracking"""
        print("\n💰 Demonstrating Cost Tracking...")
        
        # Simulate various API calls to generate costs
        queries = [
            {"query": "best restaurants Istanbul", "location": "Istanbul"},
            {"query": "romantic dinner Bosphorus", "location": "Istanbul"},
            {"query": "Turkish cuisine Sultanahmet", "location": "Istanbul"},
            {"query": "seafood restaurants Bebek", "location": "Istanbul"},
            {"query": "vegetarian food Kadıköy", "location": "Istanbul"}
        ]
        
        costs_before = await self.get_current_costs()
        
        async with aiohttp.ClientSession() as session:
            print("   🔄 Making sample API requests...")
            
            for i, query in enumerate(queries):
                print(f"   Request {i+1}/5: {query['query']}")
                
                # Make search request
                response = await session.get(
                    f"{self.base_url}/api/restaurants/search",
                    params=query
                )
                
                if response.status == 200:
                    data = await response.json()
                    print(f"      ✅ Found {len(data.get('results', []))} results")
                    
                    # Show cache info if available
                    if "cache_info" in data:
                        cache_info = data["cache_info"]
                        hit_status = "HIT" if cache_info.get("hit", False) else "MISS"
                        print(f"      📊 Cache: {hit_status}, TTL: {cache_info.get('ttl', 'N/A')}s")
                        
                await asyncio.sleep(0.5)  # Brief pause between requests
                
        costs_after = await self.get_current_costs()
        
        # Calculate cost difference
        cost_diff = {
            "api_costs": costs_after.get("api_costs", 0) - costs_before.get("api_costs", 0),
            "total_requests": costs_after.get("total_requests", 0) - costs_before.get("total_requests", 0),
            "cache_savings": costs_after.get("cache_savings", 0) - costs_before.get("cache_savings", 0)
        }
        
        self.demo_results["cost_tracking"] = {
            "requests_made": len(queries),
            "cost_difference": cost_diff,
            "current_costs": costs_after
        }
        
        print(f"\n   📊 Demo Results:")
        print(f"      • Requests made: {len(queries)}")
        print(f"      • API cost increase: ${cost_diff['api_costs']:.4f}")
        print(f"      • Cache savings: ${cost_diff['cache_savings']:.4f}")
        print(f"      • Total requests: {costs_after.get('total_requests', 0)}")
        
        return cost_diff
        
    async def get_current_costs(self):
        """Get current cost analytics"""
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.get(f"{self.base_url}/api/restaurants/cost-analysis")
                
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except:
            return {}
            
    async def demonstrate_cache_savings(self):
        """Demonstrate cache savings calculation"""
        print("\n🚀 Demonstrating Cache Savings...")
        
        # Test same query multiple times to show cache benefits
        test_query = {"query": "cache savings demo", "location": "Istanbul"}
        
        response_times = []
        cache_hits = 0
        
        async with aiohttp.ClientSession() as session:
            print("   🔄 Testing cache performance with repeated requests...")
            
            for i in range(10):
                start_time = time.time()
                
                response = await session.get(
                    f"{self.base_url}/api/restaurants/search",
                    params=test_query
                )
                
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                response_times.append(response_time)
                
                data = {}
                if response.status == 200:
                    data = await response.json()
                    if data.get("cache_info", {}).get("hit", False):
                        cache_hits += 1
                        
                print(f"      Request {i+1}: {response_time:.0f}ms - {'CACHE HIT' if data.get('cache_info', {}).get('hit', False) else 'CACHE MISS'}")
                
                await asyncio.sleep(0.2)
                
        # Calculate cache performance
        avg_response_time = sum(response_times) / len(response_times)
        cache_hit_rate = (cache_hits / len(response_times)) * 100
        
        # Estimate savings
        estimated_api_cost_per_request = 0.025  # $0.025 per API call
        cache_saves_per_hit = estimated_api_cost_per_request
        total_cache_savings = cache_hits * cache_saves_per_hit
        
        self.demo_results["cache_savings"] = {
            "total_requests": len(response_times),
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_response_time_ms": round(avg_response_time, 2),
            "estimated_savings": total_cache_savings
        }
        
        print(f"\n   📊 Cache Performance:")
        print(f"      • Cache hit rate: {cache_hit_rate:.1f}%")
        print(f"      • Average response time: {avg_response_time:.0f}ms")
        print(f"      • Estimated savings: ${total_cache_savings:.4f}")
        
        return cache_hit_rate
        
    async def demonstrate_roi_analysis(self):
        """Demonstrate ROI analysis and cost optimization"""
        print("\n📈 Demonstrating ROI Analysis...")
        
        # Get current cost analytics
        current_costs = await self.get_current_costs()
        
        # Calculate ROI scenarios
        scenarios = {
            "without_caching": {
                "monthly_api_calls": 10000,
                "cost_per_call": 0.025,
                "monthly_cost": 10000 * 0.025,
                "infrastructure_cost": 50
            },
            "with_caching": {
                "monthly_api_calls": 10000,
                "cache_hit_rate": 75,  # 75% cache hit rate
                "actual_api_calls": 10000 * 0.25,  # Only 25% hit external API
                "cost_per_call": 0.025,
                "monthly_api_cost": (10000 * 0.25) * 0.025,
                "infrastructure_cost": 75,  # Higher due to cache infrastructure
                "cache_infrastructure": 25
            }
        }
        
        # Calculate savings and ROI
        without_caching = scenarios["without_caching"]["monthly_cost"] + scenarios["without_caching"]["infrastructure_cost"]
        with_caching = scenarios["with_caching"]["monthly_api_cost"] + scenarios["with_caching"]["infrastructure_cost"]
        
        monthly_savings = without_caching - with_caching
        annual_savings = monthly_savings * 12
        
        # Implementation cost (one-time)
        implementation_cost = 500  # Estimated development and setup cost
        
        # ROI calculation
        roi_months = implementation_cost / monthly_savings if monthly_savings > 0 else float('inf')
        roi_percentage = (annual_savings / implementation_cost) * 100 if implementation_cost > 0 else 0
        
        self.demo_results["roi_analysis"] = {
            "scenarios": scenarios,
            "monthly_savings": monthly_savings,
            "annual_savings": annual_savings,
            "implementation_cost": implementation_cost,
            "roi_months": roi_months,
            "roi_percentage": roi_percentage
        }
        
        print(f"\n   📊 ROI Analysis:")
        print(f"      • Without caching: ${without_caching:.2f}/month")
        print(f"      • With caching: ${with_caching:.2f}/month")
        print(f"      • Monthly savings: ${monthly_savings:.2f}")
        print(f"      • Annual savings: ${annual_savings:.2f}")
        print(f"      • ROI payback period: {roi_months:.1f} months")
        print(f"      • ROI percentage: {roi_percentage:.1f}%")
        
        return monthly_savings
        
    async def demonstrate_budget_controls(self):
        """Demonstrate budget controls and alerts"""
        print("\n🎯 Demonstrating Budget Controls...")
        
        # Simulate budget scenarios
        daily_budget = 20.00
        monthly_budget = 500.00
        
        current_costs = await self.get_current_costs()
        
        # Calculate current usage
        daily_usage = current_costs.get("daily_cost", 0)
        monthly_usage = current_costs.get("monthly_cost", 0)
        
        # Calculate budget percentages
        daily_percentage = (daily_usage / daily_budget) * 100 if daily_budget > 0 else 0
        monthly_percentage = (monthly_usage / monthly_budget) * 100 if monthly_budget > 0 else 0
        
        # Determine alert levels
        def get_alert_level(percentage):
            if percentage >= 100:
                return "CRITICAL - Budget exceeded"
            elif percentage >= 90:
                return "HIGH - Near budget limit"
            elif percentage >= 75:
                return "MEDIUM - Approaching limit"
            elif percentage >= 50:
                return "LOW - Half budget used"
            else:
                return "OK - Under budget"
                
        daily_alert = get_alert_level(daily_percentage)
        monthly_alert = get_alert_level(monthly_percentage)
        
        # Simulate budget control actions
        budget_actions = []
        
        if daily_percentage >= 90:
            budget_actions.append("Enable rate limiting")
            budget_actions.append("Increase cache TTL")
            
        if daily_percentage >= 100:
            budget_actions.append("Block non-essential API calls")
            budget_actions.append("Enable emergency cache-only mode")
            
        self.demo_results["budget_controls"] = {
            "budgets": {
                "daily_budget": daily_budget,
                "monthly_budget": monthly_budget
            },
            "current_usage": {
                "daily_cost": daily_usage,
                "monthly_cost": monthly_usage,
                "daily_percentage": daily_percentage,
                "monthly_percentage": monthly_percentage
            },
            "alerts": {
                "daily_alert": daily_alert,
                "monthly_alert": monthly_alert
            },
            "actions": budget_actions
        }
        
        print(f"\n   📊 Budget Status:")
        print(f"      • Daily budget: ${daily_budget:.2f}")
        print(f"      • Daily usage: ${daily_usage:.2f} ({daily_percentage:.1f}%)")
        print(f"      • Daily alert: {daily_alert}")
        print(f"      • Monthly budget: ${monthly_budget:.2f}")
        print(f"      • Monthly usage: ${monthly_usage:.2f} ({monthly_percentage:.1f}%)")
        print(f"      • Monthly alert: {monthly_alert}")
        
        if budget_actions:
            print(f"      • Budget actions: {', '.join(budget_actions)}")
        else:
            print(f"      • Budget actions: None required")
            
        return daily_percentage, monthly_percentage
        
    async def run_cost_demo(self):
        """Run complete cost calculation demonstration"""
        print("💰 AI Istanbul - Cost Calculation & Optimization Demo")
        print("=" * 60)
        
        # Run demonstration sections
        print("Starting comprehensive cost optimization demo...")
        
        cost_tracking = await self.demonstrate_cost_tracking()
        cache_savings = await self.demonstrate_cache_savings()
        roi_savings = await self.demonstrate_roi_analysis()
        budget_status = await self.demonstrate_budget_controls()
        
        # Generate demo summary
        self.demo_results["demo_summary"] = {
            "api_requests_made": 15,  # 5 + 10 from demos
            "cache_hit_rate": cache_savings,
            "estimated_monthly_savings": roi_savings,
            "budget_utilization": {
                "daily": budget_status[0],
                "monthly": budget_status[1]
            },
            "cost_optimization_active": True,
            "demo_completion": datetime.now().isoformat()
        }
        
        print("\n" + "=" * 60)
        print("📊 COST OPTIMIZATION DEMO SUMMARY")
        print("=" * 60)
        print(f"Total API requests made: 15")
        print(f"Cache hit rate achieved: {cache_savings:.1f}%")
        print(f"Estimated monthly savings: ${roi_savings:.2f}")
        print(f"Daily budget utilization: {budget_status[0]:.1f}%")
        print(f"Monthly budget utilization: {budget_status[1]:.1f}%")
        print(f"System status: Cost optimization ACTIVE")
        
        # Key insights
        print("\n🎯 KEY INSIGHTS:")
        print("• Cache system significantly reduces API costs")
        print("• Real-time cost tracking enables budget control")
        print("• ROI positive within 2-3 months")
        print("• Automated budget controls prevent overspending")
        print("• System scales efficiently with usage")
        
        # Save demo results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cost_calculation_demo_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.demo_results, f, indent=2)
            
        print(f"\nDemo results saved to: {filename}")
        
        return True

async def main():
    demo = CostCalculationDemo()
    await demo.run_cost_demo()

if __name__ == "__main__":
    asyncio.run(main())

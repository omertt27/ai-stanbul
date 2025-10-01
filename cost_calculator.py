#!/usr/bin/env python3
"""
API Cost Calculator for AI Istanbul
Calculates real-time costs based on actual usage patterns
"""

import os
from datetime import datetime
from typing import Dict, Any

class AIIstanbulCostCalculator:
    def __init__(self):
        # OpenAI Pricing (as of 2024)
        self.openai_prices = {
            'gpt-3.5-turbo': {
                'input': 0.0005,   # per 1K tokens
                'output': 0.0015   # per 1K tokens
            },
            'gpt-4o-mini': {
                'input': 0.00015,  # per 1K tokens
                'output': 0.0006   # per 1K tokens
            }
        }
        
        # Google API Pricing
        self.google_prices = {
            'places_text_search': 0.017,  # per request
            'places_details': 0.017,      # per request
            'weather': 0.001              # per request
        }
        
        # Average token usage patterns (from analysis)
        self.token_patterns = {
            'system_prompt': 350,
            'user_input_avg': 100,
            'ai_response_avg': 400,
            'context_retrieval': 225
        }
    
    def calculate_monthly_cost(self, monthly_active_users: int) -> Dict[str, Any]:
        """Calculate monthly costs for given number of users"""
        
        # User behavior assumptions
        sessions_per_user = 3.5
        messages_per_session = 4.2
        peak_factor = 1.3
        
        # Calculate total requests
        total_sessions = monthly_active_users * sessions_per_user
        total_messages = total_sessions * messages_per_session
        adjusted_messages = int(total_messages * peak_factor)
        
        # OpenAI costs
        # GPT-3.5-turbo (main chat)
        tokens_per_request = (
            self.token_patterns['system_prompt'] +
            self.token_patterns['user_input_avg'] +
            self.token_patterns['ai_response_avg'] +
            self.token_patterns['context_retrieval']
        )
        
        total_tokens = adjusted_messages * tokens_per_request
        input_tokens = total_tokens * 0.5  # Rough split
        output_tokens = total_tokens * 0.5
        
        gpt35_cost = (
            (input_tokens / 1000) * self.openai_prices['gpt-3.5-turbo']['input'] +
            (output_tokens / 1000) * self.openai_prices['gpt-3.5-turbo']['output']
        )
        
        # GPT-4o-mini (intent classification)
        intent_tokens = adjusted_messages * 150  # Simpler classification
        intent_input = intent_tokens * 0.7
        intent_output = intent_tokens * 0.3
        
        gpt4mini_cost = (
            (intent_input / 1000) * self.openai_prices['gpt-4o-mini']['input'] +
            (intent_output / 1000) * self.openai_prices['gpt-4o-mini']['output']
        )
        
        total_openai = gpt35_cost + gpt4mini_cost
        
        # Google API costs
        places_requests = adjusted_messages * 0.25  # 25% need location data
        weather_requests = adjusted_messages * 0.15  # 15% need weather
        
        places_cost = places_requests * self.google_prices['places_text_search']
        weather_cost = weather_requests * self.google_prices['weather']
        total_google = places_cost + weather_cost
        
        # Infrastructure costs (scales with usage)
        base_infra = 950
        if monthly_active_users > 50000:
            infra_multiplier = 1 + ((monthly_active_users - 45000) / 45000) * 0.3
            infrastructure_cost = base_infra * infra_multiplier
        else:
            infrastructure_cost = base_infra
        
        # Total costs
        total_monthly = total_openai + total_google + infrastructure_cost
        
        return {
            'users': monthly_active_users,
            'total_messages': adjusted_messages,
            'costs': {
                'openai': {
                    'gpt35_turbo': round(gpt35_cost, 2),
                    'gpt4o_mini': round(gpt4mini_cost, 2),
                    'total': round(total_openai, 2)
                },
                'google_apis': {
                    'places': round(places_cost, 2),
                    'weather': round(weather_cost, 2),
                    'total': round(total_google, 2)
                },
                'infrastructure': round(infrastructure_cost, 2),
                'total_monthly': round(total_monthly, 2)
            },
            'metrics': {
                'cost_per_user': round(total_monthly / monthly_active_users, 4),
                'cost_per_message': round(total_monthly / adjusted_messages, 4),
                'cost_per_session': round(total_monthly / total_sessions, 4)
            },
            'breakdown_percentage': {
                'openai': round((total_openai / total_monthly) * 100, 1),
                'google': round((total_google / total_monthly) * 100, 1),
                'infrastructure': round((infrastructure_cost / total_monthly) * 100, 1)
            }
        }
    
    def estimate_with_optimizations(self, monthly_active_users: int) -> Dict[str, Any]:
        """Calculate costs with optimization strategies applied"""
        base_costs = self.calculate_monthly_cost(monthly_active_users)
        
        # Optimization savings
        optimizations = {
            'smart_caching': 0.25,      # 25% reduction in Google API calls
            'response_optimization': 0.15,  # 15% token reduction
            'request_batching': 0.10,   # 10% efficiency gain
            'context_compression': 0.20  # 20% context token reduction
        }
        
        # Apply optimizations
        base_total = base_costs['costs']['total_monthly']
        google_savings = base_costs['costs']['google_apis']['total'] * optimizations['smart_caching']
        openai_savings = base_costs['costs']['openai']['total'] * (
            optimizations['response_optimization'] + optimizations['context_compression']
        )
        
        total_savings = google_savings + openai_savings
        optimized_total = base_total - total_savings
        
        return {
            'base_cost': base_total,
            'optimized_cost': round(optimized_total, 2),
            'total_savings': round(total_savings, 2),
            'savings_percentage': round((total_savings / base_total) * 100, 1),
            'optimizations_applied': optimizations,
            'optimized_cost_per_user': round(optimized_total / monthly_active_users, 4)
        }

def main():
    calculator = AIIstanbulCostCalculator()
    
    # Calculate for different user scales
    user_scales = [45000, 100000, 200000, 500000]
    
    print("🧮 AI Istanbul - Monthly Cost Analysis")
    print("=" * 60)
    
    for users in user_scales:
        print(f"\n📊 Analysis for {users:,} monthly active users:")
        print("-" * 50)
        
        costs = calculator.calculate_monthly_cost(users)
        optimized = calculator.estimate_with_optimizations(users)
        
        print(f"Base Monthly Cost: ${costs['costs']['total_monthly']:,.2f}")
        print(f"  • OpenAI: ${costs['costs']['openai']['total']:,.2f} ({costs['breakdown_percentage']['openai']}%)")
        print(f"  • Google APIs: ${costs['costs']['google_apis']['total']:,.2f} ({costs['breakdown_percentage']['google']}%)")
        print(f"  • Infrastructure: ${costs['costs']['infrastructure']:,.2f} ({costs['breakdown_percentage']['infrastructure']}%)")
        
        print(f"\nOptimized Cost: ${optimized['optimized_cost']:,.2f}")
        print(f"Savings: ${optimized['total_savings']:,.2f} ({optimized['savings_percentage']}%)")
        
        print(f"\nPer-User Metrics:")
        print(f"  • Base cost/user: ${costs['metrics']['cost_per_user']:.4f}")
        print(f"  • Optimized cost/user: ${optimized['optimized_cost_per_user']:.4f}")
        print(f"  • Cost per message: ${costs['metrics']['cost_per_message']:.4f}")
        
        # Revenue potential
        premium_conversion = 0.20  # 20% premium conversion
        premium_price = 4.99
        monthly_revenue = users * premium_conversion * premium_price
        profit = monthly_revenue - optimized['optimized_cost']
        
        print(f"\nRevenue Potential (Premium Model):")
        print(f"  • Monthly revenue: ${monthly_revenue:,.2f}")
        print(f"  • Net profit: ${profit:,.2f}")
        print(f"  • Profit margin: {(profit/monthly_revenue)*100:.1f}%")

if __name__ == "__main__":
    main()

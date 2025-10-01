#!/usr/bin/env python3
"""
Cost Monitoring Dashboard for AI Istanbul
Tracks API usage, cost savings, and optimization metrics in real-time
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CostMetrics:
    """Cost tracking metrics"""
    timestamp: datetime
    openai_requests: int
    openai_tokens_used: int
    google_places_requests: int
    google_weather_requests: int
    cached_responses: int
    cost_savings: float
    total_estimated_cost: float
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CostMetrics':
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class CostMonitor:
    """Real-time cost monitoring and analytics"""
    
    def __init__(self, storage_path: str = "cost_metrics"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Current session metrics
        self.session_metrics = {
            'session_start': datetime.now(),
            'openai_requests': 0,
            'openai_tokens_used': 0,
            'google_places_requests': 0,
            'google_weather_requests': 0,
            'cached_responses': 0,
            'cost_savings': 0.0,
            'total_estimated_cost': 0.0
        }
        
        # Cost rates (as of 2024)
        self.cost_rates = {
            'openai_gpt35_input': 0.0005,    # per 1K tokens
            'openai_gpt35_output': 0.0015,   # per 1K tokens
            'openai_gpt4mini_input': 0.00015, # per 1K tokens
            'openai_gpt4mini_output': 0.0006, # per 1K tokens
            'google_places': 0.017,          # per request
            'google_weather': 0.001          # per request
        }
        
        logger.info(f"🔍 Cost monitor initialized (Storage: {self.storage_path})")
    
    def log_openai_request(self, model: str, tokens_used: int, cached: bool = False):
        """Log OpenAI API request"""
        self.session_metrics['openai_requests'] += 1
        
        if not cached:
            self.session_metrics['openai_tokens_used'] += tokens_used
            
            # Calculate cost based on model
            if 'gpt-3.5' in model:
                # Rough split: 40% input, 60% output
                input_tokens = tokens_used * 0.4
                output_tokens = tokens_used * 0.6
                cost = (input_tokens/1000 * self.cost_rates['openai_gpt35_input'] + 
                       output_tokens/1000 * self.cost_rates['openai_gpt35_output'])
            elif 'gpt-4' in model and 'mini' in model:
                input_tokens = tokens_used * 0.4
                output_tokens = tokens_used * 0.6
                cost = (input_tokens/1000 * self.cost_rates['openai_gpt4mini_input'] + 
                       output_tokens/1000 * self.cost_rates['openai_gpt4mini_output'])
            else:
                cost = tokens_used/1000 * 0.001  # Fallback rate
            
            self.session_metrics['total_estimated_cost'] += cost
            
            logger.debug(f"💰 OpenAI cost: ${cost:.4f} ({tokens_used} tokens, {model})")
        else:
            # Track cost savings from cache (higher savings for preloaded responses)
            if 'preloaded' in model.lower():
                # Preloaded responses save maximum cost (no API call needed)
                estimated_cost = tokens_used/1000 * 0.002  # Higher estimated savings
            else:
                estimated_cost = tokens_used/1000 * 0.0015  # Regular cache savings
            
            self.session_metrics['cost_savings'] += estimated_cost
            self.session_metrics['cached_responses'] += 1
            
            cache_type = "preloaded" if 'preloaded' in model.lower() else "cached"
            logger.debug(f"💰 Cost saved: ${estimated_cost:.4f} ({cache_type} response)")
    
    def log_google_places_request(self, cached: bool = False):
        """Log Google Places API request"""
        if not cached:
            self.session_metrics['google_places_requests'] += 1
            cost = self.cost_rates['google_places']
            self.session_metrics['total_estimated_cost'] += cost
            
            logger.debug(f"💰 Google Places cost: ${cost:.4f}")
        else:
            # Track cost savings from cache
            self.session_metrics['cost_savings'] += self.cost_rates['google_places']
            self.session_metrics['cached_responses'] += 1
            
            logger.debug(f"💰 Cost saved: ${self.cost_rates['google_places']:.4f} (cached places)")
    
    def log_google_weather_request(self, cached: bool = False):
        """Log Google Weather API request"""
        if not cached:
            self.session_metrics['google_weather_requests'] += 1
            cost = self.cost_rates['google_weather']
            self.session_metrics['total_estimated_cost'] += cost
            
            logger.debug(f"💰 Google Weather cost: ${cost:.4f}")
        else:
            self.session_metrics['cost_savings'] += self.cost_rates['google_weather']
            self.session_metrics['cached_responses'] += 1
            
            logger.debug(f"💰 Cost saved: ${self.cost_rates['google_weather']:.4f} (cached weather)")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session cost summary"""
        duration = datetime.now() - self.session_metrics['session_start']
        
        # Calculate savings percentage
        total_without_cache = self.session_metrics['total_estimated_cost'] + self.session_metrics['cost_savings']
        savings_percentage = (self.session_metrics['cost_savings'] / max(total_without_cache, 0.01)) * 100
        
        return {
            'session_duration_minutes': duration.total_seconds() / 60,
            'total_requests': (self.session_metrics['openai_requests'] + 
                             self.session_metrics['google_places_requests'] + 
                             self.session_metrics['google_weather_requests']),
            'cached_responses': self.session_metrics['cached_responses'],
            'cache_hit_ratio': (self.session_metrics['cached_responses'] / 
                              max(self.session_metrics['openai_requests'], 1)) * 100,
            'total_cost': round(self.session_metrics['total_estimated_cost'], 4),
            'cost_savings': round(self.session_metrics['cost_savings'], 4),
            'savings_percentage': round(savings_percentage, 1),
            'tokens_used': self.session_metrics['openai_tokens_used'],
            'cost_per_request': round(self.session_metrics['total_estimated_cost'] / 
                                    max(self.session_metrics['openai_requests'], 1), 4)
        }
    
    def save_daily_metrics(self):
        """Save daily metrics to file"""
        today = datetime.now().strftime('%Y-%m-%d')
        metrics_file = self.storage_path / f"daily_metrics_{today}.json"
        
        metrics = CostMetrics(
            timestamp=datetime.now(),
            openai_requests=self.session_metrics['openai_requests'],
            openai_tokens_used=self.session_metrics['openai_tokens_used'],
            google_places_requests=self.session_metrics['google_places_requests'],
            google_weather_requests=self.session_metrics['google_weather_requests'],
            cached_responses=self.session_metrics['cached_responses'],
            cost_savings=self.session_metrics['cost_savings'],
            total_estimated_cost=self.session_metrics['total_estimated_cost']
        )
        
        try:
            # Load existing metrics if file exists
            daily_metrics = []
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    daily_metrics = [CostMetrics.from_dict(item) for item in data]
            
            # Add current metrics
            daily_metrics.append(metrics)
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump([m.to_dict() for m in daily_metrics], f, indent=2)
            
            logger.info(f"📊 Daily metrics saved: {metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to save daily metrics: {e}")
    
    def get_daily_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily cost report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        metrics_file = self.storage_path / f"daily_metrics_{date}.json"
        
        if not metrics_file.exists():
            return {'error': f'No metrics found for {date}'}
        
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                daily_metrics = [CostMetrics.from_dict(item) for item in data]
            
            # Aggregate metrics
            total_requests = sum(m.openai_requests + m.google_places_requests + m.google_weather_requests for m in daily_metrics)
            total_tokens = sum(m.openai_tokens_used for m in daily_metrics)
            total_cost = sum(m.total_estimated_cost for m in daily_metrics)
            total_savings = sum(m.cost_savings for m in daily_metrics)
            total_cached = sum(m.cached_responses for m in daily_metrics)
            
            return {
                'date': date,
                'total_requests': total_requests,
                'total_tokens': total_tokens,
                'total_cost': round(total_cost, 4),
                'cost_savings': round(total_savings, 4),
                'savings_percentage': round((total_savings / max(total_cost + total_savings, 0.01)) * 100, 1),
                'cached_responses': total_cached,
                'cache_hit_ratio': round((total_cached / max(total_requests, 1)) * 100, 1),
                'cost_per_request': round(total_cost / max(total_requests, 1), 4),
                'entries_count': len(daily_metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to load daily report: {e}")
            return {'error': str(e)}
    
    def print_realtime_dashboard(self):
        """Print real-time cost dashboard"""
        summary = self.get_session_summary()
        
        print("\n" + "="*60)
        print("🎯 AI ISTANBUL - REAL-TIME COST DASHBOARD")
        print("="*60)
        print(f"Session Duration: {summary['session_duration_minutes']:.1f} minutes")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Cached Responses: {summary['cached_responses']}")
        print(f"Cache Hit Ratio: {summary['cache_hit_ratio']:.1f}%")
        print("-"*60)
        print(f"💰 Current Cost: ${summary['total_cost']:.4f}")
        print(f"💚 Cost Savings: ${summary['cost_savings']:.4f}")
        print(f"📊 Savings Rate: {summary['savings_percentage']:.1f}%")
        print(f"🔢 Tokens Used: {summary['tokens_used']:,}")
        print(f"📈 Cost/Request: ${summary['cost_per_request']:.4f}")
        print("="*60)

# Global cost monitor instance
_cost_monitor = None

def get_cost_monitor() -> CostMonitor:
    """Get the global cost monitor instance"""
    global _cost_monitor
    if _cost_monitor is None:
        _cost_monitor = CostMonitor()
    return _cost_monitor

def log_openai_cost(model: str, tokens: int, cached: bool = False):
    """Log OpenAI API cost"""
    monitor = get_cost_monitor()
    monitor.log_openai_request(model, tokens, cached)

def log_google_places_cost(cached: bool = False):
    """Log Google Places API cost"""
    monitor = get_cost_monitor()
    monitor.log_google_places_request(cached)

def log_google_weather_cost(cached: bool = False):
    """Log Google Weather API cost"""
    monitor = get_cost_monitor()
    monitor.log_google_weather_request(cached)

def print_cost_dashboard():
    """Print cost dashboard"""
    monitor = get_cost_monitor()
    monitor.print_realtime_dashboard()

if __name__ == "__main__":
    # Test the cost monitoring system
    monitor = CostMonitor()
    
    print("🧪 Testing Cost Monitoring System...")
    
    # Simulate some API calls
    monitor.log_openai_request("gpt-3.5-turbo", 500, cached=False)
    monitor.log_openai_request("gpt-3.5-turbo", 300, cached=True)  # Cached
    monitor.log_google_places_request(cached=False)
    monitor.log_google_places_request(cached=True)  # Cached
    
    # Print dashboard
    monitor.print_realtime_dashboard()
    
    # Save daily metrics
    monitor.save_daily_metrics()
    
    print("✅ Cost monitoring system ready!")

#!/usr/bin/env python3
"""
AI Istanbul Development Server Startup Script
===========================================

Starts the development server with integrated cache system,
monitoring, and TTL optimization.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print startup banner"""
    print("=" * 80)
    print("🚀 AI Istanbul - Enhanced Development Server")
    print("   Integrated Cache System with Production Monitoring")
    print("=" * 80)
    print(f"🕐 Starting at: {datetime.now().isoformat()}")
    print()

def check_environment():
    """Check required environment variables"""
    required_vars = [
        'GOOGLE_PLACES_API_KEY',
        'OPENAI_API_KEY'
    ]
    
    optional_vars = [
        'REDIS_URL',
        'DATABASE_URL'
    ]
    
    print("🔍 Environment Check:")
    
    missing_required = []
    for var in required_vars:
        if os.getenv(var):
            print(f"  ✅ {var}: configured")
        else:
            print(f"  ❌ {var}: missing (required)")
            missing_required.append(var)
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"  ✅ {var}: configured")
        else:
            print(f"  ⚠️  {var}: not set (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required environment variables: {', '.join(missing_required)}")
        print("   Please set these variables in your .env file or environment")
        return False
    
    print("✅ Environment check passed")
    return True

def check_redis_connection():
    """Check Redis connection"""
    try:
        import redis
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        client = redis.from_url(redis_url)
        client.ping()
        print("✅ Redis: connected")
        return True
    except Exception as e:
        print(f"⚠️  Redis: not available ({e})")
        print("   Cache system will use fallback mode")
        return False

def check_components():
    """Check availability of system components"""
    print("\n🧩 Component Check:")
    
    components = [
        ('google_api_integration', 'Google API Integration'),
        ('integrated_cache_system', 'Integrated Cache System'),
        ('ttl_fine_tuning', 'TTL Fine-Tuning'),
        ('unified_ai_system', 'Unified AI System'),
        ('routes.cache_monitoring', 'Cache Monitoring Routes')
    ]
    
    available_components = []
    
    for module_name, display_name in components:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}: available")
            available_components.append(module_name)
        except ImportError as e:
            print(f"  ❌ {display_name}: not available ({e})")
    
    print(f"✅ {len(available_components)}/{len(components)} components available")
    return available_components

async def initialize_cache_system():
    """Initialize the integrated cache system"""
    try:
        from integrated_cache_system import integrated_cache_system
        
        print("\n🔥 Initializing Cache System:")
        print("  📊 Starting performance monitoring...")
        print("  🔧 Loading TTL optimization settings...")
        
        # Warm some popular queries
        popular_queries = [
            "best Turkish restaurants in Sultanahmet",
            "vegetarian restaurants in Beyoğlu"
        ]
        
        print("  🔥 Warming cache with popular queries...")
        for query in popular_queries:
            try:
                await integrated_cache_system.warm_cache_for_query(query)
                print(f"    ✅ Warmed: {query[:40]}...")
            except Exception as e:
                print(f"    ⚠️  Failed to warm: {query[:40]}... ({e})")
        
        print("✅ Cache system initialized")
        return True
        
    except ImportError:
        print("⚠️  Integrated cache system not available")
        return False
    except Exception as e:
        print(f"❌ Error initializing cache system: {e}")
        return False

def show_endpoints():
    """Show available endpoints"""
    print("\n🌐 Available Endpoints:")
    print("  Main Application:")
    print("    http://localhost:8000                    - API Root")
    print("    http://localhost:8000/docs               - API Documentation")
    print("    http://localhost:8000/restaurants/search - Restaurant Search (Original)")
    print("    http://localhost:8000/restaurants/enhanced-search - Enhanced Search (With Cache)")
    print()
    print("  Cache Monitoring:")
    print("    http://localhost:8000/api/cache/dashboard    - Monitoring Dashboard")
    print("    http://localhost:8000/api/cache/analytics    - Cache Analytics")
    print("    http://localhost:8000/api/cache/performance  - Performance Metrics")
    print("    http://localhost:8000/api/cache/health       - System Health")
    print()
    print("  TTL Optimization:")
    print("    http://localhost:8000/api/cache/ttl/report   - TTL Optimization Report")
    print("    http://localhost:8000/api/cache/ttl/current  - Current TTL Values")
    print()
    print("  Cache Management:")
    print("    POST http://localhost:8000/api/cache/warm/popular - Warm Popular Queries")
    print("    POST http://localhost:8000/api/cache/ttl/optimize - Force TTL Optimization")

def main():
    """Main startup function"""
    print_banner()
    
    # Environment check
    if not check_environment():
        sys.exit(1)
    
    # Redis check
    redis_available = check_redis_connection()
    
    # Component check
    available_components = check_components()
    
    # Initialize cache system
    if 'integrated_cache_system' in available_components:
        asyncio.run(initialize_cache_system())
    
    # Show endpoints
    show_endpoints()
    
    print("\n🚀 Starting server...")
    print("   Press Ctrl+C to stop")
    print("=" * 80)
    
    # Start the server
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cache Performance Monitoring Routes
==================================

FastAPI routes for monitoring cache performance and system analytics
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

# Import integrated cache system
try:
    from integrated_cache_system import (
        get_integrated_analytics,
        warm_popular_query,
        integrated_cache_system
    )
    INTEGRATED_CACHE_AVAILABLE = True
except ImportError:
    INTEGRATED_CACHE_AVAILABLE = False

# Import unified AI system
try:
    from unified_ai_system import get_unified_ai_system
    from models import get_db
    UNIFIED_AI_AVAILABLE = True
except ImportError:
    UNIFIED_AI_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/cache",
    tags=["cache-monitoring"],
    responses={404: {"description": "Not found"}},
)

@router.get("/analytics")
async def get_cache_analytics():
    """Get comprehensive cache performance analytics"""
    try:
        if not INTEGRATED_CACHE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Integrated cache system not available")
        
        analytics = get_integrated_analytics()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "analytics": analytics
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting cache analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_cache_performance():
    """Get real-time cache performance metrics"""
    try:
        if not INTEGRATED_CACHE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Integrated cache system not available")
        
        analytics = get_integrated_analytics()
        monitoring_data = analytics.get('production_monitoring', {})
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "hit_rate_percent": monitoring_data.get('metrics', {}).get('hit_rate_percent', 0),
                "total_requests": monitoring_data.get('metrics', {}).get('total_requests', 0),
                "cache_hits": monitoring_data.get('metrics', {}).get('cache_hits', 0),
                "cache_misses": monitoring_data.get('metrics', {}).get('cache_misses', 0),
                "average_response_time_ms": monitoring_data.get('metrics', {}).get('average_response_time_ms', 0),
                "cost_savings_usd": monitoring_data.get('metrics', {}).get('cost_savings_usd', 0),
                "cache_size_mb": monitoring_data.get('metrics', {}).get('cache_size_mb', 0),
                "error_count": monitoring_data.get('metrics', {}).get('error_count', 0)
            },
            "alert_status": monitoring_data.get('alert_status', {}),
            "system_health": analytics.get('system_health', {})
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting cache performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/popular-queries")
async def get_popular_queries():
    """Get popular cached queries"""
    try:
        if not INTEGRATED_CACHE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Integrated cache system not available")
        
        analytics = get_integrated_analytics()
        monitoring_data = analytics.get('production_monitoring', {})
        popular_queries = monitoring_data.get('metrics', {}).get('popular_queries', [])
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "popular_queries": popular_queries,
            "total_unique_queries": len(popular_queries)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting popular queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/warm")
async def warm_cache_query(
    query: str = Query(..., description="Query to warm in cache"),
    location: str = Query("Istanbul, Turkey", description="Location context")
):
    """Warm cache for a specific query"""
    try:
        if not INTEGRATED_CACHE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Integrated cache system not available")
        
        success = await warm_popular_query(query, location)
        
        return {
            "success": success,
            "message": f"Cache warming {'completed' if success else 'failed'} for query: {query}",
            "query": query,
            "location": location,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error warming cache for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/warm/popular")
async def warm_popular_queries(db = Depends(get_db) if UNIFIED_AI_AVAILABLE else None):
    """Warm cache for popular queries"""
    try:
        if not INTEGRATED_CACHE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Integrated cache system not available")
        
        if UNIFIED_AI_AVAILABLE and db:
            # Use unified AI system to get popular queries and warm them
            ai_system = get_unified_ai_system(db)
            result = await ai_system.warm_cache_for_popular_queries()
        else:
            # Fallback: warm predefined popular queries
            popular_queries = [
                "best Turkish restaurants in Sultanahmet",
                "seafood restaurants near Bosphorus", 
                "vegetarian restaurants in Beyoğlu",
                "traditional Turkish breakfast places",
                "rooftop restaurants with view"
            ]
            
            warming_results = []
            for query in popular_queries:
                success = await warm_popular_query(query, "Istanbul, Turkey")
                warming_results.append({
                    'query': query,
                    'success': success
                })
            
            successful_warming = sum(1 for r in warming_results if r['success'])
            result = {
                'success': True,
                'queries_warmed': successful_warming,
                'total_queries': len(popular_queries),
                'warming_results': warming_results
            }
        
        return {
            **result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error warming popular queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization-history")
async def get_optimization_history():
    """Get cache optimization history"""
    try:
        if not INTEGRATED_CACHE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Integrated cache system not available")
        
        analytics = get_integrated_analytics()
        monitoring_data = analytics.get('production_monitoring', {})
        optimization_history = monitoring_data.get('optimization_history', [])
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "optimization_history": optimization_history,
            "total_optimizations": len(optimization_history)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting optimization history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/errors")
async def get_cache_errors():
    """Get recent cache errors"""
    try:
        if not INTEGRATED_CACHE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Integrated cache system not available")
        
        analytics = get_integrated_analytics()
        monitoring_data = analytics.get('production_monitoring', {})
        recent_errors = monitoring_data.get('recent_errors', [])
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "recent_errors": recent_errors,
            "error_count": len(recent_errors)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting cache errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard", response_class=HTMLResponse)
async def cache_monitoring_dashboard():
    """Serve cache monitoring dashboard HTML"""
    if not INTEGRATED_CACHE_AVAILABLE:
        return HTMLResponse("""
        <html>
            <head><title>Cache Monitoring - Unavailable</title></head>
            <body>
                <h1>Cache Monitoring Dashboard</h1>
                <p style="color: red;">Integrated cache system is not available.</p>
            </body>
        </html>
        """)
    
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Istanbul - Cache Performance Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 1400px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px; 
                padding: 30px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .header { 
                text-align: center; 
                margin-bottom: 40px; 
                padding-bottom: 20px; 
                border-bottom: 3px solid #667eea;
            }
            .header h1 { 
                color: #2c3e50; 
                margin: 0; 
                font-size: 2.5em; 
            }
            .header p { 
                color: #7f8c8d; 
                margin: 10px 0 0 0; 
                font-size: 1.1em; 
            }
            .metrics-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 20px; 
                margin-bottom: 40px; 
            }
            .metric-card { 
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                color: white; 
                padding: 25px; 
                border-radius: 12px; 
                text-align: center; 
                box-shadow: 0 8px 20px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
            }
            .metric-card:hover { 
                transform: translateY(-5px); 
            }
            .metric-value { 
                font-size: 2.5em; 
                font-weight: bold; 
                margin-bottom: 10px; 
            }
            .metric-label { 
                font-size: 1.1em; 
                opacity: 0.9; 
            }
            .section { 
                margin: 40px 0; 
                padding: 25px; 
                background: #f8f9fa; 
                border-radius: 12px; 
                border-left: 5px solid #667eea;
            }
            .section h2 { 
                color: #2c3e50; 
                margin-top: 0; 
                font-size: 1.8em; 
            }
            .btn { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                border: none; 
                padding: 12px 25px; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 1em; 
                margin: 5px; 
                transition: all 0.3s ease;
            }
            .btn:hover { 
                transform: translateY(-2px); 
                box-shadow: 0 5px 15px rgba(0,0,0,0.2); 
            }
            .status-good { 
                color: #27ae60; 
                font-weight: bold; 
            }
            .status-warning { 
                color: #f39c12; 
                font-weight: bold; 
            }
            .status-error { 
                color: #e74c3c; 
                font-weight: bold; 
            }
            #loading { 
                text-align: center; 
                padding: 40px; 
                color: #7f8c8d; 
                font-size: 1.2em; 
            }
            .query-list { 
                max-height: 300px; 
                overflow-y: auto; 
                background: white; 
                border-radius: 8px; 
                padding: 15px; 
            }
            .query-item { 
                padding: 10px; 
                border-bottom: 1px solid #ecf0f1; 
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 AI Istanbul Cache Performance Dashboard</h1>
                <p>Real-time monitoring and optimization analytics</p>
            </div>
            
            <div id="loading">Loading cache analytics...</div>
            
            <div id="dashboard-content" style="display: none;">
                <div class="metrics-grid" id="metrics-grid">
                    <!-- Metrics cards will be populated by JavaScript -->
                </div>
                
                <div class="section">
                    <h2>🎯 Cache Actions</h2>
                    <button class="btn" onclick="warmPopularQueries()">🔥 Warm Popular Queries</button>
                    <button class="btn" onclick="refreshDashboard()">🔄 Refresh Dashboard</button>
                    <button class="btn" onclick="downloadAnalytics()">📊 Download Analytics</button>
                </div>
                
                <div class="section">
                    <h2>📈 Popular Queries</h2>
                    <div id="popular-queries" class="query-list">
                        <!-- Popular queries will be populated by JavaScript -->
                    </div>
                </div>
                
                <div class="section">
                    <h2>⚡ System Health</h2>
                    <div id="system-health">
                        <!-- System health will be populated by JavaScript -->
                    </div>
                </div>
                
                <div class="section">
                    <h2>🚨 Recent Errors</h2>
                    <div id="recent-errors">
                        <!-- Recent errors will be populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>

        <script>
            let dashboardData = null;
            
            async function loadDashboard() {
                try {
                    const response = await fetch('/api/cache/analytics');
                    const data = await response.json();
                    
                    if (data.success) {
                        dashboardData = data.analytics;
                        renderDashboard();
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('dashboard-content').style.display = 'block';
                    } else {
                        throw new Error(data.error || 'Failed to load analytics');
                    }
                } catch (error) {
                    document.getElementById('loading').innerHTML = 
                        `<div style="color: #e74c3c;">❌ Error loading dashboard: ${error.message}</div>`;
                }
            }
            
            function renderDashboard() {
                if (!dashboardData) return;
                
                renderMetrics();
                renderPopularQueries();
                renderSystemHealth();
                renderRecentErrors();
            }
            
            function renderMetrics() {
                const monitoring = dashboardData.production_monitoring?.metrics || {};
                const cacheAnalytics = dashboardData.time_aware_cache || {};
                
                const metrics = [
                    {
                        value: (monitoring.hit_rate_percent || 0).toFixed(1) + '%',
                        label: 'Cache Hit Rate',
                        color: 'linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%)'
                    },
                    {
                        value: monitoring.total_requests || 0,
                        label: 'Total Requests',
                        color: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                    },
                    {
                        value: '$' + (monitoring.cost_savings_usd || 0).toFixed(2),
                        label: 'Cost Savings',
                        color: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
                    },
                    {
                        value: (monitoring.average_response_time_ms || 0).toFixed(0) + 'ms',
                        label: 'Avg Response Time',
                        color: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
                    }
                ];
                
                const metricsGrid = document.getElementById('metrics-grid');
                metricsGrid.innerHTML = metrics.map(metric => `
                    <div class="metric-card" style="background: ${metric.color};">
                        <div class="metric-value">${metric.value}</div>
                        <div class="metric-label">${metric.label}</div>
                    </div>
                `).join('');
            }
            
            function renderPopularQueries() {
                const popularQueries = dashboardData.production_monitoring?.metrics?.popular_queries || [];
                const queriesContainer = document.getElementById('popular-queries');
                
                if (popularQueries.length === 0) {
                    queriesContainer.innerHTML = '<p>No popular queries data available</p>';
                    return;
                }
                
                queriesContainer.innerHTML = popularQueries.map(item => `
                    <div class="query-item">
                        <span>${item.query}</span>
                        <span style="background: #3498db; color: white; padding: 2px 8px; border-radius: 12px;">
                            ${item.frequency}
                        </span>
                    </div>
                `).join('');
            }
            
            function renderSystemHealth() {
                const health = dashboardData.system_health || {};
                const alerts = dashboardData.production_monitoring?.alert_status || {};
                
                const healthItems = [
                    { label: 'Cache Manager', status: health.cache_manager_connected, key: 'cache' },
                    { label: 'Monitor Connected', status: health.monitor_connected, key: 'monitor' },
                    { label: 'Cache Warming', status: health.warming_enabled, key: 'warming' },
                    { label: 'Hit Rate OK', status: alerts.hit_rate_ok, key: 'hitrate' },
                    { label: 'Error Rate OK', status: alerts.error_rate_ok, key: 'errorrate' },
                    { label: 'Response Time OK', status: alerts.response_time_ok, key: 'responsetime' }
                ];
                
                const healthContainer = document.getElementById('system-health');
                healthContainer.innerHTML = healthItems.map(item => {
                    const statusClass = item.status ? 'status-good' : 'status-error';
                    const statusIcon = item.status ? '✅' : '❌';
                    return `
                        <div style="padding: 8px 0; display: flex; justify-content: space-between;">
                            <span>${item.label}</span>
                            <span class="${statusClass}">${statusIcon} ${item.status ? 'OK' : 'ISSUE'}</span>
                        </div>
                    `;
                }).join('');
            }
            
            function renderRecentErrors() {
                const errors = dashboardData.production_monitoring?.recent_errors || [];
                const errorsContainer = document.getElementById('recent-errors');
                
                if (errors.length === 0) {
                    errorsContainer.innerHTML = '<p style="color: #27ae60;">✅ No recent errors</p>';
                    return;
                }
                
                errorsContainer.innerHTML = errors.map(error => `
                    <div style="padding: 10px; margin: 5px 0; background: #fff5f5; border-left: 4px solid #e74c3c; border-radius: 4px;">
                        <strong>Error:</strong> ${error.error}<br>
                        <small style="color: #7f8c8d;">Time: ${error.timestamp}</small>
                    </div>
                `).join('');
            }
            
            async function warmPopularQueries() {
                try {
                    const button = event.target;
                    button.disabled = true;
                    button.innerHTML = '⏳ Warming...';
                    
                    const response = await fetch('/api/cache/warm/popular', { method: 'POST' });
                    const data = await response.json();
                    
                    if (data.success) {
                        alert(`✅ Successfully warmed ${data.queries_warmed}/${data.total_queries} queries`);
                        refreshDashboard();
                    } else {
                        throw new Error(data.error || 'Failed to warm queries');
                    }
                } catch (error) {
                    alert(`❌ Error warming queries: ${error.message}`);
                } finally {
                    const button = event.target;
                    button.disabled = false;
                    button.innerHTML = '🔥 Warm Popular Queries';
                }
            }
            
            async function refreshDashboard() {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('dashboard-content').style.display = 'none';
                await loadDashboard();
            }
            
            function downloadAnalytics() {
                if (!dashboardData) return;
                
                const dataStr = JSON.stringify(dashboardData, null, 2);
                const dataBlob = new Blob([dataStr], {type: 'application/json'});
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `cache-analytics-${new Date().toISOString().split('T')[0]}.json`;
                link.click();
                URL.revokeObjectURL(url);
            }
            
            // Load dashboard on page load
            window.addEventListener('load', loadDashboard);
            
            // Auto-refresh every 30 seconds
            setInterval(refreshDashboard, 30000);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(dashboard_html)

@router.get("/ttl/report")
async def get_ttl_optimization_report():
    """Get TTL optimization report with detailed analytics"""
    try:
        # Import TTL optimization system
        try:
            from ttl_fine_tuning import get_ttl_optimization_report
            report = get_ttl_optimization_report()
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "ttl_optimization": report
            }
            
        except ImportError:
            raise HTTPException(status_code=503, detail="TTL optimization system not available")
        
    except Exception as e:
        logger.error(f"❌ Error getting TTL optimization report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ttl/optimize")
async def force_ttl_optimization():
    """Force TTL optimization for all cache types"""
    try:
        # Import TTL optimization system
        try:
            from ttl_fine_tuning import force_ttl_optimization
            result = force_ttl_optimization()
            
            return {
                "success": True,
                "message": f"TTL optimization completed for {result['total_optimized']} cache types",
                "timestamp": datetime.now().isoformat(),
                "optimization_results": result
            }
            
        except ImportError:
            raise HTTPException(status_code=503, detail="TTL optimization system not available")
        
    except Exception as e:
        logger.error(f"❌ Error forcing TTL optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ttl/current")
async def get_current_ttl_values():
    """Get current TTL values for all cache types"""
    try:
        # Import TTL optimization system
        try:
            from ttl_fine_tuning import ttl_optimizer
            
            current_ttls = {}
            for cache_type in ttl_optimizer.ttl_configs.keys():
                current_ttls[cache_type] = {
                    'current_ttl_seconds': ttl_optimizer.get_optimized_ttl(cache_type),
                    'base_ttl_seconds': ttl_optimizer.ttl_configs[cache_type].base_ttl_seconds,
                    'current_multiplier': ttl_optimizer.ttl_configs[cache_type].current_multiplier,
                    'min_ttl_seconds': ttl_optimizer.ttl_configs[cache_type].min_ttl_seconds,
                    'max_ttl_seconds': ttl_optimizer.ttl_configs[cache_type].max_ttl_seconds
                }
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "current_ttl_values": current_ttls
            }
            
        except ImportError:
            raise HTTPException(status_code=503, detail="TTL optimization system not available")
        
    except Exception as e:
        logger.error(f"❌ Error getting current TTL values: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def cache_system_health():
    """Get cache system health status"""
    try:
        health_status = {
            "integrated_cache_available": INTEGRATED_CACHE_AVAILABLE,
            "unified_ai_available": UNIFIED_AI_AVAILABLE,
            "timestamp": datetime.now().isoformat()
        }
        
        if INTEGRATED_CACHE_AVAILABLE:
            analytics = get_integrated_analytics()
            health_status.update({
                "system_health": analytics.get('system_health', {}),
                "integration_status": analytics.get('integration_status', {})
            })
        
        return {
            "success": True,
            "health": health_status
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

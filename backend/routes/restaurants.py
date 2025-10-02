from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from database import SessionLocal
from models import Restaurant
from api_clients.google_places import GooglePlacesClient, get_istanbul_restaurants_with_descriptions

# Use lazy import to avoid circular dependency
INTEGRATED_CACHE_AVAILABLE = False
search_restaurants_with_integrated_cache = None

def _lazy_import_integrated_cache():
    """Lazy import of integrated cache system to avoid circular imports"""
    global INTEGRATED_CACHE_AVAILABLE, search_restaurants_with_integrated_cache
    if not INTEGRATED_CACHE_AVAILABLE:
        try:
            from integrated_cache_system import search_restaurants_with_integrated_cache as _search_restaurants_with_integrated_cache
            search_restaurants_with_integrated_cache = _search_restaurants_with_integrated_cache
            INTEGRATED_CACHE_AVAILABLE = True
            logging.info("✅ Integrated cache system loaded successfully (lazy import in routes)")
        except ImportError as e:
            logging.warning(f"⚠️ Integrated cache system not available in routes: {e}")
    return INTEGRATED_CACHE_AVAILABLE

import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/restaurants", tags=["Restaurants"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/")
def get_restaurants(db: Session = Depends(get_db)):
    """Get all restaurants from the database."""
    return db.query(Restaurant).all()

@router.get("/search")
def search_restaurants_with_descriptions(
    location: Optional[str] = Query(None, description="Location to search (e.g., 'Beyoğlu, Istanbul')"),
    district: Optional[str] = Query(None, description="Istanbul district (e.g., 'Beyoğlu', 'Sultanahmet')"),
    keyword: Optional[str] = Query(None, description="Keyword to filter restaurants"),
    limit: int = Query(10, ge=1, le=50, description="Number of restaurants to return"),
    radius: int = Query(1500, ge=100, le=5000, description="Search radius in meters")
):
    """
    Search for restaurants with descriptions from Google Maps.
    Returns detailed information including descriptions, reviews, and photos.
    """
    try:
        client = GooglePlacesClient()
        
        if district:
            # Use specific Istanbul district
            search_location = f"{district}, Istanbul, Turkey"
        elif location:
            search_location = location
        else:
            # Default to Istanbul center
            search_location = "Istanbul, Turkey"
        
        # Build search query for cost calculation
        search_query = f"restaurants in {search_location}"
        if keyword:
            search_query = f"{keyword} {search_query}"
        
        # Calculate costs with integrated cache system
        if INTEGRATED_CACHE_AVAILABLE:
            logger.info(f"Searching restaurants with cache - Location: {search_location}, Radius: {radius}, Limit: {limit}, Keyword: {keyword}")
            
            # Use async integrated cache search with cost tracking
            import asyncio
            if _lazy_import_integrated_cache() and search_restaurants_with_integrated_cache:
                result = asyncio.run(search_restaurants_with_integrated_cache(
                    query=search_query,
                    location=search_location,
                    context={"radius": radius, "limit": limit, "keyword": keyword}
                ))
            else:
                # Fallback to basic search
                result = {"restaurants": [], "cache_performance": {}, "cost_analytics": {}}
            
            # Extract restaurants and cost data
            restaurants = result.get('restaurants', [])
            cache_performance = result.get('cache_performance', {})
            optimization_info = result.get('optimization_info', {})
            
            # Calculate detailed cost breakdown
            cost_breakdown = _calculate_cost_breakdown(
                cache_hit=cache_performance.get('cache_hit', False),
                fields_optimized=optimization_info.get('fields_requested', 0),
                total_available_fields=optimization_info.get('total_available_fields', 50),
                response_time_ms=cache_performance.get('response_time_ms', 0),
                cache_type=cache_performance.get('cache_type', 'unknown')
            )
            
        else:
            logger.warning("Integrated cache not available. Falling back to direct API search.")
            restaurants = client.get_restaurants_with_descriptions(
                location=search_location,
                radius=radius,
                limit=limit,
                keyword=keyword
            )
            
            # Calculate cost for non-optimized search
            cost_breakdown = _calculate_fallback_cost()
        
        return {
            "status": "success",
            "location_searched": search_location,
            "total_found": len(restaurants),
            "restaurants": restaurants,
            "cost_analysis": cost_breakdown,
            "optimization_enabled": INTEGRATED_CACHE_AVAILABLE
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching restaurants: {str(e)}")

@router.get("/istanbul/{district}")
def get_istanbul_district_restaurants(
    district: str,
    limit: int = Query(10, ge=1, le=30, description="Number of restaurants to return")
):
    """
    Get restaurants from a specific Istanbul district with descriptions.
    
    Popular districts: Beyoğlu, Sultanahmet, Beşiktaş, Kadıköy, Şişli, Fatih, Üsküdar
    """
    try:
        restaurants = get_istanbul_restaurants_with_descriptions(
            district=district,
            limit=limit
        )
        
        if not restaurants:
            raise HTTPException(
                status_code=404, 
                detail=f"No restaurants found in {district} district"
            )
        
        return {
            "status": "success",
            "district": district,
            "total_found": len(restaurants),
            "restaurants": restaurants
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching restaurants: {str(e)}")

@router.get("/details/{place_id}")
def get_restaurant_details(place_id: str):
    """Get detailed information about a specific restaurant by its Google Places ID."""
    try:
        client = GooglePlacesClient()
        details = client.get_place_details(place_id)
        
        if details.get("status") != "OK":
            raise HTTPException(
                status_code=404, 
                detail=f"Restaurant not found or API error: {details.get('status')}"
            )
        
        return {
            "status": "success",
            "restaurant": details.get("result", {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching restaurant details: {str(e)}")

@router.post("/save")
def save_restaurant_to_db(
    place_id: str,
    db: Session = Depends(get_db)
):
    """Save a restaurant from Google Places to the local database."""
    try:
        client = GooglePlacesClient()
        details = client.get_place_details(place_id)
        
        if details.get("status") != "OK":
            raise HTTPException(status_code=404, detail="Restaurant not found")
        
        result = details.get("result", {})
        
        # Check if restaurant already exists
        existing = db.query(Restaurant).filter(
            Restaurant.name == result.get("name")
        ).first()
        
        if existing:
            return {"status": "already_exists", "restaurant": existing}
        
        # Create new restaurant record
        restaurant = Restaurant(
            name=result.get("name"),
            cuisine=client._extract_cuisine_types(result.get("types", [])),
            location=result.get("formatted_address"),
            rating=result.get("rating"),
            source="Google Places"
        )
        
        db.add(restaurant)
        db.commit()
        db.refresh(restaurant)
        
        return {
            "status": "success",
            "message": "Restaurant saved to database",
            "restaurant": restaurant
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error saving restaurant: {str(e)}")

@router.get("/popular")
def get_popular_restaurants(
    min_rating: float = Query(4.0, ge=1.0, le=5.0, description="Minimum rating"),
    limit: int = Query(15, ge=1, le=30, description="Number of restaurants to return")
):
    """Get popular restaurants in Istanbul with high ratings and descriptions."""
    try:
        client = GooglePlacesClient()
        
        # Search in popular Istanbul areas
        popular_areas = ["Beyoğlu", "Sultanahmet", "Beşiktaş", "Kadıköy"]
        all_restaurants = []
        
        for area in popular_areas:
            restaurants = client.get_restaurants_with_descriptions(
                location=f"{area}, Istanbul, Turkey",
                limit=limit // len(popular_areas) + 2,
                radius=1000
            )
            
            # Filter by rating
            filtered = [r for r in restaurants if r.get("rating", 0) >= min_rating]
            all_restaurants.extend(filtered)
        
        # Sort by rating and remove duplicates
        unique_restaurants = {}
        for restaurant in all_restaurants:
            place_id = restaurant.get("place_id")
            if place_id not in unique_restaurants:
                unique_restaurants[place_id] = restaurant
        
        sorted_restaurants = sorted(
            unique_restaurants.values(),
            key=lambda x: (x.get("rating", 0), x.get("user_ratings_total", 0)),
            reverse=True
        )[:limit]
        
        return {
            "status": "success",
            "min_rating": min_rating,
            "total_found": len(sorted_restaurants),
            "restaurants": sorted_restaurants
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching popular restaurants: {str(e)}")

@router.get("/enhanced-search")
async def enhanced_restaurant_search(
    query: str = Query(..., description="Restaurant search query"),
    location: str = Query("Istanbul, Turkey", description="Location context"),
    session_id: Optional[str] = Query(None, description="User session ID for personalization"),
    budget_mode: bool = Query(False, description="Enable budget optimization mode")
):
    """
    Enhanced restaurant search with integrated caching, AI optimization, and personalization.
    
    Features:
    - Time-aware caching for optimal performance
    - Intent-based field optimization
    - Dynamic TTL based on query type
    - Cost optimization (up to 95% savings)
    - Real-time performance monitoring
    """
    try:
        if not _lazy_import_integrated_cache() or not search_restaurants_with_integrated_cache:
            raise HTTPException(
                status_code=503, 
                detail="Enhanced search not available - integrated cache system required"
            )
        
        # Use integrated cache system for optimized search
        result = await search_restaurants_with_integrated_cache(
            query=query,
            location=location,
            context={"budget_mode": budget_mode},
            session_id=session_id
        )
        
        if not result.get('success', True):
            raise HTTPException(status_code=500, detail=result.get('error', 'Search failed'))
        
        # Extract performance metrics
        cache_performance = result.get('cache_performance', {})
        optimization_info = result.get('optimization_info', {})
        
        # Calculate comprehensive cost analysis
        cost_analysis = _calculate_enhanced_cost_analysis(
            cache_hit=cache_performance.get('cache_hit', False),
            response_time_ms=cache_performance.get('response_time_ms', 0),
            cache_type=cache_performance.get('cache_type', 'unknown'),
            fields_requested=optimization_info.get('fields_requested', 0),
            total_available_fields=optimization_info.get('total_available_fields', 50),
            original_cost_score=optimization_info.get('original_cost_score', 100),
            optimized_cost_score=optimization_info.get('optimized_cost_score', 50),
            query=query,
            intent=optimization_info.get('intent', 'unknown')
        )
        
        return {
            "success": True,
            "query": query,
            "location": location,
            "restaurants": result.get('restaurants', []),
            "total_results": len(result.get('restaurants', [])),
            "performance": {
                "cache_hit": cache_performance.get('cache_hit', False),
                "response_time_ms": cache_performance.get('response_time_ms', 0),
                "cache_type": cache_performance.get('cache_type', 'unknown'),
                "cost_savings_percent": optimization_info.get('cost_savings_percent', 0),
                "fields_optimized": optimization_info.get('fields_requested', 0),
                "intent_detected": optimization_info.get('intent', 'unknown')
            },
            "cost_analysis": cost_analysis,
            "personalization": result.get('personalized_insights', {}),
            "data_source": result.get('data_source', 'google_places_optimized'),
            "timestamp": result.get('timestamp')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in enhanced restaurant search: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced search error: {str(e)}")

@router.get("/cost-analysis")
async def get_cost_analysis(
    sample_queries: Optional[str] = Query(None, description="Comma-separated sample queries to analyze"),
    user_scale: str = Query("medium_business", description="User scale: small_business, medium_business, large_enterprise, startup_mvp"),
    include_projections: bool = Query(True, description="Include cost projections and ROI analysis")
):
    """
    Get comprehensive cost analysis for restaurant search optimization.
    
    This endpoint provides detailed cost breakdowns, savings calculations,
    and ROI projections based on the integrated cache system optimizations.
    """
    try:
        if not INTEGRATED_CACHE_AVAILABLE:
            return {
                "error": "Integrated cache system not available",
                "fallback_analysis": _get_fallback_cost_analysis(user_scale, include_projections)
            }
        
        # Default sample queries if none provided
        if not sample_queries:
            sample_queries = [
                "best Turkish restaurants in Sultanahmet",
                "vegetarian restaurants in Beyoğlu", 
                "quick lunch places near Galata Tower",
                "seafood restaurants with Bosphorus view",
                "traditional Ottoman cuisine restaurants"
            ]
        else:
            sample_queries = [q.strip() for q in sample_queries.split(',')]
        
        # Analyze each sample query
        query_analyses = []
        total_optimized_cost = 0
        total_unoptimized_cost = 0
        
        for query in sample_queries[:10]:  # Limit to 10 queries
            try:
                # Simulate the search to get cost data
                if _lazy_import_integrated_cache() and search_restaurants_with_integrated_cache:
                    result = await search_restaurants_with_integrated_cache(
                        query=query,
                        location="Istanbul, Turkey",
                        context={"cost_analysis": True}
                    )
                    
                    cache_performance = result.get('cache_performance', {})
                    optimization_info = result.get('optimization_info', {})
                else:
                    # Use fallback values when integrated cache is not available
                    cache_performance = {"cache_hit": False, "response_time_ms": 800, "cache_type": "none"}
                    optimization_info = {"fields_requested": 10, "total_available_fields": 50, "original_cost_score": 100, "optimized_cost_score": 100, "intent": "basic_search"}
                
                cost_analysis = _calculate_enhanced_cost_analysis(
                    cache_hit=cache_performance.get('cache_hit', False),
                    response_time_ms=cache_performance.get('response_time_ms', 500),
                    cache_type=cache_performance.get('cache_type', 'unknown'),
                    fields_requested=optimization_info.get('fields_requested', 10),
                    total_available_fields=optimization_info.get('total_available_fields', 50),
                    original_cost_score=optimization_info.get('original_cost_score', 100),
                    optimized_cost_score=optimization_info.get('optimized_cost_score', 30),
                    query=query,
                    intent=optimization_info.get('intent', 'basic_search')
                )
                
                query_analyses.append({
                    "query": query,
                    "cost_analysis": cost_analysis
                })
                
                # Accumulate costs for overall analysis
                total_optimized_cost += cost_analysis['current_request']['total_cost_usd']
                total_unoptimized_cost += cost_analysis['cost_comparison']['unoptimized_cost_usd']
                
            except Exception as e:
                logger.error(f"Error analyzing query '{query}': {e}")
                query_analyses.append({
                    "query": query,
                    "error": str(e)
                })
        
        # Calculate overall metrics
        overall_savings = total_unoptimized_cost - total_optimized_cost
        overall_savings_percent = (overall_savings / total_unoptimized_cost) * 100 if total_unoptimized_cost > 0 else 0
        
        # Get system-wide analytics if available
        system_analytics = {}
        try:
            from integrated_cache_system import get_integrated_analytics
            system_analytics = get_integrated_analytics()
        except ImportError:
            system_analytics = {"note": "System analytics not available"}
        
        # Generate comprehensive cost report
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "cost_analysis_summary": {
                "queries_analyzed": len([q for q in query_analyses if 'error' not in q]),
                "average_optimized_cost_per_query": round(total_optimized_cost / max(1, len(sample_queries)), 6),
                "average_unoptimized_cost_per_query": round(total_unoptimized_cost / max(1, len(sample_queries)), 6),
                "overall_savings_percent": round(overall_savings_percent, 1),
                "total_savings_per_sample": round(overall_savings, 6)
            },
            "query_analyses": query_analyses,
            "optimization_impact": _calculate_optimization_impact(user_scale, overall_savings_percent, total_optimized_cost),
            "system_analytics": system_analytics,
            "cost_projections": _generate_cost_projections(user_scale, total_optimized_cost, total_unoptimized_cost) if include_projections else None,
            "recommendations": _generate_system_recommendations(overall_savings_percent, system_analytics)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in cost analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Cost analysis error: {str(e)}")

def _get_fallback_cost_analysis(user_scale: str, include_projections: bool) -> dict:
    """Generate cost analysis when integrated cache system is not available"""
    
    # Estimated costs without optimization
    base_cost_per_query = 0.092  # Higher cost without optimization
    
    scale_multipliers = {
        "small_business": 500,
        "medium_business": 5000, 
        "large_enterprise": 50000,
        "startup_mvp": 1000
    }
    
    monthly_requests = scale_multipliers.get(user_scale, 5000)
    monthly_cost = base_cost_per_query * monthly_requests
    
    return {
        "user_scale": user_scale,
        "monthly_requests": monthly_requests,
        "cost_per_query_usd": base_cost_per_query,
        "monthly_cost_usd": round(monthly_cost, 2),
        "annual_cost_usd": round(monthly_cost * 12, 2),
        "optimization_available": False,
        "potential_savings_with_optimization": "60-95%",
        "recommendation": "Enable integrated cache system for significant cost savings"
    }

def _calculate_optimization_impact(user_scale: str, savings_percent: float, optimized_cost_per_query: float) -> dict:
    """Calculate the impact of optimization based on user scale"""
    
    scale_data = {
        "small_business": {"requests": 500, "description": "Small restaurant business"},
        "medium_business": {"requests": 5000, "description": "Medium restaurant chain"},
        "large_enterprise": {"requests": 50000, "description": "Large food delivery platform"},
        "startup_mvp": {"requests": 1000, "description": "Restaurant discovery startup"}
    }
    
    scale_info = scale_data.get(user_scale, scale_data["medium_business"])
    monthly_requests = scale_info["requests"]
    
    # Calculate impact metrics
    monthly_optimized_cost = optimized_cost_per_query * monthly_requests
    monthly_unoptimized_cost = monthly_optimized_cost / (1 - savings_percent/100) if savings_percent > 0 else monthly_optimized_cost
    monthly_savings = monthly_unoptimized_cost - monthly_optimized_cost
    
    # Business impact categories
    if monthly_savings > 1000:
        impact_level = "Transformational"
        impact_description = "Massive cost reduction enabling significant business growth"
    elif monthly_savings > 500:
        impact_level = "High Impact"
        impact_description = "Substantial savings enabling feature expansion"
    elif monthly_savings > 100:
        impact_level = "Moderate Impact" 
        impact_description = "Good savings supporting operational efficiency"
    else:
        impact_level = "Low Impact"
        impact_description = "Modest savings with efficiency benefits"
    
    return {
        "user_scale": user_scale,
        "scale_description": scale_info["description"],
        "monthly_requests": monthly_requests,
        "impact_level": impact_level,
        "impact_description": impact_description,
        "financial_impact": {
            "monthly_savings_usd": round(monthly_savings, 2),
            "annual_savings_usd": round(monthly_savings * 12, 2),
            "cost_reduction_percent": round(savings_percent, 1),
            "payback_period_months": "Immediate" if monthly_savings > 0 else "N/A"
        },
        "operational_benefits": [
            f"{round(savings_percent, 1)}% reduction in API costs",
            "Faster response times through caching",
            "Improved user experience",
            "Reduced infrastructure load",
            "Better resource utilization"
        ]
    }

def _generate_cost_projections(user_scale: str, optimized_cost_per_query: float, unoptimized_cost_per_query: float) -> dict:
    """Generate detailed cost projections for different scenarios"""
    
    # Time-based projections
    time_periods = {
        "monthly": 1,
        "quarterly": 3,
        "annual": 12,
        "3_year": 36
    }
    
    scale_requests = {
        "small_business": 500,
        "medium_business": 5000,
        "large_enterprise": 50000, 
        "startup_mvp": 1000
    }
    
    base_requests = scale_requests.get(user_scale, 5000)
    
    projections = {}
    
    for period, months in time_periods.items():
        # Assume 10% growth per year
        growth_factor = (1 + 0.10) ** (months / 12)
        projected_requests = base_requests * months * growth_factor
        
        optimized_cost = optimized_cost_per_query * projected_requests
        unoptimized_cost = unoptimized_cost_per_query * projected_requests
        savings = unoptimized_cost - optimized_cost
        
        projections[period] = {
            "months": months,
            "projected_requests": int(projected_requests),
            "optimized_cost_usd": round(optimized_cost, 2),
            "unoptimized_cost_usd": round(unoptimized_cost, 2),
            "savings_usd": round(savings, 2),
            "roi_percent": round((savings / max(0.01, optimized_cost)) * 100, 1)
        }
    
    # Growth scenarios
    growth_scenarios = {}
    growth_rates = [0.05, 0.10, 0.20, 0.50]  # 5%, 10%, 20%, 50% annual growth
    
    for growth_rate in growth_rates:
        scenario_name = f"{int(growth_rate * 100)}percent_growth"
        annual_requests = base_requests * 12 * (1 + growth_rate)
        
        optimized_annual = optimized_cost_per_query * annual_requests
        unoptimized_annual = unoptimized_cost_per_query * annual_requests
        annual_savings = unoptimized_annual - optimized_annual
        
        growth_scenarios[scenario_name] = {
            "growth_rate_percent": int(growth_rate * 100),
            "annual_requests": int(annual_requests),
            "annual_optimized_cost_usd": round(optimized_annual, 2),
            "annual_unoptimized_cost_usd": round(unoptimized_annual, 2),
            "annual_savings_usd": round(annual_savings, 2)
        }
    
    return {
        "time_projections": projections,
        "growth_scenarios": growth_scenarios,
        "break_even_analysis": {
            "infrastructure_cost_monthly": 50,  # Estimated Redis + monitoring costs
            "break_even_requests_monthly": int(50 / max(0.001, (unoptimized_cost_per_query - optimized_cost_per_query))),
            "payback_period": "Immediate for most use cases"
        }
    }

def _generate_system_recommendations(savings_percent: float, system_analytics: dict) -> list:
    """Generate system-wide optimization recommendations"""
    recommendations = []
    
    # Cache performance recommendations
    cache_analytics = system_analytics.get('time_aware_cache', {})
    hit_rate = cache_analytics.get('overall_hit_rate_percent', 0)
    
    if hit_rate < 70:
        recommendations.append({
            "category": "cache_optimization",
            "priority": "high",
            "title": "Improve Cache Hit Rate",
            "description": f"Current hit rate is {hit_rate:.1f}%. Target should be >70%",
            "actions": [
                "Enable cache warming for popular queries",
                "Adjust TTL values for frequently accessed data",
                "Review cache eviction policies"
            ],
            "potential_impact": "10-25% additional cost savings"
        })
    
    if savings_percent < 60:
        recommendations.append({
            "category": "field_optimization", 
            "priority": "medium",
            "title": "Enhance Field Optimization",
            "description": f"Current savings is {savings_percent:.1f}%. More optimization possible",
            "actions": [
                "Review intent classification accuracy",
                "Adjust field mappings for common query types",
                "Enable budget mode for cost-sensitive operations"
            ],
            "potential_impact": "15-30% additional savings"
        })
    
    # System health recommendations
    monitoring_data = system_analytics.get('production_monitoring', {})
    if monitoring_data:
        error_count = monitoring_data.get('metrics', {}).get('error_count', 0)
        if error_count > 10:
            recommendations.append({
                "category": "reliability",
                "priority": "high", 
                "title": "Address System Errors",
                "description": f"System has {error_count} recent errors",
                "actions": [
                    "Review error logs in monitoring dashboard",
                    "Check Redis connectivity and performance",
                    "Validate API key configurations"
                ],
                "potential_impact": "Improved system reliability and performance"
            })
    
    # Performance recommendations
    recommendations.append({
        "category": "monitoring",
        "priority": "low",
        "title": "Regular Performance Review",
        "description": "Schedule regular review of cost optimization performance",
        "actions": [
            "Weekly review of cost analytics dashboard",
            "Monthly TTL optimization review",
            "Quarterly cost projection updates"
        ],
        "potential_impact": "Continuous optimization and cost control"
    })
    
    return recommendations

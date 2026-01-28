"""
LLM Data Integration Service
=============================

Centralized service that connects the LLM to all data handlers and services.
This ensures the LLM can access restaurant, transportation, attractions,
and other data from the database and specialized handlers.

Author: AI Istanbul Team
Date: January 2026
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class DataContext:
    """Context data for LLM prompt enrichment"""
    restaurants: List[Dict[str, Any]]
    attractions: List[Dict[str, Any]]
    transportation: Optional[Dict[str, Any]]
    weather: Optional[Dict[str, Any]]
    hidden_gems: List[Dict[str, Any]]
    neighborhood_info: Optional[Dict[str, Any]]
    
    def to_prompt_context(self) -> str:
        """Convert to text context for LLM prompt"""
        parts = []
        
        if self.restaurants:
            parts.append("=== RESTAURANT DATA ===")
            for r in self.restaurants[:5]:  # Limit to 5
                parts.append(
                    f"- {r.get('name', 'Unknown')}: {r.get('cuisine', 'Various')} cuisine "
                    f"in {r.get('district', 'Istanbul')}, "
                    f"Price: {r.get('price_level', 'N/A')}, "
                    f"Rating: {r.get('rating', 'N/A')}/5"
                )
                if r.get('description'):
                    parts.append(f"  Description: {r['description'][:100]}...")
        
        if self.attractions:
            parts.append("\n=== ATTRACTIONS DATA ===")
            for a in self.attractions[:5]:
                parts.append(
                    f"- {a.get('name', 'Unknown')} ({a.get('category', 'Attraction')}): "
                    f"{a.get('district', 'Istanbul')}"
                )
                if a.get('description'):
                    parts.append(f"  {a['description'][:100]}...")
        
        if self.hidden_gems:
            parts.append("\n=== HIDDEN GEMS ===")
            for g in self.hidden_gems[:3]:
                parts.append(f"- {g.get('name', 'Unknown')}: {g.get('description', '')[:100]}...")
        
        if self.neighborhood_info:
            parts.append("\n=== NEIGHBORHOOD INFO ===")
            parts.append(str(self.neighborhood_info))
        
        return "\n".join(parts)


class LLMDataIntegrationService:
    """
    Service that provides the LLM with access to all data sources.
    
    Data Sources:
    - PostgreSQL Database (restaurants, attractions, etc.)
    - Restaurant Query Handler (parsing and map data)
    - Transportation Services (routes, schedules)
    - Weather Service
    - Hidden Gems Database
    - Neighborhood Information
    """
    
    def __init__(self, db: Session):
        self.db = db
        self._restaurant_handler = None
        self._transport_service = None
        self._hidden_gems_service = None
        logger.info("✅ LLM Data Integration Service initialized")
    
    # ==========================================================================
    # RESTAURANT DATA ACCESS
    # ==========================================================================
    
    async def get_restaurants(
        self,
        query: str,
        cuisine_type: Optional[str] = None,
        district: Optional[str] = None,
        budget: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get restaurant data from database with optional filtering.
        
        Args:
            query: User's original query
            cuisine_type: Filter by cuisine (turkish, seafood, etc.)
            district: Filter by Istanbul district
            budget: Filter by price level (cheap, moderate, expensive)
            user_location: User's GPS coordinates for distance sorting
            limit: Maximum results to return
            
        Returns:
            List of restaurant dictionaries with full details
        """
        try:
            # Build dynamic SQL query
            sql = """
                SELECT 
                    id, name, cuisine, cuisine_type, district, location, address,
                    price_level, rating, description, phone, website,
                    latitude, longitude, opening_hours, dietary_options
                FROM restaurants
                WHERE 1=1
            """
            params = {}
            
            # Add filters
            if cuisine_type:
                sql += " AND (LOWER(cuisine) LIKE :cuisine OR LOWER(cuisine_type) LIKE :cuisine)"
                params['cuisine'] = f'%{cuisine_type.lower()}%'
            
            if district:
                sql += " AND LOWER(district) LIKE :district"
                params['district'] = f'%{district.lower()}%'
            
            if budget:
                budget_map = {'cheap': 1, 'moderate': 2, 'expensive': 3, 'luxury': 4}
                if budget.lower() in budget_map:
                    sql += " AND price_level <= :price"
                    params['price'] = budget_map[budget.lower()]
            
            # Order by rating and add limit
            sql += " ORDER BY rating DESC NULLS LAST LIMIT :limit"
            params['limit'] = limit
            
            # Execute query
            result = self.db.execute(text(sql), params)
            rows = result.fetchall()
            
            restaurants = []
            for row in rows:
                restaurant = {
                    'id': row[0],
                    'name': row[1],
                    'cuisine': row[2],
                    'cuisine_type': row[3],
                    'district': row[4],
                    'location': row[5],
                    'address': row[6],
                    'price_level': row[7],
                    'rating': row[8],
                    'description': row[9],
                    'phone': row[10],
                    'website': row[11],
                    'latitude': row[12],
                    'longitude': row[13],
                    'opening_hours': row[14],
                    'dietary_options': row[15]
                }
                
                # Calculate distance if user location provided
                if user_location and restaurant.get('latitude') and restaurant.get('longitude'):
                    restaurant['distance_km'] = self._calculate_distance(
                        user_location['lat'], user_location['lng'],
                        restaurant['latitude'], restaurant['longitude']
                    )
                
                restaurants.append(restaurant)
            
            # Sort by distance if available
            if user_location:
                restaurants.sort(key=lambda x: x.get('distance_km', 999))
            
            logger.info(f"✅ Found {len(restaurants)} restaurants matching criteria")
            return restaurants
            
        except Exception as e:
            logger.error(f"❌ Error fetching restaurants: {e}")
            return []
    
    async def get_restaurant_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed restaurant info by name"""
        try:
            result = self.db.execute(
                text("""
                    SELECT * FROM restaurants 
                    WHERE LOWER(name) LIKE :name
                    LIMIT 1
                """),
                {'name': f'%{name.lower()}%'}
            )
            row = result.fetchone()
            
            if row:
                columns = result.keys()
                return dict(zip(columns, row))
            return None
            
        except Exception as e:
            logger.error(f"Error fetching restaurant by name: {e}")
            return None
    
    # ==========================================================================
    # ATTRACTIONS DATA ACCESS
    # ==========================================================================
    
    async def get_attractions(
        self,
        query: str,
        category: Optional[str] = None,
        district: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get attractions from database"""
        try:
            sql = """
                SELECT 
                    id, name, category, district, description,
                    latitude, longitude, opening_hours, entry_fee,
                    historical_significance, visitor_tips
                FROM places
                WHERE 1=1
            """
            params = {}
            
            if category:
                sql += " AND LOWER(category) LIKE :category"
                params['category'] = f'%{category.lower()}%'
            
            if district:
                sql += " AND LOWER(district) LIKE :district"
                params['district'] = f'%{district.lower()}%'
            
            sql += " LIMIT :limit"
            params['limit'] = limit
            
            result = self.db.execute(text(sql), params)
            rows = result.fetchall()
            
            attractions = []
            for row in rows:
                attraction = {
                    'id': row[0],
                    'name': row[1],
                    'category': row[2],
                    'district': row[3],
                    'description': row[4],
                    'latitude': row[5],
                    'longitude': row[6],
                    'opening_hours': row[7],
                    'entry_fee': row[8],
                    'historical_significance': row[9],
                    'visitor_tips': row[10]
                }
                
                if user_location and attraction.get('latitude') and attraction.get('longitude'):
                    attraction['distance_km'] = self._calculate_distance(
                        user_location['lat'], user_location['lng'],
                        attraction['latitude'], attraction['longitude']
                    )
                
                attractions.append(attraction)
            
            logger.info(f"✅ Found {len(attractions)} attractions")
            return attractions
            
        except Exception as e:
            logger.error(f"❌ Error fetching attractions: {e}")
            return []
    
    # ==========================================================================
    # TRANSPORTATION DATA ACCESS
    # ==========================================================================
    
    async def get_transportation_route(
        self,
        from_location: str,
        to_location: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get transportation route using the integrated route service"""
        try:
            from services.ai_chat_route_integration import get_chat_route_handler
            
            handler = get_chat_route_handler()
            
            # Build query for route handler
            query = f"How do I get from {from_location} to {to_location}"
            
            context = {}
            if user_location:
                context['gps'] = user_location
                context['location'] = user_location
            
            result = await handler.handle_route_request(
                message=query,
                user_context=context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting transportation route: {e}")
            return None
    
    # ==========================================================================
    # HIDDEN GEMS DATA ACCESS
    # ==========================================================================
    
    async def get_hidden_gems(
        self,
        district: Optional[str] = None,
        category: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get hidden gems from database or service"""
        try:
            # Try to get from hidden gems service
            from services.hidden_gems_service import get_hidden_gems_service
            
            service = get_hidden_gems_service()
            if service:
                return service.get_gems(
                    district=district,
                    category=category,
                    user_location=user_location,
                    limit=limit
                )
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Hidden gems service unavailable: {e}")
        
        # Fallback to database query
        try:
            sql = """
                SELECT name, description, district, category, latitude, longitude
                FROM hidden_gems
                WHERE 1=1
            """
            params = {}
            
            if district:
                sql += " AND LOWER(district) LIKE :district"
                params['district'] = f'%{district.lower()}%'
            
            if category:
                sql += " AND LOWER(category) LIKE :category"
                params['category'] = f'%{category.lower()}%'
            
            sql += " LIMIT :limit"
            params['limit'] = limit
            
            result = self.db.execute(text(sql), params)
            rows = result.fetchall()
            
            return [
                {
                    'name': row[0],
                    'description': row[1],
                    'district': row[2],
                    'category': row[3],
                    'latitude': row[4],
                    'longitude': row[5]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.warning(f"Hidden gems database query failed: {e}")
            return []
    
    # ==========================================================================
    # NEIGHBORHOOD DATA ACCESS
    # ==========================================================================
    
    async def get_neighborhood_info(self, district: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive neighborhood information"""
        try:
            # Query neighborhoods table
            result = self.db.execute(
                text("""
                    SELECT name, description, highlights, best_for, 
                           getting_there, local_tips
                    FROM neighborhoods
                    WHERE LOWER(name) LIKE :district
                    LIMIT 1
                """),
                {'district': f'%{district.lower()}%'}
            )
            row = result.fetchone()
            
            if row:
                return {
                    'name': row[0],
                    'description': row[1],
                    'highlights': row[2],
                    'best_for': row[3],
                    'getting_there': row[4],
                    'local_tips': row[5]
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching neighborhood info: {e}")
            return None
    
    # ==========================================================================
    # COMBINED CONTEXT BUILDER
    # ==========================================================================
    
    async def build_llm_context(
        self,
        query: str,
        signals: Dict[str, bool],
        user_location: Optional[Dict[str, float]] = None,
        language: str = 'en'
    ) -> DataContext:
        """
        Build comprehensive context for LLM based on detected signals.
        
        Args:
            query: User's query
            signals: Dictionary of detected intent signals
            user_location: User's GPS location
            language: Language code
            
        Returns:
            DataContext with all relevant data for LLM
        """
        restaurants = []
        attractions = []
        transportation = None
        weather = None
        hidden_gems = []
        neighborhood_info = None
        
        # Fetch data based on signals
        if signals.get('needs_restaurant'):
            # Parse query for restaurant parameters
            cuisine_type = self._extract_cuisine_from_query(query)
            district = self._extract_district_from_query(query)
            budget = self._extract_budget_from_query(query)
            
            restaurants = await self.get_restaurants(
                query=query,
                cuisine_type=cuisine_type,
                district=district,
                budget=budget,
                user_location=user_location
            )
        
        if signals.get('needs_attraction'):
            category = self._extract_category_from_query(query)
            district = self._extract_district_from_query(query)
            
            attractions = await self.get_attractions(
                query=query,
                category=category,
                district=district,
                user_location=user_location
            )
        
        if signals.get('needs_transportation'):
            # Extract from/to from query
            from_loc, to_loc = self._extract_route_from_query(query)
            if from_loc and to_loc:
                transportation = await self.get_transportation_route(
                    from_location=from_loc,
                    to_location=to_loc,
                    user_location=user_location
                )
        
        if signals.get('needs_hidden_gems'):
            district = self._extract_district_from_query(query)
            hidden_gems = await self.get_hidden_gems(
                district=district,
                user_location=user_location
            )
        
        if signals.get('needs_neighborhood'):
            district = self._extract_district_from_query(query)
            if district:
                neighborhood_info = await self.get_neighborhood_info(district)
        
        return DataContext(
            restaurants=restaurants,
            attractions=attractions,
            transportation=transportation,
            weather=weather,
            hidden_gems=hidden_gems,
            neighborhood_info=neighborhood_info
        )
    
    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================
    
    def _calculate_distance(
        self, lat1: float, lng1: float, lat2: float, lng2: float
    ) -> float:
        """Calculate distance between two points in kilometers"""
        import math
        
        R = 6371  # Earth's radius in km
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lng = math.radians(lng2 - lng1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return round(R * c, 2)
    
    def _extract_cuisine_from_query(self, query: str) -> Optional[str]:
        """Extract cuisine type from query"""
        query_lower = query.lower()
        
        cuisine_keywords = {
            'turkish': ['turkish', 'türk', 'turk'],
            'kebab': ['kebab', 'kebap', 'döner', 'doner'],
            'seafood': ['seafood', 'fish', 'balık', 'balik'],
            'italian': ['italian', 'pizza', 'pasta'],
            'breakfast': ['breakfast', 'kahvaltı', 'kahvalti'],
            'cafe': ['cafe', 'coffee', 'kahve'],
        }
        
        for cuisine, keywords in cuisine_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return cuisine
        
        return None
    
    def _extract_district_from_query(self, query: str) -> Optional[str]:
        """Extract district name from query"""
        query_lower = query.lower()
        
        districts = [
            'sultanahmet', 'beyoglu', 'beyoğlu', 'taksim', 'kadikoy', 'kadıköy',
            'besiktas', 'beşiktaş', 'karakoy', 'karaköy', 'galata', 'eminonu',
            'eminönü', 'uskudar', 'üsküdar', 'fatih', 'balat', 'ortakoy', 'ortaköy'
        ]
        
        for district in districts:
            if district in query_lower:
                return district.title()
        
        return None
    
    def _extract_budget_from_query(self, query: str) -> Optional[str]:
        """Extract budget preference from query"""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ['cheap', 'budget', 'affordable', 'ucuz']):
            return 'cheap'
        if any(w in query_lower for w in ['luxury', 'expensive', 'upscale', 'lüks']):
            return 'expensive'
        if any(w in query_lower for w in ['moderate', 'mid-range', 'orta']):
            return 'moderate'
        
        return None
    
    def _extract_category_from_query(self, query: str) -> Optional[str]:
        """Extract attraction category from query"""
        query_lower = query.lower()
        
        categories = {
            'museum': ['museum', 'müze'],
            'mosque': ['mosque', 'cami', 'camii'],
            'palace': ['palace', 'saray'],
            'park': ['park', 'garden', 'bahçe'],
            'market': ['market', 'bazaar', 'çarşı'],
        }
        
        for category, keywords in categories.items():
            if any(kw in query_lower for kw in keywords):
                return category
        
        return None
    
    def _extract_route_from_query(self, query: str) -> tuple:
        """Extract from/to locations from query"""
        import re
        
        # Pattern: "from X to Y" or "X to Y"
        patterns = [
            r'from\s+([a-zA-ZğüşıöçĞÜŞİÖÇ\s]+?)\s+to\s+([a-zA-ZğüşıöçĞÜŞİÖÇ\s]+)',
            r'([a-zA-ZğüşıöçĞÜŞİÖÇ]+)\s+to\s+([a-zA-ZğüşıöçĞÜŞİÖÇ]+)',
            r"([a-zA-ZğüşıöçĞÜŞİÖÇ]+)'?dan\s+([a-zA-ZğüşıöçĞÜŞİÖÇ]+)'?[ae]",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()
        
        return None, None


# Singleton
_llm_data_service = None

def get_llm_data_service(db: Session) -> LLMDataIntegrationService:
    """Get or create LLM Data Integration Service singleton"""
    global _llm_data_service
    if _llm_data_service is None:
        _llm_data_service = LLMDataIntegrationService(db)
    elif _llm_data_service.db != db:
        _llm_data_service.db = db
    return _llm_data_service

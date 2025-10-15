#!/usr/bin/env python3
"""
Transportation Integration Helper
================================

Integrates the Enhanced Transportation System with the main Istanbul AI system
and provides comprehensive transportation query processing with all required features.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
import sqlite3
import os
import time
import hashlib

# Import the enhanced transportation system
try:
    from enhanced_transportation_system import (
        generate_comprehensive_transportation_response,
        ComprehensiveTransportProcessor,
        get_enhanced_transportation_system
    )
    ENHANCED_TRANSPORT_AVAILABLE = True
except ImportError:
    ENHANCED_TRANSPORT_AVAILABLE = False

# Import the ML transportation system as fallback
try:
    from ml_enhanced_transportation_system import (
        MLEnhancedTransportationSystem,
        GPSLocation,
        RouteOptimizationType,
        TransportMode,
        create_ml_enhanced_transportation_system
    )
    ML_SYSTEM_AVAILABLE = True
except ImportError:
    ML_SYSTEM_AVAILABLE = False
    # Create placeholder classes for compatibility
    class GPSLocation:
        def __init__(self, lat, lng, address=None, district=None):
            self.latitude = lat
            self.longitude = lng
            self.address = address
            self.district = district

# Import backend services
try:
    from backend.services.user_profiling_system import UserProfilingSystem
    from backend.services.behavioral_pattern_predictor import BehavioralPatternPredictor
    from backend.services.route_cache import RouteCache
    BACKEND_SERVICES_AVAILABLE = True
except ImportError:
    BACKEND_SERVICES_AVAILABLE = False

logger = logging.getLogger(__name__)


class LocationParser:
    """Basic location parser for transportation queries"""
    
    def __init__(self):
        self.known_locations = {
            'taksim': GPSLocation(41.0363, 28.9851, address='Taksim Square', district='Beyoğlu'),
            'sultanahmet': GPSLocation(41.0086, 28.9802, address='Sultanahmet Square', district='Fatih'),
            'galata tower': GPSLocation(41.0256, 28.9744, address='Galata Tower', district='Beyoğlu'),
            'blue mosque': GPSLocation(41.0054, 28.9768, address='Blue Mosque', district='Fatih'),
            'hagia sophia': GPSLocation(41.0086, 28.9802, address='Hagia Sophia', district='Fatih'),
            'grand bazaar': GPSLocation(41.0106, 28.9681, address='Grand Bazaar', district='Fatih'),
            'topkapi palace': GPSLocation(41.0115, 28.9833, address='Topkapi Palace', district='Fatih'),
            'galata bridge': GPSLocation(41.0201, 28.9744, address='Galata Bridge', district='Beyoğlu'),
            'ortaköy': GPSLocation(41.0553, 29.0265, address='Ortaköy', district='Beşiktaş'),
            'kadıköy': GPSLocation(40.9969, 29.0264, address='Kadıköy', district='Kadıköy'),
            'istanbul airport': GPSLocation(41.2753, 28.7519, address='Istanbul Airport', district='Arnavutköy'),
            'sabiha gökçen airport': GPSLocation(40.8986, 29.3092, address='Sabiha Gökçen Airport', district='Pendik')
        }
    
    def parse_location(self, location_text: str) -> Optional[GPSLocation]:
        """Parse location text and return GPS location"""
        location_lower = location_text.lower().strip()
        
        # Direct match
        if location_lower in self.known_locations:
            return self.known_locations[location_lower]
        
        # Partial match
        for key, location in self.known_locations.items():
            if key in location_lower or location_lower in key:
                return location
        
        # Try to extract coordinates if present
        import re
        coord_pattern = r'(\d+\.\d+),\s*(\d+\.\d+)'
        match = re.search(coord_pattern, location_text)
        if match:
            try:
                lat, lng = float(match.group(1)), float(match.group(2))
                if 40.8 <= lat <= 41.3 and 28.6 <= lng <= 29.3:  # Istanbul bounds
                    return GPSLocation(lat, lng, address=location_text)
            except ValueError:
                pass
        
        return None

class TransportationQueryAnalyzer:
    """Analyze transportation queries to extract intent and entities"""
    
    def __init__(self):
        self.transport_keywords = {
            'metro': ['metro', 'subway', 'underground', 'M1', 'M2', 'M3', 'M4', 'M7'],
            'bus': ['bus', 'otobüs', 'autobus'],
            'ferry': ['ferry', 'vapur', 'boat', 'bosphorus'],
            'taxi': ['taxi', 'taksi', 'uber', 'bitaksi'],
            'walking': ['walk', 'walking', 'on foot'],
            'tram': ['tram', 'tramway']
        }
        
        self.optimization_keywords = {
            'fastest': ['fast', 'quick', 'fastest', 'quickest'],
            'cheapest': ['cheap', 'cheapest', 'affordable', 'budget'],
            'shortest': ['short', 'shortest', 'direct'],
            'scenic': ['scenic', 'beautiful', 'sightseeing', 'view']
        }
    
    def analyze_query(self, message: str) -> Dict[str, Any]:
        """Analyze transportation query and extract information"""
        message_lower = message.lower()
        
        query_info = {
            'transport_modes': [],
            'optimization_preference': 'fastest',
            'include_attractions': False,
            'poi_categories': [],
            'locations': [],
            'route_planning': False
        }
        
        # Extract transport modes
        for mode, keywords in self.transport_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                query_info['transport_modes'].append(mode)
        
        # Extract optimization preference
        for opt_type, keywords in self.optimization_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                query_info['optimization_preference'] = opt_type
                break
        
        # Check for route planning indicators
        route_indicators = ['from', 'to', 'get to', 'go to', 'route', 'directions']
        if any(indicator in message_lower for indicator in route_indicators):
            query_info['route_planning'] = True
        
        # Check for POI/attraction mentions
        poi_keywords = ['museum', 'palace', 'mosque', 'attraction', 'sightseeing', 'visit']
        if any(keyword in message_lower for keyword in poi_keywords):
            query_info['include_attractions'] = True
            query_info['poi_categories'] = [kw for kw in poi_keywords if kw in message_lower]
        
        return query_info


class EnhancedLocationProcessor:
    """Enhanced location processing with backend integration"""
    
    def __init__(self):
        self.location_db_path = 'backend/data/location_cache.db'
        self.location_parser = LocationParser()
        self._init_location_database()
        
    def _init_location_database(self):
        """Initialize location processing database"""
        try:
            os.makedirs(os.path.dirname(self.location_db_path), exist_ok=True)
            conn = sqlite3.connect(self.location_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS location_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    user_id TEXT,
                    latitude REAL,
                    longitude REAL,
                    address TEXT,
                    district TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT 1
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_location_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    address TEXT,
                    query_context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS location_preferences (
                    user_id TEXT PRIMARY KEY,
                    favorite_locations TEXT,  -- JSON array
                    transport_preferences TEXT,  -- JSON array
                    accessibility_needs TEXT,  -- JSON array
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Location database initialization failed: {e}")
    
    async def process_location_query(self, query: str, user_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process location query with enhanced backend resolution"""
        try:
            # Check cache first
            cached_result = self._get_cached_location(query, user_id)
            if cached_result:
                return cached_result
            
            # Get user preferences for context
            user_prefs = self._get_user_location_preferences(user_id)
            
            # Try ML-enhanced resolution
            if BACKEND_SERVICES_AVAILABLE:
                ml_result = await self._ml_location_resolution(query, user_id, user_prefs, context)
                if ml_result:
                    self._cache_location_result(query, user_id, ml_result)
                    self._update_user_location_history(user_id, ml_result, query)
                    return ml_result
            
            # Fall back to basic resolution
            basic_result = self._basic_location_resolution(query)
            if basic_result:
                self._cache_location_result(query, user_id, basic_result)
                self._update_user_location_history(user_id, basic_result, query)
                return basic_result
            
            # No resolution found
            self._cache_failed_location(query, user_id)
            return {'error': 'Location not found', 'suggestions': self._get_location_suggestions(query)}
            
        except Exception as e:
            logger.error(f"Location query processing failed: {e}")
            return {'error': 'Location processing error'}
    
    def _get_cached_location(self, query: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached location result"""
        try:
            conn = sqlite3.connect(self.location_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT latitude, longitude, address, district, confidence, success
                FROM location_cache 
                WHERE query = ? AND (user_id = ? OR user_id IS NULL)
                AND datetime(timestamp) > datetime('now', '-24 hours')
                ORDER BY user_id DESC, timestamp DESC
                LIMIT 1
            ''', (query.lower().strip(), user_id))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[5]:  # success = True
                return {
                    'latitude': result[0],
                    'longitude': result[1],
                    'address': result[2],
                    'district': result[3],
                    'confidence': result[4],
                    'cached': True
                }
                
        except Exception as e:
            logger.error(f"Location cache retrieval failed: {e}")
        
        return None
    
    def _cache_location_result(self, query: str, user_id: str, result: Dict[str, Any]):
        """Cache location resolution result"""
        try:
            conn = sqlite3.connect(self.location_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO location_cache 
                (query, user_id, latitude, longitude, address, district, confidence, success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                query.lower().strip(),
                user_id,
                result.get('latitude'),
                result.get('longitude'),
                result.get('address'),
                result.get('district'),
                result.get('confidence', 0.8),
                not result.get('error')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Location caching failed: {e}")
    
    def _cache_failed_location(self, query: str, user_id: str):
        """Cache failed location query to avoid repeated attempts"""
        try:
            conn = sqlite3.connect(self.location_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO location_cache 
                (query, user_id, success)
                VALUES (?, ?, ?)
            ''', (query.lower().strip(), user_id, False))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed location caching failed: {e}")
    
    async def _ml_location_resolution(self, query: str, user_id: str, user_prefs: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """ML-enhanced location resolution"""
        # This would integrate with backend ML services
        # For now, fall back to basic resolution
        return self._basic_location_resolution(query)
    
    def _basic_location_resolution(self, query: str) -> Optional[Dict[str, Any]]:
        """Basic location resolution using known locations"""
        # Use the existing LocationParser
        gps_location = self.location_parser.parse_location(query)
        
        if gps_location:
            return {
                'latitude': gps_location.latitude,
                'longitude': gps_location.longitude,
                'address': gps_location.address,
                'district': getattr(gps_location, 'district', None),
                'confidence': 0.8
            }
        
        return None


class AdvancedUserPersonalization:
    """Advanced user personalization with backend integration"""
    
    def __init__(self):
        self.personalization_db_path = 'backend/data/personalization.db'
        self._init_personalization_database()
        
        # Initialize backend services if available
        if BACKEND_SERVICES_AVAILABLE:
            try:
                self.user_profiling = UserProfilingSystem()
                self.behavioral_predictor = BehavioralPatternPredictor()
            except Exception as e:
                logger.warning(f"Backend services initialization failed: {e}")
                self.user_profiling = None
                self.behavioral_predictor = None
        else:
            self.user_profiling = None
            self.behavioral_predictor = None
    
    def _init_personalization_database(self):
        """Initialize personalization database"""
        try:
            os.makedirs(os.path.dirname(self.personalization_db_path), exist_ok=True)
            conn = sqlite3.connect(self.personalization_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    optimization_preference TEXT DEFAULT 'fastest',
                    preferred_modes TEXT,  -- JSON array
                    budget_preference TEXT DEFAULT 'moderate',
                    travel_style TEXT DEFAULT 'general',
                    accessibility_needs TEXT,  -- JSON array
                    poi_interests TEXT,  -- JSON array
                    location_preferences TEXT,  -- JSON object
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS route_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    start_location TEXT,
                    end_location TEXT,
                    optimization_type TEXT,
                    user_rating INTEGER,
                    feedback TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Personalization database initialization failed: {e}")
    
    async def get_personalized_preferences(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized preferences with backend integration"""
        try:
            # Get cached preferences
            cached_prefs = self._get_cached_preferences(user_id)
            
            # Get behavioral insights if backend available
            behavioral_insights = {}
            if self.behavioral_predictor:
                try:
                    behavioral_insights = await self.behavioral_predictor.get_user_patterns(user_id)
                except Exception as e:
                    logger.warning(f"Behavioral insights failed: {e}")
                    behavioral_insights = self._analyze_basic_patterns(user_id, context)
            else:
                behavioral_insights = self._analyze_basic_patterns(user_id, context)
            
            # Combine preferences with behavioral insights
            personalized_prefs = self._combine_preferences(cached_prefs, behavioral_insights, context)
            
            return personalized_prefs
            
        except Exception as e:
            logger.error(f"Personalized preferences failed: {e}")
            return self._get_default_preferences()
    
    def _get_cached_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get cached user preferences"""
        try:
            conn = sqlite3.connect(self.personalization_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT optimization_preference, preferred_modes, budget_preference,
                       travel_style, accessibility_needs, poi_interests, location_preferences
                FROM user_preferences 
                WHERE user_id = ?
            ''', (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'optimization_preference': result[0],
                    'preferred_modes': json.loads(result[1] or '["metro", "bus", "walking"]'),
                    'budget_preference': result[2],
                    'travel_style': result[3],
                    'accessibility_needs': json.loads(result[4] or '[]'),
                    'poi_interests': json.loads(result[5] or '[]'),
                    'location_preferences': json.loads(result[6] or '{}')
                }
                
        except Exception as e:
            logger.error(f"Failed to get cached preferences: {e}")
        
        return self._get_default_preferences()
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences"""
        return {
            'preferred_modes': ['metro', 'bus', 'walking'],
            'optimization_preference': 'fastest',
            'accessibility_needs': [],
            'budget_preference': 'moderate',
            'travel_style': 'general',
            'poi_interests': [],
            'location_preferences': {}
        }
    
    def _combine_preferences(self, cached_prefs: Dict[str, Any], behavioral_insights: Dict[str, Any], 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Combine cached preferences with behavioral insights"""
        combined = cached_prefs.copy()
        
        # Apply behavioral insights
        if behavioral_insights.get('preferred_time_periods'):
            current_hour = context.get('current_time', datetime.now()).hour
            if current_hour in behavioral_insights['preferred_time_periods']:
                combined['time_preference_match'] = True
        
        # Apply contextual adjustments
        if context.get('query_info', {}).get('include_attractions'):
            combined['poi_interests'] = combined.get('poi_interests', []) + ['historical', 'cultural']
        
        return combined
    
    def _analyze_basic_patterns(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Basic pattern analysis for users without ML backend"""
        try:
            conn = sqlite3.connect(self.personalization_db_path)
            cursor = conn.cursor()
            
            # Get recent route history
            cursor.execute('''
                SELECT optimization_type, user_rating, COUNT(*)
                FROM route_history 
                WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
                GROUP BY optimization_type
                ORDER BY COUNT(*) DESC
            ''', (user_id,))
            
            results = cursor.fetchall()
            conn.close()
            
            patterns = {'preferred_optimization': 'fastest'}
            
            if results:
                # Most used optimization type
                most_used = results[0][0]
                patterns['preferred_optimization'] = most_used
                
                # Calculate satisfaction score
                total_ratings = sum(row[1] or 3 for row in results)
                total_routes = sum(row[2] for row in results)
                patterns['satisfaction_score'] = total_ratings / max(total_routes, 1)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Basic pattern analysis failed: {e}")
            return {'preferred_optimization': 'fastest'}


class EnhancedRouteComputation:
    """Enhanced route computation with backend integration and caching"""
    
    def __init__(self):
        self.route_db_path = 'backend/data/route_computation.db'
        self._init_route_database()
        
        # Initialize backend route cache if available
        if BACKEND_SERVICES_AVAILABLE:
            try:
                self.route_cache = RouteCache()
            except Exception as e:
                logger.warning(f"Route cache initialization failed: {e}")
                self.route_cache = None
        else:
            self.route_cache = None
    
    def _init_route_database(self):
        """Initialize route computation database"""
        try:
            os.makedirs(os.path.dirname(self.route_db_path), exist_ok=True)
            conn = sqlite3.connect(self.route_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS route_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE NOT NULL,
                    route_data TEXT NOT NULL,
                    computation_time REAL,
                    optimization_type TEXT,
                    user_id TEXT,
                    hit_count INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS route_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    route_id TEXT,
                    user_id TEXT,
                    computation_method TEXT,
                    time_taken REAL,
                    accuracy_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Route database initialization failed: {e}")
    
    async def compute_optimized_route(self, start_location: GPSLocation, end_location: GPSLocation,
                                    optimization_type: RouteOptimizationType, user_id: str,
                                    personalization_prefs: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute optimized route with backend integration and caching"""
        try:
            start_time = time.time()
            
            # Generate cache key
            cache_key = self._generate_cache_key(start_location, end_location, optimization_type, user_id)
            
            # Check cache first
            cached_result = await self._get_cached_route(cache_key, user_id)
            if cached_result:
                return cached_result
            
            # Compute new route using ML transportation system
            from transportation_integration_helper import TransportationQueryProcessor
            processor = TransportationQueryProcessor()
            
            route = await processor.ml_transport_system.get_optimized_route(
                start_location, end_location, optimization_type,
                include_pois=context.get('query_info', {}).get('include_attractions', False),
                poi_preferences=personalization_prefs.get('poi_interests', [])
            )
            
            if not route:
                return {'error': 'Unable to compute route'}
            
            # Enhance route with personalization
            enhanced_route = self._enhance_route_with_personalization(route, personalization_prefs, context)
            
            # Prepare result
            computation_time = time.time() - start_time
            route_result = {
                'route': enhanced_route,
                'computation_method': 'ml_enhanced' if BACKEND_SERVICES_AVAILABLE else 'basic',
                'computation_time': computation_time,
                'personalization_applied': True,
                'cache_key': cache_key
            }
            
            # Cache the result
            await self._cache_route_result(cache_key, route_result, computation_time, user_id)
            
            # Update performance metrics
            self._update_performance_metrics(route_result, computation_time, user_id)
            
            return route_result
            
        except Exception as e:
            logger.error(f"Route computation failed: {e}")
            return {'error': f'Route computation failed: {str(e)}'}
    
    def _generate_cache_key(self, start_location: GPSLocation, end_location: GPSLocation,
                           optimization_type: RouteOptimizationType, user_id: str) -> str:
        """Generate cache key for route"""
        key_data = f"{start_location.latitude},{start_location.longitude}_{end_location.latitude},{end_location.longitude}_{optimization_type.value}_{user_id}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_cached_route(self, cache_key: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached route if available and valid"""
        try:
            conn = sqlite3.connect(self.route_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT route_data, computation_time, timestamp
                FROM route_cache 
                WHERE cache_key = ? AND datetime(timestamp) > datetime('now', '-1 hour')
            ''', (cache_key,))
            
            result = cursor.fetchone()
            
            if result:
                # Update hit count
                cursor.execute('''
                    UPDATE route_cache 
                    SET hit_count = hit_count + 1 
                    WHERE cache_key = ?
                ''', (cache_key,))
                conn.commit()
                
                route_data = json.loads(result[0])
                route_data['cached'] = True
                route_data['cache_age_minutes'] = (datetime.now() - datetime.fromisoformat(result[2])).total_seconds() / 60
                
                conn.close()
                return route_data
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _cache_route_result(self, cache_key: str, route_result: Dict[str, Any], 
                                computation_time: float, user_id: str):
        """Cache route computation result"""
        try:
            conn = sqlite3.connect(self.route_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO route_cache 
                (cache_key, route_data, computation_time, optimization_type, user_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                cache_key,
                json.dumps(route_result),
                computation_time,
                route_result.get('route', {}).get('optimization_type', {}).get('value', 'unknown'),
                user_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Route caching failed: {e}")
    
    def _update_performance_metrics(self, route_result: Dict[str, Any], computation_time: float, user_id: str):
        """Update route computation performance metrics"""
        try:
            conn = sqlite3.connect(self.route_db_path)
            cursor = conn.cursor()
            
            route = route_result.get('route', {})
            
            cursor.execute('''
                INSERT INTO route_performance 
                (route_id, user_id, computation_method, time_taken, accuracy_score)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                getattr(route, 'route_id', 'unknown'),
                user_id,
                route_result.get('computation_method', 'unknown'),
                computation_time,
                getattr(route, 'confidence_score', 0.8)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    def _enhance_route_with_personalization(self, route, prefs: Dict[str, Any], context: Dict[str, Any]):
        """Enhance route with personalization context"""
        # Add personalization metadata to route
        if hasattr(route, '__dict__'):
            route.personalization_applied = True
            route.user_preferences = prefs
            route.context_factors = context
        
        return route


class TransportationQueryProcessor:
    """Enhanced transportation query processor with ML and GPS integration"""
    
    def __init__(self):
        self.ml_transport_system = create_ml_enhanced_transportation_system()
        self.location_parser = LocationParser()
        self.query_analyzer = TransportationQueryAnalyzer()
        
        # Initialize enhanced backend components
        self.location_processor = EnhancedLocationProcessor()
        self.user_personalization = AdvancedUserPersonalization()
        self.route_computation = EnhancedRouteComputation()
        
        logger.info("🚇 Enhanced Transportation Query Processor initialized with backend integration")
    
    async def process_transportation_query_enhanced(self, message: str, user_profile, current_time, 
                                                   user_location: Optional[Tuple[float, float]] = None) -> str:
        """Process transportation query with full ML, GPS, and backend enhancement"""
        try:
            user_id = getattr(user_profile, 'user_id', 'anonymous')
            
            # Parse the query with enhanced analysis
            query_info = self.query_analyzer.analyze_query(message)
            
            # Process location queries with backend integration
            location_context = {
                'message': message,
                'user_location': user_location,
                'query_info': query_info,
                'current_time': current_time
            }
            
            # Enhanced location resolution
            start_location, end_location = await self._resolve_locations_enhanced(
                query_info, user_id, location_context
            )
            
            if not start_location or not end_location:
                return await self._handle_location_request_enhanced(message, user_id, location_context)
            
            # Get personalized preferences
            personalization_prefs = await self.user_personalization.get_personalized_preferences(
                user_id, query_info
            )
            
            # Determine optimization type with personalization
            optimization_type = self._determine_optimization_type_enhanced(
                query_info, user_profile, personalization_prefs
            )
            
            # Compute optimized route with backend integration
            route_result = await self.route_computation.compute_optimized_route(
                start_location, end_location, optimization_type, user_id, 
                personalization_prefs, location_context
            )
            
            if route_result.get('error'):
                return f"I encountered an issue planning your route: {route_result['error']}"
            
            # Format enhanced response
            return await self._format_route_response_enhanced(
                route_result, user_profile, current_time, personalization_prefs
            )
            
        except Exception as e:
            logger.error(f"Enhanced transportation query processing failed: {e}")
            return self._fallback_transport_response(message, user_profile, current_time)
    
    async def get_gps_based_recommendations(self, user_location: Tuple[float, float], 
                                          user_profile) -> str:
        """Get GPS-based transportation recommendations"""
        try:
            gps_location = GPSLocation(user_location[0], user_location[1])
            recommendations = await self.ml_transport_system.get_location_based_recommendations(gps_location)
            
            return self._format_location_recommendations(recommendations, user_profile)
            
        except Exception as e:
            logger.error(f"GPS recommendations failed: {e}")
            return "I can help you with transportation! Tell me where you'd like to go."
    
    async def handle_route_planning_with_attractions(self, start: str, end: str, 
                                                   attractions: List[str], user_profile) -> str:
        """Handle route planning that includes museums and attractions"""
        try:
            start_location = self.location_parser.parse_location(start)
            end_location = self.location_parser.parse_location(end)
            
            if not start_location or not end_location:
                return "I need more specific location information. Could you provide addresses or landmarks?"
            
            # Map attraction interests to POI categories
            poi_categories = self._map_attractions_to_categories(attractions)
            
            route = await self.ml_transport_system.get_optimized_route(
                start_location, end_location, RouteOptimizationType.POI_OPTIMIZED,
                include_pois=True, poi_preferences=poi_categories
            )
            
            return self._format_attraction_route_response(route, user_profile)
            
        except Exception as e:
            logger.error(f"Attraction route planning failed: {e}")
            return "I can help plan your route with attractions! Please provide more details about your preferences."
    
    async def _resolve_locations(self, query_info: Dict, user_location: Optional[Tuple[float, float]], 
                               message: str) -> Tuple[Optional[GPSLocation], Optional[GPSLocation]]:
        """Resolve start and end locations from query"""
        start_location = None
        end_location = None
        
        # Handle "from my location" or GPS location
        if query_info.get('from_current_location', False) and user_location:
            start_location = GPSLocation(user_location[0], user_location[1], address="Your location")
        elif query_info.get('start_location'):
            start_location = self.location_parser.parse_location(query_info['start_location'])
        
        # Handle destination
        if query_info.get('end_location'):
            end_location = self.location_parser.parse_location(query_info['end_location'])
        
        return start_location, end_location
    
    def _determine_optimization_type(self, query_info: Dict, user_profile) -> RouteOptimizationType:
        """Determine route optimization type based on query and user profile"""
        # Check explicit optimization requests
        if query_info.get('fastest', False):
            return RouteOptimizationType.FASTEST
        elif query_info.get('cheapest', False):
            return RouteOptimizationType.CHEAPEST
        elif query_info.get('scenic', False):
            return RouteOptimizationType.MOST_SCENIC
        elif query_info.get('avoid_crowds', False):
            return RouteOptimizationType.LEAST_CROWDED
        
        # Use user profile preferences
        if hasattr(user_profile, 'travel_style'):
            if user_profile.travel_style == 'budget':
                return RouteOptimizationType.CHEAPEST
            elif user_profile.travel_style == 'luxury':
                return RouteOptimizationType.MOST_SCENIC
        
        return RouteOptimizationType.FASTEST  # Default
    
    def _map_attractions_to_categories(self, attractions: List[str]) -> List[str]:
        """Map attraction interests to POI categories"""
        category_mapping = {
            'museum': ['historical', 'art', 'cultural'],
            'palace': ['historical', 'palace'],
            'mosque': ['religious', 'historical'],
            'tower': ['landmark', 'scenic'],
            'bazaar': ['shopping', 'cultural'],
            'gallery': ['art', 'cultural']
        }
        
        categories = []
        for attraction in attractions:
            attraction_lower = attraction.lower()
            for key, cats in category_mapping.items():
                if key in attraction_lower:
                    categories.extend(cats)
        
        return list(set(categories)) if categories else ['historical', 'landmark']
    
    def _format_route_response(self, route, user_profile, current_time) -> str:
        """Format optimized route response"""
        try:
            response_parts = []
            
            # Header with optimization info
            hour = current_time.hour if hasattr(current_time, 'hour') else 12
            time_context = self._get_time_context(hour)
            
            response_parts.append(f"🗺️ **Optimized Route** ({route.optimization_type.value.replace('_', ' ').title()})")
            response_parts.append(f"⏱️ **Total Time**: {route.total_duration_minutes} minutes")
            response_parts.append(f"💰 **Total Cost**: {route.total_cost_tl:.2f} TL")
            response_parts.append(f"📏 **Distance**: {route.total_distance_km:.1f} km")
            
            # Add crowding information
            if route.crowding_prediction > 0.7:
                response_parts.append(f"⚠️ **Crowding**: High ({route.crowding_prediction:.0%}) - {time_context}")
            elif route.crowding_prediction > 0.4:
                response_parts.append(f"🟡 **Crowding**: Moderate ({route.crowding_prediction:.0%})")
            else:
                response_parts.append(f"✅ **Crowding**: Low ({route.crowding_prediction:.0%})")
            
            # Add route segments
            response_parts.append("\n**🚶‍♂️ Route Steps:**")
            for i, segment in enumerate(route.segments, 1):
                mode_emoji = self._get_transport_emoji(segment.transport_mode)
                response_parts.append(
                    f"{i}. {mode_emoji} {segment.transport_mode.value.title()}: "
                    f"{segment.duration_minutes} min ({segment.distance_km:.1f} km)"
                )
                if segment.instructions:
                    response_parts.append(f"   {segment.instructions[0]}")
            
            # Add POI information if included
            if route.poi_integration:
                response_parts.append(f"\n🏛️ **Attractions Included**: {len(route.poi_integration)}")
                for poi in route.poi_integration[:2]:  # Show first 2
                    response_parts.append(f"   • {poi['name']} ({poi['visit_duration_minutes']} min visit)")
            
            # Add real-time updates
            if route.real_time_adjustments:
                response_parts.append(f"\n📡 **Live Updates**:")
                for update in route.real_time_adjustments[:2]:  # Show first 2
                    response_parts.append(f"   {update}")
            
            # Add personalized tips
            response_parts.append(f"\n{self._get_personalized_tip(route, user_profile)}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Route formatting failed: {e}")
            return f"Route found: {route.total_duration_minutes} minutes, {route.total_cost_tl:.2f} TL"
    
    def _format_location_recommendations(self, recommendations: Dict, user_profile) -> str:
        """Format location-based recommendations"""
        try:
            response_parts = []
            response_parts.append("📍 **Your Location Analysis**")
            
            # Nearby transport
            if recommendations.get('nearby_transport'):
                response_parts.append(f"\n🚇 **Nearby Transport**:")
                for transport in recommendations['nearby_transport'][:3]:
                    station = transport['station']
                    response_parts.append(
                        f"   • {station.name}: {transport['walking_time_minutes']} min walk"
                    )
            
            # Nearby POIs
            if recommendations.get('nearby_pois'):
                response_parts.append(f"\n🏛️ **Nearby Attractions**:")
                for poi in recommendations['nearby_pois'][:3]:
                    response_parts.append(
                        f"   • {poi['name']}: {poi['walking_time_minutes']} min walk"
                    )
            
            # Custom recommendations
            if recommendations.get('recommendations'):
                response_parts.append(f"\n💡 **Recommendations**:")
                for rec in recommendations['recommendations']:
                    response_parts.append(f"   {rec}")
            
            response_parts.append(f"\nTell me where you'd like to go for personalized route planning!")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Location recommendations formatting failed: {e}")
            return "I can see your location! Where would you like to go?"
    
    def _format_attraction_route_response(self, route, user_profile) -> str:
        """Format route response with attraction focus"""
        response = self._format_route_response(route, user_profile, datetime.now())
        
        if route.poi_integration:
            response += f"\n\n🎯 **Attraction Highlights**:"
            for poi in route.poi_integration:
                response += f"\n   🏛️ **{poi['name']}**"
                response += f"\n       Visit time: {poi['visit_duration_minutes']} minutes"
                response += f"\n       Category: {poi['category'].title()}"
                if poi.get('entrance_fee_tl', 0) > 0:
                    response += f"\n       Entrance: {poi['entrance_fee_tl']} TL"
        
        return response
    
    def _get_transport_emoji(self, transport_mode: TransportMode) -> str:
        """Get emoji for transport mode"""
        emoji_map = {
            TransportMode.WALKING: "🚶‍♂️",
            TransportMode.METRO: "🚇",
            TransportMode.BUS: "🚌",
            TransportMode.TRAM: "🚊",
            TransportMode.FERRY: "⛴️",
            TransportMode.TAXI: "🚖",
            TransportMode.MARMARAY: "🚄"
        }
        return emoji_map.get(transport_mode, "🚶‍♂️")
    
    def _get_time_context(self, hour: int) -> str:
        """Get time-based context"""
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return "Peak rush hour"
        elif 6 <= hour < 7 or 9 < hour <= 11 or 15 <= hour < 17 or 19 < hour <= 21:
            return "Busy period"
        else:
            return "Quiet period"
    
    def _get_personalized_tip(self, route, user_profile) -> str:
        """Get personalized tip based on route and user profile"""
        tips = []
        
        if hasattr(user_profile, 'travel_style'):
            if user_profile.travel_style == 'budget' and route.total_cost_tl > 15:
                tips.append("💰 Tip: Consider walking part of the route to save money")
            elif user_profile.travel_style == 'family':
                tips.append("👨‍👩‍👧‍👦 Tip: All suggested transport is family-friendly")
        
        if route.crowding_prediction > 0.7:
            tips.append("⏰ Tip: Consider traveling 30 minutes earlier/later to avoid crowds")
        
        if any(s.transport_mode == TransportMode.FERRY for s in route.segments):
            tips.append("📸 Tip: Great photo opportunities during the ferry ride!")
        
        return tips[0] if tips else "💡 Have a great trip!"
    
    def _handle_location_request(self) -> str:
        """Handle when location information is missing"""
        return """📍 **Location Information Needed**

I need to know your starting point and destination to plan the best route!

**You can tell me:**
• "From Taksim to Sultanahmet"
• "How to get to Galata Tower from my location"
• "Route from Kadıköy to Beyoğlu"

**Or enable GPS location sharing for:**
• "Directions to Blue Mosque from my location"
• "Best way to get to airport from here"

I'll provide optimized routes with real-time crowding data and ML predictions! 🚀"""
    
    async def _resolve_locations_enhanced(self, query_info: Dict, user_id: str, 
                                         context: Dict[str, Any]) -> Tuple[Optional[GPSLocation], Optional[GPSLocation]]:
        """Enhanced location resolution with backend integration"""
        start_location = None
        end_location = None
        
        # Process start location
        if query_info.get('from_current_location', False) and context.get('user_location'):
            user_loc = context['user_location']
            start_location = GPSLocation(user_loc[0], user_loc[1], address="Your current location")
        elif query_info.get('start_location'):
            start_result = await self.location_processor.process_location_query(
                query_info['start_location'], user_id, context
            )
            if start_result and not start_result.get('error'):
                start_location = GPSLocation(
                    start_result['latitude'], start_result['longitude'], 
                    address=start_result.get('address')
                )
        
        # Process end location
        if query_info.get('end_location'):
            end_result = await self.location_processor.process_location_query(
                query_info['end_location'], user_id, context
            )
            if end_result and not end_result.get('error'):
                end_location = GPSLocation(
                    end_result['latitude'], end_result['longitude'],
                    address=end_result.get('address')
                )
        
        return start_location, end_location
    
    async def _handle_location_request_enhanced(self, message: str, user_id: str, 
                                              context: Dict[str, Any]) -> str:
        """Enhanced location request handling with personalization"""
        try:
            # Get user's location preferences and history
            user_prefs = await self.user_personalization.get_personalized_preferences(user_id, context)
            
            # Check if user has favorite locations
            favorite_locations = user_prefs.get('location_preferences', {}).get('favorites', [])
            
            response_parts = ["📍 **I need more location information to help you!**\n"]
            
            if favorite_locations:
                response_parts.append("🌟 **Your Favorite Locations:**")
                for loc in favorite_locations[:3]:
                    response_parts.append(f"   • {loc}")
                response_parts.append("")
            
            response_parts.extend([
                "**Tell me your route like:**",
                "• \"From Taksim to Sultanahmet\"",
                "• \"How to get to Galata Tower from my location\"",
                "• \"Route from Kadıköy to Beyoğlu\"",
                "",
                "**Or enable GPS for:**",
                "• \"Directions to Blue Mosque from here\"",
                "• \"Best way to airport from my location\"",
                "",
                f"💡 Based on your preferences, I'll suggest **{user_prefs.get('optimization_preference', 'fastest')}** routes with **{user_prefs.get('budget_preference', 'moderate')}** budget options!"
            ])
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Enhanced location request handling failed: {e}")
            return self._handle_location_request()
    
    def _determine_optimization_type_enhanced(self, query_info: Dict, user_profile, 
                                            personalization_prefs: Dict[str, Any]) -> RouteOptimizationType:
        """Enhanced optimization type determination with personalization"""
        # Check explicit query preferences first
        if query_info.get('fastest', False):
            return RouteOptimizationType.FASTEST
        elif query_info.get('cheapest', False):
            return RouteOptimizationType.CHEAPEST
        elif query_info.get('scenic', False):
            return RouteOptimizationType.MOST_SCENIC
        elif query_info.get('avoid_crowds', False):
            return RouteOptimizationType.LEAST_CROWDED
        
        # Use personalized preferences
        pref_mapping = {
            'fastest': RouteOptimizationType.FASTEST,
            'cheapest': RouteOptimizationType.CHEAPEST,
            'scenic': RouteOptimizationType.MOST_SCENIC,
            'least_crowded': RouteOptimizationType.LEAST_CROWDED
        }
        
        optimization_pref = personalization_prefs.get('optimization_preference', 'fastest')
        return pref_mapping.get(optimization_pref, RouteOptimizationType.FASTEST)
    
    async def _format_route_response_enhanced(self, route_result: Dict[str, Any], user_profile, 
                                            current_time, personalization_prefs: Dict[str, Any]) -> str:
        """Enhanced route response formatting with personalization context"""
        try:
            route = route_result.get('route')
            if not route:
                return "I couldn't generate a route. Please try with different locations."
            
            response_parts = []
            
            # Personalized header
            user_style = personalization_prefs.get('travel_style', 'general')
            style_emoji = {'budget': '💰', 'luxury': '✨', 'family': '👨‍👩‍👧‍👦', 'solo': '🚶‍♂️'}.get(user_style, '🗺️')
            
            response_parts.append(f"{style_emoji} **Your Personalized Route** ({route.optimization_type.value.replace('_', ' ').title()})")
            
            # Add personalization context
            if route_result.get('personalization_applied'):
                prefs_used = personalization_prefs.get('optimization_preference', 'fastest')
                response_parts.append(f"🎯 *Optimized for your {prefs_used} preference*")
            
            # Route details with enhanced context
            hour = current_time.hour if hasattr(current_time, 'hour') else 12
            time_context = self._get_time_context(hour)
            
            response_parts.extend([
                f"⏱️ **Duration**: {route.total_duration_minutes} minutes",
                f"💰 **Cost**: {route.total_cost_tl:.2f} TL",
                f"📏 **Distance**: {route.total_distance_km:.1f} km"
            ])
            
            # Enhanced crowding information with personalization
            crowding_level = route.crowding_prediction
            if crowding_level > 0.7:
                if 'accessibility' in personalization_prefs.get('accessibility_needs', []):
                    response_parts.append(f"♿ **Accessibility Alert**: High crowding ({crowding_level:.0%}) - Consider alternative time")
                else:
                    response_parts.append(f"⚠️ **Crowding**: High ({crowding_level:.0%}) - {time_context}")
            elif crowding_level > 0.4:
                response_parts.append(f"🟡 **Crowding**: Moderate ({crowding_level:.0%})")
            else:
                response_parts.append(f"✅ **Crowding**: Low ({crowding_level:.0%}) - Great timing!")
            
            # Enhanced route segments with personalization
            response_parts.append("\n**🚶‍♂️ Your Route Steps:**")
            preferred_modes = personalization_prefs.get('preferred_modes', [])
            
            for i, segment in enumerate(route.segments, 1):
                mode_emoji = self._get_transport_emoji(segment.transport_mode)
                mode_name = segment.transport_mode.value.title()
                
                # Add preference indicator
                pref_indicator = " ⭐" if segment.transport_mode.value in preferred_modes else ""
                
                response_parts.append(
                    f"{i}. {mode_emoji} {mode_name}{pref_indicator}: "
                    f"{segment.duration_minutes} min ({segment.distance_km:.1f} km)"
                )
                
                if segment.instructions:
                    response_parts.append(f"   {segment.instructions[0]}")
            
            # Enhanced POI information
            if route.poi_integration:
                poi_interests = personalization_prefs.get('poi_interests', [])
                response_parts.append(f"\n🏛️ **Attractions Included** (matching your interests):")
                for poi in route.poi_integration[:3]:
                    interest_match = "⭐" if any(interest in poi.get('category', '').lower() 
                                                for interest in poi_interests) else ""
                    response_parts.append(f"   • {poi['name']} {interest_match} ({poi.get('visit_duration_minutes', 30)} min)")
            
            # Personalized tips and recommendations
            personalized_tip = self._get_personalized_tip_enhanced(route, personalization_prefs, current_time)
            response_parts.append(f"\n{personalized_tip}")
            
            # Add backend performance info if available
            if route_result.get('computation_method'):
                method = route_result['computation_method']
                response_parts.append(f"\n🔧 *Route computed using {method} with your personal preferences*")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Enhanced route formatting failed: {e}")
            return self._format_route_response(route_result.get('route'), user_profile, current_time)
    
    def _get_personalized_tip_enhanced(self, route, personalization_prefs: Dict[str, Any], current_time) -> str:
        """Get enhanced personalized tip with backend insights"""
        tips = []
        
        # Budget-conscious tips
        if personalization_prefs.get('budget_preference') == 'budget':
            if route.total_cost_tl > 15:
                tips.append("💰 Budget Tip: Consider walking part of the route to save money")
            elif route.total_cost_tl == 0:
                tips.append("💰 Great choice! This route is completely free")
        
        # Accessibility tips
        accessibility_needs = personalization_prefs.get('accessibility_needs', [])
        if accessibility_needs:
            if 'wheelchair' in accessibility_needs:
                tips.append("♿ All suggested transport options are wheelchair accessible")
            elif 'mobility' in accessibility_needs:
                tips.append("🚶‍♂️ Route optimized for easy mobility - minimal walking required")
        
        # Travel style tips
        travel_style = personalization_prefs.get('travel_style', 'general')
        if travel_style == 'family':
            tips.append("👨‍👩‍👧‍👦 Family-friendly route with amenities for children")
        elif travel_style == 'luxury':
            tips.append("✨ Premium experience - comfortable and scenic options selected")
        elif travel_style == 'solo':
            tips.append("🚶‍♂️ Perfect for solo exploration - safe and efficient route")
        
        # Time-based tips
        hour = current_time.hour if hasattr(current_time, 'hour') else 12
        if route.crowding_prediction > 0.7:
            if 7 <= hour <= 9:
                tips.append("⏰ Morning rush hour - consider departing 30 minutes earlier")
            elif 17 <= hour <= 19:
                tips.append("⏰ Evening rush hour - consider waiting until after 7 PM")
        
        # POI tips based on interests
        poi_interests = personalization_prefs.get('poi_interests', [])
        if route.poi_integration and poi_interests:
            matching_interests = [poi for poi in route.poi_integration 
                                if any(interest in poi.get('category', '').lower() for interest in poi_interests)]
            if matching_interests:
                tips.append(f"🎯 Route includes attractions matching your {', '.join(poi_interests[:2])} interests!")
        
        # Ferry scenic tip
        if any(s.transport_mode.value == 'ferry' for s in route.segments):
            tips.append("📸 Don't forget your camera - amazing Bosphorus views ahead!")
        
        return tips[0] if tips else "💡 Have a wonderful trip exploring Istanbul!"



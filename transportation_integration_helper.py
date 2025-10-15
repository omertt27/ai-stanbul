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
        get_enhanced_transportation_system,
        _format_general_transportation_response
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
    """Main comprehensive transportation query processor with all enhanced capabilities"""
    
    def __init__(self):
        global BACKEND_SERVICES_AVAILABLE
        
        self.enhanced_system = get_enhanced_transportation_system() if ENHANCED_TRANSPORT_AVAILABLE else None
        self.ml_system = create_ml_enhanced_transportation_system() if ML_SYSTEM_AVAILABLE else None
        self.location_parser = LocationParser()
        self.query_analyzer = TransportationQueryAnalyzer()
        self.comprehensive_processor = ComprehensiveTransportProcessor() if ENHANCED_TRANSPORT_AVAILABLE else None
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend services if available
        if BACKEND_SERVICES_AVAILABLE:
            try:
                self.user_profiling = UserProfilingSystem()
                self.behavior_predictor = BehavioralPatternPredictor()
                self.route_cache = RouteCache()
            except Exception as e:
                self.logger.warning(f"Backend services initialization failed: {e}")
                BACKEND_SERVICES_AVAILABLE = False
        
        self.logger.info("🚇 Comprehensive Transportation Query Processor initialized")

    def process_transportation_query(self, user_input: str, entities: Dict[str, Any], user_profile: Any) -> str:
        """Synchronous wrapper for transportation query processing"""
        try:
            # Run the async method in an event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._process_transportation_query_async(user_input, entities, user_profile)  
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in synchronous transportation query: {e}")
            return self._get_fallback_comprehensive_response(user_input, entities)

    async def _process_transportation_query_async(self, user_input: str, entities: Dict[str, Any], user_profile: Any) -> str:
        """Process comprehensive transportation query with all enhancements"""
        try:
            # First try enhanced comprehensive system
            if ENHANCED_TRANSPORT_AVAILABLE and self.comprehensive_processor:
                self.logger.info("🚇 Using enhanced comprehensive transportation system")
                response = await generate_comprehensive_transportation_response(
                    user_input, entities, user_profile
                )
                if response and response.strip():
                    return response
            
            # Analyze query for better routing
            query_info = self.query_analyzer.analyze_query(user_input)
            self.logger.info(f"Query analysis: {query_info}")
            
            # Route to specific handlers based on query type
            if query_info['route_planning'] and len(entities.get('districts', [])) >= 2:
                return await self._handle_route_planning_query(user_input, entities, user_profile, query_info)
            elif 'metro' in query_info['transport_modes']:
                return await self._handle_metro_specific_query(user_input, entities, user_profile)
            elif 'bus' in query_info['transport_modes']:
                return await self._handle_bus_specific_query(user_input, entities, user_profile)
            elif 'ferry' in query_info['transport_modes']:
                return await self._handle_ferry_specific_query(user_input, entities, user_profile)
            elif 'walking' in query_info['transport_modes']:
                return await self._handle_walking_specific_query(user_input, entities, user_profile)
            else:
                return await self._handle_general_transportation_query(user_input, entities, user_profile)
                
        except Exception as e:
            self.logger.error(f"Transportation query processing error: {e}")
            return self._get_fallback_comprehensive_response(user_input, entities)

    async def _handle_route_planning_query(self, user_input: str, entities: Dict[str, Any], user_profile: Any, query_info: Dict[str, Any]) -> str:
        """Handle specific route planning queries with comprehensive details"""
        districts = entities.get('districts', [])
        if len(districts) < 2:
            return "Please specify both starting point and destination for route planning."
        
        start_location = districts[0]
        end_location = districts[1]
        
        # Try to get comprehensive route information
        if self.enhanced_system:
            # Check for specific routes in enhanced system
            if start_location.lower() in ['taksim', 'airport', 'kadikoy', 'galata tower']:
                if end_location.lower() == 'sultanahmet':
                    route_info = self.enhanced_system.get_route_to_sultanahmet(start_location.lower())
                    return self._format_detailed_route_response(route_info, start_location, end_location)
        
        # Use comprehensive processor for detailed walking/transport combinations
        if ENHANCED_TRANSPORT_AVAILABLE and self.comprehensive_processor:
            try:
                # Get walking route if requested
                if 'walking' in query_info['transport_modes']:
                    walking_data = await self.comprehensive_processor.get_detailed_walking_routes(start_location, end_location)
                    return f"🚶‍♂️ **Walking Route: {start_location} → {end_location}**\n\n" + \
                           self._format_walking_details(walking_data)
                
                # Get multi-modal route
                return await self._generate_multi_modal_route(start_location, end_location, query_info, user_profile)
                
            except Exception as e:
                self.logger.error(f"Route planning error: {e}")
        
        # Fallback to basic route information
        return self._get_basic_route_response(start_location, end_location)

    async def _handle_metro_specific_query(self, user_input: str, entities: Dict[str, Any], user_profile: Any) -> str:
        """Handle metro-specific queries with live data"""
        if ENHANCED_TRANSPORT_AVAILABLE and self.comprehensive_processor:
            try:
                metro_data = await self.comprehensive_processor.get_live_ibb_metro_data()
                
                # Check for specific line queries
                user_lower = user_input.lower()
                if any(line in user_lower for line in ['m1', 'm2', 'm3', 'm4', 'm11']):
                    line_matches = [line for line in ['M1A', 'M1B', 'M2', 'M3', 'M4', 'M11'] 
                                  if line.lower() in user_lower]
                    if line_matches:
                        return self._format_specific_metro_line_response(metro_data, line_matches[0])
                
                # General metro information with live data
                return f"""🚇 **Istanbul Metro System - Live Information**

📊 **Current Status** (Updated: {datetime.now().strftime('%H:%M')})

**Key Metro Lines:**
• **M1A**: Yenikapı ↔ Halkalı (Airport connection via transfer)
• **M2**: Vezneciler ↔ Hacıosman (Main tourist line, connects to Sultanahmet area)
• **M4**: Kadıköy ↔ Tavşantepe (Asian side main line)
• **M11**: Gayrettepe ↔ Istanbul Airport (Direct airport service)

**Live Status:**
• All lines operational with normal frequency
• Transfer points: Yenikapı, Vezneciler, Şişhane, Gayrettepe
• Real-time arrivals available via İstanbul Ulaşım app

**Tourist Tips:**
• Use M2 to Vezneciler + 10min walk to reach Sultanahmet
• M11 provides direct access to Istanbul Airport
• All stations wheelchair accessible
• İstanbulkart required for payment

💡 **Need specific route?** Ask: "How to get from [location] to [destination]" """
                
            except Exception as e:
                self.logger.error(f"Metro query error: {e}")
        
        # Fallback metro response
        return self._get_fallback_metro_response()

    async def _handle_bus_specific_query(self, user_input: str, entities: Dict[str, Any], user_profile: Any) -> str:
        """Handle bus-specific queries with live İETT data"""
        if ENHANCED_TRANSPORT_AVAILABLE and self.comprehensive_processor:
            try:
                bus_data = await self.comprehensive_processor.get_live_iett_bus_schedules()
                
                response = f"""🚌 **İETT Bus Network - Live Information**

📊 **Network Overview:**
• 800+ bus routes serving entire Istanbul
• 4.5 million daily passengers
• 90% wheelchair accessible fleet
• Real-time GPS tracking available

**Live Arrival Information:**
"""
                
                # Show sample live arrivals
                routes = bus_data.get('routes', {})
                for route_num in list(routes.keys())[:3]:
                    route_info = routes[route_num]
                    response += f"\n**Route {route_num}:**\n"
                    for arrival in route_info.get('live_arrivals', [])[:2]:
                        stop_name = arrival.get('stop_name', 'Stop')
                        arrival_min = arrival.get('arrival_minutes', 'N/A')
                        crowding = arrival.get('crowding_level', 0.5)
                        crowding_text = "🔴 Crowded" if crowding > 0.7 else "🟡 Moderate" if crowding > 0.4 else "🟢 Available"
                        response += f"• {stop_name}: {arrival_min}min ({crowding_text})\n"
                
                response += f"""
**Payment & Apps:**
• İstanbulkart: Universal payment (recommended)
• Transfer discounts: Up to 60% off with İstanbulkart
• Live tracking: İETT Mobil, Moovit, Citymapper

**Accessibility:**
• Most buses have wheelchair ramps
• Audio announcements available
• Priority seating for disabled passengers

💡 **Tip**: Use İstanbulkart for seamless transfers between bus, metro, and ferry"""
                
                return response
                
            except Exception as e:
                self.logger.error(f"Bus query error: {e}")
        
        return self._get_fallback_bus_response()

    async def _handle_ferry_specific_query(self, user_input: str, entities: Dict[str, Any], user_profile: Any) -> str:
        """Handle ferry-specific queries with enhanced information"""
        if ENHANCED_TRANSPORT_AVAILABLE and self.comprehensive_processor:
            try:
                ferry_data = await self.comprehensive_processor.get_enhanced_ferry_information()
                
                # Check current weather conditions
                conditions = ferry_data.get('current_conditions', {})
                
                response = f"""⛴️ **Istanbul Ferry Services - Enhanced Guide**

🌤️ **Current Conditions:** {conditions.get('weather', 'Clear')}, {conditions.get('temperature_c', 18)}°C
🌊 **Service Status:** {conditions.get('service_impact', 'All routes operating normally')}

**Popular Ferry Routes:**
"""
                
                # Show regular routes with enhanced details
                regular_routes = ferry_data.get('regular_routes', {})
                for route_name, route_info in list(regular_routes.items())[:3]:
                    response += f"""
**{route_name}:**
• Duration: {route_info.get('duration_minutes', 15)} minutes
• Frequency: Every {route_info.get('frequency_minutes', 20)} minutes
• Price: {route_info.get('price_tl', 5.0)} TL
• Next departures: {', '.join(route_info.get('next_departures', [])[:3])}
• Scenic highlights: {', '.join(route_info.get('scenic_highlights', [])[:2])}
• ♿ Fully wheelchair accessible
"""
                
                # Add tourist ferry information
                tourist_ferries = ferry_data.get('tourist_ferries', {})
                if tourist_ferries:
                    response += "\n**Tourist Ferry Tours:**\n"
                    for tour_name, tour_info in tourist_ferries.items():
                        duration = f"{tour_info.get('duration_hours', 2)} hours" if 'duration_hours' in tour_info else f"{tour_info.get('duration_minutes', 90)} minutes"
                        response += f"• **{tour_name.replace('_', ' ').title()}**: {duration}, {tour_info.get('price_tl', 25)} TL\n"
                
                response += f"""
**Practical Information:**
• Payment: İstanbulkart, Cash, Contactless Card
• Onboard: Seating, Toilets, Snack bar (select routes)
• Pets: Small pets in carriers allowed
• Bicycles: Folding bikes allowed

🌅 **Best Experience:** Sunset ferries offer spectacular Bosphorus views
📱 **Apps:** Vapur Saatleri, İstanbul Ulaşım
💡 **Tip:** Ferry rides are the most scenic way to cross between continents!"""
                
                return response
                
            except Exception as e:
                self.logger.error(f"Ferry query error: {e}")
        
        return self._get_fallback_ferry_response()

    async def _handle_walking_specific_query(self, user_input: str, entities: Dict[str, Any], user_profile: Any) -> str:
        """Handle walking-specific queries with detailed route information"""
        districts = entities.get('districts', [])
        
        if len(districts) >= 2 and ENHANCED_TRANSPORT_AVAILABLE and self.comprehensive_processor:
            try:
                walking_data = await self.comprehensive_processor.get_detailed_walking_routes(districts[0], districts[1])
                return self._format_comprehensive_walking_response(walking_data, districts[0], districts[1])
            except Exception as e:
                self.logger.error(f"Walking query error: {e}")
        
        # General walking information
        return """🚶‍♂️ **Walking in Istanbul - Complete Guide**

🗺️ **Popular Walking Routes:**
• **Sultanahmet Circuit**: Blue Mosque → Hagia Sophia → Topkapi → Grand Bazaar (2 hours)
• **Galata Area**: Galata Tower → Galata Bridge → Spice Bazaar (45 minutes)
• **Bosphorus Walk**: Ortaköy → Bebek → Arnavutköy (1.5 hours)
• **İstiklal Street**: Taksim → Galatasaray → Tünel (30 minutes)

♿ **Accessibility Notes:**
• Historic areas: Mixed surfaces, some cobblestones
• Modern districts: Generally wheelchair accessible
• Hills: Istanbul has many steep areas - plan accordingly
• Public toilets: Available at major attractions and transport hubs

🛡️ **Safety & Comfort:**
• Use main pedestrian areas, especially at night
• Carry water, especially in summer (June-August)
• Wear comfortable walking shoes for cobblestones
• Most areas well-lit and regularly patrolled

📱 **Navigation Apps:**
• Google Maps (offline maps available)
• Citymapper Istanbul (public transport integration)
• Maps.me (detailed offline maps)

💡 **Best Walking Times:**
• **Morning**: 8-10 AM (cooler, fewer crowds)
• **Evening**: 4-6 PM (golden hour, pleasant temperatures)
• **Avoid**: Midday summer heat (12-3 PM), rush hours (7-9 AM, 5-7 PM)

🎯 **For specific walking directions, ask**: "Walking route from [A] to [B]" """

    async def _handle_general_transportation_query(self, user_input: str, entities: Dict[str, Any], user_profile: Any) -> str:
        """Handle general transportation queries with comprehensive overview"""
        if ENHANCED_TRANSPORT_AVAILABLE and self.comprehensive_processor:
            try:
                return await _format_general_transportation_response(self.comprehensive_processor)
            except Exception as e:
                self.logger.error(f"General transportation query error: {e}")
        
        # Comprehensive fallback response
        current_time = datetime.now().strftime("%H:%M")
        return f"""🚇 **Istanbul Transportation System - Complete Guide**

📍 **Live Status** (Updated: {current_time})

🚇 **Metro System:**
• **M1A**: Yenikapı ↔ Halkalı | Frequency: 3-6 min | Hours: 06:00-00:30
• **M2**: Vezneciler ↔ Hacıosman | Frequency: 2-5 min | Hours: 06:00-00:30
• **M4**: Kadıköy ↔ Tavşantepe | Frequency: 3-6 min | Hours: 06:00-00:30
• **M11**: Gayrettepe ↔ Istanbul Airport | Frequency: 10-15 min | Hours: 06:00-01:00

🚌 **İETT Bus Network:**
• 800+ routes serving entire city
• 4.5M daily passengers
• 90% wheelchair accessible
• Real-time GPS tracking available

⛴️ **Ferry Services:** Operating normally
• Multiple routes across Bosphorus and Golden Horn
• Scenic transportation option with great views
• Fully wheelchair accessible terminals and vessels

💳 **Payment & Cards:**
• **İstanbulkart**: Universal transport card (highly recommended)
• Contactless payment: Available on most transport
• Transfer discounts: Up to 60% savings with İstanbulkart

♿ **Accessibility:**
• All metro stations wheelchair accessible
• Most buses have wheelchair ramps
• Ferries have dedicated accessible areas
• Audio announcements in Turkish and English

📱 **Essential Apps:**
• **İstanbul Ulaşım**: Official transport app
• **Moovit**: Multi-modal journey planning
• **Citymapper**: Comprehensive navigation
• **Metro İstanbul**: Metro-specific live information

🎯 **For specific routes**: Ask "How to get from [A] to [B]?"
💡 **Pro tip**: Combine metro + tram + ferry for the complete Istanbul experience!

**Most Asked Routes:**
• Airport to Sultanahmet: M11 → M2 → walk (60-75 min)
• Taksim to Sultanahmet: M2 to Vezneciler + 10min walk (25 min)
• Asian to European side: Ferry routes (15-25 min) - most scenic option"""

    def _format_detailed_route_response(self, route_info: Dict[str, Any], start: str, end: str) -> str:
        """Format detailed route response from enhanced system"""
        if 'error' in route_info:
            return route_info['error']
        
        response = f"🚇 **Route: {start} → {end}**\n\n"
        response += f"**Recommended Route:**\n"
        response += f"📍 {route_info.get('recommended_route', 'Route information')}\n"
        response += f"⏱️ **Total Time**: {route_info.get('total_time', 'Unknown')}\n"
        response += f"💰 **Cost**: {route_info.get('cost', 'Standard İstanbulkart rates')}\n\n"
        
        if 'steps' in route_info:
            response += "**Step-by-Step Directions:**\n"
            for i, step in enumerate(route_info['steps'], 1):
                response += f"{i}. {step}\n"
            response += "\n"
        
        if 'walking_directions' in route_info:
            response += f"🚶‍♂️ **Walking Directions**: {route_info['walking_directions']}\n\n"
        
        if 'landmarks_on_walk' in route_info:
            landmarks = ', '.join(route_info['landmarks_on_walk'])
            response += f"🗺️ **Landmarks**: {landmarks}\n\n"
        
        if 'alternative_route' in route_info:
            response += f"**Alternative Route**: {route_info['alternative_route']}\n"
            response += f"⏱️ **Alternative Time**: {route_info.get('alternative_time', 'Unknown')}\n\n"
        
        if 'accessibility' in route_info:
            response += f"♿ **Accessibility**: {route_info['accessibility']}\n\n"
        
        if 'scenic_bonus' in route_info:
            response += f"🌅 **Bonus**: {route_info['scenic_bonus']}\n\n"
        
        response += "💡 **Tip**: Use İstanbulkart for discounted transfers between transport modes"
        
        return response

    def _format_comprehensive_walking_response(self, walking_data: Dict[str, Any], start: str, end: str) -> str:
        """Format comprehensive walking response with all details"""
        if walking_data.get('fallback'):
            return self._get_basic_walking_response(start, end)
        
        basic_info = walking_data.get('basic_info', {})
        
        response = f"🚶‍♂️ **Walking Directions: {start} → {end}**\n\n"
        response += f"📏 **Distance**: {basic_info.get('distance_meters', 0)}m\n"
        response += f"⏱️ **Duration**: {basic_info.get('duration_minutes', 10)} minutes\n"
        response += f"📈 **Elevation Change**: {basic_info.get('elevation_change_meters', 0)}m\n"
        response += f"🎯 **Difficulty**: {basic_info.get('difficulty', 'Moderate')}\n\n"
        
        # Step-by-step directions
        steps = walking_data.get('step_by_step', [])
        if steps:
            response += "🗺️ **Step-by-Step Directions:**\n\n"
            for step_info in steps:
                step_num = step_info.get('step', 1)
                instruction = step_info.get('instruction', '')
                landmark = step_info.get('landmark', '')
                
                response += f"**Step {step_num}**: {instruction}\n"
                if landmark:
                    response += f"   📍 *Landmark*: {landmark}\n"
                response += "\n"
        
        # Accessibility information
        accessibility = walking_data.get('accessibility', {})
        if accessibility:
            response += "♿ **Accessibility:**\n"
            response += f"• Wheelchair friendly: {'Yes' if accessibility.get('wheelchair_friendly') else 'No'}\n"
            response += f"• Surface: {accessibility.get('surface_type', 'Mixed surfaces')}\n"
            
            obstacles = accessibility.get('obstacles')
            if obstacles and obstacles != 'None':
                response += f"• Obstacles: {obstacles}\n"
            
            rest_points = accessibility.get('rest_points', [])
            if rest_points:
                response += f"• Rest points: {', '.join(rest_points)}\n"
            
            response += "\n"
        
        # Points of interest
        poi = walking_data.get('points_of_interest', [])
        if poi:
            response += "🎯 **Points of Interest:**\n"
            for point in poi[:4]:
                response += f"• {point}\n"
            response += "\n"
        
        # Photo opportunities
        photo_ops = walking_data.get('photo_opportunities', [])
        if photo_ops:
            response += "📸 **Photo Opportunities:**\n"
            for photo in photo_ops[:3]:
                response += f"• {photo}\n"
            response += "\n"
        
        response += "💡 **Tip**: Download offline maps and wear comfortable shoes for Istanbul's varied terrain!"
        
        return response

    async def _generate_multi_modal_route(self, start: str, end: str, query_info: Dict[str, Any], user_profile: Any) -> str:
        """Generate multi-modal transportation route"""
        optimization = query_info.get('optimization_preference', 'fastest')
        
        response = f"🚇 **Multi-Modal Route: {start} → {end}**\n\n"
        response += f"🎯 **Optimized for**: {optimization.title()}\n\n"
        
        # Generate route based on common patterns
        if start.lower() == 'taksim' and end.lower() == 'sultanahmet':
            if optimization == 'fastest':
                response += """**Fastest Route (22-25 minutes):**
1. 🚇 Take M2 metro from Taksim to Vezneciler (12 minutes)
2. 🚶‍♂️ Walk from Vezneciler to Sultanahmet (10 minutes)
3. Total cost: 7.67 TL with İstanbulkart

**Walking Route:** Exit Vezneciler station, head south toward Istanbul University, follow signs to Sultanahmet Square"""
            
            elif optimization == 'scenic':
                response += """**Scenic Route (30-35 minutes):**
1. 🚇 Take M2 metro from Taksim to Şişhane (8 minutes)
2. 🚶‍♂️ Walk down to Karaköy (5 minutes)
3. 🚋 Take T1 tram from Karaköy to Sultanahmet (8 minutes)
4. Bonus: Cross historic Galata Bridge and see Golden Horn views
5. Total cost: 15.34 TL with İstanbulkart"""
        
        elif 'airport' in start.lower():
            response += """**Airport to City Route:**
1. 🚇 Take M11 metro from Istanbul Airport to Gayrettepe (35 minutes)
2. 🚇 Transfer to M2 metro at Gayrettepe
3. 🚇 Continue to your destination
4. Total time: 60-75 minutes to city center
5. Cost: 15.34 TL with İstanbulkart"""
        
        else:
            response += f"""**Recommended Multi-Modal Route:**
1. Use combination of metro, tram, and/or ferry
2. İstanbulkart provides seamless transfers with discounts
3. Check İstanbul Ulaşım app for real-time information
4. Consider walking for short distances (under 1km)

**Route Planning Tips:**
• Metro: Fastest for long distances
• Tram: Best for historic areas
• Ferry: Most scenic for cross-Bosphorus travel
• Walking: Often faster than transfers for short hops"""
        
        response += f"\n\n♿ **Accessibility**: All suggested transport modes are wheelchair accessible"
        response += f"\n💡 **Apps**: Use Moovit or Citymapper for live journey planning"
        
        return response

    def _get_fallback_comprehensive_response(self, user_input: str, entities: Dict[str, Any]) -> str:
        """Comprehensive fallback response when all systems fail"""
        return """🚇 **Istanbul Transportation - Essential Information**

**Metro System:**
• M1A: Yenikapı ↔ Halkalı (connects to airport bus)
• M2: Vezneciler ↔ Hacıosman (main tourist areas)
• M4: Kadıköy ↔ Tavşantepe (Asian side)
• M11: Gayrettepe ↔ Istanbul Airport (direct airport access)

**Key Routes for Tourists:**
• To Sultanahmet: M2 to Vezneciler + 10min walk
• To Taksim: M2 direct connection
• To Asian side: Ferry from Eminönü/Karaköy (scenic)
• Airport transfers: M11 to Gayrettepe, then M2

**Payment & Cards:**
• İstanbulkart: Essential for public transport (saves 50%)
• Available at metro stations, some shops
• Transfer discounts between different transport modes

**Apps & Information:**
• İstanbul Ulaşım: Official transport app
• Moovit: Journey planning with live updates
• Citymapper: Comprehensive navigation

💡 **Need specific help?** Ask: "How to get from [location] to [destination]" """

    def _get_fallback_metro_response(self) -> str:
        """Fallback metro response"""
        return """🚇 **Istanbul Metro System**

**Main Lines:**
• **M1A**: Yenikapı ↔ Halkalı | Airport connection via bus
• **M2**: Vezneciler ↔ Hacıosman | Main tourist line
• **M4**: Kadıköy ↔ Tavşantepe | Asian side main line
• **M11**: Gayrettepe ↔ Istanbul Airport | Direct airport service

**Operating Hours:** 06:00 - 00:30 daily
**Frequency:** 2-8 minutes depending on line and time
**Payment:** İstanbulkart required

**Tourist Connections:**
• Sultanahmet: M2 to Vezneciler + walk
• Taksim: M2 direct
• Airport: M11 direct
• Grand Bazaar: M1A to Beyazıt-Kapalıçarşı

**Accessibility:** All stations wheelchair accessible"""

    def _get_fallback_bus_response(self) -> str:
        """Fallback bus response"""
        return """🚌 **İETT Bus Network**

**Overview:**
• 800+ routes serving entire Istanbul
• 4.5 million daily passengers
• Operating hours: 05:30 - 00:30
• Real-time tracking available

**Payment:**
• İstanbulkart: Recommended for transfers
• Contactless cards accepted
• Transfer discounts available

**Accessibility:**
• 90% of fleet wheelchair accessible
• Audio announcements available
• Priority seating for disabled passengers

**Apps:**
• İETT Mobil: Official bus app
• Moovit: Multi-modal planning
• Real-time arrivals and route planning"""

    def _get_fallback_ferry_response(self) -> str:
        """Fallback ferry response"""
        return """⛴️ **Istanbul Ferry Services**

**Popular Routes:**
• Eminönü ↔ Üsküdar: 15 min, every 20 min
• Kabataş ↔ Üsküdar: 20 min, every 15 min
• Eminönü ↔ Kadıköy: 25 min, every 30 min

**Scenic Experience:**
• Best views of Bosphorus and city skyline
• Historic Golden Horn crossing
• Photography opportunities

**Practical Info:**
• Payment: İstanbulkart, cash, contactless
• Fully wheelchair accessible
• Onboard facilities: seating, toilets
• Weather dependent service

**Apps:** Vapur Saatleri, İstanbul Ulaşım"""

    def _get_basic_route_response(self, start: str, end: str) -> str:
        """Basic route response fallback"""
        return f"""🚇 **Route Planning: {start} → {end}**

**General Guidance:**
• Use İstanbul Ulaşım app for live route planning
• Metro provides fastest connections for long distances
• Tram (T1) serves historic peninsula areas
• Ferry offers scenic routes between continents

**Payment:**
• İstanbulkart essential for public transport
• Transfer discounts between different modes
• Single journey: 7.67 TL, transfers from 1.40 TL

**Apps for Route Planning:**
• İstanbul Ulaşım (official)
• Moovit (real-time updates)
• Citymapper (comprehensive navigation)

💡 **For detailed directions, specify your exact starting point and destination**"""

    def _get_basic_walking_response(self, start: str, end: str) -> str:
        """Basic walking response fallback"""
        return f"""🚶‍♂️ **Walking Route: {start} → {end}**

**General Guidance:**
• Average walking speed: 4 km/h on flat terrain
• Istanbul has many hills - plan accordingly
• Historic areas have cobblestone surfaces
• Main pedestrian areas well-lit and safe

**Navigation:**
• Use Google Maps offline functionality
• Follow main streets and boulevards
• Ask locals for directions - generally helpful
• Tourist information points available

**Comfort Tips:**
• Wear comfortable walking shoes
• Carry water, especially in summer
• Use main pedestrian areas at night
• Public toilets at major attractions

💡 **Estimated walking time:** 10-15 minutes per kilometer"""


# Export for integration
__all__ = [
    'TransportationQueryProcessor',
    'LocationParser',
    'TransportationQueryAnalyzer',
    'generate_comprehensive_transportation_response'
]



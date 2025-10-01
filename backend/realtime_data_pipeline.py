#!/usr/bin/env python3
"""
Unified Real-Time Data Pipeline
Synchronizes multiple data sources with freshness validation and cache invalidation
"""

import asyncio
import aiohttp
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Available data sources"""
    GOOGLE_PLACES = "google_places"
    GOOGLE_WEATHER = "google_weather"
    STATIC_DATABASE = "static_database"
    CACHED_DATA = "cached_data"

class DataFreshness(Enum):
    """Data freshness levels"""
    REAL_TIME = "real_time"        # < 5 minutes
    FRESH = "fresh"                # < 1 hour
    CURRENT = "current"            # < 6 hours
    STALE = "stale"                # < 24 hours
    EXPIRED = "expired"            # > 24 hours

@dataclass
class DataSourceInfo:
    """Information about a data source"""
    source: DataSource
    last_updated: datetime
    freshness: DataFreshness
    confidence: float
    error_count: int = 0
    is_available: bool = True

@dataclass
class UnifiedDataEntry:
    """Unified data entry with metadata"""
    data_id: str
    content: Any
    sources: List[DataSourceInfo]
    primary_source: DataSource
    last_validated: datetime
    freshness_threshold: timedelta
    tags: List[str]
    location_context: Optional[str] = None

class RealTimeDataPipeline:
    """Unified pipeline for real-time data management"""
    
    def __init__(self):
        self.data_registry: Dict[str, UnifiedDataEntry] = {}
        self.source_status: Dict[DataSource, DataSourceInfo] = {}
        self.freshness_thresholds = {
            'restaurants': timedelta(hours=6),      # Restaurant data valid for 6 hours
            'museums': timedelta(hours=24),         # Museum data valid for 24 hours
            'transportation': timedelta(hours=2),   # Transport data valid for 2 hours
            'weather': timedelta(minutes=30),       # Weather data valid for 30 minutes
            'events': timedelta(hours=1)            # Events data valid for 1 hour
        }
        
        # Initialize source monitoring
        self._initialize_source_monitoring()
        
        # Data validation rules
        self.validation_rules = {
            'restaurants': self._validate_restaurant_data,
            'museums': self._validate_museum_data,
            'transportation': self._validate_transport_data,
            'weather': self._validate_weather_data
        }
    
    def _initialize_source_monitoring(self):
        """Initialize monitoring for all data sources"""
        for source in DataSource:
            self.source_status[source] = DataSourceInfo(
                source=source,
                last_updated=datetime.now(),
                freshness=DataFreshness.EXPIRED,
                confidence=1.0,
                is_available=True
            )
    
    async def get_unified_data(self, data_type: str, query_params: Dict[str, Any], 
                              location: Optional[str] = None) -> Dict[str, Any]:
        """Get unified data from multiple sources with freshness validation"""
        
        data_id = self._generate_data_id(data_type, query_params, location)
        
        # Check if we have cached data
        if data_id in self.data_registry:
            entry = self.data_registry[data_id]
            if self._is_data_fresh(entry):
                logger.info(f"🎯 Using fresh cached data for {data_type}")
                return self._format_response(entry, from_cache=True)
        
        # Fetch fresh data from available sources
        logger.info(f"🔄 Fetching fresh data for {data_type}")
        fresh_data = await self._fetch_from_sources(data_type, query_params, location)
        
        # Store in registry
        if fresh_data:
            self.data_registry[data_id] = fresh_data
        
        return self._format_response(fresh_data or self.data_registry.get(data_id))
    
    async def _fetch_from_sources(self, data_type: str, query_params: Dict[str, Any], 
                                 location: Optional[str] = None) -> Optional[UnifiedDataEntry]:
        """Fetch data from multiple sources in priority order"""
        
        # Determine source priority based on data type
        source_priority = self._get_source_priority(data_type)
        
        results = []
        errors = []
        
        # Try sources in priority order
        for source in source_priority:
            try:
                if not self.source_status[source].is_available:
                    continue
                
                data = await self._fetch_from_source(source, data_type, query_params, location)
                if data:
                    source_info = DataSourceInfo(
                        source=source,
                        last_updated=datetime.now(),
                        freshness=self._calculate_freshness(datetime.now()),
                        confidence=self._calculate_confidence(source, data)
                    )
                    results.append((data, source_info))
                    
            except Exception as e:
                error_msg = f"Failed to fetch from {source.value}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
                # Update source status
                self.source_status[source].error_count += 1
                if self.source_status[source].error_count > 3:
                    self.source_status[source].is_available = False
        
        if not results:
            logger.error(f"Failed to fetch data from all sources: {errors}")
            return None
        
        # Merge results from multiple sources
        return self._merge_source_results(results, data_type, query_params, location)
    
    def _get_source_priority(self, data_type: str) -> List[DataSource]:
        """Get source priority for different data types"""
        priority_map = {
            'restaurants': [DataSource.GOOGLE_PLACES, DataSource.STATIC_DATABASE],
            'museums': [DataSource.STATIC_DATABASE, DataSource.GOOGLE_PLACES],
            'transportation': [DataSource.STATIC_DATABASE, DataSource.GOOGLE_PLACES],
            'weather': [DataSource.GOOGLE_WEATHER],
            'events': [DataSource.GOOGLE_PLACES, DataSource.STATIC_DATABASE]
        }
        
        return priority_map.get(data_type, [DataSource.GOOGLE_PLACES, DataSource.STATIC_DATABASE])
    
    async def _fetch_from_source(self, source: DataSource, data_type: str, 
                               query_params: Dict[str, Any], 
                               location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch data from a specific source"""
        
        if source == DataSource.GOOGLE_PLACES:
            return await self._fetch_google_places_data(data_type, query_params, location)
        elif source == DataSource.GOOGLE_WEATHER:
            return await self._fetch_weather_data(query_params, location)
        elif source == DataSource.STATIC_DATABASE:
            return await self._fetch_static_data(data_type, query_params, location)
        else:
            return None
    
    async def _fetch_google_places_data(self, data_type: str, query_params: Dict[str, Any], 
                                      location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch data from Google Places API"""
        try:
            # Import here to avoid circular dependencies
            from api_clients.enhanced_google_places import EnhancedGooglePlacesClient
            
            client = EnhancedGooglePlacesClient()
            
            if data_type == 'restaurants':
                result = client.search_restaurants(
                    location=location,
                    keyword=query_params.get('keyword'),
                    min_rating=query_params.get('min_rating')
                )
                
                if result and result.get('status') == 'OK':
                    return {
                        'source': DataSource.GOOGLE_PLACES.value,
                        'data': result,
                        'timestamp': datetime.now().isoformat(),
                        'confidence': 0.9 if result.get('data_source') == 'google_places_api' else 0.6
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Google Places API error: {e}")
            return None
    
    async def _fetch_weather_data(self, query_params: Dict[str, Any], 
                                location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch weather data"""
        try:
            # Simulate weather API call (replace with actual implementation)
            weather_data = {
                'location': location or 'Istanbul',
                'temperature': 22,
                'condition': 'partly_cloudy',
                'humidity': 65,
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'source': DataSource.GOOGLE_WEATHER.value,
                'data': weather_data,
                'timestamp': datetime.now().isoformat(),
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return None
    
    async def _fetch_static_data(self, data_type: str, query_params: Dict[str, Any], 
                               location: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch data from static database"""
        try:
            # Import database modules
            from database import SessionLocal
            from models import Restaurant, Museum
            # TransportationHub may not be defined yet
            try:
                from models import TransportationHub
            except ImportError:
                TransportationHub = None
                logger.warning("TransportationHub model not available")
            
            db = SessionLocal()
            
            if data_type == 'restaurants':
                # Query restaurants from database
                query_filter = True
                if location:
                    # Use location column (not district)
                    query_filter = Restaurant.location.ilike(f"%{location}%")
                
                restaurants = db.query(Restaurant).filter(query_filter).limit(10).all()
                
                restaurant_data = [
                    {
                        'name': r.name or 'Unknown Restaurant',
                        'rating': r.rating or 0.0,
                        'location': r.location or 'Istanbul',
                        'cuisine': r.cuisine or 'Turkish',
                        'description': r.description or 'No description available',
                        'place_id': r.place_id,
                        'source': r.source or 'database'
                    }
                    for r in restaurants
                ]
                
                return {
                    'source': DataSource.STATIC_DATABASE.value,
                    'data': {'results': restaurant_data},
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.7
                }
            
            elif data_type == 'museums' or data_type == 'attractions':
                query_filter = True
                if location:
                    # Use location column (not district)
                    query_filter = Museum.location.ilike(f"%{location}%")
                
                museums = db.query(Museum).filter(query_filter).limit(10).all()
                
                museum_data = [
                    {
                        'name': m.name or 'Unknown Museum',
                        'location': m.location or 'Istanbul',
                        'hours': m.hours or 'Contact for hours',
                        'ticket_price': m.ticket_price or 0.0,
                        'highlights': m.highlights or 'Historical significance',
                        'type': 'museum'
                    }
                    for m in museums
                ]
                
                return {
                    'source': DataSource.STATIC_DATABASE.value,
                    'data': {'results': museum_data},
                    'timestamp': datetime.now().isoformat(),
                    'confidence': 0.8
                }
            
            db.close()
            return None
            
        except Exception as e:
            logger.error(f"Static database error: {e}")
            return None
    
    def _merge_source_results(self, results: List[Tuple[Dict[str, Any], DataSourceInfo]], 
                            data_type: str, query_params: Dict[str, Any], 
                            location: Optional[str] = None) -> UnifiedDataEntry:
        """Merge results from multiple sources"""
        
        # Sort by confidence and freshness
        results.sort(key=lambda x: (x[1].confidence, x[1].freshness.value), reverse=True)
        
        primary_data, primary_source_info = results[0]
        all_sources = [source_info for _, source_info in results]
        
        # Merge data intelligently based on type
        merged_data = self._intelligent_merge(data_type, [data for data, _ in results])
        
        data_id = self._generate_data_id(data_type, query_params, location)
        
        return UnifiedDataEntry(
            data_id=data_id,
            content=merged_data,
            sources=all_sources,
            primary_source=primary_source_info.source,
            last_validated=datetime.now(),
            freshness_threshold=self.freshness_thresholds.get(data_type, timedelta(hours=6)),
            tags=[data_type, location or 'general'],
            location_context=location
        )
    
    def _intelligent_merge(self, data_type: str, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Intelligently merge data from multiple sources"""
        
        if not data_list:
            return {}
        
        # Use the highest confidence source as base
        base_data = data_list[0].copy()
        
        if data_type == 'restaurants':
            # Merge restaurant lists, removing duplicates by name
            all_restaurants = []
            seen_names = set()
            
            for data in data_list:
                restaurants = data.get('data', {}).get('results') or data.get('data', {}).get('restaurants', [])
                for restaurant in restaurants:
                    name = restaurant.get('name', '').lower()
                    if name and name not in seen_names:
                        all_restaurants.append(restaurant)
                        seen_names.add(name)
            
            base_data['data'] = {'results': all_restaurants[:15]}  # Limit to 15 results
            
        elif data_type == 'museums':
            # Similar merging for museums
            all_museums = []
            seen_names = set()
            
            for data in data_list:
                museums = data.get('data', {}).get('results', [])
                for museum in museums:
                    name = museum.get('name', '').lower()
                    if name and name not in seen_names:
                        all_museums.append(museum)
                        seen_names.add(name)
            
            base_data['data'] = {'results': all_museums[:10]}
        
        # Add metadata about merge
        base_data['merged_from'] = [data.get('source') for data in data_list]
        base_data['merge_timestamp'] = datetime.now().isoformat()
        
        return base_data
    
    def _generate_data_id(self, data_type: str, query_params: Dict[str, Any], 
                         location: Optional[str] = None) -> str:
        """Generate unique data ID"""
        key_components = [data_type, str(sorted(query_params.items())), location or '']
        key_string = '|'.join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_data_fresh(self, entry: UnifiedDataEntry) -> bool:
        """Check if data entry is still fresh"""
        age = datetime.now() - entry.last_validated
        return age < entry.freshness_threshold
    
    def _calculate_freshness(self, timestamp: datetime) -> DataFreshness:
        """Calculate data freshness based on timestamp"""
        age = datetime.now() - timestamp
        
        if age < timedelta(minutes=5):
            return DataFreshness.REAL_TIME
        elif age < timedelta(hours=1):
            return DataFreshness.FRESH
        elif age < timedelta(hours=6):
            return DataFreshness.CURRENT
        elif age < timedelta(hours=24):
            return DataFreshness.STALE
        else:
            return DataFreshness.EXPIRED
    
    def _calculate_confidence(self, source: DataSource, data: Dict[str, Any]) -> float:
        """Calculate confidence score for data source"""
        base_confidence = {
            DataSource.GOOGLE_PLACES: 0.9,
            DataSource.GOOGLE_WEATHER: 0.8,
            DataSource.STATIC_DATABASE: 0.7,
            DataSource.CACHED_DATA: 0.6
        }
        
        confidence = base_confidence.get(source, 0.5)
        
        # Adjust based on data quality indicators
        if data.get('data_source') == 'real_api':
            confidence += 0.1
        elif data.get('data_source') == 'mock_data':
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))
    
    def _format_response(self, entry: Optional[UnifiedDataEntry], 
                        from_cache: bool = False) -> Dict[str, Any]:
        """Format response with metadata"""
        if not entry:
            return {
                'success': False,
                'error': 'No data available',
                'sources': [],
                'timestamp': datetime.now().isoformat()
            }
        
        return {
            'success': True,
            'data': entry.content,
            'metadata': {
                'primary_source': entry.primary_source.value,
                'all_sources': [s.source.value for s in entry.sources],
                'freshness': self._calculate_freshness(entry.last_validated).value,
                'confidence': max([s.confidence for s in entry.sources]),
                'from_cache': from_cache,
                'last_updated': entry.last_validated.isoformat(),
                'location_context': entry.location_context
            },
            'timestamp': datetime.now().isoformat()
        }
    
    # Validation methods
    def _validate_restaurant_data(self, data: Dict[str, Any]) -> bool:
        """Validate restaurant data structure"""
        if not data or 'data' not in data:
            return False
        
        results = data['data'].get('results', [])
        if not results:
            return False
        
        # Check if restaurants have required fields
        for restaurant in results[:3]:  # Check first 3
            if not restaurant.get('name'):
                return False
        
        return True
    
    def _validate_museum_data(self, data: Dict[str, Any]) -> bool:
        """Validate museum data structure"""
        return self._validate_restaurant_data(data)  # Similar structure
    
    def _validate_transport_data(self, data: Dict[str, Any]) -> bool:
        """Validate transportation data structure"""
        return bool(data and data.get('data'))
    
    def _validate_weather_data(self, data: Dict[str, Any]) -> bool:
        """Validate weather data structure"""
        if not data or 'data' not in data:
            return False
        
        weather_data = data['data']
        required_fields = ['location', 'temperature', 'condition']
        return all(field in weather_data for field in required_fields)
    
    async def invalidate_data(self, data_type: str, location: Optional[str] = None):
        """Invalidate cached data for specific type/location"""
        keys_to_remove = []
        
        for data_id, entry in self.data_registry.items():
            if (data_type in entry.tags and 
                (not location or entry.location_context == location)):
                keys_to_remove.append(data_id)
        
        for key in keys_to_remove:
            del self.data_registry[key]
            
        logger.info(f"Invalidated {len(keys_to_remove)} entries for {data_type}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status and health information"""
        return {
            'sources': {
                source.value: {
                    'available': info.is_available,
                    'last_updated': info.last_updated.isoformat(),
                    'error_count': info.error_count,
                    'confidence': info.confidence
                }
                for source, info in self.source_status.items()
            },
            'cached_entries': len(self.data_registry),
            'freshness_distribution': self._get_freshness_distribution(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_freshness_distribution(self) -> Dict[str, int]:
        """Get distribution of data freshness levels"""
        distribution = {freshness.value: 0 for freshness in DataFreshness}
        
        for entry in self.data_registry.values():
            freshness = self._calculate_freshness(entry.last_validated)
            distribution[freshness.value] += 1
        
        return distribution

# Global pipeline instance
_data_pipeline = None

def get_data_pipeline() -> RealTimeDataPipeline:
    """Get global data pipeline instance"""
    global _data_pipeline
    if _data_pipeline is None:
        _data_pipeline = RealTimeDataPipeline()
    return _data_pipeline

async def get_unified_restaurant_data(location: str, keyword: Optional[str] = None) -> Dict[str, Any]:
    """Get unified restaurant data"""
    pipeline = get_data_pipeline()
    return await pipeline.get_unified_data(
        'restaurants',
        {'keyword': keyword, 'min_rating': 4.0},
        location
    )

async def get_unified_museum_data(location: str) -> Dict[str, Any]:
    """Get unified museum data"""
    pipeline = get_data_pipeline()
    return await pipeline.get_unified_data('museums', {}, location)

if __name__ == "__main__":
    async def test_pipeline():
        print("🧪 Testing Real-Time Data Pipeline...")
        
        pipeline = RealTimeDataPipeline()
        
        # Test restaurant data
        restaurant_data = await pipeline.get_unified_data(
            'restaurants',
            {'keyword': 'turkish', 'min_rating': 4.0},
            'sultanahmet'
        )
        
        print(f"✅ Restaurant data: {restaurant_data['success']}")
        if restaurant_data['success']:
            print(f"   Sources: {restaurant_data['metadata']['all_sources']}")
            print(f"   Freshness: {restaurant_data['metadata']['freshness']}")
        
        # Test pipeline status
        status = pipeline.get_pipeline_status()
        print(f"📊 Pipeline status: {len(status['sources'])} sources monitored")
        
        print("✅ Real-Time Data Pipeline test complete!")
    
    asyncio.run(test_pipeline())

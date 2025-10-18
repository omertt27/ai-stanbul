#!/usr/bin/env python3
"""
Parallel POI Scoring Service
============================

Implements concurrent POI scoring for faster route optimization.

Features:
- Async/parallel POI evaluation
- Batch ML predictions
- Score caching
- Priority-based scheduling
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ScoringTask:
    """Single POI scoring task"""
    poi: Any
    user_location: Any
    preferences: Dict[str, Any]
    current_time: Any
    transport_mode: str
    task_id: str


@dataclass
class ScoringResult:
    """Result of POI scoring"""
    poi: Any
    score: float
    distance_km: float
    travel_time_minutes: int
    crowding_factor: float
    value_score: float
    computation_time_ms: float


class ParallelPOIScoringService:
    """
    Parallel POI scoring for high-performance route optimization
    
    Performance improvements:
    - Sequential: 10 POIs * 50ms = 500ms
    - Parallel (4 workers): 10 POIs / 4 * 50ms = 125ms (4x faster)
    - With batching: Even faster for ML predictions
    """
    
    def __init__(self, max_workers: int = 4, enable_batching: bool = True):
        """
        Initialize parallel scoring service
        
        Args:
            max_workers: Number of parallel workers (default: 4)
            enable_batching: Enable ML prediction batching
        """
        self.max_workers = max_workers
        self.enable_batching = enable_batching
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.score_cache: Dict[str, ScoringResult] = {}
        logger.info(f"⚡ Parallel POI scoring initialized with {max_workers} workers")
    
    async def score_pois_parallel(
        self,
        pois: List[Any],
        user_location: Any,
        preferences: Dict[str, Any],
        current_time: Any,
        transport_mode: str,
        ml_service: Optional[Any] = None
    ) -> List[ScoringResult]:
        """
        Score multiple POIs in parallel
        
        Args:
            pois: List of POI objects to score
            user_location: User's current location
            preferences: User preferences
            current_time: Current datetime
            transport_mode: Transport mode for scoring
            ml_service: Optional ML service for predictions
            
        Returns:
            List of ScoringResult objects
        """
        start_time = time.time()
        
        # Check cache first
        uncached_pois = []
        cached_results = []
        
        for poi in pois:
            cache_key = self._get_cache_key(poi, user_location, preferences)
            if cache_key in self.score_cache:
                cached_results.append(self.score_cache[cache_key])
            else:
                uncached_pois.append(poi)
        
        logger.debug(f"Cache hits: {len(cached_results)}/{len(pois)}")
        
        if not uncached_pois:
            return cached_results
        
        # Batch ML predictions if enabled
        ml_predictions = {}
        if self.enable_batching and ml_service:
            ml_predictions = await self._batch_ml_predictions(
                uncached_pois, current_time, ml_service
            )
        
        # Create scoring tasks
        tasks = []
        for poi in uncached_pois:
            task = ScoringTask(
                poi=poi,
                user_location=user_location,
                preferences=preferences,
                current_time=current_time,
                transport_mode=transport_mode,
                task_id=f"{poi.poi_id}_{int(time.time() * 1000)}"
            )
            
            # Add ML prediction if available
            ml_pred = ml_predictions.get(poi.poi_id, {})
            
            # Schedule scoring task
            future = asyncio.get_event_loop().run_in_executor(
                self.executor,
                self._score_poi_sync,
                task,
                ml_pred
            )
            tasks.append(future)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Cache results
        for result in results:
            cache_key = self._get_cache_key(result.poi, user_location, preferences)
            self.score_cache[cache_key] = result
        
        # Combine cached and new results
        all_results = cached_results + results
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"⚡ Scored {len(pois)} POIs in {elapsed_ms:.0f}ms ({len(pois)/elapsed_ms*1000:.0f} POIs/sec)")
        
        return all_results
    
    def _score_poi_sync(self, task: ScoringTask, ml_prediction: Dict[str, Any]) -> ScoringResult:
        """
        Synchronous POI scoring (runs in thread pool)
        
        Args:
            task: Scoring task
            ml_prediction: Pre-computed ML prediction
            
        Returns:
            ScoringResult
        """
        start_time = time.time()
        
        # Calculate distance
        distance_km = self._calculate_distance(
            task.user_location.latitude, task.user_location.longitude,
            task.poi.location.lat, task.poi.location.lon
        )
        
        # Estimate travel time (simple model)
        if task.transport_mode == 'walk':
            travel_time_minutes = int(distance_km * 12)  # 5 km/h walking speed
        elif task.transport_mode == 'metro':
            travel_time_minutes = int(distance_km * 2) + 5  # Metro + waiting
        else:
            travel_time_minutes = int(distance_km * 3)  # Mixed transport
        
        # Get crowding factor from ML or default
        crowding_factor = ml_prediction.get('crowding_level', 0.5)
        
        # Calculate value score
        value_score = self._calculate_value_score(
            task.poi, task.preferences, crowding_factor
        )
        
        # Calculate final score (weighted combination)
        score = self._calculate_final_score(
            value_score=value_score,
            distance_km=distance_km,
            crowding_factor=crowding_factor,
            rating=task.poi.rating,
            preferences=task.preferences
        )
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        return ScoringResult(
            poi=task.poi,
            score=score,
            distance_km=distance_km,
            travel_time_minutes=travel_time_minutes,
            crowding_factor=crowding_factor,
            value_score=value_score,
            computation_time_ms=computation_time_ms
        )
    
    async def _batch_ml_predictions(
        self,
        pois: List[Any],
        current_time: Any,
        ml_service: Any
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batch ML predictions for multiple POIs
        
        Args:
            pois: List of POIs
            current_time: Current datetime
            ml_service: ML prediction service
            
        Returns:
            Dict mapping poi_id to prediction
        """
        try:
            # Batch predict crowding for all POIs
            predictions = {}
            
            # Create batch request
            batch_data = []
            for poi in pois:
                batch_data.append({
                    'poi_id': poi.poi_id,
                    'hour': current_time.hour,
                    'day_of_week': current_time.weekday(),
                    'is_weekend': current_time.weekday() >= 5
                })
            
            # Single ML call for all POIs
            if hasattr(ml_service, 'predict_crowding_batch'):
                batch_results = await ml_service.predict_crowding_batch(batch_data)
                for data, result in zip(batch_data, batch_results):
                    predictions[data['poi_id']] = result
            else:
                # Fallback to individual predictions
                for data in batch_data:
                    pred = await ml_service.predict_crowding(
                        data['poi_id'], 
                        current_time
                    )
                    predictions[data['poi_id']] = {'crowding_level': pred}
            
            logger.debug(f"Batch ML predictions: {len(predictions)} POIs")
            return predictions
            
        except Exception as e:
            logger.warning(f"Batch ML prediction failed: {e}")
            return {}
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance"""
        import math
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _calculate_value_score(
        self,
        poi: Any,
        preferences: Dict[str, Any],
        crowding_factor: float
    ) -> float:
        """Calculate POI value score"""
        # Base score from rating
        base_score = poi.rating / 5.0
        
        # Interest match bonus
        interests = preferences.get('interests', [])
        interest_match = 1.0
        if interests and hasattr(poi, 'category'):
            if poi.category.lower() in [i.lower() for i in interests]:
                interest_match = 1.3
        
        # Crowding penalty
        crowding_penalty = 1.0 - (crowding_factor * 0.3)
        
        return base_score * interest_match * crowding_penalty
    
    def _calculate_final_score(
        self,
        value_score: float,
        distance_km: float,
        crowding_factor: float,
        rating: float,
        preferences: Dict[str, Any]
    ) -> float:
        """Calculate final weighted score"""
        # Distance decay (closer is better)
        distance_score = 1.0 / (1.0 + distance_km)
        
        # Crowding penalty
        crowding_score = 1.0 - crowding_factor * 0.4
        
        # Rating score
        rating_score = rating / 5.0
        
        # Weighted combination
        final_score = (
            value_score * 0.4 +
            distance_score * 0.3 +
            crowding_score * 0.2 +
            rating_score * 0.1
        )
        
        return final_score
    
    def _get_cache_key(self, poi: Any, user_location: Any, preferences: Dict[str, Any]) -> str:
        """Generate cache key for POI scoring"""
        # Simple cache key based on POI and rough location
        loc_key = f"{int(user_location.latitude * 100)}_{int(user_location.longitude * 100)}"
        pref_key = "_".join(sorted(preferences.get('interests', [])))
        return f"{poi.poi_id}_{loc_key}_{pref_key}"
    
    def clear_cache(self):
        """Clear scoring cache"""
        self.score_cache.clear()
        logger.info("Scoring cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'max_workers': self.max_workers,
            'enable_batching': self.enable_batching,
            'cache_size': len(self.score_cache),
            'executor_type': type(self.executor).__name__
        }
    
    def shutdown(self):
        """Shutdown executor"""
        self.executor.shutdown(wait=True)
        logger.info("Parallel scoring service shutdown")


# Singleton instance
_scoring_service = None


def get_parallel_scoring_service(max_workers: int = 4) -> ParallelPOIScoringService:
    """Get singleton parallel scoring service"""
    global _scoring_service
    if _scoring_service is None:
        _scoring_service = ParallelPOIScoringService(max_workers=max_workers)
    return _scoring_service

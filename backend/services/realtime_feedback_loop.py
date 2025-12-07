"""
Real-Time Feedback Loop Service
Integrates feedback collection with online learning engine
Provides the main interface for real-time personalization
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from services.feedback_collector import get_feedback_collector, FeedbackEventType
from ml.online_learning import get_online_learning_engine
from services.redis_cache import get_redis_cache
from database import SessionLocal
from models import FeedbackEvent, UserInteractionAggregate

logger = logging.getLogger(__name__)


class RealtimeFeedbackLoop:
    """
    Main service that connects feedback collection with online learning
    Manages the entire real-time personalization pipeline
    """
    
    def __init__(self):
        """Initialize the real-time feedback loop"""
        self.feedback_collector = get_feedback_collector()
        self.online_learning_engine = get_online_learning_engine()
        self.redis_cache = get_redis_cache()
        self.is_running = False
        
        # Register the online learning engine as a feedback handler
        self.feedback_collector.register_handler(self._process_feedback_batch)
        
        logger.info("✅ RealtimeFeedbackLoop initialized")
    
    async def start(self):
        """Start the real-time feedback loop"""
        if self.is_running:
            logger.warning("⚠️ Feedback loop already running")
            return
        
        await self.feedback_collector.start()
        self.is_running = True
        logger.info("🚀 Real-time feedback loop started")
    
    async def stop(self):
        """Stop the real-time feedback loop"""
        if not self.is_running:
            return
        
        await self.feedback_collector.stop()
        self.is_running = False
        logger.info("🛑 Real-time feedback loop stopped")
    
    async def track_view(
        self,
        user_id: str,
        item_id: str,
        item_type: str,
        dwell_time: Optional[float] = None,
        session_id: Optional[str] = None
    ):
        """Track a view event"""
        await self.feedback_collector.collect_view(
            user_id, item_id, item_type, dwell_time, session_id
        )
    
    async def track_click(
        self,
        user_id: str,
        item_id: str,
        item_type: str,
        position: Optional[int] = None,
        session_id: Optional[str] = None
    ):
        """Track a click event"""
        await self.feedback_collector.collect_click(
            user_id, item_id, item_type, position, session_id
        )
    
    async def track_rating(
        self,
        user_id: str,
        item_id: str,
        item_type: str,
        rating: float,
        session_id: Optional[str] = None
    ):
        """Track a rating event"""
        await self.feedback_collector.collect_rating(
            user_id, item_id, item_type, rating, session_id
        )
    
    async def track_save(
        self,
        user_id: str,
        item_id: str,
        item_type: str,
        session_id: Optional[str] = None
    ):
        """Track a save/bookmark event"""
        await self.feedback_collector.collect_save(
            user_id, item_id, item_type, session_id
        )
    
    async def track_conversion(
        self,
        user_id: str,
        item_id: str,
        item_type: str,
        session_id: Optional[str] = None
    ):
        """Track a conversion event (user visited the place)"""
        await self.feedback_collector.collect_conversion(
            user_id, item_id, item_type, session_id
        )
    
    async def _process_feedback_batch(self, events: List[Dict[str, Any]]):
        """
        Process a batch of feedback events
        This is called by the feedback collector when events are flushed
        """
        if not events:
            return
            
        try:
            # Store events in database
            await self._store_events(events)
            
            # Update online learning models
            await self.online_learning_engine.process_feedback_batch(events)
            
            # Update interaction aggregates
            await self._update_aggregates(events)
            
            # Invalidate recommendation caches for affected users
            affected_users = set(event['user_id'] for event in events)
            for user_id in affected_users:
                self.redis_cache.invalidate_user_recommendations(user_id)
            
            logger.info(f"✅ Processed batch of {len(events)} events, invalidated {len(affected_users)} user caches")
        except Exception as e:
            logger.error(f"❌ Error processing feedback batch: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    async def _store_events(self, events: List[Dict[str, Any]]):
        """Store events in database"""
        # Run database operations in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._store_events_sync, events)
    
    def _store_events_sync(self, events: List[Dict[str, Any]]):
        """Synchronous database storage"""
        db = SessionLocal()
        try:
            for event in events:
                db_event = FeedbackEvent(
                    user_id=event['user_id'],
                    session_id=event.get('session_id'),
                    event_type=event['event_type'],
                    item_id=event['item_id'],
                    item_type=event['item_type'],
                    event_metadata=event.get('metadata', {}),
                    timestamp=event['timestamp'],
                    processed=False
                )
                db.add(db_event)
            
            db.commit()
            logger.debug(f"💾 Stored {len(events)} events in database")
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error storing events: {str(e)}")
            raise  # Re-raise to help debugging
        finally:
            db.close()
    
    async def _update_aggregates(self, events: List[Dict[str, Any]]):
        """Update user interaction aggregates"""
        # Run database operations in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._update_aggregates_sync, events)
    
    def _update_aggregates_sync(self, events: List[Dict[str, Any]]):
        """Synchronous aggregate updates"""
        db = SessionLocal()
        try:
            # Group events by user and item type
            user_item_events = {}
            for event in events:
                key = (event['user_id'], event['item_type'])
                if key not in user_item_events:
                    user_item_events[key] = []
                user_item_events[key].append(event)
            
            # Update aggregates for each user-item_type combination
            for (user_id, item_type), user_events in user_item_events.items():
                aggregate = db.query(UserInteractionAggregate).filter(
                    UserInteractionAggregate.user_id == user_id,
                    UserInteractionAggregate.item_type == item_type
                ).first()
                
                if not aggregate:
                    aggregate = UserInteractionAggregate(
                        user_id=user_id,
                        item_type=item_type,
                        view_count=0,
                        click_count=0,
                        save_count=0,
                        rating_count=0,
                        conversion_count=0,
                        avg_rating=0.0,
                        avg_dwell_time=0.0,
                        click_through_rate=0.0,
                        conversion_rate=0.0,
                        recency_score=0.0
                    )
                    db.add(aggregate)
                
                # Update counts
                for event in user_events:
                    event_type = event['event_type']
                    if event_type == 'view':
                        aggregate.view_count += 1
                    elif event_type == 'click':
                        aggregate.click_count += 1
                    elif event_type == 'save':
                        aggregate.save_count += 1
                    elif event_type == 'rating':
                        aggregate.rating_count += 1
                        # Update average rating
                        rating = event.get('metadata', {}).get('rating', 0)
                        if aggregate.avg_rating == 0:
                            aggregate.avg_rating = rating
                        else:
                            # Incremental average
                            aggregate.avg_rating = (
                                (aggregate.avg_rating * (aggregate.rating_count - 1) + rating) /
                                aggregate.rating_count
                            )
                    elif event_type == 'conversion':
                        aggregate.conversion_count += 1
                    
                    # Update dwell time
                    dwell_time = event.get('metadata', {}).get('dwell_time')
                    if dwell_time:
                        if aggregate.avg_dwell_time == 0:
                            aggregate.avg_dwell_time = dwell_time
                        else:
                            # Incremental average
                            total_views = aggregate.view_count
                            aggregate.avg_dwell_time = (
                                (aggregate.avg_dwell_time * (total_views - 1) + dwell_time) /
                                total_views
                            )
                
                # Update rates
                if aggregate.view_count > 0:
                    aggregate.click_through_rate = aggregate.click_count / aggregate.view_count
                if aggregate.click_count > 0:
                    aggregate.conversion_rate = aggregate.conversion_count / aggregate.click_count
                
                # Update last interaction time
                aggregate.last_interaction = datetime.now()
                
                # Cache the updated aggregate (1-minute TTL)
                aggregate_data = {
                    'view_count': aggregate.view_count,
                    'click_count': aggregate.click_count,
                    'save_count': aggregate.save_count,
                    'rating_count': aggregate.rating_count,
                    'conversion_count': aggregate.conversion_count,
                    'avg_rating': aggregate.avg_rating,
                    'avg_dwell_time': aggregate.avg_dwell_time,
                    'click_through_rate': aggregate.click_through_rate,
                    'conversion_rate': aggregate.conversion_rate,
                    'recency_score': aggregate.recency_score,
                    'last_interaction': aggregate.last_interaction.isoformat()
                }
                self.redis_cache.cache_user_aggregate(
                    user_id, item_type, aggregate_data, ttl_seconds=60
                )
            
            db.commit()
            logger.debug(f"📊 Updated aggregates for {len(user_item_events)} user-item combinations")
        except Exception as e:
            db.rollback()
            logger.error(f"❌ Error updating aggregates: {str(e)}")
            raise  # Re-raise to help debugging
        finally:
            db.close()
    
    def get_recommendations(
        self,
        user_id: str,
        candidate_items: List[str],
        item_type: str = "hidden_gem",
        top_k: int = 10,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations using online learning
        
        Args:
            user_id: User identifier
            candidate_items: List of candidate item IDs
            item_type: Type of items (for cache key)
            top_k: Number of recommendations to return
            use_cache: Whether to use Redis cache
        
        Returns:
            List of recommendation dictionaries with item_id and score
        """
        # Try cache first
        if use_cache:
            cached_recs = self.redis_cache.get_recommendations(user_id, item_type)
            if cached_recs:
                logger.debug(f"✅ Cache hit for user {user_id} recommendations")
                return cached_recs[:top_k]
        
        # Generate fresh recommendations
        recommendations = self.online_learning_engine.get_recommendations(
            user_id, candidate_items, top_k
        )
        
        result = [
            {'item_id': item_id, 'score': float(score)}
            for item_id, score in recommendations
        ]
        
        # Cache the recommendations (5-minute TTL)
        if use_cache:
            self.redis_cache.cache_recommendations(
                user_id, item_type, result, ttl_seconds=300
            )
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        return {
            'collector_metrics': self.feedback_collector.get_metrics(),
            'learning_engine_metrics': self.online_learning_engine.get_metrics(),
            'cache_stats': self.redis_cache.get_stats(),
            'is_running': self.is_running
        }


# Global instance
_realtime_feedback_loop = None


def get_realtime_feedback_loop() -> RealtimeFeedbackLoop:
    """Get or create the global realtime feedback loop instance"""
    global _realtime_feedback_loop
    if _realtime_feedback_loop is None:
        _realtime_feedback_loop = RealtimeFeedbackLoop()
    return _realtime_feedback_loop


# Example usage
if __name__ == "__main__":
    import json
    
    async def main():
        loop = RealtimeFeedbackLoop()
        await loop.start()
        
        # Simulate user interactions
        await loop.track_view("user1", "place1", "hidden_gem", dwell_time=15.5)
        await loop.track_click("user1", "place1", "hidden_gem", position=1)
        await loop.track_rating("user1", "place1", "hidden_gem", rating=4.5)
        
        # Wait for processing
        await asyncio.sleep(6)
        
        # Get recommendations
        candidates = ["place1", "place2", "place3", "place4"]
        recs = loop.get_recommendations("user1", candidates, top_k=3)
        print("\nRecommendations for user1:")
        for rec in recs:
            print(f"  {rec['item_id']}: {rec['score']:.4f}")
        
        # Get metrics
        print("\nMetrics:", json.dumps(loop.get_metrics(), indent=2))
        
        await loop.stop()
    
    asyncio.run(main())

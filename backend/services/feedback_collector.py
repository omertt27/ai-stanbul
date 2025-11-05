"""
Real-Time Feedback Collection Service
Captures user interactions and feedback events for online learning
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json
from collections import deque
import asyncio

logger = logging.getLogger(__name__)


class FeedbackEventType(Enum):
    """Types of feedback events"""
    VIEW = "view"
    CLICK = "click"
    SAVE = "save"
    SHARE = "share"
    RATING = "rating"
    DWELL_TIME = "dwell_time"
    INTERACTION = "interaction"
    REJECTION = "rejection"
    CONVERSION = "conversion"  # User actually visited the place


class FeedbackCollector:
    """
    Collects and buffers user feedback events for real-time processing
    Implements a high-throughput, low-latency event collection system
    """
    
    def __init__(self, buffer_size: int = 1000, flush_interval: int = 5):
        """
        Initialize feedback collector
        
        Args:
            buffer_size: Maximum number of events to buffer before flushing
            flush_interval: Seconds between automatic buffer flushes
        """
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.event_handlers = []
        self._flush_task = None
        
        # Metrics for monitoring
        self.total_events = 0
        self.events_by_type = {}
        self.last_flush_time = datetime.now()
        
        logger.info(f"✅ FeedbackCollector initialized (buffer_size={buffer_size}, flush_interval={flush_interval}s)")
    
    async def start(self):
        """Start the automatic flush background task"""
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self._auto_flush())
            logger.info("🚀 Feedback collector auto-flush started")
    
    async def stop(self):
        """Stop the automatic flush task and flush remaining events"""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None
        
        # Flush remaining events
        await self.flush()
        logger.info("🛑 Feedback collector stopped")
    
    async def collect_event(
        self,
        user_id: str,
        event_type: FeedbackEventType,
        item_id: str,
        item_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Collect a single feedback event
        
        Args:
            user_id: User identifier
            event_type: Type of feedback event
            item_id: ID of the item (place, restaurant, etc.)
            item_type: Type of item (hidden_gem, restaurant, attraction, etc.)
            metadata: Additional event metadata (rating value, dwell time, etc.)
            session_id: Session identifier for tracking user journey
            timestamp: Event timestamp (defaults to now)
        
        Returns:
            bool: True if event was successfully collected
        """
        try:
            event = {
                'user_id': user_id,
                'event_type': event_type.value,
                'item_id': item_id,
                'item_type': item_type,
                'metadata': metadata or {},
                'session_id': session_id,
                'timestamp': timestamp or datetime.now()
            }
            
            self.buffer.append(event)
            self.total_events += 1
            
            # Update metrics
            event_type_str = event_type.value
            self.events_by_type[event_type_str] = self.events_by_type.get(event_type_str, 0) + 1
            
            # Auto-flush if buffer is full
            if len(self.buffer) >= self.buffer_size:
                await self.flush()
            
            logger.debug(f"📝 Collected event: {event_type.value} for item {item_id} by user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error collecting event: {str(e)}")
            return False
    
    async def collect_view(self, user_id: str, item_id: str, item_type: str, 
                          dwell_time: Optional[float] = None, session_id: Optional[str] = None):
        """Convenience method to collect a view event"""
        metadata = {'dwell_time': dwell_time} if dwell_time else {}
        return await self.collect_event(
            user_id, FeedbackEventType.VIEW, item_id, item_type, 
            metadata, session_id
        )
    
    async def collect_click(self, user_id: str, item_id: str, item_type: str, 
                           position: Optional[int] = None, session_id: Optional[str] = None):
        """Convenience method to collect a click event"""
        metadata = {'position': position} if position is not None else {}
        return await self.collect_event(
            user_id, FeedbackEventType.CLICK, item_id, item_type, 
            metadata, session_id
        )
    
    async def collect_rating(self, user_id: str, item_id: str, item_type: str, 
                            rating: float, session_id: Optional[str] = None):
        """Convenience method to collect a rating event"""
        metadata = {'rating': rating}
        return await self.collect_event(
            user_id, FeedbackEventType.RATING, item_id, item_type, 
            metadata, session_id
        )
    
    async def collect_save(self, user_id: str, item_id: str, item_type: str, 
                          session_id: Optional[str] = None):
        """Convenience method to collect a save/bookmark event"""
        return await self.collect_event(
            user_id, FeedbackEventType.SAVE, item_id, item_type, 
            None, session_id
        )
    
    async def collect_conversion(self, user_id: str, item_id: str, item_type: str,
                                session_id: Optional[str] = None):
        """Convenience method to collect a conversion event (user visited the place)"""
        return await self.collect_event(
            user_id, FeedbackEventType.CONVERSION, item_id, item_type,
            None, session_id
        )
    
    def register_handler(self, handler_func):
        """
        Register a handler function to be called on buffer flush
        Handler should accept a list of events
        """
        self.event_handlers.append(handler_func)
        logger.info(f"✅ Registered event handler: {handler_func.__name__}")
    
    async def flush(self):
        """Flush buffered events to all registered handlers"""
        if not self.buffer:
            return
        
        events = list(self.buffer)
        self.buffer.clear()
        
        logger.info(f"🔄 Flushing {len(events)} events to {len(self.event_handlers)} handlers")
        
        # Call all registered handlers
        for handler in self.event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(events)
                else:
                    handler(events)
            except Exception as e:
                logger.error(f"❌ Error in event handler {handler.__name__}: {str(e)}")
        
        self.last_flush_time = datetime.now()
    
    async def _auto_flush(self):
        """Background task for automatic buffer flushing"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                if self.buffer:
                    await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in auto-flush: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collector metrics"""
        return {
            'total_events': self.total_events,
            'events_by_type': self.events_by_type,
            'buffer_size': len(self.buffer),
            'buffer_capacity': self.buffer_size,
            'last_flush_time': self.last_flush_time.isoformat(),
            'handlers_registered': len(self.event_handlers)
        }


# Global instance
_feedback_collector = None


def get_feedback_collector() -> FeedbackCollector:
    """Get or create the global feedback collector instance"""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
    return _feedback_collector


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def example_handler(events: List[Dict]):
        print(f"Processing {len(events)} events")
        for event in events:
            print(f"  - {event['event_type']}: {event['item_id']} by {event['user_id']}")
    
    async def main():
        collector = FeedbackCollector(buffer_size=5, flush_interval=2)
        collector.register_handler(example_handler)
        await collector.start()
        
        # Simulate events
        await collector.collect_view("user123", "place456", "hidden_gem", dwell_time=15.5)
        await collector.collect_click("user123", "place456", "hidden_gem", position=1)
        await collector.collect_rating("user123", "place456", "hidden_gem", rating=4.5)
        
        await asyncio.sleep(3)  # Wait for auto-flush
        
        await collector.stop()
        print("\nMetrics:", json.dumps(collector.get_metrics(), indent=2))
    
    asyncio.run(main())

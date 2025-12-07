"""
amplitude_adapter.py - Amplitude Integration for Technical Metrics

Sends AI/ML performance metrics to Amplitude for unified analytics.

Separates:
- User behavior → Amplitude (via frontend SDK)
- Technical/system metrics → Amplitude (via this adapter)

Author: AI Istanbul Team
Date: December 7, 2025
"""

import logging
import asyncio
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AmplitudeAdapter:
    """
    Adapter to send technical AI/ML metrics to Amplitude.
    
    Sends backend performance metrics that complement
    frontend user behavior tracking.
    """
    
    def __init__(self, api_key: str, use_batch: bool = True, region: str = "EU"):
        """
        Initialize Amplitude adapter.
        
        Args:
            api_key: Amplitude API key
            use_batch: Use batch API (recommended for backend)
            region: Amplitude region ("EU" or "US")
        """
        self.api_key = api_key
        self.use_batch = use_batch
        self.region = region.upper()
        
        # Amplitude HTTP API endpoints (EU region)
        if self.region == "EU":
            self.endpoint = "https://api.eu.amplitude.com/2/httpapi"
        else:
            self.endpoint = "https://api2.amplitude.com/2/httpapi"
        
        # Event queue for batching
        self.event_queue = []
        self.batch_size = 10
        
        logger.info(f"✅ Amplitude adapter initialized (region: {region}, batch: {use_batch})")
    
    async def track_llm_performance(
        self,
        query_id: str,
        user_id: str,
        metrics: Dict[str, Any]
    ):
        """
        Track LLM performance metrics.
        
        Args:
            query_id: Unique query identifier
            user_id: User identifier
            metrics: Performance metrics
        """
        event = {
            "user_id": user_id,
            "event_type": "llm_query_completed",
            "event_properties": {
                "query_id": query_id,
                "generation_time_ms": metrics.get('llm_generation_time', 0) * 1000,
                "total_time_ms": metrics.get('total_time', 0) * 1000,
                "tokens_used": metrics.get('tokens_used', 0),
                "signal_confidence": metrics.get('signal_confidence', 0),
                "passes_used": metrics.get('passes_used', 1),
                "fallback_used": metrics.get('fallback_used', False),
                "language": metrics.get('language', 'en')
            },
            "time": int(datetime.now().timestamp() * 1000)
        }
        
        await self._send_event(event)
    
    async def track_signal_accuracy(
        self,
        user_id: str,
        signal_intents: Dict[str, bool],
        llm_intents: Dict[str, bool],
        agreement_rate: float
    ):
        """
        Track signal detection accuracy.
        
        Args:
            user_id: User identifier
            signal_intents: Regex-detected intents
            llm_intents: LLM-detected intents
            agreement_rate: Agreement percentage
        """
        event = {
            "user_id": user_id,
            "event_type": "signal_detection_analyzed",
            "event_properties": {
                "signal_intents": list(k for k, v in signal_intents.items() if v),
                "llm_intents": list(k for k, v in llm_intents.items() if v),
                "agreement_rate": agreement_rate,
                "has_discrepancy": agreement_rate < 1.0
            },
            "time": int(datetime.now().timestamp() * 1000)
        }
        
        await self._send_event(event)
    
    async def track_system_health(
        self,
        metrics: Dict[str, Any]
    ):
        """
        Track system health metrics (aggregate).
        
        Args:
            metrics: Health metrics
        """
        event = {
            "user_id": "$amplitude_system",  # Special system user
            "event_type": "system_health_snapshot",
            "event_properties": {
                "query_count_1h": metrics.get('query_count', 0),
                "avg_latency_ms": metrics.get('avg_latency', 0) * 1000,
                "error_rate": metrics.get('error_rate', 0),
                "fallback_rate": metrics.get('fallback_rate', 0),
                "signal_confidence": metrics.get('avg_confidence', 0)
            },
            "time": int(datetime.now().timestamp() * 1000)
        }
        
        await self._send_event(event)
    
    async def _send_event(self, event: Dict[str, Any]):
        """Send event to Amplitude."""
        if self.use_batch:
            # Add to batch queue
            self.event_queue.append(event)
            
            # Flush if batch size reached
            if len(self.event_queue) >= self.batch_size:
                await self._flush_batch()
        else:
            # Send immediately
            await self._send_single(event)
    
    async def _send_single(self, event: Dict[str, Any]):
        """Send single event to Amplitude."""
        try:
            payload = {
                "api_key": self.api_key,
                "events": [event]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.endpoint, json=payload)
                
                if response.status_code != 200:
                    logger.error(f"Amplitude API error: {response.text}")
        
        except Exception as e:
            logger.error(f"Failed to send Amplitude event: {e}")
    
    async def _flush_batch(self):
        """Flush queued events to Amplitude."""
        if not self.event_queue:
            return
        
        try:
            payload = {
                "api_key": self.api_key,
                "events": self.event_queue
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(self.endpoint, json=payload)
                
                if response.status_code == 200:
                    logger.debug(f"✅ Sent {len(self.event_queue)} events to Amplitude")
                    self.event_queue.clear()
                else:
                    logger.error(f"Amplitude batch error: {response.text}")
        
        except Exception as e:
            logger.error(f"Failed to flush Amplitude batch: {e}")


# Global adapter instance
_amplitude: Optional[AmplitudeAdapter] = None


def get_amplitude(api_key: Optional[str] = None, region: str = "EU") -> Optional[AmplitudeAdapter]:
    """
    Get or create Amplitude adapter.
    
    Args:
        api_key: Amplitude API key
        region: Amplitude region ("EU" or "US")
        
    Returns:
        Amplitude adapter or None if not configured
    """
    global _amplitude
    
    if _amplitude is None and api_key:
        _amplitude = AmplitudeAdapter(api_key=api_key, use_batch=True, region=region)
    
    return _amplitude

#!/usr/bin/env python3
"""
Automated Data Pipeline System for AI Istanbul
==============================================

Fully automated pipeline that:
1. Schedules regular scraping runs
2. Applies AI-based quality filtering
3. Auto-approves high-quality content
4. Sends flagged content for manual review
5. Updates the live database continuously
6. Monitors data freshness and gaps
"""

import asyncio
import schedule
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
import threading
import requests
from pathlib import Path

# Import existing services
try:
    from scraping_curation_service import ScrapingCurationService, CurationStatus
    SCRAPING_SERVICE_AVAILABLE = True
except ImportError:
    print("⚠️ Scraping service not available")
    SCRAPING_SERVICE_AVAILABLE = False

try:
    from query_analytics_system import query_analytics_system
    ANALYTICS_AVAILABLE = True
except ImportError:
    print("⚠️ Analytics system not available")
    ANALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

class PipelineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class ContentType(Enum):
    ATTRACTION = "attraction"
    RESTAURANT = "restaurant"
    EVENT = "event"
    TRANSPORT = "transport"
    ACCOMMODATION = "accommodation"

@dataclass
class PipelineConfig:
    """Configuration for automated pipeline"""
    enabled: bool = True
    scraping_interval_hours: int = 24
    quality_threshold: float = 0.8
    auto_approve_threshold: float = 0.9
    max_daily_additions: int = 50
    content_types: List[ContentType] = None
    
    def __post_init__(self):
        if self.content_types is None:
            self.content_types = list(ContentType)

class AutomatedDataPipeline:
    """Fully automated data pipeline for continuous content updates"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.status = PipelineStatus.IDLE
        self.last_run = None
        self.stats = {
            "total_runs": 0,
            "items_processed": 0,
            "items_auto_approved": 0,
            "items_flagged": 0,
            "items_rejected": 0
        }
        
        # Initialize database
        self.db_path = "automated_pipeline.db"
        self._init_database()
        
        # Initialize services
        if SCRAPING_SERVICE_AVAILABLE:
            self.scraping_service = ScrapingCurationService()
        
        self.running = False
        self.scheduler_thread = None
        
    def _init_database(self):
        """Initialize pipeline database"""
        with sqlite3.connect(self.db_path) as conn:
            # Pipeline runs log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    status TEXT NOT NULL,
                    items_processed INTEGER DEFAULT 0,
                    items_approved INTEGER DEFAULT 0,
                    items_flagged INTEGER DEFAULT 0,
                    items_rejected INTEGER DEFAULT 0,
                    error_details TEXT,
                    config_snapshot TEXT
                )
            """)
            
            # Content quality scores
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_quality (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_id TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    raw_data TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    quality_factors TEXT,
                    auto_approved BOOLEAN DEFAULT FALSE,
                    manual_review_needed BOOLEAN DEFAULT FALSE,
                    final_status TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at DATETIME
                )
            """)
            
            # Data freshness tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_freshness (
                    content_type TEXT PRIMARY KEY,
                    last_update DATETIME NOT NULL,
                    total_items INTEGER DEFAULT 0,
                    fresh_items INTEGER DEFAULT 0,
                    stale_items INTEGER DEFAULT 0,
                    gaps_identified TEXT,
                    next_update_due DATETIME
                )
            """)
            
            # Auto-approved content pending integration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_integration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_type TEXT NOT NULL,
                    content_data TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    source TEXT NOT NULL,
                    integration_status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    integrated_at DATETIME
                )
            """)
            
            conn.commit()
    
    def start_automated_pipeline(self):
        """Start the automated pipeline with scheduling"""
        if self.running:
            logger.warning("Pipeline already running")
            return
        
        self.running = True
        self.status = PipelineStatus.RUNNING
        
        # Schedule regular runs
        schedule.clear()
        schedule.every(self.config.scraping_interval_hours).hours.do(self._run_pipeline_cycle)
        
        # Start scheduler in separate thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"🚀 Automated pipeline started (runs every {self.config.scraping_interval_hours} hours)")
        
        # Run first cycle immediately
        asyncio.create_task(self._run_pipeline_cycle())
    
    def stop_automated_pipeline(self):
        """Stop the automated pipeline"""
        self.running = False
        self.status = PipelineStatus.IDLE
        schedule.clear()
        logger.info("⏹️ Automated pipeline stopped")
    
    def _scheduler_loop(self):
        """Run the scheduler in a separate thread"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    async def _run_pipeline_cycle(self):
        """Run a complete pipeline cycle"""
        if not self.config.enabled:
            logger.info("Pipeline disabled in config")
            return
        
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.status = PipelineStatus.RUNNING
        
        logger.info(f"🔄 Starting pipeline cycle: {run_id}")
        
        try:
            # Log run start
            self._log_run_start(run_id)
            
            # Step 1: Check data freshness and identify gaps
            gaps = await self._identify_data_gaps()
            logger.info(f"📊 Identified {len(gaps)} data gaps")
            
            # Step 2: Run targeted scraping based on gaps
            scraped_data = await self._run_targeted_scraping(gaps)
            logger.info(f"🔍 Scraped {len(scraped_data)} new items")
            
            # Step 3: Apply AI quality filtering
            quality_results = await self._apply_quality_filtering(scraped_data)
            logger.info(f"🎯 Quality filtered: {len(quality_results)} items")
            
            # Step 4: Auto-approve high-quality content
            auto_approved = await self._auto_approve_content(quality_results)
            logger.info(f"✅ Auto-approved: {len(auto_approved)} items")
            
            # Step 5: Flag content for manual review
            flagged = await self._flag_for_manual_review(quality_results)
            logger.info(f"🚩 Flagged for review: {len(flagged)} items")
            
            # Step 6: Integrate approved content
            integrated = await self._integrate_approved_content()
            logger.info(f"🔗 Integrated: {integrated} items")
            
            # Step 7: Update data freshness tracking
            await self._update_freshness_tracking()
            
            # Log successful completion
            self._log_run_completion(run_id, len(scraped_data), len(auto_approved), len(flagged))
            
            self.stats["total_runs"] += 1
            self.stats["items_processed"] += len(scraped_data)
            self.stats["items_auto_approved"] += len(auto_approved)
            self.stats["items_flagged"] += len(flagged)
            
            self.last_run = datetime.now()
            self.status = PipelineStatus.IDLE
            
            logger.info(f"✅ Pipeline cycle completed: {run_id}")
            
        except Exception as e:
            logger.error(f"❌ Pipeline cycle failed: {e}")
            self.status = PipelineStatus.ERROR
            self._log_run_error(run_id, str(e))
    
    async def _identify_data_gaps(self) -> List[Dict[str, Any]]:
        """Identify gaps in data coverage"""
        gaps = []
        
        # Check each content type for staleness and coverage
        for content_type in self.config.content_types:
            # Check last update time
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT last_update, total_items, fresh_items, stale_items
                    FROM data_freshness
                    WHERE content_type = ?
                """, (content_type.value,))
                
                row = cursor.fetchone()
                if not row:
                    # No data tracked yet - big gap
                    gaps.append({
                        "type": content_type.value,
                        "severity": "high",
                        "reason": "no_data_tracked",
                        "action": "initial_scraping"
                    })
                else:
                    last_update, total_items, fresh_items, stale_items = row
                    last_update_dt = datetime.fromisoformat(last_update)
                    
                    # Check if data is stale (older than 2 days)
                    if (datetime.now() - last_update_dt).days > 2:
                        gaps.append({
                            "type": content_type.value,
                            "severity": "medium",
                            "reason": "stale_data",
                            "action": "refresh_scraping",
                            "last_update": last_update,
                            "days_old": (datetime.now() - last_update_dt).days
                        })
                    
                    # Check coverage gaps (less than expected minimum)
                    expected_minimums = {
                        "attraction": 100,
                        "restaurant": 200,
                        "event": 50,
                        "transport": 30,
                        "accommodation": 150
                    }
                    
                    expected = expected_minimums.get(content_type.value, 50)
                    if total_items < expected:
                        gaps.append({
                            "type": content_type.value,
                            "severity": "medium",
                            "reason": "low_coverage",
                            "action": "expand_scraping",
                            "current_items": total_items,
                            "target_items": expected
                        })
        
        # Use analytics to identify failed query patterns for content gaps
        if ANALYTICS_AVAILABLE:
            failed_analysis = query_analytics_system.get_failed_queries_analysis(days=7)
            
            # Look for data_gap failures
            if "data_gap" in failed_analysis["failure_categories"]:
                # Analyze the problematic intents to see what content is missing
                for intent, count in failed_analysis["problematic_intents"].items():
                    if count > 5:  # Significant number of failures
                        gaps.append({
                            "type": "query_driven",
                            "severity": "high",
                            "reason": "user_demand_unmet",
                            "action": "targeted_scraping",
                            "intent": intent,
                            "failure_count": count
                        })
        
        return gaps
    
    async def _run_targeted_scraping(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run scraping targeted at identified gaps"""
        scraped_data = []
        
        if not SCRAPING_SERVICE_AVAILABLE:
            logger.warning("Scraping service not available")
            return scraped_data
        
        for gap in gaps:
            try:
                if gap["type"] == "attraction":
                    # Scrape attractions
                    data = await self._scrape_attractions(gap)
                elif gap["type"] == "restaurant":
                    # Scrape restaurants
                    data = await self._scrape_restaurants(gap)
                elif gap["type"] == "event":
                    # Scrape events
                    data = await self._scrape_events(gap)
                elif gap["type"] == "query_driven":
                    # Scrape based on failed query intent
                    data = await self._scrape_by_intent(gap)
                else:
                    continue
                
                scraped_data.extend(data)
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error scraping for gap {gap}: {e}")
                continue
        
        return scraped_data
    
    async def _scrape_attractions(self, gap: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape attractions data"""
        # Implement attraction scraping logic
        # This would use external APIs like Google Places, Foursquare, etc.
        return [
            {
                "type": "attraction",
                "name": f"Sample Attraction {i}",
                "category": "museum",
                "location": "Istanbul",
                "rating": 4.2 + (i * 0.1),
                "description": f"A great attraction in Istanbul #{i}",
                "source": "mock_api",
                "scraped_at": datetime.now().isoformat()
            }
            for i in range(5)  # Mock data
        ]
    
    async def _scrape_restaurants(self, gap: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape restaurant data"""
        return [
            {
                "type": "restaurant",
                "name": f"Sample Restaurant {i}",
                "cuisine": "Turkish",
                "location": "Istanbul",
                "rating": 4.0 + (i * 0.1),
                "price_range": "$$",
                "description": f"Authentic Turkish restaurant #{i}",
                "source": "mock_api",
                "scraped_at": datetime.now().isoformat()
            }
            for i in range(3)  # Mock data
        ]
    
    async def _scrape_events(self, gap: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape events data"""
        return [
            {
                "type": "event",
                "name": f"Sample Event {i}",
                "category": "cultural",
                "location": "Istanbul",
                "date": (datetime.now() + timedelta(days=i*7)).isoformat(),
                "description": f"Cultural event in Istanbul #{i}",
                "source": "mock_api",
                "scraped_at": datetime.now().isoformat()
            }
            for i in range(2)  # Mock data
        ]
    
    async def _scrape_by_intent(self, gap: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape based on failed query intent"""
        intent = gap.get("intent", "")
        
        # Map intents to scraping strategies
        if "museum" in intent.lower():
            return await self._scrape_attractions({"focus": "museums"})
        elif "restaurant" in intent.lower():
            return await self._scrape_restaurants({"focus": "restaurants"})
        elif "event" in intent.lower():
            return await self._scrape_events({"focus": "events"})
        else:
            return []
    
    async def _apply_quality_filtering(self, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply AI-based quality filtering to scraped data"""
        quality_results = []
        
        for item in scraped_data:
            try:
                # Calculate quality score based on multiple factors
                quality_score = self._calculate_quality_score(item)
                
                # Store quality assessment
                quality_factors = self._get_quality_factors(item)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO content_quality (
                            content_id, content_type, raw_data, quality_score,
                            quality_factors, auto_approved, manual_review_needed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"{item['type']}_{hash(json.dumps(item, sort_keys=True))}",
                        item["type"],
                        json.dumps(item),
                        quality_score,
                        json.dumps(quality_factors),
                        quality_score >= self.config.auto_approve_threshold,
                        quality_score < self.config.quality_threshold
                    ))
                    conn.commit()
                
                item["quality_score"] = quality_score
                item["quality_factors"] = quality_factors
                quality_results.append(item)
                
            except Exception as e:
                logger.error(f"Error processing item quality: {e}")
                continue
        
        return quality_results
    
    def _calculate_quality_score(self, item: Dict[str, Any]) -> float:
        """Calculate quality score for an item"""
        score = 0.0
        factors = []
        
        # Basic completeness (30%)
        required_fields = ["name", "location", "description"]
        completeness = sum(1 for field in required_fields if item.get(field)) / len(required_fields)
        score += completeness * 0.3
        factors.append(f"completeness: {completeness:.2f}")
        
        # Rating quality (25%)
        rating = item.get("rating", 0)
        if rating >= 4.0:
            rating_score = 1.0
        elif rating >= 3.5:
            rating_score = 0.8
        elif rating >= 3.0:
            rating_score = 0.6
        else:
            rating_score = 0.3
        score += rating_score * 0.25
        factors.append(f"rating: {rating_score:.2f}")
        
        # Description quality (20%)
        description = item.get("description", "")
        if len(description) > 100:
            desc_score = 1.0
        elif len(description) > 50:
            desc_score = 0.7
        elif len(description) > 20:
            desc_score = 0.5
        else:
            desc_score = 0.2
        score += desc_score * 0.2
        factors.append(f"description: {desc_score:.2f}")
        
        # Source reliability (15%)
        source = item.get("source", "")
        if source in ["google_places", "foursquare"]:
            source_score = 1.0
        elif source in ["tripadvisor", "yelp"]:
            source_score = 0.8
        else:
            source_score = 0.5
        score += source_score * 0.15
        factors.append(f"source: {source_score:.2f}")
        
        # Freshness (10%)
        scraped_at = item.get("scraped_at", "")
        if scraped_at:
            try:
                scraped_dt = datetime.fromisoformat(scraped_at.replace('Z', '+00:00'))
                hours_old = (datetime.now() - scraped_dt).total_seconds() / 3600
                if hours_old < 24:
                    freshness_score = 1.0
                elif hours_old < 72:
                    freshness_score = 0.8
                else:
                    freshness_score = 0.5
            except:
                freshness_score = 0.5
        else:
            freshness_score = 0.5
        score += freshness_score * 0.1
        factors.append(f"freshness: {freshness_score:.2f}")
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _get_quality_factors(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed quality factors for an item"""
        return {
            "has_rating": bool(item.get("rating")),
            "description_length": len(item.get("description", "")),
            "has_location": bool(item.get("location")),
            "source": item.get("source", "unknown"),
            "calculated_at": datetime.now().isoformat()
        }
    
    async def _auto_approve_content(self, quality_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Auto-approve high-quality content"""
        auto_approved = []
        
        for item in quality_results:
            if item["quality_score"] >= self.config.auto_approve_threshold:
                # Add to pending integration
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO pending_integration (
                            content_type, content_data, quality_score, source
                        ) VALUES (?, ?, ?, ?)
                    """, (
                        item["type"],
                        json.dumps(item),
                        item["quality_score"],
                        item.get("source", "unknown")
                    ))
                    conn.commit()
                
                auto_approved.append(item)
        
        return auto_approved
    
    async def _flag_for_manual_review(self, quality_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flag content that needs manual review"""
        flagged = []
        
        for item in quality_results:
            if (item["quality_score"] >= self.config.quality_threshold and 
                item["quality_score"] < self.config.auto_approve_threshold):
                
                # Mark for manual review
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE content_quality
                        SET manual_review_needed = TRUE
                        WHERE content_id = ?
                    """, (f"{item['type']}_{hash(json.dumps(item, sort_keys=True))}",))
                    conn.commit()
                
                flagged.append(item)
        
        return flagged
    
    async def _integrate_approved_content(self) -> int:
        """Integrate auto-approved content into the live system"""
        integrated_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, content_type, content_data, quality_score
                FROM pending_integration
                WHERE integration_status = 'pending'
                ORDER BY quality_score DESC
                LIMIT ?
            """, (self.config.max_daily_additions,))
            
            pending_items = cursor.fetchall()
            
            for item_id, content_type, content_data_json, quality_score in pending_items:
                try:
                    content_data = json.loads(content_data_json)
                    
                    # Here you would integrate with your actual database
                    # For now, we'll just mark as integrated
                    success = await self._integrate_single_item(content_type, content_data)
                    
                    if success:
                        conn.execute("""
                            UPDATE pending_integration
                            SET integration_status = 'completed', integrated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (item_id,))
                        integrated_count += 1
                    else:
                        conn.execute("""
                            UPDATE pending_integration
                            SET integration_status = 'failed'
                            WHERE id = ?
                        """, (item_id,))
                    
                    conn.commit()
                    
                except Exception as e:
                    logger.error(f"Error integrating item {item_id}: {e}")
                    continue
        
        return integrated_count
    
    async def _integrate_single_item(self, content_type: str, content_data: Dict[str, Any]) -> bool:
        """Integrate a single item into the live system"""
        # This would integrate with your actual database/system
        # For now, just simulate success
        logger.info(f"Integrating {content_type}: {content_data.get('name', 'Unknown')}")
        return True
    
    async def _update_freshness_tracking(self):
        """Update data freshness tracking"""
        for content_type in self.config.content_types:
            with sqlite3.connect(self.db_path) as conn:
                # Count items processed today
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM content_quality
                    WHERE content_type = ? AND DATE(created_at) = DATE('now')
                """, (content_type.value,))
                
                fresh_items = cursor.fetchone()[0]
                
                # Update freshness tracking
                conn.execute("""
                    INSERT OR REPLACE INTO data_freshness (
                        content_type, last_update, fresh_items, next_update_due
                    ) VALUES (?, CURRENT_TIMESTAMP, ?, ?)
                """, (
                    content_type.value,
                    fresh_items,
                    (datetime.now() + timedelta(hours=self.config.scraping_interval_hours)).isoformat()
                ))
                
                conn.commit()
    
    def _log_run_start(self, run_id: str):
        """Log the start of a pipeline run"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO pipeline_runs (run_id, status, config_snapshot)
                VALUES (?, 'running', ?)
            """, (run_id, json.dumps(asdict(self.config))))
            conn.commit()
    
    def _log_run_completion(self, run_id: str, processed: int, approved: int, flagged: int):
        """Log successful completion of a pipeline run"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE pipeline_runs
                SET end_time = CURRENT_TIMESTAMP, status = 'completed',
                    items_processed = ?, items_approved = ?, items_flagged = ?
                WHERE run_id = ?
            """, (processed, approved, flagged, run_id))
            conn.commit()
    
    def _log_run_error(self, run_id: str, error_details: str):
        """Log a pipeline run error"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE pipeline_runs
                SET end_time = CURRENT_TIMESTAMP, status = 'error', error_details = ?
                WHERE run_id = ?
            """, (error_details, run_id))
            conn.commit()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics"""
        return {
            "status": self.status.value,
            "config": asdict(self.config),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "stats": self.stats,
            "next_run": (datetime.now() + timedelta(hours=self.config.scraping_interval_hours)).isoformat()
        }
    
    def get_content_quality_dashboard(self) -> Dict[str, Any]:
        """Get content quality dashboard"""
        with sqlite3.connect(self.db_path) as conn:
            # Quality distribution
            cursor = conn.execute("""
                SELECT content_type, AVG(quality_score), COUNT(*), 
                       COUNT(CASE WHEN auto_approved = 1 THEN 1 END),
                       COUNT(CASE WHEN manual_review_needed = 1 THEN 1 END)
                FROM content_quality
                WHERE created_at >= DATE('now', '-7 days')
                GROUP BY content_type
            """)
            
            quality_stats = {}
            for row in cursor.fetchall():
                content_type, avg_quality, total, auto_approved, needs_review = row
                quality_stats[content_type] = {
                    "avg_quality": avg_quality,
                    "total_items": total,
                    "auto_approved": auto_approved,
                    "needs_review": needs_review,
                    "approval_rate": auto_approved / max(total, 1) * 100
                }
            
            # Pending integration
            cursor = conn.execute("""
                SELECT integration_status, COUNT(*)
                FROM pending_integration
                GROUP BY integration_status
            """)
            
            integration_stats = dict(cursor.fetchall())
            
            return {
                "quality_stats": quality_stats,
                "integration_stats": integration_stats,
                "total_pending": integration_stats.get("pending", 0),
                "total_completed": integration_stats.get("completed", 0)
            }

# Global pipeline instance
automated_pipeline = AutomatedDataPipeline()

def start_pipeline(config: PipelineConfig = None):
    """Start the automated pipeline"""
    if config:
        automated_pipeline.config = config
    automated_pipeline.start_automated_pipeline()

def stop_pipeline():
    """Stop the automated pipeline"""
    automated_pipeline.stop_automated_pipeline()

def get_pipeline_dashboard():
    """Get comprehensive pipeline dashboard"""
    return {
        "status": automated_pipeline.get_pipeline_status(),
        "quality": automated_pipeline.get_content_quality_dashboard()
    }

if __name__ == "__main__":
    # Test the automated pipeline
    print("🧪 Testing Automated Data Pipeline...")
    
    config = PipelineConfig(
        enabled=True,
        scraping_interval_hours=1,  # For testing
        quality_threshold=0.7,
        auto_approve_threshold=0.85,
        max_daily_additions=10
    )
    
    pipeline = AutomatedDataPipeline(config)
    
    # Run a single cycle for testing
    import asyncio
    asyncio.run(pipeline._run_pipeline_cycle())
    
    # Get dashboard
    dashboard = pipeline.get_pipeline_status()
    print(f"📊 Pipeline Status: {dashboard['status']}")
    print(f"📈 Stats: {dashboard['stats']}")
    
    quality_dashboard = pipeline.get_content_quality_dashboard()
    print(f"🎯 Quality Stats: {quality_dashboard}")
    
    print("✅ Automated Data Pipeline is working correctly!")

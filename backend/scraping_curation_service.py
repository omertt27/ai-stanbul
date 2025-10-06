#!/usr/bin/env python3
"""
Scraping & Curation Service for AI Istanbul
===========================================

Semi-automated scraping + manual curation system for:
1. New restaurants, museums, events discovery
2. Quality assessment and validation
3. Manual review and approval workflow
4. Data enrichment and standardization
"""

import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class CurationStatus(Enum):
    DISCOVERED = "discovered"
    QUALITY_CHECKED = "quality_checked"
    APPROVED = "approved"
    REJECTED = "rejected"
    INTEGRATED = "integrated"

class ContentQuality(Enum):
    EXCELLENT = "excellent"  # Auto-approve (>0.9)
    GOOD = "good"           # Manual review (0.7-0.9)
    POOR = "poor"           # Auto-reject (<0.7)

@dataclass
class DiscoveredContent:
    """Discovered content item waiting for curation"""
    id: str
    content_type: str  # restaurant, museum, event, transport
    name: str
    description: str
    location: str
    source_url: str
    discovery_date: datetime
    
    # Quality assessment
    quality_score: float = 0.0
    quality_reasons: List[str] = None
    
    # Curation workflow
    status: CurationStatus = CurationStatus.DISCOVERED
    curator_notes: str = ""
    approved_by: str = ""
    
    # Enhanced data
    category: str = ""
    district: str = ""
    coordinates: Tuple[float, float] = None
    opening_hours: str = ""
    price_level: str = ""
    features: List[str] = None
    
    def __post_init__(self):
        if self.quality_reasons is None:
            self.quality_reasons = []
        if self.features is None:
            self.features = []

class ScrapingCurationService:
    """Service for discovering and curating new Istanbul content"""
    
    def __init__(self, db_path: str = "curation_content.db"):
        self.db_path = db_path
        self._init_database()
        
        # Quality assessment keywords
        self.quality_keywords = {
            "positive": ["authentic", "traditional", "historic", "unique", "hidden gem", 
                        "local favorite", "award", "recommended", "popular", "famous"],
            "negative": ["tourist trap", "overpriced", "closed", "temporary", "pop-up",
                        "under construction", "renovation"]
        }
        
        # Istanbul districts for validation
        self.istanbul_districts = [
            "Sultanahmet", "Beyoğlu", "Galata", "Kadiköy", "Beşiktaş", "Şişli",
            "Fatih", "Eminönü", "Üsküdar", "Ortaköy", "Taksim", "Balat",
            "Fener", "Nişantaşı", "Bebek", "Arnavutköy", "Sarıyer"
        ]
    
    def _init_database(self):
        """Initialize curation database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discovered_content (
                    id TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    location TEXT,
                    source_url TEXT,
                    discovery_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    quality_score REAL DEFAULT 0.0,
                    quality_reasons TEXT,
                    
                    status TEXT DEFAULT 'discovered',
                    curator_notes TEXT,
                    approved_by TEXT,
                    
                    category TEXT,
                    district TEXT,
                    coordinates TEXT,
                    opening_hours TEXT,
                    price_level TEXT,
                    features TEXT,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def discover_new_restaurants(self, limit: int = 20) -> List[DiscoveredContent]:
        """Simulate discovering new restaurants (would use real APIs in production)"""
        print(f"🔍 Discovering new restaurants...")
        
        # Simulated restaurant discoveries
        simulated_restaurants = [
            {
                "name": "Lokanta Maya",
                "description": "Contemporary Turkish cuisine with a modern twist, featuring seasonal ingredients and innovative presentations.",
                "location": "Kadiköy, Istanbul",
                "category": "Fine Dining",
                "features": ["contemporary", "seasonal menu", "wine pairing"]
            },
            {
                "name": "Çukur Meyhane",
                "description": "Traditional Istanbul meyhane serving authentic meze and rakı in a historic setting.",
                "location": "Beyoğlu, Istanbul", 
                "category": "Traditional Meyhane",
                "features": ["authentic", "historic", "live music"]
            },
            {
                "name": "Balık Pazarı Tezgahı",
                "description": "Fresh seafood restaurant in the heart of Galata, known for daily catches and traditional preparation.",
                "location": "Galata, Istanbul",
                "category": "Seafood",
                "features": ["fresh seafood", "daily specials", "local favorite"]
            }
        ]
        
        discovered = []
        for i, rest_data in enumerate(simulated_restaurants[:limit]):
            content = DiscoveredContent(
                id=f"rest_{int(time.time())}_{i}",
                content_type="restaurant",
                name=rest_data["name"],
                description=rest_data["description"],
                location=rest_data["location"],
                source_url=f"https://example-source.com/restaurant/{i}",
                discovery_date=datetime.now(),
                category=rest_data["category"],
                features=rest_data["features"]
            )
            
            # Assess quality
            content.quality_score = self._assess_content_quality(content)
            discovered.append(content)
            
            # Store in database
            self._store_discovered_content(content)
            
        print(f"✅ Discovered {len(discovered)} new restaurants")
        return discovered
    
    def discover_new_museums(self, limit: int = 10) -> List[DiscoveredContent]:
        """Simulate discovering new museums and cultural sites"""
        print(f"🏛️ Discovering new museums...")
        
        simulated_museums = [
            {
                "name": "Istanbul Maritime Museum Branch",
                "description": "Specialized maritime collection showcasing Istanbul's naval history and Bosphorus heritage.",
                "location": "Beşiktaş, Istanbul",
                "category": "Specialized Museum",
                "features": ["maritime history", "naval artifacts", "Bosphorus focus"]
            },
            {
                "name": "Galata Mevlevi House Museum",
                "description": "Historic tekke showcasing Sufi culture and whirling dervish traditions in Galata.",
                "location": "Galata, Istanbul",
                "category": "Cultural Heritage",
                "features": ["Sufi culture", "historic tekke", "spiritual heritage"]
            }
        ]
        
        discovered = []
        for i, museum_data in enumerate(simulated_museums[:limit]):
            content = DiscoveredContent(
                id=f"museum_{int(time.time())}_{i}",
                content_type="museum",
                name=museum_data["name"],
                description=museum_data["description"],
                location=museum_data["location"],
                source_url=f"https://example-source.com/museum/{i}",
                discovery_date=datetime.now(),
                category=museum_data["category"],
                features=museum_data["features"]
            )
            
            content.quality_score = self._assess_content_quality(content)
            discovered.append(content)
            self._store_discovered_content(content)
            
        print(f"✅ Discovered {len(discovered)} new museums")
        return discovered
    
    def discover_seasonal_events(self, limit: int = 15) -> List[DiscoveredContent]:
        """Discover upcoming seasonal events and festivals"""
        print(f"🎉 Discovering seasonal events...")
        
        current_month = datetime.now().month
        seasonal_events = []
        
        # October events (current month in simulation)
        if current_month == 10:
            seasonal_events = [
                {
                    "name": "Istanbul Film Festival Autumn Edition",
                    "description": "International film screenings across Istanbul's historic venues, featuring Turkish and world cinema.",
                    "location": "Various venues, Istanbul",
                    "category": "Cultural Festival",
                    "features": ["international films", "multiple venues", "cultural exchange"]
                },
                {
                    "name": "Bosphorus Jazz Festival",
                    "description": "Waterfront jazz performances along the Bosphorus, featuring local and international artists.",
                    "location": "Ortaköy, Istanbul",
                    "category": "Music Festival",
                    "features": ["jazz music", "waterfront venues", "international artists"]
                }
            ]
        
        discovered = []
        for i, event_data in enumerate(seasonal_events[:limit]):
            content = DiscoveredContent(
                id=f"event_{int(time.time())}_{i}",
                content_type="event",
                name=event_data["name"],
                description=event_data["description"],
                location=event_data["location"],
                source_url=f"https://example-events.com/event/{i}",
                discovery_date=datetime.now(),
                category=event_data["category"],
                features=event_data["features"]
            )
            
            content.quality_score = self._assess_content_quality(content)
            discovered.append(content)
            self._store_discovered_content(content)
            
        print(f"✅ Discovered {len(discovered)} seasonal events")
        return discovered
    
    def _assess_content_quality(self, content: DiscoveredContent) -> float:
        """Assess content quality using various heuristics"""
        score = 0.5  # Base score
        reasons = []
        
        # Check for positive keywords
        text_to_check = f"{content.name} {content.description}".lower()
        
        positive_matches = sum(1 for keyword in self.quality_keywords["positive"] 
                             if keyword in text_to_check)
        negative_matches = sum(1 for keyword in self.quality_keywords["negative"] 
                             if keyword in text_to_check)
        
        # Adjust score based on keywords
        score += positive_matches * 0.1
        score -= negative_matches * 0.2
        
        if positive_matches > 0:
            reasons.append(f"Contains {positive_matches} positive quality indicators")
        if negative_matches > 0:
            reasons.append(f"Contains {negative_matches} negative quality indicators")
        
        # Check location validity
        location_valid = any(district.lower() in content.location.lower() 
                           for district in self.istanbul_districts)
        if location_valid:
            score += 0.2
            reasons.append("Valid Istanbul location")
        else:
            score -= 0.1
            reasons.append("Location needs verification")
        
        # Check description quality
        if len(content.description) > 100:
            score += 0.1
            reasons.append("Detailed description")
        elif len(content.description) < 50:
            score -= 0.1
            reasons.append("Description too brief")
        
        # Check for specific content type indicators
        if content.content_type == "restaurant":
            restaurant_indicators = ["cuisine", "menu", "chef", "dining", "kitchen"]
            if any(indicator in text_to_check for indicator in restaurant_indicators):
                score += 0.1
                reasons.append("Contains restaurant-specific content")
        
        elif content.content_type == "museum":
            museum_indicators = ["collection", "exhibit", "artifact", "historic", "culture"]
            if any(indicator in text_to_check for indicator in museum_indicators):
                score += 0.1
                reasons.append("Contains museum-specific content")
        
        # Ensure score is within bounds
        score = max(0.0, min(1.0, score))
        content.quality_reasons = reasons
        
        return score
    
    def _store_discovered_content(self, content: DiscoveredContent):
        """Store discovered content in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO discovered_content (
                    id, content_type, name, description, location, source_url,
                    discovery_date, quality_score, quality_reasons, status,
                    category, features
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                content.id, content.content_type, content.name, content.description,
                content.location, content.source_url, content.discovery_date,
                content.quality_score, json.dumps(content.quality_reasons), 
                content.status.value, content.category, json.dumps(content.features)
            ))
    
    def get_pending_curation(self, limit: int = 50) -> List[DiscoveredContent]:
        """Get content pending manual curation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM discovered_content 
                WHERE status IN ('discovered', 'quality_checked')
                ORDER BY quality_score DESC, discovery_date DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                content = self._row_to_content(row)
                results.append(content)
            
            return results
    
    def approve_content(self, content_id: str, curator_name: str, notes: str = "") -> bool:
        """Approve content for integration"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE discovered_content 
                SET status = 'approved', approved_by = ?, curator_notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (curator_name, notes, content_id))
            
            return conn.total_changes > 0
    
    def reject_content(self, content_id: str, curator_name: str, reason: str) -> bool:
        """Reject content"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE discovered_content 
                SET status = 'rejected', approved_by = ?, curator_notes = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (curator_name, reason, content_id))
            
            return conn.total_changes > 0
    
    def get_approved_content(self) -> List[DiscoveredContent]:
        """Get approved content ready for integration"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM discovered_content 
                WHERE status = 'approved'
                ORDER BY quality_score DESC
            """)
            
            results = []
            for row in cursor.fetchall():
                content = self._row_to_content(row)
                results.append(content)
            
            return results
    
    def mark_as_integrated(self, content_id: str) -> bool:
        """Mark content as integrated into live database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE discovered_content 
                SET status = 'integrated', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (content_id,))
            
            return conn.total_changes > 0
    
    def get_curation_stats(self) -> Dict[str, Any]:
        """Get curation statistics"""
        with sqlite3.connect(self.db_path) as conn:
            # Status counts
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM discovered_content
                GROUP BY status
            """)
            status_counts = dict(cursor.fetchall())
            
            # Quality distribution
            cursor = conn.execute("""
                SELECT 
                    CASE 
                        WHEN quality_score >= 0.9 THEN 'excellent'
                        WHEN quality_score >= 0.7 THEN 'good'
                        ELSE 'poor'
                    END as quality_level,
                    COUNT(*) as count
                FROM discovered_content
                GROUP BY quality_level
            """)
            quality_counts = dict(cursor.fetchall())
            
            # Content type distribution
            cursor = conn.execute("""
                SELECT content_type, COUNT(*) as count
                FROM discovered_content
                GROUP BY content_type
            """)
            type_counts = dict(cursor.fetchall())
            
            return {
                "total_discovered": sum(status_counts.values()),
                "status_distribution": status_counts,
                "quality_distribution": quality_counts,
                "content_type_distribution": type_counts,
                "pending_review": status_counts.get("discovered", 0) + status_counts.get("quality_checked", 0),
                "approval_rate": status_counts.get("approved", 0) / max(1, sum(status_counts.values())) * 100
            }
    
    def _row_to_content(self, row) -> DiscoveredContent:
        """Convert database row to DiscoveredContent object"""
        return DiscoveredContent(
            id=row[0],
            content_type=row[1],
            name=row[2],
            description=row[3] or "",
            location=row[4] or "",
            source_url=row[5] or "",
            discovery_date=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
            quality_score=row[7] or 0.0,
            quality_reasons=json.loads(row[8]) if row[8] else [],
            status=CurationStatus(row[9]) if row[9] else CurationStatus.DISCOVERED,
            curator_notes=row[10] or "",
            approved_by=row[11] or "",
            category=row[12] or "",
            features=json.loads(row[17]) if row[17] else []
        )

# Global service instance
scraping_curation_service = ScrapingCurationService()

if __name__ == "__main__":
    # Test the scraping and curation service
    print("🧪 Testing Scraping & Curation Service...")
    
    service = ScrapingCurationService()
    
    # Discover new content
    restaurants = service.discover_new_restaurants(3)
    museums = service.discover_new_museums(2)
    events = service.discover_seasonal_events(2)
    
    print(f"✅ Discovered {len(restaurants)} restaurants, {len(museums)} museums, {len(events)} events")
    
    # Get curation stats
    stats = service.get_curation_stats()
    print(f"📊 Total discovered: {stats['total_discovered']}")
    print(f"📋 Pending review: {stats['pending_review']}")
    print(f"✅ Approval rate: {stats['approval_rate']:.1f}%")
    
    # Test approval workflow
    pending = service.get_pending_curation(5)
    if pending:
        # Auto-approve high quality content
        high_quality = [c for c in pending if c.quality_score >= 0.9]
        for content in high_quality:
            service.approve_content(content.id, "auto_curator", "High quality auto-approval")
            print(f"✅ Auto-approved: {content.name} (score: {content.quality_score:.2f})")
    
    print("✅ Scraping & Curation Service is working correctly!")

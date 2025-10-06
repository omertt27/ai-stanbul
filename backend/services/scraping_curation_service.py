"""
Semi-Automated Scraping and Manual Curation Service
Integrates data discovery with human curation for quality control
"""

import requests
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
import logging
from urllib.parse import urlencode
import sqlite3

class ScrapingSource(Enum):
    GOOGLE_PLACES = "google_places"
    FOURSQUARE = "foursquare"
    TRIPADVISOR = "tripadvisor"  
    YELP = "yelp"
    LOCAL_BLOGS = "local_blogs"

class CurationStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"

@dataclass
class ScrapingConfig:
    source: ScrapingSource
    api_key: str
    base_url: str
    rate_limit: int  # requests per minute
    active: bool = True

class ScrapingCurationService:
    """Service for semi-automated scraping with manual curation workflow"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self.init_database()
        self.scraping_configs = {}
        self.logger = logging.getLogger(__name__)
        
    def init_database(self):
        """Initialize scraping and curation database"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Scraping configurations
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraping_configs (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            api_key TEXT,
            base_url TEXT NOT NULL,
            rate_limit INTEGER DEFAULT 60,
            active BOOLEAN DEFAULT TRUE,
            last_used DATETIME,
            total_requests INTEGER DEFAULT 0,
            successful_requests INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Scraping sessions
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraping_sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            search_query TEXT NOT NULL,
            location TEXT NOT NULL,
            radius INTEGER DEFAULT 5000,
            total_found INTEGER DEFAULT 0,
            processed INTEGER DEFAULT 0,
            approved INTEGER DEFAULT 0,
            rejected INTEGER DEFAULT 0,
            status TEXT DEFAULT 'running',  -- running, completed, failed
            started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME
        )
        ''')
        
        # Curation templates
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS curation_templates (
            id TEXT PRIMARY KEY,
            template_name TEXT NOT NULL,
            category TEXT NOT NULL,
            authenticity_criteria TEXT NOT NULL,  -- JSON array
            quality_indicators TEXT NOT NULL,     -- JSON array
            red_flags TEXT NOT NULL,              -- JSON array
            scoring_weights TEXT NOT NULL,        -- JSON object
            created_by TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Manual curation decisions
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS curation_decisions (
            id TEXT PRIMARY KEY,
            scraped_data_id TEXT NOT NULL,
            curator_id TEXT NOT NULL,
            decision TEXT NOT NULL,  -- approve, reject, needs_review
            authenticity_score REAL,
            quality_score REAL,
            curator_notes TEXT,
            decision_rationale TEXT,
            time_spent_minutes INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.conn.commit()
        self._setup_default_templates()
    
    def _setup_default_templates(self):
        """Setup default curation templates"""
        # Check if templates already exist
        self.cursor.execute('SELECT COUNT(*) FROM curation_templates')
        if self.cursor.fetchone()[0] > 0:
            return
        
        templates = [
            {
                "template_name": "Mosque Curation",
                "category": "Mosque",
                "authenticity_criteria": [
                    "Historical significance and architectural merit",
                    "Active religious community",
                    "Proper Islamic architectural features",
                    "Cultural and educational value",
                    "Accessibility to respectful visitors"
                ],
                "quality_indicators": [
                    "High rating from local community",
                    "Historical documentation available",
                    "Architectural significance",
                    "Active worship community",
                    "Educational/cultural programs"
                ],
                "red_flags": [
                    "Primarily tourist-focused",
                    "Commercialized religious experience",
                    "Disrespectful visitor behavior reported",
                    "Limited historical significance",
                    "Restricted authentic access"
                ],
                "scoring_weights": {
                    "historical_significance": 0.25,
                    "community_rating": 0.20,
                    "architectural_merit": 0.20,
                    "accessibility": 0.15,
                    "cultural_value": 0.20
                }
            },
            {
                "template_name": "Bazaar Curation",
                "category": "Bazaar",
                "authenticity_criteria": [
                    "Traditional Turkish goods and crafts",
                    "Local vendor community",
                    "Historical market setting",
                    "Authentic pricing (not tourist-inflated)",
                    "Cultural shopping experience"
                ],
                "quality_indicators": [
                    "Local customers present",
                    "Traditional goods and crafts",
                    "Reasonable pricing",
                    "Historical market building",
                    "Skilled artisan workshops"
                ],
                "red_flags": [
                    "Only tourist souvenirs",
                    "Extremely inflated prices",
                    "Aggressive tourist targeting",
                    "No local customers",
                    "Generic imported goods"
                ],
                "scoring_weights": {
                    "authenticity_of_goods": 0.30,
                    "pricing_fairness": 0.20,
                    "local_presence": 0.20,
                    "historical_setting": 0.15,
                    "cultural_experience": 0.15
                }
            },
            {
                "template_name": "Cultural Experience Curation",
                "category": "Cultural Experience",
                "authenticity_criteria": [
                    "Genuine Turkish cultural content",
                    "Educational or participatory elements",
                    "Local cultural practitioners involved",
                    "Respectful cultural representation",
                    "Accessible to different audiences"
                ],
                "quality_indicators": [
                    "Local cultural experts involved",
                    "Educational value",
                    "Authentic cultural practices",
                    "Positive community reception",
                    "Sustainable cultural preservation"
                ],
                "red_flags": [
                    "Stereotypical cultural representation",
                    "Commercial exploitation of culture",
                    "Inaccurate cultural information",
                    "Disrespectful presentation",
                    "No local community involvement"
                ],
                "scoring_weights": {
                    "cultural_accuracy": 0.30,
                    "local_involvement": 0.25,
                    "educational_value": 0.20,
                    "respectful_presentation": 0.15,
                    "accessibility": 0.10
                }
            }
        ]
        
        for template in templates:
            template_id = str(uuid.uuid4())
            self.cursor.execute('''
            INSERT INTO curation_templates (
                id, template_name, category, authenticity_criteria,
                quality_indicators, red_flags, scoring_weights, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template_id, template['template_name'], template['category'],
                json.dumps(template['authenticity_criteria']),
                json.dumps(template['quality_indicators']),
                json.dumps(template['red_flags']),
                json.dumps(template['scoring_weights']),
                'system_default'
            ))
        
        self.conn.commit()
    
    def add_scraping_config(self, source: str, api_key: str, base_url: str,
                           rate_limit: int = 60) -> str:
        """Add new scraping configuration"""
        config_id = str(uuid.uuid4())
        
        self.cursor.execute('''
        INSERT INTO scraping_configs (
            id, source, api_key, base_url, rate_limit
        ) VALUES (?, ?, ?, ?, ?)
        ''', (config_id, source, api_key, base_url, rate_limit))
        
        self.conn.commit()
        return config_id
    
    def scrape_google_places(self, query: str, location: str = "Istanbul,Turkey",
                           radius: int = 5000, max_results: int = 50) -> str:
        """Scrape Google Places API (mock implementation)"""
        session_id = str(uuid.uuid4())
        
        # Create scraping session
        self.cursor.execute('''
        INSERT INTO scraping_sessions (
            id, source, search_query, location, radius, status
        ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (session_id, 'google_places', query, location, radius, 'running'))
        
        # Mock scraping results (in production, this would use actual API)
        mock_results = self._generate_mock_google_places_data(query, max_results)
        
        # Process results
        processed = 0
        for result in mock_results:
            scraped_id = self._store_scraped_data('google_places', result)
            if scraped_id:
                processed += 1
        
        # Update session
        self.cursor.execute('''
        UPDATE scraping_sessions 
        SET total_found = ?, processed = ?, status = 'completed', completed_at = ?
        WHERE id = ?
        ''', (len(mock_results), processed, datetime.now().isoformat(), session_id))
        
        self.conn.commit()
        return session_id
    
    def _generate_mock_google_places_data(self, query: str, max_results: int) -> List[Dict]:
        """Generate mock Google Places data for demonstration"""
        categories = {
            'mosque': ['Mosque', 'Religious Site'],
            'bazaar': ['Bazaar', 'Market'],
            'restaurant': ['Dining', 'Restaurant'],
            'cafe': ['Dining', 'Cafe'],
            'museum': ['Culture', 'Museum'],
            'nightlife': ['Nightlife', 'Bar']
        }
        
        query_lower = query.lower()
        category = None
        for key, value in categories.items():
            if key in query_lower:
                category = value
                break
        
        if not category:
            category = ['Cultural Experience', 'Attraction']
        
        mock_results = []
        for i in range(min(max_results, 10)):  # Limit for demo
            result = {
                'place_id': f'mock_place_{uuid.uuid4().hex[:8]}',
                'name': f'{category[0]} Discovery {i+1}',
                'types': [category[1].lower().replace(' ', '_')],
                'rating': round(3.5 + (i % 3) * 0.5, 1),
                'user_ratings_total': 50 + (i * 20),
                'price_level': (i % 4) + 1,
                'geometry': {
                    'location': {
                        'lat': 41.0082 + (i * 0.01),
                        'lng': 28.9784 + (i * 0.01)
                    }
                },
                'formatted_address': f'Mock Address {i+1}, Istanbul, Turkey',
                'opening_hours': {
                    'open_now': True
                },
                'photos': [{'photo_reference': f'mock_photo_{i}'}]
            }
            mock_results.append(result)
        
        return mock_results
    
    def _store_scraped_data(self, source: str, data: Dict) -> Optional[str]:
        """Store scraped data for curation"""
        try:
            scraped_id = str(uuid.uuid4())
            
            # Extract key information
            name = data.get('name', 'Unknown')
            category = self._infer_category_from_types(data.get('types', []))
            rating = data.get('rating', 0.0)
            review_count = data.get('user_ratings_total', 0)
            
            geometry = data.get('geometry', {}).get('location', {})
            lat = geometry.get('lat', 0.0)
            lng = geometry.get('lng', 0.0)
            
            address = data.get('formatted_address', '')
            price_level = data.get('price_level', 1)
            
            # Store in scraped_raw_data (this table should exist from enhanced_attractions_service)
            self.cursor.execute('''
            INSERT INTO scraped_raw_data (
                id, source, external_id, name, category, rating, review_count,
                coordinates_lat, coordinates_lng, address, price_level, raw_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                scraped_id, source, data.get('place_id', ''), name, category,
                rating, review_count, lat, lng, address, str(price_level),
                json.dumps(data)
            ))
            
            return scraped_id
            
        except Exception as e:
            self.logger.error(f"Error storing scraped data: {e}")
            return None
    
    def _infer_category_from_types(self, types: List[str]) -> str:
        """Infer attraction category from Google Places types"""
        type_mapping = {
            'mosque': 'Mosque',
            'shopping_mall': 'Bazaar',
            'store': 'Bazaar',
            'restaurant': 'Dining',
            'cafe': 'Dining',
            'museum': 'Culture',
            'tourist_attraction': 'Cultural Experience',
            'night_club': 'Nightlife',
            'bar': 'Nightlife'
        }
        
        for place_type in types:
            if place_type in type_mapping:
                return type_mapping[place_type]
        
        return 'Cultural Experience'  # Default
    
    def get_curation_template(self, category: str) -> Optional[Dict]:
        """Get curation template for category"""
        self.cursor.execute('''
        SELECT template_name, authenticity_criteria, quality_indicators,
               red_flags, scoring_weights
        FROM curation_templates
        WHERE category = ?
        ORDER BY created_at DESC
        LIMIT 1
        ''', (category,))
        
        result = self.cursor.fetchone()
        if not result:
            return None
        
        return {
            'template_name': result[0],
            'authenticity_criteria': json.loads(result[1]),
            'quality_indicators': json.loads(result[2]),
            'red_flags': json.loads(result[3]),
            'scoring_weights': json.loads(result[4])
        }
    
    def submit_curation_decision(self, scraped_data_id: str, curator_id: str,
                               decision: str, authenticity_score: float,
                               quality_score: float, notes: str,
                               rationale: str, time_spent: int) -> str:
        """Submit manual curation decision"""
        decision_id = str(uuid.uuid4())
        
        self.cursor.execute('''
        INSERT INTO curation_decisions (
            id, scraped_data_id, curator_id, decision, authenticity_score,
            quality_score, curator_notes, decision_rationale, time_spent_minutes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision_id, scraped_data_id, curator_id, decision,
            authenticity_score, quality_score, notes, rationale, time_spent
        ))
        
        # Update scraped data status
        self.cursor.execute('''
        UPDATE scraped_raw_data 
        SET curation_status = ?
        WHERE id = ?
        ''', (decision, scraped_data_id))
        
        self.conn.commit()
        return decision_id
    
    def get_curation_workload(self, curator_id: str = None, limit: int = 20) -> List[Dict]:
        """Get items pending curation"""
        self.cursor.execute('''
        SELECT srd.id, srd.name, srd.category, srd.rating, srd.review_count,
               srd.source, srd.scraped_at, srd.raw_data
        FROM scraped_raw_data srd
        LEFT JOIN curation_decisions cd ON srd.id = cd.scraped_data_id
        WHERE srd.curation_status = 'pending' AND cd.id IS NULL
        ORDER BY srd.rating DESC, srd.review_count DESC
        LIMIT ?
        ''', (limit,))
        
        workload = []
        for row in self.cursor.fetchall():
            # Get curation template for category
            template = self.get_curation_template(row[2])  # category
            
            workload.append({
                'scraped_id': row[0],
                'name': row[1],
                'category': row[2],
                'rating': row[3],
                'review_count': row[4],
                'source': row[5],
                'scraped_at': row[6],
                'raw_data': json.loads(row[7]),
                'curation_template': template
            })
        
        return workload
    
    def generate_curation_report(self, session_id: str = None) -> Dict[str, Any]:
        """Generate curation performance report"""
        report = {}
        
        # Overall statistics
        self.cursor.execute('''
        SELECT 
            COUNT(*) as total_scraped,
            COUNT(CASE WHEN curation_status = 'approved' THEN 1 END) as approved,
            COUNT(CASE WHEN curation_status = 'rejected' THEN 1 END) as rejected,
            COUNT(CASE WHEN curation_status = 'pending' THEN 1 END) as pending
        FROM scraped_raw_data
        ''')
        
        result = self.cursor.fetchone()
        report['overall_stats'] = {
            'total_scraped': result[0],
            'approved': result[1],
            'rejected': result[2],
            'pending': result[3],
            'approval_rate': round((result[1] / result[0]) * 100, 1) if result[0] > 0 else 0
        }
        
        # By category
        self.cursor.execute('''
        SELECT category, 
               COUNT(*) as total,
               COUNT(CASE WHEN curation_status = 'approved' THEN 1 END) as approved,
               AVG(rating) as avg_rating
        FROM scraped_raw_data
        GROUP BY category
        ORDER BY total DESC
        ''')
        
        report['by_category'] = [
            {
                'category': row[0],
                'total': row[1],
                'approved': row[2],
                'approval_rate': round((row[2] / row[1]) * 100, 1) if row[1] > 0 else 0,
                'avg_rating': round(row[3], 2) if row[3] else 0
            }
            for row in self.cursor.fetchall()
        ]
        
        # By source
        self.cursor.execute('''
        SELECT source,
               COUNT(*) as total,
               COUNT(CASE WHEN curation_status = 'approved' THEN 1 END) as approved
        FROM scraped_raw_data
        GROUP BY source
        ''')
        
        report['by_source'] = dict(self.cursor.fetchall())
        
        # Curator productivity
        self.cursor.execute('''
        SELECT curator_id,
               COUNT(*) as decisions_made,
               AVG(time_spent_minutes) as avg_time,
               COUNT(CASE WHEN decision = 'approve' THEN 1 END) as approved
        FROM curation_decisions
        GROUP BY curator_id
        ORDER BY decisions_made DESC
        ''')
        
        report['curator_productivity'] = [
            {
                'curator_id': row[0],
                'decisions_made': row[1],
                'avg_time_minutes': round(row[2], 1) if row[2] else 0,
                'approved': row[3],
                'approval_rate': round((row[3] / row[1]) * 100, 1) if row[1] > 0 else 0
            }
            for row in self.cursor.fetchall()
        ]
        
        return report
    
    def get_scraping_templates(self) -> Dict[str, Dict]:
        """Get predefined scraping templates for different categories"""
        return {
            'mosques': {
                'query': 'mosque islamic religious site',
                'location': 'Istanbul,Turkey',
                'radius': 10000,
                'types': ['mosque', 'place_of_worship'],
                'priority_keywords': ['historic', 'ottoman', 'imperial', 'sinan']
            },
            'bazaars': {
                'query': 'bazaar market traditional shopping',
                'location': 'Istanbul,Turkey', 
                'radius': 8000,
                'types': ['shopping_mall', 'store', 'market'],
                'priority_keywords': ['traditional', 'authentic', 'artisan', 'craft']
            },
            'cultural_experiences': {
                'query': 'cultural experience traditional turkish',
                'location': 'Istanbul,Turkey',
                'radius': 12000,
                'types': ['tourist_attraction', 'cultural_center'],
                'priority_keywords': ['authentic', 'traditional', 'cultural', 'historic']
            },
            'nightlife': {
                'query': 'nightlife entertainment music venue',
                'location': 'Istanbul,Turkey',
                'radius': 6000,
                'types': ['night_club', 'bar', 'live_music_venue'],
                'priority_keywords': ['live music', 'authentic', 'local', 'jazz']
            }
        }
    
    def run_batch_scraping(self, categories: List[str] = None) -> Dict[str, str]:
        """Run batch scraping for multiple categories"""
        if categories is None:
            categories = ['mosques', 'bazaars', 'cultural_experiences', 'nightlife']
        
        templates = self.get_scraping_templates()
        session_ids = {}
        
        for category in categories:
            if category in templates:
                template = templates[category]
                session_id = self.scrape_google_places(
                    query=template['query'],
                    location=template['location'],
                    radius=template['radius'],
                    max_results=20
                )
                session_ids[category] = session_id
                
                # Rate limiting
                time.sleep(1)  # Simple rate limiting
        
        return session_ids
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

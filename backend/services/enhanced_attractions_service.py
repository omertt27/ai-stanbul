"""
Enhanced Attractions Database Service with Curated Records
Manages the expanded attraction database with scraping integration
"""

import sqlite3
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid
import random

class AttractionCategory(Enum):
    MOSQUE = "Mosque"
    BAZAAR = "Bazaar" 
    FERRY_TOUR = "Ferry Tour"
    NIGHTLIFE = "Nightlife"
    CULTURAL_EXPERIENCE = "Cultural Experience"
    NEIGHBORHOOD = "Historic Neighborhood"
    DINING = "Dining"
    VIEWPOINT = "Viewpoint"
    ART_CULTURE = "Art & Culture"

class ScrapingSource(Enum):
    GOOGLE_PLACES = "google_places"
    FOURSQUARE = "foursquare"
    TRIPADVISOR = "tripadvisor"
    LOCAL_BLOGS = "local_blogs"
    MANUAL_CURATION = "manual_curation"

@dataclass
class CuratedAttraction:
    id: int
    name: str
    category: str
    subcategory: str
    description: str
    address: str
    district: str
    coordinates_lat: float
    coordinates_lng: float
    price_range: str
    opening_hours: str
    authenticity_score: float
    local_rating: float
    tourist_rating: float
    crowd_level: str
    best_time_to_visit: str
    transportation_tips: str
    local_tips: str
    seasonal_info: str

class EnhancedAttractionsService:
    """Service for managing enhanced attractions database with scraping capabilities"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self.init_database()
        self._populate_curated_attractions()
        
    def init_database(self):
        """Initialize the enhanced attractions database schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Enhanced attractions table with comprehensive metadata
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS attractions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            description TEXT NOT NULL,
            address TEXT,
            district TEXT,
            coordinates_lat REAL,
            coordinates_lng REAL,
            price_range TEXT,
            opening_hours TEXT,
            website TEXT,
            phone TEXT,
            authenticity_score REAL DEFAULT 0.0,
            local_rating REAL DEFAULT 0.0,
            tourist_rating REAL DEFAULT 0.0,
            crowd_level TEXT DEFAULT 'medium',
            best_time_to_visit TEXT,
            transportation_tips TEXT,
            local_tips TEXT,
            seasonal_info TEXT,
            source TEXT DEFAULT 'manual_curation',
            verification_status TEXT DEFAULT 'unverified',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Scraping sources and data
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraping_sources (
            id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            base_url TEXT,
            api_key_required BOOLEAN DEFAULT FALSE,
            rate_limit INTEGER DEFAULT 1000,
            last_scraped DATETIME,
            total_records_found INTEGER DEFAULT 0,
            active BOOLEAN DEFAULT TRUE
        )
        ''')
        
        # Scraped raw data for manual curation
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS scraped_raw_data (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            external_id TEXT,
            name TEXT NOT NULL,
            category TEXT,
            rating REAL,
            review_count INTEGER,
            coordinates_lat REAL,
            coordinates_lng REAL,
            address TEXT,
            price_level TEXT,
            raw_data TEXT,  -- JSON of all scraped data
            curation_status TEXT DEFAULT 'pending',  -- pending, approved, rejected
            curated_attraction_id INTEGER,
            scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (curated_attraction_id) REFERENCES attractions (id)
        )
        ''')
        
        # Manual curation queue
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS curation_queue (
            id TEXT PRIMARY KEY,
            scraped_data_id TEXT NOT NULL,
            priority INTEGER DEFAULT 5,
            curator_notes TEXT,
            authenticity_assessment TEXT,
            recommendation TEXT,  -- approve, reject, needs_review
            processed BOOLEAN DEFAULT FALSE,
            processed_at DATETIME,
            processed_by TEXT,
            FOREIGN KEY (scraped_data_id) REFERENCES scraped_raw_data (id)
        )
        ''')
        
        self.conn.commit()
    
    def _populate_curated_attractions(self):
        """Populate database with curated authentic attractions"""
        # Check if already populated
        self.cursor.execute('SELECT COUNT(*) FROM attractions')
        if self.cursor.fetchone()[0] > 0:
            return
        
        # Curated attractions from the notebook analysis
        curated_attractions = [
            {
                "name": "Süleymaniye Mosque Complex",
                "category": "Mosque",
                "subcategory": "Ottoman Imperial",
                "description": "Magnificent Ottoman mosque by architect Sinan with panoramic city views and peaceful courtyards",
                "address": "Prof. Sıddık Sami Onar Cd. No:1, 34116 Fatih/İstanbul",
                "district": "Fatih",
                "coordinates_lat": 41.0161,
                "coordinates_lng": 28.9636,
                "price_range": "Free",
                "opening_hours": "Daily 9:00-18:00 (closed during prayer times)",
                "authenticity_score": 9.5,
                "local_rating": 9.8,
                "tourist_rating": 9.1,
                "crowd_level": "medium",
                "best_time_to_visit": "Early morning or before sunset for golden light",
                "transportation_tips": "Bus from Eminönü or metro to Vezneciler, 10-minute walk",
                "local_tips": "Respect prayer times, dress modestly. Climb to tea garden behind mosque for city views. Free but donations appreciated.",
                "seasonal_info": "Beautiful year-round, especially stunning in spring and autumn light"
            },
            {
                "name": "Rüstem Pasha Mosque",
                "category": "Mosque", 
                "subcategory": "Hidden Gem",
                "description": "Small mosque above shops near Spice Bazaar, famous for exquisite Iznik tiles covering interior walls",
                "address": "Hasırcılar Cd. No:62, 34110 Fatih/İstanbul",
                "district": "Fatih",
                "coordinates_lat": 41.0156,
                "coordinates_lng": 28.9685,
                "price_range": "Free",
                "opening_hours": "Daily except during prayer times",
                "authenticity_score": 9.2,
                "local_rating": 9.5,
                "tourist_rating": 8.3,
                "crowd_level": "low",
                "best_time_to_visit": "Mid-morning when natural light illuminates tiles best",
                "transportation_tips": "2-minute walk from Spice Bazaar, look for stairs going up",
                "local_tips": "Hidden entrance - look for narrow stairs between shops. Photography allowed but be respectful. One of Sinan's masterpieces.",
                "seasonal_info": "Indoor location makes it perfect year-round refuge from weather"
            },
            {
                "name": "Mihrimah Sultan Mosque (Edirnekapı)",
                "category": "Mosque",
                "subcategory": "Sunset Mosque",
                "description": "Sinan's architectural masterpiece designed to capture sunlight perfectly, known as the 'Mosque of Light'",
                "address": "Edirnekapı Cd., 34091 Fatih/İstanbul",
                "district": "Fatih",
                "coordinates_lat": 41.0316,
                "coordinates_lng": 28.9364,
                "price_range": "Free",
                "opening_hours": "Daily except during prayer times",
                "authenticity_score": 9.0,
                "local_rating": 9.2,
                "tourist_rating": 7.8,
                "crowd_level": "low",
                "best_time_to_visit": "Sunset - mosque designed to capture setting sun",
                "transportation_tips": "Metro to Topkapı-Ulubatlı, 15-minute walk or bus 28T",
                "local_tips": "Legend says Sinan built it for Sultan's daughter to see from Topkapi Palace. Visit at sunset for magical lighting effect.",
                "seasonal_info": "Sunset timing varies by season - winter sunsets particularly dramatic"
            },
            {
                "name": "Arasta Bazaar",
                "category": "Bazaar",
                "subcategory": "Artisan Quarter",
                "description": "Historic bazaar behind Blue Mosque specializing in authentic Turkish carpets, ceramics, and handicrafts",
                "address": "Arasta Çarşısı, Sultanahmet, 34122 Fatih/İstanbul",
                "district": "Fatih",
                "coordinates_lat": 41.0055,
                "coordinates_lng": 28.9744,
                "price_range": "Varies (expensive but authentic)",
                "opening_hours": "Daily 9:00-19:00",
                "authenticity_score": 8.5,
                "local_rating": 8.2,
                "tourist_rating": 8.8,
                "crowd_level": "medium",
                "best_time_to_visit": "Morning when artisans are working",
                "transportation_tips": "Behind Blue Mosque, walking distance from Sultanahmet tram",
                "local_tips": "Smaller than Grand Bazaar but more authentic. Watch artisans at work. Prices negotiable but quality is higher.",
                "seasonal_info": "Covered bazaar comfortable year-round, quieter in winter months"
            },
            {
                "name": "Balat Flea Market",
                "category": "Bazaar",
                "subcategory": "Vintage Market",
                "description": "Authentic Sunday flea market in historic Balat neighborhood with vintage items, antiques, and local atmosphere",
                "address": "Balat Mahallesi, various streets, 34087 Fatih/İstanbul",
                "district": "Fatih",
                "coordinates_lat": 41.0284,
                "coordinates_lng": 28.9489,
                "price_range": "Cheap to moderate",
                "opening_hours": "Sundays 8:00-16:00",
                "authenticity_score": 9.1,
                "local_rating": 9.0,
                "tourist_rating": 7.5,
                "crowd_level": "high on Sundays",
                "best_time_to_visit": "Early Sunday morning for best selection",
                "transportation_tips": "Bus 55T from Eminönü or Golden Horn ferry to Balat",
                "local_tips": "Bargaining expected and fun. Bring cash only. Mix of locals and antique hunters. Great photo opportunities.",
                "seasonal_info": "Outdoor market - check weather, more pleasant in mild seasons"
            },
            {
                "name": "Kapalıçarşı Traditional Workshops",
                "category": "Bazaar",
                "subcategory": "Master Craftsmen",
                "description": "Working workshops within Grand Bazaar where master artisans create carpets, jewelry, and traditional crafts",
                "address": "Beyazıt Mh., Kapalıçarşı, Fatih/İstanbul",
                "district": "Fatih",
                "coordinates_lat": 41.0108,
                "coordinates_lng": 28.9684,
                "price_range": "Free to watch (products expensive)",
                "opening_hours": "Mon-Sat: 9:00-19:00, closed Sundays",
                "authenticity_score": 9.0,
                "local_rating": 8.4,
                "tourist_rating": 8.9,
                "crowd_level": "moderate",
                "best_time_to_visit": "Mid-morning when masters are working",
                "transportation_tips": "Multiple entrances from Beyazıt or Sultanahmet",
                "local_tips": "Ask respectfully before photos. Some masters speak English and love sharing knowledge. Prices reflect quality and craftsmanship.",
                "seasonal_info": "Covered bazaar comfortable year-round, masters work daily except Sundays"
            },
            {
                "name": "Kadıköy Tuesday Market",
                "category": "Bazaar",
                "subcategory": "Local Market",
                "description": "Authentic neighborhood market where locals shop for fresh produce, spices, and household items",
                "address": "Kadıköy Salı Pazarı, various streets, Kadıköy/İstanbul",
                "district": "Kadıköy",
                "coordinates_lat": 40.9833,
                "coordinates_lng": 29.0264,
                "price_range": "Very affordable",
                "opening_hours": "Tuesdays 7:00-17:00",
                "authenticity_score": 9.3,
                "local_rating": 9.5,
                "tourist_rating": 7.2,
                "crowd_level": "very high on Tuesdays",
                "best_time_to_visit": "Early morning (7-10 AM) for freshest produce",
                "transportation_tips": "Ferry to Kadıköy, 5-minute walk from pier",
                "local_tips": "Bring reusable bags. Try seasonal fruits - vendors offer tastings. Best prices in Istanbul for fresh food.",
                "seasonal_info": "Seasonal produce selection varies, spring and summer have best variety"
            },
            {
                "name": "Golden Horn Ferry Route",
                "category": "Ferry Tour",
                "subcategory": "Historic Waterway",
                "description": "Scenic ferry journey along Golden Horn connecting historic neighborhoods with local commuter atmosphere",
                "address": "Eminönü - Eyüp - Sütlüce route",
                "district": "Various",
                "coordinates_lat": 41.0192,
                "coordinates_lng": 28.9735,
                "price_range": "15-25 TL",
                "opening_hours": "Daily 7:00-19:00 (seasonal variations)",
                "authenticity_score": 8.8,
                "local_rating": 8.5,
                "tourist_rating": 7.9,
                "crowd_level": "medium, higher in weekends",
                "best_time_to_visit": "Late afternoon for golden light on historic buildings",
                "transportation_tips": "Departs from Eminönü ferry terminal, regular schedule",
                "local_tips": "Mix of commuters and tourists. Stand on deck for photos. Connect to Pierre Loti Hill at Eyüp stop.",
                "seasonal_info": "Year-round service, more romantic in autumn and spring weather"
            },
            {
                "name": "Bosphorus Night Ferry",
                "category": "Ferry Tour",
                "subcategory": "Evening Cruise",
                "description": "Evening ferry service along Bosphorus with illuminated palaces and bridges, used by locals and tourists",
                "address": "Eminönü - Kavacık route",
                "district": "Various",
                "coordinates_lat": 41.0192,
                "coordinates_lng": 28.9735,
                "price_range": "25-40 TL",
                "opening_hours": "Evening departures (check seasonal schedule)",
                "authenticity_score": 8.0,
                "local_rating": 7.8,
                "tourist_rating": 9.2,
                "crowd_level": "high in summer evenings",
                "best_time_to_visit": "Sunset departure for transition from day to night",
                "transportation_tips": "Book in advance during summer, departs from Eminönü",
                "local_tips": "Bring light jacket for evening breeze. Upper deck has best views but gets crowded. Tea available on board.",
                "seasonal_info": "Most popular March-October, limited winter schedule"
            },
            {
                "name": "Üsküdar-Beşiktaş Local Ferry",
                "category": "Ferry Tour",
                "subcategory": "Commuter Experience",
                "description": "Short Bosphorus crossing used daily by locals, authentic Istanbul commuter experience with great views",
                "address": "Üsküdar - Beşiktaş route",
                "district": "Various",
                "coordinates_lat": 41.0225,
                "coordinates_lng": 29.0155,
                "price_range": "8-15 TL",
                "opening_hours": "Daily 6:00-24:00 frequent service",
                "authenticity_score": 9.2,
                "local_rating": 9.0,
                "tourist_rating": 8.1,
                "crowd_level": "high during rush hours",
                "best_time_to_visit": "Mid-morning or early afternoon to avoid commuter rush",
                "transportation_tips": "Most frequent ferry service, 15-minute journey",
                "local_tips": "Stand outside for photos. Listen to locals' daily conversations. Quick way to cross continents.",
                "seasonal_info": "Year-round reliable service, beautiful in all weather conditions"
            },
            {
                "name": "Balat Antique Neighborhood",
                "category": "Historic Neighborhood",
                "subcategory": "Colorful Houses",
                "description": "Historic Jewish quarter with colorful Ottoman houses, antique shops, and authentic café culture",
                "address": "Balat Mahallesi, 34087 Fatih/İstanbul",
                "district": "Fatih",
                "coordinates_lat": 41.0267,
                "coordinates_lng": 28.9478,
                "price_range": "Free to explore",
                "opening_hours": "Always accessible",
                "authenticity_score": 8.7,
                "local_rating": 8.5,
                "tourist_rating": 9.3,
                "crowd_level": "high on weekends",
                "best_time_to_visit": "Weekday morning for photos without crowds",
                "transportation_tips": "Bus from Eminönü or Golden Horn ferry to Fener",
                "local_tips": "Respect residents' privacy. Try local coffee shops. Combine with Fener Greek district.",
                "seasonal_info": "Beautiful year-round, spring flowers add extra color"
            },
            {
                "name": "Nardis Jazz Club",
                "category": "Nightlife",
                "subcategory": "Live Music",
                "description": "Intimate jazz club in Galata with live performances, authentic atmosphere, and local music scene",
                "address": "Kuledibi Sk. No:14, 34420 Beyoğlu/İstanbul",
                "district": "Beyoğlu",
                "coordinates_lat": 41.0258,
                "coordinates_lng": 28.9744,
                "price_range": "100-200 TL entry + drinks",
                "opening_hours": "Tue-Sun 21:00-02:00",
                "authenticity_score": 9.0,
                "local_rating": 9.2,
                "tourist_rating": 8.8,
                "crowd_level": "intimate (small venue)",
                "best_time_to_visit": "Weeknight shows for more authentic local crowd",
                "transportation_tips": "10-minute walk from Galata Tower, taxi recommended late night",
                "local_tips": "Reservations essential. Small venue with great acoustics. Mix of local musicians and international acts.",
                "seasonal_info": "Year-round programming, winter months have more regular local acts"
            }
        ]
        
        for attraction in curated_attractions:
            self.cursor.execute('''
            INSERT INTO attractions (
                name, category, subcategory, description, address, district,
                coordinates_lat, coordinates_lng, price_range, opening_hours,
                authenticity_score, local_rating, tourist_rating, crowd_level,
                best_time_to_visit, transportation_tips, local_tips, seasonal_info,
                source, verification_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                attraction['name'], attraction['category'], attraction['subcategory'],
                attraction['description'], attraction['address'], attraction['district'],
                attraction['coordinates_lat'], attraction['coordinates_lng'],
                attraction['price_range'], attraction['opening_hours'],
                attraction['authenticity_score'], attraction['local_rating'],
                attraction['tourist_rating'], attraction['crowd_level'],
                attraction['best_time_to_visit'], attraction['transportation_tips'],
                attraction['local_tips'], attraction['seasonal_info'],
                'expert_curation', 'verified'
            ))
        
        self.conn.commit()
    
    def get_attractions_by_category(self, category: str, limit: int = 10) -> List[Dict]:
        """Get attractions filtered by category"""
        self.cursor.execute('''
        SELECT id, name, category, subcategory, description, district,
               authenticity_score, local_rating, tourist_rating, crowd_level,
               best_time_to_visit, local_tips, price_range
        FROM attractions
        WHERE category = ?
        ORDER BY authenticity_score DESC, local_rating DESC
        LIMIT ?
        ''', (category, limit))
        
        return [
            {
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'subcategory': row[3],
                'description': row[4],
                'district': row[5],
                'authenticity_score': row[6],
                'local_rating': row[7],
                'tourist_rating': row[8],
                'crowd_level': row[9],
                'best_time_to_visit': row[10],
                'local_tips': row[11],
                'price_range': row[12]
            }
            for row in self.cursor.fetchall()
        ]
    
    def get_top_authentic_attractions(self, limit: int = 10) -> List[Dict]:
        """Get top attractions by authenticity score"""
        self.cursor.execute('''
        SELECT id, name, category, description, district, authenticity_score,
               local_rating, best_time_to_visit, local_tips
        FROM attractions
        ORDER BY authenticity_score DESC, local_rating DESC
        LIMIT ?
        ''', (limit,))
        
        return [
            {
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'description': row[3],
                'district': row[4],
                'authenticity_score': row[5],
                'local_rating': row[6],
                'best_time_to_visit': row[7],
                'local_tips': row[8]
            }
            for row in self.cursor.fetchall()
        ]
    
    def search_attractions(self, query: str, category: str = None) -> List[Dict]:
        """Search attractions by text query"""
        search_pattern = f"%{query.lower()}%"
        
        where_conditions = [
            "(LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(local_tips) LIKE ?)"
        ]
        params = [search_pattern, search_pattern, search_pattern]
        
        if category:
            where_conditions.append("category = ?")
            params.append(category)
        
        where_clause = " AND ".join(where_conditions)
        
        self.cursor.execute(f'''
        SELECT id, name, category, subcategory, description, district,
               authenticity_score, local_rating, tourist_rating,
               best_time_to_visit, local_tips
        FROM attractions
        WHERE {where_clause}
        ORDER BY authenticity_score DESC
        ''', params)
        
        return [
            {
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'subcategory': row[3],
                'description': row[4],
                'district': row[5],
                'authenticity_score': row[6],
                'local_rating': row[7],
                'tourist_rating': row[8],
                'best_time_to_visit': row[9],
                'local_tips': row[10]
            }
            for row in self.cursor.fetchall()
        ]
    
    def add_scraped_data(self, source: str, external_id: str, name: str,
                        category: str, rating: float, review_count: int,
                        coordinates: Tuple[float, float], address: str,
                        price_level: str, raw_data: Dict) -> str:
        """Add scraped data to curation queue"""
        scraped_id = str(uuid.uuid4())
        
        self.cursor.execute('''
        INSERT INTO scraped_raw_data (
            id, source, external_id, name, category, rating, review_count,
            coordinates_lat, coordinates_lng, address, price_level, raw_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scraped_id, source, external_id, name, category, rating, review_count,
            coordinates[0], coordinates[1], address, price_level, json.dumps(raw_data)
        ))
        
        # Add to curation queue
        curation_id = str(uuid.uuid4())
        priority = self._calculate_curation_priority(rating, review_count, category)
        
        self.cursor.execute('''
        INSERT INTO curation_queue (
            id, scraped_data_id, priority
        ) VALUES (?, ?, ?)
        ''', (curation_id, scraped_id, priority))
        
        self.conn.commit()
        return scraped_id
    
    def _calculate_curation_priority(self, rating: float, review_count: int, category: str) -> int:
        """Calculate priority for manual curation (1-10, higher = more priority)"""
        priority = 5  # Base priority
        
        # High rating boost
        if rating >= 4.5:
            priority += 2
        elif rating >= 4.0:
            priority += 1
        
        # Review count boost
        if review_count >= 100:
            priority += 2
        elif review_count >= 50:
            priority += 1
        
        # Category priority
        high_priority_categories = ['Mosque', 'Bazaar', 'Cultural Experience']
        if category in high_priority_categories:
            priority += 1
        
        return min(10, priority)
    
    def get_curation_queue(self, limit: int = 20) -> List[Dict]:
        """Get items pending manual curation"""
        self.cursor.execute('''
        SELECT cq.id, cq.priority, srd.name, srd.category, srd.rating,
               srd.review_count, srd.source, srd.scraped_at
        FROM curation_queue cq
        JOIN scraped_raw_data srd ON cq.scraped_data_id = srd.id
        WHERE cq.processed = FALSE
        ORDER BY cq.priority DESC, srd.scraped_at DESC
        LIMIT ?
        ''', (limit,))
        
        return [
            {
                'curation_id': row[0],
                'priority': row[1],
                'name': row[2],
                'category': row[3],
                'rating': row[4],
                'review_count': row[5],
                'source': row[6],
                'scraped_at': row[7]
            }
            for row in self.cursor.fetchall()
        ]
    
    def approve_scraped_attraction(self, curation_id: str, curator_notes: str = "") -> int:
        """Approve scraped data and convert to attraction"""
        # Get scraped data
        self.cursor.execute('''
        SELECT srd.name, srd.category, srd.rating, srd.coordinates_lat,
               srd.coordinates_lng, srd.address, srd.raw_data, srd.id
        FROM curation_queue cq
        JOIN scraped_raw_data srd ON cq.scraped_data_id = srd.id
        WHERE cq.id = ?
        ''', (curation_id,))
        
        result = self.cursor.fetchone()
        if not result:
            raise ValueError("Curation item not found")
        
        name, category, rating, lat, lng, address, raw_data_json, scraped_id = result
        raw_data = json.loads(raw_data_json)
        
        # Estimate authenticity score based on scraped data
        authenticity_score = self._estimate_authenticity_score(raw_data, rating)
        
        # Create new attraction
        self.cursor.execute('''
        INSERT INTO attractions (
            name, category, description, address, coordinates_lat, coordinates_lng,
            authenticity_score, local_rating, tourist_rating, source,
            verification_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, category, raw_data.get('description', f"Curated {category} in Istanbul"),
            address, lat, lng, authenticity_score, rating, rating,
            'scraping_curated', 'pending_verification'
        ))
        
        attraction_id = self.cursor.lastrowid
        
        # Update scraped data and curation queue
        self.cursor.execute('''
        UPDATE scraped_raw_data 
        SET curation_status = 'approved', curated_attraction_id = ?
        WHERE id = ?
        ''', (attraction_id, scraped_id))
        
        self.cursor.execute('''
        UPDATE curation_queue 
        SET processed = TRUE, processed_at = ?, curator_notes = ?, recommendation = 'approve'
        WHERE id = ?
        ''', (datetime.now().isoformat(), curator_notes, curation_id))
        
        self.conn.commit()
        return attraction_id
    
    def _estimate_authenticity_score(self, raw_data: Dict, rating: float) -> float:
        """Estimate authenticity score from scraped data"""
        base_score = 5.0
        
        # Rating contribution
        if rating >= 4.5:
            base_score += 2.0
        elif rating >= 4.0:
            base_score += 1.0
        
        # Review keywords analysis (mock implementation)
        description = raw_data.get('description', '').lower()
        authentic_keywords = ['local', 'traditional', 'authentic', 'historic', 'cultural']
        tourist_trap_keywords = ['tourist', 'souvenir', 'overpriced']
        
        keyword_score = 0
        for keyword in authentic_keywords:
            if keyword in description:
                keyword_score += 0.3
        
        for keyword in tourist_trap_keywords:
            if keyword in description:
                keyword_score -= 0.5
        
        final_score = min(10.0, max(1.0, base_score + keyword_score))
        return round(final_score, 1)
    
    def get_attractions_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attraction database statistics"""
        stats = {}
        
        # Total attractions
        self.cursor.execute('SELECT COUNT(*) FROM attractions')
        stats['total_attractions'] = self.cursor.fetchone()[0]
        
        # By category
        self.cursor.execute('''
        SELECT category, COUNT(*), AVG(authenticity_score)
        FROM attractions
        GROUP BY category
        ORDER BY COUNT(*) DESC
        ''')
        
        stats['by_category'] = [
            {
                'category': row[0],
                'count': row[1],
                'avg_authenticity': round(row[2], 2)
            }
            for row in self.cursor.fetchall()
        ]
        
        # By district
        self.cursor.execute('''
        SELECT district, COUNT(*)
        FROM attractions
        WHERE district IS NOT NULL
        GROUP BY district
        ORDER BY COUNT(*) DESC
        LIMIT 10
        ''')
        
        stats['by_district'] = dict(self.cursor.fetchall())
        
        # Authenticity distribution
        self.cursor.execute('''
        SELECT 
            AVG(authenticity_score) as avg_auth,
            MIN(authenticity_score) as min_auth,
            MAX(authenticity_score) as max_auth
        FROM attractions
        ''')
        
        result = self.cursor.fetchone()
        stats['authenticity_stats'] = {
            'average': round(result[0], 2),
            'minimum': result[1],
            'maximum': result[2]
        }
        
        # Curation queue stats
        self.cursor.execute('SELECT COUNT(*) FROM curation_queue WHERE processed = FALSE')
        stats['pending_curation'] = self.cursor.fetchone()[0]
        
        self.cursor.execute('SELECT COUNT(*) FROM scraped_raw_data')
        stats['total_scraped_data'] = self.cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

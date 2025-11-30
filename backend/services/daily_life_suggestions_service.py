"""
Daily Life Suggestions Service
Provides authentic daily-life suggestions for both tourists and locals
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import random

class AudienceType(Enum):
    TOURIST = "tourist"
    LOCAL = "local"
    BOTH = "both"

class SuggestionCategory(Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    COFFEE = "coffee"
    SHOPPING = "shopping"
    CULTURE = "culture"
    NATURE = "nature"
    NIGHTLIFE = "nightlife"
    TRANSPORT = "transport"
    LIFESTYLE = "lifestyle"

@dataclass
class DailyLifeSuggestion:
    id: str
    category: SuggestionCategory
    title: str
    description: str
    location: str
    audience: AudienceType
    authenticity_score: float
    local_tips: str
    cultural_context: str
    time_specific: str
    price_range: str
    accessibility: str

class DailyLifeSuggestionsService:
    """Service for providing authentic daily-life suggestions"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self.init_database()
        self._populate_default_suggestions()
        
    def init_database(self):
        """Initialize the daily life suggestions database"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Daily life suggestions table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_suggestions (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            location TEXT,
            district TEXT,
            audience TEXT NOT NULL,  -- tourist, local, both
            authenticity_score REAL DEFAULT 0.0,
            local_tips TEXT,
            cultural_context TEXT,
            time_specific TEXT,  -- morning, afternoon, evening, night
            price_range TEXT,
            accessibility TEXT,
            seasonal_relevance TEXT,  -- JSON array of months
            validation_source TEXT,  -- local_validation, user_feedback, expert_review
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Local validation table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS local_validations (
            id TEXT PRIMARY KEY,
            suggestion_id TEXT NOT NULL,
            validator_type TEXT NOT NULL,  -- local_resident, guide, expert
            validation_score REAL NOT NULL,
            validation_comment TEXT,
            validated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (suggestion_id) REFERENCES daily_suggestions (id)
        )
        ''')
        
        # Cultural context database
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS cultural_contexts (
            id TEXT PRIMARY KEY,
            context_type TEXT NOT NULL,  -- etiquette, tradition, language, behavior
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            importance_level TEXT NOT NULL,  -- high, medium, low
            applies_to TEXT,  -- tourists, locals, both
            examples TEXT,  -- JSON array of examples
            related_locations TEXT  -- JSON array of location types
        )
        ''')
        
        # Authenticity criteria
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS authenticity_criteria (
            id TEXT PRIMARY KEY,
            criterion_name TEXT NOT NULL,
            description TEXT NOT NULL,
            weight REAL DEFAULT 1.0,
            measurement_method TEXT,
            examples TEXT  -- JSON array
        )
        ''')
        
        self.conn.commit()
    
    def _populate_default_suggestions(self):
        """Populate with authentic daily life suggestions"""
        # Check if already populated
        self.cursor.execute('SELECT COUNT(*) FROM daily_suggestions')
        if self.cursor.fetchone()[0] > 0:
            return
        
        suggestions = [
            # Tourist Suggestions
            {
                "category": "BREAKFAST",
                "title": "Traditional Turkish Breakfast in Cihangir",
                "description": "Experience authentic Turkish breakfast (kahvaltı) at a local neighborhood café in Cihangir, away from tourist traps",
                "location": "Cihangir Neighborhood Cafes",
                "district": "Beyoğlu",
                "audience": "TOURIST",
                "authenticity_score": 9.2,
                "local_tips": "Order 'serpme kahvaltı' (spread breakfast) and take your time - Turks never rush breakfast. Say 'Afiyet olsun' when you start eating.",
                "cultural_context": "Turkish breakfast is a social ritual, not just a meal. It's meant to be shared and savored slowly with conversation.",
                "time_specific": "morning",
                "price_range": "40-80 TL per person",
                "accessibility": "Easy walking from Tünel",
                "seasonal_relevance": json.dumps([1,2,3,4,5,6,7,8,9,10,11,12])
            },
            {
                "category": "COFFEE",
                "title": "Turkish Coffee with Locals in Fatih",
                "description": "Join locals for traditional Turkish coffee and backgammon at century-old coffeehouses (kahvehane) in historic Fatih",
                "location": "Traditional Kahvehanes in Fatih",
                "district": "Fatih",
                "audience": "TOURIST",
                "authenticity_score": 9.5,
                "local_tips": "Don't add sugar after brewing. Watch a game of tavla (backgammon) and locals might invite you to play. Tip: order with 'az şekerli' (little sugar).",
                "cultural_context": "Coffeehouses are male-dominated social spaces. Women travelers should check the atmosphere first, though most welcome respectful visitors.",
                "time_specific": "afternoon",
                "price_range": "15-25 TL",
                "accessibility": "Near major mosques, easy to find",
                "seasonal_relevance": json.dumps([1,2,3,4,5,6,7,8,9,10,11,12])
            },
            {
                "category": "SHOPPING",
                "title": "Local Market Shopping in Beşiktaş Pazarı",
                "description": "Shop where Istanbul families shop - Beşiktaş Saturday Market for authentic Turkish products and prices",
                "location": "Beşiktaş Cumartesi Pazarı",
                "district": "Beşiktaş",
                "audience": "TOURIST",
                "authenticity_score": 8.8,
                "local_tips": "Bring a fabric bag, bargain is expected but don't go too low. Try seasonal fruits and ask for 'tadım' (taste). Best bargains after 4 PM.",
                "cultural_context": "Turkish markets are community gathering places. Vendors often become friends with regular customers and offer tea.",
                "time_specific": "morning",
                "price_range": "Much cheaper than tourist areas",
                "accessibility": "15 min walk from Beşiktaş metro",
                "seasonal_relevance": json.dumps([1,2,3,4,5,6,7,8,9,10,11,12])
            },
            
            # Local Suggestions
            {
                "category": "LUNCH",
                "title": "Hidden Lunch Spots in İstiklal Side Streets",
                "description": "Discover the small lokanta (eateries) that locals use for quick, quality lunch away from İstiklal Avenue crowds",
                "location": "Side streets of İstiklal Avenue",
                "district": "Beyoğlu",
                "audience": "LOCAL",
                "authenticity_score": 9.0,
                "local_tips": "Look for places with handwritten daily menus. 'Günün yemeği' (today's meal) is always fresh. Most locals eat lunch 12:30-2 PM.",
                "cultural_context": "Turkish workers prefer set meals (tabldot) with soup, main course, and ayran. Quick but never rushed eating.",
                "time_specific": "afternoon",
                "price_range": "25-45 TL",
                "accessibility": "5-10 min walk from İstiklal",
                "seasonal_relevance": json.dumps([1,2,3,4,5,6,7,8,9,10,11,12])
            },
            {
                "category": "CULTURE",
                "title": "Neighborhood Tea Gardens (Çay Bahçesi)",
                "description": "Evening tea at local çay bahçesi where Istanbul residents unwind after work with friends and dominos",
                "location": "Neighborhood tea gardens",
                "district": "Various",
                "audience": "LOCAL",
                "authenticity_score": 9.3,
                "local_tips": "Order tea in small glasses ('ince belli bardak'). Games of okey and tavla are social - watching is welcome. Evening crowd after 6 PM.",
                "cultural_context": "Tea gardens are democratic spaces where all social classes mix. Essential part of Istanbul social life.",
                "time_specific": "evening",
                "price_range": "8-15 TL per tea",
                "accessibility": "Found in every district",
                "seasonal_relevance": json.dumps([4,5,6,7,8,9,10])
            },
            {
                "category": "TRANSPORT",
                "title": "Master the Dolmuş System",
                "description": "Learn to use shared taxis (dolmuş) like a local for efficient, cheap transportation to hard-to-reach places",
                "location": "Major dolmuş stops",
                "district": "Various",
                "audience": "LOCAL",
                "authenticity_score": 8.5,
                "local_tips": "Wave to stop, say destination before entering. Pay when leaving. Squeeze in - there's always room for one more. Know the route names.",
                "cultural_context": "Dolmuş represents Turkish 'we'll make it work' mentality. Drivers are local area experts and often help with directions.",
                "time_specific": "all day",
                "price_range": "8-25 TL depending on distance",
                "accessibility": "Learning curve required",
                "seasonal_relevance": json.dumps([1,2,3,4,5,6,7,8,9,10,11,12])
            },
            
            # Both Audiences
            {
                "category": "NATURE",
                "title": "Sunset at Pierre Loti Hill",
                "description": "Watch sunset over Golden Horn from this authentic local viewpoint, accessible by historic cable car",
                "location": "Pierre Loti Tepesi",
                "district": "Eyüp",
                "audience": "BOTH",
                "authenticity_score": 8.7,
                "local_tips": "Take the teleferik (cable car) up. Arrive 30 min before sunset. Order Turkish tea and simit. Locals come for weekend family time.",
                "cultural_context": "Named after French writer Pierre Loti who loved Istanbul. Popular for wedding photos and family outings.",
                "time_specific": "evening",
                "price_range": "Teleferik 15 TL, tea 10 TL",
                "accessibility": "Cable car from Eyüp",
                "seasonal_relevance": json.dumps([3,4,5,6,7,8,9,10])
            },
            {
                "category": "NIGHTLIFE",
                "title": "Authentic Meyhane Experience",
                "description": "Traditional Turkish tavern (meyhane) experience with meze, rakı, and live music in Beyoğlu or Kadıköy",
                "location": "Traditional meyhanes",
                "district": "Beyoğlu/Kadıköy",
                "audience": "BOTH",
                "authenticity_score": 9.1,
                "local_tips": "Start with cold meze, then hot. Rakı is sipped, never rushed. Live music starts after 10 PM. Reservations needed weekends.",
                "cultural_context": "Meyhane culture is about slow conversation, friendship, and enjoying life. 'Keyif' (pleasure/joy) is the goal, not getting drunk.",
                "time_specific": "evening",
                "price_range": "150-300 TL per person",
                "accessibility": "Most accessible via metro/ferry",
                "seasonal_relevance": json.dumps([1,2,3,4,5,6,7,8,9,10,11,12])
            },
            {
                "category": "LIFESTYLE",
                "title": "Hamam (Turkish Bath) Tradition",
                "description": "Experience traditional Turkish bath culture at authentic neighborhood hamam, not tourist versions",
                "location": "Neighborhood hamams",
                "district": "Various",
                "audience": "BOTH",
                "authenticity_score": 8.9,
                "local_tips": "Bring your own towel or rent one. Scrub service (kese) is optional but recommended. Relax afterwards with tea. Gender-separated or family sections.",
                "cultural_context": "Hamam is about purification and relaxation, not just cleansing. Important in Turkish culture for centuries. Social bonding experience.",
                "time_specific": "afternoon",
                "price_range": "50-150 TL depending on services",
                "accessibility": "Found in most historic districts",
                "seasonal_relevance": json.dumps([1,2,3,4,5,6,7,8,9,10,11,12])
            }
        ]
        
        for suggestion in suggestions:
            suggestion_id = f"daily_{hash(suggestion['title'])}_{random.randint(1000, 9999)}"
            suggestion['id'] = suggestion_id
            
            self.cursor.execute('''
            INSERT INTO daily_suggestions (
                id, category, title, description, location, district, audience,
                authenticity_score, local_tips, cultural_context, time_specific,
                price_range, accessibility, seasonal_relevance, validation_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                suggestion_id, suggestion['category'], suggestion['title'],
                suggestion['description'], suggestion['location'], suggestion['district'],
                suggestion['audience'], suggestion['authenticity_score'],
                suggestion['local_tips'], suggestion['cultural_context'],
                suggestion['time_specific'], suggestion['price_range'],
                suggestion['accessibility'], suggestion['seasonal_relevance'],
                'expert_curation'
            ))
        
        self.conn.commit()
    
    def get_suggestions_for_audience(self, audience: str, category: str = None,
                                   time_specific: str = None, limit: int = 10) -> List[Dict]:
        """Get suggestions filtered by audience type"""
        where_conditions = ["audience = ? OR audience = 'BOTH'"]
        params = [audience.upper()]
        
        if category:
            where_conditions.append("category = ?")
            params.append(category.upper())
        
        if time_specific:
            where_conditions.append("time_specific = ?")
            params.append(time_specific.lower())
        
        where_clause = " AND ".join(where_conditions)
        params.append(limit)
        
        self.cursor.execute(f'''
        SELECT id, category, title, description, location, district,
               authenticity_score, local_tips, cultural_context,
               time_specific, price_range, accessibility
        FROM daily_suggestions
        WHERE {where_clause}
        ORDER BY authenticity_score DESC, RANDOM()
        LIMIT ?
        ''', params)
        
        return [
            {
                'id': row[0],
                'category': row[1],
                'title': row[2],
                'description': row[3],
                'location': row[4],
                'district': row[5],
                'authenticity_score': row[6],
                'local_tips': row[7],
                'cultural_context': row[8],
                'time_specific': row[9],
                'price_range': row[10],
                'accessibility': row[11]
            }
            for row in self.cursor.fetchall()
        ]
    
    def get_tourist_suggestions(self, category: str = None, time_specific: str = None) -> List[Dict]:
        """Get suggestions specifically curated for tourists"""
        return self.get_suggestions_for_audience('TOURIST', category, time_specific)
    
    def get_local_suggestions(self, category: str = None, time_specific: str = None) -> List[Dict]:
        """Get suggestions for locals discovering their own city"""
        return self.get_suggestions_for_audience('LOCAL', category, time_specific)
    
    def get_daily_schedule_suggestions(self, audience: str, date: datetime = None) -> Dict[str, List]:
        """Get a full daily schedule of suggestions"""
        if date is None:
            date = datetime.now()
        
        schedule = {
            'morning': self.get_suggestions_for_audience(audience, time_specific='morning', limit=3),
            'afternoon': self.get_suggestions_for_audience(audience, time_specific='afternoon', limit=3),
            'evening': self.get_suggestions_for_audience(audience, time_specific='evening', limit=3)
        }
        
        # Add cultural context for the day
        schedule['cultural_tips'] = self._get_daily_cultural_tips(date)
        
        return schedule
    
    def _get_daily_cultural_tips(self, date: datetime) -> List[str]:
        """Get cultural tips relevant to the current day/season"""
        month = date.month
        weekday = date.weekday()  # 0 = Monday
        
        tips = []
        
        if weekday == 4:  # Friday
            tips.append("Friday is prayer day - mosques are busiest from 12:30-1:30 PM")
        
        if weekday in [5, 6]:  # Weekend
            tips.append("Weekends are family time - parks and recreational areas will be busy")
        
        if month in [6, 7, 8]:  # Summer
            tips.append("Summer heat: locals avoid midday sun, best activities before 11 AM or after 6 PM")
        
        if month in [12, 1, 2]:  # Winter
            tips.append("Winter season: indoor activities preferred, tea consumption increases significantly")
        
        return tips
    
    def add_local_validation(self, suggestion_id: str, validator_type: str,
                           score: float, comment: str = "") -> str:
        """Add local validation for a suggestion"""
        validation_id = f"val_{random.randint(10000, 99999)}"
        
        self.cursor.execute('''
        INSERT INTO local_validations (
            id, suggestion_id, validator_type, validation_score, validation_comment
        ) VALUES (?, ?, ?, ?, ?)
        ''', (validation_id, suggestion_id, validator_type, score, comment))
        
        self.conn.commit()
        
        # Update suggestion authenticity score based on validations
        self._update_authenticity_score(suggestion_id)
        
        return validation_id
    
    def _update_authenticity_score(self, suggestion_id: str):
        """Update authenticity score based on local validations"""
        self.cursor.execute('''
        SELECT AVG(validation_score), COUNT(*)
        FROM local_validations
        WHERE suggestion_id = ?
        ''', (suggestion_id,))
        
        result = self.cursor.fetchone()
        if result and result[1] > 0:  # Has validations
            avg_validation = result[0]
            validation_count = result[1]
            
            # Weight validation score with confidence factor
            confidence = min(1.0, validation_count / 3.0)  # Full confidence at 3+ validations
            
            # Get current base score
            self.cursor.execute('''
            SELECT authenticity_score FROM daily_suggestions WHERE id = ?
            ''', (suggestion_id,))
            base_score = self.cursor.fetchone()[0]
            
            # Calculate new score (blend base with validation)
            new_score = base_score * (1 - confidence) + avg_validation * confidence
            
            # Update suggestion
            self.cursor.execute('''
            UPDATE daily_suggestions 
            SET authenticity_score = ?, updated_at = ?
            WHERE id = ?
            ''', (round(new_score, 2), datetime.now().isoformat(), suggestion_id))
            
            self.conn.commit()
    
    def get_authenticity_analytics(self) -> Dict[str, Any]:
        """Get analytics on suggestion authenticity and validation"""
        analytics = {}
        
        # Overall authenticity distribution
        self.cursor.execute('''
        SELECT 
            AVG(authenticity_score) as avg_auth,
            MIN(authenticity_score) as min_auth,
            MAX(authenticity_score) as max_auth,
            COUNT(*) as total_suggestions
        FROM daily_suggestions
        ''')
        
        result = self.cursor.fetchone()
        analytics['authenticity_stats'] = {
            'average': round(result[0], 2),
            'minimum': result[1],
            'maximum': result[2],
            'total_suggestions': result[3]
        }
        
        # Audience distribution
        self.cursor.execute('''
        SELECT audience, COUNT(*) 
        FROM daily_suggestions 
        GROUP BY audience
        ''')
        
        analytics['audience_distribution'] = dict(self.cursor.fetchall())
        
        # Category distribution
        self.cursor.execute('''
        SELECT category, COUNT(*), AVG(authenticity_score)
        FROM daily_suggestions 
        GROUP BY category
        ORDER BY AVG(authenticity_score) DESC
        ''')
        
        analytics['category_stats'] = [
            {
                'category': row[0],
                'count': row[1],
                'avg_authenticity': round(row[2], 2)
            }
            for row in self.cursor.fetchall()
        ]
        
        # Validation stats
        self.cursor.execute('''
        SELECT COUNT(*) as total_validations,
               AVG(validation_score) as avg_validation
        FROM local_validations
        ''')
        
        result = self.cursor.fetchone()
        analytics['validation_stats'] = {
            'total_validations': result[0],
            'average_validation_score': round(result[1] or 0, 2)
        }
        
        return analytics
    
    def search_suggestions(self, query: str, audience: str = None) -> List[Dict]:
        """Search suggestions by text query"""
        search_pattern = f"%{query.lower()}%"
        
        where_conditions = [
            "(LOWER(title) LIKE ? OR LOWER(description) LIKE ? OR LOWER(local_tips) LIKE ?)"
        ]
        params = [search_pattern, search_pattern, search_pattern]
        
        if audience:
            where_conditions.append("(audience = ? OR audience = 'BOTH')")
            params.append(audience.upper())
        
        where_clause = " AND ".join(where_conditions)
        
        self.cursor.execute(f'''
        SELECT id, category, title, description, location, district,
               authenticity_score, local_tips, cultural_context,
               time_specific, price_range, accessibility
        FROM daily_suggestions
        WHERE {where_clause}
        ORDER BY authenticity_score DESC
        ''', params)
        
        return [
            {
                'id': row[0],
                'category': row[1],
                'title': row[2],
                'description': row[3],
                'location': row[4],
                'district': row[5],
                'authenticity_score': row[6],
                'local_tips': row[7],
                'cultural_context': row[8],
                'time_specific': row[9],
                'price_range': row[10],
                'accessibility': row[11]
            }
            for row in self.cursor.fetchall()
        ]
    
    def get_specific_locations(self, query: str, language: str = 'en') -> Dict[str, Any]:
        """
        Get specific location recommendations for common daily life queries.
        
        Args:
            query: User query (e.g., "Where can I buy a SIM card?")
            language: Response language
            
        Returns:
            Dict with specific locations and practical tips
        """
        query_lower = query.lower()
        
        # Detect query type
        location_data = None
        
        if any(term in query_lower for term in ['sim', 'phone', 'mobile', 'turkcell', 'vodafone']):
            location_data = self._get_sim_card_locations(language)
        elif any(term in query_lower for term in ['pharmacy', 'eczane', 'medicine', 'ilaç']):
            location_data = self._get_pharmacy_locations(language)
        elif any(term in query_lower for term in ['atm', 'bank', 'banka', 'money', 'para']):
            location_data = self._get_atm_bank_locations(language)
        elif any(term in query_lower for term in ['grocery', 'supermarket', 'market', 'migros']):
            location_data = self._get_grocery_locations(language)
        elif any(term in query_lower for term in ['post', 'ptt', 'mail', 'package', 'kargo']):
            location_data = self._get_post_office_locations(language)
        elif any(term in query_lower for term in ['hospital', 'doctor', 'clinic', 'hastane']):
            location_data = self._get_medical_locations(language)
        
        return location_data or self._get_fallback_tips(language)
    
    def _get_sim_card_locations(self, language: str) -> Dict[str, Any]:
        """Get SIM card shop locations"""
        if language == 'tr':
            return {
                'type': 'sim_card',
                'title': 'SIM Kart Alabileceğiniz Yerler',
                'locations': [
                    {
                        'name': 'Turkcell Mağazaları',
                        'areas': ['İstiklal Caddesi', 'Taksim', 'Kadıköy', 'Beşiktaş', 'Havalimanı'],
                        'description': 'En yaygın operatör, İstanbul\'da her yerde mağazası var',
                        'tip': 'Pasaport ve adres kaydı gerekli. Prepaid SIM kartlar 100-200 TL civarı.'
                    },
                    {
                        'name': 'Vodafone Mağazaları',
                        'areas': ['İstiklal', 'Mecidiyeköy', 'Nişantaşı', 'Havalimanı'],
                        'description': 'İyi veri paketleri ve İngilizce destek',
                        'tip': 'Turist paketleri mevcut, 30 günlük veri + konuşma paketleri.'
                    },
                    {
                        'name': 'Türk Telekom',
                        'areas': ['Taksim', 'Bakırköy', 'Ataşehir'],
                        'description': 'Yerli operatör, güvenilir kapsama',
                        'tip': 'Havalimanında da alabilirsiniz ama şehir merkezinde daha uygun.'
                    }
                ],
                'practical_tips': [
                    'Pasaportunuzu ve Türkiye\'deki adresi (otel adresi olabilir) yanınıza alın',
                    'Aktivasyon 2-4 saat sürebilir',
                    'Tourist SIM paketleri genelde 15-30 gün geçerli',
                    'Veri paketleri ayrı satılır, ilk alımda paket dahil olanları tercih edin'
                ]
            }
        else:  # English
            return {
                'type': 'sim_card',
                'title': 'Where to Buy SIM Cards',
                'locations': [
                    {
                        'name': 'Turkcell Stores',
                        'areas': ['İstiklal Avenue', 'Taksim', 'Kadıköy', 'Beşiktaş', 'Airport'],
                        'description': 'Largest operator, stores everywhere in Istanbul',
                        'tip': 'Need passport and Turkish address. Prepaid SIMs cost 100-200 TL.'
                    },
                    {
                        'name': 'Vodafone Stores',
                        'areas': ['İstiklal', 'Mecidiyeköy', 'Nişantaşı', 'Airport'],
                        'description': 'Good data packages and English support',
                        'tip': 'Tourist packages available, 30-day data + voice plans.'
                    },
                    {
                        'name': 'Türk Telekom',
                        'areas': ['Taksim', 'Bakırköy', 'Ataşehir'],
                        'description': 'Local operator, reliable coverage',
                        'tip': 'Can buy at airport but city center usually cheaper.'
                    }
                ],
                'practical_tips': [
                    'Bring your passport and a Turkish address (hotel address works)',
                    'Activation takes 2-4 hours',
                    'Tourist SIM packages usually valid for 15-30 days',
                    'Data packages sold separately, choose plans with data included'
                ]
            }
    
    def _get_pharmacy_locations(self, language: str) -> Dict[str, Any]:
        """Get pharmacy locations"""
        if language == 'tr':
            return {
                'type': 'pharmacy',
                'title': 'Eczane Bul',
                'locations': [
                    {
                        'name': 'Nöbetçi Eczane (24/7 Eczaneler)',
                        'areas': ['Her mahallede bir tane vardır'],
                        'description': 'Gece ve hafta sonları açık nöbetçi eczaneler',
                        'tip': 'Google Maps\'te "nöbetçi eczane" arayın veya www.istanbuleczaciodasi.org.tr sitesini kontrol edin'
                    },
                    {
                        'name': 'Zincir Eczaneler',
                        'areas': ['Taksim', 'Kadıköy', 'Beşiktaş', 'AVM\'lerde'],
                        'description': 'Güvenilir zincir eczaneler (Eczacıbaşı, Memleket vb.)',
                        'tip': 'Çoğu reçetesiz ilaç rahatça satılır. Antibiyotik için reçete şart.'
                    }
                ],
                'practical_tips': [
                    'Türkiye\'de çoğu ilaç reçetesiz alınabilir (ağrı kesici, soğuk algınlığı vb.)',
                    'Antibiyotikler için doktor reçetesi gerekli',
                    'Eczacılar genelde İngilizce bilir',
                    'İlaç fiyatları Avrupa\'ya göre düşük'
                ]
            }
        else:  # English
            return {
                'type': 'pharmacy',
                'title': 'Finding Pharmacies',
                'locations': [
                    {
                        'name': '24/7 On-Duty Pharmacies (Nöbetçi Eczane)',
                        'areas': ['Every neighborhood has one'],
                        'description': 'Rotating duty pharmacies open nights and weekends',
                        'tip': 'Search "nöbetçi eczane" on Google Maps or check www.istanbuleczaciodasi.org.tr'
                    },
                    {
                        'name': 'Chain Pharmacies',
                        'areas': ['Taksim', 'Kadıköy', 'Beşiktaş', 'Shopping malls'],
                        'description': 'Reliable chains (Eczacıbaşı, Memleket, etc.)',
                        'tip': 'Most over-the-counter meds sold easily. Antibiotics need prescription.'
                    }
                ],
                'practical_tips': [
                    'Many medications available without prescription (painkillers, cold medicine, etc.)',
                    'Antibiotics require doctor\'s prescription',
                    'Pharmacists usually speak English',
                    'Medicine prices lower than Europe'
                ]
            }
    
    def _get_atm_bank_locations(self, language: str) -> Dict[str, Any]:
        """Get ATM and bank locations"""
        if language == 'tr':
            return {
                'type': 'banking',
                'title': 'ATM ve Banka Şubeleri',
                'locations': [
                    {
                        'name': 'ATM\'ler',
                        'areas': ['Her yerde - Metro istasyonları, AVM\'ler, ana caddelerde'],
                        'description': 'Yaygın bankalar: Garanti, Akbank, İş Bankası, Yapı Kredi, QNB',
                        'tip': 'Yabancı kartlar çoğu ATM\'de çalışır. İşlem başına 10-30 TL komisyon'
                    },
                    {
                        'name': 'Döviz Büroları',
                        'areas': ['Taksim', 'Sultanahmet', 'Eminönü', 'Kadıköy', 'Havalimanı'],
                        'description': 'Euro ve Dolar bozdurmak için',
                        'tip': 'Havalimanından daha iyi kur. Birkaç büroyu karşılaştırın.'
                    }
                ],
                'practical_tips': [
                    'ATM\'ler Türkçe ve İngilizce seçeneği sunar',
                    'Maksimum çekim limiti genelde 2000-5000 TL',
                    'Gece ATM kullanırken dikkatli olun',
                    'Kartınızın yurtdışı kullanımını aktifleştirin'
                ]
            }
        else:  # English
            return {
                'type': 'banking',
                'title': 'ATMs and Banks',
                'locations': [
                    {
                        'name': 'ATMs',
                        'areas': ['Everywhere - Metro stations, malls, main streets'],
                        'description': 'Common banks: Garanti, Akbank, İş Bankası, Yapı Kredi, QNB',
                        'tip': 'Foreign cards work at most ATMs. Expect 10-30 TL fee per transaction'
                    },
                    {
                        'name': 'Currency Exchange Offices',
                        'areas': ['Taksim', 'Sultanahmet', 'Eminönü', 'Kadıköy', 'Airport'],
                        'description': 'For exchanging EUR and USD',
                        'tip': 'Better rates than airport. Compare a few offices.'
                    }
                ],
                'practical_tips': [
                    'ATMs offer Turkish and English options',
                    'Maximum withdrawal usually 2000-5000 TL',
                    'Be careful using ATMs at night',
                    'Enable international usage on your card'
                ]
            }
    
    def _get_grocery_locations(self, language: str) -> Dict[str, Any]:
        """Get grocery store locations"""
        if language == 'tr':
            return {
                'type': 'grocery',
                'title': 'Market ve Süpermarketler',
                'locations': [
                    {
                        'name': 'Migros',
                        'areas': ['İstanbul\'un her yerinde'],
                        'description': 'En yaygın zincir market',
                        'tip': 'Money Card ile indirimler. 24 saat açık şubeler var.'
                    },
                    {
                        'name': 'Şok / BİM / A101',
                        'areas': ['Her mahallede'],
                        'description': 'Ucuz diskont marketler',
                        'tip': 'Fiyatlar en uygun. Temel gıda ve ev eşyaları için ideal.'
                    },
                    {
                        'name': 'Carrefour / Macro',
                        'areas': ['AVM\'lerde ve büyük şubelerde'],
                        'description': 'Geniş ürün yelpazesi',
                        'tip': 'Uluslararası ürünler burada bulunur.'
                    }
                ],
                'practical_tips': [
                    'Pazar günleri pazarlar var (daha ucuz, taze sebze-meyve)',
                    'Bakkal (köşe market) her yerde, daha pahalı ama yakın',
                    'İstanbulkart marketlerde geçer',
                    'Nakit veya kart, her ikisi de kabul edilir'
                ]
            }
        else:  # English
            return {
                'type': 'grocery',
                'title': 'Grocery Stores and Supermarkets',
                'locations': [
                    {
                        'name': 'Migros',
                        'areas': ['All over Istanbul'],
                        'description': 'Most widespread chain',
                        'tip': 'Money Card for discounts. Some locations open 24 hours.'
                    },
                    {
                        'name': 'Şok / BİM / A101',
                        'areas': ['Every neighborhood'],
                        'description': 'Budget discount stores',
                        'tip': 'Cheapest prices. Great for basics and household items.'
                    },
                    {
                        'name': 'Carrefour / Macro',
                        'areas': ['Shopping malls and large branches'],
                        'description': 'Wide product range',
                        'tip': 'International products available here.'
                    }
                ],
                'practical_tips': [
                    'Sunday markets (pazar) have cheaper, fresh produce',
                    'Corner stores (bakkal) everywhere, pricier but convenient',
                    'IstanbulCard works at stores',
                    'Cash or card both accepted'
                ]
            }
    
    def _get_post_office_locations(self, language: str) -> Dict[str, Any]:
        """Get post office locations"""
        if language == 'tr':
            return {
                'type': 'post',
                'title': 'PTT ve Kargo',
                'locations': [
                    {
                        'name': 'PTT (Türk Postası)',
                        'areas': ['Her ilçede birçok şube'],
                        'description': 'Resmi posta hizmeti',
                        'tip': 'Mektup, paket gönderimi ve fatura ödeme. Genelde kuyruğu vardır.'
                    },
                    {
                        'name': 'Özel Kargolar (Yurtiçi, Aras, MNG)',
                        'areas': ['AVM\'ler ve ana caddelerde'],
                        'description': 'Hızlı kargo hizmetleri',
                        'tip': 'Aynı gün veya ertesi gün teslimat. PTT\'den daha hızlı.'
                    }
                ],
                'practical_tips': [
                    'PTT uluslararası gönderi yapabilir',
                    'Özel kargo şirketleri şehir içi için daha hızlı',
                    'Paket almak için kimlik gerekli',
                    'Online alışverişler genelde kapıya teslim'
                ]
            }
        else:  # English
            return {
                'type': 'post',
                'title': 'Post Offices and Shipping',
                'locations': [
                    {
                        'name': 'PTT (Turkish Post)',
                        'areas': ['Multiple branches in every district'],
                        'description': 'Official postal service',
                        'tip': 'Mail, package sending, and bill payment. Usually has queues.'
                    },
                    {
                        'name': 'Private Couriers (Yurtiçi, Aras, MNG)',
                        'areas': ['Malls and main streets'],
                        'description': 'Fast shipping services',
                        'tip': 'Same-day or next-day delivery. Faster than PTT.'
                    }
                ],
                'practical_tips': [
                    'PTT can handle international shipping',
                    'Private couriers faster for domestic packages',
                    'ID required to pick up packages',
                    'Online shopping usually delivers to door'
                ]
            }
    
    def _get_medical_locations(self, language: str) -> Dict[str, Any]:
        """Get hospital and clinic locations"""
        if language == 'tr':
            return {
                'type': 'medical',
                'title': 'Sağlık Hizmetleri',
                'locations': [
                    {
                        'name': 'Acil Servisler',
                        'areas': ['Her ilçede devlet hastanesi'],
                        'description': 'Acil durumlar için 112',
                        'tip': 'Devlet hastaneleri ücretsiz ama kalabalık. Özel hastaneler ücretli.'
                    },
                    {
                        'name': 'Özel Hastaneler (American Hospital, Acibadem vb.)',
                        'areas': ['Nişantaşı, Maslak, Kadıköy, Altunizade'],
                        'description': 'Yüksek kaliteli hizmet, İngilizce bilen personel',
                        'tip': 'Sigorta kabul edilir. Fiyatlar yüksek ama hizmet çok iyi.'
                    },
                    {
                        'name': 'Poliklinikler',
                        'areas': ['Her mahallede'],
                        'description': 'Küçük sağlık sorunları için',
                        'tip': 'Randevusuz muayene. Uygun fiyatlı.'
                    }
                ],
                'practical_tips': [
                    'Acil: 112 (ambulans)',
                    'Seyahat sigortanız varsa özel hastane tercih edin',
                    'Pasaport veya kimlik yanınızda olsun',
                    'Özel hastanelerde İngilizce konuşulur'
                ]
            }
        else:  # English
            return {
                'type': 'medical',
                'title': 'Medical Services',
                'locations': [
                    {
                        'name': 'Emergency Rooms',
                        'areas': ['Public hospital in every district'],
                        'description': 'For emergencies call 112',
                        'tip': 'Public hospitals free but crowded. Private hospitals charge fees.'
                    },
                    {
                        'name': 'Private Hospitals (American Hospital, Acibadem, etc.)',
                        'areas': ['Nişantaşı, Maslak, Kadıköy, Altunizade'],
                        'description': 'High quality service, English-speaking staff',
                        'tip': 'Insurance accepted. Expensive but excellent service.'
                    },
                    {
                        'name': 'Clinics (Poliklinik)',
                        'areas': ['Every neighborhood'],
                        'description': 'For minor health issues',
                        'tip': 'Walk-in consultations. Affordable prices.'
                    }
                ],
                'practical_tips': [
                    'Emergency: 112 (ambulance)',
                    'If you have travel insurance, use private hospitals',
                    'Bring your passport or ID',
                    'Private hospitals have English-speaking staff'
                ]
            }
    
    def _get_fallback_tips(self, language: str) -> Dict[str, Any]:
        """Fallback general tips when specific location not matched"""
        if language == 'tr':
            return {
                'type': 'general',
                'title': 'İstanbul\'da Günlük Hayat İpuçları',
                'tips': [
                    'Google Maps Türkçe ve İngilizce çalışır, çoğu yeri bulabilirsiniz',
                    'İstanbulkart her yerde geçer - metro, otobüs, tramvay, vapur',
                    'Taksim, Kadıköy, Beşiktaş\'ta her türlü hizmet bulabilirsiniz',
                    'Yerel halk yardımseverdir, sorduğunuzda yol gösterirler',
                    'AVM\'lerde (Alışveriş Merkezleri) hemen her şey var'
                ]
            }
        else:  # English
            return {
                'type': 'general',
                'title': 'Daily Life Tips in Istanbul',
                'tips': [
                    'Google Maps works in English, you can find most places',
                    'IstanbulCard works everywhere - metro, bus, tram, ferry',
                    'Taksim, Kadıköy, Beşiktaş have all services you need',
                    'Locals are helpful, just ask for directions',
                    'Shopping malls (AVM) have everything you need'
                ]
            }

    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

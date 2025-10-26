#!/usr/bin/env python3
"""
Enhanced Monthly Istanbul Events Scheduler
Extended version with comprehensive İKSV subdomain coverage
"""

import asyncio
import logging
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    import aiohttp
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    print("⚠️ Web scraping libraries not available. Install: pip install aiohttp beautifulsoup4")
    WEB_SCRAPING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# İKSV Venue Database for enhanced venue mapping
IKSV_VENUES = {
    'salon_iksv': {
        'name': {'tr': 'Salon İKSV', 'en': 'Salon İKSV'},
        'address': 'Sadi Konuralp Caddesi No:5 Şişhane, Beyoğlu',
        'capacity': 300,
        'accessibility': True,
    },
    'zorlu_psm': {
        'name': {'tr': 'Zorlu PSM', 'en': 'Zorlu PSM'},
        'address': 'Levazım, Koru Sokağı No:2, 34340 Beşiktaş',
        'capacity': 2000,
        'accessibility': True,
    },
    'harbiye': {
        'name': {'tr': 'Harbiye Muhsin Ertuğrul Sahnesi', 'en': 'Harbiye Muhsin Ertuğrul Stage'},
        'address': 'Harbiye, Taşkışla Cad. No:8, Şişli',
        'capacity': 800,
        'accessibility': True,
    },
    'paribu_art': {
        'name': {'tr': 'Paribu Art', 'en': 'Paribu Art'},
        'address': 'Paribu Art, Beyoğlu',
        'capacity': 500,
        'accessibility': True,
    },
    'moda_sahnesi': {
        'name': {'tr': 'Moda Sahnesi', 'en': 'Moda Sahnesi'},
        'address': 'Caferağa, Moda Cd., 34710 Kadıköy',
        'capacity': 400,
        'accessibility': True,
    },
    'halic_historic_inn': {
        'name': {'tr': 'Haliç Konukevi (Tarihi Han)', 'en': 'Historic Inn at Haliç'},
        'address': 'Haliç, İstanbul',
        'capacity': 200,
        'accessibility': False,
    },
    'orient_institut': {
        'name': {'tr': 'Orient-Institut Istanbul', 'en': 'Orient-Institut Istanbul'},
        'address': 'Susam Sok. No:16-18, Cihangir, Beyoğlu',
        'capacity': 100,
        'accessibility': True,
    },
}

# Enhanced URL configuration for comprehensive İKSV coverage
IKSV_URL_CONFIG = {
    'main': {
        'urls': [
            'https://www.iksv.org/en',
            'https://www.iksv.org/tr',
        ],
        'category': 'general',
        'priority': 1,
    },
    'music_festival': {
        'urls': [
            'https://muzik.iksv.org/en',
            'https://muzik.iksv.org/tr',
            'https://muzik.iksv.org/en/programme',
            'https://muzik.iksv.org/tr/program',
        ],
        'category': 'Music',
        'festival': 'Istanbul Music Festival',
        'priority': 2,
    },
    'film_festival': {
        'urls': [
            'https://film.iksv.org/en',
            'https://film.iksv.org/tr',
            'https://film.iksv.org/en/programme',
            'https://film.iksv.org/tr/program',
        ],
        'category': 'Film',
        'festival': 'Istanbul Film Festival',
        'priority': 2,
    },
    'jazz_festival': {
        'urls': [
            'https://caz.iksv.org/en',
            'https://caz.iksv.org/tr',
            'https://caz.iksv.org/en/programme',
            'https://caz.iksv.org/tr/program',
        ],
        'category': 'Music',
        'festival': 'Istanbul Jazz Festival',
        'priority': 2,
    },
    'theatre_festival': {
        'urls': [
            'https://tiyatro.iksv.org/en',
            'https://tiyatro.iksv.org/tr',
            'https://tiyatro.iksv.org/en/programme',
            'https://tiyatro.iksv.org/tr/program',
        ],
        'category': 'Theatre',
        'festival': 'Istanbul Theatre Festival',
        'priority': 2,
    },
    'biennial': {
        'urls': [
            'https://bienal.iksv.org/en',
            'https://bienal.iksv.org/tr',
            'https://bienal.iksv.org/en/18th-istanbul-biennial',
        ],
        'category': 'Art',
        'festival': 'Istanbul Biennial',
        'priority': 2,
    },
    'salon': {
        'urls': [
            'https://saloniksv.com/en',
            'https://saloniksv.com/tr',
            'https://saloniksv.com/en/programme',
            'https://saloniksv.com/tr/program',
        ],
        'category': 'Music',
        'venue': 'Salon İKSV',
        'priority': 1,
    },
    'filmekimi': {
        'urls': [
            'https://filmekimi.iksv.org/en',
            'https://filmekimi.iksv.org/tr',
        ],
        'category': 'Film',
        'festival': 'Filmekimi',
        'priority': 2,
    },
}


class EnhancedEventsScheduler:
    """Enhanced scheduler with comprehensive İKSV coverage and improved data quality"""
    
    def __init__(self):
        """Initialize the enhanced scheduler"""
        self.data_directory = Path("data/events")
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.current_events_file = self.data_directory / "current_events.json"
        self.enhanced_events_file = self.data_directory / "enhanced_events.json"
        self.last_fetch_file = self.data_directory / "last_fetch_enhanced.json"
        self.fetch_stats_file = self.data_directory / "fetch_statistics.json"
        
        # Statistics tracking
        self.fetch_stats = {
            'total_fetched': 0,
            'by_source': {},
            'by_category': {},
            'last_update': None,
            'errors': []
        }
        
        logger.info("📅 Enhanced Events Scheduler initialized")
    
    async def fetch_all_iksv_events(self) -> List[Dict[str, Any]]:
        """Fetch events from all İKSV domains with enhanced data"""
        if not WEB_SCRAPING_AVAILABLE:
            logger.error("❌ Web scraping libraries not available")
            return []
        
        all_events = []
        
        try:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context, limit=10)
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # Process each URL config group
                for source_key, config in IKSV_URL_CONFIG.items():
                    logger.info(f"🎯 Processing {source_key} ({config.get('category', 'general')})")
                    
                    source_events = []
                    for url in config['urls']:
                        try:
                            events = await self._fetch_from_url(session, url, config)
                            source_events.extend(events)
                            logger.info(f"  ✅ {url}: {len(events)} events")
                            
                            # Update statistics
                            self.fetch_stats['by_source'][source_key] = \
                                self.fetch_stats['by_source'].get(source_key, 0) + len(events)
                            
                            # Small delay to be respectful to server
                            await asyncio.sleep(0.5)
                            
                        except Exception as e:
                            error_msg = f"Error fetching {url}: {str(e)}"
                            logger.warning(f"  ⚠️ {error_msg}")
                            self.fetch_stats['errors'].append({
                                'url': url,
                                'error': str(e),
                                'timestamp': datetime.now().isoformat()
                            })
                    
                    # Deduplicate within source
                    source_events = self._deduplicate_events(source_events)
                    all_events.extend(source_events)
                    
                    logger.info(f"  📊 {source_key}: {len(source_events)} unique events")
        
        except Exception as e:
            logger.error(f"❌ Error in fetch process: {e}")
        
        # Final deduplication across all sources
        all_events = self._deduplicate_events(all_events)
        
        # Update statistics
        self.fetch_stats['total_fetched'] = len(all_events)
        self.fetch_stats['last_update'] = datetime.now().isoformat()
        
        # Count by category
        for event in all_events:
            category = event.get('category', 'Unknown')
            self.fetch_stats['by_category'][category] = \
                self.fetch_stats['by_category'].get(category, 0) + 1
        
        # Save statistics
        self._save_fetch_statistics()
        
        logger.info(f"✨ Total unique events fetched: {len(all_events)}")
        
        return all_events
    
    async def _fetch_from_url(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fetch and parse events from a single URL"""
        try:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Status {response.status} for {url}")
                    return []
                
                html = await response.text()
                events = self._parse_html_with_context(html, url, config)
                
                # Enhance events with config metadata
                for event in events:
                    if 'category' not in event and 'category' in config:
                        event['category'] = config['category']
                    if 'festival' in config:
                        event['festival'] = config['festival']
                    if 'venue' in config and not event.get('venue'):
                        event['venue'] = config['venue']
                    event['source_url'] = url
                    event['fetched_at'] = datetime.now().isoformat()
                
                return events
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url}")
            return []
        except Exception as e:
            logger.warning(f"Error fetching {url}: {e}")
            return []
    
    def _parse_html_with_context(
        self, 
        html: str, 
        url: str, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse HTML with domain-specific context"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Use domain-specific selectors based on URL
            if 'salon' in url.lower():
                return self._parse_salon_events(soup)
            elif 'muzik.iksv' in url or 'caz.iksv' in url:
                return self._parse_music_events(soup)
            elif 'film.iksv' in url:
                return self._parse_film_events(soup)
            elif 'tiyatro.iksv' in url:
                return self._parse_theatre_events(soup)
            elif 'bienal.iksv' in url:
                return self._parse_biennial_events(soup)
            else:
                return self._parse_general_events(soup)
                
        except Exception as e:
            logger.error(f"Error parsing HTML from {url}: {e}")
            return []
    
    def _parse_salon_events(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse Salon İKSV specific events"""
        events = []
        # Based on actual HTML structure from saloniksv.com
        event_containers = soup.select('.event-item, .concert-item, .performance-item, article')
        
        for container in event_containers[:30]:
            try:
                event = self._extract_enhanced_event(container)
                if event:
                    event['venue'] = 'Salon İKSV'
                    event['category'] = event.get('category', 'Music')
                    # Salon events typically include music performances
                    if 'Salon İKSV' in event.get('title', ''):
                        events.append(event)
            except Exception as e:
                logger.debug(f"Error parsing Salon event: {e}")
        
        return events
    
    def _parse_music_events(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse music festival specific events"""
        events = []
        # Music festival specific selectors
        event_containers = soup.select('.concert, .performance, .event-card, .program-item')
        
        for container in event_containers[:50]:
            try:
                event = self._extract_enhanced_event(container)
                if event:
                    event['category'] = 'Music'
                    events.append(event)
            except Exception as e:
                logger.debug(f"Error parsing music event: {e}")
        
        return events
    
    def _parse_film_events(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse film festival specific events"""
        events = []
        event_containers = soup.select('.film, .screening, .movie-card, .event-item')
        
        for container in event_containers[:50]:
            try:
                event = self._extract_enhanced_event(container)
                if event:
                    event['category'] = 'Film'
                    events.append(event)
            except Exception as e:
                logger.debug(f"Error parsing film event: {e}")
        
        return events
    
    def _parse_theatre_events(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse theatre festival specific events"""
        events = []
        # Based on actual HTML structure from tiyatro.iksv.org
        # Look for event cards in the program section
        event_containers = soup.select('.event-card, article[class*="event"], .programme-item')
        
        # Also look for direct links to event pages
        event_links = soup.select('a[href*="/the-29th-istanbul-theater-festival"]')
        
        for container in event_containers[:50]:
            try:
                event = self._extract_enhanced_event(container)
                if event:
                    event['category'] = 'Theatre'
                    events.append(event)
            except Exception as e:
                logger.debug(f"Error parsing theatre event: {e}")
        
        # Process event links separately
        for link in event_links[:50]:
            try:
                title_elem = link
                title = link.get_text(strip=True)
                if title and len(title) > 5:
                    # Extract date and venue from adjacent elements
                    parent = link.find_parent()
                    date_elem = parent.select_one('.date, time') if parent else None
                    venue_elem = parent.select_one('.venue, .location') if parent else None
                    
                    event = {
                        'title': title,
                        'category': 'Theatre',
                        'source': 'İKSV',
                        'festival': 'Istanbul Theatre Festival',
                        'event_url': link.get('href', ''),
                    }
                    
                    if date_elem:
                        event['date_str'] = date_elem.get_text(strip=True)
                    if venue_elem:
                        event['venue'] = venue_elem.get_text(strip=True)
                    
                    events.append(event)
            except Exception as e:
                logger.debug(f"Error parsing theatre link: {e}")
        
        return events
    
    def _parse_biennial_events(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse biennial specific events"""
        events = []
        event_containers = soup.select('.exhibition, .artwork, .venue-item, .event-card')
        
        for container in event_containers[:50]:
            try:
                event = self._extract_enhanced_event(container)
                if event:
                    event['category'] = 'Art'
                    events.append(event)
            except Exception as e:
                logger.debug(f"Error parsing biennial event: {e}")
        
        return events
    
    def _parse_general_events(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse general event pages - main İKSV calendar"""
        events = []
        
        # Based on the actual HTML structure from iksv.org event calendar
        # Look for event categories: Theatre, Salon İKSV, etc.
        event_sections = soup.select('div[class*="event"], article')
        
        # Process event containers
        for container in event_sections[:40]:
            try:
                event = self._extract_enhanced_event(container)
                if event:
                    # Try to extract category from context
                    category_text = container.get_text().lower()
                    if 'theatre' in category_text or 'tiyatro' in category_text:
                        event['category'] = 'Theatre'
                    elif 'salon' in category_text:
                        event['category'] = 'Music'
                        event['venue'] = 'Salon İKSV'
                    elif 'biennial' in category_text or 'bienal' in category_text:
                        event['category'] = 'Art'
                    
                    events.append(event)
            except Exception as e:
                logger.debug(f"Error parsing general event: {e}")
        
        # Also parse the structured event listings from the main calendar
        calendar_events = self._parse_calendar_section(soup)
        events.extend(calendar_events)
        
        return events
    
    def _parse_calendar_section(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Parse the event calendar section from main İKSV page"""
        events = []
        
        # Look for the calendar grid or event list
        # Based on the fetched HTML, events are shown with category labels, dates, venues
        try:
            # Find all event entries with dates
            date_patterns = [
                r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\w+)\s+(\d{2}:\d{2})',
                r'(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+(\w+)\s+(\d{2}:\d{2})'
            ]
            
            # Extract text content and search for event patterns
            page_text = soup.get_text()
            
            # This is a simplified approach - in practice, you'd want more structured parsing
            # For now, we'll focus on the structured event containers
            
        except Exception as e:
            logger.debug(f"Error parsing calendar section: {e}")
        
        return events
    
    def _extract_enhanced_event(self, container) -> Optional[Dict[str, Any]]:
        """Extract event with enhanced data fields"""
        try:
            # Extract basic fields
            title = self._extract_title(container)
            if not title:
                return None
            
            event = {
                'title': title,
                'description': self._extract_description(container),
                'date_str': self._extract_date(container),
                'time': self._extract_time(container),
                'venue': self._extract_venue(container),
                'source': 'İKSV',
            }
            
            # Extract enhanced fields
            artist = self._extract_artist(container)
            if artist:
                event['artist'] = artist
            
            ticket_url = self._extract_ticket_url(container)
            if ticket_url:
                event['ticket_url'] = ticket_url
            
            price = self._extract_price(container)
            if price:
                event['price_range'] = price
            
            image_url = self._extract_image_url(container)
            if image_url:
                event['image_url'] = image_url
            
            return event
            
        except Exception as e:
            logger.debug(f"Error extracting enhanced event: {e}")
            return None
    
    def _extract_title(self, container) -> Optional[str]:
        """Extract event title"""
        selectors = ['h1', 'h2', 'h3', '.title', '.event-title', '.name', 'a']
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and 5 < len(title) < 200:
                    return title
        return None
    
    def _extract_description(self, container) -> str:
        """Extract event description"""
        selectors = ['.description', '.event-description', '.details', 'p']
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                desc = element.get_text(strip=True)
                if desc and len(desc) > 20:
                    return desc[:500]  # Limit to 500 chars
        return ""
    
    def _extract_date(self, container) -> str:
        """Extract event date"""
        # Try various selectors
        selectors = ['.date', '.event-date', '.when', 'time', '.day']
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                date_str = element.get_text(strip=True)
                if date_str:
                    return date_str
        
        # Try to find date patterns in the text
        text = container.get_text()
        date_patterns = [
            r'(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\w+\s+(\d{2}:\d{2})',
            r'(\d{1,2})\s+(Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\w+\s+(\d{2}:\d{2})',
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        
        return "N/A"
    
    def _extract_time(self, container) -> str:
        """Extract event time"""
        # Try various selectors
        selectors = ['.time', '.event-time', '.hour']
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        # Try to find time patterns in text (e.g., "17.00", "19:30")
        text = container.get_text()
        time_patterns = [
            r'\b(\d{2}[:.]\d{2})\b',
            r'\b(\d{1,2}\s*(?:AM|PM|am|pm))\b',
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return ""
    
    def _extract_venue(self, container) -> str:
        """Extract venue with enhanced mapping"""
        selectors = ['.venue', '.location', '.place', '.where']
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                venue_text = element.get_text(strip=True)
                # Try to map to known venues
                venue_lower = venue_text.lower()
                for venue_key, venue_data in IKSV_VENUES.items():
                    if venue_key.replace('_', ' ') in venue_lower:
                        return venue_data['name']['en']
                return venue_text
        return "İKSV Venue"
    
    def _extract_artist(self, container) -> Optional[str]:
        """Extract artist/performer information"""
        selectors = ['.artist', '.performer', '.creator', '.by', '.author']
        for selector in selectors:
            element = container.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return None
    
    def _extract_ticket_url(self, container) -> Optional[str]:
        """Extract ticket purchase URL"""
        # Look for links with ticket-related text or passo.com.tr
        links = container.find_all('a', href=True)
        ticket_keywords = ['ticket', 'buy', 'purchase', 'bilet', 'passo']
        
        for link in links:
            href = link['href']
            link_text = link.get_text().lower()
            
            # Check if it's a ticket link
            if any(keyword in link_text for keyword in ticket_keywords) or 'passo.com.tr' in href:
                if href.startswith('http'):
                    return href
                elif href.startswith('/'):
                    # Try to determine the base URL
                    if 'tiyatro.iksv' in str(container):
                        return 'https://tiyatro.iksv.org' + href
                    elif 'salon' in str(container):
                        return 'https://saloniksv.com' + href
                    else:
                        return 'https://www.iksv.org' + href
        
        # Also check for ticket URLs in the parent context
        parent = container.find_parent()
        if parent:
            passo_links = parent.find_all('a', href=re.compile(r'passo\.com\.tr'))
            if passo_links:
                return passo_links[0]['href']
        
        return None
    
    def _extract_price(self, container) -> Optional[str]:
        """Extract price information"""
        text = container.get_text()
        # Look for price patterns (TL, ₺, EUR, $)
        price_patterns = [
            r'(\d+(?:[.,]\d+)?)\s*(?:TL|₺)',
            r'(\d+)\s*-\s*(\d+)\s*(?:TL|₺)',
            r'(?:Free|Ücretsiz)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def _extract_image_url(self, container) -> Optional[str]:
        """Extract event image URL"""
        img = container.find('img')
        if img and img.get('src'):
            src = img['src']
            if src.startswith('http'):
                return src
            elif src.startswith('/'):
                return 'https://www.iksv.org' + src
        return None
    
    def _deduplicate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate events based on title similarity"""
        unique_events = []
        seen_titles = set()
        
        for event in events:
            title_lower = event['title'].lower().strip()
            # Simple deduplication - can be enhanced with fuzzy matching
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_events.append(event)
        
        return unique_events
    
    def _save_fetch_statistics(self):
        """Save fetch statistics to file"""
        try:
            with open(self.fetch_stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.fetch_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"📊 Statistics saved to {self.fetch_stats_file}")
        except Exception as e:
            logger.error(f"Error saving statistics: {e}")
    
    async def save_enhanced_events(self, events: List[Dict[str, Any]]):
        """Save enhanced events to JSON file"""
        try:
            data = {
                'events': events,
                'metadata': {
                    'total_events': len(events),
                    'last_updated': datetime.now().isoformat(),
                    'sources': list(IKSV_URL_CONFIG.keys()),
                    'statistics': self.fetch_stats
                }
            }
            
            with open(self.enhanced_events_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved {len(events)} enhanced events to {self.enhanced_events_file}")
            
        except Exception as e:
            logger.error(f"Error saving enhanced events: {e}")
    
    async def run_enhanced_fetch(self):
        """Run the enhanced event fetch process"""
        logger.info("🚀 Starting enhanced İKSV event fetch...")
        
        events = await self.fetch_all_iksv_events()
        
        if events:
            await self.save_enhanced_events(events)
            logger.info(f"✅ Enhanced fetch complete: {len(events)} events")
            
            # Print summary
            print("\n" + "="*60)
            print("📊 ENHANCED FETCH SUMMARY")
            print("="*60)
            print(f"Total Events: {len(events)}")
            print(f"\nBy Source:")
            for source, count in self.fetch_stats['by_source'].items():
                print(f"  {source}: {count}")
            print(f"\nBy Category:")
            for category, count in self.fetch_stats['by_category'].items():
                print(f"  {category}: {count}")
            if self.fetch_stats['errors']:
                print(f"\n⚠️ Errors: {len(self.fetch_stats['errors'])}")
            print("="*60 + "\n")
        else:
            logger.warning("⚠️ No events fetched")


async def main():
    """Main function to run enhanced scheduler"""
    scheduler = EnhancedEventsScheduler()
    await scheduler.run_enhanced_fetch()


if __name__ == "__main__":
    asyncio.run(main())

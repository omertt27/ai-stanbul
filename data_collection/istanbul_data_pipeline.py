"""
Istanbul Tourism Data Collection Pipeline
Comprehensive data gathering system for training domain-specific LLM
"""

import asyncio
import aiohttp
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from urllib.parse import urljoin, urlparse
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    base_url: str
    endpoints: List[str]
    rate_limit: float  # seconds between requests
    headers: Dict[str, str]
    enabled: bool = True

@dataclass
class CollectedData:
    """Structure for collected data points"""
    source: str
    category: str
    title: str
    content: str
    url: str
    metadata: Dict[str, Any]
    collected_at: str
    language: str = "en"

class IstanbulDataPipeline:
    """Main data collection pipeline for Istanbul tourism data"""
    
    def __init__(self, output_dir: str = "collected_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data sources
        self.data_sources = self._initialize_data_sources()
        
        # Statistics tracking
        self.stats = {
            'total_collected': 0,
            'by_source': {},
            'by_category': {},
            'errors': []
        }
    
    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize all data sources configuration"""
        return {
            'istanbul_tourism_guides': DataSource(
                name="Istanbul Tourism Guides",
                base_url="https://www.istanbul.com",
                endpoints=[
                    "/attractions/",
                    "/restaurants/",
                    "/hotels/",
                    "/shopping/",
                    "/nightlife/",
                    "/transportation/"
                ],
                rate_limit=1.0,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                }
            ),
            
            'cultural_heritage': DataSource(
                name="Cultural Heritage Data",
                base_url="https://www.istanbul.gov.tr",
                endpoints=[
                    "/en/culture/museums",
                    "/en/culture/historical-sites",
                    "/en/culture/heritage"
                ],
                rate_limit=2.0,
                headers={
                    'User-Agent': 'Mozilla/5.0 Research Bot'
                }
            ),
            
            'transportation_official': DataSource(
                name="Official Transportation Data",
                base_url="https://www.iett.istanbul",
                endpoints=[
                    "/en/main/pages/metro-map/93",
                    "/en/main/pages/bus-routes/94",
                    "/en/main/pages/ferry-schedules/95"
                ],
                rate_limit=3.0,
                headers={
                    'User-Agent': 'Mozilla/5.0 Data Collection Bot'
                }
            )
        }
    
    async def collect_all_data(self):
        """Main method to collect data from all sources"""
        logger.info("🚀 Starting Istanbul tourism data collection...")
        
        # Create output directories
        for source_name in self.data_sources.keys():
            (self.output_dir / source_name).mkdir(exist_ok=True)
        
        # Collect from each source
        tasks = []
        for source_name, source_config in self.data_sources.items():
            if source_config.enabled:
                tasks.append(self._collect_from_source(source_name, source_config))
        
        # Run collection tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Generate summary report
        self._generate_collection_report()
        
        logger.info(f"✅ Data collection completed. Total items: {self.stats['total_collected']}")
    
    async def _collect_from_source(self, source_name: str, source_config: DataSource):
        """Collect data from a specific source"""
        logger.info(f"📡 Collecting from {source_config.name}...")
        
        self.stats['by_source'][source_name] = 0
        
        async with aiohttp.ClientSession(
            headers=source_config.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            
            for endpoint in source_config.endpoints:
                try:
                    await self._collect_from_endpoint(
                        session, source_name, source_config, endpoint
                    )
                    
                    # Rate limiting
                    await asyncio.sleep(source_config.rate_limit)
                    
                except Exception as e:
                    error_msg = f"Error collecting from {source_name}{endpoint}: {str(e)}"
                    logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
    
    async def _collect_from_endpoint(self, session: aiohttp.ClientSession, 
                                   source_name: str, source_config: DataSource, 
                                   endpoint: str):
        """Collect data from a specific endpoint"""
        url = urljoin(source_config.base_url, endpoint)
        
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    html_content = await response.text()
                    data_points = await self._extract_data_from_html(
                        html_content, url, source_name
                    )
                    
                    # Save collected data
                    await self._save_data_points(source_name, data_points)
                    
                    self.stats['by_source'][source_name] += len(data_points)
                    self.stats['total_collected'] += len(data_points)
                    
                    logger.info(f"✅ Collected {len(data_points)} items from {url}")
                else:
                    logger.warning(f"⚠️ HTTP {response.status} for {url}")
        
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout collecting from {url}")
        except Exception as e:
            logger.error(f"❌ Error collecting from {url}: {str(e)}")
    
    async def _extract_data_from_html(self, html_content: str, url: str, 
                                    source_name: str) -> List[CollectedData]:
        """Extract structured data from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        data_points = []
        
        # Remove unwanted elements
        for unwanted in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            unwanted.decompose()
        
        # Extract main content based on source type
        if source_name == 'istanbul_tourism_guides':
            data_points.extend(self._extract_tourism_guide_data(soup, url))
        elif source_name == 'cultural_heritage':
            data_points.extend(self._extract_cultural_data(soup, url))
        elif source_name == 'transportation_official':
            data_points.extend(self._extract_transport_data(soup, url))
        
        return data_points
    
    def _extract_tourism_guide_data(self, soup: BeautifulSoup, url: str) -> List[CollectedData]:
        """Extract data from tourism guide pages"""
        data_points = []
        
        # Extract attraction/restaurant/venue information
        articles = soup.find_all(['article', 'div'], class_=re.compile(r'(attraction|restaurant|venue|place)'))
        
        for article in articles:
            title_elem = article.find(['h1', 'h2', 'h3'])
            if not title_elem:
                continue
            
            title = title_elem.get_text(strip=True)
            
            # Extract description
            desc_elem = article.find(['p', 'div'], class_=re.compile(r'(description|summary|content)'))
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            # Extract metadata
            metadata = {
                'address': self._extract_address(article),
                'price_range': self._extract_price_info(article),
                'rating': self._extract_rating(article),
                'category': self._determine_category(url, title, description)
            }
            
            if title and description:
                data_points.append(CollectedData(
                    source="istanbul_tourism_guides",
                    category=metadata['category'],
                    title=title,
                    content=description,
                    url=url,
                    metadata=metadata,
                    collected_at=datetime.now().isoformat()
                ))
        
        return data_points
    
    def _extract_cultural_data(self, soup: BeautifulSoup, url: str) -> List[CollectedData]:
        """Extract cultural heritage data"""
        data_points = []
        
        # Look for museum/historical site information
        content_sections = soup.find_all(['section', 'div'], 
                                       class_=re.compile(r'(museum|heritage|historical|cultural)'))
        
        for section in content_sections:
            title_elem = section.find(['h1', 'h2', 'h3'])
            if not title_elem:
                continue
            
            title = title_elem.get_text(strip=True)
            
            # Extract all text content
            paragraphs = section.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            metadata = {
                'opening_hours': self._extract_opening_hours(section),
                'admission_fee': self._extract_admission_info(section),
                'historical_period': self._extract_historical_period(content),
                'category': 'cultural_heritage'
            }
            
            if title and content:
                data_points.append(CollectedData(
                    source="cultural_heritage",
                    category="cultural_heritage",
                    title=title,
                    content=content,
                    url=url,
                    metadata=metadata,
                    collected_at=datetime.now().isoformat()
                ))
        
        return data_points
    
    def _extract_transport_data(self, soup: BeautifulSoup, url: str) -> List[CollectedData]:
        """Extract transportation data"""
        data_points = []
        
        # Extract route and schedule information
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    route_info = [cell.get_text(strip=True) for cell in cells]
                    
                    title = f"Transportation Route: {route_info[0]}"
                    content = f"Route details: {' | '.join(route_info)}"
                    
                    metadata = {
                        'transport_type': self._determine_transport_type(url),
                        'route_number': route_info[0] if route_info else None,
                        'category': 'transportation'
                    }
                    
                    data_points.append(CollectedData(
                        source="transportation_official",
                        category="transportation",
                        title=title,
                        content=content,
                        url=url,
                        metadata=metadata,
                        collected_at=datetime.now().isoformat()
                    ))
        
        return data_points
    
    def _extract_address(self, element) -> Optional[str]:
        """Extract address information"""
        address_elem = element.find(['div', 'span'], class_=re.compile(r'address'))
        return address_elem.get_text(strip=True) if address_elem else None
    
    def _extract_price_info(self, element) -> Optional[str]:
        """Extract price information"""
        price_elem = element.find(['div', 'span'], class_=re.compile(r'(price|cost)'))
        return price_elem.get_text(strip=True) if price_elem else None
    
    def _extract_rating(self, element) -> Optional[str]:
        """Extract rating information"""
        rating_elem = element.find(['div', 'span'], class_=re.compile(r'rating'))
        return rating_elem.get_text(strip=True) if rating_elem else None
    
    def _extract_opening_hours(self, element) -> Optional[str]:
        """Extract opening hours"""
        hours_elem = element.find(['div', 'span'], string=re.compile(r'(opening|hours|open)'))
        return hours_elem.get_text(strip=True) if hours_elem else None
    
    def _extract_admission_info(self, element) -> Optional[str]:
        """Extract admission fee information"""
        fee_elem = element.find(['div', 'span'], string=re.compile(r'(admission|fee|ticket)'))
        return fee_elem.get_text(strip=True) if fee_elem else None
    
    def _extract_historical_period(self, content: str) -> Optional[str]:
        """Extract historical period from content"""
        periods = ['Byzantine', 'Ottoman', 'Roman', 'Ancient', 'Medieval', 'Modern']
        for period in periods:
            if period.lower() in content.lower():
                return period
        return None
    
    def _determine_category(self, url: str, title: str, content: str) -> str:
        """Determine content category"""
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        if any(word in url_lower for word in ['restaurant', 'food', 'dining']):
            return 'dining'
        elif any(word in url_lower for word in ['attraction', 'museum', 'palace']):
            return 'attractions'
        elif any(word in url_lower for word in ['hotel', 'accommodation']):
            return 'accommodation'
        elif any(word in url_lower for word in ['shopping', 'bazaar', 'market']):
            return 'shopping'
        elif any(word in url_lower for word in ['transport', 'metro', 'bus']):
            return 'transportation'
        else:
            return 'general'
    
    def _determine_transport_type(self, url: str) -> str:
        """Determine transportation type from URL"""
        url_lower = url.lower()
        if 'metro' in url_lower:
            return 'metro'
        elif 'bus' in url_lower:
            return 'bus'
        elif 'ferry' in url_lower:
            return 'ferry'
        elif 'tram' in url_lower:
            return 'tram'
        else:
            return 'general'
    
    async def _save_data_points(self, source_name: str, data_points: List[CollectedData]):
        """Save collected data points to files"""
        if not data_points:
            return
        
        # Save as JSON
        json_file = self.output_dir / source_name / f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(dp) for dp in data_points], f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy analysis
        csv_file = self.output_dir / source_name / f"{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame([asdict(dp) for dp in data_points])
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        logger.info(f"💾 Saved {len(data_points)} data points to {json_file}")
    
    def _generate_collection_report(self):
        """Generate a summary report of data collection"""
        report = {
            'collection_date': datetime.now().isoformat(),
            'total_items_collected': self.stats['total_collected'],
            'items_by_source': self.stats['by_source'],
            'items_by_category': self.stats['by_category'],
            'errors_encountered': len(self.stats['errors']),
            'error_details': self.stats['errors']
        }
        
        report_file = self.output_dir / 'collection_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Collection report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("📊 DATA COLLECTION SUMMARY")
        print("="*50)
        print(f"Total items collected: {self.stats['total_collected']}")
        print(f"Errors encountered: {len(self.stats['errors'])}")
        print("\nItems by source:")
        for source, count in self.stats['by_source'].items():
            print(f"  {source}: {count}")
        print("="*50)

# Google Places API integration for local business data
class GooglePlacesCollector:
    """Collect local business data from Google Places API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place"
        
    async def collect_istanbul_places(self, place_types: List[str] = None):
        """Collect places data from Google Places API"""
        if place_types is None:
            place_types = ['restaurant', 'tourist_attraction', 'museum', 'shopping_mall']
        
        all_places = []
        
        for place_type in place_types:
            places = await self._search_places_by_type(place_type)
            all_places.extend(places)
            
            # Rate limiting for API
            await asyncio.sleep(1)
        
        return all_places
    
    async def _search_places_by_type(self, place_type: str) -> List[Dict]:
        """Search for places of a specific type in Istanbul"""
        url = f"{self.base_url}/textsearch/json"
        
        params = {
            'query': f'{place_type} in Istanbul Turkey',
            'key': self.api_key,
            'fields': 'place_id,name,rating,formatted_address,types,opening_hours,price_level'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('results', [])
                    else:
                        logger.error(f"Google Places API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error collecting Google Places data: {e}")
            return []

# Main execution function
async def main():
    """Main execution function"""
    # Initialize the data pipeline
    pipeline = IstanbulDataPipeline("istanbul_training_data")
    
    # Start data collection
    await pipeline.collect_all_data()
    
    # If you have Google Places API key, uncomment below:
    # google_api_key = os.getenv('GOOGLE_PLACES_API_KEY')
    # if google_api_key:
    #     google_collector = GooglePlacesCollector(google_api_key)
    #     places_data = await google_collector.collect_istanbul_places()
    #     print(f"Collected {len(places_data)} places from Google Places API")

if __name__ == "__main__":
    asyncio.run(main())

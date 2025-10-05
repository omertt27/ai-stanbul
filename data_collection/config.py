# Istanbul Tourism Data Collection Configuration
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DataSourceConfig:
    """Configuration for individual data sources"""
    name: str
    enabled: bool
    base_url: Optional[str]
    api_key: Optional[str]
    rate_limit: int  # requests per minute
    output_format: str  # json, csv, xml
    priority: int  # 1=highest, 5=lowest

class DataPipelineConfig:
    """Main configuration for Istanbul tourism data pipeline"""
    
    # Output directories
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    VALIDATED_DATA_DIR = "data/validated"
    TRAINING_DATA_DIR = "data/training"
    
    # Data source configurations
    DATA_SOURCES = {
        'istanbul_tourism_guides': DataSourceConfig(
            name='istanbul_tourism_guides',
            enabled=True,
            base_url='https://www.istanbul.com',
            api_key=None,
            rate_limit=30,  # 30 requests per minute
            output_format='json',
            priority=1
        ),
        
        'cultural_heritage_data': DataSourceConfig(
            name='cultural_heritage_data',
            enabled=True,
            base_url='https://www.kultur.gov.tr',
            api_key=None,
            rate_limit=20,
            output_format='json',
            priority=1
        ),
        
        'local_business_info': DataSourceConfig(
            name='local_business_info',
            enabled=True,
            base_url='https://maps.googleapis.com/maps/api/place',
            api_key=os.getenv('GOOGLE_PLACES_API_KEY'),
            rate_limit=100,  # Google Places API limit
            output_format='json',
            priority=2
        ),
        
        'transportation_official': DataSourceConfig(
            name='transportation_official',
            enabled=True,
            base_url='https://api.ibb.gov.tr',
            api_key=None,
            rate_limit=60,
            output_format='json',
            priority=2
        ),
        
        'user_reviews_curated': DataSourceConfig(
            name='user_reviews_curated',
            enabled=True,
            base_url='https://www.tripadvisor.com',
            api_key=None,
            rate_limit=10,  # Conservative for scraping
            output_format='json',
            priority=3
        ),
        
        'local_expert_content': DataSourceConfig(
            name='local_expert_content',  
            enabled=True,
            base_url=None,  # Manual content collection
            api_key=None,
            rate_limit=0,
            output_format='json',
            priority=4
        )
    }
    
    # Istanbul specific locations for data collection
    ISTANBUL_DISTRICTS = [
        'Sultanahmet', 'Beyoğlu', 'Beşiktaş', 'Kadıköy', 'Üsküdar',
        'Fatih', 'Galata', 'Taksim', 'Ortaköy', 'Arnavutköy',
        'Balat', 'Karaköy', 'Eminönü', 'Bakırköy', 'Şişli'
    ]
    
    ATTRACTION_CATEGORIES = [
        'historical_sites', 'museums', 'mosques', 'palaces',
        'restaurants', 'cafes', 'shopping', 'nightlife',
        'parks', 'viewpoints', 'transportation_hubs',
        'cultural_centers', 'markets', 'baths'
    ]
    
    # Data quality thresholds
    MIN_TEXT_LENGTH = 50  # Minimum characters for content
    MAX_TEXT_LENGTH = 5000  # Maximum characters to avoid spam
    MIN_RATING = 3.0  # Minimum rating for venues
    LANGUAGE_CODES = ['en', 'tr']  # English and Turkish
    
    # Collection targets
    TARGET_ATTRACTIONS = 500
    TARGET_RESTAURANTS = 1000
    TARGET_REVIEWS_PER_PLACE = 20
    TARGET_TRANSPORTATION_ROUTES = 200
    
    # File naming convention
    @staticmethod
    def get_output_filename(source_name: str, date_str: str, file_type: str = 'json') -> str:
        return f"{source_name}_{date_str}.{file_type}"
    
    # Validation rules
    REQUIRED_FIELDS = {
        'attractions': ['name', 'location', 'description', 'category'],
        'restaurants': ['name', 'location', 'cuisine_type', 'rating'],
        'transportation': ['route_name', 'type', 'schedule'],
        'reviews': ['text', 'rating', 'place_name'],
        'cultural_info': ['title', 'content', 'category']
    }

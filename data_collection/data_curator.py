"""
Istanbul Tourism Data Curation and Quality Control System
Week 2 Implementation - Production-grade data cleaning and preparation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from datetime import datetime
import re
import hashlib
from collections import Counter, defaultdict
import asyncio
from dataclasses import dataclass
import langdetect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

from config import DataPipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_curation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Data quality metrics tracking"""
    total_records: int = 0
    duplicate_records: int = 0
    invalid_language: int = 0
    insufficient_content: int = 0
    irrelevant_content: int = 0
    low_quality_content: int = 0
    valid_records: int = 0
    
    def quality_score(self) -> float:
        """Calculate overall quality score (0-1)"""
        if self.total_records == 0:
            return 0.0
        return self.valid_records / self.total_records

class DataCurator:
    """Advanced data curation and quality control system"""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.metrics = QualityMetrics()
        self.seen_hashes = set()
        self.quality_patterns = self._init_quality_patterns()
        self.relevance_keywords = self._init_relevance_keywords()
        
    def _init_quality_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for quality assessment"""
        return {
            'spam_patterns': [
                r'click here',
                r'buy now',
                r'limited time offer',
                r'www\.',
                r'http[s]?://',
                r'@\w+',  # social media handles
                r'#\w+',  # hashtags without context
            ],
            'low_quality_patterns': [
                r'^.{1,20}$',  # Too short
                r'(.)\1{5,}',  # Repeated characters
                r'[A-Z]{10,}',  # Too many caps
                r'!!!{3,}',    # Too many exclamation marks
                r'\?\?\?{3,}', # Too many question marks
            ],
            'turkish_patterns': [
                r'[çğıöşüÇĞIÖŞÜ]',  # Turkish characters
                r'\b(ve|ile|için|olan|olan|bu|şu|o)\b',  # Common Turkish words
            ],
            'location_patterns': [
                r'\b(Istanbul|İstanbul|Sultanahmet|Beyoğlu|Taksim|Galata|Kadıköy)\b',
                r'\b(Turkey|Türkiye|Turkish)\b',
            ]
        }
    
    def _init_relevance_keywords(self) -> Dict[str, List[str]]:
        """Initialize keyword sets for relevance checking"""
        return {
            'tourism': [
                'visit', 'attraction', 'tourist', 'sightseeing', 'landmark',
                'museum', 'mosque', 'palace', 'tower', 'bridge', 'bazaar',
                'tour', 'guide', 'ticket', 'entrance', 'hours', 'schedule'
            ],
            'food_dining': [
                'restaurant', 'cafe', 'food', 'cuisine', 'meal', 'breakfast',
                'lunch', 'dinner', 'menu', 'dish', 'taste', 'flavor',
                'turkish', 'kebab', 'baklava', 'tea', 'coffee'
            ],
            'transportation': [
                'metro', 'bus', 'ferry', 'tram', 'transport', 'station',
                'route', 'schedule', 'fare', 'ticket', 'travel', 'journey'
            ],
            'culture_history': [
                'history', 'culture', 'heritage', 'traditional', 'ottoman',
                'byzantine', 'empire', 'sultan', 'architecture', 'ancient',
                'historic', 'cultural', 'art', 'artifact'
            ],
            'accommodation': [
                'hotel', 'hostel', 'accommodation', 'stay', 'room', 'booking',
                'reservation', 'guest', 'service', 'amenity'
            ]
        }
    
    def detect_language(self, text: str) -> Optional[str]:
        """Detect text language with error handling"""
        try:
            # Clean text for language detection
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) < 10:
                return None
                
            lang = langdetect.detect(clean_text)
            return lang
        except:
            return None
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent processing"""
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        text = text.strip()
        
        return text
    
    def calculate_content_hash(self, content: str, title: str = "") -> str:
        """Calculate unique hash for content deduplication"""
        # Normalize content for hashing
        normalized = self.normalize_text(f"{title} {content}").lower()
        # Remove common words for better deduplication
        normalized = re.sub(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', '', normalized)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def assess_content_quality(self, text: str) -> Tuple[bool, List[str]]:
        """Assess content quality and return pass/fail with reasons"""
        issues = []
        
        # Length check
        if len(text) < self.config.MIN_TEXT_LENGTH:
            issues.append(f"Too short: {len(text)} chars")
        
        if len(text) > self.config.MAX_TEXT_LENGTH:
            issues.append(f"Too long: {len(text)} chars")
        
        # Spam pattern check
        for pattern in self.quality_patterns['spam_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"Spam pattern detected: {pattern}")
        
        # Low quality pattern check
        for pattern in self.quality_patterns['low_quality_patterns']:
            if re.search(pattern, text):
                issues.append(f"Low quality pattern: {pattern}")
        
        # Content diversity (character diversity)
        unique_chars = len(set(text.lower()))
        if unique_chars < 10:
            issues.append(f"Low character diversity: {unique_chars}")
        
        # Word count and average word length
        words = text.split()
        if len(words) < 10:
            issues.append(f"Too few words: {len(words)}")
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        if avg_word_length < 3:
            issues.append(f"Average word length too short: {avg_word_length:.1f}")
        
        return len(issues) == 0, issues
    
    def check_relevance(self, text: str, title: str = "") -> Tuple[float, Dict[str, int]]:
        """Check content relevance to Istanbul tourism"""
        combined_text = f"{title} {text}".lower()
        
        category_scores = {}
        total_matches = 0
        
        for category, keywords in self.relevance_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in combined_text)
            category_scores[category] = matches
            total_matches += matches
        
        # Location relevance bonus
        location_matches = sum(1 for pattern in self.quality_patterns['location_patterns']
                             if re.search(pattern, combined_text, re.IGNORECASE))
        
        # Calculate relevance score (0-1)
        relevance_score = min(1.0, (total_matches + location_matches * 2) / 20)
        
        return relevance_score, category_scores
    
    def find_duplicates(self, data: List[Dict]) -> Set[str]:
        """Find duplicate records using content hashing"""
        hash_to_id = {}
        duplicates = set()
        
        for item in data:
            content = (item.get('content', '') or 
                      item.get('text', '') or 
                      item.get('description', '') or
                      item.get('review_text', '') or
                      '')
            title = item.get('title', '') or item.get('name', '') or ''
            
            content_hash = self.calculate_content_hash(content, title)
            
            if content_hash in hash_to_id:
                duplicates.add(item.get('id', str(hash(str(item)))))
            else:
                hash_to_id[content_hash] = item.get('id', str(hash(str(item))))
        
        return duplicates
    
    def curate_single_record(self, record: Dict) -> Tuple[Optional[Dict], List[str]]:
        """Curate a single data record"""
        issues = []
        
        # Extract main content - try multiple field names
        content = (record.get('content', '') or 
                  record.get('text', '') or 
                  record.get('description', '') or
                  record.get('review_text', '') or
                  '')
        title = record.get('title', '') or record.get('name', '') or ''
        
        if not content:
            issues.append("No content found")
            return None, issues
        
        # Normalize content
        content = self.normalize_text(content)
        title = self.normalize_text(title) if title else ""
        
        # Language detection
        detected_lang = self.detect_language(content)
        if detected_lang not in self.config.LANGUAGE_CODES:
            issues.append(f"Invalid language: {detected_lang}")
            self.metrics.invalid_language += 1
        
        # Quality assessment
        quality_pass, quality_issues = self.assess_content_quality(content)
        if not quality_pass:
            issues.extend(quality_issues)
            self.metrics.low_quality_content += 1
        
        # Relevance check
        relevance_score, category_scores = self.check_relevance(content, title)
        if relevance_score < 0.2:  # Minimum relevance threshold
            issues.append(f"Low relevance score: {relevance_score:.2f}")
            self.metrics.irrelevant_content += 1
        
        # If record has issues, return None
        if issues:
            return None, issues
        
        # Create curated record
        curated_record = {
            'id': record.get('id', self.calculate_content_hash(content, title)),
            'title': title,
            'content': content,
            'category': record.get('category', self._infer_category(category_scores)),
            'location': record.get('location', ''),
            'source': record.get('source', 'unknown'),
            'language': detected_lang or 'en',
            'quality_score': 1.0 - (len(quality_issues) * 0.1),  # Quality score
            'relevance_score': relevance_score,
            'category_scores': category_scores,
            'original_data': record,  # Keep original for reference
            'curated_at': datetime.now().isoformat()
        }
        
        # Add optional fields if present
        for field in ['rating', 'coordinates', 'price_level', 'schedule', 'contact']:
            if field in record:
                curated_record[field] = record[field]
        
        self.metrics.valid_records += 1
        return curated_record, []
    
    def _infer_category(self, category_scores: Dict[str, int]) -> str:
        """Infer primary category from keyword matches"""
        if not category_scores:
            return 'general'
        
        # Return category with highest score
        return max(category_scores.items(), key=lambda x: x[1])[0]
    
    async def curate_dataset(self, input_files: List[str]) -> Dict[str, Any]:
        """Main curation pipeline for multiple input files"""
        logger.info("Starting data curation pipeline...")
        
        all_data = []
        file_stats = {}
        
        # Load all data files
        for file_path in input_files:
            try:
                file_path = Path(file_path)
                logger.info(f"Processing file: {file_path}")
                
                if file_path.suffix == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            data = [data]  # Convert single record to list
                        all_data.extend(data)
                        file_stats[str(file_path)] = len(data)
                
                elif file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    data = df.to_dict('records')
                    all_data.extend(data)
                    file_stats[str(file_path)] = len(data)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                file_stats[str(file_path)] = f"Error: {str(e)}"
        
        logger.info(f"Loaded {len(all_data)} total records from {len(input_files)} files")
        self.metrics.total_records = len(all_data)
        
        # Find and remove duplicates
        logger.info("Identifying duplicates...")
        duplicates = self.find_duplicates(all_data)
        logger.info(f"Found {len(duplicates)} duplicate records")
        self.metrics.duplicate_records = len(duplicates)
        
        # Curate each record
        logger.info("Curating individual records...")
        curated_data = []
        curation_issues = defaultdict(list)
        
        for i, record in enumerate(all_data):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(all_data)} records")
            
            # Skip duplicates (but log what we're skipping)
            record_id = record.get('id', str(hash(str(record))))
            if record_id in duplicates:
                logger.debug(f"Skipping duplicate record: {record_id}")
                continue
            
            curated_record, issues = self.curate_single_record(record)
            
            if curated_record:
                curated_data.append(curated_record)
            else:
                curation_issues['failed_curation'].extend(issues)
        
        # Generate curation report
        curation_summary = {
            'curation_date': datetime.now().isoformat(),
            'input_files': file_stats,
            'metrics': {
                'total_input_records': self.metrics.total_records,
                'duplicate_records': self.metrics.duplicate_records,
                'invalid_language': self.metrics.invalid_language,
                'insufficient_content': self.metrics.insufficient_content,
                'irrelevant_content': self.metrics.irrelevant_content,
                'low_quality_content': self.metrics.low_quality_content,
                'valid_output_records': len(curated_data),
                'overall_quality_score': self.metrics.quality_score(),
                'retention_rate': len(curated_data) / self.metrics.total_records if self.metrics.total_records > 0 else 0
            },
            'category_distribution': self._analyze_categories(curated_data),
            'language_distribution': self._analyze_languages(curated_data),
            'quality_distribution': self._analyze_quality_scores(curated_data),
            'curation_issues': dict(curation_issues),
            'recommendations': self._generate_recommendations(curated_data)
        }
        
        # Save curated dataset
        output_path = Path(self.config.VALIDATED_DATA_DIR) / f"curated_istanbul_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(curated_data, f, indent=2, ensure_ascii=False)
        
        # Save curation report
        report_path = Path(self.config.VALIDATED_DATA_DIR) / f"curation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(curation_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Curation complete! Curated {len(curated_data)} records")
        logger.info(f"Quality score: {curation_summary['metrics']['overall_quality_score']:.2f}")
        logger.info(f"Retention rate: {curation_summary['metrics']['retention_rate']:.2f}")
        
        return curation_summary
    
    def _analyze_categories(self, data: List[Dict]) -> Dict[str, int]:
        """Analyze category distribution"""
        categories = [record.get('category', 'unknown') for record in data]
        return dict(Counter(categories))
    
    def _analyze_languages(self, data: List[Dict]) -> Dict[str, int]:
        """Analyze language distribution"""
        languages = [record.get('language', 'unknown') for record in data]
        return dict(Counter(languages))
    
    def _analyze_quality_scores(self, data: List[Dict]) -> Dict[str, float]:
        """Analyze quality score distribution"""
        quality_scores = [record.get('quality_score', 0) for record in data]
        if not quality_scores:
            return {}
        
        return {
            'mean': float(np.mean(quality_scores)),
            'median': float(np.median(quality_scores)),
            'min': float(np.min(quality_scores)),
            'max': float(np.max(quality_scores)),
            'std': float(np.std(quality_scores))
        }
    
    def _generate_recommendations(self, data: List[Dict]) -> List[str]:
        """Generate recommendations for data improvement"""
        recommendations = []
        
        if len(data) < 1000:
            recommendations.append("Collect more data - current dataset is small for training")
        
        category_dist = self._analyze_categories(data)
        max_category_ratio = max(category_dist.values()) / len(data) if data else 0
        if max_category_ratio > 0.6:
            recommendations.append("Dataset is imbalanced - collect more diverse content categories")
        
        quality_stats = self._analyze_quality_scores(data)
        if quality_stats.get('mean', 0) < 0.7:
            recommendations.append("Overall quality is low - improve data sources and collection methods")
        
        if self.metrics.invalid_language > self.metrics.total_records * 0.3:
            recommendations.append("High rate of non-target language content - improve language filtering")
        
        return recommendations

async def main():
    """Main curation execution"""
    config = DataPipelineConfig()
    curator = DataCurator(config)
    
    # Find all raw data files
    raw_data_dir = Path(config.RAW_DATA_DIR)
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory not found: {raw_data_dir}")
        return
    
    input_files = []
    for pattern in ['*.json', '*.csv']:
        input_files.extend(raw_data_dir.glob(pattern))
    
    if not input_files:
        logger.error("No input files found in raw data directory")
        return
    
    logger.info(f"Found {len(input_files)} input files")
    
    # Run curation pipeline
    summary = await curator.curate_dataset([str(f) for f in input_files])
    
    # Print summary report
    print("\n" + "="*60)
    print("ISTANBUL TOURISM DATA CURATION REPORT")
    print("="*60)
    print(f"Input Records: {summary['metrics']['total_input_records']}")
    print(f"Output Records: {summary['metrics']['valid_output_records']}")
    print(f"Quality Score: {summary['metrics']['overall_quality_score']:.2f}")
    print(f"Retention Rate: {summary['metrics']['retention_rate']:.2f}")
    
    print("\nCategory Distribution:")
    for category, count in summary['category_distribution'].items():
        print(f"  {category}: {count} records")
    
    print("\nLanguage Distribution:")
    for language, count in summary['language_distribution'].items():
        print(f"  {language}: {count} records")
    
    if summary['recommendations']:
        print("\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
    
    print(f"\nCurated data saved to: {config.VALIDATED_DATA_DIR}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())

"""
Data Quality Validator and Cleaner
Ensures collected data meets quality standards for LLM training
"""

import json
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import langdetect
from langdetect.lang_detect_exception import LangDetectException
import logging

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Quality metrics for collected data"""
    total_items: int
    valid_items: int
    duplicate_items: int
    low_quality_items: int
    language_distribution: Dict[str, int]
    average_content_length: float
    category_distribution: Dict[str, int]

class DataQualityValidator:
    """Validates and cleans collected tourism data"""
    
    def __init__(self, min_content_length: int = 50, max_content_length: int = 5000):
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        
        # Quality patterns
        self.noise_patterns = [
            r'cookie\s+policy',
            r'privacy\s+policy',
            r'terms\s+of\s+service',
            r'subscribe\s+to\s+newsletter',
            r'follow\s+us\s+on',
            r'social\s+media',
            r'advertisement',
            r'sponsored\s+content'
        ]
        
        # Istanbul-specific validation keywords
        self.istanbul_keywords = [
            'istanbul', 'constantinople', 'bosphorus', 'galata', 'sultanahmet',
            'hagia sophia', 'blue mosque', 'topkapi', 'grand bazaar', 'taksim',
            'beyoglu', 'kadikoy', 'uskudar', 'besiktas', 'fatih', 'eminonu',
            'turkish', 'turkey', 'ottoman', 'byzantine'
        ]
    
    def validate_dataset(self, data_dir: Path) -> QualityMetrics:
        """Validate entire dataset and return quality metrics"""
        logger.info("🔍 Starting data quality validation...")
        
        all_data = []
        
        # Load all collected data
        for json_file in data_dir.rglob("*.json"):
            if json_file.name != "collection_report.json":
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_data.extend(data)
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
        
        # Validate each item
        valid_items = []
        duplicates = 0
        low_quality = 0
        language_dist = {}
        category_dist = {}
        
        seen_content = set()
        
        for item in all_data:
            # Check for duplicates
            content_hash = hash(item.get('content', ''))
            if content_hash in seen_content:
                duplicates += 1
                continue
            seen_content.add(content_hash)
            
            # Validate quality
            if self._is_high_quality(item):
                valid_items.append(item)
                
                # Track language distribution
                lang = self._detect_language(item.get('content', ''))
                language_dist[lang] = language_dist.get(lang, 0) + 1
                
                # Track category distribution
                category = item.get('category', 'unknown')
                category_dist[category] = category_dist.get(category, 0) + 1
            else:
                low_quality += 1
        
        # Calculate metrics
        avg_length = sum(len(item.get('content', '')) for item in valid_items) / len(valid_items) if valid_items else 0
        
        metrics = QualityMetrics(
            total_items=len(all_data),
            valid_items=len(valid_items),
            duplicate_items=duplicates,
            low_quality_items=low_quality,
            language_distribution=language_dist,
            average_content_length=avg_length,
            category_distribution=category_dist
        )
        
        # Save cleaned dataset
        self._save_cleaned_dataset(data_dir, valid_items)
        
        # Generate quality report
        self._generate_quality_report(data_dir, metrics)
        
        return metrics
    
    def _is_high_quality(self, item: Dict) -> bool:
        """Check if data item meets quality standards"""
        content = item.get('content', '')
        title = item.get('title', '')
        
        # Check minimum content length
        if len(content) < self.min_content_length:
            return False
        
        # Check maximum content length (avoid very long content)
        if len(content) > self.max_content_length:
            return False
        
        # Check for noise patterns
        content_lower = content.lower()
        if any(re.search(pattern, content_lower) for pattern in self.noise_patterns):
            return False
        
        # Check Istanbul relevance
        if not self._is_istanbul_relevant(content + ' ' + title):
            return False
        
        # Check for meaningful content (not just navigation/menu items)
        if self._is_navigation_content(content):
            return False
        
        # Check language (should be primarily English or Turkish)
        lang = self._detect_language(content)
        if lang not in ['en', 'tr', 'unknown']:
            return False
        
        return True
    
    def _is_istanbul_relevant(self, text: str) -> bool:
        """Check if content is relevant to Istanbul"""
        text_lower = text.lower()
        
        # Count Istanbul-related keywords
        keyword_count = sum(1 for keyword in self.istanbul_keywords 
                          if keyword in text_lower)
        
        # Should have at least 1 Istanbul-related keyword
        return keyword_count >= 1
    
    def _is_navigation_content(self, content: str) -> bool:
        """Check if content is just navigation/menu items"""
        # Short content with lots of links/buttons
        if len(content) < 100 and content.count('|') > 3:
            return True
        
        # Content that's mostly navigation words
        nav_words = ['home', 'about', 'contact', 'menu', 'search', 'login', 'register']
        words = content.lower().split()
        nav_ratio = sum(1 for word in words if word in nav_words) / len(words) if words else 0
        
        return nav_ratio > 0.3
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text content"""
        try:
            if len(text) < 20:
                return 'unknown'
            return langdetect.detect(text)
        except LangDetectException:
            return 'unknown'
    
    def _save_cleaned_dataset(self, data_dir: Path, valid_items: List[Dict]):
        """Save cleaned and validated dataset"""
        output_file = data_dir / 'cleaned_dataset.json'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(valid_items, f, indent=2, ensure_ascii=False)
        
        # Also save as JSONL for easier processing
        jsonl_file = data_dir / 'cleaned_dataset.jsonl'
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Saved {len(valid_items)} cleaned items to {output_file}")
    
    def _generate_quality_report(self, data_dir: Path, metrics: QualityMetrics):
        """Generate quality validation report"""
        report = {
            'validation_date': pd.Timestamp.now().isoformat(),
            'total_items_processed': metrics.total_items,
            'valid_items': metrics.valid_items,
            'duplicate_items_removed': metrics.duplicate_items,
            'low_quality_items_removed': metrics.low_quality_items,
            'quality_rate': f"{(metrics.valid_items / metrics.total_items * 100):.1f}%" if metrics.total_items > 0 else "0%",
            'average_content_length': f"{metrics.average_content_length:.1f} characters",
            'language_distribution': metrics.language_distribution,
            'category_distribution': metrics.category_distribution
        }
        
        report_file = data_dir / 'quality_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*50)
        print("🔍 DATA QUALITY VALIDATION REPORT")
        print("="*50)
        print(f"Total items processed: {metrics.total_items}")
        print(f"Valid items: {metrics.valid_items}")
        print(f"Quality rate: {(metrics.valid_items / metrics.total_items * 100):.1f}%" if metrics.total_items > 0 else "0%")
        print(f"Duplicates removed: {metrics.duplicate_items}")
        print(f"Low quality removed: {metrics.low_quality_items}")
        print(f"Average content length: {metrics.average_content_length:.1f} characters")
        print("\nLanguage distribution:")
        for lang, count in metrics.language_distribution.items():
            print(f"  {lang}: {count}")
        print("\nCategory distribution:")
        for category, count in metrics.category_distribution.items():
            print(f"  {category}: {count}")
        print("="*50)

def main():
    """Run data quality validation"""
    data_dir = Path("istanbul_training_data")
    
    if not data_dir.exists():
        print("❌ Data directory not found. Please run data collection first.")
        return
    
    validator = DataQualityValidator()
    metrics = validator.validate_dataset(data_dir)
    
    print(f"✅ Data validation completed. {metrics.valid_items} high-quality items ready for training.")

if __name__ == "__main__":
    main()

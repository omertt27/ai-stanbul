"""
Query Preprocessing Pipeline
Integrates dialect normalization, typo correction, and entity extraction
"""

import logging
import time
import sys
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.turkish_dialect_normalizer import get_dialect_normalizer
from services.turkish_typo_corrector import get_typo_corrector
from services.entity_extractor import get_entity_extractor

logger = logging.getLogger(__name__)


@dataclass
class QueryProcessingResult:
    """Result of query preprocessing"""
    original_query: str
    cleaned_query: str
    entities: Dict[str, Any]
    intent: Optional[str]
    
    # Processing details
    typo_corrections: list
    dialect_normalizations: list
    has_typos: bool
    has_dialect: bool
    
    # Performance metrics
    typo_correction_ms: float
    dialect_normalization_ms: float
    entity_extraction_ms: float
    total_processing_ms: float


class QueryPreprocessingPipeline:
    """
    Unified query preprocessing pipeline
    
    Order of operations:
    1. Typo correction
    2. Dialect normalization
    3. Entity extraction
    """
    
    def __init__(self):
        self.typo_corrector = get_typo_corrector()
        self.dialect_normalizer = get_dialect_normalizer()
        self.entity_extractor = get_entity_extractor()
        
        self.stats = {
            'total_queries_processed': 0,
            'queries_with_typos': 0,
            'queries_with_dialect': 0,
            'total_typo_corrections': 0,
            'total_dialect_normalizations': 0,
            'avg_processing_time_ms': 0.0
        }
        
        logger.info("✅ QueryPreprocessingPipeline initialized")
    
    def process(self, query: str, intent: Optional[str] = None) -> QueryProcessingResult:
        """
        Process query through full pipeline
        
        Args:
            query: Raw user query
            intent: Optional intent hint for entity extraction
            
        Returns:
            QueryProcessingResult with cleaned query and extracted entities
        """
        start_time = time.time()
        
        # Step 1: Typo Correction
        typo_start = time.time()
        corrected_query, typo_corrections = self.typo_corrector.correct_typos(query, aggressive=False)
        typo_time = (time.time() - typo_start) * 1000
        
        # Step 2: Dialect Normalization
        dialect_start = time.time()
        normalized_query, dialect_normalizations = self.dialect_normalizer.normalize(corrected_query)
        dialect_time = (time.time() - dialect_start) * 1000
        
        # Step 3: Entity Extraction
        entity_start = time.time()
        entities = self.entity_extractor.extract_entities(
            normalized_query,
            intent or 'general'
        )
        entity_time = (time.time() - entity_start) * 1000
        
        # Calculate total time
        total_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self._update_stats(typo_corrections, dialect_normalizations, total_time)
        
        # Create result
        result = QueryProcessingResult(
            original_query=query,
            cleaned_query=normalized_query,
            entities=entities,
            intent=intent,
            typo_corrections=typo_corrections,
            dialect_normalizations=dialect_normalizations,
            has_typos=len(typo_corrections) > 0,
            has_dialect=len(dialect_normalizations) > 0,
            typo_correction_ms=typo_time,
            dialect_normalization_ms=dialect_time,
            entity_extraction_ms=entity_time,
            total_processing_ms=total_time
        )
        
        # Log processing details
        if result.has_typos or result.has_dialect:
            logger.info(f"📝 Processed query: '{query}' → '{normalized_query}'")
            if result.has_typos:
                logger.info(f"   🔧 Typo corrections: {typo_corrections}")
            if result.has_dialect:
                logger.info(f"   📝 Dialect normalizations: {dialect_normalizations}")
            logger.info(f"   ⏱️  Processing time: {total_time:.2f}ms")
        
        return result
    
    def process_batch(self, queries: list[Tuple[str, Optional[str]]]) -> list[QueryProcessingResult]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of (query, intent) tuples
            
        Returns:
            List of QueryProcessingResult objects
        """
        results = []
        for query, intent in queries:
            result = self.process(query, intent)
            results.append(result)
        
        logger.info(f"📊 Batch processed {len(queries)} queries")
        return results
    
    def _update_stats(self, typo_corrections: list, dialect_normalizations: list, processing_time: float):
        """Update pipeline statistics"""
        self.stats['total_queries_processed'] += 1
        
        if typo_corrections:
            self.stats['queries_with_typos'] += 1
            self.stats['total_typo_corrections'] += len(typo_corrections)
        
        if dialect_normalizations:
            self.stats['queries_with_dialect'] += 1
            self.stats['total_dialect_normalizations'] += len(dialect_normalizations)
        
        # Update rolling average processing time
        n = self.stats['total_queries_processed']
        old_avg = self.stats['avg_processing_time_ms']
        self.stats['avg_processing_time_ms'] = (old_avg * (n - 1) + processing_time) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        total = self.stats['total_queries_processed']
        
        return {
            'total_queries': total,
            'queries_with_typos': self.stats['queries_with_typos'],
            'queries_with_dialect': self.stats['queries_with_dialect'],
            'typo_percentage': (self.stats['queries_with_typos'] / total * 100) if total > 0 else 0,
            'dialect_percentage': (self.stats['queries_with_dialect'] / total * 100) if total > 0 else 0,
            'total_typo_corrections': self.stats['total_typo_corrections'],
            'total_dialect_normalizations': self.stats['total_dialect_normalizations'],
            'avg_corrections_per_query': (self.stats['total_typo_corrections'] / self.stats['queries_with_typos']) if self.stats['queries_with_typos'] > 0 else 0,
            'avg_normalizations_per_query': (self.stats['total_dialect_normalizations'] / self.stats['queries_with_dialect']) if self.stats['queries_with_dialect'] > 0 else 0,
            'avg_processing_time_ms': self.stats['avg_processing_time_ms'],
        }
    
    def reset_statistics(self):
        """Reset pipeline statistics"""
        self.stats = {
            'total_queries_processed': 0,
            'queries_with_typos': 0,
            'queries_with_dialect': 0,
            'total_typo_corrections': 0,
            'total_dialect_normalizations': 0,
            'avg_processing_time_ms': 0.0
        }
        logger.info("📊 Pipeline statistics reset")


# Singleton instance
_pipeline_instance = None

def get_preprocessing_pipeline() -> QueryPreprocessingPipeline:
    """Get or create preprocessing pipeline singleton"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = QueryPreprocessingPipeline()
    return _pipeline_instance

"""
Turkish Dialect Normalizer
Normalizes regional Turkish dialects and colloquialisms to standard Turkish
"""

import re
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class TurkishDialectNormalizer:
    """Normalize regional Turkish dialects to standard Turkish"""
    
    def __init__(self):
        self.dialect_mappings = self._load_dialect_mappings()
        self.suffix_patterns = self._load_suffix_patterns()
        logger.info("✅ TurkishDialectNormalizer initialized")
    
    def _load_dialect_mappings(self) -> Dict[str, str]:
        """Load dialect to standard Turkish mappings"""
        return {
            # Istanbul dialect - informal speech
            'bi': 'bir',  # "bi yere" -> "bir yere"
            'bişey': 'bir şey',
            'bişi': 'bir şey',
            'hiçbi': 'hiçbir',
            'hepsi': 'hepsi',
            
            # Location variations
            'şurda': 'şurada',
            'burda': 'burada',
            'orda': 'orada',
            'nerde': 'nerede',
            'nere': 'nere',
            'nereye': 'nereye',
            
            # Common Aegean dialect - future tense
            'gidicem': 'gideceğim',
            'gelcem': 'geleceğim',
            'yapıcam': 'yapacağım',
            'alıcam': 'alacağım',
            'edicem': 'edeceğim',
            'bilcem': 'bileceğim',
            
            # Question words - colloquial
            'nası': 'nasıl',
            'neden': 'neden',
            'niçin': 'niçin',
            'niye': 'neden',
            'napıyım': 'ne yapayım',
            'napayım': 'ne yapayım',
            
            # Demonstratives - colloquial
            'şu': 'şu',
            'bu': 'bu',
            'o': 'o',
            
            # Time expressions
            'bugün': 'bugün',
            'dün': 'dün',
            'yarın': 'yarın',
            'sabah': 'sabah',
            'akşam': 'akşam',
            
            # Common contractions
            'değil': 'değil',
            'yok': 'yok',
            'var': 'var',
            'tamam': 'tamam',
            
            # Tourism-specific informal
            'nereye gidiyim': 'nereye gideyim',
            'nasıl gidicem': 'nasıl gideceğim',
            'nasıl gidiyim': 'nasıl gideyim',
            'ne yapıcam': 'ne yapacağım',
            'ne yapayım': 'ne yapayım',
            
            # Verb suffixes - informal to formal
            'gidiyim': 'gideyim',
            'yapıyım': 'yapayım',
            'alıyım': 'alayım',
            
            # Common Turkish internet slang
            'naber': 'ne haber',
            'nbr': 'ne haber',
            'tmm': 'tamam',
            'tamamdır': 'tamam',
            
            # Informal eating/drinking
            'yiyek': 'yiyelim',
            'yiyelim': 'yiyelim',
            'içek': 'içelim',
            'gidek': 'gidelim',
        }
    
    def _load_suffix_patterns(self) -> List[Tuple[str, str]]:
        """Load common suffix variations and their corrections"""
        return [
            # Future tense: -cem/-cek -> -ceğim/-cek
            (r'\b(\w+)cem\b', r'\1ceğim'),
            (r'\b(\w+)cak\b', r'\1cak'),
            
            # Subjunctive: -iyim/-iyim -> -eyim/-ayım
            (r'\b(\w+)iyim\b', r'\1eyim'),
            
            # Location: -da/-de (stay the same, just clean)
            # These are correct, no change needed
        ]
    
    def normalize(self, query: str) -> Tuple[str, List[str]]:
        """
        Normalize dialectal variations in query
        
        Args:
            query: Original query with potential dialect variations
            
        Returns:
            Tuple of (normalized_query, list_of_normalizations)
        """
        normalized = query
        normalizations = []
        
        # Apply word-level dialect mappings
        for dialect, standard in self.dialect_mappings.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(dialect) + r'\b'
            if re.search(pattern, normalized, re.IGNORECASE):
                old_normalized = normalized
                normalized = re.sub(pattern, standard, normalized, flags=re.IGNORECASE)
                if old_normalized != normalized:
                    normalizations.append(f"{dialect} → {standard}")
        
        # Apply suffix pattern corrections
        for pattern, replacement in self.suffix_patterns:
            old_normalized = normalized
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
            if old_normalized != normalized:
                normalizations.append(f"suffix correction: {pattern}")
        
        if normalizations:
            logger.info(f"📝 Normalized query: '{query}' → '{normalized}'")
            logger.info(f"   Applied {len(normalizations)} normalizations")
        
        return normalized, normalizations
    
    def normalize_silent(self, query: str) -> str:
        """
        Normalize query without returning normalization details
        Useful for pipeline integration where details aren't needed
        
        Args:
            query: Original query
            
        Returns:
            Normalized query string
        """
        normalized, _ = self.normalize(query)
        return normalized
    
    def has_dialect(self, query: str) -> bool:
        """
        Check if query contains dialect variations
        
        Args:
            query: Query to check
            
        Returns:
            True if dialect patterns detected
        """
        query_lower = query.lower()
        
        # Check for known dialect words
        for dialect in self.dialect_mappings.keys():
            pattern = r'\b' + re.escape(dialect) + r'\b'
            if re.search(pattern, query_lower):
                return True
        
        # Check for suffix patterns
        for pattern, _ in self.suffix_patterns:
            if re.search(pattern, query_lower):
                return True
        
        return False
    
    def get_dialect_statistics(self, queries: List[str]) -> Dict:
        """
        Get statistics about dialect usage in a list of queries
        
        Args:
            queries: List of queries to analyze
            
        Returns:
            Dictionary with dialect statistics
        """
        total = len(queries)
        with_dialect = 0
        dialect_counts = {dialect: 0 for dialect in self.dialect_mappings.keys()}
        
        for query in queries:
            if self.has_dialect(query):
                with_dialect += 1
                
                # Count individual dialect occurrences
                query_lower = query.lower()
                for dialect in self.dialect_mappings.keys():
                    pattern = r'\b' + re.escape(dialect) + r'\b'
                    if re.search(pattern, query_lower):
                        dialect_counts[dialect] += 1
        
        # Get top 10 most common dialects
        top_dialects = sorted(
            [(dialect, count) for dialect, count in dialect_counts.items() if count > 0],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_queries': total,
            'queries_with_dialect': with_dialect,
            'dialect_percentage': (with_dialect / total * 100) if total > 0 else 0,
            'top_dialects': top_dialects,
            'unique_dialects_found': len([c for c in dialect_counts.values() if c > 0])
        }


# Singleton instance
_normalizer_instance = None

def get_dialect_normalizer() -> TurkishDialectNormalizer:
    """Get or create dialect normalizer singleton"""
    global _normalizer_instance
    if _normalizer_instance is None:
        _normalizer_instance = TurkishDialectNormalizer()
    return _normalizer_instance

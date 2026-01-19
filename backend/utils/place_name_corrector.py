"""
Place Name Corrector

Fixes common misspellings of Istanbul place names in LLM responses.
This ensures consistent, correct spelling regardless of LLM output quality.

Author: AI Istanbul Team
Date: January 2026
"""

import re
import logging

logger = logging.getLogger(__name__)


# Common misspellings and their corrections
PLACE_NAME_CORRECTIONS = {
    # Galataport misspellings
    r'\bGalatport\b': 'Galataport',
    r'\bGalataport\b': 'Galataport',  # Already correct, but ensures consistency
    r'\bGalat[ao][\s-]?[Pp]ort\b': 'Galataport',
    
    # Karaköy misspellings  
    r'\bKarakoy\b': 'Karaköy',
    r'\bKarak[öo]y\b': 'Karaköy',
    
    # Beyoğlu misspellings
    r'\bBeyoglu\b': 'Beyoğlu',
    r'\bBeyolu\b': 'Beyoğlu',
    
    # Taksim misspellings
    r'\bTaxim\b': 'Taksim',
    
    # Sultanahmet misspellings
    r'\bSultanahmed\b': 'Sultanahmet',
    r'\bSultanamet\b': 'Sultanahmet',
    
    # Eminönü misspellings
    r'\bEminonu\b': 'Eminönü',
    r'\bEmin[öo]n[üu]\b': 'Eminönü',
    
    # Kadıköy misspellings
    r'\bKadikoy\b': 'Kadıköy',
    r'\bKad[ıi]k[öo]y\b': 'Kadıköy',
    
    # Beşiktaş misspellings
    r'\bBesiktas\b': 'Beşiktaş',
    r'\bBe[sş]ikta[sş]\b': 'Beşiktaş',
    
    # Ortaköy misspellings
    r'\bOrtakoy\b': 'Ortaköy',
    r'\bOrtak[öo]y\b': 'Ortaköy',
    
    # Üsküdar misspellings
    r'\bUskudar\b': 'Üsküdar',
    r'\b[ÜU]sk[üu]dar\b': 'Üsküdar',
    
    # Çemberlitaş misspellings
    r'\bCemberlitas\b': 'Çemberlitaş',
    r'\b[ÇC]emberlita[sş]\b': 'Çemberlitaş',
    
    # Kabataş misspellings
    r'\bKabatas\b': 'Kabataş',
    r'\bKabata[sş]\b': 'Kabataş',
    
    # Dolmabahçe misspellings
    r'\bDolmabahce\b': 'Dolmabahçe',
    r'\bDolmabah[çc]e\b': 'Dolmabahçe',
    
    # Şişli misspellings
    r'\bSisli\b': 'Şişli',
    r'\b[ŞS]i[sş]li\b': 'Şişli',
}


def correct_place_names(text: str) -> str:
    """
    Correct common misspellings of Istanbul place names.
    
    Uses regex patterns to find and replace misspellings while
    preserving the surrounding text and formatting.
    
    Args:
        text: Text potentially containing place name misspellings
        
    Returns:
        Text with corrected place names
        
    Example:
        >>> correct_place_names("Visit Galatport and Karakoy")
        'Visit Galataport and Karaköy'
    """
    if not text:
        return text
    
    corrected_text = text
    corrections_made = []
    
    # Apply all corrections
    for pattern, replacement in PLACE_NAME_CORRECTIONS.items():
        # Find all matches before replacing (for logging)
        matches = re.findall(pattern, corrected_text, re.IGNORECASE)
        if matches:
            # Replace case-insensitively but preserve capitalization pattern
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
            corrections_made.extend(matches)
    
    # Log corrections if any were made
    if corrections_made:
        logger.info(f"✏️ Corrected {len(corrections_made)} place name(s): {', '.join(set(corrections_made))}")
    
    return corrected_text


def correct_place_names_in_response(response_dict: dict) -> dict:
    """
    Correct place names in a response dictionary.
    
    Handles nested structures and corrects place names in all string fields.
    
    Args:
        response_dict: Dictionary containing response data
        
    Returns:
        Dictionary with corrected place names
    """
    if not isinstance(response_dict, dict):
        return response_dict
    
    corrected = {}
    for key, value in response_dict.items():
        if isinstance(value, str):
            corrected[key] = correct_place_names(value)
        elif isinstance(value, dict):
            corrected[key] = correct_place_names_in_response(value)
        elif isinstance(value, list):
            corrected[key] = [
                correct_place_names(item) if isinstance(item, str)
                else correct_place_names_in_response(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            corrected[key] = value
    
    return corrected


# Quick test
if __name__ == "__main__":
    test_cases = [
        "Visit Galatport and Karakoy for amazing views!",
        "Take the tram from Kabatas to Eminonu",
        "Explore Beyoglu and Sultanahmed",
        "Galataport is near Karaköy",  # Already correct
    ]
    
    print("Place Name Correction Test:")
    print("=" * 60)
    for test in test_cases:
        corrected = correct_place_names(test)
        print(f"Original:  {test}")
        print(f"Corrected: {corrected}")
        print("-" * 60)

#!/usr/bin/env python3
"""
Configuration Validator for AI Istanbul
Validates all required environment variables and settings.
"""

import os
from typing import Dict, List

def validate_config() -> Dict[str, bool]:
    """Validate all configuration requirements"""
    
    results = {}
    
    # Required environment variables
    required_vars = [
        "GOOGLE_PLACES_API_KEY",
        "OPENAI_API_KEY",
        "DATABASE_URL", 
        "REDIS_URL"
    ]
    
    for var in required_vars:
        results[f"env_{var}"] = bool(os.getenv(var))
        
    # Optional but recommended
    optional_vars = [
        "SECRET_KEY",
        "CORS_ORIGINS",
        "DEBUG",
        "LOG_LEVEL"
    ]
    
    for var in optional_vars:
        results[f"optional_{var}"] = bool(os.getenv(var))
        
    return results

if __name__ == "__main__":
    results = validate_config()
    for key, value in results.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")

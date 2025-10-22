"""
Entity Extraction API Routes
Provides endpoints for extracting entities from user queries
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.entity_extractor import get_entity_extractor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/entities", tags=["Entity Extraction"])


class EntityExtractionRequest(BaseModel):
    """Request model for entity extraction"""
    query: str
    intent: Optional[str] = "general"


class EntityExtractionResponse(BaseModel):
    """Response model for entity extraction"""
    success: bool
    query: str
    intent: str
    entities: Dict[str, Any]
    extracted_count: int


@router.post("/extract", response_model=EntityExtractionResponse)
async def extract_entities(request: EntityExtractionRequest):
    """
    Extract entities from user query
    
    **Example Turkish Query:**
    ```json
    {
        "query": "Sultanahmet'te bugün akşam 2 kişilik ucuz balık restoranı",
        "intent": "restaurant"
    }
    ```
    
    **Example English Query:**
    ```json
    {
        "query": "cheap seafood restaurant in Sultanahmet for 2 people tonight",
        "intent": "restaurant"
    }
    ```
    
    **Returns:**
    - `locations`: List of Istanbul locations/districts
    - `cuisines`: List of cuisine types (for restaurant queries)
    - `price_range`: budget/mid_range/luxury
    - `dates`: Date string or relative date
    - `time`: Time of day or specific time
    - `party_size`: Number of people
    - `preferences`: User preferences (family_friendly, romantic, etc.)
    - `attraction_type`: Type of attraction (museum, mosque, etc.)
    - `transport_mode`: Transportation mode (metro, ferry, etc.)
    - `from_location`: Origin location (for route queries)
    - `to_location`: Destination location (for route queries)
    """
    try:
        logger.info(f"📊 Entity extraction request - Query: '{request.query}', Intent: {request.intent}")
        
        extractor = get_entity_extractor()
        entities = extractor.extract_entities(request.query, request.intent)
        
        extracted_count = sum(1 for v in entities.values() if v)
        
        logger.info(f"✅ Extracted {extracted_count} entity types from query")
        
        return EntityExtractionResponse(
            success=True,
            query=request.query,
            intent=request.intent,
            entities=entities,
            extracted_count=extracted_count
        )
    
    except Exception as e:
        logger.error(f"❌ Entity extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check for entity extractor"""
    try:
        extractor = get_entity_extractor()
        return {
            "status": "healthy",
            "service": "entity_extractor",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"❌ Entity extractor health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Service unhealthy: {str(e)}")


# Export router
__all__ = ['router']

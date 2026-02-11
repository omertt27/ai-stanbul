"""
RunPod LLM Integration with NCF Recommendations

This script integrates the NCF deep learning model with the RunPod-hosted LLM
to provide personalized, context-aware recommendations.

Architecture:
    User Query → NCF Recommendations → Context Builder → RunPod LLM → Enhanced Response

Author: AI Istanbul Team
Date: February 10, 2026
"""

import os
import sys
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.ml.deep_learning.models.ncf import NCF
from backend.ml.deep_learning.data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RunPodNCFIntegration:
    """
    Integration layer between NCF model and RunPod LLM.
    
    Provides personalized recommendations that enhance the LLM's responses.
    """
    
    def __init__(
        self,
        ncf_model_path: Optional[str] = None,
        runpod_host: str = "localhost",
        runpod_port: int = 8000,
        enable_ncf: bool = True
    ):
        """
        Initialize RunPod NCF integration.
        
        Args:
            ncf_model_path: Path to trained NCF model (if None, will train new)
            runpod_host: RunPod LLM host (via SSH tunnel)
            runpod_port: RunPod LLM port
            enable_ncf: Whether to enable NCF recommendations
        """
        self.runpod_host = runpod_host
        self.runpod_port = runpod_port
        self.enable_ncf = enable_ncf
        
        # Initialize NCF model
        self.ncf_model: Optional[NCF] = None
        self.pipeline: Optional[DataPipeline] = None
        
        if enable_ncf:
            self._initialize_ncf(ncf_model_path)
        
        logger.info(f"✅ RunPod NCF Integration initialized (NCF enabled: {enable_ncf})")
    
    def _initialize_ncf(self, model_path: Optional[str] = None) -> None:
        """Initialize NCF model."""
        try:
            self.pipeline = DataPipeline()
            
            if model_path and os.path.exists(model_path):
                # Load existing model
                logger.info(f"Loading NCF model from {model_path}...")
                self.ncf_model = NCF(num_users=100, num_items=100)  # Will be updated on load
                self.ncf_model.load(model_path)
                logger.info("✅ NCF model loaded successfully")
            else:
                logger.warning("No NCF model found. Will train on first use.")
        except Exception as e:
            logger.error(f"Failed to initialize NCF: {e}")
            self.enable_ncf = False
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        query: str,
        context: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get personalized recommendations using NCF.
        
        Args:
            user_id: User identifier
            query: User's query
            context: Context information (location, preferences, etc.)
            top_k: Number of recommendations to return
            
        Returns:
            List of recommendations with scores
        """
        if not self.enable_ncf or self.ncf_model is None:
            logger.warning("NCF not enabled, returning fallback recommendations")
            return self._get_fallback_recommendations(query, context, top_k)
        
        try:
            # Map user ID to index
            user_idx = self.pipeline.user_to_idx.get(user_id)
            
            if user_idx is None:
                # Cold start - new user
                logger.info(f"Cold start for user {user_id}, using fallback")
                return self._get_fallback_recommendations(query, context, top_k)
            
            # Get candidate items based on context
            candidate_items = self._extract_candidate_items(query, context)
            
            if not candidate_items:
                logger.warning("No candidate items found")
                return []
            
            # Map items to indices
            candidate_indices = []
            for item_id in candidate_items:
                if item_id in self.pipeline.item_to_idx:
                    candidate_indices.append(self.pipeline.item_to_idx[item_id])
            
            if not candidate_indices:
                logger.warning("No valid candidate indices")
                return self._get_fallback_recommendations(query, context, top_k)
            
            # Get NCF predictions
            recommendations = self.ncf_model.recommend_for_user(
                user_id=user_idx,
                candidate_items=np.array(candidate_indices),
                top_k=top_k
            )
            
            # Convert back to item IDs with metadata
            results = []
            for item_idx, score in recommendations:
                item_id = self.pipeline.idx_to_item[item_idx]
                results.append({
                    "item_id": item_id,
                    "score": float(score),
                    "source": "ncf",
                    "metadata": self._get_item_metadata(item_id)
                })
            
            logger.info(f"✅ Generated {len(results)} NCF recommendations for {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error getting NCF recommendations: {e}", exc_info=True)
            return self._get_fallback_recommendations(query, context, top_k)
    
    def _extract_candidate_items(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Extract candidate items from query and context.
        
        This is a placeholder - in production, integrate with your
        existing venue/place database.
        """
        # Example: Extract from context
        candidates = []
        
        # From context categories
        if "category" in context:
            # Get items in this category
            pass
        
        # From location
        if "location" in context:
            # Get nearby items
            pass
        
        # From query intent
        # Parse query to extract relevant categories/types
        
        # For now, return all known items (replace with actual logic)
        if self.pipeline and self.pipeline.item_to_idx:
            candidates = list(self.pipeline.item_to_idx.keys())[:100]
        
        return candidates
    
    def _get_item_metadata(self, item_id: str) -> Dict[str, Any]:
        """
        Get metadata for an item.
        
        This is a placeholder - integrate with your venue database.
        """
        return {
            "name": item_id,
            "category": "unknown",
            "location": "Istanbul"
        }
    
    def _get_fallback_recommendations(
        self,
        query: str,
        context: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fallback recommendations when NCF is not available.
        
        Uses simple popularity or rule-based recommendations.
        """
        logger.info("Using fallback recommendations")
        
        # Simple fallback based on query keywords
        fallback_items = [
            {"item_id": "galata_tower", "score": 0.9, "source": "fallback"},
            {"item_id": "hagia_sophia", "score": 0.85, "source": "fallback"},
            {"item_id": "grand_bazaar", "score": 0.8, "source": "fallback"},
            {"item_id": "blue_mosque", "score": 0.75, "source": "fallback"},
            {"item_id": "topkapi_palace", "score": 0.7, "source": "fallback"},
        ]
        
        return fallback_items[:top_k]
    
    async def build_llm_context(
        self,
        user_id: str,
        query: str,
        context: Dict[str, Any],
        recommendations: List[Dict[str, Any]]
    ) -> str:
        """
        Build enhanced context for RunPod LLM based on NCF recommendations.
        
        Args:
            user_id: User identifier
            query: User's query
            context: Context information
            recommendations: NCF recommendations
            
        Returns:
            Enhanced context string for LLM
        """
        # Build context with personalized recommendations
        context_parts = [
            f"User Query: {query}",
            "",
            "Personalized Recommendations (Neural Collaborative Filtering):",
        ]
        
        for i, rec in enumerate(recommendations[:5], 1):
            item_id = rec['item_id']
            score = rec['score']
            source = rec.get('source', 'unknown')
            context_parts.append(
                f"{i}. {item_id} (relevance: {score:.2f}, source: {source})"
            )
        
        # Add user context
        if context:
            context_parts.append("")
            context_parts.append("User Context:")
            for key, value in context.items():
                context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts)
    
    async def get_llm_response(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        Get response from RunPod LLM via SSH tunnel.
        
        Args:
            prompt: Prompt to send to LLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            LLM response
        """
        try:
            import aiohttp
            
            url = f"http://{self.runpod_host}:{self.runpod_port}/v1/completions"
            
            payload = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['text']
                    else:
                        logger.error(f"LLM request failed: {response.status}")
                        return "Sorry, I couldn't process your request."
                        
        except Exception as e:
            logger.error(f"Error calling RunPod LLM: {e}", exc_info=True)
            return "Sorry, the AI service is temporarily unavailable."
    
    async def process_query(
        self,
        user_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query with NCF + RunPod LLM.
        
        Full pipeline:
        1. Get NCF recommendations
        2. Build enhanced context
        3. Query RunPod LLM
        4. Return combined response
        
        Args:
            user_id: User identifier
            query: User's query
            context: Optional context information
            
        Returns:
            Response with recommendations and LLM text
        """
        if context is None:
            context = {}
        
        logger.info(f"Processing query for user {user_id}: {query}")
        
        # Step 1: Get NCF recommendations
        recommendations = await self.get_personalized_recommendations(
            user_id=user_id,
            query=query,
            context=context,
            top_k=10
        )
        
        # Step 2: Build enhanced context for LLM
        llm_context = await self.build_llm_context(
            user_id=user_id,
            query=query,
            context=context,
            recommendations=recommendations
        )
        
        # Step 3: Create full prompt for LLM
        full_prompt = f"""You are an AI assistant for Istanbul tourism. Use the personalized recommendations below to provide a helpful response.

{llm_context}

Based on the above recommendations and context, provide a helpful, personalized response to the user's query. Be specific and mention the recommended places.

Response:"""
        
        # Step 4: Get LLM response
        llm_response = await self.get_llm_response(full_prompt)
        
        # Step 5: Return combined result
        return {
            "query": query,
            "user_id": user_id,
            "recommendations": recommendations,
            "llm_response": llm_response,
            "context": llm_context,
            "source": "ncf+runpod"
        }


async def demo():
    """Demo the RunPod NCF integration."""
    print("\n" + "="*80)
    print("🚀 RunPod + NCF Integration Demo")
    print("="*80 + "\n")
    
    # Initialize integration
    integration = RunPodNCFIntegration(
        ncf_model_path="./test_models/ncf/best_model.pth",
        runpod_host="localhost",
        runpod_port=8000,
        enable_ncf=True  # Set to False to test without NCF
    )
    
    # Example queries
    test_queries = [
        {
            "user_id": "user_123",
            "query": "What are the best places to visit in Istanbul?",
            "context": {
                "location": "Taksim",
                "preferences": ["historical", "cultural"],
                "budget": "medium"
            }
        },
        {
            "user_id": "user_456",
            "query": "I want to try authentic Turkish food",
            "context": {
                "location": "Sultanahmet",
                "preferences": ["food", "local"],
                "budget": "low"
            }
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}")
        print(f"{'='*80}\n")
        
        # Process query
        result = await integration.process_query(
            user_id=test_case["user_id"],
            query=test_case["query"],
            context=test_case["context"]
        )
        
        # Print results
        print(f"User: {result['user_id']}")
        print(f"Query: {result['query']}\n")
        
        print("Recommendations:")
        for j, rec in enumerate(result['recommendations'][:5], 1):
            print(f"  {j}. {rec['item_id']} (score: {rec['score']:.3f}, source: {rec['source']})")
        
        print(f"\nLLM Response:")
        print(f"{result['llm_response']}\n")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║  🚀 RunPod + NCF Integration                                  ║
    ║                                                                ║
    ║  Combines Neural Collaborative Filtering with RunPod LLM      ║
    ║  for personalized, AI-enhanced recommendations                ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Run demo
    asyncio.run(demo())

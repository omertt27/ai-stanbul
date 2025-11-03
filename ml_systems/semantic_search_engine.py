"""
Semantic search engine that works on both CPU and GPU
Supports multiple collections (restaurants, attractions, tips)
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import torch
import os
import logging

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    def __init__(self, model_path="./models/semantic-search", use_gpu=None):
        logger.info("🔄 Loading semantic search model...")
        
        # Auto-detect device
        if use_gpu is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        logger.info(f"📍 Using device: {device}")
        
        self.encoder = SentenceTransformer(model_path, device=device)
        self.dimension = 768
        self.collections = {}  # Store multiple collections
        self.use_gpu = (device == "cuda")
        
        logger.info(f"✅ Semantic search model loaded on {device}")
    
    async def initialize(self):
        """Async initialization - loads all available collections"""
        logger.info("🔄 Initializing semantic search collections...")
        
        # Try to load restaurants
        try:
            self.load_collection("restaurants", "./data/semantic_index.bin")
            logger.info("  ✅ Restaurants collection loaded")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not load restaurants: {e}")
        
        # Try to load attractions
        try:
            self.load_collection("attractions", "./data/attractions_index.bin")
            logger.info("  ✅ Attractions collection loaded")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not load attractions: {e}")
        
        # Try to load tips
        try:
            self.load_collection("tips", "./data/tips_index.bin")
            logger.info("  ✅ Tips collection loaded")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not load tips: {e}")
        
        logger.info(f"✨ Initialized {len(self.collections)} collections")
        
        # For backwards compatibility, set default index/items
        if "restaurants" in self.collections:
            self.index = self.collections["restaurants"]["index"]
            self.items = self.collections["restaurants"]["items"]
    
    def index_items(self, items, save_path="./data/semantic_index.bin"):
        """Index items for semantic search"""
        print(f"🔄 Indexing {len(items)} items...")
        
        # Create text representations
        texts = [self._create_text_representation(item) for item in items]
        
        # Encode in batches
        embeddings = self.encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Use GPU for FAISS if available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("🚀 Using GPU acceleration for FAISS")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            print("💻 Using CPU for FAISS")
        
        # Add embeddings
        self.index.add(embeddings.astype('float32'))
        self.items = items
        
        # Save index
        self._save_index(save_path)
        
        print(f"✅ Indexed {len(items)} items")
    
    
    def search(self, query, top_k=5, filters=None, collection="restaurants"):
        """
        Search for relevant items in specified collection
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply
            collection: Collection name (restaurants, attractions, tips)
        """
        # Get collection
        if collection not in self.collections:
            logger.warning(f"Collection '{collection}' not found, using default")
            if not self.collections:
                raise ValueError("No collections loaded")
            collection = list(self.collections.keys())[0]
        
        col = self.collections[collection]
        index = col["index"]
        items = col["items"]
        
        if index is None:
            raise ValueError(f"Index for collection '{collection}' not loaded")
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = index.search(
            query_embedding.astype('float32'),
            min(top_k * 2, len(items))  # Get more to allow for filtering
        )
        
        # Get results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(items):
                item = items[idx].copy()
                item['similarity_score'] = float(1 / (1 + distance))
                
                if self._matches_filters(item, filters):
                    results.append(item)
                    if len(results) >= top_k:
                        break
        
        return results
    
    def _create_text_representation(self, item):
        """Create rich text representation"""
        parts = []
        for key in ['name', 'description', 'category', 'location', 'cuisine', 'tags']:
            if key in item and item[key]:
                parts.append(str(item[key]))
        return " - ".join(parts) if parts else "unknown"
    
    def _matches_filters(self, item, filters):
        """Check if item matches filters"""
        if not filters:
            return True
        for key, value in filters.items():
            if key in item:
                if isinstance(value, list):
                    if item[key] not in value:
                        return False
                elif item[key] != value:
                    return False
        return True
    
    def _save_index(self, path):
        """Save index to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Move to CPU before saving
        if self.use_gpu and faiss.get_num_gpus() > 0:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            index_cpu = self.index
        
        faiss.write_index(index_cpu, path)
        with open(path + ".items", 'wb') as f:
            pickle.dump(self.items, f)
        print(f"💾 Index saved to {path}")
    
    
    def load_index(self, path):
        """Load index from disk (backwards compatibility)"""
        self.load_collection("default", path)
        if "default" in self.collections:
            self.index = self.collections["default"]["index"]
            self.items = self.collections["default"]["items"]
    
    def load_collection(self, name, path):
        """Load a named collection from disk"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
        
        logger.info(f"📥 Loading collection '{name}' from {path}...")
        
        # Load FAISS index
        index = faiss.read_index(path)
        
        # Move to GPU if available
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        # Load items
        with open(path + ".items", 'rb') as f:
            items = pickle.load(f)
        
        # Store collection
        self.collections[name] = {
            "index": index,
            "items": items
        }
        
        logger.info(f"✅ Loaded collection '{name}' with {len(items)} items")

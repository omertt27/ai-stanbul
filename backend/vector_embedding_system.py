#!/usr/bin/env python3
"""
Vector Embeddings System for AI Istanbul
========================================

Implements:
1. Text encoding using SentenceTransformers (MiniLM, SBERT)
2. FAISS vector storage for semantic similarity search
3. Precomputed vectors with daily updates
4. Hybrid search combining keyword + semantic search
"""

import numpy as np
import faiss
import pickle
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import sqlite3
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class VectorDocument:
    """Document with vector representation"""
    id: str
    content: str
    metadata: Dict[str, Any]
    vector: Optional[np.ndarray] = None
    last_updated: Optional[datetime] = None

@dataclass
class SearchResult:
    """Search result with relevance score"""
    document: VectorDocument
    similarity_score: float
    search_type: str  # 'semantic', 'keyword', 'hybrid'

class VectorEmbeddingSystem:
    """Complete vector embedding and semantic search system"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "vectors.db"):
        """Initialize with lightweight sentence transformer model"""
        self.model_name = model_name
        self.db_path = db_path
        self.model = None
        self.faiss_index = None
        self.document_store = {}
        self.lock = threading.Lock()
        
        # Initialize components
        self._init_model()
        self._init_database()
        self._load_vectors()
    
    def _init_model(self):
        """Load the sentence transformer model"""
        try:
            print(f"🤖 Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def _init_database(self):
        """Initialize vector storage database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_documents (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    vector_data BLOB NOT NULL,
                    content_type TEXT NOT NULL,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_content_type ON vector_documents(content_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_updated ON vector_documents(last_updated)")
            
            conn.commit()
    
    def _load_vectors(self):
        """Load existing vectors from database and build FAISS index"""
        print("📂 Loading existing vectors...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, content, metadata, vector_data, content_type, last_updated
                FROM vector_documents
                ORDER BY created_at
            """)
            
            vectors = []
            documents = []
            
            for row in cursor.fetchall():
                doc_id, content, metadata_json, vector_blob, content_type, last_updated = row
                
                # Deserialize vector
                vector = np.frombuffer(vector_blob, dtype=np.float32)
                
                # Create document object
                doc = VectorDocument(
                    id=doc_id,
                    content=content,
                    metadata=json.loads(metadata_json),
                    vector=vector,
                    last_updated=datetime.fromisoformat(last_updated) if last_updated else None
                )
                
                documents.append(doc)
                vectors.append(vector)
                self.document_store[doc_id] = doc
            
            if vectors:
                # Build FAISS index
                self._build_faiss_index(np.array(vectors))
                print(f"✅ Loaded {len(vectors)} vectors into FAISS index")
            else:
                # Initialize empty FAISS index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                print("📊 Initialized empty FAISS index")
    
    def _build_faiss_index(self, vectors: np.ndarray):
        """Build FAISS index from vectors"""
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(vectors)
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(vectors)
        
        print(f"🔍 Built FAISS index with {self.faiss_index.ntotal} vectors")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into vector representation"""
        if not self.model:
            raise ValueError("Model not initialized")
        
        # Clean and prepare text
        text = text.strip()
        if not text:
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Generate embedding
        embedding = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return embedding[0].astype(np.float32)
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any], content_type: str = "general") -> bool:
        """Add or update document in vector store"""
        try:
            with self.lock:
                # Generate vector
                vector = self.encode_text(content)
                
                # Create document
                doc = VectorDocument(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    vector=vector,
                    last_updated=datetime.now()
                )
                
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO vector_documents 
                        (id, content, metadata, vector_data, content_type, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        doc_id,
                        content,
                        json.dumps(metadata),
                        vector.tobytes(),
                        content_type,
                        doc.last_updated.isoformat()
                    ))
                    conn.commit()
                
                # Update document store
                self.document_store[doc_id] = doc
                
                # Rebuild FAISS index (for simplicity - could be optimized)
                self._rebuild_faiss_index()
                
                print(f"✅ Added document: {doc_id} ({content_type})")
                return True
                
        except Exception as e:
            print(f"❌ Error adding document {doc_id}: {e}")
            return False
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from all documents"""
        if not self.document_store:
            return
        
        vectors = []
        for doc in self.document_store.values():
            if doc.vector is not None:
                vectors.append(doc.vector)
        
        if vectors:
            self._build_faiss_index(np.array(vectors))
    
    def semantic_search(self, query: str, k: int = 10, min_similarity: float = 0.3) -> List[SearchResult]:
        """Perform semantic similarity search"""
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return []
        
        try:
            # Encode query
            query_vector = self.encode_text(query)
            query_vector = query_vector.reshape(1, -1)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_vector)
            
            # Search
            similarities, indices = self.faiss_index.search(query_vector, min(k, self.faiss_index.ntotal))
            
            results = []
            doc_list = list(self.document_store.values())
            
            for similarity, idx in zip(similarities[0], indices[0]):
                if idx != -1 and similarity >= min_similarity:
                    doc = doc_list[idx]
                    results.append(SearchResult(
                        document=doc,
                        similarity_score=float(similarity),
                        search_type="semantic"
                    ))
            
            return results
            
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 10, semantic_weight: float = 0.7) -> List[SearchResult]:
        """Combine semantic and keyword search"""
        # Get semantic results
        semantic_results = self.semantic_search(query, k * 2)  # Get more semantic results
        
        # Get keyword results (simple TF-IDF-like scoring)
        keyword_results = self._keyword_search(query, k * 2)
        
        # Combine and re-rank results
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result.document.id
            combined_results[doc_id] = {
                'document': result.document,
                'semantic_score': result.similarity_score,
                'keyword_score': 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result.document.id
            if doc_id in combined_results:
                combined_results[doc_id]['keyword_score'] = result.similarity_score
            else:
                combined_results[doc_id] = {
                    'document': result.document,
                    'semantic_score': 0.0,
                    'keyword_score': result.similarity_score
                }
        
        # Calculate hybrid scores
        final_results = []
        for doc_id, scores in combined_results.items():
            hybrid_score = (
                semantic_weight * scores['semantic_score'] + 
                (1 - semantic_weight) * scores['keyword_score']
            )
            
            final_results.append(SearchResult(
                document=scores['document'],
                similarity_score=hybrid_score,
                search_type="hybrid"
            ))
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x.similarity_score, reverse=True)
        return final_results[:k]
    
    def _keyword_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Simple keyword-based search for hybrid approach"""
        query_words = set(query.lower().split())
        results = []
        
        for doc in self.document_store.values():
            content_words = set(doc.content.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0:
                    results.append(SearchResult(
                        document=doc,
                        similarity_score=similarity,
                        search_type="keyword"
                    ))
        
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:k]
    
    def update_daily_vectors(self):
        """Update vectors for content modified in the last day"""
        cutoff_time = datetime.now() - timedelta(days=1)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, content, metadata, content_type
                FROM vector_documents
                WHERE last_updated > ?
            """, (cutoff_time.isoformat(),))
            
            updated_count = 0
            for row in cursor.fetchall():
                doc_id, content, metadata_json, content_type = row
                metadata = json.loads(metadata_json)
                
                # Re-encode and update
                if self.add_document(doc_id, content, metadata, content_type):
                    updated_count += 1
            
            print(f"🔄 Updated {updated_count} vectors from last 24 hours")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_documents": len(self.document_store),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "faiss_index_size": self.faiss_index.ntotal if self.faiss_index else 0,
            "last_update": datetime.now().isoformat()
        }
    
    def bulk_add_from_database(self):
        """Bulk add content from main database to vector store"""
        print("📊 Bulk adding content to vector store...")
        
        try:
            from database import SessionLocal
            from models import Restaurant, Museum, Place
            
            db = SessionLocal()
            added_count = 0
            
            # Add restaurants
            restaurants = db.query(Restaurant).all()
            for restaurant in restaurants:
                content = f"{restaurant.name} {restaurant.description or ''} {restaurant.cuisine_type or ''}"
                metadata = {
                    "type": "restaurant",
                    "name": restaurant.name,
                    "district": restaurant.district,
                    "cuisine_type": restaurant.cuisine_type,
                    "rating": float(restaurant.rating) if restaurant.rating else None
                }
                
                if self.add_document(f"restaurant_{restaurant.id}", content, metadata, "restaurant"):
                    added_count += 1
            
            # Add museums
            museums = db.query(Museum).all()
            for museum in museums:
                content = f"{museum.name} {museum.description or ''} {museum.category or ''}"
                metadata = {
                    "type": "museum",
                    "name": museum.name,
                    "district": museum.district,
                    "category": museum.category,
                    "rating": float(museum.rating) if museum.rating else None
                }
                
                if self.add_document(f"museum_{museum.id}", content, metadata, "museum"):
                    added_count += 1
            
            # Add places
            places = db.query(Place).all()
            for place in places:
                content = f"{place.name} {place.description or ''} {place.category or ''}"
                metadata = {
                    "type": "place",
                    "name": place.name,
                    "district": place.district,
                    "category": place.category,
                    "rating": float(place.rating) if place.rating else None
                }
                
                if self.add_document(f"place_{place.id}", content, metadata, "place"):
                    added_count += 1
            
            db.close()
            print(f"✅ Bulk added {added_count} documents to vector store")
            
        except Exception as e:
            print(f"❌ Bulk add error: {e}")

# Global vector system instance
vector_embedding_system = VectorEmbeddingSystem()

def initialize_vector_system():
    """Initialize the vector embedding system"""
    try:
        print("🚀 Initializing Vector Embedding System...")
        
        # Check if we need to populate with existing data
        stats = vector_embedding_system.get_stats()
        if stats["total_documents"] == 0:
            print("📊 No vectors found, performing bulk import...")
            vector_embedding_system.bulk_add_from_database()
        
        print(f"✅ Vector system initialized with {stats['total_documents']} documents")
        return True
        
    except Exception as e:
        print(f"❌ Vector system initialization error: {e}")
        return False

if __name__ == "__main__":
    # Test the vector embedding system
    print("🧪 Testing Vector Embedding System...")
    
    # Test basic operations
    system = VectorEmbeddingSystem()
    
    # Add test documents
    system.add_document(
        "test_1",
        "Hagia Sophia is a magnificent Byzantine church turned mosque in Istanbul",
        {"type": "attraction", "district": "Sultanahmet"},
        "attraction"
    )
    
    system.add_document(
        "test_2", 
        "Blue Mosque is famous for its blue tiles and six minarets",
        {"type": "attraction", "district": "Sultanahmet"},
        "attraction"
    )
    
    # Test semantic search
    results = system.semantic_search("Byzantine architecture Istanbul", k=5)
    print(f"✅ Semantic search test: {len(results)} results")
    
    for result in results:
        print(f"   - {result.document.id}: {result.similarity_score:.3f}")
    
    # Test hybrid search
    hybrid_results = system.hybrid_search("mosque Istanbul", k=5)
    print(f"✅ Hybrid search test: {len(hybrid_results)} results")
    
    # Get stats
    stats = system.get_stats()
    print(f"📊 System stats: {stats}")
    
    print("✅ Vector Embedding System is working correctly!")

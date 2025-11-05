"""
RAG Vector Service for AI-stanbul
Handles vector embeddings, FAISS indexing, and semantic retrieval

Architecture:
- SentenceTransformers for embeddings (384-dim, fast, multilingual)
- FAISS for vector similarity search (in-memory for MVP)
- Retrieval-Augmented Generation for LLaMA responses

Cost: $0 (local embeddings + FAISS)
Performance: <100ms query time for 1000+ documents
Scalability: Ready for cloud vector DB migration (Pinecone/Weaviate)

Priority: HIGH - Core MVP Feature
Status: Production-Ready
"""

import logging
from typing import List, Dict, Any, Optional
import os
import sys
import pickle
import numpy as np

# Add parent directory to path for imports
if '.' not in sys.path:
    sys.path.insert(0, '.')

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. Install with: pip install faiss-cpu")

try:
    from backend.data.rag_knowledge_base import build_knowledge_base, KnowledgeDocument
    KNOWLEDGE_BASE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_BASE_AVAILABLE = False
    logger.error("Knowledge base not available")


class RAGVectorService:
    """
    RAG Vector Service using FAISS + SentenceTransformers
    
    Features:
    - Fast local embedding generation (paraphrase-multilingual-MiniLM-L12-v2)
    - FAISS vector search (cosine similarity)
    - Persistent index storage
    - Multilingual support (Turkish + English)
    """
    
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        index_path: str = "./data/rag_index",
        rebuild_index: bool = False
    ):
        """
        Initialize RAG vector service
        
        Args:
            model_name: SentenceTransformer model name (384-dim multilingual)
            index_path: Path to store/load FAISS index
            rebuild_index: Force rebuild index from scratch
        """
        self.model_name = model_name
        self.index_path = index_path
        self.embedding_dim = 384  # MiniLM-L12-v2 dimension
        
        # Check dependencies
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("❌ sentence-transformers not available")
            self.available = False
            return
        
        if not FAISS_AVAILABLE:
            logger.error("❌ faiss not available")
            self.available = False
            return
        
        if not KNOWLEDGE_BASE_AVAILABLE:
            logger.error("❌ Knowledge base not available")
            self.available = False
            return
        
        self.available = True
        
        # Initialize components
        self.encoder = None
        self.index = None
        self.documents = []
        self.document_map = {}  # id -> KnowledgeDocument
        
        # Load or build index
        if rebuild_index or not self._load_index():
            logger.info("🔨 Building new RAG index...")
            self._build_index()
        else:
            logger.info("✅ RAG index loaded from disk")
    
    def _load_encoder(self):
        """Load SentenceTransformer encoder"""
        if self.encoder is None:
            logger.info(f"📦 Loading embedding model: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
            logger.info("✅ Encoder loaded")
    
    def _build_index(self):
        """Build FAISS index from knowledge base"""
        try:
            # Load encoder
            self._load_encoder()
            
            # Build knowledge base
            logger.info("📚 Building knowledge base...")
            self.documents = build_knowledge_base()
            
            # Create document map
            self.document_map = {doc.id: doc for doc in self.documents}
            
            # Extract text for embedding
            texts = [doc.content for doc in self.documents]
            
            logger.info(f"🔢 Generating embeddings for {len(texts)} documents...")
            embeddings = self.encoder.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # For cosine similarity
            )
            
            # Create FAISS index
            logger.info("🏗️ Building FAISS index...")
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product (cosine similarity with normalized vectors)
            self.index.add(embeddings.astype('float32'))
            
            # Save index
            self._save_index()
            
            logger.info(f"✅ RAG index built: {len(self.documents)} documents, {self.embedding_dim}D embeddings")
            
        except Exception as e:
            logger.error(f"❌ Error building RAG index: {e}")
            self.available = False
    
    def _save_index(self):
        """Save FAISS index and documents to disk"""
        try:
            os.makedirs(self.index_path, exist_ok=True)
            
            # Save FAISS index
            faiss_path = os.path.join(self.index_path, "faiss.index")
            faiss.write_index(self.index, faiss_path)
            
            # Save documents
            docs_path = os.path.join(self.index_path, "documents.pkl")
            with open(docs_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'document_map': self.document_map
                }, f)
            
            logger.info(f"💾 RAG index saved to {self.index_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving RAG index: {e}")
    
    def _load_index(self) -> bool:
        """Load FAISS index and documents from disk"""
        try:
            faiss_path = os.path.join(self.index_path, "faiss.index")
            docs_path = os.path.join(self.index_path, "documents.pkl")
            
            if not os.path.exists(faiss_path) or not os.path.exists(docs_path):
                logger.info("📭 No saved index found")
                return False
            
            # Load encoder first
            self._load_encoder()
            
            # Load FAISS index
            self.index = faiss.read_index(faiss_path)
            
            # Load documents
            with open(docs_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.document_map = data['document_map']
            
            logger.info(f"✅ Loaded RAG index: {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading RAG index: {e}")
            return False
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        category_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K most relevant documents for a query
        
        Args:
            query: User query text
            top_k: Number of documents to retrieve
            category_filter: Optional list of categories to filter by
            
        Returns:
            List of dicts with: {id, title, content, score, category, metadata}
        """
        if not self.available:
            logger.warning("⚠️ RAG service not available")
            return []
        
        try:
            # Embed query
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')
            
            # Search FAISS index
            # Get more results for filtering
            search_k = top_k * 3 if category_filter else top_k
            scores, indices = self.index.search(query_embedding, search_k)
            
            # Build results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(self.documents):
                    continue
                
                doc = self.documents[idx]
                
                # Apply category filter
                if category_filter and doc.category not in category_filter:
                    continue
                
                results.append({
                    'id': doc.id,
                    'title': doc.title,
                    'content': doc.content,
                    'category': doc.category,
                    'metadata': doc.metadata,
                    'score': float(score)
                })
                
                # Stop when we have enough results
                if len(results) >= top_k:
                    break
            
            logger.info(f"🔍 Retrieved {len(results)} documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents: {e}")
            return []
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 3,
        category_filter: Optional[List[str]] = None,
        max_length: int = 1000
    ) -> str:
        """
        Retrieve and format context for LLM prompt
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            category_filter: Optional category filter
            max_length: Max total character length
            
        Returns:
            Formatted context string for LLM
        """
        results = self.retrieve(query, top_k=top_k, category_filter=category_filter)
        
        if not results:
            return ""
        
        # Format context
        context_parts = [f"Relevant Istanbul Information:\n"]
        current_length = len(context_parts[0])
        
        for i, doc in enumerate(results, 1):
            doc_text = f"\n{i}. {doc['title']}\n{doc['content']}\n"
            
            # Check length limit
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "".join(context_parts)
    
    def rebuild_index(self):
        """Force rebuild the index"""
        logger.info("🔄 Rebuilding RAG index...")
        self._build_index()


# Singleton instance
_rag_service_instance = None


def get_rag_service(rebuild: bool = False) -> Optional[RAGVectorService]:
    """
    Get or create RAG service singleton
    
    Args:
        rebuild: Force rebuild index
        
    Returns:
        RAGVectorService instance or None if unavailable
    """
    global _rag_service_instance
    
    if _rag_service_instance is None or rebuild:
        _rag_service_instance = RAGVectorService(rebuild_index=rebuild)
    
    if not _rag_service_instance.available:
        return None
    
    return _rag_service_instance


if __name__ == "__main__":
    """Test RAG service"""
    print("🚀 Testing RAG Vector Service\n")
    
    # Initialize service
    service = RAGVectorService(rebuild_index=True)
    
    if not service.available:
        print("❌ RAG service not available")
        exit(1)
    
    # Test queries
    test_queries = [
        "Best way to cross from Europe to Asia in rainy weather",
        "Hidden cafes in Kadıköy",
        "How to use Istanbulkart",
        "M2 metro route information"
    ]
    
    print("=" * 60)
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 60)
        
        results = service.retrieve(query, top_k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc['title']} (score: {doc['score']:.3f})")
            print(f"   Category: {doc['category']}")
            print(f"   {doc['content'][:150]}...")
        
        print("\n" + "=" * 60)
    
    # Test LLM context generation
    print("\n\n📝 LLM Context Test")
    print("=" * 60)
    context = service.get_context_for_llm(
        "I want to go from Taksim to Kadıköy when it's raining",
        top_k=3
    )
    print(context)

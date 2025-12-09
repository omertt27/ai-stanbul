"""
Database-backed RAG (Retrieval-Augmented Generation) Service

This service provides semantic search over your actual databases:
- Restaurants (cuisine, location, ratings, reviews)
- Museums & Attractions (exhibits, hours, tickets)
- Events (concerts, festivals, shows)
- Places (districts, neighborhoods)
- Blog Posts (local insights, guides)

Features:
- Vector embeddings with Sentence Transformers
- ChromaDB for fast semantic search
- Real-time database sync
- Multi-lingual support (EN, TR, AR, FR, DE, RU)
- Typo-tolerant search
- Context-aware retrieval
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os

# Import your actual database models
from models import (
    Restaurant, Museum, Event, Place, 
    BlogPost, ChatHistory
)
from database import SessionLocal

logger = logging.getLogger(__name__)


class DatabaseRAGService:
    """
    RAG service that works with your actual database models
    Provides semantic search over restaurants, museums, events, and more
    """
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize RAG service with database connection and vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        if persist_directory is None:
            persist_directory = os.path.join(
                os.path.dirname(__file__),
                "../data/vector_db"
            )
        
        logger.info("🚀 Initializing Database RAG Service")
        logger.info(f"   Vector store: {persist_directory}")
        
        # Initialize embedding model (multilingual support)
        logger.info("   Loading multilingual embedding model...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Initialize ChromaDB
        logger.info("   Initializing ChromaDB...")
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collections for each data type
        self.collections = {}
        self._init_collections()
        
        logger.info("✅ Database RAG Service initialized successfully")
    
    def _init_collections(self):
        """Initialize ChromaDB collections for each data type"""
        collection_configs = [
            ("restaurants", "Restaurant information with cuisine, location, and ratings"),
            ("museums", "Museums and attractions with exhibits and visiting info"),
            ("events", "Events, concerts, festivals, and shows"),
            ("places", "Districts, neighborhoods, and landmarks"),
            ("blog_posts", "Local insights, guides, and tips from blog content"),
            ("general", "General Istanbul tourism knowledge")
        ]
        
        for name, description in collection_configs:
            try:
                self.collections[name] = self.client.get_or_create_collection(
                    name=f"istanbul_{name}",
                    metadata={"description": description}
                )
                logger.info(f"   ✓ Collection '{name}': {self.collections[name].count()} docs")
            except Exception as e:
                logger.error(f"   ✗ Failed to create collection '{name}': {e}")
    
    def sync_database(self, db: Session = None, force: bool = False):
        """
        Sync database content to vector store
        
        Args:
            db: Database session (creates new if None)
            force: If True, clears and rebuilds all collections
        """
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True
        
        try:
            logger.info("🔄 Syncing database to vector store...")
            
            # Check if we need to sync (skip if already populated and not forced)
            if not force:
                total_docs = sum(col.count() for col in self.collections.values())
                if total_docs > 0:
                    logger.info(f"   Vector store already has {total_docs} documents")
                    logger.info("   Use force=True to rebuild")
                    return
            
            # Clear collections if forced
            if force:
                logger.info("   Clearing existing collections...")
                for name, collection in self.collections.items():
                    try:
                        self.client.delete_collection(f"istanbul_{name}")
                    except:
                        pass
                self._init_collections()
            
            # Sync each data type
            self._sync_restaurants(db)
            self._sync_museums(db)
            self._sync_events(db)
            self._sync_places(db)
            self._sync_blog_posts(db)
            
            logger.info("✅ Database sync completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Database sync failed: {e}", exc_info=True)
            raise
        finally:
            if close_db:
                db.close()
    
    def _sync_restaurants(self, db: Session):
        """Sync restaurants to vector store"""
        try:
            restaurants = db.query(Restaurant).all()
            logger.info(f"   Syncing {len(restaurants)} restaurants...")
            
            if not restaurants:
                logger.warning("   No restaurants found in database")
                return
            
            documents = []
            metadatas = []
            ids = []
            
            for restaurant in restaurants:
                # Create searchable text
                doc_text = self._format_restaurant(restaurant)
                documents.append(doc_text)
                
                # Metadata for filtering
                metadatas.append({
                    'type': 'restaurant',
                    'id': str(restaurant.id),
                    'name': restaurant.name or '',
                    'cuisine': restaurant.cuisine or '',
                    'location': restaurant.location or '',
                    'rating': float(restaurant.rating) if restaurant.rating else 0.0,
                    'price_level': int(restaurant.price_level) if restaurant.price_level else 0
                })
                
                ids.append(f"restaurant_{restaurant.id}")
            
            # Generate embeddings and add to collection
            if documents:
                embeddings = self.encoder.encode(documents, show_progress_bar=False)
                self.collections['restaurants'].add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"   ✓ Added {len(documents)} restaurants")
        
        except Exception as e:
            logger.error(f"   ✗ Failed to sync restaurants: {e}")
    
    def _sync_museums(self, db: Session):
        """Sync museums/attractions to vector store"""
        try:
            museums = db.query(Museum).all()
            logger.info(f"   Syncing {len(museums)} museums/attractions...")
            
            if not museums:
                logger.warning("   No museums found in database")
                return
            
            documents = []
            metadatas = []
            ids = []
            
            for museum in museums:
                doc_text = self._format_museum(museum)
                documents.append(doc_text)
                
                metadatas.append({
                    'type': 'museum',
                    'id': str(museum.id),
                    'name': museum.name or '',
                    'location': museum.location or '',
                    'ticket_price': float(museum.ticket_price) if museum.ticket_price else 0.0
                })
                
                ids.append(f"museum_{museum.id}")
            
            if documents:
                embeddings = self.encoder.encode(documents, show_progress_bar=False)
                self.collections['museums'].add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"   ✓ Added {len(documents)} museums")
        
        except Exception as e:
            logger.error(f"   ✗ Failed to sync museums: {e}")
    
    def _sync_events(self, db: Session):
        """Sync events to vector store"""
        try:
            # Get upcoming and recent events (not events from years ago)
            now = datetime.utcnow()
            events = db.query(Event).filter(
                or_(
                    Event.date >= now,
                    Event.date == None  # Include events without dates
                )
            ).all()
            
            logger.info(f"   Syncing {len(events)} events...")
            
            if not events:
                logger.warning("   No events found in database")
                return
            
            documents = []
            metadatas = []
            ids = []
            
            for event in events:
                doc_text = self._format_event(event)
                documents.append(doc_text)
                
                metadatas.append({
                    'type': 'event',
                    'id': str(event.id),
                    'name': event.name or '',
                    'venue': event.venue or '',
                    'genre': event.genre or '',
                    'date': event.date.isoformat() if event.date else ''
                })
                
                ids.append(f"event_{event.id}")
            
            if documents:
                embeddings = self.encoder.encode(documents, show_progress_bar=False)
                self.collections['events'].add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"   ✓ Added {len(documents)} events")
        
        except Exception as e:
            logger.error(f"   ✗ Failed to sync events: {e}")
    
    def _sync_places(self, db: Session):
        """Sync places (districts, neighborhoods) to vector store"""
        try:
            places = db.query(Place).all()
            logger.info(f"   Syncing {len(places)} places...")
            
            if not places:
                logger.warning("   No places found in database")
                return
            
            documents = []
            metadatas = []
            ids = []
            
            for place in places:
                doc_text = self._format_place(place)
                documents.append(doc_text)
                
                metadatas.append({
                    'type': 'place',
                    'id': str(place.id),
                    'name': place.name or '',
                    'category': place.category or '',
                    'district': place.district or ''
                })
                
                ids.append(f"place_{place.id}")
            
            if documents:
                embeddings = self.encoder.encode(documents, show_progress_bar=False)
                self.collections['places'].add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"   ✓ Added {len(documents)} places")
        
        except Exception as e:
            logger.error(f"   ✗ Failed to sync places: {e}")
    
    def _sync_blog_posts(self, db: Session):
        """Sync published blog posts to vector store"""
        try:
            posts = db.query(BlogPost).filter(
                BlogPost.status == 'published'
            ).all()
            
            logger.info(f"   Syncing {len(posts)} blog posts...")
            
            if not posts:
                logger.warning("   No published blog posts found")
                return
            
            documents = []
            metadatas = []
            ids = []
            
            for post in posts:
                doc_text = self._format_blog_post(post)
                documents.append(doc_text)
                
                metadatas.append({
                    'type': 'blog_post',
                    'id': str(post.id),
                    'title': post.title or '',
                    'category': post.category or '',
                    'district': post.district or ''
                })
                
                ids.append(f"blog_{post.id}")
            
            if documents:
                embeddings = self.encoder.encode(documents, show_progress_bar=False)
                self.collections['blog_posts'].add(
                    embeddings=embeddings.tolist(),
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"   ✓ Added {len(documents)} blog posts")
        
        except Exception as e:
            logger.error(f"   ✗ Failed to sync blog posts: {e}")
    
    # ==========================================
    # Document Formatting Methods
    # ==========================================
    
    def _format_restaurant(self, restaurant: Restaurant) -> str:
        """Format restaurant for semantic search"""
        parts = [f"Restaurant: {restaurant.name}"]
        
        if restaurant.cuisine:
            parts.append(f"Cuisine: {restaurant.cuisine}")
        if restaurant.location:
            parts.append(f"Location: {restaurant.location}")
        if restaurant.rating:
            parts.append(f"Rating: {restaurant.rating}/5")
        if restaurant.description:
            parts.append(f"Description: {restaurant.description}")
        if restaurant.price_level:
            price_range = "$" * restaurant.price_level
            parts.append(f"Price: {price_range}")
        
        return "\n".join(parts)
    
    def _format_museum(self, museum: Museum) -> str:
        """Format museum/attraction for semantic search"""
        parts = [f"Museum/Attraction: {museum.name}"]
        
        if museum.location:
            parts.append(f"Location: {museum.location}")
        if museum.hours:
            parts.append(f"Hours: {museum.hours}")
        if museum.ticket_price:
            parts.append(f"Ticket Price: {museum.ticket_price} TL")
        if museum.highlights:
            parts.append(f"Highlights: {museum.highlights}")
        
        return "\n".join(parts)
    
    def _format_event(self, event: Event) -> str:
        """Format event for semantic search"""
        parts = [f"Event: {event.name}"]
        
        if event.venue:
            parts.append(f"Venue: {event.venue}")
        if event.date:
            parts.append(f"Date: {event.date.strftime('%B %d, %Y')}")
        if event.genre:
            parts.append(f"Genre: {event.genre}")
        
        return "\n".join(parts)
    
    def _format_place(self, place: Place) -> str:
        """Format place for semantic search"""
        parts = [f"Place: {place.name}"]
        
        if place.category:
            parts.append(f"Category: {place.category}")
        if place.district:
            parts.append(f"District: {place.district}")
        
        return "\n".join(parts)
    
    def _format_blog_post(self, post: BlogPost) -> str:
        """Format blog post for semantic search"""
        parts = [f"Guide: {post.title}"]
        
        if post.excerpt:
            parts.append(f"Summary: {post.excerpt}")
        elif post.content:
            # Use first 200 chars of content if no excerpt
            content_preview = post.content[:200] + "..." if len(post.content) > 200 else post.content
            parts.append(f"Content: {content_preview}")
        
        if post.category:
            parts.append(f"Category: {post.category}")
        if post.district:
            parts.append(f"District: {post.district}")
        
        return "\n".join(parts)
    
    # ==========================================
    # Search & Retrieval Methods
    # ==========================================
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across all data sources
        
        Args:
            query: User query (any language)
            top_k: Number of results to return
            categories: Filter by data types (e.g., ['restaurants', 'museums'])
            filters: Additional metadata filters
        
        Returns:
            List of relevant results with metadata and relevance scores
        """
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode([query])[0].tolist()
            
            # Determine which collections to search
            if categories:
                collections_to_search = [
                    (name, col) for name, col in self.collections.items()
                    if name in categories
                ]
            else:
                collections_to_search = list(self.collections.items())
            
            # Search each collection
            all_results = []
            for name, collection in collections_to_search:
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(top_k, collection.count()),
                        where=filters
                    )
                    
                    # Format results
                    if results['documents'] and len(results['documents'][0]) > 0:
                        for i in range(len(results['documents'][0])):
                            all_results.append({
                                'text': results['documents'][0][i],
                                'metadata': results['metadatas'][0][i],
                                'distance': results['distances'][0][i] if 'distances' in results else None,
                                'collection': name,
                                'relevance_score': 1.0 - (results['distances'][0][i] if 'distances' in results else 0.5)
                            })
                
                except Exception as e:
                    logger.warning(f"Search failed for collection '{name}': {e}")
            
            # Sort by relevance and return top_k
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            final_results = all_results[:top_k]
            
            logger.info(f"🔍 Search: '{query[:50]}...' -> {len(final_results)} results")
            return final_results
        
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    def search_restaurants(
        self,
        query: str,
        top_k: int = 5,
        cuisine: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_price_level: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search restaurants with specific filters"""
        filters = {}
        if cuisine:
            filters['cuisine'] = cuisine
        # Note: ChromaDB only supports exact match filters, not range queries
        # For rating/price filtering, we'll filter post-retrieval
        
        results = self.search(query, top_k=top_k * 2, categories=['restaurants'], filters=filters)
        
        # Apply post-retrieval filters
        filtered = []
        for result in results:
            metadata = result['metadata']
            if min_rating and metadata.get('rating', 0) < min_rating:
                continue
            if max_price_level and metadata.get('price_level', 99) > max_price_level:
                continue
            filtered.append(result)
        
        return filtered[:top_k]
    
    def search_events(
        self,
        query: str,
        top_k: int = 5,
        genre: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search events with specific filters"""
        filters = {}
        if genre:
            filters['genre'] = genre
        
        return self.search(query, top_k=top_k, categories=['events'], filters=filters)
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 3,
        max_length: int = 1500
    ) -> str:
        """
        Get formatted context for LLM prompt
        
        Args:
            query: User query
            top_k: Number of context items
            max_length: Maximum context length in characters
        
        Returns:
            Formatted context string for LLM
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return ""
        
        # Build context string
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            text = result['text']
            metadata = result['metadata']
            
            # Add type label
            item_type = metadata.get('type', 'item').title()
            context_item = f"[{item_type} {i}]\n{text}\n"
            
            # Check length limit
            if current_length + len(context_item) > max_length:
                break
            
            context_parts.append(context_item)
            current_length += len(context_item)
        
        return "\n".join(context_parts)
    
    def enhance_llm_prompt(
        self,
        user_query: str,
        system_prompt: str = "",
        top_k: int = 3
    ) -> str:
        """
        Enhance LLM prompt with retrieved context
        
        Args:
            user_query: User's question
            system_prompt: Existing system prompt
            top_k: Number of context items
        
        Returns:
            Enhanced prompt with retrieved context
        """
        context = self.get_context_for_llm(user_query, top_k=top_k)
        
        if not context:
            return user_query
        
        enhanced = f"""You are an expert Istanbul tourism guide. Use the following verified information to answer the user's question accurately.

RETRIEVED INFORMATION:
{context}

USER QUESTION: {user_query}

INSTRUCTIONS:
- Base your answer primarily on the retrieved information above
- Include specific details (names, locations, prices, ratings, hours)
- If the information doesn't fully answer the question, say so
- Be conversational but accurate
- Keep response concise (2-3 sentences unless more detail requested)

ANSWER:"""
        
        return enhanced
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        stats = {
            'collections': {},
            'total_documents': 0,
            'last_sync': None  # TODO: Track sync timestamp
        }
        
        for name, collection in self.collections.items():
            count = collection.count()
            stats['collections'][name] = count
            stats['total_documents'] += count
        
        return stats


# ==========================================
# Singleton Instance
# ==========================================

_rag_instance = None

def get_rag_service(db: Session = None) -> DatabaseRAGService:
    """
    Get or create RAG service instance (singleton)
    
    Args:
        db: Database session for initial sync (optional)
    
    Returns:
        DatabaseRAGService instance
    """
    global _rag_instance
    
    if _rag_instance is None:
        logger.info("Creating new RAG service instance")
        _rag_instance = DatabaseRAGService()
        
        # Check if we need to sync
        stats = _rag_instance.get_stats()
        if stats['total_documents'] == 0:
            logger.info("Vector store is empty, syncing from database...")
            try:
                _rag_instance.sync_database(db=db, force=False)
            except Exception as e:
                logger.error(f"Failed to sync database: {e}")
    
    return _rag_instance


# ==========================================
# CLI Tool for Management
# ==========================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Istanbul RAG Service CLI\n")
    
    # Initialize service
    rag = DatabaseRAGService()
    
    # Parse command
    command = sys.argv[1] if len(sys.argv) > 1 else "stats"
    
    if command == "sync":
        print("📊 Syncing database to vector store...\n")
        force = "--force" in sys.argv
        rag.sync_database(force=force)
        print("\n✅ Sync complete!")
    
    elif command == "stats":
        print("📊 Vector Store Statistics:\n")
        stats = rag.get_stats()
        for name, count in stats['collections'].items():
            print(f"   {name:20s}: {count:5d} documents")
        print(f"\n   Total: {stats['total_documents']} documents")
    
    elif command == "search":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "best restaurants in Sultanahmet"
        print(f"🔍 Searching: '{query}'\n")
        results = rag.search(query, top_k=5)
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}] (Score: {result['relevance_score']:.3f})")
            print(f"Type: {result['metadata']['type']}")
            print(f"Name: {result['metadata'].get('name', 'N/A')}")
            print(f"Text: {result['text'][:150]}...")
    
    elif command == "test":
        print("🧪 Running test queries...\n")
        test_queries = [
            "best Turkish restaurant near Sultanahmet",
            "what to see in Hagia Sophia",
            "upcoming concerts in Istanbul",
            "romantic neighborhoods for couples",
            "family-friendly activities"
        ]
        
        for query in test_queries:
            print(f"\n❓ {query}")
            results = rag.search(query, top_k=2)
            if results:
                top = results[0]
                print(f"   ✓ Found: {top['metadata'].get('name', 'N/A')} ({top['metadata']['type']})")
                print(f"   Score: {top['relevance_score']:.3f}")
            else:
                print("   ✗ No results")
    
    else:
        print(f"Unknown command: {command}")
        print("\nUsage:")
        print("  python database_rag_service.py stats          # Show statistics")
        print("  python database_rag_service.py sync [--force] # Sync database")
        print("  python database_rag_service.py search <query> # Test search")
        print("  python database_rag_service.py test           # Run test queries")

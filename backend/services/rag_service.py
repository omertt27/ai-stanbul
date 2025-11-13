"""
RAG (Retrieval-Augmented Generation) Service for Istanbul Chatbot

This service provides context-aware retrieval of Istanbul tourism information
to enhance LLM responses with accurate, up-to-date data.

Features:
- Vector-based semantic search
- Multi-category knowledge base (restaurants, attractions, neighborhoods, etc.)
- Intelligent context ranking
- Real-time data updates
"""

import json
import os
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)

class IstanbulRAG:
    """RAG system for Istanbul tourism chatbot"""
    
    def __init__(self, knowledge_base_path: str = None, persist_directory: str = None):
        """
        Initialize RAG system with knowledge base and vector store
        
        Args:
            knowledge_base_path: Path to JSON knowledge base file
            persist_directory: Directory to persist ChromaDB data
        """
        # Set default paths
        if knowledge_base_path is None:
            knowledge_base_path = os.path.join(
                os.path.dirname(__file__), 
                "../data/istanbul_knowledge_base.json"
            )
        
        if persist_directory is None:
            persist_directory = os.path.join(
                os.path.dirname(__file__),
                "../data/chroma_db"
            )
        
        logger.info(f"Initializing Istanbul RAG system")
        logger.info(f"Knowledge base: {knowledge_base_path}")
        logger.info(f"Vector store: {persist_directory}")
        
        # Initialize embedding model (lightweight, fast model)
        logger.info("Loading sentence transformer model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        logger.info("Initializing ChromaDB...")
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection("istanbul_data")
            logger.info("Loaded existing ChromaDB collection")
        except:
            self.collection = self.client.create_collection(
                name="istanbul_data",
                metadata={"description": "Istanbul tourism knowledge base"}
            )
            logger.info("Created new ChromaDB collection")
        
        # Load knowledge base
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()
        
        # Check if we need to populate the database
        if self.collection.count() == 0:
            logger.info("Empty collection detected, populating...")
            self.populate_database()
        else:
            logger.info(f"Collection has {self.collection.count()} documents")
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base from JSON file"""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded knowledge base with {len(data)} categories")
            return data
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return {}
    
    def populate_database(self):
        """Populate vector database with knowledge base content"""
        logger.info("Populating vector database...")
        
        documents = []
        metadatas = []
        ids = []
        
        # Process restaurants
        for item in self.knowledge_base.get('restaurants', []):
            doc_text = self._format_restaurant(item)
            documents.append(doc_text)
            metadatas.append({
                'category': 'restaurant',
                'id': item['id'],
                'name': item['name'],
                'location': item['neighborhood'],
                'price_range': item['price_range']
            })
            ids.append(item['id'])
        
        # Process attractions
        for item in self.knowledge_base.get('attractions', []):
            doc_text = self._format_attraction(item)
            documents.append(doc_text)
            metadatas.append({
                'category': 'attraction',
                'id': item['id'],
                'name': item['name'],
                'location': item['neighborhood'],
                'type': item['type']
            })
            ids.append(item['id'])
        
        # Process neighborhoods
        for item in self.knowledge_base.get('neighborhoods', []):
            doc_text = self._format_neighborhood(item)
            documents.append(doc_text)
            metadatas.append({
                'category': 'neighborhood',
                'id': item['id'],
                'name': item['name'],
                'district': item['district']
            })
            ids.append(item['id'])
        
        # Process transportation
        for item in self.knowledge_base.get('transportation', []):
            doc_text = self._format_transportation(item)
            documents.append(doc_text)
            metadatas.append({
                'category': 'transportation',
                'id': item['id'],
                'type': item['type'],
                'name': item['name']
            })
            ids.append(item['id'])
        
        # Process events
        for item in self.knowledge_base.get('events', []):
            doc_text = self._format_event(item)
            documents.append(doc_text)
            metadatas.append({
                'category': 'event',
                'id': item['id'],
                'name': item['name'],
                'season': item['season']
            })
            ids.append(item['id'])
        
        # Process local tips
        for tip_category in self.knowledge_base.get('local_tips', []):
            for i, tip in enumerate(tip_category['tips']):
                doc_text = f"Local tip about {tip_category['category']}: {tip}"
                documents.append(doc_text)
                metadatas.append({
                    'category': 'local_tip',
                    'id': f"tip_{tip_category['category']}_{i}",
                    'tip_category': tip_category['category']
                })
                ids.append(f"tip_{tip_category['category']}_{i}")
        
        # Process weather info
        for item in self.knowledge_base.get('weather', []):
            doc_text = self._format_weather(item)
            documents.append(doc_text)
            metadatas.append({
                'category': 'weather',
                'id': f"weather_{item['season']}",
                'season': item['season']
            })
            ids.append(f"weather_{item['season']}")
        
        # Process districts
        for item in self.knowledge_base.get('districts', []):
            doc_text = self._format_district(item)
            documents.append(doc_text)
            metadatas.append({
                'category': 'district',
                'id': item['id'],
                'name': item['name'],
                'type': item['type']
            })
            ids.append(item['id'])
        
        # Process district relationships
        for item in self.knowledge_base.get('district_relationships', []):
            doc_text = self._format_district_relationship(item)
            documents.append(doc_text)
            metadatas.append({
                'category': 'district_relationship',
                'id': item['id'],
                'type': item['type']
            })
            ids.append(item['id'])
        
        # Process query patterns
        for item in self.knowledge_base.get('query_patterns', []):
            doc_text = self._format_query_pattern(item)
            documents.append(doc_text)
            metadatas.append({
                'category': 'query_pattern',
                'id': item['id'],
                'query_type': item['query_type']
            })
            ids.append(item['id'])
        
        # Generate embeddings and add to database
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✅ Successfully populated database with {len(documents)} documents")
    
    def _format_restaurant(self, item: Dict) -> str:
        """Format restaurant data for vector search"""
        return f"""Restaurant: {item['name']}
Type: {item['cuisine']}
Location: {item['neighborhood']}, {item['address']}
Price: {item['price_range']} | Rating: {item['rating']}/5
Specialty: {item['specialty']}
Description: {item['description']}
Tips: {', '.join(item['tips'])}
Hours: {item['hours']}
Nearby: {', '.join(item['nearby_attractions'])}"""
    
    def _format_attraction(self, item: Dict) -> str:
        """Format attraction data for vector search"""
        return f"""Attraction: {item['name']}
Type: {item['type']} | Category: {item['category']}
Location: {item['neighborhood']}, {item['address']}
Price: {item['price']} | Rating: {item['rating']}/5
Description: {item['description']}
Highlights: {', '.join(item['highlights'])}
Tips: {', '.join(item['tips'])}
Hours: {item['hours']} | Duration: {item['duration']}
Best time: {item['best_time']}
Nearby: {', '.join(item['nearby'])}"""
    
    def _format_neighborhood(self, item: Dict) -> str:
        """Format neighborhood data for vector search"""
        return f"""Neighborhood: {item['name']}
District: {item['district']}
Description: {item['description']}
Atmosphere: {item['atmosphere']}
Best for: {', '.join(item['best_for'])}
Main attractions: {', '.join(item['main_attractions'])}
Vibe: {item['vibe']}
Safety: {item['safety']}
Tips: {', '.join(item['tips'])}"""
    
    def _format_transportation(self, item: Dict) -> str:
        """Format transportation data for vector search"""
        # Handle different field names
        key_stops = item.get('key_stops', item.get('routes', []))
        if isinstance(key_stops, list):
            key_stops_str = ', '.join(str(s) for s in key_stops)
        else:
            key_stops_str = str(key_stops)
        
        return f"""Transportation: {item['name']}
Type: {item['type']}
Description: {item['description']}
Key stops/routes: {key_stops_str}
Frequency: {item['frequency']}
Hours: {item['hours']} | Cost: {item['cost']}
Tips: {', '.join(item['tips'])}"""
    
    def _format_event(self, item: Dict) -> str:
        """Format event data for vector search"""
        return f"""Event: {item['name']}
Type: {item['type']} | Season: {item['season']}
Description: {item['description']}
Duration: {item['duration']}
Tips: {', '.join(item['tips'])}"""
    
    def _format_weather(self, item: Dict) -> str:
        """Format weather data for vector search"""
        return f"""Weather in {item['season']} ({', '.join(item['months'])})
Temperature: {item['temperature']}
Conditions: {item['conditions']}
Advice: {item['advice']}
Crowds: {item['crowds']}
Highlights: {', '.join(item['highlights'])}"""
    
    def _format_district(self, item: Dict) -> str:
        """Format district data for vector search"""
        characteristics = ', '.join(item['characteristics'])
        landmarks = ', '.join(item['landmarks'])
        transport_info = []
        for transport_type, lines in item['transport'].items():
            transport_info.append(f"{transport_type}: {', '.join(lines)}")
        transport_str = '; '.join(transport_info)
        
        return f"""District: {item['name']} ({item['name_tr']})
Type: {item['type']}
Characteristics: {characteristics}
Description: {item['description']}
Culture Score: {item['culture_score']}/10 | Nightlife: {item['nightlife_score']}/10
Sea Access: {item['sea_access_score']}/10 | Quiet: {item['quiet_score']}/10
Authentic: {item['authentic_score']}/10 | Luxury: {item['luxury_score']}/10
Landmarks: {landmarks}
Transport: {transport_str}
Restaurant Density: {item['restaurant_density']}
Price Level: {item['price_level']}
Best Time: {item['best_time']}
Tourist Level: {item['tourist_level']}
Searchable Info: {item['searchable_text']}"""
    
    def _format_district_relationship(self, item: Dict) -> str:
        """Format district relationship data for vector search"""
        return f"""District Relationship: {item['description']}
Type: {item['type']}
{item['description']}"""
    
    def _format_query_pattern(self, item: Dict) -> str:
        """Format query pattern data for vector search"""
        districts = ', '.join(item['preferred_districts'])
        characteristics = ', '.join(item['characteristics'])
        return f"""Query Pattern for {item['query_type']} experiences:
Recommended Districts: {districts}
Key Characteristics: {characteristics}
{item['description']}"""
    
    def retrieve(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            top_k: Number of results to return
            category_filter: Optional category filter (restaurant, attraction, etc.)
        
        Returns:
            List of relevant documents with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode([query])[0].tolist()
            
            # Prepare filter
            where_filter = None
            if category_filter:
                where_filter = {"category": category_filter}
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def enhance_prompt(self, user_query: str, top_k: int = 3) -> str:
        """
        Enhance LLM prompt with retrieved context
        
        Args:
            user_query: User's question
            top_k: Number of context items to retrieve
        
        Returns:
            Enhanced prompt with context
        """
        # Retrieve relevant context
        results = self.retrieve(user_query, top_k=top_k)
        
        if not results:
            return user_query
        
        # Build context string
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Context {i}]\n{result['text']}\n")
        
        context_str = "\n".join(context_parts)
        
        # Create enhanced prompt
        enhanced_prompt = f"""You are an expert Istanbul tourism guide. Answer the user's question using ONLY the provided context below. Be specific, accurate, and helpful.

Context Information:
{context_str}

User Question: {user_query}

Instructions:
- Use specific details from the context (names, addresses, prices, tips)
- If asked about restaurants, mention price range and specialty
- If asked about attractions, mention hours, prices, and tips
- If asked about transportation, mention routes and costs
- Be conversational but informative
- If the context doesn't fully answer the question, say so and provide what you can

Answer:"""
        
        return enhanced_prompt
    
    def get_category_stats(self) -> Dict[str, int]:
        """Get statistics about knowledge base categories"""
        stats = {}
        categories = [
            'restaurant', 'attraction', 'neighborhood', 'transportation', 
            'event', 'local_tip', 'weather', 'district', 'district_relationship', 
            'query_pattern'
        ]
        for category in categories:
            count = len(self.collection.get(where={"category": category})['ids'])
            stats[category] = count
        return stats


# Singleton instance
_rag_instance = None

def get_rag_instance() -> IstanbulRAG:
    """Get or create RAG instance (singleton pattern)"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = IstanbulRAG()
    return _rag_instance


if __name__ == "__main__":
    # Test the RAG system
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Testing Istanbul RAG System\n")
    
    # Initialize
    rag = IstanbulRAG()
    
    # Show stats
    print("📊 Knowledge Base Statistics:")
    stats = rag.get_category_stats()
    for category, count in stats.items():
        print(f"  {category}: {count} items")
    print()
    
    # Test queries
    test_queries = [
        "What are good restaurants in Sultanahmet?",
        "Tell me about Hagia Sophia",
        "How do I get to Taksim Square?",
        "What's the weather like in spring?",
        "Any hidden gems in Istanbul?"
    ]
    
    print("🧪 Testing Queries:\n")
    for query in test_queries:
        print(f"❓ Query: {query}")
        results = rag.retrieve(query, top_k=2)
        print(f"✅ Found {len(results)} relevant results")
        if results:
            print(f"   Top result category: {results[0]['metadata']['category']}")
        print()
    
    print("✅ RAG System Test Complete!")

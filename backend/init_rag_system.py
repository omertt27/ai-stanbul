#!/usr/bin/env python3
"""
RAG System Initialization and Testing Script

This script helps you:
1. Install RAG dependencies
2. Sync your database to the vector store
3. Test RAG search
4. Check RAG statistics

Usage:
    python init_rag_system.py install  # Install dependencies
    python init_rag_system.py sync     # Sync database to vector store
    python init_rag_system.py test     # Run test queries
    python init_rag_system.py stats    # Show statistics
"""

import sys
import os
import subprocess
import logging

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install required packages for RAG"""
    print("📦 Installing RAG dependencies...\n")
    
    packages = [
        "sentence-transformers",  # For embeddings
        "chromadb",  # Vector database
        "torch",  # Required by sentence-transformers
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("\n✅ All dependencies installed successfully!")
    return True


def sync_database(force=False):
    """Sync database content to vector store"""
    print("🔄 Syncing database to vector store...\n")
    
    try:
        from services.database_rag_service import DatabaseRAGService
        from database import SessionLocal
        
        # Create RAG service
        rag = DatabaseRAGService()
        
        # Get database session
        db = SessionLocal()
        
        try:
            # Sync database
            rag.sync_database(db=db, force=force)
            print("\n✅ Database sync completed successfully!")
            
            # Show stats
            stats = rag.get_stats()
            print("\n📊 Vector Store Statistics:")
            for name, count in stats['collections'].items():
                print(f"   {name:20s}: {count:5d} documents")
            print(f"\n   Total: {stats['total_documents']} documents")
            
            return True
            
        finally:
            db.close()
            
    except ImportError as e:
        print(f"❌ Failed to import RAG service: {e}")
        print("\n💡 Tip: Install dependencies first with: python init_rag_system.py install")
        return False
    except Exception as e:
        print(f"❌ Sync failed: {e}")
        logger.error("Sync error", exc_info=True)
        return False


def test_rag_system():
    """Test RAG with sample queries"""
    print("🧪 Testing RAG system with sample queries...\n")
    
    try:
        from services.database_rag_service import DatabaseRAGService
        
        rag = DatabaseRAGService()
        
        # Test queries
        test_queries = [
            ("Find Turkish restaurants in Sultanahmet", "restaurants"),
            ("What museums should I visit?", "museums"),
            ("Any upcoming concerts or events?", "events"),
            ("Tell me about Kadikoy neighborhood", "places"),
            ("Best cafes with Bosphorus view", "restaurants"),
            ("Family-friendly activities in Istanbul", None),
        ]
        
        print(f"Running {len(test_queries)} test queries...\n")
        
        for i, (query, category) in enumerate(test_queries, 1):
            print(f"{'='*60}")
            print(f"Test {i}/{len(test_queries)}: {query}")
            print(f"{'='*60}")
            
            categories = [category] if category else None
            results = rag.search(query, top_k=3, categories=categories)
            
            if results:
                print(f"\n✅ Found {len(results)} results:\n")
                for j, result in enumerate(results, 1):
                    metadata = result['metadata']
                    score = result['relevance_score']
                    
                    print(f"  [{j}] {metadata.get('name', 'N/A')}")
                    print(f"      Type: {metadata['type']}")
                    print(f"      Score: {score:.3f}")
                    if metadata.get('location'):
                        print(f"      Location: {metadata['location']}")
                    if metadata.get('rating'):
                        print(f"      Rating: {metadata['rating']}/5")
                    print()
            else:
                print("\n❌ No results found\n")
            
            print()
        
        print("\n✅ All tests completed!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import RAG service: {e}")
        print("\n💡 Tip: Install dependencies first with: python init_rag_system.py install")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error("Test error", exc_info=True)
        return False


def show_stats():
    """Show RAG statistics"""
    print("📊 RAG System Statistics\n")
    
    try:
        from services.database_rag_service import DatabaseRAGService
        
        rag = DatabaseRAGService()
        stats = rag.get_stats()
        
        print("Vector Store Collections:")
        print(f"{'Collection':<20} {'Documents':<10}")
        print("-" * 30)
        
        for name, count in stats['collections'].items():
            print(f"{name:<20} {count:<10}")
        
        print("-" * 30)
        print(f"{'TOTAL':<20} {stats['total_documents']:<10}")
        
        print("\n✅ Statistics retrieved successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import RAG service: {e}")
        print("\n💡 Tip: Install dependencies first with: python init_rag_system.py install")
        return False
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
        logger.error("Stats error", exc_info=True)
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "install":
        success = install_dependencies()
        sys.exit(0 if success else 1)
    
    elif command == "sync":
        force = "--force" in sys.argv
        if force:
            print("⚠️  Force sync enabled - will rebuild all collections\n")
        success = sync_database(force=force)
        sys.exit(0 if success else 1)
    
    elif command == "test":
        success = test_rag_system()
        sys.exit(0 if success else 1)
    
    elif command == "stats":
        success = show_stats()
        sys.exit(0 if success else 1)
    
    elif command == "all":
        print("🚀 Running full RAG setup...\n")
        
        # Install dependencies
        if not install_dependencies():
            sys.exit(1)
        
        print("\n" + "="*60 + "\n")
        
        # Sync database
        if not sync_database(force=False):
            sys.exit(1)
        
        print("\n" + "="*60 + "\n")
        
        # Run tests
        if not test_rag_system():
            sys.exit(1)
        
        print("\n" + "="*60)
        print("✅ RAG system is fully set up and ready to use!")
        print("="*60)
        sys.exit(0)
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

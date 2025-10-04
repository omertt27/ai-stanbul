"""
Quick validation script for enhanced AI Istanbul components
Tests each component individually to identify and fix remaining issues
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def test_component(component_name, test_func):
    """Test a single component"""
    print(f"\n🧪 Testing {component_name}...")
    try:
        result = test_func()
        if result:
            print(f"✅ {component_name} - Working")
            return True
        else:
            print(f"❌ {component_name} - Failed test")
            return False
    except Exception as e:
        print(f"❌ {component_name} - Error: {e}")
        return False

def test_user_profiling():
    """Test user profiling system"""
    from user_profiling_system import UserProfilingSystem
    
    profiling = UserProfilingSystem("test_profiles.db")
    
    # Test profile creation
    profile = profiling.get_or_create_profile('test_user')
    if not profile:
        return False
    
    # Test getting profile
    retrieved = profiling.get_user_profile('test_user')
    if not retrieved:
        return False
    
    # Test updating profile
    test_interaction = {
        'query': 'I love historical sites',
        'intent': 'attractions', 
        'entities': {'attraction_type': 'historical'},
        'timestamp': '2024-01-01T10:00:00',
        'satisfaction_score': 0.9
    }
    
    profiling.update_user_profile('test_user', 'I love historical sites', 'Here are some sites...', 0.9, test_interaction)
    
    return True

def test_intent_classifier():
    """Test enhanced intent classifier"""
    from enhanced_intent_classifier import EnhancedIntentClassifier
    
    classifier = EnhancedIntentClassifier()
    
    # Test basic classification
    result = classifier.classify_intent("How do I get to Hagia Sophia?")
    if not result or result.confidence <= 0:
        return False
    
    return True

def test_semantic_cache():
    """Test ML semantic cache"""
    from ml_semantic_cache import MLSemanticCache
    
    cache = MLSemanticCache('test_cache')
    
    # Test adding to cache
    try:
        cache.add_to_cache(
            "What are the best attractions?", 
            "Top attractions include Hagia Sophia...", 
            "attractions", 
            "en"
        )
    except Exception as e:
        print(f"Cache add failed: {e}")
        return False
    
    # Test retrieving from cache
    try:
        result = cache.get_cached_response("What are the top places to visit?")
        # It's OK if no result found, just that method exists
    except Exception as e:
        print(f"Cache retrieval failed: {e}")
        return False
    
    return True

def test_query_clustering():
    """Test query clustering system"""
    from query_clustering_system import QueryClusteringSystem
    
    clustering = QueryClusteringSystem('test_clustering')
    
    # Test adding queries
    try:
        # Check if method exists and works
        if hasattr(clustering, 'add_query'):
            clustering.add_query("How to get to Hagia Sophia?", "transportation")
        elif hasattr(clustering, 'add_to_cluster'):
            clustering.add_to_cluster("How to get to Hagia Sophia?", "transportation")
        else:
            print("No add_query or add_to_cluster method found")
            return False
    except Exception as e:
        print(f"Clustering add failed: {e}")
        return False
    
    return True

def test_production_integration():
    """Test production integration"""
    from enhanced_production_integration import EnhancedProductionOrchestrator
    
    config = {
        'cache_dir': 'test_cache',
        'profile_db_path': 'test_profiles.db'
    }
    
    orchestrator = EnhancedProductionOrchestrator(config)
    
    # Test query processing
    result = orchestrator.process_chat_query("What are the best attractions?", "test_user")
    
    if not result or not result.get('response'):
        return False
    
    return True

def main():
    """Run all component tests"""
    print("🔍 Enhanced AI Istanbul Component Validation")
    print("=" * 50)
    
    # Suppress some logging
    logging.getLogger().setLevel(logging.ERROR)
    
    tests = [
        ("User Profiling System", test_user_profiling),
        ("Intent Classifier", test_intent_classifier), 
        ("ML Semantic Cache", test_semantic_cache),
        ("Query Clustering", test_query_clustering),
        ("Production Integration", test_production_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        if test_component(name, test_func):
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"📊 Results: {passed}/{total} components working ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All components ready for production!")
        return True
    elif passed >= total * 0.8:
        print("✅ System mostly ready - minor fixes needed")
        return True
    else:
        print("⚠️ System needs more work before production")
        return False
    
    # Cleanup
    try:
        import shutil
        test_files = ['test_profiles.db', 'test_cache', 'test_clustering']
        for f in test_files:
            path = Path(f)
            if path.exists():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
    except:
        pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

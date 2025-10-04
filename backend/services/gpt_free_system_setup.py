"""
GPT-Free AI Istanbul System Setup and Testing Script
Comprehensive setup, testing, and validation of the GPT-free system
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    logger.info("🔍 Checking dependencies...")
    
    dependencies = {
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'sentence-transformers': 'sentence_transformers',
        'faiss': 'faiss'
    }
    
    available = {}
    missing = []
    
    for name, import_name in dependencies.items():
        try:
            __import__(import_name)
            available[name] = True
            logger.info(f"✅ {name} available")
        except ImportError:
            available[name] = False
            missing.append(name)
            logger.warning(f"⚠️ {name} not available")
    
    if missing:
        logger.warning(f"📦 Missing dependencies: {', '.join(missing)}")
        logger.info("Install with: pip install scikit-learn sentence-transformers faiss-cpu")
    else:
        logger.info("✅ All dependencies available")
    
    return available, missing

def setup_directories():
    """Setup required directories"""
    logger.info("📁 Setting up directories...")
    
    directories = [
        'cache_data',
        'clustering_data', 
        'production_cache',
        'production_clustering',
        'test_results',
        'exports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 Created/verified: {directory}")
    
    return directories

def initialize_systems():
    """Initialize the GPT-free systems"""
    logger.info("🚀 Initializing systems...")
    
    try:
        # Import our systems
        from services.enhanced_gpt_free_system import create_gpt_free_system
        from services.production_gpt_free_integration import setup_production_system
        
        # Create GPT-free system
        gpt_free_system = create_gpt_free_system({
            'cache_dir': 'test_cache_data',
            'clustering_dir': 'test_clustering_data'
        })
        
        # Create production orchestrator
        production_system = setup_production_system({
            'cache_dir': 'production_cache',
            'clustering_dir': 'production_clustering',
            'gpt_free_threshold': 0.6
        })
        
        logger.info("✅ Systems initialized successfully")
        return gpt_free_system, production_system
        
    except Exception as e:
        logger.error(f"❌ Error initializing systems: {e}")
        return None, None

def run_comprehensive_tests(gpt_free_system, production_system):
    """Run comprehensive tests on the systems"""
    logger.info("🧪 Running comprehensive tests...")
    
    test_queries = [
        # Transportation queries
        {
            'query': 'How to get to Hagia Sophia from Taksim?',
            'expected_intent': 'transportation',
            'context': {'from': 'taksim', 'to': 'hagia_sophia'}
        },
        {
            'query': 'Best way to Blue Mosque by metro',
            'expected_intent': 'transportation',
            'context': {'transport_mode': 'metro', 'destination': 'blue_mosque'}
        },
        {
            'query': 'How do I get to Galata Tower?',
            'expected_intent': 'transportation',
            'context': {'destination': 'galata_tower'}
        },
        
        # Food queries
        {
            'query': 'Best restaurants in Sultanahmet',
            'expected_intent': 'food',
            'context': {'area': 'sultanahmet', 'type': 'restaurants'}
        },
        {
            'query': 'Where to eat traditional Turkish food?',
            'expected_intent': 'food',
            'context': {'cuisine': 'turkish', 'type': 'traditional'}
        },
        {
            'query': 'Good breakfast places near Blue Mosque',
            'expected_intent': 'food',
            'context': {'meal': 'breakfast', 'location': 'blue_mosque'}
        },
        
        # Practical info queries
        {
            'query': 'Hagia Sophia opening hours',
            'expected_intent': 'practical_info',
            'context': {'attraction': 'hagia_sophia', 'info_type': 'hours'}
        },
        {
            'query': 'Blue Mosque ticket price',
            'expected_intent': 'practical_info',
            'context': {'attraction': 'blue_mosque', 'info_type': 'price'}
        },
        {
            'query': 'When does Topkapi Palace open?',
            'expected_intent': 'practical_info',
            'context': {'attraction': 'topkapi_palace', 'info_type': 'hours'}
        },
        
        # Exploration queries
        {
            'query': 'What to see in Beyoglu?',
            'expected_intent': 'exploration',
            'context': {'area': 'beyoglu', 'type': 'attractions'}
        },
        {
            'query': 'Things to do in Sultanahmet',
            'expected_intent': 'exploration', 
            'context': {'area': 'sultanahmet', 'type': 'activities'}
        },
        {
            'query': 'Best attractions for history lovers',
            'expected_intent': 'exploration',
            'context': {'interest': 'history', 'type': 'attractions'}
        },
        
        # Shopping queries
        {
            'query': 'Where to buy souvenirs in Istanbul?',
            'expected_intent': 'shopping',
            'context': {'product': 'souvenirs', 'type': 'shopping'}
        },
        {
            'query': 'Grand Bazaar shopping guide',
            'expected_intent': 'shopping',
            'context': {'location': 'grand_bazaar', 'type': 'shopping'}
        },
        
        # Complex queries
        {
            'query': 'I want to visit Hagia Sophia and Blue Mosque, then find a good restaurant nearby',
            'expected_intent': 'exploration',
            'context': {'attractions': ['hagia_sophia', 'blue_mosque'], 'need_food': True}
        },
        {
            'query': 'Planning a day in Istanbul - need transport, sightseeing, and food recommendations',
            'expected_intent': 'exploration',
            'context': {'type': 'day_plan', 'needs': ['transport', 'sightseeing', 'food']}
        }
    ]
    
    test_results = {
        'total_tests': len(test_queries),
        'gpt_free_system': {
            'successful': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'failures': 0,
            'avg_response_time': 0.0,
            'results': []
        },
        'production_system': {
            'successful': 0,
            'gpt_free_used': 0,
            'fallback_used': 0,
            'failures': 0,
            'avg_response_time': 0.0,  
            'results': []
        }
    }
    
    logger.info(f"🧪 Testing {len(test_queries)} queries...")
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        expected_intent = test_case['expected_intent']
        context = test_case['context']
        
        logger.info(f"Test {i}/{len(test_queries)}: {query[:50]}...")
        
        # Test GPT-free system
        if gpt_free_system:
            try:
                start_time = time.time()
                gpt_free_result = gpt_free_system.process_query(query, context)
                response_time = (time.time() - start_time) * 1000
                
                result = {
                    'query': query,
                    'expected_intent': expected_intent,
                    'response': gpt_free_result.response[:200] + "..." if len(gpt_free_result.response) > 200 else gpt_free_result.response,
                    'source': gpt_free_result.source,
                    'confidence': gpt_free_result.confidence,
                    'response_time_ms': response_time,
                    'metadata': gpt_free_result.metadata
                }
                
                test_results['gpt_free_system']['results'].append(result)
                test_results['gpt_free_system']['successful'] += 1
                
                if gpt_free_result.confidence >= 0.8:
                    test_results['gpt_free_system']['high_confidence'] += 1
                elif gpt_free_result.confidence >= 0.6:
                    test_results['gpt_free_system']['medium_confidence'] += 1
                else:
                    test_results['gpt_free_system']['low_confidence'] += 1
                
                logger.info(f"✅ GPT-Free: {gpt_free_result.source} ({gpt_free_result.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"❌ GPT-Free system failed: {e}")
                test_results['gpt_free_system']['failures'] += 1
        
        # Test production system
        if production_system:
            try:
                start_time = time.time()
                prod_result = production_system.process_chat_query(query, f"test_user_{i}", context)
                response_time = (time.time() - start_time) * 1000
                
                result = {
                    'query': query,
                    'expected_intent': expected_intent,
                    'response': prod_result['response'][:200] + "..." if len(prod_result['response']) > 200 else prod_result['response'],
                    'source': prod_result['source'],
                    'confidence': prod_result['confidence'],
                    'response_time_ms': response_time,
                    'cost_saved': prod_result.get('cost_saved', False),
                    'fallback_used': prod_result.get('fallback_used', False)
                }
                
                test_results['production_system']['results'].append(result)
                test_results['production_system']['successful'] += 1
                
                if prod_result['source'] in ['semantic_cache', 'query_clustering']:
                    test_results['production_system']['gpt_free_used'] += 1
                else:
                    test_results['production_system']['fallback_used'] += 1
                
                logger.info(f"✅ Production: {prod_result['source']} ({prod_result['confidence']:.2f})")
                
            except Exception as e:
                logger.error(f"❌ Production system failed: {e}")
                test_results['production_system']['failures'] += 1
        
        # Small delay between tests
        time.sleep(0.1)
    
    # Calculate averages
    if test_results['gpt_free_system']['results']:
        avg_time = sum(r['response_time_ms'] for r in test_results['gpt_free_system']['results'])
        test_results['gpt_free_system']['avg_response_time'] = avg_time / len(test_results['gpt_free_system']['results'])
    
    if test_results['production_system']['results']:
        avg_time = sum(r['response_time_ms'] for r in test_results['production_system']['results'])
        test_results['production_system']['avg_response_time'] = avg_time / len(test_results['production_system']['results'])
    
    logger.info("✅ Comprehensive testing completed")
    return test_results

def generate_test_report(test_results: Dict, dependencies: Dict, systems_initialized: bool):
    """Generate comprehensive test report"""
    logger.info("📊 Generating test report...")
    
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'system_status': {
            'dependencies_available': dependencies,
            'systems_initialized': systems_initialized
        },
        'test_results': test_results,
        'performance_analysis': {},
        'recommendations': []
    }
    
    # Performance analysis
    if test_results['gpt_free_system']['results']:
        gfs_results = test_results['gpt_free_system']
        total_tests = test_results['total_tests']
        
        report['performance_analysis']['gpt_free_system'] = {
            'success_rate': (gfs_results['successful'] / total_tests) * 100,
            'high_confidence_rate': (gfs_results['high_confidence'] / max(1, gfs_results['successful'])) * 100,
            'medium_confidence_rate': (gfs_results['medium_confidence'] / max(1, gfs_results['successful'])) * 100,
            'low_confidence_rate': (gfs_results['low_confidence'] / max(1, gfs_results['successful'])) * 100,
            'avg_response_time_ms': gfs_results['avg_response_time'],
            'failure_rate': (gfs_results['failures'] / total_tests) * 100
        }
    
    if test_results['production_system']['results']:
        prod_results = test_results['production_system']
        total_tests = test_results['total_tests']
        
        report['performance_analysis']['production_system'] = {
            'success_rate': (prod_results['successful'] / total_tests) * 100,
            'gpt_free_usage_rate': (prod_results['gpt_free_used'] / max(1, prod_results['successful'])) * 100,
            'fallback_usage_rate': (prod_results['fallback_used'] / max(1, prod_results['successful'])) * 100,
            'avg_response_time_ms': prod_results['avg_response_time'],
            'failure_rate': (prod_results['failures'] / total_tests) * 100
        }
    
    # Generate recommendations
    recommendations = []
    
    # Dependency recommendations
    if not dependencies.get('sentence-transformers', False):
        recommendations.append({
            'priority': 'high',
            'category': 'dependencies',
            'message': 'Install sentence-transformers for better semantic understanding',
            'action': 'pip install sentence-transformers'
        })
    
    if not dependencies.get('faiss', False):
        recommendations.append({
            'priority': 'medium',
            'category': 'dependencies', 
            'message': 'Install FAISS for faster similarity search',
            'action': 'pip install faiss-cpu'
        })
    
    # Performance recommendations
    if systems_initialized:
        gfs_perf = report['performance_analysis'].get('gpt_free_system', {})
        
        if gfs_perf.get('high_confidence_rate', 0) < 50:
            recommendations.append({
                'priority': 'medium',
                'category': 'performance',
                'message': 'Low high-confidence response rate - consider expanding knowledge base',
                'action': 'Add more training data and improve clustering'
            })
        
        if gfs_perf.get('avg_response_time_ms', 0) > 1000:
            recommendations.append({
                'priority': 'low',
                'category': 'performance',
                'message': 'Response time could be optimized',
                'action': 'Consider caching improvements or model optimization'
            })
    
    report['recommendations'] = recommendations
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"exports/gpt_free_system_test_report_{timestamp}.json"
    
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Test report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("GPT-FREE AI ISTANBUL SYSTEM - TEST SUMMARY")
        print("="*80)
        
        print(f"\n📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Total Tests: {test_results['total_tests']}")
        
        if systems_initialized:
            gfs_perf = report['performance_analysis'].get('gpt_free_system', {})
            prod_perf = report['performance_analysis'].get('production_system', {})
            
            print(f"\n🤖 GPT-Free System:")
            print(f"   Success Rate: {gfs_perf.get('success_rate', 0):.1f}%")
            print(f"   High Confidence: {gfs_perf.get('high_confidence_rate', 0):.1f}%")
            print(f"   Avg Response Time: {gfs_perf.get('avg_response_time_ms', 0):.1f}ms")
            
            print(f"\n🏭 Production System:")
            print(f"   Success Rate: {prod_perf.get('success_rate', 0):.1f}%")
            print(f"   GPT-Free Usage: {prod_perf.get('gpt_free_usage_rate', 0):.1f}%")
            print(f"   Avg Response Time: {prod_perf.get('avg_response_time_ms', 0):.1f}ms")
        
        print(f"\n📋 Recommendations: {len(recommendations)}")
        for rec in recommendations:
            print(f"   • [{rec['priority'].upper()}] {rec['message']}")
        
        print(f"\n📄 Full report: {report_file}")
        print("="*80)
        
        return report_file
        
    except Exception as e:
        logger.error(f"❌ Error saving test report: {e}")
        return None

def main():
    """Main setup and testing function"""
    print("🚀 GPT-Free AI Istanbul System - Setup & Testing")
    print("="*60)
    
    # Check dependencies
    dependencies, missing = check_dependencies()
    
    # Setup directories
    directories = setup_directories()
    
    # Initialize systems
    gpt_free_system, production_system = initialize_systems()
    systems_initialized = gpt_free_system is not None and production_system is not None
    
    if not systems_initialized:
        logger.error("❌ Failed to initialize systems - running limited tests")
    
    # Run tests
    if systems_initialized:
        test_results = run_comprehensive_tests(gpt_free_system, production_system)
    else:
        test_results = {
            'total_tests': 0,
            'gpt_free_system': {'results': []},
            'production_system': {'results': []}
        }
    
    # Generate report
    report_file = generate_test_report(test_results, dependencies, systems_initialized)
    
    # Final status
    if systems_initialized and test_results['total_tests'] > 0:
        success_rate = (test_results['gpt_free_system']['successful'] / test_results['total_tests']) * 100
        
        if success_rate >= 80:
            print("\n✅ SYSTEM STATUS: EXCELLENT - Ready for production")
        elif success_rate >= 60:
            print("\n🟡 SYSTEM STATUS: GOOD - Minor optimizations recommended")
        elif success_rate >= 40:
            print("\n🟠 SYSTEM STATUS: NEEDS IMPROVEMENT - Review recommendations")
        else:
            print("\n🔴 SYSTEM STATUS: CRITICAL - Major issues need addressing")
    else:
        print("\n⚠️ SYSTEM STATUS: NOT TESTED - Setup issues detected")
    
    print(f"\n📊 Complete report available at: {report_file}")
    print("🎯 Next steps: Review recommendations and optimize based on test results")

if __name__ == "__main__":
    main()

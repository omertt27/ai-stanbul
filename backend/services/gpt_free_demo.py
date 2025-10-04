#!/usr/bin/env python3
"""
GPT-Free AI Istanbul System Demo
Interactive demonstration of the GPT-free system capabilities
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, Any

def print_header():
    """Print demo header"""
    print("\n" + "="*80)
    print("🚀 GPT-FREE AI ISTANBUL SYSTEM - INTERACTIVE DEMO")
    print("="*80)
    print("This demo shows how the system handles queries without GPT dependency")
    print("Features: ML Caching, Query Clustering, Knowledge Base, Smart Fallbacks")
    print("-"*80)

def setup_demo_system():
    """Setup the demo system"""
    print("🔧 Setting up demo system...")
    
    try:
        # Create demo directories
        demo_dirs = ['demo_cache', 'demo_clustering', 'demo_exports']
        for directory in demo_dirs:
            os.makedirs(directory, exist_ok=True)
        
        # Import and initialize system
        from services.enhanced_gpt_free_system import create_gpt_free_system
        
        system = create_gpt_free_system({
            'cache_dir': 'demo_cache',
            'clustering_dir': 'demo_clustering'
        })
        
        print("✅ Demo system initialized successfully!")
        return system
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the correct directory with all dependencies installed")
        return None
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return None

def run_predefined_demos(system):
    """Run predefined demo queries"""
    print("\n🎯 PREDEFINED DEMO QUERIES")
    print("-"*50)
    
    demo_queries = [
        {
            'query': 'How to get to Hagia Sophia?',
            'description': 'Transportation query - should use clustering/templates'
        },
        {
            'query': 'Best restaurants in Sultanahmet',
            'description': 'Food recommendations - knowledge base lookup'
        },
        {
            'query': 'Blue Mosque opening hours',
            'description': 'Practical information - template response'
        },
        {
            'query': 'What to see in Beyoglu?',
            'description': 'Area exploration - knowledge base + templates'
        },
        {
            'query': 'Where to buy souvenirs in Istanbul?',
            'description': 'Shopping query - clustering system'
        },
        {
            'query': 'I need help planning my day in Istanbul',
            'description': 'Complex query - smart fallback system'
        }
    ]
    
    results = []
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n📝 Demo {i}/6: {demo['description']}")
        print(f"Q: {demo['query']}")
        print("Processing...", end="", flush=True)
        
        start_time = time.time()
        
        try:
            result = system.process_query(demo['query'], {'demo_mode': True})
            processing_time = (time.time() - start_time) * 1000
            
            print(f" ✅ Done ({processing_time:.0f}ms)")
            print(f"Source: {result.source} | Confidence: {result.confidence:.2f}")
            print(f"A: {result.response[:200]}...")
            if len(result.response) > 200:
                print("   [Response truncated for demo]")
            
            results.append({
                'query': demo['query'],
                'source': result.source,
                'confidence': result.confidence,
                'processing_time': processing_time,
                'response_length': len(result.response)
            })
            
        except Exception as e:
            print(f" ❌ Error: {e}")
            results.append({
                'query': demo['query'],
                'error': str(e)
            })
        
        time.sleep(0.5)  # Brief pause for readability
    
    return results

def run_interactive_demo(system):
    """Run interactive demo mode"""
    print("\n💬 INTERACTIVE DEMO MODE")
    print("-"*50)
    print("Ask questions about Istanbul! Type 'quit' to exit, 'stats' for statistics.")
    print("Examples: 'How to get to Galata Tower?', 'Turkish food recommendations'")
    
    query_count = 0
    
    while True:
        try:
            print(f"\n[Query {query_count + 1}]")
            user_query = input("🗣️  You: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            elif user_query.lower() in ['stats', 'statistics']:
                show_system_stats(system)
                continue
            elif not user_query:
                print("💡 Please enter a query or 'quit' to exit")
                continue
            
            print("🤖 AI Istanbul: Processing...", end="", flush=True)
            
            start_time = time.time()
            result = system.process_query(user_query, {'interactive_mode': True})
            processing_time = (time.time() - start_time) * 1000
            
            print(f" ✅ ({processing_time:.0f}ms)")
            print(f"🔍 Source: {result.source} | Confidence: {result.confidence:.2f}")
            print(f"💬 Response:\n{result.response}")
            
            query_count += 1
            
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")

def show_system_stats(system):
    """Show current system statistics"""
    try:
        stats = system.get_system_statistics()
        
        print("\n📊 SYSTEM STATISTICS")
        print("-"*30)
        
        overall = stats['overall_performance']
        print(f"Queries Processed: {overall['total_queries_processed']}")
        print(f"Cache Hit Rate: {overall['cache_hit_rate_percent']:.1f}%")
        print(f"GPT-Free Coverage: {overall['gpt_free_coverage_percent']:.1f}%")
        print(f"Avg Response Time: {overall['avg_response_time_ms']:.1f}ms")
        
        print("\nCoverage by Source:")
        for source, percentage in stats['coverage_by_source'].items():
            print(f"  {source.replace('_', ' ').title()}: {percentage:.1f}%")
        
        cache_info = stats['semantic_cache']
        print(f"\nCache Statistics:")
        print(f"  Cached Responses: {cache_info['total_cached_responses']}")
        print(f"  Cache Hits: {cache_info['cache_hits']}")
        print(f"  Cache Misses: {cache_info['cache_misses']}")
        
        clustering_info = stats['clustering_system']
        print(f"\nClustering Statistics:")
        print(f"  Total Clusters: {clustering_info['total_clusters']}")
        print(f"  Templates Created: {clustering_info['total_templates']}")
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")

def generate_demo_report(predefined_results, system):
    """Generate demo report"""
    print("\n📋 GENERATING DEMO REPORT")
    print("-"*30)
    
    try:
        # Get system stats
        stats = system.get_system_statistics()
        
        # Create report
        report = {
            'demo_timestamp': datetime.now().isoformat(),
            'predefined_demo_results': predefined_results,
            'system_statistics': stats,
            'demo_summary': {
                'total_predefined_queries': len(predefined_results),
                'successful_queries': len([r for r in predefined_results if 'error' not in r]),
                'failed_queries': len([r for r in predefined_results if 'error' in r]),
                'avg_processing_time': sum(r.get('processing_time', 0) for r in predefined_results) / max(1, len(predefined_results)),
                'sources_used': list(set(r.get('source', 'error') for r in predefined_results))
            }
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"demo_exports/demo_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Demo report saved to: {report_file}")
        
        # Print summary
        summary = report['demo_summary']
        print(f"\n📊 Demo Summary:")
        print(f"  Successful Queries: {summary['successful_queries']}/{summary['total_predefined_queries']}")
        print(f"  Average Processing Time: {summary['avg_processing_time']:.1f}ms")
        print(f"  Sources Used: {', '.join(summary['sources_used'])}")
        
        return report_file
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        return None

def show_demo_menu():
    """Show demo menu options"""
    print("\n🎯 DEMO OPTIONS")
    print("-"*20)
    print("1. Run Predefined Demo Queries")
    print("2. Interactive Demo Mode") 
    print("3. View System Statistics")
    print("4. Generate Demo Report")
    print("5. Exit Demo")
    print()

def main():
    """Main demo function"""
    print_header()
    
    # Setup system
    system = setup_demo_system()
    if not system:
        print("❌ Failed to setup demo system. Exiting...")
        return
    
    predefined_results = []
    
    while True:
        show_demo_menu()
        
        try:
            choice = input("Select option (1-5): ").strip()
            
            if choice == '1':
                predefined_results = run_predefined_demos(system)
            elif choice == '2':
                run_interactive_demo(system)
            elif choice == '3':
                show_system_stats(system)
            elif choice == '4':
                generate_demo_report(predefined_results, system)
            elif choice == '5':
                break
            else:
                print("❌ Invalid option. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Demo interrupted by user")
            break
        except Exception as e:
            print(f"❌ Demo error: {e}")
    
    print("\n🎯 DEMO COMPLETE")
    print("="*50)
    print("Thank you for trying the GPT-Free AI Istanbul System!")
    print("For production deployment, see GPT_FREE_IMPLEMENTATION_GUIDE.md")
    print("="*50)

if __name__ == "__main__":
    main()

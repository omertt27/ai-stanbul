#!/usr/bin/env python3
"""
Query Preprocessing Pipeline - Interactive Demo
Demonstrates the integrated pipeline with real examples
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from backend.services.query_preprocessing_pipeline import get_preprocessing_pipeline
import time


def print_header(text: str):
    """Print styled header"""
    print(f"\n{'='*80}")
    print(f"  {text}")
    print(f"{'='*80}\n")


def print_result(result):
    """Print formatted result"""
    print(f"📝 Original Query:")
    print(f"   '{result.original_query}'")
    print(f"\n✨ Cleaned Query:")
    print(f"   '{result.cleaned_query}'")
    
    if result.cleaned_query != result.original_query:
        print(f"\n🔧 Changes Made:")
        if result.has_typos and result.typo_corrections:
            print(f"   • Typo corrections: {len(result.typo_corrections)}")
            for correction in result.typo_corrections:
                print(f"     - {correction}")
        
        if result.has_dialect and result.dialect_normalizations:
            print(f"   • Dialect normalizations: {len(result.dialect_normalizations)}")
            for norm in result.dialect_normalizations:
                print(f"     - {norm}")
    else:
        print(f"\n✅ No changes needed - query is clean!")
    
    if result.entities:
        print(f"\n🎯 Entities Extracted: {len(result.entities)}")
        for key, value in result.entities.items():
            if isinstance(value, list):
                value_str = ', '.join(str(v) for v in value)
            else:
                value_str = str(value)
            print(f"   • {key}: {value_str}")
    else:
        print(f"\n🎯 No entities extracted")
    
    print(f"\n⚡ Performance:")
    print(f"   • Typo correction:      {result.typo_correction_ms:.3f}ms")
    print(f"   • Dialect normalization: {result.dialect_normalization_ms:.3f}ms")
    print(f"   • Entity extraction:     {result.entity_extraction_ms:.3f}ms")
    print(f"   • Total processing:      {result.total_processing_ms:.3f}ms")
    print()


def demo_restaurant_queries():
    """Demo restaurant-related queries"""
    print_header("🍽️  RESTAURANT QUERIES")
    
    pipeline = get_preprocessing_pipeline()
    
    queries = [
        ("sultanahmed'te bi balik restoranı arıyorum", "restaurant"),
        ("kadikoy'de bugün akşam 4 kişilik ucuz italyan yemek", "restaurant"),
        ("beyoglu'nda şurda bi kebap lokantası var mı", "restaurant"),
    ]
    
    for i, (query, intent) in enumerate(queries, 1):
        print(f"Example {i}:")
        result = pipeline.process(query, intent)
        print_result(result)
        print("-" * 80)


def demo_transportation_queries():
    """Demo transportation-related queries"""
    print_header("🚇 TRANSPORTATION QUERIES")
    
    pipeline = get_preprocessing_pipeline()
    
    queries = [
        ("taksim'den kadikoy'e nası gidicem", "transportation"),
        ("sultanahmed'den ayasofia'ya metro ile gitmek istiyorum", "transportation"),
        ("besiktas'tan uskudar'a vapur var mı", "transportation"),
    ]
    
    for i, (query, intent) in enumerate(queries, 1):
        print(f"Example {i}:")
        result = pipeline.process(query, intent)
        print_result(result)
        print("-" * 80)


def demo_attraction_queries():
    """Demo attraction-related queries"""
    print_header("🏛️  ATTRACTION QUERIES")
    
    pipeline = get_preprocessing_pipeline()
    
    queries = [
        ("ayasofia'yı gezmek istiyorum", "attraction"),
        ("beyoglu'nda tarihi yerler var mı", "attraction"),
        ("kadikoy'de müze görmek istiyorum", "attraction"),
    ]
    
    for i, (query, intent) in enumerate(queries, 1):
        print(f"Example {i}:")
        result = pipeline.process(query, intent)
        print_result(result)
        print("-" * 80)


def demo_batch_processing():
    """Demo batch processing"""
    print_header("📦 BATCH PROCESSING")
    
    pipeline = get_preprocessing_pipeline()
    
    queries = [
        ("sultanahmed'te bi restoran", "restaurant"),
        ("taksim'den besiktas'a", "transportation"),
        ("kadikoy'de müze", "attraction"),
        ("beyoglu'nda ucuz otel", "accommodation"),
        ("galata kulesi nası gidicem", "transportation"),
    ]
    
    print(f"Processing {len(queries)} queries in batch...\n")
    
    start_time = time.time()
    results = pipeline.process_batch(queries)
    batch_time = (time.time() - start_time) * 1000
    
    for i, result in enumerate(results, 1):
        print(f"{i}. '{result.original_query}'")
        print(f"   → '{result.cleaned_query}'")
        print(f"   Entities: {len(result.entities)}, Time: {result.total_processing_ms:.3f}ms")
    
    print(f"\n⚡ Batch Performance:")
    print(f"   Total time: {batch_time:.3f}ms")
    print(f"   Avg per query: {batch_time/len(queries):.3f}ms")
    print()


def demo_statistics():
    """Demo statistics tracking"""
    print_header("📊 STATISTICS TRACKING")
    
    pipeline = get_preprocessing_pipeline()
    pipeline.reset_statistics()
    
    # Process various queries
    test_queries = [
        ("sultanahmed'te bi restoran", "restaurant"),
        ("kadikoy'de müze", "attraction"),
        ("nası gidicem", "general"),
        ("Taksim'de otel", "accommodation"),
        ("beyoglu'nda bişey", "general"),
        ("galata kulesi", "attraction"),
        ("ucuz balik yemek istiyorum", "restaurant"),
    ]
    
    print(f"Processing {len(test_queries)} queries to collect statistics...\n")
    
    for query, intent in test_queries:
        pipeline.process(query, intent)
    
    stats = pipeline.get_statistics()
    
    print("📈 Pipeline Statistics:")
    print(f"   • Total queries processed:       {stats['total_queries']}")
    print(f"   • Queries with typos:            {stats['queries_with_typos']} ({stats['typo_percentage']:.1f}%)")
    print(f"   • Queries with dialect:          {stats['queries_with_dialect']} ({stats['dialect_percentage']:.1f}%)")
    print(f"   • Total typo corrections:        {stats['total_typo_corrections']}")
    print(f"   • Total dialect normalizations:  {stats['total_dialect_normalizations']}")
    print(f"   • Avg corrections per query:     {stats['avg_corrections_per_query']:.2f}")
    print(f"   • Avg normalizations per query:  {stats['avg_normalizations_per_query']:.2f}")
    print(f"   • Avg processing time:           {stats['avg_processing_time_ms']:.3f}ms")
    print()


def interactive_demo():
    """Interactive demo - user can enter queries"""
    print_header("🎮 INTERACTIVE MODE")
    
    pipeline = get_preprocessing_pipeline()
    
    print("Enter your queries below (type 'quit' to exit, 'stats' to see statistics):")
    print("Format: query | intent (e.g., 'sultanahmed'te restoran | restaurant')")
    print()
    
    while True:
        try:
            user_input = input("Your query: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break
            
            if user_input.lower() == 'stats':
                stats = pipeline.get_statistics()
                print("\n📊 Current Statistics:")
                print(f"   Total queries: {stats['total_queries']}")
                print(f"   Typo rate: {stats['typo_percentage']:.1f}%")
                print(f"   Dialect rate: {stats['dialect_percentage']:.1f}%")
                print(f"   Avg time: {stats['avg_processing_time_ms']:.3f}ms\n")
                continue
            
            # Parse input
            if '|' in user_input:
                query, intent = user_input.split('|', 1)
                query = query.strip()
                intent = intent.strip()
            else:
                query = user_input
                intent = 'general'
            
            # Process query
            result = pipeline.process(query, intent)
            print()
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main demo runner"""
    print("\n" + "="*80)
    print("  🚀 QUERY PREPROCESSING PIPELINE - INTERACTIVE DEMO")
    print("="*80)
    
    print("\nThis demo showcases the integrated query preprocessing pipeline:")
    print("  1. Turkish Typo Correction")
    print("  2. Turkish Dialect Normalization")
    print("  3. Entity Extraction")
    print()
    
    # Run demos
    try:
        demo_restaurant_queries()
        input("Press Enter to continue...")
        
        demo_transportation_queries()
        input("Press Enter to continue...")
        
        demo_attraction_queries()
        input("Press Enter to continue...")
        
        demo_batch_processing()
        input("Press Enter to continue...")
        
        demo_statistics()
        input("Press Enter to start interactive mode...")
        
        interactive_demo()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye! 👋")
    
    print("\n" + "="*80)
    print("  ✅ Demo Complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

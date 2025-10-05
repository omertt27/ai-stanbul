#!/usr/bin/env python3
"""
Complete Istanbul Tourism Data Pipeline Demo
Week 1-2 Implementation - Full pipeline from collection to training data
"""

import asyncio
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any

# Setup project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import components
from simple_collector import SimpleIstanbulCollector
from data_curator import DataCurator
from training_data_formatter import TrainingDataFormatter
from config import DataPipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_complete_pipeline():
    """Run the complete data pipeline"""
    
    print("=" * 80)
    print("🚀 ISTANBUL TOURISM DATA PIPELINE - COMPLETE DEMO")
    print("=" * 80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = DataPipelineConfig()
    
    # Phase 1: Data Collection
    print("\n📡 PHASE 1: DATA COLLECTION")
    print("-" * 40)
    
    collector = SimpleIstanbulCollector()
    collection_summary = await collector.collect_all_data()
    
    print(f"✅ Collected {collection_summary['total_records']} sample records")
    
    # Phase 2: Data Curation
    print("\n🔍 PHASE 2: DATA CURATION")
    print("-" * 40)
    
    curator = DataCurator(config)
    
    # Use only the main sample file to avoid duplicates
    main_data_file = None
    for file_path in collection_summary['data_files_created']:
        if 'sample_data' in file_path:
            main_data_file = file_path
            break
    
    if not main_data_file:
        print("❌ Main data file not found")
        return
    
    curation_summary = await curator.curate_dataset([main_data_file])
    
    curated_count = curation_summary['metrics']['valid_output_records']
    quality_score = curation_summary['metrics']['overall_quality_score']
    retention_rate = curation_summary['metrics']['retention_rate']
    
    print(f"✅ Curated {curated_count} records")
    print(f"📊 Quality score: {quality_score:.2f}")
    print(f"📈 Retention rate: {retention_rate:.2f}")
    
    # Phase 3: Training Data Preparation
    print("\n🎯 PHASE 3: TRAINING DATA PREPARATION")
    print("-" * 40)
    
    # Find the curated data file
    validated_dir = Path(config.VALIDATED_DATA_DIR)
    curated_files = list(validated_dir.glob('curated_istanbul_data_*.json'))
    
    if not curated_files:
        print("❌ No curated files found")
        return
    
    latest_file = max(curated_files, key=lambda x: x.stat().st_mtime)
    
    # Format the data for training
    formatter = TrainingDataFormatter(config.TRAINING_DATA_DIR)
    training_result = formatter.format_for_training(latest_file)
    
    total_training_examples = sum(training_result.values())
    print(f"✅ Generated {total_training_examples} training examples")
    print(f"   • Q&A pairs: {training_result.get('qa_pairs', 0)}")
    print(f"   • Conversations: {training_result.get('conversations', 0)}")
    print(f"   • Instructions: {training_result.get('instructions', 0)}")
    
    # Generate final summary
    print("\n" + "=" * 80)
    print("📊 PIPELINE COMPLETION SUMMARY")
    print("=" * 80)
    
    print(f"🔢 Total Records Processed:")
    print(f"   Raw collected: {collection_summary['total_records']}")
    print(f"   Successfully curated: {curated_count}")
    print(f"   Training examples generated: {total_training_examples}")
    
    print(f"\n📂 Output Directories:")
    print(f"   Raw data: {config.RAW_DATA_DIR}")
    print(f"   Curated data: {config.VALIDATED_DATA_DIR}")
    print(f"   Training data: {config.TRAINING_DATA_DIR}")
    
    print(f"\n🎯 Data Categories:")
    for category, count in curation_summary.get('category_distribution', {}).items():
        print(f"   {category}: {count} records")
    
    print(f"\n⚡ Performance Metrics:")
    amplification_factor = total_training_examples / collection_summary['total_records'] if collection_summary['total_records'] > 0 else 0
    print(f"   Data amplification: {amplification_factor:.1f}x")
    print(f"   Quality retention: {retention_rate:.1%}")
    print(f"   Content quality: {quality_score:.2f}/1.0")
    
    print(f"\n🚀 Next Steps:")
    print("   1. Review generated training data quality")
    print("   2. Set up model training environment")
    print("   3. Configure small LLM architecture (100-200M params)")
    print("   4. Begin model training/distillation")
    print("   5. Implement inference pipeline")
    print("   6. Deploy production system")
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return {
        'collection_summary': collection_summary,
        'curation_summary': curation_summary,
        'training_result': training_result,
        'total_examples': total_training_examples,
        'pipeline_success': True
    }

async def main():
    """Main execution"""
    try:
        result = await run_complete_pipeline()
        
        if result and result['pipeline_success']:
            print(f"\n🎉 Success! Generated {result['total_examples']} training examples")
            print("Ready to begin model training phase.")
        else:
            print("\n❌ Pipeline failed to complete")
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        print(f"\n❌ Pipeline error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

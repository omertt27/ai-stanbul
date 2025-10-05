#!/usr/bin/env python3
"""
Istanbul Tourism Data Pipeline Runner
Week 1-2 Implementation - Complete data collection and curation pipeline
"""

import asyncio
import sys
import argparse
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Setup project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import DataPipelineConfig
from istanbul_data_pipeline import IstanbulDataPipeline
from data_curator import DataCurator
from training_data_formatter import TrainingDataFormatter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        self.results = {}
        
    async def run_collection_phase(self) -> Dict[str, Any]:
        """Run data collection phase"""
        logger.info("=" * 60)
        logger.info("PHASE 1: DATA COLLECTION")
        logger.info("=" * 60)
        
        pipeline = IstanbulDataPipeline()
        collection_summary = await pipeline.collect_all_data()
            
        self.results['collection'] = collection_summary
        return collection_summary
    
    async def run_curation_phase(self) -> Dict[str, Any]:
        """Run data curation phase"""
        logger.info("=" * 60)
        logger.info("PHASE 2: DATA CURATION")
        logger.info("=" * 60)
        
        curator = DataCurator(self.config)
        
        # Find all raw data files
        raw_data_dir = Path(self.config.RAW_DATA_DIR)
        input_files = []
        for pattern in ['*.json', '*.csv']:
            input_files.extend(raw_data_dir.glob(pattern))
        
        if not input_files:
            logger.warning("No raw data files found for curation")
            return {}
        
        curation_summary = await curator.curate_dataset([str(f) for f in input_files])
        
        self.results['curation'] = curation_summary
        return curation_summary
    
    async def run_training_prep_phase(self) -> Dict[str, Any]:
        """Run training data preparation phase"""
        logger.info("=" * 60)
        logger.info("PHASE 3: TRAINING DATA PREPARATION")
        logger.info("=" * 60)
        
        formatter = TrainingDataFormatter(self.config)
        
        # Find most recent curated data file
        validated_dir = Path(self.config.VALIDATED_DATA_DIR)
        curated_files = list(validated_dir.glob("curated_istanbul_data_*.json"))
        
        if not curated_files:
            logger.error("No curated data files found")
            return {}
        
        latest_file = max(curated_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using curated data file: {latest_file}")
        
        prep_summary = await formatter.prepare_training_data(str(latest_file))
        
        self.results['training_prep'] = prep_summary
        return prep_summary
    
    async def run_full_pipeline(self, skip_collection: bool = False, skip_curation: bool = False) -> Dict[str, Any]:
        """Run the complete data pipeline"""
        logger.info("Starting Istanbul Tourism Data Pipeline")
        logger.info(f"Pipeline start time: {datetime.now()}")
        
        pipeline_start = datetime.now()
        
        try:
            # Phase 1: Data Collection
            if not skip_collection:
                collection_summary = await self.run_collection_phase()
                logger.info(f"Collection completed: {collection_summary['total_records']} records")
            else:
                logger.info("Skipping data collection phase")
            
            # Phase 2: Data Curation
            if not skip_curation:
                curation_summary = await self.run_curation_phase()
                if curation_summary:
                    logger.info(f"Curation completed: {curation_summary['metrics']['valid_output_records']} valid records")
                else:
                    logger.warning("Curation phase returned no results")
            else:
                logger.info("Skipping data curation phase")
            
            # Phase 3: Training Data Preparation
            prep_summary = await self.run_training_prep_phase()
            if prep_summary:
                logger.info(f"Training prep completed: {prep_summary['total_training_examples']} training examples")
            else:
                logger.warning("Training preparation phase returned no results")
            
            # Generate final pipeline summary
            pipeline_end = datetime.now()
            pipeline_duration = pipeline_end - pipeline_start
            
            final_summary = {
                'pipeline_start': pipeline_start.isoformat(),
                'pipeline_end': pipeline_end.isoformat(),
                'total_duration_seconds': pipeline_duration.total_seconds(),
                'phases_completed': list(self.results.keys()),
                'collection_summary': self.results.get('collection', {}),
                'curation_summary': self.results.get('curation', {}),
                'training_prep_summary': self.results.get('training_prep', {}),
                'final_statistics': self._generate_final_statistics(),
                'recommendations': self._generate_final_recommendations(),
                'next_steps': [
                    'Review generated training data quality',
                    'Set up model training environment',
                    'Configure model architecture (100-200M params)',
                    'Begin model pretraining/distillation',
                    'Implement evaluation metrics',
                    'Set up production inference pipeline'
                ]
            }
            
            # Save final summary
            summary_path = Path(self.config.TRAINING_DATA_DIR) / f"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def _generate_final_statistics(self) -> Dict[str, Any]:
        """Generate final pipeline statistics"""
        stats = {}
        
        if 'collection' in self.results:
            stats['total_raw_records'] = self.results['collection'].get('total_records', 0)
            stats['collection_sources'] = self.results['collection'].get('successful_sources', 0)
        
        if 'curation' in self.results:
            curation_metrics = self.results['curation'].get('metrics', {})
            stats['curated_records'] = curation_metrics.get('valid_output_records', 0)
            stats['data_quality_score'] = curation_metrics.get('overall_quality_score', 0)
            stats['retention_rate'] = curation_metrics.get('retention_rate', 0)
        
        if 'training_prep' in self.results:
            stats['training_examples'] = self.results['training_prep'].get('total_training_examples', 0)
            stats['training_formats'] = len(self.results['training_prep'].get('formats_generated', []))
        
        return stats
    
    def _generate_final_recommendations(self) -> List[str]:
        """Generate final recommendations based on pipeline results"""
        recommendations = []
        
        # Check data quantity
        stats = self._generate_final_statistics()
        training_examples = stats.get('training_examples', 0)
        
        if training_examples < 5000:
            recommendations.append("Consider collecting more data - current dataset may be small for effective training")
        
        if training_examples > 50000:
            recommendations.append("Large dataset available - consider using advanced training techniques like curriculum learning")
        
        # Check data quality
        quality_score = stats.get('data_quality_score', 0)
        if quality_score < 0.7:
            recommendations.append("Data quality could be improved - review collection sources and curation rules")
        
        retention_rate = stats.get('retention_rate', 0)
        if retention_rate < 0.5:
            recommendations.append("Low data retention rate - consider adjusting quality thresholds or improving data sources")
        
        # General recommendations
        recommendations.extend([
            "Set up continuous data collection pipeline for ongoing model improvement",
            "Implement A/B testing framework for model evaluation",
            "Consider multilingual support (Turkish/English) for broader user base",
            "Plan for seasonal data collection to capture tourism variations"
        ])
        
        return recommendations
    
    def print_summary_report(self, summary: Dict[str, Any]):
        """Print comprehensive pipeline summary report"""
        print("\n" + "=" * 80)
        print("ISTANBUL TOURISM DATA PIPELINE - FINAL REPORT")
        print("=" * 80)
        
        print(f"Pipeline Duration: {summary['total_duration_seconds']:.2f} seconds")
        print(f"Phases Completed: {', '.join(summary['phases_completed'])}")
        
        stats = summary['final_statistics']
        print(f"\nFINAL STATISTICS:")
        print(f"  Raw Records Collected: {stats.get('total_raw_records', 'N/A')}")
        print(f"  Curated Records: {stats.get('curated_records', 'N/A')}")
        print(f"  Training Examples: {stats.get('training_examples', 'N/A')}")
        print(f"  Data Quality Score: {stats.get('data_quality_score', 'N/A'):.2f}")
        print(f"  Data Retention Rate: {stats.get('retention_rate', 'N/A'):.2f}")
        
        if 'collection_summary' in summary and summary['collection_summary']:
            print(f"\nCOLLECTION DETAILS:")
            collection = summary['collection_summary']
            for source, count in collection.get('records_by_source', {}).items():
                print(f"  {source}: {count} records")
        
        if 'curation_summary' in summary and summary['curation_summary']:
            print(f"\nCURATION DETAILS:")
            curation = summary['curation_summary']
            for category, count in curation.get('category_distribution', {}).items():
                print(f"  {category}: {count} records")
        
        if 'training_prep_summary' in summary and summary['training_prep_summary']:
            print(f"\nTRAINING DATA DETAILS:")
            training = summary['training_prep_summary']
            for format_type in training.get('formats_generated', []):
                print(f"  {format_type}: Available")
            
            splits = training.get('split_sizes', {})
            print(f"  Train: {splits.get('train', 0)} examples")
            print(f"  Validation: {splits.get('validation', 0)} examples")
            print(f"  Test: {splits.get('test', 0)} examples")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
        
        print(f"\nNEXT STEPS:")
        for step in summary['next_steps']:
            print(f"  - {step}")
        
        print(f"\nOUTPUT DIRECTORIES:")
        print(f"  Raw Data: {self.config.RAW_DATA_DIR}")
        print(f"  Curated Data: {self.config.VALIDATED_DATA_DIR}")
        print(f"  Training Data: {self.config.TRAINING_DATA_DIR}")
        
        print("=" * 80)

async def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Istanbul Tourism Data Pipeline')
    parser.add_argument('--skip-collection', action='store_true', 
                       help='Skip data collection phase')
    parser.add_argument('--skip-curation', action='store_true',
                       help='Skip data curation phase')
    parser.add_argument('--collection-only', action='store_true',
                       help='Run only data collection phase')
    parser.add_argument('--curation-only', action='store_true',
                       help='Run only data curation phase')
    parser.add_argument('--training-only', action='store_true',
                       help='Run only training data preparation phase')
    
    args = parser.parse_args()
    
    # Initialize configuration and runner
    config = DataPipelineConfig()
    runner = PipelineRunner(config)
    
    try:
        if args.collection_only:
            summary = await runner.run_collection_phase()
            print(f"Collection completed: {summary['total_records']} records")
            
        elif args.curation_only:
            summary = await runner.run_curation_phase()
            if summary:
                print(f"Curation completed: {summary['metrics']['valid_output_records']} valid records")
            
        elif args.training_only:
            summary = await runner.run_training_prep_phase()
            if summary:
                print(f"Training prep completed: {summary['total_training_examples']} examples")
        
        else:
            # Run full pipeline
            summary = await runner.run_full_pipeline(
                skip_collection=args.skip_collection,
                skip_curation=args.skip_curation
            )
            runner.print_summary_report(summary)
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Monthly automated retraining script
Run this via cron job on the first day of each month
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from user_feedback_collection_system import get_feedback_collector

# Setup logging
log_dir = Path(__file__).parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'monthly_retrain.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def check_retrain_needed(min_corrections=50, min_misclassifications=20):
    """
    Check if retraining is needed based on feedback statistics
    
    Args:
        min_corrections: Minimum number of corrections to trigger retraining
        min_misclassifications: Minimum number of misclassifications
        
    Returns:
        tuple: (bool: is_needed, dict: stats)
    """
    logger.info("Checking if retraining is needed...")
    
    collector = get_feedback_collector()
    stats = collector.get_feedback_summary(days=30)
    
    total_corrections = len(stats.get('misclassifications', []))
    low_confidence = len(stats.get('low_confidence_predictions', []))
    
    logger.info(f"Total corrections: {total_corrections}")
    logger.info(f"Low confidence predictions: {low_confidence}")
    logger.info(f"Total feedback: {stats['total']}")
    
    # Check if retraining criteria are met
    needs_retrain = (
        total_corrections >= min_corrections or
        (total_corrections >= min_misclassifications and low_confidence >= 30)
    )
    
    return needs_retrain, stats


def generate_retraining_data(min_corrections_per_intent=5):
    """
    Generate retraining data from feedback
    
    Args:
        min_corrections_per_intent: Minimum corrections per intent to include
        
    Returns:
        tuple: (int: count, str: filepath)
    """
    logger.info("Generating retraining data...")
    
    collector = get_feedback_collector()
    count, filepath = collector.generate_retraining_data(
        min_corrections=min_corrections_per_intent
    )
    
    logger.info(f"Generated {count} retraining examples in {filepath}")
    return count, filepath


def merge_datasets():
    """
    Merge English expansion and feedback retraining data
    
    Returns:
        tuple: (int: total_count, str: output_path)
    """
    logger.info("Merging datasets...")
    
    base_dir = Path(__file__).parent.parent
    english_file = base_dir / 'english_expanded_training_data.json'
    retraining_file = base_dir / 'data' / 'retraining_data.json'
    output_file = base_dir / 'final_training_data_merged.json'
    
    if not english_file.exists():
        logger.error(f"English expansion file not found: {english_file}")
        return 0, None
    
    if not retraining_file.exists():
        logger.warning(f"Retraining file not found: {retraining_file}")
        # Use only English data
        with open(english_file, 'r', encoding='utf-8') as f:
            english_data = json.load(f)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(english_data, f, indent=2, ensure_ascii=False)
        
        total = len(english_data['training_data'])
        logger.info(f"Merged {total} examples (English only)")
        return total, str(output_file)
    
    # Load both datasets
    with open(english_file, 'r', encoding='utf-8') as f:
        english_data = json.load(f)
    
    with open(retraining_file, 'r', encoding='utf-8') as f:
        retraining_data = json.load(f)
    
    # Merge training data
    merged_training = english_data['training_data'].copy()
    
    # Add retraining examples (avoid duplicates)
    existing_queries = {item['query'].lower() for item in merged_training}
    new_count = 0
    
    for item in retraining_data['training_data']:
        if item['query'].lower() not in existing_queries:
            merged_training.append(item)
            existing_queries.add(item['query'].lower())
            new_count += 1
    
    # Create merged dataset
    merged_data = {
        'metadata': {
            'description': 'Merged training data with English expansion and user feedback',
            'creation_date': datetime.now().isoformat(),
            'english_examples': len(english_data['training_data']),
            'feedback_examples': new_count,
            'total_examples': len(merged_training)
        },
        'training_data': merged_training
    }
    
    # Save merged data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Merged {len(merged_training)} total examples")
    logger.info(f"  - English: {len(english_data['training_data'])}")
    logger.info(f"  - New feedback: {new_count}")
    
    return len(merged_training), str(output_file)


def check_data_quality(filepath):
    """
    Check training data quality before retraining
    
    Args:
        filepath: Path to training data file
        
    Returns:
        bool: True if quality checks pass
    """
    logger.info("Checking data quality...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    training_data = data['training_data']
    
    # Check for duplicates
    queries = [item['query'] for item in training_data]
    duplicates = len(queries) - len(set([q.lower() for q in queries]))
    
    # Check intent distribution
    intent_counts = {}
    for item in training_data:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    # Check for imbalanced intents
    if intent_counts:
        max_count = max(intent_counts.values())
        min_count = min(intent_counts.values())
        imbalance_ratio = max_count / min_count
    else:
        imbalance_ratio = 1.0
    
    logger.info(f"Total examples: {len(training_data)}")
    logger.info(f"Duplicates: {duplicates}")
    logger.info(f"Unique intents: {len(intent_counts)}")
    logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    # Show top 10 intents
    top_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 intents:")
    for intent, count in top_intents:
        logger.info(f"  {intent}: {count}")
    
    # Quality checks
    quality_ok = True
    
    if duplicates > 10:
        logger.warning(f"⚠️  WARNING: {duplicates} duplicates found!")
        quality_ok = False
    
    if imbalance_ratio > 15:
        logger.warning(f"⚠️  WARNING: High class imbalance (ratio: {imbalance_ratio:.2f})!")
        # Don't fail on imbalance, just warn
    
    if len(training_data) < 100:
        logger.warning("⚠️  WARNING: Very few training examples!")
        quality_ok = False
    
    return quality_ok


def save_retrain_report(stats, retrain_count, output_path):
    """
    Save retraining report for record-keeping
    
    Args:
        stats: Feedback statistics
        retrain_count: Number of examples used for retraining
        output_path: Path to merged training data
    """
    report_dir = Path(__file__).parent.parent / 'reports'
    report_dir.mkdir(exist_ok=True)
    
    report_path = report_dir / f'retrain_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    report = {
        'date': datetime.now().isoformat(),
        'feedback_stats': {
            'total_feedback': stats['total'],
            'misclassifications': len(stats.get('misclassifications', [])),
            'low_confidence': len(stats.get('low_confidence_predictions', [])),
            'by_function': dict(stats['by_function']),
            'by_language': dict(stats['by_language'])
        },
        'retraining': {
            'examples_generated': retrain_count,
            'training_data_path': output_path
        }
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved retraining report to {report_path}")


def main():
    """
    Main monthly retraining process
    """
    try:
        logger.info("=" * 60)
        logger.info("🤖 MONTHLY RETRAINING PROCESS STARTED")
        logger.info("=" * 60)
        
        # Step 1: Check if retraining is needed
        needs_retrain, stats = check_retrain_needed(
            min_corrections=50,
            min_misclassifications=20
        )
        
        if not needs_retrain:
            logger.info("ℹ️  Not enough feedback for retraining. Exiting.")
            logger.info("Criteria not met:")
            logger.info("  - Need at least 50 corrections OR")
            logger.info("  - Need at least 20 misclassifications + 30 low confidence")
            return 0
        
        logger.info("✅ Retraining criteria met. Proceeding...")
        
        # Step 2: Generate retraining data
        retrain_count, retrain_path = generate_retraining_data(
            min_corrections_per_intent=5
        )
        
        if retrain_count == 0:
            logger.warning("⚠️  No retraining data generated. Exiting.")
            return 0
        
        # Step 3: Merge datasets
        total_count, merged_path = merge_datasets()
        
        if not merged_path:
            logger.error("❌ Failed to merge datasets. Exiting.")
            return 1
        
        # Step 4: Check data quality
        quality_ok = check_data_quality(merged_path)
        
        if not quality_ok:
            logger.error("❌ Data quality check failed. Manual review required.")
            return 1
        
        logger.info("✅ Data quality check passed")
        
        # Step 5: Save retraining report
        save_retrain_report(stats, retrain_count, merged_path)
        
        logger.info("=" * 60)
        logger.info("✅ RETRAINING DATA READY!")
        logger.info("=" * 60)
        logger.info(f"Training data: {merged_path}")
        logger.info(f"Total examples: {total_count}")
        logger.info(f"New feedback examples: {retrain_count}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Review the merged training data")
        logger.info("2. Run: python train_intent_classifier.py \\")
        logger.info("     --data-file final_training_data_merged.json \\")
        logger.info("     --epochs 20 \\")
        logger.info("     --batch-size 16")
        logger.info("3. Test the new model")
        logger.info("4. Deploy to production")
        logger.info("")
        logger.info("💡 TIP: You can automate training by extending this script")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error during retraining process: {str(e)}")
        logger.exception(e)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

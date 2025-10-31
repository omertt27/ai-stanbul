#!/usr/bin/env python3
"""
Evaluate Fine-tuned Models for Istanbul AI
Compares base model vs fine-tuned model performance
"""

import sys
import os
import json
import torch
from pathlib import Path
from typing import Dict, List
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(test_file: str = "data/intent_training_data_augmented.json", 
                   holdout_size: int = 200) -> List[Dict]:
    """
    Load test data from training file (using holdout set)
    
    Args:
        test_file: Path to training data file
        holdout_size: Number of examples to use for testing
        
    Returns:
        List of test examples
    """
    logger.info(f"Loading test data from: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    # Use last N examples as test set (not used in training)
    test_data = all_data[-holdout_size:]
    
    logger.info(f"✅ Loaded {len(test_data)} test examples")
    return test_data


def evaluate_model(model_type: str, test_data: List[Dict]) -> Dict:
    """
    Evaluate a model on test data
    
    Args:
        model_type: 'base' or 'finetuned'
        test_data: List of test examples
        
    Returns:
        Dictionary with evaluation metrics
    """
    from neural_query_classifier import NeuralQueryClassifier
    
    # Load appropriate model
    use_finetuned = (model_type == 'finetuned')
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluating {model_type.upper()} Model")
    logger.info(f"{'='*70}\n")
    
    try:
        classifier = NeuralQueryClassifier(
            model_path='models/distilbert_intent_classifier',
            use_finetuned=use_finetuned,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            confidence_threshold=0.70,
            enable_logging=False
        )
    except Exception as e:
        logger.error(f"❌ Failed to load {model_type} model: {e}")
        return {
            'error': str(e),
            'accuracy': 0.0,
            'total': 0
        }
    
    # Run predictions
    correct = 0
    total = 0
    high_confidence = 0
    low_confidence = 0
    total_confidence = 0.0
    latencies = []
    
    intent_correct = {}
    intent_total = {}
    
    for i, item in enumerate(test_data):
        text = item['text']
        true_intent = item['intent']
        
        try:
            import time
            start = time.time()
            predicted_intent, confidence = classifier.predict(text)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
            
            # Update statistics
            total += 1
            total_confidence += confidence
            
            if confidence >= 0.75:
                high_confidence += 1
            else:
                low_confidence += 1
            
            # Check correctness
            if predicted_intent == true_intent:
                correct += 1
                intent_correct[true_intent] = intent_correct.get(true_intent, 0) + 1
            
            intent_total[true_intent] = intent_total.get(true_intent, 0) + 1
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                accuracy_so_far = (correct / total) * 100
                logger.info(f"   Progress: {i+1}/{len(test_data)} - Accuracy: {accuracy_so_far:.1f}%")
                
        except Exception as e:
            logger.warning(f"Error processing example {i}: {e}")
            continue
    
    # Calculate metrics
    accuracy = (correct / total) * 100 if total > 0 else 0
    avg_confidence = total_confidence / total if total > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    # Per-intent accuracy
    intent_accuracies = {}
    for intent in intent_total:
        intent_acc = (intent_correct.get(intent, 0) / intent_total[intent]) * 100
        intent_accuracies[intent] = intent_acc
    
    results = {
        'model_type': model_type,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_confidence': avg_confidence,
        'high_confidence_count': high_confidence,
        'low_confidence_count': low_confidence,
        'avg_latency_ms': avg_latency,
        'intent_accuracies': intent_accuracies
    }
    
    return results


def print_results(base_results: Dict, finetuned_results: Dict):
    """Print comparison results"""
    
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Base Model':<20} {'Fine-tuned Model':<20} {'Improvement':<15}")
    print("-" * 85)
    
    # Accuracy
    base_acc = base_results.get('accuracy', 0)
    fine_acc = finetuned_results.get('accuracy', 0)
    improvement = fine_acc - base_acc
    print(f"{'Overall Accuracy':<30} {base_acc:>18.2f}% {fine_acc:>18.2f}% {improvement:>13.2f}%")
    
    # Confidence
    base_conf = base_results.get('avg_confidence', 0)
    fine_conf = finetuned_results.get('avg_confidence', 0)
    conf_improvement = fine_conf - base_conf
    print(f"{'Average Confidence':<30} {base_conf:>20.3f} {fine_conf:>20.3f} {conf_improvement:>+14.3f}")
    
    # Latency
    base_lat = base_results.get('avg_latency_ms', 0)
    fine_lat = finetuned_results.get('avg_latency_ms', 0)
    lat_diff = fine_lat - base_lat
    print(f"{'Average Latency (ms)':<30} {base_lat:>20.1f} {fine_lat:>20.1f} {lat_diff:>+14.1f}")
    
    # High confidence predictions
    base_high = base_results.get('high_confidence_count', 0)
    fine_high = finetuned_results.get('high_confidence_count', 0)
    print(f"{'High Confidence (>75%)':<30} {base_high:>20} {fine_high:>20} {fine_high - base_high:>+14}")
    
    print("\n" + "="*70)
    print("📈 PERFORMANCE SUMMARY")
    print("="*70)
    
    if improvement > 0:
        print(f"✅ Fine-tuned model is BETTER by {improvement:.2f}%")
    elif improvement < 0:
        print(f"⚠️  Fine-tuned model is WORSE by {abs(improvement):.2f}%")
    else:
        print(f"➖ Models perform equally")
    
    if conf_improvement > 0:
        print(f"✅ Fine-tuned model has {conf_improvement:.1%} higher confidence")
    
    print()


def save_results(base_results: Dict, finetuned_results: Dict, output_file: str = "reports/model_evaluation_results.json"):
    """Save evaluation results to JSON file"""
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    results = {
        'base_model': base_results,
        'finetuned_model': finetuned_results,
        'comparison': {
            'accuracy_improvement': finetuned_results.get('accuracy', 0) - base_results.get('accuracy', 0),
            'confidence_improvement': finetuned_results.get('avg_confidence', 0) - base_results.get('avg_confidence', 0),
            'latency_difference': finetuned_results.get('avg_latency_ms', 0) - base_results.get('avg_latency_ms', 0)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Results saved to: {output_file}")


def main():
    """Main evaluation function"""
    
    print("\n" + "="*70)
    print("🎓 FINE-TUNED MODEL EVALUATION")
    print("="*70)
    
    # Load test data
    try:
        test_data = load_test_data()
    except Exception as e:
        logger.error(f"❌ Failed to load test data: {e}")
        return 1
    
    # Evaluate base model
    logger.info("\n📚 Evaluating BASE model...")
    base_results = evaluate_model('base', test_data)
    
    if 'error' in base_results:
        logger.error("❌ Base model evaluation failed")
        logger.info("   This might be expected if the base model doesn't exist")
        logger.info("   Continuing with fine-tuned model evaluation only...")
        base_results = {
            'accuracy': 0,
            'avg_confidence': 0,
            'avg_latency_ms': 0,
            'high_confidence_count': 0,
            'total': len(test_data)
        }
    
    # Evaluate fine-tuned model
    logger.info("\n🎓 Evaluating FINE-TUNED model...")
    finetuned_results = evaluate_model('finetuned', test_data)
    
    if 'error' in finetuned_results:
        logger.error("❌ Fine-tuned model evaluation failed")
        logger.error("   Make sure fine-tuning has completed successfully")
        logger.error("   Run: python scripts/quick_finetune.py")
        return 1
    
    # Print comparison
    print_results(base_results, finetuned_results)
    
    # Save results
    save_results(base_results, finetuned_results)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📊 View detailed results: reports/model_evaluation_results.json")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

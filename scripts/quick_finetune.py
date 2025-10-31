#!/usr/bin/env python3
"""
Quick Fine-tuning Script for Istanbul AI
Automates the fine-tuning process with sensible defaults
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(
        description='Quick fine-tuning for Istanbul AI models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tuning (uses defaults)
  python scripts/quick_finetune.py

  # Fine-tuning with custom epochs
  python scripts/quick_finetune.py --epochs 10

  # Fine-tuning with GPU-optimized settings
  python scripts/quick_finetune.py --gpu --batch-size 32

  # Fast fine-tuning for testing
  python scripts/quick_finetune.py --fast
        """
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='data/intent_training_data_augmented.json',
        help='Path to training data JSON file (default: data/intent_training_data_augmented.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/istanbul_intent_classifier_finetuned',
        help='Output directory for fine-tuned model (default: models/istanbul_intent_classifier_finetuned)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Number of training epochs (default: 15)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Training batch size (default: 16, increase for GPU)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-5,
        help='Learning rate (default: 2e-5)'
    )
    
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.15,
        help='Validation split fraction (default: 0.15)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU-optimized settings (larger batch size, etc.)'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode for testing (fewer epochs, smaller batch)'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate model after fine-tuning'
    )
    
    args = parser.parse_args()
    
    # Override settings for fast mode
    if args.fast:
        print("⚡ Fast mode enabled - using reduced settings for quick testing")
        args.epochs = 3
        args.batch_size = 8
    
    # Override settings for GPU mode
    if args.gpu:
        print("🚀 GPU mode enabled - using optimized settings for GPU training")
        args.batch_size = 32
    
    # Check if data file exists
    if not Path(args.data_file).exists():
        print(f"❌ Error: Training data file not found: {args.data_file}")
        print("   Please ensure the file exists or specify a different path with --data-file")
        return 1
    
    # Import and run fine-tuning
    try:
        from finetune_intent_classifier import finetune_intent_classifier
        
        print("\n" + "="*70)
        print("🚀 STARTING ISTANBUL AI FINE-TUNING")
        print("="*70)
        print(f"\n📋 Configuration:")
        print(f"   Data file: {args.data_file}")
        print(f"   Output dir: {args.output_dir}")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.learning_rate}")
        print(f"   Validation split: {args.validation_split}")
        print()
        
        # Run fine-tuning
        model_path = finetune_intent_classifier(
            data_file=args.data_file,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_split=args.validation_split
        )
        
        print("\n" + "="*70)
        print("✅ FINE-TUNING COMPLETE!")
        print("="*70)
        print(f"\n📁 Model saved to: {model_path}")
        print("\n🎯 Next steps:")
        print("   1. Update neural_query_classifier.py to use fine-tuned model")
        print("   2. Update main_system.py to enable use_finetuned=True")
        print("   3. Run evaluation: python scripts/evaluate_finetuned_models.py")
        print("   4. Create A/B test to compare models in production")
        print()
        
        # Optional evaluation
        if args.evaluate:
            print("\n📊 Running evaluation...")
            try:
                from evaluate_finetuned_models import evaluate_intent_classifier
                evaluate_intent_classifier()
            except ImportError:
                print("⚠️  Evaluation script not found. Skipping evaluation.")
            except Exception as e:
                print(f"⚠️  Evaluation failed: {e}")
        
        return 0
        
    except ImportError as e:
        print(f"\n❌ Error: Could not import fine-tuning script: {e}")
        print("   Make sure scripts/finetune_intent_classifier.py exists")
        return 1
    except Exception as e:
        print(f"\n❌ Error during fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

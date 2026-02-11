#!/usr/bin/env python3
"""
Simple NCF Training and ONNX Export

Uses the existing test_ncf_model.py logic to train and export.
"""

import sys
import logging
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from backend.ml.deep_learning.models.ncf import NCF
from backend.ml.deep_learning.data_pipeline import DataPipeline


def generate_sample_data(num_users=100, num_items=200, num_interactions=1000):
    """Generate sample interaction data for training."""
    logger.info(f"Generating sample data: {num_users} users, {num_items} items, {num_interactions} interactions")
    
    feedback_data = []
    for _ in range(num_interactions):
        user_id = f"user_{np.random.randint(0, num_users)}"
        item_id = f"item_{np.random.randint(0, num_items)}"
        feedback = np.random.choice(['positive', 'negative'], p=[0.7, 0.3])
        
        feedback_data.append({
            'user_id': user_id,
            'item_id': item_id,
            'feedback': feedback
        })
    
    return feedback_data


def main():
    print("=" * 80)
    print("🚀 NCF Model Training and ONNX Export")
    print("=" * 80)
    
    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating sample interaction data...")
    feedback_data = generate_sample_data(num_users=100, num_items=200, num_interactions=1000)
    
    # Step 2: Load data pipeline
    print("\n🔄 Step 2: Loading data pipeline...")
    pipeline = DataPipeline()
    user_ids, item_ids, labels = pipeline.load_from_feedback(feedback_data)
    
    # Split data
    train_data, val_data, test_data = pipeline.split_data(
        user_ids, item_ids, labels, strategy="random"
    )
    
    # Generate negative samples
    train_data_with_negatives = pipeline.generate_negative_samples(*train_data)
    
    print(f"✅ Data prepared:")
    print(f"   Users: {pipeline.num_users}")
    print(f"   Items: {pipeline.num_items}")
    print(f"   Train samples: {len(train_data_with_negatives[0])}")
    print(f"   Val samples: {len(val_data[0])}")
    print(f"   Test samples: {len(test_data[0])}")
    
    # Step 3: Create and train model
    print("\n🧠 Step 3: Training NCF model (5 epochs)...")
    model = NCF(
        num_users=pipeline.num_users,
        num_items=pipeline.num_items,
        embedding_dim=64,
        mlp_layers=[128, 64, 32],
        dropout=0.2,
        learning_rate=0.001,
        model_dir="backend/ml/deep_learning/models"
    )
    
    history = model.train(
        train_data=train_data_with_negatives,
        validation_data=val_data,
        epochs=5,
        batch_size=256,
        early_stopping_patience=3
    )
    
    print(f"✅ Training complete!")
    if history:
        if 'train_accuracy' in history:
            print(f"   Final train accuracy: {history['train_accuracy'][-1]:.4f}")
            print(f"   Final val accuracy: {history.get('val_accuracy', [0])[-1]:.4f}")
        elif 'train_loss' in history:
            print(f"   Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"   Final val loss: {history.get('val_loss', [0])[-1]:.4f}")
    
    # Step 4: Save PyTorch model (already saved during training)
    print("\n💾 Step 4: Ensuring model is saved...")
    
    # Find the saved model (NCF saves to model_dir/best_model.pth)
    model_dir = Path("backend/ml/deep_learning/models")
    best_model_path = model_dir / "best_model.pth"
    pytorch_path = model_dir / "ncf_model.pt"
    
    if best_model_path.exists():
        # Rename to standard name
        import shutil
        shutil.copy(best_model_path, pytorch_path)
        print(f"✅ Model copied to: {pytorch_path}")
    else:
        # Save the current model state
        model.save(str(pytorch_path))
        print(f"✅ Model saved to: {pytorch_path}")
    
    # Step 5: Export to ONNX
    print("\n🔄 Step 5: Exporting to ONNX...")
    
    from backend.ml.deep_learning.onnx_export import ONNXExporter
    
    onnx_path = model_dir / "ncf_model.onnx"
    exporter = ONNXExporter(
        model_path=str(pytorch_path),
        output_path=str(onnx_path)
    )
    
    if not exporter.export():
        print("❌ ONNX export failed!")
        return 1
    
    # Step 6: Benchmark
    print("\n⚡ Step 6: Benchmarking ONNX model...")
    
    from backend.ml.deep_learning.onnx_inference import ONNXNCFPredictor
    
    predictor = ONNXNCFPredictor(str(onnx_path))
    results = predictor.benchmark(num_iterations=50, batch_sizes=[1, 10, 50])
    
    print("\n📊 Performance Results:")
    print("=" * 70)
    print(f"{'Batch':<10} {'Mean (ms)':<12} {'P95 (ms)':<12} {'QPS':<15}")
    print("-" * 70)
    for batch, metrics in results.items():
        print(f"{batch:<10} {metrics['mean_ms']:<12.2f} {metrics['p95_ms']:<12.2f} {metrics['throughput_qps']:<15.0f}")
    print("=" * 70)
    
    # Step 7: Test recommendations
    print("\n🏆 Step 7: Testing recommendations...")
    top_items = predictor.predict_for_user(
        user_id=0,
        item_ids=list(range(min(50, pipeline.num_items))),
        top_k=5
    )
    
    print("Top-5 recommendations for User 0:")
    for rank, (item_id, score) in enumerate(top_items, 1):
        print(f"   {rank}. Item {item_id}: {score:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ SUCCESS! NCF model trained and exported to ONNX")
    print("=" * 80)
    print(f"\n📁 Files created:")
    print(f"   PyTorch: {pytorch_path}")
    print(f"   ONNX: {onnx_path}")
    print(f"   Metadata: {onnx_path.with_suffix('.json')}")
    print(f"\n🚀 Ready for production!")
    
    return 0


if __name__ == '__main__':
    exit(main())

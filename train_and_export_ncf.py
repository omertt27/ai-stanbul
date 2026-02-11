#!/usr/bin/env python3
"""
Quick NCF Model Training and ONNX Export Pipeline

This script:
1. Trains an NCF model on existing interaction data
2. Exports it to ONNX format
3. Benchmarks performance
4. Validates accuracy

Usage:
    python train_and_export_ncf.py
"""

import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.ml.deep_learning.models.ncf import NCFModel
from backend.ml.deep_learning.data_pipeline import DataPipeline
from backend.ml.deep_learning.onnx_export import ONNXExporter
from backend.ml.deep_learning.onnx_inference import ONNXNCFPredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train NCF model and export to ONNX."""
    
    print("=" * 80)
    print("🚀 NCF Model Training and ONNX Export Pipeline")
    print("=" * 80)
    
    # Step 1: Load and prepare data
    print("\n📊 Step 1: Loading interaction data...")
    data_pipeline = DataPipeline()
    data_pipeline.load_data()
    
    train_loader, val_loader, test_loader = data_pipeline.create_dataloaders(
        batch_size=256,
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    num_users = data_pipeline.num_users
    num_items = data_pipeline.num_items
    
    print(f"✅ Data loaded:")
    print(f"   Users: {num_users}")
    print(f"   Items: {num_items}")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print(f"   Test samples: {len(test_loader.dataset)}")
    
    # Step 2: Create and train model
    print("\n🧠 Step 2: Training NCF model...")
    model = NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=64,
        mlp_layers=[128, 64, 32]
    )
    
    # Train model (quick training - 5 epochs)
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        learning_rate=0.001,
        patience=3
    )
    
    print(f"✅ Training complete!")
    print(f"   Final train accuracy: {history['train_acc'][-1]:.4f}")
    print(f"   Final val accuracy: {history['val_acc'][-1]:.4f}")
    
    # Step 3: Save PyTorch model
    print("\n💾 Step 3: Saving PyTorch model...")
    model_dir = Path("backend/ml/deep_learning/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    pytorch_path = model_dir / "ncf_model.pt"
    model.save(str(pytorch_path))
    print(f"✅ PyTorch model saved to: {pytorch_path}")
    
    # Step 4: Export to ONNX
    print("\n🔄 Step 4: Exporting to ONNX format...")
    onnx_path = model_dir / "ncf_model.onnx"
    
    exporter = ONNXExporter(
        model_path=str(pytorch_path),
        output_path=str(onnx_path),
        opset_version=14,
        dynamic_batch=True
    )
    
    success = exporter.export()
    
    if not success:
        print("❌ ONNX export failed!")
        return 1
    
    print(f"✅ ONNX model exported to: {onnx_path}")
    
    # Step 5: Benchmark ONNX model
    print("\n⚡ Step 5: Benchmarking ONNX performance...")
    predictor = ONNXNCFPredictor(str(onnx_path))
    
    results = predictor.benchmark(
        num_iterations=100,
        batch_sizes=[1, 10, 50]
    )
    
    print("\n📊 Performance Results:")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput (QPS)':<20}")
    print("-" * 80)
    for batch_size, metrics in results.items():
        print(f"{batch_size:<12} {metrics['mean_ms']:<12.2f} "
              f"{metrics['p95_ms']:<12.2f} {metrics['throughput_qps']:<20.0f}")
    print("=" * 80)
    
    # Step 6: Test recommendations
    print("\n🏆 Step 6: Testing recommendations...")
    top_items = predictor.predict_for_user(
        user_id=0,
        item_ids=list(range(min(100, num_items))),
        top_k=5
    )
    
    print("Top-5 recommendations for User 0:")
    for rank, (item_id, score) in enumerate(top_items, 1):
        print(f"   {rank}. Item {item_id}: {score:.4f}")
    
    print("\n" + "=" * 80)
    print("✅ Pipeline Complete!")
    print("=" * 80)
    print(f"\n📁 Files created:")
    print(f"   PyTorch model: {pytorch_path}")
    print(f"   ONNX model: {onnx_path}")
    print(f"   Metadata: {onnx_path.with_suffix('.json')}")
    print(f"\n🚀 Ready for production deployment!")
    
    return 0


if __name__ == '__main__':
    exit(main())

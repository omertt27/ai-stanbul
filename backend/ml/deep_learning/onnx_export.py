#!/usr/bin/env python3
"""
ONNX Model Export Script for NCF Model

Converts the trained PyTorch NCF model to ONNX format for:
- 3-5x faster inference
- Cross-platform deployment
- Smaller model size
- Better production performance

Usage:
    python backend/ml/deep_learning/onnx_export.py --model-path models/ncf_model.pt --output models/ncf_model.onnx
"""

import argparse
import logging
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import json

from backend.ml.deep_learning.models.ncf import NCFModel as NCFPyTorchModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    Export PyTorch NCF model to ONNX format with validation.
    """
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        opset_version: int = 14,
        dynamic_batch: bool = True
    ):
        """
        Initialize ONNX exporter.
        
        Args:
            model_path: Path to trained PyTorch model (.pt file)
            output_path: Path for exported ONNX model
            opset_version: ONNX opset version (14 recommended for PyTorch 1.12+)
            dynamic_batch: Support dynamic batch sizes
        """
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_pytorch_model(self) -> Tuple[NCFPyTorchModel, Dict[str, Any]]:
        """
        Load trained PyTorch NCF model.
        
        Returns:
            Tuple of (model, metadata)
        """
        logger.info(f"Loading PyTorch model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # The NCF wrapper saves with nested structure:
        # checkpoint['model_state_dict']['model_state_dict'] = actual model weights
        # checkpoint['model_state_dict']['num_users'] = config
        
        inner_checkpoint = checkpoint.get('model_state_dict', {})
        
        # Extract model config from inner checkpoint
        num_users = inner_checkpoint.get('num_users', 0)
        num_items = inner_checkpoint.get('num_items', 0)
        embedding_dim = inner_checkpoint.get('embedding_dim', 64)
        mlp_layers = inner_checkpoint.get('mlp_layers', [128, 64, 32])
        
        logger.info(f"Model config: users={num_users}, items={num_items}, "
                   f"embedding_dim={embedding_dim}")
        
        # Create model instance (the actual PyTorch model, not the wrapper)
        model = NCFPyTorchModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            mlp_layers=mlp_layers
        )
        
        # Load the actual model weights from nested structure
        actual_state_dict = inner_checkpoint.get('model_state_dict', {})
        model.load_state_dict(actual_state_dict)
        model.eval()
        
        # Extract metadata
        metadata = {
            'num_users': num_users,
            'num_items': num_items,
            'embedding_dim': embedding_dim,
            'mlp_layers': mlp_layers,
            'training_metrics': checkpoint.get('metrics', {}),
            'pytorch_version': torch.__version__
        }
        
        logger.info(f"✅ Model loaded successfully")
        
        return model, metadata
    
    def export_to_onnx(self, model: NCFPyTorchModel) -> None:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: Trained NCF model
        """
        logger.info(f"Exporting model to ONNX format...")
        
        # Create dummy input (user_id, item_id)
        # Shape: (batch_size, 1) for each
        batch_size = 1
        dummy_user = torch.tensor([[0]], dtype=torch.long)
        dummy_item = torch.tensor([[0]], dtype=torch.long)
        
        # Define dynamic axes if enabled
        dynamic_axes = None
        if self.dynamic_batch:
            dynamic_axes = {
                'user_ids': {0: 'batch_size'},
                'item_ids': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_user, dummy_item),
            self.output_path,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=['user_ids', 'item_ids'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        logger.info(f"✅ Model exported to {self.output_path}")
    
    def validate_onnx_model(self) -> bool:
        """
        Validate exported ONNX model.
        
        Returns:
            True if validation passes
        """
        logger.info("Validating ONNX model...")
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(str(self.output_path))
            
            # Check model is well-formed
            onnx.checker.check_model(onnx_model)
            logger.info("✅ ONNX model structure is valid")
            
            # Check graph inputs/outputs
            graph = onnx_model.graph
            logger.info(f"Inputs: {[input.name for input in graph.input]}")
            logger.info(f"Outputs: {[output.name for output in graph.output]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX validation failed: {e}")
            return False
    
    def compare_outputs(
        self,
        pytorch_model: NCFPyTorchModel,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Compare PyTorch and ONNX model outputs.
        
        Args:
            pytorch_model: Original PyTorch model
            num_samples: Number of random samples to test
            
        Returns:
            Dictionary with comparison metrics
        """
        logger.info(f"Comparing PyTorch vs ONNX outputs ({num_samples} samples)...")
        
        # Create ONNX runtime session
        ort_session = ort.InferenceSession(str(self.output_path))
        
        # Generate random test samples
        max_user = pytorch_model.num_users
        max_item = pytorch_model.num_items
        
        user_ids = np.random.randint(0, max_user, size=(num_samples, 1)).astype(np.int64)
        item_ids = np.random.randint(0, max_item, size=(num_samples, 1)).astype(np.int64)
        
        # PyTorch predictions
        with torch.no_grad():
            pytorch_user = torch.from_numpy(user_ids)
            pytorch_item = torch.from_numpy(item_ids)
            pytorch_output = pytorch_model(pytorch_user, pytorch_item).numpy()
        
        # ONNX predictions
        ort_inputs = {
            'user_ids': user_ids,
            'item_ids': item_ids
        }
        onnx_output = ort_session.run(None, ort_inputs)[0]
        
        # Calculate differences
        abs_diff = np.abs(pytorch_output - onnx_output)
        rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)
        
        metrics = {
            'mean_abs_diff': float(np.mean(abs_diff)),
            'max_abs_diff': float(np.max(abs_diff)),
            'mean_rel_diff': float(np.mean(rel_diff)),
            'max_rel_diff': float(np.max(rel_diff)),
            'correlation': float(np.corrcoef(pytorch_output.flatten(), onnx_output.flatten())[0, 1])
        }
        
        logger.info("Comparison metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")
        
        # Check if outputs are close enough
        if metrics['mean_abs_diff'] < 1e-5 and metrics['correlation'] > 0.999:
            logger.info("✅ ONNX model outputs match PyTorch model")
        else:
            logger.warning("⚠️ ONNX model outputs differ from PyTorch model")
        
        return metrics
    
    def save_metadata(self, metadata: Dict[str, Any], comparison: Dict[str, float]) -> None:
        """
        Save model metadata and comparison results.
        
        Args:
            metadata: Model configuration and training info
            comparison: PyTorch vs ONNX comparison metrics
        """
        metadata_path = self.output_path.with_suffix('.json')
        
        full_metadata = {
            **metadata,
            'onnx_export': {
                'opset_version': self.opset_version,
                'dynamic_batch': self.dynamic_batch,
                'model_size_bytes': self.output_path.stat().st_size,
                'comparison_metrics': comparison
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        logger.info(f"✅ Metadata saved to {metadata_path}")
    
    def export(self) -> bool:
        """
        Full export pipeline: load → export → validate → compare.
        
        Returns:
            True if export successful
        """
        try:
            # Load PyTorch model
            model, metadata = self.load_pytorch_model()
            
            # Export to ONNX
            self.export_to_onnx(model)
            
            # Validate ONNX model
            if not self.validate_onnx_model():
                return False
            
            # Compare outputs
            comparison = self.compare_outputs(model)
            
            # Save metadata
            self.save_metadata(metadata, comparison)
            
            logger.info("🎉 ONNX export completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for ONNX export."""
    parser = argparse.ArgumentParser(description='Export NCF model to ONNX format')
    parser.add_argument(
        '--model-path',
        type=str,
        default='backend/ml/deep_learning/models/ncf_model.pt',
        help='Path to PyTorch model file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='backend/ml/deep_learning/models/ncf_model.onnx',
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--opset-version',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )
    parser.add_argument(
        '--no-dynamic-batch',
        action='store_true',
        help='Disable dynamic batch size support'
    )
    
    args = parser.parse_args()
    
    # Create exporter
    exporter = ONNXExporter(
        model_path=args.model_path,
        output_path=args.output,
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch
    )
    
    # Run export
    success = exporter.export()
    
    if success:
        logger.info("✅ ONNX export successful!")
        logger.info(f"Model saved to: {args.output}")
        logger.info(f"Metadata saved to: {args.output.replace('.onnx', '.json')}")
        return 0
    else:
        logger.error("❌ ONNX export failed!")
        return 1


if __name__ == '__main__':
    exit(main())

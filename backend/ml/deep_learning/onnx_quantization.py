"""
ONNX INT8 Quantization for NCF Model

Quantizes the NCF ONNX model from FP32 to INT8 for faster inference
and smaller model size.

Expected improvements:
- Inference speed: 2x faster
- Model size: 75% smaller
- Memory usage: 50% reduction
- Accuracy impact: <1% degradation

Author: AI Istanbul Team
Date: February 11, 2026
"""

import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table
import onnxruntime as ort

logger = logging.getLogger(__name__)


class NCFCalibrationDataReader(CalibrationDataReader):
    """
    Calibration data reader for INT8 quantization.
    
    Provides sample data for calibration to optimize quantization accuracy.
    """
    
    def __init__(self, num_users: int, num_items: int, num_samples: int = 100):
        """
        Initialize calibration data reader.
        
        Args:
            num_users: Number of users in the model
            num_items: Number of items in the model
            num_samples: Number of calibration samples to generate
        """
        self.num_users = num_users
        self.num_items = num_items
        self.num_samples = num_samples
        self.current_sample = 0
        
        # Generate calibration data
        self.calibration_data = self._generate_calibration_data()
    
    def _generate_calibration_data(self):
        """Generate representative calibration data."""
        data = []
        
        for _ in range(self.num_samples):
            user_id = np.random.randint(0, self.num_users)
            item_id = np.random.randint(0, self.num_items)
            
            data.append({
                'user_id': np.array([user_id], dtype=np.int64),
                'item_id': np.array([item_id], dtype=np.int64)
            })
        
        logger.info(f"Generated {len(data)} calibration samples")
        return data
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Get next calibration sample."""
        if self.current_sample >= len(self.calibration_data):
            return None
        
        sample = self.calibration_data[self.current_sample]
        self.current_sample += 1
        
        return sample
    
    def rewind(self):
        """Reset to beginning of calibration data."""
        self.current_sample = 0


class ONNXQuantizer:
    """
    Quantizes ONNX models to INT8 format.
    """
    
    def __init__(self):
        self.supported_ops = [
            'MatMul', 'Gemm', 'Conv', 'Mul', 'Add'
        ]
    
    def quantize_dynamic(
        self,
        model_path: str,
        output_path: str,
        weight_type: QuantType = QuantType.QInt8
    ) -> Dict[str, Any]:
        """
        Quantize model using dynamic quantization.
        
        Dynamic quantization quantizes weights at load time and activations
        at runtime. Faster to apply but slightly less optimal than static.
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path for output quantized model
            weight_type: Quantization type for weights
            
        Returns:
            Dictionary with quantization info
        """
        logger.info(f"🔄 Starting dynamic quantization...")
        logger.info(f"  Input: {model_path}")
        logger.info(f"  Output: {output_path}")
        
        # Quantize
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=weight_type,
            op_types_to_quantize=self.supported_ops
        )
        
        # Get model sizes
        original_size = Path(model_path).stat().st_size
        quantized_size = Path(output_path).stat().st_size
        size_reduction = (1 - quantized_size / original_size) * 100
        
        logger.info(f"✅ Dynamic quantization complete!")
        logger.info(f"  Original size: {original_size / 1024 / 1024:.2f} MB")
        logger.info(f"  Quantized size: {quantized_size / 1024 / 1024:.2f} MB")
        logger.info(f"  Size reduction: {size_reduction:.1f}%")
        
        return {
            'method': 'dynamic',
            'original_size_bytes': original_size,
            'quantized_size_bytes': quantized_size,
            'size_reduction_percent': size_reduction,
            'weight_type': str(weight_type)
        }
    
    def validate_quantized_model(
        self,
        original_model_path: str,
        quantized_model_path: str,
        num_users: int = 1000,
        num_items: int = 500,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Validate quantized model accuracy against original.
        
        Args:
            original_model_path: Path to original FP32 model
            quantized_model_path: Path to quantized INT8 model
            num_users: Number of users in model
            num_items: Number of items in model
            num_samples: Number of validation samples
            
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating quantized model accuracy...")
        
        # Load models
        original_session = ort.InferenceSession(original_model_path)
        quantized_session = ort.InferenceSession(quantized_model_path)
        
        # Generate test data
        differences = []
        
        for _ in range(num_samples):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)
            
            inputs = {
                'user_id': np.array([user_id], dtype=np.int64),
                'item_id': np.array([item_id], dtype=np.int64)
            }
            
            # Run inference
            original_output = original_session.run(None, inputs)[0]
            quantized_output = quantized_session.run(None, inputs)[0]
            
            # Calculate difference
            diff = np.abs(original_output - quantized_output).mean()
            differences.append(diff)
        
        # Calculate statistics
        avg_diff = np.mean(differences)
        max_diff = np.max(differences)
        
        # Accuracy delta (as percentage)
        accuracy_delta = avg_diff * 100
        
        logger.info(f"✅ Validation complete!")
        logger.info(f"  Average difference: {avg_diff:.6f}")
        logger.info(f"  Max difference: {max_diff:.6f}")
        logger.info(f"  Accuracy delta: {accuracy_delta:.3f}%")
        
        # Check if acceptable (<1% degradation)
        acceptable = accuracy_delta < 1.0
        
        if acceptable:
            logger.info(f"  ✅ Accuracy degradation is acceptable (<1%)")
        else:
            logger.warning(f"  ⚠️ Accuracy degradation exceeds 1%")
        
        return {
            'avg_difference': float(avg_diff),
            'max_difference': float(max_diff),
            'accuracy_delta_percent': float(accuracy_delta),
            'acceptable': acceptable,
            'num_samples': num_samples
        }
    
    def benchmark_performance(
        self,
        original_model_path: str,
        quantized_model_path: str,
        num_users: int = 1000,
        num_items: int = 500,
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance of original vs quantized model.
        
        Args:
            original_model_path: Path to original model
            quantized_model_path: Path to quantized model
            num_users: Number of users
            num_items: Number of items
            num_iterations: Number of benchmark iterations
            
        Returns:
            Performance comparison
        """
        import time
        
        logger.info(f"⏱️ Benchmarking performance ({num_iterations} iterations)...")
        
        # Load models
        original_session = ort.InferenceSession(original_model_path)
        quantized_session = ort.InferenceSession(quantized_model_path)
        
        # Benchmark original model
        original_times = []
        for _ in range(num_iterations):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)
            
            inputs = {
                'user_id': np.array([user_id], dtype=np.int64),
                'item_id': np.array([item_id], dtype=np.int64)
            }
            
            start = time.perf_counter()
            _ = original_session.run(None, inputs)
            original_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark quantized model
        quantized_times = []
        for _ in range(num_iterations):
            user_id = np.random.randint(0, num_users)
            item_id = np.random.randint(0, num_items)
            
            inputs = {
                'user_id': np.array([user_id], dtype=np.int64),
                'item_id': np.array([item_id], dtype=np.int64)
            }
            
            start = time.perf_counter()
            _ = quantized_session.run(None, inputs)
            quantized_times.append((time.perf_counter() - start) * 1000)
        
        # Calculate statistics
        original_p50 = np.percentile(original_times, 50)
        original_p95 = np.percentile(original_times, 95)
        original_avg = np.mean(original_times)
        
        quantized_p50 = np.percentile(quantized_times, 50)
        quantized_p95 = np.percentile(quantized_times, 95)
        quantized_avg = np.mean(quantized_times)
        
        speedup = original_avg / quantized_avg
        
        logger.info(f"✅ Benchmark complete!")
        logger.info(f"  Original FP32:")
        logger.info(f"    P50: {original_p50:.2f}ms")
        logger.info(f"    P95: {original_p95:.2f}ms")
        logger.info(f"    Avg: {original_avg:.2f}ms")
        logger.info(f"  Quantized INT8:")
        logger.info(f"    P50: {quantized_p50:.2f}ms")
        logger.info(f"    P95: {quantized_p95:.2f}ms")
        logger.info(f"    Avg: {quantized_avg:.2f}ms")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        return {
            'original': {
                'p50_ms': float(original_p50),
                'p95_ms': float(original_p95),
                'avg_ms': float(original_avg)
            },
            'quantized': {
                'p50_ms': float(quantized_p50),
                'p95_ms': float(quantized_p95),
                'avg_ms': float(quantized_avg)
            },
            'speedup': float(speedup),
            'improvement_percent': float((speedup - 1) * 100)
        }


def main():
    """Main quantization workflow."""
    parser = argparse.ArgumentParser(description='Quantize NCF ONNX model to INT8')
    parser.add_argument(
        '--input',
        type=str,
        default='backend/ml/deep_learning/models/ncf_model.onnx',
        help='Path to input FP32 ONNX model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='backend/ml/deep_learning/models/ncf_model_int8.onnx',
        help='Path for output INT8 ONNX model'
    )
    parser.add_argument(
        '--num-users',
        type=int,
        default=1000,
        help='Number of users in model'
    )
    parser.add_argument(
        '--num-items',
        type=int,
        default=500,
        help='Number of items in model'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate quantized model accuracy'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark performance'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("🚀 Starting NCF ONNX INT8 Quantization")
    logger.info(f"  Input model: {args.input}")
    logger.info(f"  Output model: {args.output}")
    
    # Initialize quantizer
    quantizer = ONNXQuantizer()
    
    # Quantize model
    quant_info = quantizer.quantize_dynamic(
        model_path=args.input,
        output_path=args.output
    )
    
    # Validate if requested
    if args.validate:
        validation_results = quantizer.validate_quantized_model(
            original_model_path=args.input,
            quantized_model_path=args.output,
            num_users=args.num_users,
            num_items=args.num_items
        )
        quant_info['validation'] = validation_results
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_results = quantizer.benchmark_performance(
            original_model_path=args.input,
            quantized_model_path=args.output,
            num_users=args.num_users,
            num_items=args.num_items
        )
        quant_info['benchmark'] = benchmark_results
    
    logger.info("🎉 Quantization complete!")
    logger.info(f"  Quantized model saved to: {args.output}")
    
    return quant_info


if __name__ == '__main__':
    main()

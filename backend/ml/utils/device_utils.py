"""
Cross-Platform Device Utilities
Support for CUDA (T4), MPS (Apple Silicon), and CPU

Handles:
- Device detection and selection
- Mixed precision compatibility (FP16 on CUDA, FP32 on MPS)
- Platform-specific optimizations
"""

import torch
import logging
from typing import Tuple, Optional
import platform

logger = logging.getLogger(__name__)


def get_optimal_device(prefer_gpu: bool = True, device_override: Optional[str] = None) -> Tuple[str, bool]:
    """
    Get the optimal device for the current platform
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        device_override: Force specific device ('cuda', 'mps', 'cpu')
    
    Returns:
        Tuple of (device_name, supports_fp16)
        
    Platform Support:
    - CUDA (T4 Production): Full FP16 mixed precision
    - MPS (M2 Pro Dev): FP32 only (MPS doesn't support FP16 well)
    - CPU (Fallback): FP32 only
    """
    
    if device_override:
        device = device_override
        supports_fp16 = (device == 'cuda')
        logger.info(f"🔧 Device override: {device} (FP16: {supports_fp16})")
        return device, supports_fp16
    
    # Check CUDA (NVIDIA GPUs - T4 production)
    if prefer_gpu and torch.cuda.is_available():
        device = 'cuda'
        supports_fp16 = True
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"🚀 CUDA GPU detected: {gpu_name}")
        logger.info(f"   Memory: {gpu_memory:.1f} GB")
        logger.info(f"   FP16 mixed precision: ✅ ENABLED")
        
        return device, supports_fp16
    
    # Check MPS (Apple Silicon - M1/M2/M3)
    if prefer_gpu and torch.backends.mps.is_available():
        device = 'mps'
        supports_fp16 = False  # MPS has poor FP16 support
        
        machine = platform.machine()
        logger.info(f"🍎 Apple Silicon MPS detected: {machine}")
        logger.info(f"   FP16 mixed precision: ❌ DISABLED (using FP32)")
        logger.warning("   Note: MPS uses FP32 for stability. Training will be slower than CUDA.")
        
        return device, supports_fp16
    
    # Fallback to CPU
    device = 'cpu'
    supports_fp16 = False
    
    logger.info(f"💻 Using CPU (no GPU available)")
    logger.info(f"   FP16 mixed precision: ❌ DISABLED")
    logger.warning("   Note: CPU training is significantly slower. Consider using GPU.")
    
    return device, supports_fp16


def setup_device_optimizations(device: str):
    """
    Configure platform-specific optimizations
    
    Args:
        device: Device name ('cuda', 'mps', or 'cpu')
    """
    
    if device == 'cuda':
        # CUDA optimizations for T4
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels
        torch.backends.cudnn.enabled = True
        
        logger.info("⚙️  CUDA optimizations enabled:")
        logger.info("   - cuDNN benchmark mode")
        logger.info("   - Mixed precision (FP16) training")
        
    elif device == 'mps':
        # MPS optimizations for Apple Silicon
        # MPS is automatically optimized by PyTorch
        logger.info("⚙️  MPS optimizations:")
        logger.info("   - Metal Performance Shaders")
        logger.info("   - Unified memory architecture")
        logger.info("   - FP32 precision (stable)")
        
    else:
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        
        logger.info("⚙️  CPU optimizations:")
        logger.info(f"   - Threads: {torch.get_num_threads()}")


def get_batch_size_recommendation(device: str, model_size_mb: float) -> int:
    """
    Get recommended batch size based on device and model size
    
    Args:
        device: Device name
        model_size_mb: Model size in megabytes
    
    Returns:
        Recommended batch size
    """
    
    if device == 'cuda':
        # T4 has 16GB VRAM
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Heuristic: ~50% of memory for batch
        available_memory = gpu_memory * 0.5
        sample_memory_mb = model_size_mb * 0.01  # Rough estimate
        recommended = int((available_memory * 1000) / sample_memory_mb)
        
        # Clamp to reasonable range
        recommended = max(512, min(recommended, 4096))
        
    elif device == 'mps':
        # M2 Pro has unified memory (shared with system)
        # Be conservative to avoid memory pressure
        recommended = 1024  # Smaller batches for MPS
        
    else:
        # CPU - use small batches
        recommended = 256
    
    logger.info(f"📊 Recommended batch size for {device}: {recommended}")
    return recommended


def move_batch_to_device(batch, device: str):
    """
    Move a batch to the specified device with proper handling
    
    Args:
        batch: Batch data (tensor, tuple, or dict)
        device: Target device
    
    Returns:
        Batch moved to device
    """
    
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=(device == 'cuda'))
    
    elif isinstance(batch, (tuple, list)):
        return type(batch)(move_batch_to_device(item, device) for item in batch)
    
    elif isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    
    else:
        return batch


def print_device_info():
    """Print comprehensive device information"""
    
    print("\n" + "="*60)
    print("🖥️  Device Information")
    print("="*60)
    
    # System info
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    # CUDA info
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    
    # MPS info
    print(f"\nMPS Available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"  Apple Silicon: {platform.machine()}")
        print(f"  Unified Memory: Yes")
    
    # CPU info
    print(f"\nCPU Threads: {torch.get_num_threads()}")
    
    print("="*60 + "\n")


# Platform-specific context managers

class CUDAMemoryOptimizer:
    """Context manager for CUDA memory optimization"""
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class MPSMemoryOptimizer:
    """Context manager for MPS memory optimization"""
    
    def __enter__(self):
        # MPS memory is managed automatically by Metal
        return self
    
    def __exit__(self, *args):
        # Force sync to avoid memory leaks
        if torch.backends.mps.is_available():
            torch.mps.synchronize()


def get_memory_optimizer(device: str):
    """Get the appropriate memory optimizer for the device"""
    
    if device == 'cuda':
        return CUDAMemoryOptimizer()
    elif device == 'mps':
        return MPSMemoryOptimizer()
    else:
        # No-op for CPU
        class NoOpOptimizer:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoOpOptimizer()


# Utility functions for model training

def get_dataloader_config(device: str) -> dict:
    """
    Get optimal DataLoader configuration for device
    
    Args:
        device: Device name
    
    Returns:
        Dict with DataLoader kwargs
    """
    
    if device == 'cuda':
        return {
            'pin_memory': True,  # Faster CPU->GPU transfer
            'num_workers': 4,    # Parallel data loading
            'persistent_workers': True,
        }
    
    elif device == 'mps':
        return {
            'pin_memory': False,  # Not needed for unified memory
            'num_workers': 2,     # Conservative for MPS
            'persistent_workers': False,
        }
    
    else:  # CPU
        return {
            'pin_memory': False,
            'num_workers': 2,
            'persistent_workers': False,
        }


if __name__ == "__main__":
    # Test device detection
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print_device_info()
    
    device, supports_fp16 = get_optimal_device()
    setup_device_optimizations(device)
    
    print(f"\n✅ Selected device: {device}")
    print(f"✅ FP16 support: {supports_fp16}")
    print(f"✅ Recommended batch size: {get_batch_size_recommendation(device, 50.0)}")

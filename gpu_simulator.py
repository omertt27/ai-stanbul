"""
GPU Simulator for Local Development
Simulates NVIDIA T4 GPU behavior on MacBook CPU/MPS
"""

import torch
import time
from typing import Optional, Dict, Any
import logging
import random
import psutil

logger = logging.getLogger(__name__)


class T4GPUSimulator:
    """
    Simulates T4 GPU for local development
    - Uses MPS (Metal) on Apple Silicon if available
    - Falls back to CPU with simulated latency
    - Tracks "GPU" metrics for monitoring
    """
    
    def __init__(self, simulate_latency: bool = True):
        self.simulate_latency = simulate_latency
        
        # Detect best device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.backend = "MPS (Apple GPU)"
            logger.info("🍎 Using Apple Metal (MPS) for GPU simulation")
        else:
            self.device = torch.device("cpu")
            self.backend = "CPU"
            logger.info("💻 Using CPU for GPU simulation")
        
        # Simulated T4 specs
        self.specs = {
            'name': 'Simulated NVIDIA T4',
            'memory_total_gb': 16,
            'cuda_cores': 2560,
            'tensor_cores': 320,
            'compute_capability': '7.5'
        }
        
        # Metrics tracking
        self.metrics = {
            'total_inferences': 0,
            'total_time_ms': 0.0,
            'avg_latency_ms': 0.0,
            'utilization': 0.0
        }
        
    def is_available(self) -> bool:
        """Check if 'GPU' is available"""
        return True  # Always available in simulation mode
    
    def get_device(self) -> torch.device:
        """Get torch device"""
        return self.device
    
    def get_memory_info(self) -> Dict[str, float]:
        """Simulate GPU memory info"""
        if self.device.type == "mps":
            # MPS doesn't expose memory info, return simulated
            return {
                'allocated_gb': 2.5,
                'reserved_gb': 4.0,
                'total_gb': 16.0,
                'utilization_percent': 15.6
            }
        else:
            # CPU - return system memory
            mem = psutil.virtual_memory()
            return {
                'allocated_gb': (mem.total - mem.available) / 1e9,
                'total_gb': mem.total / 1e9,
                'utilization_percent': mem.percent
            }
    
    def simulate_t4_latency(self, operation: str = 'inference') -> float:
        """
        Simulate T4 GPU latency characteristics
        Returns: simulated latency in milliseconds
        """
        if not self.simulate_latency:
            return 0.0
        
        # T4 GPU typical latencies (milliseconds)
        t4_latencies = {
            'inference': 2.5,      # BERT inference
            'embedding': 1.0,      # Embedding lookup
            'similarity': 0.5,     # Cosine similarity
            'ranking': 1.5,        # Ranking model
            'prediction': 2.0      # General prediction
        }
        
        # Add small random variation
        base_latency = t4_latencies.get(operation, 2.0)
        simulated = base_latency * random.uniform(0.9, 1.1)
        
        # Sleep to simulate GPU processing time
        time.sleep(simulated / 1000.0)
        
        return simulated
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to device"""
        return tensor.to(self.device)
    
    def inference_context(self):
        """Context manager for inference (like torch.cuda.amp.autocast)"""
        if self.device.type == "mps":
            # MPS doesn't support autocast yet
            return torch.no_grad()
        else:
            return torch.no_grad()
    
    def update_metrics(self, latency_ms: float):
        """Update performance metrics"""
        self.metrics['total_inferences'] += 1
        self.metrics['total_time_ms'] += latency_ms
        self.metrics['avg_latency_ms'] = (
            self.metrics['total_time_ms'] / self.metrics['total_inferences']
        )
        
        # Simulate utilization (random for now)
        self.metrics['utilization'] = random.uniform(60, 85)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'device': str(self.device),
            'backend': self.backend,
            'specs': self.specs,
            'metrics': self.metrics,
            'memory': self.get_memory_info()
        }
    
    def __repr__(self):
        return f"T4GPUSimulator(device={self.device}, backend={self.backend})"


# Global simulator instance
_simulator = None


def get_gpu_simulator() -> T4GPUSimulator:
    """Get or create global GPU simulator"""
    global _simulator
    if _simulator is None:
        _simulator = T4GPUSimulator()
    return _simulator


# Example usage
if __name__ == "__main__":
    # Initialize simulator
    sim = get_gpu_simulator()
    
    print(f"\n{'='*60}")
    print(f"GPU Simulator Test")
    print(f"{'='*60}\n")
    
    print(f"GPU Simulator: {sim}")
    print(f"Device: {sim.get_device()}")
    print(f"\nMetrics:")
    import json
    print(json.dumps(sim.get_metrics(), indent=2))
    
    # Test tensor operations
    print(f"\n{'='*60}")
    print(f"Testing Tensor Operations")
    print(f"{'='*60}\n")
    
    x = torch.randn(100, 768)
    print(f"Created tensor: {x.shape}")
    
    x_device = sim.to_device(x)
    print(f"Moved to device: {x_device.device}")
    
    # Simulate inference
    with sim.inference_context():
        start = time.time()
        y = torch.matmul(x_device, x_device.T)
        latency = (time.time() - start) * 1000
        
    print(f"Matrix multiplication result: {y.shape}")
    print(f"Actual latency: {latency:.2f}ms")
    
    # Simulate T4 latency
    simulated_latency = sim.simulate_t4_latency('inference')
    print(f"Simulated T4 latency: {simulated_latency:.2f}ms")
    
    sim.update_metrics(simulated_latency)
    
    print(f"\n{'='*60}")
    print(f"✅ GPU Simulator working correctly!")
    print(f"{'='*60}\n")

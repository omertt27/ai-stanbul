# Istanbul AI - Local Development Setup (MacBook)
**Date:** October 21, 2025  
**Platform:** macOS (Apple Silicon M1/M2/M3 or Intel)  
**Goal:** Develop and test GPU/ML enhancements locally before cloud deployment

---

## 🎯 Overview

Since MacBooks don't have NVIDIA GPUs, we'll:
1. **Develop locally** with CPU fallback simulation
2. **Test on Google Colab** with free T4 GPU access
3. **Deploy to GCP** when ready for production

---

## 💻 Phase 1: Local MacBook Setup (Day 1)

### Step 1.1: Check Your Mac Specs

```bash
# Check macOS version
sw_vers

# Check processor type (Apple Silicon or Intel)
uname -m
# Output: arm64 (Apple Silicon) or x86_64 (Intel)

# Check memory
sysctl hw.memsize | awk '{print $2/1073741824 " GB"}'

# Expected: macOS 13+ (Ventura), 16GB+ RAM recommended
```

### Step 1.2: Install Homebrew (if not already installed)

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Verify installation
brew --version
```

### Step 1.3: Install Python 3.10+

```bash
# Install Python 3.10 via Homebrew
brew install python@3.10

# Verify installation
python3.10 --version

# Create symlink (optional)
ln -s /opt/homebrew/bin/python3.10 /usr/local/bin/python3
```

### Step 1.4: Create Virtual Environment

```bash
# Navigate to project directory
cd ~/Desktop/ai-stanbul

# Create virtual environment
python3.10 -m venv venv-gpu-ml

# Activate virtual environment
source venv-gpu-ml/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# You should see (venv-gpu-ml) in your terminal prompt
```

### Step 1.5: Install PyTorch (CPU version for Mac)

```bash
# For Apple Silicon (M1/M2/M3)
pip install torch torchvision torchaudio

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

# Note: MPS = Metal Performance Shaders (Apple's GPU acceleration)
```

### Step 1.6: Install ML/DL Libraries

```bash
# Core ML libraries
pip install numpy pandas scikit-learn scipy

# Deep learning
pip install transformers sentence-transformers

# Traditional ML
pip install xgboost lightgbm catboost

# NLP
pip install spacy textblob nltk
python3 -m spacy download en_core_web_sm

# Vector search
pip install faiss-cpu  # Note: CPU version for Mac

# Database & caching
pip install redis pymongo sqlalchemy

# API & web
pip install fastapi uvicorn pydantic

# Monitoring & logging
pip install mlflow wandb prometheus-client

# Testing
pip install pytest pytest-asyncio pytest-cov

# Utilities
pip install python-dotenv pyyaml requests aiohttp tqdm

# Save requirements
pip freeze > requirements-local.txt
```

### Step 1.7: Install Redis (for caching)

```bash
# Install Redis via Homebrew
brew install redis

# Start Redis server
brew services start redis

# Test Redis connection
redis-cli ping
# Expected output: PONG

# Check Redis is running
brew services list | grep redis
```

### Step 1.8: Install PostgreSQL (for user profiles & analytics)

```bash
# Install PostgreSQL
brew install postgresql@15

# Start PostgreSQL
brew services start postgresql@15

# Create database
createdb istanbul_ai_dev

# Verify connection
psql istanbul_ai_dev -c "SELECT version();"
```

---

## 🧪 Phase 2: Development Environment Setup (Day 1-2)

### Step 2.1: Project Structure

```bash
# Create development directory structure
mkdir -p ~/Desktop/ai-stanbul/{models,data,logs,tests,notebooks}
mkdir -p ~/Desktop/ai-stanbul/models/{t4_simulator,personalization,route_optimizer}
mkdir -p ~/Desktop/ai-stanbul/data/{raw,processed,embeddings,cache}

# Your structure should look like:
# ai-stanbul/
# ├── backend/
# ├── models/              # ML models
# │   ├── t4_simulator/
# │   ├── personalization/
# │   └── route_optimizer/
# ├── data/                # Data storage
# ├── logs/                # Application logs
# ├── tests/               # Test files
# ├── notebooks/           # Jupyter notebooks
# ├── venv-gpu-ml/        # Virtual environment
# └── requirements-local.txt
```

### Step 2.2: Environment Variables

```bash
# Create .env file for local development
cat > .env.local << 'EOF'
# Istanbul AI - Local Development Environment
ENVIRONMENT=development
DEBUG=True

# Database
DATABASE_URL=postgresql://localhost/istanbul_ai_dev
REDIS_URL=redis://localhost:6379/0

# ML Models
MODEL_PATH=./models
USE_GPU=False
SIMULATE_GPU=True
DEVICE=cpu

# API Keys (for testing)
OPENAI_API_KEY=your_key_here_if_needed
GOOGLE_MAPS_API_KEY=your_key_here

# Logging
LOG_LEVEL=INFO
LOG_PATH=./logs

# Performance
BATCH_SIZE=32
MAX_WORKERS=4
CACHE_TTL=3600

# MLflow
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=istanbul-ai-local

# Weights & Biases (optional)
WANDB_PROJECT=istanbul-ai-dev
WANDB_MODE=offline
EOF

# Load environment variables
source .env.local
```

### Step 2.3: Create GPU Simulator

```python
# Create: models/t4_simulator/gpu_simulator.py

"""
GPU Simulator for Local Development
Simulates NVIDIA T4 GPU behavior on MacBook CPU/MPS
"""

import torch
import time
from typing import Optional, Dict, Any
import logging

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
            import psutil
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
        import random
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
        import random
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
    
    print(f"GPU Simulator: {sim}")
    print(f"Device: {sim.get_device()}")
    print(f"\nMetrics:")
    import json
    print(json.dumps(sim.get_metrics(), indent=2))
    
    # Test tensor operations
    x = torch.randn(100, 768)
    x_device = sim.to_device(x)
    
    # Simulate inference
    with sim.inference_context():
        start = time.time()
        y = torch.matmul(x_device, x_device.T)
        latency = (time.time() - start) * 1000
        
    print(f"\nTensor operation latency: {latency:.2f}ms")
    
    # Simulate T4 latency
    simulated_latency = sim.simulate_t4_latency('inference')
    print(f"Simulated T4 latency: {simulated_latency:.2f}ms")
```

Save this file and test it:

```bash
# Test GPU simulator
python3 models/t4_simulator/gpu_simulator.py

# Expected output:
# 🍎 Using Apple Metal (MPS) for GPU simulation  (or CPU)
# GPU Simulator: T4GPUSimulator(device=mps, backend=MPS (Apple GPU))
# Device: mps
# Metrics: {...}
```

---

## 🧪 Phase 3: Google Colab Testing (Day 2)

Since your MacBook doesn't have an NVIDIA GPU, use Google Colab for **real GPU testing**:

### Step 3.1: Setup Google Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create new notebook: "Istanbul AI - T4 GPU Testing"
3. Enable GPU:
   - Runtime → Change runtime type → Hardware accelerator → **T4 GPU**
   - Click Save

### Step 3.2: Colab Setup Code

```python
# In Colab notebook - Cell 1: Setup
!nvidia-smi  # Verify T4 GPU

# Install dependencies
!pip install torch torchvision transformers sentence-transformers faiss-gpu

# Clone your repo (if public) or upload files
from google.colab import drive
drive.mount('/content/drive')

# Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

```python
# Cell 2: Test T4 Neural Query Processor
# Upload your t4_neural_query_processor.py and test it

from t4_neural_query_processor import T4NeuralQueryProcessor

processor = T4NeuralQueryProcessor()

# Test queries
test_queries = [
    "Show me museums in Sultanahmet",
    "How to get to Galata Tower?",
    "Best restaurants in Beyoğlu"
]

for query in test_queries:
    result = await processor.process_query(query)
    print(f"Query: {query}")
    print(f"Intent: {result.intent} (confidence: {result.confidence:.2f})")
    print(f"Latency: {result.latency_ms:.1f}ms\n")
```

### Step 3.3: Download Models from Colab

```python
# After testing in Colab, download trained models
from google.colab import files

# Save and download model
torch.save(processor.model.state_dict(), 'turkish_bert_finetuned.pth')
files.download('turkish_bert_finetuned.pth')

# Upload to your MacBook and use in development
```

---

## 🔧 Phase 4: Hybrid Development Workflow (Day 3+)

### Recommended Workflow:

```
┌─────────────────────────────────────────────┐
│     Development Cycle                       │
├─────────────────────────────────────────────┤
│                                             │
│  1. Code on MacBook (local)                │
│     ├─ Write Python code                   │
│     ├─ Unit tests (CPU/MPS)                │
│     └─ Integration tests                   │
│                                             │
│  2. Test on Google Colab (T4 GPU)          │
│     ├─ Real GPU inference testing          │
│     ├─ Performance benchmarking            │
│     └─ Model training/fine-tuning          │
│                                             │
│  3. Deploy to GCP (Production)             │
│     ├─ Staging environment test            │
│     ├─ Canary deployment (10%)             │
│     └─ Full production rollout             │
│                                             │
└─────────────────────────────────────────────┘
```

### Daily Development Commands:

```bash
# Morning: Start development session
cd ~/Desktop/ai-stanbul
source venv-gpu-ml/bin/activate
brew services start redis
brew services start postgresql@15

# Check services
redis-cli ping
psql istanbul_ai_dev -c "SELECT 1"

# Run tests locally
pytest tests/ -v

# Run application locally
python backend/main.py

# Evening: Stop services (save battery)
brew services stop redis
brew services stop postgresql@15
deactivate
```

---

## 📝 Phase 5: Create Test Scripts (Day 3)

### Create test_local_gpu_simulation.py

```python
# tests/test_local_gpu_simulation.py

"""
Test GPU simulation on MacBook
Verifies that all GPU-dependent code works with CPU/MPS fallback
"""

import sys
sys.path.append('..')

import torch
import pytest
import time
from models.t4_simulator.gpu_simulator import get_gpu_simulator


def test_gpu_simulator_initialization():
    """Test GPU simulator initializes correctly"""
    sim = get_gpu_simulator()
    
    assert sim is not None
    assert sim.is_available()
    assert sim.get_device() in [torch.device('mps'), torch.device('cpu')]
    print(f"✅ GPU Simulator initialized: {sim.backend}")


def test_tensor_operations():
    """Test tensor operations on simulated GPU"""
    sim = get_gpu_simulator()
    
    # Create tensor
    x = torch.randn(100, 768)
    x_device = sim.to_device(x)
    
    # Verify device
    assert str(x_device.device) == str(sim.get_device())
    
    # Matrix multiplication
    with sim.inference_context():
        y = torch.matmul(x_device, x_device.T)
    
    assert y.shape == (100, 100)
    print("✅ Tensor operations work on simulated GPU")


def test_latency_simulation():
    """Test latency simulation"""
    sim = get_gpu_simulator()
    
    # Test different operations
    operations = ['inference', 'embedding', 'similarity']
    
    for op in operations:
        latency = sim.simulate_t4_latency(op)
        assert latency > 0
        assert latency < 10  # Should be under 10ms
        print(f"✅ {op}: {latency:.2f}ms (simulated T4 latency)")


def test_metrics_tracking():
    """Test metrics are tracked correctly"""
    sim = get_gpu_simulator()
    
    # Simulate some inferences
    for i in range(10):
        sim.simulate_t4_latency('inference')
        sim.update_metrics(2.5)
    
    metrics = sim.get_metrics()
    
    assert metrics['metrics']['total_inferences'] == 10
    assert metrics['metrics']['avg_latency_ms'] > 0
    print(f"✅ Metrics tracking works: {metrics['metrics']}")


def test_memory_info():
    """Test memory info retrieval"""
    sim = get_gpu_simulator()
    
    mem_info = sim.get_memory_info()
    
    assert 'total_gb' in mem_info
    assert mem_info['total_gb'] > 0
    print(f"✅ Memory info: {mem_info['total_gb']:.1f} GB available")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing GPU Simulation on MacBook")
    print("="*60 + "\n")
    
    test_gpu_simulator_initialization()
    test_tensor_operations()
    test_latency_simulation()
    test_metrics_tracking()
    test_memory_info()
    
    print("\n" + "="*60)
    print("✅ All tests passed! Ready for development")
    print("="*60 + "\n")
```

Run tests:

```bash
# Run GPU simulation tests
python tests/test_local_gpu_simulation.py

# Or with pytest
pytest tests/test_local_gpu_simulation.py -v -s
```

---

## 🚀 Next Steps

### You're now ready to:

1. ✅ **Develop locally** on MacBook with GPU simulation
2. ✅ **Test on Colab** with real T4 GPU (free!)
3. ✅ **Deploy to GCP** when code is production-ready

### Tomorrow (Day 4-5):

- [ ] Create `t4_neural_query_processor.py` (works with simulator)
- [ ] Create `hybrid_resource_scheduler.py` (auto-detects environment)
- [ ] Test personalization engine locally
- [ ] Upload to Colab for real GPU testing

---

## 📊 Development Environment Summary

```yaml
Local MacBook:
  - Device: CPU or MPS (Apple Silicon)
  - Purpose: Development, unit testing, integration testing
  - Advantages: Fast iteration, no cloud costs, offline work
  - Limitations: No real GPU, limited memory
  
Google Colab:
  - Device: NVIDIA T4 GPU (free!)
  - Purpose: GPU testing, model training, benchmarking
  - Advantages: Real GPU, free, easy sharing
  - Limitations: 12-hour sessions, limited storage
  
GCP Production:
  - Device: NVIDIA T4 GPU + C3 VM
  - Purpose: Production deployment
  - Advantages: 24/7 availability, scalable, monitored
  - Cost: ~$250-300/month
```

---

*Setup guide created: October 21, 2025*  
*Platform: macOS (Apple Silicon / Intel)*  
*Ready for: Local development with GPU simulation*

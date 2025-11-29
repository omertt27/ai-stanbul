#!/usr/bin/env python3
"""
vLLM Server Setup Helper for RunPod
This script helps setup and validate vLLM installation with Hugging Face
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✅ GPU detected")
            print(result.stdout)
            return True
        else:
            logger.error("❌ No GPU detected")
            return False
    except FileNotFoundError:
        logger.error("❌ nvidia-smi not found. Are you on a GPU instance?")
        return False

def install_packages():
    """Install required packages"""
    logger.info("📦 Installing required packages...")
    
    packages = [
        # Hugging Face
        "transformers",
        "huggingface-hub",
        "accelerate",
        "tokenizers",
        # vLLM
        "vllm",
        # Utilities
        "sentencepiece",
        "protobuf",
        "fastapi",
        "uvicorn[standard]",
    ]
    
    for package in packages:
        logger.info(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], 
                      check=False)
    
    logger.info("✅ Package installation complete")

def setup_huggingface():
    """Setup Hugging Face authentication"""
    logger.info("🔐 Setting up Hugging Face authentication...")
    
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        logger.warning("⚠️  HF_TOKEN not found in environment")
        logger.info("To use gated models like Llama, you need to:")
        logger.info("1. Get token from: https://huggingface.co/settings/tokens")
        logger.info("2. Set environment variable: export HF_TOKEN='your_token_here'")
        logger.info("3. Accept model license at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
        return False
    
    try:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=True)
        logger.info("✅ Hugging Face authentication successful")
        return True
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        return False

def verify_imports():
    """Verify all necessary imports work"""
    logger.info("🔍 Verifying imports...")
    
    imports_to_check = [
        ("transformers", "AutoTokenizer"),
        ("huggingface_hub", "login"),
        ("vllm", "LLM"),
        ("vllm.entrypoints.openai", "api_server"),
        ("torch", None),
    ]
    
    all_ok = True
    for module_name, obj_name in imports_to_check:
        try:
            module = __import__(module_name, fromlist=[obj_name] if obj_name else [])
            if obj_name:
                getattr(module, obj_name)
            logger.info(f"✅ {module_name} imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module_name}: {e}")
            all_ok = False
    
    return all_ok

def test_model_access(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """Test if we can access the model"""
    logger.info(f"🔍 Testing access to {model_name}...")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("✅ Model is accessible")
        logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
        return True
    except Exception as e:
        logger.error(f"❌ Cannot access model: {e}")
        logger.error("\nPossible issues:")
        logger.error("1. Model license not accepted on Hugging Face")
        logger.error("2. Invalid or missing HF_TOKEN")
        logger.error("3. No internet connection")
        return False

def create_vllm_start_script():
    """Create a startup script for vLLM"""
    script_path = Path("/workspace/start_vllm.sh")
    
    script_content = """#!/bin/bash
# vLLM Startup Script for RunPod

echo "🚀 Starting vLLM Server..."

# Set environment variables
export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Kill existing vLLM processes
echo "🔄 Stopping existing vLLM processes..."
pkill -9 -f vllm
sleep 2

# Start vLLM server
echo "▶️  Starting vLLM..."
python3 -m vllm.entrypoints.openai.api_server \\
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
  --port 8000 \\
  --host 0.0.0.0 \\
  --max-model-len 8192 \\
  --gpu-memory-utilization 0.85 \\
  --tensor-parallel-size 1 \\
  --trust-remote-code \\
  > /workspace/vllm_server.log 2>&1 &

echo "⏳ Server starting... (takes 60-120 seconds)"
echo "📋 Check logs: tail -f /workspace/vllm_server.log"
echo "🏥 Health check: curl http://localhost:8000/health"
"""
    
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        logger.info(f"✅ Startup script created at {script_path}")
        logger.info(f"Run with: {script_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create startup script: {e}")
        return False

def main():
    """Main setup routine"""
    logger.info("=" * 60)
    logger.info("🚀 RunPod vLLM Setup Helper")
    logger.info("=" * 60)
    
    # Check GPU
    if not check_gpu():
        logger.error("Setup cannot continue without GPU")
        return False
    
    # Install packages
    install_packages()
    
    # Verify imports
    if not verify_imports():
        logger.error("Some imports failed. Please check installation.")
        return False
    
    # Setup Hugging Face
    setup_huggingface()
    
    # Test model access
    test_model_access()
    
    # Create startup script
    create_vllm_start_script()
    
    logger.info("=" * 60)
    logger.info("✅ Setup complete!")
    logger.info("=" * 60)
    logger.info("\n📋 Next steps:")
    logger.info("1. Make sure HF_TOKEN is set: export HF_TOKEN='your_token'")
    logger.info("2. Start vLLM: /workspace/start_vllm.sh")
    logger.info("3. Wait 60-120 seconds for model to load")
    logger.info("4. Test: curl http://localhost:8000/health")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

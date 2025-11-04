"""
Optimized LLM Download and Test for Apple Silicon
Handles memory constraints intelligently with multiple fallback strategies
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class OptimizedLLMManager:
    """
    Intelligent LLM manager with memory-aware fallback strategies
    """
    
    # Models in order of preference (smallest to largest)
    MODELS = [
        {
            'name': 'LLaMA 3.2 1B',
            'id': 'meta-llama/Llama-3.2-1B-Instruct',
            'size_gb': 2.5,
            'recommended_vram_gb': 4,
        },
        {
            'name': 'LLaMA 3.2 3B',
            'id': 'meta-llama/Llama-3.2-3B-Instruct',
            'size_gb': 6,
            'recommended_vram_gb': 8,
        },
        {
            'name': 'LLaMA 3.1 8B',
            'id': 'meta-llama/Llama-3.1-8B-Instruct',
            'size_gb': 16,
            'recommended_vram_gb': 20,
        },
    ]
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.device = self._detect_device()
        self.available_memory = self._get_available_memory()
        
        logger.info(f"🖥️  Device: {self.device}")
        logger.info(f"💾 Available Memory: {self.available_memory:.2f} GB")
    
    def _detect_device(self) -> str:
        """Detect best available device"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _get_available_memory(self) -> float:
        """Get available memory in GB"""
        if self.device == "mps":
            # macOS Metal - conservative estimate
            # Get system memory and assume 70% available for MPS
            import subprocess
            try:
                result = subprocess.run(
                    ['sysctl', 'hw.memsize'],
                    capture_output=True,
                    text=True
                )
                total_memory = int(result.stdout.split()[1])
                # Convert to GB and use 70% as safe limit for MPS
                return (total_memory / (1024**3)) * 0.7
            except:
                return 12.0  # Conservative fallback
        elif self.device == "cuda":
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            # CPU - use system RAM
            import psutil
            return psutil.virtual_memory().available / (1024**3)
    
    def select_best_model(self) -> dict:
        """Select best model that fits in available memory"""
        logger.info(f"\n🔍 Selecting best model for {self.available_memory:.1f} GB memory...")
        
        for model_info in self.MODELS:
            if model_info['recommended_vram_gb'] <= self.available_memory:
                logger.info(f"✅ Selected: {model_info['name']}")
                logger.info(f"   Size: {model_info['size_gb']} GB")
                logger.info(f"   Recommended VRAM: {model_info['recommended_vram_gb']} GB")
                return model_info
        
        # If no model fits, use smallest with CPU fallback
        logger.warning("⚠️  No model fits in VRAM - will use CPU fallback")
        return self.MODELS[0]
    
    def download_model(self, model_info: dict) -> tuple:
        """
        Download model with optimizations
        
        Returns:
            (model, tokenizer) or (None, None) if failed
        """
        model_id = model_info['id']
        model_name = model_info['name']
        local_path = self.models_dir / model_name.lower().replace(' ', '-')
        
        logger.info(f"\n📥 Downloading {model_name}...")
        logger.info(f"   Model ID: {model_id}")
        logger.info(f"   Local Path: {local_path}")
        
        try:
            # Check if already downloaded
            if local_path.exists() and (local_path / "config.json").exists():
                logger.info(f"✅ Model already downloaded!")
                model_id = str(local_path)
            
            # Download tokenizer first (small)
            logger.info("📝 Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(local_path),
                trust_remote_code=True
            )
            
            # Download model with memory optimizations
            logger.info(f"🔄 Loading model (this may take a while)...")
            
            # Use optimal dtype based on device
            if self.device == "mps":
                # MPS works best with float32 for now
                dtype = torch.float32
                logger.info("   Using float32 for MPS compatibility")
            elif self.device == "cuda":
                # CUDA can use float16 for memory savings
                dtype = torch.float16
                logger.info("   Using float16 for CUDA optimization")
            else:
                # CPU uses float32
                dtype = torch.float32
                logger.info("   Using float32 for CPU")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=str(local_path),
                torch_dtype=dtype,
                low_cpu_mem_usage=True,  # Load directly to target device
                trust_remote_code=True,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            # Save model locally if downloaded from Hub
            if not local_path.exists():
                logger.info(f"💾 Saving model to {local_path}...")
                model.save_pretrained(local_path)
                tokenizer.save_pretrained(local_path)
            
            logger.info(f"✅ {model_name} downloaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Error downloading {model_name}: {e}")
            return None, None
    
    def test_model_inference(self, model, tokenizer, device_override=None):
        """
        Test model with actual inference
        
        Args:
            model: The loaded model
            tokenizer: The tokenizer
            device_override: Override device (for CPU fallback test)
        """
        device = device_override or self.device
        
        logger.info(f"\n🧪 Testing inference on {device}...")
        
        try:
            # Move model to device if needed
            if device_override:
                logger.info(f"   Moving model to {device_override}...")
                model = model.to(device_override)
            elif device == "mps":
                logger.info(f"   Moving model to MPS...")
                # Clear MPS cache first
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                model = model.to(device)
            
            # Test prompt
            test_prompt = "Istanbul is a beautiful city known for"
            
            logger.info(f"   Prompt: '{test_prompt}'")
            
            # Tokenize
            inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
            
            # Generate (short sequence to save memory)
            logger.info("   Generating response...")
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"\n✅ Inference successful!")
            logger.info(f"\n📝 Generated text:")
            logger.info(f"   {response}")
            
            return True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"❌ Out of memory on {device}: {e}")
                return False
            else:
                logger.error(f"❌ Runtime error: {e}")
                raise
        except Exception as e:
            logger.error(f"❌ Error during inference: {e}")
            raise


def main():
    """Main execution flow with intelligent fallbacks"""
    
    logger.info("=" * 80)
    logger.info("🚀 Optimized LLM Download & Test for Apple Silicon")
    logger.info("=" * 80)
    
    manager = OptimizedLLMManager()
    
    # Select best model
    model_info = manager.select_best_model()
    
    # Download model
    model, tokenizer = manager.download_model(model_info)
    
    if model is None:
        logger.error("❌ Failed to download model. Exiting.")
        return 1
    
    # Try inference on primary device (MPS)
    if manager.device == "mps":
        logger.info("\n" + "=" * 80)
        logger.info("🧪 Testing on MPS (Metal)")
        logger.info("=" * 80)
        
        try:
            success = manager.test_model_inference(model, tokenizer)
            
            if success:
                logger.info("\n" + "=" * 80)
                logger.info("✅ SUCCESS! Model works on MPS")
                logger.info("=" * 80)
                logger.info(f"\n📌 Model saved to: {manager.models_dir / model_info['name'].lower().replace(' ', '-')}")
                logger.info(f"🎯 Use this model in production with device='mps'")
                return 0
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("\n⚠️  MPS out of memory - trying CPU fallback...")
            else:
                raise
    
    # CPU Fallback
    logger.info("\n" + "=" * 80)
    logger.info("🧪 Testing on CPU (Fallback)")
    logger.info("=" * 80)
    
    try:
        success = manager.test_model_inference(model, tokenizer, device_override="cpu")
        
        if success:
            logger.info("\n" + "=" * 80)
            logger.info("✅ SUCCESS! Model works on CPU")
            logger.info("=" * 80)
            logger.info(f"\n📌 Model saved to: {manager.models_dir / model_info['name'].lower().replace(' ', '-')}")
            logger.info(f"⚠️  Note: CPU inference will be slower than MPS")
            logger.info(f"💡 Consider using a smaller model or cloud API for production")
            return 0
    
    except Exception as e:
        logger.error(f"\n❌ FAILED on CPU: {e}")
        return 1
    
    logger.error("\n❌ All attempts failed")
    return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

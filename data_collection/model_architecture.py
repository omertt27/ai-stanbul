"""
Istanbul Tourism Small LLM - Model Architecture Configuration
Week 3-4 Implementation - GPT-2 Medium (355M params) setup for domain-specific training
"""

import json
import torch
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for Istanbul Tourism domain-specific model"""
    
    # Model Architecture
    architecture: str = 'GPT-2 Medium'
    model_size: str = '355M'
    base_model: str = 'gpt2-medium'
    vocab_size: int = 50257  # GPT-2 default, will be customized
    
    # Domain-specific settings
    vocabulary: str = 'turkish_english_tourism_focused'
    context_length: int = 2048
    max_position_embeddings: int = 2048
    optimization_target: str = 'istanbul_tourism_accuracy'
    
    # Architecture parameters (GPT-2 Medium)
    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    n_positions: int = 2048
    
    # Training configuration
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    max_steps: int = 50000
    
    # Optimization settings
    use_gradient_checkpointing: bool = True
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Domain-specific vocabulary settings
    special_tokens: List[str] = None
    domain_vocabulary_size: int = 45000  # Reduced from 50K for domain focus
    
    # Quantization settings (for deployment)
    quantization_enabled: bool = True
    quantization_bits: int = 4
    use_onnx: bool = True
    
    # Turkish language support
    turkish_support: bool = True
    multilingual: bool = True
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [
                "<|istanbul|>",        # Istanbul context marker
                "<|attraction|>",      # Attraction information
                "<|restaurant|>",      # Restaurant/dining
                "<|transport|>",       # Transportation
                "<|culture|>",         # Cultural information
                "<|practical|>",       # Practical information
                "<|review|>",          # User review/opinion
                "<|recommendation|>",  # AI recommendation
                "<|turkish|>",         # Turkish language content
                "<|english|>",         # English language content
            ]

class ModelArchitecture:
    """Istanbul Tourism Model Architecture Manager"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_dir = Path("models/istanbul_tourism_gpt2")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def create_model_configuration(self) -> Dict[str, Any]:
        """Create detailed model configuration for training"""
        
        model_config = {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            
            # Architecture parameters
            "vocab_size": self.config.domain_vocabulary_size,
            "n_positions": self.config.n_positions,
            "n_embd": self.config.n_embd,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "n_inner": self.config.n_embd * 4,  # Standard GPT-2 ratio
            
            # Activation and normalization
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            
            # Generation settings
            "max_length": self.config.context_length,
            "use_cache": True,
            "pad_token_id": 50256,  # Will be updated with domain vocabulary
            "bos_token_id": 50256,
            "eos_token_id": 50256,
            
            # Domain-specific settings
            "domain": "istanbul_tourism",
            "languages": ["en", "tr"] if self.config.turkish_support else ["en"],
            "special_tokens": self.config.special_tokens,
            
            # Training metadata
            "created_date": datetime.now().isoformat(),
            "target_domain": "Istanbul Tourism Assistant",
            "base_model": self.config.base_model,
        }
        
        return model_config
    
    def create_training_configuration(self) -> Dict[str, Any]:
        """Create training configuration"""
        
        training_config = {
            # Model settings
            "model_name_or_path": self.config.base_model,
            "output_dir": str(self.model_dir),
            "overwrite_output_dir": True,
            
            # Training parameters
            "num_train_epochs": 3,
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "max_steps": self.config.max_steps,
            
            # Optimization
            "optim": "adamw_torch",
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            
            # Precision and efficiency
            "fp16": self.config.fp16,
            "gradient_checkpointing": self.config.use_gradient_checkpointing,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            
            # Evaluation and logging
            "evaluation_strategy": "steps",
            "eval_steps": 1000,
            "logging_steps": 100,
            "save_steps": 2000,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            
            # Early stopping
            "early_stopping_patience": 5,
            "early_stopping_threshold": 0.001,
            
            # Domain-specific settings
            "block_size": self.config.context_length,
            "mlm": False,  # Causal language modeling
            "dataset_name": "istanbul_tourism_corpus",
        }
        
        return training_config
    
    def create_vocabulary_configuration(self) -> Dict[str, Any]:
        """Create domain-specific vocabulary configuration"""
        
        vocab_config = {
            "base_tokenizer": "gpt2",
            "target_vocab_size": self.config.domain_vocabulary_size,
            
            # Domain-specific tokens to prioritize
            "domain_tokens": [
                # Turkish locations
                "Istanbul", "Sultanahmet", "Beyoğlu", "Galata", "Kadıköy", 
                "Üsküdar", "Beşiktaş", "Taksim", "Eminönü", "Fatih",
                
                # Attractions
                "Hagia", "Sophia", "Topkapi", "Mosque", "Palace", "Bazaar",
                "Bosphorus", "Galata", "Tower", "Cistern", "Museum",
                
                # Transportation
                "Metro", "Tram", "Ferry", "Bus", "Istanbulkart", "IETT",
                "Station", "Stop", "Route", "Schedule", "Fare",
                
                # Food & Dining
                "Turkish", "Kebab", "Baklava", "Döner", "Meze", "Raki",
                "Restaurant", "Café", "Breakfast", "Kahvaltı", "Lokum",
                
                # Cultural terms
                "Ottoman", "Byzantine", "Sultan", "Empire", "Heritage",
                "Traditional", "Historical", "Cultural", "Ancient",
                
                # Common Turkish words
                "ve", "ile", "için", "olan", "bu", "şu", "o", "çok",
                "güzel", "iyi", "var", "yok", "nasıl", "nerede",
            ],
            
            # Special tokens for domain
            "special_tokens_map": {
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "bos_token": "<|startoftext|>",
                "eos_token": "<|endoftext|>",
                "additional_special_tokens": self.config.special_tokens
            },
            
            # Language support
            "languages": ["en", "tr"] if self.config.turkish_support else ["en"],
            "multilingual_support": self.config.multilingual,
        }
        
        return vocab_config
    
    def create_deployment_configuration(self) -> Dict[str, Any]:
        """Create deployment configuration for production"""
        
        deployment_config = {
            # Model serving
            "model_format": "onnx" if self.config.use_onnx else "pytorch",
            "quantization": {
                "enabled": self.config.quantization_enabled,
                "bits": self.config.quantization_bits,
                "method": "dynamic" if self.config.quantization_bits == 8 else "static"
            },
            
            # Inference settings
            "max_length": 512,  # Shorter for fast inference
            "batch_size": 8,
            "num_beams": 1,  # Greedy decoding for speed
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            
            # Performance optimization
            "torch_dtype": "float16",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "use_cache": True,
            
            # Memory management
            "max_memory_gb": 12,  # Fits on RTX 4090
            "offload_folder": "models/offload",
            
            # API settings
            "api_timeout": 30,
            "max_concurrent_requests": 10,
            "response_timeout": 5,
        }
        
        return deployment_config
    
    def save_configurations(self):
        """Save all configurations to files"""
        
        configs = {
            "model_config.json": self.create_model_configuration(),
            "training_config.json": self.create_training_configuration(),
            "vocabulary_config.json": self.create_vocabulary_configuration(),
            "deployment_config.json": self.create_deployment_configuration(),
            "architecture_summary.json": asdict(self.config)
        }
        
        for filename, config in configs.items():
            config_path = self.model_dir / filename
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {filename} to {config_path}")
        
        # Create README with configuration summary
        self.create_readme()
        
        return configs
    
    def create_readme(self):
        """Create comprehensive README for the model"""
        
        readme_content = f"""# Istanbul Tourism GPT-2 Medium Model

## Model Overview
- **Architecture**: {self.config.architecture}
- **Parameters**: {self.config.model_size}
- **Base Model**: {self.config.base_model}
- **Domain**: Istanbul Tourism Assistant
- **Languages**: {'Turkish + English' if self.config.turkish_support else 'English'}

## Architecture Details
- **Embedding Dimensions**: {self.config.n_embd}
- **Attention Heads**: {self.config.n_head}
- **Layers**: {self.config.n_layer}
- **Context Length**: {self.config.context_length}
- **Vocabulary Size**: {self.config.domain_vocabulary_size}

## Training Configuration
- **Batch Size**: {self.config.batch_size}
- **Learning Rate**: {self.config.learning_rate}
- **Max Steps**: {self.config.max_steps}
- **Optimization Target**: {self.config.optimization_target}

## Special Tokens
The model uses domain-specific special tokens:
{chr(10).join(f"- `{token}`" for token in self.config.special_tokens)}

## Deployment
- **Quantization**: {'Enabled' if self.config.quantization_enabled else 'Disabled'} ({self.config.quantization_bits}-bit)
- **ONNX Export**: {'Enabled' if self.config.use_onnx else 'Disabled'}
- **Memory Requirements**: ~6-12GB GPU memory
- **Target Latency**: <200ms per query

## Usage
The model is specifically trained for Istanbul tourism queries and can handle:
- Attraction recommendations
- Restaurant suggestions
- Transportation guidance
- Cultural information
- Practical travel advice

## Files
- `model_config.json`: HuggingFace model configuration
- `training_config.json`: Training hyperparameters
- `vocabulary_config.json`: Tokenizer and vocabulary settings
- `deployment_config.json`: Production deployment settings
- `architecture_summary.json`: Complete configuration summary

## Training Data Requirements
- Target training data: 50,000+ Istanbul tourism Q&A pairs
- Languages: English and Turkish
- Content types: Attractions, dining, transport, culture, reviews

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_path = self.model_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"Created README at {readme_path}")

def main():
    """Setup Istanbul Tourism model architecture"""
    
    print("=" * 60)
    print("🤖 ISTANBUL TOURISM MODEL ARCHITECTURE SETUP")
    print("=" * 60)
    print(f"📅 Week 3-4 Implementation - {datetime.now().strftime('%Y-%m-%d')}")
    
    # Create model configuration
    config = ModelConfig()
    
    print(f"\n📋 Model Configuration:")
    print(f"   Architecture: {config.architecture}")
    print(f"   Parameters: {config.model_size}")
    print(f"   Context Length: {config.context_length}")
    print(f"   Vocabulary: {config.domain_vocabulary_size} tokens")
    print(f"   Languages: {'Turkish + English' if config.turkish_support else 'English'}")
    print(f"   Quantization: {'Enabled' if config.quantization_enabled else 'Disabled'}")
    
    # Setup architecture manager
    architecture = ModelArchitecture(config)
    
    print(f"\n⚙️ Creating configurations...")
    configs = architecture.save_configurations()
    
    print(f"\n✅ Configuration files created:")
    for filename in configs.keys():
        print(f"   📄 {filename}")
    
    print(f"\n📂 Model directory: {architecture.model_dir}")
    
    # Display next steps
    print(f"\n🚀 Next Steps (Week 5-8):")
    print("   1. Prepare training data (50K+ Istanbul tourism examples)")
    print("   2. Set up distillation from Llama-3.1-8B-Instruct")
    print("   3. Configure training environment (GPU cluster)")
    print("   4. Begin model training with domain-specific data")
    print("   5. Monitor training metrics and adjust hyperparameters")
    
    print(f"\n💡 Training Command Preview:")
    print("   python train_istanbul_model.py \\")
    print("     --config_file models/istanbul_tourism_gpt2/training_config.json \\")
    print("     --data_dir data/training \\")
    print("     --output_dir models/istanbul_tourism_gpt2 \\")
    print("     --do_train --do_eval")
    
    print("=" * 60)
    print("✅ ARCHITECTURE SETUP COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()

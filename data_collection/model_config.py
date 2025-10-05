"""
Istanbul Tourism Domain-Specific Model Configuration
GPT-2 Medium (355M parameters) optimized for Turkish-English tourism queries
"""

import torch
from transformers import GPT2Config, GPT2Tokenizer
from typing import Dict, Any, List
import json
import os

class IstanbulModelConfig:
    """Configuration class for Istanbul tourism domain-specific model"""
    
    def __init__(self):
        self.model_config = {
            'architecture': 'GPT-2 Medium',
            'base_model': 'gpt2-medium',
            'parameters': '355M',
            'vocabulary_size': 50400,  # Extended for Turkish + tourism terms
            'context_length': 2048,
            'hidden_size': 1024,
            'num_layers': 24,
            'num_heads': 16,
            'intermediate_size': 4096,
            'optimization_target': 'istanbul_tourism_accuracy',
            'languages': ['turkish', 'english'],
            'domain': 'istanbul_tourism'
        }
        
        # Istanbul-specific vocabulary extensions
        self.istanbul_vocabulary = [
            # Districts and Areas
            'sultanahmet', 'beyoğlu', 'galata', 'karaköy', 'beşiktaş', 
            'ortaköy', 'taksim', 'kadıköy', 'üsküdar', 'fatih',
            'eminönü', 'bakırköy', 'şişli', 'pendik', 'maltepe',
            
            # Transportation
            'metro', 'metrobüs', 'dolmuş', 'vapur', 'tramvay',
            'marmaray', 'istanbulkart', 'otobüs', 'taksi', 'uber',
            'bilet', 'aktarma', 'durak', 'istasyon', 'hattı',
            
            # Attractions
            'ayasofya', 'sultanahmet', 'topkapı', 'kapalıçarşı',
            'galata kulesi', 'boğaz', 'büyük mecidiye', 'dolmabahçe',
            'basilica cistern', 'hagia sophia', 'blue mosque',
            'grand bazaar', 'spice bazaar', 'bosphorus',
            
            # Food and Dining
            'kebap', 'döner', 'lahmacun', 'pide', 'simit',
            'çay', 'kahve', 'baklava', 'lokum', 'meze',
            'rakı', 'balık ekmek', 'kokoreç', 'midye dolma',
            'künefe', 'mantı', 'börek', 'çorba', 'pilav',
            
            # Cultural Terms
            'hamam', 'cami', 'müze', 'saray', 'köprü',
            'medrese', 'türbe', 'han', 'bedesten', 'çeşme',
            'Ottoman', 'Byzantine', 'Turkish', 'Islamic',
            
            # Tourism Terms
            'turist', 'rehber', 'tur', 'bilet', 'müze kart',
            'otel', 'pansiyon', 'hostel', 'rezervasyon',
            'tatil', 'gezi', 'fotoğraf', 'alışveriş', 'hediyelik'
        ]
        
        # Special tokens for conversation flow
        self.special_tokens = {
            'pad_token': '<|pad|>',
            'bos_token': '<|startoftext|>',
            'eos_token': '<|endoftext|>',
            'unk_token': '<|unk|>',
            'user_token': '<|user|>',
            'assistant_token': '<|assistant|>',
            'context_token': '<|context|>',
            'location_token': '<|location|>',
            'time_token': '<|time|>',
            'preference_token': '<|preference|>'
        }
    
    def get_model_config(self) -> GPT2Config:
        """Get GPT2Config object with Istanbul-specific settings"""
        config = GPT2Config(
            vocab_size=self.model_config['vocabulary_size'],
            n_positions=self.model_config['context_length'],
            n_embd=self.model_config['hidden_size'],
            n_layer=self.model_config['num_layers'],
            n_head=self.model_config['num_heads'],
            n_inner=self.model_config['intermediate_size'],
            activation_function="gelu_new",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            pad_token_id=50257
        )
        return config
    
    def create_tokenizer(self, save_directory: str = "./tokenizer") -> GPT2Tokenizer:
        """Create and extend tokenizer with Istanbul-specific vocabulary"""
        # Load base GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        
        # Add special tokens
        special_tokens_dict = {
            'pad_token': self.special_tokens['pad_token'],
            'additional_special_tokens': list(self.special_tokens.values())
        }
        tokenizer.add_special_tokens(special_tokens_dict)
        
        # Add Istanbul-specific vocabulary
        tokenizer.add_tokens(self.istanbul_vocabulary)
        
        # Save extended tokenizer
        os.makedirs(save_directory, exist_ok=True)
        tokenizer.save_pretrained(save_directory)
        
        print(f"Extended tokenizer saved to {save_directory}")
        print(f"Vocabulary size: {len(tokenizer)}")
        print(f"Added Istanbul terms: {len(self.istanbul_vocabulary)}")
        
        return tokenizer
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for the Istanbul model"""
        return {
            'model_name': 'istanbul-tourism-gpt2-medium',
            'base_model': 'gpt2-medium',
            'max_length': self.model_config['context_length'],
            'batch_size': 8,
            'learning_rate': 2e-5,
            'warmup_steps': 500,
            'max_steps': 10000,
            'eval_steps': 500,
            'save_steps': 1000,
            'gradient_accumulation_steps': 4,
            'fp16': True,
            'dataloader_drop_last': True,
            'remove_unused_columns': False,
            'label_names': ['labels'],
            'prediction_loss_only': True,
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,
            'optim': 'adamw_torch',
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'adam_epsilon': 1e-8,
            'weight_decay': 0.01,
            'lr_scheduler_type': 'cosine',
            'num_train_epochs': 3,
            'max_grad_norm': 1.0,
            'seed': 42,
            'data_seed': 42,
            'tf32': True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False
        }
    
    def get_quantization_config(self) -> Dict[str, Any]:
        """Get quantization configuration for deployment optimization"""
        return {
            'quantization_method': 'GPTQ',  # or 'AWQ'
            'bits': 4,
            'group_size': 128,
            'desc_act': False,
            'disable_exllama': False,
            'model_seqlen': self.model_config['context_length'],
            'cache_examples_on_gpu': True,
            'use_triton': True,
            'warmup_autotune': True,
            'fuse_layers': True
        }
    
    def save_config(self, save_directory: str = "./model_config"):
        """Save all configuration to files"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model config
        with open(f"{save_directory}/model_config.json", 'w', encoding='utf-8') as f:
            json.dump(self.model_config, f, indent=2, ensure_ascii=False)
        
        # Save training config
        training_config = self.get_training_config()
        with open(f"{save_directory}/training_config.json", 'w') as f:
            json.dump(training_config, f, indent=2)
        
        # Save quantization config
        quant_config = self.get_quantization_config()
        with open(f"{save_directory}/quantization_config.json", 'w') as f:
            json.dump(quant_config, f, indent=2)
        
        # Save vocabulary extensions
        with open(f"{save_directory}/istanbul_vocabulary.json", 'w', encoding='utf-8') as f:
            json.dump(self.istanbul_vocabulary, f, indent=2, ensure_ascii=False)
        
        # Save special tokens
        with open(f"{save_directory}/special_tokens.json", 'w', encoding='utf-8') as f:
            json.dump(self.special_tokens, f, indent=2, ensure_ascii=False)
        
        print(f"All configurations saved to {save_directory}")
    
    def estimate_requirements(self) -> Dict[str, str]:
        """Estimate hardware and memory requirements"""
        return {
            'model_size_fp16': '710 MB',
            'model_size_4bit': '178 MB',
            'training_memory_required': '12-16 GB GPU',
            'inference_memory_fp16': '1-2 GB GPU',
            'inference_memory_4bit': '512 MB GPU',
            'recommended_gpu_training': 'RTX 3090/4090, A100',
            'recommended_gpu_inference': 'RTX 3060 or better',
            'cpu_inference_possible': 'Yes (slower)',
            'context_memory_per_token': '2 bytes (fp16), 0.5 bytes (4bit)',
            'max_context_memory': '8 MB (fp16), 2 MB (4bit)'
        }
    
    def print_summary(self):
        """Print configuration summary"""
        print("=" * 60)
        print("ISTANBUL TOURISM MODEL CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Architecture: {self.model_config['architecture']}")
        print(f"Parameters: {self.model_config['parameters']}")
        print(f"Context Length: {self.model_config['context_length']}")
        print(f"Vocabulary Size: {self.model_config['vocabulary_size']}")
        print(f"Languages: {', '.join(self.model_config['languages'])}")
        print(f"Domain Focus: {self.model_config['domain']}")
        print(f"Istanbul Terms Added: {len(self.istanbul_vocabulary)}")
        print(f"Special Tokens: {len(self.special_tokens)}")
        print()
        
        requirements = self.estimate_requirements()
        print("HARDWARE REQUIREMENTS:")
        print("-" * 30)
        for key, value in requirements.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 60)

def main():
    """Main function to create and save Istanbul model configuration"""
    print("Creating Istanbul Tourism Model Configuration...")
    
    # Initialize configuration
    config = IstanbulModelConfig()
    
    # Print summary
    config.print_summary()
    
    # Save all configurations
    config.save_config("./istanbul_model_config")
    
    # Create and save extended tokenizer
    tokenizer = config.create_tokenizer("./istanbul_tokenizer")
    
    # Save the actual model config object
    model_config = config.get_model_config()
    model_config.save_pretrained("./istanbul_model_config")
    
    print("\nConfiguration setup complete!")
    print("Next steps:")
    print("1. Prepare training data using training_data_formatter.py")
    print("2. Run model training using the generated configurations")
    print("3. Quantize model for deployment using quantization_config.json")

if __name__ == "__main__":
    main()

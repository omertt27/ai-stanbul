"""
Training Environment Setup for Istanbul Tourism Model
Handles environment setup, dependencies, and training scripts
"""

import os
import subprocess
import sys
import json
import platform
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingEnvironmentSetup:
    """Setup training environment for Istanbul tourism model"""
    
    def __init__(self, base_dir: str = "./training_environment"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.model_dir = self.base_dir / "models" / "istanbul_tourism_gpt2"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Training requirements
        self.training_requirements = [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "peft>=0.6.0",  # For LoRA fine-tuning
            "bitsandbytes>=0.41.0",  # For quantization
            "auto-gptq>=0.4.0",  # For GPTQ quantization
            "optimum>=1.14.0",  # For quantization optimization
            "wandb>=0.15.0",  # For experiment tracking
            "tensorboard>=2.14.0",  # For logging
            "scikit-learn>=1.3.0",  # For evaluation metrics
            "nltk>=3.8.0",  # For text processing
            "rouge-score>=0.1.2",  # For evaluation
            "sacrebleu>=2.3.0",  # For BLEU scores
            "psutil>=5.9.0",  # For monitoring
            "tqdm>=4.65.0",  # For progress bars
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0"
        ]
        
        # Optional GPU acceleration
        self.gpu_requirements = [
            "torch-audio",  # GPU audio processing
            "torchvision",  # GPU vision processing
            "xformers",  # Memory efficient attention
            "flash-attn>=2.3.0"  # Flash attention
        ]
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system capabilities and requirements"""
        info = {
            'python_version': sys.version,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            'total_gpu_memory': [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
            'recommended_setup': self._get_recommended_setup()
        }
        
        # Convert memory to GB
        if info['total_gpu_memory']:
            info['total_gpu_memory_gb'] = [mem / (1024**3) for mem in info['total_gpu_memory']]
        
        return info
    
    def _get_recommended_setup(self) -> Dict[str, str]:
        """Get recommended setup based on available hardware"""
        if not torch.cuda.is_available():
            return {
                'training_mode': 'CPU only (very slow)',
                'batch_size': '1-2',
                'gradient_accumulation': '8-16',
                'quantization': 'Not recommended',
                'estimated_time': '5-10 days'
            }
        
        gpu_memory = max([torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]) / (1024**3)
        
        if gpu_memory >= 24:  # RTX 4090, A100
            return {
                'training_mode': 'Full precision + gradient checkpointing',
                'batch_size': '4-8',
                'gradient_accumulation': '2-4',
                'quantization': 'Optional for deployment',
                'estimated_time': '6-12 hours'
            }
        elif gpu_memory >= 12:  # RTX 3090, 4080
            return {
                'training_mode': 'Mixed precision (fp16)',
                'batch_size': '2-4',
                'gradient_accumulation': '4-8',
                'quantization': 'Recommended',
                'estimated_time': '12-24 hours'
            }
        elif gpu_memory >= 8:  # RTX 3070, 4060 Ti
            return {
                'training_mode': 'LoRA fine-tuning + quantization',
                'batch_size': '1-2',
                'gradient_accumulation': '8-16',
                'quantization': 'Required',
                'estimated_time': '1-2 days'
            }
        else:  # Lower memory GPUs
            return {
                'training_mode': 'CPU + small GPU assistance',
                'batch_size': '1',
                'gradient_accumulation': '16-32',
                'quantization': 'Required',
                'estimated_time': '2-5 days'
            }
    
    def create_training_requirements(self):
        """Create requirements.txt file for training environment"""
        requirements = """# Core ML and Deep Learning
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
datasets>=2.12.0
peft>=0.6.0

# Quantization and optimization
bitsandbytes>=0.41.0
auto-gptq>=0.4.0
optimum>=1.14.0
flash-attn>=2.0.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
datasets>=2.12.0

# Text processing
sentencepiece>=0.1.99
sacremoses>=0.0.53
langdetect>=1.0.9

# Turkish language support
turkish-stemmer>=1.3.0
zeyrek>=0.1.2

# Evaluation metrics
evaluate>=0.4.0
rouge-score>=0.1.2
bleu>=0.1.0
sacrebleu>=2.3.0

# Distributed training
fairscale>=0.4.13

# Monitoring and logging
wandb>=0.15.0
tensorboard>=2.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Development tools
jupyter>=1.0.0
ipywidgets>=8.0.0
tqdm>=4.65.0

# Configuration management
hydra-core>=1.3.0
omegaconf>=2.3.0

# Model serving (for testing)
fastapi>=0.100.0
uvicorn>=0.22.0

# Utilities
psutil>=5.9.0
nltk>=3.8.0
"""
        
        req_path = self.model_dir / "requirements.txt"
        req_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(req_path, 'w') as f:
            f.write(requirements)
        
        logger.info(f"Created training requirements at {req_path}")
        return req_path
    
    def create_training_scripts(self):
        """Create training and evaluation scripts"""
        
        # Main training script
        training_script = '''#!/usr/bin/env python3
"""
Istanbul Tourism Model Training Script
Distillation training from Llama-3.1-8B to GPT-2 Medium
"""

import os
import json
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
import wandb
from pathlib import Path

def setup_model_and_tokenizer(config_path):
    """Setup model and tokenizer with domain-specific configuration"""
    
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # Load base GPT-2 Medium model
    config = GPT2Config.from_pretrained('gpt2-medium')
    
    # Update with domain-specific settings
    config.vocab_size = model_config['vocab_size']
    config.n_positions = model_config['n_positions']
    config.n_embd = model_config['n_embd']
    config.n_layer = model_config['n_layer']
    config.n_head = model_config['n_head']
    
    # Initialize model
    model = GPT2LMHeadModel(config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    
    # Add special tokens
    special_tokens = model_config.get('special_tokens', [])
    if special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer, config

def load_training_data(data_dir):
    """Load Istanbul tourism training data"""
    
    data_files = {
        'train': str(Path(data_dir) / 'qa_training_data.jsonl'),
        'validation': str(Path(data_dir) / 'instruction_training_data.jsonl')
    }
    
    dataset = load_dataset('json', data_files=data_files)
    return dataset

def main():
    # Initialize Weights & Biases
    wandb.init(project="istanbul-tourism-gpt2", name="distillation-training")
    
    # Load configuration
    config_path = "models/istanbul_tourism_gpt2/model_config.json"
    model, tokenizer, model_config = setup_model_and_tokenizer(config_path)
    
    # Load training data
    dataset = load_training_data("data/training")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/istanbul_tourism_gpt2/checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=1000,
        logging_steps=100,
        save_steps=2000,
        eval_steps=1000,
        evaluation_strategy="steps",
        fp16=True,
        dataloader_num_workers=4,
        report_to="wandb"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is autoregressive
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    trainer.train()
    
    # Save model
    trainer.save_model("models/istanbul_tourism_gpt2/final")
    tokenizer.save_pretrained("models/istanbul_tourism_gpt2/final")

if __name__ == "__main__":
    main()
'''
        
        # Evaluation script
        evaluation_script = '''#!/usr/bin/env python3
"""
Istanbul Tourism Model Evaluation Script
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
from pathlib import Path
from datasets import load_dataset
import numpy as np
from rouge_score import rouge_scorer
from evaluate import load

def load_model_and_tokenizer(model_path):
    """Load trained Istanbul tourism model"""
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

def evaluate_model(model, tokenizer, test_data):
    """Evaluate model on Istanbul tourism tasks"""
    
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    bleu = load("bleu")
    
    results = {
        'rouge_scores': [],
        'bleu_scores': [],
        'perplexity': [],
        'domain_accuracy': 0
    }
    
    model.eval()
    with torch.no_grad():
        for example in test_data:
            input_text = example['input']
            target_text = example['output']
            
            # Generate response
            inputs = tokenizer.encode(input_text, return_tensors='pt')
            outputs = model.generate(
                inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(input_text):].strip()
            
            # Calculate metrics
            rouge_scores = rouge.score(target_text, response)
            results['rouge_scores'].append(rouge_scores)
            
            # Calculate perplexity
            target_tokens = tokenizer.encode(target_text, return_tensors='pt')
            loss = model(target_tokens, labels=target_tokens).loss
            perplexity = torch.exp(loss).item()
            results['perplexity'].append(perplexity)
    
    # Aggregate results
    avg_rouge1 = np.mean([s['rouge1'].fmeasure for s in results['rouge_scores']])
    avg_rouge2 = np.mean([s['rouge2'].fmeasure for s in results['rouge_scores']])
    avg_rougeL = np.mean([s['rougeL'].fmeasure for s in results['rouge_scores']])
    avg_perplexity = np.mean(results['perplexity'])
    
    return {
        'rouge1': avg_rouge1,
        'rouge2': avg_rouge2,
        'rougeL': avg_rougeL,
        'perplexity': avg_perplexity
    }

def main():
    model_path = "models/istanbul_tourism_gpt2/final"
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Load test data
    test_data = load_dataset('json', data_files='data/training/test_data.jsonl')['train']
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, test_data)
    
    print("\\n=== Istanbul Tourism Model Evaluation Results ===")
    print(f"ROUGE-1 F1: {results['rouge1']:.4f}")
    print(f"ROUGE-2 F1: {results['rouge2']:.4f}")
    print(f"ROUGE-L F1: {results['rougeL']:.4f}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    
    # Save results
    with open("models/istanbul_tourism_gpt2/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
'''
        
        # Save scripts
        scripts = {
            "train_istanbul_model.py": training_script,
            "evaluate_istanbul_model.py": evaluation_script
        }
        
        for filename, content in scripts.items():
            script_path = self.model_dir / filename
            with open(script_path, 'w') as f:
                f.write(content)
            
            # Make executable
            script_path.chmod(0o755)
            logger.info(f"Created training script: {script_path}")
        
        return scripts
    
    def create_distillation_config(self):
        """Create knowledge distillation configuration"""
        
        distillation_config = {
            "teacher_model": {
                "name": "meta-llama/Llama-2-7b-chat-hf",  # Free alternative to Llama-3.1-8B
                "load_in_8bit": True,
                "device_map": "auto"
            },
            
            "student_model": {
                "name": "gpt2-medium",
                "config_path": "models/istanbul_tourism_gpt2/model_config.json"
            },
            
            "distillation_params": {
                "temperature": 3.0,
                "alpha_distillation": 0.7,
                "alpha_ground_truth": 0.3,
                "alpha_cosine": 0.1
            },
            
            "training_data": {
                "train_file": "data/training/qa_training_data.jsonl",
                "validation_file": "data/training/instruction_training_data.jsonl",
                "max_train_samples": 50000,
                "max_eval_samples": 5000
            },
            
            "training_args": {
                "output_dir": "models/istanbul_tourism_gpt2/distillation",
                "num_train_epochs": 5,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "learning_rate": 3e-5,
                "warmup_ratio": 0.1,
                "fp16": True,
                "dataloader_num_workers": 4,
                "evaluation_strategy": "steps",
                "eval_steps": 500,
                "save_steps": 1000,
                "logging_steps": 50,
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False
            }
        }
        
        config_path = self.model_dir / "distillation_config.json"
        with open(config_path, 'w') as f:
            json.dump(distillation_config, f, indent=2)
        
        logger.info(f"Created distillation config: {config_path}")
        return config_path
    
    def setup_gpu_environment(self):
        """Check and setup GPU environment"""
        
        gpu_info = {
            "cuda_available": torch.cuda.is_available() if 'torch' in sys.modules else False,
            "gpu_count": torch.cuda.device_count() if 'torch' in sys.modules and torch.cuda.is_available() else 0,
            "gpu_names": [],
            "total_memory_gb": 0,
            "recommendations": []
        }
        
        if gpu_info["cuda_available"]:
            for i in range(gpu_info["gpu_count"]):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_info["gpu_names"].append(gpu_name)
                gpu_info["total_memory_gb"] += gpu_memory
            
            # Provide recommendations based on available GPU memory
            if gpu_info["total_memory_gb"] >= 24:
                gpu_info["recommendations"].append("✅ Excellent GPU setup for training GPT-2 Medium")
                gpu_info["recommendations"].append("💡 Can use larger batch sizes and full precision training")
            elif gpu_info["total_memory_gb"] >= 12:
                gpu_info["recommendations"].append("✅ Good GPU setup for training")
                gpu_info["recommendations"].append("💡 Recommend using gradient checkpointing and fp16")
            elif gpu_info["total_memory_gb"] >= 8:
                gpu_info["recommendations"].append("⚠️ Limited GPU memory - use small batch sizes")
                gpu_info["recommendations"].append("💡 Enable gradient checkpointing, fp16, and DeepSpeed ZeRO")
            else:
                gpu_info["recommendations"].append("❌ Insufficient GPU memory for efficient training")
                gpu_info["recommendations"].append("💡 Consider using cloud GPU instances or CPU training")
        else:
            gpu_info["recommendations"].append("❌ No CUDA-capable GPU detected")
            gpu_info["recommendations"].append("💡 Training will be very slow on CPU")
        
        return gpu_info
    
    def create_environment_summary(self):
        """Create comprehensive environment summary"""
        
        summary = {
            "setup_date": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "model_architecture": "GPT-2 Medium (355M parameters)",
            "target_domain": "Istanbul Tourism",
            "training_approach": "Knowledge Distillation",
            
            "files_created": [
                "training_requirements.txt",
                "train_istanbul_model.py", 
                "evaluate_istanbul_model.py",
                "distillation_config.json",
                "environment_summary.json"
            ],
            
            "next_steps": [
                "Install training requirements: pip install -r training_requirements.txt",
                "Prepare training data (Week 1-2 output)",
                "Configure Weights & Biases for monitoring",
                "Run training: python train_istanbul_model.py", 
                "Monitor training progress and adjust hyperparameters",
                "Evaluate model performance on test set",
                "Optimize model for deployment (quantization, ONNX)"
            ],
            
            "estimated_timeline": {
                "data_preparation": "1-2 days",
                "environment_setup": "0.5 days", 
                "initial_training": "3-5 days",
                "hyperparameter_tuning": "2-3 days",
                "evaluation_and_optimization": "1-2 days",
                "total": "7-12 days"
            }
        }
        
        # Add GPU information
        try:
            import torch
            summary["gpu_info"] = self.setup_gpu_environment()
        except ImportError:
            summary["gpu_info"] = {"error": "PyTorch not installed yet"}
        
        return summary

def main():
    """Setup training environment for Istanbul Tourism model"""
    
    print("=" * 70)
    print("🛠️ ISTANBUL TOURISM MODEL TRAINING ENVIRONMENT SETUP")
    print("=" * 70)
    print(f"📅 Week 3-4 Implementation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup = TrainingEnvironmentSetup()
    
    print(f"\\n📂 Setting up training infrastructure...")
    
    # Create requirements file
    print("   📦 Creating training requirements...")
    req_path = setup.create_training_requirements()
    
    # Create training scripts
    print("   🧠 Creating training scripts...")
    scripts = setup.create_training_scripts()
    
    # Create distillation configuration
    print("   🎯 Creating distillation configuration...")
    distill_config = setup.create_distillation_config()
    
    # Create environment summary
    print("   📋 Generating environment summary...")
    summary = setup.create_environment_summary()
    
    # Save summary
    summary_path = setup.model_dir / "environment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\n✅ Training environment setup complete!")
    print(f"\\n📁 Files created in {setup.model_dir}:")
    for filename in summary["files_created"]:
        print(f"   📄 {filename}")
    
    # Display GPU information if available
    if "gpu_info" in summary and "recommendations" in summary["gpu_info"]:
        print(f"\\n🖥️ GPU Environment:")
        if summary["gpu_info"]["cuda_available"]:
            print(f"   GPU Count: {summary['gpu_info']['gpu_count']}")
            print(f"   Total Memory: {summary['gpu_info']['total_memory_gb']:.1f} GB")
        
        for rec in summary["gpu_info"]["recommendations"]:
            print(f"   {rec}")
    
    print(f"\\n⏱️ Estimated Timeline:")
    for phase, duration in summary["estimated_timeline"].items():
        if phase != "total":
            print(f"   {phase.replace('_', ' ').title()}: {duration}")
    print(f"   ➡️ Total Estimated Time: {summary['estimated_timeline']['total']}")
    
    print(f"\\n🚀 Next Steps:")
    for i, step in enumerate(summary["next_steps"], 1):
        print(f"   {i}. {step}")
    
    print("=" * 70)
    print("✅ TRAINING ENVIRONMENT READY!")
    print("=" * 70)

if __name__ == "__main__":
    main()

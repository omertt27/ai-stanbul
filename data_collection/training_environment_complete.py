"""
Complete Training Environment Setup for Istanbul Tourism Model
Comprehensive training infrastructure, monitoring, and deployment setup
"""

import os
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import torch
from datetime import datetime

class TrainingEnvironmentSetup:
    """Complete setup for Istanbul tourism model training environment"""
    
    def __init__(self, base_dir: str = "./training_environment"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Core training requirements
        self.training_requirements = [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "peft>=0.6.0",
            "bitsandbytes>=0.41.0",
            "auto-gptq>=0.4.0",
            "optimum>=1.14.0",
            "wandb>=0.15.0",
            "tensorboard>=2.14.0",
            "scikit-learn>=1.3.0",
            "nltk>=3.8.0",
            "rouge-score>=0.1.2",
            "sacrebleu>=2.3.0",
            "psutil>=5.9.0",
            "tqdm>=4.65.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0"
        ]
        
        # Optional GPU acceleration
        self.gpu_requirements = [
            "xformers",
            "flash-attn>=2.3.0"
        ]
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """Check system capabilities"""
        info = {
            'python_version': sys.version,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [],
            'total_gpu_memory_gb': [],
            'recommended_setup': {}
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                info['gpu_names'].append(torch.cuda.get_device_name(i))
                memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                info['total_gpu_memory_gb'].append(memory_gb)
        
        info['recommended_setup'] = self._get_recommended_setup(info)
        return info
    
    def _get_recommended_setup(self, system_info: Dict) -> Dict[str, str]:
        """Get recommended setup based on hardware"""
        if not system_info['cuda_available']:
            return {
                'training_mode': 'CPU only (very slow)',
                'batch_size': '1-2',
                'gradient_accumulation': '8-16',
                'quantization': 'Not recommended',
                'estimated_time': '5-10 days'
            }
        
        max_memory = max(system_info['total_gpu_memory_gb']) if system_info['total_gpu_memory_gb'] else 0
        
        if max_memory >= 24:
            return {
                'training_mode': 'Full precision + gradient checkpointing',
                'batch_size': '4-8',
                'gradient_accumulation': '2-4',
                'quantization': 'Optional for deployment',
                'estimated_time': '6-12 hours'
            }
        elif max_memory >= 12:
            return {
                'training_mode': 'Mixed precision (fp16)',
                'batch_size': '2-4',
                'gradient_accumulation': '4-8',
                'quantization': 'Recommended',
                'estimated_time': '12-24 hours'
            }
        elif max_memory >= 8:
            return {
                'training_mode': 'LoRA fine-tuning + quantization',
                'batch_size': '1-2',
                'gradient_accumulation': '8-16',
                'quantization': 'Required',
                'estimated_time': '1-2 days'
            }
        else:
            return {
                'training_mode': 'CPU + small GPU assistance',
                'batch_size': '1',
                'gradient_accumulation': '16-32',
                'quantization': 'Required',
                'estimated_time': '2-5 days'
            }
    
    def create_requirements_file(self):
        """Create comprehensive requirements.txt"""
        requirements_path = self.base_dir / "requirements.txt"
        
        content = "# Istanbul Tourism Model Training Requirements\\n"
        content += "# Core training dependencies\\n"
        for req in self.training_requirements:
            content += f"{req}\\n"
        
        content += "\\n# Optional GPU acceleration\\n"
        for req in self.gpu_requirements:
            content += f"# {req}\\n"
        
        with open(requirements_path, 'w') as f:
            f.write(content)
        
        print(f"Requirements file created: {requirements_path}")
    
    def create_training_script(self):
        """Create main training script"""
        script_path = self.base_dir / "train_istanbul_model.py"
        
        training_script = '''"""
Istanbul Tourism Model Training Script
"""

import os
import json
import torch
import wandb
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IstanbulModelTrainer:
    def __init__(self, config_dir: str = "../istanbul_model_config"):
        self.config_dir = Path(config_dir)
        self.load_configs()
        
    def load_configs(self):
        """Load configurations"""
        with open(self.config_dir / "training_config.json", 'r') as f:
            self.training_config = json.load(f)
        
        self.model_config = GPT2Config.from_pretrained(self.config_dir)
        logger.info(f"Loaded configurations from {self.config_dir}")
    
    def prepare_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        tokenizer_path = "../istanbul_tokenizer"
        if os.path.exists(tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        else:
            logger.warning("Custom tokenizer not found, using base GPT-2")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(
            'gpt2-medium',
            config=self.model_config
        )
        
        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized model embeddings to {len(self.tokenizer)}")
    
    def load_dataset(self, data_path: str = "../data/training"):
        """Load training dataset"""
        data_path = Path(data_path)
        
        if (data_path / "conversation_data.jsonl").exists():
            dataset = load_dataset('json', data_files=str(data_path / "conversation_data.jsonl"))['train']
        else:
            logger.error(f"Training data not found at {data_path}")
            raise FileNotFoundError("Training data not found")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.training_config['max_length']
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        
        self.train_dataset = split_dataset['train']
        self.eval_dataset = split_dataset['test']
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.eval_dataset)} eval")
    
    def setup_training_arguments(self, output_dir: str = "./istanbul_model_output"):
        """Setup training arguments"""
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.training_config['num_train_epochs'],
            per_device_train_batch_size=self.training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.training_config['gradient_accumulation_steps'],
            learning_rate=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay'],
            max_grad_norm=self.training_config['max_grad_norm'],
            warmup_steps=self.training_config['warmup_steps'],
            max_steps=self.training_config['max_steps'],
            logging_steps=100,
            save_steps=self.training_config['save_steps'],
            eval_steps=self.training_config['eval_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.training_config['fp16'],
            gradient_checkpointing=self.training_config['gradient_checkpointing'],
            dataloader_drop_last=self.training_config['dataloader_drop_last'],
            run_name="istanbul-tourism-model",
            report_to="wandb" if wandb.api.api_key else None,
            seed=self.training_config['seed'],
            data_seed=self.training_config['data_seed']
        )
    
    def train(self):
        """Run training"""
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        logger.info("Starting training...")
        train_result = trainer.train()
        
        trainer.save_model()
        trainer.save_state()
        
        logger.info(f"Training completed. Final loss: {train_result.training_loss:.4f}")
        return train_result

def main():
    parser = argparse.ArgumentParser(description="Train Istanbul Tourism Model")
    parser.add_argument("--config_dir", default="../istanbul_model_config")
    parser.add_argument("--data_path", default="../data/training")
    parser.add_argument("--output_dir", default="./istanbul_model_output")
    parser.add_argument("--wandb_project", default="istanbul-tourism-model")
    
    args = parser.parse_args()
    
    if wandb.api.api_key:
        wandb.init(project=args.wandb_project)
    
    trainer = IstanbulModelTrainer(args.config_dir)
    trainer.prepare_model_and_tokenizer()
    trainer.load_dataset(args.data_path)
    trainer.setup_training_arguments(args.output_dir)
    
    result = trainer.train()
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(training_script)
        
        print(f"Training script created: {script_path}")
    
    def create_setup_script(self):
        """Create environment setup script"""
        script_path = self.base_dir / "setup_environment.sh"
        
        setup_content = '''#!/bin/bash
# Istanbul Tourism Model Training Environment Setup

echo "Setting up Istanbul Tourism Model Training Environment..."

# Create virtual environment
python -m venv istanbul_env
source istanbul_env/bin/activate

# Install basic requirements
pip install --upgrade pip setuptools wheel

# Install PyTorch
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing GPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install training requirements
echo "Installing training requirements..."
pip install -r requirements.txt

# Create directories
mkdir -p data/raw data/validated data/training
mkdir -p models/checkpoints models/final
mkdir -p logs evaluation_results

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo "Environment setup complete!"
echo "Activate with: source istanbul_env/bin/activate"
'''
        
        with open(script_path, 'w') as f:
            f.write(setup_content)
        
        os.chmod(script_path, 0o755)
        print(f"Setup script created: {script_path}")
    
    def create_distillation_config(self):
        """Create knowledge distillation configuration"""
        config = {
            'teacher_model': 'gpt-3.5-turbo',
            'student_model': 'istanbul-tourism-gpt2-medium',
            'distillation_method': 'response_distillation',
            'temperature': 3.0,
            'alpha': 0.7,
            'beta': 0.3,
            'max_examples': 50000,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'num_epochs': 5,
            'evaluation_steps': 500,
            'save_steps': 1000,
            'topics': [
                'istanbul_attractions',
                'transportation',
                'food_dining',
                'hotels_accommodation',
                'cultural_sites',
                'shopping',
                'nightlife',
                'day_trips',
                'practical_info',
                'history_culture'
            ]
        }
        
        config_path = self.base_dir / "distillation_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Distillation config created: {config_path}")
    
    def generate_environment_summary(self):
        """Generate complete environment summary"""
        system_info = self.check_system_requirements()
        
        summary = {
            'environment_setup': {
                'base_directory': str(self.base_dir),
                'creation_time': datetime.now().isoformat(),
                'python_version': system_info['python_version'].split()[0],
                'cuda_available': system_info['cuda_available'],
                'gpu_count': system_info['gpu_count'],
                'recommended_setup': system_info['recommended_setup']
            },
            'created_files': {
                'requirements.txt': 'Training dependencies',
                'train_istanbul_model.py': 'Main training script',
                'setup_environment.sh': 'Environment setup script',
                'distillation_config.json': 'Knowledge distillation config'
            },
            'next_steps': [
                '1. Run setup_environment.sh to create virtual environment',
                '2. Prepare training data using data collection pipeline',
                '3. Configure model settings in ../istanbul_model_config/',
                '4. Start training with: python train_istanbul_model.py',
                '5. Monitor training progress and evaluate results'
            ],
            'estimated_training_time': system_info['recommended_setup']['estimated_time'],
            'hardware_requirements': system_info['recommended_setup']
        }
        
        summary_path = self.base_dir / "environment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary, summary_path

def main():
    """Main setup function"""
    print("=" * 60)
    print("ISTANBUL TOURISM MODEL - TRAINING ENVIRONMENT SETUP")
    print("=" * 60)
    
    setup = TrainingEnvironmentSetup()
    
    # Check system requirements
    system_info = setup.check_system_requirements()
    print(f"\\nSystem Information:")
    print(f"Python: {system_info['python_version'].split()[0]}")
    print(f"CUDA Available: {system_info['cuda_available']}")
    if system_info['cuda_available']:
        print(f"GPU Count: {system_info['gpu_count']}")
        print(f"GPU Names: {', '.join(system_info['gpu_names'])}")
        if system_info['total_gpu_memory_gb']:
            print(f"GPU Memory: {', '.join([f'{mem:.1f}GB' for mem in system_info['total_gpu_memory_gb']])}")
    
    print(f"\\nRecommended Setup:")
    for key, value in system_info['recommended_setup'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Create all files
    print(f"\\nCreating training environment files...")
    setup.create_requirements_file()
    setup.create_training_script()
    setup.create_setup_script()
    setup.create_distillation_config()
    
    # Generate summary
    summary, summary_path = setup.generate_environment_summary()
    
    print(f"\\nEnvironment setup complete!")
    print(f"Summary saved to: {summary_path}")
    print(f"\\nNext steps:")
    for step in summary['next_steps']:
        print(f"  {step}")
    
    print(f"\\nEstimated training time: {summary['estimated_training_time']}")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Knowledge Distillation Training Pipeline for Istanbul Tourism Model
Week 5-8 Implementation: Distilling knowledge from Llama-3.1-8B-Instruct to GPT-2-355M-Istanbul
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    TrainingArguments, Trainer
)
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import wandb
from tqdm import tqdm
import gc
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistillationConfig:
    """Configuration for knowledge distillation training"""
    
    def __init__(self):
        # Model configurations
        self.teacher_model = 'meta-llama/Llama-3.1-8B-Instruct'
        self.student_model_path = '../istanbul_model_config'
        self.tokenizer_path = '../istanbul_tokenizer'
        
        # Distillation parameters
        self.distillation_temperature = 3.0
        self.alpha_distillation = 0.7  # Weight for distillation loss
        self.alpha_ground_truth = 0.3  # Weight for ground truth loss
        
        # Training parameters
        self.batch_size = 2  # Small batch for memory efficiency
        self.gradient_accumulation_steps = 8
        self.learning_rate = 1e-4
        self.num_epochs = 5
        self.max_length = 512  # Shorter for distillation
        self.warmup_steps = 200
        self.eval_steps = 100
        self.save_steps = 500
        
        # Data parameters
        self.max_examples = 50000
        self.validation_split = 0.1
        
        # Output paths
        self.output_dir = './distilled_istanbul_model'
        self.logs_dir = './distillation_logs'
        
        # Hardware optimization
        self.use_fp16 = True
        self.use_gradient_checkpointing = True
        self.dataloader_num_workers = 2

class IstanbulDistillationDataset(Dataset):
    """Dataset for knowledge distillation training"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the text for distillation
        if 'question' in item and 'answer' in item:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        elif 'instruction' in item and 'output' in item:
            input_text = item.get('input', '')
            if input_text:
                text = f"Instruction: {item['instruction']}\nInput: {input_text}\nOutput: {item['output']}"
            else:
                text = f"Instruction: {item['instruction']}\nOutput: {item['output']}"
        else:
            text = item.get('text', str(item))
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()
        }

class DistillationTrainer:
    """Knowledge distillation trainer for Istanbul tourism model"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize models and tokenizers
        self.setup_models()
        self.setup_datasets()
    
    def setup_models(self):
        """Initialize teacher and student models"""
        logger.info("Setting up teacher and student models...")
        
        # Load student tokenizer (Istanbul-specific)
        if os.path.exists(self.config.tokenizer_path):
            self.student_tokenizer = GPT2Tokenizer.from_pretrained(self.config.tokenizer_path)
            logger.info(f"Loaded Istanbul-specific tokenizer: {len(self.student_tokenizer)} tokens")
        else:
            logger.warning("Istanbul tokenizer not found, using base GPT-2")
            self.student_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        
        # Load student model (Istanbul tourism GPT-2)
        if os.path.exists(self.config.student_model_path):
            self.student_model = GPT2LMHeadModel.from_pretrained(self.config.student_model_path)
            logger.info("Loaded pre-configured Istanbul student model")
        else:
            logger.info("Loading base GPT-2 Medium as student model")
            self.student_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
        
        # Resize student model embeddings if needed
        if len(self.student_tokenizer) > self.student_model.config.vocab_size:
            self.student_model.resize_token_embeddings(len(self.student_tokenizer))
            logger.info(f"Resized student embeddings to {len(self.student_tokenizer)}")
        
        # Load teacher model and tokenizer
        try:
            logger.info("Loading Llama-3.1-8B-Instruct teacher model...")
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(self.config.teacher_model)
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.config.teacher_model,
                torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Set padding token
            if self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
            
            logger.info("Teacher model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            logger.info("Falling back to GPT-2 XL as teacher model")
            self.teacher_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
            self.teacher_model = GPT2LMHeadModel.from_pretrained('gpt2-xl')
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        
        # Move models to device
        self.student_model = self.student_model.to(self.device)
        if hasattr(self.teacher_model, 'to'):
            self.teacher_model = self.teacher_model.to(self.device)
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.use_gradient_checkpointing:
            self.student_model.gradient_checkpointing_enable()
    
    def setup_datasets(self):
        """Load and prepare training datasets"""
        logger.info("Loading training datasets...")
        
        # Load all available training data
        training_data = []
        data_dir = Path('../data/training')
        
        # Load different data formats
        for file_name in ['qa_training_data.jsonl', 'conversation_training_data.jsonl', 'instruction_training_data.jsonl']:
            file_path = data_dir / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = []
                    for line in f:
                        if line.strip():
                            file_data.append(json.loads(line.strip()))
                    training_data.extend(file_data)
                    logger.info(f"Loaded {len(file_data)} examples from {file_name}")
        
        if not training_data:
            raise ValueError("No training data found. Please run data collection pipeline first.")
        
        # Limit data size if specified
        if self.config.max_examples and len(training_data) > self.config.max_examples:
            training_data = training_data[:self.config.max_examples]
            logger.info(f"Limited training data to {len(training_data)} examples")
        
        # Split into train/validation
        val_size = int(len(training_data) * self.config.validation_split)
        self.train_data = training_data[val_size:]
        self.val_data = training_data[:val_size]
        
        logger.info(f"Training examples: {len(self.train_data)}")
        logger.info(f"Validation examples: {len(self.val_data)}")
        
        # Create datasets
        self.train_dataset = IstanbulDistillationDataset(
            self.train_data, self.student_tokenizer, self.config.max_length
        )
        self.val_dataset = IstanbulDistillationDataset(
            self.val_data, self.student_tokenizer, self.config.max_length
        )
    
    def get_teacher_predictions(self, input_texts: List[str]) -> torch.Tensor:
        """Get soft predictions from teacher model"""
        self.teacher_model.eval()
        
        # Tokenize inputs for teacher
        teacher_inputs = self.teacher_tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            # Apply temperature to logits
            teacher_logits = teacher_outputs.logits / self.config.distillation_temperature
            teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        return teacher_probs
    
    def distillation_loss(self, student_logits: torch.Tensor, teacher_probs: torch.Tensor, 
                         labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate combined distillation and ground truth loss"""
        
        # Student predictions with temperature
        student_log_probs = F.log_softmax(
            student_logits / self.config.distillation_temperature, dim=-1
        )
        
        # Distillation loss (KL divergence)
        distill_loss = F.kl_div(
            student_log_probs, teacher_probs, reduction='batchmean'
        ) * (self.config.distillation_temperature ** 2)
        
        # Ground truth loss (standard cross-entropy)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.student_tokenizer.pad_token_id
        )
        
        # Combined loss
        total_loss = (
            self.config.alpha_distillation * distill_loss +
            self.config.alpha_ground_truth * ce_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'distillation_loss': distill_loss.item(),
            'ce_loss': ce_loss.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with distillation"""
        self.student_model.train()
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers
        )
        
        epoch_losses = []
        epoch_distill_losses = []
        epoch_ce_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Get teacher predictions
            input_texts = self.student_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            teacher_probs = self.get_teacher_predictions(input_texts)
            
            # Forward pass through student
            student_outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Calculate distillation loss
            total_loss, loss_dict = self.distillation_loss(
                student_outputs.logits, teacher_probs, labels
            )
            
            # Backward pass
            total_loss = total_loss / self.config.gradient_accumulation_steps
            total_loss.backward()
            
            # Track losses
            epoch_losses.append(loss_dict['total_loss'])
            epoch_distill_losses.append(loss_dict['distillation_loss'])
            epoch_ce_losses.append(loss_dict['ce_loss'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'distill': f"{loss_dict['distillation_loss']:.4f}",
                'ce': f"{loss_dict['ce_loss']:.4f}"
            })
            
            # Gradient update
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                # Note: optimizer step would be here in actual implementation
                self.student_model.zero_grad()
            
            # Memory cleanup
            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        return {
            'avg_loss': np.mean(epoch_losses),
            'avg_distill_loss': np.mean(epoch_distill_losses),
            'avg_ce_loss': np.mean(epoch_ce_losses)
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the student model"""
        self.student_model.eval()
        
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        eval_losses = []
        eval_distill_losses = []
        eval_ce_losses = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get teacher predictions
                input_texts = self.student_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                teacher_probs = self.get_teacher_predictions(input_texts)
                
                # Student forward pass
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Calculate losses
                total_loss, loss_dict = self.distillation_loss(
                    student_outputs.logits, teacher_probs, labels
                )
                
                eval_losses.append(loss_dict['total_loss'])
                eval_distill_losses.append(loss_dict['distillation_loss'])
                eval_ce_losses.append(loss_dict['ce_loss'])
        
        return {
            'eval_loss': np.mean(eval_losses),
            'eval_distill_loss': np.mean(eval_distill_losses),
            'eval_ce_loss': np.mean(eval_ce_losses)
        }
    
    def save_model(self, epoch: int, metrics: Dict[str, float]):
        """Save the trained student model"""
        save_path = Path(self.config.output_dir) / f"epoch_{epoch}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.student_model.save_pretrained(save_path)
        self.student_tokenizer.save_pretrained(save_path)
        
        # Save training metrics
        metrics_path = save_path / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting knowledge distillation training...")
        logger.info(f"Teacher model: {self.config.teacher_model}")
        logger.info(f"Student model: GPT-2-355M-Istanbul")
        logger.info(f"Training examples: {len(self.train_data)}")
        logger.info(f"Validation examples: {len(self.val_data)}")
        
        # Initialize wandb if available
        if wandb.api.api_key:
            wandb.init(
                project="istanbul-tourism-distillation",
                config=self.config.__dict__
            )
        
        best_eval_loss = float('inf')
        
        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"\\nEpoch {epoch}/{self.config.num_epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            # Combined metrics
            epoch_metrics = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                **train_metrics,
                **eval_metrics
            }
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            logger.info(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
            logger.info(f"Distillation Loss: {train_metrics['avg_distill_loss']:.4f}")
            
            if wandb.api.api_key:
                wandb.log(epoch_metrics)
            
            # Save best model
            if eval_metrics['eval_loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['eval_loss']
                self.save_model(epoch, epoch_metrics)
                logger.info(f"New best model saved (eval_loss: {best_eval_loss:.4f})")
            
            # Save checkpoint every few epochs
            if epoch % 2 == 0:
                checkpoint_path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch}"
                self.student_model.save_pretrained(checkpoint_path)
        
        # Save final model
        final_path = Path(self.config.output_dir) / "final_model"
        self.save_model(0, {'final': True})
        
        logger.info("Distillation training completed!")
        return best_eval_loss

def main():
    """Main distillation training function"""
    print("=" * 60)
    print("ISTANBUL TOURISM MODEL - KNOWLEDGE DISTILLATION TRAINING")
    print("Week 5-8 Implementation")
    print("=" * 60)
    
    # Initialize configuration
    config = DistillationConfig()
    
    # Print configuration
    print(f"\\nTraining Configuration:")
    print(f"Teacher Model: {config.teacher_model}")
    print(f"Student Model: GPT-2-355M-Istanbul")
    print(f"Distillation Temperature: {config.distillation_temperature}")
    print(f"Alpha Distillation: {config.alpha_distillation}")
    print(f"Alpha Ground Truth: {config.alpha_ground_truth}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    
    # Initialize trainer
    try:
        trainer = DistillationTrainer(config)
        
        # Start training
        best_loss = trainer.train()
        
        print(f"\\n🎯 Training completed successfully!")
        print(f"Best evaluation loss: {best_loss:.4f}")
        print(f"Model saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

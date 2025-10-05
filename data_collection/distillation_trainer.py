"""
Istanbul Tourism Model Knowledge Distillation Training
Week 5-8 Implementation: Training with distillation from open-source teacher model
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, Dataset as HFDataset
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistillationTrainer:
    """Knowledge Distillation Trainer for Istanbul Tourism Model"""
    
    def __init__(self, config_path: str = "./training_environment/distillation_config.json"):
        self.config_path = Path(config_path)
        self.load_config()
        self.setup_models()
        self.setup_training()
        
    def load_config(self):
        """Load distillation configuration"""
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Loaded distillation config from {self.config_path}")
        
    def setup_models(self):
        """Initialize teacher and student models"""
        logger.info("Setting up teacher and student models...")
        
        # Setup teacher model (Llama-3.1-8B alternative)
        teacher_config = self.config['teacher_model']
        try:
            # Try to use a smaller, free teacher model for demonstration
            teacher_model_name = "microsoft/DialoGPT-medium"  # Free alternative
            logger.info(f"Loading teacher model: {teacher_model_name}")
            
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if missing
            if self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
                
        except Exception as e:
            logger.warning(f"Could not load teacher model: {e}")
            logger.info("Using GPT-2 Large as teacher model instead")
            self.teacher_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
            self.teacher_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
            if self.teacher_tokenizer.pad_token is None:
                self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        
        # Setup student model (our Istanbul tourism GPT-2 Medium)
        logger.info("Setting up student model (Istanbul GPT-2 Medium)...")
        
        # Load custom configuration if available
        istanbul_config_path = Path("./istanbul_model_config/config.json")
        if istanbul_config_path.exists():
            self.student_config = GPT2Config.from_pretrained(str(istanbul_config_path))
        else:
            self.student_config = GPT2Config.from_pretrained("gpt2-medium")
        
        # Load extended tokenizer if available
        istanbul_tokenizer_path = Path("./istanbul_tokenizer")
        if istanbul_tokenizer_path.exists():
            self.student_tokenizer = GPT2Tokenizer.from_pretrained(str(istanbul_tokenizer_path))
        else:
            self.student_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        
        # Initialize student model from scratch with config
        self.student_model = GPT2LMHeadModel(self.student_config)
        
        # Resize embeddings if tokenizer was extended
        if len(self.student_tokenizer) > self.student_model.config.vocab_size:
            self.student_model.resize_token_embeddings(len(self.student_tokenizer))
            logger.info(f"Resized student model embeddings to {len(self.student_tokenizer)}")
        
        # Move models to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model.eval()  # Teacher is always in eval mode
        self.student_model.to(self.device)
        
        logger.info(f"Models loaded on device: {self.device}")
        
    def setup_training(self):
        """Setup training components"""
        # Distillation parameters
        distill_params = self.config['distillation_params']
        self.temperature = distill_params['temperature']
        self.alpha_distillation = distill_params['alpha_distillation']
        self.alpha_ground_truth = distill_params['alpha_ground_truth']
        
        # Training arguments
        training_config = self.config['training_args']
        self.training_args = TrainingArguments(**training_config)
        
        logger.info("Training setup completed")
        
    def load_training_data(self) -> Dict[str, HFDataset]:
        """Load Istanbul tourism training data"""
        logger.info("Loading training data...")
        
        data_config = self.config['training_data']
        
        # Load training data files
        data_files = {}
        train_file = Path(data_config['train_file'])
        if train_file.exists():
            data_files['train'] = str(train_file)
        else:
            # Fallback to available training data
            available_files = list(Path("./data/training").glob("*_training_data.jsonl"))
            if available_files:
                data_files['train'] = str(available_files[0])
                logger.info(f"Using available training file: {available_files[0]}")
        
        # Load validation data
        val_file = Path(data_config['validation_file'])
        if val_file.exists():
            data_files['validation'] = str(val_file)
        elif len(available_files) > 1:
            data_files['validation'] = str(available_files[1])
        
        if not data_files:
            raise FileNotFoundError("No training data files found!")
        
        # Load dataset
        dataset = load_dataset('json', data_files=data_files)
        
        # Limit samples if specified
        max_train = data_config.get('max_train_samples')
        max_eval = data_config.get('max_eval_samples')
        
        if max_train and len(dataset['train']) > max_train:
            dataset['train'] = dataset['train'].select(range(max_train))
            
        if 'validation' in dataset and max_eval and len(dataset['validation']) > max_eval:
            dataset['validation'] = dataset['validation'].select(range(max_eval))
        
        logger.info(f"Loaded {len(dataset['train'])} training examples")
        if 'validation' in dataset:
            logger.info(f"Loaded {len(dataset['validation'])} validation examples")
        
        return dataset
    
    def prepare_data_for_distillation(self, dataset: Dict[str, HFDataset]) -> Dict[str, HFDataset]:
        """Prepare data for distillation training"""
        logger.info("Preparing data for distillation...")
        
        def tokenize_function(examples):
            # Handle different data formats
            texts = []
            if 'text' in examples:
                texts = examples['text']
            elif 'question' in examples and 'answer' in examples:
                texts = [f"Q: {q} A: {a}" for q, a in zip(examples['question'], examples['answer'])]
            elif 'instruction' in examples and 'output' in examples:
                texts = [f"{inst} {out}" for inst, out in zip(examples['instruction'], examples['output'])]
            else:
                # Fallback: use first available text field
                for key in examples.keys():
                    if isinstance(examples[key][0], str):
                        texts = examples[key]
                        break
            
            if not texts:
                raise ValueError("No text data found in examples")
            
            # Tokenize for student model
            tokenized = self.student_tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,  # Reasonable length for tourism Q&A
                return_tensors="pt"
            )
            
            # Add labels (same as input_ids for language modeling)
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        return tokenized_dataset
    
    def distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                         labels: torch.Tensor) -> torch.Tensor:
        """Calculate distillation loss"""
        
        # Standard cross-entropy loss (ground truth)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Distillation loss (soft targets from teacher)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        distill_loss = F.kl_div(
            student_probs.view(-1, student_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (
            self.alpha_ground_truth * ce_loss +
            self.alpha_distillation * distill_loss
        )
        
        return total_loss, ce_loss, distill_loss
    
    def get_teacher_outputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get teacher model outputs"""
        with torch.no_grad():
            # Convert student tokens to teacher tokens if vocabularies differ
            if len(self.teacher_tokenizer) != len(self.student_tokenizer):
                # Decode and re-encode (simple approach)
                texts = self.student_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                teacher_inputs = self.teacher_tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=input_ids.size(1)
                ).to(input_ids.device)
                
                teacher_outputs = self.teacher_model(**teacher_inputs)
            else:
                # Direct use if vocabularies match
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        
        return teacher_outputs.logits
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with distillation"""
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Get student outputs
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        student_logits = student_outputs.logits
        
        # Get teacher outputs
        teacher_logits = self.get_teacher_outputs(input_ids, attention_mask)
        
        # Ensure logits have same shape (pad or truncate as needed)
        min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_seq_len, :]
        teacher_logits = teacher_logits[:, :min_seq_len, :]
        labels = labels[:, :min_seq_len]
        
        # Calculate distillation loss
        total_loss, ce_loss, distill_loss = self.distillation_loss(
            student_logits, teacher_logits, labels
        )
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss.item(),
            'distill_loss': distill_loss.item()
        }
    
    def train(self):
        """Main training loop"""
        logger.info("Starting distillation training...")
        
        # Initialize wandb if available
        use_wandb = False
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="istanbul-tourism-distillation",
                    config=self.config,
                    name=f"distillation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )
                use_wandb = True
            except Exception as e:
                logger.warning(f"Could not initialize wandb: {e}")
        
        # Load and prepare data
        dataset = self.load_training_data()
        tokenized_dataset = self.prepare_data_for_distillation(dataset)
        
        # Create data loader
        train_dataloader = DataLoader(
            tokenized_dataset['train'],
            batch_size=self.training_args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay
        )
        
        num_training_steps = len(train_dataloader) * self.training_args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * self.training_args.warmup_ratio),
            num_training_steps=num_training_steps
        )
        
        # Training loop
        self.student_model.train()
        global_step = 0
        
        for epoch in range(int(self.training_args.num_train_epochs)):
            logger.info(f"Starting epoch {epoch + 1}/{self.training_args.num_train_epochs}")
            
            epoch_losses = {'total': [], 'ce': [], 'distill': []}
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                losses = self.train_step(batch)
                
                # Backward pass
                total_loss = losses['total_loss']
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.training_args.max_grad_norm
                )
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Log metrics
                epoch_losses['total'].append(total_loss.item())
                epoch_losses['ce'].append(losses['ce_loss'])
                epoch_losses['distill'].append(losses['distill_loss'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'ce': f"{losses['ce_loss']:.4f}",
                    'distill': f"{losses['distill_loss']:.4f}"
                })
                
                # Log to wandb
                if use_wandb and global_step % self.training_args.logging_steps == 0:
                    wandb.log({
                        'train/total_loss': total_loss.item(),
                        'train/ce_loss': losses['ce_loss'],
                        'train/distill_loss': losses['distill_loss'],
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/epoch': epoch
                    }, step=global_step)
                
                global_step += 1
                
                # Save checkpoint
                if global_step % self.training_args.save_steps == 0:
                    self.save_checkpoint(global_step)
            
            # End of epoch logging
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            logger.info(f"Epoch {epoch + 1} completed. Avg losses: {avg_losses}")
            
            if use_wandb:
                wandb.log({
                    'epoch/total_loss': avg_losses['total'],
                    'epoch/ce_loss': avg_losses['ce'],
                    'epoch/distill_loss': avg_losses['distill'],
                    'epoch/number': epoch
                })
        
        # Save final model
        self.save_model()
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        logger.info("Distillation training completed!")
    
    def collate_fn(self, batch):
        """Custom collate function for batch processing"""
        # Convert list of dicts to dict of lists
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key in ['input_ids', 'attention_mask', 'labels']:
                # Pad sequences
                sequences = [item[key] for item in batch]
                collated[key] = torch.nn.utils.rnn.pad_sequence(
                    sequences, batch_first=True, padding_value=0
                )
        
        return collated
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.training_args.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.student_model.save_pretrained(checkpoint_dir)
        self.student_tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def save_model(self):
        """Save final trained model"""
        output_dir = Path(self.training_args.output_dir) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.student_model.save_pretrained(output_dir)
        self.student_tokenizer.save_pretrained(output_dir)
        
        # Save training config
        with open(output_dir / "distillation_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Final model saved to {output_dir}")

def main():
    """Main distillation training function"""
    print("=" * 70)
    print("🎓 ISTANBUL TOURISM MODEL KNOWLEDGE DISTILLATION TRAINING")
    print("=" * 70)
    print(f"📅 Week 5-8 Implementation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize distillation trainer
        trainer = DistillationTrainer()
        
        # Start training
        trainer.train()
        
        print("\n✅ Distillation training completed successfully!")
        print(f"📁 Model saved to: {trainer.training_args.output_dir}/final")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()

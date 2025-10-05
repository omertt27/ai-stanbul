"""
Simplified Istanbul Tourism Model Training Script
Week 5-8 Implementation: Basic training without complex dependencies
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
    get_linear_schedule_with_warmup
)
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleDistillationTrainer:
    """Simplified distillation trainer for Istanbul tourism model"""
    
    def __init__(self):
        self.setup_models()
        self.setup_training_params()
        
    def setup_models(self):
        """Initialize teacher and student models"""
        logger.info("Setting up models...")
        
        # Use GPT-2 Large as teacher (more accessible than Llama)
        logger.info("Loading teacher model: GPT-2 Large")
        self.teacher_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        self.teacher_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        
        # Setup student model (Istanbul GPT-2 Medium)
        logger.info("Setting up student model...")
        
        # Load custom config if available
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
        
        # Initialize student model from scratch
        self.student_model = GPT2LMHeadModel(self.student_config)
        
        # Resize embeddings if tokenizer was extended
        if len(self.student_tokenizer) > self.student_model.config.vocab_size:
            self.student_model.resize_token_embeddings(len(self.student_tokenizer))
            logger.info(f"Resized student model embeddings to {len(self.student_tokenizer)}")
        
        # Move models to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        
        # Teacher is always in eval mode
        self.teacher_model.eval()
        
        logger.info(f"Models loaded on device: {self.device}")
        
    def setup_training_params(self):
        """Setup training parameters"""
        self.temperature = 3.0
        self.alpha_distillation = 0.7
        self.alpha_ground_truth = 0.3
        self.learning_rate = 3e-5
        self.num_epochs = 2
        self.batch_size = 1 if self.device.type == 'cpu' else 2
        self.gradient_accumulation_steps = 4
        self.max_grad_norm = 1.0
        self.output_dir = Path("./simple_distillation_output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Training setup: {self.num_epochs} epochs, batch size {self.batch_size}")
        
    def load_training_data(self):
        """Load Istanbul tourism training data"""
        logger.info("Loading training data...")
        
        # Load training data files
        data_files = []
        for file_pattern in ["*_training_data.jsonl"]:
            files = list(Path("./data/training").glob(file_pattern))
            data_files.extend(files)
        
        if not data_files:
            raise FileNotFoundError("No training data files found!")
        
        # Load all data
        all_data = []
        for file_path in data_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(all_data)} training examples")
        
        # Process data into text format
        texts = []
        for item in all_data:
            if 'question' in item and 'answer' in item:
                text = f"Q: {item['question']} A: {item['answer']}"
            elif 'text' in item:
                text = item['text']
            elif 'instruction' in item and 'output' in item:
                text = f"{item['instruction']} {item['output']}"
            else:
                continue
            texts.append(text)
        
        return texts
    
    def create_dataset(self, texts: List[str]):
        """Create PyTorch dataset"""
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                
                # Tokenize
                encoded = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoded['input_ids'].squeeze(),
                    'attention_mask': encoded['attention_mask'].squeeze(),
                    'labels': encoded['input_ids'].squeeze().clone()
                }
        
        return TextDataset(texts, self.student_tokenizer)
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Calculate distillation loss"""
        # Standard cross-entropy loss (ground truth)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.student_tokenizer.pad_token_id
        )
        
        # Distillation loss (soft targets from teacher)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Ensure same vocabulary size by truncating teacher logits if needed
        vocab_size = min(student_logits.size(-1), teacher_logits.size(-1))
        student_probs = student_probs[:, :, :vocab_size]
        teacher_probs = teacher_probs[:, :, :vocab_size]
        
        distill_loss = F.kl_div(
            student_probs.view(-1, vocab_size),
            teacher_probs.view(-1, vocab_size),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (
            self.alpha_ground_truth * ce_loss +
            self.alpha_distillation * distill_loss
        )
        
        return total_loss, ce_loss, distill_loss
    
    def get_teacher_outputs(self, input_ids, attention_mask):
        """Get teacher model outputs with vocabulary alignment"""
        with torch.no_grad():
            # Convert student tokens to teacher tokens if vocabularies differ
            batch_size = input_ids.shape[0]
            teacher_input_ids = []
            teacher_attention_mask = []
            
            for i in range(batch_size):
                # Get the actual tokens (excluding padding)
                seq_len = attention_mask[i].sum().item()
                student_tokens = input_ids[i][:seq_len]
                
                # Decode with student tokenizer and re-encode with teacher tokenizer
                text = self.student_tokenizer.decode(student_tokens, skip_special_tokens=True)
                
                # Re-encode with teacher tokenizer
                teacher_encoded = self.teacher_tokenizer(
                    text,
                    max_length=input_ids.shape[1],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                teacher_input_ids.append(teacher_encoded['input_ids'][0])
                teacher_attention_mask.append(teacher_encoded['attention_mask'][0])
            
            teacher_input_ids = torch.stack(teacher_input_ids)
            teacher_attention_mask = torch.stack(teacher_attention_mask)
            
            teacher_outputs = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask
            )
        return teacher_outputs.logits
    
    def train(self):
        """Main training loop"""
        logger.info("Starting simplified distillation training...")
        
        # Load data
        texts = self.load_training_data()
        dataset = self.create_dataset(texts)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        num_training_steps = len(dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * 0.1),
            num_training_steps=num_training_steps
        )
        
        # Training loop
        self.student_model.train()
        global_step = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            epoch_losses = {'total': [], 'ce': [], 'distill': []}
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
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
                
                # Ensure same sequence length
                min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
                student_logits = student_logits[:, :min_seq_len, :]
                teacher_logits = teacher_logits[:, :min_seq_len, :]
                labels = labels[:, :min_seq_len]
                
                # Calculate losses
                total_loss, ce_loss, distill_loss = self.distillation_loss(
                    student_logits, teacher_logits, labels
                )
                
                # Scale loss for gradient accumulation
                total_loss = total_loss / self.gradient_accumulation_steps
                
                # Backward pass
                total_loss.backward()
                
                # Record losses
                epoch_losses['total'].append(total_loss.item() * self.gradient_accumulation_steps)
                epoch_losses['ce'].append(ce_loss.item())
                epoch_losses['distill'].append(distill_loss.item())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item() * self.gradient_accumulation_steps:.4f}",
                    'ce': f"{ce_loss.item():.4f}",
                    'distill': f"{distill_loss.item():.4f}"
                })
                
                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.max_grad_norm
                    )
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Save checkpoint periodically
                    if global_step % 50 == 0:
                        self.save_checkpoint(global_step)
                
                # Early stopping for demo (remove for full training)
                if global_step >= 100:  # Limit to 100 steps for demo
                    logger.info("Demo training completed (100 steps)")
                    break
            
            # End of epoch logging
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            logger.info(f"Epoch {epoch + 1} completed. Avg losses: {avg_losses}")
            
            if global_step >= 100:  # Exit if demo limit reached
                break
        
        # Save final model
        self.save_model()
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, step: int):
        """Save training checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        self.student_model.save_pretrained(checkpoint_dir)
        self.student_tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def save_model(self):
        """Save final trained model"""
        final_dir = self.output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        
        self.student_model.save_pretrained(final_dir)
        self.student_tokenizer.save_pretrained(final_dir)
        
        # Save training info
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'epochs': self.num_epochs,
            'device': str(self.device),
            'teacher_model': 'gpt2-large',
            'student_model': 'istanbul-tourism-gpt2-medium',
            'distillation_params': {
                'temperature': self.temperature,
                'alpha_distillation': self.alpha_distillation,
                'alpha_ground_truth': self.alpha_ground_truth
            }
        }
        
        with open(final_dir / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Final model saved to {final_dir}")

def main():
    """Main training function"""
    print("=" * 70)
    print("🎓 SIMPLE ISTANBUL TOURISM MODEL DISTILLATION TRAINING")
    print("=" * 70)
    print(f"📅 Week 5-8 Implementation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize trainer
        trainer = SimpleDistillationTrainer()
        
        # Start training
        trainer.train()
        
        print("\n✅ Distillation training completed successfully!")
        print(f"📁 Model saved to: {trainer.output_dir}/final")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

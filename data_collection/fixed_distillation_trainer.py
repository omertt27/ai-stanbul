"""
Fixed Istanbul Tourism Model Training Script
Week 5-8 Implementation: Simplified training without vocabulary mismatch
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
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

class FixedDistillationTrainer:
    """Fixed distillation trainer using same tokenizer for both models"""
    
    def __init__(self):
        self.setup_models()
        self.setup_training_params()
        
    def setup_models(self):
        """Initialize teacher and student models with same tokenizer"""
        logger.info("Setting up models...")
        
        # Use same tokenizer for both models to avoid vocabulary mismatch
        logger.info("Loading shared tokenizer: GPT-2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Teacher model (GPT-2 Large)
        logger.info("Loading teacher model: GPT-2 Large")
        self.teacher_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        self.teacher_model.eval()
        
        # Student model (GPT-2 Medium) - same tokenizer
        logger.info("Setting up student model: GPT-2 Medium")
        self.student_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        
        # Add Istanbul-specific special tokens
        special_tokens = [
            "<istanbul>", "<travel>", "<tourism>", "<culture>", 
            "<history>", "<food>", "<transport>", "<accommodation>"
        ]
        
        num_added_tokens = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        
        if num_added_tokens > 0:
            # Resize both models' embeddings
            self.teacher_model.resize_token_embeddings(len(self.tokenizer))
            self.student_model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added_tokens} Istanbul-specific tokens")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)
        
        logger.info(f"Models loaded on device: {self.device}")
        
    def setup_training_params(self):
        """Setup training hyperparameters"""
        self.num_epochs = 3  # Reduced for faster testing
        self.batch_size = 1  # Small batch for memory efficiency
        self.gradient_accumulation_steps = 4
        self.learning_rate = 3e-5
        self.temperature = 3.0
        self.alpha_distillation = 0.7
        self.alpha_ground_truth = 0.3
        self.max_length = 512
        
        logger.info(f"Training setup: {self.num_epochs} epochs, batch size {self.batch_size}")
        
    def load_training_data(self):
        """Load Istanbul tourism training data"""
        logger.info("Loading training data...")
        
        texts = []
        data_dir = Path("data/training")
        
        # Load different format files
        for file_path in data_dir.glob("*.jsonl"):
            if file_path.name in ['qa_training_data.jsonl', 'instruction_training_data.jsonl', 'conversation_training_data.jsonl']:
                logger.info(f"Loading {file_path.name}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'input' in data and 'output' in data:
                                # Format as prompt-response pair
                                text = f"<istanbul>{data['input']}<travel>{data['output']}<|endoftext|>"
                                texts.append(text)
                            elif 'text' in data:
                                # Direct text format
                                texts.append(f"<istanbul>{data['text']}<|endoftext|>")
                        except json.JSONDecodeError:
                            continue
        
        logger.info(f"Loaded {len(texts)} training examples")
        return texts
    
    class TextDataset(Dataset):
        """Simple text dataset for training"""
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
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].squeeze()
            attention_mask = encoded['attention_mask'].squeeze()
            
            # Labels are same as input_ids for language modeling
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # Ignore padding tokens
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
    
    def create_dataset(self, texts):
        """Create dataset from texts"""
        return self.TextDataset(texts, self.tokenizer, self.max_length)
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Calculate distillation loss with aligned vocabularies"""
        # Standard cross-entropy loss (ground truth)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # Distillation loss (soft targets from teacher)
        # Both models have same vocabulary size now
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Create mask for valid positions (not padding)
        mask = (labels != -100).unsqueeze(-1).expand_as(student_probs)
        
        distill_loss = F.kl_div(
            student_probs.view(-1, student_logits.size(-1)),
            teacher_probs.view(-1, teacher_logits.size(-1)),
            reduction='none'
        )
        
        # Apply mask and average
        distill_loss = (distill_loss * mask.view(-1, student_logits.size(-1))).sum() / mask.sum()
        distill_loss = distill_loss * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (
            self.alpha_ground_truth * ce_loss +
            self.alpha_distillation * distill_loss
        )
        
        return total_loss, ce_loss, distill_loss
    
    def get_teacher_outputs(self, input_ids, attention_mask):
        """Get teacher model outputs"""
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        return teacher_outputs.logits
    
    def train(self):
        """Main training loop"""
        logger.info("Starting fixed distillation training...")
        
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
                        max_norm=1.0
                    )
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                # Save checkpoint every 50 steps
                if global_step > 0 and global_step % 50 == 0:
                    self.save_checkpoint(global_step)
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses['total'])
            avg_ce = np.mean(epoch_losses['ce'])
            avg_distill = np.mean(epoch_losses['distill'])
            
            logger.info(f"Epoch {epoch + 1} completed:")
            logger.info(f"  Average Total Loss: {avg_loss:.4f}")
            logger.info(f"  Average CE Loss: {avg_ce:.4f}")
            logger.info(f"  Average Distill Loss: {avg_distill:.4f}")
        
        # Final save
        self.save_final_model()
        logger.info("Training completed successfully!")
    
    def save_checkpoint(self, step):
        """Save training checkpoint"""
        checkpoint_dir = Path("models/istanbul_tourism_model/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint-{step}"
        self.student_model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        logger.info(f"Checkpoint saved at step {step}")
    
    def save_final_model(self):
        """Save final trained model"""
        model_dir = Path("models/istanbul_tourism_model")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.student_model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Save training info
        training_info = {
            "model_type": "Istanbul Tourism GPT-2 Medium",
            "training_date": datetime.now().isoformat(),
            "teacher_model": "GPT-2 Large",
            "student_model": "GPT-2 Medium",
            "vocabulary_size": len(self.tokenizer),
            "special_tokens": self.tokenizer.additional_special_tokens,
            "training_params": {
                "epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "temperature": self.temperature,
                "alpha_distillation": self.alpha_distillation,
                "alpha_ground_truth": self.alpha_ground_truth
            }
        }
        
        with open(model_dir / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Final model saved to {model_dir}")

def main():
    """Main training execution"""
    print("="*70)
    print("🎓 FIXED ISTANBUL TOURISM MODEL DISTILLATION TRAINING")
    print("="*70)
    print(f"📅 Week 5-8 Implementation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        trainer = FixedDistillationTrainer()
        trainer.train()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📁 Model saved to: models/istanbul_tourism_model/")
        print(f"🎯 Ready for evaluation and production deployment")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()

"""
Practical Knowledge Distillation for Istanbul Tourism Model
Week 5-8 Implementation: Simplified distillation using available models
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, GPT2Config,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PracticalDistillationConfig:
    """Practical distillation configuration for available models"""
    
    def __init__(self):
        # Model paths
        self.student_model_path = '../istanbul_model_config'
        self.tokenizer_path = '../istanbul_tokenizer'
        self.teacher_model = 'gpt2-large'  # Available teacher model
        
        # Distillation parameters  
        self.distillation_temperature = 3.0
        self.alpha_distillation = 0.7
        self.alpha_ground_truth = 0.3
        
        # Training parameters
        self.num_epochs = 3
        self.batch_size = 4
        self.gradient_accumulation_steps = 4
        self.learning_rate = 5e-5
        self.warmup_steps = 100
        self.max_length = 512
        self.save_steps = 250
        self.eval_steps = 100
        self.logging_steps = 50
        
        # Output paths
        self.output_dir = './distilled_istanbul_model'
        self.data_dir = './data/training'
        
        # Hardware settings
        self.fp16 = True
        self.gradient_checkpointing = True

class IstanbulDistillationDataset(Dataset):
    """Dataset for distillation training"""
    
    def __init__(self, examples: List[Dict], tokenizer: GPT2Tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format text based on data type
        if 'question' in example and 'answer' in example:
            text = f"Q: {example['question']}\\nA: {example['answer']}"
        elif 'instruction' in example and 'output' in example:
            input_text = example.get('input', '')
            if input_text.strip():
                text = f"{example['instruction']}\\nInput: {input_text}\\nOutput: {example['output']}"
            else:
                text = f"{example['instruction']}\\nOutput: {example['output']}"
        elif 'conversation' in example:
            # Format conversation
            conversation_text = []
            for turn in example['conversation']:
                role = turn.get('role', 'user')
                content = turn.get('content', '')
                if role == 'user':
                    conversation_text.append(f"User: {content}")
                else:
                    conversation_text.append(f"Assistant: {content}")
            text = '\\n'.join(conversation_text)
        else:
            text = str(example)
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze().clone()
        }

class DistillationTrainer:
    """Custom trainer for knowledge distillation"""
    
    def __init__(self, config: PracticalDistillationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.setup_models()
        self.load_data()
        
    def setup_models(self):
        """Setup teacher and student models"""
        logger.info("Setting up models...")
        
        # Load tokenizer
        if os.path.exists(self.config.tokenizer_path):
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.tokenizer_path)
            logger.info(f"Loaded Istanbul tokenizer: {len(self.tokenizer)} tokens")
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Using base GPT-2 tokenizer")
        
        # Load student model (our Istanbul model)
        if os.path.exists(self.config.student_model_path):
            self.student_model = GPT2LMHeadModel.from_pretrained(self.config.student_model_path)
            logger.info("Loaded Istanbul student model")
        else:
            config_dict = {
                'vocab_size': len(self.tokenizer),
                'n_positions': 2048,
                'n_embd': 1024,
                'n_layer': 24,
                'n_head': 16,
                'n_inner': 4096
            }
            model_config = GPT2Config(**config_dict)
            self.student_model = GPT2LMHeadModel(model_config)
            logger.info("Created new Istanbul student model")
        
        # Resize embeddings if needed
        if len(self.tokenizer) > self.student_model.config.vocab_size:
            self.student_model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Resized student embeddings to {len(self.tokenizer)}")
        
        # Load teacher model
        self.teacher_model = GPT2LMHeadModel.from_pretrained(self.config.teacher_model)
        
        # Move to device
        self.student_model = self.student_model.to(self.device)
        self.teacher_model = self.teacher_model.to(self.device)
        
        # Set teacher to eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        logger.info(f"Teacher model: {self.config.teacher_model}")
        logger.info(f"Student model: GPT-2 Medium (Istanbul)")
    
    def load_data(self):
        """Load training data"""
        logger.info("Loading training data...")
        
        all_examples = []
        data_dir = Path(self.config.data_dir)
        
        # Load all training files
        for file_name in ['qa_training_data.jsonl', 'conversation_training_data.jsonl', 'instruction_training_data.jsonl']:
            file_path = data_dir / file_name
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_examples = []
                    for line in f:
                        if line.strip():
                            file_examples.append(json.loads(line.strip()))
                    all_examples.extend(file_examples)
                    logger.info(f"Loaded {len(file_examples)} examples from {file_name}")
        
        if not all_examples:
            raise ValueError("No training data found!")
        
        # Split data
        val_size = max(1, len(all_examples) // 10)  # 10% validation
        self.val_examples = all_examples[:val_size]
        self.train_examples = all_examples[val_size:]
        
        logger.info(f"Training examples: {len(self.train_examples)}")
        logger.info(f"Validation examples: {len(self.val_examples)}")
        
        # Create datasets
        self.train_dataset = IstanbulDistillationDataset(
            self.train_examples, self.tokenizer, self.config.max_length
        )
        self.val_dataset = IstanbulDistillationDataset(
            self.val_examples, self.tokenizer, self.config.max_length
        )
    
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Calculate distillation loss"""
        # Apply temperature to logits
        student_log_probs = F.log_softmax(student_logits / self.config.distillation_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.config.distillation_temperature, dim=-1)
        
        # KL divergence loss
        distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        distill_loss = distill_loss * (self.config.distillation_temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # Combined loss
        total_loss = (
            self.config.alpha_distillation * distill_loss +
            self.config.alpha_ground_truth * ce_loss
        )
        
        return total_loss, {
            'distill_loss': distill_loss.item(),
            'ce_loss': ce_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_step(self, batch):
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Student forward pass
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Teacher forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Calculate distillation loss
        loss, loss_dict = self.distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            labels
        )
        
        return loss, loss_dict
    
    def train(self):
        """Main training loop"""
        logger.info("Starting distillation training...")
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues on some systems
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            self.student_model.train()
            epoch_losses = []
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                loss, loss_dict = self.train_step(batch)
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                epoch_losses.append(loss_dict['total_loss'])
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'distill': f"{loss_dict['distill_loss']:.4f}",
                    'ce': f"{loss_dict['ce_loss']:.4f}"
                })
                
                # Gradient update
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                # Validation and saving
                if global_step % self.config.eval_steps == 0:
                    val_loss = self.evaluate(val_loader)
                    logger.info(f"Step {global_step}, Val Loss: {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_model(f"best_model_step_{global_step}")
                        logger.info(f"New best model saved! Val loss: {val_loss:.4f}")
                    
                    self.student_model.train()
            
            # End of epoch
            avg_train_loss = np.mean(epoch_losses)
            val_loss = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch + 1} Summary:")
            logger.info(f"  Average Training Loss: {avg_train_loss:.4f}")
            logger.info(f"  Validation Loss: {val_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_model(f"epoch_{epoch + 1}")
        
        # Save final model
        self.save_model("final_model")
        logger.info(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        
        return best_val_loss
    
    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.student_model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                loss, _ = self.train_step(batch)
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save_model(self, checkpoint_name: str):
        """Save model checkpoint"""
        save_path = Path(self.config.output_dir) / checkpoint_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.student_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        config_dict = {
            'distillation_temperature': self.config.distillation_temperature,
            'alpha_distillation': self.config.alpha_distillation,
            'alpha_ground_truth': self.config.alpha_ground_truth,
            'teacher_model': self.config.teacher_model,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(save_path / 'distillation_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")

def main():
    """Main distillation function"""
    print("=" * 70)
    print("ISTANBUL TOURISM MODEL - PRACTICAL KNOWLEDGE DISTILLATION")
    print("Week 5-8 Implementation")
    print("=" * 70)
    
    # Check for training data
    data_dir = Path('./data/training')
    if not data_dir.exists():
        print("❌ Training data directory not found!")
        print("Please run the data collection pipeline first.")
        return
    
    # Initialize config
    config = PracticalDistillationConfig()
    
    print(f"\\nConfiguration:")
    print(f"Teacher Model: {config.teacher_model}")
    print(f"Student Model: GPT-2 Medium (Istanbul Tourism)")
    print(f"Distillation Temperature: {config.distillation_temperature}")
    print(f"Alpha Distillation: {config.alpha_distillation}")
    print(f"Alpha Ground Truth: {config.alpha_ground_truth}")
    print(f"Training Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    
    # Initialize and start training
    try:
        trainer = DistillationTrainer(config)
        best_loss = trainer.train()
        
        print(f"\\n🎯 Distillation training completed successfully!")
        print(f"Best validation loss: {best_loss:.4f}")
        print(f"Distilled model saved to: {config.output_dir}")
        
        # Summary
        print(f"\\n📊 Training Summary:")
        print(f"✅ Successfully distilled knowledge from {config.teacher_model}")
        print(f"✅ Istanbul-specific student model trained")
        print(f"✅ Model ready for deployment and evaluation")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
